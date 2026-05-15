#!/usr/bin/env python3
"""
Randomized-smoothing-style certification pilot for preliminary rRCM-path experiment.

Tags: controlled_debug_certification | randomized_smoothing_style
      | not_official_rrcm_reproduction | not_paper_level_certified_robustness
      | preliminary_evidence_only

No official rRCM checkpoint is used.
Models are re-trained from scratch (same seed / settings as prior experiment)
when saved checkpoints are not found.  Re-training is deterministic up to
CUDA non-determinism; expect minor weight differences vs the original runs.

Pilot settings
    200 test samples, N0=100, N=1000, sigma=0.5, alpha=0.001
    conditions: baseline, proto_only, proto_margin, (margin_only optional)
    seeds: 0, 1, 2
"""

import sys
import time
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
import accelerate

try:
    import scipy.stats as sp_stats
except ImportError:
    raise ImportError("scipy is required: pip install scipy")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from experiment_utils import build_finetune_model, set_seed, load_config
from rrcm_tune import train_one_epoch_consistency, load_data
from run_preliminary_rrcm_path_v2 import (
    make_margin_config, make_loaders,
    NOISE_AUG, LBD, ETA,
    CONDITIONS_CONFIG, HPARAM_VARIANTS,
)

# ---------------------------------------------------------------------------
# Pilot configuration
# ---------------------------------------------------------------------------

CONDITIONS_DEFAULT = ["baseline", "proto_only", "proto_margin", "margin_only"]
HPARAM   = "A"
SEEDS    = [0, 1, 2]

TRAIN_N  = 2000
TEST_N   = 500
EPOCHS   = 7
LR       = 1e-4

N_CERT   = 200       # number of test samples to certify
N0       = 100       # samples for class selection
N        = 1000      # samples for Clopper-Pearson bound
SIGMA    = 0.5       # smoothing noise
ALPHA    = 0.001     # one-sided confidence level
CBATCH   = 100       # noisy-forward batch size
RADIUS_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

OUT_DIR  = Path("results/preliminary_rrcm_path_experiment_v2")
CKPT_DIR = OUT_DIR / "checkpoints"
CERT_DIR = OUT_DIR / "certification"

# ---------------------------------------------------------------------------
# Randomized-smoothing helpers
# ---------------------------------------------------------------------------

def _cp_lower(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower confidence bound at level alpha."""
    if k == 0:
        return 0.0
    return float(sp_stats.beta.ppf(alpha, k, n - k + 1))


@torch.no_grad()
def _vote(model, x: torch.Tensor, num: int, sigma: float, device) -> np.ndarray:
    """Return class vote counts over `num` independent noisy passes of x."""
    counts = np.zeros(10, dtype=np.int64)
    rem = num
    while rem > 0:
        bs = min(CBATCH, rem)
        imgs = x.unsqueeze(0).expand(bs, -1, -1, -1).contiguous()
        preds = model(imgs, noise_aug=sigma).argmax(1).cpu().numpy()
        for p in preds:
            counts[p] += 1
        rem -= bs
    return counts


def certify_one(model, x: torch.Tensor, y: int, device):
    """
    Returns (cAbar, abstain, certified_radius, correct).
      cAbar            : smoothed top class (from N0 votes)
      abstain          : True if pA_lower <= 0.5 (cannot certify)
      certified_radius : sigma * Phi^{-1}(pA_lower); -1.0 when abstain
      correct          : cAbar == y
    """
    cAbar = int(_vote(model, x, N0, SIGMA, device).argmax())
    correct = cAbar == y

    nA = int(_vote(model, x, N, SIGMA, device)[cAbar])
    pA_lo = _cp_lower(nA, N, ALPHA)

    if pA_lo > 0.5:
        radius = float(SIGMA * sp_stats.norm.ppf(pA_lo))
        radius = min(radius, 10.0)   # cap against numerical overflow near pA_lo≈1
        return cAbar, False, radius, correct
    return cAbar, True, -1.0, correct


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_and_save(condition, seed, config, data_dir, device, ckpt_path):
    print(f"  [retrain] {condition} hpA seed={seed} ...", flush=True)
    set_seed(seed)
    acc = accelerate.Accelerator(cpu=(device.type == "cpu"))
    model = build_finetune_model(config, device=device, tiny_random=True)
    train_loader, _ = make_loaders(config, data_dir, TRAIN_N, TEST_N)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters()
         if p.requires_grad and n not in ("linear_head.weight", "linear_head.bias")],
        lr=LR, weight_decay=0.0,
    )
    loss_fn = nn.CrossEntropyLoss()
    margin_config = make_margin_config(condition, HPARAM)
    model, train_loader, optimizer = acc.prepare(model, train_loader, optimizer)

    for epoch in range(EPOCHS):
        train_one_epoch_consistency(
            acc, model, train_loader, optimizer, loss_fn,
            mixup_fn=None, noise_aug=NOISE_AUG, lbd=LBD, eta=ETA,
            margin_config=margin_config,
        )
        print(f"    ep{epoch}", flush=True)

    raw = acc.unwrap_model(model)
    raw.eval()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw.state_dict(), ckpt_path)
    print(f"  saved: {ckpt_path}", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return raw


def _load_model(ckpt_path, config, device):
    model = build_finetune_model(config, device=device, tiny_random=True)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _rk(r):
    """Turn a float radius into a safe CSV/dict key string."""
    return "r" + str(r).replace(".", "")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(all_rows, conditions_done):
    per_seed = []
    for cond in conditions_done:
        for seed in SEEDS:
            rows = [r for r in all_rows if r["condition"] == cond and r["seed"] == seed]
            if not rows:
                continue
            n = len(rows)
            n_correct = sum(r["correct"] for r in rows)
            n_abstain = sum(r["abstain"] for r in rows)
            # certified at r=0 but radius < first non-zero grid threshold
            n_low_rad = sum(
                1 for r in rows
                if r["correct"] and not r["abstain"]
                and r["certified_radius"] < RADIUS_GRID[1]
            )
            ps = {
                "condition":       cond,
                "seed":            seed,
                "n_samples":       n,
                "n_correct":       n_correct,
                "n_abstain":       n_abstain,
                "n_low_radius":    n_low_rad,
                "smoothed_acc":    100.0 * n_correct / n,
            }
            for rad in RADIUS_GRID:
                cert = sum(
                    1 for r in rows
                    if r["correct"] and not r["abstain"]
                    and r["certified_radius"] >= rad
                )
                ps[f"cert_{_rk(rad)}"] = 100.0 * cert / n
            per_seed.append(ps)

    agg = []
    for cond in conditions_done:
        rows = [r for r in per_seed if r["condition"] == cond]
        if not rows:
            continue
        def _m(k): return float(np.mean([r[k] for r in rows]))
        def _s(k): return float(np.std([r[k] for r in rows]))
        a = {
            "condition":         cond,
            "n_seeds":           len(rows),
            "smoothed_acc_mean": _m("smoothed_acc"),
            "smoothed_acc_std":  _s("smoothed_acc"),
            "n_abstain_mean":    _m("n_abstain"),
            "n_low_radius_mean": _m("n_low_radius"),
        }
        for rad in RADIUS_GRID:
            k = f"cert_{_rk(rad)}"
            a[f"{k}_mean"] = _m(k)
            a[f"{k}_std"]  = _s(k)
        agg.append(a)
    return agg, per_seed


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _make_plot(agg_rows, path):
    colors = {
        "baseline":     "#555555",
        "proto_only":   "#2196F3",
        "proto_margin": "#4CAF50",
        "margin_only":  "#FF9800",
    }
    order = ["baseline", "proto_only", "margin_only", "proto_margin"]
    agg_map = {r["condition"]: r for r in agg_rows}

    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in order:
        if cond not in agg_map:
            continue
        a = agg_map[cond]
        ym = [a[f"cert_{_rk(r)}_mean"] for r in RADIUS_GRID]
        ys = [a[f"cert_{_rk(r)}_std"]  for r in RADIUS_GRID]
        ax.plot(RADIUS_GRID, ym, "o-", label=cond,
                color=colors.get(cond, "gray"), linewidth=1.8)
        ax.fill_between(
            RADIUS_GRID,
            [m - s for m, s in zip(ym, ys)],
            [m + s for m, s in zip(ym, ys)],
            alpha=0.15, color=colors.get(cond, "gray"),
        )

    ax.set_xlabel("Certified radius r")
    ax.set_ylabel("Certified accuracy (%)")
    ax.set_title(
        f"Certified accuracy vs radius\n"
        f"pilot: {N_CERT} samples, N={N}, σ={SIGMA}, α={ALPHA}  "
        f"[NOT official rRCM | NOT paper-level certified robustness]",
        fontsize=8.5,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(-0.03, max(RADIUS_GRID) + 0.07)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _write_summary(path, agg_rows, _per_seed, total_elapsed, conditions_done):
    agg = {r["condition"]: r for r in agg_rows}

    def cm(c, r): return agg[c].get(f"cert_{_rk(r)}_mean", float("nan"))
    def cs(c, r): return agg[c].get(f"cert_{_rk(r)}_std",  float("nan"))
    def sm(c):    return agg[c].get("smoothed_acc_mean",    float("nan"))
    def ss(c):    return agg[c].get("smoothed_acc_std",     float("nan"))
    has = lambda c: c in agg

    # Q1
    if has("proto_only") and has("baseline"):
        d0  = cm("proto_only", 0.0) - cm("baseline", 0.0)
        d05 = cm("proto_only", 0.5) - cm("baseline", 0.5)
        sign = "YES" if d0 > 0 else "NO"
        q1 = (
            f"**Preliminary signal: {sign}** — "
            f"proto_only certified acc at r=0.0: "
            f"{cm('proto_only', 0.0):.1f}±{cs('proto_only', 0.0):.1f}% "
            f"vs baseline {cm('baseline', 0.0):.1f}±{cs('baseline', 0.0):.1f}% "
            f"(Δ{d0:+.1f}pp). "
            f"At r=0.5: Δ{d05:+.1f}pp. "
            "Interpret cautiously — pilot scale, random init."
        )
    else:
        q1 = "proto_only or baseline not available."

    # Q2
    if has("proto_margin") and has("baseline"):
        d0  = cm("proto_margin", 0.0) - cm("baseline", 0.0)
        d05 = cm("proto_margin", 0.5) - cm("baseline", 0.5)
        sign = "YES" if d0 > 0 else "NO"
        q2 = (
            f"**Preliminary signal: {sign}** — "
            f"proto_margin at r=0.0: "
            f"{cm('proto_margin', 0.0):.1f}±{cs('proto_margin', 0.0):.1f}% "
            f"vs baseline {cm('baseline', 0.0):.1f}±{cs('baseline', 0.0):.1f}% "
            f"(Δ{d0:+.1f}pp). At r=0.5: Δ{d05:+.1f}pp."
        )
    else:
        q2 = "proto_margin or baseline not available."

    # Q3
    if has("proto_margin") and has("proto_only"):
        d0  = cm("proto_margin", 0.0) - cm("proto_only", 0.0)
        d05 = cm("proto_margin", 0.5) - cm("proto_only", 0.5)
        sign = "YES" if d0 > 0 else "NO"
        q3 = (
            f"**Preliminary signal: {sign}** — "
            f"At r=0.0: proto_margin {cm('proto_margin', 0.0):.1f}% vs "
            f"proto_only {cm('proto_only', 0.0):.1f}% (Δ{d0:+.1f}pp). "
            f"At r=0.5: Δ{d05:+.1f}pp."
        )
    else:
        q3 = "proto_margin or proto_only not available."

    # Q4
    prior_noisy = {"baseline": 19.5, "proto_only": 26.9, "proto_margin": 26.9,
                   "margin_only": 18.8}
    q4_lines = []
    for c in ["baseline", "proto_only", "proto_margin", "margin_only"]:
        if not has(c): continue
        q4_lines.append(
            f"  - {c}: prior single-pass noisy acc (σ=0.5) = {prior_noisy.get(c,'?')}% | "
            f"smoothed acc (N0={N0} votes) = {sm(c):.1f}±{ss(c):.1f}%"
        )
    q4 = (
        "Prior noisy accuracy used a single forward pass per sample; "
        f"smoothed accuracy here uses N0={N0} majority votes. "
        "Smoothed accuracy should be close to, but can differ from, single-pass noisy accuracy "
        "because majority voting reduces variance. Qualitative ordering should agree if the "
        "training effect is real.\n\n"
        + "\n".join(q4_lines)
    )

    # Q5
    collapse = {}
    for c in ["baseline", "proto_only", "proto_margin", "margin_only"]:
        if not has(c): continue
        for r in RADIUS_GRID:
            if cm(c, r) < 5.0:
                collapse[c] = r
                break
        else:
            collapse[c] = f">{max(RADIUS_GRID)}"
    q5_parts = [
        f"{c}: first <5% at r={collapse[c]}"
        for c in ["baseline", "proto_only", "proto_margin", "margin_only"]
        if c in collapse
    ]
    q5 = (
        "Certified accuracy first drops below 5% at:\n\n  "
        + "\n  ".join(q5_parts) +
        "\n\nInterpret cautiously — high variance at pilot scale (N=1000, 200 samples)."
    )

    # Q6
    q6_lines = []
    for c in ["baseline", "proto_only", "proto_margin", "margin_only"]:
        if not has(c): continue
        ab = agg[c]["n_abstain_mean"]
        lr = agg[c]["n_low_radius_mean"]
        q6_lines.append(
            f"  - **{c}**: abstain = {ab:.1f} / {N_CERT} per seed; "
            f"certified but radius < {RADIUS_GRID[1]} = {lr:.1f} / {N_CERT} per seed"
        )
    q6 = (
        "\n".join(q6_lines) +
        "\n\n*Abstain* = pA_lower ≤ 0.5 (N noisy passes do not provide enough evidence to certify). "
        "*Low-radius* = certified but certified_radius < 0.25 (certified at r=0 in the grid but not r=0.25)."
    )

    # Q7
    n_models = len(conditions_done) * len(SEEDS)
    per_model = total_elapsed / max(n_models, 1)
    q7 = (
        f"Total: {total_elapsed/60:.1f} min for {n_models} models "
        f"({len(conditions_done)} conditions × {len(SEEDS)} seeds, "
        f"including re-training where checkpoints were absent).\n"
        f"~{per_model:.0f}s per (condition, seed) = training ({EPOCHS} epochs) + "
        f"certifying {N_CERT} samples × (N0={N0} + N={N}) noisy passes / "
        f"batch_size={CBATCH}."
    )

    # Q8
    q8 = (
        "1. **Randomly initialised model** — no official rRCM checkpoint. "
        "Absolute certified accuracy figures are near-chance; only relative comparisons are meaningful.\n"
        "2. **Not paper-level certified robustness** — formal certification (Cohen et al. 2019) requires a "
        "dedicated smoothed classifier trained with Gaussian data augmentation at the certification σ. "
        "This pilot reuses the training noise_aug mechanism, which is not the same setup.\n"
        "3. **N=1000 is small** — Clopper-Pearson bounds are conservative (loose). "
        "For reference, Cohen et al. use N=100,000. With N=1000, certified radii are underestimates.\n"
        "4. **200 test samples** — all differences ≤5pp should be treated as noise.\n"
        "5. **Re-trained models** — no checkpoints were saved in the prior experiment. "
        "Re-training with the same seed is deterministic up to CUDA non-determinism; "
        "minor weight differences vs the original runs are possible.\n"
        "6. **σ=0.5 only** — results do not characterise behaviour at other noise levels.\n"
        "7. **7 epochs, 2000 samples** — far below any production training regime."
    )

    # Table
    r_hdrs = " | ".join(f"r={r}" for r in RADIUS_GRID)
    sep = "|".join(["---"] * len(RADIUS_GRID))
    tbl_rows = []
    for c in ["baseline", "proto_only", "margin_only", "proto_margin"]:
        if not has(c): continue
        sm_str = f"{sm(c):.1f}±{ss(c):.1f}"
        r_str  = " | ".join(f"{cm(c, r):.1f}±{cs(c, r):.1f}" for r in RADIUS_GRID)
        tbl_rows.append(f"| {c} | {sm_str} | {r_str} |")

    md = f"""# Certification Pilot — Preliminary rRCM-Path Experiment v2

**Tags:** `controlled_debug_certification | randomized_smoothing_style | not_official_rrcm_reproduction | not_paper_level_certified_robustness | preliminary_evidence_only`

> **Caveats**
> - Model is **randomly initialised** — no official rRCM checkpoint.
> - This is **randomized-smoothing-style certification** on a debug-scale model,
>   not paper-level certified robustness.
> - Models were **re-trained** (same seed, same settings) because checkpoints were not saved
>   in the prior experiment. Minor weight differences from CUDA non-determinism are possible.
> - Pilot scale: {N_CERT} samples, N={N}. All certified accuracy figures carry high variance.
> - **Preliminary evidence only.**

---

## Setup

| Param | Value |
|-------|-------|
| Conditions | {conditions_done} |
| Seeds | {SEEDS} |
| Test samples | {N_CERT} (first {N_CERT} of 500-sample test subset) |
| N0 (class selection) | {N0} |
| N (certification) | {N} |
| σ (smoothing) | {SIGMA} |
| α (confidence) | {ALPHA} |
| Radius grid | {RADIUS_GRID} |
| Confidence bound | One-sided Clopper-Pearson: `scipy.stats.beta.ppf(α, k, N−k+1)` |
| Certified at r | smoothed pred correct AND certified_radius ≥ r |

---

## Results (mean±std over {len(SEEDS)} seeds)

| condition | smoothed_acc | {r_hdrs} |
|-----------|-------------|{sep}|
{chr(10).join(tbl_rows)}

*smoothed_acc = fraction of samples where N0-vote prediction matches true label*

---

## Q1 — Does proto_only improve certified accuracy over baseline?

{q1}

---

## Q2 — Does proto_margin improve certified accuracy over baseline?

{q2}

---

## Q3 — Does proto_margin outperform proto_only?

{q3}

---

## Q4 — Do certified accuracy trends match noisy accuracy trends?

{q4}

---

## Q5 — At which radius does certified accuracy collapse?

{q5}

---

## Q6 — How many samples have radius 0 or abstain?

{q6}

---

## Q7 — How expensive was certification?

{q7}

---

## Q8 — What are the limitations?

{q8}
"""
    path.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Randomized-smoothing-style certification pilot (preliminary, not paper-level)"
    )
    parser.add_argument("--data-dir",   default="./data")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS_DEFAULT)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    print(f"Device: {device}")
    print(f"Conditions: {args.conditions}")
    print(f"Pilot: N_CERT={N_CERT}, N0={N0}, N={N}, sigma={SIGMA}, alpha={ALPHA}")
    print("NOTE: models will be re-trained if checkpoints are absent.\n")

    config = load_config(Path("configs/cifar10_margin_aware_rrcm.py"))
    CERT_DIR.mkdir(parents=True, exist_ok=True)

    # Certification test subset — same first N_CERT samples from test set
    test_ds_full = load_data(
        name="cifar10", data_dir=args.data_dir,
        image_size=config.dataset.image_size, mode="test",
        value_range=config.dataset.value_range, augmentation_type="weak",
    )
    cert_ds = Subset(test_ds_full, list(range(N_CERT)))

    all_rows   = []
    t_start    = time.perf_counter()
    conditions_done = [c for c in args.conditions if c in CONDITIONS_CONFIG]

    for condition in conditions_done:
        for seed in SEEDS:
            ckpt = CKPT_DIR / f"{condition}_hpA_seed{seed}.pt"

            if ckpt.exists():
                print(f"\n=== {condition} seed={seed}: loading checkpoint ===")
                model = _load_model(ckpt, config, device)
            else:
                print(f"\n=== {condition} seed={seed}: no checkpoint - re-training ===")
                model = _train_and_save(condition, seed, config, args.data_dir, device, ckpt)

            model.eval()
            print(f"  Certifying {N_CERT} samples (N0={N0}, N={N}, sigma={SIGMA}) ...",
                  flush=True)
            t_cert = time.perf_counter()

            for idx in range(N_CERT):
                img, label = cert_ds[idx]
                img = img.to(device).float()
                cAbar, abstain, radius, correct = certify_one(
                    model, img, int(label), device
                )
                all_rows.append({
                    "condition":        condition,
                    "seed":             seed,
                    "sample_idx":       idx,
                    "true_label":       int(label),
                    "smoothed_pred":    cAbar,
                    "correct":          int(correct),
                    "abstain":          int(abstain),
                    "certified_radius": radius,
                })
                if (idx + 1) % 50 == 0:
                    el = time.perf_counter() - t_cert
                    eta = el / (idx + 1) * (N_CERT - idx - 1)
                    print(f"    [{idx+1}/{N_CERT}]  {el:.0f}s elapsed, "
                          f"~{eta:.0f}s remaining", flush=True)

            cert_s = time.perf_counter() - t_cert
            print(f"  -> done in {cert_s:.1f}s  "
                  f"({cert_s/N_CERT*1000:.0f}ms/sample)", flush=True)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal: {total_elapsed/60:.1f} min")

    # ---- Outputs ------------------------------------------------------------
    _write_csv(CERT_DIR / "per_example_certification.csv", all_rows)
    print(f"Saved: {CERT_DIR / 'per_example_certification.csv'}")

    agg_rows, per_seed_rows = _aggregate(all_rows, conditions_done)
    _write_csv(CERT_DIR / "certification_summary.csv", agg_rows)
    print(f"Saved: {CERT_DIR / 'certification_summary.csv'}")

    _make_plot(agg_rows, CERT_DIR / "certified_accuracy_vs_radius.png")
    print(f"Saved: {CERT_DIR / 'certified_accuracy_vs_radius.png'}")

    _write_summary(
        CERT_DIR / "summary_certification.md",
        agg_rows, per_seed_rows, total_elapsed, conditions_done,
    )
    print(f"Saved: {CERT_DIR / 'summary_certification.md'}")

    # Quick console summary
    print("\n--- Certified accuracy (mean over seeds) ---")
    agg_map = {r["condition"]: r for r in agg_rows}
    header = f"{'condition':<14}" + "".join(f"  r={r:<5}" for r in RADIUS_GRID)
    print(header)
    for c in ["baseline", "proto_only", "margin_only", "proto_margin"]:
        if c not in agg_map: continue
        row = f"{c:<14}"
        for r in RADIUS_GRID:
            row += f"  {agg_map[c][f'cert_{_rk(r)}_mean']:5.1f}%"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
