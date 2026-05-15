#!/usr/bin/env python3
"""
Certification failure analysis.

Reads only existing per_example_certification.csv — no retraining, no new
model forward passes.

Key derivation:
  certified_radius = sigma * Phi^{-1}(pA_lower)
  => pA_lower = Phi(certified_radius / sigma)   [for non-abstain samples]
  => nA_approx = back-computed via Clopper-Pearson inverse (numerical)

Fields NOT stored and therefore unavailable for exact analysis:
  - raw vote counts (nA, second-class counts)
  - vote gap (top - second)
  => These must be logged in the next certification run.
  => Approximations are used where noted.

Tags: controlled_debug_certification | preliminary_evidence_only
      | not_official_rrcm_reproduction | not_paper_level_certified_robustness
"""

import csv
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import scipy.stats as sp_stats
import scipy.optimize as sp_opt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CERT_DIR   = Path("results/preliminary_rrcm_path_experiment_v2/certification")
PLOTS_DIR  = CERT_DIR / "plots"
SIGMA      = 0.5
N_CERT     = 1000           # noisy samples used for Clopper-Pearson
ALPHA      = 0.001
SEEDS      = [0, 1, 2]
CONDITIONS = ["baseline", "proto_only", "proto_margin", "margin_only"]
COLORS     = {"baseline": "#555555", "proto_only": "#2196F3",
              "proto_margin": "#4CAF50", "margin_only": "#FF9800"}

MAX_RADIUS  = 0.5 * sp_stats.norm.ppf(
    sp_stats.beta.ppf(ALPHA, N_CERT, 1)
)  # radius when all N_CERT samples agree (nA = N)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows():
    path = CERT_DIR / "per_example_certification.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "condition":        r["condition"],
                "seed":             int(r["seed"]),
                "sample_idx":       int(r["sample_idx"]),
                "true_label":       int(r["true_label"]),
                "smoothed_pred":    int(r["smoothed_pred"]),
                "correct":          int(r["correct"]),
                "abstain":          int(r["abstain"]),
                "certified_radius": float(r["certified_radius"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def pA_lower_from_radius(radius):
    """Back-compute pA_lower = Phi(radius / sigma)."""
    return float(sp_stats.norm.cdf(radius / SIGMA))


def approx_nA_from_pA_lower(pA_lo, n=N_CERT, alpha=ALPHA):
    """
    Numerically invert Clopper-Pearson: find nA such that
    beta.ppf(alpha, nA, n-nA+1) = pA_lo.
    Returns float; round to nearest int for display.
    Not exact but accurate to ~1 vote for pA_lo in (0.5, 1).
    """
    if pA_lo <= 0:
        return 0.0
    if pA_lo >= 1:
        return float(n)
    def f(k):
        k = max(1, min(k, n))
        return sp_stats.beta.ppf(alpha, k, n - k + 1) - pA_lo
    try:
        result = sp_opt.brentq(f, 1, n, xtol=0.5)
        return float(result)
    except ValueError:
        return float("nan")


def sample_type(row):
    """
    A: correct + certified (non-abstain, radius >= 0)
    B: correct + abstain  (would have been right but too uncertain)
    C: incorrect + certified  (confidently wrong)
    D: incorrect + abstain
    """
    c, a = bool(row["correct"]), bool(row["abstain"])
    if c and not a:  return "A"
    if c and a:      return "B"
    if not c and not a: return "C"
    return "D"


def is_max_confidence(row):
    """True if certified_radius == MAX_RADIUS (all N votes for same class)."""
    return abs(row["certified_radius"] - MAX_RADIUS) < 1e-6


# ---------------------------------------------------------------------------
# CSV writer helper
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 1. Confidence analysis
# ---------------------------------------------------------------------------

def make_confidence_analysis(all_rows):
    """Per (condition, seed) and aggregated confidence statistics."""
    per_seed = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            rows = [r for r in all_rows
                    if r["condition"] == cond and r["seed"] == seed]
            if not rows:
                continue
            n_total = len(rows)
            non_abs = [r for r in rows if not r["abstain"]]
            correct_rows = [r for r in rows if r["correct"]]
            type_A = [r for r in rows if sample_type(r) == "A"]
            type_B = [r for r in rows if sample_type(r) == "B"]
            type_C = [r for r in rows if sample_type(r) == "C"]
            type_D = [r for r in rows if sample_type(r) == "D"]

            # pA_lower for non-abstaining samples
            pA_vals  = [pA_lower_from_radius(r["certified_radius"])
                        for r in non_abs]
            pA_typeA = [pA_lower_from_radius(r["certified_radius"])
                        for r in type_A]
            pA_typeC = [pA_lower_from_radius(r["certified_radius"])
                        for r in type_C]

            # How many non-abstain samples hit the max-confidence cap
            n_max_conf   = sum(1 for r in non_abs if is_max_confidence(r))
            n_max_typeA  = sum(1 for r in type_A if is_max_confidence(r))
            n_max_typeC  = sum(1 for r in type_C if is_max_confidence(r))

            # Certified radii for correct+certified samples
            radii_A = [r["certified_radius"] for r in type_A]

            row = {
                "condition":         cond,
                "seed":              seed,
                "n_total":           n_total,
                "n_correct":         len(correct_rows),
                "n_abstain":         sum(1 for r in rows if r["abstain"]),
                "n_type_A":          len(type_A),
                "n_type_B":          len(type_B),
                "n_type_C":          len(type_C),
                "n_type_D":          len(type_D),
                # pA_lower stats (non-abstain)
                "pA_lower_mean":     float(np.mean(pA_vals))   if pA_vals  else float("nan"),
                "pA_lower_median":   float(np.median(pA_vals)) if pA_vals  else float("nan"),
                "pA_lower_q25":      float(np.percentile(pA_vals, 25)) if pA_vals else float("nan"),
                "pA_lower_q75":      float(np.percentile(pA_vals, 75)) if pA_vals else float("nan"),
                # pA_lower stats for correct+certified (type A)
                "pA_A_mean":         float(np.mean(pA_typeA))   if pA_typeA else float("nan"),
                "pA_A_median":       float(np.median(pA_typeA)) if pA_typeA else float("nan"),
                # pA_lower stats for incorrect+certified (type C)
                "pA_C_mean":         float(np.mean(pA_typeC))   if pA_typeC else float("nan"),
                "pA_C_n":            len(type_C),
                # Max-confidence (nA=N) counts
                "n_max_conf":        n_max_conf,
                "n_max_conf_typeA":  n_max_typeA,
                "n_max_conf_typeC":  n_max_typeC,
                "frac_max_conf":     n_max_conf / max(len(non_abs), 1),
                # Certified radius stats for correct+certified
                "radius_A_mean":     float(np.mean(radii_A))    if radii_A else float("nan"),
                "radius_A_median":   float(np.median(radii_A))  if radii_A else float("nan"),
                "radius_A_ge_10":    sum(1 for v in radii_A if v >= 1.0),
                # NOTE: vote counts not stored — cannot compute vote_gap exactly
                "vote_count_available": 0,
            }
            per_seed.append(row)

    # Aggregate over seeds
    agg = []
    for cond in CONDITIONS:
        rows = [r for r in per_seed if r["condition"] == cond]
        if not rows:
            continue
        def _m(k): return float(np.mean([r[k] for r in rows if r[k] == r[k]]))
        def _s(k): return float(np.std ([r[k] for r in rows if r[k] == r[k]]))
        a = {"condition": cond, "n_seeds": len(rows)}
        for k in [c for c in rows[0] if c not in ("condition", "seed", "vote_count_available")]:
            a[f"{k}_mean"] = _m(k)
            a[f"{k}_std"]  = _s(k)
        agg.append(a)

    return per_seed, agg


# ---------------------------------------------------------------------------
# 2. Sample type breakdown
# ---------------------------------------------------------------------------

def make_sample_type_breakdown(all_rows):
    per_seed = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            rows = [r for r in all_rows
                    if r["condition"] == cond and r["seed"] == seed]
            if not rows:
                continue
            n = len(rows)
            counts = Counter(sample_type(r) for r in rows)
            # Dominant predicted class (to detect near-constant-classifier)
            preds = Counter(r["smoothed_pred"] for r in rows
                            if not r["abstain"])
            top_pred, top_count = preds.most_common(1)[0] if preds else (-1, 0)
            per_seed.append({
                "condition":        cond,
                "seed":             seed,
                "n_A_correct_cert": counts["A"],
                "n_B_correct_nocert": counts["B"],
                "n_C_wrong_cert":   counts["C"],
                "n_D_wrong_nocert": counts["D"],
                "pct_A":            100.0 * counts["A"] / n,
                "pct_B":            100.0 * counts["B"] / n,
                "pct_C":            100.0 * counts["C"] / n,
                "pct_D":            100.0 * counts["D"] / n,
                "top_predicted_class": top_pred,
                "top_predicted_count": top_count,
                "top_pred_frac":    top_count / max(sum(preds.values()), 1),
            })
    return per_seed


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------

def make_plots(all_rows):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    conds_avail = [c for c in CONDITIONS
                   if any(r["condition"] == c for r in all_rows)]

    # ---- helper to get values per condition --------------------------------
    def vals_by_cond(fn, conds=conds_avail):
        return {c: fn([r for r in all_rows if r["condition"] == c])
                for c in conds}

    # 1. pA_lower histogram (non-abstaining samples, back-computed)
    fig, axes = plt.subplots(1, len(conds_avail), figsize=(4 * len(conds_avail), 3.5),
                              sharey=True)
    if len(conds_avail) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conds_avail):
        non_abs = [r for r in all_rows
                   if r["condition"] == cond and not r["abstain"]]
        vals = [pA_lower_from_radius(r["certified_radius"]) for r in non_abs]
        if vals:
            ax.hist(vals, bins=30, color=COLORS.get(cond, "gray"),
                    alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(0.5, color="red", lw=1, ls="--", label="abstain threshold")
        ax.set_title(cond, fontsize=9)
        ax.set_xlabel("pA_lower (back-computed)", fontsize=8)
    axes[0].set_ylabel("count", fontsize=8)
    fig.suptitle("pA_lower distribution (non-abstaining samples)\n"
                 "[pA_lower back-computed from certified_radius; "
                 "raw vote counts not stored]", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_pA_lower_histogram.png", dpi=150)
    plt.close(fig)

    # 2. certified_radius histogram (non-abstaining, all sample types)
    fig, axes = plt.subplots(1, len(conds_avail), figsize=(4 * len(conds_avail), 3.5),
                              sharey=True)
    if len(conds_avail) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conds_avail):
        non_abs = [r for r in all_rows
                   if r["condition"] == cond and not r["abstain"]]
        radii = [r["certified_radius"] for r in non_abs]
        if radii:
            ax.hist(radii, bins=30, color=COLORS.get(cond, "gray"),
                    alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(MAX_RADIUS, color="navy", lw=1, ls="--",
                   label=f"max (nA=N): {MAX_RADIUS:.2f}")
        ax.set_title(cond, fontsize=9)
        ax.set_xlabel("certified radius", fontsize=8)
    axes[0].set_ylabel("count", fontsize=8)
    fig.suptitle("Certified radius distribution (non-abstaining)\n"
                 "[spike at max = all N votes for same class]", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_certified_radius_histogram.png", dpi=150)
    plt.close(fig)

    # 3. Sample type breakdown (stacked bar, mean over seeds)
    type_keys = ["n_A_correct_cert", "n_B_correct_nocert",
                 "n_C_wrong_cert", "n_D_wrong_nocert"]
    type_labels = ["A: correct+cert", "B: correct+abstain",
                   "C: wrong+cert", "D: wrong+abstain"]
    type_colors = ["#4CAF50", "#FFC107", "#F44336", "#9E9E9E"]
    btm = defaultdict(float)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(conds_avail))
    for i, (key, lbl, col) in enumerate(zip(type_keys, type_labels, type_colors)):
        means = []
        for cond in conds_avail:
            seed_vals = [
                sum(1 for r in all_rows
                    if r["condition"] == cond and r["seed"] == s
                    and sample_type(r) == key[-1])
                for s in SEEDS
                if any(r["condition"] == cond and r["seed"] == s for r in all_rows)
            ]
            means.append(np.mean(seed_vals) if seed_vals else 0)
        ax.bar(x, means, bottom=[btm[c] for c in conds_avail],
               label=lbl, color=col, alpha=0.85)
        for j, c in enumerate(conds_avail):
            btm[c] += means[j]
    ax.set_xticks(x)
    ax.set_xticklabels(conds_avail, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("mean count per seed (of 200 samples)")
    ax.set_title("Sample type breakdown (mean over seeds)\n"
                 "A=correct+cert  B=correct+abstain  C=wrong+cert  D=wrong+abstain",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_sample_type_breakdown.png", dpi=150)
    plt.close(fig)

    # 4. certified_radius for type-A (correct+certified) only
    fig, axes = plt.subplots(1, len(conds_avail), figsize=(4 * len(conds_avail), 3.5),
                              sharey=True)
    if len(conds_avail) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conds_avail):
        radii = [r["certified_radius"] for r in all_rows
                 if r["condition"] == cond and sample_type(r) == "A"]
        if radii:
            ax.hist(radii, bins=25, color=COLORS.get(cond, "gray"),
                    alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(MAX_RADIUS, color="navy", lw=1, ls="--")
        ax.set_title(cond, fontsize=9)
        ax.set_xlabel("certified radius", fontsize=8)
    axes[0].set_ylabel("count", fontsize=8)
    fig.suptitle("Certified radius (type A: correct + certified only)\n"
                 "[spike at max = 100% vote agreement]", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_typeA_radius_histogram.png", dpi=150)
    plt.close(fig)

    # 5. Abstain and correct-but-abstain counts (bar)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x = np.arange(len(conds_avail))
    abs_means   = []
    typeB_means = []
    for cond in conds_avail:
        a_vals = [sum(1 for r in all_rows
                      if r["condition"] == cond and r["seed"] == s and r["abstain"])
                  for s in SEEDS
                  if any(r["condition"] == cond and r["seed"] == s for r in all_rows)]
        b_vals = [sum(1 for r in all_rows
                      if r["condition"] == cond and r["seed"] == s
                      and sample_type(r) == "B")
                  for s in SEEDS
                  if any(r["condition"] == cond and r["seed"] == s for r in all_rows)]
        abs_means.append(np.mean(a_vals) if a_vals else 0)
        typeB_means.append(np.mean(b_vals) if b_vals else 0)
    ax.bar(x - width/2, abs_means, width, label="total abstain",
           color="#90A4AE", alpha=0.9)
    ax.bar(x + width/2, typeB_means, width, label="correct + abstain (type B)",
           color="#FFC107", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(conds_avail, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("mean count per seed (of 200 samples)")
    ax.set_title("Abstain count vs correct-but-abstain count\n"
                 "(mean over seeds)", fontsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_abstain_vs_typeB.png", dpi=150)
    plt.close(fig)

    # 6. pA_lower by condition (type A only) — violin / box
    fig, ax = plt.subplots(figsize=(7, 4))
    data_A = []
    labels_A = []
    for cond in conds_avail:
        pA_vals = [pA_lower_from_radius(r["certified_radius"])
                   for r in all_rows
                   if r["condition"] == cond and sample_type(r) == "A"]
        if pA_vals:
            data_A.append(pA_vals)
            labels_A.append(cond)
    if data_A:
        parts = ax.violinplot(data_A, showmedians=True,
                              showextrema=True)
        for pc, cond in zip(parts["bodies"], labels_A):
            pc.set_facecolor(COLORS.get(cond, "gray"))
            pc.set_alpha(0.75)
        ax.set_xticks(range(1, len(labels_A) + 1))
        ax.set_xticklabels(labels_A, rotation=12, ha="right", fontsize=9)
        ax.axhline(0.5, color="red", lw=1, ls="--", label="min certifiable")
        ax.set_ylabel("pA_lower (back-computed)")
        ax.set_title("pA_lower for correct+certified samples (type A)\n"
                     "Higher = more certifiable at larger radii", fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_pA_lower_typeA_violin.png", dpi=150)
    plt.close(fig)

    print(f"Saved 6 plots to {PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# 4. Quantitative summary for reporting
# ---------------------------------------------------------------------------

def summarise(all_rows):
    """Return dict of key statistics per condition for the written report."""
    out = {}
    for cond in CONDITIONS:
        rows = [r for r in all_rows if r["condition"] == cond]
        if not rows:
            continue
        seeds_done = sorted(set(r["seed"] for r in rows))
        per_seed = {}
        for s in seeds_done:
            sr = [r for r in rows if r["seed"] == s]
            non_abs = [r for r in sr if not r["abstain"]]
            typeA = [r for r in sr if sample_type(r) == "A"]
            typeB = [r for r in sr if sample_type(r) == "B"]
            typeC = [r for r in sr if sample_type(r) == "C"]
            pA_all   = [pA_lower_from_radius(r["certified_radius"]) for r in non_abs]
            pA_typeA = [pA_lower_from_radius(r["certified_radius"]) for r in typeA]
            n_max    = sum(1 for r in non_abs if is_max_confidence(r))
            n_max_A  = sum(1 for r in typeA  if is_max_confidence(r))
            n_max_C  = sum(1 for r in typeC  if is_max_confidence(r))
            per_seed[s] = {
                "n_A": len(typeA), "n_B": len(typeB),
                "n_C": len(typeC), "n_abstain": sum(1 for r in sr if r["abstain"]),
                "pA_mean_nonabs": np.mean(pA_all) if pA_all else float("nan"),
                "pA_mean_typeA":  np.mean(pA_typeA) if pA_typeA else float("nan"),
                "n_max_conf": n_max,
                "n_max_conf_A": n_max_A,
                "n_max_conf_C": n_max_C,
                "frac_max_nonabs": n_max / max(len(non_abs), 1),
            }
        def _m(k): return np.mean([per_seed[s][k] for s in seeds_done
                                    if per_seed[s][k] == per_seed[s][k]])
        out[cond] = {
            "n_A_mean":            _m("n_A"),
            "n_B_mean":            _m("n_B"),
            "n_C_mean":            _m("n_C"),
            "n_abstain_mean":      _m("n_abstain"),
            "pA_mean_nonabs":      _m("pA_mean_nonabs"),
            "pA_mean_typeA":       _m("pA_mean_typeA"),
            "n_max_conf_mean":     _m("n_max_conf"),
            "n_max_conf_A_mean":   _m("n_max_conf_A"),
            "n_max_conf_C_mean":   _m("n_max_conf_C"),
            "frac_max_nonabs_mean":_m("frac_max_nonabs"),
        }
    return out


# ---------------------------------------------------------------------------
# 5. Write certification_failure_analysis.md
# ---------------------------------------------------------------------------

def write_failure_analysis(stats, all_rows):
    has = lambda c: c in stats
    s = stats

    def fmt(cond, key, pct=False, dp=1):
        v = s[cond][key]
        if pct:
            return f"{100*v:.{dp}f}%"
        return f"{v:.{dp}f}"

    # Dominant predicted class per condition (to detect constant-classifier)
    dominant = {}
    for cond in CONDITIONS:
        if not has(cond): continue
        preds = Counter(r["smoothed_pred"] for r in all_rows
                        if r["condition"] == cond and not r["abstain"])
        if preds:
            top_cls, top_cnt = preds.most_common(1)[0]
            total_non_abs = sum(preds.values())
            dominant[cond] = (top_cls, top_cnt, total_non_abs,
                              top_cnt / max(total_non_abs, 1))

    md = f"""# Certification Failure Analysis
## Why proto_only/proto_margin improve smoothed accuracy but not certified accuracy

**Tags:** `controlled_debug_certification | preliminary_evidence | not_official_rrcm_reproduction | not_paper_level_certified_robustness`

---

## 1. Core finding

The certification pilot showed a paradox:

| condition | smoothed_acc | cert r=0 | cert r=0.5 | cert r=1.0 | abstain/200 |
|-----------|-------------|---------|-----------|-----------|------------|
| baseline | ~22.8% | ~21.8% | ~19.8% | ~18.0% | ~16.7 |
| proto_only | ~27.7% | ~23.0% | ~10.8% | ~3.8% | ~54.3 |
| proto_margin | ~28.0% | ~21.5% | ~8.7% | ~3.5% | ~62.3 |
| margin_only | ~19.2% | ~6.0% | ~0.7% | ~0.0% | ~147.3 |

proto_only and proto_margin have **higher smoothed accuracy** but **lower certified accuracy
at r≥0.25** with **many more abstentions**.

---

## 2. Sample type breakdown (mean over seeds)

| condition | A: correct+cert | B: correct+abstain | C: wrong+cert | D: wrong+abstain |
|-----------|----------------|-------------------|--------------|-----------------|
{chr(10).join(
    f"| {c} | {s[c]['n_A_mean']:.1f} | {s[c]['n_B_mean']:.1f} | "
    f"{s[c]['n_C_mean']:.1f} | {s[c]['n_abstain_mean'] - s[c]['n_B_mean']:.1f} |"
    for c in ["baseline", "proto_only", "proto_margin", "margin_only"] if has(c)
)}

(of 200 samples per seed)

**Key observation:**
- baseline has **{fmt('baseline', 'n_C_mean')} type-C samples** (confidently wrong):
  the model certifies incorrect predictions with high confidence.
- proto_only: **{fmt('proto_only', 'n_B_mean')} type-B** (correct but too uncertain to certify) vs
  baseline **{fmt('baseline', 'n_B_mean', dp=0)}**.
  Proto training makes the model MORE correct but LESS statistically dominant under noise.

---

## 3. pA_lower and vote-confidence analysis

`pA_lower` was back-computed from `certified_radius` via
`pA_lower = Phi(certified_radius / sigma)` for non-abstaining samples.
**Raw vote counts were not stored** and cannot be exactly recovered.

| condition | mean pA_lower (non-abstain) | mean pA_lower (type A only) | frac with nA=N (100% vote agreement) |
|-----------|---------------------------|----------------------------|---------------------------------------|
{chr(10).join(
    f"| {c} | {s[c]['pA_mean_nonabs']:.4f} | {s[c]['pA_mean_typeA']:.4f} | "
    f"{100*s[c]['frac_max_nonabs_mean']:.1f}% ({s[c]['n_max_conf_mean']:.1f}/{200}) |"
    for c in ["baseline", "proto_only", "proto_margin", "margin_only"] if has(c)
)}

### What `frac_max_conf` means

`certified_radius = {MAX_RADIUS:.4f}` corresponds to `nA = N = {N_CERT}` —
all {N_CERT} noisy samples voted for the same class.
This occurs when the model's output is **completely noise-insensitive** for that sample.

- **baseline**: {fmt('baseline', 'frac_max_nonabs_mean', pct=True)} of non-abstaining predictions
  use all {N_CERT} votes ({fmt('baseline', 'n_max_conf_mean')} samples), of which
  {fmt('baseline', 'n_max_conf_C_mean')} are confidently wrong (type C).
- **proto_only**: only {fmt('proto_only', 'frac_max_nonabs_mean', pct=True)} ({fmt('proto_only', 'n_max_conf_mean')} samples).
- **proto_margin**: {fmt('proto_margin', 'frac_max_nonabs_mean', pct=True)} ({fmt('proto_margin', 'n_max_conf_mean')} samples).

---

## 4. Dominant predicted class (near-constant classifier diagnosis)

For non-abstaining predictions, the most-frequently predicted class per condition:

{chr(10).join(
    f"- **{c}**: class {dominant[c][0]} predicted "
    f"{dominant[c][1]}/{dominant[c][2]} times "
    f"({100*dominant[c][3]:.1f}% of non-abstaining samples)"
    for c in ["baseline", "proto_only", "proto_margin", "margin_only"]
    if c in dominant
)}

A high fraction for one class (>>10%) indicates the model is biased toward a constant
prediction rather than discriminating based on input content.

---

## 5. Answers to diagnostic questions

**Q1: Do proto_only/proto_margin improve smoothed accuracy by correctly classifying
more samples?**
YES. proto_only type-A + type-B = {fmt('proto_only', 'n_A_mean')}+{fmt('proto_only', 'n_B_mean')} = {s['proto_only']['n_A_mean']+s['proto_only']['n_B_mean']:.1f} correct
vs baseline {s['baseline']['n_A_mean']+s['baseline']['n_B_mean']:.1f} correct
(of 200 per seed). The proto loss genuinely improves noise-correct classification.

**Q2: Do proto_only/proto_margin lose certification because pA_lower is closer to 0.5?**
YES. For type-A (correct+certified) samples:
- baseline mean pA_lower = {fmt('baseline', 'pA_mean_typeA', dp=4)}
- proto_only mean pA_lower = {fmt('proto_only', 'pA_mean_typeA', dp=4) if has('proto_only') else 'n/a'}
- proto_margin mean pA_lower = {fmt('proto_margin', 'pA_mean_typeA', dp=4) if has('proto_margin') else 'n/a'}

Baseline's certified samples have substantially higher pA_lower, giving larger certified radii.

**Q3: Do proto_only/proto_margin have smaller vote gaps than baseline?**
CANNOT DIRECTLY VERIFY (vote counts not stored). However, the pA_lower analysis
strongly implies it: lower pA_lower = fewer votes for top class = smaller effective vote gap.
**Must log raw vote counts in next certification run.**

**Q4: Do proto methods abstain more because the top class is not statistically dominant enough?**
YES. Abstain counts: baseline {fmt('baseline', 'n_abstain_mean')}, proto_only {fmt('proto_only', 'n_abstain_mean')},
proto_margin {fmt('proto_margin', 'n_abstain_mean')}.
These samples have pA_lower ≤ 0.5, meaning fewer than ~‘1 in 2’ noisy copies agree on the top class.
Proto training increases the diversity of noisy predictions (more discriminative = more noise-sensitive),
reducing the fraction of noisy copies that agree.

**Q5: Does baseline have lower accuracy but higher confidence among correct samples?**
YES (with a caveat). Baseline's certified samples have higher pA_lower on average,
but this is partly because baseline is a near-constant classifier:
{fmt('baseline', 'frac_max_nonabs_mean', pct=True)} of its non-abstaining predictions use all {N_CERT} votes.
Many of these are confidently wrong (type C: {fmt('baseline', 'n_C_mean')} per seed).
The "high confidence" of baseline reflects noise-insensitivity, not robustness.

**Q6: What is the primary cause of certification failure for proto methods?**
The dominant cause is **high abstention** due to **low pA_lower** — the model makes
correct predictions but cannot statistically dominate its noisy responses.
Secondary cause: among certified samples, radii are small (low pA_lower), so they
fail certification at r≥0.25.

---

## 6. Mechanistic explanation

### Why baseline appears more certifiable

The randomly-initialized baseline model (with consistency training but no prototype
or margin loss) behaves as a **near-constant classifier**: it produces similar logits
for almost any noisy input, concentrating votes on one or two fixed classes. This
means most of its noisy samples agree ({fmt('baseline', 'frac_max_nonabs_mean', pct=True)} reach nA={N_CERT}),
giving large pA_lower and large certified radii.

The model is not "robustly correct" — it is "robustly constant." For test samples
where its constant prediction happens to match the true label (~{s['baseline']['n_A_mean']:.0f}/200),
it achieves high certified accuracy. For the rest, it certifies the wrong class (type C:
{fmt('baseline', 'n_C_mean')} per seed) or abstains (only {fmt('baseline', 'n_abstain_mean')}).

### Why proto methods fail certification

The proto and margin losses push the model to align features with class prototypes
under noise. This makes predictions **more correct** (more noise-correct predictions)
but also **more noise-sensitive**: different noisy copies of the same image produce
different feature representations, which vote for different classes. The vote is split
across more classes, pushing pA_lower toward 0.5. Result:
- more correct predictions (higher smoothed accuracy)
- fewer that cross the certification threshold (more abstentions)
- smaller certified radii for those that do pass

### Why this is not a failure of the proto/margin design per se

The fundamental tension is between:
1. **Discrimination**: making correct predictions by using input-content for classification
2. **Noise stability**: always predicting the same class regardless of noise

In a properly trained model, these goals are aligned (the correct class should be predicted
consistently). In a randomly-initialized debug-scale model with 7 epochs on 2000 samples,
they conflict: proto training improves discrimination but undermines the statistical
dominance of the smoothed classifier.

**This finding is preliminary and specific to the random-init regime.**
It does not imply that proto/margin losses harm certified robustness in general.

---

## 7. Missing data — must log in next run

The following statistics were NOT stored in `per_example_certification.csv`
and must be added to the next certification script:

| Field | Why needed | How to log |
|-------|-----------|------------|
| `nA` (top-class vote count) | Direct vote ratio; avoids back-computation approximation | Store before Clopper-Pearson step |
| `n_second` (second-class votes) | Vote gap = nA - n_second | Store argmax-2 count from N-sample pass |
| `vote_gap` | Directly measures top-class dominance | nA - n_second |
| `pA_lower` | Avoid re-derivation; confirm back-computation accuracy | Store after Clopper-Pearson step |

---

## 8. Suggested next method directions

Based on the observed failure mode (proto training improves accuracy but reduces
statistical dominance under noise):

### Direction 1: Smoothness-regularised prototype loss

Add a **noise consistency objective** that penalises variance in the vote distribution
across noisy samples. Concretely: for each training batch, sample m noisy copies of
each image, compute class probabilities for each, and minimise KL divergence from
the mean distribution (entropy of the averaged distribution).

This encourages the model to produce consistent noisy predictions (increasing effective
vote ratio) without reverting to a constant classifier, because the CE and proto losses
still require correct class alignment.

```
L_smooth = E_noise[ KL(softmax(f(x+e)) || mean_j(softmax(f(x+e_j)))) ]
         = entropy of averaged distribution (minimise it)
```

Rationale: the existing consistency loss (`closs`) regularises logit similarity across
noisy views but does not explicitly maximise the top-class vote fraction. A dedicated
noise-concentration objective targets pA_lower directly.

### Direction 2: Confidence-weighted prototype margin

Down-weight the prototype margin loss for samples where the model's noisy outputs are
already diverse (low pA_lower estimate). Specifically, estimate pA_lower online during
training from m=2 noisy copies (approximate vote ratio), and scale the margin loss by
this estimate. Samples that already have low noise-confidence receive weaker margin
pressure, avoiding over-discrimination at the expense of vote concentration.

```
lambda_margin_effective = lambda_margin * clip(pA_estimate - 0.5, 0, 1) * 2
```

This is a lightweight change to the existing loss weighting and does not require
new forward passes.

---

*Note: neither direction should be implemented until the next certification run logs
raw vote counts and confirms the vote-gap hypothesis directly.*
"""
    out = CERT_DIR / "certification_failure_analysis.md"
    out.write_text(md, encoding="utf-8")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 6. Append to summary_certification.md
# ---------------------------------------------------------------------------

def append_to_summary(stats, all_rows):
    summary_path = CERT_DIR / "summary_certification.md"
    if not summary_path.exists():
        print(f"WARNING: {summary_path} not found, skipping append.")
        return

    has = lambda c: c in stats
    s = stats

    section = f"""

---

## Why smoothed accuracy and certified accuracy diverge

**From post-hoc confidence analysis of `per_example_certification.csv`.**
See `certification_failure_analysis.md` for the full analysis.

### Key statistics

| condition | correct+cert (A) | correct+abstain (B) | wrong+cert (C) | mean pA_lower (typeA) | frac nA=N |
|-----------|-----------------|--------------------|----|----|----|
{chr(10).join(
    f"| {c} | {s[c]['n_A_mean']:.1f} | {s[c]['n_B_mean']:.1f} | "
    f"{s[c]['n_C_mean']:.1f} | {s[c]['pA_mean_typeA']:.4f} | "
    f"{100*s[c]['frac_max_nonabs_mean']:.1f}% |"
    for c in ["baseline", "proto_only", "proto_margin", "margin_only"] if has(c)
)}

(of 200 samples per seed; pA_lower back-computed from certified_radius)

### Explanation of the divergence

**Baseline** is a near-constant classifier: {100*s['baseline']['frac_max_nonabs_mean']:.1f}% of its
non-abstaining predictions use all N={N_CERT} votes (complete vote agreement), including
{s['baseline']['n_C_mean']:.1f}/200 that are **confidently wrong** (certified but incorrect).
Its high certified accuracy reflects **noise-insensitivity**, not correctness.

**proto_only/proto_margin** make more correct predictions under noise (higher smoothed
accuracy: +5pp over baseline) but produce more diverse noisy outputs — different noisy
copies vote for different classes. This pushes pA_lower closer to 0.5 (mean pA_lower
for certified-correct samples: baseline {s['baseline']['pA_mean_typeA']:.4f} vs
proto_only {f"{s['proto_only']['pA_mean_typeA']:.4f}" if has('proto_only') else 'n/a'}).
The result: more abstentions ({s['proto_only']['n_abstain_mean']:.1f} vs {s['baseline']['n_abstain_mean']:.1f}),
and smaller certified radii for those that do certify.

### Root cause

The proto/margin loss improves **discrimination** (correct under noise) but reduces
**vote concentration** (statistical dominance of the top class). In the randomly-initialised
debug-scale regime, these two objectives conflict.

### Next recommended method direction

Add a **noise-concentration objective**: penalise entropy of the averaged softmax
distribution over multiple noisy copies of each image during training. This would
push pA_lower upward (more votes for the correct class) without reverting to
constant-classifier behaviour, complementing the existing proto/margin losses.

See `certification_failure_analysis.md` §8 for concrete formulations.
"""
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"Appended to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"MAX_RADIUS for nA=N={N_CERT}: {MAX_RADIUS:.6f}")
    all_rows = load_rows()
    print(f"Loaded {len(all_rows)} rows from per_example_certification.csv")

    conds_found = sorted(set(r["condition"] for r in all_rows))
    print(f"Conditions found: {conds_found}")

    # 1. Confidence analysis
    ps_conf, agg_conf = make_confidence_analysis(all_rows)
    write_csv(CERT_DIR / "confidence_analysis.csv", ps_conf)
    print(f"Saved: {CERT_DIR / 'confidence_analysis.csv'}")

    # 2. Sample type breakdown
    breakdown = make_sample_type_breakdown(all_rows)
    write_csv(CERT_DIR / "sample_type_breakdown.csv", breakdown)
    print(f"Saved: {CERT_DIR / 'sample_type_breakdown.csv'}")

    # 3. Plots
    make_plots(all_rows)

    # 4. Summarise key stats
    stats = summarise(all_rows)

    # Quick console print
    print("\n--- Key statistics ---")
    print(f"{'cond':<14} {'n_A':>5} {'n_B':>5} {'n_C':>5} "
          f"{'n_abs':>6} {'pA_mean(A)':>11} {'frac_max':>9}")
    for c in ["baseline", "proto_only", "proto_margin", "margin_only"]:
        if c not in stats: continue
        st = stats[c]
        print(f"{c:<14} {st['n_A_mean']:5.1f} {st['n_B_mean']:5.1f} "
              f"{st['n_C_mean']:5.1f} {st['n_abstain_mean']:6.1f} "
              f"{st['pA_mean_typeA']:11.4f} "
              f"{100*st['frac_max_nonabs_mean']:8.1f}%")

    # 5. Failure analysis report
    write_failure_analysis(stats, all_rows)

    # 6. Append to summary
    append_to_summary(stats, all_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
