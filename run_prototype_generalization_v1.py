#!/usr/bin/env python3
"""
Prototype-Centered Generalization Experiment v1

Tags:
  medium_scale_preliminary | prototype_centered | not_official_rrcm_reproduction
  not_paper_level_certified_robustness | controlled_experiment

Goal:  Test whether prototype-centered alignment generalises beyond the tiny debug setting.
Main question: Does proto_only remain beneficial at 5000 training samples / 15 epochs?

Conditions:
  baseline       -- consistency loss only
  proto_only     -- consistency + prototype CE  (lambda_proto=0.1)
  proto_margin_B -- consistency + proto CE + margin hinge  (lambda_proto=0.2, lambda_margin=0.05)

Seeds: 0, 1, 2
Scale: train=5000, test=1000, epochs=15
Certification diagnostic: 200 test samples, N0=100, N=1000, sigma=0.5, alpha=0.001
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import accelerate
import ml_collections
import scipy.stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))

from experiment_utils import build_finetune_model, load_config, set_seed
from rrcm_tune import (
    load_data,
    train_one_epoch_consistency,
    margin_aware_prototype_loss,
    classifier_weight_prototypes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TAG = (
    "medium_scale_preliminary | prototype_centered | "
    "not_official_rrcm_reproduction | not_paper_level_certified_robustness"
)

NOISE_AUG         = 0.5
LBD               = 10.0
ETA               = 0.5
BATCH_SIZE        = 64
PROTOTYPE_MARGIN  = 0.2
PROTOTYPE_TEMP    = 0.1

DEFAULT_N_TRAIN   = 5000
DEFAULT_N_TEST    = 1000
DEFAULT_EPOCHS    = 15
DEFAULT_SEEDS     = [0, 1, 2]

# Certification diagnostic
CERT_N_SAMPLES    = 200
CERT_N0           = 100
CERT_N            = 1000
CERT_SIGMA        = 0.5
CERT_ALPHA        = 0.001
CERT_CBATCH       = 100
CERT_RADIUS_GRID  = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_RADIUS        = float(CERT_SIGMA * sp_stats.norm.ppf(
                        sp_stats.beta.ppf(CERT_ALPHA, CERT_N, 1)))

# Condition definitions
CONDITIONS = {
    "baseline": {
        "use_margin_aware_loss": False,
        "use_proto_loss":        False,
        "use_margin_loss":       False,
        "lambda_proto":          0.0,
        "lambda_margin":         0.0,
    },
    "proto_only": {
        "use_margin_aware_loss": True,
        "use_proto_loss":        True,
        "use_margin_loss":       False,
        "lambda_proto":          0.1,
        "lambda_margin":         0.0,
    },
    "proto_margin_B": {
        "use_margin_aware_loss": True,
        "use_proto_loss":        True,
        "use_margin_loss":       True,
        "lambda_proto":          0.2,
        "lambda_margin":         0.05,
    },
}

CONDITION_ORDER = ["baseline", "proto_only", "proto_margin_B"]
CONDITION_COLORS = {
    "baseline":       "#4c72b0",
    "proto_only":     "#55a868",
    "proto_margin_B": "#8172b2",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_loaders(config, data_dir, n_train, n_test):
    train_ds = load_data(
        name="cifar10", data_dir=data_dir,
        image_size=config.dataset.image_size, mode="train",
        value_range=config.dataset.value_range, augmentation_type="weak",
    )
    test_ds = load_data(
        name="cifar10", data_dir=data_dir,
        image_size=config.dataset.image_size, mode="test",
        value_range=config.dataset.value_range, augmentation_type="weak",
    )
    tr = Subset(train_ds, list(range(min(n_train, len(train_ds)))))
    te = Subset(test_ds,  list(range(min(n_test,  len(test_ds)))))
    train_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader  = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, drop_last=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, noise_aug=0.0):
    model.eval()
    correct = total = 0
    for img, y in loader:
        img, y = img.to(device), y.to(device)
        correct += (model(img, noise_aug=noise_aug).argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total if total else 0.0


@torch.no_grad()
def compute_logit_margin(model, loader, device, noise_aug=0.0):
    model.eval()
    gaps = []
    for img, _ in loader:
        logits = model(img.to(device), noise_aug=noise_aug)
        top2 = logits.topk(2, dim=1).values
        gaps.append((top2[:, 0] - top2[:, 1]).cpu())
    return float(torch.cat(gaps).mean()) if gaps else float("nan")


@torch.no_grad()
def compute_geometry(model, loader, device):
    model.eval()
    protos = classifier_weight_prototypes(model).to(device)
    pm_list, sc_list, swm_list, fc_list = [], [], [], []
    for img, y in loader:
        img, y = img.to(device), y.to(device)
        img2 = torch.cat([img, img], dim=0)
        y2   = torch.cat([y,   y],   dim=0)
        _, feats = model(img2, noise_aug=NOISE_AUG, return_features=True)
        m = margin_aware_prototype_loss(
            feats, y2, protos,
            temperature=PROTOTYPE_TEMP, margin=PROTOTYPE_MARGIN,
            normalize_features=True, use_proto_loss=False, use_margin_loss=False,
        )
        pm_list.append(m["proto_margin"].item())
        sc_list.append(m["sim_correct"].item())
        swm_list.append(m["sim_wrong_max"].item())
        f0, f1 = feats.chunk(2)
        f0n = F.normalize(f0.flatten(1), dim=1)
        f1n = F.normalize(f1.flatten(1), dim=1)
        fc_list.append((f0n - f1n).pow(2).sum(1).sqrt().mean().item())
    return {
        "proto_margin_mean":     float(np.mean(pm_list)),
        "sim_correct_mean":      float(np.mean(sc_list)),
        "sim_wrong_max_mean":    float(np.mean(swm_list)),
        "feat_consistency_mean": float(np.mean(fc_list)),
    }


def measure_inference_cost(model, device, n_repeats=50, batch_size=32,
                            image_size=32, noise_aug=0.0):
    model.eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy, noise_aug=noise_aug)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model(dummy, noise_aug=noise_aug)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    return mean_ms, std_ms, batch_size / (mean_ms / 1000.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_margin_config(cond_name):
    cfg = CONDITIONS[cond_name]
    return ml_collections.ConfigDict({
        "use_margin_aware_loss": cfg["use_margin_aware_loss"],
        "use_proto_loss":        cfg["use_proto_loss"],
        "use_margin_loss":       cfg["use_margin_loss"],
        "lambda_proto":          cfg["lambda_proto"],
        "lambda_margin":         cfg["lambda_margin"],
        "prototype_margin":      PROTOTYPE_MARGIN,
        "prototype_temperature": PROTOTYPE_TEMP,
        "prototype_source":      "classifier_weight",
        "normalize_features":    True,
        "log_geometry_metrics":  True,
    })


def run_one(cond_name, seed, config, data_dir, device, log_dir, ckpt_dir,
            n_train, n_test, epochs):
    set_seed(seed)
    acc = accelerate.Accelerator(cpu=(device.type == "cpu"))

    model = build_finetune_model(config, device=device, tiny_random=True)
    train_loader, test_loader = make_loaders(config, data_dir, n_train, n_test)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters()
         if p.requires_grad and n not in ("linear_head.weight", "linear_head.bias")],
        lr=1e-4, weight_decay=0.0,
    )
    loss_fn      = nn.CrossEntropyLoss()
    margin_cfg   = make_margin_config(cond_name)
    model, train_loader, optimizer = acc.prepare(model, train_loader, optimizer)

    cfg = CONDITIONS[cond_name]
    epoch_rows = []
    log_lines  = []
    header = (f"=== {cond_name} seed={seed} "
              f"lp={cfg['lambda_proto']} lm={cfg['lambda_margin']} "
              f"n_train={n_train} epochs={epochs} ===")
    print(header)
    log_lines.append(header)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        metrics = train_one_epoch_consistency(
            acc, model, train_loader, optimizer, loss_fn,
            mixup_fn=None, noise_aug=NOISE_AUG, lbd=LBD, eta=ETA,
            margin_config=margin_cfg,
        ) or {}
        elapsed = time.perf_counter() - t0

        raw = acc.unwrap_model(model)
        clean_acc   = evaluate(raw, test_loader, device, noise_aug=0.0)
        noisy_acc05 = evaluate(raw, test_loader, device, noise_aug=0.5)
        noisy_acc10 = evaluate(raw, test_loader, device, noise_aug=1.0)
        noisy_acc20 = evaluate(raw, test_loader, device, noise_aug=2.0)
        lmarg       = compute_logit_margin(raw, test_loader, device, noise_aug=0.0)

        def _get(key):
            v = metrics.get(key, float("nan"))
            return float(v) if v == v else float("nan")

        row = {
            "condition":              cond_name,
            "seed":                   seed,
            "epoch":                  epoch,
            "lambda_proto":           cfg["lambda_proto"],
            "lambda_margin":          cfg["lambda_margin"],
            "train_cls_loss":         _get("avg_loss"),
            "train_cons_loss":        _get("avg_closs"),
            "train_proto_loss":       _get("avg_proto_loss"),
            "train_margin_loss":      _get("avg_margin_loss"),
            "train_proto_margin":     _get("avg_proto_margin"),
            "train_sim_correct":      _get("avg_sim_correct"),
            "train_sim_wrong_max":    _get("avg_sim_wrong_max"),
            "train_feat_consistency": _get("avg_feature_consistency"),
            "clean_acc":              clean_acc,
            "noisy_acc_sigma05":      noisy_acc05,
            "noisy_acc_sigma10":      noisy_acc10,
            "noisy_acc_sigma20":      noisy_acc20,
            "logit_margin":           lmarg,
            "epoch_wall_sec":         elapsed,
        }
        epoch_rows.append(row)
        msg = (f"  [ep{epoch:02d}] cls={row['train_cls_loss']:.4f} "
               f"cons={row['train_cons_loss']:.4f} "
               f"pm={row['train_proto_margin']:.4f} | "
               f"clean={clean_acc:.1f}% "
               f"n05={noisy_acc05:.1f}% n10={noisy_acc10:.1f}% n20={noisy_acc20:.1f}% "
               f"lmarg={lmarg:.3f}  ({elapsed:.1f}s)")
        print(msg)
        log_lines.append(msg)

    raw = acc.unwrap_model(model)
    geo = compute_geometry(raw, test_loader, device)
    inf_mean, inf_std, throughput = measure_inference_cost(
        raw, device, n_repeats=30, batch_size=32,
        image_size=config.dataset.image_size,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cond_name}_seed{seed}.pt"
    torch.save(acc.unwrap_model(model).state_dict(), ckpt_path)
    print(f"  -> checkpoint: {ckpt_path}")

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{cond_name}_seed{seed}.txt").write_text(
        "\n".join(log_lines), encoding="utf-8")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "condition":      cond_name,
        "seed":           seed,
        "lambda_proto":   cfg["lambda_proto"],
        "lambda_margin":  cfg["lambda_margin"],
        "epoch_rows":     epoch_rows,
        "geometry":       geo,
        "inf_mean_ms":    inf_mean,
        "inf_std_ms":     inf_std,
        "throughput_ips": throughput,
        "ckpt_path":      ckpt_path,
        "final_clean_acc":    epoch_rows[-1]["clean_acc"],
        "final_noisy05":      epoch_rows[-1]["noisy_acc_sigma05"],
        "final_noisy10":      epoch_rows[-1]["noisy_acc_sigma10"],
        "final_noisy20":      epoch_rows[-1]["noisy_acc_sigma20"],
        "final_logit_margin": epoch_rows[-1]["logit_margin"],
    }


# ---------------------------------------------------------------------------
# Certification diagnostic
# ---------------------------------------------------------------------------

def _cp_lower(k, n, alpha):
    if k == 0:
        return 0.0
    return float(sp_stats.beta.ppf(alpha, k, n - k + 1))


@torch.no_grad()
def _vote(model, x, num, sigma, device):
    counts = np.zeros(10, dtype=np.int64)
    rem = num
    while rem > 0:
        bs   = min(CERT_CBATCH, rem)
        imgs = x.unsqueeze(0).expand(bs, -1, -1, -1).contiguous()
        preds = model(imgs, noise_aug=sigma).argmax(1).cpu().numpy()
        for p in preds:
            counts[p] += 1
        rem -= bs
    return counts


def certify_one(model, x, y, device):
    counts0 = _vote(model, x, CERT_N0, CERT_SIGMA, device)
    cAbar   = int(counts0.argmax())
    correct = int(cAbar == int(y))

    counts  = _vote(model, x, CERT_N, CERT_SIGMA, device)
    nA      = int(counts[cAbar])
    pA_lo   = _cp_lower(nA, CERT_N, CERT_ALPHA)

    sorted_idx = np.argsort(counts)[::-1]
    n_second   = int(counts[sorted_idx[1]]) if len(sorted_idx) > 1 else 0
    vote_gap   = nA - n_second

    if pA_lo > 0.5:
        radius  = float(min(CERT_SIGMA * sp_stats.norm.ppf(pA_lo), MAX_RADIUS + 1e-9))
        abstain = 0
    else:
        radius  = -1.0
        abstain = 1

    return {
        "smoothed_pred":    cAbar,
        "correct":          correct,
        "abstain":          abstain,
        "certified_radius": radius,
        "nA":               nA,
        "n_second":         n_second,
        "vote_gap":         vote_gap,
        "pA_lower":         pA_lo,
    }


def run_certification_diagnostic(model, config, data_dir, device, cond_name, seed):
    test_ds = load_data(
        name="cifar10", data_dir=data_dir,
        image_size=config.dataset.image_size, mode="test",
        value_range=config.dataset.value_range, augmentation_type="weak",
    )
    model.eval()
    rows = []
    for idx in range(CERT_N_SAMPLES):
        x, y = test_ds[idx]
        x = x.to(device)
        diag = certify_one(model, x, int(y), device)
        rows.append({
            "condition":        cond_name,
            "seed":             seed,
            "sample_idx":       idx,
            "true_label":       int(y),
            "smoothed_pred":    diag["smoothed_pred"],
            "correct":          diag["correct"],
            "abstain":          diag["abstain"],
            "certified_radius": diag["certified_radius"],
            "nA":               diag["nA"],
            "n_second":         diag["n_second"],
            "vote_gap":         diag["vote_gap"],
            "pA_lower":         diag["pA_lower"],
        })
        if (idx + 1) % 50 == 0:
            print(f"    certified {idx+1}/{CERT_N_SAMPLES} samples")
    return rows


# ---------------------------------------------------------------------------
# Certification analysis helpers
# ---------------------------------------------------------------------------

def sample_type(row):
    c = bool(int(row["correct"]))
    a = bool(int(row["abstain"]))
    if c and not a:     return "A"
    if c and a:         return "B"
    if not c and not a: return "C"
    return "D"


def is_max_conf(row):
    return abs(float(row["certified_radius"]) - MAX_RADIUS) < 1e-6


def _mstd(vals):
    clean = [v for v in vals if v == v]
    if not clean:
        return float("nan"), 0.0
    return float(np.mean(clean)), float(np.std(clean)) if len(clean) > 1 else 0.0


def summarise_cert(rows, seeds):
    by_cs = {}
    for r in rows:
        by_cs.setdefault((r["condition"], int(r["seed"])), []).append(r)

    agg = {}
    for cond in CONDITION_ORDER:
        seeds_present = [s for s in seeds if (cond, s) in by_cs]
        if not seeds_present:
            continue

        per_seed = {}
        for s in seeds_present:
            sd = by_cs[(cond, s)]
            n  = len(sd)
            types = [sample_type(r) for r in sd]
            non_abs = [r for r in sd if not int(r["abstain"])]

            pA_vals_all = [float(r["pA_lower"]) for r in non_abs]
            pA_vals_A   = [float(r["pA_lower"]) for r in sd if sample_type(r) == "A"]
            vg_vals     = [int(r["vote_gap"])    for r in non_abs]

            cert_at_r = {
                r_thr: sum(
                    1 for r in sd
                    if int(r["correct"]) and not int(r["abstain"])
                       and float(r["certified_radius"]) >= r_thr
                )
                for r_thr in CERT_RADIUS_GRID
            }
            per_seed[s] = {
                "smoothed_acc":   100.0 * sum(int(r["correct"]) for r in sd) / n if n else 0.0,
                "n_abstain":      types.count("B") + types.count("D"),
                "n_A":            types.count("A"),
                "n_B":            types.count("B"),
                "n_C":            types.count("C"),
                "n_D":            types.count("D"),
                "n_max_conf":     sum(1 for r in non_abs if is_max_conf(r)),
                "frac_max_conf":  sum(1 for r in non_abs if is_max_conf(r)) / max(len(non_abs), 1),
                "pA_mean_nonabs": float(np.mean(pA_vals_all)) if pA_vals_all else float("nan"),
                "pA_mean_typeA":  float(np.mean(pA_vals_A))   if pA_vals_A   else float("nan"),
                "vote_gap_mean":  float(np.mean(vg_vals))     if vg_vals     else float("nan"),
                "cert_at_r":      cert_at_r,
            }

        def ms(key):
            return _mstd([per_seed[s][key] for s in seeds_present])

        def ms_r(r_thr):
            return _mstd([per_seed[s]["cert_at_r"][r_thr] for s in seeds_present])

        agg[cond] = {
            "n_seeds":        len(seeds_present),
            "smoothed_acc":   ms("smoothed_acc"),
            "n_abstain":      ms("n_abstain"),
            "n_A":            ms("n_A"),
            "n_B":            ms("n_B"),
            "n_C":            ms("n_C"),
            "n_D":            ms("n_D"),
            "n_max_conf":     ms("n_max_conf"),
            "frac_max_conf":  ms("frac_max_conf"),
            "pA_mean_nonabs": ms("pA_mean_nonabs"),
            "pA_mean_typeA":  ms("pA_mean_typeA"),
            "vote_gap_mean":  ms("vote_gap_mean"),
            "cert_at_r":      {r_thr: ms_r(r_thr) for r_thr in CERT_RADIUS_GRID},
        }
    return agg


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_per_seed_metrics(path, results):
    rows = []
    for r in results:
        rows.append({
            "condition":          r["condition"],
            "lambda_proto":       r["lambda_proto"],
            "lambda_margin":      r["lambda_margin"],
            "seed":               r["seed"],
            "final_clean_acc":    r["final_clean_acc"],
            "final_noisy05":      r["final_noisy05"],
            "final_noisy10":      r["final_noisy10"],
            "final_noisy20":      r["final_noisy20"],
            "final_logit_margin": r["final_logit_margin"],
            **r["geometry"],
            "inf_mean_ms":        r["inf_mean_ms"],
            "throughput_ips":     r["throughput_ips"],
        })
    _write_csv(path, rows)


def write_aggregate_metrics(path, results):
    by_cond = {}
    for r in results:
        by_cond.setdefault(r["condition"], []).append(r)
    rows = []
    for cond in CONDITION_ORDER:
        rs = by_cond.get(cond, [])
        if not rs:
            continue
        def ms(fn):
            return _mstd([fn(r) for r in rs])
        rows.append({
            "condition":             cond,
            "lambda_proto":          rs[0]["lambda_proto"],
            "lambda_margin":         rs[0]["lambda_margin"],
            "n_seeds":               len(rs),
            "clean_acc_mean":        ms(lambda r: r["final_clean_acc"])[0],
            "clean_acc_std":         ms(lambda r: r["final_clean_acc"])[1],
            "noisy05_mean":          ms(lambda r: r["final_noisy05"])[0],
            "noisy05_std":           ms(lambda r: r["final_noisy05"])[1],
            "noisy10_mean":          ms(lambda r: r["final_noisy10"])[0],
            "noisy10_std":           ms(lambda r: r["final_noisy10"])[1],
            "noisy20_mean":          ms(lambda r: r["final_noisy20"])[0],
            "noisy20_std":           ms(lambda r: r["final_noisy20"])[1],
            "logit_margin_mean":     ms(lambda r: r["final_logit_margin"])[0],
            "logit_margin_std":      ms(lambda r: r["final_logit_margin"])[1],
            "proto_margin_mean":     ms(lambda r: r["geometry"]["proto_margin_mean"])[0],
            "proto_margin_std":      ms(lambda r: r["geometry"]["proto_margin_mean"])[1],
            "sim_correct_mean":      ms(lambda r: r["geometry"]["sim_correct_mean"])[0],
            "sim_correct_std":       ms(lambda r: r["geometry"]["sim_correct_mean"])[1],
            "sim_wrong_max_mean":    ms(lambda r: r["geometry"]["sim_wrong_max_mean"])[0],
            "sim_wrong_max_std":     ms(lambda r: r["geometry"]["sim_wrong_max_mean"])[1],
            "feat_consistency_mean": ms(lambda r: r["geometry"]["feat_consistency_mean"])[0],
            "feat_consistency_std":  ms(lambda r: r["geometry"]["feat_consistency_mean"])[1],
            "inf_mean_ms":           float(np.mean([r["inf_mean_ms"]    for r in rs])),
            "throughput_ips":        float(np.mean([r["throughput_ips"] for r in rs])),
        })
    _write_csv(path, rows)


def write_geometry_csv(path, results):
    rows = [
        {"condition": r["condition"], "seed": r["seed"],
         **r["geometry"],
         "lambda_proto": r["lambda_proto"], "lambda_margin": r["lambda_margin"]}
        for r in results
    ]
    _write_csv(path, rows)


def write_inference_cost_csv(path, results):
    rows = [
        {"condition": r["condition"], "seed": r["seed"],
         "inf_mean_ms": r["inf_mean_ms"], "inf_std_ms": r["inf_std_ms"],
         "throughput_ips": r["throughput_ips"]}
        for r in results
    ]
    _write_csv(path, rows)


def write_certification_summary_csv(path, cert_agg):
    rows = []
    for cond in CONDITION_ORDER:
        a = cert_agg.get(cond)
        if a is None:
            continue
        row = {
            "condition":           cond,
            "n_seeds":             a["n_seeds"],
            "smoothed_acc_mean":   a["smoothed_acc"][0],
            "smoothed_acc_std":    a["smoothed_acc"][1],
            "n_abstain_mean":      a["n_abstain"][0],
            "n_abstain_std":       a["n_abstain"][1],
            "n_A_mean":            a["n_A"][0],
            "n_B_mean":            a["n_B"][0],
            "n_C_mean":            a["n_C"][0],
            "n_D_mean":            a["n_D"][0],
            "frac_max_conf_mean":  a["frac_max_conf"][0],
            "pA_mean_nonabs_mean": a["pA_mean_nonabs"][0],
            "pA_mean_typeA_mean":  a["pA_mean_typeA"][0],
            "vote_gap_mean_mean":  a["vote_gap_mean"][0],
        }
        for r_thr in CERT_RADIUS_GRID:
            key = f"r{r_thr:.2f}".replace(".", "")
            row[f"cert_{key}_mean"] = a["cert_at_r"][r_thr][0]
            row[f"cert_{key}_std"]  = a["cert_at_r"][r_thr][1]
        rows.append(row)
    _write_csv(path, rows)


def write_cert_failure_analysis_csv(path, cert_rows):
    rows = [
        {**r, "sample_type": sample_type(r), "is_max_conf": int(is_max_conf(r))}
        for r in cert_rows
    ]
    _write_csv(path, rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(out_dir, results, cert_rows, cert_agg):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    by_cond = {}
    for r in results:
        by_cond.setdefault(r["condition"], []).append(r)

    present  = [c for c in CONDITION_ORDER if c in by_cond]
    clrs     = [CONDITION_COLORS.get(c, "#888") for c in present]
    xlabels  = [c.replace("_", "\n") for c in present]
    x        = np.arange(len(present))
    W        = 0.55

    def bar_vals(fn):
        ms = [_mstd([fn(r) for r in by_cond.get(c, [])])[0] for c in present]
        ss = [_mstd([fn(r) for r in by_cond.get(c, [])])[1] for c in present]
        return ms, ss

    def simple_bar(ax, fn, ylabel, title, zero_line=False):
        means, stds = bar_vals(fn)
        ax.bar(x, means, W, yerr=stds, capsize=5, color=clrs, alpha=0.85, ecolor="#444")
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10, pad=6)
        if zero_line:
            ax.axhline(0, color="black", lw=0.5, ls="--")

    # 1. Clean accuracy by condition
    fig, ax = plt.subplots(figsize=(6, 4))
    simple_bar(ax, lambda r: r["final_clean_acc"], "Accuracy (%)",
               "Clean Accuracy (mean+/-std, 3 seeds)")
    fig.tight_layout()
    fig.savefig(plots_dir / "clean_accuracy_by_condition.png", dpi=120)
    plt.close(fig)

    # 2. Noisy accuracy by sigma
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (fn, lbl) in zip(axes, [
        (lambda r: r["final_noisy05"], "sigma=0.5"),
        (lambda r: r["final_noisy10"], "sigma=1.0"),
        (lambda r: r["final_noisy20"], "sigma=2.0"),
    ]):
        simple_bar(ax, fn, "Accuracy (%)", f"Noisy Acc {lbl}")
    fig.suptitle("Noisy Accuracy (not certified robustness)", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "noisy_accuracy_by_sigma.png", dpi=120)
    plt.close(fig)

    # 3. Prototype margin by condition
    fig, ax = plt.subplots(figsize=(6, 4))
    simple_bar(ax, lambda r: r["geometry"]["proto_margin_mean"],
               "Prototype Margin", "Prototype Margin (mean+/-std)", zero_line=True)
    fig.tight_layout()
    fig.savefig(plots_dir / "prototype_margin_by_condition.png", dpi=120)
    plt.close(fig)

    # 4. sim_correct vs sim_wrong_max (grouped)
    fig, ax = plt.subplots(figsize=(7, 4))
    w2 = 0.32
    xx = np.arange(len(present))
    sc_m,  sc_s  = zip(*[_mstd([r["geometry"]["sim_correct_mean"]   for r in by_cond.get(c, [])]) for c in present])
    swm_m, swm_s = zip(*[_mstd([r["geometry"]["sim_wrong_max_mean"] for r in by_cond.get(c, [])]) for c in present])
    ax.bar(xx - w2/2, sc_m,  w2, yerr=sc_s,  capsize=4, label="sim_correct (up)",   alpha=0.85, color="#4c72b0")
    ax.bar(xx + w2/2, swm_m, w2, yerr=swm_s, capsize=4, label="sim_wrong_max (down)", alpha=0.85, color="#c44e52")
    ax.set_xticks(xx); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Cosine Similarity", fontsize=9)
    ax.set_title("Prototype Similarities (mean+/-std)", fontsize=10)
    ax.legend(fontsize=8); ax.axhline(0, color="black", lw=0.5, ls="--")
    fig.tight_layout()
    fig.savefig(plots_dir / "sim_correct_vs_sim_wrong.png", dpi=120)
    plt.close(fig)

    # 5. Inference latency
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    simple_bar(axes[0], lambda r: r["inf_mean_ms"],    "ms/batch", "Inference Latency (batch=32)")
    simple_bar(axes[1], lambda r: r["throughput_ips"], "img/s",    "Throughput")
    fig.tight_layout()
    fig.savefig(plots_dir / "inference_latency_by_condition.png", dpi=120)
    plt.close(fig)

    # Certification plots require cert data
    if not cert_agg:
        print("  No certification data -- skipping cert plots (6-9)")
        return

    cert_present = [c for c in CONDITION_ORDER if c in cert_agg]
    cert_clrs    = [CONDITION_COLORS.get(c, "#888") for c in cert_present]
    cx           = np.arange(len(cert_present))
    cert_xlabels = [c.replace("_", "\n") for c in cert_present]

    def cert_bar(ax, fn, ylabel, title):
        means = [fn(cert_agg[c])[0] for c in cert_present]
        stds  = [fn(cert_agg[c])[1] for c in cert_present]
        ax.bar(cx, means, W, yerr=stds, capsize=5, color=cert_clrs, alpha=0.85, ecolor="#444")
        ax.set_xticks(cx); ax.set_xticklabels(cert_xlabels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10, pad=6)

    # 6. Certified accuracy vs radius (line plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, clr in zip(cert_present, cert_clrs):
        a     = cert_agg[c]
        ys_m  = [a["cert_at_r"][r][0] for r in CERT_RADIUS_GRID]
        ys_s  = [a["cert_at_r"][r][1] for r in CERT_RADIUS_GRID]
        ax.plot(CERT_RADIUS_GRID, ys_m, marker="o", label=c, color=clr)
        ax.fill_between(
            CERT_RADIUS_GRID,
            [m - s for m, s in zip(ys_m, ys_s)],
            [m + s for m, s in zip(ys_m, ys_s)],
            alpha=0.15, color=clr,
        )
    ax.set_xlabel("Certified radius")
    ax.set_ylabel("Certified samples (of 200)")
    ax.set_title("Certified Accuracy vs Radius\n(smoothing-style diagnostic, not paper-level certified robustness)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "certification_accuracy_vs_radius.png", dpi=120)
    plt.close(fig)

    # 7. Wrong-certified count (type C)
    fig, ax = plt.subplots(figsize=(6, 4))
    cert_bar(ax, lambda a: a["n_C"], "Count (of 200)",
             "Wrong + Certified (Type C)\nlower = fewer confidently-wrong predictions")
    fig.tight_layout()
    fig.savefig(plots_dir / "wrong_certified_count_by_condition.png", dpi=120)
    plt.close(fig)

    # 8. Abstain count
    fig, ax = plt.subplots(figsize=(6, 4))
    cert_bar(ax, lambda a: a["n_abstain"], "Count (of 200)", "Abstain Count")
    fig.tight_layout()
    fig.savefig(plots_dir / "abstain_count_by_condition.png", dpi=120)
    plt.close(fig)

    # 9. pA_lower histogram by condition
    if cert_rows:
        ncols = len(cert_present)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)
        if ncols == 1:
            axes = [axes]
        for ax, c in zip(axes, cert_present):
            non_abs = [r for r in cert_rows
                       if r["condition"] == c and not int(r["abstain"])]
            pA_vals = [float(r["pA_lower"]) for r in non_abs]
            if pA_vals:
                ax.hist(pA_vals, bins=20, range=(0.5, 1.0),
                        color=CONDITION_COLORS.get(c, "#888"), alpha=0.75)
            ax.set_xlabel("pA_lower"); ax.set_title(c.replace("_", "\n"), fontsize=9)
            ax.axvline(0.5, color="red", lw=0.8, ls="--")
        axes[0].set_ylabel("Count")
        fig.suptitle("pA_lower Distribution (non-abstaining, sigma=0.5)\n"
                     "red dashed line = certification threshold", fontsize=10)
        fig.tight_layout()
        fig.savefig(plots_dir / "pA_lower_histogram_by_condition.png", dpi=120)
        plt.close(fig)

    print(f"  Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def write_summary_md(path, results, cert_rows, cert_agg, args, device):
    by_cond = {}
    for r in results:
        by_cond.setdefault(r["condition"], []).append(r)

    def agg(cond):
        rs = by_cond.get(cond, [])
        if not rs:
            return None
        def ms(fn): return _mstd([fn(r) for r in rs])
        return {
            "clean": ms(lambda r: r["final_clean_acc"]),
            "n05":   ms(lambda r: r["final_noisy05"]),
            "n10":   ms(lambda r: r["final_noisy10"]),
            "n20":   ms(lambda r: r["final_noisy20"]),
            "lmarg": ms(lambda r: r["final_logit_margin"]),
            "pm":    ms(lambda r: r["geometry"]["proto_margin_mean"]),
            "sc":    ms(lambda r: r["geometry"]["sim_correct_mean"]),
            "swm":   ms(lambda r: r["geometry"]["sim_wrong_max_mean"]),
            "fc":    ms(lambda r: r["geometry"]["feat_consistency_mean"]),
            "inf":   ms(lambda r: r["inf_mean_ms"]),
            "thru":  ms(lambda r: r["throughput_ips"]),
        }

    b   = agg("baseline")
    po  = agg("proto_only")
    pmb = agg("proto_margin_B")

    def f2(d, key, dp=2):
        if d is None: return "N/A"
        return f"{d[key][0]:.{dp}f}+/-{d[key][1]:.{dp}f}"

    # Q1 proto_only clean acc
    if po and b:
        d = po["clean"][0] - b["clean"][0]
        if d > 1.0:
            q1 = (f"**YES** -- proto_only {po['clean'][0]:.1f}+/-{po['clean'][1]:.1f}% "
                  f"vs baseline {b['clean'][0]:.1f}+/-{b['clean'][1]:.1f}% "
                  f"(delta={d:+.1f}pp). Improvement persists at medium scale.")
        elif d > -0.5:
            q1 = (f"**MARGINAL** -- proto_only {po['clean'][0]:.1f}+/-{po['clean'][1]:.1f}% "
                  f"vs baseline {b['clean'][0]:.1f}+/-{b['clean'][1]:.1f}% "
                  f"(delta={d:+.1f}pp). Within noise range.")
        else:
            q1 = (f"**NO** -- proto_only {po['clean'][0]:.1f}+/-{po['clean'][1]:.1f}% "
                  f"vs baseline {b['clean'][0]:.1f}+/-{b['clean'][1]:.1f}% "
                  f"(delta={d:+.1f}pp). Improvement did not generalise.")
    else:
        q1 = "Insufficient data."

    # Q2 proto_only noisy sigma=0.5
    if po and b:
        d = po["n05"][0] - b["n05"][0]
        if d > 1.0:
            q2 = (f"**YES** -- {po['n05'][0]:.1f}+/-{po['n05'][1]:.1f}% vs "
                  f"baseline {b['n05'][0]:.1f}+/-{b['n05'][1]:.1f}% (delta={d:+.1f}pp).")
        elif d > -0.5:
            q2 = (f"**MARGINAL** -- {po['n05'][0]:.1f}+/-{po['n05'][1]:.1f}% vs "
                  f"baseline {b['n05'][0]:.1f}+/-{b['n05'][1]:.1f}% (delta={d:+.1f}pp). "
                  f"Within noise.")
        else:
            q2 = (f"**NO** -- {po['n05'][0]:.1f}+/-{po['n05'][1]:.1f}% vs "
                  f"baseline {b['n05'][0]:.1f}+/-{b['n05'][1]:.1f}% (delta={d:+.1f}pp).")
    else:
        q2 = "Insufficient data."

    # Q3 sigma=2.0 regression
    if po and b:
        d = po["n20"][0] - b["n20"][0]
        if d < -1.0:
            q3 = (f"**YES, persists** -- proto_only sigma=2.0: {po['n20'][0]:.1f}+/-{po['n20'][1]:.1f}% "
                  f"vs baseline {b['n20'][0]:.1f}+/-{b['n20'][1]:.1f}% (delta={d:+.1f}pp). "
                  f"Regression at large sigma carries over to medium scale.")
        elif d > 0.5:
            q3 = (f"**NO, resolved** -- proto_only sigma=2.0: {po['n20'][0]:.1f}+/-{po['n20'][1]:.1f}% "
                  f"vs baseline {b['n20'][0]:.1f}+/-{b['n20'][1]:.1f}% (delta={d:+.1f}pp). "
                  f"Regression resolved with longer training.")
        else:
            q3 = (f"**UNCLEAR** -- proto_only sigma=2.0: {po['n20'][0]:.1f}+/-{po['n20'][1]:.1f}% "
                  f"vs baseline {b['n20'][0]:.1f}+/-{b['n20'][1]:.1f}% (delta={d:+.1f}pp). Within noise.")
    else:
        q3 = "Insufficient data."

    # Q4 proto_margin_B vs proto_only
    if pmb and po:
        d_clean = pmb["clean"][0] - po["clean"][0]
        d_n05   = pmb["n05"][0]   - po["n05"][0]
        d_pm    = pmb["pm"][0]    - po["pm"][0]
        if d_clean > 1.0 or d_n05 > 1.0:
            q4 = (f"**proto_margin_B outperforms proto_only** -- "
                  f"clean delta={d_clean:+.1f}pp, n05 delta={d_n05:+.1f}pp, "
                  f"proto_margin delta={d_pm:+.4f}. Margin component adds value at this scale.")
        elif d_pm > 0.02 and d_clean > -0.5:
            q4 = (f"**proto_only sufficient for accuracy; proto_margin_B better for geometry** -- "
                  f"clean delta={d_clean:+.1f}pp, n05 delta={d_n05:+.1f}pp, "
                  f"pm delta={d_pm:+.4f}.")
        else:
            q4 = (f"**proto_only sufficient** -- proto_margin_B shows no clear advantage "
                  f"(clean delta={d_clean:+.1f}pp, n05 delta={d_n05:+.1f}pp, "
                  f"pm delta={d_pm:+.4f}).")
    else:
        q4 = "Insufficient data."

    # Q5 geometry mechanism
    if po and b:
        sc_d  = po["sc"][0]  - b["sc"][0]
        swm_d = po["swm"][0] - b["swm"][0]
        if abs(swm_d) > abs(sc_d) and swm_d < 0:
            mech = "**primarily via reduced sim_wrong_max (max wrong-class similarity decreases)**"
        elif sc_d > 0 and sc_d > abs(swm_d):
            mech = "**primarily via increased sim_correct (correct-class similarity increases)**"
        else:
            mech = "**mixed: both sim_correct and sim_wrong_max shift**"
        q5 = (f"{mech}.\n"
              f"sim_correct: baseline {b['sc'][0]:.4f} -> proto_only {po['sc'][0]:.4f} "
              f"(delta={sc_d:+.4f}).\n"
              f"sim_wrong_max: baseline {b['swm'][0]:.4f} -> proto_only {po['swm'][0]:.4f} "
              f"(delta={swm_d:+.4f}).")
    else:
        q5 = "Insufficient data."

    # Q6 near-constant artifact with longer training
    if cert_agg and "baseline" in cert_agg:
        ca        = cert_agg["baseline"]
        frac      = ca["frac_max_conf"][0]
        n_C       = ca["n_C"][0]
        prev_frac = 0.672
        prev_n_C  = 139.7
        if frac < prev_frac - 0.05:
            q6 = (f"**YES, artifact reduced** -- baseline frac_max_conf={100*frac:.1f}% "
                  f"(was {100*prev_frac:.1f}% in debug pilot), "
                  f"type-C (wrong+certified)={n_C:.1f}/200 (was {prev_n_C:.1f}/200). "
                  f"Longer training reduces near-constant behaviour.")
        elif frac > prev_frac - 0.02:
            q6 = (f"**NO, artifact persists** -- baseline frac_max_conf={100*frac:.1f}% "
                  f"(was {100*prev_frac:.1f}% in debug pilot), "
                  f"type-C={n_C:.1f}/200 (was {prev_n_C:.1f}/200). "
                  f"Baseline near-constant classifier behaviour still present.")
        else:
            q6 = (f"**PARTIAL** -- baseline frac_max_conf={100*frac:.1f}% "
                  f"(was {100*prev_frac:.1f}% in debug pilot), "
                  f"type-C={n_C:.1f}/200 (was {prev_n_C:.1f}/200).")
    else:
        q6 = "Certification diagnostic not run or baseline not present."

    # Q7 correct-but-uncertified
    if cert_agg:
        lines = []
        for c in CONDITION_ORDER:
            if c not in cert_agg:
                continue
            a = cert_agg[c]
            lines.append(
                f"  - {c}: A (correct+cert)={a['n_A'][0]:.1f}, "
                f"B (correct+abstain)={a['n_B'][0]:.1f}, "
                f"pA_lower(A)={a['pA_mean_typeA'][0]:.4f}, "
                f"vote_gap={a['vote_gap_mean'][0]:.1f}"
            )
        q7 = "Sample type breakdown (mean over seeds, of 200 samples):\n" + "\n".join(lines)
    else:
        q7 = "Certification diagnostic not run."

    # Q8 inference cost
    if b and po:
        d = po["inf"][0] - b["inf"][0]
        q8 = (f"**Unchanged** -- margin-aware loss is training-only; inference arch is identical. "
              f"baseline={b['inf'][0]:.2f}ms, proto_only={po['inf'][0]:.2f}ms "
              f"(delta={d:+.2f}ms, noise-level). "
              f"Throughput: baseline={b['thru'][0]:.0f}, proto_only={po['thru'][0]:.0f} img/s.")
    else:
        q8 = "Insufficient data."

    # Q9 confidence loss justified?
    q9 = (
        "A confidence-calibrated loss is justified only if ALL of the following hold at this scale:\n\n"
        "1. proto_only/proto_margin_B improve clean or noisy accuracy (Q1, Q2).\n"
        "2. Proto methods have more correct predictions than baseline (type-A + type-B > baseline).\n"
        "3. Proto methods show more abstentions or type-B (correct-but-uncertified) cases (Q7).\n"
        "4. Baseline still shows wrong-certified (type-C) or near-constant artifact (Q6).\n"
        "5. Inference cost remains unchanged (Q8).\n"
        "6. Pattern is consistent across all 3 seeds.\n\n"
        "If all 6 hold: propose 'Prototype-Centered rRCM + Noisy-View Confidence Calibration.'\n"
        "If any fail: re-examine whether the failure mode was a random-init artifact."
    )

    # Q10 stop at proto only?
    q10 = (
        "Stop at prototype alignment only (do not pursue confidence calibration) if:\n\n"
        "1. proto_only clean/noisy accuracy gains disappear at medium scale, OR\n"
        "2. Baseline near-constant artifact disappears (type-C < 20/200), OR\n"
        "3. Proto methods no longer show excess abstentions vs baseline.\n\n"
        "In those cases, the certification failure from the debug pilot was a random-init "
        "artifact and prototype alignment alone is the appropriate method."
    )

    # Main results table
    table_rows = []
    for c in CONDITION_ORDER:
        a = agg(c)
        if a is None:
            continue
        table_rows.append(
            f"| {c} | {a['clean'][0]:.1f}+/-{a['clean'][1]:.1f} "
            f"| {a['n05'][0]:.1f}+/-{a['n05'][1]:.1f} "
            f"| {a['n10'][0]:.1f}+/-{a['n10'][1]:.1f} "
            f"| {a['n20'][0]:.1f}+/-{a['n20'][1]:.1f} "
            f"| {a['pm'][0]:.4f}+/-{a['pm'][1]:.4f} "
            f"| {a['sc'][0]:.4f}+/-{a['sc'][1]:.4f} "
            f"| {a['swm'][0]:.4f}+/-{a['swm'][1]:.4f} |"
        )

    # Cert summary table
    cert_table_rows = []
    if cert_agg:
        for c in CONDITION_ORDER:
            a = cert_agg.get(c)
            if a is None:
                continue
            cr_str = " | ".join(f"{a['cert_at_r'][r][0]:.1f}" for r in CERT_RADIUS_GRID)
            cert_table_rows.append(
                f"| {c} "
                f"| {a['smoothed_acc'][0]:.1f}+/-{a['smoothed_acc'][1]:.1f} "
                f"| {cr_str} "
                f"| {a['n_abstain'][0]:.1f} "
                f"| {a['n_C'][0]:.1f} "
                f"| {100*a['frac_max_conf'][0]:.1f}% |"
            )

    cert_table_block = (
        "| condition | smoothed_acc | r=0 | r=0.25 | r=0.5 | r=0.75 | r=1.0 | abstain | "
        "type-C | frac_max_conf |\n"
        "|-----------|-------------|-----|--------|-------|--------|-------|---------|"
        "--------|---------------|\n"
        + "\n".join(cert_table_rows)
        if cert_table_rows else "*(certification diagnostic not run)*"
    )

    md = f"""# Prototype-Centered Generalization Experiment v1

**Tags:** `{EXPERIMENT_TAG}`

> **Caveats**
> - Model is **randomly initialised** (no official rRCM checkpoint available).
> - This is **not** an official rRCM reproduction.
> - Accuracy figures are **not certified robustness** -- raw noisy-accuracy under Gaussian aug.
> - Certification section is a **diagnostic pilot only** (N=1000, 200 samples) -- not paper-level.
> - Results are from a medium-scale controlled experiment.
> - Conclusions are preliminary.

---

## Setup

| Param | Value |
|-------|-------|
| Experiment | Prototype-Centered Generalization v1 |
| Model | Tiny random rRCMViT (embed_dim=64, depth=1) |
| Task | consistency (original rRCM path) |
| Dataset | CIFAR-10 subset ({args.n_train} train / {args.n_test} test) |
| Seeds | {args.seeds} |
| Epochs | {args.epochs} |
| Batch size | {BATCH_SIZE} |
| noise_aug | {NOISE_AUG} |
| lbd | {LBD} |
| eta | {ETA} |
| prototype_margin | {PROTOTYPE_MARGIN} |
| Conditions | baseline, proto_only (lp=0.1), proto_margin_B (lp=0.2, lm=0.05) |
| Device | {device} |
| Cert N0/N/sigma/alpha | {CERT_N0}/{CERT_N}/{CERT_SIGMA}/{CERT_ALPHA} |

---

## Main Results (mean+/-std over 3 seeds)

| condition | clean_acc | noisy_s0.5 | noisy_s1.0 | noisy_s2.0 | proto_margin | sim_correct | sim_wrong_max |
|-----------|-----------|-----------|-----------|-----------|--------------|-------------|---------------|
{chr(10).join(table_rows)}

---

## Certification Diagnostic (smoothing-style, not paper-level certified robustness)

{cert_table_block}

---

## Q1 -- Does proto_only still improve clean accuracy over baseline?

{q1}

---

## Q2 -- Does proto_only still improve noisy accuracy at sigma=0.5?

{q2}

---

## Q3 -- Does proto_only still regress at sigma=2.0?

{q3}

---

## Q4 -- Does proto_margin_B outperform proto_only, or is proto_only sufficient?

{q4}

---

## Q5 -- Does prototype margin improve via sim_correct increase, sim_wrong_max decrease, or both?

{q5}

---

## Q6 -- Does longer training reduce the baseline near-constant wrong-certified artifact?

{q6}

---

## Q7 -- Do proto methods still have more correct-but-uncertified samples?

{q7}

---

## Q8 -- Does inference cost remain unchanged?

{q8}

---

## Q9 -- What result would justify implementing a confidence-calibrated loss?

{q9}

---

## Q10 -- What result would suggest stopping at prototype alignment only?

{q10}

---

## Output files

| File | Description |
|------|-------------|
| `aggregate_metrics.csv` | Mean+/-std over seeds per condition |
| `per_seed_metrics.csv` | Final-epoch metrics per (condition, seed) |
| `geometry.csv` | Post-training prototype geometry per run |
| `inference_cost.csv` | Wall-clock latency + throughput per run |
| `certification_diagnostic_summary.csv` | Cert pilot aggregated per condition |
| `certification_failure_analysis.csv` | Per-example cert with sample_type, nA, vote_gap |
| `training_logs/` | Per-run console logs |
| `plots/` | 9 figures |
"""
    path.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Load prior results (resume support)
# ---------------------------------------------------------------------------

def load_prior_results(out_dir):
    ps_path  = out_dir / "per_seed_metrics.csv"
    geo_path = out_dir / "geometry.csv"
    inf_path = out_dir / "inference_cost.csv"

    if not ps_path.exists():
        return []

    geo_data = {}
    if geo_path.exists():
        with geo_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                geo_data[(row["condition"], int(row["seed"]))] = {
                    "proto_margin_mean":     float(row["proto_margin_mean"]),
                    "sim_correct_mean":      float(row["sim_correct_mean"]),
                    "sim_wrong_max_mean":    float(row["sim_wrong_max_mean"]),
                    "feat_consistency_mean": float(row["feat_consistency_mean"]),
                }

    inf_data = {}
    if inf_path.exists():
        with inf_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                inf_data[(row["condition"], int(row["seed"]))] = {
                    "inf_mean_ms":    float(row["inf_mean_ms"]),
                    "inf_std_ms":     float(row["inf_std_ms"]),
                    "throughput_ips": float(row["throughput_ips"]),
                }

    results = []
    with ps_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            seed = int(row["seed"])
            k    = (cond, seed)
            geo  = geo_data.get(k, {
                "proto_margin_mean": float("nan"), "sim_correct_mean": float("nan"),
                "sim_wrong_max_mean": float("nan"), "feat_consistency_mean": float("nan"),
            })
            inf  = inf_data.get(k, {
                "inf_mean_ms": float("nan"), "inf_std_ms": float("nan"),
                "throughput_ips": float("nan"),
            })
            cfg = CONDITIONS.get(cond, {"lambda_proto": float(row.get("lambda_proto", 0)),
                                        "lambda_margin": float(row.get("lambda_margin", 0))})
            results.append({
                "condition":      cond,
                "seed":           seed,
                "lambda_proto":   cfg.get("lambda_proto",  float(row.get("lambda_proto",  0))),
                "lambda_margin":  cfg.get("lambda_margin", float(row.get("lambda_margin", 0))),
                "epoch_rows":     [],
                "geometry":       geo,
                "inf_mean_ms":    inf["inf_mean_ms"],
                "inf_std_ms":     inf["inf_std_ms"],
                "throughput_ips": inf["throughput_ips"],
                "ckpt_path":      out_dir / "checkpoints" / f"{cond}_seed{seed}.pt",
                "final_clean_acc":    float(row["final_clean_acc"]),
                "final_noisy05":      float(row["final_noisy05"]),
                "final_noisy10":      float(row["final_noisy10"]),
                "final_noisy20":      float(row["final_noisy20"]),
                "final_logit_margin": float(row["final_logit_margin"]),
            })
    return results


def load_prior_cert_rows(out_dir):
    fa_path = out_dir / "certification_failure_analysis.csv"
    if not fa_path.exists():
        return []
    int_fields   = {"seed", "sample_idx", "true_label", "smoothed_pred",
                    "correct", "abstain", "nA", "n_second", "vote_gap", "is_max_conf"}
    float_fields = {"certified_radius", "pA_lower"}
    rows = []
    with fa_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            converted = {}
            for k, v in row.items():
                if k in int_fields:
                    converted[k] = int(v) if v != "" else 0
                elif k in float_fields:
                    converted[k] = float(v)
                else:
                    converted[k] = v
            rows.append(converted)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/cifar10_margin_aware_rrcm.py")
    parser.add_argument("--data-dir",    default="data/cifar10")
    parser.add_argument("--out-dir",     default="results/prototype_centered_generalization_v1")
    parser.add_argument("--n-train",     type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n-test",      type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--epochs",      type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seeds",       type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--conditions",  nargs="+",
                        choices=list(CONDITIONS.keys()), default=list(CONDITIONS.keys()))
    parser.add_argument("--skip-cert",   action="store_true")
    parser.add_argument("--device",      default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    out_dir  = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    log_dir  = out_dir / "training_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    # Estimate runtime
    n_runs      = len(args.conditions) * len(args.seeds)
    batches_run = (args.n_train // BATCH_SIZE) * args.epochs
    est_sec     = batches_run * 0.015  # rough 15ms/batch (training + eval)
    est_min     = est_sec * n_runs / 60

    print(f"\n{'='*62}")
    print(f"Prototype-Centered Generalization Experiment v1")
    print(f"  Chosen setting: n_train={args.n_train}, n_test={args.n_test}, "
          f"epochs={args.epochs}")
    print(f"  Conditions: {args.conditions}")
    print(f"  Seeds:      {args.seeds}    Device: {device}")
    print(f"  Total runs: {n_runs}   batches/run: {batches_run}")
    print(f"  Est. training time: ~{est_min:.0f} min (very rough)")
    print(f"  Cert diagnostic: {CERT_N_SAMPLES} samples, N0={CERT_N0}, N={CERT_N}, "
          f"sigma={CERT_SIGMA}  skip={args.skip_cert}")
    print(f"  Output: {out_dir}")
    print(f"{'='*62}\n")

    # Load any prior results
    prior_results = load_prior_results(out_dir)
    prior_keys    = {(r["condition"], r["seed"]) for r in prior_results}
    runs_to_do    = [(c, s) for c in args.conditions for s in args.seeds
                     if (c, s) not in prior_keys]
    if prior_keys:
        print(f"  Resuming: {len(prior_keys)} prior run(s) found, "
              f"{len(runs_to_do)} new run(s) to do.")

    new_results = []
    for i, (cond, seed) in enumerate(runs_to_do):
        print(f"\n[{i+1}/{len(runs_to_do)}] {cond}  seed={seed}")
        r = run_one(cond, seed, config, args.data_dir, device,
                    log_dir, ckpt_dir, args.n_train, args.n_test, args.epochs)
        new_results.append(r)

    all_results = prior_results + new_results

    # Write training CSVs
    print(f"\n{'='*62}")
    print("Writing training output files ...")
    write_per_seed_metrics( out_dir / "per_seed_metrics.csv",  all_results)
    write_aggregate_metrics(out_dir / "aggregate_metrics.csv", all_results)
    write_geometry_csv(     out_dir / "geometry.csv",          all_results)
    write_inference_cost_csv(out_dir / "inference_cost.csv",   all_results)

    # Certification diagnostic
    cert_rows = load_prior_cert_rows(out_dir)
    done_cert  = {(r["condition"], r["seed"]) for r in cert_rows}

    if not args.skip_cert:
        print(f"\n--- Certification Diagnostic ---")
        for r in all_results:
            cond, seed = r["condition"], r["seed"]
            if (cond, seed) in done_cert:
                print(f"  [SKIP] {cond} seed={seed} (already certified)")
                continue
            ckpt_path = Path(r.get("ckpt_path", ckpt_dir / f"{cond}_seed{seed}.pt"))
            if not ckpt_path.exists():
                print(f"  [WARN] checkpoint missing for {cond} seed={seed} -- re-training ...")
                r2 = run_one(cond, seed, config, args.data_dir, device,
                             log_dir, ckpt_dir, args.n_train, args.n_test, args.epochs)
                ckpt_path = r2["ckpt_path"]

            print(f"  Certifying {cond} seed={seed} ...")
            mdl = build_finetune_model(config, checkpoint=str(ckpt_path),
                                        device=device, tiny_random=True)
            mdl.eval()
            t0 = time.perf_counter()
            new_cert = run_certification_diagnostic(
                mdl, config, args.data_dir, device, cond, seed)
            elapsed = time.perf_counter() - t0
            cert_rows.extend(new_cert)
            done_cert.add((cond, seed))
            print(f"    done in {elapsed:.1f}s")
            del mdl
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if cert_rows:
            write_cert_failure_analysis_csv(
                out_dir / "certification_failure_analysis.csv", cert_rows)
            cert_agg = summarise_cert(cert_rows, args.seeds)
            write_certification_summary_csv(
                out_dir / "certification_diagnostic_summary.csv", cert_agg)
            print(f"  Certification data written.")
        else:
            cert_agg = {}
    else:
        cert_agg = summarise_cert(cert_rows, args.seeds) if cert_rows else {}

    # Plots + summary
    print("\nGenerating plots ...")
    make_plots(out_dir, all_results, cert_rows, cert_agg)

    print("Writing summary.md ...")
    write_summary_md(out_dir / "summary.md", all_results, cert_rows, cert_agg, args, device)

    print(f"\nAll outputs in: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
