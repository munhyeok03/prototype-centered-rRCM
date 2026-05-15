"""
Prototype-Centered rRCM + Noisy-View Confidence Consistency — v1

New loss: L_conf = mean_i[-log(mean_k p_{y_i,k} + eps) + beta_conf * var_k p_{y_i,k}]
Uses existing m=2 noisy views from the consistency training pass — no extra forward passes.
Training-time only; inference path unchanged.

Conditions:
  baseline                -- consistency loss only
  proto_only              -- + proto CE loss (lp=0.2)
  proto_conf              -- + proto CE + confidence consistency (lp=0.2, lc=0.05)
  proto_conf_weak_margin  -- + proto CE + conf + margin (lp=0.2, lc=0.05, lm=0.02)

Scale: n_train=5000, n_test=1000, epochs=15, seeds=[0,1,2]
Output: results/prototype_confidence_rrcm_v1/

Usage:
  python run_prototype_confidence_rrcm_v1.py                    # full run
  python run_prototype_confidence_rrcm_v1.py --smoke-test-only  # smoke tests only
  python run_prototype_confidence_rrcm_v1.py --skip-cert        # skip certification
"""

import argparse
import csv
import math
import os
import sys
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

sys.path.insert(0, str(Path(__file__).parent))

from experiment_utils import build_finetune_model, load_config, set_seed
from rrcm_tune import (
    load_data,
    train_one_epoch_consistency,
    margin_aware_prototype_loss,
    classifier_weight_prototypes,
)

# ── paths ─────────────────────────────────────────────────────────────────────
OUT = Path("results/prototype_confidence_rrcm_v1")
CONFIG_PATH = "configs/cifar10_finetune.py"
DATA_DIR = "data/cifar10"

# ── constants ─────────────────────────────────────────────────────────────────
NOISE_AUG = 0.5
LBD = 10.0
ETA = 0.5
BATCH_SIZE = 64
PROTOTYPE_MARGIN = 0.2
PROTOTYPE_TEMP = 0.1

DEFAULT_N_TRAIN = 5000
DEFAULT_N_TEST = 1000
DEFAULT_EPOCHS = 15
DEFAULT_SEEDS = [0, 1, 2]

# certification
CERT_N_SAMPLES = 200
CERT_N0 = 100
CERT_N = 1000
CERT_SIGMA = 0.5
CERT_ALPHA = 0.001
CERT_CBATCH = 100
CERT_RADIUS_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_RADIUS = float(CERT_SIGMA * sp_stats.norm.ppf(
    sp_stats.beta.ppf(CERT_ALPHA, CERT_N, 1)))

# confidence diagnostics
CONF_DIAG_M = 4
CONF_DIAG_N = 500

# ── conditions ────────────────────────────────────────────────────────────────
CONDITIONS = {
    "baseline": {
        "use_margin_aware_loss": False,
        "use_proto_loss": False,
        "use_margin_loss": False,
        "lambda_proto": 0.0,
        "lambda_margin": 0.0,
        "use_confidence_consistency_loss": False,
        "lambda_conf": 0.0,
        "beta_conf": 0.1,
        "conf_eps": 1e-6,
    },
    "proto_only": {
        "use_margin_aware_loss": True,
        "use_proto_loss": True,
        "use_margin_loss": False,
        "lambda_proto": 0.2,
        "lambda_margin": 0.0,
        "use_confidence_consistency_loss": False,
        "lambda_conf": 0.0,
        "beta_conf": 0.1,
        "conf_eps": 1e-6,
    },
    "proto_conf": {
        "use_margin_aware_loss": True,
        "use_proto_loss": True,
        "use_margin_loss": False,
        "lambda_proto": 0.2,
        "lambda_margin": 0.0,
        "use_confidence_consistency_loss": True,
        "lambda_conf": 0.05,
        "beta_conf": 0.1,
        "conf_eps": 1e-6,
    },
    "proto_conf_weak_margin": {
        "use_margin_aware_loss": True,
        "use_proto_loss": True,
        "use_margin_loss": True,
        "lambda_proto": 0.2,
        "lambda_margin": 0.02,
        "use_confidence_consistency_loss": True,
        "lambda_conf": 0.05,
        "beta_conf": 0.1,
        "conf_eps": 1e-6,
    },
}
CONDITION_ORDER = ["baseline", "proto_only", "proto_conf", "proto_conf_weak_margin"]
CONDITION_COLORS = {
    "baseline": "#4c72b0",
    "proto_only": "#55a868",
    "proto_conf": "#e8645a",
    "proto_conf_weak_margin": "#c47d27",
}


# ── data ──────────────────────────────────────────────────────────────────────
def make_loaders(config, n_train, n_test):
    train_ds = load_data(
        name="cifar10", data_dir=DATA_DIR,
        image_size=config.dataset.image_size, mode="train",
        value_range=config.dataset.value_range, augmentation_type="weak",
    )
    test_ds = load_data(
        name="cifar10", data_dir=DATA_DIR,
        image_size=config.dataset.image_size, mode="test",
        value_range=config.dataset.value_range, augmentation_type="weak",
    )
    tr = Subset(train_ds, list(range(min(n_train, len(train_ds)))))
    te = Subset(test_ds, list(range(min(n_test, len(test_ds)))))
    train_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, drop_last=False)
    return train_loader, test_loader


# ── margin config ─────────────────────────────────────────────────────────────
def make_margin_config(cond_name):
    cfg = CONDITIONS[cond_name]
    return ml_collections.ConfigDict({
        "use_margin_aware_loss": cfg["use_margin_aware_loss"],
        "use_proto_loss": cfg["use_proto_loss"],
        "use_margin_loss": cfg["use_margin_loss"],
        "lambda_proto": cfg["lambda_proto"],
        "lambda_margin": cfg["lambda_margin"],
        "use_confidence_consistency_loss": cfg["use_confidence_consistency_loss"],
        "lambda_conf": cfg["lambda_conf"],
        "beta_conf": cfg["beta_conf"],
        "conf_eps": cfg["conf_eps"],
        "prototype_margin": PROTOTYPE_MARGIN,
        "prototype_temperature": PROTOTYPE_TEMP,
        "prototype_source": "classifier_weight",
        "normalize_features": True,
        "log_geometry_metrics": True,
    })


# ── evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, noise_aug=0.0):
    model.eval()
    correct = total = 0
    for img, y in loader:
        img, y = img.to(device), y.to(device)
        correct += (model(img, noise_aug=noise_aug).argmax(1) == y).sum().item()
        total += y.size(0)
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
        y2 = torch.cat([y, y], dim=0)
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
        "proto_margin_mean": float(np.mean(pm_list)),
        "sim_correct_mean": float(np.mean(sc_list)),
        "sim_wrong_max_mean": float(np.mean(swm_list)),
        "feat_consistency_mean": float(np.mean(fc_list)),
    }


def measure_inference_cost(model, device, n_repeats=30, batch_size=32,
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
    std_ms = float(np.std(times))
    return mean_ms, std_ms, batch_size / (mean_ms / 1000.0)


# ── confidence diagnostics ────────────────────────────────────────────────────
@torch.no_grad()
def compute_confidence_diagnostics(model, config, device, n_samples=CONF_DIAG_N):
    """Measure true-class confidence consistency across CONF_DIAG_M independent noisy views."""
    model.eval()
    test_ds = load_data(
        name="cifar10", data_dir=DATA_DIR,
        image_size=config.dataset.image_size, mode="test",
        value_range=config.dataset.value_range, augmentation_type="weak",
    )
    loader = DataLoader(
        Subset(test_ds, list(range(min(n_samples, len(test_ds))))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    m = CONF_DIAG_M
    py_means, py_vars, frac_corr = [], [], []
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        B = img.size(0)
        views = torch.stack([img + torch.randn_like(img) * NOISE_AUG
                             for _ in range(m)], dim=0)  # [m, B, C, H, W]
        logits_flat = model(views.view(m * B, *img.shape[1:]),
                            noise_aug=0.0)  # [m*B, C]
        p_flat = F.softmax(logits_flat, dim=1)
        p_view = p_flat.view(m, B, -1)  # [m, B, C]
        p_y = p_view[:, torch.arange(B, device=device), label]  # [m, B]
        mean_py = p_y.mean(0)  # [B]
        var_py = p_y.var(0, unbiased=False)  # [B]
        top_cls = logits_flat.view(m, B, -1).mean(0).argmax(1)  # [B]
        py_means.append(mean_py.cpu())
        py_vars.append(var_py.cpu())
        frac_corr.append((top_cls == label).float().cpu())

    py_means = torch.cat(py_means)
    py_vars = torch.cat(py_vars)
    frac_corr = torch.cat(frac_corr)
    mask_c = frac_corr == 1
    mask_w = frac_corr == 0

    def safe_mean(t, mask):
        sub = t[mask]
        return sub.mean().item() if mask.any() else float("nan")

    return {
        "conf_py_mean": py_means.mean().item(),
        "conf_py_var": py_vars.mean().item(),
        "conf_py_mean_correct": safe_mean(py_means, mask_c),
        "conf_py_mean_wrong": safe_mean(py_means, mask_w),
        "conf_py_var_correct": safe_mean(py_vars, mask_c),
        "conf_py_var_wrong": safe_mean(py_vars, mask_w),
        "frac_correct": frac_corr.mean().item(),
    }


# ── certification ─────────────────────────────────────────────────────────────
def _cp_lower(k, n, alpha):
    if k == 0:
        return 0.0
    return float(sp_stats.beta.ppf(alpha, k, n - k + 1))


@torch.no_grad()
def _vote(model, x, num, sigma, device):
    counts = np.zeros(10, dtype=np.int64)
    rem = num
    while rem > 0:
        bs = min(CERT_CBATCH, rem)
        imgs = x.unsqueeze(0).expand(bs, -1, -1, -1).contiguous()
        preds = model(imgs, noise_aug=sigma).argmax(1).cpu().numpy()
        for p in preds:
            counts[p] += 1
        rem -= bs
    return counts


def certify_one(model, x, y, device):
    counts0 = _vote(model, x, CERT_N0, CERT_SIGMA, device)
    cAbar = int(counts0.argmax())
    correct = int(cAbar == int(y))
    counts = _vote(model, x, CERT_N, CERT_SIGMA, device)
    nA = int(counts[cAbar])
    pA_lo = _cp_lower(nA, CERT_N, CERT_ALPHA)
    sorted_idx = np.argsort(counts)[::-1]
    n_second = int(counts[sorted_idx[1]]) if len(sorted_idx) > 1 else 0
    vote_gap = nA - n_second
    if pA_lo > 0.5:
        radius = float(min(CERT_SIGMA * sp_stats.norm.ppf(pA_lo), MAX_RADIUS + 1e-9))
        abstain = 0
    else:
        radius = -1.0
        abstain = 1
    return {
        "smoothed_pred": cAbar,
        "correct": correct,
        "abstain": abstain,
        "certified_radius": radius,
        "nA": nA,
        "n_second": n_second,
        "vote_gap": vote_gap,
        "pA_lower": pA_lo,
    }


def run_certification_diagnostic(model, config, device, cond_name, seed):
    test_ds = load_data(
        name="cifar10", data_dir=DATA_DIR,
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
            "condition": cond_name,
            "seed": seed,
            "sample_idx": idx,
            "true_label": int(y),
            **diag,
        })
        if (idx + 1) % 50 == 0:
            print(f"    certified {idx+1}/{CERT_N_SAMPLES} samples", flush=True)
    return rows


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
    clean = [v for v in vals if not math.isnan(v)]
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
            n = len(sd)
            types = [sample_type(r) for r in sd]
            n_A = types.count("A")
            n_B = types.count("B")
            n_C = types.count("C")
            n_D = types.count("D")
            n_abstain = sum(1 for r in sd if int(r["abstain"]) == 1)
            smoothed_acc = 100.0 * sum(1 for r in sd if int(r["correct"]) == 1) / n
            frac_max_conf = sum(1 for r in sd if is_max_conf(r)) / n
            pA_type_A = [float(r["pA_lower"]) for r in sd if sample_type(r) == "A"]
            pA_nonabs = [float(r["pA_lower"]) for r in sd if int(r["abstain"]) == 0]
            vote_gaps_A = [int(r["vote_gap"]) for r in sd if sample_type(r) == "A"]
            cert_at = {}
            for rad in CERT_RADIUS_GRID:
                cert_at[rad] = sum(
                    1 for r in sd
                    if int(r["correct"]) == 1 and int(r["abstain"]) == 0
                    and float(r["certified_radius"]) >= rad
                )
            per_seed[s] = {
                "n": n, "n_A": n_A, "n_B": n_B, "n_C": n_C, "n_D": n_D,
                "n_abstain": n_abstain, "smoothed_acc": smoothed_acc,
                "frac_max_conf": frac_max_conf,
                "pA_mean_typeA": float(np.mean(pA_type_A)) if pA_type_A else float("nan"),
                "pA_mean_nonabs": float(np.mean(pA_nonabs)) if pA_nonabs else float("nan"),
                "vote_gap_mean": float(np.mean(vote_gaps_A)) if vote_gaps_A else float("nan"),
                "cert_at": cert_at,
            }
        agg[cond] = per_seed
    return agg


# ── training ──────────────────────────────────────────────────────────────────
def run_one(cond_name, seed, config, device, log_dir, ckpt_dir,
            n_train, n_test, epochs, skip_cert=False):
    set_seed(seed)
    acc = accelerate.Accelerator(cpu=(device.type == "cpu"))

    model = build_finetune_model(config, device=device, tiny_random=True)
    train_loader, test_loader = make_loaders(config, n_train, n_test)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters()
         if p.requires_grad and n not in ("linear_head.weight", "linear_head.bias")],
        lr=1e-4, weight_decay=0.0,
    )
    loss_fn = nn.CrossEntropyLoss()
    margin_cfg = make_margin_config(cond_name)
    model, train_loader, optimizer = acc.prepare(model, train_loader, optimizer)

    cfg = CONDITIONS[cond_name]
    epoch_rows = []
    log_lines = []
    header = (f"=== {cond_name} seed={seed} lp={cfg['lambda_proto']} "
              f"lm={cfg['lambda_margin']} lc={cfg['lambda_conf']} "
              f"n_train={n_train} epochs={epochs} ===")
    print(header, flush=True)
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
        clean_acc = evaluate(raw, test_loader, device, noise_aug=0.0)
        noisy05 = evaluate(raw, test_loader, device, noise_aug=0.5)
        noisy10 = evaluate(raw, test_loader, device, noise_aug=1.0)
        noisy20 = evaluate(raw, test_loader, device, noise_aug=2.0)
        lmarg = compute_logit_margin(raw, test_loader, device, noise_aug=0.0)

        def _get(key):
            v = metrics.get(key, float("nan"))
            return float(v) if v == v else float("nan")

        row = {
            "condition": cond_name, "seed": seed, "epoch": epoch,
            "lambda_proto": cfg["lambda_proto"],
            "lambda_margin": cfg["lambda_margin"],
            "lambda_conf": cfg["lambda_conf"],
            "train_cls_loss": _get("avg_loss"),
            "train_cons_loss": _get("avg_closs"),
            "train_proto_loss": _get("avg_proto_loss"),
            "train_margin_loss": _get("avg_margin_loss"),
            "train_conf_loss": _get("avg_conf_loss"),
            "train_conf_py_mean": _get("avg_conf_py_mean"),
            "train_conf_py_var": _get("avg_conf_py_var"),
            "train_proto_margin": _get("avg_proto_margin"),
            "train_sim_correct": _get("avg_sim_correct"),
            "train_sim_wrong_max": _get("avg_sim_wrong_max"),
            "train_feat_consistency": _get("avg_feature_consistency"),
            "clean_acc": clean_acc, "noisy_acc_sigma05": noisy05,
            "noisy_acc_sigma10": noisy10, "noisy_acc_sigma20": noisy20,
            "logit_margin": lmarg, "epoch_wall_sec": elapsed,
        }
        epoch_rows.append(row)
        msg = (f"  [ep{epoch:02d}] cls={row['train_cls_loss']:.4f} "
               f"cons={row['train_cons_loss']:.4f} "
               f"conf={row['train_conf_loss']:.4f} "
               f"pm={row['train_proto_margin']:.4f} | "
               f"clean={clean_acc:.1f}% n05={noisy05:.1f}% ({elapsed:.1f}s)")
        print(msg, flush=True)
        log_lines.append(msg)

    raw = acc.unwrap_model(model)
    geo = compute_geometry(raw, test_loader, device)
    inf_mean, inf_std, throughput = measure_inference_cost(
        raw, device, n_repeats=30, batch_size=32,
        image_size=config.dataset.image_size,
    )
    conf_diag = compute_confidence_diagnostics(raw, config, device)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cond_name}_seed{seed}.pt"
    torch.save(raw.state_dict(), ckpt_path)
    print(f"  -> checkpoint: {ckpt_path}", flush=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{cond_name}_seed{seed}.txt").write_text(
        "\n".join(log_lines), encoding="utf-8")

    cert_rows = None
    if not skip_cert:
        print(f"  Running certification diagnostic...", flush=True)
        cert_rows = run_certification_diagnostic(raw, config, device, cond_name, seed)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "condition": cond_name,
        "seed": seed,
        "lambda_proto": cfg["lambda_proto"],
        "lambda_margin": cfg["lambda_margin"],
        "lambda_conf": cfg["lambda_conf"],
        "epoch_rows": epoch_rows,
        "geometry": geo,
        "inf_mean_ms": inf_mean,
        "inf_std_ms": inf_std,
        "throughput_ips": throughput,
        "ckpt_path": ckpt_path,
        "conf_diag": conf_diag,
        "cert_rows": cert_rows,
        "final_clean_acc": epoch_rows[-1]["clean_acc"],
        "final_noisy05": epoch_rows[-1]["noisy_acc_sigma05"],
        "final_noisy10": epoch_rows[-1]["noisy_acc_sigma10"],
        "final_noisy20": epoch_rows[-1]["noisy_acc_sigma20"],
        "final_logit_margin": epoch_rows[-1]["logit_margin"],
    }


# ── CSV writers ───────────────────────────────────────────────────────────────
def write_per_seed_metrics(all_results):
    path = OUT / "per_seed_metrics.csv"
    fields = [
        "condition", "lambda_proto", "lambda_margin", "lambda_conf", "seed",
        "final_clean_acc", "final_noisy05", "final_noisy10", "final_noisy20",
        "final_logit_margin", "proto_margin_mean", "sim_correct_mean",
        "sim_wrong_max_mean", "feat_consistency_mean",
        "inf_mean_ms", "throughput_ips",
        "conf_py_mean", "conf_py_var",
        "conf_py_mean_correct", "conf_py_mean_wrong",
        "conf_py_var_correct", "conf_py_var_wrong",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            cd = r["conf_diag"]
            geo = r["geometry"]
            w.writerow({
                "condition": r["condition"],
                "lambda_proto": r["lambda_proto"],
                "lambda_margin": r["lambda_margin"],
                "lambda_conf": r["lambda_conf"],
                "seed": r["seed"],
                "final_clean_acc": r["final_clean_acc"],
                "final_noisy05": r["final_noisy05"],
                "final_noisy10": r["final_noisy10"],
                "final_noisy20": r["final_noisy20"],
                "final_logit_margin": r["final_logit_margin"],
                "proto_margin_mean": geo["proto_margin_mean"],
                "sim_correct_mean": geo["sim_correct_mean"],
                "sim_wrong_max_mean": geo["sim_wrong_max_mean"],
                "feat_consistency_mean": geo["feat_consistency_mean"],
                "inf_mean_ms": r["inf_mean_ms"],
                "throughput_ips": r["throughput_ips"],
                "conf_py_mean": cd["conf_py_mean"],
                "conf_py_var": cd["conf_py_var"],
                "conf_py_mean_correct": cd["conf_py_mean_correct"],
                "conf_py_mean_wrong": cd["conf_py_mean_wrong"],
                "conf_py_var_correct": cd["conf_py_var_correct"],
                "conf_py_var_wrong": cd["conf_py_var_wrong"],
            })


def _agg(vals):
    clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), float(np.std(clean))


def write_aggregate_metrics(all_results):
    path = OUT / "aggregate_metrics.csv"
    conds = list(dict.fromkeys(r["condition"] for r in all_results))
    fields = [
        "condition", "lambda_proto", "lambda_margin", "lambda_conf", "n_seeds",
        "clean_acc_mean", "clean_acc_std",
        "noisy05_mean", "noisy05_std",
        "noisy10_mean", "noisy10_std",
        "noisy20_mean", "noisy20_std",
        "proto_margin_mean", "proto_margin_std",
        "sim_correct_mean", "sim_correct_std",
        "sim_wrong_max_mean", "sim_wrong_max_std",
        "conf_py_mean_mean", "conf_py_mean_std",
        "conf_py_var_mean", "conf_py_var_std",
        "conf_py_mean_correct_mean", "conf_py_mean_wrong_mean",
        "inf_mean_ms", "throughput_ips",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cond in conds:
            cr = [r for r in all_results if r["condition"] == cond]
            cfg = CONDITIONS[cond]
            m, s = _agg([r["final_clean_acc"] for r in cr])
            n05m, n05s = _agg([r["final_noisy05"] for r in cr])
            n10m, n10s = _agg([r["final_noisy10"] for r in cr])
            n20m, n20s = _agg([r["final_noisy20"] for r in cr])
            pm, ps = _agg([r["geometry"]["proto_margin_mean"] for r in cr])
            scm, scs = _agg([r["geometry"]["sim_correct_mean"] for r in cr])
            swm, sws = _agg([r["geometry"]["sim_wrong_max_mean"] for r in cr])
            cpm, cps = _agg([r["conf_diag"]["conf_py_mean"] for r in cr])
            cvm, cvs = _agg([r["conf_diag"]["conf_py_var"] for r in cr])
            cmc = float(np.nanmean([r["conf_diag"]["conf_py_mean_correct"] for r in cr]))
            cmw = float(np.nanmean([r["conf_diag"]["conf_py_mean_wrong"] for r in cr]))
            w.writerow({
                "condition": cond,
                "lambda_proto": cfg["lambda_proto"],
                "lambda_margin": cfg["lambda_margin"],
                "lambda_conf": cfg["lambda_conf"],
                "n_seeds": len(cr),
                "clean_acc_mean": m, "clean_acc_std": s,
                "noisy05_mean": n05m, "noisy05_std": n05s,
                "noisy10_mean": n10m, "noisy10_std": n10s,
                "noisy20_mean": n20m, "noisy20_std": n20s,
                "proto_margin_mean": pm, "proto_margin_std": ps,
                "sim_correct_mean": scm, "sim_correct_std": scs,
                "sim_wrong_max_mean": swm, "sim_wrong_max_std": sws,
                "conf_py_mean_mean": cpm, "conf_py_mean_std": cps,
                "conf_py_var_mean": cvm, "conf_py_var_std": cvs,
                "conf_py_mean_correct_mean": cmc,
                "conf_py_mean_wrong_mean": cmw,
                "inf_mean_ms": float(np.mean([r["inf_mean_ms"] for r in cr])),
                "throughput_ips": float(np.mean([r["throughput_ips"] for r in cr])),
            })


def write_geometry_csv(all_results):
    path = OUT / "geometry.csv"
    fields = ["condition", "seed", "sim_correct_mean", "sim_wrong_max_mean",
              "proto_margin_mean", "feat_consistency_mean"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            geo = r["geometry"]
            w.writerow({"condition": r["condition"], "seed": r["seed"], **geo})


def write_inference_cost_csv(all_results):
    path = OUT / "inference_cost.csv"
    fields = ["condition", "seed", "inf_mean_ms", "inf_std_ms", "throughput_ips"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({
                "condition": r["condition"], "seed": r["seed"],
                "inf_mean_ms": r["inf_mean_ms"],
                "inf_std_ms": r["inf_std_ms"],
                "throughput_ips": r["throughput_ips"],
            })


def write_confidence_diagnostics_csv(all_results):
    path = OUT / "confidence_diagnostics.csv"
    fields = [
        "condition", "seed",
        "conf_py_mean", "conf_py_var",
        "conf_py_mean_correct", "conf_py_mean_wrong",
        "conf_py_var_correct", "conf_py_var_wrong",
        "frac_correct",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            cd = r["conf_diag"]
            w.writerow({"condition": r["condition"], "seed": r["seed"], **cd})


def write_certification_summary_csv(cert_agg, seeds):
    path = OUT / "certification_diagnostic_summary.csv"
    fields = [
        "condition", "n_seeds",
        "smoothed_acc_mean", "smoothed_acc_std",
        "n_abstain_mean", "n_abstain_std",
        "n_A_mean", "n_B_mean", "n_C_mean", "n_D_mean",
        "frac_max_conf_mean", "pA_mean_nonabs_mean",
        "pA_mean_typeA_mean", "vote_gap_mean_mean",
    ]
    for rad in CERT_RADIUS_GRID:
        fields += [f"cert_r{int(rad*100):03d}_mean", f"cert_r{int(rad*100):03d}_std"]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cond in CONDITION_ORDER:
            if cond not in cert_agg:
                continue
            ps = list(cert_agg[cond].values())
            if not ps:
                continue

            def am(key): return float(np.nanmean([p[key] for p in ps]))
            def asd(key): return float(np.nanstd([p[key] for p in ps]))

            row = {
                "condition": cond,
                "n_seeds": len(ps),
                "smoothed_acc_mean": am("smoothed_acc"),
                "smoothed_acc_std": asd("smoothed_acc"),
                "n_abstain_mean": am("n_abstain"),
                "n_abstain_std": asd("n_abstain"),
                "n_A_mean": am("n_A"), "n_B_mean": am("n_B"),
                "n_C_mean": am("n_C"), "n_D_mean": am("n_D"),
                "frac_max_conf_mean": am("frac_max_conf"),
                "pA_mean_nonabs_mean": am("pA_mean_nonabs"),
                "pA_mean_typeA_mean": am("pA_mean_typeA"),
                "vote_gap_mean_mean": am("vote_gap_mean"),
            }
            for rad in CERT_RADIUS_GRID:
                vals = [p["cert_at"][rad] for p in ps]
                row[f"cert_r{int(rad*100):03d}_mean"] = float(np.mean(vals))
                row[f"cert_r{int(rad*100):03d}_std"] = float(np.std(vals))
            w.writerow(row)


def write_cert_failure_analysis_csv(all_results):
    path = OUT / "certification_failure_analysis.csv"
    fields = [
        "condition", "seed", "sample_idx", "true_label",
        "smoothed_pred", "correct", "abstain", "certified_radius",
        "nA", "n_second", "vote_gap", "pA_lower", "stype", "max_conf",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            if not r["cert_rows"]:
                continue
            for cr in r["cert_rows"]:
                row = dict(cr)
                row["stype"] = sample_type(cr)
                row["max_conf"] = int(is_max_conf(cr))
                w.writerow(row)


def write_training_loss_csv(all_results):
    path = OUT / "training_loss_curves.csv"
    all_rows = []
    for r in all_results:
        for row in r["epoch_rows"]:
            all_rows.append(row)
    if not all_rows:
        return
    all_keys = set()
    for row in all_rows:
        all_keys.update(row.keys())
    fixed = ["condition", "seed", "epoch"]
    rest = sorted(all_keys - set(fixed))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fixed + rest, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)


# ── plots ─────────────────────────────────────────────────────────────────────
def make_plots(all_results, cert_agg, seeds):
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    conds = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in all_results)]

    def agg(cond, key):
        vals = [r[key] for r in all_results if r["condition"] == cond]
        return _agg(vals)

    def agg_cd(cond, key):
        vals = [r["conf_diag"][key] for r in all_results if r["condition"] == cond]
        return _agg(vals)

    def agg_geo(cond, key):
        vals = [r["geometry"][key] for r in all_results if r["condition"] == cond]
        return _agg(vals)

    def c(cond):
        return CONDITION_COLORS.get(cond, "#888888")

    # Plot 1: clean accuracy
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cond in enumerate(conds):
        m, s = agg(cond, "final_clean_acc")
        ax.bar(i, m, yerr=s, color=c(cond), capsize=5, alpha=0.85, label=cond)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("Clean Accuracy (%)"); ax.set_title("Clean Accuracy by Condition")
    plt.tight_layout(); plt.savefig(plot_dir / "01_clean_accuracy.png", dpi=100); plt.close()

    # Plot 2: noisy accuracy curve
    fig, ax = plt.subplots(figsize=(7, 4))
    noise_keys = ["final_clean_acc", "final_noisy05", "final_noisy10", "final_noisy20"]
    noise_labs = ["0.0", "0.5", "1.0", "2.0"]
    for cond in conds:
        means = [agg(cond, k)[0] for k in noise_keys]
        stds = [agg(cond, k)[1] for k in noise_keys]
        ax.errorbar(range(4), means, yerr=stds, label=cond, marker="o", color=c(cond))
    ax.set_xticks(range(4)); ax.set_xticklabels(noise_labs)
    ax.set_xlabel("Noise σ"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Noise Level"); ax.legend()
    plt.tight_layout(); plt.savefig(plot_dir / "02_noisy_accuracy.png", dpi=100); plt.close()

    # Plot 3: prototype margin
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cond in enumerate(conds):
        m, s = agg_geo(cond, "proto_margin_mean")
        ax.bar(i, m, yerr=s, color=c(cond), capsize=5, alpha=0.85)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("Proto Margin"); ax.set_title("Prototype Margin by Condition")
    plt.tight_layout(); plt.savefig(plot_dir / "03_proto_margin.png", dpi=100); plt.close()

    # Plot 4: sim_correct vs sim_wrong_max
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(conds)); bw = 0.35
    sc_m = [agg_geo(cnd, "sim_correct_mean")[0] for cnd in conds]
    sc_s = [agg_geo(cnd, "sim_correct_mean")[1] for cnd in conds]
    sw_m = [agg_geo(cnd, "sim_wrong_max_mean")[0] for cnd in conds]
    sw_s = [agg_geo(cnd, "sim_wrong_max_mean")[1] for cnd in conds]
    ax.bar(x - bw/2, sc_m, bw, yerr=sc_s, label="sim_correct", alpha=0.85, capsize=5)
    ax.bar(x + bw/2, sw_m, bw, yerr=sw_s, label="sim_wrong_max", alpha=0.85, capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_title("Prototype Similarity"); ax.legend()
    plt.tight_layout(); plt.savefig(plot_dir / "04_similarity_breakdown.png", dpi=100); plt.close()

    # Plot 5: conf_py_mean (overall, correct, wrong)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(conds)); bw = 0.25
    for i, (key, label) in enumerate([
        ("conf_py_mean", "overall"),
        ("conf_py_mean_correct", "correct"),
        ("conf_py_mean_wrong", "wrong"),
    ]):
        means = [agg_cd(cnd, key)[0] for cnd in conds]
        stds = [agg_cd(cnd, key)[1] for cnd in conds]
        ax.bar(x + (i - 1) * bw, means, bw, yerr=stds, label=label, alpha=0.85, capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_title("Mean True-Class Confidence (noisy views)"); ax.legend()
    plt.tight_layout(); plt.savefig(plot_dir / "05_conf_py_mean.png", dpi=100); plt.close()

    # Plot 6: conf_py_var
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (key, label) in enumerate([
        ("conf_py_var", "overall"),
        ("conf_py_var_correct", "correct"),
        ("conf_py_var_wrong", "wrong"),
    ]):
        means = [agg_cd(cnd, key)[0] for cnd in conds]
        stds = [agg_cd(cnd, key)[1] for cnd in conds]
        ax.bar(x + (i - 1) * bw, means, bw, yerr=stds, label=label, alpha=0.85, capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_title("Variance of True-Class Confidence (↓=more consistent)"); ax.legend()
    plt.tight_layout(); plt.savefig(plot_dir / "06_conf_py_var.png", dpi=100); plt.close()

    # Plots 7–11: certification
    if cert_agg:
        def ca(cond, key):
            ps = list(cert_agg.get(cond, {}).values())
            if not ps:
                return float("nan"), float("nan")
            return float(np.nanmean([p[key] for p in ps])), float(np.nanstd([p[key] for p in ps]))

        def cr_at(cond, rad):
            ps = list(cert_agg.get(cond, {}).values())
            if not ps:
                return float("nan"), float("nan")
            vals = [p["cert_at"][rad] for p in ps]
            return float(np.mean(vals)), float(np.std(vals))

        # Plot 7: cert curve
        fig, ax = plt.subplots(figsize=(7, 4))
        for cond in conds:
            means = [cr_at(cond, rad)[0] for rad in CERT_RADIUS_GRID]
            stds = [cr_at(cond, rad)[1] for rad in CERT_RADIUS_GRID]
            if not all(math.isnan(m) for m in means):
                ax.errorbar(CERT_RADIUS_GRID, means, yerr=stds,
                            label=cond, marker="o", color=c(cond))
        ax.set_xlabel("Radius"); ax.set_ylabel("Certified correct (of 200)")
        ax.set_title("Certification Curve"); ax.legend()
        plt.tight_layout(); plt.savefig(plot_dir / "07_certification_curve.png", dpi=100); plt.close()

        # Plot 8: sample types stacked bar
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(conds)); bottom = np.zeros(len(conds))
        type_keys = ["n_A", "n_B", "n_C", "n_D"]
        type_labs = ["A: correct+cert", "B: correct+abstain", "C: wrong+cert", "D: wrong+abstain"]
        type_cols = ["#4CAF50", "#8BC34A", "#F44336", "#FF9800"]
        for tk, tl, tc in zip(type_keys, type_labs, type_cols):
            means = [ca(cnd, tk)[0] for cnd in conds]
            ax.bar(x, means, bottom=bottom, label=tl, color=tc, alpha=0.85)
            bottom += np.array([m if not math.isnan(m) else 0 for m in means])
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20, ha="right")
        ax.set_title("Sample Type Breakdown (200 samples)"); ax.legend(loc="upper right")
        plt.tight_layout(); plt.savefig(plot_dir / "08_sample_types.png", dpi=100); plt.close()

        # Plot 9: abstain count and pA_mean_typeA
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for i, cond in enumerate(conds):
            m, s = ca(cond, "n_abstain")
            if not math.isnan(m):
                ax1.bar(i, m, yerr=s, color=c(cond), capsize=5, alpha=0.85)
            m2, s2 = ca(cond, "pA_mean_typeA")
            if not math.isnan(m2):
                ax2.bar(i, m2, yerr=s2, color=c(cond), capsize=5, alpha=0.85)
        ax1.set_xticks(range(len(conds))); ax1.set_xticklabels(conds, rotation=20, ha="right")
        ax1.set_title("Abstain Count"); ax1.set_ylabel("n abstain")
        ax2.set_xticks(range(len(conds))); ax2.set_xticklabels(conds, rotation=20, ha="right")
        ax2.set_title("pA_lower (type-A)"); ax2.set_ylabel("pA_lower mean")
        plt.tight_layout(); plt.savefig(plot_dir / "09_abstain_pA.png", dpi=100); plt.close()

        # Plot 10: conf_py_mean vs cert @ r=0.5
        fig, ax = plt.subplots(figsize=(6, 5))
        for r in all_results:
            cond = r["condition"]
            ps = cert_agg.get(cond, {})
            if r["seed"] in ps:
                x_val = r["conf_diag"]["conf_py_mean"]
                y_val = ps[r["seed"]]["cert_at"].get(0.5, float("nan"))
                if not math.isnan(y_val):
                    ax.scatter(x_val, y_val, color=c(cond),
                               label=cond if r["seed"] == seeds[0] else "_",
                               s=80, alpha=0.85)
        ax.set_xlabel("Mean true-class confidence (m=4 noisy views)")
        ax.set_ylabel("Certified @ r=0.5")
        ax.set_title("Confidence vs Certification"); ax.legend()
        plt.tight_layout(); plt.savefig(plot_dir / "10_conf_vs_cert.png", dpi=100); plt.close()

    # Plot 11: per-seed conf_py_mean_correct vs conf_py_mean_wrong scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    seen = set()
    for r in all_results:
        cond = r["condition"]
        ax.scatter(
            r["conf_diag"]["conf_py_mean_correct"],
            r["conf_diag"]["conf_py_mean_wrong"],
            color=c(cond),
            label=cond if cond not in seen else "_",
            s=80, alpha=0.85,
        )
        seen.add(cond)
    lim = [0, 1]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("conf_py_mean_correct"); ax.set_ylabel("conf_py_mean_wrong")
    ax.set_title("Confidence Separation: Correct vs Wrong"); ax.legend()
    plt.tight_layout(); plt.savefig(plot_dir / "11_conf_separation.png", dpi=100); plt.close()

    print(f"  Saved plots to {plot_dir}", flush=True)


# ── summary.md ────────────────────────────────────────────────────────────────
def write_summary_md(all_results, cert_agg, seeds, n_train, n_test, n_epochs):
    conds = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in all_results)]

    def agg(cond, key):
        vals = [r[key] for r in all_results if r["condition"] == cond]
        return _agg(vals)

    def agg_cd(cond, key):
        vals = [r["conf_diag"][key] for r in all_results if r["condition"] == cond]
        return _agg(vals)

    def agg_geo(cond, key):
        vals = [r["geometry"][key] for r in all_results if r["condition"] == cond]
        return _agg(vals)

    def ca(cond, key):
        ps = list(cert_agg.get(cond, {}).values())
        if not ps:
            return float("nan"), float("nan")
        return float(np.nanmean([p[key] for p in ps])), float(np.nanstd([p[key] for p in ps]))

    def cr_at(cond, rad):
        ps = list(cert_agg.get(cond, {}).values())
        if not ps:
            return float("nan"), float("nan")
        vals = [p["cert_at"][rad] for p in ps]
        return float(np.mean(vals)), float(np.std(vals))

    def fmt(m, s, dp=2):
        if math.isnan(m):
            return "n/a"
        return f"{m:.{dp}f}+/-{s:.{dp}f}"

    has_cert = bool(cert_agg)

    md = f"""# Prototype-Centered rRCM + Noisy-View Confidence Consistency — v1

**Tags:** `medium_scale_preliminary | prototype_confidence | not_official_rrcm_reproduction | not_paper_level_certified_robustness`

> **Caveats**
> - Model is **randomly initialised** (no official rRCM checkpoint available).
> - This is **not** an official rRCM reproduction.
> - Accuracy figures are **not certified robustness** -- raw noisy-accuracy under Gaussian aug.
> - Certification section is a **diagnostic pilot only** (N={CERT_N}, {CERT_N_SAMPLES} samples) -- not paper-level.
> - Results are from a medium-scale controlled experiment.
> - Conclusions are preliminary.

---

## Setup

| Param | Value |
|-------|-------|
| Experiment | Prototype-Centered rRCM + Noisy-View Confidence Consistency v1 |
| Model | Tiny random rRCMViT (embed_dim=64, depth=1) |
| Task | consistency (original rRCM path) |
| Dataset | CIFAR-10 subset ({n_train} train / {n_test} test) |
| Seeds | {seeds} |
| Epochs | {n_epochs} |
| Batch size | {BATCH_SIZE} |
| noise_aug | {NOISE_AUG} |
| lbd | {LBD} |
| eta | {ETA} |
| Conditions | baseline, proto_only (lp=0.2), proto_conf (lp=0.2, lc=0.05), proto_conf_weak_margin (lp=0.2, lc=0.05, lm=0.02) |
| Confidence loss | L_conf = mean_i[-log(mean_k p_y_i_k + eps) + beta_conf * var_k p_y_i_k] |
| beta_conf | 0.1 |
| conf_eps | 1e-6 |
| Conf diag views | {CONF_DIAG_M} |
| Device | cuda |
| Cert N0/N/sigma/alpha | {CERT_N0}/{CERT_N}/{CERT_SIGMA}/{CERT_ALPHA} |

---

## Main Results (mean+/-std over {len(seeds)} seeds)

| condition | clean_acc | noisy_s0.5 | noisy_s1.0 | noisy_s2.0 | proto_margin |
|-----------|-----------|-----------|-----------|-----------|--------------|
"""
    for cond in conds:
        m, s = agg(cond, "final_clean_acc")
        n05m, n05s = agg(cond, "final_noisy05")
        n10m, n10s = agg(cond, "final_noisy10")
        n20m, n20s = agg(cond, "final_noisy20")
        pm, ps_ = agg_geo(cond, "proto_margin_mean")
        md += f"| {cond} | {fmt(m, s)} | {fmt(n05m, n05s)} | {fmt(n10m, n10s)} | {fmt(n20m, n20s)} | {fmt(pm, ps_, dp=4)} |\n"

    md += "\n---\n\n## Confidence Diagnostics (m=4 noisy views, 500 samples)\n\n"
    md += "| condition | conf_py_mean | conf_py_var | conf_py_mean_correct | conf_py_mean_wrong |\n"
    md += "|-----------|-------------|------------|---------------------|-------------------|\n"
    for cond in conds:
        cpm, cps = agg_cd(cond, "conf_py_mean")
        cvm, cvs = agg_cd(cond, "conf_py_var")
        mcm, mcs = agg_cd(cond, "conf_py_mean_correct")
        mwm, mws = agg_cd(cond, "conf_py_mean_wrong")
        md += f"| {cond} | {fmt(cpm, cps)} | {fmt(cvm, cvs, dp=6)} | {fmt(mcm, mcs)} | {fmt(mwm, mws)} |\n"

    if has_cert:
        md += "\n---\n\n## Certification Diagnostic (smoothing-style, not paper-level certified robustness)\n\n"
        md += "| condition | smoothed_acc | r=0 | r=0.25 | r=0.5 | r=0.75 | r=1.0 | abstain | type-C | frac_max_conf |\n"
        md += "|-----------|-------------|-----|--------|-------|--------|-------|---------|--------|---------------|\n"
        for cond in conds:
            sm, ss = ca(cond, "smoothed_acc")
            abm, abs_ = ca(cond, "n_abstain")
            cm, cs = ca(cond, "n_C")
            fm, _ = ca(cond, "frac_max_conf")
            r0m, _ = cr_at(cond, 0.0)
            r25m, _ = cr_at(cond, 0.25)
            r50m, _ = cr_at(cond, 0.5)
            r75m, _ = cr_at(cond, 0.75)
            r100m, _ = cr_at(cond, 1.0)
            if math.isnan(sm):
                md += f"| {cond} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |\n"
            else:
                md += (f"| {cond} | {sm:.1f}+/-{ss:.1f} | {r0m:.1f} | {r25m:.1f} | {r50m:.1f} | "
                       f"{r75m:.1f} | {r100m:.1f} | {abm:.1f} | {cm:.1f} | {fm:.3f} |\n")

    md += "\n---\n\n"

    # Q1
    po_clean, _ = agg("proto_only", "final_clean_acc") if "proto_only" in conds else (float("nan"), float("nan"))
    pc_clean, _ = agg("proto_conf", "final_clean_acc") if "proto_conf" in conds else (float("nan"), float("nan"))
    d_clean = pc_clean - po_clean
    po_n05, _ = agg("proto_only", "final_noisy05") if "proto_only" in conds else (float("nan"), float("nan"))
    pc_n05, _ = agg("proto_conf", "final_noisy05") if "proto_conf" in conds else (float("nan"), float("nan"))
    d_n05 = pc_n05 - po_n05
    md += f"## Q1 -- Does proto_conf improve clean or noisy accuracy over proto_only?\n\n"
    md += f"proto_conf vs proto_only: clean delta={d_clean:+.1f}pp; noisy05 delta={d_n05:+.1f}pp.\n\n"
    md += "| condition | clean | noisy05 |\n|---|---|---|\n"
    for cond in conds:
        m, s = agg(cond, "final_clean_acc")
        n05m, n05s = agg(cond, "final_noisy05")
        md += f"| {cond} | {fmt(m, s)} | {fmt(n05m, n05s)} |\n"
    md += "\n---\n\n"

    # Q2
    md += "## Q2 -- Does confidence loss increase mean true-class confidence for correct predictions?\n\n"
    md += "conf_py_mean_correct (higher = more confident on correct predictions):\n\n"
    for cond in conds:
        m, s = agg_cd(cond, "conf_py_mean_correct")
        md += f"- {cond}: {fmt(m, s)}\n"
    md += "\n---\n\n"

    # Q3
    md += "## Q3 -- Does confidence loss reduce variance of true-class confidence?\n\n"
    md += "conf_py_var (lower = more consistent predictions across noisy views):\n\n"
    for cond in conds:
        m, s = agg_cd(cond, "conf_py_var")
        md += f"- {cond}: {fmt(m, s, dp=6)}\n"
    md += "\n---\n\n"

    # Q4
    md += "## Q4 -- Does confidence loss increase separation between correct and wrong confidence?\n\n"
    md += "Gap = conf_py_mean_correct - conf_py_mean_wrong:\n\n"
    for cond in conds:
        mc, _ = agg_cd(cond, "conf_py_mean_correct")
        mw, _ = agg_cd(cond, "conf_py_mean_wrong")
        gap = mc - mw if not (math.isnan(mc) or math.isnan(mw)) else float("nan")
        md += f"- {cond}: gap={gap:.4f} (correct={mc:.4f}, wrong={mw:.4f})\n"
    md += "\n---\n\n"

    # Q5
    md += "## Q5 -- Does proto_conf reduce type-C (wrong+certified) count vs proto_only?\n\n"
    if has_cert:
        for cond in conds:
            m, s = ca(cond, "n_C")
            if not math.isnan(m):
                md += f"- {cond}: n_C={m:.1f}+/-{s:.1f}\n"
    else:
        md += "Certification not run.\n"
    md += "\n---\n\n"

    # Q6
    md += "## Q6 -- Does proto_conf increase type-A (correct+certified) count vs proto_only?\n\n"
    if has_cert:
        for cond in conds:
            am, as_ = ca(cond, "n_A")
            bm, bs = ca(cond, "n_B")
            if not math.isnan(am):
                md += f"- {cond}: n_A={am:.1f}+/-{as_:.1f}, n_B={bm:.1f}+/-{bs:.1f}\n"
    else:
        md += "Certification not run.\n"
    md += "\n---\n\n"

    # Q7
    md += "## Q7 -- Does proto_conf improve certified accuracy at r=0.5 vs proto_only?\n\n"
    if has_cert:
        for cond in conds:
            m, s = cr_at(cond, 0.5)
            if not math.isnan(m):
                md += f"- {cond}: cert@r=0.5={m:.1f}+/-{s:.1f}\n"
    else:
        md += "Certification not run.\n"
    md += "\n---\n\n"

    # Q8
    md += "## Q8 -- Does proto_conf_weak_margin add value over proto_conf?\n\n"
    if "proto_conf_weak_margin" in conds and "proto_conf" in conds:
        pcwm_c, _ = agg("proto_conf_weak_margin", "final_clean_acc")
        pc_c, _ = agg("proto_conf", "final_clean_acc")
        d = pcwm_c - pc_c
        md += f"proto_conf_weak_margin vs proto_conf clean delta: {d:+.1f}pp.\n\n"
    else:
        md += "proto_conf_weak_margin not run.\n\n"
    md += "---\n\n"

    # Q9
    md += "## Q9 -- Does inference cost remain unchanged?\n\n"
    md += "Per-condition timing (mean over seeds, ms/batch):\n\n"
    for cond in conds:
        m, s = agg(cond, "inf_mean_ms")
        md += f"- {cond}: {fmt(m, s)}ms/batch\n"
    md += "\n**Note:** Margin-aware and confidence losses are training-time only. "
    md += "The inference computation graph is identical across all conditions. "
    md += "Any timing differences reflect measurement noise (GPU thermal, batch order), not real overhead.\n\n---\n\n"

    # Q10 decision
    md += """## Q10 -- Decision: does the evidence justify proceeding with confidence consistency loss?

Criteria (all must hold to justify proceeding):
1. proto_conf does NOT regress clean or noisy accuracy vs proto_only.
2. conf_py_mean increases for correct predictions (vs proto_only).
3. conf_py_var decreases vs proto_only (more consistent across noisy views).
4. Confidence separation (correct vs wrong) improves vs proto_only.
5. n_A + n_B (correctly predicted, certified or abstain) >= proto_only.
6. Inference cost remains unchanged.

**Fill in after reviewing results above.**

---

## Output files

| File | Description |
|------|-------------|
| `aggregate_metrics.csv` | Mean+/-std over seeds per condition |
| `per_seed_metrics.csv` | Final-epoch metrics per (condition, seed) |
| `geometry.csv` | Post-training prototype geometry per run |
| `inference_cost.csv` | Wall-clock latency + throughput per run |
| `confidence_diagnostics.csv` | Per-run noisy-view confidence diagnostics |
| `certification_diagnostic_summary.csv` | Cert pilot aggregated per condition |
| `certification_failure_analysis.csv` | Per-example cert data with sample_type |
| `training_loss_curves.csv` | Per-epoch training logs incl. conf_loss |
| `smoke_tests.csv` | Smoke test pass/fail log |
| `plots/` | 11 figures |
"""
    path = OUT / "summary.md"
    path.write_text(md, encoding="utf-8")
    print(f"  Wrote {path}", flush=True)


# ── smoke test ────────────────────────────────────────────────────────────────
def run_smoke_tests(config, device):
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    print("\n=== SMOKE TEST ===", flush=True)

    # 1. Model construction
    try:
        model = build_finetune_model(config, device=device, tiny_random=True)
        results.append(("model_construction", "PASS", ""))
        print("  [1/7] model construction: PASS")
    except Exception as e:
        results.append(("model_construction", "FAIL", str(e)))
        print(f"  [1/7] model construction: FAIL — {e}")
        _write_smoke_csv(results)
        return False

    # 2. Data loading
    try:
        train_loader, test_loader = make_loaders(config, 200, 50)
        results.append(("data_loading", "PASS", ""))
        print("  [2/7] data loading: PASS")
    except Exception as e:
        results.append(("data_loading", "FAIL", str(e)))
        print(f"  [2/7] data loading: FAIL — {e}")
        _write_smoke_csv(results)
        return False

    # 3. Baseline — no conf_loss in logging_dict
    ld_b = None
    try:
        set_seed(0)
        model_b = build_finetune_model(config, device=device, tiny_random=True)
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-4)
        acc_b = accelerate.Accelerator(cpu=(device.type == "cpu"))
        model_b, opt_b, tl_b = acc_b.prepare(model_b, opt_b, train_loader)
        ld_b = train_one_epoch_consistency(
            acc_b, model_b, tl_b, opt_b, nn.CrossEntropyLoss(),
            mixup_fn=None, noise_aug=NOISE_AUG, lbd=LBD, eta=ETA,
            margin_config=make_margin_config("baseline"),
        ) or {}
        assert "avg_conf_loss" not in ld_b, f"baseline should NOT have conf_loss; got {list(ld_b.keys())}"
        results.append(("baseline_no_conf_loss", "PASS", str(ld_b.get("avg_loss"))))
        print("  [3/7] baseline no conf_loss: PASS")
    except Exception as e:
        results.append(("baseline_no_conf_loss", "FAIL", str(e)))
        print(f"  [3/7] baseline no conf_loss: FAIL — {e}")

    # 4. proto_conf — conf_loss > 0
    ld_c = None
    acc_c = None
    model_c = None
    try:
        set_seed(0)
        model_c = build_finetune_model(config, device=device, tiny_random=True)
        opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-4)
        acc_c = accelerate.Accelerator(cpu=(device.type == "cpu"))
        model_c, opt_c, tl_c = acc_c.prepare(model_c, opt_c, train_loader)
        ld_c = train_one_epoch_consistency(
            acc_c, model_c, tl_c, opt_c, nn.CrossEntropyLoss(),
            mixup_fn=None, noise_aug=NOISE_AUG, lbd=LBD, eta=ETA,
            margin_config=make_margin_config("proto_conf"),
        ) or {}
        assert "avg_conf_loss" in ld_c, f"proto_conf should have conf_loss; keys={list(ld_c.keys())}"
        cv = ld_c["avg_conf_loss"]
        assert cv > 0, f"conf_loss should be > 0, got {cv}"
        results.append(("proto_conf_loss_nonzero", "PASS", f"avg_conf_loss={cv:.4f}"))
        print(f"  [4/7] proto_conf L_conf > 0: PASS (avg_conf_loss={cv:.4f})")
    except Exception as e:
        results.append(("proto_conf_loss_nonzero", "FAIL", str(e)))
        print(f"  [4/7] proto_conf L_conf > 0: FAIL — {e}")

    # 5. conf diagnostics in logging_dict
    try:
        assert ld_c is not None and "avg_conf_py_mean" in ld_c and "avg_conf_py_var" in ld_c, \
            f"missing conf diag keys; got {list(ld_c.keys()) if ld_c else 'None'}"
        results.append(("conf_diagnostics_logged", "PASS",
                        f"py_mean={ld_c['avg_conf_py_mean']:.4f} py_var={ld_c['avg_conf_py_var']:.6f}"))
        print(f"  [5/7] conf diagnostics in log: PASS")
    except Exception as e:
        results.append(("conf_diagnostics_logged", "FAIL", str(e)))
        print(f"  [5/7] conf diagnostics in log: FAIL — {e}")

    # 6. compute_confidence_diagnostics
    try:
        raw_c = acc_c.unwrap_model(model_c) if acc_c and model_c else None
        assert raw_c is not None
        cd = compute_confidence_diagnostics(raw_c, config, device, n_samples=50)
        assert "conf_py_mean" in cd and "conf_py_var" in cd
        results.append(("confidence_diagnostics_fn", "PASS",
                        f"conf_py_mean={cd['conf_py_mean']:.4f}"))
        print(f"  [6/7] compute_confidence_diagnostics: PASS")
    except Exception as e:
        results.append(("confidence_diagnostics_fn", "FAIL", str(e)))
        print(f"  [6/7] compute_confidence_diagnostics: FAIL — {e}")

    # 7. Inference cost unchanged
    try:
        raw_b = acc_b.unwrap_model(model_b)
        ms_b, _, _ = measure_inference_cost(raw_b, device, n_repeats=5, batch_size=32,
                                            image_size=config.dataset.image_size)
        raw_c2 = acc_c.unwrap_model(model_c)
        ms_c, _, _ = measure_inference_cost(raw_c2, device, n_repeats=5, batch_size=32,
                                             image_size=config.dataset.image_size)
        ratio = ms_c / ms_b if ms_b > 0 else float("inf")
        status = "PASS" if ratio < 5.0 else "WARN"
        results.append(("inference_unchanged", status,
                        f"baseline={ms_b:.2f}ms proto_conf={ms_c:.2f}ms ratio={ratio:.2f}"))
        print(f"  [7/7] inference cost: {status} (ratio={ratio:.2f})")
    except Exception as e:
        results.append(("inference_unchanged", "FAIL", str(e)))
        print(f"  [7/7] inference cost: FAIL — {e}")

    _write_smoke_csv(results)
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_warn = sum(1 for _, s, _ in results if s == "WARN")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\nSmoke summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
    return n_fail == 0


def _write_smoke_csv(results):
    path = OUT / "smoke_tests.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "status", "detail"])
        w.writerows(results)
    print(f"  Wrote {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument("--skip-cert", action="store_true")
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--conditions", nargs="+", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    config = load_config(CONFIG_PATH)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.smoke_test_only:
        ok = run_smoke_tests(config, device)
        sys.exit(0 if ok else 1)

    print("Running smoke tests before full experiment...", flush=True)
    ok = run_smoke_tests(config, device)
    if not ok:
        print("Smoke tests FAILED. Aborting.", flush=True)
        sys.exit(1)
    print("Smoke tests passed.\n", flush=True)

    conds_to_run = args.conditions if args.conditions else CONDITION_ORDER
    ckpt_dir = OUT / "checkpoints"
    log_dir = OUT / "training_logs"
    all_results = []

    for cond_name in conds_to_run:
        if cond_name not in CONDITIONS:
            print(f"Unknown condition: {cond_name}. Skipping.")
            continue
        for seed in DEFAULT_SEEDS:
            ckpt = ckpt_dir / f"{cond_name}_seed{seed}.pt"
            if ckpt.exists():
                print(f"  Skipping {cond_name} seed={seed} (checkpoint exists).")
                continue
            result = run_one(
                cond_name, seed, config, device, log_dir, ckpt_dir,
                args.n_train, args.n_test, args.epochs,
                skip_cert=args.skip_cert,
            )
            all_results.append(result)

    if not all_results:
        print("No new runs.")
        sys.exit(0)

    cert_rows_all = [cr for r in all_results if r["cert_rows"] for cr in r["cert_rows"]]
    cert_agg = summarise_cert(cert_rows_all, DEFAULT_SEEDS) if cert_rows_all else {}

    print("\nWriting outputs...", flush=True)
    write_per_seed_metrics(all_results)
    write_aggregate_metrics(all_results)
    write_geometry_csv(all_results)
    write_inference_cost_csv(all_results)
    write_confidence_diagnostics_csv(all_results)
    write_training_loss_csv(all_results)
    if cert_agg:
        write_certification_summary_csv(cert_agg, DEFAULT_SEEDS)
        write_cert_failure_analysis_csv(all_results)
    make_plots(all_results, cert_agg, DEFAULT_SEEDS)
    write_summary_md(all_results, cert_agg, DEFAULT_SEEDS,
                     args.n_train, args.n_test, args.epochs)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
