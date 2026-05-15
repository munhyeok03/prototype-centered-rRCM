"""
Stronger preliminary rRCM-path experiment (v2).

Tags:
  preliminary_debug_experiment
  not_official_rrcm_reproduction
  not_certified_accuracy

Runs 4 conditions using the original rRCM code path:
  1. baseline     – consistency loss only
  2. proto_only   – consistency + prototype cross-entropy
  3. margin_only  – consistency + margin hinge loss
  4. proto_margin – consistency + both

By default (--mode minimal) runs only baseline + proto_margin for 3 seeds.
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

from experiment_utils import build_finetune_model, load_config, set_seed
from rrcm_tune import (
    load_data,
    train_one_epoch_consistency,
    margin_aware_prototype_loss,
    classifier_weight_prototypes,
)

# --------------------------------------------------------------------------- #
EXPERIMENT_TAG = (
    "preliminary_debug_experiment | "
    "not_official_rrcm_reproduction | "
    "not_certified_accuracy"
)

NOISE_AUG = 0.5
LBD = 10.0
ETA = 0.5
BATCH_SIZE = 64
PROTOTYPE_MARGIN = 0.2
PROTOTYPE_TEMPERATURE = 0.1

CONDITIONS_CONFIG = {
    "baseline":    dict(use_margin_aware_loss=False, use_proto_loss=False, use_margin_loss=False),
    "proto_only":  dict(use_margin_aware_loss=True,  use_proto_loss=True,  use_margin_loss=False),
    "margin_only": dict(use_margin_aware_loss=True,  use_proto_loss=False, use_margin_loss=True),
    "proto_margin":dict(use_margin_aware_loss=True,  use_proto_loss=True,  use_margin_loss=True),
}

HPARAM_VARIANTS = {
    "A": dict(lambda_proto=0.1, lambda_margin=0.1),
    "B": dict(lambda_proto=0.2, lambda_margin=0.05),
    "C": dict(lambda_proto=0.2, lambda_margin=0.1),
    "D": dict(lambda_proto=0.1, lambda_margin=0.05),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_margin_config(condition_name, hparam_variant):
    cond = CONDITIONS_CONFIG[condition_name]
    hp   = HPARAM_VARIANTS[hparam_variant]
    return ml_collections.ConfigDict({
        **cond,
        **hp,
        "prototype_margin":      PROTOTYPE_MARGIN,
        "prototype_temperature": PROTOTYPE_TEMPERATURE,
        "prototype_source":      "classifier_weight",
        "normalize_features":    True,
        "log_geometry_metrics":  True,
    })


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
    train_subset = Subset(train_ds, list(range(min(n_train, len(train_ds)))))
    test_subset  = Subset(test_ds,  list(range(min(n_test,  len(test_ds)))))
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, drop_last=False)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, noise_aug=0.0):
    model.eval()
    correct = total = 0
    for img, y in loader:
        img, y = img.to(device), y.to(device)
        pred = model(img, noise_aug=noise_aug)
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_logit_margin(model, loader, device, noise_aug=0.0):
    """Mean top-1 minus top-2 logit gap."""
    model.eval()
    gaps = []
    for img, _ in loader:
        img = img.to(device)
        logits = model(img, noise_aug=noise_aug)
        top2 = logits.topk(2, dim=1).values
        gaps.append((top2[:, 0] - top2[:, 1]).cpu())
    return float(torch.cat(gaps).mean()) if gaps else float("nan")


@torch.no_grad()
def compute_geometry(model, loader, device):
    """Post-training prototype geometry and feature consistency on test set."""
    model.eval()
    protos = classifier_weight_prototypes(model).to(device)
    pm_list, sc_list, swm_list, fc_list = [], [], [], []
    for img, y in loader:
        img, y = img.to(device), y.to(device)
        img_rep = torch.cat([img, img], dim=0)
        y_rep   = torch.cat([y,   y],   dim=0)
        _, feats = model(img_rep, noise_aug=NOISE_AUG, return_features=True)
        m = margin_aware_prototype_loss(
            feats, y_rep, protos,
            temperature=PROTOTYPE_TEMPERATURE, margin=PROTOTYPE_MARGIN,
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
    """Returns (mean_ms, std_ms, throughput_ips)."""
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
    throughput = batch_size / (mean_ms / 1000.0)
    return mean_ms, std_ms, throughput


# --------------------------------------------------------------------------- #
# Single-run trainer
# --------------------------------------------------------------------------- #

def run_one(condition_name, hparam_variant, seed,
            config, data_dir, device, log_dir, n_train, n_test, epochs):
    set_seed(seed)
    acc = accelerate.Accelerator(cpu=(device.type == "cpu"))

    model = build_finetune_model(config, device=device, tiny_random=True)
    train_loader, test_loader = make_loaders(config, data_dir, n_train, n_test)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters()
         if p.requires_grad and n not in ("linear_head.weight", "linear_head.bias")],
        lr=1e-4, weight_decay=0.0,
    )
    loss_fn = nn.CrossEntropyLoss()
    margin_config = make_margin_config(condition_name, hparam_variant)
    model, train_loader, optimizer = acc.prepare(model, train_loader, optimizer)

    hp = HPARAM_VARIANTS[hparam_variant]
    epoch_rows = []
    log_lines = []
    header = (f"=== {condition_name} hp={hparam_variant} "
              f"(lp={hp['lambda_proto']} lm={hp['lambda_margin']}) seed={seed} "
              f"n_train={n_train} epochs={epochs} ===")
    print(header)
    log_lines.append(header)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        metrics = train_one_epoch_consistency(
            acc, model, train_loader, optimizer, loss_fn,
            mixup_fn=None, noise_aug=NOISE_AUG, lbd=LBD, eta=ETA,
            margin_config=margin_config,
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
            return float(v) if v == v else float("nan")  # handle nan

        row = {
            "condition": condition_name, "hparam_variant": hparam_variant,
            "lambda_proto": hp["lambda_proto"], "lambda_margin": hp["lambda_margin"],
            "seed": seed, "epoch": epoch,
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
               f"pm_train={row['train_proto_margin']:.4f} | "
               f"clean={clean_acc:.1f}% "
               f"n05={noisy_acc05:.1f}% n10={noisy_acc10:.1f}% n20={noisy_acc20:.1f}% "
               f"lmarg={lmarg:.3f}  ({elapsed:.1f}s)")
        print(msg)
        log_lines.append(msg)

    raw = acc.unwrap_model(model)
    geo = compute_geometry(raw, test_loader, device)
    inf_mean, inf_std, throughput = measure_inference_cost(
        raw, device, n_repeats=30, batch_size=32,
        image_size=config.dataset.image_size, noise_aug=0.0,
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"{condition_name}_hp{hparam_variant}_seed{seed}.txt"
    (log_dir / log_name).write_text("\n".join(log_lines), encoding="utf-8")
    print(f"  -> log: {log_dir / log_name}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "condition": condition_name,
        "hparam_variant": hparam_variant,
        "lambda_proto": hp["lambda_proto"],
        "lambda_margin": hp["lambda_margin"],
        "seed": seed,
        "epoch_rows": epoch_rows,
        "geometry": geo,
        "inf_mean_ms": inf_mean,
        "inf_std_ms": inf_std,
        "throughput_ips": throughput,
        "final_clean_acc":     epoch_rows[-1]["clean_acc"],
        "final_noisy05":       epoch_rows[-1]["noisy_acc_sigma05"],
        "final_noisy10":       epoch_rows[-1]["noisy_acc_sigma10"],
        "final_noisy20":       epoch_rows[-1]["noisy_acc_sigma20"],
        "final_logit_margin":  epoch_rows[-1]["logit_margin"],
    }


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #

def _group(results, keyfn):
    groups = {}
    for r in results:
        groups.setdefault(keyfn(r), []).append(r)
    return groups


def _mstd(vals):
    clean = [v for v in vals if v == v and v != float("nan")]
    if not clean:
        return float("nan"), 0.0
    return float(np.mean(clean)), float(np.std(clean)) if len(clean) > 1 else 0.0


# --------------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------------- #

def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_per_epoch_csv(path, all_results):
    rows = []
    for r in all_results:
        rows.extend(r["epoch_rows"])
    _write_csv(path, rows)


def write_per_seed_csv(path, all_results):
    rows = []
    for r in all_results:
        geo = r["geometry"]
        rows.append({
            "condition": r["condition"], "hparam_variant": r["hparam_variant"],
            "lambda_proto": r["lambda_proto"], "lambda_margin": r["lambda_margin"],
            "seed": r["seed"],
            "final_clean_acc":        r["final_clean_acc"],
            "final_noisy05":          r["final_noisy05"],
            "final_noisy10":          r["final_noisy10"],
            "final_noisy20":          r["final_noisy20"],
            "final_logit_margin":     r["final_logit_margin"],
            "proto_margin_mean":      geo["proto_margin_mean"],
            "sim_correct_mean":       geo["sim_correct_mean"],
            "sim_wrong_max_mean":     geo["sim_wrong_max_mean"],
            "feat_consistency_mean":  geo["feat_consistency_mean"],
            "inf_mean_ms":            r["inf_mean_ms"],
            "throughput_ips":         r["throughput_ips"],
        })
    _write_csv(path, rows)


def write_geometry_csv(path, all_results):
    rows = []
    for r in all_results:
        rows.append({
            "condition": r["condition"], "hparam_variant": r["hparam_variant"],
            "seed": r["seed"],
            **r["geometry"],
            "experiment_tag": EXPERIMENT_TAG,
        })
    _write_csv(path, rows)


def write_inference_cost_csv(path, all_results):
    rows = []
    for r in all_results:
        rows.append({
            "condition": r["condition"], "hparam_variant": r["hparam_variant"],
            "seed": r["seed"],
            "inf_mean_ms":    r["inf_mean_ms"],
            "inf_std_ms":     r["inf_std_ms"],
            "throughput_ips": r["throughput_ips"],
            "experiment_tag": EXPERIMENT_TAG,
        })
    _write_csv(path, rows)


def write_aggregate_metrics_csv(path, all_results):
    groups = _group(all_results, lambda r: (r["condition"], r["hparam_variant"]))
    rows = []
    for (cond, hp), rs in sorted(groups.items()):
        def _f(key):   return _mstd([r["final_" + key]           for r in rs]) if "final_" + key in r else _mstd([r[key] for r in rs])
        def _g(key):   return _mstd([r["geometry"][key]          for r in rs])
        def _i(key):   return float(np.mean([r[key]              for r in rs]))
        rows.append({
            "condition": cond, "hparam_variant": hp,
            "lambda_proto": rs[0]["lambda_proto"], "lambda_margin": rs[0]["lambda_margin"],
            "n_seeds": len(rs),
            "clean_acc_mean":  _mstd([r["final_clean_acc"] for r in rs])[0],
            "clean_acc_std":   _mstd([r["final_clean_acc"] for r in rs])[1],
            "noisy05_mean":    _mstd([r["final_noisy05"]   for r in rs])[0],
            "noisy05_std":     _mstd([r["final_noisy05"]   for r in rs])[1],
            "noisy10_mean":    _mstd([r["final_noisy10"]   for r in rs])[0],
            "noisy10_std":     _mstd([r["final_noisy10"]   for r in rs])[1],
            "noisy20_mean":    _mstd([r["final_noisy20"]   for r in rs])[0],
            "noisy20_std":     _mstd([r["final_noisy20"]   for r in rs])[1],
            "logit_margin_mean":  _mstd([r["final_logit_margin"]          for r in rs])[0],
            "logit_margin_std":   _mstd([r["final_logit_margin"]          for r in rs])[1],
            "proto_margin_mean":  _mstd([r["geometry"]["proto_margin_mean"]  for r in rs])[0],
            "proto_margin_std":   _mstd([r["geometry"]["proto_margin_mean"]  for r in rs])[1],
            "sim_correct_mean":   _mstd([r["geometry"]["sim_correct_mean"]   for r in rs])[0],
            "sim_correct_std":    _mstd([r["geometry"]["sim_correct_mean"]   for r in rs])[1],
            "sim_wrong_max_mean": _mstd([r["geometry"]["sim_wrong_max_mean"] for r in rs])[0],
            "sim_wrong_max_std":  _mstd([r["geometry"]["sim_wrong_max_mean"] for r in rs])[1],
            "feat_consistency_mean": _mstd([r["geometry"]["feat_consistency_mean"] for r in rs])[0],
            "feat_consistency_std":  _mstd([r["geometry"]["feat_consistency_mean"] for r in rs])[1],
            "inf_mean_ms":    float(np.mean([r["inf_mean_ms"]    for r in rs])),
            "throughput_ips": float(np.mean([r["throughput_ips"] for r in rs])),
        })
    _write_csv(path, rows)


def write_ablation_summary_csv(path, all_results):
    ablation = [r for r in all_results if r["hparam_variant"] == "A"] or all_results
    groups = _group(ablation, lambda r: r["condition"])
    order = ["baseline", "proto_only", "margin_only", "proto_margin"]
    rows = []
    for cond in order:
        if cond not in groups:
            continue
        rs = groups[cond]
        rows.append({
            "condition": cond, "n_seeds": len(rs),
            "clean_acc_mean":     _mstd([r["final_clean_acc"]                   for r in rs])[0],
            "clean_acc_std":      _mstd([r["final_clean_acc"]                   for r in rs])[1],
            "noisy05_mean":       _mstd([r["final_noisy05"]                     for r in rs])[0],
            "noisy05_std":        _mstd([r["final_noisy05"]                     for r in rs])[1],
            "noisy10_mean":       _mstd([r["final_noisy10"]                     for r in rs])[0],
            "noisy10_std":        _mstd([r["final_noisy10"]                     for r in rs])[1],
            "noisy20_mean":       _mstd([r["final_noisy20"]                     for r in rs])[0],
            "noisy20_std":        _mstd([r["final_noisy20"]                     for r in rs])[1],
            "logit_margin_mean":  _mstd([r["final_logit_margin"]                for r in rs])[0],
            "logit_margin_std":   _mstd([r["final_logit_margin"]                for r in rs])[1],
            "proto_margin_mean":  _mstd([r["geometry"]["proto_margin_mean"]     for r in rs])[0],
            "proto_margin_std":   _mstd([r["geometry"]["proto_margin_mean"]     for r in rs])[1],
            "sim_correct_mean":   _mstd([r["geometry"]["sim_correct_mean"]      for r in rs])[0],
            "sim_correct_std":    _mstd([r["geometry"]["sim_correct_mean"]      for r in rs])[1],
            "sim_wrong_max_mean": _mstd([r["geometry"]["sim_wrong_max_mean"]    for r in rs])[0],
            "sim_wrong_max_std":  _mstd([r["geometry"]["sim_wrong_max_mean"]    for r in rs])[1],
        })
    _write_csv(path, rows)


def write_hyperparameter_summary_csv(path, all_results):
    hp_results = [r for r in all_results if r["condition"] == "proto_margin"] or all_results
    groups = _group(hp_results, lambda r: r["hparam_variant"])
    rows = []
    for hv in sorted(groups.keys()):
        rs = groups[hv]
        rows.append({
            "hparam_variant": hv,
            "lambda_proto": rs[0]["lambda_proto"], "lambda_margin": rs[0]["lambda_margin"],
            "n_seeds": len(rs),
            "clean_acc_mean":     _mstd([r["final_clean_acc"]                  for r in rs])[0],
            "clean_acc_std":      _mstd([r["final_clean_acc"]                  for r in rs])[1],
            "noisy05_mean":       _mstd([r["final_noisy05"]                    for r in rs])[0],
            "noisy05_std":        _mstd([r["final_noisy05"]                    for r in rs])[1],
            "proto_margin_mean":  _mstd([r["geometry"]["proto_margin_mean"]    for r in rs])[0],
            "proto_margin_std":   _mstd([r["geometry"]["proto_margin_mean"]    for r in rs])[1],
            "sim_correct_mean":   _mstd([r["geometry"]["sim_correct_mean"]     for r in rs])[0],
            "sim_wrong_max_mean": _mstd([r["geometry"]["sim_wrong_max_mean"]   for r in rs])[0],
        })
    _write_csv(path, rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def make_plots(out_dir, all_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ablation = [r for r in all_results if r["hparam_variant"] == "A"] or all_results
    groups_a = _group(ablation, lambda r: r["condition"])

    order = ["baseline", "proto_only", "margin_only", "proto_margin"]
    colors = {"baseline": "#4c72b0", "proto_only": "#55a868",
              "margin_only": "#c44e52", "proto_margin": "#8172b2"}
    present = [c for c in order if c in groups_a]
    if not present:
        return

    def agg_val(cond, fn):
        rs = groups_a.get(cond, [])
        vals = [fn(r) for r in rs]
        vals = [v for v in vals if v == v]
        return (float(np.mean(vals)) if vals else float("nan"),
                float(np.std(vals)) if len(vals) > 1 else 0.0)

    x = np.arange(len(present))
    W = 0.55
    clrs = [colors.get(c, "#888") for c in present]

    def simple_bar(ax, fn, ylabel, title):
        means = [agg_val(c, fn)[0] for c in present]
        stds  = [agg_val(c, fn)[1] for c in present]
        ax.bar(x, means, W, yerr=stds, capsize=5, color=clrs, alpha=0.85, ecolor="#444")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in present], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axhline(0, color="black", lw=0.5, ls="--")

    # 1. Clean accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    simple_bar(ax, lambda r: r["final_clean_acc"], "Accuracy (%)",
               "Clean Accuracy (mean±std over seeds)")
    fig.tight_layout(); fig.savefig(plots_dir / "clean_accuracy.png", dpi=120); plt.close(fig)

    # 2. Noisy accuracy vs sigma
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (fn, lbl) in zip(axes, [
        (lambda r: r["final_noisy05"], "σ=0.5"),
        (lambda r: r["final_noisy10"], "σ=1.0"),
        (lambda r: r["final_noisy20"], "σ=2.0"),
    ]):
        simple_bar(ax, fn, "Accuracy (%)", f"Noisy Acc {lbl}")
    fig.suptitle("Noisy Accuracy (not certified robustness)", fontsize=11)
    fig.tight_layout(); fig.savefig(plots_dir / "noisy_accuracy_vs_sigma.png", dpi=120); plt.close(fig)

    # 3. Prototype margin
    fig, ax = plt.subplots(figsize=(6, 4))
    simple_bar(ax, lambda r: r["geometry"]["proto_margin_mean"],
               "Prototype Margin", "Prototype Margin (↑ better)")
    fig.tight_layout(); fig.savefig(plots_dir / "prototype_margin.png", dpi=120); plt.close(fig)

    # 4. sim_correct vs sim_wrong_max (grouped)
    fig, ax = plt.subplots(figsize=(7, 4))
    w2 = 0.32
    xx = np.arange(len(present))
    sc_m  = [agg_val(c, lambda r: r["geometry"]["sim_correct_mean"])[0]  for c in present]
    sc_s  = [agg_val(c, lambda r: r["geometry"]["sim_correct_mean"])[1]  for c in present]
    swm_m = [agg_val(c, lambda r: r["geometry"]["sim_wrong_max_mean"])[0] for c in present]
    swm_s = [agg_val(c, lambda r: r["geometry"]["sim_wrong_max_mean"])[1] for c in present]
    ax.bar(xx - w2/2, sc_m,  w2, yerr=sc_s,  capsize=4, label="sim_correct (↑)",   alpha=0.85, color="#4c72b0")
    ax.bar(xx + w2/2, swm_m, w2, yerr=swm_s, capsize=4, label="sim_wrong_max (↓)", alpha=0.85, color="#c44e52")
    ax.set_xticks(xx)
    ax.set_xticklabels([c.replace("_", "\n") for c in present], fontsize=9)
    ax.set_ylabel("Cosine Similarity", fontsize=9)
    ax.set_title("Prototype Similarities (mean±std over seeds)", fontsize=10)
    ax.legend(fontsize=8); ax.axhline(0, color="black", lw=0.5, ls="--")
    fig.tight_layout(); fig.savefig(plots_dir / "prototype_similarities.png", dpi=120); plt.close(fig)

    # 5. Top-1/top-2 logit margin
    fig, ax = plt.subplots(figsize=(6, 4))
    simple_bar(ax, lambda r: r["final_logit_margin"],
               "Logit Gap", "Top-1 / Top-2 Logit Margin (↑ better)")
    fig.tight_layout(); fig.savefig(plots_dir / "logit_margin.png", dpi=120); plt.close(fig)

    # 6. Inference cost (latency + throughput)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    simple_bar(axes[0], lambda r: r["inf_mean_ms"],    "ms / batch", "Inference Latency (batch=32)")
    simple_bar(axes[1], lambda r: r["throughput_ips"], "img/s",      "Throughput (↑ better)")
    fig.tight_layout(); fig.savefig(plots_dir / "inference_cost.png", dpi=120); plt.close(fig)

    # 7. Feature consistency distance
    fig, ax = plt.subplots(figsize=(6, 4))
    simple_bar(ax, lambda r: r["geometry"]["feat_consistency_mean"],
               "L2 dist (normalized)", "Feature Consistency Distance (↓ = more stable)")
    fig.tight_layout(); fig.savefig(plots_dir / "feature_consistency.png", dpi=120); plt.close(fig)

    # 8. Hparam sensitivity (if >1 variant)
    hp_rs = [r for r in all_results if r["condition"] == "proto_margin"]
    hp_groups = _group(hp_rs, lambda r: r["hparam_variant"])
    if len(hp_groups) > 1:
        hvs = sorted(hp_groups.keys())
        hx  = np.arange(len(hvs))
        pm_m = [float(np.mean([r["geometry"]["proto_margin_mean"] for r in hp_groups[hv]])) for hv in hvs]
        ca_m = [float(np.mean([r["final_clean_acc"]               for r in hp_groups[hv]])) for hv in hvs]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(hx, pm_m, W, alpha=0.85, color="#8172b2")
        axes[0].set_xticks(hx); axes[0].set_xticklabels(hvs)
        axes[0].set_ylabel("Prototype Margin"); axes[0].set_title("proto_margin: Prototype Margin")
        axes[1].bar(hx, ca_m, W, alpha=0.85, color="#8172b2")
        axes[1].set_xticks(hx); axes[1].set_xticklabels(hvs)
        axes[1].set_ylabel("Clean Acc (%)"); axes[1].set_title("proto_margin: Clean Accuracy")
        fig.suptitle("Hyperparameter Sensitivity (proto_margin, mean over seeds)", fontsize=11)
        fig.tight_layout(); fig.savefig(plots_dir / "hparam_sensitivity.png", dpi=120); plt.close(fig)

    print(f"  Plots saved to {plots_dir}")


# --------------------------------------------------------------------------- #
# Load prior results (for incremental / append runs)
# --------------------------------------------------------------------------- #

def load_existing_results(out_dir: Path) -> list:
    """Reconstruct result dicts from previously written per_seed + geometry + inference CSVs."""
    per_seed_path = out_dir / "per_seed_metrics.csv"
    if not per_seed_path.exists():
        return []

    geo_data, inf_data = {}, {}

    geo_path = out_dir / "geometry.csv"
    if geo_path.exists():
        with geo_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                k = (row["condition"], row["hparam_variant"], int(row["seed"]))
                geo_data[k] = {
                    "proto_margin_mean":     float(row["proto_margin_mean"]),
                    "sim_correct_mean":      float(row["sim_correct_mean"]),
                    "sim_wrong_max_mean":    float(row["sim_wrong_max_mean"]),
                    "feat_consistency_mean": float(row["feat_consistency_mean"]),
                }

    inf_path = out_dir / "inference_cost.csv"
    if inf_path.exists():
        with inf_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                k = (row["condition"], row["hparam_variant"], int(row["seed"]))
                inf_data[k] = {
                    "inf_mean_ms":    float(row["inf_mean_ms"]),
                    "inf_std_ms":     float(row["inf_std_ms"]),
                    "throughput_ips": float(row["throughput_ips"]),
                }

    results = []
    with per_seed_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            hp   = row["hparam_variant"]
            seed = int(row["seed"])
            k    = (cond, hp, seed)
            geo  = geo_data.get(k, {
                "proto_margin_mean":     float(row.get("proto_margin_mean", "nan")),
                "sim_correct_mean":      float(row.get("sim_correct_mean",  "nan")),
                "sim_wrong_max_mean":    float(row.get("sim_wrong_max_mean","nan")),
                "feat_consistency_mean": float(row.get("feat_consistency_mean","nan")),
            })
            inf  = inf_data.get(k, {})
            results.append({
                "condition": cond, "hparam_variant": hp,
                "lambda_proto": float(row["lambda_proto"]),
                "lambda_margin": float(row["lambda_margin"]),
                "seed": seed,
                "epoch_rows": [],   # not stored in per_seed CSV
                "geometry": geo,
                "inf_mean_ms":    inf.get("inf_mean_ms",    float("nan")),
                "inf_std_ms":     inf.get("inf_std_ms",     float("nan")),
                "throughput_ips": inf.get("throughput_ips", float("nan")),
                "final_clean_acc":    float(row["final_clean_acc"]),
                "final_noisy05":      float(row["final_noisy05"]),
                "final_noisy10":      float(row["final_noisy10"]),
                "final_noisy20":      float(row["final_noisy20"]),
                "final_logit_margin": float(row["final_logit_margin"]),
            })
    return results


def load_existing_epoch_rows(out_dir: Path) -> list:
    """Load raw epoch-level rows from per_epoch_metrics.csv."""
    p = out_dir / "per_epoch_metrics.csv"
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Summary markdown
# --------------------------------------------------------------------------- #

def write_summary_md(path, all_results, args, device):
    ablation = [r for r in all_results if r["hparam_variant"] == "A"] or all_results
    groups_a = _group(ablation, lambda r: r["condition"])
    n_seeds = len(set(r["seed"] for r in ablation))

    def agg(cond):
        rs = groups_a.get(cond, [])
        if not rs:
            return None
        def m(fn): return _mstd([fn(r) for r in rs])
        return {
            "clean": m(lambda r: r["final_clean_acc"]),
            "n05":   m(lambda r: r["final_noisy05"]),
            "n10":   m(lambda r: r["final_noisy10"]),
            "n20":   m(lambda r: r["final_noisy20"]),
            "lmarg": m(lambda r: r["final_logit_margin"]),
            "pm":    m(lambda r: r["geometry"]["proto_margin_mean"]),
            "sc":    m(lambda r: r["geometry"]["sim_correct_mean"]),
            "swm":   m(lambda r: r["geometry"]["sim_wrong_max_mean"]),
            "fc":    m(lambda r: r["geometry"]["feat_consistency_mean"]),
            "inf":   m(lambda r: r["inf_mean_ms"]),
            "thru":  m(lambda r: r["throughput_ips"]),
        }

    b  = agg("baseline")
    pm = agg("proto_margin")
    po = agg("proto_only")
    mo = agg("margin_only")
    run_conditions = sorted(set(r["condition"] for r in all_results))
    run_hparams    = sorted(set(r["hparam_variant"] for r in all_results))

    def fmt(d, key, fmt_=".3f"):
        if d is None: return "N/A"
        mu, sd = d[key]
        return f"{mu:{fmt_}}±{sd:{fmt_}}"

    def delta_str(a, bval, key, pos_good=True, fmt_=".2f"):
        if a is None or bval is None: return "N/A"
        d = a[key][0] - bval[key][0]
        sign = "+" if d >= 0 else ""
        verb = ("improvement" if (pos_good and d > 0.3) or (not pos_good and d < -0.3)
                else "regression" if (pos_good and d < -0.3) or (not pos_good and d > 0.3)
                else "neutral")
        return f"{a[key][0]:{fmt_}}±{a[key][1]:{fmt_}} vs {bval[key][0]:{fmt_}}±{bval[key][1]:{fmt_}} (Δ{sign}{d:{fmt_}}, {verb})"

    # Q1
    if pm and b:
        better = pm["pm"][0] > b["pm"][0]
        q1 = ("**Preliminary signal: YES** — proto_margin shows larger prototype margin than baseline "
              if better else
              "**Preliminary signal: UNCLEAR** — proto_margin did not clearly outperform baseline ")
        q1 += (f"({fmt(pm,'pm')}) vs baseline ({fmt(b,'pm')}) "
               f"over {n_seeds} seeds. Interpret cautiously — random init, small subset.")
    else:
        q1 = "Insufficient conditions run."

    # Q2
    if pm and b:
        sc_d  = pm["sc"][0]  - b["sc"][0]
        swm_d = pm["swm"][0] - b["swm"][0]
        if abs(swm_d) >= abs(sc_d) and swm_d < 0:
            driver = "**Primarily via reduced max wrong similarity (sim_wrong_max ↓).**"
        elif sc_d > 0 and sc_d >= abs(swm_d):
            driver = "**Primarily via increased correct similarity (sim_correct ↑).**"
        else:
            driver = "**Mixed: both components shift, neither clearly dominant.**"
        q2 = (f"{driver} "
              f"sim_correct: {delta_str(pm,b,'sc',pos_good=True,fmt_='.4f')}. "
              f"sim_wrong_max: {delta_str(pm,b,'swm',pos_good=False,fmt_='.4f')}.")
    else:
        q2 = "Insufficient data."

    # Q3
    if pm and b:
        def noisy_line(key, lbl):
            d = pm[key][0] - b[key][0]
            verdict = "preliminary improvement" if d > 0.5 else ("slight regression" if d < -0.5 else "neutral")
            return f"  - {lbl}: baseline {b[key][0]:.1f}% → proto_margin {pm[key][0]:.1f}% (Δ{d:+.1f}%, {verdict})"
        q3 = "\n".join([noisy_line("n05","σ=0.5"), noisy_line("n10","σ=1.0"), noisy_line("n20","σ=2.0")])
    else:
        q3 = "Insufficient data."

    # Q4
    if pm and b:
        d = pm["clean"][0] - b["clean"][0]
        if abs(d) < 0.5:
            q4 = f"**No meaningful drop** (Δ={d:+.2f}%, within measurement noise)."
        elif d < 0:
            q4 = f"**Small drop**: {b['clean'][0]:.1f}% → {pm['clean'][0]:.1f}% (Δ{d:+.1f}%). Expected from robustness–accuracy trade-off."
        else:
            q4 = f"**Slight improvement**: {b['clean'][0]:.1f}% → {pm['clean'][0]:.1f}% (Δ{d:+.1f}%)."
    else:
        q4 = "Insufficient data."

    # Q5
    if pm and b:
        d_ms = pm["inf"][0] - b["inf"][0]
        q5 = (f"**No change** (margin-aware loss is training-only; inference arch is identical). "
              f"Latency: {b['inf'][0]:.2f}ms vs {pm['inf'][0]:.2f}ms (Δ{d_ms:+.2f}ms — noise). "
              f"Throughput: {b['thru'][0]:.0f} vs {pm['thru'][0]:.0f} img/s.")
    else:
        q5 = "Insufficient data."

    # Q6
    if po and mo and b:
        d_po = po["pm"][0] - b["pm"][0]
        d_mo = mo["pm"][0] - b["pm"][0]
        winner = "proto loss" if d_po > d_mo else "margin loss"
        q6 = (f"**{winner} contributes more.** "
              f"proto_only Δpm vs baseline: {d_po:+.4f}; "
              f"margin_only Δpm vs baseline: {d_mo:+.4f}.")
    else:
        q6 = (f"Ablation conditions (proto_only, margin_only) not run in this stage. "
              f"Run `--conditions baseline proto_only margin_only proto_margin` to compare components.")

    # Q7
    hp_rs = [r for r in all_results if r["condition"] == "proto_margin"]
    hp_grps = _group(hp_rs, lambda r: r["hparam_variant"])
    if len(hp_grps) > 1:
        best_hv, best_pm_val = None, float("-inf")
        for hv, rs in hp_grps.items():
            v = float(np.mean([r["geometry"]["proto_margin_mean"] for r in rs]))
            if v > best_pm_val:
                best_pm_val, best_hv = v, hv
        hp_lines = "; ".join(
            f"hp={hv}: pm={np.mean([r['geometry']['proto_margin_mean'] for r in rs]):.4f}"
            for hv, rs in sorted(hp_grps.items())
        )
        q7 = f"**Best by prototype margin: hp={best_hv}** (pm={best_pm_val:.4f}). All: {hp_lines}."
    else:
        q7 = ("Only hparam variant A (λ_proto=0.1, λ_margin=0.1) was run. "
              "Run with `--hparam-variants A B C D` to compare.")

    q8 = """\
1. **Certified robustness** — Cannot run `certify.py` without a properly pre-trained rRCM backbone.
2. **Meaningful absolute accuracy** — Random init produces near-chance accuracy; numbers are for relative comparison only.
3. **Official rRCM reproduction** — Requires the official checkpoint and full-dataset training.
4. **Paper Table comparison** — All numbers here are from a randomly-initialised tiny model.
5. **Certified radius ablation** — Meaningful only after proper pre-training."""

    # Condition table
    cond_table_rows = []
    for c in ["baseline", "proto_only", "margin_only", "proto_margin"]:
        a = agg(c)
        if a is None:
            continue
        cond_table_rows.append(
            f"| {c} | {a['clean'][0]:.1f}±{a['clean'][1]:.1f} "
            f"| {a['n05'][0]:.1f}±{a['n05'][1]:.1f} "
            f"| {a['n10'][0]:.1f}±{a['n10'][1]:.1f} "
            f"| {a['n20'][0]:.1f}±{a['n20'][1]:.1f} "
            f"| {a['pm'][0]:.4f}±{a['pm'][1]:.4f} "
            f"| {a['sc'][0]:.4f}±{a['sc'][1]:.4f} "
            f"| {a['swm'][0]:.4f}±{a['swm'][1]:.4f} "
            f"| {a['lmarg'][0]:.3f}±{a['lmarg'][1]:.3f} |"
        )

    md = f"""# Preliminary rRCM-Path Experiment v2

**Tags:** `{EXPERIMENT_TAG}`

> **Caveats**
> - Model is **randomly initialised** (no official rRCM checkpoint available).
> - This is **not** an official rRCM reproduction.
> - Accuracy figures are **not certified robustness** — raw noisy-accuracy under Gaussian aug.
> - All results are from a debug-scale controlled experiment ({args.epochs} epochs, {args.n_train} training samples).
> - Conclusions are preliminary and may not generalise.

---

## Setup

| Param | Value |
|-------|-------|
| Model | Tiny random rRCMViT (embed_dim=64, depth=1) |
| Task | consistency (rRCM path) |
| Dataset | CIFAR-10 subset ({args.n_train} train / {args.n_test} test) |
| Seeds | {args.seeds} |
| Epochs | {args.epochs} |
| Batch size | {BATCH_SIZE} |
| noise_aug (σ) | {NOISE_AUG} |
| lbd | {LBD} |
| eta | {ETA} |
| prototype_margin | {PROTOTYPE_MARGIN} |
| prototype_temperature | {PROTOTYPE_TEMPERATURE} |
| Conditions run | {run_conditions} |
| Hparam variants run | {run_hparams} |
| Device | {device} |

---

## Aggregate Results (hparam A, mean±std over seeds)

| condition | clean_acc | noisy_σ0.5 | noisy_σ1.0 | noisy_σ2.0 | proto_margin | sim_correct | sim_wrong_max | logit_margin |
|-----------|-----------|------------|------------|------------|--------------|-------------|---------------|--------------|
{chr(10).join(cond_table_rows)}

---

## Q1 — Is prototype margin consistently improved across seeds?

{q1}

---

## Q2 — Is improvement from sim_correct ↑ or sim_wrong_max ↓?

{q2}

---

## Q3 — Does noisy accuracy improve at any sigma?

{q3}

*Not certified robustness — raw noisy-accuracy under Gaussian augmentation.*

---

## Q4 — Does clean accuracy drop?

{q4}

---

## Q5 — Does inference cost change?

{q5}

---

## Q6 — Which component helps more: proto loss or margin loss?

{q6}

---

## Q7 — Which hyperparameter setting looks best?

{q7}

---

## Q8 — What remains blocked without an official checkpoint?

{q8}

---

## Output files

| File | Description |
|------|-------------|
| `aggregate_metrics.csv` | Mean±std over seeds per (condition, hparam_variant) |
| `per_seed_metrics.csv` | Final-epoch metrics per run (condition×hparam×seed) |
| `per_epoch_metrics.csv` | Per-epoch training+eval metrics for all runs |
| `geometry.csv` | Post-training prototype geometry per run |
| `inference_cost.csv` | Wall-clock latency + throughput per run |
| `ablation_summary.csv` | Ablation table across 4 conditions (hparam A) |
| `hyperparameter_summary.csv` | Hparam sensitivity for proto_margin condition |
| `training_logs/` | Per-run console logs |
| `plots/` | Matplotlib figures |
"""
    path.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Preliminary rRCM-path experiment v2")
    parser.add_argument("--config",           default="configs/cifar10_margin_aware_rrcm.py")
    parser.add_argument("--data-dir",         default="data/cifar10")
    parser.add_argument("--out-dir",          default="results/preliminary_rrcm_path_experiment_v2")
    parser.add_argument("--n-train",          type=int, default=2000)
    parser.add_argument("--n-test",           type=int, default=500)
    parser.add_argument("--epochs",           type=int, default=7)
    parser.add_argument("--seeds",            type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--conditions",       nargs="+",
                        choices=list(CONDITIONS_CONFIG.keys()),
                        default=["baseline", "proto_margin"])
    parser.add_argument("--hparam-variants",  nargs="+",
                        choices=list(HPARAM_VARIANTS.keys()), default=["A"])
    parser.add_argument("--mode",             choices=["minimal", "ablation", "full"], default=None,
                        help="minimal=baseline+proto_margin+A; "
                             "ablation=all4conditions+A; full=all4+allhparams")
    parser.add_argument("--device",           default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.mode == "minimal":
        args.conditions = ["baseline", "proto_margin"]
        args.hparam_variants = ["A"]
    elif args.mode == "ablation":
        args.conditions = list(CONDITIONS_CONFIG.keys())
        args.hparam_variants = ["A"]
    elif args.mode == "full":
        args.conditions = list(CONDITIONS_CONFIG.keys())
        args.hparam_variants = list(HPARAM_VARIANTS.keys())

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "training_logs"

    config = load_config(args.config)

    all_requested = [(c, h, s)
                     for c in args.conditions
                     for h in args.hparam_variants
                     for s in args.seeds]

    # Load any prior results already in the output dir and skip those runs
    prior_results = load_existing_results(out_dir)
    prior_keys    = {(r["condition"], r["hparam_variant"], r["seed"]) for r in prior_results}
    runs_to_do    = [(c, h, s) for c, h, s in all_requested if (c, h, s) not in prior_keys]
    prior_epoch_rows = load_existing_epoch_rows(out_dir)

    print(f"\n{'='*60}")
    print(f"Preliminary rRCM-Path Experiment v2")
    print(f"  Device:     {device}")
    print(f"  Conditions: {args.conditions}")
    print(f"  Hparams:    {args.hparam_variants}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  n_train={args.n_train}  n_test={args.n_test}  epochs={args.epochs}")
    if prior_results:
        print(f"  Loaded {len(prior_results)} prior result(s) from {out_dir}")
    print(f"  New runs:   {len(runs_to_do)}  (of {len(all_requested)} requested)")
    print(f"  Output:     {out_dir}")
    print(f"{'='*60}\n")

    new_results = []
    for i, (cond, hp, seed) in enumerate(runs_to_do):
        print(f"\n[{i+1}/{len(runs_to_do)}] {cond}  hp={hp}  seed={seed}")
        r = run_one(
            condition_name=cond, hparam_variant=hp, seed=seed,
            config=config, data_dir=args.data_dir, device=device,
            log_dir=log_dir, n_train=args.n_train, n_test=args.n_test,
            epochs=args.epochs,
        )
        new_results.append(r)

    all_results = prior_results + new_results

    print(f"\n{'='*60}")
    print("Writing output files ...")

    # per_epoch: combine pre-existing rows with new epoch_rows
    new_epoch_rows = [row for r in new_results for row in r["epoch_rows"]]
    all_epoch_rows = prior_epoch_rows + new_epoch_rows
    if all_epoch_rows:
        _write_csv(out_dir / "per_epoch_metrics.csv", all_epoch_rows)
    write_per_seed_csv(         out_dir / "per_seed_metrics.csv",        all_results)
    write_geometry_csv(         out_dir / "geometry.csv",                all_results)
    write_inference_cost_csv(   out_dir / "inference_cost.csv",          all_results)
    write_aggregate_metrics_csv(out_dir / "aggregate_metrics.csv",       all_results)
    write_ablation_summary_csv( out_dir / "ablation_summary.csv",        all_results)
    write_hyperparameter_summary_csv(out_dir / "hyperparameter_summary.csv", all_results)
    make_plots(out_dir, all_results)
    write_summary_md(           out_dir / "summary.md",  all_results, args, device)
    print(f"\nAll outputs in: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
