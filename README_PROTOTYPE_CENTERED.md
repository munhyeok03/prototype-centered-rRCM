# Prototype-Centered rRCM

**An exploratory extension of rRCM that adds prototype alignment to the consistency training objective.**

> **Attribution:** This repository is a fork of [rRCM](https://github.com/jiachenlei/rRCM) by Jia-Chen Lei et al.
> The base rRCM code (all files not listed under "What this fork adds") belongs to the original authors.
> This extension is independent research and is not affiliated with or endorsed by the rRCM authors.

---

## What this fork adds

### Core implementation

| File | Description |
|------|-------------|
| `rrcm_tune.py` | Added `confidence_consistency_loss()` and prototype/margin loss integration in `train_one_epoch_consistency` |
| `rrcm/utils.py` | Minor diagnostic hook for geometry logging |
| `experiment_utils.py` | `build_finetune_model`, `load_config`, `set_seed` helpers used by experiment scripts |

### Experiment scripts

| File | Description |
|------|-------------|
| `run_preliminary_rrcm_path_v2.py` | Debug-scale ablation (2K train, 7 epochs): baseline vs proto_only vs margin_only vs proto_margin |
| `run_prototype_generalization_v1.py` | Medium-scale generalization (5K train, 15 epochs): baseline vs proto_only vs proto_margin_B |
| `run_prototype_confidence_rrcm_v1.py` | Medium-scale confidence consistency study (5K train, 15 epochs): proto_only vs proto_conf — null result |
| `run_certification_pilot.py` | Diagnostic certification pilot (200 samples, N=1000) |
| `smoke_test_margin_aware.py` | 7-item smoke test for all new loss components |

### Config files

| File | Description |
|------|-------------|
| `configs/cifar10_rrcm_baseline_margin_experiment.py` | Baseline config for margin/prototype experiments |
| `configs/cifar10_rrcm_proto_only.py` | proto_only condition config |
| `configs/cifar10_rrcm_margin_only.py` | margin_only condition config |
| `configs/cifar10_margin_aware_rrcm.py` | Combined proto+margin config |

### Analysis scripts

| File | Description |
|------|-------------|
| `analyze_geometry.py` | Post-hoc prototype geometry analysis |
| `analyze_certification.py` | Certification failure analysis (type-A/B/C breakdown) |
| `evaluate_metrics.py` | Aggregate metrics from CSVs |
| `make_plots.py` | Standalone plot generation |
| `measure_inference_cost.py` | Wall-clock inference timing |
| `verify_cuda_env.py` | Environment / CUDA sanity check |

---

## What this fork does NOT provide

- Official rRCM pre-trained checkpoints (not available publicly)
- Paper-level certified robustness numbers (experiments use N=1,000; Cohen et al. use N=100,000)
- Full-scale CIFAR-10 training (experiments use up to 5,000 training samples)
- Reproduction of original rRCM paper results

All experiment results were obtained with a **randomly initialized tiny model** (embed_dim=64, depth=1, ~70K params). Results are controlled comparisons, not absolute benchmarks.

---

## Key findings

**Prototype alignment (proto_only, λp=0.2) consistently improves over the rRCM consistency baseline:**

| Condition | Clean Acc | Noisy σ=0.5 | Proto Margin |
|-----------|-----------|-------------|--------------|
| baseline | 18.6±2.8% | 18.4±2.4% | −0.291 |
| proto_only | 32.7±1.6% | 32.3±1.1% | −0.050 |

**Confidence consistency loss (proto_conf) is a null result** in the random-init regime: softmax outputs remain near-uniform (~0.1/class) throughout training, making the loss constant and its gradient near-zero.

---

## Environment setup

```bash
# Requires Python 3.9+ and CUDA (tested on CUDA 11.x / PyTorch 2.x)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm ml_collections scipy accelerate absl-py
```

---

## Smoke test

```bash
python smoke_test_margin_aware.py
```

Runs 7 checks: loss forward pass, gradient flow, conf loss constant detection, geometry metric computation, certification logic, and timing.

---

## Reproducing experiments

```bash
# Debug-scale ablation (fast, ~20 min on GPU)
python run_preliminary_rrcm_path_v2.py

# Medium-scale generalization (5K samples, 15 epochs, 3 seeds)
python run_prototype_generalization_v1.py

# Confidence consistency null result (same scale)
python run_prototype_confidence_rrcm_v1.py
```

Results are written to `results/` (excluded from version control).

---

## Scientific caution

> This is **not** an official rRCM reproduction.
> Results use a randomly initialized model with no pre-training.
> Certified robustness numbers are diagnostic-grade only (N=1,000, not paper-grade N=100,000).
> Do not compare absolute numbers to published rRCM results.

The primary contribution is a controlled demonstration that **prototype alignment reduces the near-constant wrong-certified artifact** in the random-init regime, and a documented null result for confidence consistency loss in the same regime.
