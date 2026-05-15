# Prototype-Centered rRCM

This repository is a fork of [rRCM](https://github.com/jiachenlei/rRCM) (Lei et al.) that adds prototype alignment and confidence consistency losses to the rRCM consistency training objective. It contains implementation code and scripts for controlled preliminary experiments only — it is not an official rRCM reproduction and does not provide paper-level results.

> The original rRCM README is preserved in [`UPSTREAM_README.md`](UPSTREAM_README.md).

---

## What this is not

- **Not** an official rRCM reproduction
- **No** official rRCM pre-trained checkpoints (none are publicly available)
- **No** paper-level certified robustness (experiments use N=1,000; Cohen et al. use N=100,000)
- **No** full-scale training (experiments use up to 5,000 CIFAR-10 samples)

All results come from a randomly initialized tiny model (embed_dim=64, depth=1, ~70K params). Do not compare absolute numbers to published rRCM results.

---

## What this fork adds

**Modified files**

| File | Change |
|------|--------|
| `rrcm_tune.py` | Added `confidence_consistency_loss()` and prototype/margin loss hooks in `train_one_epoch_consistency` |
| `rrcm/utils.py` | Minor diagnostic hook for geometry logging |

**New files**

| File | Description |
|------|-------------|
| `experiment_utils.py` | `build_finetune_model`, `load_config`, `set_seed` helpers |
| `smoke_test_margin_aware.py` | 7-item smoke test for all new loss components |
| `run_preliminary_rrcm_path_v2.py` | Debug-scale ablation: baseline vs proto_only vs margin_only vs proto_margin (2K train, 7 epochs) |
| `run_prototype_generalization_v1.py` | Medium-scale: baseline vs proto_only vs proto_margin_B (5K train, 15 epochs, 3 seeds) |
| `run_prototype_confidence_rrcm_v1.py` | Confidence consistency study: proto_only vs proto_conf — null result (5K train, 15 epochs, 3 seeds) |
| `run_certification_pilot.py` | Diagnostic certification pilot (200 samples, N=1,000) |
| `analyze_geometry.py` | Post-hoc prototype geometry analysis |
| `analyze_certification.py` | Certification failure analysis (type-A/B/C breakdown) |
| `evaluate_metrics.py` | Aggregate metrics from result CSVs |
| `make_plots.py` | Standalone plot generation |
| `measure_inference_cost.py` | Wall-clock inference timing |
| `verify_cuda_env.py` | Environment / CUDA sanity check |
| `configs/cifar10_rrcm_baseline_margin_experiment.py` | Baseline config |
| `configs/cifar10_rrcm_proto_only.py` | proto_only config |
| `configs/cifar10_rrcm_margin_only.py` | margin_only config |
| `configs/cifar10_margin_aware_rrcm.py` | Combined proto+margin config |

---

## Setup

Requires Python 3.9+, CUDA (tested on CUDA 11.x / PyTorch 2.x).

```bash
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
# Debug-scale ablation (~20 min on GPU)
python run_preliminary_rrcm_path_v2.py

# Medium-scale generalization (3 seeds, ~2 hr on GPU)
python run_prototype_generalization_v1.py

# Confidence consistency null result (same scale)
python run_prototype_confidence_rrcm_v1.py
```

Results are written to `results/` (excluded from version control).

---

## Preliminary findings

These results are from a randomly initialized tiny model in a controlled setting. They are not comparable to published rRCM benchmarks.

**Prototype alignment improves classification over the consistency-only baseline:**

| Condition | Clean Acc | Noisy (σ=0.5) | Proto Margin |
|-----------|-----------|---------------|--------------|
| baseline | 18.6±2.8% | 18.4±2.4% | −0.291 |
| proto_only | 32.7±1.6% | 32.3±1.1% | −0.050 |

**Confidence consistency loss (proto_conf) is a null result** in this regime: with a randomly initialized model, softmax outputs stay near-uniform throughout training (~0.1/class), making the loss approximately constant and its gradient near-zero. This is a regime limitation, not a flaw in the loss formulation.
