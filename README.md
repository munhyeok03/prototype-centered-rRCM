# Prototype-Centered rRCM

This repository contains implementation code for prototype-centered extensions of the public [rRCM](https://github.com/jiachenlei/rRCM) codebase. It is designed for controlled preliminary experiments on representation geometry and noisy robustness behavior under the rRCM consistency training path. It uses the upstream rRCM codebase as the base implementation and adds prototype-alignment objectives, margin-ablation experiments, and diagnostic analysis utilities on top. The original upstream README is preserved as [`UPSTREAM_README.md`](UPSTREAM_README.md).

---

## Overview

rRCM trains a model to produce consistent representations across clean and noisy views of the same input. This fork asks a focused question: does adding a class-prototype alignment objective provide a useful additional training signal within the rRCM consistency framework?

The main extension is a prototype alignment loss that encourages noisy representations to be similar to their correct class prototype (defined by the classifier weight vectors). A margin separation loss is included as an ablation. A confidence consistency diagnostic was also explored but did not show gains in the current setting. The goal is not to replace or compete with rRCM, but to study this extension under reproducible controlled conditions.

---

## Relation to upstream rRCM

This repository is an independent research fork of [jiachenlei/rRCM](https://github.com/jiachenlei/rRCM) by Lei et al. The original upstream README is preserved as [`UPSTREAM_README.md`](UPSTREAM_README.md). This fork is not affiliated with or endorsed by the original rRCM authors.

---

## What this fork adds

- **Prototype alignment loss** — encourages noisy representations toward correct class prototypes
- **Margin separation loss** — ablation objective for studying prototype geometry
- **Confidence consistency diagnostic** — exploratory extension; did not improve over proto_only in the current setting
- **Controlled experiment scripts** — three stages from debug-scale to medium-scale, 3 seeds each
- **Geometry analysis utilities** — prototype margin, cosine similarity breakdowns
- **Certification diagnostic utilities** — type-A/B/C sample analysis, smoothed-accuracy curves
- **Inference-cost measurement utilities** — per-condition wall-clock timing

---

## Repository contents

| File / Folder | Purpose |
|---|---|
| `rrcm_tune.py` | Core training loop; adds prototype/margin/confidence loss hooks to `train_one_epoch_consistency` |
| `rrcm/utils.py` | Adds geometry logging hook; upstream code otherwise unchanged |
| `experiment_utils.py` | Shared helpers: `build_finetune_model`, `load_config`, `set_seed` |
| `configs/` | Per-condition config files (baseline, proto_only, margin_only, proto+margin) |
| `smoke_test_margin_aware.py` | 7-item smoke test for all new loss components |
| `run_preliminary_rrcm_path_v2.py` | Debug-scale ablation (2K samples, 7 epochs, 4 conditions) |
| `run_prototype_generalization_v1.py` | Medium-scale generalization study (5K samples, 15 epochs, 3 seeds) |
| `run_prototype_confidence_rrcm_v1.py` | Confidence consistency diagnostic (5K samples, 15 epochs, 3 seeds) |
| `run_certification_pilot.py` | Diagnostic certification pilot (200 samples, N=1,000) |
| `analyze_geometry.py` | Post-hoc prototype geometry analysis |
| `analyze_certification.py` | Certification failure analysis (type-A/B/C breakdown) |
| `measure_inference_cost.py` | Wall-clock inference timing per condition |
| `README.md` | This file |
| `UPSTREAM_README.md` | Original rRCM README |

---

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

GPU experiments require a CUDA-compatible PyTorch build. If `requirements.txt` installs a CPU-only version, replace with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm ml_collections scipy accelerate absl-py
```

---

## Quick sanity checks

Verify the environment and all new loss components before running full experiments:

```bash
# Check CUDA availability and package versions
python verify_cuda_env.py

# Smoke test with baseline config
python smoke_test_margin_aware.py \
    --config configs/cifar10_rrcm_baseline_margin_experiment.py \
    --data-dir data/cifar10

# Smoke test with prototype-alignment config
python smoke_test_margin_aware.py \
    --config configs/cifar10_margin_aware_rrcm.py \
    --data-dir data/cifar10
```

All 7 checks should pass before proceeding.

---

## Running controlled experiments

```bash
# Stage 1: Debug-scale ablation (~20 min on GPU)
python run_preliminary_rrcm_path_v2.py

# Stage 2: Medium-scale generalization study (~2 hr on GPU, 3 seeds)
python run_prototype_generalization_v1.py

# Stage 3: Confidence consistency diagnostic (same scale as Stage 2)
python run_prototype_confidence_rrcm_v1.py

# Optional: Diagnostic certification pilot
python run_certification_pilot.py
```

Generated outputs (CSVs, plots, checkpoints, logs) are written to `results/`, which is excluded from version control.

---

## Preliminary findings

*These results are from randomly initialized tiny models (embed_dim=64, depth=1, ~70K params). They are controlled preliminary experiments, not official rRCM paper benchmarks. Absolute numbers are not comparable to original rRCM results, which require official pre-trained checkpoints and full-scale evaluation.*

Across controlled experiments, prototype alignment was the most consistent positive signal:

- **Medium-scale generalization v1** used a 5K train / 1K test CIFAR-10 subset for 15 epochs. In this experiment, `proto_only` used λp=0.1 and improved clean accuracy from 18.6% to 32.7% (+14.1pp) and noisy σ=0.5 accuracy from 18.4% to 32.3% (+13.8pp), while improving prototype margin from −0.291 to −0.050.
- **Debug-scale ablation v2** used a 2K train / 500 test CIFAR-10 subset for 7 epochs. `margin_only` improved prototype geometry in isolation (prototype margin −0.281 → −0.012) but was not a reliable standalone objective: noisy σ=1.0 accuracy dropped from 19.1% to 15.5%, and noisy σ=2.0 accuracy dropped from 18.6% to 13.1%.
- The combined prototype + margin conditions performed comparably to `proto_only` with no clear additional gain from the margin component. In medium-scale generalization, `proto_margin_B` used λp=0.2, λm=0.05 and produced clean 32.7%, noisy σ=0.5 32.0%, and prototype margin −0.049.
- **Confidence consistency v1** used a 5K train / 1K test CIFAR-10 subset for 15 epochs. In this experiment, `proto_only` used λp=0.2, while `proto_conf` used λp=0.2 and λc=0.05. `proto_conf` did not provide additional gains over `proto_only`: clean accuracy was 32.8% vs 32.7%, noisy σ=0.5 accuracy was 32.3% for both, and prototype margin remained −0.050. Because the softmax outputs remained close to uniform, the confidence objective did not provide a clearly distinct optimization signal beyond the existing supervised loss. This appears to be a regime limitation rather than a flaw in the loss formulation.

---

## Scope and limitations

- No official rRCM checkpoints are included; all experiments use randomly initialized models.
- This is not an official reproduction of the rRCM paper.
- Datasets, checkpoints, and generated result files are not tracked in version control.
- Certification scripts use small sampling budgets (N=1,000) and are intended as diagnostics, not benchmark evaluations.
- Reported results should be interpreted as controlled preliminary evidence within the described experimental setting.

---

## Citation / attribution

For citation details of the original rRCM paper and method, refer to [`UPSTREAM_README.md`](UPSTREAM_README.md) and the [upstream repository](https://github.com/jiachenlei/rRCM).

---
---

# 한국어 설명

이 레포지토리는 공개된 [rRCM](https://github.com/jiachenlei/rRCM) 코드베이스를 기반으로 한 prototype-centered 확장 구현입니다. rRCM 일관성 학습 경로 위에서 표현 기하학(representation geometry)과 노이즈 강건성 행동을 연구하기 위한 통제된 예비 실험용으로 설계되었습니다. upstream rRCM 코드베이스를 기반 구현으로 사용하며, 이 fork는 prototype alignment 목적함수, margin ablation 실험, 진단 분석 유틸리티를 추가합니다. 원본 upstream README는 [`UPSTREAM_README.md`](UPSTREAM_README.md)로 보존되어 있습니다.

---

## 개요

rRCM은 같은 입력의 클린 뷰와 노이즈 뷰 사이의 표현 일관성을 학습합니다. 이 fork는 핵심 질문을 하나 던집니다: rRCM 일관성 프레임워크 내에서 클래스 prototype alignment 목적함수를 추가하는 것이 유용한 추가 학습 신호를 제공하는가?

주요 확장은 prototype alignment loss로, 노이즈 표현이 정답 클래스 prototype(분류기 가중치 벡터로 정의)에 가까워지도록 유도합니다. margin separation loss는 ablation으로 포함됩니다. confidence consistency diagnostic도 탐색했으나 현재 설정에서 추가 이득을 보이지 않았습니다. 목표는 rRCM을 대체하거나 경쟁하는 것이 아니라, 재현 가능한 통제 조건 하에서 이 확장을 연구하는 것입니다.

---

## 원본 rRCM과의 관계

이 레포지토리는 Lei et al.의 [jiachenlei/rRCM](https://github.com/jiachenlei/rRCM)의 독립 연구 fork입니다. 원본 upstream README는 [`UPSTREAM_README.md`](UPSTREAM_README.md)로 보존되어 있습니다. 이 fork는 원본 rRCM 저자들과 제휴하거나 공식 승인을 받은 것이 아닙니다.

---

## 이 fork에서 추가한 내용

- **Prototype alignment loss** — 노이즈 표현이 정답 클래스 prototype에 가까워지도록 유도
- **Margin separation loss** — prototype 기하학 연구를 위한 ablation 목적함수
- **Confidence consistency diagnostic** — 탐색적 확장; 현재 설정에서 proto_only 대비 개선 없음
- **통제 실험 스크립트** — 디버그 스케일부터 중간 스케일까지 3단계, 각 3시드
- **기하학 분석 유틸리티** — prototype margin, cosine similarity 분석
- **인증 진단 유틸리티** — type-A/B/C 샘플 분석, smoothed accuracy 곡선
- **추론 비용 측정 유틸리티** — 조건별 wall-clock 타이밍

---

## 레포 구성

| 파일 / 폴더 | 역할 |
|---|---|
| `rrcm_tune.py` | 핵심 학습 루프; `train_one_epoch_consistency`에 prototype/margin/confidence loss 훅 추가 |
| `rrcm/utils.py` | 기하학 로깅 훅 추가; 나머지 upstream 코드 유지 |
| `experiment_utils.py` | 공유 헬퍼: `build_finetune_model`, `load_config`, `set_seed` |
| `configs/` | 조건별 설정 파일 (baseline, proto_only, margin_only, proto+margin) |
| `smoke_test_margin_aware.py` | 새로운 loss 컴포넌트 전체에 대한 7개 항목 스모크 테스트 |
| `run_preliminary_rrcm_path_v2.py` | 디버그 스케일 ablation (2K 샘플, 7 epoch, 4가지 조건) |
| `run_prototype_generalization_v1.py` | 중간 스케일 일반화 연구 (5K 샘플, 15 epoch, 3시드) |
| `run_prototype_confidence_rrcm_v1.py` | Confidence consistency 진단 (5K 샘플, 15 epoch, 3시드) |
| `run_certification_pilot.py` | 진단용 인증 파일럿 (200 샘플, N=1,000) |
| `analyze_geometry.py` | 사후 prototype 기하학 분석 |
| `analyze_certification.py` | 인증 실패 분석 (type-A/B/C 분류) |
| `measure_inference_cost.py` | 조건별 wall-clock 추론 타이밍 |
| `README.md` | 이 파일 |
| `UPSTREAM_README.md` | 원본 rRCM README |

---

## 설치

Python 3.9 이상이 필요합니다.

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

GPU 실험을 위해 CUDA 호환 PyTorch가 필요할 경우:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm ml_collections scipy accelerate absl-py
```

---

## 빠른 확인

전체 실험 전에 환경과 새로운 loss 컴포넌트를 검증합니다:

```bash
# CUDA 가용성 및 패키지 버전 확인
python verify_cuda_env.py

# baseline 설정으로 스모크 테스트
python smoke_test_margin_aware.py \
    --config configs/cifar10_rrcm_baseline_margin_experiment.py \
    --data-dir data/cifar10

# prototype alignment 설정으로 스모크 테스트
python smoke_test_margin_aware.py \
    --config configs/cifar10_margin_aware_rrcm.py \
    --data-dir data/cifar10
```

7개 항목이 모두 통과한 후 실험을 진행합니다.

---

## 실험 실행

```bash
# 1단계: 디버그 스케일 ablation (~20분, GPU)
python run_preliminary_rrcm_path_v2.py

# 2단계: 중간 스케일 일반화 연구 (~2시간, GPU, 3시드)
python run_prototype_generalization_v1.py

# 3단계: Confidence consistency 진단 (2단계와 동일 규모)
python run_prototype_confidence_rrcm_v1.py

# 선택: 진단용 인증 파일럿
python run_certification_pilot.py
```

생성된 결과물(CSV, 플롯, 체크포인트, 로그)은 `results/`에 저장되며 버전 관리에서 제외됩니다.

---

## 예비 결과 요약

*이 결과는 무작위 초기화된 소형 모델(embed_dim=64, depth=1, 약 70K 파라미터)을 사용한 controlled preliminary experiment입니다. 공식 사전학습 checkpoint와 full-scale 평가를 사용하는 원본 rRCM 논문 벤치마크와 절대 수치를 비교할 수 없습니다.*

통제 실험 전반에서 prototype alignment가 가장 일관된 양성 신호였습니다:

- **Medium-scale generalization v1**은 CIFAR-10 5K train / 1K test subset, 15 epochs로 수행되었습니다. 이 실험의 `proto_only`는 λp=0.1을 사용했으며, clean accuracy를 18.6%에서 32.7%로(+14.1pp), noisy σ=0.5 accuracy를 18.4%에서 32.3%로(+13.8pp) 개선했습니다. Prototype margin도 −0.291에서 −0.050으로 개선되었습니다.
- **Debug-scale ablation v2**는 CIFAR-10 2K train / 500 test subset, 7 epochs로 수행되었습니다. `margin_only`는 단독으로 prototype geometry를 개선했지만(prototype margin −0.281 → −0.012), 신뢰할 수 있는 단독 목적함수는 아니었습니다. Noisy σ=1.0 accuracy는 19.1%에서 15.5%로, noisy σ=2.0 accuracy는 18.6%에서 13.1%로 하락했습니다.
- Prototype + margin 결합 조건은 `proto_only`와 유사한 성능을 보였으며 margin 컴포넌트의 명확한 추가 이득은 없었습니다. Medium-scale generalization에서 `proto_margin_B`는 λp=0.2, λm=0.05를 사용했고 clean 32.7%, noisy σ=0.5 32.0%, prototype margin −0.049를 기록했습니다.
- **Confidence consistency v1**은 CIFAR-10 5K train / 1K test subset, 15 epochs로 수행되었습니다. 이 실험의 `proto_only`는 λp=0.2를, `proto_conf`는 λp=0.2와 λc=0.05를 사용했습니다. `proto_conf`는 `proto_only` 대비 추가 이득을 제공하지 않았습니다: clean accuracy는 32.8% vs 32.7%, noisy σ=0.5 accuracy는 둘 다 32.3%, prototype margin은 둘 다 −0.050 수준이었습니다. Softmax 출력이 균등 분포에 가깝게 유지되어 confidence 목적함수가 기존 supervised loss와 구별되는 추가 최적화 신호를 충분히 제공하지 못한 것으로 해석합니다.

---

## 범위와 한계

- 공식 rRCM 체크포인트는 포함되지 않으며, 모든 실험은 무작위 초기화 모델을 사용합니다.
- 이 레포지토리는 rRCM 논문의 공식 재현이 아닙니다.
- 데이터셋, 체크포인트, 생성된 결과 파일은 버전 관리에서 추적되지 않습니다.
- 인증 스크립트는 소규모 샘플링 예산(N=1,000)을 사용하며 벤치마크 평가가 아닌 진단 목적으로 설계되었습니다.
- 보고된 결과는 기술된 실험 설정 내에서의 통제된 예비 증거로 해석해야 합니다.

---

## 출처 표기

원본 rRCM 논문과 방법의 인용 정보는 [`UPSTREAM_README.md`](UPSTREAM_README.md)와 [upstream 레포지토리](https://github.com/jiachenlei/rRCM)를 참고하시기 바랍니다.
