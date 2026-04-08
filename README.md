# AdaptSci: Training-Free Adaptive Test-Time Compute for Agentic Scientific Equation Discovery

> **Ahsan Bilal · Preprint · 2026**

AdaptSci is a training-free adaptive test-time compute framework built on top of
[SR-Scientist](https://arxiv.org/abs/2510.11661) (ICLR 2026). It introduces
Hessian-geometry-grounded signals that dynamically allocate computation across
three axes — *optimisation effort*, *exploration vs. exploitation*, and
*early stopping* — without adding new tools, new training, or modifying
the SR-Scientist inner iteration loop.

---

## Table of Contents

- [Overview](#overview)
- [Key Ideas](#key-ideas)
  - [Parameter Hessian](#1-parameter-hessian)
  - [Condition Number κ → Adaptive BFGS Restarts](#2-condition-number-κ--adaptive-bfgs-restarts)
  - [Coupling Matrix C → Buffer Diversity](#3-coupling-matrix-c--buffer-diversity)
  - [Effective Parameter Count n_eff → Over-parameterisation Signal](#4-effective-parameter-count-n_eff--over-parameterisation-signal)
  - [Stratified 70/30 Buffer](#5-stratified-7030-buffer)
  - [Early Stopping](#6-early-stopping)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Inference](#inference)
  - [Setup](#inference-setup)
  - [Running AdaptSci](#running-adaptsci)
  - [Running SR-Scientist Baseline](#running-sr-scientist-baseline)
- [Ablation Studies](#ablation-studies)
- [Configuration Reference](#configuration-reference)
- [Citation](#citation)

---

## Overview

SR-Scientist applies a **fixed** test-time compute schedule (25 turns × 40 iterations)
uniformly to all 129 benchmark problems regardless of difficulty. AdaptSci identifies
three structural gaps in this approach and addresses each with a geometry-grounded signal
derived from the Hessian of the fitting loss at the BFGS optimum:

| Gap | Signal | AdaptSci mechanism |
|-----|--------|--------------------|
| Uniform budget | κ (condition number) | Adaptive BFGS restart count (1/3/5) |
| Greedy buffer (monoculture) | C (coupling matrix) | Stratified 70/30 buffer with C-diversity frontier |
| Uninstrumented fitting | n_eff (effective param count) | Degeneracy detection → diversity-only mode; early stopping |

**Key property:** κ, C, and n_eff are all derived from the same p×p Hessian
computed purely numerically at no LLM-call cost.

---

## Key Ideas

### 1. Parameter Hessian

At the BFGS optimum θ*, we compute:

```
H = ∂²L / ∂θ ∂θᵀ  |_{θ*}   ∈ ℝ^{p×p}
```

via central finite differences. All three signals follow from this single matrix.

### 2. Condition Number κ → Adaptive BFGS Restarts

```
κ = λ_max(H) / λ_min(H)
```

High κ means the loss surface is ill-conditioned; a single BFGS run is likely to
report a poor result for a skeleton that would score well from a better starting point.

| κ range | BFGS restarts | Intuition |
|---------|---------------|-----------|
| κ < τ₁ (100) | 1 | Round bowl; one run suffices |
| τ₁ ≤ κ < τ₂ (1000) | 3 | Moderately elongated valley |
| κ ≥ τ₂ | 5 | Highly ill-conditioned; thorough search |

### 3. Coupling Matrix C → Buffer Diversity

```
C_ij = H_ij / sqrt(H_ii · H_jj)
```

C is the normalised off-diagonal Hessian — the second-order analogue of a correlation
matrix applied to parameter interactions. If |C_ij| ≈ 1, parameters θ_i and θ_j are
structurally entangled (e.g. `f(x) = a·b·x` is over-parameterised).

**Buffer diversity:** two equations with nearly identical C matrices are functionally
the same skeleton under different parameterisations. AdaptSci uses the Frobenius
distance ‖C_e − C_j‖_F as a diversity metric for the frontier pool, preventing the
buffer from accumulating structurally identical hypotheses.

### 4. Effective Parameter Count n_eff → Over-parameterisation Signal

```
n_eff = (tr H)² / tr(H²)
```

n_eff is the participation ratio of the Hessian eigenvalue spectrum.
When n_eff ≪ p, the skeleton has more free parameters than the data can constrain.
This is the most reliable signal to abandon the current skeleton and explore a
structurally simpler one.

### 5. Stratified 70/30 Buffer

AdaptSci replaces SR-Scientist's pure top-K exploitation buffer with:

| Pool | Share | Selection criterion |
|------|-------|---------------------|
| Top-performing | 70% | Lowest training NMSE (exploitation) |
| Frontier | 30% | Highest frontier score s_fr(e) (exploration) |

Frontier score:
```
s_fr(e) = α · |ΔNMSE|_w / calls_w  +  (1−α) · min_{j∈S} ‖C_e − C_j‖_F
```

When a degenerate skeleton is detected (|C_ij| > 0.95 or n_eff < 0.5p),
the system switches to **diversity-only mode** (α ← 0).

### 6. Early Stopping

```
β = calls / max(NMSE_t − ε_target, ε)
```

Stop when:
- NMSE < ε_target  (convergence), or
- β > β*           (marginal returns too small for remaining budget)

This releases budget from easy problems for redistribution to harder ones.

---

## Repository Structure

```
adaptsci/
├── main.py                          # Entry point (extended with all AdaptSci args)
├── run_ablations.py                 # Ablation study runner (all 12 conditions)
├── infer/
│   ├── __init__.py
│   ├── inference.py                 # Main orchestrator with AdaptSci logic
│   ├── hessian.py                   # κ, C, n_eff computation (NEW)
│   ├── buffer.py                    # Stratified 70/30 AdaptSci buffer (NEW)
│   ├── early_stop.py                # Efficiency-ratio early stopping (NEW)
│   ├── tool.py                      # Multi-restart equation_evaluator (updated)
│   ├── prompt.py                    # Prompts with Hessian guidance (updated)
│   ├── get_results.py               # Multi-restart evaluation (updated)
│   └── analysis.py                  # Results analysis + Hessian stats (updated)
├── inference/scripts/
│   ├── inference_adaptsci.sh        # Full AdaptSci run script
│   └── inference_sr_scientist_baseline.sh  # Baseline (SR-Scientist mode)
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-repo/adaptsci
cd adaptsci

conda create -n adaptsci python=3.11
conda activate adaptsci

pip install torch==2.8.0
pip install "sglang[all]==0.5.2"
pip install h5py scipy numpy pandas openai tqdm transformers httpx

# Sandbox (SandboxFusion)
git clone https://github.com/bytedance/SandboxFusion.git
cd SandboxFusion
conda create -n sandbox-runtime python=3.11
conda activate sandbox-runtime
poetry install
mkdir -p docs/build
cd runtime/python && pip install -r requirements.txt
cd ../.. && make run-online
```

---

## Inference

### Inference Setup

```bash
# Download and preprocess the benchmark
hf download nnheui/llm-srbench --repo-type dataset --local-dir ./data/inference
python utils/data_preprocessing.py
```

### Running AdaptSci

```bash
# Full AdaptSci (all mechanisms enabled)
bash inference/scripts/inference_adaptsci.sh

# Or directly with all options visible:
python main.py \
    --model-name Qwen3-Coder-480B-A35B-Instruct \
    --model-url http://localhost:8000/v1 \
    --sandbox-urls http://localhost:8080/run_code \
    --parquet-file-path data/inference/lsr_synth.parquet \
    --mape-threshold 0.001 \
    --num-turns 40 \
    --top-k 3 \
    --max-assistant-turns 25 \
    --output-json-path results/adaptsci.json \
    --adaptive-compute \
    --tau1 100 --tau2 1000 \
    --epsilon-target 1e-4 --beta-star 5000 \
    --exploit-frac 0.70 \
    --frontier-alpha 0.5 \
    --frontier-window 3
```

### Running SR-Scientist Baseline

```bash
# Identical to original SR-Scientist (pure SR-Scientist mode)
python main.py \
    --model-name Qwen3-Coder-480B-A35B-Instruct \
    --model-url http://localhost:8000/v1 \
    --sandbox-urls http://localhost:8080/run_code \
    --parquet-file-path data/inference/lsr_synth.parquet \
    --mape-threshold 0.001 \
    --num-turns 40 \
    --top-k 3 \
    --max-assistant-turns 25 \
    --output-json-path results/sr_scientist_baseline.json \
    --no-adaptive-compute
```

---

## Ablation Studies

All 12 ablation conditions from Section 6 of the paper can be run with a single script:

```bash
python run_ablations.py \
    --model-name Qwen3-Coder-480B-A35B-Instruct \
    --model-url http://localhost:8000/v1 \
    --sandbox-urls http://localhost:8080/run_code \
    --parquet-file-path data/inference/lsr_synth.parquet \
    --output-dir results/ablations \
    --mape-threshold 0.001 \
    --num-turns 40 \
    --top-k 3
```

This runs the following conditions:

| Ablation name | Description |
|---------------|-------------|
| `full_adaptsci` | Full AdaptSci (all mechanisms on) |
| `no_early_stopping` | Ablation (i): remove early stopping |
| `fix_n_restarts` | Ablation (ii): fix n_restarts=1 |
| `alpha_0` | Ablation (iii): α=0 (diversity only) |
| `alpha_0.25` | Ablation (iii): α=0.25 |
| `alpha_0.5` | Ablation (iii): α=0.5 (default) |
| `alpha_0.75` | Ablation (iii): α=0.75 |
| `alpha_1` | Ablation (iii): α=1 (improvement only) |
| `no_c_diversity` | Ablation (iv): C-diversity term off |
| `random_frontier` | Ablation (iv): random frontier sampling |
| `hessian_at_first_run` | Ablation (v): Hessian at first BFGS run, not best-of-n |
| `sr_scientist_baseline` | Baseline: SR-Scientist (no adaptive compute) |

You can also run individual ablations via `--run-only`:

```bash
python run_ablations.py ... --run-only full_adaptsci no_early_stopping fix_n_restarts
```

Results are saved as `results/ablations/<name>.json` and a summary CSV is written to
`results/ablations/ablation_summary.csv`.

Individual ablations can also be invoked directly:

```bash
# Ablation (i): no early stopping
python main.py ... --adaptive-compute --ablation-disable-early-stopping

# Ablation (ii): fix n_restarts=1
python main.py ... --adaptive-compute --ablation-fix-n-restarts

# Ablation (iii): sweep α
python main.py ... --adaptive-compute --ablation-alpha-override 0.0
python main.py ... --adaptive-compute --ablation-alpha-override 0.25
# ... etc

# Ablation (iv): C-diversity off
python main.py ... --adaptive-compute --ablation-disable-c-diversity

# Ablation (iv): random frontier
python main.py ... --adaptive-compute --ablation-random-frontier

# Ablation (v): Hessian at first BFGS run
python main.py ... --adaptive-compute --ablation-hessian-at-first-run
```

---

## Configuration Reference

### Core Arguments (inherited from SR-Scientist)

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--model-name` | str | ✓ | vLLM model identifier |
| `--model-url` | str | ✓ | vLLM server URL |
| `--sandbox-urls` | str+ | ✓ | Sandbox execution URL(s) |
| `--parquet-file-path` | str | ✓ | Input dataset path |
| `--mape-threshold` | float | ✓ | MAPE success threshold (e.g. 0.001) |
| `--num-turns` | int | ✓ | Number of outer turns (iterations) |
| `--top-k` | int | 3 | Seed set size K |
| `--max-assistant-turns` | int | 100 | Max tool calls per turn |
| `--output-json-path` | str | ✓ | Output path |

### AdaptSci Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--adaptive-compute` | True | Enable AdaptSci (master switch) |
| `--no-adaptive-compute` | — | Run as SR-Scientist (disables all AdaptSci) |
| `--tau1` | 100 | κ threshold: below → 1 BFGS restart |
| `--tau2` | 1000 | κ threshold: above → 5 BFGS restarts |
| `--epsilon-target` | 1e-4 | NMSE convergence target ε_target |
| `--beta-star` | 5000 | Efficiency ratio threshold β* |
| `--coupling-threshold` | 0.95 | \|C_ij\| threshold for degeneracy |
| `--n-eff-ratio-threshold` | 0.5 | n_eff/p threshold for degeneracy |
| `--exploit-frac` | 0.70 | Exploitation pool fraction (70/30 split) |
| `--frontier-alpha` | 0.5 | α in s_fr (improvement vs diversity) |
| `--frontier-window` | 3 | Window w for momentum computation |
| `--no-hessian-in-prompt` | False | Suppress κ/n_eff guidance from LLM prompt |

### Ablation Flags

| Argument | Ablation | Description |
|----------|----------|-------------|
| `--ablation-disable-early-stopping` | (i) | Remove early stopping |
| `--ablation-fix-n-restarts` | (ii) | Fix n_restarts=1 throughout |
| `--ablation-alpha-override ALPHA` | (iii) | Fix α ∈ {0,0.25,0.5,0.75,1} |
| `--ablation-disable-c-diversity` | (iv) | C-diversity term off (α=1) |
| `--ablation-random-frontier` | (iv) | Random frontier sampling |
| `--ablation-hessian-at-first-run` | (v) | Hessian at first BFGS run, not best-of-n |

---

## Supported Models

All models from SR-Scientist are supported:

| Model | HuggingFace ID |
|-------|----------------|
| Qwen3-Coder-480B (paper backbone) | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` |
| Qwen3-Coder-30B | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| GLM-4.5-Air | `zai-org/GLM-4.5-Air-FP8` |
| GPT-OSS-120B | `openai/gpt-oss-120b` |
| GPT-OSS-20B | `openai/gpt-oss-20b` |

---

## Citation

If you use AdaptSci, please cite:

```bibtex
@article{bilal2026adaptsci,
  title   = {AdaptSci: Training-Free Adaptive Test-Time Compute for Agentic Scientific Equation Discovery},
  author  = {Ahsan Bilal},
  year    = {2026},
}
```

Please also cite SR-Scientist, on which AdaptSci is built:

```bibtex
@misc{xia2025srscientist,
  title         = {SR-Scientist: Scientific Equation Discovery With Agentic AI},
  author        = {Shijie Xia and Yuhan Sun and Pengfei Liu},
  year          = {2025},
  eprint        = {2510.11661},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
}
```

---

## License

Apache License 2.0.
