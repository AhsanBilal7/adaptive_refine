# AdaptSci: Training-Free Adaptive Test-Time Compute for Agentic Scientific Equation Discovery

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue.svg)]()
[![Built on LLM-SR](https://img.shields.io/badge/Built%20on-LLM--SR%20(ICLR%202025%20Oral)-b31b1b.svg)](https://arxiv.org/abs/2404.18400)

**AdaptSci** extends [LLM-SR](../llmsr/README.md) with training-free *adaptive test-time compute allocation* along two axes:

| Component | What it does | Proposal §  |
|---|---|:---:|
| **Adaptive Buffer Allocation** | Online exploit/explore balance; favours diverse islands when progress stalls | §1.2 |
| **Hessian-Guided Optimization** | Allocates extra BFGS restarts for ill-conditioned candidates; skips degenerate ones | §1.3–1.4 |

---

## Project layout

```
project_root/
├── llmsr/          ← pure LLM-SR (no AdaptSci)
└── adaptsci/       ← AdaptSci (imports llmsr as sibling package)
```

AdaptSci does **not** copy LLM-SR code. It imports from `llmsr` and only
overrides what it needs: buffer, evaluator, sampler, pipeline, profiler, config.

---

## Hessian signals (§1.3)

For a fitted skeleton **f(x; θ⋆)** with fitting-loss Hessian **H(f)**:

| Signal | Formula | Interpretation |
|---|---|---|
| Condition number κ | λ_max / (λ_min + ε) | Large → ill-conditioned → extra restarts |
| Coupling | mean \|C_ij\|, C_ij = H_ij/√(H_ii H_jj+ε) | High → redundant params → redirect explore |
| n_eff / p | tr(H)²/(tr(H²)+ε) / p | Low → over-parameterized → redirect explore |

### Adaptive policy (§1.4)

```
if n_eff/p < eff_dim_ratio_threshold   →  REDIRECT_EXPLORE
elif coupling > coupling_threshold     →  REDIRECT_EXPLORE
elif κ > condition_number_threshold    →  EXTRA_RESTARTS (k ∝ log severity)
else                                   →  WELL_CONDITIONED
```

---

## Ablation modes

| `--ablation_mode` | Buffer adaptive | Hessian | Reproduces |
|---|:---:|:---:|---|
| `full` | ✅ | ✅ | AdaptSci (paper) |
| `buffer_only` | ✅ | ❌ | AdaptSci-Buffer |
| `hessian_only` | ❌ | ✅ | AdaptSci-Hessian |
| `none` | ❌ | ❌ | Plain LLM-SR |

---

## Supported Models

Same model families as LLM-SR (server is shared). See [llmsr/README.md](../llmsr/README.md) for full table.

**API backends:**

| Backend | `--api_model` | Env var |
|---|---|---|
| OpenAI | `gpt-3.5-turbo` · `gpt-4o` | `API_KEY` |
| Anthropic | `claude-3-5-sonnet-20241022` · `claude-3-opus-20240229` · `claude-3-5-haiku-20241022` | `ANTHROPIC_API_KEY` |

---

## Installation

```bash
# Install from project root (llmsr must be on the Python path)
conda create -n adaptsci python=3.11.7
conda activate adaptsci
pip install -r requirements.txt
```

AdaptSci shares the same `requirements.txt` as LLM-SR (adds only `scipy`, which is already present).

---

## Quick Start

### Full AdaptSci + Qwen 2.5-7B (local)

```bash
# Step 1: start server (from project root)
python llmsr/llm_engine/engine.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --gpu_ids 0 --port 5000 --quantization

# Step 2: run AdaptSci
PYTHONPATH=. python adaptsci/main.py \
    --local_server_url http://127.0.0.1:5000 \
    --problem_name oscillator2 \
    --spec_path ./adaptsci/specs/specification_oscillator2_numpy.txt \
    --log_path ./logs/adaptsci/qwen2.5_7b/full/oscillator2 \
    --ablation_mode full
```

### Full AdaptSci + GPT-4o (API)

```bash
export API_KEY=sk-...
PYTHONPATH=. python adaptsci/main.py \
    --use_api True --api_model gpt-4o \
    --problem_name oscillator2 \
    --spec_path ./adaptsci/specs/specification_oscillator2_numpy.txt \
    --log_path ./logs/adaptsci/gpt4o/full/oscillator2 \
    --ablation_mode full
```

### Full AdaptSci + Claude 3.5 Sonnet (API)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
PYTHONPATH=. python adaptsci/main.py \
    --use_api True --api_model claude-3-5-sonnet-20241022 \
    --problem_name oscillator2 \
    --spec_path ./adaptsci/specs/specification_oscillator2_numpy.txt \
    --log_path ./logs/adaptsci/claude/full/oscillator2 \
    --ablation_mode full
```

### Full ablation matrix (4 modes × 4 problems)

```bash
export API_KEY=sk-...
for MODE in full buffer_only hessian_only none; do
  for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
    PYTHONPATH=. python adaptsci/main.py \
      --use_api True --api_model gpt-3.5-turbo \
      --problem_name $PROB \
      --spec_path ./adaptsci/specs/specification_${PROB}_numpy.txt \
      --log_path ./logs/adaptsci/ablation/${MODE}/$PROB \
      --ablation_mode $MODE
  done
done
```

---

## CLI Reference

### AdaptSci-specific flags

| Argument | Default | Description |
|---|---|---|
| `--ablation_mode` | `full` | `full` / `buffer_only` / `hessian_only` / `none` |
| `--kappa_threshold` | `50.0` | κ(f) → extra restarts above this |
| `--coupling_threshold` | `0.85` | Mean \|C_ij\| → redirect explore above this |
| `--eff_dim_ratio_threshold` | `0.3` | n_eff/p → redirect explore below this |
| `--max_extra_restarts` | `3` | Max additional BFGS restarts |
| `--hessian_step` | `1e-4` | Finite-difference step h |
| `--ab_window_size` | `50` | Sliding window for progress estimation |
| `--ab_progress_threshold` | `0.01` | Min relative improvement for strong-progress |
| `--ab_diversity_fn` | `token_jaccard` | `token_jaccard` / `edit_distance` / `length_bucket` |

### Shared backend flags (same as LLM-SR)

| Argument | Default | Description |
|---|---|---|
| `--llm_backend` | `local` | `local` or `vllm` |
| `--local_server_url` | `http://127.0.0.1:5000` | Flask engine URL |
| `--vllm_url` | `http://127.0.0.1:8000` | vLLM server URL |
| `--use_api` | `False` | Use external API |
| `--api_model` | `gpt-3.5-turbo` | API model name |

---

## Structure

```
adaptsci/
├── adaptsci/
│   ├── __init__.py
│   ├── config.py           HessianConfig, AdaptiveBufferConfig, ablation_mode
│   ├── hessian.py          κ, coupling, n_eff diagnostics + adaptive policy
│   ├── adaptive_buffer.py  Online exploit/explore scheduler
│   ├── buffer.py           ExperienceBuffer + adaptive island sampling hook
│   ├── evaluator.py        Evaluator + Hessian-guided refit
│   ├── pipeline.py         Wires all AdaptSci components
│   ├── sampler.py          Passes scheduler + hessian_cfg through the loop
│   └── profile.py          TensorBoard + JSON with Hessian/scheduler metrics
├── main.py                 Entry point
├── run_server.sh           Server launch (symlinks to llmsr/run_server.sh)
├── run_adaptsci.sh         All ablation + sweep run commands
├── specs/                  Problem specification files
└── README.md               This file
```

---

## Logging

```
logs/adaptsci/<run>/
├── samples/
│   └── samples_N.json      score + Hessian diagnostics + policy action
└── events.out.tfevents.*   TensorBoard
```

### New TensorBoard scalars (vs LLM-SR)

| Scalar | Description |
|---|---|
| `AdaptSci/exploit_weight` | Current exploitation weight of the scheduler |
| `AdaptSci/Policy Actions` | Cumulative: well_conditioned / extra_restarts / redirect_explore |
| `AdaptSci/Mean Kappa` | Running mean κ(f) across all evaluated candidates |
| `AdaptSci/Mean Coupling` | Running mean parameter coupling |
| `AdaptSci/Mean n_eff` | Running mean effective dimensionality |
| `Hessian/kappa` | Per-sample κ |
| `Hessian/coupling` | Per-sample coupling |
| `Hessian/eff_dim_ratio` | Per-sample n_eff/p |

### Per-sample JSON format

```json
{
  "sample_order": 42,
  "function": "def equation(x, params): ...",
  "score": -0.0031,
  "sample_time": 1.23,
  "evaluate_time": 0.87,
  "hessian": {
    "kappa": 312.4,
    "coupling": 0.23,
    "n_eff": 3.1,
    "n_params": 5,
    "eff_dim_ratio": 0.62,
    "policy_action": "extra_restarts",
    "policy_reason": "kappa=312.4 > threshold 50.0"
  }
}
```

---

## Citation

```bibtex
@article{shojaee2024llm,
  title   = {LLM-SR: Scientific Equation Discovery via Programming with Large Language Models},
  author  = {Shojaee, Parshin and Meidani, Kazem and Gupta, Shashank and Farimani, Amir Barati and Reddy, Chandan K},
  journal = {ICLR 2025},
  year    = {2025}
}
```

---

## License

MIT License. Built on [LLM-SR](https://github.com/deep-symbolic-mathematics/LLM-SR), [FunSearch](https://github.com/google-deepmind/funsearch), and [PySR](https://github.com/MilesCranmer/PySR).
