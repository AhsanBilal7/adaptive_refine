# CIMR Paired Ablation Experiments

This repository runs the paired-ablation CIMR pipeline used in the experiment scripts. The main path is depth-based rollout generation with optional self-critique and self-correction, followed by PRM scoring and aggregation across rollouts.

---

## Project Structure

```text
adaptive_scientific_discovery/
├── run_paired_ablation.py   main Python driver for one dataset run
├── run_paired_ablation.sh   shell wrapper around the driver
├── run_nd_sweep.sh          loops over N and depth values from a config
├── batch_nd_sweep.sh        Slurm wrapper for the same sweep
├── configs/                 model / sweep configs used by the bash scripts
├── src/                     CIMR implementation and model / verifier helpers
└── experiments_results/     output directory created by the scripts
```

---

## Methodology

The paired-ablation pipeline is organized around one question at a time:

1. Run `N` independent rollouts.
2. For each rollout, run `D` depth steps.
3. At each step, generate reasoning, optionally critique it, optionally correct it, then score the resulting trace with the verifier.
4. Aggregate rollout outputs using majority vote and max-PRM selection.

The key ablations exposed by the scripts are:

1. `--no_critique` to skip the critique phase.
2. `--no_correction` to skip the correction phase.
3. `--cimr_n_rollouts` and `--cimr_depth` to control the search budget.

The default sweep configs set `N_VALUES`, `D_VALUES`, `DATASETS`, and the ablation flags for each model.

---

## Installation

```bash
pip install -r requirements.txt
```

If you want 4-bit or 8-bit loading, install `bitsandbytes` as well:

```bash
pip install bitsandbytes>=0.43.0
```

---

## Datasets Used By The Scripts

The bash entrypoints currently run on these datasets:

| Dataset | Notes |
|---|---|
| `aime24` | AIME 2024 |
| `aime25` | AIME 2025 |
| `amc23` | AMC 2023 |
| `math500` | MATH-500 |
| `olympiadbench` | OlympiadBench |
| `gsm8k` | Supported by `run_paired_ablation.sh` directly |

---

## Bash Usage

### 1. Run one paired-ablation job

`run_paired_ablation.sh` is the direct entrypoint. It accepts `--model_name`, `--cimr_n_rollouts`, `--cimr_depth`, `--no_critique`, `--no_correction`, and an optional dataset list.

```bash
bash run_paired_ablation.sh \
  --model_name Qwen/Qwen2.5-Math-7B-Instruct \
  --cimr_n_rollouts 8 \
  --cimr_depth 4 \
  aime24 amc23 math500
```

If you omit the dataset list, it defaults to `aime24 aime25 amc23 math500 olympiadbench`.

### 2. Sweep over `N` and depth

`run_nd_sweep.sh` loads a config from `configs/*.conf` and loops over `N_VALUES` and `D_VALUES`.

```bash
bash run_nd_sweep.sh configs/qwen_7b.conf
```

Example config values:

```bash
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"
N_VALUES=(8 3 5 10)
D_VALUES=(4 2 3 5)
DATASETS=(aime24 aime25 amc23 math500 olympiadbench)
NO_CRITIQUE=1
NO_CORRECTION=0
```

### 3. Submit the same sweep to Slurm

`batch_nd_sweep.sh` wraps the same sweep in a Slurm job.

```bash
sbatch batch_nd_sweep.sh configs/qwen_7b.conf
```

---

## Outputs

`run_paired_ablation.sh` writes results under `experiments_results/results_paired/<model_short>_mnt2048_t0.7_p0.95/`.

For each dataset run, it produces:

```text
logs/paired_<dataset>_N<N>_d<D>.log
summary_paired_<dataset>_N<N>_d<D>.json
results_paired_<dataset>_N<N>_d<D>.jsonl
analysis_paired_<dataset>_N<N>_d<D>.json
sample_cache_<dataset>_N<N>.jsonl   only when cache loading is enabled
```

The CIMR store is also saved as `cimr_full_results_<dataset>_N<N>_d<D>.json` so partially completed runs can resume.

---

## Notes

1. The sweep configs in `configs/` are the source of truth for model name, dataset list, and the `N`/depth grids.
2. `run_paired_ablation.py` is the driver used by the bash scripts; the older `run_eval.py` path in the previous README does not match the current experiment workflow.
3. If you want, I can also rewrite the remaining README sections to document the Python driver flags one by one.
