# Adaptive Delta Refinement (ADR): repair only the reasoning that must change

This repository contains:

1. **The base method, *Refining Over Resampling* (CIMR):** `N` independent rollouts, each refined `D` times by *rewrite → critique → correct*, then a plurality vote. The untouched original code is on the **`cimr` branch**. On `main`, the original paired-ablation scripts are kept, and a faithful CIMR baseline is built into the ADR runner.
2. **Adaptive Delta Refinement** (`updated_methodology.md`), on `main`. It is the same independent rollouts and the same one-answer-per-rollout vote, with one change inside each rollout: **instead of regenerating the whole solution, regenerate only the faulty step and the steps that depend on it**, and reuse everything else verbatim.

ADR is training-free and works on the current question alone. It has no development set, no learned controller, no reward model, no procedure file, and no cross-question memory.

---

## The method in one example (methodology §4)

A rectangle is 12 m by 5 m; paint costs $3/m², fence $2/m. The model writes a numbered solution with dependency ids:

```text
1. uses: []      Area: 12×5 = 50            ← wrong
2. uses: []      Perimeter: 2(12+5) = 34
3. uses: [1]     Paint: 3×50 = 150
4. uses: [2]     Fence: 2×34 = 68
5. uses: [3, 4]  Total: 150+68 = \boxed{218}
```

The critic says `FIX 1 | 12×5 is 60`. The repair set is step 1 plus everything that (indirectly) uses it: **A(1) = {1, 3, 5}**. Steps 2 and 4 stay **byte-for-byte**. The corrector outputs only three replacement lines (60, 180, `\boxed{248}`).

For comparison, a full rewrite regenerates all 5 steps, and suffix regeneration from step 1 also regenerates all 5.

---

## How the code implements the methodology

| Methodology | Implementation | File |
|---|---|---|
| §3.1 structured solutions | One call per rollout produces `k. uses: [...]` steps. They are parsed deterministically into a DAG (ids 1..m, references only backward). A malformed solution takes the full-rewrite fallback until it is valid. | `src/adr/structure.py`, `src/adr/prompts.py` |
| §3.2 critic | The same frozen model returns `DECISION: KEEP`, `FIX <step> \| <error>`, or `REBUILD \| <reason>`. It checks both the math and missing dependency references, and never sees other rollouts or the majority. | `src/adr/prompts.py` |
| §3.3 repair set | `A(j) = {j} ∪ Desc(j)` by BFS over the DAG. There are no scores, thresholds, or top-k. `A(j) ⊆ {j..m}` always. | `structure.repair_set_adr` |
| §3.4 patch | The corrector gets the question, the current solution, the error, and the exact ids allowed to change. It must return exactly those ids. Other steps are kept verbatim. Dependency lists may change but must point backward. An invalid patch triggers the full-rewrite fallback if affordable; otherwise the previous solution is kept. | `structure.apply_patch`, `methods.direct_gen` |
| §3.5 / §5 loop | Round-robin over active rollouts: critique → KEEP freezes / FIX patches / REBUILD rewrites. A repeated complete state freezes the rollout (cycle). The loop stops when all rollouts are frozen or no turn is affordable. The last complete solution of each rollout votes (plurality; ties go to the lowest rollout index). | `methods.direct_gen`, `voting.py` |
| §5 budget | Every call, failed output, and fallback is charged with its **real** tokens. Each turn reserves the maximum cost of critique + update. Only actual cost is charged, so a KEEP returns the unused allowance. An incomplete update never replaces a complete solution. | `budget.py`, `methods.py` |
| §6 configuration | Breadth `N` plus a predeclared set of total budgets `B` replace depth `D`. All methods share the same decoding and per-call caps. | `config.py`, `.env` |
| §7 comparison | `cimr` (original three-stage, fixed D), `full_rewrite` (direct critique + full rewrite, same stopping rule; = ablation 2), `suffix` (direct critique + regenerate steps j..end), **`adr`**, `adr_all_deps` (ablation 1: every step uses all earlier steps, so A(j) = suffix), and `maj_matched` (compute-matched majority sampling under the same B). The four direct-critique variants start from the **same structured solutions and identical first critiques** (shared prefix). | `methods.py`, `runner.py` |
| §7 reporting | Accuracy and solved count vs. total input/output tokens and FLOPs; wall-clock latency (same batching); fraction of reasoning regenerated per repair and `\|A\|/suffix`; wrong→correct and correct→wrong changes; format failures, fallback rate, invalid-update rate; KEEP/FIX/REBUILD counts; paired question-level comparisons against ADR (sign-flip p-value and 95% bootstrap CI). | `runner.summarize`, `stats.py` |
| §7 pilot | `pilot`: ADR on 50 questions. Reports dependency well-formedness, repair-set vs. suffix sizes, and fallback rate, then gives a verdict ("do not begin a large sweep" if the solutions are mostly linear or fallbacks dominate). It fits nothing. | `run_adr.py pilot`, `runner.pilot_report` |

**Not included:**
* **Thought-ICS** (autonomous), the external full-system baseline in §7. Run it from its own release under the same budgets.
* The **dependency-omission audit** on a labeled sample (§8). This is a manual analysis; the per-turn records in `results_*.jsonl` contain every solution version and repair set it needs.

---

## Project structure

```text
├── .env / sample.env           machine paths + predefined variables (sourced by every script)
├── configs/
│   ├── adr_<model>.conf        model-specific overrides (context length, batch size)
│   ├── adr_aime24.conf         experiment: §7 comparison on AIME24
│   ├── adr_math500.conf        experiment: §7 main study on MATH500
│   └── <model>[_cric_corr].conf  original CIMR N×D sweep configs
├── run_adr.py                  CLI: sanity | pilot | run
├── run_adr.sh / batch_adr.sh   launcher (sources .env + confs, venv, lock, log) / Slurm wrapper
├── src/adr/                    Adaptive Delta Refinement
│   ├── structure.py            parse / validate DAG, repair sets, patch & suffix splicing
│   ├── prompts.py              generation, critic (KEEP/FIX/REBUILD), patch, suffix, rewrite
│   ├── methods.py              shared prefix, direct-critique loop, CIMR, compute-matched majority
│   ├── runner.py               cells per budget, grading, metrics, paired comparisons, pilot
│   ├── backends.py, budget.py  model calls with real token counts; compute meter
│   ├── scheduler.py            continuous batching across all solves
│   ├── checkpoint.py, progress.py   continuous saving / resume; readable progress
│   └── voting.py, stats.py, config.py, state.py
├── src/cimr_*.py, model_wrapper.py, dataset_loader.py, evaluator_classes.py, ...   shared + original pipeline
├── run_paired_ablation.{py,sh}, run_nd_sweep.sh, batch_nd_sweep.sh   original CIMR scripts
└── tests/test_adr.py           18 unit tests (scripted backend, no GPU)
```

---

## Installation

The environment is pinned by `pyproject.toml` + `uv.lock` (Python 3.12, torch 2.10+cu128, transformers 4.57.6):

```bash
set -a; source .env; set +a       # UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, UV_PROJECT_ENVIRONMENT=$VENV_PATH (scratch)
uv sync --frozen --python 3.12 --no-install-package vllm
```

`--no-install-package vllm` is needed on these RHEL8 nodes. The locked vllm wheel requires glibc ≥ 2.31; the nodes ship 2.28. The default backend is HF.

Compute nodes are offline, so cache models and datasets from a login node first:

```bash
source $VENV_PATH/bin/activate
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-Math-7B-Instruct')"
python -c "import sys; sys.path.insert(0,'.'); from src.dataset_loader import load_dataset_split as l; [l(x) for x in ['aime24','aime25','amc23','math500','olympiadbench']]"
```

---

## Configuration: `.env` → `configs/*.conf` → CLI

* **`.env`** holds paths and the predefined variables. Big files go to scratch: `ADR_OUTPUT_ROOT=$SCRATCH/adr_results`, plus the HF/uv/pip caches and the venv.
* **`configs/*.conf`** are sourced after `.env` and override it. Stack them with commas (model conf first, then the experiment conf).
* **CLI flags** (`--<field>`) override both. Every field of `src/adr/config.py:ADRConfig` reads `ADR_<FIELD>`.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | per conf | frozen model used in every role |
| `ADR_N_ROLLOUTS` | 4 | breadth N |
| `ADR_BUDGETS` | `30000,60000` | predeclared total per-question budgets B |
| `ADR_COST_MEASURE` | `tokens` | budget unit (`tokens`, `tflops`, `calls`; FLOPs = 2·params·tokens) |
| `ADR_TEMPERATURE`, `ADR_TOP_P` | 0.7, 0.95 | shared decoding for solutions / rewrites / patches |
| `ADR_CRITIQUE_TEMPERATURE` | 0.0 | deterministic critic |
| `ADR_GEN_MAX_NEW_TOKENS`, `ADR_PATCH_MAX_NEW_TOKENS`, `ADR_CRITIQUE_MAX_NEW_TOKENS` | 1536 / 1024 / 384 | shared per-call output caps |
| `ADR_CIMR_DEPTH` | 4 | D of the original-method baseline |
| `ADR_METHODS`, `ADR_DATASETS`, `ADR_SEEDS` | all six, `math500`, 1 | what to run |

---

## How to run

```bash
# 0) checks (no GPU): grader, extraction, parser, the §4 worked example (A(1) = {1,3,5}), patch splicing
bash run_adr.sh sanity

# 1) §7 feasibility pilot: 50 MATH500 questions, ADR only
sbatch batch_adr.sh pilot configs/adr_qwen_7b.conf

# 2) the comparison on AIME24 (or MATH500)
sbatch batch_adr.sh run configs/adr_qwen_7b.conf,configs/adr_aime24.conf
#    interactive node instead:  ssh <node>; HF_HUB_OFFLINE=1 nohup bash run_adr.sh run configs/adr_qwen_7b.conf,configs/adr_aime24.conf &

# 3) tests
python -m unittest tests/test_adr.py -v
```

**Monitoring.** The log prints a progress line every minute:

```text
aime24 · cell 2/3 B=30000 · method 4/5 adr · 30 solves:  40%|████  | 12/30 [0:12:11<0:18:00] | left=18 in-flight=18 model-calls=640
```

Each cell has `ckpt*.status.json` files with done/left counts.

**Resume.** Everything is saved continuously on scratch:
* finished solves are appended immediately;
* in-flight question states are snapshotted every `ADR_CHECKPOINT_EVERY_S` seconds;
* finished (cell, method) results are skipped.

After a killed job, **run the same command again**. A lock prevents two identical runs.

### Output layout

```text
$ADR_OUTPUT_ROOT/<model>/                       (default $SCRATCH/adr_results/<model>/)
├── experiment.log
└── <dataset>/                                  e.g. aime24/
    ├── config.json, summary_all.json           every cell × method in one table
    ├── prefix/                                 shared structured solutions + first critiques (+ checkpoints)
    ├── cimr_D4/                                original method: results_cimr.jsonl, summary.json
    └── B30000/, B60000/                        one folder per predeclared budget
        ├── results_<method>.jsonl              per question: answers, cost, latency, every turn
        │                                        (feedback, repair ids, |A|, suffix size, fallback, answers before/after)
        ├── summary.json                        metrics per method + paired comparisons vs adr
        └── ckpt/
```

**Success criterion (§7).** At comparable accuracy, `adr` must cost less than **both** `full_rewrite` and `suffix`. Beating only `cimr` would support a simpler implementation change, not the research claim.

---

## Original CIMR pipeline

The untouched original code lives on the `cimr` branch (`git checkout cimr`). On `main` the original scripts still work. They source `.env` for paths.

```bash
bash run_paired_ablation.sh --model_name Qwen/Qwen2.5-Math-7B-Instruct --cimr_n_rollouts 8 --cimr_depth 4 aime24 amc23 math500
bash run_nd_sweep.sh configs/qwen_7b.conf           # N_VALUES × D_VALUES from the conf
sbatch batch_nd_sweep.sh configs/qwen_7b.conf
```

Outputs go under `experiments_results/results_paired/<model>_mnt<T>_t<temp>_p<top_p>/`. `extract_cimr_metrics.py`, `extract_compute_flop.py`, and `export_paper_csvs.py` build the paper's figures and tables from them.

Supported datasets: `aime24`, `aime25`, `amc23`, `math500`, `olympiadbench`, `gsm8k`.
