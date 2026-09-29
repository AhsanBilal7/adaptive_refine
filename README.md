# Refining Over Resampling + Training-Free Procedure RSI

This repository contains two pipelines on top of the same frozen LLM:

1. **Paired-ablation CIMR pipeline** (the original paper, *Refining Over Resampling*): `N` independent rollouts, each refined for `D` rounds of *generate → critique → correct*, then aggregated by plurality vote.
2. **Procedure RSI** (`updated_methodology.md`): an outer loop that improves *the procedure* deciding which rollout to refine, how to refine it, and when to stop. It uses measured action effects from copied states and reusable, guarded two-slot repair plans. All model weights stay frozen.

---

## Project structure

```text
adaptive_scientific_discovery/
├── .env                        machine paths + PREDEFINED experiment variables (sourced by every script)
├── sample.env                  template copy of .env
├── configs/
│   ├── rsi_*.conf              Procedure-RSI per-model overrides (sourced after .env)
│   ├── rsi_smoke.conf          minutes-long end-to-end smoke test
│   ├── rsi_qwen_7b_aime.conf   FULL experiment: Qwen2.5-Math-7B, N=8, 5 rounds, AIME24+25 test
│   ├── rsi_paper_scale.conf    N=8, D=4 overlay (after the pilot)
│   ├── ablations/*.conf        stackable §8 ablation overlays
│   └── <model>[_cric_corr].conf  original N×D sweep configs
├── run_procedure_rsi.py        Procedure-RSI CLI: sanity | splits | develop | evaluate
├── run_procedure_rsi.sh        launcher: sources .env + conf(s), activates venv, logs
├── run_rsi_experiment.sh       full experiment (develop → evaluate), resumable, lock-protected
├── batch_procedure_rsi.sh      Slurm wrapper for the launcher
├── run_paired_ablation.{py,sh} original CIMR driver
├── run_nd_sweep.sh / batch_nd_sweep.sh   original N×D sweep (+ Slurm)
├── src/
│   ├── rsi/                    Procedure RSI (see table below)
│   ├── cimr_*.py               original CIMR pipeline (prompts reused by rsi/)
│   ├── model_wrapper.py        HF wrapper (+ generate_with_usage: realized token counts)
│   ├── dataset_loader.py       benchmarks + math_train / gsm8k_train dev pools
│   └── evaluator_classes.py, inference_methods.py   answer equivalence / grading
└── tests/test_rsi.py           24 unit tests incl. crash/resume (scripted backend, no GPU)
```

---

## Procedure RSI: what is implemented

The procedure is `H_t = (P_t, R_t, M_t)`: editable critique/correction prompts `P_t`, an ordered list of rules and guarded plans `R_t`, and signed evidence memory `M_t`. The generator, critic, corrector, and editor all use the **same frozen model**. The answer extractor, gold grader, voting rule, and budget meter are fixed and cannot be edited.

| Methodology | What the code does | File |
|---|---|---|
| §4.1 initialize | `N` independent problem-only traces; one active trace per slot; only the current version votes; `RESTART` replaces a slot and never adds a vote | `solver.py`, `state.py` |
| §4.2 state / features | Fixed schema: `depth, answer_freq, answer_changes, recent_changes, agrees_plurality, valid, critique` (per slot) and `budget_frac, top_frac, margin, n_distinct_frac, invalid_frac` (global). The critique category comes from a fixed `ERROR CATEGORY:` line (`unknown` until a counted critique exists). No problem id or gold answer is available to rules | `state.py`, `prompts.py` |
| §4.3 actions | `FULL_REFINE` (3 calls), `DIRECT_REPAIR` (2), `RESTART` (1), `KEEP` (0, freezes the slot), `STOP`, `PAIR(m,i,j)` with a continuation guard checked after the first op. The first matching rule wins; ties go to slot order; otherwise H0's next affordable action applies. A decision cap guarantees termination | `procedure.py`, `solver.py` |
| §4.3 compute | Realized input/output tokens are metered per call; FLOPs = 2·P·(T_in+T_out). One predeclared `cost_measure` (tflops/tokens/calls) sets the hard cap `B` and the utility; the other measures are always reported. Each op **reserves** its worst case (capped inputs + `max_new_tokens`) before starting. An incomplete op leaves the slot unchanged but is still charged | `budget.py`, `backends.py`, `solver.py` |
| §4.4 vote | Plurality over normalized answers (paper App. A.5 η); invalid extractions abstain; ties go to the lowest slot index; all-invalid returns `[invalid]`. The same conventions apply to every baseline | `voting.py` |
| §5 action values | Complete states reached by `H_t` are saved; each is forked into incumbent / alternatives / `STOP` on 2 rotating or disagreement-targeted slots, with K seeds and common random numbers. `Q̂ = mean[1{ŷ=y*} − λ·C_future/B]`, `Δ̂ = Q̂(a) − Q̂(a0)`. Untested or infeasible actions stay *unknown* | `counterfactual.py` |
| §5 negative evidence | Every record stores features, action, parent/child answers, terminal vote, future cost, seeds, model/procedure hashes, and recovery/regression flags. Positive, zero, and negative effects are all kept | `memory.py` |
| §5.1 interactions | For ≤2 ordered pairs per state: `Q_∅, Q_a, Q_b, Q_{a→b}` under the same remaining cap. Computes `I = Q_ab − Q_a − Q_b + Q_∅`, the same contrast on success rates (`I_acc`), and `util_gain = Q_ab − max(Q_∅,Q_a,Q_b)` | `counterfactual.py` |
| §5.1 plans | A pair becomes a plan only with reproducible `I>0` (`pair_repro_frac`), `I_acc>0`, `util_gain>0` and support ≥ `min_support`. Plans are compiled into guarded two-slot plans and then re-evaluated as the actual guarded plan | `editor.py` |
| §6 outer loop | Per round: solve-and-log → counterfactuals → memory → patches (deterministic compiler and/or frozen-LLM editor; one patch = add/revise/delete rule, add plan, or edit prompt, and it must cite supporting and contradicting evidence ids) → choose on search data → paired full solve on a **fresh** confirmation block → accept only if the sign-flip test on question-level utility passes at `α_t = α/R` **and** the accuracy lower bound > −margin | `outer_loop.py`, `stats.py` |
| §6 data roles | Search / confirmation / sealed test, deduplicated by problem and number-templated source; template groups never straddle splits; dev problems overlapping the test templates are dropped. The sealed test is written separately and asserted never to enter memory | `splits.py` |
| §3 restrictions | The editor only sees binned evidence (no problem text, ids, or gold). Rules are validated against the feature schema. Prompt patches containing answer lists, dev-problem n-grams, dev gold answers, or lookup language are rejected. Rule, plan, and prompt budgets are declared | `editor.py`, `procedure.py` |
| §8 reporting | Accuracy, solved counts, seed std, utility, calls / tokens / TFLOPs, extraction failures, candidate accuracy, action frequencies, conditional W→C / C→W, vote recoveries / regressions, evidence yield, accepted edits, development-cost ledger | `runner.py`, `counterfactual.py`, `outer_loop.py` |
| §9 sanity | Grader checks on known (in)equivalent answers plus deliberately mutated predictions; `develop` refuses to run if they fail. `evidence_yield` reports how many comparisons actually distinguish actions | `voting.py`, `counterfactual.py` |

**Budget headroom.** Reservation uses worst cases (every call at `max_new_tokens`), so realized H0 cost is typically well below `B`. In the 1.5B smoke run it was ~40% of `B`. Adaptive procedures can spend that headroom, which is why every comparison reports utility (`λ·C/B`) and realized TFLOPs/calls, not accuracy alone. Use `maj_matched` for the equal-cap comparison and the realized cost columns for comparable-realized-cost comparisons (§8).

**H0** is the paper's round-robin `FULL_REFINE` schedule to depth `h0_depth`. With `budget_mode=h0_cycles`, `B` is the worst-case cost of `N` initial samples plus `N·budget_cycles` full refinements (plus the declared prompt-edit allowance), so H0 with `h0_depth ≤ budget_cycles` always completes its schedule.

**Baselines implemented** (identical extraction, vote, and cap): `h0` (original fixed refinement), `heuristic` (fixed hand-written adaptive rules), `maj_matched` (compute-matched majority sampling under the same `B`), and any saved procedure JSON. GEPA, AFlow/Meta-Harness, Reflexion, CoRefine, RSA, LW2S, and C3 are external systems and are **not** included. Run them under their own access regimes.

### Decisive ablations (§8): stack an overlay after a model conf

| Overlay | Compares |
|---|---|
| `ablations/primitive_only.conf` | guarded pair plans vs primitive-only control |
| `ablations/random_plans.conf`, `greedy_vote_plans.conf` | interaction-selected vs random / immediate-vote-greedy plans |
| `ablations/trace_credit.conf` | population-vote credit vs individual-trace credit |
| `ablations/trajectory_evidence.conf` | matched-state experiments vs trajectory summaries |
| `ablations/success_only_memory.conf` | signed vs success-only memory |
| `ablations/prompts_only.conf` | evolved rules vs evolved prompts alone |
| `ablations/no_keep.conf`, `no_restart.conf` | `KEEP` / `RESTART` removed |
| `ablations/reset_procedure.conf` | persistent vs reset procedure, same accumulated evidence |
| `ablations/frozen_after2.conf` | frozen procedure + memory after round 2 vs continued adaptation |

---

## Installation

The project environment is pinned by `pyproject.toml` + `uv.lock` (Python 3.12, torch 2.10+cu128, transformers 4.57.6). Build it on scratch:

```bash
set -a; source .env; set +a          # sets UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, UV_PROJECT_ENVIRONMENT=$VENV_PATH
uv sync --frozen --python 3.12 --no-install-package vllm
```

`--no-install-package vllm` is required on this cluster. The lock pins `vllm==0.19.1`, whose only x86_64 wheel needs glibc ≥ 2.31, and these RHEL8 nodes ship glibc 2.28, so uv falls back to a source build that fails without `CUDA_HOME`. The default `RSI_BACKEND=hf` does not need vllm. `RSI_BACKEND=vllm` is implemented (per-request seeds) for hosts where the wheel installs.

The older `$SCRATCH/uv_envs/train_free_kv` venv is **not** compatible with this code: Python 3.11, torch 2.1.2, transformers 4.44 (`model_wrapper` passes `dtype=`, which needs ≥ 4.56), and no scikit-learn or sentence-transformers.

Compute nodes have no internet. Pre-fetch models and datasets from a login node (for example `huggingface_hub.snapshot_download(...)`), then run with `HF_HUB_OFFLINE=1`.

---

## Configuration: `.env` → `configs/*.conf` → CLI

* **`.env`** holds machine paths (venv, scratch caches, HF caches, logs, output root) and the **predefined experiment variables** (model, decoding, and every `RSI_*` default). Each value is `${VAR:-default}`, so an exported value wins.
* **`configs/*.conf`** are sourced after `.env` and override what they set. Several can be stacked with commas.
* **CLI** flags override both: `--<field> value` or `--set field=value` for any field of `src/rsi/config.py:RSIConfig`.

Every `RSIConfig` field `foo_bar` reads `RSI_FOO_BAR`. `MODEL_NAME`, `BATCH_SIZE`, and `SEED` are shared with the original pipeline. Main knobs:

| Variable | Default | Meaning |
|---|---|---|
| `RSI_N_SLOTS` | 4 | N slots (paper: 8) |
| `RSI_H0_DEPTH`, `RSI_BUDGET_CYCLES` | 2, 2 | H0 depth; cap ≈ two full refinement cycles per slot |
| `RSI_COST_MEASURE` | tflops | predeclared cap/utility measure (`tflops`/`tokens`/`calls`) |
| `RSI_UTILITY_LAMBDA` | 0.2 | λ in `Q = 1{correct} − λ·C/B` |
| `RSI_CF_SEEDS` | 4 | K continuation seeds per tested action |
| `RSI_CF_STATES_PER_PROBLEM`, `RSI_CF_ALT_SLOTS`, `RSI_CF_PAIRS_PER_STATE` | 1, 2, 2 | sampled states, slots per state, ordered pairs per state |
| `RSI_ROUNDS`, `RSI_SEARCH_BATCH_SIZE`, `RSI_CONFIRM_BLOCK_SIZE` | 5, 20, 40 | outer rounds and block sizes |
| `RSI_ALPHA_TOTAL`, `RSI_NONINFERIORITY_MARGIN` | 0.05, 0.02 | α split as α/R; accuracy margin |
| `RSI_EDITOR_MODE` | both | `compiler`, `llm`, or `both` |
| `RSI_REWRITE_TEMPERATURE` | 0.7 | sampled rewrite as in the paper's equations; `0` reproduces the old greedy CIMR code path |
| `RSI_DEV_DATASETS`, `RSI_TEST_DATASETS` | math_train, math500 | development pool / sealed evaluation |

---

## How to run

### Step 0: once per machine

```bash
cd /home/u.ab353540/adaptive_scientific_discovery
set -a; source .env; set +a
uv sync --frozen --python 3.12 --no-install-package vllm      # builds $VENV_PATH (see Installation)
```

Compute nodes are offline. Cache the model and datasets from a **login node**:

```bash
source $VENV_PATH/bin/activate
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-Math-7B-Instruct')"
python -c "import sys; sys.path.insert(0,'.'); from src.dataset_loader import load_dataset_split as l; [l(x) for x in ['math_train','math500','aime24','aime25','amc23','olympiadbench']]"
```

### Step 1: pick a node that has free GPU memory

```bash
squeue -u $USER                     # NODELIST of your RUNNING allocations
ssh ac055                           # enter one of them
nvidia-smi                          # check free memory (7B bf16 needs ~15 GB + KV cache)
cd /home/u.ab353540/adaptive_scientific_discovery
```

### Step 2: run (resumable)

```bash
# the FULL AIME experiment: develop (5 rounds on MATH-train Level 5) → evaluate h0 / maj_matched / heuristic / final on AIME24+25
HF_HUB_OFFLINE=1 nohup bash run_rsi_experiment.sh configs/rsi_qwen_7b.conf,configs/rsi_qwen_7b_aime.conf \
    > logs/aime_N8_qwen7b_experiment.out 2>&1 &

# or as its own Slurm job (not tied to another allocation's time limit; resubmit to resume)
sbatch batch_procedure_rsi.sh experiment configs/rsi_qwen_7b.conf,configs/rsi_qwen_7b_aime.conf
```

Other entry points:

```bash
bash run_procedure_rsi.sh sanity                                          # grader checks, no GPU
bash run_procedure_rsi.sh develop  configs/rsi_smoke.conf                 # minutes-long smoke test
bash run_procedure_rsi.sh develop  configs/rsi_qwen_7b.conf,configs/rsi_qwen_7b_sample_N8.conf   # small N=8 sample
bash run_procedure_rsi.sh evaluate configs/rsi_qwen_1b.conf -- --methods h0,maj_matched,heuristic,final --eval_seeds 3
bash run_procedure_rsi.sh evaluate configs/rsi_qwen_1b.conf -- --methods h0,final --sealed
bash run_procedure_rsi.sh develop  configs/rsi_qwen_1b.conf,configs/ablations/primitive_only.conf  # an ablation arm
python run_procedure_rsi.py develop --env_file .env --conf configs/rsi_qwen_1b.conf --cf_seeds 2  # direct python
python -m unittest tests/test_rsi.py -v                                   # unit tests
```

### Step 3: monitor

```bash
M=$SCRATCH/procedure_rsi_results/Qwen2.5-Math-7B-Instruct
cat $M/development/status.json                         # current round / phase
cat $M/development/rounds/round_*/ckpt_*.status.json   # solves done / left in the current phase
grep progress logs/aime_N8_qwen7b_experiment.out | tail -3
```

The log prints a tqdm-style progress line every minute (a live tqdm bar on an interactive terminal):

```text
Round 1/5 · phase 2/5 · counterfactual:  33%|██████▋             | 312/932 [1:02:11<2:04:00, 5.0solves/min] | left=620 in-flight=256 model-calls=6,480
```

* **Round r/5**: outer-loop round.
* **phase p/5**: 1 solve-and-log, 2 test alternatives (counterfactual), 3 propose edits, 4 score on search, 5 confirm.
* **312/932**: solves finished / total in this phase.
* **left**: solves still to finish.
* **in-flight**: solves currently active.
* **model-calls**: generations made in this session.

Evaluation lines read `Eval dataset 1/2 aime24 · method 2/4 maj_matched · 90 solves: …`.

### Output layout (one folder per model, one folder per test dataset)

```text
$RSI_OUTPUT_ROOT/<model>/                 (default $SCRATCH/procedure_rsi_results/<model>/)
├── final_procedure.json                  the model's frozen procedure (written when development ends)
├── experiment.log                        full log of run_rsi_experiment.sh
├── eval_summary.json                     every dataset × method in one table
├── development/                          everything used to BUILD the procedure
│   ├── config.json, status.json, splits.json, dev_problems.jsonl, sealed_test/
│   ├── memory.jsonl, lineage.json, dev_ledger.json, procedures/HXXX_<hash>.json
│   ├── rounds/round_XX.json              per-round summary (evidence, patches, search scores, confirmation test)
│   └── rounds/round_XX/                  base_procedure.json, log.pkl, evidence.jsonl, proposals.pkl,
│                                         search.pkl, confirm.pkl, ckpt_*.{done,inflight}.pkl, ckpt_*.status.json
├── aime24/                               results of every method on AIME24
│   ├── results_h0.jsonl, results_maj_matched.jsonl, results_heuristic.jsonl, results_final.jsonl
│   ├── summary.json, config.json, ckpt/
├── aime25/  math500/  olympiadbench/ ... same structure
└── runs/<RSI_RUN_NAME>/                  smoke / sample / ablation runs, same inner layout
```

The main run of a model has no `RSI_RUN_NAME` and writes directly to `<model>/`. Any named run (e.g. `configs/ablations/*.conf`, `rsi_smoke.conf`) goes to `<model>/runs/<name>/`, so it never overwrites the main procedure. To evaluate the model's procedure on more benchmarks later:

```bash
bash run_procedure_rsi.sh evaluate configs/rsi_qwen_7b.conf,configs/rsi_qwen_7b_aime.conf -- \
    --methods h0,maj_matched,heuristic,final --datasets math500,olympiadbench,amc23
```

Each dataset lands in its own folder; finished (dataset, method) results are skipped.

### Continuous saving and resume

Everything is written **continuously** under `<model>/`. When the allocation ends or the job is killed, **run the same command again**. The run picks up where it stopped:

* **Finished solves** are appended to `*.done.pkl` as soon as they complete and are never re-run.
* **In-flight solves** save their full state (slots, histories, charged compute, decisions) to `*.inflight.pkl`, at most every `RSI_CHECKPOINT_EVERY_S` seconds (default 60). They restart from that state. An operation cut off mid-way is treated as an incomplete operation: its compute stays charged, the slot is unchanged, and the decision is made again.
* **Scheduling.** The scheduler uses continuous batching. Each GPU chunk (`BATCH_SIZE` requests) advances its solves immediately, so a checkpoint opportunity comes every few minutes even when hundreds of solves are active.
* **Each outer-loop phase** saves its output once. The branch plan of the counterfactual phase is fixed on disk, and evidence memory is rebuilt from per-round files (never double-counted).
* **`evaluate`** skips finished (dataset, method) results. It re-runs a method only if its procedure changed (hash) or the number of seeds changed.
* **Locking.** `run_rsi_experiment.sh` holds a lock file, so two copies of the same experiment never run at once.

### Cost to expect (measured)

Rates measured for Qwen2.5-Math-7B, N=8, HF backend, on an H100 shared with another job: about **13–25 model calls per minute**.

| Phase (one round, default AIME config) | ≈ model calls | ≈ time |
|---|---|---|
| 1 solve-and-log (20 problems) | 1,100 | 45–60 min |
| 2 test alternatives (≈930 continuations) | 22,000 | 18–28 h |
| 3 propose edits | few | minutes |
| 4 score H + ≤8 candidates on search | 10,000 | 8–13 h |
| 5 confirmation (only if a candidate wins on search) | 9,000 | 7–11 h |
| **one round** | **~40,000** | **~1.5–2 days** |
| **5 rounds → `final_procedure.json`** | **~200,000** | **~7–10 GPU-days** |

Ways to cut this, all expressible as conf values:
* a dedicated (unshared) GPU: roughly 2×
* `RSI_CF_SEEDS=2`: halves phase 2
* `RSI_CF_ALT_OPS=DIRECT_REPAIR,RESTART,KEEP`: fewer branches per state
* `RSI_EDITOR_NUM_PROPOSALS=2`: fewer candidates to score in phase 4
* `RSI_CONFIRM_SEEDS=1`: halves phase 5
* shorter generations (`RSI_*_MAX_NEW_TOKENS`)
* a vLLM backend (typically ≥5× faster), which needs a host or container with glibc ≥ 2.31

`dev_ledger.json` reports the development compute actually spent, separately from inference cost.

### Caveats

* The HF backend has one global RNG per batch chunk, so per-request seeds are best effort. Branches share seed bases (common random numbers), but exact replay depends on batch composition. The vLLM backend honours per-request seeds.
* Critique categories are model assertions, not correctness labels. Correct final answers are not verified rationales.
* Each solve is sequential (one decision at a time). Throughput comes from batching many solves and branches together (`scheduler.py`).

---

## Original paired-ablation pipeline

Unchanged in behaviour. The scripts now source `.env` for paths and defaults instead of hard-coded `/home/ahsan/crisp`, and `uv run --frozen --no-sync` uses the `crisp` venv.

```bash
bash run_paired_ablation.sh --model_name Qwen/Qwen2.5-Math-7B-Instruct --cimr_n_rollouts 8 --cimr_depth 4 aime24 amc23 math500
bash run_nd_sweep.sh configs/qwen_7b.conf          # loops N_VALUES × D_VALUES from the conf
sbatch batch_nd_sweep.sh configs/qwen_7b.conf
```

Per dataset it writes `logs/paired_<dataset>_N<N>_d<D>.log`, `summary_paired_*.json`, `results_paired_*.jsonl`, `analysis_paired_*.json`, and the resumable CIMR store `cimr_full_results_<dataset>_N<N>_d<D>.json` under `experiments_results/results_paired/<model_short>_mnt<T>_t<temp>_p<top_p>/`. `extract_cimr_metrics.py`, `extract_compute_flop.py`, and `export_paper_csvs.py` build the paper's figures and tables from those stores.

Supported datasets: `aime24`, `aime25`, `amc23`, `math500`, `olympiadbench`, `gsm8k` (plus `math_train` / `gsm8k_train` as development pools for Procedure RSI).
