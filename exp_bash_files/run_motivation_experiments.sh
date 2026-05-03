#!/usr/bin/env bash
# =============================================================================
# run_motivation_experiments.sh
# =============================================================================
# Runs all experiments needed to motivate DN-BoN and then analyses the
# results for all 5 motivation experiments.
#
# What this script does
# ---------------------
# PHASE 1 — Generation (if SKIP_GEN=0):
#   Run methods for each dataset × each N in N_LIST.
#   Each (dataset, method, N) is ONE separate python call.
#
# PHASE 2 — Analysis:
#   Run analyse_motivation.py on the generated JSONL files.
#   Produces 6 CSV files in ANALYSIS_OUT/.
#
# Usage
# -----
#   bash run_motivation_experiments.sh <datasets> <methods>
#
#   <datasets> : comma-separated list or "all"
#     Valid: math500, aime24, aime25, amc23, gsm8k,
#            mmlu, bbh, arc_challenge, truthfulqa, strategyqa
#     "all"  expands to all of the above
#
#   <methods>  : comma-separated list or "all"
#     Valid: dn_bon, dn_bon_updated, bon, bon_majority, greedy, greedy_pass_n
#     "all"  expands to: dn_bon,dn_bon_updated,bon,bon_majority
#     Note: dn_bon is required for the 5 motivation experiments.
#           bon and bon_majority are required for comparisons.
#           dn_bon_updated enables direct DORA vs DN-BoN comparison.
#
#   To skip generation and only run analysis on existing JSONL files:
#     SKIP_GEN=1 bash run_motivation_experiments.sh <datasets> <methods>
#
# Examples
# --------
#   # ── single dataset, single method ──────────────────────────────────
#   bash run_motivation_experiments.sh aime24 dn_bon
#   bash run_motivation_experiments.sh math500 bon
#
#   # ── single dataset, multiple methods ───────────────────────────────
#   bash run_motivation_experiments.sh aime24 dn_bon,bon,bon_majority
#   bash run_motivation_experiments.sh math500 dn_bon,dn_bon_updated,bon,bon_majority
#
#   # ── single dataset, ALL methods ────────────────────────────────────
#   bash run_motivation_experiments.sh aime24 all
#   bash run_motivation_experiments.sh math500 all
#
#   # ── multiple datasets, single method ───────────────────────────────
#   bash run_motivation_experiments.sh aime24,aime25 dn_bon
#   bash run_motivation_experiments.sh math500,amc23 bon,dn_bon
#
#   # ── multiple datasets, multiple methods ────────────────────────────
#   bash run_motivation_experiments.sh aime24,aime25 dn_bon,bon,bon_majority
#   bash run_motivation_experiments.sh math500,aime24,aime25 dn_bon,dn_bon_updated,bon,bon_majority
#
#   # ── ALL datasets, ALL methods ───────────────────────────────────────
#   bash run_motivation_experiments.sh all all
#
#   # ── ALL datasets, specific methods ─────────────────────────────────
#   bash run_motivation_experiments.sh all dn_bon,bon
#   bash run_motivation_experiments.sh all dn_bon,dn_bon_updated
#
#   # ── specific datasets, ALL methods ─────────────────────────────────
#   bash run_motivation_experiments.sh aime24,aime25 all
#   bash run_motivation_experiments.sh mmlu,bbh,arc_challenge all
#
#   # ── general reasoning datasets ──────────────────────────────────────
#   bash run_motivation_experiments.sh mmlu,bbh,arc_challenge,truthfulqa,strategyqa dn_bon,bon,bon_majority
#
#   # ── skip generation, re-run analysis only ──────────────────────────
#   SKIP_GEN=1 bash run_motivation_experiments.sh aime24,aime25 dn_bon,bon,bon_majority
#
# Output layout
# -------------
#   ${RESULTS_DIR}/
#     dn_bon_math500_N4.jsonl
#     dn_bon_math500_N8.jsonl
#     ...
#   ${ANALYSIS_OUT}/
#     summary_stats.csv
#     exp1_redundancy_vs_budget.csv
#     exp2_accuracy_by_redundancy_bin.csv
#     exp3_majority_cluster_analysis.csv
#     exp4_score_gap_vs_redundancy.csv
#     exp5_oracle_cluster_analysis.csv
# =============================================================================

set -euo pipefail


# =============================================================================
# CONFIG
# =============================================================================

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
PYTHON="python"

# N values to sweep — Exp 1 needs at least 3 values to show the budget curve
N_LIST="2 4 8 16 32"

# Generation parameters
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.95
LM_BATCH_SIZE=32
DTYPE="auto"
LOAD_4BIT=0
LOAD_8BIT=0
GPUS=""              # e.g. "0,1" for multi-GPU; "" = all visible

# Output directories
BASE_OUT="./experiments_results/results_updated"
ANALYSIS_OUT="./motivation_analysis"

# HuggingFace cache
HF_CACHE="/scratch/${USER}/hf_cache"

# Verifier for BoN / DN-BoN candidate selection
# -----------------------------------------------
# MUST match the verifier used in your main experiments.
# Using a different verifier here would mean the motivation experiments
# operate under a different scoring regime than the method being motivated.
#
# prm     — Process Reward Model (Qwen2.5-Math-PRM-7B)
#            Continuous per-step quality score.  This is what the main
#            method uses, so use it here too for consistency.
#            Makes Exp 4 (score gap analysis) meaningful: gaps are
#            real-valued rather than binary 0/1.
# outcome — Binary correctness vs gold answer.  Simpler but inconsistent
#            with PRM-based experiments.
# length  — Proxy heuristic; not recommended for motivation experiments.
VERIFIER="prm"

# PRM model — only used when VERIFIER=prm
# Recommended: Qwen/Qwen2.5-Math-PRM-7B  (~14 GB fp16)
# Must be the SAME model used in your main experiments.
PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B"

# PRM aggregation: how to reduce per-step scores to one trace score
#   min   — most conservative; penalises any weak reasoning step
#   mean  — average quality across all steps
#   last  — quality of the final step only
# Use the same aggregation as your main experiments.
PRM_AGGREGATION="mean"

EMBEDDING_MODEL="all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD=0.85
CLUSTERING_METHOD="agglomerative"

# DORA temperatures — only used by dn_bon_updated; ignored by all other methods.
# Must match the values used in your main run_experiments.sh for a fair comparison.
#   T_B — quality softmax temperature (default 1.0)
#   T_S — similarity softmax temperature (default 0.1)
T_B=1.0
T_S=0.1

# Number of bins for Exp 2 and Exp 4 (3 = low/medium/high)
N_BINS=3

# =============================================================================
# END CONFIG
# =============================================================================


# =============================================================================
# CLI argument parsing
# $1 = datasets  (comma-separated or "all"; default "all")
# $2 = methods   (comma-separated or "all"; default "all")
# =============================================================================

DATASETS_ARG="${1:-all}"
METHODS_ARG="${2:-all}"

# Canonical full lists
ALL_DATASETS="math500 aime24 aime25 amc23 gsm8k mmlu bbh arc_challenge truthfulqa strategyqa"
# Default "all" for methods means the core motivation set
ALL_METHODS="dn_bon dn_bon_updated bon bon_majority"

# Expand "all"
if [[ "${DATASETS_ARG}" == "all" ]]; then
    DATASETS="${ALL_DATASETS}"
else
    DATASETS="${DATASETS_ARG//,/ }"
fi

if [[ "${METHODS_ARG}" == "all" ]]; then
    METHODS="${ALL_METHODS}"
else
    METHODS="${METHODS_ARG//,/ }"
fi


# =============================================================================
# Validation
# =============================================================================

_valid_datasets="math500 aime24 aime25 amc23 gsm8k mmlu bbh arc_challenge truthfulqa strategyqa olympiadbench"
_valid_methods="dn_bon dn_bon_updated bon bon_majority greedy greedy_pass_n"

_check_dataset() {
    local d="$1"
    if ! echo "${_valid_datasets}" | grep -qw "${d}"; then
        echo "ERROR: unknown dataset '${d}'"
        echo "Valid datasets: ${_valid_datasets}"
        exit 1
    fi
}

_check_method() {
    local m="$1"
    if ! echo "${_valid_methods}" | grep -qw "${m}"; then
        echo "ERROR: unknown method '${m}'"
        echo "Valid methods: ${_valid_methods}"
        exit 1
    fi
}

for D in ${DATASETS}; do _check_dataset "${D}"; done
for M in ${METHODS};  do _check_method  "${M}"; done


# =============================================================================
# Per-dataset sample caps
# =============================================================================
_max_samples() {
    case "$1" in
        aime24|aime25) echo 32  ;;
        amc23)         echo 40  ;;
        math500|gsm8k) echo 500 ;;
        mmlu)          echo 500 ;;
        bbh)           echo 500 ;;
        arc_challenge) echo 500 ;;
        truthfulqa)    echo 500 ;;
        strategyqa)    echo 500 ;;
        *)             echo 500 ;;
    esac
}


# =============================================================================
# Setup
# =============================================================================

# Build result folder name — encodes model, token budget, sampling params, verifier
# so PRM-scored and outcome-scored results land in separate folders and never mix.
model_short="${MODEL##*/}"
model_short="${model_short//[: ]/_}"
RUN_TAG="${model_short}_mnt${MAX_NEW_TOKENS}_t${TEMPERATURE}_p${TOP_P}"
[[ "${LOAD_4BIT}" == "1" ]] && RUN_TAG+="_4bit"
[[ "${LOAD_8BIT}" == "1" ]] && RUN_TAG+="_8bit"
RESULTS_DIR="${BASE_OUT}/${RUN_TAG}"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${ANALYSIS_OUT}"

NOW=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

# HuggingFace cache
if [[ -n "${HF_CACHE}" ]]; then
    export HF_HOME="${HF_CACHE}"
    export TRANSFORMERS_CACHE="${HF_CACHE}"
    export HF_DATASETS_CACHE="${HF_CACHE}"
    export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
    mkdir -p "${HF_CACHE}"
fi

# GPU
if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    echo "[gpu] CUDA_VISIBLE_DEVICES=${GPUS}"
fi

SKIP_GEN="${SKIP_GEN:-0}"


# =============================================================================
# Common flags for run_eval.py
# =============================================================================
common_flags() {
    local flags=""
    flags+=" --model ${MODEL}"
    flags+=" --batch_size ${LM_BATCH_SIZE}"
    flags+=" --max_new_tokens ${MAX_NEW_TOKENS}"
    flags+=" --temperature ${TEMPERATURE}"
    flags+=" --top_p ${TOP_P}"
    flags+=" --dtype ${DTYPE}"
    flags+=" --verifier ${VERIFIER}"
    flags+=" --embedding_model ${EMBEDDING_MODEL}"
    flags+=" --similarity_threshold ${SIMILARITY_THRESHOLD}"
    flags+=" --clustering_method ${CLUSTERING_METHOD}"
    # DORA temperatures — only consumed by dn_bon_updated; harmlessly passed to others
    flags+=" --T_b ${T_B}"
    flags+=" --T_s ${T_S}"
    [[ -n "${HF_CACHE}" ]]         && flags+=" --hf_cache_dir ${HF_CACHE}"
    [[ "${LOAD_4BIT}" == "1" ]]    && flags+=" --load_in_4bit"
    [[ "${LOAD_8BIT}" == "1" ]]    && flags+=" --load_in_8bit"
    [[ -n "${GPUS}" ]]             && flags+=" --gpus ${GPUS}"
    [[ "${VERIFIER}" == "prm" ]]   && flags+=" --prm_model ${PRM_MODEL}"
    [[ "${VERIFIER}" == "prm" ]]   && flags+=" --prm_aggregation ${PRM_AGGREGATION}"
    echo "${flags}"
}


# =============================================================================
# Single experiment runner
# =============================================================================
run_exp() {
    local tag="$1"; shift
    local log_file="${LOG_DIR}/${tag}_${NOW}.log"
    echo ""
    echo "------------------------------------------------------------"
    echo " START : ${tag}"
    echo " LOG   : ${log_file}"
    echo "------------------------------------------------------------"
    ${PYTHON} run_eval.py "$@" 2>&1 | tee "${log_file}"
    local ec=${PIPESTATUS[0]}
    [[ ${ec} -eq 0 ]] && echo "  [DONE]  ${tag}" \
                      || echo "  [FAIL]  ${tag}  (exit ${ec})"
}


# =============================================================================
# PHASE 1 — Generation
# =============================================================================

if [[ "${SKIP_GEN}" == "1" ]]; then
    echo ""
    echo "SKIP_GEN=1 — skipping generation phase, using existing JSONL files."
    echo "Results dir: ${RESULTS_DIR}"
else

    # Count total runs for progress display
    N_DS=0; for D in ${DATASETS}; do N_DS=$((N_DS+1)); done
    N_N=0;  for N in ${N_LIST};   do N_N=$((N_N+1));   done
    N_ME=0; for M in ${METHODS};  do N_ME=$((N_ME+1));  done
    TOTAL=$((N_DS * N_N * N_ME))

    echo ""
    echo "============================================================"
    echo "  MOTIVATION EXPERIMENTS — GENERATION PHASE"
    echo "  Model    : ${MODEL}"
    echo "  Datasets : ${DATASETS}"
    echo "  N values : ${N_LIST}"
    echo "  Methods  : ${METHODS}"
    echo "  Total    : ${TOTAL} python calls  (${N_DS}ds × ${N_N}N × ${N_ME}methods)"
    echo "  Output   : ${RESULTS_DIR}/"
    echo "============================================================"

    RUN_IDX=0

    for DATASET in ${DATASETS}; do
        MAX_S="$(_max_samples "${DATASET}")"

        for N in ${N_LIST}; do
            FLAGS="$(common_flags)"

            for METHOD in ${METHODS}; do
                RUN_IDX=$((RUN_IDX+1))
                TAG="${METHOD}_${DATASET}_N${N}"
                echo ""
                echo "  [${RUN_IDX}/${TOTAL}]  dataset=${DATASET}  method=${METHOD}  N=${N}"

                if [[ "${METHOD}" == "dn_bon" ]]; then
                    # DN-BoN (hard cluster penalty) — requires --alpha
                    run_exp "${TAG}" \
                        ${FLAGS} \
                        --dataset     "${DATASET}" \
                        --methods     dn_bon \
                        --N           "${N}" \
                        --alpha       0.5 \
                        --max_samples "${MAX_S}" \
                        --output_dir  "${RESULTS_DIR}"

                elif [[ "${METHOD}" == "dn_bon_updated" ]]; then
                    # DN-BoN Updated (DORA) — no --alpha; T_b/T_s come from common_flags
                    run_exp "${TAG}" \
                        ${FLAGS} \
                        --dataset     "${DATASET}" \
                        --methods     dn_bon_updated \
                        --N           "${N}" \
                        --max_samples "${MAX_S}" \
                        --output_dir  "${RESULTS_DIR}"

                else
                    # bon, bon_majority, greedy, greedy_pass_n — no extra flags
                    run_exp "${TAG}" \
                        ${FLAGS} \
                        --dataset     "${DATASET}" \
                        --methods     "${METHOD}" \
                        --N           "${N}" \
                        --max_samples "${MAX_S}" \
                        --output_dir  "${RESULTS_DIR}"
                fi

            done   # METHOD
        done   # N
    done   # DATASET

    echo ""
    echo "============================================================"
    echo "  Generation complete.  JSONL files in: ${RESULTS_DIR}/"
    echo "============================================================"

fi   # SKIP_GEN


# =============================================================================
# PHASE 2 — Analysis
# =============================================================================

echo ""
echo "============================================================"
echo "  MOTIVATION EXPERIMENTS — ANALYSIS PHASE"
echo "  Reading  : ${RESULTS_DIR}/"
echo "  Writing  : ${ANALYSIS_OUT}/"
echo "============================================================"

${PYTHON} analyse_motivation.py \
    --results_dir "${RESULTS_DIR}" \
    --model       "${MODEL}" \
    --out_dir     "${ANALYSIS_OUT}" \
    --n_bins      "${N_BINS}" \
    --exps        1 2 3 4 5 summary \
    2>&1 | tee "${LOG_DIR}/analyse_motivation_${NOW}.log"


# =============================================================================
# Print output summary
# =============================================================================

echo ""
echo "============================================================"
echo "  ALL DONE"
echo ""
echo "  Datasets ran  : ${DATASETS}"
echo "  Methods ran   : ${METHODS}"
echo "  JSONL results : ${RESULTS_DIR}/"
echo "  Analysis CSVs : ${ANALYSIS_OUT}/"
echo ""
echo "  CSV files:"
for f in "${ANALYSIS_OUT}"/*.csv; do
    [[ -f "$f" ]] && printf "    %-50s  %s bytes\n" \
        "$(basename "$f")" "$(wc -c < "$f")"
done
echo ""
echo "  To re-run analysis only (reuse existing JSONL):"
echo "    SKIP_GEN=1 bash run_motivation_experiments.sh ${DATASETS_ARG} ${METHODS_ARG}"
echo ""
echo "  To analyse a different results folder:"
echo "    python analyse_motivation.py \\"
echo "        --results_dir ${RESULTS_DIR} \\"
echo "        --model ${MODEL} \\"
echo "        --out_dir ${ANALYSIS_OUT}"
echo "============================================================"






# # ── single dataset, single method ────────────────────────────────────
# bash run_motivation_experiments.sh aime24 dn_bon
# bash run_motivation_experiments.sh math500 bon

# # ── single dataset, multiple methods ─────────────────────────────────
# bash run_motivation_experiments.sh aime24 dn_bon,bon,bon_majority
# bash run_motivation_experiments.sh math500 dn_bon,dn_bon_updated,bon,bon_majority

# # ── single dataset, ALL methods ──────────────────────────────────────
# bash run_motivation_experiments.sh aime24 all
# bash run_motivation_experiments.sh math500 all

# # ── multiple datasets, multiple methods ──────────────────────────────
# bash run_motivation_experiments.sh aime24,aime25 dn_bon,bon,bon_majority
# bash run_motivation_experiments.sh math500,aime24,aime25 dn_bon,dn_bon_updated,bon,bon_majority

# # ── ALL datasets, ALL methods ─────────────────────────────────────────
# bash run_motivation_experiments.sh all all

# # ── ALL datasets, specific methods ───────────────────────────────────
# bash run_motivation_experiments.sh all dn_bon,bon
# bash run_motivation_experiments.sh all dn_bon,dn_bon_updated

# # ── general reasoning benchmarks ─────────────────────────────────────
# bash run_motivation_experiments.sh mmlu,bbh,arc_challenge,truthfulqa,strategyqa dn_bon,bon,bon_majority

# # ── skip generation, re-run analysis only ────────────────────────────
# SKIP_GEN=1 bash run_motivation_experiments.sh aime24,aime25 dn_bon,bon,bon_majority