#!/usr/bin/env bash
# =============================================================================
# run_motivation_experiments.sh
# =============================================================================
# Runs all experiments needed to motivate DN-BoN and then analyses the
# results for all 5 motivation experiments.
#
# What this script does
# ---------------------
# PHASE 1 — Generation (if --skip_gen is not set):
#   Run dn_bon, bon, bon_majority for each dataset × each N in N_LIST.
#   Each (dataset, method, N) is ONE separate python call.
#
# PHASE 2 — Analysis:
#   Run analyse_motivation.py on the generated JSONL files.
#   Produces 6 CSV files in ANALYSIS_OUT/.
#
# Usage
# -----
#   bash run_motivation_experiments.sh               # generate + analyse
#   SKIP_GEN=1 bash run_motivation_experiments.sh    # analyse only (reuse existing)
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

# Datasets to run
DATASETS="math500 aime24 aime25 amc23 gsm8k"

# N values to sweep — Exp 1 needs at least 3 values to show the budget curve
N_LIST="4 8 16 32"

# Methods to run for each (dataset, N)
#   dn_bon       — required for all 5 experiments
#   bon          — required for Exp 1, 2, 3, 4, 5 comparisons
#   bon_majority — required for Exp 1, 2, 3 comparisons
METHODS="dn_bon bon bon_majority"

# Generation parameters
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.95
LM_BATCH_SIZE=64
DTYPE="auto"
LOAD_4BIT=0
LOAD_8BIT=0
GPUS=""              # e.g. "0,1" for multi-GPU; "" = all visible

# Output directories
BASE_OUT="./results"
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
PRM_AGGREGATION="min"

EMBEDDING_MODEL="all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD=0.85
CLUSTERING_METHOD="agglomerative"

# Number of bins for Exp 2 and Exp 4 (3 = low/medium/high)
N_BINS=3

# =============================================================================
# END CONFIG
# =============================================================================


# =============================================================================
# Setup
# =============================================================================

# Build result folder name — encodes model, token budget, sampling params, verifier
# so PRM-scored and outcome-scored results land in separate folders and never mix.
model_short="${MODEL##*/}"
model_short="${model_short//[: ]/_}"
RUN_TAG="${model_short}_mnt${MAX_NEW_TOKENS}_t${TEMPERATURE}_p${TOP_P}_v${VERIFIER}"
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
# Per-dataset sample caps
# =============================================================================
_max_samples() {
    case "$1" in
        aime24|aime25) echo 32  ;;
        amc23)         echo 40  ;;
        math500|gsm8k) echo 500 ;;
        *)             echo 500 ;;
    esac
}

_check_dataset() {
    case "$1" in
        math500|aime24|aime25|amc23|gsm8k) return 0 ;;
        *) echo "ERROR: unknown dataset '$1'"; exit 1 ;;
    esac
}


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

    # Count total runs
    N_DS=0; for D in ${DATASETS};  do N_DS=$((N_DS+1)); done
    N_N=0;  for N in ${N_LIST};    do N_N=$((N_N+1)); done
    N_ME=0; for M in ${METHODS};   do N_ME=$((N_ME+1)); done
    TOTAL=$((N_DS * N_N * N_ME))

    echo ""
    echo "============================================================"
    echo "  MOTIVATION EXPERIMENTS — GENERATION PHASE"
    echo "  Model   : ${MODEL}"
    echo "  Datasets: ${DATASETS}"
    echo "  N values: ${N_LIST}"
    echo "  Methods : ${METHODS}"
    echo "  Total   : ${TOTAL} python calls"
    echo "  Output  : ${RESULTS_DIR}/"
    echo "============================================================"

    RUN_IDX=0

    for DATASET in ${DATASETS}; do
        _check_dataset "${DATASET}"
        MAX_S="$(_max_samples "${DATASET}")"

        for N in ${N_LIST}; do
            FLAGS="$(common_flags)"

            for METHOD in ${METHODS}; do
                RUN_IDX=$((RUN_IDX+1))
                TAG="${METHOD}_${DATASET}_N${N}"
                echo ""
                echo "  [${RUN_IDX}/${TOTAL}]  dataset=${DATASET}  method=${METHOD}  N=${N}"

                if [[ "${METHOD}" == "dn_bon" ]]; then
                    run_exp "${TAG}" \
                        ${FLAGS} \
                        --dataset     "${DATASET}" \
                        --methods     dn_bon \
                        --N           "${N}" \
                        --alpha       0.5 \
                        --max_samples "${MAX_S}" \
                        --output_dir  "${RESULTS_DIR}"
                else
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
echo "  JSONL results  : ${RESULTS_DIR}/"
echo "  Analysis CSVs  : ${ANALYSIS_OUT}/"
echo ""
echo "  CSV files:"
for f in "${ANALYSIS_OUT}"/*.csv; do
    [[ -f "$f" ]] && printf "    %-50s  %s bytes\n" \
        "$(basename "$f")" "$(wc -c < "$f")"
done
echo ""
echo "  To re-run analysis only (reuse existing JSONL):"
echo "    SKIP_GEN=1 bash run_motivation_experiments.sh"
echo ""
echo "  To analyse a different results folder:"
echo "    python analyse_motivation.py \\"
echo "        --results_dir ${RESULTS_DIR} \\"
echo "        --model ${MODEL} \\"
echo "        --out_dir ${ANALYSIS_OUT}"
echo "============================================================"