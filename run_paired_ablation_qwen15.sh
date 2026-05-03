#!/usr/bin/env bash
# =============================================================================
# run_paired_ablation.sh
# =============================================================================
# Runs the paired ablation harness with the new rollout-based CIMR pipeline.
#
# DESIGN PRINCIPLE
# ─────────────────
#   1. Generate N traces once   → sample_cache_<tag>.jsonl
#   2. Score with PRM once      → stored in the same cache file
#   3. BASELINES from CACHE     → BoN / Majority / Oracle / DN-BoN(α, τ)
#   4. CIMR LIVE per question   → N rollouts × d steps each:
#                                   step k:  generate → critique → correct
#                                   step k's corrected output → step k+1
#                                 Final answer of each rollout = step d's
#                                 corrected answer.  Aggregate across N
#                                 rollouts via:
#                                   * majority vote
#                                   * max PRM (Best-of-N over corrected)
#                                 Both predictions and accuracies saved.
#
# RESUME SUPPORT
# ─────────────────
#   * BASELINES skip generation when sample cache is found.
#   * CIMR persists after EVERY step into
#       cimr_full_results_<dataset>_N<N>_d<D>.json
#     so reruns automatically resume per-question, per-rollout, per-step.
#
# USAGE
# ─────
#   bash run_paired_ablation.sh                   # all datasets
#   bash run_paired_ablation.sh aime24            # one dataset
#   bash run_paired_ablation.sh math500 aime24    # multiple
#   LOAD_CACHE=1 bash run_paired_ablation.sh      # force --load_cache
#
# OUTPUTS (per dataset)
# ─────────────────────
#   $OUT/sample_cache_<tag>.jsonl                          raw traces + PRM
#   $OUT/results_paired_<tag>.jsonl                        per-Q baselines
#   $OUT/summary_paired_<tag>.json                         accuracy summary
#   $OUT/analysis_paired_<tag>.json                        per-Q meta
#   $OUT/cimr_full_results_<dataset>_N<N>_d<D>.json        full CIMR log
#   $OUT/figures/, $OUT/logs/
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Generator LLM
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
# MODEL="Qwen/Qwen2.5-Math-7B-Instruct"

# PRM verifier
VERIFIER_MODEL="Qwen/Qwen2.5-Math-PRM-7B"

# Sentence encoder for clustering
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Generation settings (baselines)
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.95

# Sample budget for baseline cache
N_DEFAULT=8

# Ablation sweeps
ALPHA_VALUES="0.0 0.25 0.5 0.75 1.0"
TAU_VALUES="0.70 0.75 0.80 0.85 0.90 0.95"

# Batch size for baselines / cache build
BATCH_SIZE=32

# ─────────────────────────────────────────────────────────────────────────────
# CIMR — rollout-based step-wise pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline:
#   for rollout i in 1..N_ROLLOUTS:
#       for step k in 1..DEPTH:
#           generate reasoning  given (question, prev_corrected_step)
#           self-critique
#           self-correction
#           PRM-score the corrected trace
#       rollout_i.final_answer    = step DEPTH's corrected answer
#       rollout_i.final_prm_score = step DEPTH's corrected PRM
#   pred_majority = mode of [rollout_i.final_answer for i in 1..N_ROLLOUTS]
#   pred_max_prm  = answer of argmax_i rollout_i.final_prm_score
#
# Both accuracies are saved.  Results are persisted after EVERY step into
#   cimr_full_results_<dataset>_N<N_ROLLOUTS>_d<DEPTH>.json
# so re-running the script transparently resumes from the last finished
# step on a per-question, per-rollout, per-step granularity.
# ─────────────────────────────────────────────────────────────────────────────
ENABLE_CIMR=1
CIMR_N_ROLLOUTS=8                  # ⇐ N — independent rollouts per question
CIMR_DEPTH=4                       # ⇐ d — refinement steps per rollout

# Skip baseline Pass 1 and run CIMR-only when set to 1
SKIP_BASELINES=1

# Reasoning generation knobs (per step)
CIMR_REASONING_MAX_NEW_TOKENS=2048
CIMR_REASONING_TEMPERATURE=0.7
CIMR_REASONING_TOP_P=0.95

# Critique knobs
CIMR_CRITIQUE_MAX_TOKENS=384
CIMR_CRITIQUE_MAX_TRACE_CHARS=4000

# Correction knobs
CIMR_CORRECTION_MAX_TOKENS=1024
CIMR_CORRECTION_MAX_TRACE_CHARS=4000
CIMR_CORRECTION_MAX_CRITIQUE_CHARS=1500

# Continuation prompt (how much of the prior corrected step the next step sees)
CIMR_CONTINUATION_MAX_PRIOR_CHARS=4000

# Surface sibling-rollout corrected answers to critique + correction
CIMR_OTHER_AWARE=1

# Resolve paths relative to this script so cache lookup works from any cwd.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output root
BASE_OUT="${ROOT_DIR}/experiments_results/results_paired"

# GPU(s) to use (comma-separated indices, blank = auto)
GPUS="0,2,3"

# Python executable
PYTHON="python"

# Dataset size limits (set to "" for full)
MAX_SAMPLES_AIME="30"
MAX_SAMPLES_AMC="40"
MAX_SAMPLES_MATH="500"
MAX_SAMPLES_OLYMPIAD="80"
MAX_SAMPLES_GSM=""

# ─────────────────────────────────────────────────────────────────────────────
# Derived settings
# ─────────────────────────────────────────────────────────────────────────────

MODEL_SHORT=$(basename "${MODEL}")
RUN_TAG="${MODEL_SHORT}_mnt${MAX_NEW_TOKENS}_t${TEMPERATURE}_p${TOP_P}"
OUT="${BASE_OUT}/${RUN_TAG}"
mkdir -p "${OUT}/figures" "${OUT}/logs"

GPU_FLAG=""
if [[ -n "${GPUS}" ]]; then
    GPU_FLAG="--gpus ${GPUS}"
fi

CIMR_FLAGS=""
if [[ "${ENABLE_CIMR}" == "1" ]]; then
    CIMR_FLAGS="--enable_cimr"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_n_rollouts ${CIMR_N_ROLLOUTS}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_depth ${CIMR_DEPTH}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_reasoning_max_new_tokens ${CIMR_REASONING_MAX_NEW_TOKENS}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_reasoning_temperature   ${CIMR_REASONING_TEMPERATURE}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_reasoning_top_p         ${CIMR_REASONING_TOP_P}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_critique_max_tokens     ${CIMR_CRITIQUE_MAX_TOKENS}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_critique_max_trace_chars ${CIMR_CRITIQUE_MAX_TRACE_CHARS}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_correction_max_tokens          ${CIMR_CORRECTION_MAX_TOKENS}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_correction_max_trace_chars     ${CIMR_CORRECTION_MAX_TRACE_CHARS}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_correction_max_critique_chars  ${CIMR_CORRECTION_MAX_CRITIQUE_CHARS}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_continuation_max_prior_chars   ${CIMR_CONTINUATION_MAX_PRIOR_CHARS}"
    if [[ "${CIMR_OTHER_AWARE}" == "1" ]]; then
        CIMR_FLAGS="${CIMR_FLAGS} --cimr_other_answer_aware"
    fi
fi

COMMON_ARGS=(
    --model            "${MODEL}"
    --verifier_model   "${VERIFIER_MODEL}"
    --embedding_model  "${EMBEDDING_MODEL}"
    --max_new_tokens   "${MAX_NEW_TOKENS}"
    --temperature      "${TEMPERATURE}"
    --top_p            "${TOP_P}"
    --alpha_values     ${ALPHA_VALUES}
    --tau_values       ${TAU_VALUES}
    --output_dir       "${OUT}"
    --batch_size       "${BATCH_SIZE}"
    ${GPU_FLAG}
    ${CIMR_FLAGS}
)

# Optionally add skip-baselines flag to python args
if [[ "${SKIP_BASELINES}" == "1" ]]; then
    COMMON_ARGS+=(--skip_baselines)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Runner functions
# ─────────────────────────────────────────────────────────────────────────────

run_dataset() {
    local DATASET="$1"
    local N="$2"
    local MAX_SAMPLES="$3"

    local TAG="${DATASET}_N${N}"
    local LOG="${OUT}/logs/paired_${TAG}_d${CIMR_DEPTH}_R${CIMR_N_ROLLOUTS}.log"

    CACHE_FILE="${OUT}/sample_cache_${TAG}.jsonl"
    CACHE_FLAG=()
    if [[ "${LOAD_CACHE:-0}" == "1" && -f "${CACHE_FILE}" ]]; then
        CACHE_FLAG=(--load_cache "${CACHE_FILE}")
    fi

    MAX_FLAG=()
    if [[ -n "${MAX_SAMPLES}" ]]; then
        MAX_FLAG=(--max_samples "${MAX_SAMPLES}")
    fi

    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  Dataset : ${DATASET}    N(baseline)=${N}"
    echo "  CIMR    : N_rollouts=${CIMR_N_ROLLOUTS}  depth=${CIMR_DEPTH}"
    echo "  Out     : ${OUT}"
    echo "  Log     : ${LOG}"
    echo "──────────────────────────────────────────────────────────────────────"

    ${PYTHON} run_paired_ablation.py \
        "${COMMON_ARGS[@]}" \
        --batch_size  "${BATCH_SIZE}" \
        --dataset     "${DATASET}" \
        --N           "${N}" \
        "${MAX_FLAG[@]}" \
        "${CACHE_FLAG[@]}" \
        2>&1 | tee "${LOG}"

    echo "  ✓  Done. Log: ${LOG}"
}

# Sweep over depths d ∈ {1, 2, 3, 4} for one dataset, keeping N fixed.
run_depth_sweep() {
    local DATASET="$1"
    local MAX_SAMPLES="$2"
    local DEPTHS=(1 2 3 4)

    for D in "${DEPTHS[@]}"; do
        local LOCAL_FLAGS="--enable_cimr --cimr_n_rollouts ${CIMR_N_ROLLOUTS} --cimr_depth ${D}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_reasoning_max_new_tokens ${CIMR_REASONING_MAX_NEW_TOKENS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_reasoning_temperature   ${CIMR_REASONING_TEMPERATURE}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_reasoning_top_p         ${CIMR_REASONING_TOP_P}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_critique_max_tokens     ${CIMR_CRITIQUE_MAX_TOKENS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_critique_max_trace_chars ${CIMR_CRITIQUE_MAX_TRACE_CHARS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_correction_max_tokens         ${CIMR_CORRECTION_MAX_TOKENS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_correction_max_trace_chars    ${CIMR_CORRECTION_MAX_TRACE_CHARS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_correction_max_critique_chars ${CIMR_CORRECTION_MAX_CRITIQUE_CHARS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_continuation_max_prior_chars  ${CIMR_CONTINUATION_MAX_PRIOR_CHARS}"
        if [[ "${CIMR_OTHER_AWARE}" == "1" ]]; then
            LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_other_answer_aware"
        fi

        local TAG="${DATASET}_N${N_DEFAULT}"
        local LOG="${OUT}/logs/paired_${TAG}_d${D}_R${CIMR_N_ROLLOUTS}_depthsweep.log"

        CACHE_FILE="${OUT}/sample_cache_${TAG}.jsonl"
        CACHE_FLAG=()
        if [[ -f "${CACHE_FILE}" ]]; then
            CACHE_FLAG=(--load_cache "${CACHE_FILE}")
        fi

        MAX_FLAG=()
        if [[ -n "${MAX_SAMPLES}" ]]; then
            MAX_FLAG=(--max_samples "${MAX_SAMPLES}")
        fi

        echo ""
        echo "  ── Depth sweep: ${DATASET}  N_rollouts=${CIMR_N_ROLLOUTS}  d=${D} ──"

        ${PYTHON} run_paired_ablation.py \
            --model            "${MODEL}" \
            --verifier_model   "${VERIFIER_MODEL}" \
            --embedding_model  "${EMBEDDING_MODEL}" \
            --max_new_tokens   "${MAX_NEW_TOKENS}" \
            --temperature      "${TEMPERATURE}" \
            --top_p            "${TOP_P}" \
            --alpha_values     ${ALPHA_VALUES} \
            --tau_values       ${TAU_VALUES} \
            --output_dir       "${OUT}" \
            --batch_size       "${BATCH_SIZE}" \
            ${GPU_FLAG} \
            ${LOCAL_FLAGS} \
            --dataset          "${DATASET}" \
            --N                "${N_DEFAULT}" \
            "${MAX_FLAG[@]}" \
            "${CACHE_FLAG[@]}" \
            2>&1 | tee "${LOG}"
    done
}

# Sweep over rollout counts N ∈ {2, 4, 8, 16} keeping d fixed.
run_rollout_sweep() {
    local DATASET="$1"
    local MAX_SAMPLES="$2"
    local ROLLOUTS=(2 4 8 16)

    for R in "${ROLLOUTS[@]}"; do
        local LOCAL_FLAGS="--enable_cimr --cimr_n_rollouts ${R} --cimr_depth ${CIMR_DEPTH}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_reasoning_max_new_tokens ${CIMR_REASONING_MAX_NEW_TOKENS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_reasoning_temperature   ${CIMR_REASONING_TEMPERATURE}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_reasoning_top_p         ${CIMR_REASONING_TOP_P}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_critique_max_tokens     ${CIMR_CRITIQUE_MAX_TOKENS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_critique_max_trace_chars ${CIMR_CRITIQUE_MAX_TRACE_CHARS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_correction_max_tokens         ${CIMR_CORRECTION_MAX_TOKENS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_correction_max_trace_chars    ${CIMR_CORRECTION_MAX_TRACE_CHARS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_correction_max_critique_chars ${CIMR_CORRECTION_MAX_CRITIQUE_CHARS}"
        LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_continuation_max_prior_chars  ${CIMR_CONTINUATION_MAX_PRIOR_CHARS}"
        if [[ "${CIMR_OTHER_AWARE}" == "1" ]]; then
            LOCAL_FLAGS="${LOCAL_FLAGS} --cimr_other_answer_aware"
        fi

        local TAG="${DATASET}_N${N_DEFAULT}"
        local LOG="${OUT}/logs/paired_${TAG}_d${CIMR_DEPTH}_R${R}_rolloutsweep.log"

        CACHE_FILE="${OUT}/sample_cache_${TAG}.jsonl"
        CACHE_FLAG=()
        if [[ -f "${CACHE_FILE}" ]]; then
            CACHE_FLAG=(--load_cache "${CACHE_FILE}")
        fi

        MAX_FLAG=()
        if [[ -n "${MAX_SAMPLES}" ]]; then
            MAX_FLAG=(--max_samples "${MAX_SAMPLES}")
        fi

        echo ""
        echo "  ── Rollout sweep: ${DATASET}  N_rollouts=${R}  d=${CIMR_DEPTH} ──"

        ${PYTHON} run_paired_ablation.py \
            --model            "${MODEL}" \
            --verifier_model   "${VERIFIER_MODEL}" \
            --embedding_model  "${EMBEDDING_MODEL}" \
            --max_new_tokens   "${MAX_NEW_TOKENS}" \
            --temperature      "${TEMPERATURE}" \
            --top_p            "${TOP_P}" \
            --alpha_values     ${ALPHA_VALUES} \
            --tau_values       ${TAU_VALUES} \
            --output_dir       "${OUT}" \
            --batch_size       "${BATCH_SIZE}" \
            ${GPU_FLAG} \
            ${LOCAL_FLAGS} \
            --dataset          "${DATASET}" \
            --N                "${N_DEFAULT}" \
            "${MAX_FLAG[@]}" \
            "${CACHE_FLAG[@]}" \
            2>&1 | tee "${LOG}"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Main dispatch
# ─────────────────────────────────────────────────────────────────────────────

DATASETS=("${@}")

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(aime24 aime25 amc23 math500 olympiadbench)
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DN-BoN Paired Ablation  (CIMR rollouts × steps)"
echo "  Model        : ${MODEL}"
echo "  Verifier     : ${VERIFIER_MODEL}"
echo "  N (baseline) : ${N_DEFAULT}"
echo "  CIMR rollouts: ${CIMR_N_ROLLOUTS}"
echo "  CIMR depth   : ${CIMR_DEPTH}"
echo "  α sweep      : ${ALPHA_VALUES}"
echo "  τ sweep      : ${TAU_VALUES}"
echo "  Datasets     : ${DATASETS[*]}"
echo "  Out dir      : ${OUT}"
echo "════════════════════════════════════════════════════════════════"

for DATASET in "${DATASETS[@]}"; do
    case "${DATASET}" in

        aime24)        run_dataset aime24        "${N_DEFAULT}" "${MAX_SAMPLES_AIME}" ;;
        aime25)        run_dataset aime25        "${N_DEFAULT}" "${MAX_SAMPLES_AIME}" ;;
        amc23)         run_dataset amc23         "${N_DEFAULT}" "${MAX_SAMPLES_AMC}"  ;;
        math500)       run_dataset math500       "${N_DEFAULT}" "${MAX_SAMPLES_MATH}" ;;
        olympiadbench) run_dataset olympiadbench "${N_DEFAULT}" "${MAX_SAMPLES_OLYMPIAD}" ;;
        gsm8k)         run_dataset gsm8k         "${N_DEFAULT}" "${MAX_SAMPLES_GSM}"  ;;

        depth_aime24)        run_depth_sweep aime24        "${MAX_SAMPLES_AIME}" ;;
        depth_math500)       run_depth_sweep math500       "${MAX_SAMPLES_MATH}" ;;
        depth_amc23)         run_depth_sweep amc23         "${MAX_SAMPLES_AMC}"  ;;
        depth_olympiad)      run_depth_sweep olympiadbench "${MAX_SAMPLES_OLYMPIAD}" ;;
        depth_all)
            run_depth_sweep aime24        "${MAX_SAMPLES_AIME}"
            run_depth_sweep aime25        "${MAX_SAMPLES_AIME}"
            run_depth_sweep amc23         "${MAX_SAMPLES_AMC}"
            run_depth_sweep math500       "${MAX_SAMPLES_MATH}"
            run_depth_sweep olympiadbench "${MAX_SAMPLES_OLYMPIAD}"
            ;;

        rollout_aime24)        run_rollout_sweep aime24        "${MAX_SAMPLES_AIME}" ;;
        rollout_math500)       run_rollout_sweep math500       "${MAX_SAMPLES_MATH}" ;;
        rollout_amc23)         run_rollout_sweep amc23         "${MAX_SAMPLES_AMC}"  ;;
        rollout_olympiad)      run_rollout_sweep olympiadbench "${MAX_SAMPLES_OLYMPIAD}" ;;

        all)
            run_dataset aime24        "${N_DEFAULT}" "${MAX_SAMPLES_AIME}"
            run_dataset aime25        "${N_DEFAULT}" "${MAX_SAMPLES_AIME}"
            run_dataset amc23         "${N_DEFAULT}" "${MAX_SAMPLES_AMC}"
            run_dataset math500       "${N_DEFAULT}" "${MAX_SAMPLES_MATH}"
            run_dataset olympiadbench "${N_DEFAULT}" "${MAX_SAMPLES_OLYMPIAD}"
            ;;

        *)
            echo "Unknown dataset: '${DATASET}'"
            echo ""
            echo "Usage: bash run_paired_ablation.sh [dataset ...]"
            echo ""
            echo "Datasets:  aime24 aime25 amc23 math500 olympiadbench gsm8k"
            echo "Depth   :  depth_aime24 depth_math500 depth_amc23 depth_olympiad depth_all"
            echo "Rollouts:  rollout_aime24 rollout_math500 rollout_amc23 rollout_olympiad"
            echo "All     :  all"
            exit 1
            ;;
    esac
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All paired ablation experiments finished."
echo "  Results : ${OUT}"
echo "════════════════════════════════════════════════════════════════"
