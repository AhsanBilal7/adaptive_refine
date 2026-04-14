#!/usr/bin/env bash
# =============================================================================
# run_token_sweep.sh
# =============================================================================
# Sweeps over a list of MAX_NEW_TOKENS values, running a chosen set of
# datasets and methods for each token budget.
#
# Every (max_new_tokens × dataset × method) triple is ONE separate python
# process — the GPU is fully released between runs.
#
# Result folders are named per-token-budget:
#   results/Qwen2.5-Math-7B-Instruct_mnt1024_t0.7_p0.95/
#   results/Qwen2.5-Math-7B-Instruct_mnt2048_t0.7_p0.95/
#   results/Qwen2.5-Math-7B-Instruct_mnt4096_t0.7_p0.95/
#
# Usage
# -----
#   bash run_token_sweep.sh
#
# Edit the CONFIG section below to choose:
#   TOKEN_LIST   — which max_new_tokens values to sweep
#   DATASETS     — which datasets to run
#   METHODS      — which inference methods to run
# =============================================================================

set -euo pipefail


# =============================================================================
# CONFIG  — edit these
# =============================================================================

# -----------------------------------------------------------------------------
# MODEL
# HuggingFace model id or local path.
#   Qwen/Qwen2.5-Math-7B-Instruct    — strong math, ~16 GB fp16
#   Qwen/Qwen2.5-Math-1.5B-Instruct  — lightweight, ~4 GB fp16
#   deepseek-ai/deepseek-math-7b-instruct
#   meta-llama/Llama-3.1-8B-Instruct
# -----------------------------------------------------------------------------
MODEL="Qwen/Qwen2.5-Math-7B-Instruct"

# -----------------------------------------------------------------------------
# TOKEN_LIST
# Space-separated list of max_new_tokens values to sweep.
# Each value becomes a separate results folder.
#   1024  — fast; enough for MATH-500 / GSM-8K
#   2048  — safer for AIME (long derivations)
#   4096  — very long CoT; slow but complete
#   8192  — maximum; only for models that support it
# -----------------------------------------------------------------------------
TOKEN_LIST="1024 2048 4096"

# -----------------------------------------------------------------------------
# DATASETS
# Space-separated list of datasets to run for each token budget.
# Valid: math500  aime24  aime25  amc23  gsm8k
# -----------------------------------------------------------------------------
DATASETS="math500 aime24 aime25 amc23 gsm8k"

# -----------------------------------------------------------------------------
# METHODS
# Space-separated list of inference methods to run for each (token, dataset).
# Valid: greedy  bon  dn_bon  greedy_pass_n  bon_majority
#
# Notes:
#   greedy        — no sampling; ignores N_SAMPLES and TEMPERATURE
#   bon           — needs VERIFIER; use VERIFIER=prm (no gold leakage)
#   dn_bon        — needs VERIFIER + EMBEDDING_MODEL; adds --alpha 0.5
#   greedy_pass_n — no verifier needed; uses model log-probability
#   bon_majority  — no verifier needed; uses majority vote
# -----------------------------------------------------------------------------
METHODS="greedy bon dn_bon greedy_pass_n bon_majority"

# -----------------------------------------------------------------------------
# N_SAMPLES
# Number of candidate traces for BoN / DN-BoN / greedy_pass_n / bon_majority.
#   8   — good default
#   16  — stronger; use for AIME
#   32  — near-oracle; slow
# -----------------------------------------------------------------------------
N_SAMPLES=8

# -----------------------------------------------------------------------------
# LM_BATCH_SIZE
# Sequences per model.generate() call.
# Set equal to N_SAMPLES for BoN so all N prompts decode in one call.
# For H100 60 GB with a 7B model: 64–128 is safe.
# -----------------------------------------------------------------------------
LM_BATCH_SIZE=64

# -----------------------------------------------------------------------------
# TEMPERATURE / TOP_P
# Sampling parameters (greedy ignores these).
# -----------------------------------------------------------------------------
TEMPERATURE=0.7
TOP_P=0.95

# -----------------------------------------------------------------------------
# DTYPE / Quantisation
# -----------------------------------------------------------------------------
DTYPE="auto"
LOAD_4BIT=0   # set 1 for NF4 4-bit (~4-5 GB for 7B)
LOAD_8BIT=0   # set 1 for int8  (~8 GB for 7B)

# -----------------------------------------------------------------------------
# BASE_OUT
# Root output directory. One sub-folder is created per token budget.
# -----------------------------------------------------------------------------
BASE_OUT="./results"

# -----------------------------------------------------------------------------
# PYTHON
# Override for a specific conda env:
#   PYTHON="/home/ahsan/.conda/envs/dora/bin/python"
# -----------------------------------------------------------------------------
PYTHON="python"

# -----------------------------------------------------------------------------
# HF_CACHE
# HuggingFace cache dir. Move to scratch on HPC clusters.
# -----------------------------------------------------------------------------
HF_CACHE="/scratch/${USER}/hf_cache"

# -----------------------------------------------------------------------------
# GPUS
# Which GPU(s) to use.  Sets CUDA_VISIBLE_DEVICES for every python call.
#   ""      — use all visible GPUs (default)
#   "0"     — single GPU index 0
#   "0,1"   — two GPUs, model sharded automatically
#   "2,3"   — GPUs 2 and 3 (useful on shared nodes)
# -----------------------------------------------------------------------------
GPUS=""

# -----------------------------------------------------------------------------
# VERIFIER
# Scoring back-end for BoN / DN-BoN candidate selection.
#
# IMPORTANT — gold leakage:
#   outcome  — scores by comparing against the gold answer directly.
#              This LEAKS the gold answer into the selection step, which
#              inflates BoN / DN-BoN accuracy and makes comparisons unfair.
#              Only valid as an oracle upper bound — do NOT use for reported
#              experiments.
#
#   prm      — Process Reward Model; no gold leakage; correct for fair eval.
#              Requires PRM_MODEL below.  Use this for all experiments.
#
#   length   — Training-free proxy; no gold needed; deployment baseline.
# -----------------------------------------------------------------------------
VERIFIER="prm"

# -----------------------------------------------------------------------------
# PRM_MODEL / PRM_AGGREGATION
# Only used when VERIFIER=prm.
#   PRM_MODEL       : Qwen/Qwen2.5-Math-PRM-7B  (~14 GB fp16)
#   PRM_AGGREGATION : min (conservative) | mean | last
# -----------------------------------------------------------------------------
PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B"
PRM_AGGREGATION="min"

EMBEDDING_MODEL="all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD=0.85
CLUSTERING_METHOD="agglomerative"

# =============================================================================
# END CONFIG
# =============================================================================


# =============================================================================
# Validation
# =============================================================================

_valid_datasets="math500 aime24 aime25 amc23 gsm8k"
_valid_methods="greedy bon dn_bon greedy_pass_n bon_majority"

for D in ${DATASETS}; do
    if ! echo "${_valid_datasets}" | grep -qw "${D}"; then
        echo "ERROR: unknown dataset '${D}'"
        echo "Valid: ${_valid_datasets}"
        exit 1
    fi
done

for M in ${METHODS}; do
    if ! echo "${_valid_methods}" | grep -qw "${M}"; then
        echo "ERROR: unknown method '${M}'"
        echo "Valid: ${_valid_methods}"
        exit 1
    fi
done


# =============================================================================
# HuggingFace cache env vars
# =============================================================================
if [[ -n "${HF_CACHE}" ]]; then
    export HF_HOME="${HF_CACHE}"
    export TRANSFORMERS_CACHE="${HF_CACHE}"
    export HF_DATASETS_CACHE="${HF_CACHE}"
    export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
    mkdir -p "${HF_CACHE}"
fi

# =============================================================================
# GPU setup
# =============================================================================
if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    echo "[gpu] CUDA_VISIBLE_DEVICES=${GPUS}"
fi


# =============================================================================
# Helpers
# =============================================================================

NOW=$(date +"%Y%m%d_%H%M%S")

# Per-dataset sample caps
_max_samples() {
    case "$1" in
        aime24|aime25) echo 32  ;;
        amc23)         echo 40  ;;
        math500|gsm8k) echo 500 ;;
        *)             echo 500 ;;
    esac
}

# Build result folder tag for a given MAX_NEW_TOKENS
_build_tag() {
    local mnt="$1"
    local model_short="${MODEL##*/}"
    model_short="${model_short//[: ]/_}"
    local tag="${model_short}_mnt${mnt}_t${TEMPERATURE}_p${TOP_P}"
    [[ "${LOAD_4BIT}" == "1" ]] && tag+="_4bit"
    [[ "${LOAD_8BIT}" == "1" ]] && tag+="_8bit"
    echo "${tag}"
}

# Build common flags for a given MAX_NEW_TOKENS
_common_flags() {
    local mnt="$1"
    local flags=""
    flags+=" --model ${MODEL}"
    flags+=" --batch_size ${LM_BATCH_SIZE}"
    flags+=" --max_new_tokens ${mnt}"
    flags+=" --temperature ${TEMPERATURE}"
    flags+=" --top_p ${TOP_P}"
    flags+=" --dtype ${DTYPE}"
    flags+=" --verifier ${VERIFIER}"
    flags+=" --embedding_model ${EMBEDDING_MODEL}"
    flags+=" --similarity_threshold ${SIMILARITY_THRESHOLD}"
    flags+=" --clustering_method ${CLUSTERING_METHOD}"
    [[ -n "${HF_CACHE}" ]]      && flags+=" --hf_cache_dir ${HF_CACHE}"
    [[ "${LOAD_4BIT}" == "1" ]] && flags+=" --load_in_4bit"
    [[ "${LOAD_8BIT}" == "1" ]] && flags+=" --load_in_8bit"
    [[ "${VERIFIER}"  == "prm" ]] && flags+=" --prm_model ${PRM_MODEL}"
    [[ "${VERIFIER}"  == "prm" ]] && flags+=" --prm_aggregation ${PRM_AGGREGATION}"
    [[ -n "${GPUS}" ]]          && flags+=" --gpus ${GPUS}"
    echo "${flags}"
}

# run one experiment — streams to terminal and log file
run_exp() {
    local tag="$1"; local out_dir="$2"; shift 2
    local log_dir="${out_dir}/logs"
    mkdir -p "${log_dir}"
    local log_file="${log_dir}/${tag}_${NOW}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo " START : ${tag}"
    echo " OUT   : ${out_dir}"
    echo " LOG   : ${log_file}"
    echo "------------------------------------------------------------"

    ${PYTHON} run_eval.py "$@" 2>&1 | tee "${log_file}"

    local exit_code=${PIPESTATUS[0]}
    if [[ ${exit_code} -eq 0 ]]; then
        echo "  [DONE]  ${tag}"
    else
        echo "  [FAIL]  ${tag}  (exit ${exit_code}) — see ${log_file}"
    fi
}


# =============================================================================
# Count total runs for progress display
# =============================================================================

N_TOKENS=0; for T in ${TOKEN_LIST}; do N_TOKENS=$((N_TOKENS+1)); done
N_DS=0;     for D in ${DATASETS};   do N_DS=$((N_DS+1)); done
N_ME=0;     for M in ${METHODS};    do N_ME=$((N_ME+1)); done
TOTAL=$((N_TOKENS * N_DS * N_ME))


# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "  TOKEN SWEEP"
echo "------------------------------------------------------------"
echo "  Model      : ${MODEL}"
echo "  Tokens     : ${TOKEN_LIST}"
echo "  Datasets   : ${DATASETS}"
echo "  Methods    : ${METHODS}"
echo "  N_SAMPLES  : ${N_SAMPLES}"
echo "  Batch size : ${LM_BATCH_SIZE}"
echo "  Total runs : ${TOTAL}  (${N_TOKENS} tokens × ${N_DS} datasets × ${N_ME} methods)"
echo "  Base out   : ${BASE_OUT}"
echo "============================================================"
echo ""


# =============================================================================
# Main sweep loop
# =============================================================================

RUN_IDX=0

for MNT in ${TOKEN_LIST}; do

    TAG="$(_build_tag "${MNT}")"
    OUT="${BASE_OUT}/${TAG}"

    echo ""
    echo "############################################################"
    echo "  max_new_tokens = ${MNT}   →   ${OUT}"
    echo "############################################################"

    for DATASET in ${DATASETS}; do

        MAX_S="$(_max_samples "${DATASET}")"
        FLAGS="$(_common_flags "${MNT}")"

        echo "  ----- dataset=${DATASET}  max_samples=${MAX_S} -----"

        for METHOD in ${METHODS}; do

            RUN_IDX=$((RUN_IDX+1))
            EXP_TAG="${DATASET}_${METHOD}_N${N_SAMPLES}_mnt${MNT}"

            echo "  [${RUN_IDX}/${TOTAL}]  mnt=${MNT}  dataset=${DATASET}  method=${METHOD}"

            # dn_bon needs --alpha; others don't
            if [[ "${METHOD}" == "dn_bon" ]]; then
                run_exp "${EXP_TAG}" "${OUT}" \
                    ${FLAGS} \
                    --dataset  "${DATASET}" \
                    --methods  "${METHOD}" \
                    --N        "${N_SAMPLES}" \
                    --alpha    0.5 \
                    --max_samples "${MAX_S}" \
                    --output_dir  "${OUT}"
            else
                run_exp "${EXP_TAG}" "${OUT}" \
                    ${FLAGS} \
                    --dataset  "${DATASET}" \
                    --methods  "${METHOD}" \
                    --N        "${N_SAMPLES}" \
                    --max_samples "${MAX_S}" \
                    --output_dir  "${OUT}"
            fi

        done   # METHOD
    done   # DATASET
done   # MNT


# =============================================================================
# Done
# =============================================================================

echo ""
echo "============================================================"
echo "  TOKEN SWEEP COMPLETE — ${TOTAL} runs finished"
echo ""
for MNT in ${TOKEN_LIST}; do
    TAG="$(_build_tag "${MNT}")"
    echo "  mnt=${MNT}  →  ${BASE_OUT}/${TAG}/"
done
echo "============================================================"