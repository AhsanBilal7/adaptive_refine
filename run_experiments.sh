#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# =============================================================================
# Every experiment is a SEPARATE python process so the GPU is fully released
# between runs.
#
# Result folder naming
# --------------------
# All results go into ONE flat directory under BASE_OUT:
#   ./results/mnt2048_t0.7_p0.95/
#   ./results/mnt2048_t0.7_p0.95_4bit/
#   ./results/mnt4096_t0.8_p0.95/
#
# The folder name encodes:
#   mnt<MAX_NEW_TOKENS>   — generation length cap
#   t<TEMPERATURE>        — sampling temperature
#   p<TOP_P>              — nucleus sampling p
#   _4bit                 — if LOAD_4BIT=1  (omitted otherwise)
#   _8bit                 — if LOAD_8BIT=1  (omitted otherwise)
#
# Usage
# -----
#   bash run_experiments.sh              # run ALL blocks sequentially
#   bash run_experiments.sh math500      # MATH-500 (greedy+BoN+DN-BoN)
#   bash run_experiments.sh aime24       # AIME-24
#   bash run_experiments.sh amc23        # AMC-23
#   bash run_experiments.sh gsm8k        # GSM-8K
#   bash run_experiments.sh alpha        # alpha ablation (α ∈ 0..1)
#   bash run_experiments.sh scaling      # budget scaling (N = 4,8,16,32)
#   bash run_experiments.sh clustering   # similarity-threshold ablation
#   bash run_experiments.sh verifier     # outcome vs length verifier
# =============================================================================

set -euo pipefail


# =============================================================================
# CONFIG
# =============================================================================

# -----------------------------------------------------------------------------
# MODEL
# Recommended choices:
#   Qwen/Qwen2.5-Math-7B-Instruct    — strong math, chat-templated, ~16 GB fp16
#   Qwen/Qwen2.5-Math-1.5B-Instruct  — lightweight, ~4 GB fp16
#   deepseek-ai/deepseek-math-7b-instruct — strong on competition math
#   meta-llama/Llama-3.1-8B-Instruct  — general purpose
# For a local checkpoint: MODEL="/data/checkpoints/my_model"
# -----------------------------------------------------------------------------
MODEL="Qwen/Qwen2.5-Math-7B-Instruct"

# -----------------------------------------------------------------------------
# PYTHON
# Override for a specific conda env:
#   PYTHON="/home/ahsan/.conda/envs/dora/bin/python"
# -----------------------------------------------------------------------------
PYTHON="python"

# -----------------------------------------------------------------------------
# BASE_OUT
# Root output directory.  A single sub-folder is created under it whose name
# encodes MAX_NEW_TOKENS, TEMPERATURE, TOP_P, and quantisation flags.
# All experiment JSONL + summary JSON files go into that one folder.
# -----------------------------------------------------------------------------
BASE_OUT="./results"

# -----------------------------------------------------------------------------
# MAX_NEW_TOKENS
# Maximum tokens the model can generate per response.
#   1024  — fast, enough for MATH-500 / GSM-8K
#   2048  — safer for AIME / competition math with long derivations
#   4096  — very long chain-of-thought (slow but complete)
# Encoded in the result folder name as  mnt<value>
# -----------------------------------------------------------------------------
MAX_NEW_TOKENS=2048

# -----------------------------------------------------------------------------
# TEMPERATURE
# Sampling temperature for BoN / DN-BoN (greedy ignores this).
#   0.6   — low diversity; well-calibrated models
#   0.7   — recommended default
#   0.8   — more diverse; useful for harder problems (AIME)
#   1.0   — high diversity; may add noise
# Encoded in the result folder name as  t<value>
# -----------------------------------------------------------------------------
TEMPERATURE=0.7

# -----------------------------------------------------------------------------
# TOP_P
# Nucleus sampling cutoff.
#   0.90  — tighter nucleus
#   0.95  — recommended default
#   1.0   — no cutoff (pure temperature sampling)
# Encoded in the result folder name as  p<value>
# -----------------------------------------------------------------------------
TOP_P=0.95

# -----------------------------------------------------------------------------
# LM_BATCH_SIZE  —  the actual HuggingFace model.generate() batch size
#
# This single value controls how many sequences are processed in ONE
# model.generate() call for the main language model.  It maps directly to
# the --batch_size argument of run_eval.py and to HFModelWrapper.batch_size.
#
# What "batch size" means per method
# -----------------------------------
#   Greedy:
#     LM_BATCH_SIZE questions are stacked into one padded tensor and decoded
#     together in a single model.generate() call.
#     → actual HF batch = LM_BATCH_SIZE
#     → recommended: 32–128 (questions are independent, limited only by VRAM)
#
#   BoN / DN-BoN:
#     For each question, its prompt is replicated N_SAMPLES times.
#     All N_SAMPLES copies go into ONE padded batch → one model.generate() call.
#     LM_BATCH_SIZE is passed as --batch_size but the effective batch for
#     BoN/DN-BoN is always N_SAMPLES (the replication count), not LM_BATCH_SIZE.
#     Setting LM_BATCH_SIZE == N_SAMPLES is the recommended setup so there is
#     no ambiguity.  If LM_BATCH_SIZE < N_SAMPLES the N_SAMPLES prompts are
#     split across ceil(N_SAMPLES/LM_BATCH_SIZE) generate() calls.
#     → actual HF batch per question = N_SAMPLES
#     → recommended: LM_BATCH_SIZE = N_SAMPLES (set both to the same value)
#
# VRAM guidance (fp16, single A100 80 GB, Qwen2.5-Math-7B-Instruct):
#   LM_BATCH_SIZE=8   — very safe; works with any N_SAMPLES value
#   LM_BATCH_SIZE=16  — comfortable for N_SAMPLES ≤ 16
#   LM_BATCH_SIZE=32  — good for greedy throughput; fine for N_SAMPLES=32
#   LM_BATCH_SIZE=64  — high throughput greedy; N_SAMPLES=64 needs ~40 GB
#   LM_BATCH_SIZE=128 — maximum greedy throughput; BoN rarely needs this
#
# If you hit CUDA OOM: halve LM_BATCH_SIZE first, then reduce N_SAMPLES.
# -----------------------------------------------------------------------------
LM_BATCH_SIZE=128

# -----------------------------------------------------------------------------
# N_SAMPLES  —  number of candidate traces sampled per question for BoN/DN-BoN
#
# This is the --N argument of run_eval.py.  It determines:
#   • how many i.i.d. traces are generated per question
#   • the effective HF batch size for BoN/DN-BoN (see LM_BATCH_SIZE above)
#   • the test-time compute budget  (higher N = better accuracy, more time)
#
# Recommended values:
#   4   — fast baseline; limited diversity
#   8   — good default; balances quality and speed
#   16  — stronger; use for AIME/competition problems
#   32  — near-oracle; slow but high accuracy ceiling
#
# Set LM_BATCH_SIZE = N_SAMPLES so all N prompts decode in one generate() call.
# -----------------------------------------------------------------------------
N_SAMPLES=8

# -----------------------------------------------------------------------------
# DTYPE
# Floating-point precision for weights.
# Allowed: auto | float16 | bfloat16 | float32
#   auto     — float16 on CUDA, float32 on CPU (recommended)
#   bfloat16 — better range; requires Ampere+ GPU (A100, 3090+)
# -----------------------------------------------------------------------------
DTYPE="auto"

# -----------------------------------------------------------------------------
# LOAD_4BIT / LOAD_8BIT
# bitsandbytes quantisation.  Set exactly ONE to 1, leave the other 0.
#   LOAD_4BIT=1  — NF4 4-bit (~4-5 GB for 7B); quality drop ~1-2 pp
#   LOAD_8BIT=1  — int8 8-bit (~8 GB for 7B); negligible quality drop
#   Both 0       — full dtype precision (default)
# Encoded in the result folder name as  _4bit  or  _8bit  when enabled.
# Requires:  pip install bitsandbytes
# -----------------------------------------------------------------------------
LOAD_4BIT=0
LOAD_8BIT=0

# -----------------------------------------------------------------------------
# HF_CACHE
# HuggingFace cache directory.
# On HPC clusters, move this to scratch to avoid home-quota issues.
# Recommended:  HF_CACHE="/scratch/${USER}/hf_cache"
#
# To check current usage:    du -sh ~/.cache/*
# To wipe old default cache (ONE-TIME, after confirming new cache works):
#   rm -rf ~/.cache/huggingface/hub
# -----------------------------------------------------------------------------
HF_CACHE="/scratch/${USER}/hf_cache"

# -----------------------------------------------------------------------------
# VERIFIER
# Scoring back-end for BoN / DN-BoN candidate selection.
#   outcome  — binary correctness vs gold answer (best for eval)
#   length   — training-free proxy (no gold needed; use for deployment)
#   prm      — Process Reward Model (set PRM_MODEL below; loads 2nd model)
# -----------------------------------------------------------------------------
VERIFIER="outcome"

# -----------------------------------------------------------------------------
# PRM_MODEL
# HF model id for the PRM verifier.  Only used when VERIFIER=prm.
# Recommended: Qwen/Qwen2.5-Math-PRM-7B
# -----------------------------------------------------------------------------
PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B"

# -----------------------------------------------------------------------------
# EMBEDDING_MODEL
# sentence-transformers model for DN-BoN trace clustering.
#   all-MiniLM-L6-v2   — 22 MB, fast, good quality  (default)
#   all-mpnet-base-v2  — 420 MB, slower, better quality
#   tfidf              — no download; sparse lexical fallback
# -----------------------------------------------------------------------------
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# -----------------------------------------------------------------------------
# SIMILARITY_THRESHOLD
# Cosine-similarity cutoff for grouping traces into one reasoning direction.
#   0.70 — coarse (large clusters, strong normalisation)
#   0.85 — recommended default
#   0.95 — fine-grained (almost no grouping)
# -----------------------------------------------------------------------------
SIMILARITY_THRESHOLD=0.85

# -----------------------------------------------------------------------------
# CLUSTERING_METHOD
#   agglomerative — sklearn globally-consistent clusters (recommended)
#   greedy        — fast O(N²) threshold assignment
# -----------------------------------------------------------------------------
CLUSTERING_METHOD="agglomerative"

# =============================================================================
# END CONFIG
# =============================================================================


# =============================================================================
# HuggingFace cache environment variables
# =============================================================================
# All four env vars point to HF_CACHE so no library falls back to ~/.cache/.
# Exported here so every python subprocess spawned below inherits them.
# =============================================================================
if [[ -n "${HF_CACHE}" ]]; then
    export HF_HOME="${HF_CACHE}"
    export TRANSFORMERS_CACHE="${HF_CACHE}"
    export HF_DATASETS_CACHE="${HF_CACHE}"
    export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
    mkdir -p "${HF_CACHE}"
    echo "[cache] HF_HOME=${HF_HOME}"
fi


# =============================================================================
# Internal helpers
# =============================================================================

NOW=$(date +"%Y%m%d_%H%M%S")

# Build the result-folder tag from the hyperparameters that actually affect
# outputs (batch size excluded — it is a throughput knob only).
#   mnt2048_t0.7_p0.95
#   mnt2048_t0.7_p0.95_4bit
#   mnt2048_t0.7_p0.95_8bit
_build_run_tag() {
    local tag="mnt${MAX_NEW_TOKENS}_t${TEMPERATURE}_p${TOP_P}"
    [[ "${LOAD_4BIT}" == "1" ]] && tag+="_4bit"
    [[ "${LOAD_8BIT}" == "1" ]] && tag+="_8bit"
    echo "${tag}"
}
RUN_TAG="$(_build_run_tag)"
OUT="${BASE_OUT}/${RUN_TAG}"

# Build the flags shared by every run_eval.py call.
# --batch_size here is LM_BATCH_SIZE — the actual HF model.generate() batch size.
# See CONFIG for a detailed explanation of what this means per method.
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
    [[ -n "${HF_CACHE}" ]]      && flags+=" --hf_cache_dir ${HF_CACHE}"
    [[ "${LOAD_4BIT}" == "1" ]] && flags+=" --load_in_4bit"
    [[ "${LOAD_8BIT}" == "1" ]] && flags+=" --load_in_8bit"
    [[ "${VERIFIER}"  == "prm" ]] && flags+=" --prm_model ${PRM_MODEL}"
    echo "${flags}"
}

# run_exp <tag> [extra run_eval.py args...]
# Streams output to terminal AND a timestamped log file.
run_exp() {
    local tag="$1"; shift
    local log_dir="${OUT}/logs"
    mkdir -p "${log_dir}"
    local log_file="${log_dir}/${tag}_${NOW}.log"

    echo ""
    echo "============================================================"
    echo " START : ${tag}"
    echo " OUT   : ${OUT}"
    echo " LOG   : ${log_file}"
    echo "============================================================"

    ${PYTHON} run_eval.py $(common_flags) "$@" 2>&1 | tee "${log_file}"

    local exit_code=${PIPESTATUS[0]}
    if [[ ${exit_code} -eq 0 ]]; then
        echo "  [DONE]  ${tag}"
    else
        echo "  [FAIL]  ${tag}  (exit ${exit_code}) — see ${log_file}"
    fi
    echo ""
}


# =============================================================================
# BLOCK 1 : MATH-500  (500 problems, full benchmark)
# Three separate python calls: greedy, BoN, DN-BoN.
# All results written to ${OUT}/  (no sub-folder).
# =============================================================================
run_math500() {
    echo "===== MATH-500 ====="

    # 1a. Greedy
    run_exp "math500_greedy" \
        --dataset math500 \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    # 1b. Best-of-N  (N=8)
    run_exp "math500_bon_N8" \
        --dataset math500 \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    # 1c. DN-BoN  (N=8, alpha=0.5)
    run_exp "math500_dnbon_N8_a0.5" \
        --dataset math500 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 2 : AIME-24  (32 competition problems)
# =============================================================================
run_aime24() {
    echo "===== AIME-24 ====="

    run_exp "aime24_greedy" \
        --dataset aime24 \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 32 \
        --output_dir "${OUT}"

    run_exp "aime24_bon_N8" \
        --dataset aime24 \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 32 \
        --output_dir "${OUT}"

    run_exp "aime24_dnbon_N8_a0.5" \
        --dataset aime24 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 32 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 3 : AMC-23
# =============================================================================
run_amc23() {
    echo "===== AMC-23 ====="

    run_exp "amc23_greedy" \
        --dataset amc23 \
        --methods greedy \
        --N ${N_SAMPLES} \
        --output_dir "${OUT}"

    run_exp "amc23_bon_N8" \
        --dataset amc23 \
        --methods bon \
        --N ${N_SAMPLES} \
        --output_dir "${OUT}"

    run_exp "amc23_dnbon_N8_a0.5" \
        --dataset amc23 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 4 : GSM-8K  (500 problems)
# =============================================================================
run_gsm8k() {
    echo "===== GSM-8K ====="

    run_exp "gsm8k_greedy" \
        --dataset gsm8k \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "gsm8k_bon_N8" \
        --dataset gsm8k \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "gsm8k_dnbon_N8_a0.5" \
        --dataset gsm8k \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 5 : Alpha ablation
# DN-BoN with α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} on MATH-500 and AIME-24.
# α=0 exactly recovers plain BoN (clean ablation control).
# 10 separate python calls.
# =============================================================================
run_alpha() {
    echo "===== ALPHA ABLATION ====="

    for DATASET in math500 aime24; do
        MAX_S=500
        [[ "${DATASET}" == "aime24" ]] && MAX_S=32

        for ALPHA in 0.0 0.25 0.5 0.75 1.0; do
            run_exp "${DATASET}_dnbon_N8_a${ALPHA}" \
                --dataset "${DATASET}" \
                --methods dn_bon \
                --N ${N_SAMPLES} \
                --alpha "${ALPHA}" \
                --max_samples "${MAX_S}" \
                --output_dir "${OUT}"
        done
    done
}


# =============================================================================
# BLOCK 6 : Budget scaling
# BoN and DN-BoN with N ∈ {4, 8, 16, 32} on MATH-500 and AIME-24.
# 16 separate python calls.
# =============================================================================
run_scaling() {
    echo "===== BUDGET SCALING ====="

    for DATASET in math500 aime24; do
        MAX_S=200
        [[ "${DATASET}" == "aime24" ]] && MAX_S=32

        for N in 4 8 16 32; do
            run_exp "${DATASET}_bon_N${N}" \
                --dataset "${DATASET}" \
                --methods bon \
                --N "${N}" --batch_size "${N}" \
                --max_samples "${MAX_S}" \
                --output_dir "${OUT}"

            run_exp "${DATASET}_dnbon_N${N}_a0.5" \
                --dataset "${DATASET}" \
                --methods dn_bon \
                --N "${N}" --batch_size "${N}" \
                --alpha 0.5 \
                --max_samples "${MAX_S}" \
                --output_dir "${OUT}"
        done
    done
}


# =============================================================================
# BLOCK 7 : Clustering threshold ablation
# DN-BoN on MATH-500 (200 samples) with similarity_threshold ∈ {0.70..0.95}.
# 5 separate python calls.
# =============================================================================
run_clustering() {
    echo "===== CLUSTERING THRESHOLD ABLATION ====="

    for THRESH in 0.70 0.80 0.85 0.90 0.95; do
        run_exp "math500_dnbon_N8_thresh${THRESH}" \
            --dataset math500 \
            --methods dn_bon \
            --N ${N_SAMPLES} \
            --alpha 0.5 \
            --similarity_threshold "${THRESH}" \
            --max_samples 200 \
            --output_dir "${OUT}"
    done
}


# =============================================================================
# BLOCK 8 : Verifier ablation
# DN-BoN on MATH-500 (200 samples) with outcome vs length verifier.
#
#   outcome — binary correctness vs gold answer (best for eval)
#   length  — training-free proxy; use for deployment without gold answers
#
# To add PRM: set VERIFIER=prm in CONFIG above; an extra run_exp call is
# commented out below.
# 2 separate python calls.
# =============================================================================
run_verifier() {
    echo "===== VERIFIER ABLATION ====="

    run_exp "math500_dnbon_verifier_outcome" \
        --dataset math500 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --verifier outcome \
        --max_samples 200 \
        --output_dir "${OUT}"

    run_exp "math500_dnbon_verifier_length" \
        --dataset math500 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --verifier length \
        --max_samples 200 \
        --output_dir "${OUT}"

    # Uncomment to add PRM (set VERIFIER=prm in CONFIG first):
    # run_exp "math500_dnbon_verifier_prm" \
    #     --dataset math500 \
    #     --methods dn_bon \
    #     --N ${N_SAMPLES} \
    #     --verifier prm \
    #     --prm_model "${PRM_MODEL}" \
    #     --max_samples 200 \
    #     --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 9 : greedy + BoN + DN-BoN, batch_size=128
#
# Runs one or all datasets depending on the optional second argument.
#
#   bash run_experiments.sh bs128_all              # all datasets
#   bash run_experiments.sh bs128_all aime24       # aime24 only
#   bash run_experiments.sh bs128_all aime25       # aime25 only
#   bash run_experiments.sh bs128_all math500      # math500 only
#   bash run_experiments.sh bs128_all amc23        # amc23 only
#   bash run_experiments.sh bs128_all gsm8k        # gsm8k only
#
# Each dataset × method is ONE separate python call (GPU freed between runs).
# With all datasets: 5 × 3 = 15 python calls.
# With one dataset:       3 python calls.
# =============================================================================

# Per-dataset sample caps
_max_samples() {
    case "$1" in
        aime24|aime25) echo 32  ;;
        amc23)         echo 40  ;;
        math500|gsm8k) echo 500 ;;
        *)             echo 500 ;;
    esac
}

# Validate that a dataset name is supported
_check_dataset() {
    case "$1" in
        math500|aime24|aime25|amc23|gsm8k) return 0 ;;
        *)
            echo "ERROR: unknown dataset '$1'"
            echo "Valid datasets: math500 | aime24 | aime25 | amc23 | gsm8k"
            exit 1
            ;;
    esac
}

run_bs128_all() {
    local DATASET_FILTER="${1:-all}"
    # LM_BATCH_SIZE from CONFIG controls the actual HF model.generate() batch.
    # For Greedy: LM_BATCH_SIZE questions decoded together per generate() call.
    # For BoN/DN-BoN: N_SAMPLES prompts stacked per question per generate() call.
    # This block passes LM_BATCH_SIZE via common_flags (already wired through).
    local BS="${LM_BATCH_SIZE}"

    # Build the list of datasets to run
    local DATASETS
    if [[ "${DATASET_FILTER}" == "all" ]]; then
        DATASETS="math500 aime24 aime25 amc23 gsm8k"
    else
        _check_dataset "${DATASET_FILTER}"
        DATASETS="${DATASET_FILTER}"
    fi

    echo "===== greedy+BoN+DN-BoN  N=${N_SAMPLES}  batch_size=${BS}  datasets=[${DATASETS}] ====="

    for DATASET in ${DATASETS}; do
        local MAX_S
        MAX_S="$(_max_samples "${DATASET}")"
        echo "----- dataset=${DATASET}  max_samples=${MAX_S} -----"

        # Greedy
        run_exp "${DATASET}_greedy_N8" \
            --dataset "${DATASET}" \
            --methods greedy \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        # BoN (batch_size > N accepted by CLI; sampling still uses N=8 per question)
        run_exp "${DATASET}_bon_N8" \
            --dataset "${DATASET}" \
            --methods bon \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        # DN-BoN
        run_exp "${DATASET}_dnbon_N8_a0.5" \
            --dataset "${DATASET}" \
            --methods dn_bon \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --alpha 0.5 \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"
    done
}


# =============================================================================
# BLOCK 10 : DN-BoN only, batch_size=64
#
# Runs only DN-BoN across one or all datasets.
#
#   bash run_experiments.sh dnbon_bs64              # all datasets
#   bash run_experiments.sh dnbon_bs64 aime24       # aime24 only
#   bash run_experiments.sh dnbon_bs64 aime25       # aime25 only
#   bash run_experiments.sh dnbon_bs64 math500      # math500 only
#   bash run_experiments.sh dnbon_bs64 amc23        # amc23 only
#   bash run_experiments.sh dnbon_bs64 gsm8k        # gsm8k only
#
# With all datasets: 5 python calls.
# With one dataset:  1 python call.
# =============================================================================
run_dnbon_bs64() {
    local DATASET_FILTER="${1:-all}"
    # Uses LM_BATCH_SIZE from CONFIG — same batch size as every other block.
    local BS="${LM_BATCH_SIZE}"

    local DATASETS
    if [[ "${DATASET_FILTER}" == "all" ]]; then
        DATASETS="math500 aime24 aime25 amc23 gsm8k"
    else
        _check_dataset "${DATASET_FILTER}"
        DATASETS="${DATASET_FILTER}"
    fi

    echo "===== DN-BoN only  N=${N_SAMPLES}  batch_size=${BS}  datasets=[${DATASETS}] ====="

    for DATASET in ${DATASETS}; do
        local MAX_S
        MAX_S="$(_max_samples "${DATASET}")"
        echo "----- dataset=${DATASET}  max_samples=${MAX_S} -----"

        run_exp "${DATASET}_dnbon_N8_a0.5" \
            --dataset "${DATASET}" \
            --methods dn_bon \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --alpha 0.5 \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"
    done
}


# =============================================================================
# BLOCK 11 : ALL experiments for a SINGLE dataset
#
# Runs every experiment type (greedy, BoN, DN-BoN, alpha ablation, budget
# scaling, clustering threshold, verifier ablation) for one dataset.
#
# Usage:
#   bash run_experiments.sh all_for_dataset math500
#   bash run_experiments.sh all_for_dataset aime24
#   bash run_experiments.sh all_for_dataset aime25
#   bash run_experiments.sh all_for_dataset amc23
#   bash run_experiments.sh all_for_dataset gsm8k
#
# Each method/config is a separate python call.  GPU freed between runs.
# =============================================================================
run_all_for_dataset() {
    local DATASET="${1}"

    if [[ -z "${DATASET}" ]]; then
        echo "ERROR: all_for_dataset requires a dataset argument."
        echo "Usage: bash run_experiments.sh all_for_dataset <dataset>"
        echo "Valid: math500 | aime24 | aime25 | amc23 | gsm8k"
        exit 1
    fi
    _check_dataset "${DATASET}"

    local MAX_S
    MAX_S="$(_max_samples "${DATASET}")"
    local BS="${LM_BATCH_SIZE}"

    echo ""
    echo "############################################################"
    echo "  ALL EXPERIMENTS  →  dataset=${DATASET}"
    echo "  max_samples=${MAX_S}  N=${N_SAMPLES}  batch_size=${BS}"
    echo "  output → ${OUT}"
    echo "############################################################"

    # ── 1. Greedy ─────────────────────────────────────────────────────────────
    echo "----- [1/7] Greedy -----"
    run_exp "${DATASET}_greedy" \
        --dataset "${DATASET}" \
        --methods greedy \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    # ── 2. Best-of-N  (N from CONFIG) ─────────────────────────────────────────
    echo "----- [2/7] Best-of-N  N=${N_SAMPLES} -----"
    run_exp "${DATASET}_bon_N${N_SAMPLES}" \
        --dataset "${DATASET}" \
        --methods bon \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    # ── 3. DN-BoN  (alpha=0.5, N from CONFIG) ─────────────────────────────────
    echo "----- [3/7] DN-BoN  N=${N_SAMPLES}  alpha=0.5 -----"
    run_exp "${DATASET}_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset "${DATASET}" \
        --methods dn_bon \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --alpha 0.5 \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    # ── 4. Alpha ablation  (DN-BoN, α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}) ─────────
    # α=0 recovers plain BoN — clean ablation control point.
    echo "----- [4/7] Alpha ablation -----"
    for ALPHA in 0.0 0.25 0.5 0.75 1.0; do
        run_exp "${DATASET}_dnbon_N${N_SAMPLES}_a${ALPHA}" \
            --dataset "${DATASET}" \
            --methods dn_bon \
            --N "${N_SAMPLES}" \
            --batch_size "${BS}" \
            --alpha "${ALPHA}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"
    done

    # ── 5. Budget scaling  (BoN + DN-BoN, N ∈ {4, 8, 16, 32}) ───────────────
    # Shows accuracy vs test-time compute budget curve.
    echo "----- [5/7] Budget scaling  N=4,8,16,32 -----"
    for N in 4 8 16 32; do
        run_exp "${DATASET}_bon_N${N}" \
            --dataset "${DATASET}" \
            --methods bon \
            --N "${N}" \
            --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        run_exp "${DATASET}_dnbon_N${N}_a0.5" \
            --dataset "${DATASET}" \
            --methods dn_bon \
            --N "${N}" \
            --batch_size "${BS}" \
            --alpha 0.5 \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"
    done

    # ── 6. Clustering threshold ablation  (DN-BoN, threshold ∈ {0.70..0.95}) ─
    echo "----- [6/7] Clustering threshold ablation -----"
    for THRESH in 0.70 0.80 0.85 0.90 0.95; do
        run_exp "${DATASET}_dnbon_N${N_SAMPLES}_thresh${THRESH}" \
            --dataset "${DATASET}" \
            --methods dn_bon \
            --N "${N_SAMPLES}" \
            --batch_size "${BS}" \
            --alpha 0.5 \
            --similarity_threshold "${THRESH}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"
    done

    # ── 7. Verifier ablation  (outcome vs length) ─────────────────────────────
    echo "----- [7/7] Verifier ablation -----"
    run_exp "${DATASET}_dnbon_verifier_outcome" \
        --dataset "${DATASET}" \
        --methods dn_bon \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --verifier outcome \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    run_exp "${DATASET}_dnbon_verifier_length" \
        --dataset "${DATASET}" \
        --methods dn_bon \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --verifier length \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    echo ""
    echo "############################################################"
    echo "  DONE  →  dataset=${DATASET}"
    echo "  Results : ${OUT}"
    echo "############################################################"
}


# =============================================================================
# Dispatcher
# $1 = block name   $2 = optional dataset argument
# =============================================================================
BLOCK="${1:-all}"
DATASET_ARG="${2:-all}"   # passed through to blocks that accept a dataset filter

# Print resolved output directory so the user knows where files will land.
echo ""
echo "  Result folder : ${OUT}"
echo "  Run tag       : ${RUN_TAG}"
echo ""

case "${BLOCK}" in
    math500)          run_math500                         ;;
    aime24)           run_aime24                          ;;
    amc23)            run_amc23                           ;;
    gsm8k)            run_gsm8k                           ;;
    alpha)            run_alpha                           ;;
    scaling)          run_scaling                         ;;
    clustering)       run_clustering                      ;;
    verifier)         run_verifier                        ;;
    bs128_all)        run_bs128_all    "${DATASET_ARG}"   ;;
    dnbon_bs64)       run_dnbon_bs64   "${DATASET_ARG}"   ;;
    all_for_dataset)  run_all_for_dataset "${DATASET_ARG}" ;;
    all)
        run_math500
        run_aime24
        run_amc23
        run_gsm8k
        run_alpha
        run_scaling
        run_clustering
        run_verifier
        run_bs128_all  all
        run_dnbon_bs64 all
        ;;
    *)
        echo "Unknown block: '${BLOCK}'"
        echo ""
        echo "Usage:  bash run_experiments.sh <block> [dataset]"
        echo ""
        echo "Single-dataset blocks:"
        echo "  math500 | aime24 | amc23 | gsm8k"
        echo ""
        echo "Multi-experiment single-dataset block:"
        echo "  all_for_dataset <dataset>    — ALL experiments for one dataset"
        echo "  Valid datasets: math500 | aime24 | aime25 | amc23 | gsm8k"
        echo ""
        echo "Ablation blocks (run on fixed datasets internally):"
        echo "  alpha | scaling | clustering | verifier"
        echo ""
        echo "All-dataset blocks (optional dataset filter):"
        echo "  bs128_all  [dataset]         default: all datasets"
        echo "  dnbon_bs64 [dataset]         default: all datasets"
        echo ""
        echo "Run everything:"
        echo "  all"
        echo ""
        echo "Examples:"
        echo "  bash run_experiments.sh all_for_dataset math500"
        echo "  bash run_experiments.sh all_for_dataset aime24"
        echo "  bash run_experiments.sh all_for_dataset aime25"
        echo "  bash run_experiments.sh all_for_dataset gsm8k"
        echo "  bash run_experiments.sh bs128_all aime24"
        echo "  bash run_experiments.sh dnbon_bs64 math500"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " All experiments finished."
echo " Results : ${OUT}"
echo " Logs    : ${OUT}/logs/"
echo "============================================================"