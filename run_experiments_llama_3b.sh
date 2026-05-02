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
#   ./results/Qwen2.5-Math-7B-Instruct_mnt2048_t0.7_p0.95/
#   ./results/Qwen2.5-Math-7B-Instruct_mnt2048_t0.7_p0.95_4bit/
#   ./results/deepseek-math-7b_mnt4096_t0.8_p0.95/
#   ./results/Llama-3.1-8B-Instruct_mnt2048_t0.7_p0.95_8bit/
#
# The folder name encodes:
#   <ModelName>           — HF model id with org prefix stripped
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
#   bash run_experiments.sh minerva_math # Minerva Math
#   bash run_experiments.sh college_math # College Math
#   bash run_experiments.sh olympiadbench # OlympiadBench
#   bash run_experiments.sh alpha        # alpha ablation (α ∈ 0..1)
#   bash run_experiments.sh scaling      # budget scaling (N = 4,8,16,32)
#   bash run_experiments.sh clustering   # similarity-threshold ablation
#   bash run_experiments.sh verifier     # outcome vs length verifier
#   bash run_experiments.sh dora         # DN-BoN Updated (DORA) on all datasets
#   bash run_experiments.sh mmlu          # MMLU (all 57 subjects)
#   bash run_experiments.sh bbh           # BIG-Bench Hard (23 tasks)
#   bash run_experiments.sh arc_challenge # ARC-Challenge
#   bash run_experiments.sh truthfulqa    # TruthfulQA MC1
#   bash run_experiments.sh strategyqa    # StrategyQA yes/no
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
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# MODEL="meta-llama/Llama-3.2-1B-Instruct"
# MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
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
BASE_OUT="./experiments_results/results_updated"

# -----------------------------------------------------------------------------
# MAX_NEW_TOKENS
# Maximum tokens the model can generate per response.
#   1024  — fast, enough for MATH-500 / GSM-8K
#   2048  — safer for AIME / competition math with long derivations
#   4096  — very long chain-of-thought (slow but complete)
# Encoded in the result folder name as  mnt<value>
# -----------------------------------------------------------------------------
MAX_NEW_TOKENS=2048
# MAX_NEW_TOKENS=32768

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
LM_BATCH_SIZE=16

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
# Scoring back-end for BoN / DN-BoN / DN-BoN candidate selection.
#
# IMPORTANT — gold leakage:
#   outcome  — scores by comparing against the gold answer directly.
#              This LEAKS the gold answer into the selection step, which
#              artificially inflates BoN / DN-BoN accuracy and makes the
#              comparison with greedy unfair.  Do NOT use for experiments
#              that report accuracy — it is only valid as an oracle upper bound.
#
#   prm      — Process Reward Model scores each reasoning step without seeing
#              the gold answer.  This is the correct verifier for fair
#              evaluation.  Use this for all reported experiments.
#              Requires PRM_MODEL below.
#
#   length   — Training-free proxy (shorter + \boxed = higher score).
#              No gold needed.  Use only for deployment / no-label settings.
# -----------------------------------------------------------------------------
VERIFIER="prm"

# -----------------------------------------------------------------------------
# PRM_MODEL
# HF model id for the PRM verifier.  Only used when VERIFIER=prm.
# Recommended: Qwen/Qwen2.5-Math-PRM-7B  (~14 GB fp16)
# -----------------------------------------------------------------------------
PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B"

# -----------------------------------------------------------------------------
# PRM_AGGREGATION
# How to reduce per-step PRM scores to one score per trace.
#   min   — most conservative; penalises any weak step  (recommended)
#   mean  — average quality across all steps
#   last  — quality of the final answer step only
# -----------------------------------------------------------------------------
PRM_AGGREGATION="mean"

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

# -----------------------------------------------------------------------------
# T_B / T_S  —  DORA (DN-BoN Updated) temperature hyperparameters
#
# Used only when running method dn_bon_updated.  Ignored by all other methods.
#
# T_B (quality softmax temperature):
#   Controls how sharply PRM scores are converted to quality weights.
#   Lower T_B → winner-takes-all (high-scoring trace gets almost all weight).
#   Higher T_B → more uniform weights across all candidates.
#   Recommended default: 1.0
#
# T_S (similarity softmax temperature):
#   Controls how sharply the soft affinity matrix peaks on the diagonal.
#   Lower T_S → γ_i is close to 0 for near-duplicates, close to 1 for unique traces.
#   Higher T_S → softer penalty; uniqueness scores become more uniform.
#   Recommended default: 0.1
# -----------------------------------------------------------------------------
T_B=1.0
T_S=0.1

# -----------------------------------------------------------------------------
# GPUS
# Which GPU(s) to use.  Sets CUDA_VISIBLE_DEVICES for every python subprocess.
#
# Single GPU examples:
#   GPUS="0"      — use GPU 0 only
#   GPUS="1"      — use GPU 1 only
#
# Multi-GPU examples:
#   GPUS="0,1"    — use GPUs 0 and 1 (model sharded automatically)
#   GPUS="0,1,2,3"— use all four GPUs
#
# Leave empty to use whatever CUDA_VISIBLE_DEVICES is already set in the
# environment (or all GPUs if it is not set).
# -----------------------------------------------------------------------------
GPUS=""

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
# GPU setup
# =============================================================================
if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    echo "[gpu] CUDA_VISIBLE_DEVICES=${GPUS}"
fi


# =============================================================================
# Internal helpers
# =============================================================================

NOW=$(date +"%Y%m%d_%H%M%S")

# Build the result-folder tag from model name + hyperparameters.
#
# Format:  <ModelName>_mnt<tokens>_t<temp>_p<top_p>[_4bit|_8bit]
#
# Model name sanitisation:
#   - Strip the HF org prefix (everything up to and including the last '/')
#   - Replace any remaining '/' or ':' with '_'
#
# Examples:
#   Qwen/Qwen2.5-Math-7B-Instruct  →  Qwen2.5-Math-7B-Instruct_mnt2048_t0.7_p0.95
#   deepseek-ai/deepseek-math-7b   →  deepseek-math-7b_mnt2048_t0.7_p0.95
#   /local/path/my_model           →  my_model_mnt2048_t0.7_p0.95
#   meta-llama/Llama-3.1-8B-Instruct_4bit  →  Llama-3.1-8B-Instruct_4bit_mnt2048_t0.7_p0.95_4bit
_build_run_tag() {
    # Strip org prefix: take only the part after the last '/'
    local model_short="${MODEL##*/}"
    # Replace any remaining unsafe chars (colons, spaces) with underscores
    model_short="${model_short//[: ]/_}"

    local tag="${model_short}_mnt${MAX_NEW_TOKENS}_t${TEMPERATURE}_p${TOP_P}"
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
    # DORA temperatures — only consumed by dn_bon_updated; harmlessly passed to others
    flags+=" --T_b ${T_B}"
    flags+=" --T_s ${T_S}"
    [[ -n "${HF_CACHE}" ]]      && flags+=" --hf_cache_dir ${HF_CACHE}"
    [[ "${LOAD_4BIT}" == "1" ]] && flags+=" --load_in_4bit"
    [[ "${LOAD_8BIT}" == "1" ]] && flags+=" --load_in_8bit"
    [[ "${VERIFIER}"  == "prm" ]] && flags+=" --prm_model ${PRM_MODEL}"
    [[ "${VERIFIER}"  == "prm" ]] && flags+=" --prm_aggregation ${PRM_AGGREGATION}"
    [[ -n "${GPUS}" ]]          && flags+=" --gpus ${GPUS}"
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

    # 1b. Best-of-N  (verifier score)
    run_exp "math500_bon_N${N_SAMPLES}" \
        --dataset math500 \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    # 1c. DN-BoN  (alpha=0.5)
    run_exp "math500_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset math500 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"

    # 1c2. DN-BoN Updated  (DORA: soft quality × uniqueness, T_b=${T_B}, T_s=${T_S})
    run_exp "math500_dnbon_updated_N${N_SAMPLES}" \
        --dataset math500 \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    # 1d. Greedy Pass@N  (select by max log-probability — no verifier needed)
    run_exp "math500_greedy_pass_n_N${N_SAMPLES}" \
        --dataset math500 \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    # 1e. BoN Majority  (self-consistency / majority vote)
    run_exp "math500_bon_majority_N${N_SAMPLES}" \
        --dataset math500 \
        --methods bon_majority \
        --N ${N_SAMPLES} \
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

    run_exp "aime24_bon_N${N_SAMPLES}" \
        --dataset aime24 \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 32 \
        --output_dir "${OUT}"

    run_exp "aime24_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset aime24 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 32 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "aime24_dnbon_updated_N${N_SAMPLES}" \
        --dataset aime24 \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 32 \
        --output_dir "${OUT}"

    run_exp "aime24_greedy_pass_n_N${N_SAMPLES}" \
        --dataset aime24 \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 32 \
        --output_dir "${OUT}"

    run_exp "aime24_bon_majority_N${N_SAMPLES}" \
        --dataset aime24 \
        --methods bon_majority \
        --N ${N_SAMPLES} \
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

    run_exp "amc23_bon_N${N_SAMPLES}" \
        --dataset amc23 \
        --methods bon \
        --N ${N_SAMPLES} \
        --output_dir "${OUT}"

    run_exp "amc23_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset amc23 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "amc23_dnbon_updated_N${N_SAMPLES}" \
        --dataset amc23 \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --output_dir "${OUT}"

    run_exp "amc23_greedy_pass_n_N${N_SAMPLES}" \
        --dataset amc23 \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --output_dir "${OUT}"

    run_exp "amc23_bon_majority_N${N_SAMPLES}" \
        --dataset amc23 \
        --methods bon_majority \
        --N ${N_SAMPLES} \
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

    run_exp "gsm8k_bon_N${N_SAMPLES}" \
        --dataset gsm8k \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "gsm8k_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset gsm8k \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "gsm8k_dnbon_updated_N${N_SAMPLES}" \
        --dataset gsm8k \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "gsm8k_greedy_pass_n_N${N_SAMPLES}" \
        --dataset gsm8k \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "gsm8k_bon_majority_N${N_SAMPLES}" \
        --dataset gsm8k \
        --methods bon_majority \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 4b : Minerva Math  (272 university-level problems)
#
# Covers calculus, linear algebra, differential equations, and other
# undergraduate topics written in natural mathematical language.
# All 272 test problems are run (max_samples=500 catches all of them).
# =============================================================================
run_minerva_math() {
    echo "===== MINERVA MATH ====="

    run_exp "minerva_math_greedy" \
        --dataset minerva_math \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "minerva_math_bon_N${N_SAMPLES}" \
        --dataset minerva_math \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "minerva_math_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset minerva_math \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "minerva_math_dnbon_updated_N${N_SAMPLES}" \
        --dataset minerva_math \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "minerva_math_greedy_pass_n_N${N_SAMPLES}" \
        --dataset minerva_math \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "minerva_math_bon_majority_N${N_SAMPLES}" \
        --dataset minerva_math \
        --methods bon_majority \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 4c : College Math  (TIGER-Lab/college-math, ~2818 problems)
#
# Covers a wide range of undergraduate mathematics topics.
# Capped at 500 for reasonable runtime; adjust MAX_SAMPLES below if needed.
# =============================================================================
run_college_math() {
    echo "===== COLLEGE MATH ====="

    run_exp "college_math_greedy" \
        --dataset college_math \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "college_math_bon_N${N_SAMPLES}" \
        --dataset college_math \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "college_math_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset college_math \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "college_math_dnbon_updated_N${N_SAMPLES}" \
        --dataset college_math \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "college_math_greedy_pass_n_N${N_SAMPLES}" \
        --dataset college_math \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "college_math_bon_majority_N${N_SAMPLES}" \
        --dataset college_math \
        --methods bon_majority \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 4d : OlympiadBench  (lmms-lab/OlympiadBench, competition-level)
#
# International math/physics olympiad problems — very hard.
# Capped at 200 for practical runtime (full set is ~900 EN math problems).
# Increase max_samples if compute allows.
# =============================================================================
run_olympiadbench() {
    echo "===== OLYMPIAD BENCH ====="

    run_exp "olympiadbench_greedy" \
        --dataset olympiadbench \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 200 \
        --output_dir "${OUT}"

    run_exp "olympiadbench_bon_N${N_SAMPLES}" \
        --dataset olympiadbench \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 200 \
        --output_dir "${OUT}"

    run_exp "olympiadbench_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset olympiadbench \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 200 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "olympiadbench_dnbon_updated_N${N_SAMPLES}" \
        --dataset olympiadbench \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 200 \
        --output_dir "${OUT}"

    run_exp "olympiadbench_greedy_pass_n_N${N_SAMPLES}" \
        --dataset olympiadbench \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 200 \
        --output_dir "${OUT}"

    run_exp "olympiadbench_bon_majority_N${N_SAMPLES}" \
        --dataset olympiadbench \
        --methods bon_majority \
        --N ${N_SAMPLES} \
        --max_samples 200 \
        --output_dir "${OUT}"
}
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
# DN-BoN on MATH-500 (200 samples) comparing the three verifier back-ends.
#
# Correct interpretation:
#   prm     — Process Reward Model; no gold leakage; THIS IS THE REAL METHOD.
#             Used as the default VERIFIER in CONFIG.
#   length  — Training-free proxy; no gold; realistic deployment baseline.
#   outcome — Sees the gold answer during selection; ORACLE UPPER BOUND ONLY.
#             Shows the ceiling if we had a perfect verifier.
#             Do NOT report this as a fair comparison to greedy/BoN baselines.
#
# 3 separate python calls.
# =============================================================================
run_verifier() {
    echo "===== VERIFIER ABLATION ====="

    # PRM — the real method; no gold leakage
    run_exp "math500_dnbon_verifier_prm" \
        --dataset math500 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --verifier prm \
        --prm_model "${PRM_MODEL}" \
        --prm_aggregation "${PRM_AGGREGATION}" \
        --max_samples 200 \
        --output_dir "${OUT}"

    # Length heuristic — no gold, no PRM; deployment baseline
    run_exp "math500_dnbon_verifier_length" \
        --dataset math500 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --verifier length \
        --max_samples 200 \
        --output_dir "${OUT}"

    # Outcome — oracle upper bound ONLY (leaks gold into selection)
    run_exp "math500_dnbon_verifier_outcome_ORACLE" \
        --dataset math500 \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --verifier outcome \
        --max_samples 200 \
        --output_dir "${OUT}"
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
        aime24|aime25)  echo 32   ;;
        amc23)          echo 40   ;;
        math500|gsm8k)  echo 500  ;;
        minerva_math)   echo 500  ;;   # 272 test problems — cap at 500 (all)
        college_math)   echo 500  ;;   # ~2818 problems — cap at 500 for speed
        olympiadbench)  echo 200  ;;   # competition-level; 200 is representative
        mmlu)           echo 500  ;;   # 57 subjects × ~100 per subject; 500 balanced sample
        bbh)            echo 500  ;;   # 23 tasks × ~250 per task; 500 balanced sample
        arc_challenge)  echo 500  ;;   # 1172 test questions; cap at 500 for speed
        truthfulqa)     echo 500  ;;   # 817 validation questions; all fit under cap
        strategyqa)     echo 500  ;;   # 2290 test questions; cap at 500
        *)              echo 500  ;;
    esac
}

# Validate that a dataset name is supported
_check_dataset() {
    case "$1" in
        math500|aime24|aime25|amc23|gsm8k|\
        minerva_math|college_math|olympiadbench|\
        mmlu|bbh|arc_challenge|truthfulqa|strategyqa) return 0 ;;
        *)
            echo "ERROR: unknown dataset '$1'"
            echo "Valid datasets: math500 | aime24 | aime25 | amc23 | gsm8k"
            echo "               minerva_math | college_math | olympiadbench"
            echo "               mmlu | bbh | arc_challenge | truthfulqa | strategyqa"
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
        DATASETS="math500 aime24 aime25 amc23 gsm8k minerva_math college_math olympiadbench mmlu bbh arc_challenge truthfulqa strategyqa"
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
        run_exp "${DATASET}_greedy_N${N_SAMPLES}" \
            --dataset "${DATASET}" \
            --methods greedy \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        # BoN (verifier score)
        run_exp "${DATASET}_bon_N${N_SAMPLES}" \
            --dataset "${DATASET}" \
            --methods bon \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        # DN-BoN
        run_exp "${DATASET}_dnbon_N${N_SAMPLES}_a0.5" \
            --dataset "${DATASET}" \
            --methods dn_bon \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --alpha 0.5 \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        # DN-BoN Updated  (DORA: soft quality × uniqueness)
        run_exp "${DATASET}_dnbon_updated_N${N_SAMPLES}" \
            --dataset "${DATASET}" \
            --methods dn_bon_updated \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        # Greedy Pass@N  (log-probability selection)
        run_exp "${DATASET}_greedy_pass_n_N${N_SAMPLES}" \
            --dataset "${DATASET}" \
            --methods greedy_pass_n \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        # BoN Majority  (self-consistency)
        run_exp "${DATASET}_bon_majority_N${N_SAMPLES}" \
            --dataset "${DATASET}" \
            --methods bon_majority \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
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
        DATASETS="math500 aime24 aime25 amc23 gsm8k minerva_math college_math olympiadbench mmlu bbh arc_challenge truthfulqa strategyqa"
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
# BLOCK 10b : DN-BoN Updated (DORA) only
#
# Runs only dn_bon_updated across one or all datasets.
# Uses soft quality × uniqueness scoring; no hard cluster labels.
# T_b and T_s are taken from CONFIG.
#
#   bash run_experiments.sh dora              # all datasets
#   bash run_experiments.sh dora aime24       # aime24 only
#   bash run_experiments.sh dora math500      # math500 only
#
# With all datasets: 8 python calls.
# With one dataset:  1 python call.
# =============================================================================
run_dora() {
    local DATASET_FILTER="${1:-all}"
    local BS="${LM_BATCH_SIZE}"

    local DATASETS
    if [[ "${DATASET_FILTER}" == "all" ]]; then
        DATASETS="math500 aime24 aime25 amc23 gsm8k minerva_math college_math olympiadbench mmlu bbh arc_challenge truthfulqa strategyqa"
    else
        _check_dataset "${DATASET_FILTER}"
        DATASETS="${DATASET_FILTER}"
    fi

    echo "===== DORA (DN-BoN Updated)  N=${N_SAMPLES}  T_b=${T_B}  T_s=${T_S}  batch_size=${BS}  datasets=[${DATASETS}] ====="

    for DATASET in ${DATASETS}; do
        local MAX_S
        MAX_S="$(_max_samples "${DATASET}")"
        echo "----- dataset=${DATASET}  max_samples=${MAX_S} -----"

        run_exp "${DATASET}_dnbon_updated_N${N_SAMPLES}" \
            --dataset "${DATASET}" \
            --methods dn_bon_updated \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"
    done
}


# =============================================================================
# BLOCK 12 : New methods only — Greedy Pass@N and BoN Majority
#
# Runs only the two new selection strategies across one or all datasets.
# Neither method requires a verifier or cluster manager.
#
#   bash run_experiments.sh new_methods              # all datasets
#   bash run_experiments.sh new_methods math500      # math500 only
#   bash run_experiments.sh new_methods aime24       # aime24 only
#
# Total per dataset: 2 python calls (1 per method).
# =============================================================================
run_new_methods() {
    local DATASET_FILTER="${1:-all}"
    local BS="${LM_BATCH_SIZE}"

    local DATASETS
    if [[ "${DATASET_FILTER}" == "all" ]]; then
        DATASETS="math500 aime24 aime25 amc23 gsm8k minerva_math college_math olympiadbench mmlu bbh arc_challenge truthfulqa strategyqa"
    else
        _check_dataset "${DATASET_FILTER}"
        DATASETS="${DATASET_FILTER}"
    fi

    echo "===== NEW METHODS  (greedy_pass_n + bon_majority)  N=${N_SAMPLES}  datasets=[${DATASETS}] ====="

    for DATASET in ${DATASETS}; do
        local MAX_S
        MAX_S="$(_max_samples "${DATASET}")"
        echo "----- dataset=${DATASET}  max_samples=${MAX_S} -----"

        # ── Greedy Pass@N ─────────────────────────────────────────────────────
        # Sample N traces, select by max mean per-token log-probability.
        # No verifier or cluster manager needed.
        run_exp "${DATASET}_greedy_pass_n_N${N_SAMPLES}" \
            --dataset "${DATASET}" \
            --methods greedy_pass_n \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        # ── BoN Majority  (self-consistency) ──────────────────────────────────
        # Sample N traces, group by equivalent answer, pick majority group.
        # No verifier or cluster manager needed.
        run_exp "${DATASET}_bon_majority_N${N_SAMPLES}" \
            --dataset "${DATASET}" \
            --methods bon_majority \
            --N ${N_SAMPLES} \
            --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"
    done
}


# =============================================================================
# BLOCK 11 : ALL experiments for a SINGLE dataset
#
# Runs every experiment type for one dataset:
#   greedy | bon | dn_bon | greedy_pass_n | bon_majority
#   + alpha ablation + budget scaling + clustering threshold + verifier ablation
#
# Usage:
#   bash run_experiments.sh all_for_dataset math500
#   bash run_experiments.sh all_for_dataset aime24
#   bash run_experiments.sh all_for_dataset aime25
#   bash run_experiments.sh all_for_dataset amc23
#   bash run_experiments.sh all_for_dataset gsm8k
#
# Total per dataset: 21 separate python calls.
# =============================================================================
run_all_for_dataset() {
    local DATASET="${1}"

    if [[ -z "${DATASET}" ]]; then
        echo "ERROR: all_for_dataset requires a dataset argument."
        echo "Usage: bash run_experiments.sh all_for_dataset <dataset>"
        echo "Valid: math500 | aime24 | aime25 | amc23 | gsm8k"
        echo "       minerva_math | college_math | olympiadbench"
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
    echo "----- [1/9] Greedy -----"
    run_exp "${DATASET}_greedy" \
        --dataset "${DATASET}" \
        --methods greedy \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    # ── 2. Best-of-N  (verifier score) ────────────────────────────────────────
    echo "----- [2/9] Best-of-N  N=${N_SAMPLES} -----"
    run_exp "${DATASET}_bon_N${N_SAMPLES}" \
        --dataset "${DATASET}" \
        --methods bon \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    # ── 3. DN-BoN  (alpha=0.5) ────────────────────────────────────────────────
    echo "----- [3/10] DN-BoN  N=${N_SAMPLES}  alpha=0.5 -----"
    run_exp "${DATASET}_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset "${DATASET}" \
        --methods dn_bon \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --alpha 0.5 \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    # ── 4. DN-BoN Updated  (DORA: soft quality × uniqueness) ─────────────────
    # Uses the DORA scoring rule: quality weights (PRM softmax) multiplied by
    # uniqueness scores (diagonal of the soft affinity matrix).  T_b and T_s
    # are taken from CONFIG.  No --alpha needed; no hard cluster labels used.
    echo "----- [4/10] DN-BoN Updated (DORA)  N=${N_SAMPLES}  T_b=${T_B}  T_s=${T_S} -----"
    run_exp "${DATASET}_dnbon_updated_N${N_SAMPLES}" \
        --dataset "${DATASET}" \
        --methods dn_bon_updated \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    # ── 5. Greedy Pass@N  (log-probability selection) ─────────────────────────
    # Select from N sampled traces by max mean per-token log-prob.
    # No verifier needed — uses the LM's own probability as the signal.
    echo "----- [5/10] Greedy Pass@N  N=${N_SAMPLES} -----"
    run_exp "${DATASET}_greedy_pass_n_N${N_SAMPLES}" \
        --dataset "${DATASET}" \
        --methods greedy_pass_n \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    # ── 6. BoN Majority  (self-consistency / majority vote) ───────────────────
    # Select from N sampled traces by the most-voted answer group.
    # No verifier or cluster manager needed.
    echo "----- [6/10] BoN Majority  N=${N_SAMPLES} -----"
    run_exp "${DATASET}_bon_majority_N${N_SAMPLES}" \
        --dataset "${DATASET}" \
        --methods bon_majority \
        --N "${N_SAMPLES}" \
        --batch_size "${BS}" \
        --max_samples "${MAX_S}" \
        --output_dir "${OUT}"

    # ── 7. Alpha ablation  (DN-BoN, α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}) ─────────
    echo "----- [7/10] Alpha ablation -----"
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

    # ── 8. Budget scaling  (all methods including dn_bon_updated, N ∈ {4,8,16,32}) ──
    echo "----- [8/10] Budget scaling  N=4,8,16,32 -----"
    for N in 4 8 16 32; do
        run_exp "${DATASET}_bon_N${N}" \
            --dataset "${DATASET}" \
            --methods bon \
            --N "${N}" --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        run_exp "${DATASET}_dnbon_N${N}_a0.5" \
            --dataset "${DATASET}" \
            --methods dn_bon \
            --N "${N}" --batch_size "${BS}" \
            --alpha 0.5 \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        # DN-BoN Updated at each budget N
        run_exp "${DATASET}_dnbon_updated_N${N}" \
            --dataset "${DATASET}" \
            --methods dn_bon_updated \
            --N "${N}" --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        run_exp "${DATASET}_greedy_pass_n_N${N}" \
            --dataset "${DATASET}" \
            --methods greedy_pass_n \
            --N "${N}" --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"

        run_exp "${DATASET}_bon_majority_N${N}" \
            --dataset "${DATASET}" \
            --methods bon_majority \
            --N "${N}" --batch_size "${BS}" \
            --max_samples "${MAX_S}" \
            --output_dir "${OUT}"
    done

    # ── 9. Clustering threshold ablation  (DN-BoN) ────────────────────────────
    echo "----- [9/10] Clustering threshold ablation -----"
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

    # ── 10. Verifier ablation  (outcome vs length) ─────────────────────────────
    echo "----- [10/10] Verifier ablation -----"
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
    echo "  DONE  →  dataset=${DATASET}  (all 10 experiment types)"
    echo "  Results : ${OUT}"
    echo "############################################################"
}



# =============================================================================
# BLOCK 13 : Custom run — any combination of datasets and methods
#
# The most flexible entry point.  Pass a comma-separated list of datasets
# and a comma-separated list of methods.  Use "all" as a shorthand for
# every dataset or every method.
#
# Usage
# -----
#   bash run_experiments.sh run <datasets> <methods>
#
#   <datasets> : comma-separated list or "all"
#     Valid values: math500, aime24, aime25, amc23, gsm8k,
#                    minerva_math, college_math, olympiadbench,
#                    mmlu, bbh, arc_challenge, truthfulqa, strategyqa
#     "all"  expands to all of the above
#
#   <methods>  : comma-separated list or "all"
#     Valid values: greedy, bon, dn_bon, dn_bon_updated, greedy_pass_n, bon_majority
#     "all"  expands to all of the above
#
# Examples
# --------
#   # Single dataset, single method
#   bash run_experiments.sh run math500 greedy
#
#   # Single dataset, multiple methods
#   bash run_experiments.sh run math500 greedy,bon,dn_bon
#
#   # Multiple datasets, single method
#   bash run_experiments.sh run math500,aime24,aime25 dn_bon
#
#   # Multiple datasets, multiple methods
#   bash run_experiments.sh run math500,aime24 greedy,bon,dn_bon,greedy_pass_n,bon_majority
#
#   # All datasets, all methods
#   bash run_experiments.sh run all all
#
#   # All datasets, specific methods
#   bash run_experiments.sh run all greedy,bon_majority
#
#   # Specific datasets, all methods
#   bash run_experiments.sh run math500,aime24 all
#
# Each (dataset, method) pair is ONE separate python call — GPU freed between.
# Total calls = len(datasets) × len(methods).
# =============================================================================

ALL_DATASETS="math500 aime24 aime25 amc23 gsm8k minerva_math college_math olympiadbench mmlu bbh arc_challenge truthfulqa strategyqa"
ALL_METHODS="greedy bon dn_bon dn_bon_updated greedy_pass_n bon_majority"

# Validate a single method name
_check_method() {
    case "$1" in
        greedy|bon|dn_bon|dn_bon_updated|greedy_pass_n|bon_majority) return 0 ;;
        *)
            echo "ERROR: unknown method '$1'"
            echo "Valid methods: greedy | bon | dn_bon | dn_bon_updated | greedy_pass_n | bon_majority"
            exit 1
            ;;
    esac
}

run_custom() {
    local DATASETS_ARG="${1:-all}"
    local METHODS_ARG="${2:-all}"
    local BS="${LM_BATCH_SIZE}"

    # ── Expand "all" shorthand ─────────────────────────────────────────────────
    local DATASETS
    if [[ "${DATASETS_ARG}" == "all" ]]; then
        DATASETS="${ALL_DATASETS}"
    else
        # Replace commas with spaces
        DATASETS="${DATASETS_ARG//,/ }"
    fi

    local METHODS
    if [[ "${METHODS_ARG}" == "all" ]]; then
        METHODS="${ALL_METHODS}"
    else
        METHODS="${METHODS_ARG//,/ }"
    fi

    # ── Validate all dataset and method names before running anything ──────────
    for D in ${DATASETS}; do _check_dataset "${D}"; done
    for M in ${METHODS};  do _check_method  "${M}"; done

    # ── Count total runs ───────────────────────────────────────────────────────
    local N_D=0; for D in ${DATASETS}; do N_D=$((N_D+1)); done
    local N_M=0; for M in ${METHODS};  do N_M=$((N_M+1)); done
    local TOTAL=$((N_D * N_M))

    echo ""
    echo "############################################################"
    echo "  CUSTOM RUN"
    echo "  datasets : ${DATASETS}"
    echo "  methods  : ${METHODS}"
    echo "  N        : ${N_SAMPLES}   batch_size : ${BS}"
    echo "  total    : ${TOTAL} python calls  (${N_D} datasets × ${N_M} methods)"
    echo "  output   : ${OUT}"
    echo "############################################################"
    echo ""

    local RUN_IDX=0

    for DATASET in ${DATASETS}; do
        local MAX_S
        MAX_S="$(_max_samples "${DATASET}")"

        echo "----- dataset=${DATASET}  max_samples=${MAX_S} -----"

        for METHOD in ${METHODS}; do
            RUN_IDX=$((RUN_IDX+1))
            echo "  [${RUN_IDX}/${TOTAL}]  method=${METHOD}"

            local TAG="${DATASET}_${METHOD}_N${N_SAMPLES}"

            # dn_bon needs --alpha; dn_bon_updated uses T_b/T_s (already in common_flags)
            if [[ "${METHOD}" == "dn_bon" ]]; then
                run_exp "${TAG}_a0.5" \
                    --dataset "${DATASET}" \
                    --methods "${METHOD}" \
                    --N "${N_SAMPLES}" \
                    --batch_size "${BS}" \
                    --alpha 0.5 \
                    --max_samples "${MAX_S}" \
                    --output_dir "${OUT}"
            else
                run_exp "${TAG}" \
                    --dataset "${DATASET}" \
                    --methods "${METHOD}" \
                    --N "${N_SAMPLES}" \
                    --batch_size "${BS}" \
                    --max_samples "${MAX_S}" \
                    --output_dir "${OUT}"
            fi
        done
    done

    echo ""
    echo "############################################################"
    echo "  CUSTOM RUN COMPLETE — ${TOTAL} experiments finished"
    echo "  Results : ${OUT}"
    echo "############################################################"
}


# =============================================================================
# BLOCK 4e : MMLU  (cais/mmlu, 57 subjects, multiple-choice A/B/C/D)
#
# The 'all' config loads all 57 subjects together.
# Cap at 500 for a balanced, computationally feasible sample.
# Answer format: single letter A/B/C/D.
# =============================================================================
run_mmlu() {
    echo "===== MMLU ====="

    run_exp "mmlu_greedy" \
        --dataset mmlu \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "mmlu_bon_N${N_SAMPLES}" \
        --dataset mmlu \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "mmlu_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset mmlu \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "mmlu_dnbon_updated_N${N_SAMPLES}" \
        --dataset mmlu \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "mmlu_greedy_pass_n_N${N_SAMPLES}" \
        --dataset mmlu \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "mmlu_bon_majority_N${N_SAMPLES}" \
        --dataset mmlu \
        --methods bon_majority \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 4f : BBH  (lukaemon/bbh, BIG-Bench Hard — 23 reasoning tasks)
#
# Free-form answers; evaluation uses normalised string matching.
# Cap at 500 to cover a representative sample across tasks.
# =============================================================================
run_bbh() {
    echo "===== BIG-BENCH HARD ====="

    run_exp "bbh_greedy" \
        --dataset bbh \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "bbh_bon_N${N_SAMPLES}" \
        --dataset bbh \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "bbh_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset bbh \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "bbh_dnbon_updated_N${N_SAMPLES}" \
        --dataset bbh \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "bbh_greedy_pass_n_N${N_SAMPLES}" \
        --dataset bbh \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "bbh_bon_majority_N${N_SAMPLES}" \
        --dataset bbh \
        --methods bon_majority \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 4g : ARC-Challenge  (allenai/ai2_arc, 1172 test questions)
#
# Science multiple-choice; harder subset of ARC.
# Answer format: single letter A/B/C/D.
# Cap at 500 for speed; reduce if needed.
# =============================================================================
run_arc_challenge() {
    echo "===== ARC-CHALLENGE ====="

    run_exp "arc_challenge_greedy" \
        --dataset arc_challenge \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "arc_challenge_bon_N${N_SAMPLES}" \
        --dataset arc_challenge \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "arc_challenge_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset arc_challenge \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "arc_challenge_dnbon_updated_N${N_SAMPLES}" \
        --dataset arc_challenge \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "arc_challenge_greedy_pass_n_N${N_SAMPLES}" \
        --dataset arc_challenge \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "arc_challenge_bon_majority_N${N_SAMPLES}" \
        --dataset arc_challenge \
        --methods bon_majority \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 4h : TruthfulQA  (truthfulqa/truthful_qa, MC1 split, 817 questions)
#
# Tests whether the model avoids generating false beliefs.
# Gold answer is the correct choice text (not a letter).
# All 817 validation questions fit under the 500-cap if reduced; use 500.
# =============================================================================
run_truthfulqa() {
    echo "===== TRUTHFULQA ====="

    run_exp "truthfulqa_greedy" \
        --dataset truthfulqa \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "truthfulqa_bon_N${N_SAMPLES}" \
        --dataset truthfulqa \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "truthfulqa_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset truthfulqa \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "truthfulqa_dnbon_updated_N${N_SAMPLES}" \
        --dataset truthfulqa \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "truthfulqa_greedy_pass_n_N${N_SAMPLES}" \
        --dataset truthfulqa \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "truthfulqa_bon_majority_N${N_SAMPLES}" \
        --dataset truthfulqa \
        --methods bon_majority \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"
}


# =============================================================================
# BLOCK 4i : StrategyQA  (wics/strategy-qa, 2290 test questions)
#
# Binary yes/no questions requiring multi-step implicit reasoning.
# Answer format: "yes" or "no".
# Cap at 500 for speed.
# =============================================================================
run_strategyqa() {
    echo "===== STRATEGYQA ====="

    run_exp "strategyqa_greedy" \
        --dataset strategyqa \
        --methods greedy \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "strategyqa_bon_N${N_SAMPLES}" \
        --dataset strategyqa \
        --methods bon \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "strategyqa_dnbon_N${N_SAMPLES}_a0.5" \
        --dataset strategyqa \
        --methods dn_bon \
        --N ${N_SAMPLES} \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${OUT}"

    # DN-BoN Updated  (DORA)
    run_exp "strategyqa_dnbon_updated_N${N_SAMPLES}" \
        --dataset strategyqa \
        --methods dn_bon_updated \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "strategyqa_greedy_pass_n_N${N_SAMPLES}" \
        --dataset strategyqa \
        --methods greedy_pass_n \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"

    run_exp "strategyqa_bon_majority_N${N_SAMPLES}" \
        --dataset strategyqa \
        --methods bon_majority \
        --N ${N_SAMPLES} \
        --max_samples 500 \
        --output_dir "${OUT}"
}


# =============================================================================
# Dispatcher
# $1 = block name
# $2 = dataset argument  (for blocks that take a dataset, or <datasets> for run)
# $3 = methods argument  (only for the "run" block)
# =============================================================================
BLOCK="${1:-all}"
DATASET_ARG="${2:-all}"
METHODS_ARG="${3:-all}"

echo ""
echo "  Result folder : ${OUT}"
echo "  Run tag       : ${RUN_TAG}"
echo ""

case "${BLOCK}" in
    math500)          run_math500                               ;;
    aime24)           run_aime24                                ;;
    amc23)            run_amc23                                 ;;
    gsm8k)            run_gsm8k                                 ;;
    minerva_math)     run_minerva_math                          ;;
    college_math)     run_college_math                          ;;
    olympiadbench)    run_olympiadbench                         ;;
    alpha)            run_alpha                                 ;;
    scaling)          run_scaling                               ;;
    clustering)       run_clustering                            ;;
    verifier)         run_verifier                              ;;
    dora)             run_dora            "${DATASET_ARG}"      ;;
    new_methods)      run_new_methods     "${DATASET_ARG}"      ;;
    bs128_all)        run_bs128_all       "${DATASET_ARG}"      ;;
    dnbon_bs64)       run_dnbon_bs64      "${DATASET_ARG}"      ;;
    all_for_dataset)  run_all_for_dataset "${DATASET_ARG}"      ;;
    run)              run_custom          "${DATASET_ARG}" "${METHODS_ARG}" ;;
    all)
        run_math500
        run_aime24
        run_amc23
        run_gsm8k
        run_minerva_math
        run_college_math
        run_olympiadbench
        run_alpha
        run_scaling
        run_clustering
        run_verifier
        run_dora        all
        run_new_methods all
        run_bs128_all   all
        run_dnbon_bs64  all
        ;;
    *)
        echo "Unknown block: '${BLOCK}'"
        echo ""
        echo "Usage:  bash run_experiments.sh <block> [args]"
        echo ""
        echo "──────────────────────────────────────────────────────────────"
        echo "  FLEXIBLE  run block (most useful)"
        echo "──────────────────────────────────────────────────────────────"
        echo "  run <datasets> <methods>"
        echo ""
        echo "  <datasets>: comma-separated or 'all'"
        echo "    math500 | aime24 | aime25 | amc23 | gsm8k"
        echo "    minerva_math | college_math | olympiadbench"
        echo "    mmlu | bbh | arc_challenge | truthfulqa | strategyqa"
        echo ""
        echo "  <methods>: comma-separated or 'all'"
        echo "    greedy | bon | dn_bon | dn_bon_updated | greedy_pass_n | bon_majority"
        echo ""
        echo "  Examples:"
        echo "    bash run_experiments.sh run math500 greedy"
        echo "    bash run_experiments.sh run math500 greedy,bon,dn_bon"
        echo "    bash run_experiments.sh run math500,aime24 dn_bon"
        echo "    bash run_experiments.sh run math500,aime24 dn_bon,dn_bon_updated"
        echo "    bash run_experiments.sh run mmlu,bbh,arc_challenge all"
        echo "    bash run_experiments.sh run minerva_math,college_math,olympiadbench all"
        echo "    bash run_experiments.sh run math500,aime24,aime25 greedy,bon,dn_bon,dn_bon_updated,greedy_pass_n,bon_majority"
        echo "    bash run_experiments.sh run all all"
        echo "    bash run_experiments.sh run all greedy,bon_majority"
        echo "    bash run_experiments.sh run math500,aime24 all"
        echo ""
        echo "──────────────────────────────────────────────────────────────"
        echo "  FIXED blocks"
        echo "──────────────────────────────────────────────────────────────"
        echo "  math500 | aime24 | amc23 | gsm8k        per-dataset (all 6 methods)"
        echo "  minerva_math | college_math | olympiadbench"
        echo "  mmlu | bbh | arc_challenge | truthfulqa | strategyqa"
        echo "  dora [dataset]                           DN-BoN Updated (DORA) only"
        echo "  new_methods [dataset]                    greedy_pass_n + bon_majority"
        echo "  all_for_dataset <dataset>                all 10 experiment types"
        echo "  alpha | scaling | clustering | verifier  ablation studies"
        echo "  bs128_all  [dataset]                     all methods, all datasets"
        echo "  dnbon_bs64 [dataset]                     DN-BoN (hard) only"
        echo "  all                                      run everything"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " All experiments finished."
echo " Results : ${OUT}"
echo " Logs    : ${OUT}/logs/"
echo "============================================================"

# =============================================================================
# Auto-plot: accuracy vs effective token length
# =============================================================================
# Run plot_token_accuracy.py automatically if matplotlib is available and
# there are JSONL result files in the output directory.
# Set AUTO_PLOT=0 in CONFIG to disable.
# =============================================================================
AUTO_PLOT="${AUTO_PLOT:-1}"   # default on; override with: AUTO_PLOT=0 bash run_experiments.sh ...

if [[ "${AUTO_PLOT}" == "1" ]]; then
    JSONL_COUNT=$(ls "${OUT}"/*.jsonl 2>/dev/null | grep -v "_summary" | wc -l || echo 0)
    if [[ "${JSONL_COUNT}" -gt 0 ]]; then
        echo ""
        echo "  Generating accuracy vs token-length plots..."
        PLOT_ACC="${OUT}/accuracy_vs_tokens.png"
        PLOT_DIST="${OUT}/token_distribution.png"
        CSV_OUT="${OUT}/accuracy_vs_tokens.csv"
        ${PYTHON} plot_token_accuracy.py \
            --dir      "${OUT}" \
            --model    "${MODEL}" \
            --bins     10 \
            --min_count 3 \
            --out      "${PLOT_ACC}" \
            --out_dist "${PLOT_DIST}" \
            --csv      "${CSV_OUT}" \
            --title    "Accuracy vs Token Length  [${RUN_TAG}]" \
            2>&1 || echo "  [warn] plot_token_accuracy.py failed — run manually"
        echo "  Accuracy plot  : ${PLOT_ACC}"
        echo "  Distribution   : ${PLOT_DIST}"
        echo "  CSV            : ${CSV_OUT}"
        echo ""
        echo "  To regenerate or customise:"
        echo "    python plot_token_accuracy.py --dir ${OUT} --model ${MODEL} --bins 15"
        echo "    python plot_token_accuracy.py --dir ${OUT} --model ${MODEL} --filter_method greedy_pass_n"
        echo "    python plot_token_accuracy.py --dir ${OUT} --use_chars   # raw character length"
    fi
fi

# # ── single dataset, single method ──────────────────────────────────
# bash run_experiments.sh run math500 greedy
# bash run_experiments.sh run aime24 dn_bon
# bash run_experiments.sh run gsm8k bon_majority

# # ── single dataset, multiple methods ───────────────────────────────
# bash run_experiments.sh run math500 greedy,bon,dn_bon
# bash run_experiments.sh run aime24 greedy,bon,dn_bon,greedy_pass_n,bon_majority

# # ── single dataset, ALL methods ────────────────────────────────────
# bash run_experiments.sh run math500 all
# bash run_experiments.sh run aime25 all

# # ── multiple datasets, single method ───────────────────────────────
# bash run_experiments.sh run math500,aime24 dn_bon
# bash run_experiments.sh run math500,aime24,aime25 greedy_pass_n

# # ── multiple datasets, multiple methods ────────────────────────────
# bash run_experiments.sh run math500,aime24 greedy,bon,dn_bon
# bash run_experiments.sh run math500,aime24,aime25 greedy,bon,dn_bon,greedy_pass_n,bon_majority

# # ── ALL datasets, ALL methods ───────────────────────────────────────
# bash run_experiments.sh run all all

# # ── ALL datasets, specific methods ─────────────────────────────────
# bash run_experiments.sh run all greedy,bon_majority
# bash run_experiments.sh run all dn_bon

# # ── specific datasets, ALL methods ─────────────────────────────────
# bash run_experiments.sh run math500,aime24 all



# bash run_experiments.sh run olympiadbench,math500,aime24,aime25,amc23 greedy,bon,dn_bon,greedy_pass_n,bon_majority

# bash run_experiments.sh run mmlu,bbh,arc_challenge,truthfulqa,strategyqa greedy,greedy_pass_n,bon,dn_bon,bon_majority

