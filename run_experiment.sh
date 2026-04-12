#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# =============================================================================
# Runs every experiment as a SEPARATE python process so the GPU is fully
# released and its memory freed between runs.
#
# Usage
# -----
#   bash run_experiments.sh              # run ALL blocks sequentially
#   bash run_experiments.sh math500      # MATH-500 only  (greedy+BoN+DN-BoN)
#   bash run_experiments.sh aime24       # AIME-24 only
#   bash run_experiments.sh amc23        # AMC-23 only
#   bash run_experiments.sh gsm8k        # GSM-8K only
#   bash run_experiments.sh alpha        # alpha ablation (α ∈ 0..1)
#   bash run_experiments.sh scaling      # budget scaling (N = 4,8,16,32)
#   bash run_experiments.sh clustering   # similarity-threshold ablation
#   bash run_experiments.sh verifier     # outcome vs length verifier
#
# Every run streams stdout/stderr to the terminal AND writes a timestamped
# log file under ${BASE_OUT}/logs/.
# =============================================================================

set -euo pipefail


# =============================================================================
# CONFIG
# =============================================================================
# Edit this block to match your environment.
# Every variable below is documented with its allowed values and effect.
# =============================================================================

# -----------------------------------------------------------------------------
# MODEL
# Recommended choices:
#   Qwen/Qwen2.5-Math-7B-Instruct    — strong math, chat-templated, ~16 GB fp16
#   Qwen/Qwen2.5-Math-1.5B-Instruct  — lightweight, ~4 GB fp16
#   deepseek-ai/deepseek-math-7b-instruct — strong on competition math
#   meta-llama/Llama-3.1-8B-Instruct  — general purpose
#
# For a local checkpoint use the absolute path, e.g.:
#   MODEL="/data/checkpoints/my_math_model"
# -----------------------------------------------------------------------------
MODEL="Qwen/Qwen2.5-Math-7B-Instruct"

# -----------------------------------------------------------------------------
# PYTHON
# The Python binary to use.  Override if you need a specific conda env:
#   PYTHON="/home/ahsan/.conda/envs/dora/bin/python"
# -----------------------------------------------------------------------------
PYTHON="python"

# -----------------------------------------------------------------------------
# BASE_OUT
# Root directory for all result files (JSONL, summary JSON, log files).
# Sub-directories are created automatically per experiment block.
#   ./results/math500/
#   ./results/aime24/
#   ./results/scaling/
#   ./results/alpha_ablation/
#   ...
# -----------------------------------------------------------------------------
BASE_OUT="./results"

# -----------------------------------------------------------------------------
# MAX_NEW_TOKENS
# Maximum number of tokens the model can generate per response.
# Recommended values:
#   1024  — fast, enough for most MATH-500 / GSM-8K problems
#   2048  — safer for AIME / competition math with longer derivations
#   4096  — very long chain-of-thought (slow but complete)
# -----------------------------------------------------------------------------
MAX_NEW_TOKENS=2048

# -----------------------------------------------------------------------------
# TEMPERATURE
# Sampling temperature for BoN / DN-BoN.
# Greedy always uses temperature=0 (argmax), so this only affects sampling runs.
#   0.0   — deterministic (same as greedy); all N samples will be identical
#   0.6   — low diversity; good when model is already well-calibrated
#   0.7   — recommended default for math sampling
#   0.8   — more diverse; useful for harder problems (AIME)
#   1.0   — high diversity; may produce more noise
# -----------------------------------------------------------------------------
TEMPERATURE=0.7

# -----------------------------------------------------------------------------
# TOP_P
# Nucleus sampling probability mass cutoff.  Tokens are sampled from the
# smallest set whose cumulative probability ≥ TOP_P.
#   0.90  — tighter, less tail diversity
#   0.95  — recommended default
#   1.0   — no nucleus cutoff (pure temperature sampling)
# -----------------------------------------------------------------------------
TOP_P=0.95

# -----------------------------------------------------------------------------
# GREEDY_BATCH
# Number of questions decoded together in one model.generate() call for the
# Greedy method.  This does NOT affect BoN/DN-BoN (they always use batch=N).
#
# How to set:
#   - Start with GREEDY_BATCH=32 for a 7B model on a single A100 80 GB.
#   - Halve it if you see CUDA OOM during the greedy run.
#   - Increase it (up to ~64) if GPU utilisation looks low during greedy.
#   - For quantised models (4-bit/8-bit) you can typically double this.
#
# Effect: purely on throughput, not on accuracy.
# -----------------------------------------------------------------------------
GREEDY_BATCH=32

# -----------------------------------------------------------------------------
# DTYPE
# Floating-point precision for model weights and activations.
# Allowed values: auto | float16 | bfloat16 | float32
#   auto      — float16 on CUDA, float32 on CPU (recommended)
#   float16   — standard half precision; works on all NVIDIA GPUs
#   bfloat16  — better numerical range; requires Ampere or newer (A100, 3090+)
#   float32   — full precision; 2× memory, use only for debugging
# -----------------------------------------------------------------------------
DTYPE="auto"

# -----------------------------------------------------------------------------
# LOAD_4BIT / LOAD_8BIT
# Enable bitsandbytes quantisation to reduce VRAM usage.
# Set exactly ONE of these to 1; set both to 0 to disable quantisation.
#
#   LOAD_4BIT=1  — NF4 4-bit quantisation (~4–5 GB for a 7B model)
#                  Quality drop: small (~1–2 pp on MATH-500)
#   LOAD_8BIT=1  — int8 quantisation (~8 GB for a 7B model)
#                  Quality drop: negligible
#   Both 0       — full dtype precision (default)
#
# Requires:  pip install bitsandbytes
# -----------------------------------------------------------------------------
LOAD_4BIT=0
LOAD_8BIT=0

# -----------------------------------------------------------------------------
# HF_CACHE
# HuggingFace model/dataset cache directory.
# Leave empty to use the default: ~/.cache/huggingface/
# Override example:
#   HF_CACHE="/scratch/ahsan/hf_cache"
# -----------------------------------------------------------------------------
HF_CACHE=""

# -----------------------------------------------------------------------------
# VERIFIER
# Which verifier to use for scoring candidate traces in BoN / DN-BoN.
# Allowed values: outcome | length | prm
#
#   outcome  — binary correctness against the gold answer (best for eval)
#              Requires gold answers; uses evaluator_classes.check_answers_equiv
#   length   — training-free proxy: rewards shorter, boxed answers
#              Use when gold answers are unavailable at inference time
#   prm      — Process Reward Model (continuous quality score per step)
#              Requires PRM_MODEL to be set; loads a second HF model
#
# Note: for the main benchmark runs "outcome" is correct because we have gold
# answers.  Switch to "prm" or "length" only for deployment / inference-only.
# -----------------------------------------------------------------------------
VERIFIER="outcome"

# -----------------------------------------------------------------------------
# PRM_MODEL
# HuggingFace model id for the Process Reward Model verifier.
# Only used when VERIFIER=prm.
# Recommended: Qwen/Qwen2.5-Math-PRM-7B
# -----------------------------------------------------------------------------
PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B"

# -----------------------------------------------------------------------------
# EMBEDDING_MODEL
# sentence-transformers model used to embed reasoning traces for DN-BoN
# clustering.  Smaller = faster; larger = more semantically accurate clusters.
#
# Recommended choices:
#   all-MiniLM-L6-v2       — 22 MB, very fast, good quality  (default)
#   all-mpnet-base-v2      — 420 MB, slower, better quality
#   tfidf                  — no model download needed; uses sparse TF-IDF
#                            (fast but purely lexical, misses paraphrases)
# -----------------------------------------------------------------------------
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# -----------------------------------------------------------------------------
# SIMILARITY_THRESHOLD
# Cosine-similarity threshold for grouping traces into one reasoning direction.
# Two traces with similarity ≥ THRESHOLD are put in the same cluster.
#
#   0.70  — very coarse; large clusters, strong diversity penalty
#   0.80  — moderate grouping
#   0.85  — recommended default; groups near-paraphrases
#   0.90  — fine-grained; only near-duplicates are grouped
#   0.95  — very fine; almost no grouping
#
# A higher threshold → more, smaller clusters → weaker normalisation.
# A lower threshold  → fewer, larger clusters → stronger normalisation.
# -----------------------------------------------------------------------------
SIMILARITY_THRESHOLD=0.85

# -----------------------------------------------------------------------------
# CLUSTERING_METHOD
# Algorithm used to assign traces to clusters.
# Allowed values: agglomerative | greedy
#
#   agglomerative — sklearn AgglomerativeClustering with cosine distance
#                   (recommended; globally consistent clusters)
#   greedy        — fast O(N²) threshold-based; good for large N
# -----------------------------------------------------------------------------
CLUSTERING_METHOD="agglomerative"

# =============================================================================
# END CONFIG — do not edit below this line unless you know what you are doing
# =============================================================================


# =============================================================================
# Internal helpers
# =============================================================================

NOW=$(date +"%Y%m%d_%H%M%S")

# Build the flags that are shared by every single run_eval.py invocation.
common_flags() {
    local flags=""
    flags+=" --model ${MODEL}"
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
# Streams output to terminal AND to a timestamped log file.
run_exp() {
    local tag="$1"; shift
    local log_dir="${BASE_OUT}/logs"
    mkdir -p "${log_dir}"
    local log_file="${log_dir}/${tag}_${NOW}.log"

    echo ""
    echo "============================================================"
    echo " START : ${tag}"
    echo " CMD   : ${PYTHON} run_eval.py $(common_flags) $*"
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
# BLOCK 1 : MATH-500
# Full 500-problem benchmark.  Three separate processes:
#   greedy  → 1 generate() per 32 questions (GREEDY_BATCH chunks)
#   bon     → 1 generate() per question, batch of N=8 samples
#   dn_bon  → same as bon + clustering + normalisation on CPU
# =============================================================================
run_math500() {
    echo "===== MATH-500 ====="

    # ── 1a. Greedy ────────────────────────────────────────────────────────────
    # --batch_size ${GREEDY_BATCH} : chunk 500 questions into groups of 32
    # --N 8 is ignored for greedy but required by the CLI
    run_exp "math500_greedy" \
        --dataset math500 \
        --methods greedy \
        --batch_size ${GREEDY_BATCH} \
        --N 8 \
        --max_samples 500 \
        --output_dir "${BASE_OUT}/math500"

    # ── 1b. Best-of-N  (N=8) ─────────────────────────────────────────────────
    # --batch_size 8 == --N 8 : all 8 samples for one question in one call
    run_exp "math500_bon_N8" \
        --dataset math500 \
        --methods bon \
        --N 8 --batch_size 8 \
        --max_samples 500 \
        --output_dir "${BASE_OUT}/math500"

    # ── 1c. DN-BoN  (N=8, alpha=0.5) ─────────────────────────────────────────
    # --alpha 0.5 : s_i = V(x_i) / |C(x_i)|^0.5  (moderate direction penalty)
    run_exp "math500_dnbon_N8_a0.5" \
        --dataset math500 \
        --methods dn_bon \
        --N 8 --batch_size 8 \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${BASE_OUT}/math500"
}


# =============================================================================
# BLOCK 2 : AIME-24
# 30 AMC/AIME competition problems.  max_samples=32 covers the full set.
# Longer max_new_tokens (set in CONFIG) is important here.
# =============================================================================
run_aime24() {
    echo "===== AIME-24 ====="

    # ── 2a. Greedy ────────────────────────────────────────────────────────────
    run_exp "aime24_greedy" \
        --dataset aime24 \
        --methods greedy \
        --batch_size ${GREEDY_BATCH} \
        --N 8 \
        --max_samples 32 \
        --output_dir "${BASE_OUT}/aime24"

    # ── 2b. Best-of-N ─────────────────────────────────────────────────────────
    run_exp "aime24_bon_N8" \
        --dataset aime24 \
        --methods bon \
        --N 8 --batch_size 8 \
        --max_samples 32 \
        --output_dir "${BASE_OUT}/aime24"

    # ── 2c. DN-BoN ────────────────────────────────────────────────────────────
    run_exp "aime24_dnbon_N8_a0.5" \
        --dataset aime24 \
        --methods dn_bon \
        --N 8 --batch_size 8 \
        --alpha 0.5 \
        --max_samples 32 \
        --output_dir "${BASE_OUT}/aime24"
}


# =============================================================================
# BLOCK 3 : AMC-23
# Multiple-choice competition problems.  No --max_samples cap (use all).
# =============================================================================
run_amc23() {
    echo "===== AMC-23 ====="

    run_exp "amc23_greedy" \
        --dataset amc23 \
        --methods greedy \
        --batch_size ${GREEDY_BATCH} \
        --N 8 \
        --output_dir "${BASE_OUT}/amc23"

    run_exp "amc23_bon_N8" \
        --dataset amc23 \
        --methods bon \
        --N 8 --batch_size 8 \
        --output_dir "${BASE_OUT}/amc23"

    run_exp "amc23_dnbon_N8_a0.5" \
        --dataset amc23 \
        --methods dn_bon \
        --N 8 --batch_size 8 \
        --alpha 0.5 \
        --output_dir "${BASE_OUT}/amc23"
}


# =============================================================================
# BLOCK 4 : GSM-8K
# Grade-school math, 500 test problems.  Answers are integers so the
# GSM8KEvaluator handles them directly without LaTeX equivalence checks.
# =============================================================================
run_gsm8k() {
    echo "===== GSM-8K ====="

    run_exp "gsm8k_greedy" \
        --dataset gsm8k \
        --methods greedy \
        --batch_size ${GREEDY_BATCH} \
        --N 8 \
        --max_samples 500 \
        --output_dir "${BASE_OUT}/gsm8k"

    run_exp "gsm8k_bon_N8" \
        --dataset gsm8k \
        --methods bon \
        --N 8 --batch_size 8 \
        --max_samples 500 \
        --output_dir "${BASE_OUT}/gsm8k"

    run_exp "gsm8k_dnbon_N8_a0.5" \
        --dataset gsm8k \
        --methods dn_bon \
        --N 8 --batch_size 8 \
        --alpha 0.5 \
        --max_samples 500 \
        --output_dir "${BASE_OUT}/gsm8k"
}


# =============================================================================
# BLOCK 5 : Alpha ablation
# Tests DN-BoN sensitivity to the normalisation exponent alpha.
#
# Formula:  s_i = V(x_i) / |C(x_i)|^alpha
#
# alpha = 0.00 → no normalisation; identical to plain BoN (control)
# alpha = 0.25 → gentle penalty; large clusters discounted ~12% per doubling
# alpha = 0.50 → recommended default; sqrt(cluster_size) penalty
# alpha = 0.75 → strong penalty; biases toward underrepresented directions
# alpha = 1.00 → full proportional penalty; score ÷ cluster_size
#
# Run on both MATH-500 and AIME-24 so the ablation table spans difficulty.
# Each (dataset, alpha) pair is one separate python call = 10 total runs.
# =============================================================================
run_alpha() {
    echo "===== ALPHA ABLATION ====="

    for DATASET in math500 aime24; do
        # MATH-500: 500 samples gives statistical power; AIME-24: use all 32
        MAX_S=500
        [[ "${DATASET}" == "aime24" ]] && MAX_S=32

        for ALPHA in 0.0 0.25 0.5 0.75 1.0; do
            # tag encodes both dataset and alpha for easy file lookup
            run_exp "${DATASET}_dnbon_N8_a${ALPHA}" \
                --dataset "${DATASET}" \
                --methods dn_bon \
                --N 8 --batch_size 8 \
                --alpha "${ALPHA}" \
                --max_samples "${MAX_S}" \
                --output_dir "${BASE_OUT}/alpha_ablation/${DATASET}"
        done
    done
}


# =============================================================================
# BLOCK 6 : Budget scaling
# Shows how accuracy scales with the number of samples N (test-time compute).
# Runs BoN and DN-BoN side by side so improvement of DN-BoN over BoN is
# visible as a function of N.
#
# N=4   → 4 generate() calls per question per run (small budget)
# N=8   → default budget
# N=16  → 2× default budget
# N=32  → 4× default budget
#
# --batch_size == N ensures all N samples are decoded in one forward pass.
# Total runs: 2 datasets × 4 N values × 2 methods = 16 separate processes.
# =============================================================================
run_scaling() {
    echo "===== BUDGET SCALING ====="

    for DATASET in math500 aime24; do
        MAX_S=200
        [[ "${DATASET}" == "aime24" ]] && MAX_S=32

        for N in 4 8 16 32; do
            # ── BoN at this N ─────────────────────────────────────────────────
            # batch_size=N : all N samples for one question in one generate()
            run_exp "${DATASET}_bon_N${N}" \
                --dataset "${DATASET}" \
                --methods bon \
                --N "${N}" --batch_size "${N}" \
                --max_samples "${MAX_S}" \
                --output_dir "${BASE_OUT}/scaling/${DATASET}"

            # ── DN-BoN at this N  (alpha=0.5, the default) ────────────────────
            run_exp "${DATASET}_dnbon_N${N}_a0.5" \
                --dataset "${DATASET}" \
                --methods dn_bon \
                --N "${N}" --batch_size "${N}" \
                --alpha 0.5 \
                --max_samples "${MAX_S}" \
                --output_dir "${BASE_OUT}/scaling/${DATASET}"
        done
    done
}


# =============================================================================
# BLOCK 7 : Clustering similarity-threshold ablation
# Tests how sensitive DN-BoN accuracy is to the threshold used to group traces.
#
# SIMILARITY_THRESHOLD controls cosine-similarity cutoff for agglomerative
# clustering.  Traces with similarity >= threshold share a cluster.
#
# 0.70 → coarse grouping; large clusters → strong normalisation
#         Risk: semantically different traces merged into one cluster
# 0.80 → moderate
# 0.85 → default; groups near-paraphrases of the same reasoning path
# 0.90 → fine-grained; only near-duplicates are merged
# 0.95 → very fine; almost no merging, weak normalisation effect
#
# All runs use MATH-500 with 200 samples for speed.
# 5 separate python calls.
# =============================================================================
run_clustering() {
    echo "===== CLUSTERING THRESHOLD ABLATION ====="

    for THRESH in 0.70 0.80 0.85 0.90 0.95; do
        run_exp "math500_dnbon_N8_thresh${THRESH}" \
            --dataset math500 \
            --methods dn_bon \
            --N 8 --batch_size 8 \
            --alpha 0.5 \
            --similarity_threshold "${THRESH}" \
            --max_samples 200 \
            --output_dir "${BASE_OUT}/clustering_ablation"
    done
}


# =============================================================================
# BLOCK 8 : Verifier ablation
# Compares the three verifier back-ends for scoring BoN / DN-BoN candidates.
#
# outcome  — binary correctness via evaluator_classes.check_answers_equiv
#             Uses the gold answer.  Best accuracy oracle.
# length   — training-free proxy: shorter + has \boxed{} = higher score
#             No gold answer needed.  Use for real-world deployment.
# prm      — Process Reward Model (Qwen2.5-Math-PRM-7B)
#             Loads a second model; scores each reasoning step.
#             Set PRM_MODEL in CONFIG and set VERIFIER=prm to use.
#
# Runs DN-BoN on MATH-500 (200 samples) with outcome and length.
# To add prm: set VERIFIER=prm in CONFIG, then add a third run_exp call.
# 2 separate python calls.
# =============================================================================
run_verifier() {
    echo "===== VERIFIER ABLATION ====="

    # ── outcome verifier (ground-truth comparison) ────────────────────────────
    run_exp "math500_dnbon_verifier_outcome" \
        --dataset math500 \
        --methods dn_bon \
        --N 8 --batch_size 8 \
        --verifier outcome \
        --max_samples 200 \
        --output_dir "${BASE_OUT}/verifier_ablation"

    # ── length heuristic verifier (no gold needed) ────────────────────────────
    run_exp "math500_dnbon_verifier_length" \
        --dataset math500 \
        --methods dn_bon \
        --N 8 --batch_size 8 \
        --verifier length \
        --max_samples 200 \
        --output_dir "${BASE_OUT}/verifier_ablation"

    # ── PRM verifier (uncomment + set VERIFIER=prm in CONFIG to enable) ───────
    # run_exp "math500_dnbon_verifier_prm" \
    #     --dataset math500 \
    #     --methods dn_bon \
    #     --N 8 --batch_size 8 \
    #     --verifier prm \
    #     --prm_model "${PRM_MODEL}" \
    #     --max_samples 200 \
    #     --output_dir "${BASE_OUT}/verifier_ablation"
}


# =============================================================================
# Dispatcher
# =============================================================================
BLOCK="${1:-all}"

case "${BLOCK}" in
    math500)    run_math500    ;;
    aime24)     run_aime24     ;;
    amc23)      run_amc23      ;;
    gsm8k)      run_gsm8k      ;;
    alpha)      run_alpha      ;;
    scaling)    run_scaling    ;;
    clustering) run_clustering ;;
    verifier)   run_verifier   ;;
    all)
        run_math500
        run_aime24
        run_amc23
        run_gsm8k
        run_alpha
        run_scaling
        run_clustering
        run_verifier
        ;;
    *)
        echo "Unknown block: '${BLOCK}'"
        echo "Valid choices: math500 | aime24 | amc23 | gsm8k | alpha | scaling | clustering | verifier | all"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " All experiments finished."
echo " Results : ${BASE_OUT}"
echo " Logs    : ${BASE_OUT}/logs/"
echo "============================================================"