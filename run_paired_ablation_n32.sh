#!/usr/bin/env bash
# =============================================================================
# run_paired_ablation.sh
# =============================================================================
# Runs the paired ablation harness for every dataset used in the DN-BoN paper.
#
# DESIGN PRINCIPLE
# ─────────────────
# For each (model, dataset, N) triple:
#   1. Generate N traces once  → saved in sample_cache_<tag>.jsonl
#   2. Score with PRM once     → stored in the same cache file
#   3. Apply BoN / Majority / Oracle / DN-BoN(α, τ) on the SAME samples
#
# Subsequent reruns skip generation automatically (cache is detected).
#
# USAGE
# ─────
#   bash run_paired_ablation.sh                   # all datasets
#   bash run_paired_ablation.sh aime24            # one dataset
#   bash run_paired_ablation.sh math500 aime24    # two datasets
#   LOAD_CACHE=1 bash run_paired_ablation.sh      # force cache load
#
# OUTPUTS (per dataset)
# ─────────────────────
#   $OUT/sample_cache_<tag>.jsonl          raw traces + PRM scores
#   $OUT/results_paired_<tag>.jsonl        per-question per-config
#   $OUT/summary_paired_<tag>.json         accuracy table all configs
#   $OUT/analysis_paired_<tag>.json        help/hurt meta per question
#   $OUT/latex_table_<tag>.txt             copy-paste NeurIPS table
#   $OUT/help_cases_<tag>.jsonl            DN-BoN helps vs BoN
#   $OUT/hurt_cases_<tag>.jsonl            DN-BoN hurts vs BoN
#   $OUT/figures/                          all plots (pdf + png)
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Generator LLM
MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"

# PRM verifier (must be Qwen2.5-Math-PRM-7B or compatible)
VERIFIER_MODEL="Qwen/Qwen2.5-Math-PRM-7B"

# Sentence encoder for clustering (cached after first run)
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Generation settings (match the paper)
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.95

# Sample budget — default run
N_DEFAULT=32

# Ablation sweeps  (these are applied to EVERY dataset)
ALPHA_VALUES="0.0 0.25 0.5 0.75 1.0"
TAU_VALUES="0.70 0.75 0.80 0.85 0.90 0.95"

# Batch sizes — set to N for BoN (all N prompts in one generate() call)
BATCH_SIZE=32       # matches N_DEFAULT; adjust per VRAM

# CIMR (Cluster-Informed Meta-Reasoning) — answer-clustered, per-cluster judge
# ─────────────────────────────────────────────────────────────────────────────
# Algorithm (revised):
#   1. Group the N traces by their EXTRACTED FINAL ANSWER → K answer-clusters.
#   2. For each cluster c, run the judge on (cluster c's traces +
#      answers from the OTHER clusters)   →  K judge responses per question.
#   3. Score each judge response with the PRM verifier.
#   4. Two final selections per question (both reported in the summary):
#        cimr_<variant>     : answer of the judge-response with highest PRM
#        cimr_<variant>_mv  : majority vote of the K judge-extracted answers
#                              (ties broken by sum-of-PRM)
#
# CIMR (Cluster-Informed Meta-Reasoning) — TRACE-LEVEL pipeline
# ─────────────────────────────────────────────────────────────────────────────
# For each question:
#   1. Cluster the N original traces by extracted answer  → K clusters.
#   2. For each cluster take the top-K_top traces by PRM   → flat list.
#   3. Judge each TRACE with the strict two-line verdict:
#         VERIFY: YES/NO
#         VALID  /  INVALID: <reason>
#   4. PRM-score the judge responses → S_judge,t per trace.
#   5. Select an answer with 8 trace-level rules (majority, judge_prm,
#      sum_o_plus_j, mean_o_plus_j, mean_sum_oj, sum_x_pvalid,
#      mean_x_pvalid, ensemble) — all use the strict-validity gate.
#   6. (Optional) Run a 9th rule `resolve_from_valid`: feed every
#      strict-valid trace into the judge and ask for a consolidated
#      \boxed{} answer.
#
# Variant semantics (trace-level prompt content):
#   full        : full trace text shown to the judge
#   concise     : trace text head/tail truncated to ~1500 chars
#   answer_only : no trace text — only the candidate answer
ENABLE_CIMR=1
CIMR_VARIANTS="full concise answer_only"
CIMR_MAX_JUDGE_TOKENS=256       # 2-line verdict only; 128–256 is plenty
CIMR_JUDGE_BATCH_SIZE=32        # trace-level prompts are short; 16–32 fits
CIMR_TOP_K_TRACES=4             # top-PRM traces per cluster sent to judge
# CIMR_DISABLE_RESOLVE_FROM_VALID=1   # uncomment to skip the generative pass
CIMR_RESOLVE_MAX_TOKENS=4096    # max tokens for resolve_from_valid output
# CIMR_DISABLE_CROSS_EXAMINE=1       # uncomment to skip cross-examination
CIMR_XC_MAX_TOKENS=4096         # max tokens for cross-examination output

# Output root
BASE_OUT="./experiments_results/results_paired"

# GPU(s) to use (comma-separated indices, or leave blank for auto)
GPUS="1"           # e.g. "0" or "0,1"

# Python executable
PYTHON="python"

# Limits (set to "" for full dataset)
MAX_SAMPLES_AIME="30"
MAX_SAMPLES_AMC="40"
MAX_SAMPLES_MATH="500"
MAX_SAMPLES_OLYMPIAD="80"
MAX_SAMPLES_GSM=""     # full GSM-8K

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
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_variants ${CIMR_VARIANTS}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_max_judge_tokens ${CIMR_MAX_JUDGE_TOKENS}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_judge_batch_size ${CIMR_JUDGE_BATCH_SIZE}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_top_k_traces ${CIMR_TOP_K_TRACES}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_resolve_max_tokens ${CIMR_RESOLVE_MAX_TOKENS}"
    CIMR_FLAGS="${CIMR_FLAGS} --cimr_xc_max_tokens ${CIMR_XC_MAX_TOKENS}"
    if [[ "${CIMR_DISABLE_RESOLVE_FROM_VALID:-0}" == "1" ]]; then
        CIMR_FLAGS="${CIMR_FLAGS} --cimr_disable_resolve_from_valid"
    fi
    if [[ "${CIMR_DISABLE_CROSS_EXAMINE:-0}" == "1" ]]; then
        CIMR_FLAGS="${CIMR_FLAGS} --cimr_disable_cross_examine"
    fi
fi

COMMON_ARGS=(
    --model          "${MODEL}"
    --verifier_model "${VERIFIER_MODEL}"
    --embedding_model "${EMBEDDING_MODEL}"
    --max_new_tokens  "${MAX_NEW_TOKENS}"
    --temperature     "${TEMPERATURE}"
    --top_p           "${TOP_P}"
    --batch_size      "${BATCH_SIZE}"
    --output_dir      "${OUT}"
    --alpha_values    ${ALPHA_VALUES}
    --tau_values      ${TAU_VALUES}
    --log_every       5
    ${GPU_FLAG}
    ${CIMR_FLAGS}
)

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset run functions
# ─────────────────────────────────────────────────────────────────────────────

run_dataset() {
    local DATASET="$1"
    local N="$2"
    local MAX_SAMPLES="$3"
    local LOG="${OUT}/logs/paired_${DATASET}_N${N}.log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Dataset : ${DATASET}   N=${N}   max_samples=${MAX_SAMPLES:-all}"
    echo "  Output  : ${OUT}"
    echo "════════════════════════════════════════════════════════════════"

    # Build max_samples flag
    MAX_FLAG=()
    if [[ -n "${MAX_SAMPLES}" ]]; then
        MAX_FLAG=(--max_samples "${MAX_SAMPLES}")
    fi

    # If a sample cache already exists, load it (skip generation)
    CACHE_FILE="${OUT}/sample_cache_${DATASET}_N${N}.jsonl"
    CACHE_FLAG=()
    if [[ -f "${CACHE_FILE}" ]]; then
        echo "  ↺ Cache found — skipping generation: ${CACHE_FILE}"
        CACHE_FLAG=(--load_cache "${CACHE_FILE}")
    fi

    ${PYTHON} run_paired_ablation.py \
        "${COMMON_ARGS[@]}" \
        --dataset     "${DATASET}" \
        --N           "${N}" \
        "${MAX_FLAG[@]}" \
        "${CACHE_FLAG[@]}" \
        2>&1 | tee "${LOG}"

    echo "  ✓  Done. Log: ${LOG}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Budget-scaling variant: run the same dataset at N ∈ {4, 8, 16, 32}
# ─────────────────────────────────────────────────────────────────────────────

run_scaling() {
    local DATASET="$1"
    local MAX_SAMPLES="$2"

    for N in 4 8 16 32; do
        # Adjust batch_size = N for single generate() call
        local LOG="${OUT}/logs/paired_${DATASET}_N${N}_scaling.log"

        CACHE_FILE="${OUT}/sample_cache_${DATASET}_N${N}.jsonl"
        CACHE_FLAG=()
        if [[ -f "${CACHE_FILE}" ]]; then
            CACHE_FLAG=(--load_cache "${CACHE_FILE}")
        fi

        MAX_FLAG=()
        if [[ -n "${MAX_SAMPLES}" ]]; then
            MAX_FLAG=(--max_samples "${MAX_SAMPLES}")
        fi

        echo ""
        echo "  ── Scaling: ${DATASET}  N=${N} ──────────────────────────"

        ${PYTHON} run_paired_ablation.py \
            "${COMMON_ARGS[@]}" \
            --batch_size  "${N}" \
            --dataset     "${DATASET}" \
            --N           "${N}" \
            "${MAX_FLAG[@]}" \
            "${CACHE_FLAG[@]}" \
            2>&1 | tee "${LOG}"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Main dispatch
# ─────────────────────────────────────────────────────────────────────────────

DATASETS=("${@}")   # datasets passed as positional arguments

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    # Default: run all paper datasets
    DATASETS=(aime24 aime25 amc23 math500 olympiadbench)
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DN-BoN Paired Ablation"
echo "  Model   : ${MODEL}"
echo "  Verifier: ${VERIFIER_MODEL}"
echo "  N       : ${N_DEFAULT}"
echo "  α sweep : ${ALPHA_VALUES}"
echo "  τ sweep : ${TAU_VALUES}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Out dir : ${OUT}"
echo "════════════════════════════════════════════════════════════════"

for DATASET in "${DATASETS[@]}"; do
    case "${DATASET}" in

        aime24)
            run_dataset aime24 "${N_DEFAULT}" "${MAX_SAMPLES_AIME}"
            ;;

        aime25)
            run_dataset aime25 "${N_DEFAULT}" "${MAX_SAMPLES_AIME}"
            ;;

        amc23)
            run_dataset amc23 "${N_DEFAULT}" "${MAX_SAMPLES_AMC}"
            ;;

        math500)
            run_dataset math500 "${N_DEFAULT}" "${MAX_SAMPLES_MATH}"
            ;;

        olympiadbench)
            run_dataset olympiadbench "${N_DEFAULT}" "${MAX_SAMPLES_OLYMPIAD}"
            ;;

        gsm8k)
            run_dataset gsm8k "${N_DEFAULT}" "${MAX_SAMPLES_GSM}"
            ;;

        # ── Scaling sweep (generates N=4,8,16,32 for one dataset) ───────────
        scaling_aime24)
            run_scaling aime24 "${MAX_SAMPLES_AIME}"
            ;;

        scaling_math500)
            run_scaling math500 "${MAX_SAMPLES_MATH}"
            ;;

        scaling_amc23)
            run_scaling amc23 "${MAX_SAMPLES_AMC}"
            ;;

        scaling_olympiad)
            run_scaling olympiadbench "${MAX_SAMPLES_OLYMPIAD}"
            ;;

        scaling_all)
            run_scaling aime24        "${MAX_SAMPLES_AIME}"
            run_scaling aime25        "${MAX_SAMPLES_AIME}"
            run_scaling amc23         "${MAX_SAMPLES_AMC}"
            run_scaling math500       "${MAX_SAMPLES_MATH}"
            run_scaling olympiadbench "${MAX_SAMPLES_OLYMPIAD}"
            ;;

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
            echo "Available datasets:"
            echo "  aime24  aime25  amc23  math500  olympiadbench  gsm8k"
            echo ""
            echo "Scaling sweeps (N ∈ {4,8,16,32}):"
            echo "  scaling_aime24  scaling_math500  scaling_amc23"
            echo "  scaling_olympiad  scaling_all"
            echo ""
            echo "Run everything:"
            echo "  bash run_paired_ablation.sh all"
            exit 1
            ;;
    esac
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All paired ablation experiments finished."
echo "  Results : ${OUT}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Auto-generate a consolidated summary table across all completed datasets
# ─────────────────────────────────────────────────────────────────────────────

echo "  Building consolidated summary across all completed datasets..."

${PYTHON} - <<'PYEOF' 2>&1 || echo "  [warn] consolidated summary failed — run manually"
import json, os, glob, sys

out_dir = os.environ.get("OUT_DIR", "./experiments_results/results_paired")
# Find any OUT dir from this run
import glob
dirs = glob.glob("./experiments_results/results_paired/*") + ["./experiments_results/results_paired/"]
out_dir_candidates = [d for d in dirs if os.path.isdir(d)]

summaries = {}
for d in out_dir_candidates:
    for p in glob.glob(os.path.join(d, "summary_paired_*.json")):
        tag = os.path.basename(p).replace("summary_paired_", "").replace(".json", "")
        with open(p) as f:
            summaries[tag] = json.load(f)

if not summaries:
    print("  No summary files found.")
    sys.exit(0)

print("\n  ══ Consolidated Accuracy Table ═══════════════════════════")
print(f"  {'Config':<40s}  {'Accuracy':>10s}  {'Δ vs BoN':>10s}")
print("  " + "─" * 65)
for tag, s in sorted(summaries.items()):
    bon_acc = s.get("bon", {}).get("accuracy", None)
    best_dn = None
    best_acc = -1
    for k, v in s.items():
        if k.startswith("dn_bon") and v.get("accuracy", -1) > best_acc:
            best_acc = v["accuracy"]
            best_dn = k
    if bon_acc is not None:
        print(f"  {tag:<40s}  BoN={bon_acc:.1f}%")
    if best_dn:
        delta = best_acc - (bon_acc or 0)
        print(f"  {'  Best DN-BoN: ' + best_dn:<40s}  {best_acc:.1f}%  {delta:+.1f}pp")
    print()
PYEOF

echo "  Done."