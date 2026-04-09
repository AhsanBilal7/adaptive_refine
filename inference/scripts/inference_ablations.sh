#!/bin/bash
# inference/scripts/inference_ablations.sh
# Runs all AdaptSci ablation conditions from Section 6 of the paper,
# one after the other, each with its own timestamped log file.
# Follows the same env/path/logging pattern as the main inference scripts.

set -x

# ── Conda ──────────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PARQUET_FILE_PATH="$(cd "$SCRIPT_DIR/../../data/inference" && pwd)/llmsrbench.parquet"
MODEL_PATH="$(cd "$SCRIPT_DIR/../../models/gpt-oss-20b" && pwd)"

OUTPUT_BASE="$(cd "$SCRIPT_DIR/.." && pwd)/output/ablations"
mkdir -p "$OUTPUT_BASE"

# ── Server URLs ────────────────────────────────────────────────────────────
MODEL_URL="http://0.0.0.0:30000/v1"
SANDBOX_URLS=(
    "http://127.0.0.1:9010/run_code"
    "http://127.0.0.1:9020/run_code"
    "http://127.0.0.1:9030/run_code"
    "http://127.0.0.1:9040/run_code"
    "http://127.0.0.1:9050/run_code"
    "http://127.0.0.1:9060/run_code"
    "http://127.0.0.1:9070/run_code"
    "http://127.0.0.1:9080/run_code"
)

# ── Shared hyperparameters ─────────────────────────────────────────────────
MAPE_THRESHOLD=0.001
NUM_TURNS=40
MAX_ASSISTANT_TURNS=25
TOP_K=3

SOURCE=(
    "lsr_synth/matsci"
    "lsr_synth/chem_react"
    "lsr_synth/bio_pop_growth"
    "lsr_synth/phys_osc"
)

# ── Activate env and cd to inference/ ─────────────────────────────────────
conda activate srscientist
cd "$SCRIPT_DIR/.."

# ── Helper: run one ablation condition ─────────────────────────────────────
run_ablation() {
    local ABLATION_NAME="$1"
    shift
    local EXTRA_ARGS=("$@")

    OUTPUT_JSON_PATH="${OUTPUT_BASE}/${ABLATION_NAME}.json"
    LOG_DIR="${OUTPUT_BASE/output/log}"
    mkdir -p "$LOG_DIR"
    mkdir -p "$OUTPUT_BASE"

    TIMESTAMP=$(TZ='UTC-8' date +'%Y%m%d_%H%M%S')
    LOG_FILE="${LOG_DIR}/${ABLATION_NAME}_${TIMESTAMP}.log"

    echo "" | tee -a "$LOG_FILE"
    echo "======================================================" | tee -a "$LOG_FILE"
    echo "  Ablation: ${ABLATION_NAME}" | tee -a "$LOG_FILE"
    echo "  Started:  $(TZ='UTC-8' date)" | tee -a "$LOG_FILE"
    echo "======================================================" | tee -a "$LOG_FILE"

    python main.py \
        --model-name "$MODEL_PATH" \
        --model-url "$MODEL_URL" \
        --sandbox-urls "${SANDBOX_URLS[@]}" \
        --parquet-file-path "$PARQUET_FILE_PATH" \
        --mape-threshold $MAPE_THRESHOLD \
        --num-turns $NUM_TURNS \
        --max-assistant-turns $MAX_ASSISTANT_TURNS \
        --top-k $TOP_K \
        --source "${SOURCE[@]}" \
        --output-json-path "$OUTPUT_JSON_PATH" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"

    echo "--- Finished ${ABLATION_NAME} at $(TZ='UTC-8' date) ---" | tee -a "$LOG_FILE"
}

# ══════════════════════════════════════════════════════════════════════════
# Ablation conditions  (Section 6 of AdaptSci paper)
# ══════════════════════════════════════════════════════════════════════════

# Full AdaptSci (all mechanisms on) — paper main result
run_ablation "full_adaptsci" \
    --adaptive-compute \
    --tau1 100 --tau2 1000 \
    --epsilon-target 1e-4 --beta-star 5000 \
    --exploit-frac 0.70 --frontier-alpha 0.5 --frontier-window 3

# SR-Scientist baseline (no adaptive compute)
run_ablation "sr_scientist_baseline" \
    --no-adaptive-compute

# Ablation (i): remove early stopping
run_ablation "ablation_i_no_early_stopping" \
    --adaptive-compute \
    --ablation-disable-early-stopping

# Ablation (ii): fix n_restarts=1 throughout
run_ablation "ablation_ii_fix_n_restarts" \
    --adaptive-compute \
    --ablation-fix-n-restarts

# Ablation (iii): α sweep
run_ablation "ablation_iii_alpha_0" \
    --adaptive-compute --ablation-alpha-override 0

run_ablation "ablation_iii_alpha_0.25" \
    --adaptive-compute --ablation-alpha-override 0.25

run_ablation "ablation_iii_alpha_0.5" \
    --adaptive-compute --ablation-alpha-override 0.5

run_ablation "ablation_iii_alpha_0.75" \
    --adaptive-compute --ablation-alpha-override 0.75

run_ablation "ablation_iii_alpha_1" \
    --adaptive-compute --ablation-alpha-override 1

# Ablation (iv-a): C-diversity term off (equivalent to α=1)
run_ablation "ablation_iv_no_c_diversity" \
    --adaptive-compute \
    --ablation-disable-c-diversity

# Ablation (iv-b): random frontier sampling instead of C-distance
run_ablation "ablation_iv_random_frontier" \
    --adaptive-compute \
    --ablation-random-frontier

# Ablation (v): Hessian computed at first BFGS run, not best-of-n
run_ablation "ablation_v_hessian_first_run" \
    --adaptive-compute \
    --ablation-hessian-at-first-run

echo ""
echo "All ablations finished. Results in: ${OUTPUT_BASE}"
echo "Logs in:    ${OUTPUT_BASE/output/log}"
