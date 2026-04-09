#!/bin/bash
# inference/scripts/inference_sr_scientist_baseline.sh
# Pure SR-Scientist mode — no AdaptSci adaptive compute.
# Use this to reproduce the SR-Scientist baseline numbers from the AdaptSci paper.

set -x

# ── Conda ──────────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PARQUET_FILE_PATH="$(cd "$SCRIPT_DIR/../../data/inference" && pwd)/llmsrbench.parquet"

OUTPUT_JSON_PATH="$(cd "$SCRIPT_DIR/.." && pwd)/output/sr_scientist_baseline/gpt-oss-20b_0_001_s40_max_25_top3_baseline.json"
mkdir -p "$(dirname "$OUTPUT_JSON_PATH")"

MODEL_PATH="$(cd "$SCRIPT_DIR/../../models/gpt-oss-20b" && pwd)"

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

# ── Inference hyperparameters ──────────────────────────────────────────────
MAPE_THRESHOLD=0.001
NUM_TURNS=25
MAX_ASSISTANT_TURNS=25
TOP_K=3

SOURCE=(
    # "lsr_synth/matsci"
    # "lsr_synth/chem_react"
    "lsr_synth/bio_pop_growth"
    # "lsr_synth/phys_osc"
)

# ── Activate env and cd to inference/ ─────────────────────────────────────
conda activate srscientist
cd "$SCRIPT_DIR/.."   # cd to inference/ where main.py lives

# ── Logging ───────────────────────────────────────────────────────────────
TIMESTAMP=$(TZ='UTC-8' date +'%Y%m%d_%H%M%S')
BASENAME=$(basename "$OUTPUT_JSON_PATH" .json)
OUTPUT_DIR=$(dirname "$OUTPUT_JSON_PATH")
LOG_DIR="${OUTPUT_DIR/output/log}"
LOG_FILE="${LOG_DIR}/${BASENAME}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "--- Starting SR-Scientist baseline inference at $(TZ='UTC-8' date) ---" | tee -a "$LOG_FILE"

# ── Run (--no-adaptive-compute disables all AdaptSci mechanisms) ───────────
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
    --no-adaptive-compute \
    2>&1 | tee "$LOG_FILE"

echo "--- Finished SR-Scientist baseline inference at $(TZ='UTC-8' date) ---" | tee -a "$LOG_FILE"