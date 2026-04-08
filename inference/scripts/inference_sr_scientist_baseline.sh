#!/bin/bash
# inference/scripts/inference_sr_scientist_baseline.sh
# Run in pure SR-Scientist mode (no AdaptSci adaptive compute).
# Use this to reproduce the SR-Scientist baseline numbers.

set -e

MODEL_NAME="Qwen3-Coder-480B-A35B-Instruct"
MODEL_URL="http://localhost:8000/v1"
SANDBOX_URL="http://localhost:8080/run_code"
PARQUET="./data/inference/lsr_synth.parquet"
OUTPUT="./results/sr_scientist_baseline_qwen3_480b.json"
MAPE_THRESHOLD=0.001
NUM_TURNS=40
TOP_K=3
MAX_TURNS=25

python main.py \
    --model-name "${MODEL_NAME}" \
    --model-url "${MODEL_URL}" \
    --sandbox-urls "${SANDBOX_URL}" \
    --parquet-file-path "${PARQUET}" \
    --mape-threshold ${MAPE_THRESHOLD} \
    --num-turns ${NUM_TURNS} \
    --top-k ${TOP_K} \
    --max-assistant-turns ${MAX_TURNS} \
    --output-json-path "${OUTPUT}" \
    --no-adaptive-compute

echo "Done. Results at: ${OUTPUT}"
