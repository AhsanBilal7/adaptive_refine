#!/bin/bash
# inference/scripts/inference_adaptsci.sh
# Example inference script for AdaptSci with full adaptive compute.
# Matches the paper's main experimental setting:
#   Backbone: Qwen3-Coder-480B, Budget: 1000 calls, Dataset: LSR-Synth.

set -e

MODEL_NAME="Qwen3-Coder-480B-A35B-Instruct"
MODEL_URL="http://localhost:8000/v1"
SANDBOX_URL="http://localhost:8080/run_code"
PARQUET="./data/inference/lsr_synth.parquet"
OUTPUT="./results/adaptsci_qwen3_480b.json"
MAPE_THRESHOLD=0.001
NUM_TURNS=40
TOP_K=3
MAX_TURNS=25

# ---- AdaptSci hyperparameters ----
TAU1=100        # κ < 100    → 1 restart
TAU2=1000       # κ >= 1000  → 5 restarts
EPS_TARGET=1e-4 # NMSE convergence target
BETA_STAR=5000  # efficiency ratio threshold
EXPLOIT_FRAC=0.70
FRONTIER_ALPHA=0.5
FRONTIER_WINDOW=3

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
    --adaptive-compute \
    --tau1 ${TAU1} \
    --tau2 ${TAU2} \
    --epsilon-target ${EPS_TARGET} \
    --beta-star ${BETA_STAR} \
    --exploit-frac ${EXPLOIT_FRAC} \
    --frontier-alpha ${FRONTIER_ALPHA} \
    --frontier-window ${FRONTIER_WINDOW}

echo "Done. Results at: ${OUTPUT}"
