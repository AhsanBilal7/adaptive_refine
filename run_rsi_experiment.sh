#!/usr/bin/env bash
# Full Procedure-RSI experiment: develop (outer loop) → evaluate (baselines + final procedure).
#
#   bash run_rsi_experiment.sh <conf[,overlay.conf...]>
#   e.g. bash run_rsi_experiment.sh configs/rsi_qwen_7b.conf,configs/rsi_qwen_7b_aime.conf
#
# Idempotent and resumable: everything is saved continuously under
# $RSI_OUTPUT_ROOT/<model>/<RSI_RUN_NAME>/ (scratch by default). If the job is
# killed (allocation ends, OOM, ...) just run the SAME command again — finished
# phases/solves are skipped and in-flight solves restart from their last saved
# state. A lock file prevents two copies of the same experiment running at once.
set -euo pipefail
CONFS="${1:?usage: bash run_rsi_experiment.sh <conf[,conf...]>}"
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${ROOT_DIR}"

set -a
# shellcheck disable=SC1091
source .env
IFS=',' read -r -a CONF_LIST <<< "${CONFS}"
for c in "${CONF_LIST[@]}"; do [[ -n "${c}" ]] && source "${c}"; done
set +a

RUN_DIR="${RSI_OUTPUT_ROOT}/$(basename "${MODEL_NAME}")/${RSI_RUN_NAME}"
mkdir -p "${RUN_DIR}"
exec 9>"${RUN_DIR}/.lock"
if ! flock -n 9; then
    echo "another process is already running ${RUN_DIR} (lock held) — exiting"; exit 1
fi
echo "[experiment] run_dir=${RUN_DIR}  host=$(hostname)  start=$(date)" | tee -a "${RUN_DIR}/experiment.log"

bash run_procedure_rsi.sh develop "${CONFS}" 2>&1 | tee -a "${RUN_DIR}/experiment.log"
bash run_procedure_rsi.sh evaluate "${CONFS}" -- \
    --methods "${EVAL_METHODS:-h0,maj_matched,heuristic,final}" --eval_seeds "${EVAL_SEEDS:-1}" \
    2>&1 | tee -a "${RUN_DIR}/experiment.log"
echo "[experiment] finished $(date)" | tee -a "${RUN_DIR}/experiment.log"
