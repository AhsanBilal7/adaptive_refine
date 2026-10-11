#!/usr/bin/env bash
# Launcher of the ARR experiments.
#
#   bash run_arr.sh [sanity|pilot|run] [conf[,conf...]] [-- extra python args]
#   bash run_arr.sh run   configs/arr_qwen_math_7b.conf,configs/arr_aime24.conf
#   bash run_arr.sh pilot configs/arr_qwen_math_7b.conf,configs/arr_pilot.conf
#   (no conf → $ARR_CONFS from .env)
#
# Sources .env (paths + predefined variables) then the conf file(s), activates
# $VENV_PATH and runs run_arr.py.  Results go to $ARR_OUTPUT_ROOT and
# are appended per finished solve: re-running the same command resumes.
# A lock prevents two copies of the same (command, model, datasets) at once.
set -euo pipefail
CMD="${1:-run}"
[[ $# -gt 0 ]] && shift
CONFS=""
if [[ $# -gt 0 && "$1" != "--" ]]; then CONFS="$1"; shift; fi
[[ $# -gt 0 && "$1" == "--" ]] && shift

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${ROOT_DIR}/run_arr.py" ]] || ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"
set -a
# shellcheck disable=SC1091
source "${ROOT_DIR}/.env"
CONFS="${CONFS:-${ARR_CONFS:-}}"
IFS=',' read -r -a CONF_LIST <<< "${CONFS}"
for c in "${CONF_LIST[@]}"; do [[ -n "${c}" ]] && source "${c}"; done
set +a
mkdir -p "${LOG_DIR}" "${TMPDIR}" "${HF_HOME}" "${ARR_OUTPUT_ROOT}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
fi

DATASETS_SHOWN="${ARR_DATASETS}"
[[ "${CMD}" == "pilot" ]] && DATASETS_SHOWN="${ARR_PILOT_DATASET}"
TAG="${CMD}_$(basename "${MODEL_NAME}")_${DATASETS_SHOWN//,/_}"
[[ -n "${ARR_SHARD:-}" ]] && TAG="${TAG}_shard${ARR_SHARD//\//of}"
[[ -n "${ARR_SEED_IDS:-}" ]] && TAG="${TAG}_seed${ARR_SEED_IDS//,/_}"
MODEL_DIR="${ARR_OUTPUT_ROOT}/$(basename "${MODEL_NAME}")"
mkdir -p "${MODEL_DIR}"
exec 9>"${MODEL_DIR}/.lock_${TAG}"
flock -n 9 || { echo "an identical experiment is already running (lock held) — exiting"; exit 1; }

LOG="${LOG_DIR}/arr_${TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "[run_arr] host=$(hostname) cmd=${CMD} model=${MODEL_NAME} datasets=${DATASETS_SHOWN} budgets=${ARR_BUDGETS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} out=${MODEL_DIR} log=${LOG}"
python run_arr.py "${CMD}" "$@" 2>&1 | tee "${LOG}" | tee -a "${MODEL_DIR}/experiment.log"
exit "${PIPESTATUS[0]}"
