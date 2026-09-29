#!/usr/bin/env bash
# Adaptive Delta Refinement launcher.
#
#   bash run_adr.sh <sanity|pilot|run> [conf[,conf...]] [-- extra python args]
#   bash run_adr.sh run   configs/adr_qwen_7b.conf,configs/adr_aime24.conf
#   bash run_adr.sh pilot configs/adr_qwen_7b.conf
#
# Sources .env (paths + predefined variables) then the conf file(s), activates
# $VENV_PATH and runs run_adr.py.  Results go to $ADR_OUTPUT_ROOT (scratch) and
# are saved continuously: re-running the same command resumes where it stopped.
# A lock prevents two copies of the same (model, datasets) experiment at once.
set -euo pipefail
CMD="${1:?usage: bash run_adr.sh <sanity|pilot|run> [conf[,conf...]] [-- args]}"
shift
CONFS=""
if [[ $# -gt 0 && "$1" != "--" ]]; then CONFS="$1"; shift; fi
[[ $# -gt 0 && "$1" == "--" ]] && shift

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${ROOT_DIR}"
set -a
# shellcheck disable=SC1091
source "${ROOT_DIR}/.env"
IFS=',' read -r -a CONF_LIST <<< "${CONFS}"
for c in "${CONF_LIST[@]}"; do [[ -n "${c}" ]] && source "${c}"; done
set +a
mkdir -p "${LOG_DIR}" "${TMPDIR}" "${HF_HOME}" "${ADR_OUTPUT_ROOT}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
fi

MODEL_DIR="${ADR_OUTPUT_ROOT}/$(basename "${MODEL_NAME}")"
mkdir -p "${MODEL_DIR}"
exec 9>"${MODEL_DIR}/.lock_${CMD}_${ADR_DATASETS//,/_}"
flock -n 9 || { echo "an identical experiment is already running (lock held) — exiting"; exit 1; }

LOG="${LOG_DIR}/adr_${CMD}_$(basename "${MODEL_NAME}")_${ADR_DATASETS//,/_}_$(date +%Y%m%d_%H%M%S).log"
echo "[run_adr] cmd=${CMD} model=${MODEL_NAME} datasets=${ADR_DATASETS} budgets=${ADR_BUDGETS} out=${MODEL_DIR} log=${LOG}"
python run_adr.py "${CMD}" "$@" 2>&1 | tee "${LOG}" | tee -a "${MODEL_DIR}/experiment.log"
exit "${PIPESTATUS[0]}"
