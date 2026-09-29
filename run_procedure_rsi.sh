#!/usr/bin/env bash
# Procedure RSI launcher.
#
#   bash run_procedure_rsi.sh <sanity|splits|develop|evaluate> [conf[,overlay.conf...]] [-- python args]
#
# Sources .env (paths + predefined variables), then the conf file(s) (overrides),
# activates $VENV_PATH and runs run_procedure_rsi.py.  Extra python args after
# `--` have the highest precedence, e.g.
#   bash run_procedure_rsi.sh develop configs/rsi_qwen_1b.conf
#   bash run_procedure_rsi.sh develop configs/rsi_qwen_1b.conf,configs/ablations/primitive_only.conf
#   bash run_procedure_rsi.sh evaluate configs/rsi_qwen_1b.conf -- --methods h0,maj_matched,heuristic,final
#   bash run_procedure_rsi.sh evaluate configs/rsi_qwen_1b.conf -- --methods h0,final --sealed
set -euo pipefail

CMD="${1:?usage: bash run_procedure_rsi.sh <sanity|splits|develop|evaluate> [conf[,conf...]] [-- args]}"
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
for c in "${CONF_LIST[@]}"; do
    [[ -z "${c}" ]] && continue
    # shellcheck disable=SC1090
    source "${c}"
done
set +a

mkdir -p "${LOG_DIR}" "${TMPDIR}" "${HF_HOME}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
else
    echo "WARNING: ${VENV_PATH}/bin/activate not found — using $(command -v python)" >&2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/rsi_${CMD}_$(basename "${MODEL_NAME}")_${RSI_RUN_NAME:-default}_${STAMP}.log"
echo "[run_procedure_rsi] cmd=${CMD} model=${MODEL_NAME} confs=${CONFS:-none} venv=${VENV_PATH} log=${LOG}"

python run_procedure_rsi.py "${CMD}" "$@" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
