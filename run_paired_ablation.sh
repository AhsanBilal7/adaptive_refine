#!/usr/bin/env bash
set -euo pipefail

# Paths and predefined variables come from .env (sourced below); every value
# here is only a fallback and any exported / conf value wins.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -f "${ROOT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.env"
    set +a
fi

VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen2.5-Math-PRM-7B}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_DEFAULT="${N_DEFAULT:-8}"
ALPHA_VALUES="${ALPHA_VALUES:-0.0 0.25 0.5 0.75 1.0}"
TAU_VALUES="${TAU_VALUES:-0.70 0.75 0.80 0.85 0.90 0.95}"

SKIP_BASELINES="${SKIP_BASELINES:-1}"
CIMR_N_ROLLOUTS="${CIMR_N_ROLLOUTS:-8}"
CIMR_DEPTH="${CIMR_DEPTH:-4}"
CIMR_REASONING_MAX_NEW_TOKENS="${CIMR_REASONING_MAX_NEW_TOKENS:-2048}"
CIMR_REASONING_TEMPERATURE="${CIMR_REASONING_TEMPERATURE:-0.7}"
CIMR_REASONING_TOP_P="${CIMR_REASONING_TOP_P:-0.95}"
CIMR_CRITIQUE_MAX_TOKENS="${CIMR_CRITIQUE_MAX_TOKENS:-384}"
CIMR_CRITIQUE_MAX_TRACE_CHARS="${CIMR_CRITIQUE_MAX_TRACE_CHARS:-4000}"
CIMR_CORRECTION_MAX_TOKENS="${CIMR_CORRECTION_MAX_TOKENS:-1024}"
CIMR_CORRECTION_MAX_TRACE_CHARS="${CIMR_CORRECTION_MAX_TRACE_CHARS:-4000}"
CIMR_CORRECTION_MAX_CRITIQUE_CHARS="${CIMR_CORRECTION_MAX_CRITIQUE_CHARS:-1500}"
CIMR_CONTINUATION_MAX_PRIOR_CHARS="${CIMR_CONTINUATION_MAX_PRIOR_CHARS:-4000}"
CIMR_OTHER_AWARE="${CIMR_OTHER_AWARE:-1}"

NO_CRITIQUE="${NO_CRITIQUE:-0}"
NO_CORRECTION="${NO_CORRECTION:-0}"

MAX_SAMPLES_AIME="${MAX_SAMPLES_AIME:-30}"
MAX_SAMPLES_AMC="${MAX_SAMPLES_AMC:-40}"
MAX_SAMPLES_MATH="${MAX_SAMPLES_MATH:-500}"
MAX_SAMPLES_OLYMPIAD="${MAX_SAMPLES_OLYMPIAD:-}"
MAX_SAMPLES_GSM="${MAX_SAMPLES_GSM:-}"

BASE_OUT="${BASE_OUT:-${ROOT_DIR}/experiments_results/results_paired}"
GPUS="${GPUS:-}"
PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
DATASET_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name|--model)     MODEL_NAME="$2";       shift 2 ;;
        --model_name=*|--model=*) MODEL_NAME="${1#*=}";  shift   ;;
        --cimr_n_rollouts)        CIMR_N_ROLLOUTS="$2";  shift 2 ;;
        --cimr_n_rollouts=*)      CIMR_N_ROLLOUTS="${1#*=}"; shift ;;
        --cimr_depth)             CIMR_DEPTH="$2";        shift 2 ;;
        --cimr_depth=*)           CIMR_DEPTH="${1#*=}";   shift   ;;
        --no_critique)            NO_CRITIQUE=1;           shift   ;;
        --no_correction)          NO_CORRECTION=1;         shift   ;;
        --help|-h)
            echo "Usage: bash run_paired_ablation.sh [--model_name M] [--cimr_n_rollouts N] [--cimr_depth D] [--no_critique] [--no_correction] [dataset ...]"
            exit 0 ;;
        *) DATASET_ARGS+=("$1"); shift ;;
    esac
done

if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
fi
cd "${ROOT_DIR}"

MODEL_SHORT=$(basename "${MODEL_NAME}")
OUT="${BASE_OUT}/${MODEL_SHORT}_mnt${MAX_NEW_TOKENS}_t${TEMPERATURE}_p${TOP_P}"
mkdir -p "${OUT}/figures" "${OUT}/logs"

GPU_FLAG=""
[[ -n "${GPUS}" ]] && GPU_FLAG="--gpus ${GPUS}"

CIMR_FLAGS="--cimr_n_rollouts              ${CIMR_N_ROLLOUTS}"
CIMR_FLAGS="${CIMR_FLAGS} --cimr_depth                   ${CIMR_DEPTH}"
CIMR_FLAGS="${CIMR_FLAGS} --cimr_reasoning_max_new_tokens ${CIMR_REASONING_MAX_NEW_TOKENS}"
CIMR_FLAGS="${CIMR_FLAGS} --cimr_reasoning_temperature    ${CIMR_REASONING_TEMPERATURE}"
CIMR_FLAGS="${CIMR_FLAGS} --cimr_reasoning_top_p          ${CIMR_REASONING_TOP_P}"
CIMR_FLAGS="${CIMR_FLAGS} --cimr_critique_max_tokens      ${CIMR_CRITIQUE_MAX_TOKENS}"
CIMR_FLAGS="${CIMR_FLAGS} --cimr_critique_max_trace_chars ${CIMR_CRITIQUE_MAX_TRACE_CHARS}"
CIMR_FLAGS="${CIMR_FLAGS} --cimr_correction_max_tokens          ${CIMR_CORRECTION_MAX_TOKENS}"
CIMR_FLAGS="${CIMR_FLAGS} --cimr_correction_max_trace_chars     ${CIMR_CORRECTION_MAX_TRACE_CHARS}"
CIMR_FLAGS="${CIMR_FLAGS} --cimr_correction_max_critique_chars  ${CIMR_CORRECTION_MAX_CRITIQUE_CHARS}"
CIMR_FLAGS="${CIMR_FLAGS} --cimr_continuation_max_prior_chars   ${CIMR_CONTINUATION_MAX_PRIOR_CHARS}"
[[ "${CIMR_OTHER_AWARE}" == "1" ]] && CIMR_FLAGS="${CIMR_FLAGS} --cimr_other_answer_aware"
[[ "${NO_CRITIQUE}"      == "1" ]] && CIMR_FLAGS="${CIMR_FLAGS} --no_critique"
[[ "${NO_CORRECTION}"    == "1" ]] && CIMR_FLAGS="${CIMR_FLAGS} --no_correction"

COMMON_ARGS=(
    --model           "${MODEL_NAME}"
    --verifier_model  "${VERIFIER_MODEL}"
    --embedding_model "${EMBEDDING_MODEL}"
    --max_new_tokens  "${MAX_NEW_TOKENS}"
    --temperature     "${TEMPERATURE}"
    --top_p           "${TOP_P}"
    --alpha_values    ${ALPHA_VALUES}
    --tau_values      ${TAU_VALUES}
    --output_dir      "${OUT}"
    --batch_size      "${BATCH_SIZE}"
    ${GPU_FLAG}
    ${CIMR_FLAGS}
)
[[ "${SKIP_BASELINES}" == "1" ]] && COMMON_ARGS+=(--skip_baselines)

run_dataset() {
    local DATASET="$1" N="$2" MAX_SAMPLES="$3"
    local TAG="${DATASET}_N${N}"
    local LOG="${OUT}/logs/paired_${TAG}_d${CIMR_DEPTH}_R${CIMR_N_ROLLOUTS}.log"
    local CACHE_FILE="${OUT}/sample_cache_${TAG}.jsonl"
    local CACHE_FLAG=() MAX_FLAG=()

    [[ "${LOAD_CACHE:-0}" == "1" && -f "${CACHE_FILE}" ]] && CACHE_FLAG=(--load_cache "${CACHE_FILE}")
    [[ -n "${MAX_SAMPLES}" ]] && MAX_FLAG=(--max_samples "${MAX_SAMPLES}")

    echo "  dataset=${DATASET}  N=${N}  rollouts=${CIMR_N_ROLLOUTS}  depth=${CIMR_DEPTH}"

    ${PYTHON} run_paired_ablation.py \
        "${COMMON_ARGS[@]}" \
        --dataset "${DATASET}" \
        --N       "${N}" \
        "${MAX_FLAG[@]}" \
        "${CACHE_FLAG[@]}" \
        2>&1 | tee "${LOG}"
}

DATASETS=("${DATASET_ARGS[@]}")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(aime24 aime25 amc23 math500 olympiadbench)

echo "model=${MODEL_NAME}  rollouts=${CIMR_N_ROLLOUTS}  depth=${CIMR_DEPTH}  datasets=${DATASETS[*]}"

for DATASET in "${DATASETS[@]}"; do
    case "${DATASET}" in
        aime24)        run_dataset aime24        "${N_DEFAULT}" "${MAX_SAMPLES_AIME}"     ;;
        aime25)        run_dataset aime25        "${N_DEFAULT}" "${MAX_SAMPLES_AIME}"     ;;
        amc23)         run_dataset amc23         "${N_DEFAULT}" "${MAX_SAMPLES_AMC}"      ;;
        math500)       run_dataset math500       "${N_DEFAULT}" "${MAX_SAMPLES_MATH}"     ;;
        olympiadbench) run_dataset olympiadbench "${N_DEFAULT}" "${MAX_SAMPLES_OLYMPIAD}" ;;
        gsm8k)         run_dataset gsm8k         "${N_DEFAULT}" "${MAX_SAMPLES_GSM}"      ;;
        *) echo "Unknown dataset: '${DATASET}'"; exit 1 ;;
    esac
done
