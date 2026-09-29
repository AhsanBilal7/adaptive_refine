#!/usr/bin/env bash
set -e

CONFIG="${1:?Usage: bash run_nd_sweep.sh configs/<model>.conf}"
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${ROOT_DIR}"

# .env: machine paths (venv, HF caches, logs) + predefined variables.
set -a
# shellcheck disable=SC1091
source "${ROOT_DIR}/.env"
set +a
# The conf overrides MODEL_NAME / N_VALUES / D_VALUES / DATASETS / ablation flags.
# shellcheck disable=SC1090
source "${CONFIG}"
mkdir -p "${HF_HOME}" "${LOG_DIR}"
export MODEL_NAME

ABLATION_FLAGS=()
[[ "${NO_CRITIQUE:-0}"   == "1" ]] && ABLATION_FLAGS+=(--no_critique)
[[ "${NO_CORRECTION:-0}" == "1" ]] && ABLATION_FLAGS+=(--no_correction)

for N in "${N_VALUES[@]}"; do
    for D in "${D_VALUES[@]}"; do
        echo "=== N=${N} D=${D} ==="
        uv run --frozen --no-sync -- bash "${ROOT_DIR}/run_paired_ablation.sh" \
            --model_name "${MODEL_NAME}" \
            --cimr_n_rollouts "${N}" \
            --cimr_depth "${D}" \
            "${ABLATION_FLAGS[@]}" \
            "${DATASETS[@]}"
    done
done

# bash run_nd_sweep.sh configs/llama3_8b.conf
# bash run_nd_sweep.sh configs/ministral_8b.conf
# bash run_nd_sweep.sh configs/qwen_1b.conf
# bash run_nd_sweep.sh configs/qwen_7b.conf


# bash run_nd_sweep.sh configs/llama3_8b_cric_corr.conf
# bash run_nd_sweep.sh configs/ministral_8b_cric_corr.conf
# bash run_nd_sweep.sh configs/qwen_1b_cric_corr.conf
# bash run_nd_sweep.sh configs/qwen_7b_cric_corr.conf