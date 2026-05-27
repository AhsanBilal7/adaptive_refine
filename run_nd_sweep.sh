#!/usr/bin/env bash
set -e

CONFIG="${1:?Usage: bash run_nd_sweep.sh configs/<model>.conf}"
source "${CONFIG}"

cd /home/ahsan/crisp

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache
mkdir -p "$HF_HOME"

ABLATION_FLAGS=()
[[ "${NO_CRITIQUE:-0}"   == "1" ]] && ABLATION_FLAGS+=(--no_critique)
[[ "${NO_CORRECTION:-0}" == "1" ]] && ABLATION_FLAGS+=(--no_correction)

for N in "${N_VALUES[@]}"; do
    for D in "${D_VALUES[@]}"; do
        echo "=== N=${N} D=${D} ==="
        uv run -- bash /home/ahsan/crisp/run_paired_ablation.sh \
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