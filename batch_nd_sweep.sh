#!/bin/bash
#SBATCH -p sooner_gpu_test_h100
#SBATCH --gres=gpu:1
#SBATCH --job-name=critic_batch_nd_sweep
#SBATCH --output=logs/out_nd_sweep_%j.log
#SBATCH --error=logs/err_nd_sweep_%j.log
#SBATCH --time=2-00:00:00

set -e

CONFIG="${1:?Usage: sbatch batch_nd_sweep.sh configs/<model>.conf}"
source "${CONFIG}"

cd /home/ahsan/crisp

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache
mkdir -p "$HF_HOME"

source ~/.bashrc

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



# sbatch batch_nd_sweep.sh configs/llama3_8b_cric_corr.conf
# sbatch batch_nd_sweep.sh configs/ministral_8b_cric_corr.conf
# sbatch batch_nd_sweep.sh configs/qwen_1b_cric_corr.conf
# sbatch batch_nd_sweep.sh configs/qwen_7b_cric_corr.conf




# sbatch batch_nd_sweep.sh configs/llama3_8b.conf
# sbatch batch_nd_sweep.sh configs/ministral_8b.conf
# sbatch batch_nd_sweep.sh configs/qwen_1b.conf
# sbatch batch_nd_sweep.sh configs/qwen_7b.conf