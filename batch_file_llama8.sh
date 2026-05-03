#!/bin/bash
#SBATCH -p sooner_gpu_test_h100
#SBATCH --gres=gpu:1
#SBATCH --job-name=cimr_llama8_paired_ablation
#SBATCH --output=logs/out_cimr_llama8_paired_ablation.log
#SBATCH --error=logs/err_cimr_llama8_paired_ablation.log
#SBATCH --time=2-00:00:00

set -e

cd /home/ahsan/crisp

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache

mkdir -p $HF_HOME

source ~/.bashrc

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"

uv run -- bash /home/ahsan/crisp/run_paired_ablation.sh --model_name "${MODEL_NAME}" aime24 aime25 amc23 math500