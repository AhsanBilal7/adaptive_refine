#!/bin/bash
#SBATCH -p sooner_gpu_test_h100
#SBATCH --gres=gpu:1
#SBATCH --job-name=cimr_paired_qwen8b_exp
#SBATCH --output=logs/out_cimr_paired_qwen8b.log
#SBATCH --error=logs/err_cimr_paired_qwen8b.log
#SBATCH --time=2-00:00:00

set -e

cd /home/ahsan/crisp

# -----------------------------
# 🔹 Clean old cache (optional)
# -----------------------------
echo "Before cleanup:"
du -sh ~/.cache/* || true

rm -rf ~/.cache/huggingface/hub || true

echo "After cleanup:"
du -sh ~/.cache/* || true

# -----------------------------
# 🔹 Set scratch cache (IMPORTANT)
# -----------------------------
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache

# Create directory if not exists
mkdir -p $HF_HOME

echo "HF cache set to: $HF_HOME"

# -----------------------------
# 🔹 Load environment
# -----------------------------
source ~/.bashrc
# conda activate dora

# -----------------------------
# 🔹 Debug info (very useful)
# -----------------------------
echo "Running on $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# -----------------------------
# 🔹 Run your pipeline
# -----------------------------
# bash run_token_sweep.sh
# bash run_experiments.sh run math500,aime24,aime25,amc23,olympiadbench dn_bon_updated

# bash run_motivation_experiments.sh aime24,aime25 dn_bon,dn_bon_updated,bon,bon_majority
# bash run_motivation_experiments.sh olympiadbench dn_bon,dn_bon_updated,bon,bon_majority
# bash run_experiments.sh run mmlu,bbh,arc_challenge,truthfulqa,strategyqa greedy,greedy_pass_n,bon,dn_bon,bon_majority

# bash run_experiments.sh run aime24,aime25,amc23,olympiadbench,math500 greedy,bon,greedy_pass_n,bon_majority,dn_bon,dn_bon_updated


# Check uv environment
# uv run which python


# bash run_experiments.sh run math500 dn_bon_updated

# uv run -- bash run_paired_ablation.sh
uv run -- bash run_paired_ablation.sh aime24 aime25 amc23 math500