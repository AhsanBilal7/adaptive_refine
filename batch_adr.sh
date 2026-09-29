#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=adr
#SBATCH --output=logs/out_adr_%j.log
#SBATCH --error=logs/err_adr_%j.log
#SBATCH --time=2-00:00:00
#
# Slurm wrapper for run_adr.sh (submit from the repo root). Resubmitting the
# same command resumes from the scratch checkpoints.
#   sbatch batch_adr.sh run   configs/adr_qwen_7b.conf,configs/adr_aime24.conf
#   sbatch batch_adr.sh pilot configs/adr_qwen_7b.conf
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"     # compute nodes have no internet; models/datasets are cached
bash run_adr.sh "$@"
