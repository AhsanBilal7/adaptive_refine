#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=procedure_rsi
#SBATCH --output=logs/out_rsi_%j.log
#SBATCH --error=logs/err_rsi_%j.log
#SBATCH --time=2-00:00:00
#
# Slurm wrapper around run_procedure_rsi.sh (submit from the repo root):
#   sbatch batch_procedure_rsi.sh develop  configs/rsi_qwen_1b.conf
#   sbatch batch_procedure_rsi.sh develop  configs/rsi_qwen_1b.conf,configs/ablations/primitive_only.conf
#   sbatch batch_procedure_rsi.sh evaluate configs/rsi_qwen_1b.conf -- --methods h0,maj_matched,heuristic,final
# Override the partition at submit time if needed: sbatch -p <partition> ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
bash run_procedure_rsi.sh "$@"
