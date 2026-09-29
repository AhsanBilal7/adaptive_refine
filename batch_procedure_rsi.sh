#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=procedure_rsi
#SBATCH --output=logs/out_rsi_%j.log
#SBATCH --error=logs/err_rsi_%j.log
#SBATCH --time=2-00:00:00
#
# Slurm wrapper (submit from the repo root):
#   sbatch batch_procedure_rsi.sh experiment configs/rsi_qwen_7b.conf,configs/rsi_qwen_7b_aime.conf   # develop → evaluate
#   sbatch batch_procedure_rsi.sh develop  configs/rsi_qwen_1b.conf
#   sbatch batch_procedure_rsi.sh develop  configs/rsi_qwen_1b.conf,configs/ablations/primitive_only.conf
#   sbatch batch_procedure_rsi.sh evaluate configs/rsi_qwen_1b.conf -- --methods h0,maj_matched,heuristic,final
# Override the partition at submit time if needed: sbatch -p <partition> ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # compute nodes have no internet; models/datasets are cached
if [[ "${1:-}" == "experiment" ]]; then
    shift
    bash run_rsi_experiment.sh "$@"                 # resumable develop → evaluate
else
    bash run_procedure_rsi.sh "$@"
fi
