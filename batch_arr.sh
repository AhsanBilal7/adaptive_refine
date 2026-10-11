#!/bin/bash
# Example Slurm job: one GPU, one ARR run.  Add the partition / account of your cluster, then
#   sbatch batch_arr.sh run configs/arr_qwen_math_7b.conf,configs/arr_aime24.conf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=arr
#SBATCH --output=logs/out_arr_%j.log
#SBATCH --error=logs/err_arr_%j.log
#SBATCH --time=2-00:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs

if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

bash run_arr.sh "$@"
