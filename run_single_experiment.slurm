#!/bin/bash
#SBATCH -p sooner_gpu_test_h100
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

set -e

DATASET="$1"
METHOD="$2"

if [ -z "$DATASET" ] || [ -z "$METHOD" ]; then
    echo "Usage: sbatch run_single_experiment.slurm <dataset> <method>"
    exit 1
fi

cd /home/ahsan/adaptive_scientific_discovery

# -----------------------------
# Clean old cache (optional)
# -----------------------------
echo "Before cleanup:"
du -sh ~/.cache/* 2>/dev/null || true

rm -rf ~/.cache/huggingface/hub || true

echo "After cleanup:"
du -sh ~/.cache/* 2>/dev/null || true

# -----------------------------
# Set scratch cache
# -----------------------------
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache
mkdir -p "$HF_HOME"

echo "HF cache set to: $HF_HOME"

# -----------------------------
# Load environment
# -----------------------------
source ~/.bashrc
conda activate dora

# -----------------------------
# Debug info
# -----------------------------
echo "Running on $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "Dataset: $DATASET"
echo "Method: $METHOD"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# -----------------------------
# Run single dataset + single method
# -----------------------------
bash run_experiments_llama_3b.sh run "$DATASET" "$METHOD"