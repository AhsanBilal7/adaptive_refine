#!/bin/bash
# Original DORA with cosine similarity (baseline)
set -e

# pip install latex2sympy2 word2number FlagEmbedding "transformers>=4.36.0" scikit-learn scipy datasets pyyaml accelerate

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/allmodels/Qwen2.5-Math-PRM-7B"

# ── Defaults ──────────────────────────────────────────────────────────────────
n=16
seed=0
generation_batch_size=32   # beams per model.generate() call
prm_batch_size=32           # completions per PRM forward pass
search_batch_size=32        # dataset samples processed together
embedding_batch_size=32

# ── Parse named arguments (--key=value) ───────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --n=*)                   n="${arg#*=}" ;;
        --seed=*)                seed="${arg#*=}" ;;
        --generation_batch_size=*) generation_batch_size="${arg#*=}" ;;
        --prm_batch_size=*)      prm_batch_size="${arg#*=}" ;;
        --search_batch_size=*)   search_batch_size="${arg#*=}" ;;
        --embedding_batch_size=*) embedding_batch_size="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

export CONFIG=recipes/Llama-3.2-1B-Instruct/dora_nli_qwen.yaml
export prm_path=Qwen/Qwen2.5-Math-PRM-7B
export model_path=meta-llama/Llama-3.2-1B-Instruct
export output_dir=result/MATH500/dora_cosine

echo "Running DORA-Cosine: n=$n seed=$seed gen_bs=$generation_batch_size prm_bs=$prm_batch_size"

python scripts/test_time_compute.py $CONFIG \
    --n=$n --balance_alpha=0.01 --seed=$seed \
    --prm_path=$prm_path --model_path=$model_path \
    --similarity_mode=cosine \
    --generation_batch_size=$generation_batch_size \
    --prm_batch_size=$prm_batch_size \
    --search_batch_size=$search_batch_size \
    --embedding_batch_size=$embedding_batch_size \
    --output_dir=$output_dir
