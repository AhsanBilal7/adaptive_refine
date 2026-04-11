#!/bin/bash
# DORA with Hybrid (Cosine + NLI) direction similarity
set -e

pip install latex2sympy2 word2number FlagEmbedding "transformers>=4.36.0" scikit-learn scipy datasets pyyaml accelerate

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/allmodels/Qwen2.5-Math-PRM-7B"

export CONFIG=recipes/Llama-3.2-1B-Instruct/dora_hybrid_qwen.yaml
export prm_path=allmodels/Qwen2.5-Math-PRM-7B
export model_path=allmodels/Llama-3.2-1B-Instruct
export nli_model_path=allmodels/roberta-large-mnli
export output_dir=result/MATH500/dora_hybrid

for n in 16 32 64 128 256; do
    for seed in 0 1 2 3 4; do
        python scripts/test_time_compute.py $CONFIG \
            --n=$n --balance_alpha=0.01 --seed=$seed \
            --prm_path=$prm_path --model_path=$model_path \
            --nli_model_path=$nli_model_path \
            --similarity_mode=hybrid --nli_lambda_penalty=1.0 \
            --nli_top_k=10 --output_dir=$output_dir
    done
done
