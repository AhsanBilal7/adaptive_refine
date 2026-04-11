#!/bin/bash
# DORA with NLI-based direction similarity
set -e

# pip install latex2sympy2 word2number FlagEmbedding "transformers>=4.36.0" scikit-learn scipy datasets pyyaml accelerate

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/allmodels/Qwen2.5-Math-PRM-7B"

export CONFIG=recipes/Llama-3.2-1B-Instruct/dora_nli_qwen.yaml
export prm_path=Qwen/Qwen2.5-Math-PRM-7B
export model_path=meta-llama/Llama-3.2-1B-Instruct
# export prm_path=allmodels/Qwen2.5-Math-PRM-7B
# export model_path=allmodels/Llama-3.2-1B-Instruct
export nli_model_path=allmodels/roberta-large-mnli
export output_dir=result/MATH500/dora_nli

# for n in 16 32 64 128 256; do
#     for seed in 0 1 2 3 4; do
#         python scripts/test_time_compute.py $CONFIG \
#             --n=$n --balance_alpha=0.01 --seed=$seed \
#             --prm_path=$prm_path --model_path=$model_path \
#             --nli_model_path=$nli_model_path \
#             --similarity_mode=nli --nli_lambda_penalty=1.0 \
#             --output_dir=$output_dir
#     done
# done



n=64
seed=0

python scripts/test_time_compute.py $CONFIG \
    --n=$n --balance_alpha=0.01 --seed=$seed \
    --prm_path=$prm_path --model_path=$model_path \
    --nli_model_path=$nli_model_path \
    --similarity_mode=nli --nli_lambda_penalty=1.0 \
    --output_dir=$output_dir