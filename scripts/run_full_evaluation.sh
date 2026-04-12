#!/bin/bash
# Full evaluation: 3 models x 3 benchmarks x 3 similarity modes x 5 budgets
set -e

pip install latex2sympy2 word2number FlagEmbedding "transformers>=4.36.0" scikit-learn scipy datasets pyyaml accelerate

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/allmodels/Qwen2.5-Math-PRM-7B"

export prm_path=allmodels/Qwen2.5-Math-PRM-7B
export nli_model_path=FacebookAI/roberta-large-mnli

declare -A MODELS
MODELS["Llama-3.2-1B-Instruct"]="allmodels/Llama-3.2-1B-Instruct"
MODELS["Llama-3.2-3B-Instruct"]="allmodels/Llama-3.2-3B-Instruct"
MODELS["Qwen2.5-1.5B-Instruct"]="allmodels/Qwen2.5-1.5B-Instruct"

DATASETS=("MATH500" "AIME2024" "AIME2025")
SIM_MODES=("cosine" "nli" "hybrid")
BUDGETS=(16 32 64 128 256)
MATH_SEEDS=(0 1 2 3 4)
AIME_SEEDS=(0 1 2 3 4 5 6 7 8 9)

for model_name in "${!MODELS[@]}"; do
    model_path=${MODELS[$model_name]}
    config_file=recipes/${model_name}/dora_nli_qwen.yaml

    for dataset in "${DATASETS[@]}"; do
        if [ "$dataset" = "MATH500" ]; then seeds=("${MATH_SEEDS[@]}"); else seeds=("${AIME_SEEDS[@]}"); fi

        for sim_mode in "${SIM_MODES[@]}"; do
            for n in "${BUDGETS[@]}"; do
                for seed in "${seeds[@]}"; do
                    output_dir=result/${dataset}/${model_name}/dora_${sim_mode}
                    extra=""
                    if [ "$sim_mode" = "nli" ]; then
                        extra="--nli_model_path=$nli_model_path --nli_lambda_penalty=1.0"
                    elif [ "$sim_mode" = "hybrid" ]; then
                        extra="--nli_model_path=$nli_model_path --nli_lambda_penalty=1.0 --nli_top_k=10"
                    fi

                    echo "model=$model_name dataset=$dataset mode=$sim_mode n=$n seed=$seed"
                    python scripts/test_time_compute.py $config_file \
                        --n=$n --balance_alpha=0.01 --seed=$seed \
                        --prm_path=$prm_path --model_path=$model_path \
                        --similarity_mode=$sim_mode $extra \
                        --output_dir=$output_dir
                done
            done
        done
    done
done

echo "Full evaluation complete!"
