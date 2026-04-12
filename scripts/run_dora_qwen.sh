# pip install vllm==0.6.5
# pip install latex2sympy2
# pip install word2number
# # pip install openai==0.28.0
# pip install -U FlagEmbedding
export PYTHONPATH="${PYTHONPATH}:$(pwd)/allmodels/Qwen2.5-Math-PRM-7B"


export CONFIG=recipes/Llama-3.2-1B-Instruct/dora_nli_qwen.yaml
export prm_path=Qwen/Qwen2.5-Math-PRM-7B
export model_path=meta-llama/Llama-3.2-1B-Instruct
export output_dir=./result/MATH500/Qwen2.5-Math-PRM-7B/Llama-3.2-1B-Instruct/dora
export nli_model_path=FacebookAI/roberta-large-mnli


python ./scripts/test_time_compute.py $CONFIG \
    --n=16 \
    --balance_alpha=0.01 \
    --seed=0 \
    --prm_path=$prm_path \
    --model_path=$model_path \
    --output_dir=$output_dir