#!/usr/bin/env bash
# =============================================================================
#  AdaptSci – Experiment Run Script
#
#  Run from the project root (where both llmsr/ and adaptsci/ sit):
#    bash adaptsci/run_adaptsci.sh
#
#  §1   OpenAI API
#  §2   Anthropic Claude API
#  §3   Local server – Qwen 2.5
#  §4   Local server – Llama-3.1
#  §5   Local server – DeepSeek
#  §6   Local server – Mixtral / Mistral
#  §7   Local server – Phi-3.5
#  §8   Local server – Gemma-2
#  §9   vLLM backend
#  §10  Ablation matrix (all 4 modes × all 4 problems)
#  §11  Torch optimizer variants
#  §12  Sensitivity sweeps (κ, restarts, diversity_fn, progress_threshold)
# =============================================================================

# Shared paths
SPECS=./specs
DATA=./data
LOGS=./logs


# ── §1  OpenAI API ────────────────────────────────────────────────────────────
# export API_KEY=<your_key>

## GPT-3.5 – full AdaptSci – all problems
# for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#   python main.py --use_api True --api_model "gpt-3.5-turbo" \
#     --problem_name $PROB --spec_path $SPECS/specification_${PROB}_numpy.txt \
#     --log_path $LOGS/gpt35/full/$PROB --ablation_mode full
# done

## GPT-4o – full AdaptSci – all problems
# for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#   python main.py --use_api True --api_model "gpt-4o" \
#     --problem_name $PROB --spec_path $SPECS/specification_${PROB}_numpy.txt \
#     --log_path $LOGS/gpt4o/full/$PROB --ablation_mode full
# done

## GPT-3.5 – baseline (plain LLM-SR) – all problems
# for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#   python main.py --use_api True --api_model "gpt-3.5-turbo" \
#     --problem_name $PROB --spec_path $SPECS/specification_${PROB}_numpy.txt \
#     --log_path $LOGS/gpt35/none/$PROB --ablation_mode none
# done


# ── §2  Anthropic Claude API ──────────────────────────────────────────────────
# export ANTHROPIC_API_KEY=sk-ant-...

## Claude 3.5 Sonnet – full – all problems
# for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#   python main.py --use_api True --api_model "claude-3-5-sonnet-20241022" \
#     --problem_name $PROB --spec_path $SPECS/specification_${PROB}_numpy.txt \
#     --log_path $LOGS/claude35sonnet/full/$PROB --ablation_mode full
# done

## Claude 3 Opus – oscillator2
# python main.py --use_api True --api_model "claude-3-opus-20240229" \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/claude3opus/full/oscillator2 --ablation_mode full

## Claude 3.5 Haiku – oscillator2  (fastest / cheapest)
# python main.py --use_api True --api_model "claude-3-5-haiku-20241022" \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/claude35haiku/full/oscillator2 --ablation_mode full


# ── §3  Local server – Qwen 2.5 ───────────────────────────────────────────────
# Start server first:  bash adaptsci/run_server.sh  (uncomment Qwen 2.5 block)

## Qwen2.5-7B – full – all problems
# for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#   python main.py --local_server_url http://127.0.0.1:5000 \
#     --problem_name $PROB --spec_path $SPECS/specification_${PROB}_numpy.txt \
#     --log_path $LOGS/qwen2.5_7b/full/$PROB --ablation_mode full
# done

## Qwen2.5-14B – oscillator2
# python main.py --local_server_url http://127.0.0.1:5000 \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/qwen2.5_14b/full/oscillator2 --ablation_mode full

## Qwen2.5-Coder-7B – oscillator2
# python main.py --local_server_url http://127.0.0.1:5000 \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/qwen2.5_coder7b/full/oscillator2 --ablation_mode full


# ── §4  Local server – Llama-3.1 ──────────────────────────────────────────────

## Llama-3.1-8B – full – all problems
# for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#   python main.py --local_server_url http://127.0.0.1:5000 \
#     --problem_name $PROB --spec_path $SPECS/specification_${PROB}_numpy.txt \
#     --log_path $LOGS/llama31_8b/full/$PROB --ablation_mode full
# done

## Llama-3.1-70B – oscillator2  (server with 4 GPUs)
# python main.py --local_server_url http://127.0.0.1:5000 \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/llama31_70b/full/oscillator2 --ablation_mode full

## Llama-3.2-3B – oscillator1  (tiny, single GPU)
# python main.py --local_server_url http://127.0.0.1:5000 \
#   --problem_name oscillator1 --spec_path $SPECS/specification_oscillator1_numpy.txt \
#   --log_path $LOGS/llama32_3b/full/oscillator1 --ablation_mode full


# ── §5  Local server – DeepSeek ───────────────────────────────────────────────

## DeepSeek-R1-Distill-Qwen-7B – full – oscillator2
# python main.py --local_server_url http://127.0.0.1:5000 \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/deepseek_r1_7b/full/oscillator2 --ablation_mode full

## DeepSeek-R1-Distill-Qwen-7B – full – all problems
# for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#   python main.py --local_server_url http://127.0.0.1:5000 \
#     --problem_name $PROB --spec_path $SPECS/specification_${PROB}_numpy.txt \
#     --log_path $LOGS/deepseek_r1_7b/full/$PROB --ablation_mode full
# done

## DeepSeek-Coder-V2-Lite – bactgrow
# python main.py --local_server_url http://127.0.0.1:5000 \
#   --problem_name bactgrow --spec_path $SPECS/specification_bactgrow_numpy.txt \
#   --log_path $LOGS/deepseek_coderv2/full/bactgrow --ablation_mode full


# ── §6  Local server – Mixtral / Mistral ──────────────────────────────────────

## Mixtral-8x7B – full – all problems
# for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#   python main.py --local_server_url http://127.0.0.1:5000 \
#     --problem_name $PROB --spec_path $SPECS/specification_${PROB}_numpy.txt \
#     --log_path $LOGS/mixtral/full/$PROB --ablation_mode full
# done

## Mistral-7B-v0.3 – oscillator2
python main.py --local_server_url http://127.0.0.1:5000 \
  --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
  --log_path $LOGS/mistral7b/full/oscillator2 --ablation_mode full


# ── §7  Local server – Phi-3.5 ────────────────────────────────────────────────

## Phi-3.5-mini – full – oscillator2
# python main.py --local_server_url http://127.0.0.1:5000 \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/phi35mini/full/oscillator2 --ablation_mode full

## Phi-3.5-mini – full – all problems
# for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#   python main.py --local_server_url http://127.0.0.1:5000 \
#     --problem_name $PROB --spec_path $SPECS/specification_${PROB}_numpy.txt \
#     --log_path $LOGS/phi35mini/full/$PROB --ablation_mode full
# done


# ── §8  Local server – Gemma-2 ────────────────────────────────────────────────

## Gemma-2-9B – full – oscillator2
# python main.py --local_server_url http://127.0.0.1:5000 \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/gemma2_9b/full/oscillator2 --ablation_mode full


# ── §9  vLLM backend ──────────────────────────────────────────────────────────
# Start vLLM server:
#   python -m vllm.entrypoints.openai.api_server \
#       --model Qwen/Qwen2.5-7B-Instruct --port 8000 --served-model-name qwen2.5-7b

## vLLM + Qwen2.5-7B – full – all problems
# for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#   python main.py --llm_backend vllm --vllm_url http://127.0.0.1:8000 \
#     --problem_name $PROB --spec_path $SPECS/specification_${PROB}_numpy.txt \
#     --log_path $LOGS/vllm_qwen2.5_7b/full/$PROB --ablation_mode full
# done

## vLLM + Llama-3.1-8B – full – oscillator2
# python main.py --llm_backend vllm --vllm_url http://127.0.0.1:8000 \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/vllm_llama31_8b/full/oscillator2 --ablation_mode full

## vLLM + DeepSeek-R1-7B – full – oscillator2
# python main.py --llm_backend vllm --vllm_url http://127.0.0.1:8000 \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/vllm_deepseek_r1/full/oscillator2 --ablation_mode full


# ── §10  Ablation matrix ───────────────────────────────────────────────────────
# Runs all 4 modes × all 4 problems.
# Change BACKEND_FLAGS to switch the LLM.

# BACKEND_FLAGS="--use_api True --api_model gpt-3.5-turbo"
# BACKEND_FLAGS="--use_api True --api_model gpt-4o"
# BACKEND_FLAGS="--use_api True --api_model claude-3-5-sonnet-20241022"
# BACKEND_FLAGS="--local_server_url http://127.0.0.1:5000"

# for MODE in full buffer_only hessian_only none; do
#   for PROB in oscillator1 oscillator2 bactgrow stressstrain; do
#     python main.py $BACKEND_FLAGS \
#       --problem_name $PROB \
#       --spec_path $SPECS/specification_${PROB}_numpy.txt \
#       --log_path $LOGS/ablation/${MODE}/$PROB \
#       --ablation_mode $MODE
#   done
# done


# ── §11  Torch optimizer variants ─────────────────────────────────────────────

## GPT-4o + torch – oscillator2
# python main.py --use_api True --api_model "gpt-4o" \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_torch.txt \
#   --log_path $LOGS/gpt4o/full/oscillator2_torch --ablation_mode full

## GPT-3.5 – numpy vs torch comparison
# python main.py --use_api True --api_model "gpt-3.5-turbo" \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#   --log_path $LOGS/gpt35/none/oscillator2_numpy --ablation_mode none

# python main.py --use_api True --api_model "gpt-3.5-turbo" \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_torch.txt \
#   --log_path $LOGS/gpt35/none/oscillator2_torch --ablation_mode none

## Qwen2.5-7B + torch – oscillator2
# python main.py --local_server_url http://127.0.0.1:5000 \
#   --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_torch.txt \
#   --log_path $LOGS/qwen2.5_7b/full/oscillator2_torch --ablation_mode full


# ── §12  Sensitivity sweeps ────────────────────────────────────────────────────

## κ threshold sweep {10, 50, 200}
# for KAPPA in 10 50 200; do
#   python main.py --use_api True --api_model "gpt-3.5-turbo" \
#     --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#     --log_path $LOGS/sweep/kappa${KAPPA}/oscillator2 \
#     --ablation_mode hessian_only --kappa_threshold ${KAPPA}
# done

## max_extra_restarts sweep {1, 3, 5}
# for R in 1 3 5; do
#   python main.py --use_api True --api_model "gpt-3.5-turbo" \
#     --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#     --log_path $LOGS/sweep/restarts${R}/oscillator2 \
#     --ablation_mode hessian_only --max_extra_restarts ${R}
# done

## diversity function sweep {token_jaccard, edit_distance, length_bucket}
# for DIV in token_jaccard edit_distance length_bucket; do
#   python main.py --use_api True --api_model "gpt-3.5-turbo" \
#     --problem_name bactgrow --spec_path $SPECS/specification_bactgrow_numpy.txt \
#     --log_path $LOGS/sweep/div_${DIV}/bactgrow \
#     --ablation_mode buffer_only --ab_diversity_fn ${DIV}
# done

## progress threshold sweep {0.005, 0.01, 0.05}
# for PT in 0.005 0.01 0.05; do
#   python main.py --use_api True --api_model "gpt-3.5-turbo" \
#     --problem_name oscillator2 --spec_path $SPECS/specification_oscillator2_numpy.txt \
#     --log_path $LOGS/sweep/pt${PT}/oscillator2 \
#     --ablation_mode buffer_only --ab_progress_threshold ${PT}
# done

## coupling threshold sweep {0.7, 0.85, 0.95}
# for CT in 0.7 0.85 0.95; do
#   python main.py --use_api True --api_model "gpt-3.5-turbo" \
#     --problem_name stressstrain --spec_path $SPECS/specification_stressstrain_numpy.txt \
#     --log_path $LOGS/sweep/coupling${CT}/stressstrain \
#     --ablation_mode hessian_only --coupling_threshold ${CT}
# done

echo "Run script ready.  Uncomment the desired block and run:  bash adaptsci/run_adaptsci.sh"



