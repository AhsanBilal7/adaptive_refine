#!/usr/bin/env bash
# =============================================================================
#  LLM-SR – Model Server Launch Script
#  Supports every model family via engine.py auto-detection.
#
#  Usage:  bash run_server.sh
#  Uncomment the block for the model you want to serve.
# =============================================================================


# ── Qwen 2.5 ─────────────────────────────────────────────────────────────────

## 7B  (1× GPU, 4-bit quant)
# python llm_engine/engine.py \
#     --model_path Qwen/Qwen2.5-7B-Instruct \
#     --gpu_ids 0 --port 5000 --quantization

## 14B  (2× GPUs, 4-bit)
# python llm_engine/engine.py \
#     --model_path Qwen/Qwen2.5-14B-Instruct \
#     --gpu_ids 0 1 --port 5000 --quantization

## 32B  (4× GPUs, 4-bit)
# python llm_engine/engine.py \
#     --model_path Qwen/Qwen2.5-32B-Instruct \
#     --gpu_ids 0 1 2 3 --port 5000 --quantization

## Coder-7B
# python llm_engine/engine.py \
#     --model_path Qwen/Qwen2.5-Coder-7B-Instruct \
#     --gpu_ids 0 --port 5000 --quantization


# ── Llama 3.x ─────────────────────────────────────────────────────────────────

## Llama-3.1-8B  (1× GPU)
# python llm_engine/engine.py \
#     --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --gpu_ids 0 --port 5000 --quantization

## Llama-3.1-70B  (4× GPUs)
# python llm_engine/engine.py \
#     --model_path meta-llama/Meta-Llama-3.1-70B-Instruct \
#     --gpu_ids 0 1 2 3 --port 5000 --quantization

## Llama-3.2-3B  (1× GPU, tiny)
# python llm_engine/engine.py \
#     --model_path meta-llama/Llama-3.2-3B-Instruct \
#     --gpu_ids 0 --port 5000


# ── DeepSeek ──────────────────────────────────────────────────────────────────

## DeepSeek-R1-Distill-Qwen-7B  (set max_new_tokens higher for reasoning)
# python llm_engine/engine.py \
#     --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --gpu_ids 0 --port 5000 --quantization --max_new_tokens 1024

## DeepSeek-R1-Distill-Llama-8B
# python llm_engine/engine.py \
#     --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#     --gpu_ids 0 --port 5000 --quantization --max_new_tokens 1024

## DeepSeek-Coder-V2-Lite-Instruct
# python llm_engine/engine.py \
#     --model_path deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
#     --gpu_ids 0 --port 5000 --quantization


# ── Mistral / Mixtral  (original LLM-SR models) ───────────────────────────────

## Mixtral-8x7B  (4× GPUs)
# python llm_engine/engine.py \
#     --model_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
#     --gpu_ids 0 1 2 3 --port 5000 --quantization

## Mistral-7B-v0.3  (1× GPU)
# python llm_engine/engine.py \
#     --model_path mistralai/Mistral-7B-Instruct-v0.3 \
#     --gpu_ids 0 --port 5000 --quantization


# ── Phi-3 / Phi-3.5 ──────────────────────────────────────────────────────────

## Phi-3.5-mini  (1× GPU, very fast)
# python llm_engine/engine.py \
#     --model_path microsoft/Phi-3.5-mini-instruct \
#     --gpu_ids 0 --port 5000 --quantization

## Phi-3-medium-4k
# python llm_engine/engine.py \
#     --model_path microsoft/Phi-3-medium-4k-instruct \
#     --gpu_ids 0 1 --port 5000 --quantization


# ── Gemma-2 ───────────────────────────────────────────────────────────────────

## Gemma-2-9B
# python llm_engine/engine.py \
#     --model_path google/gemma-2-9b-it \
#     --gpu_ids 0 1 --port 5000 --quantization


echo "Uncomment one block above and run:  bash run_server.sh"
