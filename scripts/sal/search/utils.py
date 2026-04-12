"""
Beam dataclass and generation utilities for HuggingFace Transformers backend.

Replaces the original vLLM-specific utils (Beam, build_conv, generate_k_steps)
with a pure Transformers implementation using model.generate().
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List
from copy import deepcopy

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class Beam:
    """A single reasoning trajectory being explored during search."""
    prompt: str
    index: int
    current_text: str = ""
    next_texts: Optional[List[str]] = None
    lookahead_texts: Optional[List[str]] = None
    pruned: bool = False
    completed: bool = False
    stop_reasons: Optional[List[str]] = None
    history: List[str] = field(default_factory=list)
    best_scores: List[float] = field(default_factory=list)
    all_scores: List[float] = field(default_factory=list)
    previous_text: Optional[str] = None
    completion_tokens: int = 0
    store_tokens: List = field(default_factory=list)
    history_embedding: List = field(default_factory=list)


@dataclass
class GenResult:
    """Result from a single generation step."""
    next_texts: List[str]
    stop_reasons: List[str]
    lookahead_texts: Optional[List[str]]
    completion_tokens: int


def build_conv(prompt: str, current_text: str, system_prompt: Optional[str] = None) -> list:
    """
    Build a chat conversation for the tokenizer's apply_chat_template.

    Returns a list of message dicts: [{"role": ..., "content": ...}, ...]
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if current_text:
        messages.append({"role": "assistant", "content": current_text})
    return messages


def _decode_single_result(
    new_token_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    stop_strings: Optional[List[str]],
) -> GenResult:
    """Decode one generated sequence and determine stop reason."""
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    token_list = new_token_ids.tolist()

    # Find the actual end: stop at the first EOS (inclusive)
    actual_len = len(token_list)
    hit_eos = False
    for t_idx, tok in enumerate(token_list):
        if tok == eos_id:
            actual_len = t_idx + 1
            hit_eos = True
            break

    num_new_tokens = actual_len
    generated_text = tokenizer.decode(token_list[:actual_len], skip_special_tokens=True)

    stop_reason = "EOS" if (hit_eos or num_new_tokens == 0) else "length"

    # Handle stop_strings: truncate at first match
    if stop_strings and stop_reason != "EOS":
        for ss in stop_strings:
            pos = generated_text.find(ss)
            if pos != -1:
                generated_text = generated_text[: pos + len(ss)]
                stop_reason = "stop"
                num_new_tokens = len(
                    tokenizer.encode(generated_text, add_special_tokens=False)
                )
                break

    return GenResult(
        next_texts=[generated_text],
        stop_reasons=[stop_reason],
        lookahead_texts=None,
        completion_tokens=num_new_tokens,
    )


def _generate_mini_batch(
    prompt_texts: List[str],
    budgets: List[int],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_strings: Optional[List[str]],
) -> List[GenResult]:
    """
    Run model.generate() on a single mini-batch of prompts.

    Uses left-padding so causal attention is correct for all sequences.
    max_new_tokens for the batch is the minimum individual budget to avoid
    exceeding any beam's remaining token allowance.
    """
    device = model.device
    batch_budget = min(min(budgets), max_new_tokens)
    model_max = (
        model.config.max_position_embeddings
        if hasattr(model.config, "max_position_embeddings")
        else 4096
    )

    # Left-padding: required for batched causal-LM generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model_max,
    ).to(device)
    tokenizer.padding_side = orig_padding_side

    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": batch_budget,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    results = []
    for i in range(len(prompt_texts)):
        new_token_ids = output_ids[i, input_len:]
        results.append(_decode_single_result(new_token_ids, tokenizer, stop_strings))

    return results


def generate_step(
    templated_prompts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    temperature: float = 0.8,
    top_p: float = 1.0,
    max_new_tokens: int = 256,
    stop_strings: Optional[List[str]] = None,
    left_completion_tokens: Optional[List[int]] = None,
    generation_batch_size: int = 1,
) -> List[GenResult]:
    """
    Generate one reasoning step for each prompt using HuggingFace Transformers.

    Args:
        templated_prompts: List of fully-templated prompt strings.
        model: HuggingFace causal LM.
        tokenizer: HuggingFace tokenizer.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_new_tokens: Maximum new tokens to generate per step.
        stop_strings: Strings that stop generation (e.g. ["\\n\\n"]).
        left_completion_tokens: Per-beam remaining token budget.
        generation_batch_size: How many prompts to process in one model.generate()
            call. Values > 1 amortise the GPU kernel launch overhead across beams
            and are significantly faster than the default of 1.

    Returns:
        List of GenResult, one per prompt.
    """
    budgets = [
        min(max_new_tokens, max(1, left_completion_tokens[i]))
        if left_completion_tokens is not None
        else max_new_tokens
        for i in range(len(templated_prompts))
    ]

    results: List[GenResult] = []
    for start in range(0, len(templated_prompts), generation_batch_size):
        end = min(start + generation_batch_size, len(templated_prompts))
        batch_prompts = templated_prompts[start:end]
        batch_budgets = budgets[start:end]

        batch_results = _generate_mini_batch(
            batch_prompts, batch_budgets, model, tokenizer,
            temperature, top_p, max_new_tokens, stop_strings,
        )
        results.extend(batch_results)

    return results


def last(scores):
    """Return the last score in a list (default aggregation)."""
    if not scores:
        return 0.0
    return scores[-1]