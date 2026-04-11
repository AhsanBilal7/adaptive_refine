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


def generate_step(
    templated_prompts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    temperature: float = 0.8,
    top_p: float = 1.0,
    max_new_tokens: int = 256,
    stop_strings: Optional[List[str]] = None,
    left_completion_tokens: Optional[List[int]] = None,
) -> List[GenResult]:
    """
    Generate one reasoning step for each prompt using HuggingFace Transformers.

    This replaces vLLM's batched generation. Each prompt is generated individually
    to handle per-beam max_new_tokens constraints.

    Args:
        templated_prompts: List of fully-templated prompt strings.
        model: HuggingFace causal LM.
        tokenizer: HuggingFace tokenizer.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_new_tokens: Maximum new tokens to generate per step.
        stop_strings: Strings that stop generation (e.g. ["\\n\\n"]).
        left_completion_tokens: Per-beam remaining token budget.

    Returns:
        List of GenResult, one per prompt.
    """
    results = []
    device = model.device

    for idx, prompt_text in enumerate(templated_prompts):
        # Per-beam token budget
        budget = max_new_tokens
        if left_completion_tokens is not None:
            budget = min(budget, max(1, left_completion_tokens[idx]))

        # Tokenize
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_position_embeddings
            if hasattr(model.config, "max_position_embeddings")
            else 4096,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": budget,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        # Stop at EOS or stop_strings via stopping_criteria not needed —
        # we'll post-process stop_strings from the decoded text
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_token_ids = output_ids[0, input_len:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        num_new_tokens = len(new_token_ids)

        # Determine stop reason
        stop_reason = "length"
        if num_new_tokens == 0:
            stop_reason = "EOS"
        elif output_ids[0, -1].item() == tokenizer.eos_token_id:
            stop_reason = "EOS"

        # Handle stop_strings: truncate at the first occurrence
        if stop_strings and stop_reason != "EOS":
            for ss in stop_strings:
                pos = generated_text.find(ss)
                if pos != -1:
                    # Include the stop string in output (matches vLLM include_stop_str_in_output=True)
                    generated_text = generated_text[: pos + len(ss)]
                    stop_reason = "stop"
                    # Recount tokens after truncation
                    num_new_tokens = len(
                        tokenizer.encode(generated_text, add_special_tokens=False)
                    )
                    break

        results.append(
            GenResult(
                next_texts=[generated_text],
                stop_reasons=[stop_reason],
                lookahead_texts=None,
                completion_tokens=num_new_tokens,
            )
        )

    return results


def last(scores):
    """Return the last score in a list (default aggregation)."""
    if not scores:
        return 0.0
    return scores[-1]