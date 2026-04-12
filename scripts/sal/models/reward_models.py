"""
Process Reward Model (PRM) wrapper using HuggingFace Transformers.

Loads Qwen2.5-Math-PRM-7B (or compatible PRM) via AutoModelForCausalLM
with trust_remote_code=True, since these models ship custom modeling files.
"""

import logging
import sys
import os
from typing import List
from copy import deepcopy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

logger = logging.getLogger(__name__)


class PRM:
    """
    Process Reward Model that scores partial or complete reasoning trajectories.

    The Qwen2.5-Math-PRM uses a special format:
      - The conversation is formatted with the problem as user message
        and the solution steps as assistant message.
      - Step separators are marked with a special token.
      - The model outputs a reward score at each step boundary.
    """

    def __init__(self, model_path: str, device: str = "auto", dtype: str = "bfloat16", prm_batch_size: int = 1):
        logger.info(f"Loading PRM from {model_path}")

        # Add model path to sys.path so custom modules can be imported
        if model_path not in sys.path:
            sys.path.insert(0, model_path)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.config.use_cache = False
        self.model.eval()
        self.device = self.model.device
        self.prm_batch_size = prm_batch_size
        logger.info(f"PRM loaded on {self.device}")

    @torch.no_grad()
    def score(
        self,
        prompts: List[str],
        completions: List[List[str]],
    ) -> List[List[List[float]]]:
        """
        Score completions for given prompts.

        Args:
            prompts: List of problem strings.
            completions: For each prompt, a list of candidate completions.
                         Each completion is a string of reasoning steps.

        Returns:
            Nested list: [prompt_idx][completion_idx] -> list of step scores.
            Each step score is a float in [0, 1].
        """
        # Flatten all (prompt, completion) pairs into a single batch
        flat_texts = []
        shape = []  # number of completions per prompt, to reconstruct nesting

        for prompt, comps in zip(prompts, completions):
            shape.append(len(comps))
            for completion in comps:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    text = f"User: {prompt}\nAssistant: {completion}"
                flat_texts.append(text)

        # Single batched forward pass for all (prompt, completion) pairs
        flat_scores = self._score_batch(flat_texts)

        # Reconstruct nested [prompt_idx][completion_idx] -> [step_scores]
        all_scores = []
        idx = 0
        for n_comps in shape:
            prompt_scores = []
            for _ in range(n_comps):
                prompt_scores.append(flat_scores[idx])
                idx += 1
            all_scores.append(prompt_scores)

        return all_scores

    def _score_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Run a batched forward pass over all texts and return one score per text.

        Falls back to mini-batch scoring when the model exposes a custom `score`
        method (e.g. Qwen2.5-Math-PRM).  prm_batch_size controls how many texts
        are scored in each call; increase it to improve GPU utilisation.
        """
        if hasattr(self.model, "score"):
            results = []
            for start in range(0, len(texts), self.prm_batch_size):
                chunk = texts[start : start + self.prm_batch_size]
                results.extend(self._score_via_model_method_text(t) for t in chunk)
            return results

        return self._score_via_logits_batch(texts)

    def _score_via_model_method_text(self, text: str) -> List[float]:
        """Score a pre-formatted text string using the model's built-in score head."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True
        ).to(self.device)
        outputs = self.model(**inputs)

        logits = outputs.logits[0, -1, :]
        if logits.shape[-1] == 1:
            score = torch.sigmoid(logits).item()
        else:
            probs = torch.softmax(logits[:2], dim=-1)
            score = probs[1].item()

        return [score]

    def _score_via_logits_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batched logits-based scoring.

        Tokenizes all texts together with padding, runs one forward pass,
        then extracts the per-sample average log-probability using each
        sample's actual (non-padding) token positions.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        outputs = self.model(**enc, output_hidden_states=False)
        logits = outputs.logits  # (B, T, V)

        shift_logits = logits[:, :-1, :]            # (B, T-1, V)
        shift_labels = enc["input_ids"][:, 1:]      # (B, T-1)
        attention_mask = enc["attention_mask"][:, 1:]  # (B, T-1)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)

        # Mask padding tokens before averaging
        masked = token_log_probs * attention_mask
        avg_log_probs = masked.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

        import math
        scores = []
        for avg in avg_log_probs.tolist():
            score = 1.0 / (1.0 + math.exp(-avg * 10))
            scores.append([score])

        return scores

    # ------------------------------------------------------------------
    # Legacy single-sample methods kept for reference / external callers
    # ------------------------------------------------------------------

    def _score_single(self, prompt: str, completion: str) -> List[float]:
        if hasattr(self.model, "score"):
            return self._score_via_model_method(prompt, completion)
        return self._score_via_logits(prompt, completion)

    def _score_via_model_method(self, prompt: str, completion: str) -> List[float]:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return self._score_via_model_method_text(text)

    def _score_via_logits(self, prompt: str, completion: str) -> List[float]:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = f"User: {prompt}\nAssistant: {completion}"
        return self._score_via_logits_batch([text])[0]


def load_prm(config) -> PRM:
    """Load PRM from config."""
    return PRM(
        model_path=config.prm_path,
        device=config.device if config.device != "auto" else "auto",
        dtype=config.dtype,
        prm_batch_size=getattr(config, "prm_batch_size", 1),
    )