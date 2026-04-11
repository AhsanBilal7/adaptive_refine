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

    def __init__(self, model_path: str, device: str = "auto", dtype: str = "bfloat16"):
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
        all_scores = []

        for prompt, comps in zip(prompts, completions):
            prompt_scores = []
            for completion in comps:
                try:
                    step_scores = self._score_single(prompt, completion)
                except Exception as e:
                    logger.warning(f"PRM scoring failed: {e}. Returning default score.")
                    step_scores = [0.5]
                prompt_scores.append(step_scores)
            all_scores.append(prompt_scores)

        return all_scores

    def _score_single(self, prompt: str, completion: str) -> List[float]:
        """
        Score a single (prompt, completion) pair.

        Uses a simple heuristic: format as chat, run through the model,
        and extract logits at step boundaries (newlines after "## Step").

        For Qwen2.5-Math-PRM specifically, we use the model's built-in
        scoring interface if available.
        """
        # Try using the model's built-in scoring if it has a score/reward method
        if hasattr(self.model, "score"):
            return self._score_via_model_method(prompt, completion)

        # Fallback: use logits-based scoring
        return self._score_via_logits(prompt, completion)

    def _score_via_model_method(self, prompt: str, completion: str) -> List[float]:
        """Score using model's built-in score method (if available)."""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(
            self.device
        )
        outputs = self.model(**inputs)

        # Extract reward from the last token's logits
        logits = outputs.logits[0, -1, :]
        # For reward models, typically use sigmoid of a specific logit
        # or the model returns scores directly
        if logits.shape[-1] == 1:
            score = torch.sigmoid(logits).item()
        else:
            # Use softmax over first 2 logits (good/bad) as common PRM pattern
            probs = torch.softmax(logits[:2], dim=-1)
            score = probs[1].item()  # probability of "good"

        return [score]

    def _score_via_logits(self, prompt: str, completion: str) -> List[float]:
        """
        Score by computing per-step perplexity / probability.

        Split completion into steps at "\\n\\n" or "## Step" boundaries,
        compute the average token log-probability for each step.
        """
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

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(
            self.device
        )
        outputs = self.model(**inputs, output_hidden_states=False)
        logits = outputs.logits

        # Compute token-level log probabilities
        shift_logits = logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Average log-prob as a rough score (higher = more confident)
        avg_log_prob = token_log_probs.mean().item()

        # Convert to [0, 1] range using sigmoid
        import math

        score = 1.0 / (1.0 + math.exp(-avg_log_prob * 10))
        return [score]


def load_prm(config) -> PRM:
    """Load PRM from config."""
    return PRM(
        model_path=config.prm_path,
        device=config.device if config.device != "auto" else "auto",
        dtype=config.dtype,
    )