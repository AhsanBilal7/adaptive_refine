"""
backends.py
===========
Capped role-call wrapper (§9 module 1).

Every model call made by the procedure goes through ``GenerationBackend.generate``
which returns the realized input/output token counts so the budget meter can
charge actual compute (§4.3: "Meter actual input/output tokens and estimated
model FLOPs, including longer contexts").

Backends
--------
  hf    — wraps ``src.model_wrapper.HFModelWrapper`` (single model.generate per
          chunk; one global torch RNG, so per-request seeds are best effort).
  vllm  — optional; per-request seeds are honoured.  The vLLM wheel pinned in
          uv.lock needs glibc ≥ 2.31, so it is not installable on RHEL8 nodes.
  fake  — deterministic scripted backend used by the unit tests.

All roles (generator / critic / corrector / editor) use the SAME frozen model.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GenRequest:
    role: str                    # init | restart | rewrite | critique | correct | editor
    system: str
    user: str
    max_new_tokens: int
    temperature: float = 0.0     # 0 → greedy
    top_p: float = 0.95
    seed: Optional[int] = None
    prompt: str = ""             # filled by backend.format()

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0

    def sampling_key(self) -> Tuple:
        return (self.max_new_tokens, round(self.temperature, 4), round(self.top_p, 4))


@dataclass
class GenResult:
    text: str
    n_in: int
    n_out: int
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())


class GenerationBackend:
    """Interface. Subclasses implement ``_generate_group`` and ``count_tokens``."""

    name = "base"
    num_params: float = 0.0
    model_id: str = ""
    max_input_tokens: int = 4096

    def format(self, system: str, user: str) -> str:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def prompt_tokens(self, system: str, user: str) -> int:
        return min(self.count_tokens(self.format(system, user)), self.max_input_tokens)

    def generate(self, requests: List[GenRequest]) -> List[GenResult]:
        """Batch requests grouped by sampling parameters; order preserved."""
        for r in requests:
            if not r.prompt:
                r.prompt = self.format(r.system, r.user)
        results: List[Optional[GenResult]] = [None] * len(requests)
        groups: Dict[Tuple, List[int]] = {}
        for i, r in enumerate(requests):
            groups.setdefault(r.sampling_key(), []).append(i)
        for _, idxs in groups.items():
            outs = self._generate_group([requests[i] for i in idxs])
            for i, o in zip(idxs, outs):
                results[i] = o
        return [r if r is not None else GenResult("", 0, 0, "missing") for r in results]

    def _generate_group(self, reqs: List[GenRequest]) -> List[GenResult]:
        raise NotImplementedError

    def model_hash(self) -> str:
        return hashlib.sha256(self.model_id.encode()).hexdigest()[:12]


def _chat_format(tokenizer, system: str, user: str) -> str:
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Some templates (e.g. older Mistral) reject a system turn.
            msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}"


# =============================================================================
# HuggingFace backend
# =============================================================================

class HFBackend(GenerationBackend):
    name = "hf"

    def __init__(self, cfg):
        from src.model_wrapper import HFModelWrapper  # type: ignore

        self.wrapper = HFModelWrapper(
            model_name_or_path=cfg.model_name,
            batch_size=cfg.batch_size,
            device=cfg.device,
            dtype=cfg.dtype,
            max_new_tokens=cfg.init_max_new_tokens,
            temperature=cfg.init_temperature,
            top_p=cfg.top_p,
            trust_remote_code=cfg.trust_remote_code,
            cache_dir=cfg.hf_cache_dir or None,
            max_input_tokens=cfg.max_input_tokens,
        )
        self.tokenizer = self.wrapper.tokenizer
        self.max_input_tokens = int(cfg.max_input_tokens)
        self.model_id = cfg.model_name + "@" + str(
            getattr(getattr(self.wrapper.model, "config", None), "_commit_hash", "") or "local")
        self.num_params = float(cfg.model_num_params) or float(
            sum(p.numel() for p in self.wrapper.model.parameters()))
        self._base_seed = int(cfg.seed)
        self._n_batches = 0

    def format(self, system: str, user: str) -> str:
        return _chat_format(self.tokenizer, system, user)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _generate_group(self, reqs: List[GenRequest]) -> List[GenResult]:
        import torch

        out: List[GenResult] = []
        bs = max(1, int(self.wrapper.batch_size))
        r0 = reqs[0]
        for start in range(0, len(reqs), bs):
            chunk = reqs[start:start + bs]
            # HF has one global RNG; derive a reproducible seed from the chunk.
            seeds = [c.seed for c in chunk if c.seed is not None]
            if seeds and r0.do_sample:
                torch.manual_seed((sum(seeds) + self._base_seed) % (2 ** 31))
            try:
                texts, n_in, n_out = self.wrapper.generate_with_usage(
                    [c.prompt for c in chunk],
                    do_sample=r0.do_sample,
                    temperature=r0.temperature if r0.do_sample else None,
                    top_p=r0.top_p if r0.do_sample else None,
                    max_new_tokens=r0.max_new_tokens,
                )
                out.extend(GenResult(t, a, b) for t, a, b in zip(texts, n_in, n_out))
            except Exception as e:  # charge the prompt even when generation fails
                logger.error(f"[HFBackend] generation failed: {e}")
                for c in chunk:
                    out.append(GenResult("", self.prompt_tokens(c.system, c.user), 0, str(e)))
            self._n_batches += 1
        return out


# =============================================================================
# vLLM backend (optional)
# =============================================================================

class VLLMBackend(GenerationBackend):
    name = "vllm"

    def __init__(self, cfg):
        try:
            from vllm import LLM  # type: ignore
        except Exception as e:  # pragma: no cover - depends on host
            raise RuntimeError(
                "backend=vllm requested but vllm is not importable. The vllm wheel in "
                "uv.lock needs glibc>=2.31; use backend=hf on RHEL8 nodes.") from e
        self.llm = LLM(
            model=cfg.model_name,
            dtype=cfg.dtype if cfg.dtype != "auto" else "auto",
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            max_model_len=cfg.vllm_max_model_len,
            seed=cfg.seed,
            trust_remote_code=cfg.trust_remote_code,
            download_dir=cfg.hf_cache_dir or None,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_input_tokens = int(cfg.max_input_tokens)
        self.model_id = cfg.model_name
        self.num_params = float(cfg.model_num_params) or _estimate_params_from_config(cfg)

    def format(self, system: str, user: str) -> str:
        return _chat_format(self.tokenizer, system, user)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _generate_group(self, reqs: List[GenRequest]) -> List[GenResult]:
        from vllm import SamplingParams  # type: ignore

        prompts, params = [], []
        for r in reqs:
            ids = self.tokenizer(r.prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > self.max_input_tokens:
                ids = ids[-self.max_input_tokens:]
            prompts.append({"prompt_token_ids": ids})
            params.append(SamplingParams(
                n=1, max_tokens=r.max_new_tokens,
                temperature=r.temperature if r.do_sample else 0.0,
                top_p=r.top_p if r.do_sample else 1.0,
                seed=r.seed if r.do_sample else None,
            ))
        try:
            outs = self.llm.generate(prompts, params, use_tqdm=False)
        except Exception as e:
            logger.error(f"[VLLMBackend] generation failed: {e}")
            return [GenResult("", len(p["prompt_token_ids"]), 0, str(e)) for p in prompts]
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(GenResult(c.text.strip(), len(o.prompt_token_ids), len(c.token_ids)))
        return res


def _estimate_params_from_config(cfg) -> float:
    """Rough dense-decoder parameter count from the HF config (vLLM path only)."""
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(cfg.model_name, cache_dir=cfg.hf_cache_dir or None,
                                       trust_remote_code=cfg.trust_remote_code)
        d = c.hidden_size; L = c.num_hidden_layers; V = c.vocab_size
        ff = getattr(c, "intermediate_size", 4 * d)
        kv = getattr(c, "num_key_value_heads", c.num_attention_heads) / c.num_attention_heads
        per_layer = d * d * (2 + 2 * kv) + 3 * d * ff
        emb = V * d * (1 if getattr(c, "tie_word_embeddings", False) else 2)
        return float(L * per_layer + emb)
    except Exception:
        logger.warning("Could not estimate parameter count; set RSI_MODEL_NUM_PARAMS.")
        return 0.0


# =============================================================================
# Fake backend (tests / dry runs)
# =============================================================================

class FakeBackend(GenerationBackend):
    """Scripted backend: ``responder(request, call_index) -> text``.

    Token counts are whitespace-token lengths, so tests can reason about cost.
    """

    name = "fake"

    def __init__(self, responder: Callable[[GenRequest, int], str], num_params: float = 1e9,
                 max_input_tokens: int = 100000):
        self.responder = responder
        self.num_params = float(num_params)
        self.model_id = "fake-model"
        self.max_input_tokens = max_input_tokens
        self.n_calls = 0
        self.batch_sizes: List[int] = []

    def format(self, system: str, user: str) -> str:
        return f"<sys>{system}</sys><user>{user}</user>"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _generate_group(self, reqs: List[GenRequest]) -> List[GenResult]:
        self.batch_sizes.append(len(reqs))
        out = []
        for r in reqs:
            self.n_calls += 1
            text = self.responder(r, self.n_calls)
            words = text.split()[: r.max_new_tokens]
            out.append(GenResult(" ".join(words), min(self.count_tokens(r.prompt), self.max_input_tokens),
                                 len(words)))
        return out


def build_backend(cfg) -> GenerationBackend:
    if cfg.backend == "hf":
        return HFBackend(cfg)
    if cfg.backend == "vllm":
        return VLLMBackend(cfg)
    raise ValueError(f"backend {cfg.backend!r} cannot be built from config (fake is test-only)")


__all__ = [
    "GenRequest", "GenResult", "GenerationBackend",
    "HFBackend", "VLLMBackend", "FakeBackend", "build_backend",
]
