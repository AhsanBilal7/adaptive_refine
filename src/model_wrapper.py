"""
model_wrapper.py
----------------
HuggingFace-backed language model wrapper with full single/multi-GPU support.

GPU handling
------------
Single GPU  (default):
  device="auto"  →  device_map="auto"  →  model lands on cuda:0

Multiple GPUs  (automatic sharding):
  device="auto"  →  device_map="auto"  →  transformers shards layers across
  all GPUs visible in CUDA_VISIBLE_DEVICES automatically.
  No code change needed; just set CUDA_VISIBLE_DEVICES before running.

Specific GPU(s):
  device="cuda:0"   →  single GPU, index 0
  device="cuda:1"   →  single GPU, index 1

CPU:
  device="cpu"      →  runs on CPU (slow; for debugging only)

Batching design
---------------
ONE batch_size argument controls every model.generate() call.

  Greedy:
    Questions chunked into groups of batch_size.
    Each chunk → one model.generate() call.

  BoN / DN-BoN (generate_samples_flat generator):
    Flat list of Q*N prompts chunked by batch_size.
    Yields (q_idx, traces) as each question's N samples complete.
    With batch_size=64, N=8  → 8 questions saved per generate() call.
"""

import logging
import math
import os
from typing import List, Optional, Dict, Any

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE    = 0.7
DEFAULT_TOP_P          = 0.95
DEFAULT_TOP_K          = 50

MATH_SYSTEM_PROMPT = (
    "You are a careful mathematics assistant. "
    "Solve the problem step by step and put your final answer in \\boxed{}."
)


# ──────────────────────────────────────────────────────────────────────────────
# GPU introspection helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log_gpu_info():
    """Log available CUDA devices at startup."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available — running on CPU")
        return
    n = torch.cuda.device_count()
    logger.info(f"CUDA available: {n} GPU(s)")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024 ** 3
        logger.info(f"  GPU {i}: {props.name}  {total_gb:.1f} GB")


def _resolve_device_map(device: str, load_quantised: bool):
    """
    Return (device_map, target_device) for from_pretrained().

    Rules
    -----
    "auto"          →  device_map="auto"  (HF spreads across all visible GPUs)
    "cuda"          →  device_map="auto"  (same as auto)
    "cuda:N"        →  device_map={"": N}  (pin all layers to GPU N)
    "cpu"           →  device_map=None, move manually after load
    quantised       →  always device_map="auto" (bitsandbytes requirement)
    """
    if load_quantised:
        return "auto", None   # bitsandbytes requires device_map="auto"

    if device in ("auto", "cuda"):
        if torch.cuda.is_available():
            return "auto", None
        else:
            return None, torch.device("cpu")

    if device.startswith("cuda:"):
        gpu_idx = int(device.split(":")[1])
        return {"": gpu_idx}, None

    if device == "cpu":
        return None, torch.device("cpu")

    # fallback
    return "auto", None


def _primary_device(model) -> torch.device:
    """
    Return the primary device for tensor operations (tokeniser input, etc.).
    For multi-GPU models, returns the device of the first parameter.
    For pipeline-parallel models, returns the device of the embedding layer
    or the first parameter found.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper class
# ──────────────────────────────────────────────────────────────────────────────

class HFModelWrapper:
    """
    HuggingFace causal / seq2seq LM wrapper.

    Parameters
    ----------
    model_name_or_path : HF Hub model id or local path
    batch_size         : sequences per model.generate() call
    device             : "auto" | "cuda" | "cuda:N" | "cpu"
                         "auto" uses all visible GPUs via device_map="auto"
    """

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 8,
        device: str = "auto",
        dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        system_prompt: str = MATH_SYSTEM_PROMPT,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        max_input_tokens: int = 2048,
    ):
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            AutoModelForSeq2SeqLM, BitsAndBytesConfig,
        )

        self.model_name     = model_name_or_path
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p
        self.top_k          = top_k
        self.system_prompt  = system_prompt
        self.max_input_tokens = int(max_input_tokens)

        # ── Log GPU availability ───────────────────────────────────────────────
        _log_gpu_info()

        # ── dtype ─────────────────────────────────────────────────────────────
        if dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        # ── Quantisation config ───────────────────────────────────────────────
        load_quantised = load_in_4bit or load_in_8bit
        bnb_cfg = None
        if load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

        # ── Resolve device_map ────────────────────────────────────────────────
        device_map, manual_device = _resolve_device_map(device, load_quantised)
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        logger.info(
            f"Device config: device='{device}'  device_map={device_map}  "
            f"n_gpus={n_gpus}  dtype={torch_dtype}  "
            f"4bit={load_in_4bit}  8bit={load_in_8bit}"
        )

        # ── Build from_pretrained kwargs ──────────────────────────────────────
        common_kw: Dict[str, Any] = dict(
            pretrained_model_name_or_path=model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if cache_dir:
            common_kw["cache_dir"] = cache_dir
        if bnb_cfg:
            common_kw["quantization_config"] = bnb_cfg
        if device_map is not None:
            common_kw["device_map"] = device_map
        # Only set dtype when not using quantisation (bnb manages its own)
        if not bnb_cfg:
            common_kw["dtype"] = torch_dtype

        # ── Load tokeniser ────────────────────────────────────────────────────
        logger.info(f"Loading tokenizer: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── Load model ────────────────────────────────────────────────────────
        logger.info(f"Loading model: {model_name_or_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(**common_kw)
            self._is_seq2seq = False
        except Exception:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(**common_kw)
            self._is_seq2seq = True
            self.tokenizer.padding_side = "right"

        # Move to specific device if device_map was not used
        if manual_device is not None:
            self.model = self.model.to(manual_device)

        self.model.eval()

        # Primary device for tokeniser → device movement
        # For multi-GPU device_map="auto", this is typically cuda:0 (input layer)
        self.device = _primary_device(self.model)

        # ── Log final placement ───────────────────────────────────────────────
        if n_gpus > 1 and device_map == "auto":
            # Show which layers landed on which GPU
            try:
                hf_device_map = getattr(self.model, "hf_device_map", {})
                if hf_device_map:
                    devices_used = sorted(set(str(v) for v in hf_device_map.values()))
                    logger.info(
                        f"Multi-GPU: model sharded across devices {devices_used}  "
                        f"(primary={self.device})"
                    )
                else:
                    logger.info(f"Model loaded  |  primary_device={self.device}")
            except Exception:
                logger.info(f"Model loaded  |  primary_device={self.device}")
        else:
            logger.info(
                f"Model loaded on {self.device}  |  batch_size={self.batch_size}"
            )

        # ── Log per-GPU memory after load ─────────────────────────────────────
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024 ** 3
                reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                logger.info(
                    f"  GPU {i} memory: allocated={alloc:.2f} GB  "
                    f"reserved={reserved:.2f} GB"
                )

    # ── prompt building ───────────────────────────────────────────────────────

    def _build_prompt(self, question: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            messages: List[Dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": question})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        if self.system_prompt:
            return f"{self.system_prompt}\n\nProblem: {question}\nSolution:"
        return f"Problem: {question}\nSolution:"

    # ── core generation  (ONLY place model.generate() is called) ─────────────

    @torch.no_grad()
    def _generate_batch_raw(
        self,
        prompts: List[str],
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Tokenise `prompts` into one left-padded batch and call model.generate()
        exactly once.

        Multi-GPU note: enc tensors are placed on self.device (the primary /
        input device).  HF device_map handles moving intermediate activations
        across GPUs automatically during the forward pass.
        """
        mnt = max_new_tokens or self.max_new_tokens
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_input_tokens,
        ).to(self.device)      # primary device — HF moves layers internally
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=mnt,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature or self.temperature
            gen_kwargs["top_p"]       = top_p       or self.top_p
            gen_kwargs["top_k"]       = top_k       or self.top_k

        output_ids = self.model.generate(**enc, **gen_kwargs)
        if not self._is_seq2seq:
            output_ids = output_ids[:, prompt_len:]
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in output_ids
        ]

    @torch.no_grad()
    def generate_with_usage(
        self,
        prompts: List[str],
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        """Like ``_generate_batch_raw`` but also returns realized token counts.

        Returns (texts, n_input_tokens, n_output_tokens) — one entry per prompt.
        Over-long prompts are truncated from the LEFT so the generation prompt
        (the end of the chat template) is never cut.  Output tokens are counted
        up to and including the first EOS, i.e. the tokens actually decoded.
        """
        mnt = max_new_tokens or self.max_new_tokens
        old_side = getattr(self.tokenizer, "truncation_side", "right")
        self.tokenizer.truncation_side = "left"
        try:
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_input_tokens,
                add_special_tokens=False,
            ).to(self.device)
        finally:
            self.tokenizer.truncation_side = old_side
        prompt_len = enc["input_ids"].shape[1]
        n_in = [int(x) for x in enc["attention_mask"].sum(dim=1).tolist()]

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=mnt,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature or self.temperature
            gen_kwargs["top_p"]       = top_p       or self.top_p
            gen_kwargs["top_k"]       = top_k       or self.top_k
        else:
            # Silence "generation flags are not valid" warnings from sampling
            # defaults stored in generation_config.
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)

        output_ids = self.model.generate(**enc, **gen_kwargs)
        if not self._is_seq2seq:
            output_ids = output_ids[:, prompt_len:]

        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        eos_set = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
        texts, n_out = [], []
        for ids in output_ids.tolist():
            n = len(ids)
            for k, t in enumerate(ids):
                if t in eos_set:
                    n = k + 1
                    break
            n_out.append(n)
            texts.append(self.tokenizer.decode(ids[:n], skip_special_tokens=True).strip())
        return texts, n_in, n_out

    # ── public API ────────────────────────────────────────────────────────────

    def generate_greedy_batch(
        self,
        questions: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Greedy-decode questions in chunks of batch_size."""
        prompts  = [self._build_prompt(q) for q in questions]
        bs       = self.batch_size
        n_chunks = math.ceil(len(prompts) / bs)
        results: List[str] = []

        chunk_bar = tqdm(
            range(0, len(prompts), bs), total=n_chunks,
            desc=f"[Greedy] generate (bs={bs})", unit="chunk",
            dynamic_ncols=True, colour="green",
        )
        for start in chunk_bar:
            chunk = prompts[start : start + bs]
            chunk_bar.set_postfix(
                q=f"{min(start+bs, len(prompts))}/{len(prompts)}", refresh=False
            )
            results.extend(
                self._generate_batch_raw(chunk, do_sample=False,
                                         max_new_tokens=max_new_tokens)
            )
        chunk_bar.close()
        return results

    def generate_samples(
        self,
        question: str,
        N: int = 8,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate N samples for ONE question in ceil(N/batch_size) calls."""
        prompt    = self._build_prompt(question)
        samples: List[str] = []
        remaining = N
        while remaining > 0:
            this_bs = min(remaining, self.batch_size)
            decoded = self._generate_batch_raw(
                [prompt] * this_bs,
                do_sample=True,
                temperature=temperature, top_p=top_p, top_k=top_k,
                max_new_tokens=max_new_tokens,
            )
            samples.extend(decoded)
            remaining -= this_bs
        return samples

    def generate_samples_flat(
        self,
        questions: List[str],
        N: int = 8,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        """
        Generator: yield (q_idx, traces) as each question's N samples become
        available after each chunk of batch_size sequences completes.

        With batch_size=64, N=8  → 8 questions yielded per generate() call.
        With batch_size=128, N=8 → 16 questions yielded per call.

        Works identically on single GPU and multi-GPU — HF device_map handles
        layer placement; the input tensor goes to self.device (primary).
        """
        prompts_flat: List[str] = []
        for q in questions:
            p = self._build_prompt(q)
            prompts_flat.extend([p] * N)

        total    = len(prompts_flat)
        bs       = self.batch_size
        n_chunks = math.ceil(total / bs)

        decoded_flat: List[str] = []
        next_q_to_yield = 0

        chunk_bar = tqdm(
            range(0, total, bs), total=n_chunks,
            desc=f"[generate] flat  bs={bs}  Q={len(questions)}  N={N}",
            unit="chunk", dynamic_ncols=True, colour="blue", leave=True,
        )

        for start in chunk_bar:
            chunk = prompts_flat[start : start + bs]
            decoded_flat.extend(
                self._generate_batch_raw(
                    chunk, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    top_k=top_k, max_new_tokens=max_new_tokens,
                )
            )
            n_complete = len(decoded_flat) // N
            while next_q_to_yield < n_complete:
                q_idx  = next_q_to_yield
                traces = decoded_flat[q_idx * N : (q_idx + 1) * N]
                chunk_bar.set_postfix(
                    saved=f"{q_idx+1}/{len(questions)}",
                    seqs=f"{len(decoded_flat)}/{total}",
                    refresh=False,
                )
                yield q_idx, traces
                next_q_to_yield += 1

        chunk_bar.close()

    def generate_greedy(self, question: str, max_new_tokens: Optional[int] = None) -> str:
        return self.generate_greedy_batch([question], max_new_tokens=max_new_tokens)[0]

    # ── sequence log-probability scoring (used by GreedyPassN) ───────────────

    @torch.no_grad()
    def score_sequences_logprob(
        self,
        prompt: str,
        completions: List[str],
    ) -> List[float]:
        """
        Mean per-token log-probability of each completion given the prompt.
        Scored in batches of batch_size — one forward pass per batch, no generation.
        Works on single and multi-GPU (input placed on self.device).
        """
        scores: List[float] = []
        enc_prompt = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        prompt_len = enc_prompt["input_ids"].shape[1]

        for i in range(0, len(completions), self.batch_size):
            batch      = completions[i : i + self.batch_size]
            full_texts = [prompt + c for c in batch]
            enc = self.tokenizer(
                full_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=4096,
            ).to(self.device)

            outputs   = self.model(**enc)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

            for j in range(len(batch)):
                ids        = enc["input_ids"][j]
                mask       = enc["attention_mask"][j]
                seq_len    = int(mask.sum().item())
                real_start = len(ids) - seq_len
                comp_start = real_start + prompt_len
                comp_end   = len(ids)
                if comp_start >= comp_end:
                    scores.append(-1e9)
                    continue
                token_ids = ids[comp_start:comp_end]
                lp        = log_probs[j, comp_start - 1 : comp_end - 1, :]
                token_lp  = lp.gather(1, token_ids.unsqueeze(1)).squeeze(1)
                scores.append(token_lp.mean().item())

        return scores