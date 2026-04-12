"""
model_wrapper.py
----------------
HuggingFace-backed language model wrapper with true batched generation
and tqdm progress bars on every generation loop.

Public API
----------
  generate_greedy_batch(questions)       -> List[str]
  generate_samples_batch(questions, N)   -> List[List[str]]
  generate_greedy(question)              -> str          (single-question shortcut)
  generate_samples(question, N)          -> List[str]    (single-question shortcut)

Batching strategy
-----------------
Greedy  : 500 questions with batch_size=32 → 16 chunks of 32.
          One tqdm bar ticks once per chunk (each chunk = one generate() call).

BoN/DN  : For each question, replicate its prompt N times → batch of N.
          One tqdm bar (owned by inference_methods.py) ticks once per question.
          model_wrapper does NOT add an extra bar here.

Fixes vs previous version
--------------------------
* `torch_dtype` kwarg renamed to `dtype` (transformers ≥ 4.45 deprecation fix).
* tqdm bar added to generate_greedy_batch chunk loop so Greedy never looks stuck.
* generate_samples_batch is silent (bar lives in inference_methods.run_batch).
"""

import logging
import math
from typing import List, Optional, Dict, Any

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE    = 0.7
DEFAULT_TOP_P          = 0.95
DEFAULT_TOP_K          = 50

MATH_SYSTEM_PROMPT = (
    "You are a careful mathematics assistant. "
    "Solve the problem step by step and put your final answer in \\boxed{}."
)


class HFModelWrapper:
    """
    HuggingFace causal / seq2seq LM wrapper.

    Parameters
    ----------
    model_name_or_path : HF Hub model id or local path
    batch_size         : sequences per model.generate() call.
                         Greedy  → set to however many fit in VRAM (e.g. 32).
                         BoN/DN  → set equal to N so all N samples per question
                                   are decoded in one call (recommended).
    device             : 'auto' | 'cuda' | 'cpu' | 'cuda:0'
    dtype              : 'auto' | 'float16' | 'bfloat16' | 'float32'
    load_in_4bit       : bitsandbytes NF4 quantisation
    load_in_8bit       : bitsandbytes int8 quantisation
    max_new_tokens     : default generation length cap
    temperature        : default sampling temperature
    top_p              : nucleus sampling p
    top_k              : top-k
    system_prompt      : prepended to every query
    trust_remote_code  : forwarded to from_pretrained
    cache_dir          : HuggingFace cache directory
    """

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 1,
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
    ):
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            BitsAndBytesConfig,
        )

        self.model_name     = model_name_or_path
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p
        self.top_k          = top_k
        self.system_prompt  = system_prompt

        # ── dtype ──────────────────────────────────────────────────────────────
        if dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        # ── quantisation ───────────────────────────────────────────────────────
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

        # ── from_pretrained kwargs ─────────────────────────────────────────────
        common_kw: Dict[str, Any] = dict(
            pretrained_model_name_or_path=model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if cache_dir:
            common_kw["cache_dir"] = cache_dir

        if bnb_cfg:
            common_kw["quantization_config"] = bnb_cfg
            common_kw["device_map"] = "auto"
        elif device == "auto":
            common_kw["device_map"] = "auto"
            common_kw["dtype"] = torch_dtype   # ← 'dtype', not 'torch_dtype'
        else:
            common_kw["dtype"] = torch_dtype   # ← 'dtype', not 'torch_dtype'

        # ── tokenizer ──────────────────────────────────────────────────────────
        logger.info(f"Loading tokenizer for {model_name_or_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        # Causal-LM requires LEFT padding so real tokens are flush-right
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── model ──────────────────────────────────────────────────────────────
        logger.info(f"Loading model {model_name_or_path} ...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(**common_kw)
            self._is_seq2seq = False
        except Exception:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(**common_kw)
            self._is_seq2seq = True
            self.tokenizer.padding_side = "right"   # seq2seq needs right-padding

        if device not in ("auto",) and not bnb_cfg:
            self.model = self.model.to(device)

        self.device = next(self.model.parameters()).device
        self.model.eval()
        logger.info(f"Model loaded on {self.device}  |  batch_size={self.batch_size}")

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt building
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # Core generation  ← ONLY place model.generate() is called
    # ──────────────────────────────────────────────────────────────────────────

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
        Tokenise `prompts` into ONE left-padded batch and call model.generate()
        exactly once.  Returns decoded strings with the prompt prefix stripped.
        """
        mnt = max_new_tokens or self.max_new_tokens

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

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

    # ──────────────────────────────────────────────────────────────────────────
    # Public API – batched (primary interface)
    # ──────────────────────────────────────────────────────────────────────────

    def generate_greedy_batch(
        self,
        questions: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Greedy-decode a list of questions in chunks of self.batch_size.

        A tqdm bar ticks once per chunk so progress is always visible.

        With batch_size=32, 500 questions → 16 chunks:
          chunk 0 : questions[0:32]   → one generate() call
          chunk 1 : questions[32:64]  → one generate() call
          ...
          chunk 15: questions[480:500]→ one generate() call
        """
        prompts    = [self._build_prompt(q) for q in questions]
        bs         = self.batch_size
        n_chunks   = math.ceil(len(prompts) / bs)
        results: List[str] = []

        chunk_bar = tqdm(
            range(0, len(prompts), bs),
            total=n_chunks,
            desc=f"[Greedy] generate (bs={bs})",
            unit="chunk",
            dynamic_ncols=True,
            colour="green",
        )

        for start in chunk_bar:
            chunk = prompts[start : start + bs]
            chunk_bar.set_postfix(
                questions=f"{min(start + bs, len(prompts))}/{len(prompts)}",
                refresh=False,
            )
            results.extend(
                self._generate_batch_raw(
                    chunk, do_sample=False, max_new_tokens=max_new_tokens
                )
            )

        chunk_bar.close()
        return results

    def generate_samples_batch(
        self,
        questions: List[str],
        N: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Sample N traces for each question.

        For each question: replicate prompt N times → one batch of N →
        one model.generate() call.  Progress is tracked by the tqdm bar
        in inference_methods.run_batch(), so no bar is added here.

        Returns List[List[str]] shape [len(questions), N].
        """
        all_results: List[List[str]] = []

        for question in questions:
            prompt    = self._build_prompt(question)
            prompts_n = [prompt] * N
            decoded   = self._generate_batch_raw(
                prompts_n,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )
            all_results.append(decoded)

        return all_results

    # ──────────────────────────────────────────────────────────────────────────
    # Single-question convenience wrappers
    # ──────────────────────────────────────────────────────────────────────────

    def generate_greedy(
        self,
        question: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        return self.generate_greedy_batch(
            [question], max_new_tokens=max_new_tokens
        )[0]

    def generate_samples(
        self,
        question: str,
        N: int = 8,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        return self.generate_samples_batch(
            [question], N=N,
            temperature=temperature, top_p=top_p, top_k=top_k,
            max_new_tokens=max_new_tokens,
        )[0]