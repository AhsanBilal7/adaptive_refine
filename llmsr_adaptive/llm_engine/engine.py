"""AdaptSci / LLM-SR model server.

Supported model families (auto-detected from --model_path)
──────────────────────────────────────────────────────────
  Qwen2.5 / Qwen2 / Qwen1.5  –  Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct,
                                  Qwen/Qwen2.5-32B-Instruct, Qwen/Qwen2-7B-Instruct, …
  Llama-3 / Llama-3.1 / 3.2  –  meta-llama/Meta-Llama-3-8B-Instruct,
                                  meta-llama/Meta-Llama-3.1-70B-Instruct, …
  Llama-2                     –  meta-llama/Llama-2-7b-chat-hf, …
  DeepSeek-R1 / DeepSeek-V2   –  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,
                                  deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct, …
  Mistral / Mixtral            –  mistralai/Mistral-7B-Instruct-v0.3,
                                  mistralai/Mixtral-8x7B-Instruct-v0.1, …
  Phi-3 / Phi-3.5              –  microsoft/Phi-3-mini-4k-instruct,
                                  microsoft/Phi-3.5-mini-instruct, …
  Gemma-2 / Gemma              –  google/gemma-2-9b-it, google/gemma-7b-it, …
  CodeLlama                    –  codellama/CodeLlama-13b-Instruct-hf, …
  StarCoder2                   –  bigcode/starcoder2-15b-instruct-v0.1, …
  Yi-1.5                       –  01-ai/Yi-1.5-9B-Chat, …
  InternLM-2.5                 –  internlm/internlm2_5-7b-chat, …

Each family has a ModelProfile that defines:
  • chat_template_style   – how to format the user turn
  • default EOS/PAD token ids  (looked up from tokenizer when possible)
  • default generation hyperparameters tuned for code generation
  • whether the model needs trust_remote_code=True
  • vLLM compatibility flag (for future vLLM backend)

The Flask server exposes a single POST /completions endpoint with the same
JSON interface used by the original LLM-SR sampler so no client changes are
needed when switching models.

Usage
─────
  python engine.py --model_path Qwen/Qwen2.5-7B-Instruct \
                   --gpu_ids 0 --port 5000 --quantization

  python engine.py --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
                   --gpu_ids 0 1 --port 5000 --quantization

  python engine.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                   --gpu_ids 0 --port 5000 --quantization --quant_bits 4

  python engine.py --model_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
                   --gpu_ids 0 1 2 3 --port 5000 --quantization
"""
from __future__ import annotations

import gc
import logging
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model profiles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Per-family generation defaults and chat formatting rules.

    Attributes
    ──────────
    family            Human-readable family name (for logging)
    chat_style        How the user turn is formatted before tokenisation.
                      'chat_template' → use tokenizer.apply_chat_template()
                      'alpaca'        → ### Instruction: / ### Response:
                      'chatml'        → <|im_start|>user\n…<|im_end|>
                      'llama2'        → [INST] … [/INST]
                      'deepseek'      → User: … \nAssistant:
    eos_token_ids     Default EOS token ids (overridden by tokenizer lookup).
    pad_token_id      Default PAD token id.
    temperature       Default sampling temperature.
    top_p             Default nucleus sampling p.
    top_k             Default top-k value.
    max_new_tokens    Default max tokens to generate.
    trust_remote_code Whether to pass trust_remote_code=True to from_pretrained.
    attn_impl         Optional attention implementation ('flash_attention_2', …).
    extra_model_kwargs Additional kwargs forwarded to from_pretrained.
    """
    family: str
    chat_style: str = 'chat_template'
    eos_token_ids: List[int] = field(default_factory=lambda: [2])
    pad_token_id: int = 2
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 512
    trust_remote_code: bool = False
    attn_impl: Optional[str] = None
    extra_model_kwargs: Dict = field(default_factory=dict)


# Registry: (substring_in_lower_model_path) → ModelProfile
# Checked in order; first match wins.
_MODEL_PROFILES: List[tuple[str, ModelProfile]] = [

    # ── Qwen 2.5 (instruction-tuned) ──────────────────────────────────────
    ('qwen2.5', ModelProfile(
        family='Qwen2.5',
        chat_style='chat_template',
        eos_token_ids=[151645, 151643],   # <|im_end|>, <|endoftext|>
        pad_token_id=151643,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_new_tokens=512,
        trust_remote_code=False,
    )),

    # ── Qwen 2 ────────────────────────────────────────────────────────────
    ('qwen2', ModelProfile(
        family='Qwen2',
        chat_style='chat_template',
        eos_token_ids=[151645, 151643],
        pad_token_id=151643,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_new_tokens=512,
        trust_remote_code=False,
    )),

    # ── Qwen 1.5 ──────────────────────────────────────────────────────────
    ('qwen1.5', ModelProfile(
        family='Qwen1.5',
        chat_style='chat_template',
        eos_token_ids=[151645, 151643],
        pad_token_id=151643,
        temperature=0.7,
        top_p=0.9,
        top_k=30,
        max_new_tokens=512,
        trust_remote_code=True,
    )),

    # ── Qwen (generic fallback) ───────────────────────────────────────────
    ('qwen', ModelProfile(
        family='Qwen',
        chat_style='chatml',
        eos_token_ids=[151645],
        pad_token_id=151643,
        temperature=0.7,
        top_p=0.9,
        top_k=30,
        max_new_tokens=512,
        trust_remote_code=True,
    )),

    # ── DeepSeek-R1 distilled on Qwen/Llama ──────────────────────────────
    ('deepseek-r1', ModelProfile(
        family='DeepSeek-R1',
        chat_style='chat_template',
        eos_token_ids=[151645, 151643, 32021, 2],
        pad_token_id=151643,
        temperature=0.6,
        top_p=0.9,
        top_k=50,
        max_new_tokens=1024,   # R1 tends to produce chain-of-thought
        trust_remote_code=False,
    )),

    # ── DeepSeek-Coder-V2 ─────────────────────────────────────────────────
    ('deepseek-coder-v2', ModelProfile(
        family='DeepSeek-Coder-V2',
        chat_style='chat_template',
        eos_token_ids=[100001, 2],
        pad_token_id=100001,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=True,
    )),

    # ── DeepSeek-V2 ───────────────────────────────────────────────────────
    ('deepseek-v2', ModelProfile(
        family='DeepSeek-V2',
        chat_style='deepseek',
        eos_token_ids=[100001, 2],
        pad_token_id=100001,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=True,
    )),

    # ── DeepSeek-Coder ────────────────────────────────────────────────────
    ('deepseek-coder', ModelProfile(
        family='DeepSeek-Coder',
        chat_style='deepseek',
        eos_token_ids=[32021, 2],
        pad_token_id=32021,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=True,
    )),

    # ── DeepSeek (generic) ────────────────────────────────────────────────
    ('deepseek', ModelProfile(
        family='DeepSeek',
        chat_style='deepseek',
        eos_token_ids=[32021, 2],
        pad_token_id=32021,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=True,
    )),

    # ── Llama 3.1 / 3.2 / 3.3 ────────────────────────────────────────────
    ('llama-3', ModelProfile(
        family='Llama-3',
        chat_style='chat_template',
        eos_token_ids=[128009, 128001, 2],  # <|eot_id|>, <|end_of_text|>
        pad_token_id=128009,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=False,
    )),

    # ── Llama 2 ───────────────────────────────────────────────────────────
    ('llama-2', ModelProfile(
        family='Llama-2',
        chat_style='llama2',
        eos_token_ids=[2],
        pad_token_id=2,
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        max_new_tokens=512,
        trust_remote_code=False,
    )),

    # ── CodeLlama ─────────────────────────────────────────────────────────
    ('codellama', ModelProfile(
        family='CodeLlama',
        chat_style='llama2',
        eos_token_ids=[2],
        pad_token_id=2,
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        max_new_tokens=512,
        trust_remote_code=False,
    )),

    # ── Mixtral ───────────────────────────────────────────────────────────
    ('mixtral', ModelProfile(
        family='Mixtral',
        chat_style='chat_template',
        eos_token_ids=[32021, 2],
        pad_token_id=32021,
        temperature=0.8,
        top_p=0.9,
        top_k=30,
        max_new_tokens=512,
        trust_remote_code=False,
    )),

    # ── Mistral ───────────────────────────────────────────────────────────
    ('mistral', ModelProfile(
        family='Mistral',
        chat_style='chat_template',
        eos_token_ids=[2],
        pad_token_id=2,
        temperature=0.8,
        top_p=0.9,
        top_k=30,
        max_new_tokens=512,
        trust_remote_code=False,
    )),

    # ── Phi-3.5 ───────────────────────────────────────────────────────────
    ('phi-3.5', ModelProfile(
        family='Phi-3.5',
        chat_style='chat_template',
        eos_token_ids=[32007, 32000, 32001],  # <|end|>, <|endoftext|>
        pad_token_id=32000,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=True,
    )),

    # ── Phi-3 ─────────────────────────────────────────────────────────────
    ('phi-3', ModelProfile(
        family='Phi-3',
        chat_style='chat_template',
        eos_token_ids=[32007, 32000, 32001],
        pad_token_id=32000,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=True,
    )),

    # ── Gemma-2 ───────────────────────────────────────────────────────────
    ('gemma-2', ModelProfile(
        family='Gemma-2',
        chat_style='chat_template',
        eos_token_ids=[1, 107],   # <eos>, <end_of_turn>
        pad_token_id=0,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=False,
        attn_impl='eager',        # flash-attn not always available for Gemma-2
    )),

    # ── Gemma ─────────────────────────────────────────────────────────────
    ('gemma', ModelProfile(
        family='Gemma',
        chat_style='chat_template',
        eos_token_ids=[1],
        pad_token_id=0,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=False,
    )),

    # ── StarCoder2 ────────────────────────────────────────────────────────
    ('starcoder2', ModelProfile(
        family='StarCoder2',
        chat_style='alpaca',
        eos_token_ids=[0],
        pad_token_id=0,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=False,
    )),

    # ── Yi-1.5 ────────────────────────────────────────────────────────────
    ('yi-1.5', ModelProfile(
        family='Yi-1.5',
        chat_style='chat_template',
        eos_token_ids=[7],
        pad_token_id=0,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=False,
    )),

    # ── InternLM-2.5 ──────────────────────────────────────────────────────
    ('internlm2', ModelProfile(
        family='InternLM2',
        chat_style='chat_template',
        eos_token_ids=[2, 92542],
        pad_token_id=2,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        trust_remote_code=True,
    )),
]

# Fallback for unknown models
_DEFAULT_PROFILE = ModelProfile(
    family='Unknown',
    chat_style='chat_template',
    eos_token_ids=[2],
    pad_token_id=2,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    max_new_tokens=512,
    trust_remote_code=False,
)


def get_model_profile(model_path: str) -> ModelProfile:
    """Return the matching ModelProfile for the given model path."""
    lower = model_path.lower()
    for key, profile in _MODEL_PROFILES:
        if key in lower:
            logger.info('Matched model family: %s (key=%r)', profile.family, key)
            return profile
    logger.warning(
        'No profile matched for %r – using default profile. '
        'Generation quality may be suboptimal.', model_path)
    return _DEFAULT_PROFILE


# ─────────────────────────────────────────────────────────────────────────────
# Chat formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_prompt(
    user_message: str,
    tokenizer,
    profile: ModelProfile,
    system_message: str = '',
) -> str | list:
    """Format `user_message` according to `profile.chat_style`.

    Returns either a pre-formatted string (for tokenizer.encode) or a list of
    message dicts (for tokenizer.apply_chat_template).
    """
    style = profile.chat_style

    if style == 'chat_template':
        # Preferred: let the tokenizer handle everything correctly
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': user_message})
        return messages   # caller uses apply_chat_template

    if style == 'chatml':
        sys_block = f'<|im_start|>system\n{system_message}<|im_end|>\n' if system_message else ''
        return (f'{sys_block}'
                f'<|im_start|>user\n{user_message}<|im_end|>\n'
                f'<|im_start|>assistant\n')

    if style == 'llama2':
        sys_block = f'<<SYS>>\n{system_message}\n<</SYS>>\n\n' if system_message else ''
        return f'<s>[INST] {sys_block}{user_message} [/INST]'

    if style == 'deepseek':
        sys_block = f'{system_message}\n\n' if system_message else ''
        return f'{sys_block}User: {user_message}\n\nAssistant:'

    if style == 'alpaca':
        sys_block = f'{system_message}\n\n' if system_message else ''
        return (f'{sys_block}'
                f'### Instruction:\n{user_message}\n\n'
                f'### Response:\n')

    # fallback
    return user_message


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

parser = ArgumentParser(description='AdaptSci / LLM-SR Model Server')

parser.add_argument('--model_path', type=str,
                    default='mistralai/Mixtral-8x7B-Instruct-v0.1',
                    help='HuggingFace model path or local directory.')
parser.add_argument('--gpu_ids', nargs='+', default=['0'],
                    help='GPU device IDs to use, e.g. 0 1 2 3')
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=5000)

# ── Quantisation ──────────────────────────────────────────────────────────────
parser.add_argument('--quantization', default=False, action='store_true',
                    help='Enable BitsAndBytes quantization.')
parser.add_argument('--quant_bits', type=int, default=4, choices=[4, 8],
                    help='Quantization precision: 4 or 8 bit (default: 4).')

# ── Generation overrides (all optional; profile defaults used otherwise) ──────
parser.add_argument('--temperature',        type=float, default=None)
parser.add_argument('--top_p',              type=float, default=None)
parser.add_argument('--top_k',              type=int,   default=None)
parser.add_argument('--max_new_tokens',     type=int,   default=None)
parser.add_argument('--do_sample',          type=bool,  default=True)
parser.add_argument('--num_return_sequences', type=int, default=1)

# ── Advanced ──────────────────────────────────────────────────────────────────
parser.add_argument('--dtype', type=str, default='auto',
                    choices=['auto', 'float16', 'bfloat16', 'float32'],
                    help='Model weight dtype (ignored when quantization is on).')
parser.add_argument('--flash_attn', default=False, action='store_true',
                    help='Use flash_attention_2 (requires flash-attn package).')
parser.add_argument('--system_prompt', type=str, default='',
                    help='Optional system prompt to prepend to every user message.')
parser.add_argument('--max_repeat_prompt', type=int, default=10,
                    help='Max batch size for repeated prompts.')

args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# GPU / device setup
# ─────────────────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    gpu_ids = args.gpu_ids
    device  = torch.device(f'cuda:{gpu_ids[0]}')
    torch.cuda.set_device(device)
    logger.info('Using GPU(s): %s', gpu_ids)
else:
    device  = torch.device('cpu')
    gpu_ids = []
    logger.info('CUDA not available – running on CPU.')


# ─────────────────────────────────────────────────────────────────────────────
# Model / tokenizer loading
# ─────────────────────────────────────────────────────────────────────────────

profile = get_model_profile(args.model_path)
logger.info('Model family: %s | chat_style: %s', profile.family, profile.chat_style)

# ── Quantisation config ───────────────────────────────────────────────────────
if args.quantization:
    if args.quant_bits == 4:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
        logger.info('4-bit NF4 quantization enabled.')
    else:
        quant_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        logger.info('8-bit quantization enabled.')
else:
    quant_cfg = None

# ── Dtype ─────────────────────────────────────────────────────────────────────
if quant_cfg is not None:
    torch_dtype = None   # BitsAndBytes manages dtype internally
elif args.dtype == 'float16':
    torch_dtype = torch.float16
elif args.dtype == 'bfloat16':
    torch_dtype = torch.bfloat16
elif args.dtype == 'float32':
    torch_dtype = torch.float32
else:
    torch_dtype = 'auto'

# ── Attention implementation ──────────────────────────────────────────────────
attn_impl = None
if args.flash_attn:
    attn_impl = 'flash_attention_2'
elif profile.attn_impl:
    attn_impl = profile.attn_impl

# ── Extra kwargs from profile ─────────────────────────────────────────────────
extra_kwargs = dict(profile.extra_model_kwargs)
if attn_impl:
    extra_kwargs['attn_implementation'] = attn_impl
if torch_dtype is not None:
    extra_kwargs['torch_dtype'] = torch_dtype

# ── Load tokenizer ────────────────────────────────────────────────────────────
logger.info('Loading tokenizer from %s …', args.model_path)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    trust_remote_code=profile.trust_remote_code,
    padding_side='left',
)
# Ensure pad token is set
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ── Resolve EOS / PAD ids from tokenizer (more reliable than hardcoded) ───────
def _resolve_token_id(name: str, fallback: int) -> int:
    """Try to get token id by special-token name; fall back to profile default."""
    tok_id = getattr(tokenizer, f'{name}_token_id', None)
    return int(tok_id) if tok_id is not None else fallback

resolved_eos_ids  = profile.eos_token_ids   # keep profile list as primary
resolved_pad_id   = _resolve_token_id('pad', profile.pad_token_id)

# ── Load model ────────────────────────────────────────────────────────────────
logger.info('Loading model from %s …', args.model_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    quantization_config=quant_cfg,
    device_map='auto',
    trust_remote_code=profile.trust_remote_code,
    **extra_kwargs,
)
model.eval()
logger.info('Model loaded. Parameters: ~%dM',
            sum(p.numel() for p in model.parameters()) // 1_000_000)

# ── Resolve effective generation defaults ─────────────────────────────────────
eff_temperature    = args.temperature    if args.temperature    is not None else profile.temperature
eff_top_p          = args.top_p          if args.top_p          is not None else profile.top_p
eff_top_k          = args.top_k          if args.top_k          is not None else profile.top_k
eff_max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else profile.max_new_tokens

logger.info(
    'Generation defaults: temp=%.2f top_p=%.2f top_k=%d max_new_tokens=%d',
    eff_temperature, eff_top_p, eff_top_k, eff_max_new_tokens)
logger.info('EOS ids: %s  PAD id: %d', resolved_eos_ids, resolved_pad_id)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation helper
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (args.system_prompt or
    'You are a helpful assistant tasked with discovering mathematical function '
    'structures for scientific systems. Complete the equation function below, '
    'considering the physical meaning and relationships of inputs. '
    'Explain your reasoning briefly before writing the code.')


def _tokenise_prompt(user_message: str, repeat: int = 1) -> torch.Tensor:
    """Return a batched input tensor (repeat × seq_len)."""
    formatted = format_prompt(user_message, tokenizer, profile,
                              system_message=SYSTEM_PROMPT)

    if isinstance(formatted, list):
        # chat_template path
        ids = tokenizer.apply_chat_template(
            formatted,
            add_generation_prompt=True,
            return_tensors='pt',
        )
    else:
        # raw string path (llama2, deepseek, alpaca, chatml manual)
        ids = tokenizer.encode(
            formatted,
            return_tensors='pt',
            add_special_tokens=False,
        )

    # Repeat for batch inference
    ids = torch.vstack([ids] * repeat)
    return ids.to(model.device)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing: strip reasoning blocks (DeepSeek-R1 <think>…</think>)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

_THINK_RE = _re.compile(r'<think>.*?</think>', _re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Remove <think>…</think> chain-of-thought blocks (DeepSeek-R1)."""
    return _THINK_RE.sub('', text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Flask server
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.route('/completions', methods=['POST'])
def completions():
    """Main completion endpoint – identical interface to original LLM-SR engine.

    Request JSON fields
    ───────────────────
    prompt          (str)  – user message / prompt body
    repeat_prompt   (int)  – batch size, default 1
    params          (dict) – optional generation parameter overrides:
                             temperature, top_p, top_k, max_new_tokens,
                             do_sample, num_return_sequences,
                             eos_token_id (int or list), pad_token_id

    Response JSON fields
    ────────────────────
    content  (list[str]) – generated text for each batch item
    model    (str)       – model path
    family   (str)       – detected model family
    """
    body = request.json
    user_prompt  = body.get('prompt', '')
    repeat_prompt = min(int(body.get('repeat_prompt', 1)), args.max_repeat_prompt)

    # ── Parse per-request generation overrides ────────────────────────────
    params = body.get('params', {}) or {}

    temperature        = params.get('temperature')    or eff_temperature
    top_p              = params.get('top_p')          or eff_top_p
    top_k              = params.get('top_k')          or eff_top_k
    max_new_tokens     = params.get('max_new_tokens') or eff_max_new_tokens
    do_sample          = params.get('do_sample', args.do_sample)
    num_return_seqs    = params.get('num_return_sequences', args.num_return_sequences)
    skip_special_tokens = params.get('skip_special_tokens', True)

    # EOS / PAD: accept single int or list from caller
    req_eos = params.get('eos_token_id', resolved_eos_ids)
    if isinstance(req_eos, int):
        req_eos = [req_eos]
    req_pad = int(params.get('pad_token_id', resolved_pad_id))

    # ── Retry loop (handles OOM) ──────────────────────────────────────────
    while True:
        try:
            input_ids = _tokenise_prompt(user_prompt, repeat=repeat_prompt)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_return_seqs,
                    eos_token_id=req_eos,
                    pad_token_id=req_pad,
                )

            # Decode only the newly generated tokens
            generated = []
            for i, out in enumerate(output_ids):
                input_len = input_ids.shape[1]
                new_tokens = out[input_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
                text = _strip_reasoning(text)
                generated.append(text)

            # VRAM cleanup
            del input_ids, output_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return jsonify({
                'content': generated,
                'model':   args.model_path,
                'family':  profile.family,
            })

        except torch.cuda.OutOfMemoryError:
            logger.warning('OOM – clearing cache and retrying …')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        except Exception as exc:
            logger.error('Generation error: %s', exc, exc_info=True)
            return jsonify({'error': str(exc), 'content': []}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model': args.model_path, 'family': profile.family})


@app.route('/model_info', methods=['GET'])
def model_info():
    """Return model metadata (useful for debugging)."""
    return jsonify({
        'model_path':    args.model_path,
        'family':        profile.family,
        'chat_style':    profile.chat_style,
        'eos_token_ids': resolved_eos_ids,
        'pad_token_id':  resolved_pad_id,
        'temperature':   eff_temperature,
        'top_p':         eff_top_p,
        'top_k':         eff_top_k,
        'max_new_tokens': eff_max_new_tokens,
        'quantization':  args.quantization,
        'quant_bits':    args.quant_bits if args.quantization else None,
        'gpu_ids':       gpu_ids,
    })


if __name__ == '__main__':
    logger.info('Starting server on %s:%d', args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=False)
