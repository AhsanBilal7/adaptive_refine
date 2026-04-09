"""AdaptSci sampler – model-aware, all backends, AdaptSci kwargs forwarding.

LLM backends
────────────
  LocalLLM      – local Flask engine.py server (any HuggingFace model)
  OpenAILLM     – OpenAI Chat Completions (gpt-3.5-turbo, gpt-4o, …)
  AnthropicLLM  – Anthropic Claude (claude-3-5-sonnet-20241022, …)
  vLLMClient    – vLLM OpenAI-compatible server (any HF model via vLLM)

AdaptSci extensions
────────────────────
  • Sampler passes adaptive_scheduler to buffer.get_prompt() so island
    selection uses adaptive exploit/explore weights.
  • hessian_cfg, adaptive_scheduler, ablation_mode are forwarded to every
    evaluator.analyse() call.
  • Per-family instruction prompts tuned for scientific code generation.
  • Body extraction handles Markdown fences and DeepSeek-R1 <think> blocks.
  • Exponential back-off retry on all backends.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Collection, List, Optional, Sequence, Type

import numpy as np
import requests

from adaptsci import config as config_lib

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-family instruction prompts
# ─────────────────────────────────────────────────────────────────────────────

_INSTRUCTION_PROMPTS: dict = {
    'qwen': (
        'You are a scientific AI assistant with expertise in mathematical modeling. '
        'Discover the mathematical function structure that best describes the given '
        'scientific system. Complete the equation function below using your knowledge '
        'of physics, biology, or materials science. Think step-by-step about the '
        'physical meaning of each variable before writing code.\n\n'
    ),
    'deepseek': (
        'You are an expert in scientific equation discovery and symbolic regression. '
        'Complete the equation function below to find a mathematical model that fits '
        'the data. Consider physical relationships between variables carefully. '
        'Write clean, commented Python code.\n\n'
    ),
    'llama3': (
        'You are a helpful scientific AI assistant. Your goal is to discover the '
        'mathematical equation governing a scientific system. Complete the equation '
        'function below, reasoning about the physical meaning and expected '
        'relationships among input variables before writing code.\n\n'
    ),
    'llama2': (
        'You are a helpful assistant for scientific equation discovery. '
        'Complete the equation function below based on the physical context.\n\n'
    ),
    'mistral': (
        'You are a helpful assistant tasked with discovering mathematical function '
        'structures for scientific systems. Complete the equation function below, '
        'considering the physical meaning and relationships of inputs.\n\n'
    ),
    'phi': (
        'You are a scientific assistant specialising in mathematical modeling. '
        'Complete the equation function below. Think about domain knowledge and '
        'physical constraints before proposing a structure.\n\n'
    ),
    'gemma': (
        'You are an AI assistant for scientific equation discovery. '
        'Complete the Python equation function below using your knowledge of '
        'the physical system described.\n\n'
    ),
    'codellama': (
        '# Scientific equation discovery task.\n'
        '# Complete the equation function below to model the described system.\n\n'
    ),
    'starcoder': (
        '# Scientific equation discovery task.\n'
        '# Complete the equation function body below.\n\n'
    ),
    'default': (
        'You are a helpful assistant tasked with discovering mathematical function '
        'structures for scientific systems. Complete the equation function below, '
        'considering the physical meaning and relationships of inputs.\n\n'
    ),
}

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _get_instruction_prompt(model_name: str) -> str:
    lower = model_name.lower()
    if 'qwen' in lower:            return _INSTRUCTION_PROMPTS['qwen']
    if 'deepseek' in lower:        return _INSTRUCTION_PROMPTS['deepseek']
    if 'llama-3' in lower or 'llama3' in lower: return _INSTRUCTION_PROMPTS['llama3']
    if 'llama-2' in lower or 'llama2' in lower: return _INSTRUCTION_PROMPTS['llama2']
    if 'mixtral' in lower or 'mistral' in lower: return _INSTRUCTION_PROMPTS['mistral']
    if 'phi' in lower:             return _INSTRUCTION_PROMPTS['phi']
    if 'gemma' in lower:           return _INSTRUCTION_PROMPTS['gemma']
    if 'codellama' in lower:       return _INSTRUCTION_PROMPTS['codellama']
    if 'starcoder' in lower:       return _INSTRUCTION_PROMPTS['starcoder']
    return _INSTRUCTION_PROMPTS['default']


# ─────────────────────────────────────────────────────────────────────────────
# Body extraction
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = re.sub(r'```(?:python|py)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    return text


def _extract_body(sample: str, config: config_lib.Config) -> str:
    """Extract and normalise the function body from an LLM response."""
    sample = _THINK_RE.sub('', sample).strip()
    sample = _strip_fences(sample)
    lines  = sample.splitlines()
    found, lineno = False, 0
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            lineno, found = i, True
            break
    if found:
        body_lines = lines[lineno + 1:]
        if config.use_api:
            return '\n'.join(body_lines) + '\n'
        indent = '    '
        fixed  = []
        for line in body_lines:
            if line and not line.startswith(indent) and not line.startswith('\t'):
                line = indent + line
            fixed.append(line)
        return '\n'.join(fixed) + '\n'
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# Retry
# ─────────────────────────────────────────────────────────────────────────────

def _with_retry(fn, max_retries: int = 10, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            wait = base_delay * (2 ** min(attempt, 5))
            logger.warning('Attempt %d/%d failed (%s) – retry in %.1fs',
                           attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f'All {max_retries} attempts failed.')


# ─────────────────────────────────────────────────────────────────────────────
# Abstract LLM
# ─────────────────────────────────────────────────────────────────────────────

class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    @abstractmethod
    def draw_samples(self, prompt: str,
                     config: config_lib.Config) -> Collection[str]: ...


# ─────────────────────────────────────────────────────────────────────────────
# LocalLLM
# ─────────────────────────────────────────────────────────────────────────────

class LocalLLM(LLM):
    """Client for the local Flask engine.py server (any HF model family)."""

    def __init__(self, samples_per_prompt: int,
                 server_url: str = 'http://127.0.0.1:5000',
                 batch_inference: bool = True,
                 trim: bool = True) -> None:
        super().__init__(samples_per_prompt)
        self._url         = server_url.rstrip('/') + '/completions'
        self._batch       = batch_inference
        self._trim        = trim
        self._model_name  = self._fetch_model_name(server_url)
        self._instruction = _get_instruction_prompt(self._model_name)
        logger.info('LocalLLM: model=%s', self._model_name or 'unknown')

    def draw_samples(self, prompt: str,
                     config: config_lib.Config) -> Collection[str]:
        full = self._instruction + prompt

        def _call() -> List[str]:
            samples: List[str] = []
            if self._batch:
                samples.extend(self._post(full, repeat=self._samples_per_prompt))
            else:
                for _ in range(self._samples_per_prompt):
                    samples.extend(self._post(full, repeat=1))
            if self._trim:
                samples = [_extract_body(s, config) for s in samples]
            return samples

        return _with_retry(_call)

    def _post(self, content: str, repeat: int) -> List[str]:
        data = {
            'prompt': content.strip(), 'repeat_prompt': repeat,
            'params': {'do_sample': True, 'temperature': None, 'top_k': None,
                       'top_p': None, 'add_special_tokens': False,
                       'skip_special_tokens': True},
        }
        resp = requests.post(self._url, data=json.dumps(data),
                             headers={'Content-Type': 'application/json'},
                             timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f'Server returned {resp.status_code}')
        return resp.json()['content']

    @staticmethod
    def _fetch_model_name(server_url: str) -> str:
        try:
            r = requests.get(server_url.rstrip('/') + '/model_info', timeout=5)
            if r.status_code == 200:
                return r.json().get('model_path', '')
        except Exception:
            pass
        return ''


# ─────────────────────────────────────────────────────────────────────────────
# OpenAILLM
# ─────────────────────────────────────────────────────────────────────────────

class OpenAILLM(LLM):
    """OpenAI Chat Completions API (gpt-3.5-turbo, gpt-4o, o1-mini, …)."""

    def __init__(self, samples_per_prompt: int, trim: bool = True) -> None:
        super().__init__(samples_per_prompt)
        self._trim = trim

    def draw_samples(self, prompt: str,
                     config: config_lib.Config) -> Collection[str]:
        full     = _INSTRUCTION_PROMPTS['default'] + prompt
        model_id = getattr(config, 'api_model', 'gpt-3.5-turbo')
        samples: List[str] = []

        for _ in range(self._samples_per_prompt):
            def _call() -> str:
                conn = http.client.HTTPSConnection('api.openai.com')
                payload = json.dumps({
                    'max_tokens': 512, 'model': model_id, 'temperature': 0.8,
                    'messages': [{'role': 'user', 'content': full}]})
                headers = {'Authorization': f"Bearer {os.environ['API_KEY']}",
                           'Content-Type': 'application/json'}
                conn.request('POST', '/v1/chat/completions', payload, headers)
                data = json.loads(conn.getresponse().read().decode())
                if 'error' in data:
                    raise RuntimeError(str(data['error']))
                return data['choices'][0]['message']['content']

            response = _with_retry(_call)
            if self._trim:
                response = _extract_body(response, config)
            samples.append(response)

        return samples


# ─────────────────────────────────────────────────────────────────────────────
# AnthropicLLM
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicLLM(LLM):
    """Anthropic Claude API.  Set ANTHROPIC_API_KEY env var."""
    _HOST, _ENDPOINT, _VERSION = 'api.anthropic.com', '/v1/messages', '2023-06-01'

    def __init__(self, samples_per_prompt: int, trim: bool = True) -> None:
        super().__init__(samples_per_prompt)
        self._trim = trim

    def draw_samples(self, prompt: str,
                     config: config_lib.Config) -> Collection[str]:
        full     = _INSTRUCTION_PROMPTS['default'] + prompt
        model_id = getattr(config, 'api_model', 'claude-3-5-sonnet-20241022')
        samples: List[str] = []

        for _ in range(self._samples_per_prompt):
            def _call() -> str:
                conn = http.client.HTTPSConnection(self._HOST)
                payload = json.dumps({
                    'model': model_id, 'max_tokens': 1024,
                    'messages': [{'role': 'user', 'content': full}]})
                headers = {
                    'x-api-key': os.environ.get('ANTHROPIC_API_KEY', ''),
                    'anthropic-version': self._VERSION,
                    'content-type': 'application/json'}
                conn.request('POST', self._ENDPOINT, payload, headers)
                data = json.loads(conn.getresponse().read().decode())
                if 'error' in data:
                    raise RuntimeError(str(data['error']))
                return data['content'][0]['text']

            response = _with_retry(_call)
            if self._trim:
                response = _extract_body(response, config)
            samples.append(response)

        return samples


# ─────────────────────────────────────────────────────────────────────────────
# vLLMClient
# ─────────────────────────────────────────────────────────────────────────────

class vLLMClient(LLM):
    """vLLM OpenAI-compatible server client.

    Start server:
        python -m vllm.entrypoints.openai.api_server \\
            --model Qwen/Qwen2.5-7B-Instruct --port 8000
    """

    def __init__(self, samples_per_prompt: int,
                 server_url: str = 'http://127.0.0.1:8000',
                 trim: bool = True) -> None:
        super().__init__(samples_per_prompt)
        self._base  = server_url.rstrip('/')
        self._trim  = trim
        self._model = self._fetch_model_id()
        self._instruction = _get_instruction_prompt(self._model)
        logger.info('vLLMClient: model=%s  url=%s', self._model, server_url)

    def draw_samples(self, prompt: str,
                     config: config_lib.Config) -> Collection[str]:
        full    = self._instruction + prompt
        samples: List[str] = []

        for _ in range(self._samples_per_prompt):
            def _call() -> str:
                r = requests.post(
                    self._base + '/v1/chat/completions',
                    json={'model': self._model,
                          'messages': [{'role': 'user', 'content': full}],
                          'max_tokens': 512, 'temperature': 0.8, 'top_p': 0.9},
                    timeout=120)
                if r.status_code != 200:
                    raise RuntimeError(f'vLLM returned {r.status_code}')
                return r.json()['choices'][0]['message']['content']

            response = _with_retry(_call)
            if self._trim:
                response = _extract_body(response, config)
            samples.append(response)

        return samples

    def _fetch_model_id(self) -> str:
        try:
            r = requests.get(self._base + '/v1/models', timeout=5)
            if r.status_code == 200:
                models = r.json().get('data', [])
                if models:
                    return models[0].get('id', 'unknown')
        except Exception:
            pass
        return 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_llm(samples_per_prompt: int, config: config_lib.Config) -> LLM:
    """Select and instantiate the correct LLM backend.

    Priority
    ────────
    use_api=True + api_model starts with 'claude-'  → AnthropicLLM
    use_api=True + llm_backend='vllm'               → vLLMClient
    use_api=True                                     → OpenAILLM
    use_api=False + llm_backend='vllm'              → vLLMClient
    use_api=False                                    → LocalLLM
    """
    backend   = getattr(config, 'llm_backend', 'local')
    api_model = getattr(config, 'api_model', '')

    if config.use_api:
        if api_model.startswith('claude-'):
            logger.info('Backend: Anthropic Claude  model=%s', api_model)
            return AnthropicLLM(samples_per_prompt)
        if backend == 'vllm':
            url = getattr(config, 'vllm_url', 'http://127.0.0.1:8000')
            logger.info('Backend: vLLM (API)  url=%s', url)
            return vLLMClient(samples_per_prompt, server_url=url)
        logger.info('Backend: OpenAI  model=%s', api_model)
        return OpenAILLM(samples_per_prompt)

    if backend == 'vllm':
        url = getattr(config, 'vllm_url', 'http://127.0.0.1:8000')
        logger.info('Backend: vLLM (local)  url=%s', url)
        return vLLMClient(samples_per_prompt, server_url=url)

    url = getattr(config, 'local_server_url', 'http://127.0.0.1:5000')
    logger.info('Backend: LocalLLM  url=%s', url)
    return LocalLLM(samples_per_prompt, server_url=url)


# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────

class Sampler:
    """Continuously samples prompts, generates skeletons, dispatches to evaluators."""

    _global_samples_nums: int = 1

    def __init__(
        self,
        database,
        evaluators: Sequence,
        samples_per_prompt: int,
        config: config_lib.Config,
        max_sample_nums: Optional[int] = None,
        llm_class: Type[LLM] = LocalLLM,   # kept for API compat; build_llm used
        hessian_cfg=None,
        adaptive_scheduler=None,
    ) -> None:
        self._database           = database
        self._evaluators         = evaluators
        self._samples_per_prompt = samples_per_prompt
        self._max_sample_nums    = max_sample_nums
        self.config              = config
        self._hessian_cfg        = hessian_cfg
        self._adaptive_scheduler = adaptive_scheduler
        self._llm                = build_llm(samples_per_prompt, config)

    def sample(self, **kwargs) -> None:
        hessian_cfg        = kwargs.pop('hessian_cfg',        self._hessian_cfg)
        adaptive_scheduler = kwargs.pop('adaptive_scheduler', self._adaptive_scheduler)
        ablation_mode      = kwargs.pop('ablation_mode',      self.config.ablation_mode)

        while True:
            if (self._max_sample_nums
                    and self.__class__._global_samples_nums >= self._max_sample_nums):
                break

            # adaptive island selection when scheduler is active
            prompt = self._database.get_prompt(scheduler=adaptive_scheduler)

            t0          = time.time()
            samples     = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - t0) / max(len(samples), 1)

            for sample in samples:
                self._global_sample_nums_plus_one()
                cur = self._get_global_sample_nums()
                ev  = np.random.choice(self._evaluators)
                ev.analyse(
                    sample, prompt.island_id, prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur,
                    sample_time=sample_time,
                    hessian_cfg=hessian_cfg,
                    adaptive_scheduler=adaptive_scheduler,
                    ablation_mode=ablation_mode,
                )

            if adaptive_scheduler is not None:
                state    = adaptive_scheduler.state_dict()
                profiler = kwargs.get('profiler', None)
                if profiler is not None and hasattr(profiler, 'log_scheduler_state'):
                    profiler.log_scheduler_state(state, self._get_global_sample_nums())

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num: int) -> None:
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self) -> None:
        self.__class__._global_samples_nums += 1
