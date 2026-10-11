"""All settings of ARR and of the baselines.

The method has two settings: the per-question FLOP budget B (``budgets``) and
the number of active roots N (``n_active``).  Replay decoding adds its block
size W (``block_size``), which changes speed but not the output.  There are no
confidence thresholds, learned controllers, or tuned revision depths.  Decoding
settings and output caps are shared by every method.

Precedence (lowest to highest): dataclass defaults < ``.env`` <
``configs/*.conf`` < command-line flags.  A field ``foo_bar`` is read from the
environment variable ``ARR_FOO_BAR``; ``MODEL_NAME`` and ``SEED`` are also read.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence

# Method keys; see methods.py (and conma.py) for the paper name of each.
METHODS = ("arr", "replace_ordinary", "replace_pld", "arr_no_replace", "direct_fixed",
           "critic_correct", "continuation", "maj", "conma", "arr_answer", "arr_once")


@dataclass
class ARRConfig:
    # Model / engine
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    dtype: str = "bfloat16"
    hf_cache_dir: str = ""                    # "" → HF_HOME / HF_HUB_CACHE
    trust_remote_code: bool = False
    thinking: str = "off"                     # off | on — chat-template thinking mode (recorded; Qwen3-style models)
    max_input_tokens: int = 2560              # prompts are left-truncated beyond this
    engine_max_rows: int = 0                  # concurrent sequences (KV slots); 0 → sized from free GPU memory
    gpu_memory_utilization: float = 0.92      # fraction of GPU memory the engine may use (weights + KV + work)
    engine_workspace_gb: float = 5.0          # memory kept free for activations / logits / attention scores
    sampler: str = "per_row"                  # per_row: one GPU draw per row | batched: per-request CPU uniforms, one GPU pass
    attn_query_chunk: int = 4096              # query-block size inside attention (bounds score memory)
    seed: int = 1234

    # Shared decoding (identical for every method)
    temperature: float = 0.7                  # sampling of roots (and of sampled baseline updates)
    top_p: float = 0.95
    max_new_tokens: int = 1536                # one output cap for every call (solutions, revisions, critiques)
    critique_max_new_tokens: int = 384        # output cap of the SETS critique
    extract_fallback_last_number: bool = True  # same answer extractor for all methods

    # ARR
    n_active: int = 4                         # N: active roots (execution slots)
    budgets: str = "150,300,600"              # per-question budgets B (cost_measure units)
    cost_measure: str = "tflops"              # tflops | passes (serial forward passes of the question)
    block_size: int = 32                      # W: draft tokens checked per parallel pass
    revise_prompt: int = 0                    # 0 = the paper's rho; 1, 2 = paraphrases (prompt ablation)

    # Baselines
    fixed_depth: int = 4                      # D of Self-Refine (direct_fixed)
    pld_max_ngram: int = 3                    # prompt-lookup decoding (reference-implementation defaults)
    pld_num_pred: int = 10
    continuation_max_prior_chars: int = 4000  # input truncation of Sequential Revision and SETS
    critique_max_trace_chars: int = 4000
    correction_max_critique_chars: int = 1500
    # ConMA (Wang et al., 2026): defaults of its paper; conma_temp is not given there
    conma_width: int = 8                      # n: trajectories per round
    conma_max_rounds: int = 0                 # 0 → rounds limited only by the budget (paper: T_max = 4)
    conma_alpha: float = 0.3                  # weight of the full-sequence confidence
    conma_beta: float = 0.7                   # weight of the tail confidence
    conma_tail: int = 128                     # W: tail window in tokens
    conma_lambda: float = 0.5                 # consensus vs. mean path quality in the group score
    conma_rho: float = 0.75                   # cumulative-probability threshold of the kernel
    conma_delta: float = 0.5                  # DSE trigger: |kernel| > delta * n
    conma_temp: float = 0.1                   # softmax temperature over group scores (our choice)

    # Experiment
    methods: str = "arr,replace_ordinary,replace_pld,arr_no_replace,direct_fixed,critic_correct,continuation,maj"
    datasets: str = "math500"
    max_samples: int = 0                      # 0 → whole dataset
    shard: str = ""                           # "i/n": run only questions with index % n == i (own files; merged on read)
    seeds: int = 1                            # independent repetitions per question
    seed_ids: str = ""                        # run only these seed indices (e.g. "1"); "" → 0 .. seeds-1
    pilot_dataset: str = "math500"            # 50-question feasibility pilot
    pilot_samples: int = 50

    # Output
    output_root: str = "experiments_results/arr"
    max_active_questions: int = 0             # concurrent (question, method, budget) solves; 0 → 2 × rows / N
    log_every_s: float = 60.0

    def list_of(self, name: str) -> List[str]:
        return [x.strip() for x in str(getattr(self, name) or "").split(",") if x.strip()]

    def seed_indices(self) -> List[int]:
        if self.seed_ids:
            return [int(x) for x in self.list_of("seed_ids")]
        return list(range(max(1, self.seeds)))

    def shard_spec(self):
        if not self.shard:
            return None
        i, n = (int(x) for x in self.shard.split("/"))
        return i, n

    def shard_suffix(self) -> str:
        sp = self.shard_spec()
        return f".s{sp[0]}of{sp[1]}" if sp else ""

    def budget_list(self) -> List[float]:
        return [float(x) for x in self.list_of("budgets")]

    def model_dir(self) -> str:
        return os.path.join(self.output_root, os.path.basename(self.model_name.rstrip("/")))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


_LEGACY_ENV = {"model_name": "MODEL_NAME", "seed": "SEED"}


def _coerce(value: str, template: Any) -> Any:
    if isinstance(template, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(template, int):
        return int(float(value))
    if isinstance(template, float):
        return float(value)
    return str(value)


def config_from_env(env: Optional[Dict[str, str]] = None) -> ARRConfig:
    env = dict(os.environ if env is None else env)
    cfg = ARRConfig()
    for f in fields(ARRConfig):
        default = getattr(cfg, f.name)
        key = f"ARR_{f.name.upper()}"
        if env.get(key, "") != "":
            setattr(cfg, f.name, _coerce(env[key], default))
        elif f.name in _LEGACY_ENV and env.get(_LEGACY_ENV[f.name], "") != "":
            setattr(cfg, f.name, _coerce(env[_LEGACY_ENV[f.name]], default))
    return cfg


def load_shell_env(paths: Sequence[Optional[str]]) -> Dict[str, str]:
    """Source bash files (``.env`` / ``*.conf``) in order and import the result."""
    paths = [p for p in paths if p]
    if not paths:
        return {}
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    script = "set -a; " + " ".join(f'source "{os.path.abspath(p)}";' for p in paths) + " env -0"
    out = subprocess.run(["bash", "-c", script], capture_output=True, check=True).stdout
    loaded: Dict[str, str] = {}
    for item in out.split(b"\0"):
        if b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        ks, vs = k.decode(), v.decode(errors="replace")
        if os.environ.get(ks) != vs:
            os.environ[ks] = vs
            loaded[ks] = vs
    return loaded


def add_config_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("ARR config overrides (highest precedence)")
    for f in fields(ARRConfig):
        g.add_argument(f"--{f.name}", default=None, help=f"env ARR_{f.name.upper()}")
    p.add_argument("--env_file", default=None, help="bash env file to source first (e.g. .env)")
    p.add_argument("--conf", default=None, help="conf file(s) sourced after --env_file (comma-separated)")


def config_from_args(args: argparse.Namespace) -> ARRConfig:
    confs = [c for c in (getattr(args, "conf", None) or "").split(",") if c.strip()]
    load_shell_env([getattr(args, "env_file", None)] + confs)
    cfg = config_from_env()
    for f in fields(ARRConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, _coerce(v, getattr(cfg, f.name)))
    validate_config(cfg)
    return cfg


def validate_config(cfg: ARRConfig) -> None:
    if cfg.cost_measure not in ("tflops", "passes"):
        raise ValueError("cost_measure must be tflops|passes")
    for m in cfg.list_of("methods"):
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}; choose from {METHODS}")
    if cfg.n_active < 1:
        raise ValueError("n_active must be >= 1")
    if cfg.block_size < 1:
        raise ValueError("block_size must be >= 1")
    if not cfg.budget_list():
        raise ValueError("budgets must list at least one budget")
    if cfg.revise_prompt not in (0, 1, 2):
        raise ValueError("revise_prompt must be 0, 1 or 2")
    if cfg.shard:
        i, n = cfg.shard_spec()
        if not (n >= 1 and 0 <= i < n):
            raise ValueError("shard must be i/n with 0 <= i < n")
    if cfg.sampler not in ("per_row", "batched"):
        raise ValueError("sampler must be per_row|batched")
    if cfg.thinking not in ("off", "on"):
        raise ValueError("thinking must be off|on")


def dump_config(cfg: ARRConfig, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = cfg.to_dict()
    d.update(extra or {})
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


__all__ = ["ARRConfig", "METHODS", "config_from_env", "config_from_args", "add_config_args",
           "load_shell_env", "validate_config", "dump_config"]
