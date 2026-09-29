"""
config.py
=========
All knobs of Adaptive Delta Refinement (ADR) and its baselines.

Principal configuration (updated_methodology.md §6): breadth N and a set of
predeclared total per-question budgets B (replacing fixed depth D).  Decoding
settings and per-call output caps are shared by every method.  There are no
confidence thresholds, learned scheduling parameters, development sets or
procedure files.

Precedence (lowest → highest): dataclass defaults < .env < configs/*.conf < CLI.
Every field ``foo_bar`` is read from ``ADR_FOO_BAR``; ``MODEL_NAME``,
``BATCH_SIZE`` and ``SEED`` are shared with the original CIMR scripts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence

# Methods of the essential comparison (§7) + the two essential ablations.
METHODS = ("cimr", "full_rewrite", "suffix", "adr", "adr_all_deps", "maj_matched")
DIRECT_CRITIQUE_METHODS = ("full_rewrite", "suffix", "adr", "adr_all_deps")


@dataclass
class ADRConfig:
    # ── Model / backend ─────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    backend: str = "hf"                       # hf | vllm
    dtype: str = "bfloat16"
    device: str = "auto"
    batch_size: int = 64                      # sequences per GPU chunk
    hf_cache_dir: str = ""                    # "" → HF_HOME / HF_HUB_CACHE
    trust_remote_code: bool = False
    max_input_tokens: int = 3072              # prompts are left-truncated beyond this
    model_num_params: float = 0.0             # 0 → counted from the loaded model (for FLOPs)
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int = 4096
    seed: int = 1234

    # ── Shared decoding (identical for every method) ─────────────────────────
    temperature: float = 0.7                  # generation of solutions / rewrites / patches
    top_p: float = 0.95
    gen_max_new_tokens: int = 1536            # a complete solution (initial, rewrite, CIMR rewrite/correct)
    critique_max_new_tokens: int = 384
    critique_temperature: float = 0.0         # critic is deterministic (as in the paper)
    patch_max_new_tokens: int = 1024          # local patch / suffix regeneration
    extract_fallback_last_number: bool = True  # same extractor for all methods

    # ── Breadth / budget ────────────────────────────────────────────────────
    n_rollouts: int = 4                       # N (§7 main study starts with N=4)
    budgets: str = "30000,60000"              # predeclared total budgets B per question (cost_measure units)
    cost_measure: str = "tokens"              # tokens | tflops | calls (tokens ∝ FLOPs = 2·P·tokens)
    max_turns_per_rollout: int = 64           # safety net only; the budget stops runs first

    # ── Original CIMR baseline (fixed depth, three-stage cycle) ──────────────
    cimr_depth: int = 4                       # D (paper default 4)
    cimr_max_prior_chars: int = 4000
    cimr_critique_max_trace_chars: int = 4000
    cimr_correction_max_critique_chars: int = 1500

    # ── Experiment ──────────────────────────────────────────────────────────
    methods: str = "cimr,full_rewrite,suffix,adr,adr_all_deps,maj_matched"
    datasets: str = "aime24"
    max_samples: int = 0                      # 0 → whole dataset
    seeds: int = 1                            # independent repetitions per question
    pilot_samples: int = 50                   # §7 feasibility pilot size
    pilot_dataset: str = "math500"

    # ── Output (big files → scratch) ────────────────────────────────────────
    output_root: str = "experiments_results/adr"
    checkpoint_every_s: float = 60.0
    max_inflight_solves: int = 0              # 0 → 4 × batch_size
    log_every_s: float = 60.0

    # ------------------------------------------------------------------
    def list_of(self, name: str) -> List[str]:
        return [x.strip() for x in str(getattr(self, name) or "").split(",") if x.strip()]

    def budget_list(self) -> List[float]:
        return [float(x) for x in self.list_of("budgets")]

    def resolved_max_inflight(self) -> int:
        return int(self.max_inflight_solves) if self.max_inflight_solves > 0 else 4 * max(1, self.batch_size)

    def model_dir(self) -> str:
        return os.path.join(self.output_root, os.path.basename(self.model_name.rstrip("/")))

    def cell_dir(self, dataset: str, cell: str) -> str:
        """<root>/<model>/<dataset>/<cell>/ — cell = B<budget> or cimr_D<depth> or pilot."""
        return os.path.join(self.model_dir(), dataset, cell)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


_LEGACY_ENV = {"model_name": "MODEL_NAME", "batch_size": "BATCH_SIZE", "seed": "SEED"}


def _coerce(value: str, template: Any) -> Any:
    if isinstance(template, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(template, int):
        return int(float(value))
    if isinstance(template, float):
        return float(value)
    return str(value)


def config_from_env(env: Optional[Dict[str, str]] = None) -> ADRConfig:
    env = dict(os.environ if env is None else env)
    cfg = ADRConfig()
    for f in fields(ADRConfig):
        default = getattr(cfg, f.name)
        key = f"ADR_{f.name.upper()}"
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
    g = p.add_argument_group("ADR config overrides (highest precedence)")
    for f in fields(ADRConfig):
        g.add_argument(f"--{f.name}", default=None, help=f"env ADR_{f.name.upper()}")
    p.add_argument("--env_file", default=None, help="bash env file to source first (e.g. .env)")
    p.add_argument("--conf", default=None, help="conf file(s) sourced after --env_file (comma-separated)")


def config_from_args(args: argparse.Namespace) -> ADRConfig:
    confs = [c for c in (getattr(args, "conf", None) or "").split(",") if c.strip()]
    load_shell_env([getattr(args, "env_file", None)] + confs)
    cfg = config_from_env()
    for f in fields(ADRConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, _coerce(v, getattr(cfg, f.name)))
    validate_config(cfg)
    return cfg


def validate_config(cfg: ADRConfig) -> None:
    if cfg.cost_measure not in ("tokens", "tflops", "calls"):
        raise ValueError("cost_measure must be tokens|tflops|calls")
    for m in cfg.list_of("methods"):
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}; choose from {METHODS}")
    if cfg.n_rollouts < 1:
        raise ValueError("n_rollouts must be >= 1")
    if not cfg.budget_list():
        raise ValueError("budgets must list at least one budget")
    if cfg.backend not in ("hf", "vllm"):
        raise ValueError("backend must be hf|vllm")


def dump_config(cfg: ADRConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


__all__ = ["ADRConfig", "METHODS", "DIRECT_CRITIQUE_METHODS", "config_from_env", "config_from_args",
           "add_config_args", "load_shell_env", "validate_config", "dump_config"]
