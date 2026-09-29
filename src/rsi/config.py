"""
config.py
=========
Single source of truth for every knob of the Procedure-RSI pipeline.

Precedence (lowest → highest)
-----------------------------
  1. dataclass defaults below
  2. ``.env``                 (machine paths + predefined experiment defaults)
  3. ``configs/<name>.conf``  (per-model / per-experiment overrides)
  4. command-line flags       (``--<field> VALUE`` or ``--set field=value``)

Every field ``foo_bar`` is read from the environment variable ``RSI_FOO_BAR``.
A handful of fields shared with the original paired-ablation scripts also
accept their legacy un-prefixed names (``MODEL_NAME``, ``BATCH_SIZE`` ...), see
``_LEGACY_ENV``.

``.env`` / ``.conf`` files are bash (they contain ``if`` blocks and
``${VAR:-default}`` expansions), so they are *sourced by bash* rather than
parsed in Python: ``load_shell_env`` runs ``bash -c 'set -a; source ...'`` and
imports the resulting environment.  The shell entrypoints source them directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence

# Actions the procedure may use (fixed vocabulary; the editor cannot add new ones).
PRIMITIVE_OPS = ("FULL_REFINE", "DIRECT_REPAIR", "RESTART", "KEEP")
PAIR_OPS = ("FULL_REFINE", "DIRECT_REPAIR", "RESTART")


@dataclass
class RSIConfig:
    # ── Model / backend ─────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    backend: str = "hf"                      # hf | vllm
    dtype: str = "auto"
    device: str = "auto"
    batch_size: int = 32
    hf_cache_dir: str = ""                   # "" → HF_HOME / HF_HUB_CACHE from the environment
    trust_remote_code: bool = False
    max_input_tokens: int = 4096             # hard truncation of any prompt (left side)
    model_num_params: float = 0.0            # 0 → counted from the loaded model
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int = 8192
    seed: int = 1234

    # ── Role decoding (generator / critic / corrector share one frozen model) ─
    init_max_new_tokens: int = 2048          # initial + RESTART generations
    init_temperature: float = 0.7
    top_p: float = 0.95
    rewrite_max_new_tokens: int = 2048       # FULL_REFINE continuation/rewrite g_i
    rewrite_temperature: float = 0.7         # 0 → greedy (the old CIMR code path)
    critique_max_new_tokens: int = 384
    critique_temperature: float = 0.0
    correct_max_new_tokens: int = 1024
    correct_temperature: float = 0.0
    critique_max_trace_chars: int = 4000
    correct_max_trace_chars: int = 4000
    correct_max_critique_chars: int = 1500
    rewrite_max_prior_chars: int = 4000
    extract_fallback_last_number: bool = True

    # ── Inner loop (§4) ─────────────────────────────────────────────────────
    n_slots: int = 4                         # N (pilot: 4; paper: 8)
    h0_depth: int = 2                        # H0 = paper round-robin to this depth
    budget_mode: str = "h0_cycles"           # h0_cycles | absolute
    budget_cycles: float = 2.0               # B ≈ N initial + N·cycles FULL_REFINE (worst case)
    budget_value: float = 0.0                # used when budget_mode=absolute (cost_measure units)
    cost_measure: str = "tflops"             # tflops | tokens | calls — predeclared cap/utility measure
    max_decisions: int = 0                   # 0 → auto: 4·N·cycles + 8
    reserve_slack_tokens: int = 64
    disabled_actions: str = ""               # e.g. "KEEP,RESTART" (ablation §8)

    # ── Counterfactual measurement (§5, §5.1) ───────────────────────────────
    utility_lambda: float = 0.2              # λ in Q = 1{correct} − λ·C/B
    cf_seeds: int = 4                        # K continuation seeds per branch
    cf_states_per_problem: int = 1
    cf_alt_slots: int = 2                    # slots studied per state (rotating coverage)
    cf_alt_ops: str = "FULL_REFINE,DIRECT_REPAIR,RESTART,KEEP"
    cf_include_stop: bool = True
    cf_pairs_per_state: int = 2              # ≤ 2 ordered pairs per state (pilot)
    cf_pair_ops: str = "FULL_REFINE,DIRECT_REPAIR,RESTART"
    cf_explore_frac: float = 0.3             # fraction of states whose slots are chosen by rotation
    evidence_mode: str = "counterfactual"    # counterfactual | trajectory  (ablation)
    credit_mode: str = "population"          # population | trace          (ablation)
    memory_signed: bool = True               # False → success-only memory  (ablation)

    # ── Outer loop (§6) ─────────────────────────────────────────────────────
    rounds: int = 5
    search_batch_size: int = 20
    confirm_block_size: int = 40
    confirm_seeds: int = 2
    search_eval_seeds: int = 1
    editor_mode: str = "both"                # llm | compiler | both
    editor_num_proposals: int = 4
    editor_max_new_tokens: int = 768
    editor_temperature: float = 0.7
    edit_types: str = "rule,plan,prompt,delete"
    max_rules: int = 12                      # declared rule budget (sensitivity to be measured)
    max_plans: int = 4
    max_prompt_chars: int = 1200
    allow_plans: bool = True                 # False → primitive-only control (ablation)
    plan_selection: str = "interaction"      # interaction | random | greedy_vote (ablation)
    min_support: int = 3                     # evidence count before compiling a rule/plan
    pair_repro_frac: float = 0.6             # fraction of states with I>0 for a "reproducible" pair
    rule_delta_min: float = 0.02
    alpha_total: float = 0.05                # α_t = alpha_total / rounds
    noninferiority_margin: float = 0.02      # accuracy margin (absolute fraction)
    utility_min_gain: float = 0.0
    persist_procedure: bool = True           # False → reset H to H0 each round, keep evidence
    freeze_after_round: int = 0              # >0: stop adapting after this round (frozen arm)
    permutation_samples: int = 10000

    # ── Data (§6) ───────────────────────────────────────────────────────────
    dev_datasets: str = "math_train"
    test_datasets: str = "math500"
    dev_pool_max: int = 2000
    test_max_samples: int = 0                # 0 → all
    split_seed: int = 17

    # ── Output ──────────────────────────────────────────────────────────────
    output_root: str = "experiments_results/procedure_rsi"
    checkpoint_every_s: float = 60.0         # in-flight solve states are saved at least this often
    max_inflight_solves: int = 0             # 0 → 4 × batch_size; bounds tick length (= checkpoint granularity)
    run_name: str = ""                       # "" / "main" → <root>/<model>/ ; else <root>/<model>/runs/<run_name>/
    log_every: int = 5

    # ------------------------------------------------------------------
    def list_of(self, name: str) -> List[str]:
        raw = getattr(self, name) or ""
        return [x.strip() for x in str(raw).split(",") if x.strip()]

    def resolved_max_decisions(self) -> int:
        if self.max_decisions > 0:
            return int(self.max_decisions)
        return int(4 * self.n_slots * max(self.budget_cycles, 1.0) + 8)

    def resolved_max_inflight(self) -> int:
        """Solves active at once.  The HF backend already works in batch_size
        chunks, so a few chunks per tick keeps the GPU full while each tick
        (and therefore each checkpoint) stays short."""
        return int(self.max_inflight_solves) if self.max_inflight_solves > 0 else 4 * max(1, self.batch_size)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def model_dir(self) -> str:
        """<root>/<model>/ for the main run; <root>/<model>/runs/<run_name>/ for
        ablations / smoke / sample runs (same inner layout)."""
        short = os.path.basename(self.model_name.rstrip("/"))
        base = os.path.join(self.output_root, short)
        if self.run_name and self.run_name != "main":
            return os.path.join(base, "runs", self.run_name)
        return base

    def run_dir(self) -> str:
        """Development artifacts (splits, rounds, memory, lineage, checkpoints)."""
        return os.path.join(self.model_dir(), "development")

    def final_procedure_path(self) -> str:
        return os.path.join(self.model_dir(), "final_procedure.json")

    def dataset_dir(self, dataset: str) -> str:
        return os.path.join(self.model_dir(), dataset)


# Legacy / shared env names accepted in addition to RSI_<FIELD>.
_LEGACY_ENV = {
    "model_name": "MODEL_NAME",
    "batch_size": "BATCH_SIZE",
    "seed": "SEED",
}


def _coerce(value: str, template: Any) -> Any:
    if isinstance(template, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(template, int) and not isinstance(template, bool):
        return int(float(value))
    if isinstance(template, float):
        return float(value)
    return str(value)


def config_from_env(env: Optional[Dict[str, str]] = None) -> RSIConfig:
    env = dict(os.environ if env is None else env)
    cfg = RSIConfig()
    for f in fields(RSIConfig):
        default = getattr(cfg, f.name)
        key = f"RSI_{f.name.upper()}"
        if key in env and env[key] != "":
            setattr(cfg, f.name, _coerce(env[key], default))
        elif f.name in _LEGACY_ENV and env.get(_LEGACY_ENV[f.name], "") != "":
            setattr(cfg, f.name, _coerce(env[_LEGACY_ENV[f.name]], default))
    return cfg


def load_shell_env(paths: Sequence[str]) -> Dict[str, str]:
    """Source bash files in order and copy the resulting variables into os.environ.

    Variables already exported by the caller take precedence only where the
    file itself uses ``${VAR:-default}``; plain assignments in a ``.conf``
    override, exactly as when the shell scripts source them.
    """
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
        k_s, v_s = k.decode(), v.decode(errors="replace")
        if os.environ.get(k_s) != v_s:
            os.environ[k_s] = v_s
            loaded[k_s] = v_s
    return loaded


def add_config_args(p: argparse.ArgumentParser) -> None:
    """Expose every RSIConfig field as ``--<field>`` (default None = keep env)."""
    g = p.add_argument_group("RSI config overrides (highest precedence)")
    for f in fields(RSIConfig):
        g.add_argument(f"--{f.name}", default=None, help=f"env RSI_{f.name.upper()}")
    g.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="generic override, may repeat")
    p.add_argument("--env_file", default=None, help="bash env file to source (e.g. .env)")
    p.add_argument("--conf", default=None,
                   help="bash conf file(s) sourced after --env_file; comma-separate to stack "
                        "(e.g. configs/rsi_qwen_1b.conf,configs/ablations/primitive_only.conf)")


def config_from_args(args: argparse.Namespace) -> RSIConfig:
    confs = [c for c in (getattr(args, "conf", None) or "").split(",") if c.strip()]
    load_shell_env([getattr(args, "env_file", None)] + confs)
    cfg = config_from_env()
    for f in fields(RSIConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, _coerce(v, getattr(cfg, f.name)))
    for item in getattr(args, "set", []) or []:
        if "=" not in item:
            raise ValueError(f"--set expects FIELD=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if not hasattr(cfg, k):
            raise ValueError(f"unknown config field {k!r}")
        setattr(cfg, k, _coerce(v, getattr(cfg, k)))
    validate_config(cfg)
    return cfg


def validate_config(cfg: RSIConfig) -> None:
    if cfg.cost_measure not in ("tflops", "tokens", "calls"):
        raise ValueError(f"cost_measure must be tflops|tokens|calls, got {cfg.cost_measure}")
    if cfg.budget_mode not in ("h0_cycles", "absolute"):
        raise ValueError(f"budget_mode must be h0_cycles|absolute, got {cfg.budget_mode}")
    if cfg.budget_mode == "absolute" and cfg.budget_value <= 0:
        raise ValueError("budget_mode=absolute requires budget_value > 0")
    if cfg.n_slots < 1:
        raise ValueError("n_slots must be >= 1")
    for op in cfg.list_of("cf_alt_ops"):
        if op not in PRIMITIVE_OPS:
            raise ValueError(f"cf_alt_ops: unknown op {op}")
    for op in cfg.list_of("cf_pair_ops"):
        if op not in PAIR_OPS:
            raise ValueError(f"cf_pair_ops: unknown op {op}")
    for op in cfg.list_of("disabled_actions"):
        if op not in PRIMITIVE_OPS + ("STOP", "PAIR"):
            raise ValueError(f"disabled_actions: unknown action {op}")
    if cfg.evidence_mode not in ("counterfactual", "trajectory"):
        raise ValueError("evidence_mode must be counterfactual|trajectory")
    if cfg.credit_mode not in ("population", "trace"):
        raise ValueError("credit_mode must be population|trace")
    if cfg.plan_selection not in ("interaction", "random", "greedy_vote"):
        raise ValueError("plan_selection must be interaction|random|greedy_vote")
    if cfg.editor_mode not in ("llm", "compiler", "both"):
        raise ValueError("editor_mode must be llm|compiler|both")
    if cfg.backend not in ("hf", "vllm", "fake"):
        raise ValueError("backend must be hf|vllm|fake")


def dump_config(cfg: RSIConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)


__all__ = [
    "RSIConfig", "PRIMITIVE_OPS", "PAIR_OPS",
    "config_from_env", "config_from_args", "add_config_args",
    "load_shell_env", "validate_config", "dump_config",
]
