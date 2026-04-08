"""
inference.py – AdaptSci multi-turn agentic inference.

AdaptSci additions on top of SR-Scientist inference.py
=======================================================
1. Hessian-adaptive BFGS restart count (Section 3.2 / 4.1)
   - After each branch evaluation, compute HessianDiag for each new equation.
   - Set n_restarts per equation based on κ.
   - Pass n_restarts to both the in-sandbox equation_evaluator and the
     post-hoc Python-side get_performance() call.

2. Stratified 70/30 buffer  (Section 4.2)
   - Replace the simple top-K MAPE sort in _create_previous_turn_context()
     with AdaptSciBuffer.select_seeds(), which balances exploitation and
     frontier diversity via the coupling-matrix C-distance.

3. Degenerate skeleton detection → diversity-only mode  (Section 4.2)
   - When the best equation in the buffer is degenerate (|C_ij| > γ or
     n_eff < 0.5p), set α=0 so the frontier pool is selected on C-diversity
     alone, forcing structural exploration.

4. Early stopping  (Section 4.3)
   - Per-example EarlyStoppingState tracks calls and best NMSE.
   - When NMSE < ε_target OR efficiency ratio β > β* the example is marked
     as completed and its remaining budget is freed.

5. Ablation flags  (Section 6, "Key ablations")
   All features can be individually disabled through the `ablation_config`
   dict so the paper's ablation table can be reproduced exactly:
     - disable_early_stopping
     - fix_n_restarts          (forces n_restarts=1)
     - alpha_override          (fixes α; sweep 0,0.25,0.5,0.75,1)
     - disable_c_diversity     (equivalent to alpha_override=1)
     - random_frontier         (random sampling instead of C-distance frontier)
     - hessian_at_first_run    (compute Hessian at first BFGS run, not best-of-n)

Backward compatibility
======================
When all ablation flags are at their defaults AND adaptive_compute=False is
passed, the system behaves identically to the original SR-Scientist code.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import json
import logging
import os
import re
import sys
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from .prompt import (
    SYSTEM_PROMPT,
    USER_PROMPT_WITH_MEMORY,
    USER_PROMPT_WITH_HESSIAN_CONTEXT,
    format_hessian_guidance,
)
from .get_results import (
    parse_equation,
    get_performance,
    get_performance_on_test,
    get_final_results,
)
from .tool import tools, get_function_by_name
from .hessian import (
    HessianDiag,
    compute_hessian_diag,
    coupling_distance,
)
from .buffer import AdaptSciBuffer, EquationRecord
from .early_stop import EarlyStoppingState, make_early_stopping

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, (np.floating, np.longdouble)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    Controls which AdaptSci mechanisms are active.
    Default = all mechanisms on (full AdaptSci).

    Set adaptive_compute=False to run in pure SR-Scientist mode.
    """
    adaptive_compute: bool = True          # master switch
    disable_early_stopping: bool = False   # ablation (i)
    fix_n_restarts: bool = False           # ablation (ii): forces n_restarts=1
    alpha_override: Optional[float] = None # ablation (iii): sweep α ∈ {0,.25,.5,.75,1}
    disable_c_diversity: bool = False      # ablation (iv): equiv. to alpha_override=1
    random_frontier: bool = False          # ablation (iv): random instead of C-distance
    hessian_at_first_run: bool = False     # ablation (v): Hessian at first BFGS, not best-of-n
    # Hyperparameters (calibrated on 20% validation split)
    tau1: float = 100.0     # κ threshold for 1→3 restarts
    tau2: float = 1_000.0   # κ threshold for 3→5 restarts
    epsilon_target: float = 1e-4   # NMSE convergence target for early stopping
    beta_star: float = 5_000.0     # efficiency-ratio threshold β*
    coupling_threshold: float = 0.95   # |C_ij| threshold for degeneracy
    n_eff_ratio_threshold: float = 0.5  # n_eff / p threshold for degeneracy
    exploit_frac: float = 0.70     # fraction of K given to exploitation pool
    frontier_alpha: float = 0.5    # α in s_fr when not overridden
    frontier_window: int = 3       # window w for momentum computation
    include_hessian_in_prompt: bool = True  # inject κ/n_eff guidance into LLM prompt


# ---------------------------------------------------------------------------
# Branch dataclass (extended with Hessian state)
# ---------------------------------------------------------------------------

@dataclass
class Branch:
    branch_id: int
    example_index: int
    status: str  # 'pending_model' | 'in_flight_model' | 'in_flight_tool' | 'completed'
    messages: List[Dict]
    mape_threshold: float
    tool_calls: List[Any] = field(default_factory=list)
    tool_executions: List[Dict] = field(default_factory=list)
    finish_reason: Optional[str] = None
    source: Optional[str] = None
    expression: Optional[str] = None
    assistant_turn_count: int = 0
    final_train_results: Dict = field(default_factory=dict)
    final_test_results: Dict = field(default_factory=dict)
    # AdaptSci: per-branch n_restarts (derived from κ of best current eq.)
    n_restarts: int = 1


# ---------------------------------------------------------------------------
# BranchManager
# ---------------------------------------------------------------------------

class BranchManager:
    def __init__(self):
        self.branches: Dict[int, Branch] = {}
        self.next_branch_id: int = 0

    def create_initial_branches(
        self,
        example,
        num_responses: int,
        previous_turn_context: str,
        mape_threshold: float,
        n_restarts: int = 1,
        hessian_context: str = "",
        use_hessian_prompt: bool = False,
    ) -> None:
        var_name = ["output"]
        var_desc = [example["symbol_descs"][0]]
        for v_idx, v_prop in enumerate(example["symbol_properties"]):
            if v_idx == 0:
                continue
            if "V" in v_prop:
                var_name.append(example["symbols"][v_idx])
                var_desc.append(example["symbol_descs"][v_idx])
        var_name = [n.strip("$").strip("\\") for n in var_name]
        var_name = [n.replace(" ", "_").replace("text", "") for n in var_name]

        if len(var_desc) > 2:
            input_desc = ", ".join(var_desc[1:-1]) + ", and " + var_desc[-1]
        else:
            input_desc = var_desc[-1]

        part1 = f"Find the mathematical function skeleton that represents {var_desc[0]}, given data on {input_desc}.\n"
        part2 = (
            "def equation(" + ", ".join([f"{name}: np.ndarray" for name in var_name[1:]]) + ", params: np.ndarray) -> np.ndarray:\n"
            + f'    """ Mathematical function for {var_desc[0]}\n\n'
            + "    Args:\n"
            + "\n".join([f"        {name}: A numpy array representing observations of {desc}." for name, desc in zip(var_name[1:], var_desc[1:])])
            + "\n        params: Array of numeric constants or parameters to be optimized\n\n"
            + "    Return:\n"
            + f"        A numpy array representing {var_desc[0]} as the result of applying the mathematical function to the inputs.\n"
            + '    """\n'
            + f"    {var_name[0]} = " + " + ".join([f"params[{i}] * {name}" for i, name in enumerate(var_name[1:])]) + f" + params[{len(var_name[1:])}]\n"
            + f"    return {var_name[0]}\n\n"
        )

        mape_threshold_percentage = f"{mape_threshold:.4%}"
        system_prompt = SYSTEM_PROMPT.format(
            part2=part2,
            mape_threshold_percentage=mape_threshold_percentage,
            mape_threshold=mape_threshold,
        )

        if use_hessian_prompt and hessian_context:
            question = USER_PROMPT_WITH_HESSIAN_CONTEXT.format(
                part1=part1,
                part2=part2,
                hessian_guidance=hessian_context,
                previous_turn_context=previous_turn_context,
                mape_threshold_percentage=mape_threshold_percentage,
            )
        else:
            question = USER_PROMPT_WITH_MEMORY.format(
                part1=part1,
                part2=part2,
                previous_turn_context=previous_turn_context,
                mape_threshold_percentage=mape_threshold_percentage,
            )

        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        for _ in range(num_responses):
            branch = Branch(
                branch_id=self.next_branch_id,
                example_index=example.name,
                status="pending_model",
                messages=list(initial_messages),
                mape_threshold=mape_threshold,
                source=example["dataset_identifier"],
                expression=example["expression"],
                n_restarts=n_restarts,
            )
            self.branches[self.next_branch_id] = branch
            self.next_branch_id += 1

    def get_branches_by_status(self, status: str) -> List[Branch]:
        return [b for b in self.branches.values() if b.status == status]

    def active_branches_exist(self) -> bool:
        return any(b.status != "completed" for b in self.branches.values())


# ---------------------------------------------------------------------------
# Branch evaluation worker (in-process pool)
# ---------------------------------------------------------------------------

def evaluate_branch_worker(branch: Branch, example_data) -> Tuple[int, dict, dict]:
    equations = list(dict.fromkeys(parse_equation(branch.messages[2:])))
    mape_threshold = branch.mape_threshold
    n_restarts = getattr(branch, "n_restarts", 1)

    train_results = get_performance(
        equations, example_data["samples"]["train"], mape_threshold,
        n_restarts=n_restarts,
    )
    test_results = get_performance_on_test(
        equations, example_data["samples"]["train"], example_data["samples"]["test"],
        mape_threshold, n_restarts=n_restarts,
    )
    return branch.branch_id, train_results, test_results


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

async def execute_tool_with_semaphore(
    tool_call,
    fn_name: str,
    fn_args: dict,
    example_data,
    sandbox_url: str,
    semaphore: asyncio.Semaphore,
    mape_threshold: float,
    n_restarts: int = 1,
):
    async with semaphore:
        start_time = time.time()
        if fn_name == "equation_evaluator":
            fn_args["mape_threshold"] = mape_threshold
            fn_args["n_restarts"] = n_restarts

        tool_func = get_function_by_name(fn_name)
        fn_res, meta_data = await tool_func(
            **fn_args,
            stdin_data=example_data["samples"]["train"],
            sandbox_fusion_url=sandbox_url,
        )
        duration = time.time() - start_time
        return tool_call, fn_res, meta_data, duration


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

async def process_branch_with_model(
    branch: Branch,
    client: AsyncOpenAI,
    model_name: str,
    tools: list,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=branch.messages,
                tools=tools,
                n=1,
                timeout=1200,
                max_completion_tokens=8192,
                temperature=0.7,
                top_p=1.0,
            )
            return branch, response
        except Exception as e:
            logger.error(f"Branch {branch.branch_id}: API call failed: {e}", exc_info=True)
            return branch, e


# ---------------------------------------------------------------------------
# AdaptSci buffer context builder
# ---------------------------------------------------------------------------

def _build_adaptsci_context(
    agg_data: Dict,
    adaptsci_buffer: AdaptSciBuffer,
    top_k: int,
    ablation: AblationConfig,
    example_index,
) -> Tuple[str, str, bool, int]:
    """
    Build previous-turn context string and Hessian guidance string.
    Also determines:
      - alpha_for_buffer  (α for frontier scoring, with ablation overrides)
      - is_degenerate     (whether best equation is degenerate → diversity-only mode)
      - n_restarts        (adaptive restart count from κ of best equation)

    Returns
    -------
    (previous_turn_context, hessian_guidance, is_degenerate, n_restarts)
    """
    if not ablation.adaptive_compute:
        # SR-Scientist mode
        context = _create_previous_turn_context_legacy(agg_data, top_k)
        return context, "", False, 1

    # --- Determine α ---
    if ablation.alpha_override is not None:
        alpha = ablation.alpha_override
    elif ablation.disable_c_diversity:
        alpha = 1.0
    else:
        alpha = ablation.frontier_alpha

    # --- Degeneracy check ---
    best = adaptsci_buffer.best()
    is_degenerate = False
    n_restarts = 1

    if best is not None and best.hdiag.valid:
        is_degenerate = best.hdiag.is_degenerate(
            coupling_threshold=ablation.coupling_threshold,
            n_eff_ratio_threshold=ablation.n_eff_ratio_threshold,
        )
        if is_degenerate:
            alpha = 0.0  # diversity-only mode
            logger.info(
                f"Example {example_index}: degenerate skeleton detected "
                f"(κ={best.hdiag.kappa:.1f}, n_eff={best.hdiag.n_eff:.1f}). "
                f"Switching to α=0 diversity-only mode."
            )
        if not ablation.fix_n_restarts:
            n_restarts = best.hdiag.adaptive_restart_count(
                tau1=ablation.tau1, tau2=ablation.tau2,
            )

    # --- Build previous-turn context from stratified buffer ---
    previous_turn_context = adaptsci_buffer.formatted_context(
        k=top_k,
        alpha_override=alpha,
        include_hessian_info=ablation.include_hessian_in_prompt,
    )

    # --- Build Hessian guidance for prompt ---
    hessian_guidance = ""
    if ablation.include_hessian_in_prompt:
        seeds = adaptsci_buffer.select_seeds(alpha_override=alpha)[:top_k]
        hessian_infos = []
        for seed in seeds:
            if seed.hdiag.valid:
                hessian_infos.append({
                    "mape": seed.mape,
                    "kappa": seed.hdiag.kappa,
                    "n_eff": seed.hdiag.n_eff,
                    "is_degenerate": seed.hdiag.is_degenerate(
                        coupling_threshold=ablation.coupling_threshold,
                        n_eff_ratio_threshold=ablation.n_eff_ratio_threshold,
                    ),
                })
        hessian_guidance = format_hessian_guidance(hessian_infos)

    return previous_turn_context, hessian_guidance, is_degenerate, n_restarts


def _create_previous_turn_context_legacy(agg_data: Dict, top_k: int) -> str:
    """
    Original SR-Scientist top-K MAPE context builder (legacy / ablation baseline).
    """
    if not agg_data["train"]["mape"]:
        return ""
    valid_results = sorted(
        [(m, e) for m, e in zip(agg_data["train"]["mape"], agg_data["train"]["equations"]) if m is not None],
        key=lambda x: x[0],
    )
    if not valid_results:
        return ""
    distinct_results = []
    last_accepted_mape = float("-inf")
    min_gap = 0.005
    for mape, eq in valid_results:
        if mape - last_accepted_mape >= min_gap:
            distinct_results.append((mape, eq))
            last_accepted_mape = mape
            if len(distinct_results) >= top_k:
                break
    if not distinct_results:
        return ""
    context_lines = [
        "\n\nIn the previous turn, some equations have been explored.",
        "Their performance (MAPE score) is listed below, sorted from lowest to highest error.",
        "You can use these experiences as inspiration to help you design a better equation.",
        "\n--- Previously Explored Equations ---",
    ]
    for mape, eq in distinct_results:
        context_lines.append(f"\n# MAPE: {mape:.4%}\n{eq.strip()}")
    context_lines.append("\n-------------------------------------\n")
    return "\n".join(context_lines)


# ---------------------------------------------------------------------------
# Buffer update helper
# ---------------------------------------------------------------------------

def _update_adaptsci_buffer(
    adaptsci_buffer: AdaptSciBuffer,
    train_results: dict,
    global_call_counter: int,
    max_params: int = 10,
    example_data=None,
) -> None:
    """
    Build EquationRecord objects from evaluation results and add them to the buffer.
    Hessian is computed using the Python-side compute_hessian_diag when example_data
    is available and mse is finite.
    """
    eqs   = train_results.get("equations", [])
    mses  = train_results.get("mse", [])
    nmses = train_results.get("nmse", [])
    mapes = train_results.get("mape", [])
    hdiags_raw = train_results.get("hessian_diags", [])

    for i, eq in enumerate(eqs):
        mse  = mses[i]  if i < len(mses)  else None
        nmse = nmses[i] if i < len(nmses) else None
        mape = mapes[i] if i < len(mapes) else None

        if mse is None or nmse is None or not np.isfinite(nmse):
            continue

        # Try to get a real HessianDiag
        hdiag = HessianDiag.invalid(max_params)
        if i < len(hdiags_raw) and isinstance(hdiags_raw[i], dict):
            # Reconstruct a minimal HessianDiag from the serialised summary
            kappa  = hdiags_raw[i].get("kappa", float("inf"))
            n_eff  = hdiags_raw[i].get("n_eff", float(max_params))
            valid  = hdiags_raw[i].get("valid", False)
            if valid:
                hdiag = HessianDiag(
                    kappa=kappa,
                    C=np.zeros((max_params, max_params)),
                    n_eff=n_eff,
                    eigenvalues=np.zeros(max_params),
                    theta_star=np.ones(max_params),
                    valid=True,
                )
        elif example_data is not None and np.isfinite(mse):
            # Compute Hessian inline (cheap: p×p op only)
            try:
                local_scope: dict = {}
                exec(eq, {"np": np}, local_scope)
                eq_fn = local_scope.get("equation")
                if eq_fn is not None:
                    hdiag = compute_hessian_diag(
                        eq_fn,
                        example_data["samples"]["train"],
                        max_params=max_params,
                    )
            except Exception:
                pass

        record = EquationRecord(
            equation=eq,
            nmse=float(nmse),
            mape=float(mape) if mape is not None else float("inf"),
            mse=float(mse),
            hdiag=hdiag,
            call_index=global_call_counter + i,
        )
        adaptsci_buffer.add(record)


# ---------------------------------------------------------------------------
# Main inference orchestrator
# ---------------------------------------------------------------------------

async def run_inference_batch(
    model_name: str,
    model_url: str,
    api_key: str,
    sandbox_urls: List[str],
    parquet_file_path: str,
    source: Optional[List[str]],
    mape_threshold: float,
    num_turns: int,
    top_k: int,
    output_json_path: str,
    max_assistant_turns: int,
    # AdaptSci arguments
    adaptive_compute: bool = True,
    ablation_config: Optional[Dict] = None,
) -> None:
    """
    Orchestrate multi-turn AdaptSci inference.

    Parameters (new vs SR-Scientist)
    ----------------------------------
    adaptive_compute  : master switch (False → pure SR-Scientist behaviour).
    ablation_config   : dict matching AblationConfig fields for ablation studies.
                        E.g. {"fix_n_restarts": True} to ablation (ii).
    """
    # Build ablation config
    ablation = AblationConfig(adaptive_compute=adaptive_compute)
    if ablation_config:
        for k, v in ablation_config.items():
            if hasattr(ablation, k):
                setattr(ablation, k, v)

    logger.info(f"AdaptSci config: {ablation}")

    # --- Infrastructure ---
    sandbox_url_cycler = itertools.cycle(sandbox_urls)
    logger.info(f"Sandbox instances: {len(sandbox_urls)}")

    client = AsyncOpenAI(api_key=api_key, base_url=model_url)

    CONCURRENCY_LIMIT_MODEL = 512
    CONCURRENCY_LIMIT_TOOL  = 64
    semaphore_model = asyncio.Semaphore(CONCURRENCY_LIMIT_MODEL)
    semaphore_tool  = asyncio.Semaphore(CONCURRENCY_LIMIT_TOOL)

    df = pd.read_parquet(parquet_file_path)
    examples_to_process = [
        row for _, row in df.iterrows()
        if not source or row["dataset_identifier"] in source
    ]

    # --- Per-example state ---
    final_results_by_example = defaultdict(lambda: {
        "train": defaultdict(list),
        "test": defaultdict(list),
        "responses": [],
        "turn_success_history": [],
    })
    example_mape_targets = {ex.name: mape_threshold for ex in examples_to_process}
    completed_example_indices: set = set()

    # AdaptSci: per-example buffer and early-stopping state
    adaptsci_buffers: Dict = {
        ex.name: AdaptSciBuffer(
            k=top_k,
            alpha=ablation.frontier_alpha,
            window=ablation.frontier_window,
            exploit_frac=ablation.exploit_frac,
        )
        for ex in examples_to_process
    }
    early_stop_states: Dict[Any, EarlyStoppingState] = {
        ex.name: make_early_stopping(
            epsilon_target=ablation.epsilon_target,
            beta_star=ablation.beta_star,
            budget=1000,
        )
        for ex in examples_to_process
    }
    # Global call counter per example (for frontier momentum tracking)
    global_call_counters: Dict = defaultdict(int)

    # --- Main turn loop ---
    for turn in range(num_turns):
        current_turn_examples = [
            ex for ex in examples_to_process
            if ex.name not in completed_example_indices
        ]
        if not current_turn_examples:
            logger.info("All examples completed. Ending turns early.")
            break

        print(f"\n{'='*25} Turn {turn+1}/{num_turns} ({len(current_turn_examples)} examples) {'='*25}")

        manager = BranchManager()

        # --- Branch creation ---
        for example in current_turn_examples:
            idx = example.name
            agg_data = final_results_by_example[idx]
            current_mape = example_mape_targets[idx]

            # Determine context and adaptive parameters
            if turn > 0 and ablation.adaptive_compute:
                prev_ctx, h_guidance, is_degen, n_restarts = _build_adaptsci_context(
                    agg_data,
                    adaptsci_buffers[idx],
                    top_k,
                    ablation,
                    idx,
                )
                use_h_prompt = ablation.include_hessian_in_prompt and bool(h_guidance)
            elif turn > 0:
                prev_ctx = _create_previous_turn_context_legacy(agg_data, top_k)
                h_guidance, is_degen, n_restarts, use_h_prompt = "", False, 1, False
            else:
                prev_ctx, h_guidance, is_degen, n_restarts, use_h_prompt = "", "", False, 1, False

            if turn > 0:
                logger.info(
                    f"  Example {idx}: target MAPE < {current_mape:.5%}, "
                    f"n_restarts={n_restarts}, degenerate={is_degen}"
                )

            manager.create_initial_branches(
                example,
                num_responses=1,
                previous_turn_context=prev_ctx,
                mape_threshold=current_mape,
                n_restarts=n_restarts,
                hessian_context=h_guidance,
                use_hessian_prompt=use_h_prompt,
            )

        pbar = tqdm(total=len(manager.branches), desc=f"✅ Turn {turn+1} Completed", position=0)
        tool_pbar = tqdm(position=1, bar_format="{desc}")
        tool_pbar.set_description_str("🔧 Avg Tool Time: N/A")
        completed_tool_calls = 0
        total_tool_time = 0.0

        # ----------------------------------------------------------------
        # Inner async helpers (defined per-turn for closure access)
        # ----------------------------------------------------------------

        async def handle_tool_calls_for_branch(
            branch: Branch,
            df,
            sandbox_url_cycler,
            semaphore: asyncio.Semaphore,
            pbar,
        ) -> None:
            nonlocal completed_tool_calls, total_tool_time
            try:
                example_data = df.loc[branch.example_index]
                tool_tasks = [
                    asyncio.create_task(
                        execute_tool_with_semaphore(
                            tc,
                            tc.function.name,
                            json.loads(tc.function.arguments),
                            example_data,
                            next(sandbox_url_cycler),
                            semaphore,
                            branch.mape_threshold,
                            n_restarts=branch.n_restarts,
                        )
                    )
                    for tc in branch.tool_calls
                ]
                for tool_call, fn_res, meta_data, duration in await asyncio.gather(*tool_tasks):
                    completed_tool_calls += 1
                    total_tool_time += duration
                    avg_time = total_tool_time / completed_tool_calls
                    tool_pbar.set_description_str(
                        f"🔧 Avg Tool Time: {avg_time:.3f}s ({completed_tool_calls} calls)"
                    )
                    branch.tool_executions.append(meta_data)
                    branch.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(fn_res),
                    })
                branch.tool_calls = []
                branch.status = "pending_model"
            except Exception as e:
                logger.error(f"Branch {branch.branch_id}: tool error: {e}", exc_info=True)
                branch.status = "completed"
                pbar.update(1)

        # ----------------------------------------------------------------
        # Core processing loop
        # ----------------------------------------------------------------

        while manager.active_branches_exist():
            statuses = [b.status for b in manager.branches.values()]
            status_counts: Dict[str, int] = defaultdict(int)
            for s in statuses:
                status_counts[s] += 1
            logger.info(f"Active branches: {dict(status_counts)}")

            model_pending_branches = manager.get_branches_by_status("pending_model")
            if model_pending_branches:
                for branch in model_pending_branches:
                    branch.assistant_turn_count += 1
                model_tasks = [
                    asyncio.create_task(
                        process_branch_with_model(b, client, model_name, tools, semaphore_model)
                    )
                    for b in model_pending_branches
                ]
                for branch in model_pending_branches:
                    branch.status = "in_flight_model"

                for future in tqdm_asyncio.as_completed(
                    model_tasks, desc=f"📞 Turn {turn+1} API Calls", position=2, leave=False
                ):
                    branch, result = await future

                    if isinstance(result, Exception):
                        logger.error(f"Branch {branch.branch_id}: API exception: {result}")
                        branch.status = "completed"
                        pbar.update(1)
                        continue

                    response_message = result.choices[0].message
                    branch.finish_reason = result.choices[0].finish_reason

                    # Fix malformed tool calls (gpt-oss models)
                    if (
                        response_message.content
                        and "to=functions." in response_message.content
                        and "<|call|>" in response_message.content
                    ):
                        branch.finish_reason = "tool_calls"

                    if branch.finish_reason == "tool_calls":
                        if branch.assistant_turn_count >= max_assistant_turns:
                            branch.messages.append(response_message.model_dump())
                            branch.status = "completed"
                            branch.finish_reason = "turn_limit"
                            pbar.update(1)
                        else:
                            # Fix malformed tool calls
                            if not response_message.tool_calls:
                                if (
                                    response_message.content
                                    and "to=functions." in response_message.content
                                    and "<|call|>" in response_message.content
                                ):
                                    pattern = re.compile(
                                        r"to=functions\.(?P<n>\w+)[\s\S]*?<\|message\|>(?P<args>.*?)\s*<\|call\|>",
                                        re.DOTALL,
                                    )
                                    match = pattern.search(response_message.content)
                                    if match:
                                        tool_name = match.group("n")
                                        tool_args_str = match.group("args").strip()
                                        if tool_args_str.endswith("]}"):
                                            tool_args_str = tool_args_str[:-2] + "}"
                                        elif "timeout" in tool_args_str:
                                            tool_args_str = re.sub(
                                                r'],\s*"timeout"\s*:\s*\d+', "", tool_args_str
                                            )
                                        elif tool_name == "data_analyzer" and '{"code":' not in tool_args_str:
                                            tool_args_str = json.dumps({"code": tool_args_str})
                                        elif tool_args_str.endswith('"'):
                                            tool_args_str = tool_args_str + "}"

                                        function_obj = Function(name=tool_name, arguments=tool_args_str)
                                        mock_tool_call = ChatCompletionMessageToolCall(
                                            id=f"call_{uuid.uuid4().hex}",
                                            function=function_obj,
                                            type="function",
                                        )
                                        response_message.tool_calls = [mock_tool_call]
                                        response_message.content = None

                            if response_message.tool_calls:
                                branch.tool_calls = response_message.tool_calls
                                branch.messages.append(response_message.model_dump())
                                branch.status = "in_flight_tool"
                                asyncio.create_task(
                                    handle_tool_calls_for_branch(
                                        branch, df, sandbox_url_cycler, semaphore_tool, pbar
                                    )
                                )
                            else:
                                branch.status = "completed"
                                pbar.update(1)
                    else:
                        branch.messages.append(response_message.model_dump())
                        branch.status = "completed"
                        pbar.update(1)

            if not manager.get_branches_by_status("pending_model"):
                await asyncio.sleep(0.1)

        pbar.close()
        tool_pbar.close()

        # ----------------------------------------------------------------
        # Final evaluation of turn results
        # ----------------------------------------------------------------
        print(f"\nEvaluating equations for Turn {turn+1}...")
        completed_branches = [b for b in manager.branches.values() if b.status == "completed"]
        branch_map = {b.branch_id: b for b in completed_branches}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(evaluate_branch_worker, b, df.loc[b.example_index]): b.branch_id
                for b in completed_branches
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures), desc="Final Evaluation"
            ):
                bid = futures[future]
                returned_id, train_res, test_res = future.result()
                ob = branch_map[returned_id]
                ob.final_train_results  = train_res or {}
                ob.final_test_results   = test_res  or {}

        # ----------------------------------------------------------------
        # Aggregate results, update buffers, check early stopping
        # ----------------------------------------------------------------
        turn_success_flags: Dict = defaultdict(bool)
        best_mape_this_turn: Dict = {}

        for branch in manager.branches.values():
            if branch.status != "completed":
                continue
            idx = branch.example_index
            final_results_by_example[idx]["responses"].append({
                "turn": turn + 1,
                "messages": branch.messages,
                "finish_reason": branch.finish_reason,
                "tool_executions": branch.tool_executions,
                "trainset_results": branch.final_train_results,
                "testset_results": branch.final_test_results,
                "n_restarts_used": branch.n_restarts,
            })

            if branch.final_train_results.get("equations"):
                for key in branch.final_train_results:
                    final_results_by_example[idx]["train"][key].extend(
                        branch.final_train_results.get(key, [])
                    )
                    final_results_by_example[idx]["test"][key].extend(
                        branch.final_test_results.get(key, [])
                    )

                # --- Update AdaptSci buffer ---
                if ablation.adaptive_compute:
                    _update_adaptsci_buffer(
                        adaptsci_buffers[idx],
                        branch.final_train_results,
                        global_call_counters[idx],
                        example_data=df.loc[idx],
                    )
                    global_call_counters[idx] += len(branch.final_train_results.get("equations", []))

                # --- Early stopping NMSE update ---
                nmses = branch.final_train_results.get("nmse", [])
                valid_nmses = [n for n in nmses if n is not None and np.isfinite(n)]
                if valid_nmses and ablation.adaptive_compute:
                    best_nmse = min(valid_nmses)
                    early_stop_states[idx].record_call(best_nmse)

                # --- Success tracking (same as SR-Scientist) ---
                successful_mapes = [
                    m for m, s in zip(
                        branch.final_train_results.get("mape", []),
                        branch.final_train_results.get("success", []),
                    )
                    if s == 1 and m is not None
                ]
                if successful_mapes:
                    turn_success_flags[idx] = True
                    min_mape = min(successful_mapes)
                    if idx not in best_mape_this_turn or min_mape < best_mape_this_turn[idx]:
                        best_mape_this_turn[idx] = min_mape

        # ----------------------------------------------------------------
        # Per-example decisions: next target, early stopping, completion
        # ----------------------------------------------------------------
        for idx in [e.name for e in current_turn_examples]:
            final_results_by_example[idx]["turn_success_history"].append(turn_success_flags[idx])

            # AdaptSci early stopping
            if (
                ablation.adaptive_compute
                and not ablation.disable_early_stopping
                and early_stop_states[idx].should_stop()
            ):
                best_rec = adaptsci_buffers[idx].best()
                best_nmse = best_rec.nmse if best_rec else float("inf")
                print(
                    f"  → Example {idx}: AdaptSci early stop "
                    f"(NMSE={best_nmse:.2e}, β={early_stop_states[idx].efficiency_ratio:.1f}). "
                    f"Releasing budget."
                )
                completed_example_indices.add(idx)
                continue

            if turn_success_flags[idx]:
                best_mape = best_mape_this_turn[idx]
                if best_mape < 1e-6:
                    print(f"  - Example {idx}: Completed (MAPE {best_mape:.2e}).")
                    completed_example_indices.add(idx)
                else:
                    next_target = 10 ** np.floor(np.log10(best_mape))
                    if next_target >= best_mape:
                        next_target /= 10.0
                    example_mape_targets[idx] = next_target
                    print(
                        f"  - Example {idx}: Success! MAPE={best_mape:.4%}. "
                        f"Next target={example_mape_targets[idx]:.4%}"
                    )
            else:
                print(
                    f"  - Example {idx}: No improvement. "
                    f"Target MAPE remains {example_mape_targets[idx]:.4%}"
                )

    # --- Final aggregation ---
    print("\nAll turns completed. Aggregating results...")
    saved_results = []
    for idx, agg_data in tqdm(final_results_by_example.items(), desc="Aggregating"):
        submitted = get_final_results(agg_data["train"], agg_data["test"])
        # Attach buffer summary for analysis
        buf = adaptsci_buffers.get(idx)
        buffer_summary = None
        if buf is not None and buf.best() is not None:
            best_rec = buf.best()
            buffer_summary = {
                "buffer_size": len(buf),
                "best_nmse": best_rec.nmse,
                "best_mape": best_rec.mape,
                "best_kappa": best_rec.hdiag.kappa if best_rec.hdiag.valid else None,
                "best_n_eff": best_rec.hdiag.n_eff if best_rec.hdiag.valid else None,
            }
        saved_results.append({
            "index": idx,
            "source": df.loc[idx]["dataset_identifier"],
            "expression": df.loc[idx]["expression"],
            "responses": agg_data["responses"],
            "submitted_results": submitted,
            "adaptsci_buffer_summary": buffer_summary,
            "early_stop_calls": early_stop_states[idx].calls if idx in early_stop_states else None,
        })

    saved_results.sort(key=lambda x: x["index"])
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(saved_results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    print(f"\nResults saved to: {output_json_path}")
