"""
prompts.py
==========
Prompt contracts for Adaptive Delta Refinement (the same for every model; only
the chat template differs).  Roles use the same frozen model.

  structured generation  — N independent numbered solutions with dependency ids (§3.1)
  critic                 — KEEP / FIX(step, error) / REBUILD(reason) on ONE rollout (§3.2)
                           (never sees other rollouts' answers or the majority)
  patch                  — replacements for exactly the repair-set ids (§3.4)
  suffix                 — regenerate steps j..end, keep 1..j-1 (suffix baseline)
  full rewrite           — a new complete structured solution (REBUILD / fallback /
                           full-rewrite baseline)

The original CIMR baseline keeps using the paper's prompts in src/cimr_prompts.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

FORMAT_RULES = (
    "Write the solution as a numbered list of steps, one line per step, in exactly this format:\n"
    "1. uses: []  <one calculation or deduction>\n"
    "2. uses: []  <...>\n"
    "3. uses: [1]  <a step that uses the result of step 1>\n"
    "4. uses: [2, 3]  <a step that uses the results of steps 2 and 3>\n"
    "Rules:\n"
    "- Each step contains ONE meaningful calculation or deduction.\n"
    "- `uses` lists EVERY earlier step whose result this step uses; use [] when the step only "
    "uses information from the question.\n"
    "- References may only point to earlier steps.\n"
    "- The last step states the final answer as \\boxed{<answer>}.\n"
    "- Write nothing before step 1 and nothing after the last step."
)

GEN_SYSTEM = "You are a precise mathematical reasoner who writes solutions as numbered, dependency-annotated steps."

CRITIC_SYSTEM = (
    "You are a careful mathematical critic. You check one numbered solution step by step. "
    "You do not solve the problem from scratch."
)

PATCH_SYSTEM = (
    "You are a careful mathematical editor. You repair only the steps you are told to repair and "
    "leave every other step untouched."
)


def generation_prompt(question: str):
    return GEN_SYSTEM, (
        f"PROBLEM\n{question.strip()}\n\n{FORMAT_RULES}\n\nSOLUTION:\n"
    )


def critique_prompt(question: str, solution: str):
    return CRITIC_SYSTEM, (
        f"PROBLEM\n{question.strip()}\n\n"
        f"NUMBERED SOLUTION\n{solution.strip()}\n\n"
        "Check every step: (a) is its calculation or deduction mathematically correct, and "
        "(b) does its `uses` list include every earlier step whose result it relies on?\n"
        "Then end with exactly ONE decision line, chosen from:\n"
        "DECISION: KEEP\n"
        "DECISION: FIX <step number> | <the concrete error in that step>\n"
        "DECISION: REBUILD | <why the approach or the step decomposition must change globally>\n"
        "Use FIX for the FIRST faulty step (a wrong calculation, a wrong deduction, or a missing "
        "dependency). Use REBUILD only when the overall approach is wrong. Use KEEP when you find no "
        "actionable error.\n\nCRITIQUE:\n"
    )


def patch_prompt(question: str, solution: str, error: str, allowed: Sequence[int]):
    ids = ", ".join(str(i) for i in allowed)
    return PATCH_SYSTEM, (
        f"PROBLEM\n{question.strip()}\n\n"
        f"CURRENT NUMBERED SOLUTION\n{solution.strip()}\n\n"
        f"DETECTED ERROR (step {allowed[0]}): {error.strip()}\n\n"
        f"Rewrite ONLY steps {ids}. All other steps stay exactly as they are and you must not output them.\n"
        "- Output the replacement steps in increasing order, one per line, in the same format "
        "`k. uses: [...]  <calculation>`.\n"
        "- Keep each step's number and its purpose; fix the error and recompute every value that "
        "depended on it.\n"
        "- `uses` may change but may only reference earlier steps.\n"
        "- If the last step is among them, it must end with \\boxed{<answer>}.\n\n"
        f"REPLACEMENT STEPS {ids}:\n"
    )


def suffix_prompt(question: str, prefix: str, error: str, j: int):
    kept = prefix.strip() if prefix.strip() else "(no earlier steps)"
    return PATCH_SYSTEM, (
        f"PROBLEM\n{question.strip()}\n\n"
        f"KEPT STEPS 1..{j - 1}\n{kept}\n\n"
        f"DETECTED ERROR in step {j}: {error.strip()}\n\n"
        f"Continue the solution from step {j} onward: output new steps numbered {j}, {j + 1}, ... in the "
        "same format `k. uses: [...]  <calculation>`, fixing the error, until the last step gives "
        "\\boxed{<answer>}. Do not repeat the kept steps.\n\n"
        f"STEPS FROM {j}:\n"
    )


def rewrite_prompt(question: str, solution: str, reason: str):
    return GEN_SYSTEM, (
        f"PROBLEM\n{question.strip()}\n\n"
        f"PREVIOUS SOLUTION\n{solution.strip() or '(none)'}\n\n"
        f"PROBLEM FOUND: {reason.strip() or 'the previous solution is not in the required format'}\n\n"
        f"Write a complete new solution.\n{FORMAT_RULES}\n\nSOLUTION:\n"
    )


# ── critic output ──────────────────────────────────────────────────────────

@dataclass
class Feedback:
    kind: str                  # KEEP | FIX | REBUILD | INVALID
    step: int = -1
    text: str = ""
    raw: str = ""


_DEC = re.compile(r"DECISION\s*:\s*\**\s*(KEEP|FIX|REBUILD)\b(.*)$", re.IGNORECASE | re.MULTILINE)


def parse_feedback(raw: str) -> Feedback:
    """Parse the LAST decision line.  Accepts `FIX 3 | msg`, `FIX(3, msg)`,
    `FIX step 3: msg`, `REBUILD(reason)` / `REBUILD | reason`."""
    last = None
    for m in _DEC.finditer(raw or ""):
        last = m
    if last is None:
        return Feedback("INVALID", raw=raw or "")
    kind = last.group(1).upper()
    rest = last.group(2).strip().strip("()").strip()
    if kind == "KEEP":
        return Feedback("KEEP", raw=raw)
    if kind == "REBUILD":
        return Feedback("REBUILD", text=rest.lstrip("|:,- ").strip(), raw=raw)
    m = re.match(r"(?:step\s*)?#?\s*(\d+)\s*[|:,\-)]?\s*(.*)", rest, re.IGNORECASE | re.DOTALL)
    if not m:
        return Feedback("INVALID", raw=raw)
    return Feedback("FIX", step=int(m.group(1)), text=m.group(2).strip().strip("|").strip(), raw=raw)


__all__ = ["generation_prompt", "critique_prompt", "patch_prompt", "suffix_prompt", "rewrite_prompt",
           "Feedback", "parse_feedback", "FORMAT_RULES"]
