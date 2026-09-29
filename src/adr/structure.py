"""
structure.py
============
Structured solutions and the dependency-closed repair set (§3.1, §3.3, §3.4).

A structured solution is a numbered list generated in ONE call per rollout:

    1. uses: []       statement and calculation
    2. uses: []       statement and calculation
    3. uses: [1]      statement using the result of step 1
    5. uses: [3,4]    final calculation ... \\boxed{answer}

It is the DAG G = (V, E) with an edge u → v when step v uses step u.  No extra
model call builds the graph; parsing is deterministic.

Repair scopes (deterministic, O(|V|+|E|), no scores / thresholds / top-k):
  * ADR          A(j) = {j} ∪ Desc_G(j)        (dependency locality)
  * all-deps     every step uses every earlier step → A(j) = {j..m}   (ablation 1)
  * suffix       {j..m}                         (position locality, Thought-ICS-like)
The requested ADR scope is always ⊆ the suffix {j..m}.

Patches replace exactly the listed step ids; every other step is kept
byte-for-byte (its raw text block is reused verbatim when rendering).
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

# "3. uses: [1, 2] text" — tolerant of **bold**, "Step 3:", "3)" and missing colon
_STEP_RE = re.compile(
    r"^\s*(?:\*\*|__)?\s*(?:step\s*)?(\d+)\s*[\.\):]\s*(?:\*\*|__)?\s*"
    r"(?:\(|\*\*|__)?\s*uses\s*:?\s*\[([^\]]*)\]\s*(?:\)|\*\*|__)?\s*:?\s*(.*)$",
    re.IGNORECASE)


@dataclass
class Step:
    id: int
    uses: List[int]
    raw: str                    # the step's complete text block, kept verbatim

    def body(self) -> str:
        return self.raw


@dataclass
class ParseResult:
    steps: List[Step] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.steps)


def _parse_uses(s: str) -> Optional[List[int]]:
    s = s.strip()
    if not s:
        return []
    out = []
    for tok in re.split(r"[,\s]+", s):
        tok = tok.strip().lstrip("#").strip()
        if not tok:
            continue
        if not tok.isdigit():
            return None
        out.append(int(tok))
    return out


def parse_steps(text: str) -> ParseResult:
    """Parse every ``k. uses: [...] body`` block; continuation lines belong to
    the preceding step.  Text before the first step is ignored."""
    res = ParseResult()
    cur: Optional[Tuple[int, Optional[List[int]], List[str]]] = None
    blocks: List[Tuple[int, Optional[List[int]], List[str]]] = []
    for line in (text or "").splitlines():
        m = _STEP_RE.match(line)
        if m:
            if cur is not None:
                blocks.append(cur)
            cur = (int(m.group(1)), _parse_uses(m.group(2)), [line.rstrip()])
        elif cur is not None:
            cur[2].append(line.rstrip())
    if cur is not None:
        blocks.append(cur)
    for sid, uses, lines in blocks:
        while lines and not lines[-1].strip():
            lines.pop()
        if uses is None:
            res.errors.append(f"step {sid}: unparseable uses list")
            uses = []
        res.steps.append(Step(sid, uses, "\n".join(lines)))
    if not res.steps:
        res.errors.append("no numbered 'k. uses: [...]' steps found")
    return res


def validate_solution(steps: Sequence[Step]) -> List[str]:
    """A complete structured solution: ids 1..m in order, every reference points
    to an existing EARLIER step (hence a DAG)."""
    errs = []
    if not steps:
        return ["empty solution"]
    for k, st in enumerate(steps, start=1):
        if st.id != k:
            errs.append(f"step ids must be 1..m in order (found {st.id} at position {k})")
            break
    for st in steps:
        bad = [u for u in st.uses if not (1 <= u < st.id)]
        if bad:
            errs.append(f"step {st.id}: invalid references {bad} (must point to earlier steps)")
    return errs


def render(steps: Sequence[Step]) -> str:
    return "\n".join(st.raw for st in steps)


def solution_hash(steps_or_text) -> str:
    text = render(steps_or_text) if not isinstance(steps_or_text, str) else steps_or_text
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


# ── repair sets ────────────────────────────────────────────────────────────

def descendants(steps: Sequence[Step], j: int) -> Set[int]:
    """Desc_G(j): every step that directly or indirectly uses step j (BFS)."""
    children: Dict[int, List[int]] = {st.id: [] for st in steps}
    for st in steps:
        for u in st.uses:
            if u in children:
                children[u].append(st.id)
    seen: Set[int] = set()
    q = deque([j])
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def repair_set_adr(steps: Sequence[Step], j: int) -> List[int]:
    """A(j) = {j} ∪ Desc_G(j), in topological (= id) order."""
    return sorted({j} | descendants(steps, j))


def repair_set_suffix(steps: Sequence[Step], j: int) -> List[int]:
    return [st.id for st in steps if st.id >= j]


def with_all_deps(steps: Sequence[Step]) -> List[Step]:
    """Ablation 1: every step uses every earlier step (dependency info removed)."""
    return [Step(st.id, list(range(1, st.id)), st.raw) for st in steps]


# ── applying updates ─────────────────────────────────────────────────────────

def apply_patch(steps: Sequence[Step], patch_text: str, allowed: Sequence[int]) -> Tuple[Optional[List[Step]], List[str]]:
    """G ⊕ Patch: replace exactly the ``allowed`` ids, keep all others verbatim.
    The patch may revise the dependency lists of the replaced steps, provided
    every reference points to an existing earlier step."""
    pr = parse_steps(patch_text)
    if not pr.steps:
        return None, pr.errors or ["empty patch"]
    got: Dict[int, Step] = {}
    for st in pr.steps:
        if st.id in got:
            return None, [f"patch repeats step {st.id}"]
        got[st.id] = st
    want = set(allowed)
    if set(got) != want:
        extra, missing = sorted(set(got) - want), sorted(want - set(got))
        return None, [f"patch ids mismatch: missing={missing} extra={extra}"]
    if pr.errors:
        return None, pr.errors
    new = [got[st.id] if st.id in want else st for st in steps]
    errs = validate_solution(new)
    return (None, errs) if errs else (new, [])


def apply_suffix(steps: Sequence[Step], suffix_text: str, j: int) -> Tuple[Optional[List[Step]], List[str]]:
    """Keep steps 1..j-1 verbatim, replace everything from j on with newly
    generated steps numbered j, j+1, ... (any count)."""
    pr = parse_steps(suffix_text)
    if not pr.ok:
        return None, pr.errors
    if pr.steps[0].id != j:
        return None, [f"suffix must start at step {j} (starts at {pr.steps[0].id})"]
    new = [st for st in steps if st.id < j] + list(pr.steps)
    errs = validate_solution(new)
    return (None, errs) if errs else (new, [])


def parse_full(text: str) -> Tuple[Optional[List[Step]], List[str]]:
    pr = parse_steps(text)
    if not pr.ok:
        return None, pr.errors
    errs = validate_solution(pr.steps)
    return (None, errs) if errs else (pr.steps, [])


__all__ = ["Step", "ParseResult", "parse_steps", "validate_solution", "render", "solution_hash",
           "descendants", "repair_set_adr", "repair_set_suffix", "with_all_deps",
           "apply_patch", "apply_suffix", "parse_full"]
