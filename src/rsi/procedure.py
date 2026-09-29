"""
procedure.py
============
Editable procedure H_t = (P_t, R_t, M_t) (§3).

  P_t  prompts   — critique / correction instructions only (EDITABLE_PROMPT_KEYS)
  R_t  items     — ordered list of conditional rules and guarded two-slot plans
  M_t  evidence  — lives in ``memory.EvidenceMemory`` (referenced by hash)

Controller semantics (§4.3)
---------------------------
* Items are scanned in order; the FIRST item that yields a legal, affordable
  action is taken.  Slot ties use fixed slot order.
* Rule   : ``when`` guard on the fixed feature schema → one primitive action.
           scope "slot" (FULL_REFINE | DIRECT_REPAIR | RESTART | KEEP) or
           scope "global" (STOP).
* Plan   : start guards select two DISTINCT eligible slots (first, second);
           first op → continuation guard (checked on the second slot AFTER the
           first op) → second op.  Budget for BOTH ops is reserved up front.
* No item matches → H0's next affordable action: FULL_REFINE on the
  shallowest unfrozen slot (lowest index) below ``h0_depth``; else STOP.
  H0 therefore reproduces the paper's round-robin schedule.
* KEEP freezes a slot (no zero-cost loops); a decision cap guarantees
  termination (enforced by the solver).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .config import PAIR_OPS, PRIMITIVE_OPS
from .prompts import DEFAULT_PROMPTS, EDITABLE_PROMPT_KEYS
from .state import CAT_OPS, CRITIQUE_CATEGORIES, FEATURE_SCHEMA, NUM_OPS, SolveState, match_all

SLOT_ACTIONS = ("FULL_REFINE", "DIRECT_REPAIR", "RESTART", "KEEP")


@dataclass
class Rule:
    id: str
    when: List[list]
    action: str
    scope: str = "slot"          # slot | global
    note: str = ""
    origin: str = ""             # H0 | compiler | llm | manual
    kind: str = "rule"


@dataclass
class Plan:
    id: str
    first_guard: List[list]
    second_guard: List[list]
    first_op: str
    second_op: str
    continuation_guard: List[list] = field(default_factory=list)
    note: str = ""
    origin: str = ""
    kind: str = "plan"


Item = Union[Rule, Plan]


@dataclass
class Decision:
    action: str                          # FULL_REFINE | DIRECT_REPAIR | RESTART | KEEP | STOP | PAIR
    slot: int = -1
    slot2: int = -1
    plan: Optional[Plan] = None
    item_id: str = "H0"
    forced: bool = False

    def key(self) -> str:
        if self.action == "STOP":
            return "STOP"
        if self.action == "PAIR":
            p = self.plan
            return f"PAIR({p.first_op}@{self.slot}->{p.second_op}@{self.slot2})" if p else "PAIR"
        return f"{self.action}@{self.slot}"


@dataclass
class Procedure:
    prompts: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PROMPTS))
    items: List[Item] = field(default_factory=list)
    h0_depth: int = 2
    version: int = 0
    parent: str = ""
    lineage: List[Dict[str, Any]] = field(default_factory=list)

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dict(prompts=dict(self.prompts), items=[asdict(it) for it in self.items],
                    h0_depth=self.h0_depth, version=self.version, parent=self.parent,
                    lineage=list(self.lineage), hash=self.hash())

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Procedure":
        items: List[Item] = []
        for it in d.get("items", []):
            it = dict(it)
            kind = it.pop("kind", "rule")
            items.append(Plan(**it) if kind == "plan" else Rule(**it))
        prompts = dict(DEFAULT_PROMPTS)
        prompts.update(d.get("prompts", {}))
        return Procedure(prompts=prompts, items=items, h0_depth=int(d.get("h0_depth", 2)),
                         version=int(d.get("version", 0)), parent=d.get("parent", ""),
                         lineage=list(d.get("lineage", [])))

    def canonical(self) -> str:
        core = dict(prompts=self.prompts, items=[asdict(it) for it in self.items], h0_depth=self.h0_depth)
        return json.dumps(core, sort_keys=True)

    def hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:12]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "Procedure":
        with open(path, "r", encoding="utf-8") as f:
            return Procedure.from_dict(json.load(f))

    def copy(self) -> "Procedure":
        return Procedure.from_dict(json.loads(json.dumps(self.to_dict())))

    def n_rules(self) -> int:
        return sum(1 for it in self.items if isinstance(it, Rule))

    def n_plans(self) -> int:
        return sum(1 for it in self.items if isinstance(it, Plan))

    # -- controller ---------------------------------------------------------
    def choose(self, st: SolveState, afford: Callable[[str, SolveState, int], bool],
               allow_plans: bool = True, disabled: Tuple[str, ...] = ()) -> Decision:
        """Pick the next decision.  ``afford(op, state, slot)`` → bool,
        ``afford("PAIR2", ...)`` is emulated via ``pair_affordable``."""
        eligible = st.eligible_slots()
        g = st.global_features()
        feats = {i: st.slot_features(i, g) for i in eligible}
        for it in self.items:
            if isinstance(it, Rule):
                if it.action in disabled:
                    continue
                if it.scope == "global":
                    if it.action == "STOP" and match_all(g, it.when):
                        return Decision("STOP", item_id=it.id)
                    continue
                for i in eligible:
                    if match_all(feats[i], it.when) and afford(it.action, st, i):
                        return Decision(it.action, slot=i, item_id=it.id)
            elif isinstance(it, Plan) and allow_plans and "PAIR" not in disabled:
                if it.first_op in disabled or it.second_op in disabled:
                    continue
                for i in eligible:
                    if not match_all(feats[i], it.first_guard):
                        continue
                    for j in eligible:
                        if j == i or not match_all(feats[j], it.second_guard):
                            continue
                        if afford(("PAIR", it.first_op, it.second_op), st, (i, j)):
                            return Decision("PAIR", slot=i, slot2=j, plan=it, item_id=it.id)
        return self.h0_next(st, afford)

    def h0_next(self, st: SolveState, afford) -> Decision:
        cands = [s for s in st.slots if not s.frozen and s.depth < self.h0_depth]
        if not cands:
            return Decision("STOP", item_id="H0:done")
        s = min(cands, key=lambda s: (s.depth, s.idx))
        if afford("FULL_REFINE", st, s.idx):
            return Decision("FULL_REFINE", slot=s.idx, item_id="H0")
        return Decision("STOP", item_id="H0:budget")


# =============================================================================
# Validation (restricted editor contract)
# =============================================================================

def validate_conditions(conds, scope_ok=("slot", "global")) -> List[str]:
    errs = []
    if not isinstance(conds, list):
        return ["guard must be a list of [feature, op, value]"]
    for c in conds:
        if not (isinstance(c, (list, tuple)) and len(c) == 3):
            errs.append(f"bad condition {c!r}")
            continue
        name, op, val = c
        if name not in FEATURE_SCHEMA:
            errs.append(f"unknown feature {name!r}")
            continue
        scope, typ = FEATURE_SCHEMA[name]
        if scope not in scope_ok:
            errs.append(f"feature {name!r} not allowed here")
        if typ == "num":
            if op not in NUM_OPS:
                errs.append(f"bad numeric op {op!r}")
            elif not isinstance(val, (int, float)) or isinstance(val, bool):
                errs.append(f"numeric feature {name!r} needs a number")
        else:
            if op not in CAT_OPS:
                errs.append(f"bad categorical op {op!r}")
            vals = val if op in ("in", "not_in") else [val]
            if not isinstance(vals, list) or any(v not in CRITIQUE_CATEGORIES for v in vals):
                errs.append(f"bad critique category value {val!r}")
    return errs


def validate_item(it: Item) -> List[str]:
    if isinstance(it, Rule):
        errs = []
        if it.scope == "global":
            if it.action != "STOP":
                errs.append("global rules may only STOP")
            errs += validate_conditions(it.when, ("global",))
            if not it.when:
                errs.append("an unconditional STOP rule is not allowed")
        elif it.scope == "slot":
            if it.action not in SLOT_ACTIONS:
                errs.append(f"slot action must be one of {SLOT_ACTIONS}")
            errs += validate_conditions(it.when)
        else:
            errs.append("scope must be slot|global")
        return errs
    errs = []
    if it.first_op not in PAIR_OPS or it.second_op not in PAIR_OPS:
        errs.append(f"plan ops must be in {PAIR_OPS}")
    errs += validate_conditions(it.first_guard)
    errs += validate_conditions(it.second_guard)
    errs += validate_conditions(it.continuation_guard)
    return errs


def validate_procedure(p: Procedure, max_rules: int, max_plans: int, max_prompt_chars: int) -> List[str]:
    errs: List[str] = []
    ids = [it.id for it in p.items]
    if len(ids) != len(set(ids)):
        errs.append("duplicate item ids")
    for it in p.items:
        errs += [f"{it.id}: {e}" for e in validate_item(it)]
    if p.n_rules() > max_rules:
        errs.append(f"rule budget exceeded ({p.n_rules()} > {max_rules})")
    if p.n_plans() > max_plans:
        errs.append(f"plan budget exceeded ({p.n_plans()} > {max_plans})")
    for k, v in p.prompts.items():
        if k not in EDITABLE_PROMPT_KEYS:
            errs.append(f"prompt key {k!r} is not editable")
        if len(v or "") > max(max_prompt_chars, len(DEFAULT_PROMPTS.get(k, "")) + max_prompt_chars):
            errs.append(f"prompt {k!r} exceeds the declared prompt budget")
    return errs


# =============================================================================
# Built-in procedures
# =============================================================================

def h0_procedure(h0_depth: int) -> Procedure:
    """The paper's fixed round-robin refinement schedule."""
    return Procedure(items=[], h0_depth=int(h0_depth), version=0)


def heuristic_procedure(h0_depth: int) -> Procedure:
    """Fixed hand-written adaptive baseline (§8 'fixed heuristic adaptive refinement').

    Not learned; used only as a baseline.  The first rule is the §6 example
    hypothesis; the others are common-sense allocation heuristics.
    """
    items: List[Item] = [
        Rule("heur_stop_unanimous", [["top_frac", ">=", 1.0], ["invalid_frac", "==", 0.0]],
             "STOP", scope="global", origin="manual", note="all slots agree"),
        Rule("heur_restart_unstable",
             [["answer_changes", ">=", 2], ["critique", "==", "no_specific_error"], ["budget_frac", ">=", 0.3]],
             "RESTART", origin="manual", note="§6 example rule hypothesis"),
        Rule("heur_restart_invalid", [["valid", "==", 0], ["budget_frac", ">=", 0.2]], "RESTART",
             origin="manual"),
        Rule("heur_keep_confirmed",
             [["critique", "==", "no_specific_error"], ["agrees_plurality", "==", 1], ["depth", ">=", 1]],
             "KEEP", origin="manual"),
        Rule("heur_repair_minority", [["agrees_plurality", "==", 0], ["depth", "<", h0_depth + 1]],
             "DIRECT_REPAIR", origin="manual"),
    ]
    return Procedure(items=items, h0_depth=int(h0_depth), version=0)


__all__ = [
    "Rule", "Plan", "Item", "Decision", "Procedure", "SLOT_ACTIONS",
    "validate_conditions", "validate_item", "validate_procedure",
    "h0_procedure", "heuristic_procedure",
]
