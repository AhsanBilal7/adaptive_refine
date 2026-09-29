"""
editor.py
=========
Restricted procedure editor (§6 step 2; §9 module 4).

A patch is ONE of:
  add_rule | revise_rule | delete_item | add_plan | edit_prompt
and must name its supporting and contradicting development evidence ids.

Two proposers share the same patch format and validation:
  compiler — deterministic: turns reproducible evidence groups into rules /
             guarded two-slot plans / deletions (plan selection mode:
             interaction | random | greedy_vote, §8 ablation).
  llm      — the SAME frozen model, prompted with the procedure, the fixed
             feature schema and the binned memory summary.

The editor cannot redefine correctness, read test labels, alter vote weights,
or exempt calls from accounting: none of those are representable in a patch.
Rules may only use the fixed feature schema; prompt patches that contain
answer lists, problem text, or lookup instructions are rejected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .backends import GenRequest, GenerationBackend
from .config import PAIR_OPS, RSIConfig
from .memory import (EvidenceMemory, agree_bin_conditions, budget_bin_conditions, describe_slot_bin,
                     slot_bin_conditions)
from .procedure import Plan, Procedure, Rule, SLOT_ACTIONS, validate_procedure
from .prompts import EDITABLE_PROMPT_KEYS
from .state import CRITIQUE_CATEGORIES, FEATURE_SCHEMA

logger = logging.getLogger(__name__)

PATCH_TYPES = ("add_rule", "revise_rule", "delete_item", "add_plan", "edit_prompt")


@dataclass
class Patch:
    type: str
    source: str = "compiler"
    rule: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    target_id: str = ""
    position: int = 0
    prompt_key: str = ""
    prompt_text: str = ""
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def pid(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:10]


# =============================================================================
# Leakage guard for prompt patches
# =============================================================================

_LEAK_PAT = re.compile(
    r"problem\s*(id|#|number|no\.)|question\s*(id|#)|answer\s*key|look\s*up|lookup|"
    r"the\s+(correct\s+)?answer\s+(to\s+(problem|question)|is)\b|benchmark|dataset|memori[sz]e",
    re.IGNORECASE)


class LeakGuard:
    def __init__(self, dev_questions: Sequence[str], dev_golds: Sequence[str]):
        self.q_ngrams = set()
        for q in dev_questions:
            w = re.findall(r"\w+", q.lower())
            for i in range(max(0, len(w) - 7)):
                self.q_ngrams.add(" ".join(w[i:i + 8]))
        self.golds = {str(g).strip() for g in dev_golds if g is not None and len(str(g).strip()) >= 3}

    def check(self, text: str) -> List[str]:
        errs = []
        if _LEAK_PAT.search(text or ""):
            errs.append("prompt patch contains problem-specific lookup / answer language")
        w = re.findall(r"\w+", (text or "").lower())
        for i in range(max(0, len(w) - 7)):
            if " ".join(w[i:i + 8]) in self.q_ngrams:
                errs.append("prompt patch copies development problem text")
                break
        nums = re.findall(r"\d+(?:\.\d+)?", text or "")
        if len([n for n in nums if len(n) >= 2]) > 4:
            errs.append("prompt patch contains a list of numbers (possible answer list)")
        toks = set(re.findall(r"[^\s,;]+", text or ""))
        if self.golds & toks:
            errs.append("prompt patch contains development gold answers")
        return errs


# =============================================================================
# Applying patches
# =============================================================================

def _as_rule(d: Dict[str, Any], origin: str) -> Rule:
    return Rule(id=str(d.get("id") or ""), when=[list(c) for c in d.get("when", [])],
                action=str(d.get("action", "")), scope=str(d.get("scope", "slot")),
                note=str(d.get("note", ""))[:200], origin=origin)


def _as_plan(d: Dict[str, Any], origin: str) -> Plan:
    return Plan(id=str(d.get("id") or ""), first_guard=[list(c) for c in d.get("first_guard", [])],
                second_guard=[list(c) for c in d.get("second_guard", [])],
                first_op=str(d.get("first_op", "")), second_op=str(d.get("second_op", "")),
                continuation_guard=[list(c) for c in d.get("continuation_guard", [])],
                note=str(d.get("note", ""))[:200], origin=origin)


def apply_patch(base: Procedure, patch: Patch, cfg: RSIConfig, guard: Optional[LeakGuard],
                known_eids: Optional[set] = None) -> Tuple[Optional[Procedure], List[str]]:
    errs: List[str] = []
    allowed = set(cfg.list_of("edit_types"))
    kind = {"add_rule": "rule", "revise_rule": "rule", "delete_item": "delete",
            "add_plan": "plan", "edit_prompt": "prompt"}.get(patch.type)
    if kind is None:
        return None, [f"unknown patch type {patch.type!r}"]
    if kind not in allowed:
        return None, [f"patch type {patch.type} disabled by edit_types={cfg.edit_types}"]
    if kind == "plan" and not cfg.allow_plans:
        return None, ["plans disabled (primitive-only ablation)"]
    if not patch.supporting_evidence:
        errs.append("patch must name supporting evidence")
    if known_eids is not None:
        bad = [e for e in patch.supporting_evidence + patch.contradicting_evidence if e not in known_eids]
        if bad:
            errs.append(f"unknown evidence ids {bad[:5]}")
    new = base.copy()
    pos = max(0, min(int(patch.position or 0), len(new.items)))
    if patch.type in ("add_rule", "revise_rule"):
        r = _as_rule(patch.rule or {}, patch.source)
        if not r.id:
            r.id = f"r_{patch.pid()}"
        if patch.type == "revise_rule":
            idx = [i for i, it in enumerate(new.items) if it.id == (patch.target_id or r.id)]
            if not idx:
                return None, errs + [f"revise target {patch.target_id!r} not found"]
            r.id = new.items[idx[0]].id
            new.items[idx[0]] = r
        else:
            if any(it.id == r.id for it in new.items):
                r.id = f"{r.id}_{patch.pid()[:4]}"
            new.items.insert(pos, r)
    elif patch.type == "add_plan":
        p = _as_plan(patch.plan or {}, patch.source)
        if not p.id:
            p.id = f"p_{patch.pid()}"
        if any(it.id == p.id for it in new.items):
            p.id = f"{p.id}_{patch.pid()[:4]}"
        new.items.insert(pos, p)
    elif patch.type == "delete_item":
        before = len(new.items)
        new.items = [it for it in new.items if it.id != patch.target_id]
        if len(new.items) == before:
            return None, errs + [f"delete target {patch.target_id!r} not found"]
    elif patch.type == "edit_prompt":
        if patch.prompt_key not in EDITABLE_PROMPT_KEYS:
            return None, errs + [f"prompt key {patch.prompt_key!r} not editable"]
        if guard is not None:
            errs += guard.check(patch.prompt_text)
        new.prompts[patch.prompt_key] = patch.prompt_text.strip()
    errs += validate_procedure(new, cfg.max_rules, cfg.max_plans, cfg.max_prompt_chars)
    if errs:
        return None, errs
    new.version = base.version + 1
    new.parent = base.hash()
    return new, []


# =============================================================================
# Deterministic evidence compiler
# =============================================================================

def compile_patches(memory: EvidenceMemory, proc: Procedure, cfg: RSIConfig, rng: random.Random,
                    limit: int) -> List[Patch]:
    patches: List[Patch] = []
    credit = cfg.credit_mode
    existing = {json.dumps([getattr(it, "when", None), getattr(it, "action", None)], sort_keys=True)
                for it in proc.items}

    # -- single-action rules ------------------------------------------------
    sg = memory.single_groups(credit)
    for (op, sb, bb), st in sorted(sg.items(), key=lambda kv: -kv[1]["mean"] * kv[1]["n"] ** 0.5):
        if op in cfg.list_of("disabled_actions"):
            continue
        if st["n_states"] < cfg.min_support or st["mean"] < cfg.rule_delta_min:
            continue
        if st["n_pos"] < st["n_neg"] or st["n_pos"] == 0:
            continue
        when = slot_bin_conditions(sb) + budget_bin_conditions(bb)
        if op in ("FULL_REFINE", "DIRECT_REPAIR"):
            when.append(["depth", "<", proc.h0_depth + 2])       # bounded repeat
        if json.dumps([when, op], sort_keys=True) in existing:
            continue
        contra = [e for e in _contradicting(memory, op, sb, bb, credit)][:8]
        patches.append(Patch("add_rule", "compiler",
                             rule=dict(when=when, action=op, scope="slot",
                                       note=f"{op} when {describe_slot_bin(sb)}, budget={bb}"),
                             supporting_evidence=st["eids"][:12], contradicting_evidence=contra,
                             rationale=f"mean Δ={st['mean']:+.3f} over n={st['n']} "
                                       f"(+{st['n_pos']}/0:{st['n_zero']}/-{st['n_neg']})"))
    # -- STOP rules -----------------------------------------------------------
    if "STOP" not in cfg.list_of("disabled_actions"):
        for (_, ab, bb), st in sorted(memory.stop_groups().items(), key=lambda kv: -kv[1]["mean"]):
            if st["n_states"] < cfg.min_support or st["mean"] < cfg.rule_delta_min or st["n_pos"] < st["n_neg"]:
                continue
            when = agree_bin_conditions(ab) + budget_bin_conditions(bb)
            patches.append(Patch("add_rule", "compiler",
                                 rule=dict(when=when, action="STOP", scope="global",
                                           note=f"STOP when votes={ab}, budget={bb}"),
                                 supporting_evidence=st["eids"][:12],
                                 rationale=f"mean Δ={st['mean']:+.3f} over n={st['n']}"))
    # -- guarded two-slot plans -----------------------------------------------
    if cfg.allow_plans:
        patches += _plan_patches(memory, cfg, rng)
    # -- deletions --------------------------------------------------------------
    for item_id, st in memory.item_regret().items():
        if not any(it.id == item_id for it in proc.items):
            continue
        if st["n"] >= cfg.min_support and st["mean_best_alt_delta"] > cfg.rule_delta_min \
                and st["frac_alt_better"] >= 0.5:
            patches.append(Patch("delete_item", "compiler", target_id=item_id,
                                 supporting_evidence=st["eids"],
                                 rationale=f"alternatives beat {item_id} by {st['mean_best_alt_delta']:+.3f}"))
    return patches[:limit]


def _contradicting(memory: EvidenceMemory, op, sb, bb, credit) -> List[str]:
    out = []
    for k, st in memory.single_groups(credit).items():
        if k[0] == op and (k[1] != sb or k[2] != bb) and st["mean"] < 0:
            out.extend(st["eids"][:3])
    return out


def _plan_patches(memory: EvidenceMemory, cfg: RSIConfig, rng: random.Random) -> List[Patch]:
    pg = memory.pair_groups()
    items = list(pg.items())
    chosen: List[Tuple[Tuple, Dict[str, Any]]] = []
    if cfg.plan_selection == "interaction":
        for k, st in sorted(items, key=lambda kv: -kv[1]["mean_I"]):
            if (st["n_states"] >= cfg.min_support and st["mean_I"] > 0 and st["frac_I_pos"] >= cfg.pair_repro_frac
                    and st["mean_I_acc"] > 0 and st["mean_util_gain"] > 0):
                chosen.append((k, st))
    elif cfg.plan_selection == "greedy_vote":
        for k, st in sorted(items, key=lambda kv: -kv[1]["mean_imm_vote_gain"]):
            if st["n_states"] >= cfg.min_support and st["mean_imm_vote_gain"] > 0:
                chosen.append((k, st))
    else:  # random control: equally many plans, chosen without looking at effects
        n_target = sum(1 for _, st in items if st["n_states"] >= cfg.min_support and st["mean_I"] > 0
                       and st["frac_I_pos"] >= cfg.pair_repro_frac)
        pool = [kv for kv in items if kv[1]["n_states"] >= 1]
        rng.shuffle(pool)
        chosen = pool[:max(n_target, 1)] if pool else []
    patches = []
    for (a, b, sa, sb), st in chosen:
        cont = [["agrees_plurality", "==", 0]] if (sb[0] and not sb[1]) else []
        if not sb[0]:
            cont = [["valid", "==", 0]]
        plan = dict(first_guard=slot_bin_conditions(sa), second_guard=slot_bin_conditions(sb),
                    first_op=a, second_op=b, continuation_guard=cont,
                    note=f"{a} on [{describe_slot_bin(sa)}] then {b} on [{describe_slot_bin(sb)}]")
        patches.append(Patch("add_plan", "compiler", plan=plan, supporting_evidence=st["eids"][:12],
                             rationale=f"{cfg.plan_selection}: mean I={st['mean_I']:+.3f} "
                                       f"frac_I>0={st['frac_I_pos']:.2f} I_acc={st['mean_I_acc']:+.3f} "
                                       f"util_gain={st['mean_util_gain']:+.3f} n={st['n']}"))
    return patches


# =============================================================================
# Frozen-LLM editor
# =============================================================================

EDITOR_SYSTEM = (
    "You edit a decision procedure that allocates compute during mathematical self-refinement. "
    "You never see problems or answers. You propose exactly ONE small patch as JSON, grounded in the "
    "measured evidence you are given. Output only the JSON object."
)


def _schema_text() -> str:
    feats = "\n".join(f"  {k}: {v[0]}-level {'number' if v[1] == 'num' else 'category'}"
                      for k, v in FEATURE_SCHEMA.items())
    return (f"FEATURES (the only names allowed in guards):\n{feats}\n"
            f"critique categories: {', '.join(CRITIQUE_CATEGORIES)}\n"
            "numeric ops: < <= > >= == !=   categorical ops: == != in not_in\n"
            f"slot actions: {', '.join(SLOT_ACTIONS)}; global action: STOP; plan ops: {', '.join(PAIR_OPS)}\n"
            f"editable prompt keys: {', '.join(EDITABLE_PROMPT_KEYS)}")


_PATCH_FORMAT = """PATCH FORMAT (choose one type):
{"type": "add_rule", "position": 0, "rule": {"id": "short_id", "when": [["feature", "op", value], ...], "action": "RESTART", "scope": "slot", "note": "..."}, "supporting_evidence": ["E12"], "contradicting_evidence": ["E40"], "rationale": "..."}
{"type": "revise_rule", "target_id": "existing_id", "rule": {...same as add_rule...}, "supporting_evidence": [...], "contradicting_evidence": [...], "rationale": "..."}
{"type": "delete_item", "target_id": "existing_id", "supporting_evidence": [...], "contradicting_evidence": [...], "rationale": "..."}
{"type": "add_plan", "position": 0, "plan": {"id": "short_id", "first_guard": [[...]], "second_guard": [[...]], "first_op": "DIRECT_REPAIR", "second_op": "DIRECT_REPAIR", "continuation_guard": [["agrees_plurality", "==", 0]], "note": "..."}, "supporting_evidence": [...], "contradicting_evidence": [...], "rationale": "..."}
{"type": "edit_prompt", "prompt_key": "critique_extra", "prompt_text": "general checking instruction (no numbers, no problem text)", "supporting_evidence": [...], "contradicting_evidence": [...], "rationale": "..."}
Global STOP rules use "scope": "global" and only global features."""


def build_editor_prompt(proc: Procedure, memory: EvidenceMemory, cfg: RSIConfig) -> str:
    cur = proc.to_dict()
    cur.pop("lineage", None)
    return (
        "CURRENT PROCEDURE (items are checked in order; first legal match acts; otherwise the default "
        f"round-robin FULL_REFINE up to depth {proc.h0_depth}):\n"
        f"{json.dumps(dict(items=cur['items'], prompts=cur['prompts']), indent=1)[:6000]}\n\n"
        f"{_schema_text()}\n\n"
        f"MEASURED EVIDENCE (development data, same-state comparisons):\n"
        f"{memory.summary_text(cfg.credit_mode)[:6000]}\n\n"
        f"CONSTRAINTS: at most {cfg.max_rules} rules and {cfg.max_plans} plans; prompt text ≤ "
        f"{cfg.max_prompt_chars} chars; allowed patch types: {cfg.edit_types}. Name supporting AND "
        "contradicting evidence ids from the list above. Prefer patches whose evidence shows reproducible "
        "gains; two-slot plans need positive interaction I.\n\n"
        f"{_PATCH_FORMAT}\n\nJSON patch:"
    )


def _lenient_json(s: str) -> str:
    """Repair common small-model JSON slips: bare decimals (.5), Python/JS
    literals, single '=' comparison ops, trailing commas.  The result is still
    fully validated against the feature schema afterwards."""
    s = re.sub(r'(?<=[\[,:\s])(-?)\.(\d)', r'\g<1>0.\2', s)
    s = re.sub(r'\bTrue\b', '1', s)
    s = re.sub(r'\bFalse\b', '0', s)
    s = re.sub(r'\bNone\b', 'null', s)
    s = s.replace('"="', '"=="')
    s = re.sub(r',\s*([\]}])', r'\1', s)
    return s


def _parse_json_obj(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    while start >= 0:
        depth = 0
        for k in range(start, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:k + 1]
                    for cand in (chunk, _lenient_json(chunk)):
                        try:
                            return json.loads(cand)
                        except Exception:
                            pass
                    break
        start = text.find("{", start + 1)
    return None


def llm_patches(backend: GenerationBackend, proc: Procedure, memory: EvidenceMemory, cfg: RSIConfig,
                n: int, seed: int) -> Tuple[List[Patch], List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (patches, raw_log, cost)."""
    if n <= 0:
        return [], [], dict(calls=0, tokens_in=0, tokens_out=0)
    user = build_editor_prompt(proc, memory, cfg)
    reqs = [GenRequest("editor", EDITOR_SYSTEM, user, cfg.editor_max_new_tokens, cfg.editor_temperature,
                       cfg.top_p, seed + i) for i in range(n)]
    outs = backend.generate(reqs)
    patches, log = [], []
    cost = dict(calls=len(outs), tokens_in=sum(o.n_in for o in outs), tokens_out=sum(o.n_out for o in outs))
    for o in outs:
        obj = _parse_json_obj(o.text)
        log.append(dict(raw=o.text[:4000], parsed=obj is not None))
        if not obj or obj.get("type") not in PATCH_TYPES:
            continue
        patches.append(Patch(
            type=obj["type"], source="llm", rule=obj.get("rule"), plan=obj.get("plan"),
            target_id=str(obj.get("target_id", "")), position=int(obj.get("position", 0) or 0),
            prompt_key=str(obj.get("prompt_key", "")), prompt_text=str(obj.get("prompt_text", "")),
            supporting_evidence=[str(x) for x in obj.get("supporting_evidence", []) or []],
            contradicting_evidence=[str(x) for x in obj.get("contradicting_evidence", []) or []],
            rationale=str(obj.get("rationale", ""))[:500]))
    return patches, log, cost


__all__ = ["Patch", "LeakGuard", "apply_patch", "compile_patches", "llm_patches",
           "build_editor_prompt", "PATCH_TYPES"]
