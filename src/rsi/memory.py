"""
memory.py
=========
Evidence memory M_t (§3, §5 "Keep negative evidence").

Stores every comparison — positive, zero, AND negative — with its state
features, action, parent/child answers, terminal values, costs, seeds and
model/procedure hashes.  Summaries shown to the editor are aggregated over
coarse feature bins and contain no problem text, problem ids or gold answers.

Bins (all derived from the fixed feature schema) are also what the
deterministic compiler turns back into guard conditions, so every compiled
rule / plan is expressible in the schema.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# -- bins ---------------------------------------------------------------------


def slot_bin(f: Dict[str, Any]) -> Tuple:
    crit = f.get("critique", "unknown")
    crit_b = crit if crit in ("unknown", "no_specific_error") else "error"
    ch = int(f.get("answer_changes", 0))
    return (int(f.get("valid", 0)), int(f.get("agrees_plurality", 0)), crit_b, min(ch, 2))


def budget_bin(g: Dict[str, Any]) -> str:
    b = float(g.get("budget_frac", 0.0))
    return "low" if b < 0.33 else ("mid" if b < 0.66 else "high")


def agree_bin(g: Dict[str, Any]) -> str:
    t = float(g.get("top_frac", 0.0))
    return "split" if t <= 0.5 else ("majority" if t < 1.0 else "unanimous")


def slot_bin_conditions(b: Tuple) -> List[list]:
    valid, agrees, crit_b, ch = b
    conds: List[list] = [["valid", "==", valid]]
    if valid:
        conds.append(["agrees_plurality", "==", agrees])
    if crit_b == "error":
        conds.append(["critique", "not_in", ["unknown", "no_specific_error"]])
    else:
        conds.append(["critique", "==", crit_b])
    conds.append(["answer_changes", ">=", 2] if ch >= 2 else ["answer_changes", "==", ch])
    return conds


def budget_bin_conditions(b: str) -> List[list]:
    return {"low": [["budget_frac", "<", 0.33]],
            "mid": [["budget_frac", ">=", 0.33], ["budget_frac", "<", 0.66]],
            "high": [["budget_frac", ">=", 0.66]]}[b]


def agree_bin_conditions(b: str) -> List[list]:
    return {"split": [["top_frac", "<=", 0.5]],
            "majority": [["top_frac", ">", 0.5], ["top_frac", "<", 1.0]],
            "unanimous": [["top_frac", ">=", 1.0]]}[b]


def describe_slot_bin(b: Tuple) -> str:
    valid, agrees, crit_b, ch = b
    if not valid:
        return f"slot has no extractable answer, critique={crit_b}, changes={'2+' if ch >= 2 else ch}"
    return (f"slot {'agrees' if agrees else 'disagrees'} with plurality, critique={crit_b}, "
            f"answer_changes={'2+' if ch >= 2 else ch}")


# -- memory -------------------------------------------------------------------


class EvidenceMemory:
    def __init__(self, path: Optional[str] = None, signed: bool = True):
        self.path = path
        self.signed = signed
        self.records: List[Dict[str, Any]] = []
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))

    def add(self, recs: Iterable[Dict[str, Any]]) -> None:
        recs = list(recs)
        for i, r in enumerate(recs):
            r.setdefault("eid", f"E{len(self.records) + i}")
        self.records.extend(recs)
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, default=str) + "\n")

    # -- views ----------------------------------------------------------------
    def visible(self) -> List[Dict[str, Any]]:
        """Signed memory keeps everything; success-only memory (ablation)
        keeps only records whose measured effect was positive."""
        if self.signed:
            return self.records
        out = []
        for r in self.records:
            d = r.get("I") if r["kind"] == "pair" else r.get("delta")
            if d is not None and d > 0:
                out.append(r)
        return out

    def single_groups(self, credit: str = "population") -> Dict[Tuple, Dict[str, Any]]:
        groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
        for r in self.visible():
            if r["kind"] not in ("single", "incumbent") or r.get("slot", -1) < 0:
                continue
            if r["kind"] == "incumbent" and r.get("action") in ("STOP", "PAIR"):
                continue
            key = r.get("delta_trace") if credit == "trace" else r.get("delta")
            if r["kind"] == "incumbent":
                key = 0.0 if r.get("delta") is not None else None
            if key is None or "slot_features" not in r:
                continue
            gk = (r["action"], slot_bin(r["slot_features"]), budget_bin(r["global_features"]))
            groups[gk].append(dict(r, _credit=float(key)))
        return {k: _stats(v, "_credit") for k, v in groups.items()}

    def stop_groups(self) -> Dict[Tuple, Dict[str, Any]]:
        groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
        for r in self.visible():
            if r["kind"] == "stop" and r.get("delta") is not None:
                gk = ("STOP", agree_bin(r["global_features"]), budget_bin(r["global_features"]))
                groups[gk].append(dict(r, _credit=float(r["delta"])))
        return {k: _stats(v, "_credit") for k, v in groups.items()}

    def pair_groups(self) -> Dict[Tuple, Dict[str, Any]]:
        groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
        for r in self.visible():
            if r["kind"] != "pair" or r.get("I") is None:
                continue
            gk = (r["op_a"], r["op_b"], slot_bin(r["features_a"]), slot_bin(r["features_b"]))
            groups[gk].append(r)
        out = {}
        for k, v in groups.items():
            I = np.array([x["I"] for x in v]); Ia = np.array([x["I_acc"] for x in v])
            ug = np.array([x["util_gain"] for x in v]); ivg = np.array([x["imm_vote_gain"] for x in v])
            out[k] = dict(n=len(v), mean_I=float(I.mean()), frac_I_pos=float((I > 1e-9).mean()),
                          mean_I_acc=float(Ia.mean()), mean_util_gain=float(ug.mean()),
                          mean_imm_vote_gain=float(ivg.mean()),
                          n_states=len({(x["qid"], x["state_decision_idx"]) for x in v}),
                          eids=[x["eid"] for x in v])
        return out

    def item_regret(self) -> Dict[str, Dict[str, Any]]:
        """For each incumbent rule/plan id: how much better the best tested
        alternative was in the states where it fired (deletion evidence)."""
        by_state: Dict[Tuple, Dict[str, Any]] = {}
        for r in self.visible():
            key = (r["round"], r["qid"], r["state_decision_idx"])
            s = by_state.setdefault(key, dict(item=r.get("incumbent_item"), best=None, eids=[]))
            d = r.get("delta") if r["kind"] != "pair" else None
            if d is not None and r["kind"] != "incumbent":
                s["best"] = d if s["best"] is None else max(s["best"], d)
                s["eids"].append(r["eid"])
        out: Dict[str, List[float]] = defaultdict(list)
        eids: Dict[str, List[str]] = defaultdict(list)
        for s in by_state.values():
            if s["best"] is not None and s["item"]:
                out[s["item"]].append(s["best"])
                eids[s["item"]].extend(s["eids"][:3])
        return {k: dict(n=len(v), mean_best_alt_delta=float(np.mean(v)),
                        frac_alt_better=float(np.mean([x > 1e-9 for x in v])), eids=eids[k][:12])
                for k, v in out.items()}

    # -- editor-facing text ---------------------------------------------------
    def summary_text(self, credit: str = "population", top_k: int = 8) -> str:
        lines: List[str] = []
        sg = self.single_groups(credit)
        ranked = sorted(sg.items(), key=lambda kv: kv[1]["mean"] * np.sqrt(kv[1]["n"]), reverse=True)
        lines.append("SINGLE-ACTION EVIDENCE (delta = measured utility of forcing the action minus the "
                     "incumbent continuation; population vote credit)" if credit == "population" else
                     "SINGLE-ACTION EVIDENCE (trace credit)")
        for (op, sb, bb), st in ranked[:top_k]:
            lines.append(f"  [+] {op} when {describe_slot_bin(sb)}, budget={bb}: n={st['n']} "
                         f"mean_delta={st['mean']:+.3f} pos/zero/neg={st['n_pos']}/{st['n_zero']}/{st['n_neg']} "
                         f"evidence={','.join(st['eids'][:4])}")
        for (op, sb, bb), st in ranked[::-1][:top_k]:
            if st["mean"] >= 0:
                break
            lines.append(f"  [-] {op} when {describe_slot_bin(sb)}, budget={bb}: n={st['n']} "
                         f"mean_delta={st['mean']:+.3f} pos/zero/neg={st['n_pos']}/{st['n_zero']}/{st['n_neg']} "
                         f"evidence={','.join(st['eids'][:4])}")
        stg = self.stop_groups()
        if stg:
            lines.append("STOP EVIDENCE")
            for (_, ab, bb), st in sorted(stg.items(), key=lambda kv: -kv[1]["mean"])[:top_k]:
                lines.append(f"  STOP when votes={ab}, budget={bb}: n={st['n']} mean_delta={st['mean']:+.3f} "
                             f"pos/neg={st['n_pos']}/{st['n_neg']} evidence={','.join(st['eids'][:4])}")
        pg = self.pair_groups()
        if pg:
            lines.append("TWO-SLOT INTERACTION EVIDENCE (I = Q_ab - Q_a - Q_b + Q_0; util_gain = Q_ab - "
                         "max(Q_0,Q_a,Q_b); I_acc on success rates)")
            for (a, b, sa, sb), st in sorted(pg.items(), key=lambda kv: -kv[1]["mean_I"])[:top_k]:
                lines.append(f"  {a} on [{describe_slot_bin(sa)}] then {b} on [{describe_slot_bin(sb)}]: "
                             f"n={st['n']} mean_I={st['mean_I']:+.3f} frac_I>0={st['frac_I_pos']:.2f} "
                             f"I_acc={st['mean_I_acc']:+.3f} util_gain={st['mean_util_gain']:+.3f} "
                             f"evidence={','.join(st['eids'][:4])}")
        reg = self.item_regret()
        if reg:
            lines.append("INCUMBENT ITEM REGRET (best tested alternative minus incumbent, where the item fired)")
            for k, st in sorted(reg.items(), key=lambda kv: -kv[1]["mean_best_alt_delta"])[:top_k]:
                lines.append(f"  {k}: n={st['n']} mean_best_alt_delta={st['mean_best_alt_delta']:+.3f} "
                             f"frac_alt_better={st['frac_alt_better']:.2f}")
        return "\n".join(lines)

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = defaultdict(int)
        for r in self.records:
            c[r["kind"]] += 1
            d = r.get("I") if r["kind"] == "pair" else r.get("delta")
            if d is None:
                c[f"{r['kind']}_unknown"] += 1
            elif d > 1e-9:
                c[f"{r['kind']}_pos"] += 1
            elif d < -1e-9:
                c[f"{r['kind']}_neg"] += 1
            else:
                c[f"{r['kind']}_zero"] += 1
        return dict(c)


def _stats(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    v = np.array([r[key] for r in rows], dtype=float)
    return dict(n=len(v), mean=float(v.mean()), std=float(v.std()),
                n_pos=int((v > 1e-9).sum()), n_neg=int((v < -1e-9).sum()),
                n_zero=int((np.abs(v) <= 1e-9).sum()),
                n_states=len({(r["round"], r["qid"], r["state_decision_idx"]) for r in rows}),
                eids=[r["eid"] for r in rows])


__all__ = [
    "EvidenceMemory", "slot_bin", "budget_bin", "agree_bin", "slot_bin_conditions",
    "budget_bin_conditions", "agree_bin_conditions", "describe_slot_bin",
]
