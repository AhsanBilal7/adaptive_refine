"""
counterfactual.py
=================
State-copy / action-comparison runner (§5, §5.1; §9 module 3).
DEVELOPMENT DATA ONLY — every branch is graded against the dev gold answer.

1. ``solve_and_log`` runs H_t on a search batch and saves complete copies of
   states it reaches (reservoir sample of ``cf_states_per_problem`` decision
   points per problem), together with the incumbent decision a0.
2. ``compare_actions`` forks each saved state S into branches:
     incumbent        force a0, continue with H_t
     single (a)       force a on one of ``cf_alt_slots`` selected slots, continue
     STOP             vote immediately
     pair (a→b)       force a on slot i then b on slot j≠i unconditionally, continue
   each with K = ``cf_seeds`` continuation seeds (common seed bases across
   branches for variance reduction).  Every branch starts from an exact copy
   (population, history, procedure, remaining budget).
3. Values:
     Q̂(S,a)  = 1/K Σ_k [ 1{ŷ = y*} − λ C_k / B ]      C = FUTURE cost from S
     Δ̂(S,a)  = Q̂(S,a) − Q̂(S,a0)
     I(S,a,b) = Q_{a→b} − Q_a − Q_b + Q_∅             (also on success rates)
   Untested actions stay unknown; infeasible branches are recorded as such,
   never as failures.  All counterfactual calls are charged to the
   procedure-development cost ledger.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backends import GenerationBackend
from .procedure import Decision, Plan, Procedure
from .runner import grade_result, seed_base
from .checkpoint import Job, atomic_pickle, load_pickle, run_jobs
from .solver import SolveContext, solve_gen
from .state import SolveState
from .voting import Grader

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    qid: str
    state: SolveState
    incumbent: Decision
    decision_idx: int
    seed: int
    gold_ref: str = ""            # key into the dev gold table; never inside the state


@dataclass
class DevLedger:
    """Procedure-development compute (counterfactual + search + confirmation)."""
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tflops: float = 0.0
    by_phase: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add(self, phase: str, cost: Dict[str, Any]) -> None:
        self.calls += int(cost["calls"]); self.tokens_in += int(cost["tokens_in"])
        self.tokens_out += int(cost["tokens_out"]); self.tflops += float(cost["tflops"])
        p = self.by_phase.setdefault(phase, dict(calls=0, tokens_in=0, tokens_out=0, tflops=0.0))
        for k in ("calls", "tokens_in", "tokens_out"):
            p[k] += int(cost[k])
        p["tflops"] += float(cost["tflops"])

    def to_dict(self) -> Dict[str, Any]:
        return dict(calls=self.calls, tokens_in=self.tokens_in, tokens_out=self.tokens_out,
                    tflops=self.tflops, by_phase=self.by_phase)


# =============================================================================
# 1. Solve and log
# =============================================================================

def solve_and_log(ctx: SolveContext, backend: GenerationBackend, problems: Sequence[Dict[str, Any]],
                  proc: Procedure, grader: Grader, rng: random.Random, seed: int = 0,
                  ckpt: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Snapshot]]:
    cfg = ctx.cfg
    k_keep = max(0, int(cfg.cf_states_per_problem))
    # reservoir-sampling state lives in `hs` so it is checkpointed with the solves
    hs: Dict[str, Any] = dict(reservoirs={}, seen={}, rng=rng.getstate())
    hrng = random.Random()
    hrng.setstate(hs["rng"])

    def make_hook(qid: str, sb: int):
        def hook(st: SolveState, d: Decision):
            hs["seen"][qid] = hs["seen"].get(qid, 0) + 1
            n = hs["seen"][qid]
            res = hs["reservoirs"].setdefault(qid, [])
            if len(res) < k_keep:
                res.append(Snapshot(qid, st.copy(), _copy_decision(d), len(st.decisions), sb, qid))
            else:
                j = hrng.randrange(n)
                if j < k_keep:
                    res[j] = Snapshot(qid, st.copy(), _copy_decision(d), len(st.decisions), sb, qid)
        return hook

    def aux_get():
        hs["rng"] = hrng.getstate()
        return hs

    def aux_set(saved):
        hs.update(saved)
        hrng.setstate(hs["rng"])

    jobs, metas = [], []
    for p in problems:
        sb = seed_base(cfg.seed, p["qid"], seed)
        st = ctx.new_state(p["qid"], p["question"], sb)
        hook = make_hook(p["qid"], sb) if k_keep else None
        jobs.append(Job(f"log|{p['qid']}|{seed}", st,
                        (lambda h: (lambda s, forced: solve_gen(ctx, s, proc, snapshot_hook=h)))(hook)))
        metas.append(p)
    outs = run_jobs(backend, jobs, ckpt, desc="solve_and_log", save_every_s=cfg.checkpoint_every_s,
                    aux_get=aux_get, aux_set=aux_set)
    recs = []
    for p, res in zip(metas, outs):
        r = grade_result(res, p.get("gold"), grader, cfg.utility_lambda)
        r["seed"] = seed
        recs.append(r)
    snaps = [s for p in problems for s in hs["reservoirs"].get(p["qid"], [])]
    return recs, snaps


def _copy_decision(d: Decision) -> Decision:
    return Decision(d.action, d.slot, d.slot2, d.plan, d.item_id, False)


# =============================================================================
# Slot / pair selection
# =============================================================================

class SlotRotation:
    def __init__(self):
        self.counter = 0

    def select(self, st: SolveState, n_slots: int, rng: random.Random, explore_frac: float
               ) -> Tuple[List[int], str]:
        eligible = st.eligible_slots()
        if not eligible:
            return [], "none"
        k = min(n_slots, len(eligible))
        if rng.random() >= explore_frac:
            winner = st.vote()[0]
            targeted = [s.idx for s in st.slots if not s.frozen and (not s.norm or s.norm != winner)]
            if len(targeted) >= k:
                return targeted[:k], "disagreement"
        start = self.counter % len(eligible)
        self.counter += 1
        chosen = [eligible[(start + t) % len(eligible)] for t in range(k)]
        return chosen, "rotation"


# =============================================================================
# 2. Branch construction and execution
# =============================================================================

@dataclass
class Branch:
    snap_i: int
    kind: str                    # incumbent | single | stop | pair
    decisions: List[Decision]
    label: str
    op: str = ""
    slot: int = -1
    op2: str = ""
    slot2: int = -1


def _branches_for(snap_i: int, snap: Snapshot, cfg, rot: SlotRotation, rng: random.Random
                  ) -> Tuple[List[Branch], Dict[str, Any]]:
    st = snap.state
    a0 = snap.incumbent
    branches = [Branch(snap_i, "incumbent", [_copy_decision(a0)], "a0:" + a0.key())]
    info: Dict[str, Any] = {}
    if cfg.evidence_mode == "trajectory":
        return branches, dict(selection="trajectory", slots=[])
    slots, mode = rot.select(st, cfg.cf_alt_slots, rng, cfg.cf_explore_frac)
    info.update(selection=mode, slots=slots)
    alt_ops = cfg.list_of("cf_alt_ops")
    pair_ops = cfg.list_of("cf_pair_ops")
    need_ops = list(dict.fromkeys(alt_ops + (pair_ops if cfg.allow_plans else [])))
    for s in slots:
        for op in need_ops:
            d = Decision(op, slot=s)
            if d.key() == a0.key():
                continue
            branches.append(Branch(snap_i, "single", [d], d.key(), op=op, slot=s))
    if cfg.cf_include_stop and a0.action != "STOP":
        branches.append(Branch(snap_i, "stop", [Decision("STOP")], "STOP"))
    if cfg.allow_plans and len(slots) >= 2 and cfg.cf_pairs_per_state > 0:
        ordered = [(slots[0], slots[1]), (slots[1], slots[0])]
        rng.shuffle(ordered)
        for (i, j) in ordered[: cfg.cf_pairs_per_state]:
            a, b = rng.choice(pair_ops), rng.choice(pair_ops)
            branches.append(Branch(snap_i, "pair", [Decision(a, slot=i), Decision(b, slot=j)],
                                   f"PAIR({a}@{i}->{b}@{j})", op=a, slot=i, op2=b, slot2=j))
    return branches, info


def _branch_job(ctx: SolveContext, snap: Snapshot, br: Branch, proc: Procedure, bi: int, k: int) -> Job:
    st = snap.state.copy()
    st.seed_base = (snap.seed * 1000003 + 7919 * (k + 1)) & 0x7FFFFFFF   # CRN across branches
    st.call_counter = 0
    forced = [Decision(d.action, d.slot, d.slot2, d.plan, d.item_id) for d in br.decisions]
    return Job(f"cf|{bi}|{k}", st, lambda s, fr: solve_gen(ctx, s, proc, forced=fr),
               forced=forced, start_n_dec=len(snap.state.decisions))


def compare_actions(ctx: SolveContext, backend: GenerationBackend, snaps: List[Snapshot],
                    proc: Procedure, gold: Dict[str, Optional[str]], grader: Grader,
                    rng: random.Random, round_idx: int, ledger: Optional[DevLedger] = None,
                    max_inflight: int = 0, ckpt: Optional[str] = None) -> List[Dict[str, Any]]:
    cfg = ctx.cfg
    lam = cfg.utility_lambda
    plan = load_pickle(ckpt + ".plan.pkl") if ckpt else None
    if plan is None:          # the branch plan is fixed once, so resumes run the same experiments
        rot = SlotRotation()
        all_branches: List[Branch] = []
        infos: List[Dict[str, Any]] = []
        for si, snap in enumerate(snaps):
            brs, info = _branches_for(si, snap, cfg, rot, rng)
            all_branches.extend(brs)
            infos.append(info)
        if ckpt:
            atomic_pickle((all_branches, infos), ckpt + ".plan.pkl")
    else:
        all_branches, infos = plan
    K = max(1, int(cfg.cf_seeds))
    jobs, keys = [], []
    for bi, br in enumerate(all_branches):
        snap = snaps[br.snap_i]
        for k in range(K):
            jobs.append(_branch_job(ctx, snap, br, proc, bi, k))
            keys.append((bi, k))
    logger.info(f"[CF] round={round_idx} states={len(snaps)} branches={len(all_branches)} "
                f"continuations={len(jobs)}")
    outs = run_jobs(backend, jobs, ckpt, desc="counterfactual", max_inflight=max_inflight,
                    save_every_s=cfg.checkpoint_every_s)

    # -- collect per-branch per-seed outcomes --------------------------------
    per_branch: Dict[int, List[Dict[str, Any]]] = {}
    for (bi, k), res in zip(keys, outs):
        br = all_branches[bi]
        snap = snaps[br.snap_i]
        g = gold.get(snap.gold_ref)
        st = res.state
        B = st.meter.budget
        fut = st.meter.spent - snap.state.meter.spent
        if ledger is not None:
            ledger.add("counterfactual", dict(
                calls=st.meter.calls - snap.state.meter.calls,
                tokens_in=st.meter.tokens_in - snap.state.meter.tokens_in,
                tokens_out=st.meter.tokens_out - snap.state.meter.tokens_out,
                tflops=st.meter.tflops - snap.state.meter.tflops))
        correct = bool(grader.is_correct(res.final_answer, g))
        forced_recs = [d for d in st.decisions[snap.decision_idx:] if d["forced"]]
        imm_answer = forced_recs[-1]["vote_after_answer"] if forced_recs else res.final_answer
        imm_correct = bool(grader.is_correct(imm_answer, g))
        child = {}
        for s in (br.slot, br.slot2):
            if s is not None and s >= 0 and forced_recs:
                # slot answer right after the forced ops (before the continuation)
                child[s] = bool(grader.is_correct(forced_recs[-1]["answers_after"][s], g))
        per_branch.setdefault(bi, []).append(dict(
            seed=k, correct=correct, future_cost=fut, future_frac=fut / B if B > 0 else 0.0,
            final=res.final_answer, infeasible=res.infeasible_forced, imm_correct=imm_correct,
            child_correct=child))

    # -- values ----------------------------------------------------------------
    def value(bi: int) -> Optional[Dict[str, float]]:
        rows = [r for r in per_branch.get(bi, []) if not r["infeasible"]]
        if not rows:
            return None
        acc = float(np.mean([r["correct"] for r in rows]))
        cf = float(np.mean([r["future_frac"] for r in rows]))
        v = dict(Q=acc - lam * cf, acc=acc, cost_frac=cf, n=len(rows),
                 imm_acc=float(np.mean([r["imm_correct"] for r in rows])),
                 Q_seeds=[float(r["correct"]) - lam * r["future_frac"] for r in rows])
        br = all_branches[bi]
        if br.slot >= 0:
            tc = [r["child_correct"].get(br.slot) for r in rows if br.slot in r["child_correct"]]
            if tc:
                v["Q_trace"] = float(np.mean(tc)) - lam * cf
        return v

    by_snap: Dict[int, Dict[str, int]] = {}
    for bi, br in enumerate(all_branches):
        by_snap.setdefault(br.snap_i, {})[br.label] = bi

    evidence: List[Dict[str, Any]] = []
    for si, snap in enumerate(snaps):
        idx = by_snap.get(si, {})
        inc_bi = next(bi for lbl, bi in idx.items() if lbl.startswith("a0:"))
        v0 = value(inc_bi)
        st = snap.state
        g = gold.get(snap.gold_ref)
        vote_correct_at_S = bool(grader.is_correct(st.vote()[1], g))
        gfeat = st.global_features()
        base = dict(round=round_idx, qid=snap.qid, state_decision_idx=snap.decision_idx,
                    selection=infos[si].get("selection"), incumbent=snap.incumbent.key(),
                    incumbent_action=snap.incumbent.action, incumbent_item=snap.incumbent.item_id,
                    global_features=gfeat, vote_correct_at_state=vote_correct_at_S,
                    model_hash=ctx.backend.model_hash(), procedure_hash=proc.hash(), K=K)
        # incumbent record (also the only record in trajectory mode)
        inc_rec = dict(base, kind="incumbent", action=snap.incumbent.action, slot=snap.incumbent.slot,
                       value=v0, delta=0.0 if v0 else None)
        if snap.incumbent.slot >= 0:
            inc_rec["slot_features"] = st.slot_features(snap.incumbent.slot, gfeat)
        evidence.append(inc_rec)
        if v0 is None:
            continue
        singles: Dict[Tuple[str, int], Dict[str, float]] = {}
        for lbl, bi in idx.items():
            br = all_branches[bi]
            if br.kind not in ("single", "stop"):
                continue
            v = value(bi)
            rec = dict(base, kind="single" if br.kind == "single" else "stop", action=br.op or "STOP",
                       slot=br.slot, value=v)
            if br.slot >= 0:
                rec["slot_features"] = st.slot_features(br.slot, gfeat)
                rec["parent_answer"] = st.slots[br.slot].answer
                rec["parent_correct"] = bool(grader.is_correct(st.slots[br.slot].answer, g))
            if v is None:
                rec.update(delta=None, status="infeasible")
            else:
                rec.update(delta=v["Q"] - v0["Q"], delta_acc=v["acc"] - v0["acc"],
                           # individual-trace credit (§8 ablation): acted slot's own
                           # correctness gain over its parent, cost-penalized
                           delta_trace=(v["Q_trace"] - float(rec.get("parent_correct", False)))
                           if "Q_trace" in v else None,
                           recovery=(not vote_correct_at_S) and v["acc"] > v0["acc"],
                           regression=vote_correct_at_S and v["acc"] < v0["acc"], status="ok")
                if br.slot >= 0:
                    singles[(br.op, br.slot)] = v
            evidence.append(rec)
        # the incumbent is itself a single action on its slot
        if snap.incumbent.action in ("FULL_REFINE", "DIRECT_REPAIR", "RESTART", "KEEP"):
            singles[(snap.incumbent.action, snap.incumbent.slot)] = v0
        for lbl, bi in idx.items():
            br = all_branches[bi]
            if br.kind != "pair":
                continue
            vab = value(bi)
            va, vb = singles.get((br.op, br.slot)), singles.get((br.op2, br.slot2))
            rec = dict(base, kind="pair", op_a=br.op, slot_a=br.slot, op_b=br.op2, slot_b=br.slot2,
                       features_a=st.slot_features(br.slot, gfeat), features_b=st.slot_features(br.slot2, gfeat),
                       value_ab=vab, value_a=va, value_b=vb, value_0=v0)
            if vab is None or va is None or vb is None:
                rec.update(status="incomplete", I=None, I_acc=None)
            else:
                I = vab["Q"] - va["Q"] - vb["Q"] + v0["Q"]
                I_acc = vab["acc"] - va["acc"] - vb["acc"] + v0["acc"]
                rec.update(status="ok", I=I, I_acc=I_acc,
                           util_gain=vab["Q"] - max(v0["Q"], va["Q"], vb["Q"]),
                           imm_vote_gain=vab["imm_acc"] - float(vote_correct_at_S),
                           delta=vab["Q"] - v0["Q"])
            evidence.append(rec)
    return evidence


def evidence_yield(evidence: List[Dict[str, Any]], eps: float = 1e-9) -> Dict[str, Any]:
    """Fraction of comparisons that distinguish actions (|Δ| > 0) — §9 check."""
    singles = [e for e in evidence if e["kind"] in ("single", "stop") and e.get("delta") is not None]
    pairs = [e for e in evidence if e["kind"] == "pair" and e.get("I") is not None]
    return dict(
        n_states=len({(e["qid"], e["state_decision_idx"]) for e in evidence}),
        n_single=len(singles),
        informative_single=sum(1 for e in singles if abs(e["delta"]) > eps),
        n_pairs=len(pairs), informative_pairs=sum(1 for e in pairs if abs(e["I"]) > eps),
        positive_pairs=sum(1 for e in pairs if e["I"] > eps),
    )


__all__ = ["Snapshot", "DevLedger", "solve_and_log", "compare_actions", "evidence_yield"]
