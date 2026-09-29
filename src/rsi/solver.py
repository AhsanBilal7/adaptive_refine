"""
solver.py
=========
Inner loop (§4): adaptive refinement with one vote per slot.

``solve_gen`` is a *generator*: it yields ``List[GenRequest]`` and receives
``List[GenResult]``.  ``scheduler.run_generators`` drives many of them at once
so that independent solves (problems, seeds, counterfactual branches) share
batched model calls, while each individual solve stays strictly sequential —
every decision sees the state produced by the previous one.

Operations (all on the same frozen model):
  FULL_REFINE i   : rewrite r_i → g_i, critique g_i → c_i, correct(g_i, c_i) → r_i'   (3 calls)
  DIRECT_REPAIR i : critique r_i → c_i, correct(r_i, c_i) → r_i'                      (2 calls)
  RESTART i       : fresh problem-only sample replaces slot i                          (1 call)
  KEEP i          : freeze slot i                                                      (0 calls)
  STOP            : vote now                                                           (0 calls)
  PAIR(m,i,j)     : plan m — op1 on i, continuation guard, op2 on j                    (sum)

Budget rules: each op reserves its worst-case cost first (capped inputs +
max_new_tokens per call); an incomplete op leaves the slot unchanged but its
spent compute is charged; RESTART never restores budget.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.cimr_step import truncate_at_first_boxed  # type: ignore

from .backends import GenRequest, GenResult, GenerationBackend
from .budget import CostMeter
from .config import RSIConfig
from .procedure import Decision, Plan, Procedure
from .prompts import (DEFAULT_PROMPTS, correct_prompt, critique_prompt, initial_prompt, parse_critique_category,
                      rewrite_prompt)
from .state import SolveState, match_all
from .voting import extract_answer, normalize_answer

SolveGen = Generator[List[GenRequest], List[GenResult], "SolveResult"]


@dataclass
class SolveResult:
    qid: str
    final_norm: str
    final_answer: str
    state: SolveState
    infeasible_forced: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


class SolveContext:
    """Per-run helpers shared by all solves: prompt construction, token
    bounds for reservation, and budget sizing.  Holds NO gold answers."""

    def __init__(self, cfg: RSIConfig, backend: GenerationBackend):
        self.cfg = cfg
        self.backend = backend
        self._tok_cache: Dict[Tuple, int] = {}
        self.disabled = tuple(cfg.list_of("disabled_actions"))

    # -- token helpers -------------------------------------------------------
    def _ptok(self, system: str, user: str) -> int:
        key = (hashlib.md5(system.encode()).hexdigest(), hashlib.md5(user.encode()).hexdigest())
        if key not in self._tok_cache:
            self._tok_cache[key] = self.backend.prompt_tokens(system, user)
        return self._tok_cache[key]

    def _cost(self, meter: CostMeter, n_in: int, n_out: int) -> float:
        return meter.cost_of(min(n_in, self.backend.max_input_tokens), n_out)

    def _bound_trace_tokens(self, producer_max_new: int, cap_chars: int) -> int:
        cap = producer_max_new if cap_chars <= 0 else min(producer_max_new, cap_chars)
        return int(cap) + int(self.cfg.reserve_slack_tokens)

    # -- per-call worst cases -------------------------------------------------
    def wc_init(self, meter, question: str) -> float:
        s, u = initial_prompt(question)
        return self._cost(meter, self._ptok(s, u), self.cfg.init_max_new_tokens)

    def wc_rewrite(self, meter, question: str, prior: Optional[str], step_idx: int, total: int) -> float:
        c = self.cfg
        if prior is None:  # unknown prior trace → bound it
            s, u = rewrite_prompt(question, "", step_idx, total, c.rewrite_max_prior_chars)
            n_in = self._ptok(s, u) + self._bound_trace_tokens(
                max(c.init_max_new_tokens, c.correct_max_new_tokens), c.rewrite_max_prior_chars)
        else:
            s, u = rewrite_prompt(question, prior, step_idx, total, c.rewrite_max_prior_chars)
            n_in = self._ptok(s, u)
        return self._cost(meter, n_in, c.rewrite_max_new_tokens)

    def wc_critique(self, meter, prompts, question: str, trace: Optional[str], producer_max_new: int) -> float:
        c = self.cfg
        if trace is None:
            s, u = critique_prompt(prompts, question, "", "", c.critique_max_trace_chars)
            n_in = self._ptok(s, u) + self._bound_trace_tokens(producer_max_new, c.critique_max_trace_chars)
        else:
            s, u = critique_prompt(prompts, question, trace, "", c.critique_max_trace_chars)
            n_in = self._ptok(s, u) + 16
        return self._cost(meter, n_in, c.critique_max_new_tokens)

    def wc_correct(self, meter, prompts, question: str, trace: Optional[str], producer_max_new: int) -> float:
        c = self.cfg
        s, u = correct_prompt(prompts, question, trace or "", "", "", c.correct_max_trace_chars,
                              c.correct_max_critique_chars)
        n_in = self._ptok(s, u) + 16
        if trace is None:
            n_in += self._bound_trace_tokens(producer_max_new, c.correct_max_trace_chars)
        n_in += self._bound_trace_tokens(c.critique_max_new_tokens, c.correct_max_critique_chars)
        return self._cost(meter, n_in, c.correct_max_new_tokens)

    # -- per-op reservation ----------------------------------------------------
    def reserve(self, op: str, st: SolveState, i: int, prompts: Dict[str, str],
                h0_depth: int, known_trace: bool = True) -> float:
        m = st.meter
        c = self.cfg
        if op in ("KEEP", "STOP"):
            return 0.0
        if op == "RESTART":
            return self.wc_init(m, st.question)
        trace = st.slots[i].trace if known_trace else None
        if op == "DIRECT_REPAIR":
            prod = max(c.init_max_new_tokens, c.correct_max_new_tokens)
            return (self.wc_critique(m, prompts, st.question, trace, prod)
                    + self.wc_correct(m, prompts, st.question, trace, prod))
        if op == "FULL_REFINE":
            step = st.slots[i].depth + 1
            return (self.wc_rewrite(m, st.question, trace, step, h0_depth)
                    + self.wc_critique(m, prompts, st.question, None, c.rewrite_max_new_tokens)
                    + self.wc_correct(m, prompts, st.question, None, c.rewrite_max_new_tokens))
        raise ValueError(op)

    def afford_fn(self, proc: Procedure):
        def afford(op, st: SolveState, slot) -> bool:
            if isinstance(op, tuple) and op[0] == "PAIR":
                (i, j), (_, a, b) = slot, op
                # second op sees a trace that does not exist yet only if i == j (never);
                # slot j is untouched by op1, so its trace is known.
                need = self.reserve(a, st, i, proc.prompts, proc.h0_depth) + \
                    self.reserve(b, st, j, proc.prompts, proc.h0_depth)
                return st.meter.can_afford(need)
            if op in self.disabled:
                return False
            return st.meter.can_afford(self.reserve(op, st, slot, proc.prompts, proc.h0_depth))
        return afford

    # -- budget sizing -----------------------------------------------------------
    def budget_for(self, question: str, meter_template: CostMeter) -> float:
        c = self.cfg
        if c.budget_mode == "absolute":
            return float(c.budget_value)
        m = meter_template
        prompts = DEFAULT_PROMPTS
        init = self.wc_init(m, question)
        full = (self.wc_rewrite(m, question, None, 1, c.h0_depth)
                + self.wc_critique(m, prompts, question, None, c.rewrite_max_new_tokens)
                + self.wc_correct(m, prompts, question, None, c.rewrite_max_new_tokens))
        # Editing the critique/correction prompts may lengthen them (≤ max_prompt_chars
        # chars ≥ tokens each); include that allowance so H0 schedules stay affordable
        # for every admissible procedure.
        extra = 2 * m.cost_of(c.max_prompt_chars, 0) if c.cost_measure != "calls" else 0.0
        return c.n_slots * init + c.n_slots * c.budget_cycles * (full + extra)

    def new_state(self, qid: str, question: str, seed_base: int) -> SolveState:
        c = self.cfg
        tmp = CostMeter(c.cost_measure, self.backend.num_params, budget=0.0)
        B = self.budget_for(question, tmp)
        meter = CostMeter(c.cost_measure, self.backend.num_params, budget=B)
        return SolveState(qid=qid, question=question, n_slots=c.n_slots, meter=meter, seed_base=seed_base)

    # -- request builders ------------------------------------------------------
    def _seed(self, st: SolveState) -> int:
        st.call_counter += 1
        h = hashlib.md5(f"{st.seed_base}:{st.qid}:{st.call_counter}".encode()).hexdigest()
        return int(h[:8], 16)

    def req_init(self, st: SolveState, role: str = "init") -> GenRequest:
        s, u = initial_prompt(st.question)
        return GenRequest(role, s, u, self.cfg.init_max_new_tokens, self.cfg.init_temperature,
                          self.cfg.top_p, self._seed(st))

    def req_rewrite(self, st: SolveState, i: int, h0_depth: int) -> GenRequest:
        slot = st.slots[i]
        s, u = rewrite_prompt(st.question, slot.trace, slot.depth + 1, h0_depth,
                              self.cfg.rewrite_max_prior_chars)
        return GenRequest("rewrite", s, u, self.cfg.rewrite_max_new_tokens, self.cfg.rewrite_temperature,
                          self.cfg.top_p, self._seed(st))

    def req_critique(self, st: SolveState, prompts, trace: str, answer: str) -> GenRequest:
        s, u = critique_prompt(prompts, st.question, trace, answer, self.cfg.critique_max_trace_chars)
        return GenRequest("critique", s, u, self.cfg.critique_max_new_tokens, self.cfg.critique_temperature,
                          self.cfg.top_p, self._seed(st))

    def req_correct(self, st: SolveState, prompts, trace: str, critique: str, answer: str) -> GenRequest:
        s, u = correct_prompt(prompts, st.question, trace, critique, answer, self.cfg.correct_max_trace_chars,
                              self.cfg.correct_max_critique_chars)
        return GenRequest("correct", s, u, self.cfg.correct_max_new_tokens, self.cfg.correct_temperature,
                          self.cfg.top_p, self._seed(st))

    def extract(self, text: str) -> Tuple[str, str]:
        a = extract_answer(text, self.cfg.extract_fallback_last_number)
        return a, normalize_answer(a)


# =============================================================================
# Operation executors (sub-generators)
# =============================================================================

def _charge(st: SolveState, req: GenRequest, res: GenResult) -> None:
    st.meter.charge(req.role, res.n_in, res.n_out)


def _call(st: SolveState, req: GenRequest):
    results = yield [req]
    res = results[0]
    _charge(st, req, res)
    return res


def op_full_refine(ctx: SolveContext, st: SolveState, i: int, proc: Procedure, didx: int):
    rw = ctx.req_rewrite(st, i, proc.h0_depth)
    r1 = yield from _call(st, rw)
    if not r1.ok:
        return dict(complete=False, failed_at="rewrite")
    g = truncate_at_first_boxed(r1.text)
    g_ans, _ = ctx.extract(g)
    r2 = yield from _call(st, ctx.req_critique(st, proc.prompts, g, g_ans))
    if not r2.ok:
        return dict(complete=False, failed_at="critique")
    cat = parse_critique_category(r2.text)
    r3 = yield from _call(st, ctx.req_correct(st, proc.prompts, g, r2.text, g_ans))
    if not r3.ok:
        return dict(complete=False, failed_at="correct")
    new = truncate_at_first_boxed(r3.text)
    ans, norm = ctx.extract(new)
    st.slots[i].commit("FULL_REFINE", new, ans, norm, critique=r2.text, category=cat,
                       decision_idx=didx, intermediate=g)
    return dict(complete=True, category=cat, pre_answer=g_ans)


def op_direct_repair(ctx: SolveContext, st: SolveState, i: int, proc: Procedure, didx: int):
    slot = st.slots[i]
    r1 = yield from _call(st, ctx.req_critique(st, proc.prompts, slot.trace, slot.answer))
    if not r1.ok:
        return dict(complete=False, failed_at="critique")
    cat = parse_critique_category(r1.text)
    r2 = yield from _call(st, ctx.req_correct(st, proc.prompts, slot.trace, r1.text, slot.answer))
    if not r2.ok:
        return dict(complete=False, failed_at="correct")
    new = truncate_at_first_boxed(r2.text)
    ans, norm = ctx.extract(new)
    slot.commit("DIRECT_REPAIR", new, ans, norm, critique=r1.text, category=cat, decision_idx=didx)
    return dict(complete=True, category=cat, pre_answer=slot.history[-2].answer if len(slot.history) > 1 else "")


def op_restart(ctx: SolveContext, st: SolveState, i: int, proc: Procedure, didx: int):
    r = yield from _call(st, ctx.req_init(st, role="restart"))
    if not r.ok:
        return dict(complete=False, failed_at="restart")
    new = truncate_at_first_boxed(r.text)
    ans, norm = ctx.extract(new)
    st.slots[i].commit("RESTART", new, ans, norm, decision_idx=didx, restart=True)
    return dict(complete=True)


def run_op(ctx, st, op: str, i: int, proc: Procedure, didx: int):
    if op == "FULL_REFINE":
        return (yield from op_full_refine(ctx, st, i, proc, didx))
    if op == "DIRECT_REPAIR":
        return (yield from op_direct_repair(ctx, st, i, proc, didx))
    if op == "RESTART":
        return (yield from op_restart(ctx, st, i, proc, didx))
    if op == "KEEP":
        st.slots[i].frozen = True
        return dict(complete=True)
    raise ValueError(op)


# =============================================================================
# Main solve generator
# =============================================================================

def initialize_gen(ctx: SolveContext, st: SolveState):
    """Draw N independent problem-only traces (one per slot)."""
    reqs = [ctx.req_init(st) for _ in range(st.n_slots)]
    results = yield reqs
    for i, (req, res) in enumerate(zip(reqs, results)):
        _charge(st, req, res)
        text = truncate_at_first_boxed(res.text) if res.ok else ""
        ans, norm = ctx.extract(text)
        st.slots[i].commit("INIT", text, ans, norm)
    st.initialized = True


def _decision_legal(ctx: SolveContext, st: SolveState, d: Decision, proc: Procedure) -> bool:
    if d.action == "STOP":
        return True
    afford = ctx.afford_fn(proc)
    if d.action == "PAIR":
        if d.slot == d.slot2 or st.slots[d.slot].frozen or st.slots[d.slot2].frozen:
            return False
        return afford(("PAIR", d.plan.first_op, d.plan.second_op), st, (d.slot, d.slot2))
    if st.slots[d.slot].frozen:
        return False
    return st.meter.can_afford(ctx.reserve(d.action, st, d.slot, proc.prompts, proc.h0_depth))


def solve_gen(ctx: SolveContext, st: SolveState, proc: Procedure,
              forced: Optional[List[Decision]] = None, snapshot_hook=None) -> SolveGen:
    """Run (or continue) one solve.  ``forced`` decisions are executed first
    (counterfactual branches); afterwards ``proc`` decides.  ``snapshot_hook``
    (state, decision) is called before each NON-forced decision so the caller
    can save copies of reached states."""
    if not st.initialized:
        yield from initialize_gen(ctx, st)
    forced = list(forced or [])
    infeasible = False
    max_dec = ctx.cfg.resolved_max_decisions()
    afford = ctx.afford_fn(proc)

    while not st.stopped:
        if st.n_decisions >= max_dec:
            st.stopped, st.stop_reason = True, "decision_cap"
            break
        if forced:
            d = forced.pop(0)
            d.forced = True
            if not _decision_legal(ctx, st, d, proc):
                infeasible = True
                st.stopped, st.stop_reason = True, "forced_infeasible"
                break
        else:
            d = proc.choose(st, afford, allow_plans=ctx.cfg.allow_plans, disabled=ctx.disabled)
            if snapshot_hook is not None:
                snapshot_hook(st, d)
        yield from execute_decision(ctx, st, d, proc)
    winner, disp, _ = st.vote()
    return SolveResult(st.qid, winner, disp, st, infeasible_forced=infeasible)


def execute_decision(ctx: SolveContext, st: SolveState, d: Decision, proc: Procedure):
    didx = len(st.decisions)
    g = st.global_features()
    rec: Dict[str, Any] = dict(
        idx=didx, action=d.action, key=d.key(), slot=d.slot, slot2=d.slot2, item_id=d.item_id,
        forced=d.forced, spent_before=st.meter.spent, global_features=g,
        slot_features={i: st.slot_features(i, g) for i in range(st.n_slots)},
        norms_before=st.norm_answers(), answers_before=[s.answer for s in st.slots],
        vote_before=st.vote()[0], vote_before_answer=st.vote()[1],
    )
    if d.action == "STOP":
        st.stopped, st.stop_reason = True, d.item_id
        st.n_decisions += 1
    elif d.action == "PAIR":
        p: Plan = d.plan
        rec["plan"] = p.id
        out1 = yield from run_op(ctx, st, p.first_op, d.slot, proc, didx)
        st.n_decisions += 1
        gf = st.global_features()
        guard_ok = (not st.slots[d.slot2].frozen
                    and match_all(st.slot_features(d.slot2, gf), p.continuation_guard))
        rec.update(first=out1, guard_passed=guard_ok)
        if guard_ok and st.meter.can_afford(ctx.reserve(p.second_op, st, d.slot2, proc.prompts, proc.h0_depth)):
            out2 = yield from run_op(ctx, st, p.second_op, d.slot2, proc, didx)
            st.n_decisions += 1
            rec["second"] = out2
        else:
            rec["second"] = dict(complete=False, skipped=True)
    else:
        out = yield from run_op(ctx, st, d.action, d.slot, proc, didx)
        st.n_decisions += 1
        rec["outcome"] = out
    rec.update(spent_after=st.meter.spent, norms_after=st.norm_answers(),
               answers_after=[s.answer for s in st.slots],
               vote_after=st.vote()[0], vote_after_answer=st.vote()[1])
    st.decisions.append(rec)


__all__ = ["SolveContext", "SolveResult", "solve_gen", "execute_decision", "initialize_gen", "run_op"]
