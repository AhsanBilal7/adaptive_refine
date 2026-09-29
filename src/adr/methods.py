"""
methods.py
==========
Every compared method as a generator that yields ``List[GenRequest]`` and
receives ``List[GenResult]`` (driven, batched and checkpointed by
scheduler.py / checkpoint.py).

  prefix_gen      N independent structured solutions + ONE critique of each.
                  Shared by all direct-critique variants so they start from the
                  same solutions and identical first critiques (§7).
  direct_gen      round-robin critique → KEEP / FIX / REBUILD loop (§3.5, §5) with
                  the update operator of the method:
                    adr           patch A(j) = {j} ∪ Desc(j)            (proposed)
                    adr_all_deps  patch {j..m}, all-earlier dependencies (ablation 1)
                    suffix        regenerate steps j..end, keep 1..j-1   (position locality)
                    full_rewrite  complete rewrite on FIX and REBUILD    (= ablation 2)
  cimr_gen        the original paper: N rollouts × D × (rewrite → critique → correct)
  maj_gen         compute-matched majority sampling under the same budget B

Budget rules (§5): every call, failed output and fallback is charged; before a
turn the maximum cost of a critique-plus-update is reserved (only actual cost
is charged, so KEEP returns the unused allowance); an incomplete or invalid
update never replaces a complete solution.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from src.cimr_prompts import (  # type: ignore
    CORRECTION_SYSTEM_PROMPT, CRITIQUE_SYSTEM_PROMPT, REASONING_SYSTEM_PROMPT,
    build_continuation_reasoning_prompt, build_initial_reasoning_prompt,
    build_self_correction_prompt, build_self_critique_prompt,
)
from src.cimr_step import truncate_at_first_boxed  # type: ignore

from .backends import GenRequest, GenResult, GenerationBackend
from .budget import CostMeter
from .config import ADRConfig
from .prompts import (Feedback, critique_prompt, generation_prompt, parse_feedback, patch_prompt,
                      rewrite_prompt, suffix_prompt)
from .state import QState, Rollout
from .structure import (apply_patch, apply_suffix, parse_full, render, repair_set_adr,
                        repair_set_suffix, solution_hash, with_all_deps)
from .voting import extract_answer, normalize_answer


class Ctx:
    """Shared helpers: request building, token bounds for reservation."""

    def __init__(self, cfg: ADRConfig, backend: GenerationBackend):
        self.cfg = cfg
        self.backend = backend
        self._tok: Dict[str, int] = {}

    def ptok(self, system: str, user: str) -> int:
        key = hashlib.md5((system + "\x00" + user).encode()).hexdigest()
        if key not in self._tok:
            self._tok[key] = self.backend.prompt_tokens(system, user)
        return self._tok[key]

    def cost(self, meter: CostMeter, n_in: int, n_out: int) -> float:
        return meter.cost_of(min(n_in, self.backend.max_input_tokens), n_out)

    def new_meter(self, budget: float) -> CostMeter:
        return CostMeter(self.cfg.cost_measure, self.backend.num_params, budget=budget)

    def seed(self, st: QState) -> int:
        st.call_counter += 1
        return int(hashlib.md5(f"{st.seed_base}:{st.qid}:{st.call_counter}".encode()).hexdigest()[:8], 16)

    def req(self, st: QState, role: str, sys_user, max_new: int, temp: float) -> GenRequest:
        s, u = sys_user
        return GenRequest(role, s, u, max_new, temp, self.cfg.top_p, self.seed(st))

    def extract(self, text: str) -> Tuple[str, str]:
        a = extract_answer(text, self.cfg.extract_fallback_last_number)
        return a, normalize_answer(a)

    # -- worst-case reservations ------------------------------------------
    def wc_critique(self, meter, q: str, sol: str) -> float:
        return self.cost(meter, self.ptok(*critique_prompt(q, sol)), self.cfg.critique_max_new_tokens)

    def wc_rewrite(self, meter, q: str, sol: str) -> float:
        n_in = self.ptok(*rewrite_prompt(q, sol, "")) + self.cfg.critique_max_new_tokens
        return self.cost(meter, n_in, self.cfg.gen_max_new_tokens)

    def wc_generation(self, meter, q: str) -> float:
        return self.cost(meter, self.ptok(*generation_prompt(q)), self.cfg.gen_max_new_tokens)


def _call(st: QState, req: GenRequest):
    res = (yield [req])[0]
    st.meter.charge(req.role, res.n_in, res.n_out)
    return res


def _set_complete(ctx: Ctx, r: Rollout, steps, text: str, kind: str, turn: int) -> None:
    r.steps = steps
    r.text = text
    r.answer, r.norm = ctx.extract(text)
    h = solution_hash(text)
    r.history.append(dict(kind=kind, turn=turn, answer=r.answer, n_steps=len(steps) if steps else 0))
    r.seen.append(h)


# =============================================================================
# Shared prefix: structured solutions + first critique
# =============================================================================

def prefix_gen(ctx: Ctx, st: QState):
    t0 = time.time()
    if st.extra.get("prefix_init_done"):          # resumed after the initial solutions were charged
        reqs, outs = [], []
    else:
        reqs = [ctx.req(st, "generate", generation_prompt(st.question), ctx.cfg.gen_max_new_tokens,
                        ctx.cfg.temperature) for _ in range(st.n)]
        outs = yield reqs
    for r, req, res in zip(st.rollouts, reqs, outs):
        st.meter.charge(req.role, res.n_in, res.n_out)
        steps, errs = parse_full(res.text) if res.ok else (None, ["empty output"])
        r.history.append(dict(kind="initial", raw_ok=res.ok, format_errors=errs))
        if steps is not None:
            _set_complete(ctx, r, steps, render(steps), "initial", 0)
        else:
            # malformed: keep the raw text as the (unstructured) complete answer holder so it can
            # still vote; the rollout will use the full-rewrite fallback until it is structured
            r.text = res.text or ""
            r.answer, r.norm = ctx.extract(r.text)
            r.steps = None
    st.extra["prefix_init_done"] = True
    creqs, idxs = [], []
    for r in st.rollouts:
        if r.steps is not None and r.pending is None:
            creqs.append(ctx.req(st, "critique", critique_prompt(st.question, r.text),
                                 ctx.cfg.critique_max_new_tokens, ctx.cfg.critique_temperature))
            idxs.append(r.idx)
    if creqs:
        couts = yield creqs
        for i, req, res in zip(idxs, creqs, couts):
            st.meter.charge(req.role, res.n_in, res.n_out)
            st.rollouts[i].pending = parse_feedback(res.text) if res.ok else Feedback("INVALID", raw="")
    st.extra["prefix"] = dict(cost=st.meter.snapshot(), elapsed=time.time() - t0,
                              format_ok=[r.steps is not None for r in st.rollouts])
    st.elapsed += time.time() - t0
    return st


# =============================================================================
# Direct-critique variants
# =============================================================================

def _turn_reserve(ctx: Ctx, st: QState, r: Rollout, method: str) -> float:
    m = st.meter
    need = 0.0
    if r.steps is not None and r.pending is None:
        need += ctx.wc_critique(m, st.question, r.text)
    # worst update: a full rewrite (REBUILD is always possible); patches are smaller
    return need + ctx.wc_rewrite(m, st.question, r.text)


def direct_gen(ctx: Ctx, st: QState, method: str):
    """Round-robin repair loop for adr | adr_all_deps | suffix | full_rewrite."""
    t0 = time.time()
    cfg = ctx.cfg
    while True:
        # the next active rollout whose turn is affordable (round-robin)
        chosen = None
        for k in range(st.n):
            i = (st.rr + k) % st.n
            r = st.rollouts[i]
            if not r.active:
                continue
            if r.n_turns >= cfg.max_turns_per_rollout:
                r.active, r.freeze_reason = False, "turn_cap"
                continue
            if st.meter.can_afford(_turn_reserve(ctx, st, r, method)):
                chosen = i
                break
        if chosen is None:
            st.stop_reason = "all_frozen" if not any(r.active for r in st.rollouts) else "budget"
            break
        st.rr = (chosen + 1) % st.n
        r = st.rollouts[chosen]
        r.n_turns += 1
        ev: Dict[str, Any] = dict(turn=len(st.events), rollout=chosen, method=method,
                                  spent_before=st.meter.spent, answer_before=r.answer,
                                  n_steps_before=len(r.steps) if r.steps else 0)
        # 1. critique (the shared first critique is reused, not re-run)
        if r.steps is None:
            fb = Feedback("REBUILD", text="the previous solution is not in the required numbered format")
            ev["critique"] = "skipped_malformed"
        elif r.pending is not None:
            fb, r.pending = r.pending, None
            ev["critique"] = "shared_first"
        else:
            res = yield from _call(st, ctx.req(st, "critique", critique_prompt(st.question, r.text),
                                               cfg.critique_max_new_tokens, cfg.critique_temperature))
            fb = parse_feedback(res.text) if res.ok else Feedback("INVALID")
            ev["critique"] = "called"
        ev.update(feedback=fb.kind, fix_step=fb.step, error=fb.text[:300])

        # 2. KEEP / unparseable critique → freeze (a greedy critic would repeat itself)
        if fb.kind in ("KEEP", "INVALID"):
            r.active = False
            r.freeze_reason = "keep" if fb.kind == "KEEP" else "critic_unparseable"
            ev.update(update="none", frozen=r.freeze_reason)
            _finish_event(st, r, ev)
            continue

        # 3. update
        m = len(r.steps) if r.steps else 0
        valid_fix = fb.kind == "FIX" and r.steps is not None and 1 <= fb.step <= m
        candidate, text, errs = None, "", []
        if valid_fix and method in ("adr", "adr_all_deps", "suffix"):
            j = fb.step
            base_steps = with_all_deps(r.steps) if method == "adr_all_deps" else r.steps
            A = repair_set_suffix(r.steps, j) if method != "adr" else repair_set_adr(base_steps, j)
            ev.update(repair_ids=A, repair_size=len(A), suffix_size=m - j + 1, m=m,
                      update="suffix" if method == "suffix" else "patch")
            if method == "suffix":
                prefix = render([s for s in r.steps if s.id < j])
                res = yield from _call(st, ctx.req(st, "suffix", suffix_prompt(st.question, prefix, fb.text, j),
                                                   cfg.patch_max_new_tokens, cfg.temperature))
                if res.ok:
                    candidate, errs = apply_suffix(r.steps, res.text, j)
                else:
                    errs = ["empty output"]
            else:
                res = yield from _call(st, ctx.req(st, "patch", patch_prompt(st.question, r.text, fb.text, A),
                                                   cfg.patch_max_new_tokens, cfg.temperature))
                if res.ok:
                    candidate, errs = apply_patch(r.steps, res.text, A)
                else:
                    errs = ["empty output"]
            ev["out_tokens"] = res.n_out
            if candidate is not None:
                ev["regenerated_steps"] = (len(candidate) - (j - 1)) if method == "suffix" else len(A)
        # full rewrite: REBUILD, malformed FIX, the full_rewrite method, or fallback after an invalid patch
        need_rewrite = candidate is None
        if need_rewrite:
            reason = fb.text if fb.kind in ("REBUILD", "FIX") else ""
            if valid_fix and method in ("adr", "adr_all_deps", "suffix"):
                ev.update(fallback=True, fallback_errors=errs[:3])
                if not st.meter.can_afford(ctx.wc_rewrite(st.meter, st.question, r.text)):
                    ev.update(update_result="fallback_unaffordable_kept_previous")
                    _finish_event(st, r, ev)
                    continue
            elif fb.kind == "FIX" and not valid_fix and r.steps is not None:
                ev.update(fallback=True, fallback_errors=[f"FIX step {fb.step} out of range 1..{m}"])
            res = yield from _call(st, ctx.req(st, "rewrite", rewrite_prompt(st.question, r.text, reason),
                                               cfg.gen_max_new_tokens, cfg.temperature))
            ev["update"] = "fallback_rewrite" if ev.get("fallback") else "rewrite"
            ev["rewrite_out_tokens"] = res.n_out
            candidate, errs = parse_full(res.text) if res.ok else (None, ["empty output"])
            if candidate is not None:
                ev["regenerated_steps"] = len(candidate)
        # 4. commit (incomplete / invalid never replaces a complete solution)
        if candidate is None:
            ev.update(update_result="invalid_kept_previous", errors=errs[:3])
        else:
            new_text = render(candidate)
            h = solution_hash(new_text)
            repeated = h in r.seen
            _set_complete(ctx, r, candidate, new_text, ev.get("update", "rewrite"), ev["turn"])
            ev.update(update_result="applied", n_steps_after=len(candidate),
                      regenerated_frac=ev.get("regenerated_steps", len(candidate)) / max(1, len(candidate)))
            if repeated:
                r.active, r.freeze_reason = False, "cycle"
                ev["frozen"] = "cycle"
        _finish_event(st, r, ev)
    st.elapsed += time.time() - t0
    return st


def _finish_event(st: QState, r: Rollout, ev: Dict[str, Any]) -> None:
    ev.update(answer_after=r.answer, spent_after=st.meter.spent)
    ev["cost"] = ev["spent_after"] - ev["spent_before"]
    st.events.append(ev)


# =============================================================================
# Original CIMR (paper) baseline: fixed depth, three-stage cycle, no budget cap
# =============================================================================

def cimr_gen(ctx: Ctx, st: QState):
    t0 = time.time()
    cfg = ctx.cfg
    if "cimr_depth_done" not in st.extra:            # resumable: stages are recorded in the state
        reqs = [ctx.req(st, "generate", (REASONING_SYSTEM_PROMPT, build_initial_reasoning_prompt(st.question)),
                        cfg.gen_max_new_tokens, cfg.temperature) for _ in range(st.n)]
        outs = yield reqs
        for r, req, res in zip(st.rollouts, reqs, outs):
            st.meter.charge(req.role, res.n_in, res.n_out)
            r.text = truncate_at_first_boxed(res.text or "")
            r.answer, r.norm = ctx.extract(r.text)
        st.extra["cimr_depth_done"] = 0
    for d in range(st.extra["cimr_depth_done"] + 1, cfg.cimr_depth + 1):
        before = [r.answer for r in st.rollouts]
        gq = [ctx.req(st, "rewrite", (REASONING_SYSTEM_PROMPT, build_continuation_reasoning_prompt(
            st.question, r.text, d + 1, cfg.cimr_depth + 1, cfg.cimr_max_prior_chars)),
            cfg.gen_max_new_tokens, cfg.temperature) for r in st.rollouts]
        gouts = yield gq
        gs = []
        for req, res in zip(gq, gouts):
            st.meter.charge(req.role, res.n_in, res.n_out)
            gs.append(truncate_at_first_boxed(res.text or ""))
        cq = [ctx.req(st, "critique", (CRITIQUE_SYSTEM_PROMPT, build_self_critique_prompt(
            st.question, g, ctx.extract(g)[0], cfg.cimr_critique_max_trace_chars)),
            cfg.critique_max_new_tokens, cfg.critique_temperature) for g in gs]
        couts = yield cq
        for req, res in zip(cq, couts):
            st.meter.charge(req.role, res.n_in, res.n_out)
        rq = [ctx.req(st, "correct", (CORRECTION_SYSTEM_PROMPT, build_self_correction_prompt(
            st.question, g, c.text or "", ctx.extract(g)[0], cfg.cimr_critique_max_trace_chars,
            cfg.cimr_correction_max_critique_chars)), cfg.gen_max_new_tokens, cfg.temperature)
            for g, c in zip(gs, couts)]
        routs = yield rq
        for r, g, req, res in zip(st.rollouts, gs, rq, routs):
            st.meter.charge(req.role, res.n_in, res.n_out)
            corr = truncate_at_first_boxed(res.text or "").strip()
            r.text = corr if corr else g                    # paper code: never forward a blank trace
            r.answer, r.norm = ctx.extract(r.text)
        st.events.append(dict(turn=d, method="cimr", depth=d, answers_before=before,
                              answers_after=[r.answer for r in st.rollouts], spent_after=st.meter.spent))
        st.extra["cimr_depth_done"] = d
    st.stop_reason = f"depth_{cfg.cimr_depth}"
    st.elapsed += time.time() - t0
    return st


# =============================================================================
# Compute-matched majority sampling
# =============================================================================

def maj_gen(ctx: Ctx, st: QState, max_samples: int = 512):
    """Draw structured solutions (same prompt as the initial ADR solutions)
    while the SAME budget B affords another one; every sample votes."""
    t0 = time.time()
    answers = st.extra.setdefault("answers", [])
    norms = st.extra.setdefault("norms", [])
    while len(answers) < max_samples:
        per = ctx.wc_generation(st.meter, st.question)
        k = int(st.meter.remaining // per) if per > 0 else 0
        k = min(k, max_samples - len(answers), max(ctx.cfg.batch_size, 1))
        if k <= 0:
            break
        reqs = [ctx.req(st, "generate", generation_prompt(st.question), ctx.cfg.gen_max_new_tokens,
                        ctx.cfg.temperature) for _ in range(k)]
        outs = yield reqs
        for req, res in zip(reqs, outs):
            st.meter.charge(req.role, res.n_in, res.n_out)
            a, n = ctx.extract(res.text or "")
            answers.append(a)
            norms.append(n)
    st.stop_reason = "budget"
    st.elapsed += time.time() - t0
    return st


__all__ = ["Ctx", "prefix_gen", "direct_gen", "cimr_gen", "maj_gen"]
