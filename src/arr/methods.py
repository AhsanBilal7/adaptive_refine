"""Per-question controllers of ARR and of every baseline.

A ``Solve`` runs one (question, seed, method, budget).  It submits engine
``Request``s and reacts to each finished one.  All solves share the engine's
first-in-first-out queue, so roots are served in turn and a new root never
runs ahead of an unfinished one.

Every method starts from the same N sampled roots of the question (the
"cohort", charged to each method's budget), and new roots use the same per-root
seeds, so the methods see identical samples where they overlap.

Method keys (paper name in brackets):

  arr               [ARR] replay revision + fixed-point replacement (proposed)
  replace_ordinary  [ARR_AR] same rule, revision by ordinary greedy decoding
  replace_pld       [ARR_PLD] same rule, revision by prompt-lookup decoding
  arr_no_replace    [ARR_Depth] a fixed point retires its root, but no new root
                    is sampled; the remaining roots keep the budget
  direct_fixed      [Self-Refine] greedy revision for a fixed depth D
  critic_correct    [SETS] critique then correction, with fixed-point replacement
  continuation      [Sequential Revision] sampled continuation until the budget ends
  maj               [Majority Voting] independent samples only
  arr_answer        stopping-rule ablation: retire a root when its answer is unchanged
  arr_once          stopping-rule ablation: retire every root after one revision

Life cycle of a root under ARR:

* A root is born by sampling sigma(x).  A sample that does not finish within
  the output cap or the budget is invalid: it is charged and does not vote, and
  a new root is started if a complete sample is affordable.
* A revision that reproduces the root exactly (termination included) is a fixed
  point: the root keeps its solution as its final vote and a new root is started
  in its slot if affordable.  A changed, complete revision replaces the
  solution and the root is revised again.  An incomplete revision ends the
  root with its last complete solution.
* "Affordable": the worst-case cost of a complete sample fits in the budget
  left after the worst-case cost of the work in flight.
* Vote: one final answer per valid root, majority vote with ties to the lowest
  root index.  Intermediate revisions never vote.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .engine import Request
from .ledger import Ledger
from .prompts import continuation_prompt, correction_prompt, critique_prompt, revise_prompt, solve_prompt
from .voting import extract_answer, majority_vote, normalize_answer, truncate_at_first_boxed

REPLACE = {"arr", "replace_ordinary", "replace_pld", "critic_correct", "maj", "arr_answer", "arr_once"}
REVISE_MODE = {"arr": "replay", "arr_no_replace": "replay", "arr_answer": "replay", "arr_once": "replay",
               "replace_ordinary": "greedy",
               "replace_pld": "pld", "direct_fixed": "greedy"}


def root_seed(seed: int, seed_idx: int, qid: str, root: int) -> int:
    return int(hashlib.sha256(f"{seed}|{seed_idx}|{qid}|{root}".encode()).hexdigest()[:15], 16)


@dataclass
class Root:
    id: int                                   # birth index = root ID (tie-break order)
    lane: int                                 # execution slot it occupies
    ids: List[int] = field(default_factory=list)
    text: str = ""
    answer: str = ""
    norm: str = ""
    valid: bool = False                       # holds a complete solution
    status: str = "active"                    # active | fixed_point | incomplete | depth_done | invalid | budget
    revisions: int = 0                        # completed revision operations
    answers: List[str] = field(default_factory=list)   # answer after 0, 1, 2 ... completed revisions


class Solve:
    """One (question, seed, method, budget) solve driven by engine results."""

    def __init__(self, method: str, cfg, engine, qid: str, question: str, seed_idx: int, budget: float,
                 cohort: List[Dict[str, Any]], task=None, meta: Optional[Dict[str, Any]] = None):
        self.method, self.cfg, self.engine = method, cfg, engine
        self.qid, self.question, self.seed_idx = qid, question, seed_idx
        self.ledger = Ledger(budget=budget, measure=cfg.cost_measure)
        self.cohort = cohort
        self.roots: List[Root] = []
        self.inflight: List[Request] = []
        self.events: List[Dict[str, Any]] = []
        self.lane_passes: Dict[int, int] = {}
        self.t0 = time.time()
        self.t_end = 0.0
        self.stop_reason = ""
        from .tasks import get_task
        self.task = task or get_task("math500")
        self.meta = meta or {}
        self._sys = self.task.system
        self._fmt = self.task.format_text
        self._solve_ids = engine.format(*solve_prompt(question, self._sys))

    def _extract(self, text: str):
        return self.task.extract(text, self.meta, self.cfg.extract_fallback_last_number)

    def _new_root_affordable(self) -> bool:
        wc = self.engine.worst_case(len(self._solve_ids), self.cfg.max_new_tokens, self.ledger.measure)
        if self.ledger.measure == "passes":
            return self.ledger.remaining >= wc
        outstanding = sum(max(0.0, r.wc - r.spent) for r in self.inflight)
        return self.ledger.remaining - outstanding >= wc

    def _submit(self, req: Request) -> Request:
        self.inflight.append(req)
        return req

    def _sample_req(self, root: Root) -> Request:
        return self._submit(Request(
            self._solve_ids, self.cfg.max_new_tokens, "sample", self.ledger, role="sample",
            temperature=self.cfg.temperature, top_p=self.cfg.top_p,
            seed=root_seed(self.cfg.seed, self.seed_idx, self.qid, root.id), tag=("sample", root)))

    def _revise_req(self, root: Root) -> Request:
        ids = self.engine.format(*revise_prompt(self.question, root.text, self.cfg.revise_prompt, self._sys))
        return self._submit(Request(ids, self.cfg.max_new_tokens, REVISE_MODE[self.method], self.ledger,
                                    role="revise", draft=list(root.ids), tag=("revise", root)))

    def _critique_req(self, root: Root, trace: str, stage: str) -> Request:
        ids = self.engine.format(*critique_prompt(self.question, trace, self._extract(trace)[0],
                                                  self.cfg.critique_max_trace_chars, self._fmt))
        return self._submit(Request(ids, self.cfg.critique_max_new_tokens, "greedy", self.ledger, role="critique",
                                    tag=(stage, root, trace)))

    def _correct_req(self, root: Root, trace: str, critique: str, sampled: bool) -> Request:
        ids = self.engine.format(*correction_prompt(self.question, trace, critique, self._extract(trace)[0],
                                                    self.cfg.critique_max_trace_chars,
                                                    self.cfg.correction_max_critique_chars, self._fmt))
        mode = "sample" if sampled else "greedy"
        return self._submit(Request(ids, self.cfg.max_new_tokens, mode, self.ledger, role="correct",
                                    draft=list(root.ids), temperature=self.cfg.temperature if sampled else 0.0,
                                    top_p=self.cfg.top_p, seed=root_seed(self.cfg.seed, self.seed_idx, self.qid,
                                                                         1_000_000 + len(self.events)),
                                    tag=("correct", root, trace)))

    def _continue_req(self, root: Root, stage: str) -> Request:
        d = root.revisions + 2                        # depth index of the next generation
        total = max(self.cfg.fixed_depth + 1, d) if self.method == "continuation" else self.cfg.fixed_depth + 1
        ids = self.engine.format(*continuation_prompt(self.question, root.text, d, total,
                                                      self.cfg.continuation_max_prior_chars, self._fmt))
        return self._submit(Request(ids, self.cfg.max_new_tokens, "sample", self.ledger, role="continue",
                                    temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                                    seed=root_seed(self.cfg.seed, self.seed_idx, self.qid,
                                                   2_000_000 + len(self.events)), tag=(stage, root)))

    def _spawn(self, lane: int) -> List[Request]:
        if self.method not in REPLACE or not self._new_root_affordable():
            return []
        root = Root(id=len(self.roots), lane=lane)
        self.roots.append(root)
        return [self._sample_req(root)]

    def _archive(self, root: Root, status: str) -> None:
        root.status = status

    def _next_op(self, root: Root) -> List[Request]:
        m = self.method
        if m == "maj":
            self._archive(root, "sampled")
            return self._spawn(root.lane)
        if m in REVISE_MODE:
            return [self._revise_req(root)]
        if m == "critic_correct":
            return [self._critique_req(root, root.text, "critique")]
        if m == "continuation":
            return [self._continue_req(root, "continue")]
        raise ValueError(m)

    def _set(self, root: Root, req: Request, text: Optional[str] = None) -> None:
        root.ids = list(req.out)
        root.text = req.text if text is None else text
        root.answer, root.norm = self._extract(root.text)
        root.valid = True

    def start(self) -> List[Request]:
        """Charge the shared initial cohort, then schedule the first operation of each root."""
        self.t0 = time.time()                         # latency starts when the solve starts, not when queued
        out: List[Request] = []
        for k, c in enumerate(self.cohort[: self.cfg.n_active]):
            cost = c["tflops"] if self.ledger.measure == "tflops" else float(c["passes"])
            root = Root(id=len(self.roots), lane=k)
            self.roots.append(root)
            if self.ledger.measure == "tflops" and cost > self.ledger.remaining + 1e-9:
                root.status = "budget"                # an initial root that the budget cannot fund
                continue
            if self.ledger.measure == "tflops":
                self.ledger.spent += cost
            else:
                self.ledger.spent = max(self.ledger.spent, cost)
                self.ledger.passes = max(self.ledger.passes, int(c["passes"]))
            self.ledger.tflops += c["tflops"]
            for kind in ("prefill", "draft_checked", "decoded"):
                self.ledger.counts[kind] += c["stats"].get(kind, 0)
            self.ledger.counts["calls"] += 1
            r = self.ledger.by_role.setdefault("sample", dict(tflops=0.0, positions=0))
            r["tflops"] += c["tflops"]
            r["positions"] += c["stats"].get("prefill", 0) + c["stats"].get("decoded", 0)
            self.lane_passes[k] = c["passes"]
            self.events.append(dict(root=root.id, lane=k, op="sample", finish=c["finish"], cohort=True,
                                    n_out=len(c["ids"]), passes=c["passes"]))
            if c["finish"] != "eos":
                root.status = "invalid"
                out += self._spawn(k)
                continue
            root.ids, root.text = list(c["ids"]), c["text"]
            root.answer, root.norm = self._extract(root.text)
            root.valid = True
            root.answers.append(root.answer)
            out += self._next_op(root)
        self._maybe_done(out)
        return out

    def on_result(self, req: Request) -> List[Request]:
        self.inflight = [r for r in self.inflight if r is not req]
        self.ledger.counts["calls"] += 1
        stage, root = req.tag[0], req.tag[1]
        self.lane_passes[root.lane] = self.lane_passes.get(root.lane, 0) + req.stats["passes"]
        ev = dict(root=root.id, lane=root.lane, op=stage, mode=req.mode, finish=req.finish,
                  n_out=len(req.out), passes=req.stats["passes"], spent=req.spent,
                  checked=req.stats["draft_checked"], accepted=req.stats["draft_accepted"],
                  discarded=req.stats["draft_discarded"], decoded=req.stats["decoded"],
                  prefill=req.stats["prefill"], answer_before=root.answer, revisions_before=root.revisions)
        if req.draft is not None:
            ev.update(draft_len=len(req.draft), lcp=req.lcp, identical=req.identical)
        out = self._handle(stage, root, req, ev)
        ev["answer_after"] = root.answer
        ev["status_after"] = root.status
        self.events.append(ev)
        self._maybe_done(out)
        return out

    def _handle(self, stage: str, root: Root, req: Request, ev: Dict[str, Any]) -> List[Request]:
        m = self.method
        if stage == "sample":                          # a newly born root
            if not req.complete:
                root.status = "invalid"
                return self._spawn(root.lane)
            self._set(root, req)
            root.answers.append(root.answer)
            return self._next_op(root)

        if stage == "revise" or stage == "correct":    # a revision (or the correction of critic_correct)
            if not req.complete:
                self._archive(root, "incomplete" if req.finish == "length" else "budget")
                return []
            root.revisions += 1
            if req.identical:
                ev["fixed_point"] = True
                if m == "direct_fixed":
                    root.answers.append(root.answer)
                    if root.revisions >= self.cfg.fixed_depth:
                        self._archive(root, "depth_done")
                        return []
                    return self._next_op(root)
                self._archive(root, "fixed_point")
                root.answers.append(root.answer)
                return self._spawn(root.lane) if m != "arr_no_replace" else []
            prev_norm = root.norm
            self._set(root, req)
            root.answers.append(root.answer)
            if (m == "arr_answer" and root.norm and root.norm == prev_norm) or m == "arr_once":
                self._archive(root, "answer_stable" if m == "arr_answer" else "depth_done")
                return self._spawn(root.lane)
            if m == "direct_fixed" and root.revisions >= self.cfg.fixed_depth:
                self._archive(root, "depth_done")
                return []
            return self._next_op(root)

        if stage == "critique":                        # critic_correct: critique → correction
            if not req.complete and req.finish == "budget":
                self._archive(root, "budget")
                return []
            return [self._correct_req(root, root.text, req.text, sampled=False)]

        if stage == "continue":                        # continuation-only
            if not req.complete and req.finish == "budget":
                self._archive(root, "budget")
                return []
            text = truncate_at_first_boxed(req.text or "").strip()
            if text:
                self._set(root, req, text)
            root.revisions += 1
            root.answers.append(root.answer)
            return self._next_op(root)

        raise ValueError(stage)

    def _maybe_done(self, new: List[Request]) -> None:
        if not self.inflight and not new and not self.t_end:
            self.t_end = time.time()
            active = [r for r in self.roots if r.status == "active"]
            self.stop_reason = "budget" if active or self.ledger.remaining < 1e-9 else "all_retired"

    @property
    def done(self) -> bool:
        return bool(self.t_end)

    def vote(self):
        voters = [r for r in self.roots if r.valid and r.status != "invalid"]
        return majority_vote([r.norm for r in voters], [r.answer for r in voters]), voters

    def record(self) -> Dict[str, Any]:
        (_, final, counts), voters = self.vote()
        return dict(
            qid=self.qid, seed=self.seed_idx, method=self.method, budget=self.ledger.budget,
            final_answer=final, votes=counts, n_voting_roots=len(voters),
            roots=[dict(id=r.id, lane=r.lane, status=r.status, valid=r.valid, revisions=r.revisions,
                        answer=r.answer, answers=r.answers, n_tokens=len(r.ids), text=r.text) for r in self.roots],
            cost=self.ledger.snapshot(), by_role=self.ledger.by_role,
            latency_sec=(self.t_end or time.time()) - self.t0,
            critical_path_passes=max(self.lane_passes.values()) if self.lane_passes else 0,
            stop_reason=self.stop_reason, events=self.events,
        )


def sample_cohort_requests(cfg, engine, qid: str, question: str, seed_idx: int, ledger: Ledger,
                           system: Optional[str] = None) -> List[Request]:
    """The N initial roots of a question (shared by every method)."""
    ids = engine.format(*(solve_prompt(question) if system is None else solve_prompt(question, system)))
    return [Request(ids, cfg.max_new_tokens, "sample", ledger, role="sample", temperature=cfg.temperature,
                    top_p=cfg.top_p, seed=root_seed(cfg.seed, seed_idx, qid, k), tag=("cohort", qid, seed_idx, k))
            for k in range(cfg.n_active)]


__all__ = ["Solve", "Root", "REPLACE", "REVISE_MODE", "root_seed", "sample_cohort_requests"]
