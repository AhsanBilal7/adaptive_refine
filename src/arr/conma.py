"""
ConMA baseline (Wang et al., "ConMA: Confidence-Guided Kernel Sampling with Multi-Stage Aggregation for
LLM Reasoning", Findings of ACL 2026), run under the same per-question FLOP budget as every other method.

One round of ConMA (Section 3 and Figure 2 of the ConMA paper):
  CKS  every trajectory gets S(τ) = α·C(τ_full) + β·C(τ_tail), C = geometric mean of token probabilities
       (Eq. 1–2; the last W tokens form the tail).  Trajectories are grouped by final answer and scored
       S(G) = λ·|G|/n + (1−λ)·mean S(τ) (Eq. 3); a temperature softmax over groups (Eq. 4) and the smallest
       top set with cumulative probability ≥ ρ (Eq. 5) form the kernel K.
  stop |kernel| = 1 → converged: the kernel's answer is the final answer.
  DSE  if |K| > δ·n, n − |K| trajectories are sampled from the critical-review prompt that lists the kernel
       representatives (highest-S member of each group, ranked by P(G)); a new answer adds a group, a known
       answer replaces that group's representative only if its S is higher.
  MSA  n trajectories are sampled from the multiple-choice prompt over the representatives; they form
       the next round's pool.
The loop starts from n trajectories of the shared solve prompt.  It ends on convergence, or when the next
stage no longer fits the budget; then the highest-confidence trajectory of the last round is the answer.

Choices that the paper leaves open, fixed here before any run:
  * the softmax temperature of Eq. 4 is not given; we use τ_temp = 0.1 (S values lie in [0, 1]);
  * the paper caps the work at T_max = 4 rounds of n = 8; under our FLOP budget the number of rounds is set by
    the budget (``conma_max_rounds`` = 0) so that ConMA, like every method, may spend the whole budget;
  * each candidate shown in a DSE / MSA prompt keeps the tail of its solution (the conclusion and boxed answer)
    so that the prompt fits the model's input cap without truncating the question;
  * an MSA answer that is a bare option letter is mapped back to that option's answer;
  * sampling uses the shared temperature / top-p of all methods (the paper uses T = 0.7, top-p = 0.9).

Critical path: stages are barriers (all n calls of a stage run in parallel; the next stage needs all of
them), so the critical path is the sum over stages of the longest call of the stage.
"""

from __future__ import annotations

import math
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .engine import Request
from .ledger import Ledger
from .methods import root_seed
from .prompts import conma_dse_prompt, conma_msa_prompt, solve_prompt


@dataclass
class Traj:
    id: int
    stage: str                      # sample | dse | msa
    round: int
    lane: int
    text: str = ""
    ids: List[int] = field(default_factory=list)
    answer: str = ""
    norm: str = ""
    valid: bool = False
    S: float = 0.0
    finish: str = ""


class ConmaSolve:
    """One (question, seed, budget) ConMA solve driven by engine results (same protocol as methods.Solve)."""

    def __init__(self, method: str, cfg, engine, qid: str, question: str, seed_idx: int, budget: float,
                 cohort=None, task=None, meta: Optional[Dict[str, Any]] = None):
        self.method, self.cfg, self.engine = method, cfg, engine
        self.qid, self.question, self.seed_idx = qid, question, seed_idx
        self.ledger = Ledger(budget=budget, measure=cfg.cost_measure)
        from .tasks import get_task
        self.task = task or get_task("math500")
        self.meta = meta or {}
        self._sys = self.task.system
        self._fmt = self.task.format_text
        self.n = int(cfg.conma_width)
        self.trajs: List[Traj] = []
        self.inflight: List[Request] = []
        self.stage = ""
        self.round = 0
        self.stage_passes: List[int] = []          # longest call of every finished stage
        self._stage_max = 0
        self.last_pool: List[Traj] = []
        self._prev_pool: List[Traj] = []           # last round that had at least one answered trajectory
        self.reps: Dict[str, Traj] = {}             # norm -> representative, in rank order
        self.kernel_sizes: List[int] = []
        self.events: List[Dict[str, Any]] = []
        self.final: Optional[Traj] = None
        self.converged = False
        self.t0 = time.time()
        self.t_end = 0.0
        self.stop_reason = ""
        self._solve_ids = engine.format(*solve_prompt(question, self._sys))

    # scoring (CKS)
    def _score(self, lp: List[float]) -> float:
        if not lp:
            return 0.0
        c = self.cfg
        full = math.exp(sum(lp) / len(lp))
        tail = lp[-int(c.conma_tail):]
        return c.conma_alpha * full + c.conma_beta * math.exp(sum(tail) / len(tail))

    def _kernel(self, pool: List[Traj]) -> List[Dict[str, Any]]:
        """Answer groups of the pool, ranked by P(G); the kernel is the smallest top set with Σ P ≥ ρ."""
        c = self.cfg
        groups: Dict[str, List[Traj]] = {}
        for t in pool:
            if t.valid and t.norm:
                groups.setdefault(t.norm, []).append(t)
        if not groups:
            return []
        n = max(1, len(pool))
        scored = []
        for norm, ts in groups.items():
            s = c.conma_lambda * len(ts) / n + (1 - c.conma_lambda) * sum(t.S for t in ts) / len(ts)
            scored.append(dict(norm=norm, members=ts, score=s, rep=max(ts, key=lambda t: t.S)))
        mx = max(g["score"] for g in scored)
        z = sum(math.exp((g["score"] - mx) / c.conma_temp) for g in scored)
        for g in scored:
            g["p"] = math.exp((g["score"] - mx) / c.conma_temp) / z
        scored.sort(key=lambda g: (-g["p"], -g["rep"].S))
        kernel, cum = [], 0.0
        for g in scored:
            kernel.append(g)
            cum += g["p"]
            if cum >= c.conma_rho - 1e-12:
                break
        return kernel

    # prompts
    def _cand_tail(self, text: str, max_tokens: int) -> str:
        """The last ``max_tokens`` tokens of a candidate solution (its conclusion and boxed answer)."""
        tok = self.engine.tokenizer
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            return text.strip()
        return "... " + tok.decode(ids[-max_tokens:], skip_special_tokens=True).strip()

    def _per_candidate(self, k: int) -> int:
        """Equal share of the input cap for each of k candidates (question and template reserved)."""
        base = len(self.engine.format(*conma_msa_prompt(self.question, "", self._sys, self._fmt)))
        room = int(self.cfg.max_input_tokens) - base - 24 * k - 16
        return max(64, room // max(1, k))

    def _options(self) -> List[Traj]:
        return list(self.reps.values())

    def _letters(self, k: int) -> List[str]:
        return list(string.ascii_uppercase[:k])

    def _dse_ids(self) -> List[int]:
        opts = self._options()
        m = self._per_candidate(len(opts))
        txt = "".join(f"\n\nCandidate solution {i + 1} (final answer: {t.answer}):\n{self._cand_tail(t.text, m)}"
                      for i, t in enumerate(opts))
        return self.engine.format(*conma_dse_prompt(self.question, txt, self._sys, self._fmt))

    def _msa_ids(self) -> List[int]:
        opts = self._options()
        m = self._per_candidate(len(opts))
        txt = "".join(f"\n\n({L}) Final answer: {t.answer}\n{self._cand_tail(t.text, m)}"
                      for L, t in zip(self._letters(len(opts)), opts))
        return self.engine.format(*conma_msa_prompt(self.question, txt, self._sys, self._fmt))

    # requests
    def _affordable(self, prompt_len: int, k: int) -> int:
        """How many of k parallel calls with this prompt fit the remaining budget in the worst case."""
        wc = self.engine.worst_case(prompt_len, self.cfg.max_new_tokens, self.ledger.measure)
        if self.ledger.measure == "passes":
            return k if self.ledger.remaining >= wc else 0
        outstanding = sum(max(0.0, r.wc - r.spent) for r in self.inflight)
        return max(0, min(k, int((self.ledger.remaining - outstanding) // max(wc, 1e-9))))

    def _launch(self, stage: str, ids: List[int], k: int) -> List[Request]:
        k = self._affordable(len(ids), k)
        if k <= 0:
            return []
        self.stage, self._stage_max = stage, 0
        if stage != "dse":
            self.round += 1
        out = []
        for lane in range(k):
            t = Traj(id=len(self.trajs), stage=stage, round=self.round, lane=lane)
            self.trajs.append(t)
            salt = {"sample": 0, "dse": 3_000_000, "msa": 4_000_000}[stage] + 1000 * self.round
            req = Request(ids, self.cfg.max_new_tokens, "sample", self.ledger, role=stage,
                          temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                          seed=root_seed(self.cfg.seed, self.seed_idx, self.qid, salt + lane), tag=(stage, t),
                          logprobs=[])
            self.inflight.append(req)
            out.append(req)
        return out

    # protocol
    def start(self) -> List[Request]:
        self.t0 = time.time()
        out = self._launch("sample", self._solve_ids, self.n)
        if not out:
            self._finish("budget")
        return out

    def on_result(self, req: Request) -> List[Request]:
        self.inflight = [r for r in self.inflight if r is not req]
        self.ledger.counts["calls"] += 1
        stage, t = req.tag
        t.finish = req.finish
        self._stage_max = max(self._stage_max, req.stats["passes"])
        if req.complete:
            t.ids, t.text = list(req.out), req.text
            t.answer, t.norm = self.task.extract(t.text, self.meta, self.cfg.extract_fallback_last_number)
            if stage == "msa":
                self._map_letter(t)
            t.valid = bool(t.norm)
            t.S = self._score(req.logprobs or [])
        self.events.append(dict(root=t.id, lane=t.lane, op=stage, round=t.round, mode=req.mode, finish=req.finish,
                                n_out=len(req.out), passes=req.stats["passes"], spent=req.spent,
                                prefill=req.stats["prefill"], decoded=req.stats["decoded"], answer_after=t.answer,
                                S=t.S))
        if self.inflight:
            return []
        self.stage_passes.append(self._stage_max)
        return self._advance(stage)

    def _map_letter(self, t: Traj) -> None:
        opts = self._options()
        a = t.answer.strip().strip("()").strip()
        if len(a) == 1 and a.upper() in self._letters(len(opts)):
            o = opts[self._letters(len(opts)).index(a.upper())]
            t.answer, t.norm = o.answer, o.norm

    def _advance(self, stage: str) -> List[Request]:
        pool = [t for t in self.trajs if t.stage == stage and t.round == self.round]
        if stage == "dse":
            for t in pool:                            # competitive expansion
                if not t.valid:
                    continue
                cur = self.reps.get(t.norm)
                if cur is None or t.S > cur.S:
                    self.reps[t.norm] = t
            if len(self.reps) == 1:
                return self._converge(next(iter(self.reps.values())))
            return self._msa()
        # a sampling round (initial or MSA) finished: CKS
        self.last_pool = pool
        kernel = self._kernel(pool)
        self.kernel_sizes.append(len(kernel))
        if not kernel:                                # no complete, answered trajectory this round: sample again
            self.last_pool = self._prev_pool or pool  # (the paper does not cover this case)
            if self.reps:                             # MSA again over the current options
                return self._msa()
            out = self._launch("sample", self._solve_ids, self.n)
            return out if out else self._out_of_budget()
        self._prev_pool = pool
        self.reps = {g["norm"]: g["rep"] for g in kernel}
        if len(kernel) == 1:
            return self._converge(kernel[0]["rep"])
        if self.cfg.conma_max_rounds and self.round >= self.cfg.conma_max_rounds:
            return self._out_of_budget("max_rounds")
        if len(kernel) > self.cfg.conma_delta * self.n and self.n - len(kernel) > 0:
            out = self._launch("dse", self._dse_ids(), self.n - len(kernel))
            if out:
                return out
        return self._msa()

    def _msa(self) -> List[Request]:
        out = self._launch("msa", self._msa_ids(), self.n)
        return out if out else self._out_of_budget()

    def _converge(self, t: Traj) -> List[Request]:
        self.final, self.converged = t, True
        self._finish("converged")
        return []

    def _out_of_budget(self, why: str = "budget") -> List[Request]:
        valid = [t for t in self.last_pool if t.valid]
        if valid:
            self.final = max(valid, key=lambda t: t.S)
        elif self.reps:
            self.final = max(self.reps.values(), key=lambda t: t.S)
        self._finish(why)
        return []

    def _finish(self, why: str) -> None:
        self.stop_reason = why
        self.t_end = time.time()

    @property
    def done(self) -> bool:
        return bool(self.t_end) and not self.inflight

    # result
    def record(self) -> Dict[str, Any]:
        last = [t for t in self.last_pool if t.valid]
        counts: Dict[str, int] = {}
        for t in last:
            counts[t.norm] = counts.get(t.norm, 0) + 1
        final = self.final.answer if self.final is not None else "[invalid]"
        keep = {t.id for t in self.reps.values()} | ({self.final.id} if self.final else set())
        return dict(
            qid=self.qid, seed=self.seed_idx, method=self.method, budget=self.ledger.budget,
            final_answer=final, votes=counts, n_voting_roots=len(last),
            roots=[dict(id=t.id, lane=t.lane, status=t.stage, valid=t.valid, revisions=0, answer=t.answer,
                        answers=[t.answer] if t.valid else [], n_tokens=len(t.ids), round=t.round, S=t.S,
                        text=t.text if t.id in keep else "") for t in self.trajs],
            cost=self.ledger.snapshot(), by_role=self.ledger.by_role,
            latency_sec=(self.t_end or time.time()) - self.t0,
            critical_path_passes=int(sum(self.stage_passes)),
            stop_reason=self.stop_reason, events=self.events,
            conma=dict(rounds=self.round, converged=self.converged, n_samples=len(self.trajs),
                       kernel_sizes=self.kernel_sizes, n_options_final=len(self.reps)),
        )


__all__ = ["ConmaSolve"]
