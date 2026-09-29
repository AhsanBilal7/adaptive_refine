"""
outer_loop.py
=============
Outer loop: turn evidence into a reusable procedure (§6).

    H = original_refinement_procedure
    memory = empty
    for round in 1..R:
        states   = solve_and_log(search_batch[round], H, frozen_model)
        evidence = compare_actions_from_copied_states(states, H)      # + two-slot pairs
        memory.add(evidence)                                           # positive, zero, negative
        patches  = frozen_editor.propose(H, memory)
        candidate = choose_on_search_data(patches, H)
        result   = paired_full_solve(candidate, H, fresh_confirmation[round])
        if acceptance_rule(result): H = candidate
    return freeze(H)            # final test answers never enter memory

Continuous saving / resume
--------------------------
Every round is split into phases whose outputs are persisted in
``<run_dir>/rounds/round_XX/``:

    log.pkl        solve-and-log records + saved states
    evidence.jsonl counterfactual evidence of this round
    proposals.pkl  patches, valid candidates, rejections, editor log
    search.pkl     search-data scores of H and every candidate
    confirm.pkl    paired confirmation records (only if a block was opened)

Inside each phase every batch of solves is checkpointed continuously
(``checkpoint.py``: finished solves appended immediately, in-flight solve
states snapshotted every ``checkpoint_every_s``).  Re-running the SAME command
(same run_dir) therefore resumes mid-phase.  ``memory.jsonl`` is rebuilt from
the per-round evidence files, so evidence is never double-counted.  Completed
rounds are summarised in ``rounds/round_XX.json`` and ``lineage.json``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .backends import GenerationBackend
from .config import RSIConfig, dump_config
from .counterfactual import DevLedger, compare_actions, evidence_yield, solve_and_log
from .editor import LeakGuard, Patch, apply_patch, compile_patches, llm_patches
from .memory import EvidenceMemory
from .procedure import Procedure, h0_procedure
from .checkpoint import atomic_json, atomic_pickle, load_pickle
from .progress import set_context
from .runner import run_solves, run_solves_multi, summarize
from .solver import SolveContext
from .splits import load_jsonl
from .stats import acceptance_test, required_block_size
from .voting import Grader

logger = logging.getLogger(__name__)


def _ledger_add_solves(ledger: DevLedger, phase: str, recs: List[Dict[str, Any]]) -> None:
    for r in recs:
        ledger.add(phase, r["cost"])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


class OuterLoop:
    def __init__(self, cfg: RSIConfig, backend: GenerationBackend, run_dir: str,
                 problems: Dict[str, Dict[str, Any]], splits: Dict[str, Any],
                 final_path: Optional[str] = None):
        self.cfg = cfg
        # the frozen procedure lives at the MODEL level (<root>/<model>/final_procedure.json)
        self.final_path = final_path or os.path.join(run_dir, "final_procedure.json")
        self.backend = backend
        self.run_dir = run_dir
        self.problems = problems
        self.splits = splits
        self.ctx = SolveContext(cfg, backend)
        self.grader = Grader("math500")
        os.makedirs(os.path.join(run_dir, "rounds"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "procedures"), exist_ok=True)
        self.memory = EvidenceMemory(None, signed=cfg.memory_signed)
        self.ledger = DevLedger()
        self.lineage_path = os.path.join(run_dir, "lineage.json")
        self.lineage: List[Dict[str, Any]] = []
        if os.path.exists(self.lineage_path):
            with open(self.lineage_path) as f:
                self.lineage = json.load(f)
        dev_ids = {q for b in splits["search"] + splits["confirm"] for q in b}
        self.guard = LeakGuard([problems[q]["question"] for q in dev_ids],
                               [problems[q]["gold"] for q in dev_ids])
        self.test_ids = set(splits.get("test", []))
        dump_config(cfg, os.path.join(run_dir, "config.json"))

    # ------------------------------------------------------------------
    def _batch(self, ids: List[str]) -> List[Dict[str, Any]]:
        bad = [q for q in ids if q in self.test_ids]
        assert not bad, f"sealed test ids leaked into development: {bad[:3]}"
        return [self.problems[q] for q in ids if q in self.problems]

    def _save_proc(self, proc: Procedure, tag: str) -> str:
        path = os.path.join(self.run_dir, "procedures", f"{tag}_{proc.hash()}.json")
        proc.save(path)
        return path

    def _gold_map(self, probs) -> Dict[str, Optional[str]]:
        return {p["qid"]: p.get("gold") for p in probs}

    def _rng(self, r: int, phase: str) -> random.Random:
        return random.Random(f"{self.cfg.seed}|{r}|{phase}")

    def _status(self, **kw) -> None:
        atomic_json(dict(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S")),
                    os.path.join(self.run_dir, "status.json"))

    def _rebuild_memory(self, upto_round: int) -> None:
        """Memory M_t = evidence of all rounds ≤ upto_round (rebuilt, never appended twice)."""
        self.memory = EvidenceMemory(None, signed=self.cfg.memory_signed)
        for rr in range(upto_round + 1):
            p = os.path.join(self.run_dir, "rounds", f"round_{rr:02d}", "evidence.jsonl")
            if os.path.exists(p):
                with open(p) as f:
                    self.memory.add(json.loads(l) for l in f if l.strip())
        with open(os.path.join(self.run_dir, "memory.jsonl.tmp"), "w") as f:
            for r in self.memory.records:
                f.write(json.dumps(r, default=str) + "\n")
        os.replace(os.path.join(self.run_dir, "memory.jsonl.tmp"), os.path.join(self.run_dir, "memory.jsonl"))

    def _phase(self, rdir: str, name: str, fn):
        """Run fn() once; its (result, ledger) is persisted so a resume skips it."""
        path = os.path.join(rdir, f"{name}.pkl")
        saved = load_pickle(path)
        if saved is not None:
            logger.info(f"[outer] phase {name}: loaded from {path}")
            out, ledger = saved
            self.ledger = ledger
            return out
        out = fn()
        atomic_pickle((out, self.ledger), path)
        return out

    # ------------------------------------------------------------------
    def run(self) -> Procedure:
        cfg = self.cfg
        H0 = h0_procedure(cfg.h0_depth)
        H = H0
        start_round = 0
        for entry in self.lineage:           # resume after the last completed round
            if entry.get("round") is not None:
                start_round = max(start_round, int(entry["round"]) + 1)
                if entry.get("procedure_path") and os.path.exists(entry["procedure_path"]):
                    H = Procedure.load(entry["procedure_path"])
        led = load_pickle(os.path.join(self.run_dir, "dev_ledger.pkl"))
        if led is not None:
            self.ledger = led
        if start_round == 0:
            self._save_proc(H, "H000")
        alpha_t = cfg.alpha_total / max(cfg.rounds, 1)

        for r in range(start_round, cfg.rounds):
            if cfg.freeze_after_round and r >= cfg.freeze_after_round:
                logger.info(f"[outer] freeze_after_round={cfg.freeze_after_round}: procedure frozen.")
                break
            t0 = time.time()
            rdir = os.path.join(self.run_dir, "rounds", f"round_{r:02d}")
            os.makedirs(rdir, exist_ok=True)
            base = H if cfg.persist_procedure else H0
            base_path = os.path.join(rdir, "base_procedure.json")
            if os.path.exists(base_path):
                base = Procedure.load(base_path)       # same incumbent after a resume
            else:
                base.save(base_path)
            search = self._batch(self.splits["search"][r])
            confirm_ids = self.splits["confirm"][r]
            logger.info("=" * 100)
            logger.info(f"[outer] ROUND {r + 1}/{cfg.rounds}: search={len(search)} problems, "
                        f"confirmation={len(confirm_ids)} problems | procedure H={base.hash()} "
                        f"rules={base.n_rules()} plans={base.n_plans()}")
            logger.info("[outer] phases: 1 solve-and-log → 2 test alternatives → 3 propose edits → "
                        "4 score on search → 5 confirm on fresh block")
            logger.info("=" * 100)

            # 1. solve-and-log
            self._status(round=r, phase="solve_and_log")
            set_context(f"Round {r + 1}/{cfg.rounds} · phase 1/5")
            def _log():
                logged, snaps = solve_and_log(self.ctx, self.backend, search, base, self.grader,
                                              self._rng(r, "log"), seed=0, ckpt=os.path.join(rdir, "ckpt_log"))
                _ledger_add_solves(self.ledger, "solve_and_log", logged)
                return logged, snaps
            logged, snaps = self._phase(rdir, "log", _log)

            # 2. counterfactual comparisons from copied states → evidence file
            ev_path = os.path.join(rdir, "evidence.jsonl")
            if not os.path.exists(ev_path):
                self._status(round=r, phase="counterfactual", states=len(snaps))
                set_context(f"Round {r + 1}/{cfg.rounds} · phase 2/5")
                evidence = compare_actions(self.ctx, self.backend, snaps, base, self._gold_map(search),
                                           self.grader, self._rng(r, "cf"), r, ledger=self.ledger,
                                           ckpt=os.path.join(rdir, "ckpt_cf"))
                for i, e in enumerate(evidence):
                    e["eid"] = f"R{r}E{i}"
                with open(ev_path + ".tmp", "w") as f:
                    for e in evidence:
                        f.write(json.dumps(e, default=str) + "\n")
                os.replace(ev_path + ".tmp", ev_path)
                atomic_pickle(self.ledger, os.path.join(self.run_dir, "dev_ledger.pkl"))
            else:
                led = load_pickle(os.path.join(self.run_dir, "dev_ledger.pkl"))
                if led is not None:
                    self.ledger = led
            # 3. memory (signed: positive, zero and negative) rebuilt from evidence files
            self._rebuild_memory(r)
            evidence = [e for e in self.memory.records if e.get("round") == r]
            ey = evidence_yield(evidence)
            logger.info(f"[outer] round {r}: evidence {ey}")

            # 4. propose patches
            self._status(round=r, phase="proposals")
            set_context(f"Round {r + 1}/{cfg.rounds} · phase 3/5")
            def _propose():
                rng = self._rng(r, "propose")
                patches: List[Patch] = []
                editor_log: List[Dict[str, Any]] = []
                n_llm = cfg.editor_num_proposals if cfg.editor_mode in ("llm", "both") else 0
                if cfg.editor_mode in ("compiler", "both"):
                    patches += compile_patches(self.memory, base, cfg, rng, limit=cfg.editor_num_proposals)
                if n_llm:
                    lp, editor_log, ecost = llm_patches(self.backend, base, self.memory, cfg, n_llm,
                                                        seed=cfg.seed + r)
                    ecost["tflops"] = 2 * self.backend.num_params * (ecost["tokens_in"] + ecost["tokens_out"]) / 1e12
                    self.ledger.add("editor", ecost)
                    patches += lp
                known = {e["eid"] for e in self.memory.records}
                candidates, rejected = [], []
                seen_hash = {base.hash()}
                for pt in patches:
                    newp, errs = apply_patch(base, pt, cfg, self.guard, known_eids=known)
                    if newp is None:
                        rejected.append(dict(patch=pt.to_dict(), errors=errs))
                        continue
                    if newp.hash() in seen_hash:
                        rejected.append(dict(patch=pt.to_dict(), errors=["duplicate procedure"]))
                        continue
                    seen_hash.add(newp.hash())
                    candidates.append((pt, newp))
                return patches, candidates, rejected, editor_log
            patches, candidates, rejected, editor_log = self._phase(rdir, "proposals", _propose)
            logger.info(f"[outer] round {r}: {len(candidates)} valid candidates, {len(rejected)} rejected")

            # 5. choose on search data (incumbent + all candidates in one batched, checkpointed pass)
            self._status(round=r, phase="search_eval", candidates=len(candidates))
            set_context(f"Round {r + 1}/{cfg.rounds} · phase 4/5")
            seeds = list(range(max(1, cfg.search_eval_seeds)))
            def _search():
                arm_recs = run_solves_multi(self.ctx, self.backend, search,
                                            [("procedure", base)] + [("procedure", np_) for _, np_ in candidates],
                                            seeds, self.grader, cfg.utility_lambda, desc="search_eval",
                                            ckpt=os.path.join(rdir, "ckpt_search"))
                for recs in arm_recs:
                    _ledger_add_solves(self.ledger, "search_eval", recs)
                return arm_recs
            arm_recs = self._phase(rdir, "search", _search)
            inc_U = _mean([x["utility"] for x in arm_recs[0]])
            scored = []
            for (pt, newp), recs in zip(candidates, arm_recs[1:]):
                scored.append(dict(patch=pt.to_dict(), procedure_hash=newp.hash(),
                                   utility=_mean([x["utility"] for x in recs]),
                                   accuracy=_mean([float(x["correct"]) for x in recs if x["correct"] is not None]),
                                   proc=newp))
            best = max(scored, key=lambda s: s["utility"], default=None)

            round_rec: Dict[str, Any] = dict(
                round=r, base_hash=base.hash(), evidence_yield=ey, memory_counts=self.memory.counts(),
                logged_summary=summarize(logged), n_patches=len(patches), rejected=rejected,
                editor_log=editor_log, search_incumbent_utility=inc_U,
                search_scores=[{k: v for k, v in s.items() if k != "proc"} for s in scored],
            )
            decision = "retain"
            conf_stats = None
            # 6-7. paired confirmation on a FRESH block; accept or retain
            if best is not None and inc_U is not None and best["utility"] > inc_U:
                self._status(round=r, phase="confirmation")
                set_context(f"Round {r + 1}/{cfg.rounds} · phase 5/5")
                confirm = self._batch(confirm_ids)
                cseeds = list(range(max(1, cfg.confirm_seeds)))
                def _confirm():
                    cand, inc = run_solves_multi(
                        self.ctx, self.backend, confirm, [("procedure", best["proc"]), ("procedure", base)],
                        cseeds, self.grader, cfg.utility_lambda, desc="confirmation",
                        ckpt=os.path.join(rdir, "ckpt_confirm"))
                    _ledger_add_solves(self.ledger, "confirmation", cand + inc)
                    return cand, inc
                cand_recs, inc_recs = self._phase(rdir, "confirm", _confirm)
                conf_stats = acceptance_test(cand_recs, inc_recs, alpha_t, cfg.noninferiority_margin,
                                             cfg.utility_min_gain, cfg.permutation_samples, cfg.seed + r)
                if conf_stats.get("discordance_rate"):
                    conf_stats["suggested_block_size"] = {
                        f"effect_{e}": required_block_size(conf_stats["discordance_rate"], e, alpha_t)
                        for e in (0.01, 0.02, 0.05)}
                conf_stats["candidate_summary"] = summarize(cand_recs)
                conf_stats["incumbent_summary"] = summarize(inc_recs)
                if conf_stats["accept"]:
                    decision = "accept"
                    newH = best["proc"]
                    newH.lineage = list(base.lineage) + [dict(round=r, patch=best["patch"], parent=base.hash())]
                    H = newH
                elif not cfg.persist_procedure:
                    H = H0
            else:
                round_rec["note"] = "no candidate beat the incumbent on search data; confirmation block not opened"
            round_rec.update(decision=decision, confirmation=conf_stats,
                             chosen_patch=best["patch"] if best else None, alpha_t=alpha_t,
                             elapsed_sec=time.time() - t0, dev_ledger=self.ledger.to_dict())
            path = self._save_proc(H, f"H{r + 1:03d}")
            round_rec["procedure_path"] = path
            atomic_json(round_rec, os.path.join(self.run_dir, "rounds", f"round_{r:02d}.json"))
            atomic_pickle(self.ledger, os.path.join(self.run_dir, "dev_ledger.pkl"))
            self.lineage.append(dict(round=r, decision=decision, procedure_hash=H.hash(), procedure_path=path,
                                     patch=best["patch"] if best else None,
                                     accepted=decision == "accept", confirmation=conf_stats and {
                                         k: conf_stats[k] for k in ("mean_utility_diff", "utility_p_value",
                                                                    "mean_accuracy_diff", "accuracy_lower_bound",
                                                                    "accept", "n_questions")},
                                     rejected_patches=len(rejected), evidence=ey))
            atomic_json(self.lineage, self.lineage_path)
            logger.info(f"[outer] round {r}: decision={decision} H={H.hash()} ({time.time() - t0:.0f}s)")

        final_path = self.final_path
        os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
        H.save(final_path)
        atomic_json(self.ledger.to_dict(), os.path.join(self.run_dir, "dev_ledger.json"))
        self._status(phase="done", final_procedure=final_path, procedure_hash=H.hash())
        logger.info(f"[outer] frozen procedure → {final_path}")
        return H


def load_dev_problems(run_dir: str) -> Dict[str, Dict[str, Any]]:
    return {p["qid"]: p for p in load_jsonl(os.path.join(run_dir, "dev_problems.jsonl"))}


__all__ = ["OuterLoop", "load_dev_problems"]
