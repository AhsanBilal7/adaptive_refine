"""
Experiment driver: runs every method and budget on one dataset and writes

  <root>/<model>/<dataset>/
      config.json                      configuration + checkpoint / template / engine facts
      cohort/seed<k>.jsonl             the shared initial roots (N per question and seed)
      B<budget>/results_<method>.jsonl one record per question × seed (roots, votes, cost, events)
      B<budget>/summary.json           metrics per method + paired comparisons against arr
      summary_all.json                 every budget × method in one table

Everything is appended as soon as a solve finishes; re-running the same
command skips finished (question, seed) records (resume).

Reported: accuracy with k/n per seed (every question retained;
no vote → incorrect); estimated TFLOPs incl. prefill and rejected draft work;
latency p50/p95 (wall-clock and serial passes on the critical path); prompt
positions, checked draft positions and ordinarily decoded positions separately;
root births, revisions per root, fixed-point retirements, accepted-prefix
fraction, final voting roots; wrong→correct and correct→wrong after every
completed revision; the fixed-initial-cohort depth accounting
A(d)−A(d−1) = ρ_rec(d) − ρ_reg(d); paired question-level CIs.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ARRConfig, dump_config
from .engine import Engine, Request
from .ledger import Ledger
from .methods import Solve, sample_cohort_requests
from .conma import ConmaSolve
from .stats import paired_compare
from .voting import majority_vote, normalize_answer

logger = logging.getLogger(__name__)
INF = float("inf")


# Data

def load_questions(dataset: str, max_samples: int = 0, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    from .tasks import load_questions as _load
    return _load(dataset, max_samples, cache_dir)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:          # truncated tail from a kill mid-write
                logger.warning(f"[resume] ignoring a truncated line in {path}")
    return out


def _read_all(path: str) -> List[Dict[str, Any]]:
    """Records of ``path`` and of every shard file ``<stem>.s<i>of<n>.jsonl`` next to it."""
    stem = path[:-len(".jsonl")]
    out = _read_jsonl(path)
    for p in sorted(glob.glob(glob.escape(stem) + ".s*of*.jsonl")):
        out += _read_jsonl(p)
    seen, uniq = set(), []
    for r in out:                                # one record per (question, seed) even if a shard overlapped
        k = (r.get("qid"), r.get("seed"))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def _shard_path(path: str, suffix: str) -> str:
    return path[:-len(".jsonl")] + suffix + ".jsonl" if suffix else path


def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_json(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


# Driving the engine

class Progress:
    def __init__(self, label: str, total: int, every: float):
        self.label, self.total, self.every = label, total, every
        self.t0 = self.last = time.time()
        self.done = 0

    def tick(self, engine: Engine, active: int, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        el = now - self.t0
        eta = el / self.done * (self.total - self.done) if self.done else float("nan")
        logger.info(f"[{self.label}] {self.done}/{self.total} solves ({100 * self.done / max(1, self.total):.0f}%) "
                    f"elapsed={el / 60:.1f}m eta={eta / 60:.1f}m active_solves={active} "
                    f"running_seqs={engine.n_running()}/{engine.rows} queued={len(engine.queue)} "
                    f"passes={engine.stats['passes']:,} pad_frac="
                    f"{engine.stats['padded_positions'] / max(1, engine.stats['padded_positions'] + engine.stats['real_positions']):.2f}")


def drive(engine: Engine, solves: Sequence[Solve], on_done, cap: int, label: str, every: float) -> None:
    """Run solves to completion on one engine, at most ``cap`` concurrently."""
    pending = deque(solves)
    owner: Dict[int, Solve] = {}
    active = 0
    prog = Progress(label, len(solves), every)

    def submit(s: Solve, reqs: List[Request]) -> None:
        for r in reqs:
            engine.submit(r)
            owner[r.rid] = s

    while pending or active:
        while pending and active < cap:
            s = pending.popleft()
            active += 1
            submit(s, s.start())
            if s.done:
                active -= 1
                prog.done += 1
                on_done(s)
        if not engine.busy:
            if active:
                raise RuntimeError("solves are active but the engine has no work")
            continue
        for r in engine.step():
            s = owner.pop(r.rid)
            submit(s, s.on_result(r))
            if s.done:
                active -= 1
                prog.done += 1
                on_done(s)
        prog.tick(engine, active)
    prog.tick(engine, active, force=True)


def run_cohort(cfg: ARRConfig, engine: Engine, questions, seeds, path_fmt: str, label: str,
               system: Optional[str] = None) -> Dict[Tuple[str, int], List[Dict]]:
    """N initial roots per (question, seed), shared by every method.  Uncapped ledger;
    each sample's own cost is recorded and charged later to every method's ledger."""
    cohort: Dict[Tuple[str, int], List[Dict]] = {}
    todo: List[Request] = []
    for k in seeds:
        for rec in _read_all(path_fmt.format(k)):
            cohort[(rec["qid"], k)] = rec["samples"]
        for q in questions:
            if (q["qid"], k) not in cohort:
                todo += sample_cohort_requests(cfg, engine, q["qid"], q["question"], k, Ledger(INF, "tflops"), system)
    if not todo:
        return cohort
    got: Dict[Tuple[str, int], List[Optional[Dict]]] = {}
    for r in todo:
        engine.submit(r)
    prog = Progress(label, len(todo), cfg.log_every_s)
    while engine.busy:
        for r in engine.step():
            _, qid, k, idx = r.tag
            slot = got.setdefault((qid, k), [None] * cfg.n_active)
            slot[idx] = dict(ids=r.out, text=r.text, finish=r.finish, tflops=r.spent, passes=r.stats["passes"],
                             stats=dict(r.stats), latency_sec=r.t_end - r.t_submit)
            prog.done += 1
            if all(x is not None for x in slot):
                cohort[(qid, k)] = slot
                _append_jsonl(_shard_path(path_fmt.format(k), cfg.shard_suffix()), dict(qid=qid, seed=k, samples=slot))
        prog.tick(engine, 0)
    return cohort


# Grading + metrics

def grade(rec: Dict[str, Any], gold: str, task, n_active: int, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score the voted answer, every root version and every revision (labels are used for
    evaluation only), and add the confidence / trajectory metadata used by the figures."""
    meta = meta or {}
    cache: Dict[str, bool] = {}

    def ok(a: str) -> bool:
        if not a or a == "[invalid]":
            return False
        if a not in cache:
            cache[a] = bool(task.grade(a, gold, meta))
        return cache[a]

    rec["gold"] = gold
    rec["correct"] = ok(rec["final_answer"])
    # confidence signals of the vote (for AUROC), computed without labels
    votes = rec.get("votes") or {}
    total = sum(votes.values())
    win = task.norm(rec["final_answer"]) if rec["final_answer"] != "[invalid]" else ""
    fixed = [r for r in rec["roots"] if r["status"] == "fixed_point" and r["valid"]]
    rec["vote_share"] = votes.get(win, 0) / total if total else 0.0
    rec["vote_margin"] = ((votes.get(win, 0) - max([c for k, c in votes.items() if k != win] + [0])) / total
                          if total else 0.0)
    rec["n_distinct_answers"] = len(votes)
    rec["n_fixed_point_roots"] = len(fixed)
    rec["fixed_point_support"] = (sum(task.norm(r["answer"]) == win for r in fixed) / rec["n_voting_roots"]
                                  if rec["n_voting_roots"] else 0.0)
    rec["output_tokens"] = int(sum(e.get("n_out", 0) for e in rec["events"]))
    rec["n_model_calls"] = int(sum(1 for e in rec["events"] if "n_out" in e))
    for r in rec["roots"]:
        r["correct"] = [ok(a) for a in r["answers"]]
    for e in rec["events"]:
        if "answer_before" in e and e.get("op") in ("revise", "correct", "continue"):
            e["before_correct"], e["after_correct"] = ok(e["answer_before"]), ok(e["answer_after"])
    # fixed-initial-cohort depth trajectory (roots born in the cohort only)
    cohort_roots = [r for r in rec["roots"] if r["id"] < n_active]
    depth = max([len(r["answers"]) - 1 for r in cohort_roots if r["answers"]] + [0])
    traj = []
    for d in range(depth + 1):
        norms, disp = [], []
        for r in cohort_roots:
            if r["answers"]:
                a = r["answers"][min(d, len(r["answers"]) - 1)]
                norms.append(task.norm(a))
                disp.append(a)
            else:
                norms.append("")
                disp.append("")
        traj.append(ok(majority_vote(norms, disp)[1]))
    rec["cohort_vote_correct_by_depth"] = traj
    return rec


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Area under the ROC curve of ``scores`` for predicting ``labels`` (Mann–Whitney U,
    ties count 1/2).  None when only one class is present."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    ranks = {}
    allv = sorted(scores)
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j] == allv[i]:
            j += 1
        ranks[allv[i]] = (i + j + 1) / 2.0          # average 1-based rank of the tie group
        i = j
    r_pos = sum(ranks[s] for s in pos)
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def summarize(recs: List[Dict[str, Any]], W: int) -> Dict[str, Any]:
    if not recs:
        return {}
    by_q: Dict[str, List[float]] = {}
    by_seed: Dict[int, List[int]] = {}
    for r in recs:
        by_q.setdefault(r["qid"], []).append(float(r["correct"]))
        by_seed.setdefault(r["seed"], []).append(int(r["correct"]))
    q_acc = [np.mean(v) for v in by_q.values()]
    ev = [e for r in recs for e in r["events"]]
    rev = [e for e in ev if e.get("op") in ("revise", "correct") and "draft_len" in e]
    rev_done = [e for e in rev if e.get("finish") == "eos"]
    fixed = [e for e in rev_done if e.get("identical")]
    changed = [e for e in rev_done if not e.get("identical")]
    upd = [e for e in ev if "before_correct" in e and e.get("finish") == "eos"]
    roots = [x for r in recs for x in r["roots"]]
    cost = lambda k: [r["cost"].get(k, 0) for r in recs]
    root_pairs = [(bool(x["correct"][-1]), x["status"]) for x in roots if x.get("valid") and x.get("correct")]
    solution_tokens = [x["n_tokens"] for x in roots if x.get("valid") and x.get("n_tokens")]
    lat = [r["latency_sec"] for r in recs]
    cpp = [r["critical_path_passes"] for r in recs]
    out = dict(
        n_solves=len(recs), n_questions=len(by_q),
        accuracy=float(np.mean(q_acc)), solved_count=float(np.sum(q_acc)),
        correct_k_of_n={str(s): f"{sum(v)}/{len(v)}" for s, v in sorted(by_seed.items())},
        # compute
        mean_tflops=float(np.mean(cost("tflops"))), median_tflops=_pct(cost("tflops"), 50),
        mean_budget_spent=float(np.mean(cost("spent"))),
        mean_prompt_positions=float(np.mean(cost("prefill"))),
        mean_draft_checked=float(np.mean(cost("draft_checked"))),
        mean_draft_accepted=float(np.mean(cost("draft_accepted"))),
        mean_draft_discarded=float(np.mean(cost("draft_discarded"))),
        mean_draft_recompute=float(np.mean(cost("draft_recompute"))),
        mean_decoded=float(np.mean(cost("decoded"))),
        mean_calls=float(np.mean(cost("calls"))),
        mean_question_passes=float(np.mean(cost("passes"))),
        latency_sec_p50=_pct(lat, 50), latency_sec_p95=_pct(lat, 95),
        critical_path_passes_p50=_pct(cpp, 50), critical_path_passes_p95=_pct(cpp, 95),
        # mechanism
        mean_root_births=float(np.mean([len(r["roots"]) for r in recs])),
        mean_voting_roots=float(np.mean([r["n_voting_roots"] for r in recs])),
        mean_revisions_per_root=float(np.mean([x["revisions"] for x in roots])) if roots else None,
        root_status=dict(Counter(x["status"] for x in roots)),
        mean_fixed_point_retirements=len(fixed) / len(recs),
        n_revisions=len(rev), n_revisions_complete=len(rev_done),
        fixed_point_rate=len(fixed) / len(rev_done) if rev_done else None,
        incomplete_revision_rate=(len(rev) - len(rev_done)) / len(rev) if rev else None,
        mean_accepted_prefix_frac=float(np.mean([e["lcp"] / max(1, e["draft_len"]) for e in rev_done])) if rev_done else None,
        mean_first_mismatch=float(np.mean([e["lcp"] for e in changed])) if changed else None,
        frac_changed_within_first_block=float(np.mean([e["lcp"] < W for e in changed])) if changed else None,
        frac_changed_at_first_token=float(np.mean([e["lcp"] == 0 for e in changed])) if changed else None,
        # accuracy transitions after every completed revision (labels used for evaluation only)
        n_completed_updates=len(upd),
        wrong_to_correct=sum(1 for e in upd if not e["before_correct"] and e["after_correct"]),
        correct_to_wrong=sum(1 for e in upd if e["before_correct"] and not e["after_correct"]),
        invalid_samples=sum(1 for e in ev if e.get("op") == "sample" and e.get("finish") != "eos"),
        # output tokens (every generated token of every call, incl. the shared initial roots)
        mean_output_tokens_per_question=float(np.mean([r.get("output_tokens", 0) for r in recs])),
        mean_output_tokens_per_call=(float(np.sum([r.get("output_tokens", 0) for r in recs]) /
                                           max(1, np.sum([r.get("n_model_calls", 0) for r in recs])))),
        mean_final_solution_tokens=(float(np.mean(solution_tokens)) if solution_tokens else None),
        # label-free confidence of the vote vs correctness (evaluation only)
        auroc_vote_share=auroc([r.get("vote_share", 0.0) for r in recs], [r["correct"] for r in recs]),
        auroc_vote_margin=auroc([r.get("vote_margin", 0.0) for r in recs], [r["correct"] for r in recs]),
        auroc_fixed_point_support=auroc([r.get("fixed_point_support", 0.0) for r in recs],
                                        [r["correct"] for r in recs]),
        mean_vote_share=float(np.mean([r.get("vote_share", 0.0) for r in recs])),
        # root-level: is a fixed-point root more often correct than a non-fixed-point voting root?
        fixed_point_root_accuracy=_mean([c for c, st in root_pairs if st == "fixed_point"]),
        non_fixed_point_root_accuracy=_mean([c for c, st in root_pairs if st != "fixed_point"]),
        stop_reasons=dict(Counter(r["stop_reason"] for r in recs)),
    )
    out["depth_accounting"] = depth_accounting(recs)
    return out


def depth_accounting(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Depth accounting on a fixed initial cohort: same question denominator, adjacent complete depths.
    A(d) − A(d−1) = ρ_rec(d) − ρ_reg(d) and A(D) − A(0) = Σ_d [ρ_rec(d) − ρ_reg(d)]."""
    trajs = [r.get("cohort_vote_correct_by_depth") or [False] for r in recs]
    D = max(len(t) for t in trajs) - 1
    at = lambda t, d: t[min(d, len(t) - 1)]
    n = len(trajs)
    A = [sum(at(t, d) for t in trajs) / n for d in range(D + 1)]
    rec_ = [sum((not at(t, d - 1)) and at(t, d) for t in trajs) / n for d in range(1, D + 1)]
    reg = [sum(at(t, d - 1) and (not at(t, d)) for t in trajs) / n for d in range(1, D + 1)]
    ok = all(abs((A[d] - A[d - 1]) - (rec_[d - 1] - reg[d - 1])) < 1e-12 for d in range(1, D + 1))
    return dict(A=A, rho_rec=rec_, rho_reg=reg, telescopes=ok, n_questions=n)


# Experiment

def _cell(budget: float) -> str:
    return f"B{int(budget) if float(budget).is_integer() else budget}"


def run_dataset(cfg: ARRConfig, engine: Engine, dataset: str, methods: Sequence[str],
                questions: Optional[List[Dict[str, Any]]] = None, tag: str = "",
                budgets: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    from .tasks import get_task
    questions = questions or load_questions(dataset, cfg.max_samples, cfg.hf_cache_dir)
    if cfg.shard_spec():
        si, sn = cfg.shard_spec()
        questions = [q for k, q in enumerate(questions) if k % sn == si]
        logger.info(f"[arr] {dataset}: shard {si}/{sn} → {len(questions)} questions")
    task = get_task(dataset)
    items = {q["qid"]: q for q in questions}
    seeds = cfg.seed_indices()
    budgets = list(budgets or cfg.budget_list())
    ds_dir = os.path.join(cfg.model_dir(), dataset + (f"_{tag}" if tag else ""))
    os.makedirs(os.path.join(ds_dir, "cohort"), exist_ok=True)
    dump_config(cfg, os.path.join(ds_dir, "config.json"), engine_facts(engine))

    if set(methods) <= {"conma"}:                     # ConMA samples its own trajectories (it needs their log-probs)
        cohort = {(q["qid"], k): [] for q in questions for k in seeds}
    else:
        cohort = run_cohort(cfg, engine, questions, seeds, os.path.join(ds_dir, "cohort", "seed{}.jsonl"),
                            f"{dataset} · shared initial cohort", task.system)

    solves: List[Solve] = []
    paths: Dict[Tuple[float, str], str] = {}
    for B in budgets:
        cdir = os.path.join(ds_dir, _cell(B))
        os.makedirs(cdir, exist_ok=True)
        for m in methods:
            p = os.path.join(cdir, f"results_{m}.jsonl")
            paths[(B, m)] = p
            have = {(r["qid"], r["seed"]) for r in _read_all(p)}
            for k in seeds:
                for q in questions:
                    if (q["qid"], k) not in have:
                        cls = ConmaSolve if m == "conma" else Solve
                        solves.append(cls(m, cfg, engine, q["qid"], q["question"], k, B, cohort[(q["qid"], k)],
                                          task=task, meta=q.get("meta")))
    logger.info(f"[arr] {dataset}: {len(questions)} questions × {len(seeds)} seeds × {len(budgets)} budgets × "
                f"{len(methods)} methods → {len(solves)} solves to run")

    def on_done(s: Solve) -> None:
        q = items[s.qid]
        rec = grade(s.record(), q["gold"], task, cfg.n_active, q.get("meta"))
        _append_jsonl(_shard_path(paths[(s.ledger.budget, s.method)], cfg.shard_suffix()), rec)

    cap = cfg.max_active_questions or max(8, 2 * engine.rows // max(1, cfg.n_active))
    drive(engine, solves, on_done, cap, f"{dataset}", cfg.log_every_s)

    all_summary: Dict[str, Any] = {}
    for B in budgets:
        cdir = os.path.join(ds_dir, _cell(B))
        recs = {m: _read_all(paths[(B, m)]) for m in methods}
        summ = {m: summarize(r, cfg.block_size) for m, r in recs.items()}
        if "arr" in recs:
            summ["paired_vs_arr"] = {
                m: dict(accuracy=paired_compare(recs["arr"], r, "correct"),
                        tflops=paired_compare([dict(x, t=x["cost"]["tflops"]) for x in recs["arr"]],
                                              [dict(x, t=x["cost"]["tflops"]) for x in r], "t"))
                for m, r in recs.items() if m != "arr" and r}
        atomic_json(summ, os.path.join(cdir, "summary.json"))
        for m, s in summ.items():
            if m != "paired_vs_arr":
                all_summary[f"{_cell(B)}/{m}"] = s
    atomic_json(all_summary, os.path.join(ds_dir, "summary_all.json"))
    return all_summary


def engine_facts(engine: Engine) -> Dict[str, Any]:
    """Record the checkpoint, thinking mode, chat template, precision, engine, seed."""
    import torch
    import transformers
    tpl = getattr(engine.tokenizer, "chat_template", None) or ""
    return dict(
        checkpoint=engine.model_id, precision=str(engine.dtype), inference_engine=f"arr.engine (transformers "
        f"{transformers.__version__}, torch {torch.__version__}, attention={engine.model.config._attn_implementation})",
        chat_template_sha256=__import__("hashlib").sha256(tpl.encode()).hexdigest()[:16],
        engine_rows=engine.rows, engine_max_len=engine.max_len, eos_ids=sorted(engine.eos),
        params_without_embedding=engine.params,
        gpu=(torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else "cpu"),
    )


# Feasibility pilot

def pilot(cfg: ARRConfig, engine: Engine) -> Dict[str, Any]:
    """Predeclared 50-question pilot on one model (fits nothing).

    1. replay vs ordinary greedy revision of every complete initial root, run as two
       separate phases under the same batching: exact agreement, accepted-prefix
       lengths, fixed-point frequency, discarded draft work, latency;
    2. arr at the first budget: the mechanism statistics.
    Verdict: little useful signal if output
    changes USUALLY (> 50 %) occur immediately (within the first draft block) AND exact
    fixed points are RARE (< 10 % of complete revisions)."""
    from .tasks import get_task
    ds = cfg.pilot_dataset
    task = get_task(ds)
    qs = load_questions(ds, cfg.pilot_samples, cfg.hf_cache_dir)
    ds_dir = os.path.join(cfg.model_dir(), f"{ds}_pilot")
    os.makedirs(os.path.join(ds_dir, "cohort"), exist_ok=True)
    dump_config(cfg, os.path.join(ds_dir, "config.json"), engine_facts(engine))
    cohort = run_cohort(cfg, engine, qs, [0], os.path.join(ds_dir, "cohort", "seed{}.jsonl"), f"{ds} pilot · cohort",
                        task.system)

    from .prompts import revise_prompt
    pairs = []
    for q in qs:
        for c in cohort[(q["qid"], 0)]:
            if c["finish"] == "eos":
                pairs.append((q, c))
    phase = {}
    for mode in ("replay", "greedy"):
        reqs = [Request(engine.format(*revise_prompt(q["question"], c["text"], cfg.revise_prompt, task.system)), cfg.max_new_tokens,
                        mode, Ledger(INF, "tflops"), role="revise", draft=list(c["ids"])) for q, c in pairs]
        t0 = time.time()
        p0 = engine.stats["passes"]
        engine.run(reqs)
        phase[mode] = dict(reqs=reqs, wall_sec=time.time() - t0, engine_passes=engine.stats["passes"] - p0)
    rp, og = phase["replay"]["reqs"], phase["greedy"]["reqs"]
    agree = [a.out == b.out and a.finish == b.finish for a, b in zip(rp, og)]
    comp = [a for a in rp if a.complete]
    changed = [a for a in comp if not a.identical]
    check = dict(
        n_revisions=len(rp),
        exact_agreement_replay_vs_ordinary=float(np.mean(agree)) if agree else None,
        n_disagreements=int(len(agree) - sum(agree)),
        fixed_point_rate=(sum(a.identical for a in comp) / len(comp)) if comp else None,
        mean_accepted_prefix_frac=float(np.mean([a.lcp / max(1, len(a.draft)) for a in comp])) if comp else None,
        accepted_prefix_len_p50=_pct([a.lcp for a in comp], 50),
        frac_changed_within_first_block=float(np.mean([a.lcp < cfg.block_size for a in changed])) if changed else None,
        frac_changed_at_first_token=float(np.mean([a.lcp == 0 for a in changed])) if changed else None,
        draft_discarded_per_revision=float(np.mean([a.stats["draft_discarded"] for a in rp])),
        draft_checked_per_revision=float(np.mean([a.stats["draft_checked"] for a in rp])),
        passes_per_revision_replay=float(np.mean([a.stats["passes"] for a in rp])),
        passes_per_revision_ordinary=float(np.mean([a.stats["passes"] for a in og])),
        tflops_per_revision_replay=float(np.mean([a.spent for a in rp])),
        tflops_per_revision_ordinary=float(np.mean([a.spent for a in og])),
        wall_sec_replay_phase=phase["replay"]["wall_sec"], wall_sec_ordinary_phase=phase["greedy"]["wall_sec"],
        engine_passes_replay_phase=phase["replay"]["engine_passes"],
        engine_passes_ordinary_phase=phase["greedy"]["engine_passes"],
    )
    with open(os.path.join(ds_dir, "replay_check.jsonl"), "w") as f:
        for (q, c), a, b in zip(pairs, rp, og):
            f.write(json.dumps(dict(qid=q["qid"], draft_len=len(c["ids"]), replay_finish=a.finish,
                                    ordinary_finish=b.finish, identical_outputs=a.out == b.out, lcp=a.lcp,
                                    fixed_point=a.identical, replay_stats=a.stats, ordinary_stats=b.stats)) + "\n")
    summ = run_dataset(cfg, engine, ds, ["arr"], questions=qs, tag="pilot", budgets=cfg.budget_list()[:1])
    arr = next(iter(summ.values()), {})
    little = ((check["frac_changed_within_first_block"] or 0) > 0.5 and
              (check["fixed_point_rate"] if check["fixed_point_rate"] is not None else 0) < 0.10)
    rep = dict(replay_check=check, arr_first_budget={k: arr.get(k) for k in (
        "accuracy", "correct_k_of_n", "mean_tflops", "mean_root_births", "mean_voting_roots",
        "mean_revisions_per_root", "fixed_point_rate", "mean_fixed_point_retirements", "mean_accepted_prefix_frac",
        "frac_changed_within_first_block", "incomplete_revision_rate", "latency_sec_p50", "latency_sec_p95",
        "critical_path_passes_p50", "critical_path_passes_p95", "root_status")},
        verdict=("little useful signal — do not start a large sweep" if little
                 else "replay engages: proceed to the main study"))
    atomic_json(rep, os.path.join(ds_dir, "pilot_report.json"))
    return rep


__all__ = ["load_questions", "run_dataset", "pilot", "summarize", "grade", "depth_accounting", "drive", "auroc"]
