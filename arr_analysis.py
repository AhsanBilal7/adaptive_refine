#!/usr/bin/env python3
"""
Question-level analyses of the finished ARR runs for the paper figures.

Reads the per-question records (results_<method>*.jsonl) of every output root
$ARR_OUTPUT_ROOT* and writes results/analysis.json, which
make_matlab_figure_script.py turns into hard-coded MATLAB figures.

  dynamics   revision depth, revision stage, and anytime accuracy (primary model, MATH500, B=800)
  regimes    net repair of the revision operator against ARR - Majority Voting, every model x benchmark
  stopping   stopping-rule ablation: exact fixed point vs. answer-level vs. one revision (paired by question and seed)
  ablations  N, temperature, budget and replay window W (seed 0, paired with seed 0 of the main runs)
  timing     wall-clock latency on an exclusive GPU, one solve at a time (if the timing run exists)
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import statistics
from collections import defaultdict

RES = os.environ.get("ARR_OUTPUT_ROOT", "experiments_results/arr").rstrip("/")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "analysis.json")
P = "Qwen2.5-Math-7B-Instruct"
MATH = ["math500", "olympiadbench", "amc23", "aime24", "aime25"]
MODELS = ["Qwen2.5-Math-7B-Instruct", "Qwen3-8B", "Ministral-8B-Instruct-2410", "gemma-4-E4B-it",
          "Qwen3.5-9B", "Qwen3.5-2B", "Qwen2.5-7B-Instruct"]
DSETS = MATH + ["gsm8k", "livebench_reasoning"]
SIZE = {"math500": 500, "olympiadbench": 674, "amc23": 40, "aime24": 30, "aime25": 30, "gsm8k": 1319,
        "livebench_reasoning": 150}


def records(root, model, ds, b, method):
    """One record per (qid, seed) of a cell; combined and sharded files."""
    pat = re.compile(rf"results_{re.escape(method)}(\.s\d+of\d+)?\.jsonl$")
    seen = {}
    for f in sorted(glob.glob(f"{root}/{model}/{ds}/B{b:g}/results_{method}*.jsonl")):
        if not pat.search(os.path.basename(f)):
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.setdefault((r["qid"], r["seed"]), r)
    return seen


def budgets(model, ds, root=RES):
    return sorted(float(os.path.basename(d)[1:]) for d in glob.glob(f"{root}/{model}/{ds}/B*"))


def complete_seeds(recs, ds):
    """Seeds whose every question is graded."""
    n = defaultdict(int)
    for (_, s) in recs:
        n[s] += 1
    return sorted(s for s, k in n.items() if k == SIZE[ds])


def acc(recs, seeds=None):
    v = [r["correct"] for (q, s), r in recs.items() if seeds is None or s in seeds]
    return 100.0 * sum(v) / len(v) if v else float("nan")


def paired_ci(a, b, keys, n_boot=2000, seed=0):
    """Mean of b - a (points) over shared (qid, seed) keys, with a 95% bootstrap CI over questions."""
    qs = sorted({q for (q, _) in keys})
    by_q = defaultdict(list)
    for k in keys:
        by_q[k[0]].append(100.0 * (b[k]["correct"] - a[k]["correct"]))
    per_q = [statistics.mean(by_q[q]) for q in qs]
    m = statistics.mean(per_q)
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(per_q, k=len(per_q))) for _ in range(n_boot))
    return m, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


# dynamics
def depth_curves(recs, kmax=8):
    """Accuracy of roots after k revisions, of the same roots before revision, and of new samples."""
    at_k = [[] for _ in range(kmax + 1)]
    base_k = [[] for _ in range(kmax + 1)]
    new_samples, n_root = [], 0
    for r in recs.values():
        cohort = {e["root"] for e in r["events"] if e.get("cohort")}
        for ro in r["roots"]:
            c = ro.get("correct") or []
            if not ro["valid"] or not c:
                continue
            n_root += 1
            if ro["id"] not in cohort:
                new_samples.append(c[0])
            for k in range(min(len(c), kmax + 1)):
                at_k[k].append(c[k])
                base_k[k].append(c[0])
    f = lambda v: 100.0 * sum(v) / len(v) if v else float("nan")
    return dict(k=list(range(kmax + 1)), revised=[f(v) for v in at_k], initial=[f(v) for v in base_k],
                share=[100.0 * len(v) / max(1, n_root) for v in at_k], new_sample=f(new_samples))


def stage_rates(recs, jmax=6):
    """Per revision index j: P(W->C), P(C->W) and P(fixed point) among complete revisions."""
    n, w2c, c2w, fp = ([0] * (jmax + 1) for _ in range(4))
    for r in recs.values():
        for e in r["events"]:
            if e.get("op") != "revise" or e.get("finish") != "eos" or "before_correct" not in e:
                continue
            j = min(e["revisions_before"] + 1, jmax)
            n[j] += 1
            w2c[j] += (not e["before_correct"]) and e["after_correct"]
            c2w[j] += e["before_correct"] and not e["after_correct"]
            fp[j] += bool(e.get("identical"))
    js = list(range(1, jmax + 1))
    g = lambda a: [100.0 * a[j] / n[j] if n[j] else float("nan") for j in js]
    return dict(j=js, n=[n[j] for j in js], w2c=g(w2c), c2w=g(c2w), fp=g(fp))


def anytime(recs, B, grid):
    """Accuracy of the running plurality vote after a fraction p of the budget B is spent."""
    out = [[] for _ in grid]
    for r in recs.values():
        roots = {ro["id"]: ro for ro in r["roots"]}
        ev_cost = sum(e.get("spent", 0.0) for e in r["events"] if not e.get("cohort"))
        t = max(0.0, r["cost"]["spent"] - ev_cost)          # the shared initial cohort is charged first
        idx = defaultdict(lambda: -1)
        timeline = []

        def snapshot():
            groups = {}
            for rid in sorted(idx):
                i = idx[rid]
                ro = roots.get(rid)
                if i < 0 or not ro or not ro["valid"] or i >= len(ro.get("answers") or []):
                    continue
                a = ro["answers"][i]
                if a in (None, ""):
                    continue
                g = groups.setdefault(a, [0, rid, bool(ro["correct"][i])])
                g[0] += 1
            if not groups:
                return False
            best = max(groups.values(), key=lambda g: (g[0], -g[1]))
            return best[2]

        for e in r["events"]:                                # the cohort events come first
            if e.get("cohort") and e.get("finish") == "eos":
                idx[e["root"]] = 0
        start = snapshot()                                   # vote of the N initial samples
        for e in r["events"]:
            if e.get("cohort"):
                continue
            t += e.get("spent", 0.0)
            if e.get("op") == "sample" and e.get("finish") == "eos":
                idx[e["root"]] = 0
            elif e.get("op") in ("revise", "correct") and e.get("finish") == "eos":
                idx[e["root"]] += 1
            timeline.append((t, snapshot()))
        for gi, p in enumerate(grid):
            cur = start
            for (tt, ok) in timeline:
                if tt <= p * B + 1e-9:
                    cur = ok
                else:
                    break
            out[gi].append(cur)
    return [100.0 * sum(v) / len(v) if v else float("nan") for v in out]


# regimes
def net_repair(recs):
    n = w2c = c2w = 0
    for r in recs.values():
        for e in r["events"]:
            if e.get("op") == "revise" and e.get("finish") == "eos" and "before_correct" in e:
                n += 1
                w2c += (not e["before_correct"]) and e["after_correct"]
                c2w += e["before_correct"] and not e["after_correct"]
    return (100.0 * (w2c - c2w) / n if n else float("nan")), n


def main():
    A = {}
    # ---- dynamics (primary model, MATH500, B=800, all seeds)
    arr = records(RES, P, "math500", 800, "arr")
    A["dyn_depth"] = depth_curves(arr)
    A["dyn_stage"] = stage_rates(arr)
    grid = [i / 20 for i in range(1, 21)]
    A["dyn_anytime"] = dict(p=[100 * g for g in grid], **{
        m: anytime(records(RES, P, "math500", 800, m), 800, grid)
        for m in ("arr", "replace_ordinary", "arr_no_replace", "maj")})
    # ---- regimes: every model x benchmark at the largest budget, seeds complete for both ARR and SC
    reg = []
    for m in MODELS:
        for d in DSETS:
            bs = budgets(m, d)
            if not bs:
                continue
            b = bs[-1]
            a, s = records(RES, m, d, b, "arr"), records(RES, m, d, b, "maj")
            seeds = sorted(set(complete_seeds(a, d)) & set(complete_seeds(s, d)))
            if not seeds:
                continue
            a = {k: v for k, v in a.items() if k[1] in seeds}
            s = {k: v for k, v in s.items() if k[1] in seeds}
            rate, nrev = net_repair(a)
            reg.append(dict(model=m, dataset=d, budget=b, seeds=len(seeds), net_repair=rate, n_rev=nrev,
                            arr=acc(a), maj=acc(s), gap=acc(a) - acc(s)))
    A["regimes"] = reg
    # ---- stopping rule (paired by question and seed with the main ARR runs)
    stop = []
    for d in MATH + ["gsm8k"]:
        for b in budgets(P, d, f"{RES}_abl_stop"):
            main_arr, main_maj = records(RES, P, d, b, "arr"), records(RES, P, d, b, "maj")
            row = dict(dataset=d, budget=b)
            ans, once = (records(f"{RES}_abl_stop", P, d, b, m) for m in ("arr_answer", "arr_once"))
            seeds = sorted(set(complete_seeds(ans, d)) & set(complete_seeds(once, d)) & set(complete_seeds(main_arr, d)))
            if not seeds:
                continue
            keys = [k for k in main_arr if k[1] in seeds]
            row["seeds"] = seeds
            row["acc"] = {"arr": acc(main_arr, seeds), "arr_answer": acc(ans, seeds), "arr_once": acc(once, seeds),
                          "maj": acc(main_maj, seeds)}
            row["delta"] = {m: paired_ci(main_arr, x, keys) for m, x in (("arr_answer", ans), ("arr_once", once),
                                                                          ("maj", main_maj))}
            row["roots"] = {m: statistics.mean(len(r["roots"]) for k, r in x.items() if k[1] in seeds)
                            for m, x in (("arr", main_arr), ("arr_answer", ans), ("arr_once", once))}
            row["passes"] = {m: statistics.median(r["critical_path_passes"] for k, r in x.items() if k[1] in seeds)
                             for m, x in (("arr", main_arr), ("arr_answer", ans), ("arr_once", once))}
            stop.append(row)
    A["stopping"] = stop
    # ---- ablations: seed 0 of each variant against seed 0 of the main run
    abl = {}
    for var in ("n2", "n8", "temp05", "temp10", "budget1600", "w8", "w16", "w64", "revprompt1", "revprompt2"):
        rows = []
        for d in DSETS:
            for b in budgets(P, d, f"{RES}_{var}"):
                for m in ("arr", "arr_no_replace", "replace_ordinary", "maj"):
                    x = records(f"{RES}_{var}", P, d, b, m)
                    if 0 not in complete_seeds(x, d):
                        continue
                    x0 = {k: v for k, v in x.items() if k[1] == 0}
                    base = {k: v for k, v in records(RES, P, d, b, m).items() if k[1] == 0}
                    rows.append(dict(dataset=d, budget=b, method=m, acc=acc(x0),
                                     base=acc(base) if len(base) == SIZE[d] else float("nan"),
                                     passes=statistics.median(r["critical_path_passes"] for r in x0.values()),
                                     base_passes=(statistics.median(r["critical_path_passes"] for r in base.values())
                                                  if base else float("nan")),
                                     tflops=statistics.mean(r["cost"]["tflops"] for r in x0.values()),
                                     roots=statistics.mean(len(r["roots"]) for r in x0.values())))
        abl[var] = rows
    A["ablations"] = abl
    # ---- timing (exclusive GPU, one solve at a time)
    tim = {}
    for m in ("arr", "replace_ordinary", "replace_pld", "maj", "conma"):
        x = records(f"{RES}_timing", P, "math500", 400, m)
        if x:
            lat = [r["latency_sec"] for r in x.values()]
            tim[m] = dict(n=len(x), acc=acc(x), lat_med=statistics.median(lat), lat_mean=statistics.mean(lat),
                          tflops=statistics.mean(r["cost"]["tflops"] for r in x.values()),
                          tokens=statistics.mean(r.get("output_tokens", 0) for r in x.values()),
                          votes=statistics.mean(r.get("n_voting_roots", 0) for r in x.values()),
                          passes=statistics.median(r["critical_path_passes"] for r in x.values()), qids=sorted(q for q, _ in x))
    A["timing"] = tim
    A["vcells"] = variant_cells()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(A, open(OUT, "w"), indent=1, default=float)
    print(f"wrote {OUT}: {len(reg)} regime settings, {len(stop)} stopping rows, "
          f"{ {k: len(v) for k, v in abl.items()} }, timing methods {list(tim)}")


def variant_cells():
    """Per-seed accuracy and median serial passes of every complete (seed, method) in every ablation root,
    read from the question-level records (summary files of sharded runs can be stale)."""
    out = {}
    for root in sorted(glob.glob(f"{RES}_*")):
        var = os.path.basename(root)[len(os.path.basename(RES)) + 1:]
        if var in ("conma", "timing") or var.startswith(("old_", "tmax")):
            continue
        for bdir in glob.glob(f"{root}/*/*/B*"):
            model, ds, b = bdir.split("/")[-3], bdir.split("/")[-2], float(os.path.basename(bdir)[1:])
            if ds not in SIZE:
                continue
            for meth in {re.sub(r"\.s\d+of\d+$", "", os.path.basename(f)[len("results_"):-len(".jsonl")])
                         for f in glob.glob(f"{bdir}/results_*.jsonl")}:
                recs = records(root, model, ds, b, meth)
                for s in complete_seeds(recs, ds):
                    x = [r for (q, ss), r in recs.items() if ss == s]
                    out.setdefault(f"{var}|{model}|{ds}|{b:g}|{meth}", {})[str(s)] = dict(
                        acc=100.0 * sum(r["correct"] for r in x) / len(x),
                        passes=statistics.median(r["critical_path_passes"] for r in x))
    return out


if __name__ == "__main__" and os.environ.get("ARR_VARIANTS_ONLY"):
    J = json.load(open(OUT))
    J["vcells"] = variant_cells()
    json.dump(J, open(OUT, "w"), indent=1, default=float)
    print("vcells", len(J["vcells"]))


if __name__ == "__main__" and not os.environ.get("ARR_VARIANTS_ONLY"):
    main()
