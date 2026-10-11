#!/usr/bin/env python3
"""
collect_results.py
==================
Collect every ARR result into ONE csv and ONE self-contained html report.

Reads the per-dataset ``summary_all.json`` (merged over shards and seeds by the runner) and the
``config.json`` beside it, for every output root ``$ARR_OUTPUT_ROOT*``:

    <root>/<model>/<dataset>/summary_all.json     {"B<budget>/<method>": {metrics...}, ...}
    <root>/<model>/<dataset>/config.json          run settings (T_max, budgets, cost measure, ...)

Roots are reported as experiment variants: ``<root>`` → main, ``<root>_passes`` → passes,
``<root>_revprompt1`` → revprompt1, ...  Archived roots that were superseded
(``*old_per_row*``, ``*tmax1536*``) are skipped unless --all is given.

A cell is marked partial when some seed has fewer graded questions than the dataset holds
(e.g. a shard still running); its numbers are then provisional.

Usage:
    python collect_results.py                                  # results/all_results.{csv,html}
    python collect_results.py --out results/arr_all            # other output prefix
    python collect_results.py --roots $ARR_OUTPUT_ROOT --all
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

# questions per dataset (to flag partial cells); unknown datasets fall back to the largest seed count
DATASET_SIZE = {"aime24": 30, "aime25": 30, "amc23": 40, "math500": 500, "olympiadbench": 674,
                "gsm8k": 1319, "livebench_reasoning": 150, "math500_pilot": 50,
                "np_calendar": 1000, "np_meeting": 1000, "np_trip": 1600}
DATASET_ORDER = ["aime24", "aime25", "amc23", "math500", "olympiadbench", "gsm8k", "livebench_reasoning"]
METHOD_ORDER = ["arr", "replace_ordinary", "replace_pld", "arr_no_replace", "direct_fixed",
                "critic_correct", "continuation", "maj", "conma", "arr_answer", "arr_once"]
ARCHIVED = ("old_per_row", "tmax1536")

# summary metric → csv column (in this order, after the identifying columns)
METRICS = [
    ("mean_tflops", "mean_tflops"), ("mean_budget_spent", "mean_budget_spent"),
    ("latency_sec_p50", "latency_sec_p50"), ("latency_sec_p95", "latency_sec_p95"),
    ("critical_path_passes_p50", "critical_path_passes_p50"),
    ("critical_path_passes_p95", "critical_path_passes_p95"),
    ("mean_root_births", "roots_per_question"), ("mean_voting_roots", "voting_roots_per_question"),
    ("mean_revisions_per_root", "revisions_per_root"), ("fixed_point_rate", "fixed_point_rate"),
    ("mean_accepted_prefix_frac", "accepted_prefix_frac"), ("mean_draft_recompute", "mean_draft_recompute"),
    ("wrong_to_correct", "wrong_to_correct"), ("correct_to_wrong", "correct_to_wrong"),
    ("mean_vote_share", "mean_vote_share"), ("auroc_vote_share", "auroc_vote_share"),
    ("auroc_vote_margin", "auroc_vote_margin"), ("auroc_fixed_point_support", "auroc_fixed_point_support"),
    ("mean_output_tokens_per_question", "output_tokens_per_question"),
    ("mean_output_tokens_per_call", "output_tokens_per_call"), ("mean_calls", "calls_per_question"),
    ("invalid_samples", "invalid_samples"), ("n_solves", "n_solves"),
]
CONFIG_COLS = [("max_new_tokens", "t_max"), ("cost_measure", "cost_measure"), ("n_active", "n_active"),
               ("revise_prompt", "revise_prompt"), ("sampler", "sampler"), ("thinking", "thinking")]


BASE = os.environ.get("ARR_OUTPUT_ROOT", "experiments_results/arr").rstrip("/")


def variant_of(root: str) -> str:
    """Variant of an output root: ``<base>`` -> main, ``<base>_n8`` -> n8, ..."""
    name = os.path.basename(root.rstrip("/"))
    rest = name[len(os.path.basename(BASE)):].lstrip("_")
    return "main" if rest in ("", "conma") else rest     # ConMA runs in its own root only to keep summaries apart


def parse_k_of_n(v: Dict[str, str]) -> List[tuple]:
    out = []
    for seed in sorted(v, key=lambda s: int(s) if str(s).isdigit() else s):
        k, n = str(v[seed]).split("/")
        out.append((seed, int(float(k)), int(float(n))))
    return out


def budget_key(b: str) -> float:
    try:
        return float(b)
    except ValueError:
        return float("inf")


def collect(roots: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "*", "*", "summary_all.json"))):
            ds_dir = os.path.dirname(path)
            dataset = os.path.basename(ds_dir)
            model = os.path.basename(os.path.dirname(ds_dir))
            try:
                summary = json.load(open(path))
            except (OSError, json.JSONDecodeError):
                continue
            cfg: Dict[str, Any] = {}
            if os.path.exists(os.path.join(ds_dir, "config.json")):
                try:
                    cfg = json.load(open(os.path.join(ds_dir, "config.json")))
                except (OSError, json.JSONDecodeError):
                    cfg = {}
            updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
            for key, m in summary.items():
                if "/" not in key or not isinstance(m, dict):
                    continue
                budget, method = key.split("/", 1)
                seeds = parse_k_of_n(m.get("correct_k_of_n") or {})
                expected = DATASET_SIZE.get(dataset) or max([n for _, _, n in seeds] or [m.get("n_questions") or 0])
                partial = (not seeds) or any(n < expected for _, _, n in seeds)
                row = dict(
                    variant=variant_of(root), model=model, dataset=dataset,
                    budget=budget.lstrip("B"), method=method,
                    accuracy=m.get("accuracy"), solved_mean=m.get("solved_count"),
                    n_questions=m.get("n_questions"), dataset_size=expected, n_seeds=len(seeds),
                    per_seed=" ".join(f"s{s}:{k}/{n}" for s, k, n in seeds),
                    complete=not partial,
                )
                for src, col in CONFIG_COLS:
                    row[col] = cfg.get(src)
                for src, col in METRICS:
                    row[col] = m.get(src)
                row["summary_updated"] = updated
                row["source"] = path
                rows.append(row)
    # gaps to arr and maj within (variant, model, dataset, budget)
    groups: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        groups[(r["variant"], r["model"], r["dataset"], r["budget"])][r["method"]] = r
    for g in groups.values():
        ranked = sorted((r for r in g.values() if r["accuracy"] is not None), key=lambda r: -r["accuracy"])
        best = ranked[0]["accuracy"] if ranked else None
        for r in g.values():
            for ref in ("arr", "maj"):
                a = g.get(ref, {}).get("accuracy")
                r[f"delta_vs_{ref}"] = (None if a is None or r["accuracy"] is None
                                        else round(r["accuracy"] - a, 4))
            r["rank"] = 1 + sum(1 for x in ranked if x["accuracy"] > (r["accuracy"] or -1))
            r["is_best"] = best is not None and r["accuracy"] is not None and abs(r["accuracy"] - best) < 1e-12
    dorder = {d: i for i, d in enumerate(DATASET_ORDER)}
    morder = {m: i for i, m in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda r: (r["variant"] != "main", r["variant"], r["model"], dorder.get(r["dataset"], 99),
                             r["dataset"], budget_key(r["budget"]), morder.get(r["method"], 99), r["method"]))
    return rows


COLUMNS = ["variant", "model", "dataset", "budget", "method", "accuracy", "rank", "is_best",
           "delta_vs_arr", "delta_vs_maj", "solved_mean", "n_questions", "dataset_size", "n_seeds",
           "per_seed", "complete"] + [c for _, c in CONFIG_COLS] + [c for _, c in METRICS] + \
          ["summary_updated", "source"]


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})


# html
def fmt_pct(x: Optional[float]) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def fmt_num(x: Any, nd: int = 1) -> str:
    if x is None or x == "":
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}" if abs(x) < 1e6 else f"{x:.3g}"
    return html.escape(str(x))


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#e2e2e0;--card:#ffffff;--best:#e6f4ea;
--bestfg:#14532d;--partial:#fff7e0;--head:#f3f3f1;--accent:#2f5d9e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141416;--fg:#e8e8ea;
--muted:#9a9aa2;--line:#2c2c31;--card:#1c1c20;--best:#16351f;--bestfg:#bfe8c9;--partial:#3a3115;
--head:#232328;--accent:#8fb3e8}}
:root[data-theme="dark"]{--bg:#141416;--fg:#e8e8ea;--muted:#9a9aa2;--line:#2c2c31;--card:#1c1c20;
--best:#16351f;--bestfg:#bfe8c9;--partial:#3a3115;--head:#232328;--accent:#8fb3e8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}h3{font-size:14px;margin:0 0 6px}
.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}thead th{background:var(--head);font-weight:600;position:sticky;top:0}
td.best{background:var(--best);color:var(--bestfg);font-weight:600}td.partial{background:var(--partial)}
.tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:9px;border:1px solid var(--line);
color:var(--muted);margin-left:6px}
input{width:100%;max-width:420px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;
background:var(--card);color:var(--fg);margin:6px 0 10px}
.wide{max-height:70vh;overflow:auto}a{color:var(--accent)}
"""


def accuracy_card(title: str, sub: str, cells: Dict[tuple, Dict[str, Any]]) -> str:
    budgets = sorted({b for b, _ in cells}, key=budget_key)
    methods = sorted({m for _, m in cells}, key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m))
    out = [f'<div class="card"><h3>{html.escape(title)}</h3><div class="muted">{sub}</div><table><thead><tr>'
           '<th>method</th>' + "".join(f"<th>B{html.escape(b)}</th>" for b in budgets) + "</tr></thead><tbody>"]
    for m in methods:
        tds = []
        for b in budgets:
            r = cells.get((b, m))
            if r is None:
                tds.append("<td>–</td>")
                continue
            cls = "best" if r["is_best"] else ("partial" if not r["complete"] else "")
            mark = "" if r["complete"] else "*"
            tip = html.escape(f'{r["per_seed"]} | TFLOPs {fmt_num(r.get("mean_tflops"))} | '
                              f'Δarr {fmt_num(r.get("delta_vs_arr"), 3)} | Δmaj {fmt_num(r.get("delta_vs_maj"), 3)}')
            tds.append(f'<td class="{cls}" title="{tip}">{fmt_pct(r["accuracy"])}{mark}</td>')
        out.append(f"<tr><td>{html.escape(m)}</td>{''.join(tds)}</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def write_html(rows: List[Dict[str, Any]], path: str) -> None:
    by_exp: Dict[tuple, Dict[tuple, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_exp[(r["variant"], r["model"], r["dataset"])][(r["budget"], r["method"])] = r
    variants = sorted({v for v, _, _ in by_exp}, key=lambda v: (v != "main", v))

    # arr vs maj at the largest budget, per model × dataset (main variant)
    gap_rows = []
    for (v, model, ds), cells in by_exp.items():
        if v != "main":
            continue
        bmax = max((b for b, _ in cells), key=budget_key)
        a, mj = cells.get((bmax, "arr")), cells.get((bmax, "maj"))
        if a and mj and a["accuracy"] is not None and mj["accuracy"] is not None:
            best = max((r for (b, _), r in cells.items() if b == bmax and r["accuracy"] is not None),
                       key=lambda r: r["accuracy"])
            gap_rows.append((model, ds, bmax, a, mj, best))
    dorder = {d: i for i, d in enumerate(DATASET_ORDER)}
    gap_rows.sort(key=lambda t: (t[0], dorder.get(t[1], 99)))

    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             "<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>ARR Results</title><style>{CSS}</style></head><body><main>",
             "<h1>Adaptive Replay Refinement — results</h1>",
             f"<div class='muted'>Generated {time.strftime('%Y-%m-%d %H:%M')} from {len(by_exp)} experiments "
             f"({len(rows)} budget × method cells). Accuracy in %, averaged over seeds; "
             "<span style='background:var(--best);padding:0 4px'>green</span> = best at that budget; "
             "<span style='background:var(--partial);padding:0 4px'>*</span> = partial (a seed or shard still "
             "running). Hover a cell for per-seed counts, compute and gaps to arr / maj.</div>"]

    parts.append("<h2>arr vs. majority voting at the largest budget (main runs)</h2><div class='card wide'>"
                 "<table><thead><tr><th>model</th><th>dataset</th><th>budget</th><th>seeds</th><th>arr</th>"
                 "<th>maj</th><th>arr − maj</th><th>best method</th></tr></thead><tbody>")
    for model, ds, b, a, mj, best in gap_rows:
        d = a["accuracy"] - mj["accuracy"]
        part = "" if (a["complete"] and mj["complete"]) else " *"
        parts.append(f"<tr><td>{html.escape(model)}</td><td>{html.escape(ds)}</td><td>B{html.escape(b)}</td>"
                     f"<td>{a['n_seeds']}</td><td>{fmt_pct(a['accuracy'])}</td><td>{fmt_pct(mj['accuracy'])}</td>"
                     f"<td>{100 * d:+.1f}{part}</td><td>{html.escape(best['method'])} ({fmt_pct(best['accuracy'])})</td></tr>")
    parts.append("</tbody></table></div>")

    for v in variants:
        parts.append(f"<h2>Accuracy tables — {html.escape(v)}</h2><div class='grid'>")
        for (vv, model, ds), cells in sorted(by_exp.items(), key=lambda kv: (kv[0][1], dorder.get(kv[0][2], 99))):
            if vv != v:
                continue
            any_r = next(iter(cells.values()))
            seeds = max(r["n_seeds"] for r in cells.values())
            complete = all(r["complete"] for r in cells.values())
            sub = (f"{any_r['dataset_size']} questions · {seeds} seed{'s' if seeds != 1 else ''} · "
                   f"T_max {fmt_num(any_r.get('t_max'))} · budget in {fmt_num(any_r.get('cost_measure'))}"
                   + ("" if complete else " <span class='tag'>partial</span>"))
            parts.append(accuracy_card(f"{model} · {ds}", sub, cells))
        parts.append("</div>")

    show = ["variant", "model", "dataset", "budget", "method", "accuracy", "delta_vs_arr", "delta_vs_maj",
            "n_seeds", "complete", "mean_tflops", "latency_sec_p50", "critical_path_passes_p50",
            "roots_per_question", "revisions_per_root", "fixed_point_rate", "wrong_to_correct",
            "correct_to_wrong", "auroc_vote_share", "output_tokens_per_question", "summary_updated"]
    parts.append("<h2>All cells</h2><input id='q' placeholder='Filter rows (e.g. qwen3 math500 arr)…'>"
                 "<div class='card wide'><table id='all'><thead><tr>"
                 + "".join(f"<th>{c}</th>" for c in show) + "</tr></thead><tbody>")
    for r in rows:
        tds = []
        for c in show:
            x = r.get(c)
            tds.append(f"<td>{fmt_pct(x) if c == 'accuracy' else fmt_num(x, 3 if c.startswith(('delta', 'auroc', 'fixed')) else 1)}</td>")
        parts.append("<tr>" + "".join(tds) + "</tr>")
    parts.append("</tbody></table></div>")
    parts.append("""<script>
const q=document.getElementById('q'),rows=[...document.querySelectorAll('#all tbody tr')];
q.addEventListener('input',()=>{const t=q.value.toLowerCase().split(/\\s+/).filter(Boolean);
rows.forEach(r=>{const s=r.textContent.toLowerCase();r.style.display=t.every(w=>s.includes(w))?'':'none';});});
</script></main></body></html>""")
    with open(path, "w") as f:
        f.write("".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=None,
                    help="output roots (default: $ARR_OUTPUT_ROOT*)")
    ap.add_argument("--all", action="store_true", help="also include archived/superseded roots")
    ap.add_argument("--out", default="results/all_results", help="output prefix (.csv and .html are added)")
    args = ap.parse_args()
    roots = args.roots or sorted(d for d in glob.glob(BASE + "*") if os.path.isdir(d))
    if not args.all:
        roots = [r for r in roots if not any(a in os.path.basename(r) for a in ARCHIVED)]
    rows = collect(roots)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_csv(rows, args.out + ".csv")
    write_html(rows, args.out + ".html")
    n_exp = len({(r["variant"], r["model"], r["dataset"]) for r in rows})
    n_part = len({(r["variant"], r["model"], r["dataset"]) for r in rows if not r["complete"]})
    print(f"{len(rows)} cells from {n_exp} experiments ({n_part} partial) in {len(roots)} roots")
    print(f"wrote {args.out}.csv and {args.out}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
