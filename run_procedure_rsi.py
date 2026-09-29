#!/usr/bin/env python3
"""
run_procedure_rsi.py
====================
Entry point for training-free Procedure RSI (see updated_methodology.md).

Subcommands
-----------
  sanity    grader / extraction checks (no model needed)                 §9
  splits    build search / confirmation / sealed-test splits            §6
  develop   outer loop: evidence → patches → paired confirmation        §5–§6
  evaluate  run procedures and baselines on benchmarks or the sealed test §8

Configuration precedence: dataclass defaults < .env < --conf file < CLI flags.
Every field of ``src/rsi/config.py:RSIConfig`` is settable as ``--<field>``,
``--set field=value`` or the env var ``RSI_<FIELD>``.

Examples
--------
  python run_procedure_rsi.py sanity
  python run_procedure_rsi.py develop  --env_file .env --conf configs/rsi_qwen_1b.conf
  python run_procedure_rsi.py evaluate --env_file .env --conf configs/rsi_qwen_1b.conf \\
      --methods h0,maj_matched,heuristic,final --datasets math500,amc23
  python run_procedure_rsi.py evaluate ... --sealed      # the run's sealed test split
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("procedure_rsi")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rsi.config import add_config_args, config_from_args, dump_config  # noqa: E402


def _parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sanity", "splits", "develop", "evaluate"):
        sp = sub.add_parser(name)
        add_config_args(sp)
        if name == "evaluate":
            sp.add_argument("--methods", default="h0,maj_matched,final",
                            help="comma list: h0 | heuristic | maj_matched | final | <procedure.json>")
            sp.add_argument("--datasets", default=None, help="comma list; default RSI_TEST_DATASETS")
            sp.add_argument("--sealed", action="store_true", help="evaluate on the run's sealed test split")
            sp.add_argument("--eval_seeds", type=int, default=1)
            sp.add_argument("--eval_tag", default="")
    return p.parse_args()


def _set_seeds(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def cmd_sanity(cfg) -> int:
    from src.rsi.voting import Grader, sanity_check_grader
    fails = []
    for ds in ["math500"] + cfg.list_of("test_datasets"):
        fails += [f"[{ds}] {x}" for x in sanity_check_grader(Grader(ds))]
    if fails:
        for f in fails:
            logger.error(f"sanity FAIL: {f}")
        return 1
    logger.info("sanity: grader + extraction checks passed")
    return 0


def _build_splits(cfg, run_dir: str):
    from src.rsi.splits import load_problems, make_splits, save_splits
    dev = load_problems(cfg.list_of("dev_datasets"), max_per_dataset=cfg.dev_pool_max,
                        cache_dir=cfg.hf_cache_dir)
    test = load_problems(cfg.list_of("test_datasets"), max_per_dataset=cfg.test_max_samples,
                         cache_dir=cfg.hf_cache_dir)
    splits = make_splits(dev, test, cfg.rounds, cfg.search_batch_size, cfg.confirm_block_size, cfg.split_seed)
    problems = {p["qid"]: p for p in dev + test}
    save_splits(splits, problems, run_dir)
    logger.info(f"splits: dev pool={len(dev)} (dropped {splits['dropped_overlapping_test']} overlapping test) "
                f"search={[len(b) for b in splits['search']]} confirm={[len(b) for b in splits['confirm']]} "
                f"sealed test={len(splits['test'])} → {run_dir}")
    return splits


def _load_splits(run_dir: str):
    with open(os.path.join(run_dir, "splits.json")) as f:
        splits = json.load(f)
    from src.rsi.splits import load_jsonl
    test_path = os.path.join(run_dir, "sealed_test", "test_problems.jsonl")
    splits["test"] = [p["qid"] for p in load_jsonl(test_path)] if os.path.exists(test_path) else []
    return splits


def cmd_splits(cfg) -> int:
    run_dir = cfg.run_dir()
    os.makedirs(run_dir, exist_ok=True)
    _build_splits(cfg, run_dir)
    return 0


def cmd_develop(cfg) -> int:
    from src.rsi.backends import build_backend
    from src.rsi.outer_loop import OuterLoop, load_dev_problems
    if cmd_sanity(cfg) != 0:
        logger.error("refusing to develop with a failing grader")
        return 1
    run_dir = cfg.run_dir()
    os.makedirs(run_dir, exist_ok=True)
    splits = _load_splits(run_dir) if os.path.exists(os.path.join(run_dir, "splits.json")) \
        else _build_splits(cfg, run_dir)
    problems = load_dev_problems(run_dir)
    _set_seeds(cfg.seed)
    backend = build_backend(cfg)
    loop = OuterLoop(cfg, backend, run_dir, problems, splits)
    H = loop.run()
    logger.info(f"final procedure {H.hash()}: rules={H.n_rules()} plans={H.n_plans()}")
    return 0


def cmd_evaluate(cfg, args) -> int:
    from src.rsi.backends import build_backend
    from src.rsi.procedure import Procedure, h0_procedure, heuristic_procedure
    from src.rsi.runner import run_solves, summarize
    from src.rsi.solver import SolveContext
    from src.rsi.splits import load_jsonl, load_problems
    from src.rsi.voting import Grader

    run_dir = cfg.run_dir()
    out_dir = os.path.join(run_dir, "eval" + (f"_{args.eval_tag}" if args.eval_tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    dump_config(cfg, os.path.join(out_dir, "config.json"))

    if args.sealed:
        probs = load_jsonl(os.path.join(run_dir, "sealed_test", "test_problems.jsonl"))
        by_ds = {}
        for p in probs:
            by_ds.setdefault(p["dataset"], []).append(p)
    else:
        dsets = (args.datasets.split(",") if args.datasets else cfg.list_of("test_datasets"))
        by_ds = {ds: load_problems([ds], max_per_dataset=cfg.test_max_samples, cache_dir=cfg.hf_cache_dir)
                 for ds in dsets}

    methods = []
    for m in args.methods.split(","):
        m = m.strip()
        if m == "h0":
            methods.append(("h0", "procedure", h0_procedure(cfg.h0_depth)))
        elif m == "heuristic":
            methods.append(("heuristic", "procedure", heuristic_procedure(cfg.h0_depth)))
        elif m == "maj_matched":
            methods.append(("maj_matched", "maj_matched", None))
        elif m == "final":
            fp = os.path.join(run_dir, "final_procedure.json")
            if not os.path.exists(fp):
                logger.warning(f"no {fp}; skipping 'final'")
                continue
            methods.append((f"final_{Procedure.load(fp).hash()}", "procedure", Procedure.load(fp)))
        elif m:
            pr = Procedure.load(m)
            methods.append((f"proc_{pr.hash()}", "procedure", pr))

    _set_seeds(cfg.seed)
    backend = build_backend(cfg)
    ctx = SolveContext(cfg, backend)
    seeds = list(range(max(1, args.eval_seeds)))
    summ_path = os.path.join(out_dir, "summary.json")
    table = {}
    if os.path.exists(summ_path):            # resume: keep finished (method, dataset) cells
        with open(summ_path) as f:
            table = json.load(f)
    for ds, probs in by_ds.items():
        grader = Grader(ds)
        for name, kind, proc in methods:
            tag = f"{name}_{ds}"
            path = os.path.join(out_dir, f"results_{tag}.jsonl")
            if tag in table and table[tag].get("n_solves") == len(probs) * len(seeds) and os.path.exists(path):
                logger.info(f"[eval] {tag}: already complete, skipping")
                continue
            logger.info(f"[eval] {tag}: {len(probs)} problems × {len(seeds)} seeds")
            recs = run_solves(ctx, backend, probs, kind, proc, seeds, grader, cfg.utility_lambda, desc=tag,
                              ckpt=os.path.join(out_dir, "ckpt", tag))
            with open(path + ".tmp", "w") as f:
                for r in recs:
                    f.write(json.dumps(r, default=str) + "\n")
            os.replace(path + ".tmp", path)
            s = summarize(recs)
            s.update(method=name, dataset=ds, procedure_hash=proc.hash() if proc else None)
            table[tag] = s
            logger.info(f"[eval] {tag}: acc={s.get('accuracy')} util={s.get('utility')} "
                        f"TFLOPs/q={s.get('mean_tflops'):.2f} calls/q={s.get('mean_calls'):.1f}")
            with open(summ_path + ".tmp", "w") as f:
                json.dump(table, f, indent=1)
            os.replace(summ_path + ".tmp", summ_path)
    print("\n" + "=" * 96)
    print(f"{'method':<28}{'dataset':<14}{'acc':>8}{'utility':>10}{'TFLOPs/q':>11}{'calls/q':>9}"
          f"{'W→C|W':>8}{'C→W|C':>8}")
    for tag, s in table.items():
        fmt = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"
        print(f"{s['method']:<28}{s['dataset']:<14}{fmt(s['accuracy']):>8}{fmt(s['utility']):>10}"
              f"{fmt(s['mean_tflops'], 2):>11}{fmt(s['mean_calls'], 1):>9}"
              f"{fmt(s['cond_wrong_to_correct']):>8}{fmt(s['cond_correct_to_wrong']):>8}")
    print("=" * 96)
    return 0


def main() -> int:
    args = _parse()
    cfg = config_from_args(args)
    if args.cmd == "sanity":
        return cmd_sanity(cfg)
    if args.cmd == "splits":
        return cmd_splits(cfg)
    if args.cmd == "develop":
        return cmd_develop(cfg)
    if args.cmd == "evaluate":
        return cmd_evaluate(cfg, args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
