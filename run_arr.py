#!/usr/bin/env python3
"""Command-line entry point of the ARR experiments.

Subcommands:

  sanity   CPU checks: grader and answer extraction, and on a tiny random model
           that replay decoding equals ordinary greedy decoding, that fixed points
           are detected, that PLD equals greedy, and that budgets truncate calls
  pilot    50-question feasibility pilot on ``pilot_dataset``: exact agreement of
           replay and ordinary decoding, accepted prefixes, fixed-point rate,
           discarded work, latency
  run      every method in ``methods`` at every budget in ``budgets`` on ``datasets``

Settings: dataclass defaults < ``.env`` < ``--conf`` files < command-line flags.
Every ``ARRConfig`` field is a flag ``--<field>`` and an environment variable
``ARR_<FIELD>``.

  python run_arr.py run   --env_file .env --conf configs/arr_qwen_math_7b.conf,configs/arr_aime24.conf
  python run_arr.py pilot --env_file .env --conf configs/arr_qwen_math_7b.conf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("arr")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.arr.config import add_config_args, config_from_args  # noqa: E402


def cmd_sanity(cfg) -> int:
    from src.arr.sanity import run_sanity
    fails = run_sanity(cfg.list_of("datasets"))
    for f in fails:
        logger.error(f"sanity FAIL: {f}")
    if not fails:
        logger.info("sanity: grader, extraction, replay==greedy, fixed points, PLD==greedy and budget checks passed")
    return 1 if fails else 0


def cmd_run(cfg, pilot: bool) -> int:
    from src.arr.engine import Engine
    from src.arr.runner import pilot as run_pilot, run_dataset
    if cmd_sanity(cfg) != 0:
        return 1
    engine = Engine(cfg)
    if pilot:
        rep = run_pilot(cfg, engine)
        logger.info("[pilot] " + json.dumps(rep, indent=1, default=str))
        return 0
    for ds in cfg.list_of("datasets"):
        summ = run_dataset(cfg, engine, ds, cfg.list_of("methods"))
        f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"
        print("\n" + "=" * 132)
        print(f"{ds}: {'budget/method':<26}{'acc':>7}{'k/n':>10}{'TFLOPs':>9}{'roots':>7}{'votes':>7}{'rev/rt':>7}"
              f"{'fixpt':>7}{'acc.pref':>9}{'lat50':>7}{'lat95':>7}{'cp50':>7}{'W→C':>5}{'C→W':>5}")
        for k, s in summ.items():
            kn = ",".join(s["correct_k_of_n"].values())
            print(f"{'':<{len(ds) + 2}}{k:<26}{f(s['accuracy']):>7}{kn:>10}{f(s['mean_tflops'], 1):>9}"
                  f"{f(s['mean_root_births'], 1):>7}{f(s['mean_voting_roots'], 1):>7}"
                  f"{f(s['mean_revisions_per_root'], 2):>7}{f(s['fixed_point_rate'], 2):>7}"
                  f"{f(s['mean_accepted_prefix_frac'], 2):>9}{f(s['latency_sec_p50'], 0):>7}"
                  f"{f(s['latency_sec_p95'], 0):>7}{f(s['critical_path_passes_p50'], 0):>7}"
                  f"{s['wrong_to_correct']:>5}{s['correct_to_wrong']:>5}")
        print("=" * 132)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sanity", "pilot", "run"):
        add_config_args(sub.add_parser(name))
    args = p.parse_args()
    cfg = config_from_args(args)
    if args.cmd == "sanity":
        return cmd_sanity(cfg)
    return cmd_run(cfg, pilot=args.cmd == "pilot")


if __name__ == "__main__":
    sys.exit(main())
