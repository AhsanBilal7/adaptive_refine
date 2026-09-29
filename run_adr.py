#!/usr/bin/env python3
"""
run_adr.py — Adaptive Delta Refinement experiments (updated_methodology.md).

Subcommands
-----------
  sanity   answer-extraction / grading and structure-parser checks (no model)
  pilot    §7 feasibility pilot: ADR on ``pilot_samples`` questions of ``pilot_dataset``
           (dependency well-formedness, repair-set vs suffix sizes, fallback rate)
  run      the §7 comparison on ``datasets``: cimr, full_rewrite, suffix, adr,
           adr_all_deps, maj_matched at every predeclared budget in ``budgets``

Configuration precedence: dataclass defaults < .env < --conf file(s) < CLI flags
(every ADRConfig field is a flag ``--<field>`` or env var ``ADR_<FIELD>``).

  python run_adr.py run   --env_file .env --conf configs/adr_qwen_7b.conf,configs/adr_aime24.conf
  python run_adr.py pilot --env_file .env --conf configs/adr_qwen_7b.conf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("adr")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.adr.config import add_config_args, config_from_args  # noqa: E402


def cmd_sanity(cfg) -> int:
    from src.adr.structure import apply_patch, parse_full, repair_set_adr, repair_set_suffix
    from src.adr.voting import Grader, sanity_check_grader
    fails = []
    for ds in ["math500"] + cfg.list_of("datasets"):
        fails += [f"[{ds}] {x}" for x in sanity_check_grader(Grader(ds))]
    # worked example of §4: repair set of step 1 is {1,3,5}, suffix is {1..5}
    sol = ("1. uses: [] Area: 12*5 = 50\n2. uses: [] Perimeter: 2(12+5) = 34\n3. uses: [1] Paint: 3*50 = 150\n"
           "4. uses: [2] Fence: 2*34 = 68\n5. uses: [3, 4] Total: 150+68 = \\boxed{218}")
    steps, errs = parse_full(sol)
    if steps is None:
        fails.append(f"structure parser failed on the worked example: {errs}")
    else:
        if repair_set_adr(steps, 1) != [1, 3, 5]:
            fails.append(f"A(1) = {repair_set_adr(steps, 1)}, expected [1, 3, 5]")
        if repair_set_suffix(steps, 1) != [1, 2, 3, 4, 5]:
            fails.append("suffix(1) != [1..5]")
        new, e2 = apply_patch(steps, "1. uses: [] Area: 12*5 = 60\n3. uses: [1] Paint: 3*60 = 180\n"
                                     "5. uses: [3, 4] Total: 180+68 = \\boxed{248}", [1, 3, 5])
        if new is None or new[1].raw != steps[1].raw or new[3].raw != steps[3].raw or "248" not in new[4].raw:
            fails.append(f"patch did not keep steps 2 and 4 verbatim / apply 1,3,5: {e2}")
    for f in fails:
        logger.error(f"sanity FAIL: {f}")
    if not fails:
        logger.info("sanity: grader, extraction, structure parser, repair set and patch checks passed")
    return 1 if fails else 0


def cmd_run(cfg, pilot: bool = False) -> int:
    from src.adr.backends import build_backend
    from src.adr.runner import load_questions, pilot_report, run_dataset
    if cmd_sanity(cfg) != 0:
        return 1
    backend = build_backend(cfg)
    if pilot:
        cfg.budgets = cfg.list_of("budgets")[0]
        ds = cfg.pilot_dataset
        qs = load_questions(ds, cfg.pilot_samples, cfg.hf_cache_dir)
        summ = run_dataset(cfg, backend, ds, ["adr"], questions=qs, tag="pilot")
        rep = pilot_report(summ)
        path = os.path.join(cfg.model_dir(), f"{ds}_pilot", "pilot_report.json")
        with open(path, "w") as f:
            json.dump(rep, f, indent=1)
        logger.info(f"[pilot] {json.dumps(rep)}  → {path}")
        return 0
    for ds in cfg.list_of("datasets"):
        summ = run_dataset(cfg, backend, ds, cfg.list_of("methods"))
        print("\n" + "=" * 118)
        print(f"{ds}: {'cell/method':<28}{'acc':>7}{'solved':>8}{'tok/q':>10}{'TFLOPs/q':>10}{'calls/q':>9}"
              f"{'lat(s)':>8}{'|A|/suf':>9}{'fallbk':>8}{'W→C':>5}{'C→W':>5}")
        f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"
        for k, s in summ.items():
            print(f"{'':<{len(ds) + 2}}{k:<28}{f(s['accuracy']):>7}{f(s['solved_count'], 1):>8}"
                  f"{f(s['mean_tokens_total'], 0):>10}{f(s['mean_tflops'], 1):>10}{f(s['mean_calls'], 1):>9}"
                  f"{f(s['mean_latency_sec'], 0):>8}{f(s['mean_repair_over_suffix'], 2):>9}"
                  f"{f(s['fallback_rate'], 2):>8}{s['wrong_to_correct']:>5}{s['correct_to_wrong']:>5}")
        print("=" * 118)
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
