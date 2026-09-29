"""
splits.py
=========
Three disjoint data roles (§6): search data, fresh confirmation blocks, and a
sealed final test set.

* Deduplicate by problem AND source template before splitting: questions are
  normalized (lower-case, whitespace collapsed) and a template key replaces
  every number with ``#``; all problems sharing a template land in the same
  split, so every state and descendant of a problem stays in its split.
* Development pool problems whose template also appears in the sealed test
  pool are dropped from development.
* Search batch r and confirmation block r are consumed only by round r;
  a confirmation block is never reused for another adaptive decision.
* The sealed test manifest is written separately; the outer loop only reads
  ``search`` / ``confirm`` and asserts no test id ever enters memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def template_key(q: str) -> str:
    t = re.sub(r"\d+(?:\.\d+)?", "#", _norm_q(q))
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def problem_id(dataset: str, q: str) -> str:
    return f"{dataset}:{hashlib.sha256(_norm_q(q).encode()).hexdigest()[:12]}"


def load_problems(datasets: List[str], max_per_dataset: int = 0, cache_dir: Optional[str] = None,
                  split: Optional[str] = None) -> List[Dict[str, Any]]:
    from src.dataset_loader import load_dataset_split  # type: ignore
    out: List[Dict[str, Any]] = []
    for ds in datasets:
        base, level = ds, None
        if ds.endswith("_l5"):             # e.g. math_train_l5 → MATH train, Level 5 only
            base, level = ds[:-3], "Level 5"
        exs = load_dataset_split(data_name=base, split=split, max_samples=None,
                                 cache_dir=cache_dir or None)
        if level is not None:
            exs = [e for e in exs if str(e.get("_raw", {}).get("level", "")) == level]
        seen = set()
        n = 0
        for ex in exs:
            q = ex.get("question", "")
            gold = ex.get("answer")
            if not q or gold in (None, ""):
                continue
            pid = problem_id(ds, q)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(dict(qid=pid, question=q, gold=str(gold), dataset=ds, template=template_key(q)))
            n += 1
            if max_per_dataset and n >= max_per_dataset:
                break
    return out


def _group_by_template(problems: List[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    g: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for p in problems:
        g.setdefault(p["template"], []).append(p)
    return g


def make_splits(dev_problems: List[Dict[str, Any]], test_problems: List[Dict[str, Any]], rounds: int,
                search_size: int, confirm_size: int, seed: int) -> Dict[str, Any]:
    test_templates = {p["template"] for p in test_problems}
    test_norms = {_norm_q(p["question"]) for p in test_problems}
    dev = [p for p in dev_problems
           if p["template"] not in test_templates and _norm_q(p["question"]) not in test_norms]
    dropped = len(dev_problems) - len(dev)
    groups = list(_group_by_template(dev).values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    need = rounds * (search_size + confirm_size)
    pool: List[List[Dict[str, Any]]] = groups
    search: List[List[Dict[str, Any]]] = [[] for _ in range(rounds)]
    confirm: List[List[Dict[str, Any]]] = [[] for _ in range(rounds)]
    # Fill blocks group-by-group so template duplicates never straddle splits.
    targets = [("search", r) for r in range(rounds)] + [("confirm", r) for r in range(rounds)]
    ti = 0
    for grp in pool:
        placed = False
        for _ in range(len(targets)):
            kind, r = targets[ti % len(targets)]
            ti += 1
            blk = search[r] if kind == "search" else confirm[r]
            cap = search_size if kind == "search" else confirm_size
            if len(blk) + len(grp) <= cap:
                blk.extend(grp)
                placed = True
                break
        if not placed and all(len(b) >= search_size for b in search) and \
                all(len(b) >= confirm_size for b in confirm):
            break
    got = sum(len(b) for b in search) + sum(len(b) for b in confirm)
    if got < need:
        logger.warning(f"[splits] development pool too small: filled {got}/{need} slots")
    return dict(
        seed=seed, rounds=rounds, search_size=search_size, confirm_size=confirm_size,
        dropped_overlapping_test=dropped,
        search=[[p["qid"] for p in b] for b in search],
        confirm=[[p["qid"] for p in b] for b in confirm],
        test=[p["qid"] for p in test_problems],
    )


def save_splits(splits: Dict[str, Any], problems: Dict[str, Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dev_ids = {q for b in splits["search"] + splits["confirm"] for q in b}
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump({k: v for k, v in splits.items() if k != "test"}, f, indent=1)
    with open(os.path.join(out_dir, "dev_problems.jsonl"), "w") as f:
        for q in sorted(dev_ids):
            f.write(json.dumps(problems[q]) + "\n")
    sealed = os.path.join(out_dir, "sealed_test")
    os.makedirs(sealed, exist_ok=True)
    with open(os.path.join(sealed, "test_problems.jsonl"), "w") as f:
        for q in splits["test"]:
            if q in problems:
                f.write(json.dumps(problems[q]) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


__all__ = ["load_problems", "make_splits", "save_splits", "template_key", "problem_id", "load_jsonl"]
