"""
scheduler.py
============
Drive many ``solve_gen`` generators concurrently and batch their model calls.

Each generator is sequential internally (it yields one list of requests and
blocks until they are answered).  Across generators the scheduler collects
every pending request into ONE backend call per tick, which the backend then
groups by sampling parameters and chunks by ``batch_size``.  This keeps the
GPU busy even though every individual solve makes one decision at a time.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Generator, List, Optional

from .backends import GenerationBackend

logger = logging.getLogger(__name__)


def run_generators(backend: GenerationBackend, gens: List[Generator],
                   on_done: Optional[Callable[[int, Any], None]] = None,
                   max_inflight: int = 0, desc: str = "",
                   on_tick: Optional[Callable[[], None]] = None) -> List[Any]:
    """Run generators to completion; returns their return values in order.

    ``max_inflight`` > 0 limits how many generators are active at once
    (bounded memory for large counterfactual sweeps).
    """
    n = len(gens)
    results: List[Any] = [None] * n
    pending = {}                       # idx -> list of requests
    next_start = 0
    active = 0
    finished = 0
    t0 = time.time()
    ticks = 0

    def start(idx: int):
        nonlocal active, finished
        try:
            reqs = next(gens[idx])
            pending[idx] = reqs
            active += 1
        except StopIteration as stop:
            results[idx] = stop.value
            finished += 1
            if on_done:
                on_done(idx, stop.value)

    limit = max_inflight if max_inflight > 0 else n
    while next_start < n and active < limit:
        start(next_start)
        next_start += 1

    while pending:
        order = list(pending.keys())
        flat, spans = [], []
        for idx in order:
            reqs = pending[idx]
            spans.append((idx, len(flat), len(reqs)))
            flat.extend(reqs)
        outs = backend.generate(flat) if flat else []
        ticks += 1
        for idx, off, k in spans:
            del pending[idx]
            try:
                reqs = gens[idx].send(outs[off:off + k])
                pending[idx] = reqs
            except StopIteration as stop:
                results[idx] = stop.value
                active -= 1
                finished += 1
                if on_done:
                    on_done(idx, stop.value)
                while next_start < n and active < limit:
                    start(next_start)
                    next_start += 1
        if on_tick is not None:
            on_tick()
        if ticks % 20 == 0:
            logger.info(f"[sched{(' ' + desc) if desc else ''}] tick={ticks} done={finished}/{n} "
                        f"active={active} batch={len(flat)} elapsed={time.time() - t0:.0f}s")
    return results


__all__ = ["run_generators"]
