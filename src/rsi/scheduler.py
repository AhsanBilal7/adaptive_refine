"""
scheduler.py
============
Drive many ``solve_gen`` generators concurrently with continuous batching.

Each generator is sequential internally (it yields a list of requests and
blocks until ALL of them are answered).  Across generators the scheduler
keeps one FIFO queue of pending requests and repeatedly sends the next chunk
(up to ``backend.chunk_size`` requests with identical sampling parameters) to
the model.  As soon as every request of a generator has an answer, that
generator is advanced and its next requests join the queue — nobody waits for
the slowest request of a huge batch, and every chunk is a checkpoint point.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable, Dict, Generator, List, Optional

from .backends import GenerationBackend

logger = logging.getLogger(__name__)


def run_generators(backend: GenerationBackend, gens: List[Generator],
                   on_done: Optional[Callable[[int, Any], None]] = None,
                   max_inflight: int = 0, desc: str = "",
                   on_tick: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Any]:
    """Run generators to completion; returns their return values in order.

    ``max_inflight`` > 0 limits how many generators are active at once.
    ``on_tick(info)`` is called after every model chunk with progress counters.
    """
    n = len(gens)
    results: List[Any] = [None] * n
    slots: Dict[int, Dict[str, Any]] = {}      # idx -> {"reqs", "outs", "left"}
    queue: deque = deque()                     # (idx, position)
    chunk = int(getattr(backend, "chunk_size", 0) or 0)
    info = dict(done=0, total=n, active=0, calls=0, chunks=0, t0=time.time())
    next_start = 0
    limit = max_inflight if max_inflight > 0 else n

    def push(idx: int, reqs) -> None:
        if not reqs:                           # a generator must always ask for something
            raise RuntimeError(f"generator {idx} yielded an empty request list")
        slots[idx] = dict(reqs=reqs, outs=[None] * len(reqs), left=len(reqs))
        for pos in range(len(reqs)):
            queue.append((idx, pos))

    def finish(idx: int, value: Any) -> None:
        results[idx] = value
        info["done"] += 1
        info["active"] -= 1
        if on_done:
            on_done(idx, value)

    def start_more() -> None:
        nonlocal next_start
        while next_start < n and info["active"] < limit:
            idx = next_start
            next_start += 1
            info["active"] += 1
            try:
                push(idx, next(gens[idx]))
            except StopIteration as stop:
                finish(idx, stop.value)

    start_more()
    while queue:
        # next chunk: the oldest request's sampling key, FIFO among equals
        key = slots[queue[0][0]]["reqs"][queue[0][1]].sampling_key()
        take, keep = [], deque()
        while queue:
            idx, pos = queue.popleft()
            if (not chunk or len(take) < chunk) and slots[idx]["reqs"][pos].sampling_key() == key:
                take.append((idx, pos))
            else:
                keep.append((idx, pos))
        queue = keep
        outs = backend.generate([slots[i]["reqs"][p] for i, p in take])
        info["calls"] += len(take)
        info["chunks"] += 1
        ready = []
        for (idx, pos), out in zip(take, outs):
            s = slots[idx]
            s["outs"][pos] = out
            s["left"] -= 1
            if s["left"] == 0:
                ready.append(idx)
        for idx in ready:
            s = slots.pop(idx)
            try:
                push(idx, gens[idx].send(s["outs"]))
            except StopIteration as stop:
                finish(idx, stop.value)
        start_more()
        if on_tick is not None:
            on_tick(dict(info, batch=len(take), queued=len(queue)))
    return results


__all__ = ["run_generators"]
