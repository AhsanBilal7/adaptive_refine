"""
progress.py
===========
Human-readable progress for long batches of solves.

* Interactive terminal → a live tqdm bar.
* Log file (nohup / sbatch) → the SAME tqdm-formatted bar written as one log
  line every ``interval`` seconds (a ``\\r`` bar is unreadable in a file).

Example line:

  Round 1/5 · phase 2/5 counterfactual |████▌     | 312/932 solves [1:02:11<2:04:00] left=620 in-flight=256 model-calls=6,480
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict

from tqdm import tqdm

logger = logging.getLogger("progress")

# Set by the outer loop / evaluator so every bar says where we are.
CONTEXT: Dict[str, str] = {"label": ""}


def set_context(label: str) -> None:
    CONTEXT["label"] = label


class Progress:
    def __init__(self, total: int, done: int, desc: str, unit: str = "solves", interval: float = 60.0):
        self.total, self.done0, self.unit = total, done, unit
        label = CONTEXT["label"]
        self.prefix = f"{label} · {desc}" if label else desc
        self.interval = interval
        self.t0 = time.time()
        self.last = 0.0
        self.tty = sys.stderr.isatty()
        self.bar = None
        if self.tty:
            self.bar = tqdm(total=total, initial=done, desc=self.prefix, unit=unit, dynamic_ncols=True)
        else:
            logger.info(self._line(done, dict(active=0, calls=0)))

    def _line(self, done: int, info: Dict[str, Any]) -> str:
        elapsed = time.time() - self.t0
        # rate from THIS session only, so the ETA is not inflated by resumed work
        meter = tqdm.format_meter(done, self.total, elapsed, ncols=100, prefix=self.prefix, unit=self.unit,
                                  initial=self.done0)
        return (f"{meter} | left={self.total - done} in-flight={info.get('active', 0)} "
                f"model-calls={info.get('calls', 0):,}")

    def update(self, done: int, info: Dict[str, Any], force: bool = False) -> None:
        if self.bar is not None:
            self.bar.n = done
            self.bar.set_postfix(left=self.total - done, inflight=info.get("active", 0),
                                 calls=info.get("calls", 0), refresh=False)
            self.bar.refresh()
            return
        if force or time.time() - self.last >= self.interval:
            logger.info(self._line(done, info))
            self.last = time.time()

    def close(self, done: int, info: Dict[str, Any]) -> None:
        if self.bar is not None:
            self.bar.n = done
            self.bar.close()
        else:
            self.update(done, info, force=True)


__all__ = ["Progress", "set_context", "CONTEXT"]
