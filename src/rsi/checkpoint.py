"""
checkpoint.py
=============
Continuous, crash-safe saving for every batch of solves (development phases
and evaluation), so a killed job resumes where it stopped.

For a job family stored under ``<prefix>``:

  <prefix>.done.pkl      append-only pickle stream: (job_id, result) per finished solve
  <prefix>.inflight.pkl  atomic snapshot (every ``save_every_s``): the CURRENT
                         SolveState of every unfinished solve + auxiliary hook state
  <prefix>.status.json   progress (done / total) for monitoring

Resume rules
------------
* finished jobs are loaded from the done-stream and never re-run;
* an in-flight job restarts from its last saved SolveState.  States are only
  saved between scheduler ticks (every generator is suspended at a yield), so
  a saved state is always consistent: calls already made are charged, and an
  operation interrupted mid-way is treated like an incomplete operation (the
  slot is unchanged, its spent compute stays charged, the decision is re-made);
* forced counterfactual decisions already executed (recorded in
  ``state.decisions`` with ``forced=True``) are not re-applied.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .backends import GenerationBackend
from .procedure import Decision
from .scheduler import run_generators
from .state import SolveState

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    state: SolveState
    make_gen: Callable[[SolveState, List[Decision]], Any]   # (state, remaining forced) -> generator
    forced: List[Decision] = field(default_factory=list)
    start_n_dec: int = 0

    def remaining_forced(self) -> List[Decision]:
        done = sum(1 for d in self.state.decisions[self.start_n_dec:] if d.get("forced"))
        return [Decision(d.action, d.slot, d.slot2, d.plan, d.item_id) for d in self.forced[done:]]


def atomic_pickle(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def load_pickle(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_done_stream(path: str) -> Dict[str, Any]:
    done: Dict[str, Any] = {}
    if not os.path.exists(path):
        return done
    with open(path, "rb") as f:
        while True:
            try:
                jid, res = pickle.load(f)
                done[jid] = res
            except EOFError:
                break
            except Exception:           # truncated tail from a kill mid-write
                logger.warning(f"[ckpt] truncated record at end of {path}; ignoring it")
                break
    return done


def run_jobs(backend: GenerationBackend, jobs: List[Job], prefix: Optional[str], desc: str = "",
             save_every_s: float = 60.0, max_inflight: int = 0,
             aux_get: Optional[Callable[[], Any]] = None,
             aux_set: Optional[Callable[[Any], None]] = None) -> List[Any]:
    """Run jobs to completion with continuous checkpointing (``prefix=None`` → no files)."""
    if prefix is None:
        gens = [j.make_gen(j.state, j.remaining_forced()) for j in jobs]
        return run_generators(backend, gens, max_inflight=max_inflight, desc=desc)

    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    done_path, infl_path, stat_path = prefix + ".done.pkl", prefix + ".inflight.pkl", prefix + ".status.json"
    done = _read_done_stream(done_path)
    infl = load_pickle(infl_path, default={}) or {}
    if infl.get("aux") is not None and aux_set is not None:
        aux_set(infl["aux"])
    saved_states: Dict[str, SolveState] = infl.get("states", {})
    todo: List[Job] = []
    for j in jobs:
        if j.id in done:
            continue
        if j.id in saved_states:
            j.state = saved_states[j.id]
        todo.append(j)
    n_resumed = sum(1 for j in todo if j.id in saved_states)
    if done or n_resumed:
        logger.info(f"[ckpt {desc}] resume: {len(done)} done, {n_resumed} in-flight restored, "
                    f"{len(todo) - n_resumed} fresh (of {len(jobs)})")

    done_f = open(done_path, "ab")
    live = {j.id: j for j in todo}
    last = [time.time()]

    def snapshot(force: bool = False):
        if not force and time.time() - last[0] < save_every_s:
            return
        atomic_pickle(dict(states={jid: j.state for jid, j in live.items()},
                           aux=aux_get() if aux_get else None), infl_path)
        atomic_json(dict(desc=desc, total=len(jobs), done=len(jobs) - len(live), inflight=len(live),
                         updated=time.strftime("%Y-%m-%d %H:%M:%S")), stat_path)
        last[0] = time.time()

    def on_done(idx: int, res: Any):
        j = todo[idx]
        pickle.dump((j.id, res), done_f, protocol=pickle.HIGHEST_PROTOCOL)
        done_f.flush()
        os.fsync(done_f.fileno())
        done[j.id] = res
        live.pop(j.id, None)

    try:
        gens = [j.make_gen(j.state, j.remaining_forced()) for j in todo]
        run_generators(backend, gens, on_done=on_done, on_tick=lambda: snapshot(False),
                       max_inflight=max_inflight, desc=desc)
    finally:
        done_f.close()
        snapshot(force=True)
    return [done[j.id] for j in jobs]


__all__ = ["Job", "run_jobs", "atomic_pickle", "atomic_json", "load_pickle"]
