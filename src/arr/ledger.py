"""
Per-question FLOP accounting under the hard budget B (``Ledger``).

Everything the decoder computes is charged, per forward pass, BEFORE the pass
runs: prompt prefill, every checked draft position (accepted or discarded),
ordinarily decoded positions, work of attempts that end incomplete, and — for
hybrid models with recurrent layers — the re-fed accepted prefix after a
partially rejected draft block (``draft_recompute``).
Nothing is refunded.

Cost measures
-------------
tflops  estimated FLOPs of every processed position:
            2·P + 4·L·(H·d_head)·ctx
        P = parameters excluding the input embedding (dense matmuls incl. the
        LM head), ctx = number of keys the position attends to (the QKᵀ and AV
        products).  This counts prompt prefill and rejected draft positions.
passes  serial forward passes that contain work of this question (a
        hardware-independent latency proxy under the shared batching): every
        pass costs the question 1, however many of its sequences it contains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FlopModel:
    params: float            # P (excluding input embeddings)
    n_layers: int
    attn_dim: int            # n_heads · head_dim

    def positions(self, ctx0: int, n: int) -> float:
        """FLOPs of n consecutive positions whose contexts are ctx0+1 .. ctx0+n keys."""
        if n <= 0:
            return 0.0
        sum_ctx = n * ctx0 + n * (n + 1) / 2.0
        return 2.0 * self.params * n + 4.0 * self.n_layers * self.attn_dim * sum_ctx

    def tflops(self, ctx0: int, n: int) -> float:
        return self.positions(ctx0, n) / 1e12

    @classmethod
    def from_hf_config(cls, c, params: float) -> "FlopModel":
        n_heads = c.num_attention_heads
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // n_heads
        return cls(params=float(params), n_layers=int(c.num_hidden_layers), attn_dim=int(n_heads * head_dim))


@dataclass
class Ledger:
    budget: float                          # B in `measure` units (inf → uncapped)
    measure: str = "tflops"
    spent: float = 0.0
    tflops: float = 0.0                    # always recorded
    passes: int = 0                        # passes containing work of this question
    last_pass: int = -1
    counts: Dict[str, int] = field(default_factory=lambda: dict(
        prefill=0, draft_checked=0, draft_accepted=0, draft_discarded=0, decoded=0, draft_recompute=0, calls=0))
    by_role: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.spent)

    def cost(self, tflops: float, pass_id: int) -> float:
        if self.measure == "tflops":
            return tflops
        return 0.0 if pass_id == self.last_pass else 1.0

    def affordable(self, tflops: float, pass_id: int) -> bool:
        return self.cost(tflops, pass_id) <= self.remaining + 1e-9

    def charge(self, tflops: float, pass_id: int, role: str, kind: str, n: int) -> None:
        self.spent += self.cost(tflops, pass_id)
        if pass_id != self.last_pass:
            self.passes += 1
            self.last_pass = pass_id
        self.tflops += tflops
        self.counts[kind] = self.counts.get(kind, 0) + int(n)
        r = self.by_role.setdefault(role, dict(tflops=0.0, positions=0))
        r["tflops"] += tflops
        r["positions"] += int(n)

    def note(self, kind: str, n: int) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + int(n)

    def snapshot(self) -> Dict[str, float]:
        return dict(measure=self.measure, budget=self.budget, spent=self.spent, tflops=self.tflops,
                    passes=self.passes, **self.counts)


__all__ = ["FlopModel", "Ledger"]
