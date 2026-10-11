"""
Batched decoder of ARR and of the baselines.

A persistent, continuously batched decoding loop over a HuggingFace causal LM.
Each request is one call (initial sample, revision, critique, ...) and runs in
one of four modes:

  sample   ordinary autoregressive sampling (temperature / top-p, per-request RNG)
  greedy   ordinary greedy decoding
  replay   greedy decoding that uses the previous solution as a token draft:
           the old tokens are checked in blocks of at most W positions in one
           parallel pass; the longest matching prefix is accepted; at the first
           mismatch the target's own argmax is emitted, the speculative entries
           after the accepted prefix are discarded, and the revision finishes by
           ordinary greedy decoding (no realignment, at most one failed block)
  pld      greedy decoding with prompt-lookup drafting (PLD baseline: n-gram
           lookup in prompt + output, draft of ``pld_num_pred`` tokens,
           verification of the longest matching prefix, repeated after mismatches)

All greedy modes produce the same output as ordinary greedy decoding up to
finite-precision differences between parallel and serial evaluation
(Proposition 1 of the paper); ``run_arr.py pilot`` measures the exact agreement on the backend.

KV cache
--------
A static slot cache: every running request owns one slot (row) and a cursor
(number of committed positions).  A pass feeds each row s tokens written at
cursor..cursor+s-1; accepted positions advance the cursor, rejected speculative
positions are simply left behind the cursor and overwritten later, so the
cache never holds invalid entries inside the committed prefix.  Rows that are
part of a pass's slot range but not of its group are fed padding written behind
their cursor (harmless).  KV entries are only reused within the current call
as in the paper: every new call is prefilled from its own prompt.

Budget
------
Before every forward pass, each row's cost is checked against its question's
ledger (ledger.py); a replay / PLD block is shortened to fit, and a row that
cannot afford even one position ends with ``finish='budget'``.  All processed
positions — prefill, checked draft (accepted and discarded), ordinary decoding
— are charged, including the work of attempts that end incomplete.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .ledger import FlopModel, Ledger

logger = logging.getLogger(__name__)

MODES = ("sample", "greedy", "replay", "pld")
ATTN_IMPL = "arr_grouped"
_SCORE_ELEMS = 1 << 27          # attention-score elements per query chunk (bounds memory)


# Request

@dataclass
class Request:
    prompt_ids: List[int]
    max_new: int
    mode: str
    ledger: Ledger
    role: str = "generate"
    draft: Optional[List[int]] = None       # previous solution tokens incl. termination (replay / identity check)
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    tag: Any = None                         # opaque payload of the caller
    wc: float = 0.0                         # worst-case cost in ledger units (set by the engine)
    logprobs: Optional[List[float]] = None  # set to [] to record log p(token) of every sampled token (ConMA)

    # filled by the engine
    rid: int = -1
    out: List[int] = field(default_factory=list)
    finish: str = ""                        # eos | length | budget
    text: str = ""
    slot: int = -1
    cursor: int = 0
    on_draft: bool = False
    spent: float = 0.0                      # ledger units charged by this request
    stats: Dict[str, int] = field(default_factory=lambda: dict(
        prefill=0, draft_checked=0, draft_accepted=0, draft_discarded=0, decoded=0, draft_recompute=0, passes=0))
    t_submit: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0
    _gen: Any = None

    @property
    def complete(self) -> bool:
        return self.finish == "eos"

    @property
    def lcp(self) -> int:
        """Length of the common prefix of the output and the draft."""
        if not self.draft:
            return 0
        n = 0
        for a, b in zip(self.out, self.draft):
            if a != b:
                break
            n += 1
        return n

    @property
    def identical(self) -> bool:
        """The complete output, including termination, equals the draft."""
        return self.complete and self.draft is not None and self.out == list(self.draft)


# Attention without materialising repeated KV heads (grouped-query aware)

def grouped_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Eager attention (same arithmetic as transformers' eager path: bf16 QKᵀ,
    fp32 softmax, bf16 AV) computed per KV group, in query chunks.
    query [B,Hq,s,D], key/value [B,Hkv,T,D], attention_mask bool [B,1,s,T]."""
    B, Hq, s, D = query.shape
    Hkv, T = key.shape[1], key.shape[2]
    g = Hq // Hkv
    scaling = scaling if scaling is not None else D ** -0.5
    q = query.view(B, Hkv, g, s, D)
    out = torch.empty_like(q)
    kt = key.transpose(-1, -2)
    step = max(1, _SCORE_ELEMS // max(1, B * Hq * T))
    neg = torch.finfo(query.dtype).min
    for a in range(0, s, step):
        b = min(s, a + step)
        qc = q[:, :, :, a:b].reshape(B, Hkv, g * (b - a), D)
        sc = torch.matmul(qc, kt) * scaling
        if attention_mask is not None:
            m = attention_mask[:, :, a:b, :T].unsqueeze(2).expand(B, 1, g, b - a, T).reshape(B, 1, g * (b - a), T)
            sc = sc.masked_fill(~m, neg)
        p = torch.softmax(sc, dim=-1, dtype=torch.float32).to(query.dtype)
        out[:, :, :, a:b] = torch.matmul(p, value).view(B, Hkv, g, b - a, D)
    return out.view(B, Hq, s, D).transpose(1, 2).contiguous(), None


def _register_attention() -> None:
    from transformers import AttentionInterface
    AttentionInterface.register(ATTN_IMPL, grouped_attention)


# Slot cache

class _LinearLayerView:
    """What transformers' linear-attention layers read from ``cache.layers[i]`` during one pass."""

    def __init__(self):
        self.record_past = False
        self.conv_states = {0: None}
        self.recurrent_states = {0: None}


def _make_slot_cache_cls():
    from transformers.cache_utils import Cache

    class SlotCache(Cache):
        """Static per-slot caches.  For attention layers ``update`` writes the new keys/values at
        per-row positions and returns what the pass attends to.  For linear-attention (recurrent)
        layers, the conv / recurrent states of every slot are stored here; a pass works on gathered
        copies that are written back only for rows whose fed tokens were all accepted (rollback =
        keeping the stored state)."""

        is_compileable = False

        def __init__(self, specs, rows, max_len, device):  # noqa: D401
            self.specs = specs
            self.k, self.v, self.lin_conv, self.lin_rec = {}, {}, {}, {}
            for l, sp in enumerate(specs):
                if sp.get("kv"):
                    n_kv, hd, dt = sp["kv"]
                    self.k[l] = torch.zeros(rows, n_kv, max_len, hd, dtype=dt, device=device)
                    self.v[l] = torch.zeros(rows, n_kv, max_len, hd, dtype=dt, device=device)
                if sp.get("lin"):
                    (cshape, cdt), (rshape, rdt) = sp["lin"]
                    self.lin_conv[l] = torch.zeros(rows, *cshape, dtype=cdt, device=device)
                    self.lin_rec[l] = torch.zeros(rows, *rshape, dtype=rdt, device=device)
            self.layers = [_LinearLayerView() if l in self.lin_conv else None for l in range(len(specs))] \
                if self.lin_conv else []
            self.max_len = max_len
            self._mode = None

        def view(self, lo: int, hi: int, write_pos: torch.Tensor, t_len: int) -> None:
            n, s = write_pos.shape
            self._mode = ("view", lo, hi, torch.arange(n, device=write_pos.device)[:, None].expand(n, s), write_pos, t_len)

        def fresh(self, slots: torch.Tensor, write_pos: torch.Tensor) -> None:
            n, s = write_pos.shape
            self._mode = ("fresh", slots[:, None].expand(n, s), write_pos, slots)

        def gather(self, slots: torch.Tensor, write_pos: torch.Tensor, t_len: int) -> None:
            """Exactly these rows (no other slot is touched); linear states are gathered copies."""
            n, s = write_pos.shape
            self._mode = ("gather", slots, slots[:, None].expand(n, s), write_pos, t_len)
            for l, L in enumerate(self.layers):
                if L is not None:
                    L.conv_states[0] = self.lin_conv[l][slots].clone()
                    L.recurrent_states[0] = self.lin_rec[l][slots].clone()

        def commit(self, keep: List[bool]) -> None:
            """Write back the linear states of the gathered rows where ``keep`` is True (integer
            indices built on the host, so no device synchronisation)."""
            if self._mode is None or self._mode[0] != "gather" or not any(keep):
                return
            slots = self._mode[1]
            sel = None
            if not all(keep):
                sel = torch.tensor([i for i, k in enumerate(keep) if k], device=slots.device)
                slots = slots.index_select(0, sel)
            for l, L in enumerate(self.layers):
                if L is not None:
                    c, r = L.conv_states[0], L.recurrent_states[0]
                    if sel is not None:
                        c, r = c.index_select(0, sel), r.index_select(0, sel)
                    self.lin_conv[l].index_copy_(0, slots, c.to(self.lin_conv[l].dtype))
                    self.lin_rec[l].index_copy_(0, slots, r.to(self.lin_rec[l].dtype))

        # attention layers
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None, **kwargs):
            ks, vs = key_states.permute(0, 2, 1, 3), value_states.permute(0, 2, 1, 3)
            mode = self._mode[0]
            if mode == "view":
                _, lo, hi, ri, pi, t_len = self._mode
                kb, vb = self.k[layer_idx][lo:hi], self.v[layer_idx][lo:hi]
                kb[ri, :, pi] = ks
                vb[ri, :, pi] = vs
                return kb[:, :, :t_len], vb[:, :, :t_len]
            if mode == "gather":
                _, slots, ri, pi, t_len = self._mode
                self.k[layer_idx][ri, :, pi] = ks
                self.v[layer_idx][ri, :, pi] = vs
                return self.k[layer_idx][slots, :, :t_len], self.v[layer_idx][slots, :, :t_len]
            _, ri, pi, _ = self._mode
            self.k[layer_idx][ri, :, pi] = ks
            self.v[layer_idx][ri, :, pi] = vs
            return key_states, value_states

        # linear-attention layers (transformers' LinearAttention cache interface)
        def has_previous_state(self, layer_idx: int = 0, state_idx: int = 0) -> bool:
            return self._mode is not None and self._mode[0] == "gather"

        def update_conv_state(self, conv_states, layer_idx, state_idx: int = 0, conv_kernel_size=None, **kwargs):
            k = self.lin_conv[layer_idx].shape[-1]
            if self._mode[0] == "fresh":
                tail = conv_states[..., -k:]
                if tail.shape[-1] < k:
                    tail = torch.nn.functional.pad(tail, (k - tail.shape[-1], 0))
                self.lin_conv[layer_idx][self._mode[3]] = tail.to(self.lin_conv[layer_idx].dtype)
                return conv_states
            L = self.layers[layer_idx]
            full = torch.cat([L.conv_states[0].to(conv_states.dtype), conv_states], dim=-1)
            L.conv_states[0] = full[..., -k:].contiguous()
            return full

        def update_recurrent_state(self, recurrent_states, layer_idx, state_idx: int = 0, **kwargs):
            if self._mode[0] == "fresh":
                self.lin_rec[layer_idx][self._mode[3]] = recurrent_states.to(self.lin_rec[layer_idx].dtype)
                return recurrent_states
            self.layers[layer_idx].recurrent_states[0] = recurrent_states
            return recurrent_states

        def swap(self, a: int, na: int, b: int, nb: int) -> None:
            """Exchange the committed KV prefixes of slots a (na positions) and b (nb positions)."""
            for buf in (self.k, self.v):
                for l in buf:
                    ta = buf[l][a, :, :na].clone()
                    buf[l][a, :, :nb] = buf[l][b, :, :nb]
                    buf[l][b, :, :na] = ta

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return 0

    return SlotCache


def discover_layer_specs(backbone, device) -> List[Dict[str, Any]]:
    """One tiny forward through the model with transformers' own cache reveals, per layer, whether it
    keeps keys/values (heads, head_dim, dtype), linear-attention conv/recurrent states (shapes, dtypes)
    or nothing (KV-shared layers).  Works for uniform, sliding-window, KV-sharing and hybrid models."""
    from transformers import DynamicCache
    try:
        cache = DynamicCache(config=backbone.config)
    except TypeError:
        cache = DynamicCache()
    ids = torch.zeros(1, 2, dtype=torch.long, device=device)
    with torch.no_grad():
        backbone(input_ids=ids, past_key_values=cache, use_cache=True,
                 position_ids=torch.arange(2, device=device)[None])
    specs = []
    for layer in getattr(cache, "layers", []):
        sp: Dict[str, Any] = {}
        keys = getattr(layer, "keys", None)
        if isinstance(keys, torch.Tensor) and keys.dim() == 4 and keys.numel():
            sp["kv"] = (int(keys.shape[1]), int(keys.shape[3]), keys.dtype)
        conv = getattr(layer, "conv_states", {}) or {}
        rec = getattr(layer, "recurrent_states", {}) or {}
        c0, r0 = conv.get(0) if isinstance(conv, dict) else None, rec.get(0) if isinstance(rec, dict) else None
        if isinstance(c0, torch.Tensor) and isinstance(r0, torch.Tensor):
            sp["lin"] = ((tuple(c0.shape[1:]), c0.dtype), (tuple(r0.shape[1:]), r0.dtype))
        specs.append(sp)
    return specs


# Prompt-lookup drafting (the PLD baseline; reference-implementation semantics)

def prompt_lookup(seq: np.ndarray, max_ngram: int, num_pred: int) -> List[int]:
    """Earliest occurrence of the trailing n-gram (n = max_ngram .. 1) in ``seq``
    whose continuation is non-empty; returns up to num_pred following tokens."""
    L = len(seq)
    for n in range(min(max_ngram, L - 1), 0, -1):
        tail = seq[L - n:]
        win = np.lib.stride_tricks.sliding_window_view(seq[:L - 1], n)
        hits = np.flatnonzero((win == tail).all(axis=1))
        for h in hits:
            start = h + n
            if start < L:
                return seq[start:min(L, start + num_pred)].tolist()
    return []


# Engine

class Engine:
    def __init__(self, cfg, device: Optional[str] = None, model=None, tokenizer="auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _register_attention()
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = getattr(torch, cfg.dtype) if cfg.dtype != "auto" else torch.bfloat16
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.hf_cache_dir or None,
                                                      trust_remote_code=cfg.trust_remote_code)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, dtype=dtype, cache_dir=cfg.hf_cache_dir or None,
                trust_remote_code=cfg.trust_remote_code, attn_implementation=ATTN_IMPL,
            ).to(self.device)
        else:
            model.config._attn_implementation = ATTN_IMPL
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.dtype = next(self.model.parameters()).dtype
        # text backbone + LM head (composite vision-language checkpoints keep the text model under
        # ``model.language_model``; only the text path is used)
        self.backbone = self.model.model
        self.composite = hasattr(self.backbone, "language_model")
        if self.composite:
            self.backbone = self.backbone.language_model
        self.lm_head = self.model.lm_head
        c = self.backbone.config
        self.hf_config = c
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos = set(gen_eos if isinstance(gen_eos, (list, tuple)) else ([gen_eos] if gen_eos is not None else []))
        if tokenizer is not None and tokenizer.eos_token_id is not None:
            eos.add(tokenizer.eos_token_id)
        self.eos = eos
        self.pad_id = (tokenizer.pad_token_id if tokenizer is not None and tokenizer.pad_token_id is not None
                       else next(iter(eos)))
        per_layer_emb = any(isinstance(m, torch.nn.Embedding) and m is not self.backbone.get_input_embeddings()
                            for m in self.backbone.modules())
        if self.composite or per_layer_emb:
            # matmul parameters only: text backbone without any embedding table, plus the LM head
            emb_ids = {id(p) for m in self.backbone.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters()}
            self.params = float(sum(p.numel() for p in self.backbone.parameters() if id(p) not in emb_ids)
                                + self.lm_head.weight.numel())
        else:
            inp = self.model.get_input_embeddings().weight
            self.params = float(sum(p.numel() for p in self.model.parameters()) - inp.numel())
            if self.lm_head.weight.data_ptr() == inp.data_ptr():     # tied output head: still a matmul per position
                self.params += float(inp.numel())
        lt = list(getattr(c, "layer_types", None) or [])
        self.hybrid = any(t == "linear_attention" for t in lt)
        n_attn_layers = sum(1 for t in lt if t != "linear_attention") if lt else c.num_hidden_layers
        self._n_attn_layers = int(n_attn_layers)             # recurrent layers have no growing context term
        self.softcap = getattr(c, "final_logit_softcapping", None) or None
        import inspect
        self._fwd_cache_position = "cache_position" in inspect.signature(self.backbone.forward).parameters
        self.model_id = cfg.model_name + "@" + str(getattr(c, "_commit_hash", "") or "local")

        self.W = int(cfg.block_size)
        self.max_len = int(cfg.max_input_tokens + cfg.max_new_tokens + max(self.W, cfg.pld_num_pred) + 2)
        # Layer kinds supported: full attention, sliding-window attention (windowed mask), KV-shared
        # layers (no cache of their own) and linear-attention / recurrent layers (hybrid models).
        unknown = set(lt) - {"full_attention", "sliding_attention", "linear_attention"}
        if unknown:
            raise NotImplementedError(f"layer types {unknown} are not supported by the ARR engine")
        sw = getattr(c, "sliding_window", None)
        self.window = int(sw) if (sw and "sliding_attention" in lt and sw < self.max_len) else 0
        self.specs = discover_layer_specs(self.backbone, self.device)
        # FLOP model: per-position 2·P plus the attention term over attention layers; the head size
        # comes from the discovered layers (heterogeneous models have per-layer head sizes)
        head_dims = [sp["kv"][1] for sp in self.specs if sp.get("kv")] or [c.hidden_size // max(1, c.num_attention_heads)]
        self.flops = FlopModel(params=self.params, n_layers=self._n_attn_layers,
                               attn_dim=int(round(c.num_attention_heads * float(np.mean(head_dims)))))
        self.row_bytes = self._row_bytes()
        self.rows = int(cfg.engine_max_rows) or self._auto_rows()
        self.cache = _make_slot_cache_cls()(self.specs, self.rows, self.max_len, self.device)
        self.slots: List[Optional[Request]] = [None] * self.rows
        self.queue: Deque[Request] = deque()
        self.pass_id = 0
        self.n_rid = 0
        self.stats = dict(passes=0, padded_positions=0, real_positions=0, migrations=0)
        logger.info(f"[engine] {cfg.model_name} dtype={self.dtype} rows={self.rows} max_len={self.max_len} "
                    f"W={self.W} params(no-emb)={self.params / 1e9:.2f}B eos={sorted(self.eos)}"
                    + (f" hybrid(linear layers={sum(1 for x in self.specs if x.get('lin'))})" if self.hybrid else "")
                    + (f" sliding_window={self.window}" if self.window else "")
                    + (f" logit_softcap={self.softcap}" if self.softcap else "")
                    + f" row={self.row_bytes / 2 ** 30:.3f}GB")

    # sizing
    def _row_bytes(self) -> int:
        """Bytes of one slot: keys+values of every caching layer at max_len, plus recurrent states."""
        b = 0
        for sp in self.specs:
            if sp.get("kv"):
                n_kv, hd, dt = sp["kv"]
                b += 2 * n_kv * hd * self.max_len * torch.finfo(dt).bits // 8
            if sp.get("lin"):
                for shape, dt in sp["lin"]:
                    b += int(np.prod(shape)) * torch.finfo(dt).bits // 8
        return max(1, b)

    def _auto_rows(self) -> int:
        if self.device.type != "cuda":
            return 8
        free, total = torch.cuda.mem_get_info(self.device)
        usable = total * float(self.cfg.gpu_memory_utilization) - (total - free) - self.cfg.engine_workspace_gb * 2 ** 30
        rows = int(max(1, usable // self.row_bytes))
        return min(rows, 512)

    # forward helpers
    def _masks(self, mask: torch.Tensor, pos: torch.Tensor, keys: torch.Tensor):
        """The causal mask, or per-layer-type masks (sliding window / linear layers) when needed."""
        if not self.window and not self.hybrid:
            return mask
        d: Dict[str, Any] = {"full_attention": mask}
        if self.window:
            d["sliding_attention"] = mask & ((pos[:, None, :, None] - keys[None, None, None, :]) < self.window)
        if self.hybrid:
            d["linear_attention"] = None
        return d

    def _forward(self, ids, mask, pos, width):
        kw = dict(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=self.cache, use_cache=True)
        if self._fwd_cache_position:
            kw["cache_position"] = torch.arange(width, device=self.device)
        return self.backbone(**kw).last_hidden_state

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(h)
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    # API
    def format(self, system: str, user: str) -> List[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = {"enable_thinking": self.cfg.thinking == "on"}
        try:
            text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except Exception:
            msgs = [{"role": "user", "content": (system + "\n\n" if system else "") + user}]
            text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids[-self.cfg.max_input_tokens:]

    def decode(self, ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            return " ".join(str(t) for t in ids)
        return self.tokenizer.decode(list(ids), skip_special_tokens=True).strip()

    def worst_case(self, prompt_len: int, max_new: int, measure: str) -> float:
        if measure == "passes":
            return float(1 + max_new)
        extra = max(self.W, self.cfg.pld_num_pred)
        return self.flops.tflops(0, prompt_len) + self.flops.tflops(prompt_len, max_new + extra)

    def submit(self, req: Request) -> None:
        assert req.mode in MODES, req.mode
        if req.mode == "replay":
            assert req.draft, "replay needs a draft"
        req.prompt_ids = list(req.prompt_ids)[-self.cfg.max_input_tokens:]
        req.max_new = min(int(req.max_new), self.max_len - len(req.prompt_ids) - max(self.W, self.cfg.pld_num_pred) - 2)
        req.rid = self.n_rid
        self.n_rid += 1
        req.t_submit = time.time()
        req.wc = self.worst_case(len(req.prompt_ids), req.max_new, req.ledger.measure)
        self.queue.append(req)

    @property
    def busy(self) -> bool:
        return bool(self.queue) or any(s is not None for s in self.slots)

    def n_running(self) -> int:
        return sum(s is not None for s in self.slots)

    def step(self) -> List[Request]:
        """Admit + prefill queued requests into free slots, then run one pass for
        every group of running requests.  Returns the requests that finished."""
        done: List[Request] = []
        self._admit(done)
        self._decode_tick(done)
        for r in done:
            if r.slot >= 0 and self.slots[r.slot] is r:
                self.slots[r.slot] = None
            r.t_end = time.time()
            r.text = self.decode([t for t in r.out if t not in self.eos])
        return done

    def run(self, reqs: Sequence[Request]) -> List[Request]:
        """Convenience: run requests to completion (order preserved)."""
        for r in reqs:
            self.submit(r)
        while self.busy:
            self.step()
        return list(reqs)

    # internals
    def _charge(self, r: Request, n: int, kind: str) -> bool:
        """Check + charge n positions starting at r.cursor for this pass."""
        c = self.flops.tflops(r.cursor, n)
        if not r.ledger.affordable(c, self.pass_id):
            return False
        before = r.ledger.spent
        r.ledger.charge(c, self.pass_id, r.role, kind, n)
        r.spent += r.ledger.spent - before
        r.stats[kind] += n
        return True

    def _finish(self, r: Request, why: str, done: List[Request]) -> None:
        if not r.finish:
            r.finish = why
            done.append(r)

    def _free_slot(self, low: bool) -> int:
        idx = range(self.rows) if low else range(self.rows - 1, -1, -1)
        for i in idx:
            if self.slots[i] is None:
                return i
        return -1

    def _admit(self, done: List[Request]) -> None:
        new: List[Request] = []
        while self.queue:
            r = self.queue[0]
            slot = self._free_slot(low=r.mode in ("replay", "pld"))
            if slot < 0:
                break
            self.queue.popleft()
            r.slot = slot
            r.t_start = time.time()
            self.slots[slot] = r
            new.append(r)
        if not new:
            return
        # prefill in memory-bounded groups (rows sorted by prompt length)
        new.sort(key=lambda r: len(r.prompt_ids))
        budget_tokens = 8192                    # prefill tokens per forward (bounds the activation peak)
        i = 0
        while i < len(new):
            j, longest = i, 0
            while j < len(new):
                longest = max(longest, len(new[j].prompt_ids))
                if (j - i + 1) * longest > budget_tokens and j > i:
                    break
                if self.hybrid and j > i and len(new[j].prompt_ids) != len(new[i].prompt_ids):
                    break
                j += 1
            group = new[i:j]
            i = j
            self.pass_id += 1
            live = []
            for r in group:
                r.cursor = 0
                if self._charge(r, len(r.prompt_ids), "prefill"):
                    live.append(r)
                    r.stats["passes"] += 1
                else:
                    self._finish(r, "budget", done)
            if live:
                self._prefill(live)
            for r in live:
                self._after_first_token(r, done)

    @torch.no_grad()
    def _prefill(self, rows: List[Request]) -> None:
        s = max(len(r.prompt_ids) for r in rows)
        n = len(rows)
        ids = torch.full((n, s), self.pad_id, dtype=torch.long)
        for k, r in enumerate(rows):
            ids[k, :len(r.prompt_ids)] = torch.tensor(r.prompt_ids)
        ids = ids.to(self.device)
        pos = torch.arange(s, device=self.device)[None].expand(n, s)
        mask = torch.ones(s, s, dtype=torch.bool, device=self.device).tril()[None, None].expand(n, 1, s, s)
        slots = torch.tensor([r.slot for r in rows], device=self.device)
        self.cache.fresh(slots, pos)
        h = self._forward(ids, self._masks(mask, pos, torch.arange(s, device=self.device)), pos, s)
        last = torch.tensor([len(r.prompt_ids) - 1 for r in rows], device=self.device)
        logits = self._logits(h[torch.arange(n, device=self.device), last])
        self.stats["passes"] += 1
        self.stats["real_positions"] += sum(len(r.prompt_ids) for r in rows)
        toks = self._choose(rows, logits)
        for r, t in zip(rows, toks):
            r.cursor = len(r.prompt_ids)
            r.out = [t]

    def _after_first_token(self, r: Request, done: List[Request]) -> None:
        if r.mode == "replay":
            r.on_draft = r.out[0] == r.draft[0]
            if r.on_draft:
                r.stats["draft_accepted"] += 1
                r.ledger.note("draft_accepted", 1)
        self._check_end(r, done)

    def _check_end(self, r: Request, done: List[Request]) -> None:
        for k, t in enumerate(r.out):
            if t in self.eos:
                del r.out[k + 1:]
                self._finish(r, "eos", done)
                return
        if len(r.out) >= r.max_new:
            del r.out[r.max_new:]
            self._finish(r, "length", done)

    def _choose(self, rows: List[Request], logits: torch.Tensor) -> List[int]:
        """One token per row from its last-position logits (sample or argmax)."""
        out = logits.float().argmax(-1).tolist()
        ks = [k for k, r in enumerate(rows) if r.mode == "sample" and r.temperature > 0]
        for k, t in zip(ks, self._sample_rows([rows[k] for k in ks], logits, ks)):
            out[k] = t
        return out

    def _sample_rows(self, rs: List[Request], logits: torch.Tensor, idx: List[int]) -> List[int]:
        """One sampled token for each request rs[i] from logits[idx[i]]."""
        if not rs:
            return []
        if self.cfg.sampler == "per_row":
            toks = [self._sample(r, logits[k]) for r, k in zip(rs, idx)]
        else:
            toks = self._sample_batched(rs, logits[torch.tensor(idx, device=logits.device)])
        need = [i for i, r in enumerate(rs) if r.logprobs is not None]
        if need:                              # log-probability of the drawn token under the model (temperature 1)
            dev = logits.device
            rows = logits[torch.tensor([idx[i] for i in need], device=dev)].float()
            tok = torch.tensor([toks[i] for i in need], device=dev)[:, None]
            for i, v in zip(need, torch.log_softmax(rows, -1).gather(1, tok)[:, 0].tolist()):
                rs[i].logprobs.append(float(v))
        return toks

    def _sample_batched(self, rs: List[Request], logits: torch.Tensor) -> List[int]:
        """Temperature + top-p sampling of all rows in one pass.  Each request draws its uniform from
        its own CPU generator (seeded like the per-row sampler), so a request's tokens do not depend
        on which other rows share the batch; the token is the inverse CDF of the renormalised nucleus
        (descending-probability order, as in the per-row sampler)."""
        u = []
        for r in rs:
            if r._gen is None:
                r._gen = torch.Generator()
                r._gen.manual_seed(int(r.seed) % (2 ** 63))
            u.append(float(torch.rand((), generator=r._gen)))
        dev = logits.device
        temp = torch.tensor([r.temperature for r in rs], device=dev)[:, None]
        top_p = torch.tensor([r.top_p if r.top_p < 1.0 else 2.0 for r in rs], device=dev)[:, None]
        p = torch.softmax(logits.float() / temp, dim=-1)
        sp, si = p.sort(dim=-1, descending=True)
        sp = sp * ((sp.cumsum(-1) - sp) < top_p)
        cdf = sp.cumsum(-1)
        thr = torch.tensor(u, device=dev)[:, None] * cdf[:, -1:]
        j = torch.searchsorted(cdf, thr, right=True).clamp(max=cdf.shape[-1] - 1)
        return si.gather(1, j)[:, 0].tolist()

    def _sample(self, r: Request, logits: torch.Tensor) -> int:
        if r._gen is None:
            r._gen = torch.Generator(device=self.device)
            r._gen.manual_seed(int(r.seed) % (2 ** 63))
        p = torch.softmax(logits.float() / r.temperature, dim=-1)
        if r.top_p < 1.0:
            sp, si = p.sort(descending=True)
            keep = (sp.cumsum(-1) - sp) < r.top_p
            sp = sp * keep
            j = torch.multinomial(sp / sp.sum(), 1, generator=r._gen)
            return int(si[j])
        return int(torch.multinomial(p, 1, generator=r._gen))

    # decode tick
    def _plan(self, r: Request) -> Tuple[List[int], str]:
        """Tokens to feed this pass (pending token + draft candidates) and the charge kind."""
        t = len(r.out)
        room = max(0, r.max_new - t - 1)
        if r.mode == "replay" and r.on_draft and t < len(r.draft):
            m = min(self.W - 1, len(r.draft) - t, room)
            return [r.out[-1]] + list(r.draft[t:t + m]), "draft_checked"
        if r.mode == "pld":
            seq = np.asarray(r.prompt_ids + r.out, dtype=np.int64)
            c = prompt_lookup(seq, self.cfg.pld_max_ngram, self.cfg.pld_num_pred)[:room]
            if c:
                return [r.out[-1]] + c, "draft_checked"
        return [r.out[-1]], "decoded"

    def _decode_tick(self, done: List[Request]) -> None:
        plans: Dict[int, Tuple[List[int], str]] = {}
        for r in self.slots:
            if r is None or r.finish:
                continue
            plans[r.slot] = self._plan(r)
        if self.hybrid:
            self._decode_tick_hybrid(plans, done)
            return
        self._compact(plans)
        groups: Dict[int, List[int]] = {}
        for s in sorted(plans):
            groups.setdefault(self._rank(s, plans), []).append(s)
        for rank in sorted(groups):
            self._run_group(groups[rank], plans, done)

    def _rank(self, s: int, plans) -> int:
        """Pass group: 0 = replay blocks, 1 = PLD blocks, 2 = one token."""
        if s not in plans:
            return 3
        if len(plans[s][0]) == 1:
            return 2
        return 0 if self.slots[s].mode == "replay" else 1

    def _compact(self, plans) -> None:
        """Keep each multi-token group contiguous at the low end of the slot range (KV swaps),
        so a pass never processes rows of another group as W-wide padding."""
        lo = 0
        for rank in (0, 1):
            members = [s for s in plans if self._rank(s, plans) == rank]
            if not members:
                continue
            k = lo + len(members)
            outside = sorted(s for s in members if s >= k)
            inside_other = [t for t in range(lo, k) if self._rank(t, plans) != rank]
            for s, t in zip(outside, inside_other):
                self._swap(s, t, plans)
            lo = k

    def _swap(self, a: int, b: int, plans) -> None:
        ra, rb = self.slots[a], self.slots[b]
        self.cache.swap(a, ra.cursor if ra else 0, b, rb.cursor if rb else 0)
        self.slots[a], self.slots[b] = rb, ra
        if ra is not None:
            ra.slot = b
        if rb is not None:
            rb.slot = a
        pa, pb = plans.pop(a, None), plans.pop(b, None)
        if pa is not None:
            plans[b] = pa
        if pb is not None:
            plans[a] = pb
        self.stats["migrations"] += 1

    # hybrid (recurrent-layer) models
    def _decode_tick_hybrid(self, plans, done: List[Request]) -> None:
        """Rows are batched only with rows of the SAME feed length (padding would advance recurrent
        states), and exactly those rows are processed (gathered), so no other slot's state moves."""
        groups: Dict[int, List[int]] = {}
        for s in sorted(plans):
            r = self.slots[s]
            feed, kind = plans[s]
            while len(feed) > 1 and not r.ledger.affordable(self.flops.tflops(r.cursor, len(feed)), self.pass_id + 1):
                feed = feed[:-1]
            if len(feed) == 1 and r.mode == "pld":
                kind = "decoded"
            plans[s] = (feed, kind)
            groups.setdefault(len(feed), []).append(s)
        for L in sorted(groups, reverse=True):
            self._run_rows_hybrid(groups[L], plans, done)

    @torch.no_grad()
    def _run_rows_hybrid(self, group: List[int], plans, done: List[Request]) -> None:
        self.pass_id += 1
        live = []
        for s in group:
            r = self.slots[s]
            feed, kind = plans[s]
            if not self._charge(r, len(feed), kind):
                self._finish(r, "budget", done)
                continue
            r.stats["passes"] += 1
            live.append(s)
        if not live:
            return
        L = len(plans[live[0]][0])
        n = len(live)
        ids = torch.tensor([plans[s][0] for s in live], dtype=torch.long, device=self.device)
        cur = torch.tensor([self.slots[s].cursor for s in live], dtype=torch.long, device=self.device)
        pos = cur[:, None] + torch.arange(L, device=self.device)[None]
        t_len = max(self.slots[s].cursor for s in live) + L
        keys = torch.arange(t_len, device=self.device)
        mask = (keys[None, None, None, :] <= pos[:, None, :, None])
        self.cache.gather(torch.tensor(live, device=self.device), pos, t_len)
        h = self._forward(ids, self._masks(mask, pos, keys), pos, L)
        self.stats["passes"] += 1
        self.stats["real_positions"] += n * L
        logits = self._logits(h.reshape(n * L, -1))
        preds = logits.argmax(-1).view(n, L).tolist()
        ks = [k for k, s in enumerate(live) if self.slots[s].mode == "sample" and self.slots[s].temperature > 0]
        sampled = dict(zip(ks, self._sample_rows([self.slots[live[k]] for k in ks], logits, [k * L for k in ks])))
        keep, refeed = [], []
        for k, s in enumerate(live):
            r = self.slots[s]
            feed = plans[s][0]
            if k in sampled:
                p = [sampled[k]]
            else:
                p = preds[k]
            m = L - 1
            a = 0
            while a < m and p[a] == feed[a + 1]:
                a += 1
            old_cursor = r.cursor
            self._accept(r, plans[s], p, done)
            keep.append(a == m)                      # every fed token accepted: the new state is valid
            if a < m and not r.finish:
                refeed.append((s, feed[:a + 1], old_cursor))
        self.cache.commit(keep)
        if refeed:
            self._refeed(refeed, done)

    @torch.no_grad()
    def _refeed(self, items, done: List[Request]) -> None:
        """Rollback for recurrent layers: the stored state is the pre-pass state (not committed);
        re-feed only the accepted tokens to advance it exactly.  These positions are charged
        (kind ``draft_recompute``)."""
        groups: Dict[int, list] = {}
        for it in items:
            groups.setdefault(len(it[1]), []).append(it)
        for L, its in groups.items():
            self.pass_id += 1
            live = []
            for s, toks, c0 in its:
                r = self.slots[s]
                c = self.flops.tflops(c0, L)
                if not r.ledger.affordable(c, self.pass_id):
                    self._finish(r, "budget", done)
                    continue
                before = r.ledger.spent
                r.ledger.charge(c, self.pass_id, r.role, "draft_recompute", L)
                r.spent += r.ledger.spent - before
                r.stats["draft_recompute"] += L
                r.stats["passes"] += 1
                live.append((s, toks, c0))
            if not live:
                continue
            ids = torch.tensor([t for _, t, _ in live], dtype=torch.long, device=self.device)
            cur = torch.tensor([c0 for _, _, c0 in live], dtype=torch.long, device=self.device)
            pos = cur[:, None] + torch.arange(L, device=self.device)[None]
            t_len = max(c0 for _, _, c0 in live) + L
            keys = torch.arange(t_len, device=self.device)
            mask = (keys[None, None, None, :] <= pos[:, None, :, None])
            self.cache.gather(torch.tensor([s for s, _, _ in live], device=self.device), pos, t_len)
            self._forward(ids, self._masks(mask, pos, keys), pos, L)
            self.cache.commit([True] * len(live))
            self.stats["passes"] += 1
            self.stats["real_positions"] += len(live) * L

    @torch.no_grad()
    def _run_group(self, group: List[int], plans, done: List[Request]) -> None:
        self.pass_id += 1
        live = []
        for s in group:
            r = self.slots[s]
            feed, kind = plans[s]
            # the allowance is checked in row order, so re-shorten against the live ledger
            while len(feed) > 1 and not r.ledger.affordable(self.flops.tflops(r.cursor, len(feed)), self.pass_id):
                feed = feed[:-1]
            if len(feed) == 1 and r.mode == "pld":
                kind = "decoded"
            plans[s] = (feed, kind)
            if not self._charge(r, len(feed), kind):
                self._finish(r, "budget", done)
                continue
            r.stats["passes"] += 1
            live.append(s)
        if not live:
            return
        lo, hi = min(live), max(live) + 1
        width = max(len(plans[s][0]) for s in live)
        n = hi - lo
        ids = torch.full((n, width), self.pad_id, dtype=torch.long)
        cur = torch.zeros(n, dtype=torch.long)
        live_set = set(live)
        for k in range(n):
            r = self.slots[lo + k]
            if r is not None:
                cur[k] = r.cursor
            if lo + k in live_set:
                f = plans[lo + k][0]
                ids[k, :len(f)] = torch.tensor(f)
        t_len = min(int(cur.max()) + width, self.max_len)    # host-side: max(pos) + 1 after the clamp
        cur = cur.to(self.device)
        ids = ids.to(self.device)
        pos = cur[:, None] + torch.arange(width, device=self.device)[None]
        pos = pos.clamp(max=self.max_len - 1)
        keys = torch.arange(t_len, device=self.device)
        mask = (keys[None, None, None, :] <= pos[:, None, :, None])
        self.cache.view(lo, hi, pos, t_len)
        h = self._forward(ids, self._masks(mask, pos, keys), pos, width)
        self.stats["passes"] += 1
        real = sum(len(plans[s][0]) for s in live)
        self.stats["real_positions"] += real
        self.stats["padded_positions"] += n * width - real
        # logits only where they are needed: every fed position of a multi-token row
        rows_idx, cols_idx, owner = [], [], []
        for s in live:
            L = len(plans[s][0])
            rows_idx += [s - lo] * L
            cols_idx += list(range(L))
            owner.append((s, L))
        logits = self._logits(h[torch.tensor(rows_idx, device=self.device),
                                torch.tensor(cols_idx, device=self.device)])
        preds = logits.argmax(-1).tolist()           # exact: bf16 → fp32 is monotone
        starts, k = [], 0
        for s, L in owner:
            starts.append(k)
            k += L
        ks = [i for i, (s, _) in enumerate(owner) if self.slots[s].mode == "sample" and self.slots[s].temperature > 0]
        sampled = dict(zip(ks, self._sample_rows([self.slots[owner[i][0]] for i in ks], logits, [starts[i] for i in ks])))
        k = 0
        for i, (s, L) in enumerate(owner):
            r = self.slots[s]
            if i in sampled:
                p = [sampled[i]]
            else:
                p = preds[k:k + L]
            k += L
            self._accept(r, plans[s], p, done)

    def _accept(self, r: Request, plan, preds: List[int], done: List[Request]) -> None:
        feed, kind = plan
        m = len(feed) - 1                     # draft candidates fed after the pending token
        a = 0
        while a < m and preds[a] == feed[a + 1]:
            a += 1
        new = feed[1:1 + a] + [preds[a]]
        t = len(r.out)
        r.cursor += 1 + a
        if m:
            r.stats["draft_discarded"] += m - a
            r.ledger.note("draft_discarded", m - a)
        if r.mode == "replay" and r.on_draft:
            for x in new:
                if t < len(r.draft) and x == r.draft[t]:
                    r.stats["draft_accepted"] += 1
                    r.ledger.note("draft_accepted", 1)
                    t += 1
                else:
                    r.on_draft = False
                    break
        elif r.mode == "pld":
            r.stats["draft_accepted"] += a
            r.ledger.note("draft_accepted", a)
        r.out.extend(new)
        self._check_end(r, done)


__all__ = ["Engine", "Request", "MODES", "grouped_attention", "prompt_lookup"]
