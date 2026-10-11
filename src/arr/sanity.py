"""
Model-free and tiny-model checks of the decoder and the allocation rule
(``run_arr.py sanity``, also used by tests/test_arr.py).  CPU, fp32, random
2-layer model — no download.

  * exactness (Proposition 1): replay revision == ordinary greedy decoding (exact tokens), for
    drafts that match fully, partly, or not at all, several block sizes, batched
  * fixed-point detection: a draft equal to the greedy output is detected as a fixed point
    (identical incl. termination) and costs fewer serial passes than ordinary decoding
  * PLD decoding == ordinary greedy decoding
  * budget: a block is shortened / a request ends with finish='budget' and the
    budget is never exceeded; every processed position is charged
  * grader / answer extraction checks
"""

from __future__ import annotations

import dataclasses
from typing import List

import torch

from .config import ARRConfig
from .engine import Engine, Request
from .ledger import Ledger
from .voting import Grader, sanity_check_grader

INF = float("inf")


def tiny_engine(block_size: int = 4, rows: int = 6, vocab: int = 64, eos: int = 1, seed: int = 0) -> Engine:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    c = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                    eos_token_id=eos, bos_token_id=eos, pad_token_id=0)
    model = Qwen2ForCausalLM(c).float()
    with torch.no_grad():                      # EOS reachable at varied lengths (4..26 tokens) for most prompts
        model.lm_head.weight[eos] = model.model.norm.weight.sign() * 0.11
    model.generation_config.eos_token_id = eos
    cfg = ARRConfig(model_name="tiny", dtype="float32", max_input_tokens=64, max_new_tokens=40,
                    block_size=block_size, engine_max_rows=rows, pld_num_pred=5)
    return Engine(cfg, device="cpu", model=model, tokenizer=None)


def tiny_mistral_engine(block_size: int = 4, rows: int = 6, vocab: int = 64, eos: int = 1) -> Engine:
    """Mistral architecture with interleaved sliding layers whose window exceeds max_len (as Ministral-8B)."""
    from transformers import MistralConfig, MistralForCausalLM
    torch.manual_seed(0)
    c = MistralConfig(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, sliding_window=4096,
                      eos_token_id=eos, bos_token_id=eos, pad_token_id=0)
    model = MistralForCausalLM(c).float()
    with torch.no_grad():
        model.lm_head.weight[eos] = model.model.norm.weight.sign() * 0.11
    model.generation_config.eos_token_id = eos
    cfg = ARRConfig(model_name="tiny-mistral", dtype="float32", max_input_tokens=64, max_new_tokens=40,
                    block_size=block_size, engine_max_rows=rows, pld_num_pred=5)
    return Engine(cfg, device="cpu", model=model, tokenizer=None)


def _greedy(e: Engine, prompts: List[List[int]]) -> List[Request]:
    return e.run([Request(p, e.cfg.max_new_tokens, "greedy", Ledger(INF)) for p in prompts])


def hf_reference_check(e: Engine, prompts: List[List[int]], ref: List[Request]) -> List[str]:
    """The engine's greedy decoding must equal transformers' own generate() with eager attention
    (validates the slot cache, per-row masks, positions and the grouped attention)."""
    import copy
    m = copy.deepcopy(e.model)
    m.config._attn_implementation = "eager"
    fails = []
    eos = next(iter(e.eos))
    for k, (p, r) in enumerate(zip(prompts, ref)):
        out = m.generate(torch.tensor([p]), attention_mask=torch.ones(1, len(p), dtype=torch.long),
                         max_new_tokens=e.cfg.max_new_tokens, do_sample=False, eos_token_id=eos,
                         pad_token_id=0)[0, len(p):].tolist()
        if eos in out:
            out = out[:out.index(eos) + 1]
        if out != r.out:
            fails.append(f"row {k}: engine greedy {r.out} != transformers generate {out}")
    return fails


def engine_checks(block_sizes=(1, 4, 7)) -> List[str]:
    fails: List[str] = []
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(2, 64, (n,), generator=g).tolist() for n in (5, 9, 13, 17, 6, 11)]
    n_fixed = n_changed = 0
    for W in block_sizes:
        e = tiny_engine(block_size=W)
        ref = _greedy(e, prompts)
        if W == block_sizes[0]:
            fails += hf_reference_check(e, prompts, ref)
            if sum(r.complete for r in ref) < 3 or all(r.complete for r in ref):
                fails.append(f"tiny model does not exercise both EOS and length termination: {[r.finish for r in ref]}")
        # drafts: exact greedy output (fixed point), a copy with one token changed in the middle,
        # a draft wrong from the first token, and a draft that is a strict prefix-extension
        drafts = []
        for k, r in enumerate(ref):
            y = list(r.out)
            if k % 3 == 0:
                drafts.append(y)
            elif k % 3 == 1:
                z = list(y)
                z[len(z) // 2] = (z[len(z) // 2] % 60) + 2
                drafts.append(z)
            else:
                drafts.append([(t % 60) + 2 for t in y[:3]] + y[3:])
        rep = e.run([Request(p, e.cfg.max_new_tokens, "replay", Ledger(INF), draft=d) for p, d in zip(prompts, drafts)])
        pld = e.run([Request(p, e.cfg.max_new_tokens, "pld", Ledger(INF)) for p in prompts])
        for k, (a, b, c) in enumerate(zip(ref, rep, pld)):
            if a.out != b.out or a.finish != b.finish:
                fails.append(f"W={W} row {k}: replay {b.out} != greedy {a.out}")
            if a.out != c.out:
                fails.append(f"W={W} row {k}: pld {c.out} != greedy {a.out}")
            if k % 3 == 0 and a.complete and not b.identical:
                fails.append(f"W={W} row {k}: exact draft not detected as a fixed point")
            if k % 3 != 0 and b.identical:
                fails.append(f"W={W} row {k}: a changed draft was reported as a fixed point")
            n_fixed += int(b.identical)
            n_changed += int(b.complete and not b.identical)
            if k % 3 == 0 and a.complete and W > 1 and len(a.out) > W and b.stats["passes"] >= a.stats["passes"]:
                fails.append(f"W={W} row {k}: full replay used {b.stats['passes']} passes vs {a.stats['passes']} ordinary")
            if b.stats["draft_discarded"] > max(0, W - 1):
                fails.append(f"W={W} row {k}: more than one failed block ({b.stats['draft_discarded']} discarded)")
            pos = b.stats["prefill"] + b.stats["draft_checked"] + b.stats["decoded"]
            if pos != b.ledger.counts["prefill"] + b.ledger.counts["draft_checked"] + b.ledger.counts["decoded"]:
                fails.append(f"W={W} row {k}: ledger position counts disagree")
    # every mode mixed in one engine, more requests than slots (admission churn, slot swaps,
    # separate replay / PLD / single-token passes) must equal each request run alone
    e = tiny_engine(block_size=4, rows=4)
    ref = _greedy(e, prompts)
    def mk():
        out = []
        for k, (p, r) in enumerate(zip(prompts, ref)):
            y = list(r.out)
            out.append(Request(p, e.cfg.max_new_tokens, "replay", Ledger(INF), draft=y if k % 2 else y[:2] + [5] + y[3:]))
            out.append(Request(p, e.cfg.max_new_tokens, "pld", Ledger(INF)))
            out.append(Request(p, e.cfg.max_new_tokens, "sample", Ledger(INF), temperature=0.8, top_p=0.9, seed=k))
            out.append(Request(p, e.cfg.max_new_tokens, "greedy", Ledger(INF)))
        return out
    for sampler in ("per_row", "batched"):
        e.cfg.sampler = sampler
        mixed = e.run(mk())
        alone = [e.run([q])[0] for q in mk()]
        for k, (a, b) in enumerate(zip(mixed, alone)):
            if a.out != b.out or a.finish != b.finish:
                fails.append(f"[{sampler} sampler] mixed-batch request {k} ({a.mode}) differs from its isolated run")
    e.cfg.sampler = "per_row"
    fails += sampler_checks(e)
    if e.stats["migrations"] == 0:
        fails.append("mixed-batch check did not exercise slot swaps")
    # a second architecture (Mistral) must also match transformers.generate and replay == greedy
    em = tiny_mistral_engine()
    refm = _greedy(em, prompts)
    fails += [f"[mistral] {x}" for x in hf_reference_check(em, prompts, refm)]
    repm = em.run([Request(p, em.cfg.max_new_tokens, "replay", Ledger(INF), draft=list(r.out)) for p, r in zip(prompts, refm)])
    for k, (a, b) in enumerate(zip(refm, repm)):
        if a.out != b.out:
            fails.append(f"[mistral] row {k}: replay != greedy")
    if not n_fixed or not n_changed:
        fails.append(f"checks did not exercise fixed points ({n_fixed}) and changed revisions ({n_changed})")
    # budget: a small allowance ends requests with 'budget' and is never exceeded
    e = tiny_engine(block_size=4)
    ref = _greedy(e, prompts)
    for r in ref:
        full = r.ledger.tflops
        led = Ledger(full * 0.6, "tflops")
        b = e.run([Request(r.prompt_ids, e.cfg.max_new_tokens, "replay", led, draft=list(r.out))])[0]
        if led.spent > led.budget + 1e-12:
            fails.append("budget exceeded")
        if r.complete and b.finish != "budget":
            fails.append(f"expected finish=budget under 60% allowance, got {b.finish}")
        if b.out != r.out[:len(b.out)]:
            fails.append("budget-truncated replay is not a prefix of the greedy output")
    return fails


def sampler_checks(e: Engine) -> List[str]:
    """The batched sampler draws only from the top-p nucleus, follows the renormalised nucleus
    probabilities, and is reproducible per request seed."""
    fails: List[str] = []
    logits = torch.full((2, 64), -30.0, device=e.device)
    logits[:, 3], logits[:, 7], logits[:, 9] = 2.0, 1.0, 0.0      # p ≈ .665 / .245 / .090
    def draws(seed: int, n: int = 400) -> List[int]:
        r = Request([2], 4, "sample", Ledger(INF), temperature=1.0, top_p=0.8, seed=seed)
        return [e._sample_batched([r], logits[:1])[0] for _ in range(n)]
    d = draws(0)
    if set(d) - {3, 7}:
        fails.append(f"batched sampler left the top-p nucleus: {sorted(set(d))}")
    share = d.count(3) / len(d)
    if not 0.66 < share < 0.80:                                     # .665 / (.665 + .245) = .73
        fails.append(f"batched sampler nucleus share {share:.2f} far from 0.73")
    if draws(0) != d or draws(1) == d:
        fails.append("batched sampler is not reproducible per request seed")
    return fails


def tiny_new_arch_engine(kind: str):
    """Tiny random models of the newer architectures (transformers >= 5): ``qwen3_5`` (hybrid: linear-
    attention layers with recurrent state + full attention) and ``gemma4`` (sliding window 8, KV-shared
    layers, per-layer input embeddings, two head sizes, final-logit softcapping)."""
    torch.manual_seed(0)
    if kind == "qwen3_5":
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        c = Qwen3_5TextConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                              layer_types=["linear_attention"] * 3 + ["full_attention"], linear_num_key_heads=2,
                              linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                              linear_conv_kernel_dim=4, max_position_embeddings=512, eos_token_id=1,
                              bos_token_id=1, pad_token_id=0)
        m = Qwen3_5ForCausalLM(c).float().eval()
        # EOS row along the mean final hidden state (its norm weights start at zero, so the sign trick
        # used for the other tiny models gives no EOS): outputs then end at varied lengths
        g = torch.Generator().manual_seed(1)
        ps = [torch.randint(2, 64, (n,), generator=g).tolist() for n in (5, 9, 13, 17, 6, 11)]
        with torch.no_grad():
            hs = torch.cat([m.model(torch.tensor([p])).last_hidden_state[0] for p in ps])
            d = hs.mean(0) / hs.mean(0).norm()
            m.lm_head.weight[1] = d * 0.7 * m.lm_head.weight.norm(dim=1).mean()
        bias = None
    else:
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig
        c = Gemma4TextConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
                             layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
                             sliding_window=8, num_kv_shared_layers=2, hidden_size_per_layer_input=8,
                             vocab_size_per_layer_input=64, max_position_embeddings=512, final_logit_softcapping=30.0,
                             eos_token_id=1, bos_token_id=1, pad_token_id=0)
        m = Gemma4ForCausalLM(c).float()
        bias = 0.11
    if bias is not None:
        with torch.no_grad():
            m.lm_head.weight[1] = m.model.norm.weight.sign() * bias
    m.generation_config.eos_token_id = 1
    cfg = ARRConfig(model_name=f"tiny-{kind}", dtype="float32", max_input_tokens=64, max_new_tokens=40,
                    block_size=4, engine_max_rows=4, pld_num_pred=5)
    return Engine(cfg, device="cpu", model=m, tokenizer=None)


class _reference_linear_kernels:
    """Use transformers' reference PyTorch implementations of the linear-attention kernels while the
    CPU checks run (optimized kernels such as flash-linear-attention are GPU-only); restore after."""

    NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
             "causal_conv1d_update")

    def __enter__(self):
        import inspect
        self.saved = []
        try:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
        except ImportError:
            return self
        for n in self.NAMES:
            fn = getattr(mod, n, None)
            if fn is not None:
                self.saved.append((mod, n, fn))
                setattr(mod, n, inspect.unwrap(fn))
        return self

    def __exit__(self, *exc):
        for mod, n, fn in self.saved:
            setattr(mod, n, fn)
        return False


def new_arch_checks() -> List[str]:
    with _reference_linear_kernels():
        return _new_arch_checks()


def _new_arch_checks() -> List[str]:
    """engine == transformers.generate, replay == greedy (incl. mid-block rejections → rollback of the
    recurrent state), PLD == greedy, mixed batches, fixed points — for each newer architecture available."""
    fails: List[str] = []
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(2, 64, (n,), generator=g).tolist() for n in (5, 9, 13, 17, 6, 11)]
    for kind in ("qwen3_5", "gemma4"):
        try:
            e = tiny_new_arch_engine(kind)
        except ImportError:
            continue                                   # transformers < 5: architecture not available
        ref = _greedy(e, prompts)
        fails += [f"[{kind}] {x}" for x in hf_reference_check(e, prompts, ref)]
        reqs, refs = [], []
        for p, r in zip(prompts, ref):
            for j in (None, 2, 6, 10, 18, 22, 30):
                y = list(r.out)
                if j is not None and j < len(y):
                    y[j] = (y[j] % 60) + 2
                reqs.append(Request(p, e.cfg.max_new_tokens, "replay", Ledger(INF), draft=y))
                refs.append(r)
        rep = e.run(reqs)
        for a, b in zip(refs, rep):
            if a.out != b.out or a.finish != b.finish:
                fails.append(f"[{kind}] replay != greedy")
                break
        pld = e.run([Request(p, e.cfg.max_new_tokens, "pld", Ledger(INF)) for p in prompts])
        if any(a.out != b.out for a, b in zip(ref, pld)):
            fails.append(f"[{kind}] pld != greedy")
        if not any(x.identical for x in rep):
            fails.append(f"[{kind}] no fixed point exercised")
        if e.hybrid and not sum(x.stats["draft_recompute"] for x in rep):
            fails.append(f"[{kind}] recurrent-state rollback not exercised")
        # sampled rows next to replay rows (partial commits) must equal their isolated runs
        e.cfg.sampler = "batched"
        def mk():
            out = []
            for k, (p, r) in enumerate(zip(prompts, ref)):
                y = list(r.out)
                out.append(Request(p, e.cfg.max_new_tokens, "sample", Ledger(INF), temperature=0.8, top_p=0.9, seed=k))
                out.append(Request(p, e.cfg.max_new_tokens, "replay", Ledger(INF), draft=y[:2] + [5] + y[3:]))
            return out
        mixed = e.run(mk())
        alone = [e.run([q])[0] for q in mk()]
        if any(a.out != b.out or a.finish != b.finish for a, b in zip(mixed, alone)):
            fails.append(f"[{kind}] batched-sampler mixed batch differs from isolated runs")
        e.cfg.sampler = "per_row"
    return fails


def run_sanity(datasets=("math500",)) -> List[str]:
    fails: List[str] = []
    for ds in dict.fromkeys(["math500", *datasets]):
        fails += [f"[{ds}] {x}" for x in sanity_check_grader(Grader(ds))]
    fails += engine_checks()
    fails += new_arch_checks()
    from .tasks import sanity_check_tasks
    fails += [f"[tasks] {x}" for x in sanity_check_tasks()]
    return fails


__all__ = ["tiny_engine", "engine_checks", "run_sanity"]
