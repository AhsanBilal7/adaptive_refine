"""
cimr_inference.py
=================
Cluster-Informed Meta-Reasoning (CIMR) — answer-clustered, per-cluster
judge variant.

Algorithm (revised)
-------------------
  Phase 1: GENERATE  — Sample N traces from π(·|q) at temperature T.
  Phase 2: SCORE     — Compute PRM verifier score V(x_i) for every trace
                        (kept for downstream metrics; not used to form clusters).
  Phase 3: ANSWER-CLUSTER — Group the N traces by their EXTRACTED FINAL ANSWER.
                            All traces with the same answer form one cluster.
                            Empty / unparsable answers form a single
                            "(no answer)" cluster (or are dropped).
                            => K clusters total, where K = number of distinct
                               extracted answers.
  Phase 4: PER-CLUSTER JUDGE — For each cluster c ∈ {1,…,K}, build a judge
                                prompt containing
                                  • the original problem,
                                  • the reasoning content selected by the
                                    chosen variant (see below), and
                                  • the answers proposed by the OTHER clusters.
                                Run the judge on all K prompts (batched).
                                This produces K cluster-responses
                                  R_1, …, R_K, each ending in \\boxed{a_k}.
  Phase 5: PRM-SCORE THE JUDGE RESPONSES — Run the verifier on every R_k to
                                obtain S_1, …, S_K.
  Phase 6: SELECT —
                    cimr_prm_score        := a_{argmax_k S_k}
                    cimr_majority_voting  := mode of (a_1,…,a_K), tie-broken
                                              by sum-of-PRM within each modal
                                              answer.

Variant semantics (NEW)
-----------------------
The `variant` string selects what reasoning content the judge sees per
cluster — there is no longer a per-trace character budget:

  "full"        : every trace in the cluster, in full.
  "concise"     : only the single trace with the highest PRM score in the
                  cluster.
  "answer_only" : no reasoning at all — the judge sees only the candidate
                  answer and the answers from the other clusters.

The legacy parameters `max_trace_chars` / `CIMR_MAX_TRACE_CHARS` and
`CIMR_TAU` are no longer used; they are accepted for API compatibility and
silently ignored.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Answer extraction from judge response
# =============================================================================

def _extract_boxed(text: str) -> str:
    """
    Extract the content of the LAST \\boxed{...} in text, handling nested
    braces.  Falls back to the last number/expression in the text if no
    \\boxed{} is found.
    """
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return nums[-1] if nums else ""

    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else text[start:].strip()


def _extract_answer(judge_response: str, ev, data_name: str) -> str:
    """
    Multi-strategy answer extraction from judge output.

    Priority:
      1. \\boxed{} brace-matching (most reliable for math)
      2. Evaluator's own extraction (dataset-specific)
      3. Last number/expression in the response
    """
    ans = _extract_boxed(judge_response)
    if ans:
        return ans

    if ev is not None:
        try:
            from inference_methods import _extract_pred  # type: ignore
            ans = _extract_pred(judge_response, ev, data_name)
            if ans:
                return ans
        except Exception:
            pass

    nums = re.findall(r"-?\d+(?:\.\d+)?", judge_response.replace(",", ""))
    return nums[-1] if nums else ""


# =============================================================================
# Judge-response post-processing
# =============================================================================
#
# Some math judges (Qwen2.5-Math-7B-Instruct in particular) occasionally fall
# into degenerate decoding loops on long contexts:  the same sentence — or
# even the same algebraic chunk like "(xyz)^4 \cdot (xyz)^{-1}" — gets
# repeated dozens of times before the budget is exhausted, with no \boxed{}
# ever produced.  Two cheap mitigations:
#
#   1. Detect such responses post-hoc and treat them as candidates for retry
#      with a stronger repetition_penalty.
#   2. Truncate at the first \boxed{...} line so any rambling AFTER an answer
#      is dropped, which both saves tokens downstream and makes
#      _extract_answer robust.
#
# These are best-effort heuristics; they do not change correctness when the
# response is well-formed.

_BOXED_RE = re.compile(r"\\boxed\{")


def _truncate_at_first_boxed(text: str) -> str:
    """
    If the response contains a \\boxed{...} expression, return the substring
    that ends just after the first such expression's closing brace.  This
    discards any post-answer rambling.  If no \\boxed is found, return the
    text unchanged.
    """
    if not text:
        return text
    m = _BOXED_RE.search(text)
    if not m:
        return text

    start = m.end()
    depth = 1
    i = start
    n = len(text)
    while i < n and depth > 0:
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        i += 1

    if depth != 0:
        # Unbalanced — just keep what we have.
        return text

    return text[:i]


def _is_degenerate_repetition(text: str) -> bool:
    """
    Cheap heuristic: a response is "degenerate" if the same line is repeated
    a large number of times, OR the same short phrase appears very many times.

    Used to decide whether a retry with stronger anti-repetition pressure is
    warranted.  Conservative thresholds — well-formed responses pass.
    """
    if not text or len(text) < 200:
        return False

    # Strip whitespace-only lines, normalise spacing
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip()
    ]
    if not lines:
        return False

    # 1) Same non-trivial line repeated >= 5 times
    from collections import Counter
    line_counts = Counter(ln for ln in lines if len(ln) > 20)
    if line_counts:
        most_common_line, n = line_counts.most_common(1)[0]
        if n >= 5:
            return True

    # 2) Repeated short n-gram, OR very high zlib-compressibility — both
    #    catch the "(xyz)^4 \cdot (xyz)^{-1}" style of in-line loop, even
    #    when the period doesn't align nicely with window steps.
    s = " ".join(lines)
    if len(s) >= 200:
        # Compression ratio: any text where a long stretch is the same phrase
        # repeated many times compresses very small.  A normal math response
        # of similar length compresses to about 35-55% of its original size.
        try:
            import zlib
            compressed = len(zlib.compress(s.encode("utf-8")))
            ratio = compressed / max(len(s), 1)
            if ratio < 0.20:
                return True
        except Exception:
            pass

        # Fallback: rolling-window n-gram count, relaxed thresholds.
        best = 0
        seen = {}
        step = 4
        for i in range(0, len(s) - 24, step):
            chunk = s[i : i + 24]
            seen[chunk] = seen.get(chunk, 0) + 1
            if seen[chunk] > best:
                best = seen[chunk]
        if best >= 8 and len(s) >= 400:
            return True

    return False


def _judge_generate_with_retry(
    judge_model,
    prompts: List[str],
    max_new_tokens: int,
) -> List[str]:
    """
    Call the judge model once on a batch.  If any response looks
    degenerate (long, no \\boxed, repeated phrase), re-run JUST those
    prompts with a stronger repetition_penalty.  Always truncate at the
    first \\boxed{...} so that any post-answer rambling is discarded
    before downstream PRM scoring.

    The HuggingFace generate() kwargs passed below (do_sample=False,
    repetition_penalty, no_repeat_ngram_size) are forwarded by
    HFModelWrapper._generate_batch_raw via **kwargs.  If the wrapper
    does not accept them, this falls back gracefully to the legacy call.
    """
    # First pass — greedy with mild repetition penalty.
    primary_kwargs = dict(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.10,
        no_repeat_ngram_size=12,
    )
    try:
        responses = judge_model._generate_batch_raw(prompts, **primary_kwargs)
    except TypeError:
        # Wrapper rejected one of the new kwargs — strip and retry once.
        responses = judge_model._generate_batch_raw(
            prompts,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    # Identify degenerate responses for retry.
    bad_idxs = [
        i for i, r in enumerate(responses)
        if _is_degenerate_repetition(r) and "\\boxed{" not in r
    ]

    if bad_idxs:
        logger.warning(
            f"[CIMR] {len(bad_idxs)}/{len(prompts)} judge responses look "
            f"degenerate; retrying with stronger repetition_penalty."
        )
        retry_prompts = [prompts[i] for i in bad_idxs]
        retry_kwargs = dict(
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.20,
            no_repeat_ngram_size=8,
        )
        try:
            retry_responses = judge_model._generate_batch_raw(
                retry_prompts, **retry_kwargs
            )
        except TypeError:
            retry_responses = judge_model._generate_batch_raw(
                retry_prompts,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        for orig_i, new_r in zip(bad_idxs, retry_responses):
            # Keep retry only if it's strictly better (shorter and not
            # itself degenerate). Otherwise keep the original.
            if new_r and not _is_degenerate_repetition(new_r):
                responses[orig_i] = new_r

    # Truncate every response at first \boxed{...}
    return [_truncate_at_first_boxed(r) for r in responses]


# =============================================================================
# Prompt formatting — bypasses HFModelWrapper._build_prompt()
# =============================================================================

def _build_cimr_prompt_text(
    judge_model,            # HFModelWrapper instance
    system_content: str,
    user_content:   str,
) -> str:
    """
    Format a CIMR judge prompt using the model's chat template, injecting our
    own system + user content (rather than the math-solver default that
    HFModelWrapper._build_prompt() hardcodes).
    """
    tok = judge_model.tokenizer

    if getattr(tok, "chat_template", None):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(
                f"apply_chat_template failed ({e}); falling back to plain concat."
            )

    return f"{system_content}\n\n{user_content}"


# =============================================================================
# Answer-clustering: group traces by extracted final answer
# =============================================================================

def _normalize_answer_key(ans: str) -> str:
    """
    Normalise an extracted answer to a canonical string used as the cluster
    key.  We strip whitespace, drop trailing zeros on plain decimals, and
    collapse a few obvious latex variants.  Equality of the key defines
    cluster membership.

    NOTE: this is intentionally light-touch.  We do NOT mathematically equate
    "1/2" and "0.5"; the evaluator handles that at correctness-check time.
    Here we just want to group obvious string duplicates.
    """
    s = str(ans or "").strip()
    if not s:
        return ""

    s = s.replace(" ", "")
    s = s.replace("\\,", "").replace("\\!", "")

    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]

    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")

    return s


def cluster_traces_by_answer(
    traces:     Sequence[str],
    answers:    Sequence[str],
    prm_scores: Sequence[float],
    drop_empty_answer: bool = False,
) -> List[Dict[str, Any]]:
    """
    Group the N traces by their extracted final answer.

    Returns a list[dict] of clusters sorted by size descending, each with:
      answer        : canonical answer string (cluster key)
      raw_answer    : the first non-canonical occurrence (display)
      trace_idxs    : original indices of traces in this cluster
      traces        : trace strings (in trace_idxs order)
      prm_scores    : per-trace PRM scores (in trace_idxs order)
      size          : len(trace_idxs)
      max_prm       : max PRM in cluster
      mean_prm      : mean PRM in cluster
    """
    if not (len(traces) == len(answers) == len(prm_scores)):
        raise ValueError(
            "traces, answers, and prm_scores must have the same length"
        )

    groups: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for i, (tr, ans, pr) in enumerate(zip(traces, answers, prm_scores)):
        key = _normalize_answer_key(ans)

        if key == "" and drop_empty_answer:
            continue

        if key not in groups:
            groups[key] = dict(
                answer=key,
                raw_answer=str(ans).strip(),
                trace_idxs=[],
                traces=[],
                prm_scores=[],
            )
        groups[key]["trace_idxs"].append(i)
        groups[key]["traces"].append(tr)
        groups[key]["prm_scores"].append(float(pr))

    clusters = []
    for g in groups.values():
        scores = g["prm_scores"]
        clusters.append(
            dict(
                answer=g["answer"],
                raw_answer=g["raw_answer"],
                trace_idxs=g["trace_idxs"],
                traces=g["traces"],
                prm_scores=scores,
                size=len(scores),
                max_prm=float(np.max(scores)) if scores else 0.0,
                mean_prm=float(np.mean(scores)) if scores else 0.0,
            )
        )

    # Sort by size desc, then max_prm desc — deterministic, makes the
    # majority-vote tie-break behave intuitively.
    clusters.sort(key=lambda c: (c["size"], c["max_prm"]), reverse=True)
    return clusters


# =============================================================================
# Variant → trace-content selection
# =============================================================================
#
# Variant semantics (per-cluster judge prompt content):
#
#   "full"        : every trace in the cluster, in full (no truncation).
#   "concise"     : only the trace with the highest PRM score in the cluster.
#   "answer_only" : no traces at all — the judge sees only the candidate
#                   answer and the answers proposed by other clusters.
#
# The legacy CIMR_MAX_TRACE_CHARS / max_trace_chars knob is no longer used;
# it is accepted for API compatibility but ignored.

# Backwards-compat shim: some callers still import this name.  It now always
# returns a sentinel value (effectively "no truncation").
def _trace_budget_for_variant(variant: str, max_trace_chars: int) -> int:
    """Deprecated.  Kept only for API compatibility — no longer used."""
    return 10**9


# =============================================================================
# Final selection over the K judge responses
# =============================================================================

def _select_from_cluster_responses(
    cluster_answers:   Sequence[str],
    cluster_prm:       Sequence[float],
    cluster_responses: Sequence[str],
    cluster_meta:      Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Given K judge responses, each with an extracted answer and a PRM score,
    decide:

      pred_prm       : answer of the cluster-response with the highest PRM
      pred_majority  : modal answer across the K responses; ties broken by
                       higher summed-PRM, then by larger original cluster.

    Returns a dict with both predictions plus per-cluster diagnostic info.
    """
    K = len(cluster_answers)
    assert K == len(cluster_prm) == len(cluster_responses) == len(cluster_meta)

    if K == 0:
        return dict(
            pred_prm="",
            pred_majority="",
            best_idx=-1,
            majority_idx=-1,
            per_cluster=[],
        )

    arr_prm = np.asarray(cluster_prm, dtype=float)
    best_idx = int(np.argmax(arr_prm))
    pred_prm = str(cluster_answers[best_idx]).strip()

    # Majority over the K judge-extracted answers
    bucket: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for k in range(K):
        key = _normalize_answer_key(cluster_answers[k])
        if key not in bucket:
            bucket[key] = dict(
                key=key,
                display=str(cluster_answers[k]).strip(),
                count=0,
                prm_sum=0.0,
                size_sum=0,
                first_idx=k,
            )
        bucket[key]["count"]    += 1
        bucket[key]["prm_sum"]  += float(arr_prm[k])
        bucket[key]["size_sum"] += int(cluster_meta[k].get("size", 1))

    ranked = sorted(
        bucket.values(),
        key=lambda b: (b["count"], b["prm_sum"], b["size_sum"]),
        reverse=True,
    )
    winning = ranked[0]
    pred_majority = winning["display"]
    majority_idx = winning["first_idx"]

    per_cluster = []
    for k in range(K):
        per_cluster.append(dict(
            cluster_idx=k,
            cluster_answer=str(cluster_meta[k].get("answer", "")),
            cluster_size=int(cluster_meta[k].get("size", 0)),
            cluster_max_prm=float(cluster_meta[k].get("max_prm", 0.0)),
            cluster_mean_prm=float(cluster_meta[k].get("mean_prm", 0.0)),
            judge_extracted=str(cluster_answers[k]).strip(),
            judge_prm_score=float(arr_prm[k]),
        ))

    return dict(
        pred_prm=pred_prm,
        pred_majority=pred_majority,
        best_idx=best_idx,
        majority_idx=majority_idx,
        per_cluster=per_cluster,
    )


# =============================================================================
# Batched judge inference  (NEW: K prompts per question)
# =============================================================================

def _flatten_per_q_per_cluster(
    per_q_clusters: List[List[Dict[str, Any]]],
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    """Flatten Q×K_q nested clusters into (q,c) index pairs + cluster list."""
    flat_index: List[Tuple[int, int]] = []
    flat_clusters: List[Dict[str, Any]] = []
    for q, clusters in enumerate(per_q_clusters):
        for c, cl in enumerate(clusters):
            flat_index.append((q, c))
            flat_clusters.append(cl)
    return flat_index, flat_clusters


def cimr_judge_batch(
    questions:           List[str],
    traces_list:         List[List[str]],
    prm_scores_list:     List[List[float]],
    cluster_labels_list: List[List[int]],            # API-compat; IGNORED
    answers_list:        List[List[str]],
    judge_model,
    ev,
    data_name:           str,
    verifier             = None,                      # NEW: required for PRM
    variant:             str   = "full",
    max_trace_chars:     int   = 2500,
    max_judge_tokens:    int   = 1024,
    judge_batch_size:    int   = 4,
    fallback_alpha:      float = 0.5,                 # API-compat; IGNORED
    gold_answers:        Optional[List[Optional[str]]] = None,
    drop_empty_answer:   bool  = False,
) -> List[Dict[str, Any]]:
    """
    Run the answer-clustered, per-cluster CIMR judge for a batch of Q
    questions.  See module docstring for the full algorithm.

    Returns one dict per question (length Q) with keys:
      pred_prm                    : answer chosen by max-PRM across K responses
      pred_majority               : answer chosen by majority vote across K
      is_correct_prm              : bool|None
      is_correct_majority         : bool|None
      predicted_answer            : alias for pred_prm (legacy harness key)
      is_correct                  : alias for is_correct_prm (legacy)
      judge_response              : concatenation of all K judge responses
      n_clusters                  : K
      clusters_meta               : per-cluster summary list
      judge_responses_per_cluster : list of K raw judge responses
      judge_prm_scores            : list of K PRM-on-judge scores
      judge_extracted_answers     : list of K extracted answers
      approx_input_tokens         : sum of per-cluster prompt token estimates
      variant                     : echo of input
      fallback                    : True if a fallback path was used
      cimr_full_log               : NEW — comprehensive per-question record
                                    capturing every cluster, every trace
                                    inside it (raw + extracted answer + raw
                                    PRM + normalised PRM), the judge's raw
                                    response per cluster, the judge's
                                    extracted answer and PRM-on-response
                                    score per cluster, and the final
                                    aggregated predictions.  This is the
                                    structure persisted to
                                    cimr_results_<tag>.json.
    """
    from prompts_meta_judge import (  # type: ignore
        build_per_cluster_judge_prompt,
        PER_CLUSTER_SYSTEM_PROMPT,
        PER_CLUSTER_SYSTEM_PROMPT_ANSWER_ONLY,
    )

    sys_prompt_for_variant = (
        PER_CLUSTER_SYSTEM_PROMPT_ANSWER_ONLY
        if variant == "answer_only"
        else PER_CLUSTER_SYSTEM_PROMPT
    )

    Q = len(questions)
    if gold_answers is None:
        gold_answers = [None] * Q

    if verifier is None:
        logger.warning(
            "[CIMR] No verifier passed to cimr_judge_batch; cimr_prm_score "
            "will degrade to first-cluster selection. Pass `verifier=...` "
            "to enable PRM-on-judge-response scoring."
        )

    t_start = time.time()

    # ── Phase 3 (Answer-cluster every question) ──────────────────────────────
    per_q_clusters: List[List[Dict[str, Any]]] = []
    for i in range(Q):
        try:
            clusters = cluster_traces_by_answer(
                traces=traces_list[i],
                answers=answers_list[i],
                prm_scores=prm_scores_list[i],
                drop_empty_answer=drop_empty_answer,
            )
        except Exception as e:
            logger.error(f"[CIMR] Answer-cluster failed for q[{i}]: {e}")
            clusters = []
        per_q_clusters.append(clusters)

    # ── Phase 4a (Build all per-cluster judge prompts, flat) ─────────────────
    # The legacy `trace_budget` knob has been removed: the variant string now
    # controls trace content directly inside the prompt builder.
    flat_index, flat_clusters = _flatten_per_q_per_cluster(per_q_clusters)
    flat_prompts:       List[str] = []   # full chat-templated prompt sent to model
    flat_user_msgs:     List[str] = []   # user-message body only (for logging)
    flat_system_msgs:   List[str] = []   # system-message body (for logging)
    flat_input_tokens:  List[int] = []

    for (q_idx, c_idx), cluster in zip(flat_index, flat_clusters):
        siblings = per_q_clusters[q_idx]
        other_answers = [
            s["raw_answer"] for j, s in enumerate(siblings) if j != c_idx
        ]
        try:
            user_msg = build_per_cluster_judge_prompt(
                question              = questions[q_idx],
                cluster_answer        = cluster["raw_answer"],
                cluster_traces        = cluster["traces"],
                other_cluster_answers = other_answers,
                variant               = variant,
                cluster_prm_scores    = cluster["prm_scores"],
            )
            full_prompt = _build_cimr_prompt_text(
                judge_model, sys_prompt_for_variant, user_msg
            )
        except Exception as e:
            logger.error(
                f"[CIMR] Prompt build failed q[{q_idx}] cluster[{c_idx}]: {e}"
            )
            user_msg    = ""
            full_prompt = ""

        flat_prompts.append(full_prompt)
        flat_user_msgs.append(user_msg)
        flat_system_msgs.append(sys_prompt_for_variant)
        flat_input_tokens.append(len(full_prompt) // 4)

    F = len(flat_prompts)
    logger.info(
        f"[CIMR] Built {F} per-cluster judge prompts across Q={Q} questions "
        f"(variant={variant})."
    )

    # ── Phase 4b (Batched judge inference across ALL clusters of ALL Qs) ─────
    flat_responses: List[str] = [""] * F

    if F > 0:
        n_chunks = math.ceil(F / judge_batch_size)
        pbar = tqdm(
            range(0, F, judge_batch_size),
            total=n_chunks,
            desc=f"[CIMR-judge {variant}]",
            unit="batch",
            dynamic_ncols=True,
            colour="cyan",
        )

        for chunk_start in pbar:
            chunk_end     = min(chunk_start + judge_batch_size, F)
            chunk_prompts = flat_prompts[chunk_start:chunk_end]

            valid_mask    = [len(p) > 0 for p in chunk_prompts]
            valid_prompts = [p for p, v in zip(chunk_prompts, valid_mask) if v]

            if not valid_prompts:
                pbar.set_postfix(status="all-failed", refresh=False)
                continue

            try:
                chunk_responses = _judge_generate_with_retry(
                    judge_model,
                    valid_prompts,
                    max_new_tokens=max_judge_tokens,
                )
            except Exception as e:
                logger.error(
                    f"[CIMR] Batch generate failed "
                    f"chunk=[{chunk_start},{chunk_end}) error={e}"
                )
                chunk_responses = [""] * len(valid_prompts)

            it = iter(chunk_responses)
            for local_i, ok in enumerate(valid_mask):
                if ok:
                    flat_responses[chunk_start + local_i] = next(it)

            pbar.set_postfix(done=f"{chunk_end}/{F}", refresh=False)

        pbar.close()

    # ── Phase 4c (Extract one answer per cluster-response) ───────────────────
    flat_extracted: List[str] = [
        _extract_answer(r, ev, data_name) if r else "" for r in flat_responses
    ]

    # ── Phase 5 (PRM-score each judge response) ─────────────────────────────
    flat_prm_scores: List[float] = [0.0] * F
    if verifier is not None and F > 0:
        try:
            from verifier import score_candidates  # type: ignore
        except Exception as e:
            logger.warning(
                f"[CIMR] Could not import score_candidates: {e}. "
                f"Skipping PRM-on-judge scoring."
            )
            score_candidates = None  # type: ignore

        if score_candidates is not None:
            grouped: Dict[int, List[int]] = defaultdict(list)
            for idx, (q_idx, _c) in enumerate(flat_index):
                grouped[q_idx].append(idx)

            for q_idx, idxs in grouped.items():
                cands = [flat_responses[i] for i in idxs]
                non_empty_pos = [k for k, c in enumerate(cands) if c.strip()]
                if not non_empty_pos:
                    continue
                cands_ne = [cands[k] for k in non_empty_pos]
                try:
                    scored = score_candidates(
                        cands_ne,
                        verifier,
                        gold_answers[q_idx],
                        question=questions[q_idx],
                    )
                    scored = np.asarray(scored, dtype=float).tolist()
                except Exception as e:
                    logger.warning(
                        f"[CIMR] PRM scoring of judge responses failed "
                        f"for q[{q_idx}]: {e}. Falling back to zeros."
                    )
                    scored = [0.0] * len(cands_ne)

                for local_pos, ne_pos in enumerate(non_empty_pos):
                    flat_prm_scores[idxs[ne_pos]] = float(scored[local_pos])

    # ── Phase 6 (Per-question selection + correctness) ───────────────────────
    results: List[Dict[str, Any]] = []

    q_to_flat: Dict[int, List[int]] = defaultdict(list)
    for idx, (q_idx, _c) in enumerate(flat_index):
        q_to_flat[q_idx].append(idx)

    for q_idx in range(Q):
        clusters     = per_q_clusters[q_idx]
        gold         = gold_answers[q_idx]
        flat_idxs    = q_to_flat.get(q_idx, [])

        c_responses  = [flat_responses[i]  for i in flat_idxs]
        c_extracted  = [flat_extracted[i]  for i in flat_idxs]
        c_prm        = [flat_prm_scores[i] for i in flat_idxs]
        c_user_msgs  = [flat_user_msgs[i]  for i in flat_idxs]
        c_sys_msgs   = [flat_system_msgs[i] for i in flat_idxs]
        c_meta       = clusters

        approx_in_tok = sum(flat_input_tokens[i] for i in flat_idxs)

        fallback = False
        if not clusters or all(not r for r in c_responses):
            # Total failure path — fall back to plain BoN over original PRM.
            fallback = True
            src_prm = np.asarray(prm_scores_list[q_idx], dtype=float)
            if src_prm.size > 0:
                bidx = int(np.argmax(src_prm))
                fb_pred = str(answers_list[q_idx][bidx]).strip()
            else:
                fb_pred = ""
            sel = dict(
                pred_prm=fb_pred,
                pred_majority=fb_pred,
                best_idx=-1,
                majority_idx=-1,
                per_cluster=[],
            )
        else:
            sel = _select_from_cluster_responses(
                cluster_answers   = c_extracted,
                cluster_prm       = c_prm,
                cluster_responses = c_responses,
                cluster_meta      = c_meta,
            )

        is_correct_prm: Optional[bool] = None
        is_correct_maj: Optional[bool] = None
        if gold is not None and ev is not None:
            try:
                from inference_methods import _check_correct  # type: ignore
                is_correct_prm = bool(
                    _check_correct(sel["pred_prm"], gold, ev, data_name)
                )
                is_correct_maj = bool(
                    _check_correct(sel["pred_majority"], gold, ev, data_name)
                )
            except Exception as e:
                logger.warning(f"[CIMR] _check_correct failed q[{q_idx}]: {e}")

        # ── Compute normalised per-trace PRM scores (within the question) ────
        # Used for analysis / paper plots: normalised across the N original
        # generation traces so they sum / scale comparably across questions.
        all_trace_prm = list(prm_scores_list[q_idx])
        if all_trace_prm:
            arr_all = np.asarray(all_trace_prm, dtype=float)
            mn, mx = float(arr_all.min()), float(arr_all.max())
            if mx > mn:
                normed_all = ((arr_all - mn) / (mx - mn)).tolist()
            else:
                normed_all = [0.5] * len(all_trace_prm)
        else:
            normed_all = []

        # Build per-cluster summary metadata (used by old logging path)
        clusters_meta_log = [
            dict(
                cluster_id=k,
                answer=cl.get("answer", ""),
                raw_answer=cl.get("raw_answer", ""),
                size=cl.get("size", 0),
                max_prm=round(float(cl.get("max_prm", 0.0)), 4),
                mean_prm=round(float(cl.get("mean_prm", 0.0)), 4),
                trace_idxs=cl.get("trace_idxs", []),
                judge_extracted=c_extracted[k] if k < len(c_extracted) else "",
                judge_prm_on_response=round(
                    float(c_prm[k]) if k < len(c_prm) else 0.0, 4
                ),
            )
            for k, cl in enumerate(clusters)
        ]

        # ── Comprehensive per-question record (NEW) ───────────────────────────
        # This is the structure that ends up in cimr_results_<tag>.json.  It
        # contains EVERYTHING needed to reproduce the selection: every cluster
        # with all of its traces, raw and normalised PRM scores, the judge
        # response per cluster + its extracted answer + its PRM-on-response
        # score, and the final aggregated predictions.
        full_clusters_log = []
        for k, cl in enumerate(clusters):
            trace_idxs = list(cl.get("trace_idxs", []))
            cluster_traces_text = list(cl.get("traces", []))
            cluster_trace_prm = [float(p) for p in cl.get("prm_scores", [])]

            cluster_normed_prm = (
                [float(normed_all[idx]) for idx in trace_idxs]
                if normed_all else []
            )

            traces_block = []
            for local_i, t_idx in enumerate(trace_idxs):
                traces_block.append(dict(
                    trace_idx          = int(t_idx),
                    trace_text         = (
                        cluster_traces_text[local_i]
                        if local_i < len(cluster_traces_text) else ""
                    ),
                    extracted_answer   = (
                        answers_list[q_idx][t_idx]
                        if t_idx < len(answers_list[q_idx]) else ""
                    ),
                    prm_score          = (
                        cluster_trace_prm[local_i]
                        if local_i < len(cluster_trace_prm) else 0.0
                    ),
                    prm_score_normed   = (
                        cluster_normed_prm[local_i]
                        if local_i < len(cluster_normed_prm) else 0.0
                    ),
                ))

            full_clusters_log.append(dict(
                cluster_id            = k,
                cluster_answer        = cl.get("raw_answer", ""),
                cluster_answer_key    = cl.get("answer", ""),
                size                  = int(cl.get("size", 0)),
                max_prm               = float(cl.get("max_prm", 0.0)),
                mean_prm              = float(cl.get("mean_prm", 0.0)),
                trace_idxs            = trace_idxs,
                traces                = traces_block,
                # Judge artifacts for this cluster
                judge_system_prompt   = (
                    c_sys_msgs[k] if k < len(c_sys_msgs) else ""
                ),
                judge_prompt          = (
                    c_user_msgs[k] if k < len(c_user_msgs) else ""
                ),
                judge_raw_response    = (
                    c_responses[k] if k < len(c_responses) else ""
                ),
                judge_extracted_answer= (
                    c_extracted[k] if k < len(c_extracted) else ""
                ),
                judge_prm_score       = (
                    float(c_prm[k]) if k < len(c_prm) else 0.0
                ),
            ))

        cimr_full_log = dict(
            question                 = questions[q_idx],
            gold_answer              = gold,
            n_traces                 = len(all_trace_prm),
            all_trace_prm_scores     = [float(p) for p in all_trace_prm],
            all_trace_prm_normalized = [float(p) for p in normed_all],
            all_trace_extracted_answers = list(answers_list[q_idx]),
            n_clusters               = len(clusters),
            clusters                 = full_clusters_log,
            # Final predictions
            pred_prm_score           = sel["pred_prm"],
            pred_majority            = sel["pred_majority"],
            is_correct_prm           = is_correct_prm,
            is_correct_majority      = is_correct_maj,
            best_cluster_idx_prm     = int(sel.get("best_idx", -1)),
            majority_cluster_idx     = int(sel.get("majority_idx", -1)),
            variant                  = variant,
            fallback                 = fallback,
        )

        joined_response = "\n\n".join(
            [
                f"[CLUSTER {k+1}/{len(c_responses)}  "
                f"answer={c_meta[k].get('raw_answer','')!r}  "
                f"size={c_meta[k].get('size','?')}  "
                f"prm_on_response={c_prm[k]:.4f}]\n{c_responses[k]}"
                for k in range(len(c_responses))
            ]
        ) if c_responses else ""

        results.append(dict(
            # New canonical keys
            pred_prm                    = sel["pred_prm"],
            pred_majority               = sel["pred_majority"],
            is_correct_prm              = is_correct_prm,
            is_correct_majority         = is_correct_maj,
            judge_responses_per_cluster = c_responses,
            judge_extracted_answers     = c_extracted,
            judge_prm_scores            = c_prm,

            # Legacy aliases (so existing harness paths keep working)
            predicted_answer    = sel["pred_prm"],
            is_correct          = is_correct_prm,
            judge_response      = joined_response,
            approx_input_tokens = approx_in_tok,
            variant             = variant,
            n_clusters          = len(clusters),
            fallback            = fallback,
            clusters_meta       = clusters_meta_log,

            # NEW: comprehensive per-question record for cimr_results_*.json
            cimr_full_log       = cimr_full_log,
        ))

    elapsed   = time.time() - t_start
    n_known_p = sum(1 for r in results if r["is_correct_prm"] is not None)
    n_corr_p  = sum(1 for r in results if r["is_correct_prm"])
    n_known_m = sum(1 for r in results if r["is_correct_majority"] is not None)
    n_corr_m  = sum(1 for r in results if r["is_correct_majority"])
    n_fb      = sum(1 for r in results if r["fallback"])
    logger.info(
        f"[CIMR] Done  Q={Q}  elapsed={elapsed:.1f}s  "
        f"prm_acc={n_corr_p}/{n_known_p}  "
        f"maj_acc={n_corr_m}/{n_known_m}  "
        f"fallbacks={n_fb}/{Q}"
    )

    return results


# =============================================================================
# Single-question wrapper (used by the paired ablation harness)
# =============================================================================

def cimr_select_from_cache(
    question:           str,
    traces:             List[str],
    prm_scores:         List[float],
    cluster_labels:     List[int],          # API-compat; ignored
    answers:            List[str],
    judge_model,
    ev,
    data_name:          str,
    verifier            = None,             # NEW (recommended)
    variant:            str   = "full",
    max_trace_chars:    int   = 2500,
    max_judge_tokens:   int   = 1024,
    gold_answer:        Optional[str] = None,
    fallback_alpha:     float = 0.5,        # API-compat; ignored
    judge_batch_size:   int   = 4,
    drop_empty_answer:  bool  = False,
) -> Dict[str, Any]:
    """
    Single-question CIMR call on pre-computed traces.  Thin wrapper around
    cimr_judge_batch() with Q=1.  All new return keys (pred_prm,
    pred_majority, is_correct_prm, is_correct_majority, judge_prm_scores,
    judge_extracted_answers, judge_responses_per_cluster) are passed through.
    """
    t0 = time.time()
    res_list = cimr_judge_batch(
        questions           = [question],
        traces_list         = [traces],
        prm_scores_list     = [prm_scores],
        cluster_labels_list = [cluster_labels],
        answers_list        = [answers],
        judge_model         = judge_model,
        ev                  = ev,
        data_name           = data_name,
        verifier            = verifier,
        variant             = variant,
        max_trace_chars     = max_trace_chars,
        max_judge_tokens    = max_judge_tokens,
        judge_batch_size    = max(judge_batch_size, 1),
        fallback_alpha      = fallback_alpha,
        gold_answers        = [gold_answer],
        drop_empty_answer   = drop_empty_answer,
    )
    r = res_list[0]
    r["elapsed_sec"] = round(time.time() - t0, 3)
    return r


# =============================================================================
# CIMRInference — full pipeline class (compatible with run_eval.py)
# =============================================================================

class CIMRInference:
    """
    Cluster-Informed Meta-Reasoning (CIMR), answer-clustered per-cluster
    judge variant.

    Phases: generate N traces → PRM score → answer-cluster → batched per-cluster
            judge → PRM-score-on-judge-response → select.

    The legacy `cluster_manager` argument is accepted for API compatibility
    but is no longer used: clustering is now derived from extracted answers,
    not embeddings.
    """

    def __init__(
        self,
        model,
        verifier,
        cluster_manager      = None,        # accepted but ignored
        N:                int   = 8,
        alpha:            float = 0.5,
        data_name:        str   = "math",
        variant:          str   = "full",
        max_trace_chars:  int   = 2500,
        max_judge_tokens: int   = 1024,
        judge_model              = None,
        judge_batch_size: int   = 4,
        temperature:      float = 0.7,
        top_p:            float = 0.95,
        drop_empty_answer: bool = False,
    ):
        self.model              = model
        self.verifier           = verifier
        self.cluster_manager    = cluster_manager
        self.N                  = N
        self.alpha              = alpha
        self.data_name          = data_name
        self.variant            = variant
        self.max_trace_chars    = max_trace_chars
        self.max_judge_tokens   = max_judge_tokens
        self.judge_model        = judge_model or model
        self.judge_batch_size   = judge_batch_size
        self.temperature        = temperature
        self.top_p              = top_p
        self.drop_empty_answer  = drop_empty_answer

        from inference_methods import _build_evaluator  # type: ignore
        self.ev = _build_evaluator(data_name)

    def run(
        self,
        question:        str,
        gold_answer:     Optional[str] = None,
        max_new_tokens:  Optional[int] = None,
    ):
        results = self.run_batch(
            [{"question": question, "answer": gold_answer}],
            max_new_tokens=max_new_tokens,
        )
        return results[0]

    def run_batch(
        self,
        examples:       List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> List[Any]:
        """
        Full CIMR pipeline over a list of examples.

        The standard `predicted_answer` and `is_correct` fields hold the
        prm_score selection.  Two extra attributes are attached for the
        majority-vote variant:
            result._cimr_pred_majority
            result._cimr_is_correct_majority
        """
        from inference_methods import (  # type: ignore
            InferenceResult, _extract_pred, _check_correct,
            _make_error_result,
        )
        from verifier import score_candidates  # type: ignore

        Q = len(examples)
        logger.info(
            f"[CIMRInference.run_batch] Q={Q}  N={self.N}  "
            f"variant={self.variant}  judge_batch_size={self.judge_batch_size}"
        )

        questions:        List[str]               = []
        gold_answers:     List[Optional[str]]     = []
        all_traces_list:  List[List[str]]         = []
        all_prm_list:     List[List[float]]       = []
        all_answers_list: List[List[str]]         = []
        all_correct_list: List[List[bool]]        = []
        gen_times:        List[float]             = []
        error_mask:       List[bool]              = []

        gen_pbar = tqdm(
            examples,
            desc=f"[CIMR Phase1-2  N={self.N}]",
            unit="q", dynamic_ncols=True, colour="blue",
        )

        for ex in gen_pbar:
            q    = ex["question"]
            gold = ex.get("answer")
            questions.append(q)
            gold_answers.append(gold)

            try:
                t0 = time.time()
                traces = self.model.generate_samples(
                    q, N=self.N,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=max_new_tokens,
                )
                gen_elapsed = time.time() - t0

                prm_scores = score_candidates(
                    traces, self.verifier, gold, question=q
                ).tolist()

                answers = [_extract_pred(t, self.ev, self.data_name) for t in traces]

                all_correct: List[bool] = []
                if gold is not None:
                    all_correct = [
                        bool(_check_correct(a, gold, self.ev, self.data_name))
                        for a in answers
                    ]

                all_traces_list.append(traces)
                all_prm_list.append(prm_scores)
                all_answers_list.append(answers)
                all_correct_list.append(all_correct)
                gen_times.append(gen_elapsed)
                error_mask.append(False)

            except Exception as e:
                logger.error(f"[CIMR] Phase1-2 failed for '{q[:60]}': {e}")
                all_traces_list.append([])
                all_prm_list.append([])
                all_answers_list.append([])
                all_correct_list.append([])
                gen_times.append(0.0)
                error_mask.append(True)

        gen_pbar.close()

        valid_idxs = [i for i, err in enumerate(error_mask) if not err]

        if valid_idxs:
            judge_results_valid = cimr_judge_batch(
                questions           = [questions[i]        for i in valid_idxs],
                traces_list         = [all_traces_list[i]  for i in valid_idxs],
                prm_scores_list     = [all_prm_list[i]     for i in valid_idxs],
                cluster_labels_list = [[]                  for _ in valid_idxs],
                answers_list        = [all_answers_list[i] for i in valid_idxs],
                judge_model         = self.judge_model,
                ev                  = self.ev,
                data_name           = self.data_name,
                verifier            = self.verifier,
                variant             = self.variant,
                max_trace_chars     = self.max_trace_chars,
                max_judge_tokens    = self.max_judge_tokens,
                judge_batch_size    = self.judge_batch_size,
                gold_answers        = [gold_answers[i] for i in valid_idxs],
                drop_empty_answer   = self.drop_empty_answer,
            )
            judge_results: List[Optional[Dict]] = [None] * Q
            for new_i, orig_i in enumerate(valid_idxs):
                judge_results[orig_i] = judge_results_valid[new_i]
        else:
            judge_results = [None] * Q

        results = []
        n_correct_prm = 0
        n_correct_maj = 0

        for i, ex in enumerate(examples):
            q    = ex["question"]
            gold = ex.get("answer")

            if error_mask[i] or judge_results[i] is None:
                results.append(_make_error_result("cimr", q, gold, self.N))
                continue

            jr = judge_results[i]

            result = InferenceResult(
                method            = "cimr",
                question          = q,
                gold_answer       = gold,
                selected_trace    = jr["judge_response"],
                predicted_answer  = jr["pred_prm"],
                is_correct        = jr["is_correct_prm"],
                N                 = self.N,
                elapsed_sec       = gen_times[i] + jr.get("elapsed_sec", 0.0),
                all_traces        = all_traces_list[i],
                raw_scores        = all_prm_list[i],
                normalized_scores = all_prm_list[i],
                cluster_labels    = [],
                cluster_sizes     = [],
                selected_index    = -1,
                alpha             = self.alpha,
            )
            result._judge_result               = jr
            result._all_preds                  = all_answers_list[i]
            result._all_correct                = all_correct_list[i]
            result._cimr_pred_prm              = jr["pred_prm"]
            result._cimr_pred_majority         = jr["pred_majority"]
            result._cimr_is_correct_prm        = jr["is_correct_prm"]
            result._cimr_is_correct_majority   = jr["is_correct_majority"]
            result._cimr_judge_prm_scores      = jr["judge_prm_scores"]
            result._cimr_judge_extracted       = jr["judge_extracted_answers"]
            result._cimr_full_log              = jr.get("cimr_full_log", {})

            results.append(result)

            if jr["is_correct_prm"]:
                n_correct_prm += 1
            if jr["is_correct_majority"]:
                n_correct_maj += 1

        n_known = sum(1 for r in results if r.is_correct is not None)
        logger.info(
            f"[CIMRInference] Finished  Q={Q}  "
            f"prm_correct={n_correct_prm}/{n_known}  "
            f"maj_correct={n_correct_maj}/{n_known}"
        )
        return results
