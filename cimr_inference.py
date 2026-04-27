"""
cimr_inference.py
=================
Cluster-Informed Meta-Reasoning (CIMR) — LLM-based post-hoc selection that
replaces the numerical score-normalisation of DN-BoN with a structured
meta-reasoning step over ALL clustered solutions.

Algorithm (Phase 4 of the full pipeline)
-----------------------------------------
  Phase 1: GENERATE  — Sample N traces from π(·|q) at temperature T.
  Phase 2: SCORE     — Compute PRM verifier score V(x_i) for every trace.
  Phase 3: CLUSTER   — Embed via ϕ, agglomerative-cluster at threshold τ.
                       Every trace gets a cluster label and cluster size.
  Phase 4: META-REASON (this module) —
    4a. Build a structured judge prompt per question containing:
          • The original query q
          • EVERY raw solution grouped by cluster (all traces, not just the
            representative), with per-trace PRM scores, extracted answers,
            and intra-cluster agreement metadata.
        See prompts_meta_judge.py for full prompt structure.
    4b. Format prompts using the model's own chat template so system / user
        roles are respected exactly as during training.  The CIMR system
        prompt (CIMR_SYSTEM_PROMPT, from prompts_meta_judge.py) replaces the
        default math-solver system prompt for these calls.
    4c. Call the judge model in BATCHES of judge_batch_size, greedy (T=0).
        All Q judge prompts are assembled first, then dispatched in chunks
        through _generate_batch_raw — the same internal call path used by
        generate_greedy_batch — so the GPU is never idle between questions.
  Phase 5: EXTRACT   — Parse the final \\boxed{} answer from each response.

Batching design
---------------
The core bottleneck for large datasets is Phase 4c: judge inference.
This module batches judge calls across questions, not within questions,
because:
  • Each judge prompt is one sequence (system + all clusters + task).
  • Variable-length prompts are left-padded by the tokeniser (same as
    generate_greedy_batch in model_wrapper.py) and packed into one tensor.
  • A batch of B judge prompts is handled in a single model.generate() call,
    giving full GPU utilisation.

The judge_batch_size should be set equal to the number of GPU(s) × memory
budget per sequence.  For an H100 80 GB with N=8 and full traces, 4–8 judge
prompts fit per batch.  For concise/answer_only variants, 16–32 are feasible.

Prompt injection into the chat template
-----------------------------------------
HFModelWrapper._build_prompt() uses self.system_prompt (the math-solver
prompt) and wraps questions in a chat template.  CIMR needs a different
system prompt (the judge protocol) and a structured user message, so we
bypass _build_prompt entirely and format prompts ourselves:

    _build_cimr_prompt_text(judge_model, system_prompt, user_message)

This function:
  1. Uses the tokeniser's apply_chat_template() if a template is available,
     injecting the CIMR system + user content directly.
  2. Falls back to plain "SYSTEM\\n\\nUSER" concatenation otherwise.

The resulting string is passed directly to _generate_batch_raw(), which
tokenises, pads, and generates — identical to generate_greedy_batch() but
without the extra _build_prompt() wrapping.

Compute cost
------------
Phase 4 adds ≤ ceil(Q / judge_batch_size) generate() calls, each processing
B judge prompts.  The FLOPs cost per question is:

    ΔF = 2 · P_judge · (T_in + T_out)

where T_in ≈ (M · max_trace_chars) / 4  tokens (M = number of clusters,
typically 2–5) and T_out ≈ max_judge_tokens (default 1024).

For Qwen2.5-Math-7B, N=8, M=3 clusters, max_trace_chars=2500:
    T_in ≈ 5 000 tokens,  T_out ≈ 1 000 tokens  → ΔF ≈ 12% of one trace.
The overhead does NOT grow with N; it grows at most linearly with M (which
saturates quickly as N increases).
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction from judge response
# ─────────────────────────────────────────────────────────────────────────────

def _extract_boxed(text: str) -> str:
    """
    Extract the content of the LAST \\boxed{...} in text, handling nested braces.
    Falls back to the last number/expression in the text if no \\boxed{} found.
    """
    import re
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
    # 1. \\boxed{}
    ans = _extract_boxed(judge_response)
    if ans:
        return ans

    # 2. Evaluator extraction
    if ev is not None:
        try:
            from inference_methods import _extract_pred
            ans = _extract_pred(judge_response, ev, data_name)
            if ans:
                return ans
        except Exception:
            pass

    # 3. Last number
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", judge_response.replace(",", ""))
    return nums[-1] if nums else ""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting — bypasses HFModelWrapper._build_prompt()
# ─────────────────────────────────────────────────────────────────────────────

def _build_cimr_prompt_text(
    judge_model,        # HFModelWrapper instance
    system_content: str,
    user_content:   str,
) -> str:
    """
    Format a CIMR judge prompt using the model's chat template.

    This function bypasses HFModelWrapper._build_prompt() (which hardcodes
    self.system_prompt and wraps the input as a plain math question) and
    instead directly calls the tokeniser's apply_chat_template() with the
    CIMR system prompt and structured user content.

    If the tokeniser has no chat template, falls back to plain concatenation.

    Parameters
    ----------
    judge_model    : HFModelWrapper — provides .tokenizer
    system_content : CIMR SYSTEM_PROMPT string
    user_content   : formatted judge user message (from build_judge_prompt)

    Returns
    -------
    str : fully formatted prompt string ready for tokenisation
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

    # Plain fallback: system \n\n user
    return f"{system_content}\n\n{user_content}"


# ─────────────────────────────────────────────────────────────────────────────
# DN-BoN fallback (used when judge call fails for a question)
# ─────────────────────────────────────────────────────────────────────────────

def _dnbon_fallback(
    prm_scores:     List[float],
    cluster_labels: List[int],
    answers:        List[str],
    alpha:          float = 0.5,
) -> Tuple[str, int]:
    """
    Compute DN-BoN selection as a fallback when the judge call fails.
    Returns (predicted_answer, best_idx).
    """
    scores = np.array(prm_scores, dtype=float)
    sizes  = np.array([
        sum(1 for l in cluster_labels if l == cluster_labels[i])
        for i in range(len(prm_scores))
    ], dtype=float)
    sizes  = np.maximum(sizes, 1.0)
    norm   = scores / (sizes ** alpha)
    idx    = int(np.argmax(norm))
    return (answers[idx] if idx < len(answers) else ""), idx


# ─────────────────────────────────────────────────────────────────────────────
# Batched judge inference (the core of this module)
# ─────────────────────────────────────────────────────────────────────────────

def cimr_judge_batch(
    questions:        List[str],
    traces_list:      List[List[str]],       # outer: Q questions, inner: N traces
    prm_scores_list:  List[List[float]],
    cluster_labels_list: List[List[int]],
    answers_list:     List[List[str]],
    judge_model,                              # HFModelWrapper
    ev,                                       # evaluator instance
    data_name:        str,
    variant:          str  = "full",
    max_trace_chars:  int  = 2500,
    max_judge_tokens: int  = 1024,
    judge_batch_size: int  = 4,
    fallback_alpha:   float = 0.5,
    gold_answers:     Optional[List[Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Run the CIMR meta-judge for a batch of Q questions in one pass.

    This is the primary entry point for batched evaluation.  It:
      1. Builds one judge prompt per question.
      2. Formats each using the model's chat template (CIMR system prompt).
      3. Dispatches all Q prompts through _generate_batch_raw in chunks of
         judge_batch_size — same internal path as generate_greedy_batch.
      4. Extracts the \\boxed{} answer from each judge response.
      5. Checks correctness if gold answers are supplied.

    Parameters
    ----------
    questions             : list[str]         — Q original problem strings
    traces_list           : list[list[str]]   — Q × N trace strings
    prm_scores_list       : list[list[float]] — Q × N PRM scores
    cluster_labels_list   : list[list[int]]   — Q × N cluster labels
    answers_list          : list[list[str]]   — Q × N extracted answers
    judge_model           : HFModelWrapper
    ev                    : evaluator (for correctness checking & fallback extraction)
    data_name             : dataset name
    variant               : 'full' | 'concise' | 'answer_only'
    max_trace_chars       : max chars per trace body in judge prompt
    max_judge_tokens      : max NEW tokens for judge response
    judge_batch_size      : prompts per model.generate() call (tune to VRAM)
    fallback_alpha        : DN-BoN alpha used if a judge call fails per-question
    gold_answers          : list[str|None] — gold for correctness; None = unknown

    Returns
    -------
    list[dict] of length Q, each with keys:
      predicted_answer : str
      is_correct       : bool | None
      judge_response   : str   (full judge output)
      approx_input_tokens : int
      variant          : str
      n_clusters       : int
      fallback         : bool  (True if judge failed → DN-BoN used)
      clusters_meta    : list[dict]  (cluster header info for logging)
    """
    from prompts_meta_judge import build_judge_prompt, SYSTEM_PROMPT as CIMR_SYS

    Q = len(questions)
    if gold_answers is None:
        gold_answers = [None] * Q

    t_start = time.time()

    # ── Step 1: Build all judge prompts ──────────────────────────────────────
    logger.info(
        f"[CIMR] Building {Q} judge prompts  "
        f"variant={variant}  max_chars={max_trace_chars}"
    )

    formatted_prompts: List[str]       = []   # ready for _generate_batch_raw
    clusters_per_q:    List[List[Dict]] = []  # cluster metadata for results
    input_token_counts: List[int]      = []

    for i in range(Q):
        try:
            _sys, user_msg, clusters = build_judge_prompt(
                question       = questions[i],
                traces         = traces_list[i],
                prm_scores     = prm_scores_list[i],
                answers        = answers_list[i],
                cluster_labels = cluster_labels_list[i],
                variant        = variant,
                max_trace_chars= max_trace_chars,
            )
            # Format with the model's chat template using CIMR system prompt
            full_prompt = _build_cimr_prompt_text(judge_model, CIMR_SYS, user_msg)
        except Exception as e:
            logger.error(f"[CIMR] Prompt build failed for q[{i}]: {e}")
            # Placeholder empty prompt — will trigger fallback in step 3
            full_prompt = ""
            clusters    = []

        formatted_prompts.append(full_prompt)
        clusters_per_q.append(clusters)
        input_token_counts.append(len(full_prompt) // 4)   # rough token estimate

    # ── Step 2: Batched judge inference ──────────────────────────────────────
    n_chunks = math.ceil(Q / judge_batch_size)
    logger.info(
        f"[CIMR] Running judge inference  "
        f"Q={Q}  judge_batch_size={judge_batch_size}  "
        f"n_chunks={n_chunks}  max_new_tokens={max_judge_tokens}"
    )

    raw_responses: List[str] = [""] * Q

    pbar = tqdm(
        range(0, Q, judge_batch_size),
        total=n_chunks,
        desc=f"[CIMR-judge {variant}]",
        unit="batch",
        dynamic_ncols=True,
        colour="cyan",
    )

    for chunk_start in pbar:
        chunk_end    = min(chunk_start + judge_batch_size, Q)
        chunk_prompts = formatted_prompts[chunk_start:chunk_end]

        # Filter out empty prompts (prompt-build failures)
        valid_mask   = [len(p) > 0 for p in chunk_prompts]
        valid_prompts = [p for p, v in zip(chunk_prompts, valid_mask) if v]

        if not valid_prompts:
            pbar.set_postfix(status="all-failed", refresh=False)
            continue

        try:
            # Use _generate_batch_raw directly — bypasses _build_prompt()
            # do_sample=False → greedy (temperature=0), deterministic judge
            chunk_responses = judge_model._generate_batch_raw(
                valid_prompts,
                do_sample=False,
                max_new_tokens=max_judge_tokens,
            )

            # print("🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩")
            # print("Prompt: ", valid_prompts)
            # print("Response: ", chunk_responses)
            # print("🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩🧩")

        except Exception as e:
            logger.error(
                f"[CIMR] Batch generate failed  "
                f"chunk=[{chunk_start},{chunk_end})  error={e}"
            )
            chunk_responses = [""] * len(valid_prompts)

        # Map back to global indices
        valid_iter = iter(chunk_responses)
        for local_i, is_valid in enumerate(valid_mask):
            if is_valid:
                raw_responses[chunk_start + local_i] = next(valid_iter)

        pbar.set_postfix(
            done=f"{chunk_end}/{Q}",
            refresh=False,
        )

    pbar.close()

    # ── Step 3: Extract answers and check correctness ─────────────────────────
    results: List[Dict[str, Any]] = []

    for i in range(Q):
        response = raw_responses[i]
        gold     = gold_answers[i]
        fallback = False

        if response:
            pred = _extract_answer(response, ev, data_name)
        else:
            pred = ""

        # If extraction failed or prompt build failed → DN-BoN fallback
        if not pred and traces_list[i]:
            fallback_ans, _ = _dnbon_fallback(
                prm_scores_list[i],
                cluster_labels_list[i],
                answers_list[i],
                alpha=fallback_alpha,
            )
            pred     = fallback_ans
            fallback = True
            if not response:
                response = f"[CIMR prompt-build failure; DN-BoN fallback: {pred}]"
            else:
                response = f"[CIMR extraction failure; DN-BoN fallback: {pred}]\n{response}"

        # Correctness
        is_correct: Optional[bool] = None
        if gold is not None and ev is not None:
            try:
                from inference_methods import _check_correct
                is_correct = bool(_check_correct(pred, gold, ev, data_name))
            except Exception:
                pass

        # Cluster summary for logging (no full trace text — just headers)
        clusters_meta = [
            dict(
                cluster_id  = c.get("cluster_id", i),
                size        = c["size"],
                max_prm     = round(c["max_prm"], 4),
                mean_prm    = round(c["mean_prm"], 4),
                agreement   = round(c.get("answer_agreement", 1.0), 3),
                modal_answer= c.get("modal_answer", ""),
                n_variants  = len(c.get("all_answer_variants", [])),
            )
            for c in clusters_per_q[i]
        ]

        results.append(dict(
            predicted_answer    = pred,
            is_correct          = is_correct,
            judge_response      = response,
            approx_input_tokens = input_token_counts[i],
            variant             = variant,
            n_clusters          = len(clusters_per_q[i]),
            fallback            = fallback,
            clusters_meta       = clusters_meta,
        ))

    elapsed = time.time() - t_start
    n_correct = sum(1 for r in results if r["is_correct"])
    n_known   = sum(1 for r in results if r["is_correct"] is not None)
    n_fallback = sum(1 for r in results if r["fallback"])
    logger.info(
        f"[CIMR] Done  Q={Q}  elapsed={elapsed:.1f}s  "
        f"accuracy={n_correct}/{n_known}  "
        f"fallbacks={n_fallback}/{Q}"
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Single-question wrapper (used by paired ablation for one cached question)
# ─────────────────────────────────────────────────────────────────────────────

def cimr_select_from_cache(
    question:         str,
    traces:           List[str],
    prm_scores:       List[float],
    cluster_labels:   List[int],
    answers:          List[str],
    judge_model,
    ev,
    data_name:        str,
    variant:          str   = "full",
    max_trace_chars:  int   = 2500,
    max_judge_tokens: int   = 1024,
    gold_answer:      Optional[str] = None,
    fallback_alpha:   float = 0.5,
    judge_batch_size: int   = 4,
) -> Dict[str, Any]:
    """
    Single-question CIMR call on pre-computed traces.

    Thin wrapper around cimr_judge_batch() with Q=1.  Used by
    run_paired_ablation.py in single-question mode.

    NOTE: judge_batch_size is passed through to cimr_judge_batch and controls
    how many judge prompts are packed into one model.generate() call.  For a
    single-question call (Q=1) it is clamped to 1 internally, but the parameter
    is kept so callers can pass a consistent value without special-casing.

    For dataset-scale evaluation, prefer calling cimr_judge_batch() directly
    with all Q questions at once — it gives full GPU utilisation across the
    entire dataset.

    Returns
    -------
    dict with keys: predicted_answer, is_correct, judge_response,
                    approx_input_tokens, variant, n_clusters,
                    fallback, clusters_meta, elapsed_sec
    """
    t0 = time.time()

    results = cimr_judge_batch(
        questions            = [question],
        traces_list          = [traces],
        prm_scores_list      = [prm_scores],
        cluster_labels_list  = [cluster_labels],
        answers_list         = [answers],
        judge_model          = judge_model,
        ev                   = ev,
        data_name            = data_name,
        variant              = variant,
        max_trace_chars      = max_trace_chars,
        max_judge_tokens     = max_judge_tokens,
        judge_batch_size     = min(judge_batch_size, 1),  # Q=1, cap at 1
        fallback_alpha       = fallback_alpha,
        gold_answers         = [gold_answer],
    )

    r = results[0]
    r["elapsed_sec"] = round(time.time() - t0, 3)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# CIMRInference — full pipeline class (compatible with run_eval.py)
# ─────────────────────────────────────────────────────────────────────────────

class CIMRInference:
    """
    Cluster-Informed Meta-Reasoning (CIMR): full pipeline.

    Phases: generate N traces → PRM score → embed+cluster → batched judge → extract.

    The judge call is batched at the run_batch() level: all questions in a
    batch share a single sequence of judge generate() calls, giving full GPU
    utilisation.

    Parameters
    ----------
    model              : HFModelWrapper — generator (Phase 1).
    verifier           : PRM verifier (Phase 2).
    cluster_manager    : ClusterManager (Phase 3).
    N                  : number of candidate traces per question.
    alpha              : DN-BoN exponent for fallback (used if judge fails).
    data_name          : dataset name for evaluator/answer-extraction.
    variant            : CIMR prompt variant ('full' | 'concise' | 'answer_only').
    max_trace_chars    : max chars per trace body in judge prompt.
    max_judge_tokens   : max new tokens for judge response.
    judge_model        : separate model for judge calls; None = use self.model.
    judge_batch_size   : questions per judge generate() call.
                         Tune to VRAM: start at 4, increase if memory allows.
    temperature        : trace sampling temperature.
    top_p              : nucleus sampling for trace generation.
    """

    def __init__(
        self,
        model,
        verifier,
        cluster_manager,
        N:                int   = 8,
        alpha:            float = 0.5,
        data_name:        str   = "math",
        variant:          str   = "full",
        max_trace_chars:  int   = 2500,
        max_judge_tokens: int   = 1024,
        judge_model             = None,
        judge_batch_size: int   = 4,
        temperature:      float = 0.7,
        top_p:            float = 0.95,
    ):
        self.model            = model
        self.verifier         = verifier
        self.cluster_manager  = cluster_manager
        self.N                = N
        self.alpha            = alpha
        self.data_name        = data_name
        self.variant          = variant
        self.max_trace_chars  = max_trace_chars
        self.max_judge_tokens = max_judge_tokens
        self.judge_model      = judge_model or model
        self.judge_batch_size = judge_batch_size
        self.temperature      = temperature
        self.top_p            = top_p

        from inference_methods import _build_evaluator
        self.ev = _build_evaluator(data_name)

    # ── Single-question run (calls run_batch with one element) ────────────────

    def run(
        self,
        question:        str,
        gold_answer:     Optional[str] = None,
        max_new_tokens:  Optional[int] = None,
    ):
        """Full single-question pipeline.  Prefer run_batch() for datasets."""
        results = self.run_batch(
            [{"question": question, "answer": gold_answer}],
            max_new_tokens=max_new_tokens,
        )
        return results[0]

    # ── Batched run (all phases, all questions) ────────────────────────────────

    def run_batch(
        self,
        examples:       List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> List[Any]:
        """
        Full CIMR pipeline over a list of examples.

        Phase 1+2 (generate + score) run question-by-question (existing
        model_wrapper.generate_samples path, unchanged).

        Phase 3 (cluster) runs question-by-question (fast; CPU-side).

        Phase 4 (judge) runs in batches of judge_batch_size across ALL
        questions — this is the efficiency gain vs. the old sequential design.

        Returns list of InferenceResult objects.
        """
        from inference_methods import (
            InferenceResult, _extract_pred, _check_correct,
            _make_error_result,
        )
        from verifier import score_candidates

        Q = len(examples)
        logger.info(
            f"[CIMRInference.run_batch] Q={Q}  N={self.N}  "
            f"variant={self.variant}  judge_batch_size={self.judge_batch_size}"
        )

        # ── Phases 1–3: generate, score, cluster (per question) ───────────────
        questions:            List[str]         = []
        gold_answers:         List[Optional[str]] = []
        all_traces_list:      List[List[str]]   = []
        all_prm_list:         List[List[float]] = []
        all_labels_list:      List[List[int]]   = []
        all_sizes_list:       List[List[int]]   = []
        all_answers_list:     List[List[str]]   = []
        all_correct_list:     List[List[bool]]  = []
        gen_times:            List[float]       = []
        error_mask:           List[bool]        = []  # True = failed → placeholder

        gen_pbar = tqdm(
            examples,
            desc=f"[CIMR Phase1-3  N={self.N}]",
            unit="q", dynamic_ncols=True, colour="blue",
        )

        for ex in gen_pbar:
            q    = ex["question"]
            gold = ex.get("answer")
            questions.append(q)
            gold_answers.append(gold)

            try:
                t0     = time.time()
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

                labels, sizes = self.cluster_manager.cluster(traces)
                labels = labels.tolist()
                sizes  = sizes.tolist()

                answers = [_extract_pred(t, self.ev, self.data_name) for t in traces]

                all_correct: List[bool] = []
                if gold is not None:
                    all_correct = [
                        bool(_check_correct(a, gold, self.ev, self.data_name))
                        for a in answers
                    ]

                all_traces_list.append(traces)
                all_prm_list.append(prm_scores)
                all_labels_list.append(labels)
                all_sizes_list.append(sizes)
                all_answers_list.append(answers)
                all_correct_list.append(all_correct)
                gen_times.append(gen_elapsed)
                error_mask.append(False)

            except Exception as e:
                logger.error(f"[CIMR] Phase1-3 failed for '{q[:60]}': {e}")
                # Fill with empty placeholders so indexing stays aligned
                all_traces_list.append([])
                all_prm_list.append([])
                all_labels_list.append([])
                all_sizes_list.append([])
                all_answers_list.append([])
                all_correct_list.append([])
                gen_times.append(0.0)
                error_mask.append(True)

        gen_pbar.close()

        # ── Phase 4: Batched judge inference ─────────────────────────────────
        # Only pass non-errored questions to the judge
        valid_idxs = [i for i, err in enumerate(error_mask) if not err]

        if valid_idxs:
            judge_results_valid = cimr_judge_batch(
                questions            = [questions[i]        for i in valid_idxs],
                traces_list          = [all_traces_list[i]  for i in valid_idxs],
                prm_scores_list      = [all_prm_list[i]     for i in valid_idxs],
                cluster_labels_list  = [all_labels_list[i]  for i in valid_idxs],
                answers_list         = [all_answers_list[i] for i in valid_idxs],
                judge_model          = self.judge_model,
                ev                   = self.ev,
                data_name            = self.data_name,
                variant              = self.variant,
                max_trace_chars      = self.max_trace_chars,
                max_judge_tokens     = self.max_judge_tokens,
                judge_batch_size     = self.judge_batch_size,
                fallback_alpha       = self.alpha,
                gold_answers         = [gold_answers[i] for i in valid_idxs],
            )
            # Map back to full-size list
            judge_results: List[Optional[Dict]] = [None] * Q
            for new_i, orig_i in enumerate(valid_idxs):
                judge_results[orig_i] = judge_results_valid[new_i]
        else:
            judge_results = [None] * Q

        # ── Phase 5: Build InferenceResult objects ────────────────────────────
        results = []
        n_correct = 0

        for i, ex in enumerate(examples):
            q    = ex["question"]
            gold = ex.get("answer")

            if error_mask[i] or judge_results[i] is None:
                results.append(_make_error_result("cimr", q, gold, self.N))
                continue

            jr   = judge_results[i]
            pred = jr["predicted_answer"]
            correct = jr["is_correct"]

            result = InferenceResult(
                method            = "cimr",
                question          = q,
                gold_answer       = gold,
                selected_trace    = jr["judge_response"],
                predicted_answer  = pred,
                is_correct        = correct,
                N                 = self.N,
                elapsed_sec       = gen_times[i] + jr.get("elapsed_sec", 0.0),
                all_traces        = all_traces_list[i],
                raw_scores        = all_prm_list[i],
                normalized_scores = all_prm_list[i],  # CIMR doesn't normalise numerically
                cluster_labels    = all_labels_list[i],
                cluster_sizes     = all_sizes_list[i],
                selected_index    = -1,                # judge produces free-form answer
                alpha             = self.alpha,
            )
            # Extra metadata for analysis
            result._judge_result  = jr
            result._all_preds     = all_answers_list[i]
            result._all_correct   = all_correct_list[i]

            results.append(result)

            if correct:
                n_correct += 1

        logger.info(
            f"[CIMRInference] Finished  Q={Q}  "
            f"correct={n_correct}/{sum(1 for r in results if r.is_correct is not None)}"
        )
        return results