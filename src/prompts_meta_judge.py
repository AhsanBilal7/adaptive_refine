"""
prompts_meta_judge.py
=====================

CIMR prompt: evidence-first, reasoning-as-veto verifier.

Core idea:
- First rank candidate answers by count + PRM evidence.
- Then inspect reasoning only to veto clearly invalid candidates.
- Avoid letting long flawed reasoning override strong answer evidence.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np


SYSTEM_PROMPT = (
    "You are a strict mathematical answer verifier.\n\n"

    "You are given a math problem, N candidate reasoning traces, extracted answers, "
    "PRM scores, and cluster information.\n\n"

    "Your goal is to select the candidate answer most likely to be correct.\n\n"

    "Important principle:\n"
    "Use reasoning traces as evidence to CONFIRM or VETO candidate answers. "
    "Do not let a fluent but isolated reasoning trace override stronger answer-level evidence.\n\n"

    "Decision process:\n"
    "1. First rank candidate answers using answer-level evidence: frequency, PRM support, "
    "and cluster support.\n"
    "2. Start from the strongest candidate answer.\n"
    "3. Inspect traces supporting that answer.\n"
    "4. Keep that answer unless every strong supporting trace has a concrete mathematical error.\n"
    "5. Move to a lower-ranked answer only if the stronger answer is clearly invalidated.\n\n"

    "Rules:\n"
    "- Do not re-solve the problem from scratch.\n"
    "- Do not invent a new answer unless all candidate answers are invalidated.\n"
    "- Do not choose an isolated minority answer merely because its reasoning sounds plausible.\n"
    "- A candidate answer is invalidated only if you can identify a specific error in all strong traces supporting it.\n"
    "- If a candidate answer appears multiple times and has high PRM support, it should be preferred unless clearly wrong.\n\n"

    "Output:\n"
    "- Briefly state the selected answer and why stronger/equal alternatives are accepted or rejected.\n"
    "- The final line must contain only \\boxed{...}."
)


def _truncate_trace(trace: str, max_chars: int) -> str:
    """Keep head and tail; truncate the middle."""
    text = str(trace).strip()

    if len(text) <= max_chars:
        return text

    keep = max_chars // 2
    head = text[:keep].rstrip()
    tail = text[-keep:].lstrip()
    omit = len(text) - max_chars

    return f"{head}\n\n[... {omit:,} chars omitted ...]\n\n{tail}"


def build_cluster_info(
    traces:         List[str],
    prm_scores:     List[float],
    answers:        List[str],
    cluster_labels: List[int],
) -> List[Dict[str, Any]]:
    """
    Convert flat per-trace arrays into structured cluster dictionaries.
    """
    if not (
        len(traces) == len(prm_scores) == len(answers) == len(cluster_labels)
    ):
        raise ValueError(
            "traces, prm_scores, answers, and cluster_labels must have the same length"
        )

    cluster_map: Dict[int, List[int]] = defaultdict(list)

    for i, cid in enumerate(cluster_labels):
        cluster_map[int(cid)].append(i)

    clusters = []

    for cid in sorted(cluster_map.keys()):
        idxs = cluster_map[cid]

        scores = [float(prm_scores[i]) for i in idxs]
        ans = [str(answers[i]).strip() for i in idxs]
        txts = [str(traces[i]) for i in idxs]

        order = sorted(
            range(len(idxs)),
            key=lambda k: scores[k],
            reverse=True,
        )

        non_empty = [a for a in ans if a]

        if non_empty:
            modal_ans, modal_count = Counter(non_empty).most_common(1)[0]
            agreement = modal_count / len(ans)
        else:
            modal_ans = ""
            agreement = 1.0

        all_variants = sorted(set(non_empty))

        cluster_traces = [
            dict(
                trace_idx=idxs[k],
                prm_score=scores[k],
                answer=ans[k],
                text=txts[k],
            )
            for k in order
        ]

        clusters.append(
            dict(
                cluster_id=cid,
                original_cluster_id=cid,
                size=len(idxs),
                mean_prm=float(np.mean(scores)),
                max_prm=float(np.max(scores)),
                answer_agreement=float(agreement),
                modal_answer=modal_ans,
                all_answer_variants=all_variants,
                traces=cluster_traces,
            )
        )

    clusters.sort(key=lambda c: (c["max_prm"], c["size"]), reverse=True)

    for new_id, c in enumerate(clusters):
        c["cluster_id"] = new_id

    return clusters


def _collect_all_traces(
    clusters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    all_traces = []

    for c in clusters:
        for t in c["traces"]:
            all_traces.append(
                {
                    "trace_idx": t["trace_idx"],
                    "cluster_id": c["cluster_id"],
                    "prm_score": t["prm_score"],
                    "answer": str(t["answer"]).strip(),
                    "text": t["text"],
                }
            )

    all_traces.sort(key=lambda x: x["trace_idx"])
    return all_traces

# def _collect_all_traces(
#     clusters: List[Dict[str, Any]],
# ) -> List[Dict[str, Any]]:
#     all_traces = []

#     for c in clusters:
#         if not c["traces"]:
#             continue

#         # pick trace with highest PRM score in this cluster
#         best_trace = max(c["traces"], key=lambda t: t["prm_score"])

#         all_traces.append(
#             {
#                 "trace_idx": best_trace["trace_idx"],
#                 "cluster_id": c["cluster_id"],
#                 "prm_score": best_trace["prm_score"],
#                 "answer": str(best_trace["answer"]).strip(),
#                 "text": best_trace["text"],
#             }
#         )

#     # optional: sort by cluster_id instead of trace_idx (more logical here)
#     all_traces.sort(key=lambda x: x["cluster_id"])
#     return all_traces


def _answer_stats(
    clusters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    all_traces = _collect_all_traces(clusters)
    answers = sorted(set(t["answer"] for t in all_traces if t["answer"]))

    stats = []

    for ans in answers:
        supporting = [t for t in all_traces if t["answer"] == ans]
        prms = [t["prm_score"] for t in supporting]
        cluster_ids = sorted(set(t["cluster_id"] + 1 for t in supporting))
        trace_ids = [t["trace_idx"] + 1 for t in supporting]

        # Evidence score is only for ordering in the prompt, not hidden gating.
        evidence_score = (
            len(supporting)
            + max(prms)
            + 0.25 * len(cluster_ids)
        )

        stats.append(
            {
                "answer": ans,
                "count": len(supporting),
                "best_prm": max(prms),
                "mean_prm": sum(prms) / len(prms),
                "clusters": cluster_ids,
                "traces": trace_ids,
                "evidence_score": evidence_score,
            }
        )

    stats.sort(
        key=lambda x: (
            x["evidence_score"],
            x["count"],
            x["best_prm"],
            x["mean_prm"],
        ),
        reverse=True,
    )

    return stats


def _build_ranked_answer_section(
    clusters: List[Dict[str, Any]],
) -> List[str]:
    stats = _answer_stats(clusters)

    lines = [
        "=" * 70,
        "RANKED CANDIDATE ANSWERS",
        "=" * 70,
        "These candidate answers are ranked by answer-level evidence.",
        "Start verification from Rank 1. Only move downward if the stronger answer is clearly invalidated.",
        "",
    ]

    for rank, s in enumerate(stats, start=1):
        lines.append(
            f"Rank {rank}: answer={s['answer']!r} | "
            f"count={s['count']} | "
            f"best_PRM={s['best_prm']:.4f} | "
            f"mean_PRM={s['mean_prm']:.4f} | "
            f"clusters={s['clusters']} | "
            f"traces={s['traces']}"
        )

    lines.append("")
    return lines


def _build_all_traces_section(
    clusters: List[Dict[str, Any]],
    max_chars_per_trace: int,
) -> List[str]:
    all_traces = _collect_all_traces(clusters)

    lines = [
        "=" * 70,
        "TRACE TEXTS FOR VERIFICATION",
        "=" * 70,
        "Use these only to verify or veto ranked candidate answers.",
        "Do not let an isolated fluent trace override stronger ranked evidence.",
        "",
    ]

    for t in all_traces:
        body = _truncate_trace(t["text"], max_chars_per_trace)

        lines += [
            "-" * 70,
            f"TRACE {t['trace_idx'] + 1}",
            f"Cluster: {t['cluster_id'] + 1}",
            f"PRM score: {t['prm_score']:.4f}",
            f"Extracted answer: {t['answer'] or '(none)'}",
            "",
            body,
            "",
        ]

    return lines


def _build_cluster_summary_section(
    clusters: List[Dict[str, Any]],
) -> List[str]:
    lines = [
        "=" * 70,
        "CLUSTER SUMMARY",
        "=" * 70,
    ]

    for c in clusters:
        variants = c["all_answer_variants"]

        ans_display = (
            ", ".join(f'"{v}"' for v in variants)
            if variants
            else "(no extracted answer)"
        )

        trace_ids = [
            str(t["trace_idx"] + 1)
            for t in sorted(c["traces"], key=lambda x: x["trace_idx"])
        ]

        lines += [
            "-" * 70,
            f"CLUSTER {c['cluster_id'] + 1}",
            f"Trace numbers: {', '.join(trace_ids)}",
            f"Size: {c['size']}",
            f"PRM max: {c['max_prm']:.4f}",
            f"PRM mean: {c['mean_prm']:.4f}",
            f"Modal answer: {c['modal_answer'] or '(none)'}",
            f"All answers: {ans_display}",
            f"Answer agreement: {c['answer_agreement']:.0%}",
            "",
        ]

    return lines


def _build_final_task_section() -> List[str]:
    return [
        "=" * 70,
        "FINAL DECISION TASK",
        "=" * 70,
        "Follow this procedure exactly:",
        "",
        "1. Begin with the Rank 1 candidate answer.",
        "2. Check the strongest traces supporting it.",
        "3. If at least one strong supporting trace has no clear mathematical error, choose Rank 1.",
        "4. Reject Rank 1 only if every strong supporting trace has a specific identified error.",
        "5. Then consider Rank 2, Rank 3, and so on.",
        "",
        "Do not perform a fresh solution unless every ranked candidate is invalidated.",
        "Do not select a lower-ranked answer without explaining why all stronger-ranked answers fail.",
        "",
        "Write 2-4 sentences.",
        "The final line must contain only:",
        "\\boxed{<answer>}",
    ]


def build_full_analysis_prompt(
    question: str,
    clusters: List[Dict[str, Any]],
    max_chars_per_trace: int = 1800,
) -> str:
    N_total = sum(c["size"] for c in clusters)
    M = len(clusters)

    lines = [
        "=" * 70,
        "PROBLEM",
        "=" * 70,
        question.strip(),
        "",
        f"There are {N_total} traces grouped into {M} cluster(s).",
        "Select the correct candidate answer by ranked evidence plus reasoning verification.",
        "",
    ]

    lines += _build_ranked_answer_section(clusters)
    lines += _build_all_traces_section(clusters, max_chars_per_trace)
    lines += _build_cluster_summary_section(clusters)
    lines += _build_final_task_section()

    return "\n".join(lines)


def build_concise_prompt(
    question: str,
    clusters: List[Dict[str, Any]],
    max_chars_per_trace: int = 700,
) -> str:
    N_total = sum(c["size"] for c in clusters)
    M = len(clusters)

    lines = [
        "PROBLEM:",
        question.strip(),
        "",
        f"There are {N_total} traces in {M} cluster(s).",
        "Select the correct candidate answer by ranked evidence plus verification.",
        "",
    ]

    lines += _build_ranked_answer_section(clusters)
    lines += _build_all_traces_section(clusters, max_chars_per_trace)

    lines += [
        "TASK:",
        "Start with Rank 1.",
        "Choose Rank 1 if any strong supporting trace has no clear mathematical error.",
        "Move to a lower rank only if every stronger candidate is clearly invalidated.",
        "Do not invent a new answer.",
        "The final line must contain only:",
        "\\boxed{<answer>}",
    ]

    return "\n".join(lines)


def build_answer_only_prompt(
    question: str,
    clusters: List[Dict[str, Any]],
) -> str:
    stats = _answer_stats(clusters)

    lines = [
        "PROBLEM:",
        question.strip(),
        "",
        "Only extracted answers, PRM scores, and clusters are shown.",
        "Choose the most reliable candidate answer. Do not invent a new answer.",
        "",
        "=" * 70,
        "RANKED CANDIDATE ANSWERS",
        "=" * 70,
    ]

    for rank, s in enumerate(stats, start=1):
        lines.append(
            f"Rank {rank}: answer={s['answer']!r} | "
            f"count={s['count']} | "
            f"best_PRM={s['best_prm']:.4f} | "
            f"mean_PRM={s['mean_prm']:.4f} | "
            f"clusters={s['clusters']} | "
            f"traces={s['traces']}"
        )

    lines += [
        "",
        "TASK:",
        "Choose Rank 1 unless there is a compelling reason from the evidence above to choose otherwise.",
        "Prefer repeated answers and high-PRM answers.",
        "The final line must contain only:",
        "\\boxed{<answer>}",
    ]

    return "\n".join(lines)


def build_judge_prompt(
    question:        str,
    traces:          List[str],
    prm_scores:      List[float],
    answers:         List[str],
    cluster_labels:  List[int],
    variant:         str = "full",
    max_trace_chars: int = 1800,
) -> tuple:
    clusters = build_cluster_info(
        traces=traces,
        prm_scores=prm_scores,
        answers=answers,
        cluster_labels=cluster_labels,
    )

    if variant == "full":
        user = build_full_analysis_prompt(
            question,
            clusters,
            max_chars_per_trace=max_trace_chars,
        )
    elif variant == "concise":
        user = build_concise_prompt(
            question,
            clusters,
            max_chars_per_trace=min(max_trace_chars, 700),
        )
    elif variant == "answer_only":
        user = build_answer_only_prompt(question, clusters)
    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose: 'full' | 'concise' | 'answer_only'"
        )

    return SYSTEM_PROMPT, user, clusters

# =============================================================================
# Per-cluster judge prompt (answer-clustered CIMR variant)
# =============================================================================
#
# Design
# ------
# Traces are grouped by their EXTRACTED FINAL ANSWER (not by embedding-based
# clustering). For each answer-cluster c, the judge receives:
#
#   * The reasoning traces in cluster c
#   * The set of answers proposed by OTHER clusters (so the judge sees that
#     alternative numerical solutions exist, but not their full reasoning)
#
# The judge re-checks the reasoning in cluster c, decides whether it is sound,
# and emits a final answer in \boxed{...}. This produces one judge response
# per answer-cluster (K total). Each response is then scored by the PRM, and
# the final answer is selected by either highest-PRM or majority-vote across
# the K responses.
#
# Why a separate prompt builder?
#   The original prompts (full / concise / answer_only) ask the judge to pick
#   ONE answer from N traces all at once.  The per-cluster design instead
#   asks the judge to verify ONE candidate answer at a time, with knowledge
#   of competing alternatives.
#
# Variant semantics (per-cluster prompts only)
# --------------------------------------------
#   "full"        : pass every trace in the cluster, in full (no truncation).
#   "concise"     : pass only the single trace with the highest PRM score.
#   "answer_only" : pass NO traces — the judge sees only the candidate answer
#                   and the alternatives proposed by other clusters.
#
# The legacy `max_chars_per_trace` / `CIMR_MAX_TRACE_CHARS` knob is gone:
# trace content is now controlled entirely by the variant.

PER_CLUSTER_SYSTEM_PROMPT = (
    "You are a verifier. You are NOT solving the problem from scratch.\n"
    "Your only task is to check whether the candidate reasoning supports "
    "the candidate final answer.\n\n"
    "You must return EXACTLY one of these two formats, and nothing else:\n"
    "  VALID\n"
    "  INVALID: <one short sentence describing the concrete error>\n\n"
    "Hard rules:\n"
    "- Do NOT propose a new answer.\n"
    "- Do NOT compare with other answers.\n"
    "- Do NOT re-solve the problem.\n"
    "- Do NOT output \\boxed{}.\n"
    "- Do NOT output anything except the verdict line.\n"
    "- If the reasoning has no clear mathematical error and arrives at the "
    "candidate answer, output VALID.\n"
    "- If the reasoning contains a concrete mistake, invalid step, or "
    "contradiction, output INVALID followed by a short reason."
)


PER_CLUSTER_SYSTEM_PROMPT_ANSWER_ONLY = (
    "You are a verifier. You are NOT solving the problem from scratch.\n"
    "You will see only the candidate final answer (no reasoning is shown). "
    "Your only task is to judge whether the candidate answer is plausibly "
    "correct for the given problem.\n\n"
    "You must return EXACTLY one of these two formats, and nothing else:\n"
    "  VALID\n"
    "  INVALID: <one short sentence describing why it is wrong>\n\n"
    "Hard rules:\n"
    "- Do NOT propose a new answer.\n"
    "- Do NOT compare with other answers.\n"
    "- Do NOT re-solve the problem.\n"
    "- Do NOT output \\boxed{}.\n"
    "- Do NOT output anything except the verdict line."
)


# =============================================================================
# Few-shot examples — VALID / INVALID veto format
# =============================================================================
# Demonstrate the exact verdict format the judge must produce.  Examples
# carry no PRM scores, no rankings, and never propose alternatives.

_FEW_SHOT_EXAMPLES_FULL_CONCISE = """\
Here are short examples of the exact output format required.

#EXAMPLE 1
Problem: What is 7 + 5?
Candidate final answer: 12
Reasoning (excerpt): "7 + 5 = 12."
Verdict:
VALID

#EXAMPLE 2
Problem: What is the smallest prime greater than 20?
Candidate final answer: 21
Reasoning (excerpt): "21 comes right after 20, so 21 is prime."
Verdict:
INVALID: 21 = 3 x 7 is composite, not prime.

#EXAMPLE 3
Problem: Compute 6!.
Candidate final answer: 720
Reasoning (excerpt): "6! = 6*5*4*3*2*1 = 720."
Verdict:
VALID
"""


_FEW_SHOT_EXAMPLES_ANSWER_ONLY = """\
Here are short examples of the exact output format required.

#EXAMPLE 1
Problem: What is 7 + 5?
Candidate final answer: 12
Verdict:
VALID

#EXAMPLE 2
Problem: What is the smallest prime greater than 20?
Candidate final answer: 21
Verdict:
INVALID: 21 = 3 x 7 is composite.

#EXAMPLE 3
Problem: Compute 6!.
Candidate final answer: 720
Verdict:
VALID
"""


def build_per_cluster_judge_prompt(
    question:              str,
    cluster_answer:        str,
    cluster_traces:        List[str],
    other_cluster_answers: List[str],
    variant:               str = "full",
    cluster_prm_scores:    List[float] = None,
) -> str:
    """
    Build a VETO judge prompt for ONE answer-cluster.

    The judge's job is binary:  VALID  or  INVALID: <reason>.
    It does NOT propose answers and does NOT compare with other clusters.

    The `variant` parameter controls *what reasoning content* the judge sees:

      "full"        : every trace in the cluster, in full (no truncation).
      "concise"     : only the single trace with the highest PRM score.
      "answer_only" : no traces — only the candidate answer is shown.

    The prompt does NOT include any PRM scores, ranking information, mean/max
    statistics, or cluster sizes.  The `other_cluster_answers` argument is
    accepted for API compatibility but DELIBERATELY NOT shown to the judge,
    because the judge is no longer comparing clusters.

    cluster_prm_scores is used ONLY to pick the top trace under the
    "concise" variant; it is never shown to the judge.
    """
    if variant not in ("full", "concise", "answer_only"):
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose: 'full' | 'concise' | 'answer_only'"
        )

    # other_cluster_answers is intentionally unused (parked for API compat)
    _ = other_cluster_answers

    cand_display = cluster_answer.strip() if cluster_answer else "(none)"

    # ── answer_only: no reasoning shown ──────────────────────────────────
    if variant == "answer_only":
        lines = [
            "Decide whether the candidate final answer is plausibly correct "
            "for the given problem. Respond ONLY with `VALID` or "
            "`INVALID: <reason>`. Do not propose a new answer.",
            "",
            "#PROBLEM",
            question.strip(),
            "",
            "#CANDIDATE ANSWER",
            cand_display,
            "",
            _FEW_SHOT_EXAMPLES_ANSWER_ONLY.rstrip(),
            "",
            "#YOUR VERDICT",
            "Now produce one verdict line for the actual problem above. "
            "Output ONLY `VALID` or `INVALID: <reason>` and nothing else.",
        ]
        return "\n".join(lines)

    # ── full / concise: pick traces to include ───────────────────────────
    if variant == "concise":
        if cluster_traces:
            if cluster_prm_scores and len(cluster_prm_scores) == len(cluster_traces):
                best_k = int(max(
                    range(len(cluster_traces)),
                    key=lambda i: float(cluster_prm_scores[i]),
                ))
            else:
                best_k = 0
            selected_traces = [cluster_traces[best_k]]
        else:
            selected_traces = []
    else:  # full
        selected_traces = list(cluster_traces)

    lines = [
        "Decide whether the reasoning shown supports the candidate final "
        "answer. Respond ONLY with `VALID` or `INVALID: <reason>`. Do not "
        "propose a new answer. Do not re-solve the problem.",
        "",
        "#PROBLEM",
        question.strip(),
        "",
        "#CANDIDATE ANSWER",
        cand_display,
        "",
        "#REASONING",
    ]

    if not selected_traces:
        lines.append("(no reasoning supplied)")
    else:
        for k, trace in enumerate(selected_traces, start=1):
            body = str(trace).strip()
            lines += [
                f"##TRACE {k}",
                body,
                "",
            ]

    lines += [
        "",
        _FEW_SHOT_EXAMPLES_FULL_CONCISE.rstrip(),
        "",
        "#YOUR VERDICT",
        "Now produce one verdict line for the actual problem above. "
        "Output ONLY `VALID` or `INVALID: <reason>` and nothing else.",
    ]

    return "\n".join(lines)


# =============================================================================
# PER-TRACE veto + verify prompts  (NEW)
# =============================================================================
#
# The judge is shown ONE candidate trace at a time and must produce a strict
# two-line verdict:
#
#     VERIFY: YES   (or NO)
#     VALID         (or INVALID: <one-sentence reason>)
#
#   VERIFY  : does the trace's final boxed answer satisfy the problem
#             constraints?  This is a fast end-to-end check.
#   VALID   : is the reasoning that arrived at that answer free of
#             concrete mathematical errors?
#
# Both must be (YES, VALID) for a trace to be considered fully validated.
# Wrong-but-popular answers cannot get credit from frequency: every trace is
# judged on its own merits.
#
# Variant semantics at the trace level:
#   "full"        : show the trace text in full
#   "concise"     : show the trace text head/tail truncated to ~1500 chars
#   "answer_only" : show only the trace's extracted final answer
# =============================================================================


PER_TRACE_SYSTEM_PROMPT = (
    "You are a strict verifier for a single mathematical reasoning trace. "
    "You are NOT solving the problem from scratch.\n\n"
    "Your output MUST be exactly two lines, in this order, and nothing else:\n"
    "  Line 1: VERIFY: YES   or   VERIFY: NO\n"
    "  Line 2: VALID          or   INVALID: <one short sentence>\n\n"
    "Definitions:\n"
    "- VERIFY asks: does the trace's final boxed answer satisfy the "
    "problem's constraints? Reason briefly in your head, then answer YES/NO.\n"
    "- VALID asks: does the reasoning leading to that answer have any "
    "concrete mathematical mistake, invalid step, or contradiction?\n\n"
    "Hard rules:\n"
    "- Do NOT propose a new answer.\n"
    "- Do NOT compare with other traces.\n"
    "- Do NOT re-solve the problem.\n"
    "- Output ONLY the two verdict lines, nothing else.\n"
    "- Do NOT output \\boxed{}."
)


PER_TRACE_SYSTEM_PROMPT_ANSWER_ONLY = (
    "You are a strict verifier for a single mathematical answer. "
    "No reasoning is shown. You are NOT solving the problem from scratch.\n\n"
    "Your output MUST be exactly two lines, in this order, and nothing else:\n"
    "  Line 1: VERIFY: YES   or   VERIFY: NO\n"
    "  Line 2: VALID          or   INVALID: <one short sentence>\n\n"
    "Definitions:\n"
    "- VERIFY asks: does the candidate answer satisfy the problem's "
    "constraints? Reason briefly in your head, then answer YES/NO.\n"
    "- VALID asks: is the candidate answer plausibly correct? "
    "(Without reasoning, this is essentially the same check as VERIFY; "
    "still output both lines.)\n\n"
    "Hard rules:\n"
    "- Do NOT propose a new answer.\n"
    "- Do NOT compare with other answers.\n"
    "- Output ONLY the two verdict lines, nothing else.\n"
    "- Do NOT output \\boxed{}."
)


_PER_TRACE_FEW_SHOT_FULL_CONCISE = """\
Here are short examples of the exact two-line format required.

#EXAMPLE 1
Problem: What is 7 + 5?
Candidate trace (excerpt): "7 + 5 = 12"
Candidate answer: 12
Verdict:
VERIFY: YES
VALID

#EXAMPLE 2
Problem: What is the smallest prime greater than 20?
Candidate trace (excerpt): "21 comes right after 20, so 21."
Candidate answer: 21
Verdict:
VERIFY: NO
INVALID: 21 = 3 x 7 is composite, not prime.

#EXAMPLE 3
Problem: Compute 6!.
Candidate trace (excerpt): "6! = 6*5*4*3*2 = 360, so 360."
Candidate answer: 360
Verdict:
VERIFY: NO
INVALID: 6! includes the factor 1 only trivially, but the product was missing a 1; correct value is 720, not 360.
"""


_PER_TRACE_FEW_SHOT_ANSWER_ONLY = """\
Here are short examples of the exact two-line format required.

#EXAMPLE 1
Problem: What is 7 + 5?
Candidate answer: 12
Verdict:
VERIFY: YES
VALID

#EXAMPLE 2
Problem: What is the smallest prime greater than 20?
Candidate answer: 21
Verdict:
VERIFY: NO
INVALID: 21 = 3 x 7 is composite, not prime.

#EXAMPLE 3
Problem: Compute 6!.
Candidate answer: 720
Verdict:
VERIFY: YES
VALID
"""


def _truncate_trace_concise(trace: str, max_chars: int = 1500) -> str:
    """Head + tail truncation. Used for the `concise` variant at trace level."""
    text = str(trace).strip()
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    head = text[:keep].rstrip()
    tail = text[-keep:].lstrip()
    omit = len(text) - max_chars
    return f"{head}\n\n[... {omit:,} chars omitted ...]\n\n{tail}"


def build_per_trace_judge_prompt(
    question:       str,
    trace_answer:   str,
    trace_text:     str,
    variant:        str = "full",
) -> str:
    """
    Build the user-message body for a per-trace VERIFY+VETO judge call.

    Variant semantics (NEW — trace level):
      "full"        : full trace text
      "concise"     : trace text head/tail truncated to ~1500 chars
      "answer_only" : no trace text — only the candidate answer

    The judge must respond with exactly two lines:
      Line 1: VERIFY: YES   or   VERIFY: NO
      Line 2: VALID          or   INVALID: <reason>
    """
    if variant not in ("full", "concise", "answer_only"):
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose: 'full' | 'concise' | 'answer_only'"
        )

    cand_display = trace_answer.strip() if trace_answer else "(none)"

    # ── answer_only ──────────────────────────────────────────────────────
    if variant == "answer_only":
        lines = [
            "Verify the candidate answer for the problem below. Respond ONLY "
            "with the two-line verdict.",
            "",
            "#PROBLEM",
            question.strip(),
            "",
            "#CANDIDATE ANSWER",
            cand_display,
            "",
            _PER_TRACE_FEW_SHOT_ANSWER_ONLY.rstrip(),
            "",
            "#YOUR VERDICT",
            "Now output exactly two lines for the problem above:",
            "  Line 1: VERIFY: YES   or   VERIFY: NO",
            "  Line 2: VALID          or   INVALID: <one short sentence>",
        ]
        return "\n".join(lines)

    # ── full / concise ───────────────────────────────────────────────────
    if variant == "concise":
        trace_body = _truncate_trace_concise(trace_text)
    else:
        trace_body = str(trace_text).strip() or "(empty)"

    lines = [
        "Verify whether the reasoning trace below is correct and supports "
        "its candidate answer. Respond ONLY with the two-line verdict.",
        "",
        "#PROBLEM",
        question.strip(),
        "",
        "#CANDIDATE ANSWER",
        cand_display,
        "",
        "#REASONING TRACE",
        trace_body,
        "",
        _PER_TRACE_FEW_SHOT_FULL_CONCISE.rstrip(),
        "",
        "#YOUR VERDICT",
        "Now output exactly two lines for the problem above:",
        "  Line 1: VERIFY: YES   or   VERIFY: NO",
        "  Line 2: VALID          or   INVALID: <one short sentence>",
    ]
    return "\n".join(lines)


# =============================================================================
# RESOLVE-FROM-VALID prompt  (NEW selection method)
# =============================================================================
#
# After per-trace verification, gather every trace whose verdict was
# (VERIFY=YES, VALID).  Pass the curated set to the judge as a SINGLE prompt
# and ask it to produce one final answer to the problem.
#
# This selection rule is called `resolve_from_valid` and produces one new
# answer per question.  If no trace passed the strict validity gate, the
# caller should fall back to the best original-PRM trace's answer.

RESOLVE_FROM_VALID_SYSTEM_PROMPT = (
    "You are a math problem solver. You will be shown a problem and several "
    "reasoning traces that have all been verified to be correct.\n\n"
    "Your task is to produce ONE final answer to the problem.  You may "
    "borrow steps from any of the verified traces.\n\n"
    "Hard rules:\n"
    "- Be concise. A short paragraph of analysis is plenty.\n"
    "- Do not repeat sentences or phrases.\n"
    "- The very last line must be exactly: \\boxed{<answer>}"
)


_RESOLVE_FEW_SHOT = """\
Here is an example of the exact output shape required.

#EXAMPLE
Problem: What is 7 + 5?
Verified trace 1: "7 + 5 = 12"
Verified trace 2: "Counting up from 7 by 5 gives 12."
Response:
Both verified traces agree that 7 + 5 = 12.
\\boxed{12}
"""


def build_resolve_from_valid_prompt(
    question:       str,
    valid_traces:   List[str],
) -> str:
    """
    Build the user-message body for a resolve-from-valid generative call.

    `valid_traces` is the (curated, possibly empty) list of trace texts that
    passed the strict verify-and-validate gate. The prompt asks the model to
    produce one consolidated final answer.
    """
    lines = [
        "Below is a math problem and a list of verified reasoning traces. "
        "Produce ONE final answer to the problem. End your response with "
        "one line: \\boxed{<answer>}.",
        "",
        "#PROBLEM",
        question.strip(),
        "",
        "#VERIFIED TRACES",
    ]
    if not valid_traces:
        lines.append("(no verified traces were available)")
    else:
        for k, t in enumerate(valid_traces, start=1):
            body = str(t).strip() or "(empty)"
            lines += [
                f"##VERIFIED TRACE {k}",
                body,
                "",
            ]

    lines += [
        "",
        _RESOLVE_FEW_SHOT.rstrip(),
        "",
        "#YOUR FINAL ANSWER",
        "Now produce the final answer for the actual problem above. "
        "End with one line: \\boxed{<answer>}.",
    ]
    return "\n".join(lines)


# =============================================================================
# CROSS-EXAMINATION prompt  (NEW — per cluster)
# =============================================================================
#
# After per-trace VERIFY+VALID veto, every cluster has a mix of VALID and
# INVALID traces.  The cross-examination judge takes ALL of them and produces
# the cluster's final answer.
#
# CRITICAL bias-mitigation requirement: do NOT defer to the per-trace labels.
# The previous judge call can be wrong in BOTH directions:
#   - a VALID-labeled trace may actually be wrong
#   - an INVALID-labeled trace may actually be correct
# The cross-examiner must verify the mathematics independently.
#
# Prompt design choices to enforce this:
#   1. Labels are described as "preliminary checks", not as ground truth.
#   2. The prompt explicitly says both kinds of label errors are common.
#   3. Three balanced few-shot examples cover (a) labels right, (b) VALID
#      label was wrong, (c) INVALID label was wrong.
#   4. The judge is told to verify the math, not to count labels.
#
# Output format: short analysis followed by \boxed{<answer>} on the last line.
# =============================================================================


CROSS_EXAMINE_SYSTEM_PROMPT = (
    "You are a careful mathematical answer checker. You will be shown a "
    "problem and several reasoning traces, each carrying a preliminary "
    "label of VALID or INVALID from a separate, fast verifier.\n\n"
    "Important: those preliminary labels are NOT ground truth and are wrong "
    "often enough that you must verify the mathematics independently. "
    "BOTH of these failure modes are common:\n"
    "  - a trace labeled VALID can still arrive at a WRONG answer\n"
    "  - a trace labeled INVALID can be CORRECT (the verifier was mistaken)\n\n"
    "Your task is to determine the correct answer to the problem by checking "
    "the math yourself, not by counting labels or by trusting any single "
    "trace. You may borrow steps from any trace (VALID or INVALID), and you "
    "may also reach a different answer than any of them if your own check "
    "shows they are all wrong.\n\n"
    "Hard rules:\n"
    "- Be brief. A short paragraph of analysis is plenty.\n"
    "- Do not repeat sentences or phrases.\n"
    "- Do not output reasoning past the final answer.\n"
    "- The very last line must be exactly: \\boxed{<answer>}"
)


_CROSS_EXAMINE_FEW_SHOT = """\
Here are short examples that show how to weigh traces independently of
their preliminary labels.

#EXAMPLE 1  (labels right; both VALID, both correct)
Problem: What is 7 + 5?
[Label: VALID]    Trace 1: "7 + 5 = 12"
[Label: VALID]    Trace 2: "Counting up from 7 by 5 gives 12"
Response:
Both traces compute 7+5=12 and the arithmetic checks out.
\\boxed{12}

#EXAMPLE 2  (a VALID label was WRONG)
Problem: What is the smallest prime greater than 20?
[Label: VALID]    Trace 1: "21 is right after 20, so 21."
[Label: INVALID]  Trace 2: "21 is composite; the next prime is 23."
Response:
Trace 1 was labeled VALID but 21 = 3 x 7 is composite, so its answer 21 is wrong. Trace 2's reasoning is correct: the smallest prime above 20 is 23.
\\boxed{23}

#EXAMPLE 3  (an INVALID label was WRONG)
Problem: Compute 6!.
[Label: INVALID]  Trace 1: "6 x 5 x 4 x 3 x 2 x 1 = 720."
[Label: VALID]    Trace 2: "6! = 6 x 5 x 4 x 3 x 2 = 360."
Response:
Trace 1 is actually correct: 6! = 6 x 5 x 4 x 3 x 2 x 1 = 720. The INVALID label was wrong. Trace 2 dropped the factor 1 (effectively miscomputed); its 360 is incorrect.
\\boxed{720}
"""


def build_cross_examine_prompt(
    question:         str,
    cluster_answer:   str,
    judged_traces:    List[Dict[str, Any]],
) -> str:
    """
    Build a cross-examination prompt for ONE cluster.

    `judged_traces` is the list of (top-k) traces in this cluster that were
    veto-judged in Phase A.  Each entry must contain:
        trace_text      : the reasoning text
        verdict_label   : "VALID" | "INVALID" | "UNCLEAR"
        verify_label    : "YES"   | "NO"      | "UNCLEAR"
        error_message   : the INVALID reason, if any
        is_valid_strict : bool

    The prompt frames each trace with a "[Label: VALID]" or "[Label: INVALID:
    <reason>]" tag, but the surrounding instructions and few-shot examples
    make clear that the cross-examiner must verify the math independently.

    Note: `cluster_answer` is shown for context only.  The cross-examiner is
    NOT instructed to confirm it; its job is to find the correct answer for
    the problem, regardless of which answer the cluster carries.
    """
    cand_display = (cluster_answer or "").strip() or "(none)"
    lines = [
        "Determine the correct final answer to the problem below by "
        "cross-examining the reasoning traces. Do NOT defer to the labels — "
        "verify the math yourself. End with one line: \\boxed{<answer>}.",
        "",
        "#PROBLEM",
        question.strip(),
        "",
        "#CLUSTER CANDIDATE ANSWER (the answer the traces below all reach)",
        cand_display,
        "",
        "#TRACES UNDER REVIEW",
    ]

    if not judged_traces:
        lines.append("(no judged traces in this cluster)")
    else:
        for k, jt in enumerate(judged_traces, start=1):
            v_label = str(jt.get("verdict_label", "UNCLEAR")).upper()
            verify_label = str(jt.get("verify_label", "UNCLEAR")).upper()
            err     = str(jt.get("error_message", "") or "").strip()
            body    = str(jt.get("trace_text", "") or "").strip() or "(empty)"
            if v_label == "INVALID":
                tag_inner = f"INVALID: {err}" if err else "INVALID"
            else:
                tag_inner = v_label
            label_tag = (
                f"[Preliminary label: {tag_inner};  "
                f"verify-answer-check: {verify_label}]"
            )
            lines += [
                f"##TRACE {k}  {label_tag}",
                body,
                "",
            ]

    lines += [
        "",
        _CROSS_EXAMINE_FEW_SHOT.rstrip(),
        "",
        "#YOUR FINAL ANSWER",
        "Now produce the final answer for the actual problem above. Verify "
        "the math yourself; do not just count labels. End with one line: "
        "\\boxed{<answer>}.",
    ]
    return "\n".join(lines)
