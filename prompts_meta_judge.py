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
    "You are a careful math answer checker. "
    "You will be shown a problem, a candidate final answer, and reasoning that supports it. "
    "Decide what the correct final answer to the problem is, then output it once.\n\n"
    "Hard rules:\n"
    "- Be brief. At most a short paragraph of analysis.\n"
    "- Do not repeat sentences or phrases. Never restart the analysis mid-response.\n"
    "- Do not output reasoning past the final answer.\n"
    "- The very last line must be exactly: \\boxed{<answer>}"
)


PER_CLUSTER_SYSTEM_PROMPT_ANSWER_ONLY = (
    "You are a careful math answer checker. "
    "You will be shown a problem, a candidate final answer, and a few alternative answers. "
    "No reasoning is provided. "
    "Decide what the correct final answer is and output it once.\n\n"
    "Hard rules:\n"
    "- Be brief. At most a few short sentences.\n"
    "- Do not repeat sentences or phrases. Never restart the analysis mid-response.\n"
    "- Do not output reasoning past the final answer.\n"
    "- The very last line must be exactly: \\boxed{<answer>}"
)


# =============================================================================
# Few-shot examples
# =============================================================================
# These are appended near the end of every per-cluster prompt so the judge
# sees concrete demonstrations of the expected output shape: a short analysis
# followed by a single final \boxed{...} line.  No PRM scores, no rankings,
# and no mention of "cluster" — just answer-checking shape.
#
# Three small examples cover:
#   (1) candidate is correct  → confirm
#   (2) candidate is wrong, an alternative is right  → switch to alternative
#   (3) no alternative shown   → confirm or correct on its own merits

_FEW_SHOT_EXAMPLES_FULL_CONCISE = """\
Here are short examples that show the exact output shape required.

#EXAMPLE 1
Problem: What is 7 + 5?
Candidate final answer: 12
Reasoning (excerpt): "7 + 5 equals 12."
Other answers proposed: 11
Response:
The arithmetic 7 + 5 = 12 is correct.
\\boxed{12}

#EXAMPLE 2
Problem: What is the smallest prime greater than 20?
Candidate final answer: 21
Reasoning (excerpt): "21 comes right after 20, so 21."
Other answers proposed: 23
Response:
21 = 3 x 7 is composite, so the candidate is wrong. The smallest prime above 20 is 23.
\\boxed{23}

#EXAMPLE 3
Problem: Compute 6!.
Candidate final answer: 720
Reasoning (excerpt): "6! = 6*5*4*3*2*1 = 720."
Other answers proposed: (none)
Response:
6! = 720 is correct.
\\boxed{720}
"""


_FEW_SHOT_EXAMPLES_ANSWER_ONLY = """\
Here are short examples that show the exact output shape required.

#EXAMPLE 1
Problem: What is 7 + 5?
Candidate final answer: 12
Other answers proposed: 11
Response:
7 + 5 = 12, so the candidate is correct.
\\boxed{12}

#EXAMPLE 2
Problem: What is the smallest prime greater than 20?
Candidate final answer: 21
Other answers proposed: 23
Response:
21 is composite (3 x 7), so 23 is correct.
\\boxed{23}

#EXAMPLE 3
Problem: Compute 6!.
Candidate final answer: 720
Other answers proposed: (none)
Response:
6! = 720.
\\boxed{720}
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
    Build a judge prompt for ONE answer-cluster.

    The `variant` parameter controls *what reasoning content* the judge sees:

      "full"        : every trace in the cluster, in full (no truncation).
      "concise"     : only the single trace with the highest PRM score in
                      the cluster, in full.
      "answer_only" : no traces — the judge sees only the candidate answer
                      and the answers proposed by other clusters.

    The prompt deliberately does NOT include any PRM scores, ranking
    information, mean/max statistics, or cluster sizes.  The judge sees only
    the problem, the candidate answer, the supporting reasoning (when
    applicable), and the alternative answers, framed by simple `#HEADING`
    sections and three short worked examples.

    cluster_prm_scores is used internally to pick the top trace under the
    "concise" variant; it is NOT shown to the judge.
    """
    if variant not in ("full", "concise", "answer_only"):
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose: 'full' | 'concise' | 'answer_only'"
        )

    # Deduplicate and tidy the alternatives
    others = [str(a).strip() for a in other_cluster_answers if str(a).strip()]
    others = list(dict.fromkeys(others))
    others_display = ", ".join(others) if others else "(none)"
    cand_display   = cluster_answer.strip() if cluster_answer else "(none)"

    # ── answer_only ──────────────────────────────────────────────────────
    if variant == "answer_only":
        lines = [
            "Decide what the correct final answer to the problem is. "
            "End your response with one line: \\boxed{<answer>}.",
            "",
            "#PROBLEM",
            question.strip(),
            "",
            "#CANDIDATE ANSWER",
            cand_display,
            "",
            "#OTHER ANSWERS PROPOSED",
            others_display,
            "",
            _FEW_SHOT_EXAMPLES_ANSWER_ONLY.rstrip(),
            "",
            "#YOUR TASK",
            "Now answer the actual problem above. Keep it brief, do not "
            "repeat yourself, and end with one line: \\boxed{<answer>}.",
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
        "Check the candidate answer against the reasoning shown. Decide "
        "what the correct final answer to the problem is. End your response "
        "with one line: \\boxed{<answer>}.",
        "",
        "#PROBLEM",
        question.strip(),
        "",
        "#CANDIDATE ANSWER",
        cand_display,
        "",
        "#OTHER ANSWERS PROPOSED",
        others_display,
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
        "#YOUR TASK",
        "Now answer the actual problem above. Keep it brief, do not repeat "
        "yourself, and end with one line: \\boxed{<answer>}.",
    ]

    return "\n".join(lines)
