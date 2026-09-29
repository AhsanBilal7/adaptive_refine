"""
dataset_loader.py
-----------------
Unified dataset loading for math reasoning and general reasoning benchmarks.

Ground-truth answer extraction
-------------------------------
Every dataset stores answers differently.  This module handles the
extraction explicitly per dataset so the `answer` field in every
returned example is always a clean, stripped final answer string
(e.g. "2", "\\frac{1}{3}", "16", "A", "yes") — never a full solution paragraph.

The extraction pipeline for each dataset:
  math500      : extract_answer(solution_text, "math")   via parser.py
  aime24/25    : row["Answer"]                            (already clean int)
  amc23        : row["Answer"]                            (already clean)
  gsm8k        : row["answer"].split("####")[-1].strip()
  olympiadbench: row["final_answer"][0].strip("$")
  college_math : row["answer"]
  minerva_math : row["answer"]
  mmlu         : ABCD[row["answer"]]  (int 0-3 mapped to letter)
                 question rebuilt as "{stem}\\nA. ...\\nB. ...\\nC. ...\\nD. ..."
  bbh          : row["target"]        (free-form string, used directly)
  arc_challenge: row["answerKey"]     (letter A-D; normalised if given as 1-4)
                 question rebuilt with labelled choices appended
  truthfulqa   : row["mc1_targets"]["choices"][labels.index(1)]  (correct choice text)
  strategyqa   : "yes" or "no"  (converted from bool row["answer"])
  custom       : row["answer"] or row["solution"]
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace dataset registry
# ──────────────────────────────────────────────────────────────────────────────

_HF_CONFIGS: Dict[str, Dict[str, Any]] = {
    "math500": {
        "hf_path":      "HuggingFaceH4/MATH-500",
        "hf_name":      "default",
        "split":        "test",
        "question_key": "problem",
        "answer_key":   "solution",      # full worked solution; \boxed{} extracted
    },
    "aime24": {
        "hf_path":      "Maxwell-Jia/AIME_2024",
        "hf_name":      None,
        "split":        "train",
        "question_key": "Problem",
        "answer_key":   "Answer",        # clean integer string, used directly
    },
    "aime25": {
        "hf_path":      "math-ai/aime25",
        "hf_name":      None,
        "split":        "test",
        "question_key": "problem",
        "answer_key":   "answer",        # clean integer string, used directly
    },
    "amc23": {
        "hf_path":      "math-ai/amc23",
        "hf_name":      None,
        "split":        "test",
        "question_key": "question",
        "answer_key":   "answer",        # clean string, used directly
    },
    "gsm8k": {
        "hf_path":      "openai/gsm8k",
        "hf_name":      "main",
        "split":        "test",
        "question_key": "question",
        "answer_key":   "answer",        # contains "#### <number>"; split on ####
    },
    "olympiadbench": {
        "hf_path":      "math-ai/olympiadbench",
        "hf_name":      None,
        "split":        "test",
        "question_key": "question",
        "answer_key":   "final_answer",  # list; take [0] and strip $
    },
    "college_math": {
        "hf_path":      "TIGER-Lab/college-math",
        "hf_name":      None,
        "split":        "test",
        "question_key": "problem",
        "answer_key":   "answer",        # may be prefixed "answer: ..." or wrapped in $
    },
    "minerva_math": {
        "hf_path":      "math-ai/minerva_math",
        "hf_name":      None,
        "split":        "test",
        "question_key": "problem",
        "answer_key":   "answer",        # may contain \boxed{}; extracted if present
    },
    # ── General reasoning / knowledge benchmarks ──────────────────────────────
    "mmlu": {
        # Massive Multitask Language Understanding (Hendrycks et al., 2021)
        # 57 subjects, multiple-choice A/B/C/D.
        # answer field: int (0-3); we map to the letter A/B/C/D.
        # question field contains only the stem; choices are a separate list.
        # We build the full question string here:
        #   "{stem}\nA. {choice0}\nB. {choice1}\nC. {choice2}\nD. {choice3}"
        "hf_path":      "cais/mmlu",
        "hf_name":      "all",
        "split":        "test",
        "question_key": "question",      # raw stem; choices appended in _extract_gt
        "answer_key":   "answer",        # int 0-3 → mapped to "A"/"B"/"C"/"D"
    },
    "bbh": {
        # BIG-Bench Hard (Suzgun et al., 2022) — 23 challenging reasoning tasks.
        # question_key: "input"   (the task prompt, already self-contained)
        # answer_key:   "target"  (free-form string, used directly)
        "hf_path":      "lukaemon/bbh",
        "hf_name":      "all",
        "split":        "test",
        "question_key": "input",
        "answer_key":   "target",        # string; used directly
    },
    "arc_challenge": {
        # ARC Challenge set (Clark et al., 2018) — science MCQ, harder subset.
        # answerKey is a single letter "A"/"B"/"C"/"D" (or "1"/"2"/"3"/"4").
        # We also format the full question with labelled choices.
        "hf_path":      "allenai/ai2_arc",
        "hf_name":      "ARC-Challenge",
        "split":        "test",
        "question_key": "question",      # stem; choices appended in _extract_gt
        "answer_key":   "answerKey",     # "A"/"B"/"C"/"D" or "1"/"2"/"3"/"4"
    },
    "truthfulqa": {
        # TruthfulQA MC1 split (Lin et al., 2022) — 817 questions.
        # mc1_targets is a dict: {"choices": [...], "labels": [...]}
        # Exactly one label == 1; that choice is the correct answer.
        "hf_path":      "truthfulqa/truthful_qa",
        "hf_name":      "multiple_choice",
        "split":        "validation",
        "question_key": "question",
        "answer_key":   "mc1_targets",   # dict; correct = choices[labels.index(1)]
    },
    "strategyqa": {
        # StrategyQA (Geva et al., 2021) — yes/no questions requiring multi-step
        # reasoning via implicit strategy decomposition.
        # answer field: bool (True/False) → normalised to "yes"/"no".
        "hf_path":      "wics/strategy-qa",
        "hf_name":      None,
        "split":        "test",
        "question_key": "question",
        "answer_key":   "answer",        # bool → "yes" / "no"
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset ground-truth extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_gt(row: Dict, data_name: str) -> str:
    """
    Return the clean final-answer string for one dataset row.

    For datasets whose answer_key already contains a clean value (AIME, GSM8K…)
    we read it directly.  For math500 / minerva_math the answer is buried inside
    a full solution string with \\boxed{}, so we call extract_answer() from
    parser.py to pull it out.

    The answer_key for each dataset is declared in _HF_CONFIGS and used here
    as the primary field lookup so the registry and extraction stay in sync.
    """
    cfg = _HF_CONFIGS.get(data_name, {})
    answer_key = cfg.get("answer_key", "answer")

    # ── math500: solution column contains the full worked solution ─────────────
    if data_name in ("math500", "math"):
        solution = str(row.get(answer_key, ""))
        return _extract_boxed_from_solution(solution)

    # ── minerva_math: same structure as math500 ────────────────────────────────
    if data_name == "minerva_math":
        solution = str(row.get(answer_key, row.get("solution", "")))
        if "\\boxed" in solution or "boxed" in solution:
            return _extract_boxed_from_solution(solution)
        return solution.strip("$").strip()

    # ── aime24 / amc23: Answer field is a clean integer string ────────────────
    if data_name in ("aime24", "amc23"):
        return str(row.get(answer_key, "")).strip()

    # ── aime25: answer field (lowercase) is a clean integer string ────────────
    if data_name == "aime25":
        return str(row.get(answer_key, "")).strip()

    # ── gsm8k: "#### <number>" at the end of the answer field ─────────────────
    if data_name == "gsm8k":
        raw = str(row.get(answer_key, ""))
        if "####" in raw:
            return raw.split("####")[-1].strip()
        return raw.strip()

    # ── olympiadbench: final_answer is a list ──────────────────────────────────
    if data_name == "olympiadbench":
        fa = row.get(answer_key, [""])
        val = fa[0] if isinstance(fa, list) and fa else str(fa)
        return str(val).strip("$").strip()

    # ── college_math ───────────────────────────────────────────────────────────
    if data_name == "college_math":
        ans = str(row.get(answer_key, "")).strip()
        ans = ans.strip("$").strip()
        if ans.lower().startswith("answer:"):
            ans = ans[len("answer:"):].strip()
        return ans

    # ── mmlu ───────────────────────────────────────────────────────────────────
    # answer is an int 0-3 indexing into the choices list.
    # We map it to the corresponding letter "A"/"B"/"C"/"D".
    if data_name == "mmlu":
        idx = row.get(answer_key)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return str(idx).strip()
        return ["A", "B", "C", "D"][idx] if 0 <= idx <= 3 else str(idx)

    # ── bbh ────────────────────────────────────────────────────────────────────
    # target is a free-form string; use directly.
    if data_name == "bbh":
        return str(row.get(answer_key, "")).strip()

    # ── arc_challenge ──────────────────────────────────────────────────────────
    # answerKey is "A"/"B"/"C"/"D" or "1"/"2"/"3"/"4".
    # Normalise numeric keys to letters.
    if data_name == "arc_challenge":
        key = str(row.get(answer_key, "")).strip()
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
        return num_to_letter.get(key, key)

    # ── truthfulqa ─────────────────────────────────────────────────────────────
    # mc1_targets is {"choices": [...], "labels": [...]}
    # The single entry with label == 1 is the correct answer.
    # We return the choice TEXT so the model can match it verbatim.
    if data_name == "truthfulqa":
        mc1 = row.get(answer_key, {})
        choices = mc1.get("choices", [])
        labels  = mc1.get("labels", [])
        try:
            correct_idx = labels.index(1)
            return str(choices[correct_idx]).strip()
        except (ValueError, IndexError):
            return str(choices[0]).strip() if choices else ""

    # ── strategyqa ─────────────────────────────────────────────────────────────
    # answer is a Python bool (True/False).  Normalise to "yes"/"no".
    if data_name == "strategyqa":
        raw = row.get(answer_key)
        if isinstance(raw, bool):
            return "yes" if raw else "no"
        s = str(raw).strip().lower()
        if s in ("true", "yes", "1"):
            return "yes"
        if s in ("false", "no", "0"):
            return "no"
        return s

    # ── generic fallback: use answer_key from registry, then common names ──────
    val = row.get(answer_key)
    if val is None:
        for key in ("answer", "Answer", "solution", "target", "label"):
            val = row.get(key)
            if val is not None:
                break
    if val is not None:
        raw = str(val).strip()
        if "####" in raw:
            return raw.split("####")[-1].strip()
        if "\\boxed" in raw or "boxed" in raw:
            return _extract_boxed_from_solution(raw)
        return raw.strip("$").strip()

    return ""


def _extract_boxed_from_solution(solution: str) -> str:
    """
    Extract the content of the last \\boxed{} in a solution string.

    Pipeline:
      1. Try parser.extract_answer (handles edge cases, strip_string, etc.)
      2. Fall back to a direct \\boxed{} parser.
      3. Return empty string if neither works.
    """
    # Try parser.py's extract_answer first (handles all math formatting edge cases)
    try:
        from parser import extract_answer  # type: ignore
        result = extract_answer(solution, "math")
        if result and result != "NONE":
            return result.strip()
    except Exception:
        pass

    # Direct \\boxed{} extraction fallback
    return _parse_boxed(solution)


def _parse_boxed(text: str) -> str:
    """Extract content of the last \\boxed{} with brace-matching."""
    if "boxed" not in text:
        return ""
    ans = text.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack, result = 1, ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                result += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                result += c
            else:
                result += c
        return result.strip()
    # no brace — take up to next $
    return ans.split("$")[0].strip()


def _build_mcq_question(stem: str, choices: list, labels: list = None) -> str:
    """
    Format a multiple-choice question as a self-contained prompt string.

    Produces:
        "{stem}
         A. {choices[0]}
         B. {choices[1]}
         C. {choices[2]}
         D. {choices[3]}"

    Parameters
    ----------
    stem    : raw question text
    choices : list of answer-option strings
    labels  : optional label list (e.g. ["A","B","C","D"]).
              If None, defaults to A/B/C/D/E/F/G/H.
    """
    default_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    if labels is None:
        labels = default_labels[:len(choices)]
    lines = [stem.strip()]
    for label, choice in zip(labels, choices):
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset_split(
    data_name: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    custom_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load a benchmark and return normalised examples.

    Each example contains:
        question   (str)  – problem text
        answer     (str)  – clean final answer  (e.g. "2", "\\frac{1}{3}")
        _data_name (str)  – canonical dataset name
        _raw       (dict) – original HuggingFace row

    Parameters
    ----------
    data_name   : key from _HF_CONFIGS, or "custom"
    split       : override the default split
    max_samples : cap on number of examples
    custom_path : path to local JSON/JSONL (required for data_name=="custom")
    cache_dir   : HuggingFace cache directory
    """
    if data_name == "custom":
        return _load_custom(custom_path, max_samples)

    if data_name not in _HF_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{data_name}'. "
            f"Choose from: {sorted(_HF_CONFIGS)} or 'custom'."
        )

    cfg = _HF_CONFIGS[data_name]
    effective_split = split or cfg["split"]

    try:
        from datasets import load_dataset as hf_load  # type: ignore
        kwargs: Dict[str, Any] = {"split": effective_split}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if cfg.get("hf_name"):
            raw_ds = hf_load(cfg["hf_path"], cfg["hf_name"], **kwargs)
        else:
            raw_ds = hf_load(cfg["hf_path"], **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load '{data_name}' from HuggingFace Hub: {e}"
        ) from e

    examples: List[Dict[str, Any]] = []
    n_empty = 0

    for row in raw_ds:
        row_dict = dict(row)
        question = str(row_dict.get(cfg["question_key"], "")).strip()
        if not question:
            continue

        # ── MCQ datasets: append labelled choices to the question stem ─────────
        # MMLU and ARC-Challenge store choices separately from the question stem.
        # We build a self-contained prompt so the model sees all options.
        if data_name == "mmlu":
            choices = row_dict.get("choices", [])
            if choices:
                question = _build_mcq_question(question, choices)

        elif data_name == "arc_challenge":
            ch = row_dict.get("choices", {})
            ch_texts  = ch.get("text",  []) if isinstance(ch, dict) else []
            ch_labels = ch.get("label", []) if isinstance(ch, dict) else []
            if ch_texts:
                question = _build_mcq_question(question, ch_texts,
                                               labels=ch_labels if ch_labels else None)

        answer = _extract_gt(row_dict, data_name)

        if not answer:
            n_empty += 1
            logger.debug(f"Empty answer for row: {row_dict}")

        examples.append({
            "question":   question,
            "answer":     answer,
            "_data_name": data_name,
            "_raw":       row_dict,
        })

        if max_samples and len(examples) >= max_samples:
            break

    if n_empty:
        logger.warning(
            f"{n_empty}/{len(examples)} examples had empty answers — "
            f"check the dataset fields."
        )

    logger.info(
        f"Loaded {len(examples)} examples from '{data_name}' "
        f"({effective_split} split).  "
        f"Sample answers: {[e['answer'] for e in examples[:3]]}"
    )
    return examples


def _load_custom(path: Optional[str], max_samples: Optional[int]) -> List[Dict]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Custom file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = json.load(f)
            if isinstance(rows, dict):
                rows = list(rows.values())

    examples = []
    for row in rows:
        question = str(row.get("question", row.get("problem", ""))).strip()
        # For custom files: support both clean answer and solution-with-boxed
        raw_ans = str(row.get("answer", row.get("solution", ""))).strip()
        if "\\boxed" in raw_ans or "boxed" in raw_ans:
            answer = _extract_boxed_from_solution(raw_ans)
        else:
            answer = raw_ans
        if question:
            examples.append({
                "question":   question,
                "answer":     answer,
                "_data_name": "custom",
                "_raw":       row,
            })
        if max_samples and len(examples) >= max_samples:
            break

    logger.info(f"Loaded {len(examples)} custom examples from {path}.")
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# CLI smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--max_samples", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    exs = load_dataset_split(args.dataset, max_samples=args.max_samples)
    for i, ex in enumerate(exs):
        print(f"\n[{i}] Q: {ex['question'][:100]}...")
        print(f"    A (raw):    {ex['_raw'].get('solution', ex['_raw'].get('Answer', ''))[:80]}")
        print(f"    A (clean):  {ex['answer']}")