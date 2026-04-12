"""
dataset_loader.py
-----------------
Unified dataset loading for math reasoning benchmarks.

Ground-truth answer extraction
-------------------------------
Every dataset stores answers differently.  This module handles the
extraction explicitly per dataset so the `answer` field in every
returned example is always a clean, stripped final answer string
(e.g. "2", "\\frac{1}{3}", "16") — never a full solution paragraph.

The extraction pipeline for each dataset:
  math500      : extract_answer(solution_text, "math")   via parser.py
  aime24/25    : row["Answer"]                            (already clean int)
  amc23        : row["Answer"]                            (already clean)
  gsm8k        : row["answer"].split("####")[-1].strip()
  olympiadbench: row["final_answer"][0].strip("$")
  college_math : row["answer"]
  minerva_math : row["answer"]
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
        "hf_path":    "HuggingFaceH4/MATH-500",
        "hf_name":    "default",
        "split":      "test",
        "question_key": "problem",
        # answer extracted by _extract_gt, not read directly
    },
    "aime24": {
        "hf_path":    "Maxwell-Jia/AIME_2024",
        "hf_name":    None,
        "split":      "train",
        "question_key": "Problem",
    },
    "aime25": {
        "hf_path":    "math-ai/aime25",
        "hf_name":    None,
        "split":      "test",
        "question_key": "Problem",
    },
    "amc23": {
        "hf_path":    "Maxwell-Jia/AMC_2023",
        "hf_name":    None,
        "split":      "train",
        "question_key": "Problem",
    },
    "gsm8k": {
        "hf_path":    "openai/gsm8k",
        "hf_name":    "main",
        "split":      "test",
        "question_key": "question",
    },
    "olympiadbench": {
        "hf_path":    "lmms-lab/OlympiadBench",
        "hf_name":    "OE_TO_maths_en_COMP",
        "split":      "test",
        "question_key": "question",
    },
    "college_math": {
        "hf_path":    "TIGER-Lab/college-math",
        "hf_name":    None,
        "split":      "test",
        "question_key": "problem",
    },
    "minerva_math": {
        "hf_path":    "math-ai/minerva_math",
        "hf_name":    None,
        "split":      "test",
        "question_key": "problem",
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
    """

    # ── math500: solution column contains the full worked solution ─────────────
    if data_name in ("math500", "math"):
        solution = str(row.get("solution", ""))
        return _extract_boxed_from_solution(solution)

    # ── minerva_math: same structure as math500 ────────────────────────────────
    if data_name == "minerva_math":
        solution = str(row.get("answer", row.get("solution", "")))
        if "\\boxed" in solution or "boxed" in solution:
            return _extract_boxed_from_solution(solution)
        return solution.strip("$").strip()

    # ── aime24 / aime25 / amc23: Answer field is already a clean integer/string
    if data_name in ("aime24", "aime25", "amc23"):
        return str(row.get("Answer", "")).strip()

    # ── gsm8k: "#### <number>" at the end of the answer field ─────────────────
    if data_name == "gsm8k":
        raw = str(row.get("answer", ""))
        if "####" in raw:
            return raw.split("####")[-1].strip()
        return raw.strip()

    # ── olympiadbench: final_answer is a list ──────────────────────────────────
    if data_name == "olympiadbench":
        fa = row.get("final_answer", [""])
        val = fa[0] if isinstance(fa, list) and fa else str(fa)
        return str(val).strip("$").strip()

    # ── college_math ───────────────────────────────────────────────────────────
    if data_name == "college_math":
        ans = str(row.get("answer", "")).strip()
        # some entries are like "answer: 3" or "$3$"
        ans = ans.strip("$").strip()
        if ans.lower().startswith("answer:"):
            ans = ans[len("answer:"):].strip()
        return ans

    # ── generic fallback: try common key names ─────────────────────────────────
    for key in ("answer", "Answer", "solution", "target", "label"):
        val = row.get(key)
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