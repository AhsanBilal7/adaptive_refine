"""Loading of the math benchmarks from the Hugging Face Hub.

``load_dataset_split(name)`` returns one dict per problem with the problem text
(``question``) and its clean final answer (``answer``), for example ``"2"`` or
``"\\frac{1}{3}"``, never a full worked solution.  Each benchmark stores its
answer differently:

  math500        last ``\\boxed{}`` of the reference solution
  aime24, amc23  ``Answer`` / ``answer`` field (already clean)
  aime25         ``answer`` field (already clean)
  gsm8k          text after ``####`` in the reference solution
  olympiadbench  first element of ``final_answer``, without ``$``

LiveBench and NATURAL PLAN are loaded by ``tasks.py``.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HF_CONFIGS: Dict[str, Dict[str, Any]] = {
    "math500": dict(hf_path="HuggingFaceH4/MATH-500", hf_name="default", split="test",
                    question_key="problem", answer_key="solution"),
    "aime24": dict(hf_path="Maxwell-Jia/AIME_2024", hf_name=None, split="train",
                   question_key="Problem", answer_key="Answer"),
    "aime25": dict(hf_path="math-ai/aime25", hf_name=None, split="test",
                   question_key="problem", answer_key="answer"),
    "amc23": dict(hf_path="math-ai/amc23", hf_name=None, split="test",
                  question_key="question", answer_key="answer"),
    "gsm8k": dict(hf_path="openai/gsm8k", hf_name="main", split="test",
                  question_key="question", answer_key="answer"),
    "olympiadbench": dict(hf_path="math-ai/olympiadbench", hf_name=None, split="test",
                          question_key="question", answer_key="final_answer"),
}


def parse_boxed(text: str) -> str:
    """Content of the last ``\\boxed{}`` in ``text`` (nested braces handled)."""
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
    return ans.split("$")[0].strip()


def gold_answer(row: Dict[str, Any], data_name: str) -> str:
    """Clean final answer of one dataset row."""
    key = HF_CONFIGS[data_name]["answer_key"]
    if data_name == "math500":
        return parse_boxed(str(row.get(key, "")))
    if data_name == "gsm8k":
        raw = str(row.get(key, ""))
        return raw.split("####")[-1].strip() if "####" in raw else raw.strip()
    if data_name == "olympiadbench":
        fa = row.get(key, [""])
        val = fa[0] if isinstance(fa, list) and fa else str(fa)
        return str(val).strip("$").strip()
    return str(row.get(key, "")).strip()


def load_dataset_split(data_name: str, max_samples: Optional[int] = None,
                       cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load a benchmark as a list of ``{"question", "answer"}`` dicts."""
    if data_name not in HF_CONFIGS:
        raise ValueError(f"unknown dataset {data_name!r}; choose from {sorted(HF_CONFIGS)}")
    cfg = HF_CONFIGS[data_name]
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {"split": cfg["split"]}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    args = (cfg["hf_path"], cfg["hf_name"]) if cfg["hf_name"] else (cfg["hf_path"],)
    examples: List[Dict[str, Any]] = []
    for row in load_dataset(*args, **kwargs):
        row = dict(row)
        question = str(row.get(cfg["question_key"], "")).strip()
        if not question:
            continue
        examples.append(dict(question=question, answer=gold_answer(row, data_name)))
        if max_samples and len(examples) >= max_samples:
            break
    logger.info(f"loaded {len(examples)} problems from {data_name}")
    return examples


__all__ = ["HF_CONFIGS", "load_dataset_split", "gold_answer", "parse_boxed"]
