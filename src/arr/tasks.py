"""
Benchmarks of the ARR experiments.  A Task supplies, per item: the prompt, the
answer extractor used for voting (one canonical answer per root), and the
scorer.  Every method uses the same Task, so extraction, tie-break and scoring
are identical across methods.

  math        aime24, aime25, amc23, math500, olympiadbench, gsm8k
              (last \\boxed{} answer; the repository's math grader)
  NATURAL PLAN (google-deepmind/natural-plan; 0-shot prompts + a fixed output-format line)
              np_calendar (1,000), np_meeting (1,000), np_trip (1,600)
  LiveBench reasoning, original 2024 release (livebench/reasoning)
              livebench_reasoning = web_of_lies_v2 (2024-06-24) + zebra_puzzle (2024-06-24)
              + spatial (2024-07-26), 50 each = 150; livebench_<task> for one subset

Scoring uses the OFFICIAL functions (src/arr/benchmarks/*_official.py, vendored
verbatim).  The voted answer is a canonical rendering of what the official parser
extracts from a response, so the official scorer applies to it unchanged:

  np_calendar  calendar_parse_response → "Day, HH:MM - HH:MM"; the answer after the
               last "proposed time" if present, else the parser's first match
  np_trip      parse_response → [(city, stay)] → compute_example_score
  np_meeting   parse_text_plan → plan steps; correct iff validator_from_text(plan)
               equals the golden plan's number of valid meetings (official criterion)
  livebench    web_of_lies_v2 / spatial: the bold answer; zebra_puzzle (2024): the
               ***bold*** answer; scored by the official process_results of the release
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .voting import Grader, extract_answer, normalize_answer

MATH_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
NP_URL = "https://raw.githubusercontent.com/google-deepmind/natural-plan/main/data/{}.json"
NP_FILES = {"np_calendar": "calendar_scheduling", "np_meeting": "meeting_planning", "np_trip": "trip_planning"}
LIVEBENCH_2024 = {"web_of_lies_v2": "2024-06-24", "zebra_puzzle": "2024-06-24", "spatial": "2024-07-26"}

# Fixed output-format line appended to every NATURAL PLAN 0-shot prompt (the 0-shot prompts
# do not state the format the official parsers read; the 5-shot prompts show it by example).
NP_FORMAT = {
    "np_calendar": ("Give your final answer on its own line in exactly this format:\n"
                    "Here is the proposed time: <Day>, <HH:MM> - <HH:MM>"),
    "np_meeting": ("Write your final plan after 'SOLUTION:' as sentences in exactly these forms:\n"
                   "You start at <Location> at <H:MMAM/PM>.\n"
                   "You travel to <Location> in <N> minutes and arrive at <H:MMAM/PM>.\n"
                   "You wait until <H:MMAM/PM>.\n"
                   "You meet <Name> for <N> minutes from <H:MMAM/PM> to <H:MMAM/PM>."),
    "np_trip": ("Write your final plan in exactly this format (one line per stay and per flight):\n"
                "**Day 1-<d>:** Arriving in <City> and visit <City> for <n> days.\n"
                "**Day <d>:** Fly from <City> to <City>.\n"
                "**Day <d>-<e>:** Visit <City> for <n> days.\n"
                "... until the last day."),
}
BOXED_LINE = "Final answer: \\boxed{<answer>}"
_THOUSANDS = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?")
GENERIC_ANSWER = "the final answer in the exact format the problem asks for"


@dataclass
class Task:
    name: str
    system: str
    kind: str                                    # math | natural_plan | livebench
    answer_format: Optional[str] = None          # replaces the "\\boxed{}" instruction for non-math tasks
    _grader: Any = None

    # extraction (voting)
    def extract(self, text: str, meta: Dict[str, Any], fallback_last_number: bool = True) -> Tuple[str, str]:
        text = text or ""
        if self.kind == "math":
            a = extract_answer(text, fallback_last_number)
            if self.name == "gsm8k" and _THOUSANDS.fullmatch(a.strip()):
                a = a.strip().replace(",", "")          # "1,000" → "1000" (numeric answers only)
        else:
            a = _EXTRACT[meta.get("task", self.name)](text) or ""
        return a, self.norm(a)

    def norm(self, answer: str) -> str:
        """Vote key of an extracted answer ("" = abstain)."""
        if self.kind == "math":
            return normalize_answer(answer)
        return re.sub(r"\s+", " ", (answer or "").strip().lower())

    # scoring
    def grade(self, answer: str, gold: str, meta: Dict[str, Any]) -> bool:
        if not answer:
            return False
        if self.kind == "math":
            if self._grader is None:
                self._grader = Grader(self.name)
            return bool(self._grader.is_correct(answer, gold))
        return bool(_GRADE[meta.get("task", self.name)](answer, gold, meta))

    def format_text(self, s: str) -> str:
        """Baseline prompts ask for '\\boxed{}'; non-math tasks keep their own answer format."""
        return s if self.kind == "math" else s.replace(BOXED_LINE, GENERIC_ANSWER).replace(
            "\\boxed{{<answer>}}", GENERIC_ANSWER)


# NATURAL PLAN

def _np():
    from .benchmarks import natural_plan_official as npo
    return npo


def _num_to_hhmm(x: float) -> str:
    return f"{int(x)}:{'30' if abs(x - int(x) - 0.5) < 1e-9 else '00'}"


def extract_calendar(text: str) -> str:
    npo = _np()
    tail = text[text.lower().rfind("proposed time"):] if "proposed time" in text.lower() else text
    day, s, e = npo.calendar_parse_response(tail)
    if not day:
        day, s, e = npo.calendar_parse_response(text)
    return f"{day}, {_num_to_hhmm(s)} - {_num_to_hhmm(e)}" if day else ""


def grade_calendar(answer: str, gold: str, meta) -> bool:
    npo = _np()
    return npo.calendar_parse_response(answer) == npo.calendar_parse_response(gold)


def extract_trip(text: str) -> str:
    plan = _np().parse_response(text)
    return " | ".join(f"{c}:{d}" for c, d in plan)


def _trip_plan(answer: str):
    out = []
    for piece in answer.split(" | "):
        if ":" in piece:
            c, d = piece.rsplit(":", 1)
            out.append((c, int(d)))
    return out


def grade_trip(answer: str, gold: str, meta) -> bool:
    return _np().compute_example_score(meta["cities"], meta["durations"], _trip_plan(answer)) == 1.0


_MEET_STEP = re.compile(r"^You (start|travel|wait|meet)\b")


def extract_meeting(text: str) -> str:
    steps = _np().parse_text_plan(text)
    plan, started = [], False
    for s in steps:
        s = s.strip().strip("*").strip()
        if _MEET_STEP.match(s):
            plan.append(s)
            started = True
        elif started:
            break
    return ". ".join(plan) + ("." if plan else "")


def _meeting_score(steps: List[str], meta) -> int:
    """Official validator_from_text.  It stops at the first invalid step when that step raises
    ValueError, but an unknown location / person raises KeyError and would crash the official
    script; such a step is treated like any other invalid step: the plan is scored up to it."""
    npo = _np()
    start_location, initial_time = meta["constraints"][0]
    cons = npo.process_constraints(meta["constraints"][1:])
    with contextlib.redirect_stdout(io.StringIO()):          # the official validator prints its errors
        for k in range(len(steps), -1, -1):
            try:
                return npo.validator_from_text(steps[:k], cons, start_location, initial_time, meta["dist_matrix"])
            except (KeyError, IndexError, TypeError):
                continue
    return 0


def grade_meeting(answer: str, gold: str, meta) -> bool:
    pred = _np().parse_text_plan(answer)
    return _meeting_score(pred, meta) == _meeting_score(meta["golden_plan"], meta)


# LiveBench reasoning (2024)

def _lb():
    from .benchmarks import livebench_official as lbo
    return lbo


def extract_web_of_lies(text: str) -> str:
    """The official web_of_lies_v2 parse (solution tags → bold yes/no → boxed → plain)."""
    lbo = _lb()
    m = re.findall(r"<solution>(.*?)</solution>", text)
    if m:
        return m[-1].strip().lower()
    bold = re.findall(r"\*\*(.*?)\*\*", text)
    if bold:
        words = [w.lower().strip().replace(",", "").replace(".", "") for b in bold for w in b.split()]
        ans = []
        i = len(words) - 1
        while i >= 0 and len(ans) < 3:
            if words[i] in ("yes", "no", "unknown"):
                ans = [words[i]] + ans
            i -= 1
        if ans:
            return ", ".join(ans)
    t = text.replace("\\textbf{", "\\boxed{").replace("\\fbox{", "\\boxed{")
    lb = lbo.last_boxed_only_string(t)
    if lb:
        try:
            return lbo.remove_boxed(lb).strip().lower()
        except AssertionError:
            pass
    import itertools
    best, idx = "", -1
    for comb in itertools.product(["yes", "no", "unknown"], repeat=3):
        k = text.lower().find(", ".join(comb))
        if k > idx:
            best, idx = ", ".join(comb), k
    return best


def extract_spatial(text: str) -> str:
    lbo = _lb()
    bold = re.findall(r"\*\*([^\*]+)\*\*", text)
    if bold:
        return bold[-1].strip()
    lb = lbo.last_boxed_only_string(text.replace("\\fbox{", "\\boxed{"))
    if lb:
        try:
            a = lbo.remove_boxed(lb)
        except AssertionError:
            return ""
        for p in ("\\textbf{", "\\mathbf{", "\\text{", "}"):
            a = a.replace(p, "")
        return a.strip()
    return ""


def extract_zebra_2024(text: str) -> str:
    bold = re.findall(r"\*\*\*(\w+)\*\*\*", text)
    if bold:
        return bold[-1]
    words = re.findall(r"\b\w+\b", text)
    return words[-1] if words else ""


def grade_web_of_lies(answer, gold, meta) -> bool:
    return _lb().web_of_lies_process_results(gold, f"**{answer}**") == 1


def grade_spatial(answer, gold, meta) -> bool:
    return _lb().spatial_process_results(gold, f"**{answer}**") == 1


def grade_zebra(answer, gold, meta) -> bool:
    fn = _lb().get_zebra_puzzle_evaluator(meta.get("release_date", "2024-06-24"))
    return fn(gold, f"***{answer}***") == 1


_EXTRACT: Dict[str, Callable[[str], str]] = {
    "np_calendar": extract_calendar, "np_trip": extract_trip, "np_meeting": extract_meeting,
    "web_of_lies_v2": extract_web_of_lies, "spatial": extract_spatial, "zebra_puzzle": extract_zebra_2024,
}
_GRADE: Dict[str, Callable[[str, str, Dict], bool]] = {
    "np_calendar": grade_calendar, "np_trip": grade_trip, "np_meeting": grade_meeting,
    "web_of_lies_v2": grade_web_of_lies, "spatial": grade_spatial, "zebra_puzzle": grade_zebra,
}


# Registry + loaders

def get_task(dataset: str) -> Task:
    if dataset in NP_FILES:
        return Task(dataset, "", "natural_plan", answer_format=NP_FORMAT[dataset])
    if dataset.startswith("livebench"):
        return Task(dataset, "", "livebench", answer_format=GENERIC_ANSWER)
    return Task(dataset, MATH_SYSTEM, "math")


def _cache_dir() -> str:
    d = os.path.join(os.environ.get("SCRATCH", os.path.expanduser("~")), "benchmarks")
    os.makedirs(d, exist_ok=True)
    return d


def load_natural_plan(dataset: str) -> List[Dict[str, Any]]:
    import urllib.request
    path = os.path.join(_cache_dir(), "natural_plan", NP_FILES[dataset] + ".json")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(NP_URL.format(NP_FILES[dataset]), path)
    with open(path) as f:
        data = json.load(f)
    out = []
    for key, x in data.items():
        meta = dict(task=dataset, source_id=key)
        if dataset == "np_trip":
            meta.update(cities=x["cities"], durations=x["durations"], num_cities=x["num_cities"])
        elif dataset == "np_meeting":
            meta.update(constraints=x["constraints"], dist_matrix=x["dist_matrix"], golden_plan=x["golden_plan"],
                        num_people=x["num_people"])
        else:
            meta.update(num_people=x["num_people"], num_days=x["num_days"])
        gold = x["golden_plan"] if isinstance(x["golden_plan"], str) else " ".join(x["golden_plan"])
        out.append(dict(question=x["prompt_0shot"].rstrip() + "\n\n" + NP_FORMAT[dataset], gold=gold, meta=meta))
    return out


def load_livebench(dataset: str, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    d = load_dataset("livebench/reasoning", split="test", cache_dir=cache_dir or None)
    tasks = list(LIVEBENCH_2024) if dataset == "livebench_reasoning" else [dataset[len("livebench_"):]]
    out = []
    for t in tasks:
        rows = [r for r in d if r["task"] == t and str(r["livebench_release_date"])[:10] == LIVEBENCH_2024[t]]
        rows.sort(key=lambda r: r["question_id"])
        for r in rows:
            out.append(dict(question=r["turns"][0], gold=str(r["ground_truth"]),
                            meta=dict(task=t, question_id=r["question_id"],
                                      release_date=str(r["livebench_release_date"])[:10])))
    return out


def load_questions(dataset: str, max_samples: int = 0, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if dataset in NP_FILES:
        items = load_natural_plan(dataset)
    elif dataset.startswith("livebench"):
        items = load_livebench(dataset, cache_dir)
    else:
        from .data import load_dataset_split
        exs = load_dataset_split(data_name=dataset, max_samples=None, cache_dir=cache_dir or None)
        items = [dict(question=e["question"], gold=str(e.get("answer", "")), meta={}) for e in exs if e.get("question")]
    if max_samples:
        items = items[:max_samples]
    return [dict(qid=f"{dataset}:{i:04d}", **x) for i, x in enumerate(items)]


def sanity_check_tasks() -> List[str]:
    """Round-trip checks: canonical answers extracted from the golden plans are graded correct."""
    fails = []
    for ds in NP_FILES:
        items = load_natural_plan(ds)[:40]
        t = get_task(ds)
        for it in items:
            gold_text = it["gold"] if ds != "np_meeting" else "SOLUTION: " + it["gold"]
            a, _ = t.extract(gold_text, it["meta"])
            if not t.grade(a, it["gold"], it["meta"]):
                fails.append(f"{ds}: golden plan not graded correct after extraction: {a!r}")
                break
            if t.grade("", it["gold"], it["meta"]):
                fails.append(f"{ds}: empty answer graded correct")
                break
    it = load_natural_plan("np_meeting")[0]                  # an unknown location must not crash scoring
    get_task("np_meeting").grade("SOLUTION: You start at Nowhere at 9:00AM. You travel to the Moon in 5 minutes "
                                 "and arrive at 9:05AM.", it["gold"], it["meta"])
    t = get_task("livebench_reasoning")
    for text, gold, meta in [("so **no, yes, yes**", "no, yes, yes", dict(task="web_of_lies_v2")),
                             ("thus **3**", "3", dict(task="spatial")),
                             ("position ***2***", "2", dict(task="zebra_puzzle", release_date="2024-06-24"))]:
        a, _ = t.extract(text, meta)
        if not t.grade(a, gold, meta):
            fails.append(f"livebench {meta['task']}: {text!r} → {a!r} not graded correct")
        if t.grade(a + "x", gold, meta):
            fails.append(f"livebench {meta['task']}: mutated answer graded correct")
    return fails


__all__ = ["Task", "get_task", "load_questions", "sanity_check_tasks", "NP_FILES", "LIVEBENCH_2024"]
