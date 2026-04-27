"""
Utility helpers expected by parser.py
"""
import re
import regex


def extract_program(pred_str: str) -> str:
    """Extract Python/SymPy code blocks from model output."""
    pattern = r"```python(.*?)```"
    programs = re.findall(pattern, pred_str, re.DOTALL)
    if programs:
        return programs[-1].strip()
    # fallback: return as-is
    return pred_str.strip()


def extract_program_output(pred_str: str) -> str:
    """Extract the last printed/returned value from program output."""
    lines = [l.strip() for l in pred_str.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def floatify_ans(ans):
    """Try to convert answer string to float."""
    try:
        return float(ans)
    except Exception:
        return None
