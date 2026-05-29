"""MATH exact-match reward.

The function is lenient about boxed answers, trailing punctuation, and the
``Answer:`` prefix used by our protocol prompts. It does not try to do
algebraic simplification; for that we recommend `sympy`, but in our pipeline
the protocols are prompted to return a final-line answer in canonical form.
"""
from __future__ import annotations

import re


_ANSWER_LINE = re.compile(r"(?:answer\s*[:=]\s*)(.+)$", re.IGNORECASE | re.MULTILINE)
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")


def _normalize(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    m = _BOXED.search(s)
    if m:
        return _normalize(m.group(1))
    m = _ANSWER_LINE.search(s)
    if m:
        s = m.group(1)
    s = s.strip().rstrip(".")
    s = s.replace(" ", "")
    return s.lower()


def math_exact_match(prediction: str, target: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(target) else 0.0
