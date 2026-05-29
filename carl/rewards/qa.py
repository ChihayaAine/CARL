"""QA token-level F1 reward (SQuAD-style)."""
from __future__ import annotations

import re
import string
from collections import Counter


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_ANSWER_LINE = re.compile(r"(?:answer\s*[:=]\s*)(.+)$", re.IGNORECASE | re.MULTILINE)


def _normalize(s: str) -> str:
    s = (s or "").strip()
    m = _ANSWER_LINE.search(s)
    if m:
        s = m.group(1)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def qa_token_f1(prediction: str, target: str) -> float:
    pred = _normalize(prediction).split()
    gold = _normalize(target).split()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    p = n_same / len(pred)
    r = n_same / len(gold)
    return 2 * p * r / (p + r)
