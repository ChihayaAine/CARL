"""Lightweight BM25 retrieval used for the Code task's pre-treatment features.

The pre-treatment feature vector includes the BM25 score against the top-k
retrieved files and basic doc-length statistics (length, mean length). This
mirrors the Code task description in the paper (Methodology and Appendix B).
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, Tuple


_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


class BM25Index:
    """In-process BM25 with optional ``rank-bm25`` acceleration."""

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75):
        self.docs = list(documents)
        self.tokens = [_tokenize(d) for d in self.docs]
        self.k1 = k1
        self.b = b
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
            self._idx = BM25Okapi(self.tokens, k1=k1, b=b)
            self._fallback = False
        except Exception:
            self._idx = None
            self._fallback = True
            self._build_fallback()

    def _build_fallback(self) -> None:
        N = len(self.tokens)
        df: dict = {}
        for toks in self.tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1.0 + (N - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._dl = [len(toks) for toks in self.tokens]
        self._avgdl = sum(self._dl) / max(N, 1)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_toks = _tokenize(query)
        if not q_toks or not self.docs:
            return []
        if not self._fallback:
            scores = self._idx.get_scores(q_toks)
        else:
            scores = self._fallback_scores(q_toks)
        order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))[:top_k]
        return [(i, float(scores[i])) for i in order]

    def _fallback_scores(self, q_toks: Iterable[str]) -> List[float]:
        scores = [0.0] * len(self.tokens)
        for q in q_toks:
            idf = self._idf.get(q)
            if idf is None:
                continue
            for i, toks in enumerate(self.tokens):
                tf = toks.count(q)
                if tf == 0:
                    continue
                dl = self._dl[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1.0))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores

    def doc_length_stats(self, indices: Sequence[int]) -> dict:
        if not indices:
            return {"top_doc_len_mean": 0.0, "top_doc_len_max": 0.0}
        lens = [len(self.tokens[i]) for i in indices]
        return {"top_doc_len_mean": float(sum(lens) / len(lens)),
                "top_doc_len_max":  float(max(lens))}
