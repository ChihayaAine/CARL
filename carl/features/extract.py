"""Pre-treatment feature extraction.

Produces a :class:`FeatureBundle` containing:
    X      : (n, d_total) dense feature matrix used by every learner in CARL
    enc    : (n, d_enc)   text encoder embedding (also used for kNN shrinkage)
    meta   : list of dicts with per-example diagnostic fields

The numeric per-example side-channel features are (in order):
    0 input_length_chars / 1000
    1 input_length_tokens / 100     (rough whitespace token proxy)
    2 task_id (0=math, 1=qa, 2=code)
    3 benchmark_id_hash (0-1 stable hash bucket)
    4 difficulty_hint (float in [0, 1])
    5 problem_class_hash (0-1 stable hash bucket)
    6 hop_count (QA) or 0
    7 bm25_top1 score (Code) or 0
    8 bm25_top5 mean (Code) or 0
    9 top_doc_len_mean / 1000 (Code) or 0
   10 solo_probe_entropy
   11 has_passages (1 if passages provided)
   12 num_passages (QA only)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import hashlib
import numpy as np

from .bm25 import BM25Index
from .solo_probe import solo_probe_entropy

SIDE_DIM = 13


def _h_unit(s: str) -> float:
    h = int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(),
                       "little")
    return (h % 10_000) / 10_000.0


def _side_features(example: dict, *, solo_entropy: float) -> np.ndarray:
    x = np.zeros(SIDE_DIM, dtype=np.float32)
    inp = example.get("input") or ""
    x[0] = len(inp) / 1000.0
    x[1] = len(inp.split()) / 100.0
    task = example.get("task", "math")
    x[2] = {"math": 0.0, "qa": 1.0, "code": 2.0}.get(task, 0.0)
    x[3] = _h_unit(str(example.get("benchmark_id") or ""))
    x[4] = float(example.get("difficulty_hint") or 0.0)
    meta = example.get("meta") or {}
    x[5] = _h_unit(str(meta.get("problem_class") or meta.get("repo") or ""))
    x[6] = float(meta.get("hop_count") or 0)
    x[7] = float(meta.get("bm25_top1") or 0.0)
    x[8] = float(meta.get("bm25_top5_mean") or 0.0)
    x[9] = float(meta.get("top_doc_len_mean") or 0.0) / 1000.0
    x[10] = float(solo_entropy)
    pas = meta.get("passages") or []
    x[11] = 1.0 if pas else 0.0
    x[12] = float(len(pas)) / 10.0
    return x


@dataclass
class FeatureBundle:
    X: np.ndarray          # (n, d_enc + SIDE_DIM)
    enc: np.ndarray        # (n, d_enc)
    side: np.ndarray       # (n, SIDE_DIM)
    ids: List[str]
    examples: List[dict]   # original normalised examples


def _attach_code_retrieval(examples: Sequence[dict], top_k: int) -> None:
    """For Code, compute BM25 over the in-example "files" lists.

    Each Code example carries a small set of files in ``meta['files']`` for
    Synthetic data and SWE-bench Lite. We index them per-example and store
    aggregate scores on the example's meta dict.
    """
    for ex in examples:
        if ex.get("task") != "code":
            continue
        meta = ex.get("meta") or {}
        files = meta.get("files") or []
        if not files:
            # synthesise a single document from the input itself
            files = [ex.get("input") or ""]
        idx = BM25Index([str(f) for f in files])
        hits = idx.search(ex.get("input") or "", top_k=top_k)
        if hits:
            top_score = hits[0][1]
            mean_score = float(sum(s for _, s in hits) / len(hits))
            meta["bm25_top1"] = float(top_score)
            meta["bm25_top5_mean"] = mean_score
            meta.update(idx.doc_length_stats([i for i, _ in hits]))
        ex["meta"] = meta


def featurize_examples(examples: Sequence[dict], encoder, backend=None,
                       *, run_solo_probe: bool = True,
                       bm25_top_k: int = 5,
                       precomputed_entropy: Optional[Sequence[float]] = None,
                       ) -> FeatureBundle:
    """Compute the full pre-treatment feature matrix for a list of examples."""
    examples = [dict(e) for e in examples]
    _attach_code_retrieval(examples, top_k=bm25_top_k)

    texts = [ex.get("input") or "" for ex in examples]
    enc = encoder.encode(texts)

    if precomputed_entropy is not None:
        ent = np.asarray(precomputed_entropy, dtype=np.float32)
    elif run_solo_probe and backend is not None:
        ent = np.asarray([solo_probe_entropy(backend, t, seed=i)
                          for i, t in enumerate(texts)], dtype=np.float32)
    else:
        ent = np.zeros(len(texts), dtype=np.float32)

    side = np.stack([_side_features(ex, solo_entropy=float(ent[i]))
                     for i, ex in enumerate(examples)], axis=0)
    X = np.concatenate([enc, side], axis=1).astype(np.float32)
    ids = [ex["id"] for ex in examples]
    return FeatureBundle(X=X, enc=enc, side=side, ids=ids, examples=examples)
