"""Text encoders.

The default is ``HashingEncoder`` (zero deps; deterministic 256-dim feature
hashing). ``SBertEncoder`` is preferred when ``sentence-transformers`` is
installed (the paper uses an LLM mean-pooled representation; we expose
both flavours since SBert is far cheaper and still produces a useful X for
the kNN/MLP heads).
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence

import numpy as np


class _Encoder:
    dim: int = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEncoder(_Encoder):
    """Deterministic feature-hashing encoder over character 3- and 4-grams.

    No external dependencies and no network access. Suitable for tests, CI,
    smoke runs, or as a fallback when SBert is unavailable.
    """

    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            self._encode_one(t.lower(), out[i])
        # L2 normalise
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def _encode_one(self, text: str, target: np.ndarray) -> None:
        for k in (3, 4):
            for j in range(max(len(text) - k + 1, 0)):
                gram = text[j:j + k]
                h = int.from_bytes(
                    hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest(),
                    "little",
                )
                idx = h % self.dim
                sign = 1.0 if (h & 1) else -1.0
                target[idx] += sign


class SBertEncoder(_Encoder):
    """Sentence-Transformers wrapper. Lazy-import."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vec = self._model.encode(list(texts), normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)


def build_encoder(cfg) -> _Encoder:
    kind = (cfg.encoder_kind or "sbert").lower()
    if kind == "sbert":
        try:
            return SBertEncoder(cfg.encoder_name)
        except Exception:
            return HashingEncoder(dim=cfg.encoder_dim or 256)
    if kind == "mock":
        return HashingEncoder(dim=cfg.encoder_dim or 256)
    if kind == "hashing":
        return HashingEncoder(dim=cfg.encoder_dim or 256)
    raise ValueError(f"Unknown encoder kind: {kind!r}")
