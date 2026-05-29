"""Routing-only cheap solo probe.

The probe asks the LLM for a single 32-token forward pass and computes the
Shannon entropy of the first-token distribution. The probe's *text* is never
fed to a downstream protocol; only the entropy enters X. This is the
``Solo probe proxy`` row of Table 6 in the paper.

When the backend cannot return top-token logprobs (e.g. the OpenAI backend
talking to an endpoint that does not expose logprobs), we fall back to a
length-based deterministic heuristic so the pipeline still produces a
consistent feature.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _entropy(logprobs: Sequence[float]) -> float:
    # logprobs may be unnormalised; renormalise to a proper distribution
    if not logprobs:
        return 0.0
    m = max(logprobs)
    exps = [math.exp(x - m) for x in logprobs]
    z = sum(exps)
    probs = [e / z for e in exps]
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))


def solo_probe_entropy(backend, prompt: str, *, max_tokens: int = 32,
                       top_logprobs: int = 8, seed: int | None = 0) -> float:
    """Run the cheap solo probe and return the entropy of the first token."""
    try:
        resp = backend.chat(prompt,
                            max_tokens=max_tokens,
                            temperature=0.0,
                            request_logprobs=True,
                            logprobs_top_k=top_logprobs,
                            seed=seed)
    except Exception:
        # Length-based fallback. Longer prompts -> higher (proxy) entropy.
        return float(min(1.5, 0.3 + 0.0003 * len(prompt)))
    lp = resp.top_token_logprobs or []
    if not lp:
        # Fall back when the server returned no logprobs.
        return float(min(1.5, 0.3 + 0.0003 * len(prompt)))
    return _entropy([v for _, v in lp])


def batch_solo_probe(backend, prompts: Iterable[str], **kwargs) -> np.ndarray:
    return np.asarray([solo_probe_entropy(backend, p, **kwargs) for p in prompts],
                      dtype=np.float32)
