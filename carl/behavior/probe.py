"""Randomized probe distribution q_psi.

q_psi(T | x) is a softmax of three terms:
    epistemic uncertainty bonus    (higher for inputs the advantage learner is uncertain about)
    inverse-count coverage bonus   (favours under-sampled (x-cluster, treatment) pairs)
    small negative cost term       (mild distaste for expensive protocols)

q_psi depends only on X. Until an advantage learner exists, ``uncertainty``
defaults to zero, which collapses q_psi to a coverage + cost-weighted uniform
distribution. The behavior policy then enforces b_min on top.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def probe_distribution(
    K: int,
    *,
    cost: Optional[Sequence[float]] = None,
    counts: Optional[Sequence[int]] = None,
    uncertainty: Optional[Sequence[float]] = None,
    temperature: float = 1.0,
    cost_weight: float = 0.6,
    coverage_weight: float = 0.8,
    uncertainty_weight: float = 0.8,
) -> np.ndarray:
    """Return a (K,) probability vector over the protocols.

    Parameters
    ----------
    K           : number of protocols
    cost        : per-protocol expected cost (token + latency, normalised)
    counts      : per-protocol count of prior assignments to similar X
    uncertainty : per-protocol epistemic uncertainty (sigma in the paper)
    """
    cost_arr = np.asarray(cost if cost is not None else [0.0] * K, dtype=np.float64)
    cnt_arr = np.asarray(counts if counts is not None else [0] * K, dtype=np.float64)
    unc_arr = np.asarray(uncertainty if uncertainty is not None else [0.0] * K,
                         dtype=np.float64)
    coverage = 1.0 / np.sqrt(1.0 + cnt_arr)
    logits = (uncertainty_weight * unc_arr
              + coverage_weight * coverage
              - cost_weight * cost_arr)
    logits = logits / max(temperature, 1e-6)
    logits -= logits.max()
    p = np.exp(logits)
    p = p / p.sum()
    return p
