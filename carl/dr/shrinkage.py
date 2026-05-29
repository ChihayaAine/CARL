"""kNN shrinkage of DR advantage estimates toward the direct contrast.

For each (x_i, k) the paper shrinks the DR-trained predictor toward the
direct-contrast prediction with weight ``gamma * sqrt(K_NN) / (sqrt(K_NN) + 1)``
when local nuisance noise is high. We implement a simple, deterministic
variant that returns the convex combination

    shrunk = (1 - alpha) * dr + alpha * direct

with ``alpha`` increasing in the local empirical variance of the DR
pseudo-outcome (estimated from the K_NN nearest training neighbours).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ShrinkageResult:
    advantage: np.ndarray       # (N, K)
    alpha: np.ndarray           # (N, K)
    direct: np.ndarray          # (N, K)
    dr_pred: np.ndarray         # (N, K)


def _knn_indices(query: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    """Return (N_q, k) indices into ``ref`` of nearest neighbours by L2."""
    if ref.shape[0] == 0:
        return np.zeros((query.shape[0], 0), dtype=np.int64)
    k = int(min(k, ref.shape[0]))
    # chunked to keep memory bounded
    out = np.empty((query.shape[0], k), dtype=np.int64)
    chunk = 256
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk]
        # ||q-r||^2 = ||q||^2 + ||r||^2 - 2 q.r
        d = np.einsum("ij,ij->i", q, q)[:, None] + \
            np.einsum("ij,ij->i", ref, ref)[None, :] - 2.0 * q @ ref.T
        np.maximum(d, 0.0, out=d)
        out[i:i + chunk] = np.argpartition(d, k - 1, axis=1)[:, :k]
    return out


def shrink_to_direct(
    psi: np.ndarray,
    direct: np.ndarray,
    X: np.ndarray,
    *,
    K_NN: int = 50,
    gamma: float = 0.5,
    train_mask: Optional[np.ndarray] = None,
) -> ShrinkageResult:
    """Apply kNN-based shrinkage of psi toward the direct contrast.

    Parameters
    ----------
    psi : (N, K) DR pseudo-outcomes from build_dr_scores.
    direct : (N, K) direct contrast (mu_Y[k] - mu_Y[0]).
    X : (N, d) features used for the nearest-neighbour metric.
    K_NN : neighbourhood size.
    gamma : maximum shrinkage strength (alpha is bounded by this).
    train_mask : optional bool array selecting rows usable as neighbours.
    """
    N, K = psi.shape
    if N == 0:
        return ShrinkageResult(psi.copy(), np.zeros_like(psi),
                               direct.copy(), psi.copy())
    if train_mask is None:
        train_mask = np.ones(N, dtype=bool)
    ref = X[train_mask]
    nn = _knn_indices(X, ref, K_NN)
    base_idx = np.flatnonzero(train_mask)
    if nn.shape[1] == 0:
        alpha = np.zeros_like(psi)
        return ShrinkageResult(psi.copy(), alpha, direct.copy(), psi.copy())
    psi_train = psi[base_idx]
    # local variance per row per arm (use train neighbours)
    nbr_psi = psi_train[nn]                            # (N, K_NN, K)
    local_var = nbr_psi.var(axis=1)                     # (N, K)
    # bound alpha in [0, gamma] using a saturating function of local variance.
    # alpha = gamma * tanh(local_var / scale); scale picked from global var.
    scale = max(float(psi_train.var()), 1e-6)
    alpha = float(gamma) * np.tanh(local_var / scale)
    alpha = np.clip(alpha, 0.0, float(gamma))
    shrunk = (1.0 - alpha) * psi + alpha * direct
    return ShrinkageResult(advantage=shrunk, alpha=alpha,
                           direct=direct.copy(), dr_pred=psi.copy())
