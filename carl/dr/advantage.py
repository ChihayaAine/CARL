"""Advantage estimator: bootstrap ensemble of regressors trained on shrunk
DR pseudo-outcomes. Returns mean (the advantage) and epistemic sigma.

We use sklearn's HistGradientBoostingRegressor to stay dependency-light.
For per-arm advantages we train K-1 regressors (one per non-solo arm).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from carl.nuisances.heads import build_nuisance_estimator, fit_head_ensemble, HeadEnsemble


@dataclass
class AdvantageModel:
    ensembles: List[Optional[HeadEnsemble]]   # one per arm, ensembles[solo_index] is None
    solo_index: int
    K: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = X.shape[0]
        mean = np.zeros((N, self.K), dtype=np.float64)
        sigma = np.zeros((N, self.K), dtype=np.float64)
        for k, ens in enumerate(self.ensembles):
            if ens is None:
                continue
            m, s = ens.predict_mean_std(X)
            mean[:, k] = m
            sigma[:, k] = s
        return mean, sigma


def fit_advantage_ensemble(
    X: np.ndarray,
    advantage_targets: np.ndarray,
    *,
    solo_index: int = 0,
    n_boot: int = 5,
    kind: str = "boost",
    seed: int = 0,
) -> AdvantageModel:
    """Fit one bootstrap ensemble per non-solo arm.

    advantage_targets : (N, K) the shrunk DR pseudo-outcomes for each arm.
    """
    N, K = advantage_targets.shape
    make = lambda s: build_nuisance_estimator(kind, seed=s)
    ensembles: List[Optional[HeadEnsemble]] = [None] * K
    for k in range(K):
        if k == solo_index:
            continue
        if N == 0:
            ensembles[k] = HeadEnsemble(members=[make(seed + k)])
            continue
        ensembles[k] = fit_head_ensemble(
            make, X, advantage_targets[:, k], n_boot=n_boot, seed=seed + k * 31,
        )
    return AdvantageModel(ensembles=ensembles, solo_index=solo_index, K=K,
                          meta={"n_boot": n_boot, "kind": kind})
