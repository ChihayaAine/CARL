"""Mixed behavior policy b = (1 - eps) * pi_phi + eps * q_psi, projected onto
the simplex with floor b_min >= 0.02 on every (x, k)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _project_to_simplex_with_floor(p: np.ndarray, floor: float) -> np.ndarray:
    """Set every entry to >= floor and renormalise the simplex."""
    K = p.shape[0]
    floor = float(min(max(floor, 0.0), 1.0 / K))
    p = np.clip(p, floor, None)
    # If sum>1, scale only the excess above floor so the floor is preserved.
    excess = p - floor
    s = excess.sum()
    target = 1.0 - K * floor
    if s <= 0:
        return np.full_like(p, 1.0 / K)
    excess *= target / s
    return floor + excess


def behavior_policy(pi_phi: np.ndarray, q_psi: np.ndarray, *, eps: float, b_min: float
                    ) -> np.ndarray:
    """Build the live behavior policy b(T | x)."""
    pi_phi = np.asarray(pi_phi, dtype=np.float64)
    q_psi = np.asarray(q_psi, dtype=np.float64)
    pi_phi = pi_phi / pi_phi.sum()
    q_psi = q_psi / q_psi.sum()
    b = (1.0 - eps) * pi_phi + eps * q_psi
    return _project_to_simplex_with_floor(b, floor=b_min)


def sample_treatment(b: np.ndarray, rng: np.random.Generator
                     ) -> Tuple[int, float, str]:
    """Sample an action k ~ b. Returns (k, prob, source) where source
    indicates whether the row was an exploration (probe) draw or not."""
    k = int(rng.choice(b.shape[0], p=b))
    return k, float(b[k]), "behavior"
