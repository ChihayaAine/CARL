"""Fixed-arm baselines (Always-SOLO, Always-VERIFY, ...) and Best-Fixed.

Always-X just routes every input to arm X. Best-Fixed picks the single arm
that achieves the highest validation utility and routes every input to it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BaselineDecision:
    name: str
    decisions: np.ndarray


def always_arm(N: int, arm: int, *, name: str | None = None) -> BaselineDecision:
    return BaselineDecision(
        name=name or f"Always-{arm}",
        decisions=np.full(N, int(arm), dtype=np.int64),
    )


def best_fixed_from_dr(
    psi: np.ndarray,
    *,
    N_test: int,
    solo_index: int = 0,
    arm_names: Sequence[str] | None = None,
) -> BaselineDecision:
    """Pick the arm with the highest mean DR pseudo-outcome on validation,
    and route every test row to it."""
    if psi.shape[0] == 0:
        arm = solo_index
    else:
        arm = int(np.argmax(psi.mean(axis=0)))
    name = "Best-Fixed"
    if arm_names is not None and 0 <= arm < len(arm_names):
        name = f"Best-Fixed[{arm_names[arm]}]"
    return BaselineDecision(name=name,
                            decisions=np.full(N_test, arm, dtype=np.int64))
