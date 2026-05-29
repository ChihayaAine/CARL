"""Off-policy evaluation wrappers for non-full-matrix splits.

When the test split is logged with a known behavior policy but only one arm
was actually pulled per row, we estimate the policy value via SNIPW or DR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from carl.dr.ipw import ipw_value, dr_value, OPEResult


@dataclass
class OPESummary:
    name: str
    snipw: OPEResult
    dr: OPEResult


def evaluate_off_policy(
    name: str,
    decisions: np.ndarray,
    *,
    T: np.ndarray,
    Y: np.ndarray,
    behavior_prob: np.ndarray,
    mu_Y: np.ndarray,
    e: np.ndarray,
) -> OPESummary:
    snipw = ipw_value(T, decisions, Y, behavior_prob, self_normalised=True)
    dr = dr_value(T, decisions, Y, mu_Y, e)
    return OPESummary(name=name, snipw=snipw, dr=dr)
