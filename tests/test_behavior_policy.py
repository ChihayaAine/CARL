import numpy as np

from carl.behavior.policy import behavior_policy, _project_to_simplex_with_floor


def test_floor_is_enforced():
    p = np.array([0.9, 0.05, 0.04, 0.01])
    out = _project_to_simplex_with_floor(p, floor=0.02)
    assert (out >= 0.02 - 1e-9).all()
    assert abs(out.sum() - 1.0) < 1e-9


def test_mix_respects_floor_and_simplex():
    pi = np.array([0.6, 0.3, 0.1])
    q = np.array([0.4, 0.4, 0.2])
    b = behavior_policy(pi, q, eps=0.15, b_min=0.05)
    assert (b >= 0.05 - 1e-9).all()
    assert abs(b.sum() - 1.0) < 1e-9
