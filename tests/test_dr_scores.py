import numpy as np

from carl.dr import build_dr_scores
from carl.nuisances.crossfit import NuisancePredictions


def _make_dummy_nuis(N=20, K=3):
    rng = np.random.default_rng(0)
    mu_R = rng.normal(0.5, 0.1, size=(N, K))
    mu_R[:, 0] = 0.4  # solo baseline
    mu_Ct = np.zeros((N, K))
    mu_Cl = np.zeros((N, K))
    sigma_R = np.full((N, K), 0.05)
    e = np.full((N, K), 1.0 / K)
    return NuisancePredictions(mu_R=mu_R, sigma_R=sigma_R,
                               mu_Ct=mu_Ct, mu_Cl=mu_Cl,
                               e=e, fold_ids=np.zeros(N, dtype=int))


def test_dr_psi_solo_column_is_zero():
    n = _make_dummy_nuis()
    T = np.zeros(n.n, dtype=int)
    R = np.full(n.n, 0.5)
    dr = build_dr_scores(n, T=T, R=R, Ct=np.zeros(n.n), Cl=np.zeros(n.n),
                         lam=0.0, mu=0.0, solo_index=0)
    assert np.allclose(dr.psi[:, 0], 0.0)
    assert dr.psi.shape == (n.n, n.K)
