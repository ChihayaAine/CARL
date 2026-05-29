import numpy as np

from carl.calibration import fit_conformal_quantiles, apply_lcb


def test_conformal_lcb_shape_and_solo():
    rng = np.random.default_rng(0)
    N, K = 50, 3
    psi = rng.normal(size=(N, K))
    a_hat = psi + rng.normal(scale=0.05, size=(N, K))
    sigma = np.full((N, K), 0.1)
    cal = fit_conformal_quantiles(psi, a_hat, sigma, delta=0.1, solo_index=0)
    lcb = apply_lcb(a_hat, sigma, cal, kappa=1.0)
    assert lcb.shape == (N, K)
    assert np.allclose(lcb[:, 0], 0.0)
