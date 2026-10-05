"""Poisson non-Gaussian terms (thecov.discreteness)."""
import numpy as np
import pytest

from thecov import Tracer, PowerSpectrumModel, GaussianCovariance, DiscretenessCovariance


def p_lin():
    k = np.logspace(-4, 1, 3000)
    return k, 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)


@pytest.fixture(scope='module')
def setup():
    rng = np.random.default_rng(0)
    R, nbar = 300.0, 1e-3
    x = rng.uniform(-R, R, size=(60000, 3))
    pos = x[np.sum(x * x, 1) < R * R][:20000] + [0, 0, 1e5]
    V = 4 / 3 * np.pi * R ** 3
    tr = Tracer('T', {'POSITION': pos, 'WEIGHT': np.ones(len(pos)), 'NZ': np.full(len(pos), nbar)}, nbar * V / len(pos))
    cov = GaussianCovariance([tr], np.array([0.05, 0.06, 0.15, 0.16]), ells=(0, 2), L_max=0)
    return cov, nbar, V


def test_window_integrals_uniform(setup):
    cov, nbar, V = setup
    d = DiscretenessCovariance(cov, p_lin(), b1=2.0, f=0.0)
    J_Smm, J_SS = d.window_integrals('T')
    assert np.isclose(J_Smm, 1 / (nbar * V), rtol=0.02) and np.isclose(J_SS, 1 / (nbar ** 2 * V), rtol=0.02)


def test_real_space_against_monte_carlo(setup):
    """f = 0: the monopole block equals the isotropic average over k1^, k2^ (Monte Carlo), and the l = 0 x 2
    block vanishes (an isotropic function of k1^.k2^ only couples equal multipoles)."""
    cov, nbar, V = setup
    d = DiscretenessCovariance(cov, p_lin(), b1=2.0, f=0.0, b2=0.4, n_mu=16, n_phi=24, n_k=2)
    C, _ = d.covariance([('T', 'T')], ells=(0, 2))
    nb = 3
    assert np.max(np.abs(C[:nb, nb:])) < 1e-3 * np.max(np.abs(C[:nb, :nb]))
    J_Smm, J_SS = d.window_integrals('T')
    rng = np.random.default_rng(1)
    n = 400000

    def unit(n):
        v = rng.normal(size=(3, n))
        return v / np.linalg.norm(v, axis=0)
    for (i, j) in [(0, 0), (0, 2), (2, 2)]:
        e = cov.k_edges
        ka = np.cbrt(rng.uniform(e[i] ** 3, e[i + 1] ** 3, n))
        kb = np.cbrt(rng.uniform(e[j] ** 3, e[j + 1] ** 3, n))
        k1, k2 = ka * unit(n), kb * unit(n)
        F = 2 * J_Smm * (d._B(k1, k2) + d._B(k1, -k2)) + J_SS * (d._Ps(k1 + k2) + d._Ps(k1 - k2))
        assert np.isclose(C[i, j], F.mean(), rtol=0.03), (i, j, C[i, j], F.mean())
