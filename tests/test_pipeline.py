"""End-to-end checks with synthetic randoms in a spherical window.

1. Q_000(s) from pair counts vs the analytic self-overlap volume of a sphere.
2. The monopole covariance vs a brute-force evaluation of
       C_00(i,j) = 2/I^2 <P(k1) P(k2) |W~(k1-k2)|^2>_{shells}
   for an isotropic P(k).
3. A far-away sphere (distant observer): ratios C_{l1 l2}/C_00 close to the periodic-box values.
"""
import math

import numpy as np
import pytest
from scipy.special import spherical_jn, eval_legendre

from thecov import Tracer, PowerSpectrumModel, GaussianCovariance
from thecov.tracers import Window
from thecov.windows import TripolarWindow
from thecov.wigner import FOUR_PI


def sphere_randoms(R, n, center=(0.0, 0.0, 0.0), nbar=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while sum(len(p) for p in pts) < n:
        x = rng.uniform(-R, R, size=(2 * n, 3))
        pts.append(x[np.sum(x * x, axis=1) < R * R])
    pos = np.concatenate(pts)[:n] + np.asarray(center)
    V = 4 * np.pi / 3 * R ** 3
    alpha = nbar * V / n
    return {'POSITION': pos, 'WEIGHT': np.ones(n), 'NZ': np.full(n, nbar)}, alpha


def pk_model():
    k = np.geomspace(1e-3, 1.0, 400)
    P0 = 2e4 * (k / 0.05) / (1 + (k / 0.05) ** 2.2)     # smooth, isotropic
    return k, P0


def test_Q000_sphere():
    R, n, nbar = 500.0, 12000, 2e-4
    rnd, alpha = sphere_randoms(R, n, nbar=nbar)
    tr = Tracer('T', rnd, alpha)
    w = Window('W', tr, tr)
    s_edges = np.arange(0, 2 * R + 20, 20.0)
    tw = TripolarWindow(w, w, [(0, 0, 0), (0, 0, 2)], s_edges, n_sub=3000).compute()
    s = tw.s_centers
    Vov = np.pi / 12 * (4 * R + s) * (2 * R - s) ** 2
    Q_th = FOUR_PI ** (-1.5) * FOUR_PI * nbar ** 4 * Vov
    sel = (s > 40) & (s < 1.6 * R)
    assert np.allclose(tw.Q[(0, 0, 0)][sel], Q_th[sel], rtol=0.04)
    # isotropy: the s-quadrupole of an origin-centred sphere vanishes
    assert np.max(np.abs(tw.Q[(0, 0, 2)][sel])) < 0.03 * np.max(np.abs(Q_th[sel]))


def bruteforce_monopole(R, k_edges, kfun, nq=24):
    """2/I^2 * <P P |W~(k1-k2)|^2> over shells, for a top-hat sphere of radius R."""
    I = 4 * np.pi / 3 * R ** 3                                # per nbar^2 (nbar cancels)
    q = np.linspace(1e-6, 2 * k_edges[-1] + 0.05, 200001)
    wt = 4 * np.pi * R ** 3 * spherical_jn(1, q * R) / (q * R)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] * wt[1:] ** 2 + q[:-1] * wt[:-1] ** 2) * np.diff(q))])
    G = lambda x: np.interp(x, q, cum)
    x, wgl = np.polynomial.legendre.leggauss(nq)
    nb = len(k_edges) - 1
    C = np.zeros((nb, nb))
    for i in range(nb):
        k1 = 0.5 * (k_edges[i + 1] - k_edges[i]) * x + 0.5 * (k_edges[i + 1] + k_edges[i])
        w1 = 0.5 * (k_edges[i + 1] - k_edges[i]) * wgl * k1 ** 2 * kfun(k1)
        n1 = (k_edges[i + 1] ** 3 - k_edges[i] ** 3) / 3
        for j in range(nb):
            k2 = 0.5 * (k_edges[j + 1] - k_edges[j]) * x + 0.5 * (k_edges[j + 1] + k_edges[j])
            w2 = 0.5 * (k_edges[j + 1] - k_edges[j]) * wgl * k2 ** 2 * kfun(k2)
            n2 = (k_edges[j + 1] ** 3 - k_edges[j] ** 3) / 3
            K1, K2 = k1[:, None], k2[None, :]
            F = (G(K1 + K2) - G(np.abs(K1 - K2))) / (2 * K1 * K2)      # int dmu/2 |W~|^2
            C[i, j] = 2.0 / I ** 2 * np.sum(w1[:, None] * w2[None, :] * F) / (n1 * n2)
    return C


def test_monopole_vs_bruteforce():
    R, n, nbar = 500.0, 12000, 2e-4
    rnd, alpha = sphere_randoms(R, n, nbar=nbar)
    tr = Tracer('T', rnd, alpha)
    k, P0 = pk_model()
    model = PowerSpectrumModel()
    model.add(('T', 'T'), {0: (k, P0)})
    k_edges = np.arange(0.02, 0.13, 0.02)
    cov = GaussianCovariance([tr], k_edges, ells=(0,), L_max=0, s_max=2 * R, ds=2.0, ds_pair=10.0,
                             shot_noise=False, n_sub=3000)
    cov.compute_windows([('T', 'T')]).set_model(model)
    C = cov.block(('T', 'T'), ('T', 'T'), 0, 0)
    Cb = bruteforce_monopole(R, k_edges, lambda kk: np.interp(kk, k, P0))
    d = np.diag(C) / np.diag(Cb) - 1
    assert np.all(np.abs(d) < 0.03), d
    # first off-diagonal (window leakage) at the level set by pair-count noise
    o = np.diag(C, 1) / np.diag(Cb, 1) - 1
    assert np.all(np.abs(o) < 0.25), o


def _legendre4(l1, l2, L1, L2):
    x, w = np.polynomial.legendre.leggauss(40)
    return 0.5 * np.sum(w * eval_legendre(l1, x) * eval_legendre(l2, x) * eval_legendre(L1, x) * eval_legendre(L2, x))


def test_far_sphere_multipole_ratios():
    """Sphere far from the observer: x^ ~ n^ constant, so ratios C_{l1 l2}/C_00 approach the box values."""
    R, n, nbar = 500.0, 12000, 2e-4
    rnd, alpha = sphere_randoms(R, n, center=(0, 0, 40000.0), nbar=nbar)
    tr = Tracer('T', rnd, alpha)
    k, P0 = pk_model()
    f, b = 0.8, 2.0
    beta = f / b
    # Kaiser multipoles
    P2 = P0 * (4 * beta / 3 + 4 * beta ** 2 / 7) / (1 + 2 * beta / 3 + beta ** 2 / 5)
    P4 = P0 * (8 * beta ** 2 / 35) / (1 + 2 * beta / 3 + beta ** 2 / 5)
    model = PowerSpectrumModel()
    model.add(('T', 'T'), {0: (k, P0), 2: (k, P2), 4: (k, P4)})
    k_edges = np.array([0.06, 0.08, 0.10])
    cov = GaussianCovariance([tr], k_edges, ells=(0, 2, 4), L_max=4, s_max=2 * R, ds=2.0, ds_pair=10.0,
                             shot_noise=False, n_sub=2500)
    cov.compute_windows([('T', 'T')]).set_model(model)
    kc = 0.5 * (k_edges[:-1] + k_edges[1:])
    PL = {0: np.interp(kc, k, P0), 2: np.interp(kc, k, P2), 4: np.interp(kc, k, P4)}
    C = {(l1, l2): cov.block(('T', 'T'), ('T', 'T'), l1, l2) for l1 in (0, 2, 4) for l2 in (0, 2, 4)}
    for (l1, l2) in [(2, 2), (0, 2), (4, 4), (2, 4)]:
        box = (2 * l1 + 1) * (2 * l2 + 1) * sum(PL[L1] * PL[L2] * _legendre4(l1, l2, L1, L2)
                                                for L1 in (0, 2, 4) for L2 in (0, 2, 4))
        box00 = sum(PL[L1] * PL[L2] * _legendre4(0, 0, L1, L2) for L1 in (0, 2, 4) for L2 in (0, 2, 4))
        ratio_code = np.diag(C[(l1, l2)]) / np.diag(C[(0, 0)])
        ratio_box = box / box00
        assert np.allclose(ratio_code, ratio_box, rtol=0.10), (l1, l2, ratio_code, ratio_box)
        # exchange symmetry (exact for T1, approximate for T2)
        assert np.allclose(C[(l1, l2)], C[(l2, l1)].T, rtol=0.05, atol=0.02 * np.max(np.abs(C[(l1, l2)])))
