r"""Three-point window invariants Q_n(r1, r2) from anchored pair sums (thecov.bispectrum_windows) on a uniform sphere
centred on the observer, against Monte-Carlo quadrature of int d^3x dOmega1 dOmega2 W W W Phi_n
(desi_validation/bispectrum_dev/sphere_check.py).

The pair-sum estimate is unbiased but noisy: the anchors sample W(x) (a 2000-anchor subsample has ~1.5 % radial
fluctuations, which on a R = 500 sphere move the edge fraction by ~2 %), and the far neighbours are one catalogue shared
by all anchors (its density fluctuations do not average over anchors).  Tolerances below reflect n_anchor / n_sub.
Checked: the isotropic and anisotropic invariants at several (r1, r2); the r1 = r2 bins (self-pair subtraction) follow
the trend of the neighbouring off-diagonal bins; the small-r limit approaches the box value V sqrt(N/4pi) H times the
edge fraction; shot-noise kinds scale exactly; save/load and interpolation.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from thecov.tracers import Window
from thecov.bispectrum_algebra import box_window_value
from thecov.bispectrum_windows import AnchoredMoments, ThreePointWindowLibrary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'desi_validation' / 'bispectrum_dev'))
from sphere_check import sphere_tracer, mc_reference, R_SPHERE, ALPHA  # noqa: E402

R_EDGES = np.arange(0.0, 800.1, 20.0)
C = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
NS = [(0, 0, 0, 0, 0, 0, 0), (1, 0, 1, 1, 0, 1, 0), (0, 2, 2, 0, 2, 2, 4), (2, 2, 2, 0, 2, 2, 2)]


@pytest.fixture(scope='module')
def sphere():
    T, nbar, V = sphere_tracer()
    return T, nbar, V


def _moments(T, n_anchor, n_sub):
    wins = [Window('W', T, T), Window('S', T)]
    am = AnchoredMoments(wins, wins, R_EDGES, n_anchor=n_anchor, n_sub=n_sub, n_near=200000, s_split=80.0, seed=0)
    am.compute({(0, 0, 0): set(NS), (1, 0, 0): set(NS[:1]), (0, 1, 0): set(NS[:1])})
    return am


def test_sphere_invariants(sphere):
    T, nbar, V = sphere
    am = _moments(T, n_anchor=1500, n_sub=4000)
    for n in NS[:3]:
        Q = am.Q[(0, 0, 0, n)]
        for (b1, b2) in [(1, 1), (3, 10), (10, 25), (25, 25)]:
            est, err = mc_reference(n, C[b1], C[b2], nbar)
            if abs(est) > 20 * err:
                assert abs(Q[b1, b2] / est - 1) < 0.08, (n, b1, b2, Q[b1, b2], est)
    # small-r limit: the box value times the edge fraction (1 - 3 r / 4 R)^2 of the two legs
    n = NS[0]
    edge = (1 - 0.75 * C[1] / R_SPHERE) ** 2
    assert abs(am.Q[(0, 0, 0, n)][1, 1] / (box_window_value(n, V) * nbar ** 6 * edge) - 1) < 0.05
    # shot-noise kinds are the same pairs with other weights: exact ratios
    ratio = (1 + ALPHA) / nbar
    assert np.allclose(am.Q[(1, 0, 0, n)], ratio * am.Q[(0, 0, 0, n)], rtol=1e-10)
    assert np.allclose(am.Q[(0, 1, 0, n)], ratio * am.Q[(0, 0, 0, n)], rtol=1e-10)


@pytest.mark.slow
def test_sphere_invariants_converged(sphere):
    T, nbar, V = sphere
    am = _moments(T, n_anchor=5000, n_sub=10000)
    for n in NS:
        Q = am.Q[(0, 0, 0, n)]
        for (b1, b2) in [(1, 1), (2, 3), (3, 10), (6, 6), (10, 25), (25, 25), (15, 30)]:
            est, err = mc_reference(n, C[b1], C[b2], nbar, n_mc=800000)
            if abs(est) > 30 * err:
                assert abs(Q[b1, b2] / est - 1) < 0.03, (n, b1, b2, Q[b1, b2], est)
    # self-pair subtraction: the diagonal follows the off-diagonal trend (isotropic invariant, near regime)
    Q = am.Q[(0, 0, 0, NS[0])]
    for b in (1, 2, 3):
        est, _ = mc_reference(NS[0], C[b], C[b], nbar, n_mc=800000)
        est2, _ = mc_reference(NS[0], C[b], C[b + 1], nbar, n_mc=800000)
        assert abs(Q[b, b] / est - Q[b, b + 1] / est2) < 0.01


def test_library_roundtrip(sphere, tmp_path):
    T, nbar, V = sphere
    lib = ThreePointWindowLibrary(R_EDGES, n_anchor=500, n_sub=1500, n_near=50000)
    lib.request(T, ('W', 'W', 'W'), NS[:2])
    lib.request(T, ('S', 'W', 'W'), NS[:1])
    assert lib.missing()
    lib.compute_all()
    assert not lib.missing()
    r = np.arange(5.0, 600.0, 10.0)
    Q = lib.get(T.name, ('W', 'W', 'W'), NS[0], r)
    assert Q.shape == (len(r), len(r)) and np.all(np.isfinite(Q))
    # interpolation reproduces the bin values at the bin centres
    am = lib._windows[T.name]
    Qc = lib.get(T.name, ('W', 'W', 'W'), NS[0], C[2:20])
    assert np.allclose(Qc, am.Q[(0, 0, 0, NS[0])][2:20, 2:20], rtol=1e-8)
    assert lib.get(T.name, ('W', 'W', 'W'), (3, 0, 3, 3, 0, 3, 0), r) is None
    lib.save(tmp_path / 'tp.npz')
    lib2 = ThreePointWindowLibrary(R_EDGES)
    lib2.load(tmp_path / 'tp.npz')
    assert np.allclose(lib2.get(T.name, ('S', 'W', 'W'), NS[0], r), lib.get(T.name, ('S', 'W', 'W'), NS[0], r))
