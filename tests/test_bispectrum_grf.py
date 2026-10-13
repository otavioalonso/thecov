"""Gaussian-field smoke test of the bispectrum estimator and the class-1 covariance (slow; the full validation run is
desi_validation/bispectrum_dev/grf_sphere.py + grf_theory.py + grf_compare.py, documented in
report/bispectrum_implementation.md).

Checked here, on a 96^3 mesh (k_Nyquist = 0.23) with shells up to k = 0.16 and few realisations: (i) the GRF power is normalised; (ii) the Sugiyama estimator's
normalisation through the mean B_000 of a quadratic field, B = 2 eps [P P + 2 perms] (shell-averaged tree level), on
the high-k elements where the bin-centre approximation of the third side is adequate; (iii) the periodic-box GRF
variance exceeds the analytic class-1 term by the class-2 fraction, i.e. lies between 1 and 2.5 times it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'desi_validation' / 'bispectrum_dev'))
import grf_sphere as G  # noqa: E402

pytestmark = pytest.mark.slow


K_EDGES = np.array([0.04, 0.08, 0.12, 0.16])


@pytest.fixture(scope='module')
def mesh():
    return G.Mesh(L=1300.0, N=96, k_edges=K_EDGES)


def test_grf_power(mesh):
    d = mesh.grf(3)
    Pk = np.abs(sfft.fftn(d) * mesh.dV) ** 2 / mesh.L ** 3
    for s in mesh.shells:
        assert abs(Pk[s].mean() / G.P_model(mesh.k[s]).mean() - 1) < 0.05


def test_estimator_normalisation(mesh):
    nbar, eps, n_real = 3e-4, 0.3, 60
    I3 = nbar ** 3 * mesh.L ** 3
    acc = []
    for seed in range(n_real):
        d = mesh.grf(1000 + seed)
        d = d + eps * (d ** 2 - np.mean(d ** 2))
        acc.append(mesh.measure(nbar * d, I3)[(0, 0, 0)])
    acc = np.array(acc)
    mean, err = acc.mean(0), acc.std(0) / np.sqrt(n_real)
    kc = 0.5 * (K_EDGES[1:] + K_EDGES[:-1])
    nb = len(kc)
    n = 0
    for i in range(nb):
        for j in range(i, nb):
            if i >= 1 and j >= 1:
                mu = np.linspace(-1, 1, 2001)
                k3 = np.sqrt(kc[i] ** 2 + kc[j] ** 2 + 2 * kc[i] * kc[j] * mu)
                Pi, Pj, P3 = G.P_model(kc[i]), G.P_model(kc[j]), np.mean(G.P_model(k3))
                pred = 2 * eps * (Pi * Pj + Pi * P3 + Pj * P3)
                assert abs(mean[n] / pred - 1) < 0.1 + 3 * err[n] / pred, (i, j, mean[n], pred)
            n += 1


def test_box_variance_vs_class1(mesh):
    from grf_theory import box_class1
    nbar, n_real = 3e-4, 150
    I3 = nbar ** 3 * mesh.L ** 3
    rows = np.array([mesh.measure(nbar * mesh.grf(5000 + s), I3)[(0, 0, 0)] for s in range(n_real)])
    var = rows.var(0, ddof=1)
    Vi = mesh.nmodes / mesh.L ** 3
    C4 = box_class1((0, 0, 0), (0, 0, 0), {0: G.P_model}, K_EDGES, mesh.L ** 3, Vi, nq=4, nth=12, nph=24)
    nb = len(Vi)
    n = 0
    for i in range(nb):
        for j in range(i, nb):
            ratio = var[n] / C4[i, j, i, j]
            assert 0.95 < ratio < 2.6, (i, j, ratio)
            n += 1
