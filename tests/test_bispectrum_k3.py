"""The k3 ~ 0 term (pairing (1 2)(1' 2')(3 3')) and the weight-proportional sampling of the three-point windows."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'desi_validation' / 'bispectrum_dev'))
from thecov import PowerSpectrumModel, BispectrumGaussianCovariance  # noqa: E402
from thecov.bispectrum import extrapolated_power  # noqa: E402

K_EDGES = np.arange(0.04, 0.2001, 0.04)
ELLS = ((0, 0, 0), (2, 0, 2))


def P_model(k):
    return 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)


def _cov(tracer, nbar, I3):
    kk = np.logspace(-4, 0.7, 4000)
    model = PowerSpectrumModel()
    model.add(('S', 'S'), {0: (kk, P_model(kk))})
    cov = BispectrumGaussianCovariance(tracer, K_EDGES, multipoles=ELLS, L_max=0, lam_w=0, r_max=1000.0, dr=4.0,
                                       shot_noise=False, class2=False)
    cov.set_model(model).set_normalization(I3)
    return cov


def test_k3_coefficient_and_overlap():
    from sphere_check import smooth_tracer
    T, nbar, V = smooth_tracer(n_ran=20000, R=600.0)
    cov = _cov(T, nbar, nbar ** 3 * (2 * np.pi * 200.0 ** 2 / 3) ** 1.5)
    assert cov.k3_coefficient((0, 0, 0)) == 1.0
    assert abs(cov.k3_coefficient((2, 0, 2)) - 1.0) < 1e-12
    P = extrapolated_power(cov.model, 'S', 'S', 0)
    K = cov.kernels
    for i in range(cov.nbins):
        pbar = np.sum(K.wq[i] * P(K.kq[i])) / K.norm[i]
        Vi = K.norm[i] / (2 * np.pi ** 2)
        assert abs(cov.shell_overlap(i, i, 0.0, P)[0] / (pbar / Vi) - 1) < 1e-10
        assert cov.shell_overlap(i, (i + 1) % cov.nbins, 0.0, P)[0] == 0.0
        # a displacement much smaller than the shell width barely changes the self-overlap; beyond twice the outer
        # edge no pair (q, -q + eps) can stay in the shell
        assert cov.shell_overlap(i, i, 1e-4, P)[0] / (pbar / Vi) > 0.99
        assert cov.shell_overlap(i, i, 2 * K_EDGES[i + 1] + 0.01, P)[0] == 0.0
        assert 0.3 < cov.shell_overlap(i, i, 0.5 * (K_EDGES[i + 1] - K_EDGES[i]), P)[0] / (pbar / Vi) < 0.9


def test_k3_vanishes_for_uniform_window():
    """int W F = 0 for a uniform omega after the alpha step: the leading k3 term is zero and the edge-driven overlap
    corrections are at the per-cent level of the Gaussian variance."""
    from sphere_check import sphere_tracer
    T, nbar, V = sphere_tracer(n_ran=1000000, R=500.0)      # the residual scales with the random shot noise (1/n_ran)
    cov = _cov(T, nbar, nbar ** 3 * V)
    K0 = cov.k3_zero_block((0, 0, 0), (0, 0, 0), n_mesh=64, overlap=False)
    K1 = cov.k3_zero_block((0, 0, 0), (0, 0, 0), n_mesh=64, overlap=True)
    ref = np.load(Path(__file__).resolve().parents[1] / 'desi_validation' / 'bispectrum_dev' / 'grf_sphere_theory_c12_lam2.npz')
    C = (ref['C_class1'] + ref['C_class2']).diagonal()[[0, 4, 7, 9]]            # (i, i) B000 elements
    d0 = np.array([K0[i, i, i, i] for i in range(4)]); d1 = np.array([K1[i, i, i, i] for i in range(4)])
    assert np.all(np.abs(d0) / C < 0.03)
    assert np.all(np.abs(d1) / C < 0.1)


def test_k3_gaussian_window_matches_analytic_mesh():
    """Gaussian window: the leading k3 term from the randoms agrees with the analytic-window mesh computation."""
    from sphere_check import smooth_tracer
    ref = np.load(Path(__file__).resolve().parents[1] / 'desi_validation' / 'bispectrum_dev' / 'grf_smooth_k3_B000.npz')
    T, nbar, V = smooth_tracer(n_ran=1000000, R=600.0)
    cov = _cov(T, nbar, nbar ** 3 * (2 * np.pi * 200.0 ** 2 / 3) ** 1.5)
    K0 = cov.k3_zero_block((0, 0, 0), (0, 0, 0), n_mesh=64, overlap=False)
    d0 = np.array([[K0[i, i, j, j] for j in range(4)] for i in range(4)])
    assert np.all(np.abs(d0 / ref['k3_diag_block'] - 1) < 0.08)
    K1 = cov.k3_zero_block((0, 0, 0), (0, 0, 0), n_mesh=64, overlap=True)
    d1 = np.array([[K1[i, i, j, j] for j in range(4)] for i in range(4)])
    r = d1.diagonal() / d0.diagonal()
    assert np.all((r > 0.6) & (r < 0.9))                                           # k_w / dk = 0.2: a 25-30 % reduction
    assert np.abs(cov.k3_zero_block((2, 0, 2), (2, 0, 2), n_mesh=64)).max() == 0.0   # isotropic P: Pbar_2 = 0


@pytest.mark.slow
def test_importance_sampled_q0_gaussian_window():
    """Q_0(r1, r2) of a Gaussian window from the pair sums against the closed form."""
    from thecov.tracers import Window
    from thecov.bispectrum_windows import AnchoredMoments
    from sphere_check import smooth_tracer
    R = 600.0; sig = R / 3
    T, nbar, V = smooth_tracer(R=R)
    edges = np.arange(0, 1201, 20.0)
    n0 = (0, 0, 0, 0, 0, 0, 0)
    wins = [Window('W', T, T)]
    am = AnchoredMoments(wins, wins, edges, n_anchor=3000, n_sub=6000, n_near=200000, seed=1)
    am.compute({(0, 0, 0): {n0}})
    Q = am.Q[(0, 0, 0, n0)]
    c = 0.5 * (edges[1:] + edges[:-1]); r1, r2 = np.meshgrid(c, c, indexing='ij'); a = 2 * r1 * r2 / (3 * sig ** 2)
    Qan = (nbar ** 6 * (4 * np.pi) ** -2.5 * (np.pi * sig ** 2 / 3) ** 1.5 * np.exp(-2 * (r1 ** 2 + r2 ** 2) / (3 * sig ** 2))
           * (4 * np.pi) ** 2 * np.sinh(a) / a)
    sel = (r1 < 400) & (r2 < 400)
    assert abs(np.sum(Q[sel] * r1[sel] ** 2 * r2[sel] ** 2) / np.sum(Qan[sel] * r1[sel] ** 2 * r2[sel] ** 2) - 1) < 0.05
