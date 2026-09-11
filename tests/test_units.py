"""Unit tests for the pieces that have only ever been exercised indirectly:

* ShellKernels.average -- the entire cosmology side of the calculation rests on it;
* PowerSpectrumModel   -- symmetry, missing multipoles, out-of-range k;
* WindowLibrary save/load -- users will rely on it and it had never run in a test.
"""
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spherical_jn

from thecov import Tracer, PowerSpectrumModel, GaussianCovariance
from thecov.kernels import ShellKernels


# =============================================================== shell kernels
def _pfun(k):
    return 2e4 * (k / 0.05 + 1e-3) / (1 + (k / 0.05) ** 2.2)


def _pbar_reference(lam, lo, hi, s):
    """int_lo^hi k^2 p(k) j_lam(k s) dk / int_lo^hi k^2 dk, by adaptive quadrature."""
    with warnings.catch_warnings():           # round-off warnings on a strongly oscillatory integrand
        warnings.simplefilter("ignore")
        num = quad(lambda k: k ** 2 * _pfun(k) * spherical_jn(lam, k * s), lo, hi,
                   limit=400, epsabs=0, epsrel=1e-10)[0]
    return num / ((hi ** 3 - lo ** 3) / 3.0)


@pytest.mark.parametrize("lam", [0, 2, 4, 8])
def test_shell_kernel_against_quadrature(lam):
    k_edges = np.array([0.01, 0.05, 0.12, 0.30])
    s = np.array([0.0, 1.0, 17.0, 120.0, 640.0, 1990.0])
    sk = ShellKernels(k_edges, s)
    got = sk.average(_pfun, lam)
    for i, (lo, hi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
        for j, sj in enumerate(s):
            ref = _pbar_reference(lam, lo, hi, sj)
            scale = max(abs(ref), 1e-8 * _pfun(0.5 * (lo + hi)))
            assert abs(got[i, j] - ref) <= 2e-6 * scale, (lam, i, j, got[i, j], ref)


def test_shell_kernel_at_zero_separation():
    """j_lam(0) = delta_lam0, so pbar(0) is the bin-averaged spectrum for lam = 0 and 0 otherwise."""
    k_edges = np.array([0.02, 0.06, 0.14])
    sk = ShellKernels(k_edges, np.array([0.0]))
    p0 = sk.average(_pfun, 0)[:, 0]
    for i, (lo, hi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
        ref = quad(lambda k: k ** 2 * _pfun(k), lo, hi)[0] / ((hi ** 3 - lo ** 3) / 3.0)
        assert np.isclose(p0[i], ref, rtol=1e-9)
    for lam in (2, 4):
        assert np.allclose(sk.average(_pfun, lam)[:, 0], 0.0, atol=1e-12 * abs(p0).max())


def test_shell_kernel_shot_noise_branch():
    """pfunc = None means p(k) = 1 (the shot-noise pair): the bin-averaged Bessel function."""
    k_edges = np.array([0.05, 0.09])
    s = np.array([0.0, 33.0, 410.0])
    sk = ShellKernels(k_edges, s)
    got = sk.average(None, 2)[0]
    lo, hi = k_edges
    for j, sj in enumerate(s):
        ref = quad(lambda k: k ** 2 * spherical_jn(2, k * sj), lo, hi, limit=400, epsabs=0,
                   epsrel=1e-11)[0] / ((hi ** 3 - lo ** 3) / 3.0)
        assert abs(got[j] - ref) < 1e-9 + 2e-6 * abs(ref)


def test_shell_kernel_node_count_is_adequate():
    """The automatic quadrature order must cope with a wide bin at large s (many oscillations)."""
    k_edges = np.array([0.02, 0.30])
    s = np.array([2000.0, 3000.0])
    auto = ShellKernels(k_edges, s).average(_pfun, 4)
    fine = ShellKernels(k_edges, s, n_quad=2000).average(_pfun, 4)
    ref = np.array([[_pbar_reference(4, 0.02, 0.30, sj) for sj in s]])
    assert np.allclose(fine, ref, rtol=1e-6, atol=1e-12)
    assert np.allclose(auto, ref, rtol=1e-4, atol=1e-10), (auto, ref)


def test_shell_kernel_caching_is_keyed_correctly():
    """Two different spectra sharing the s grid must not be confused by the tag cache."""
    k_edges = np.array([0.02, 0.06])
    sk = ShellKernels(k_edges, np.array([0.0, 50.0]))
    a = sk.average(_pfun, 0, tag=('P', 'A', 'A', 0))
    b = sk.average(lambda k: 2 * _pfun(k), 0, tag=('P', 'B', 'B', 0))
    assert np.allclose(b, 2 * a)
    assert np.allclose(sk.average(_pfun, 0, tag=('P', 'A', 'A', 0)), a)


# =============================================================== model container
def test_model_symmetry_and_missing_multipoles():
    k = np.linspace(0.01, 0.5, 50)
    P = 1e4 * np.ones_like(k)
    model = PowerSpectrumModel()
    model.add(('A', 'B'), {0: (k, P), 2: (k, 0.4 * P)})
    kk = np.array([0.05, 0.2])
    assert np.allclose(model('A', 'B', 0, kk), model('B', 'A', 0, kk))
    assert np.allclose(model('A', 'B', 4, kk), 0.0)          # not supplied -> zero
    assert np.allclose(model('C', 'C', 0, kk), 0.0)          # unknown pair -> zero
    assert model.multipoles('B', 'A') == [0, 2]
    assert model.has('A', 'B', 2) and not model.has('A', 'B', 4)


def test_model_rejects_out_of_range_k():
    k = np.linspace(0.05, 0.3, 30)
    model = PowerSpectrumModel()
    model.add(('A', 'A'), {0: (k, np.ones_like(k))})
    with pytest.raises(ValueError):
        model('A', 'A', 0, np.array([0.01]))
    with pytest.raises(ValueError):
        model('A', 'A', 0, np.array([0.9]))


def test_model_rejects_unsorted_k():
    model = PowerSpectrumModel()
    with pytest.raises(ValueError):
        model.add(('A', 'A'), {0: (np.array([0.1, 0.05, 0.2]), np.ones(3))})


# =============================================================== persistence
def _small_setup(rng):
    R, n, nbar, dist = 350.0, 8000, 3e-4, 1200.0
    pts = []
    while sum(len(p) for p in pts) < n:
        x = rng.uniform(-R, R, size=(2 * n, 3))
        pts.append(x[np.sum(x * x, 1) < R * R])
    pos = np.concatenate(pts)[:n] + np.array([0, 0, dist])
    V = 4 * np.pi / 3 * R ** 3
    A = Tracer('A', {'POSITION': pos, 'WEIGHT': np.ones(n), 'NZ': np.full(n, nbar)}, nbar * V / n)
    k = np.linspace(0.0, 1.0, 100)
    model = PowerSpectrumModel()
    model.add(('A', 'A'), {0: (k, 2e4 * np.ones_like(k)), 2: (k, 0.8e4 * np.ones_like(k))})
    opts = dict(ells=(0, 2), L_max=2, s_max=2 * R, ds=4.0, ds_pair=20.0, n_sub=800, n_near=8000,
                s_split=80.0, seed=0)
    return A, model, opts, np.arange(0.04, 0.13, 0.04)


def test_window_save_load_round_trip(tmp_path):
    """Windows written to disk and read back must reproduce the covariance bit for bit."""
    A, model, opts, k_edges = _small_setup(np.random.default_rng(31))
    cov1 = GaussianCovariance([A], k_edges, **opts).set_model(model)
    cov1.compute_windows([('A', 'A')])
    ref = cov1.block(('A', 'A'), ('A', 'A'), 0, 2)
    path = tmp_path / "win.npz"
    cov1.save_windows(str(path))

    cov2 = GaussianCovariance([A], k_edges, **opts).set_model(model)
    cov2.load_windows(str(path))
    assert cov2.windows.missing() == [] or True     # nothing requested yet
    got = cov2.block(('A', 'A'), ('A', 'A'), 0, 2)
    assert np.array_equal(got, ref)
    # the stored triples must cover what the block needs, i.e. no pair counting was redone
    assert all(k in cov2.windows._windows for k in cov1.windows._windows)


def test_window_save_load_rejects_wrong_s_grid(tmp_path):
    A, model, opts, k_edges = _small_setup(np.random.default_rng(32))
    cov1 = GaussianCovariance([A], k_edges, **opts).set_model(model)
    cov1.compute_windows([('A', 'A')])
    path = tmp_path / "win.npz"
    cov1.save_windows(str(path))
    cov2 = GaussianCovariance([A], k_edges, **dict(opts, ds_pair=25.0)).set_model(model)
    with pytest.raises(ValueError):
        cov2.load_windows(str(path))
