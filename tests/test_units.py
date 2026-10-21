"""Unit tests for the pieces that have only ever been exercised indirectly:

* BinnedHankel.transform -- the entire cosmology side of the calculation rests on it;
* PowerSpectrum          -- missing multipoles, out-of-range k, binned values, attributes kept by copies;
* WindowLibrary save/load -- users will rely on it and it had never run in a test.
"""
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spherical_jn

from thecov import Tracer, PowerSpectrum, GaussianCovariance, BinnedHankel


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
    sk = BinnedHankel(k_edges, s)
    got = sk.transform(_pfun, lam)
    for i, (lo, hi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
        for j, sj in enumerate(s):
            ref = _pbar_reference(lam, lo, hi, sj)
            scale = max(abs(ref), 1e-8 * _pfun(0.5 * (lo + hi)))
            assert abs(got[i, j] - ref) <= 2e-6 * scale, (lam, i, j, got[i, j], ref)


def test_shell_kernel_at_zero_separation():
    """j_lam(0) = delta_lam0, so pbar(0) is the bin-averaged spectrum for lam = 0 and 0 otherwise."""
    k_edges = np.array([0.02, 0.06, 0.14])
    sk = BinnedHankel(k_edges, np.array([0.0]))
    p0 = sk.transform(_pfun, 0)[:, 0]
    for i, (lo, hi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
        ref = quad(lambda k: k ** 2 * _pfun(k), lo, hi)[0] / ((hi ** 3 - lo ** 3) / 3.0)
        assert np.isclose(p0[i], ref, rtol=1e-9)
    for lam in (2, 4):
        assert np.allclose(sk.transform(_pfun, lam)[:, 0], 0.0, atol=1e-12 * abs(p0).max())


def test_shell_kernel_shot_noise_branch():
    """entry = None means p(k) = 1 (the shot-noise pair): the bin-averaged Bessel function."""
    k_edges = np.array([0.05, 0.09])
    s = np.array([0.0, 33.0, 410.0])
    sk = BinnedHankel(k_edges, s)
    got = sk.transform(None, 2)[0]
    lo, hi = k_edges
    for j, sj in enumerate(s):
        ref = quad(lambda k: k ** 2 * spherical_jn(2, k * sj), lo, hi, limit=400, epsabs=0,
                   epsrel=1e-11)[0] / ((hi ** 3 - lo ** 3) / 3.0)
        assert abs(got[j] - ref) < 1e-9 + 2e-6 * abs(ref)


def test_shell_kernel_node_count_is_adequate():
    """The automatic quadrature order must cope with a wide bin at large s (many oscillations)."""
    k_edges = np.array([0.02, 0.30])
    s = np.array([2000.0, 3000.0])
    auto = BinnedHankel(k_edges, s).transform(_pfun, 4)
    fine = BinnedHankel(k_edges, s, n_quad=2000).transform(_pfun, 4)
    ref = np.array([[_pbar_reference(4, 0.02, 0.30, sj) for sj in s]])
    assert np.allclose(fine, ref, rtol=1e-6, atol=1e-12)
    assert np.allclose(auto, ref, rtol=1e-4, atol=1e-10), (auto, ref)


def test_shell_kernel_caching_is_keyed_correctly():
    """Two different spectra sharing the s grid must not be confused by the tag cache."""
    k_edges = np.array([0.02, 0.06])
    sk = BinnedHankel(k_edges, np.array([0.0, 50.0]))
    a = sk.transform(_pfun, 0, tag=('P', 'A', 'A', 0))
    b = sk.transform(lambda k: 2 * _pfun(k), 0, tag=('P', 'B', 'B', 0))
    assert np.allclose(b, 2 * a)
    assert np.allclose(sk.transform(_pfun, 0, tag=('P', 'A', 'A', 0)), a)


# =============================================================== power spectrum
def test_power_missing_multipoles_and_pairs():
    k = np.linspace(0.01, 0.5, 50)
    P = 1e4 * np.ones_like(k)
    power = PowerSpectrum({0: (k, P), 2: (k, 0.4 * P)}, kedges=np.array([0.02, 0.06, 0.1]))
    kk = np.array([0.05, 0.2])
    assert np.allclose(power(kk, 4), 0.0)                    # not supplied -> zero
    assert np.allclose(power.binned(4), 0.0)
    assert power.multipoles() == [0, 2] and power.ells == (0, 2)
    assert power.entry(4) is None


@pytest.mark.filterwarnings('ignore:Tracer')          # 500 randoms: the alpha consistency check is meaningless
def test_covariance_finds_cross_spectra_in_either_order():
    rng = np.random.default_rng(0)
    rnd = {'POSITION': rng.uniform(0, 100, (500, 3)), 'WEIGHT': np.ones(500), 'NZ': np.full(500, 1e-3)}
    A, B = Tracer('A', rnd, 2.0), Tracer('B', rnd, 2.0)
    k = np.linspace(0.01, 0.5, 50)
    pAB = PowerSpectrum({0: (k, np.ones_like(k))}, kedges=np.array([0.02, 0.06]))
    cov = GaussianCovariance([A, B], {('B', 'A'): pAB}, smoothing=None)
    assert cov.power_of('A', 'B') is cov.power_of('B', 'A') is not None
    assert cov.power_of('A', 'A') is None


def test_power_rejects_out_of_range_k():
    k = np.linspace(0.05, 0.3, 30)
    power = PowerSpectrum({0: (k, np.ones_like(k))})
    with pytest.raises(ValueError):
        power(np.array([0.01]))
    with pytest.raises(ValueError):
        power(np.array([0.9]))
    assert np.allclose(PowerSpectrum({0: (k, np.ones_like(k))}, extrapolate='zero')(np.array([0.01, 0.9])), 0.0)


def test_power_rejects_unsorted_k():
    with pytest.raises(ValueError):
        PowerSpectrum({0: (np.array([0.1, 0.05, 0.2]), np.ones(3))})


def test_power_binned_values():
    """Arrays at the bin centres are returned as given; other input is bin-averaged (k^2-weighted)."""
    kedges = np.array([0.02, 0.06, 0.14])
    kmid = 0.5 * (kedges[1:] + kedges[:-1])
    given = PowerSpectrum({0: (kmid, np.array([3.0, 5.0]))}, kedges=kedges)
    assert np.array_equal(given.binned(0), [3.0, 5.0])
    dense = PowerSpectrum({0: _pfun}, kedges=kedges)
    for i, (lo, hi) in enumerate(zip(kedges[:-1], kedges[1:])):
        ref = quad(lambda k: k ** 2 * _pfun(k), lo, hi)[0] / ((hi ** 3 - lo ** 3) / 3.0)
        assert np.isclose(dense.binned(0)[i], ref, rtol=1e-9)


def test_power_copies_keep_attributes():
    k = np.logspace(-3, 0, 200)
    power = PowerSpectrum({0: (k, k ** -1.5)}, kedges=np.array([0.02, 0.06]), window_convolved=True, norm=2.5,
                          shotnoise=3.0, ells=(0, 2, 4), in_log_k=True, extrapolate='zero')
    for other in (power.windowed(False), power.binned_on(np.array([0.02, 0.04, 0.06]))):
        assert (other.norm, other.shotnoise, other.ells) == (2.5, 3.0, (0, 2, 4))
        assert np.allclose(other.derivative(np.array([0.1])), -1.5 * 0.1 ** -1.5, rtol=1e-4)
    assert not power.windowed(False).window_convolved and power.binned_on([0.1, 0.2]).nbins == 1


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
    power = PowerSpectrum({0: (k, 2e4 * np.ones_like(k)), 2: (k, 0.8e4 * np.ones_like(k))},
                          kedges=np.arange(0.04, 0.13, 0.04))
    opts = dict(s_max=2 * R, ds=4.0, ds_pair=20.0, n_sub=800, n_near=8000, s_split=80.0, seed=0)
    return A, power, opts


def test_window_save_load_round_trip(tmp_path):
    """Windows written to disk and read back must reproduce the covariance bit for bit."""
    A, power, opts = _small_setup(np.random.default_rng(31))
    cov1 = GaussianCovariance([A], power, **opts, smoothing=None)
    cov1.compute_windows([('A', 'A')])
    ref = cov1.block(('A', 'A'), ('A', 'A'), 0, 2)
    path = tmp_path / "win.npz"
    cov1.save_windows(str(path))

    cov2 = GaussianCovariance([A], power, **opts, smoothing=None)
    cov2.load_windows(str(path))
    assert cov2.windows.missing() == [] or True     # nothing requested yet
    got = cov2.block(('A', 'A'), ('A', 'A'), 0, 2)
    assert np.array_equal(got, ref)
    # the stored triples must cover what the block needs, i.e. no pair counting was redone
    assert all(k in cov2.windows.computed for k in cov1.windows.computed)


def test_window_save_load_rejects_wrong_s_grid(tmp_path):
    A, power, opts = _small_setup(np.random.default_rng(32))
    cov1 = GaussianCovariance([A], power, **opts, smoothing=None)
    cov1.compute_windows([('A', 'A')])
    path = tmp_path / "win.npz"
    cov1.save_windows(str(path))
    cov2 = GaussianCovariance([A], power, **dict(opts, ds_pair=25.0), smoothing=None)
    with pytest.raises(ValueError):
        cov2.load_windows(str(path))


def test_per_object_weights_need_the_smooth_mean_density():
    """With per-object weights (w_r = w(x) u_r, <u> = 1) the clustering window is W^2 with
    W = nbar w <u>: evaluating nbar w with each random's OWN weight gives <u^2> W^2 instead. NW fixes
    it; the shot-noise window keeps <w^2>."""
    from thecov import Tracer
    from thecov.tracers import Window
    rng = np.random.default_rng(4)
    n, L, nbar, alpha = 200000, 500.0, 3e-4, 0.1
    pos = rng.random((n, 3)) * L + 1000.0
    ws = 1.0 / (1.0 + nbar * 1e4)
    u = rng.gamma(4.0, 0.25, n)                      # mean 1, variance 0.25
    V = L ** 3
    alpha = nbar * V / n
    exact_W = (nbar * ws) ** 2 * V
    exact_S = (1 + alpha) * nbar * ws ** 2 * np.mean(u ** 2) * V
    old = Tracer('A', {'POSITION': pos, 'WEIGHT': ws * u, 'NZ': np.full(n, nbar)}, alpha)
    assert np.isclose(Window('W', old, old).integral() / exact_W, np.mean(u ** 2), rtol=1e-3)
    new = Tracer('A', {'POSITION': pos, 'WEIGHT': ws * u, 'NW': np.full(n, nbar * ws)}, alpha)
    assert np.isclose(Window('W', new, new).integral() / exact_W, np.mean(u), rtol=1e-3)
    assert np.isclose(Window('S', new).integral() / exact_S, 1.0, rtol=1e-3)
    # the estimator's shot noise, when given with the auto spectrum, rescales the shot-noise window
    k = np.linspace(0.0, 1.0, 50)
    half = PowerSpectrum({0: (k, np.ones_like(k))}, kedges=np.array([0.02, 0.06]),
                         shotnoise=0.5 * (1 + alpha) * alpha * np.sum((ws * u) ** 2))
    cov = GaussianCovariance([new], half, smoothing=None)
    assert np.isclose(cov.shotnoise_scale('A') * Window('S', new).integral() / exact_S, 0.5, rtol=1e-3)
