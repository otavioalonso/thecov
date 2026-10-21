"""Pair-averaged clustering windows (thecov.smoothing): kernels, smoothing, I_k, covariance options."""
import numpy as np
import pytest

from thecov import Tracer, PowerSpectrum, GaussianCovariance, WindowSmoothing
from thecov.smoothing import xi0_from_power, bin_kernels, hankel0, taper, DensityField


def pk_shape():
    k = np.linspace(0.005, 0.4, 200)
    return k, 2e4 * (k / 0.05) / (1 + (k / 0.05) ** 2.2)


def fiducial():
    return PowerSpectrum({0: pk_shape()})


def sphere(R, n, holes=(), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while sum(len(p) for p in out) < n:
        x = rng.uniform(-R, R, size=(4 * n, 3))
        keep = np.sum(x * x, 1) < R * R
        for c, rh in holes:
            keep &= np.sum((x - c) ** 2, 1) > rh * rh
        out.append(x[keep])
    return np.concatenate(out)[:n] + np.array([0.0, 0.0, 2000.0])    # observer far away


def test_kernels_unit_integral_and_basis():
    r = np.arange(0.125, 200, 0.25)
    xi = xi0_from_power(fiducial(), r)
    K = bin_kernels(np.arange(0.02, 0.31, 0.01), r, xi)
    assert np.allclose(hankel0(r, K, np.array([0.0]))[:, 0], 1.0, atol=1e-6)
    # the high-k kernels are compact (a few Mpc/h): their transform is ~1 at q = 0.05
    Kq = hankel0(r, K, np.array([0.05]))[:, 0]
    assert abs(Kq[-1] - 1) < 0.02
    T = taper(r, 8.0, 2.0)
    assert T[0] == 1.0 and abs(T[-1]) < 1e-12


def test_uniform_window_smoothing_is_identity_inside():
    R, nbar = 300.0, 1e-3
    ref = sphere(R, 600_000, seed=1)
    V = 4 / 3 * np.pi * R ** 3
    alpha = nbar * V / len(ref)
    sm = WindowSmoothing(cell=6.0, r_max=120.0, tol=5e-3)
    sm.add_density('T', ref, np.ones(len(ref)), alpha)
    sm.set_power('T', 'T', fiducial())
    pos = sphere(R, 20_000, seed=2)
    tr = Tracer('T', {'POSITION': pos, 'WEIGHT': np.ones(len(pos)), 'NZ': np.full(len(pos), nbar)}, nbar * V / len(pos))
    sm.build(np.arange(0.02, 0.21, 0.02), {'T': tr}, [('T', 'T')], log=lambda *a: None)
    c = sm.coeffs('T', 'T')
    vals = np.stack([sm.values('T', 'T', b, 'T') for b in range(sm.n_basis('T', 'T'))])   # (B, N)
    W_k = c @ vals                                                       # (nbins, N): (K_k * W)(x)
    inner = np.linalg.norm(pos - [0, 0, 2000.0], axis=1) < R - 130
    assert np.allclose(W_k[:, inner].mean(1) / nbar, 1.0, atol=0.02)
    # near the edge the partner can fall outside: the pair-averaged window is lower there
    edge = np.linalg.norm(pos - [0, 0, 2000.0], axis=1) > R - 10
    assert np.all(W_k[:, edge].mean(1) < W_k[:, inner].mean(1))


def test_I_k_matches_pair_count_integral_with_holes():
    """I_k = int W (K_k * W) = int 4 pi r^2 K_k(r) Q_WW(r) dr, with Q_WW from brute-force pair counts."""
    R, nbar = 250.0, 1e-3
    rng = np.random.default_rng(3)
    holes = [(rng.uniform(-200, 200, 3) + [0, 0, 2000.0], rng.uniform(3, 12)) for _ in range(400)]
    holes = [(c - [0, 0, 2000.0], rh) for c, rh in holes]
    ref = sphere(R, 500_000, holes=holes, seed=4)
    V_eff = len(ref) / (len(sphere(R, 500_000, seed=4)) / (4 / 3 * np.pi * R ** 3))   # occupied volume
    alpha = nbar * V_eff / len(ref)
    sm = WindowSmoothing(cell=5.0, r_max=120.0, tol=2e-3)
    sm.add_density('T', ref, np.ones(len(ref)), alpha)
    sm.set_power('T', 'T', fiducial())
    pos = ref[::10]
    tr = Tracer('T', {'POSITION': pos, 'WEIGHT': np.ones(len(pos)), 'NZ': np.full(len(pos), nbar)}, alpha * 10)
    k_edges = np.arange(0.02, 0.21, 0.03)
    sm.build(k_edges, {'T': tr}, [('T', 'T')], log=lambda *a: None)
    Ik = sm.I_k('T', 'T')
    # brute force: Q_WW(r) from all pairs of a subsample of ref, in fine r bins
    from scipy.spatial import cKDTree
    sub = ref[::5]
    a_sub = alpha * 5
    r_edges = np.arange(0.0, 121.0, 1.0)
    tree = cKDTree(sub)
    cnt = tree.count_neighbors(tree, r_edges) - len(sub)                 # ordered pairs, self excluded
    dd = np.diff(cnt).astype(float)
    shell = 4 / 3 * np.pi * np.diff(r_edges ** 3)
    Q = a_sub ** 2 * dd / shell                                          # Q_WW(r) = int W W(x + r)
    rc = 0.5 * (r_edges[1:] + r_edges[:-1])
    xi = xi0_from_power(fiducial(), sm.r)
    Ki = bin_kernels(k_edges, sm.r, xi)
    Qr = np.interp(sm.r, rc, Q)
    I_bf = np.trapezoid(4 * np.pi * sm.r ** 2 * Ki * Qr[None, :], sm.r, axis=1)
    I_loc = tr.alpha * np.sum(tr.weights * tr.weighted_density)
    assert np.all(Ik < I_loc)                                             # holes dilute the pair-averaged window
    assert np.allclose(Ik / I_bf, 1.0, atol=0.02)
    # another binning: coefficients projected on the same basis (no new smoothing or pair counts)
    k2 = np.arange(0.02, 0.21, 0.06)
    K2 = bin_kernels(k2, sm.r, xi)
    I2_bf = np.trapezoid(4 * np.pi * sm.r ** 2 * K2 * Qr[None, :], sm.r, axis=1)
    assert np.allclose(sm.I_k('T', 'T', k2) / I2_bf, 1.0, atol=0.02)
    # bins beyond the built range: their (more compact) kernels are not in the basis, and the projection says so
    with pytest.warns(UserWarning, match='rebuild the smoothing'):
        sm.I_k('T', 'T', np.arange(0.25, 0.41, 0.05))


@pytest.fixture(scope='module')
def small_cov_setup():
    R, nbar = 300.0, 1e-3
    pos = sphere(R, 15_000, seed=5)
    V = 4 / 3 * np.pi * R ** 3
    tr = Tracer('T', {'POSITION': pos, 'WEIGHT': np.ones(len(pos)), 'NZ': np.full(len(pos), nbar)}, nbar * V / len(pos))
    k, p = pk_shape()
    return tr, {0: (k, p)}, k, p, nbar, V


def test_window_convolved_power_equals_rescaled_power(small_cov_setup):
    tr, multipoles, k, p, nbar, V = small_cov_setup
    k_edges = np.arange(0.04, 0.17, 0.04)
    kw = dict(n_sub=1500, n_near=8000)
    cov = GaussianCovariance([tr], k_edges, **kw, smoothing=None)
    cov.compute_windows([('T', 'T')])
    norm = 0.9 * cov.I_local('T', 'T')
    cov.set_power(PowerSpectrum(multipoles, kedges=k_edges, window_convolved=True, norm=norm))
    C1, _ = cov.covariance([('T', 'T')])
    cov.set_power(PowerSpectrum({0: (k, p * norm / cov.I_local('T', 'T'))}, kedges=k_edges, norm=norm))
    C2, _ = cov.covariance([('T', 'T')])
    assert np.allclose(C1, C2, rtol=1e-10)


def test_smoothed_covariance_close_to_local_in_hole_free_window(small_cov_setup):
    tr, multipoles, k, p, nbar, V = small_cov_setup
    ref = sphere(300.0, 400_000, seed=6)
    sm = WindowSmoothing(cell=6.0, r_max=100.0, tol=5e-3)
    sm.add_density('T', ref, np.ones(len(ref)), nbar * V / len(ref))
    sm.set_power('T', 'T', PowerSpectrum({0: (k, p)}))
    power = PowerSpectrum(multipoles, kedges=np.arange(0.06, 0.17, 0.05))
    kw = dict(n_sub=1500, n_near=8000)
    c0 = GaussianCovariance([tr], power, **kw, smoothing=None)
    c1 = GaussianCovariance([tr], power, smoothing=sm, **kw)
    C0, _ = c0.covariance([('T', 'T')])
    C1, _ = c1.covariance([('T', 'T')])
    d0, d1 = np.diag(C0), np.diag(C1)
    # no holes: only the edge layer (kernel width / R) differs; the smoothed window is slightly smaller
    assert np.all(d1 < d0 * 1.01) and np.all(d1 > d0 * 0.85)
    assert sm.n_basis('T', 'T') >= 1


def test_automatic_smoothing_is_the_explicit_one(small_cov_setup):
    """smoothing='auto' (the default) = a WindowSmoothing with the tracer's own randoms and the spectrum's P_0, set by
    hand; and a WindowSmoothing given nothing takes the same defaults."""
    tr, multipoles, k, p, nbar, V = small_cov_setup
    power = PowerSpectrum({0: (k, p)}, kedges=np.arange(0.04, 0.17, 0.04))
    kw = dict(n_sub=1500, n_near=8000)
    auto = GaussianCovariance([tr], power, **kw)
    auto.compute_windows([('T', 'T')])
    sm = WindowSmoothing()
    sm.add_density('T', tr.positions, tr.weights, tr.alpha)
    sm.set_power('T', 'T', power)
    explicit = GaussianCovariance([tr], power, smoothing=sm, **kw)
    explicit.compute_windows([('T', 'T')])
    assert np.allclose(auto.I_k('T', 'T'), explicit.I_k('T', 'T'), rtol=1e-12)
    assert np.allclose(auto.covariance([('T', 'T')])[0], explicit.covariance([('T', 'T')])[0], rtol=1e-12)
    defaults = GaussianCovariance([tr], power, smoothing=WindowSmoothing(), **kw)
    defaults.compute_windows([('T', 'T')])
    assert np.allclose(defaults.I_k('T', 'T'), explicit.I_k('T', 'T'), rtol=1e-12)


def test_automatic_smoothing_needs_the_power_first(small_cov_setup):
    tr, multipoles, k, p, nbar, V = small_cov_setup
    cov = GaussianCovariance([tr], np.arange(0.04, 0.17, 0.04), n_sub=1500, n_near=8000)
    with pytest.raises(RuntimeError, match='set the power'):
        cov.compute_windows([('T', 'T')])
