"""Tree-level trispectrum (T0) covariance (thecov.trispectrum)."""
import numpy as np
import pytest

from thecov import Tracer, GaussianCovariance, TrispectrumCovariance, CovarianceTemplates, DiscretenessCovariance
from thecov.trispectrum import (FG3, Z2, Z3, Bias, galileon_bias, trispectrum, parallelogram, multipoles, response_split,
                                LinearPower)
from thecov.ssc import _Z2


def p_lin():
    k = np.logspace(-4, 1, 3000)
    return k, 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)


BIAS = Bias(2.0, 0.8, -0.3, 0.2, 0.1, 0.1, 0.3)


def test_F3_angle_average_is_the_P13_kernel():
    x, w = np.polynomial.legendre.leggauss(64)
    k = np.repeat(np.array([[0.0], [0.0], [1.0]]), 64, 1)
    for r in (0.3, 0.7, 1.6):
        q = r * np.stack([np.sqrt(1 - x ** 2), 0 * x, x])
        lhs = np.sum(w * FG3(k, q, -q)[0]) / 2
        rhs = (12 / r ** 2 - 158 + 100 * r ** 2 - 42 * r ** 4
               + 3 / r ** 3 * (r ** 2 - 1) ** 3 * (7 * r ** 2 + 2) * np.log(abs((1 + r) / (1 - r)))) / (3024 * r ** 2)
        assert np.isclose(lhs, rhs, rtol=1e-10)


def test_kernels():
    rng = np.random.default_rng(1)
    a, b, c = rng.normal(size=(3, 3, 7))
    b1, b2, bs2, f = 2.0, 0.7, -0.4, 0.8
    assert np.allclose(Z2(a, b, galileon_bias(b1, b2, bs2), f), _Z2(a, b, b1, b2, bs2, f), rtol=1e-12)
    assert np.allclose(Z3(a, b, c, Bias(1, 0, 0, 0, 0, 0, 0), 0.0), FG3(a, b, c)[0], rtol=1e-12)    # real space, unbiased
    assert np.allclose(Z3(a, b, c, BIAS, f), Z3(c, a, b, BIAS, f), rtol=1e-12)                       # symmetric
    z0, z1 = Z3(a, -a, b, BIAS, f), Z3(a, -a * (1 - 1e-7), b, BIAS, f)                                # finite at q1 + q2 = 0
    assert np.all(np.isfinite(z0)) and np.allclose(z0, z1, rtol=1e-4)
    P = LinearPower(*p_lin())
    k1, k2 = 0.1 * rng.normal(size=(2, 3, 50))
    full, fast = trispectrum([k1, -k1, k2, -k2], P, BIAS, f), parallelogram(k1, k2, P, BIAS, f)
    for part in full:
        assert np.allclose(full[part], fast[part], rtol=1e-10)


def test_multipoles_independent_quadrature():
    """the (mu12, mu1, psi) quadrature against (mu1, mu2, phi) with n^ = z^, for l1 != l2 and k1 > k2"""
    P, f = LinearPower(*p_lin()), 0.7
    k1, k2 = 0.1, 0.05
    a = multipoles(k1, k2, P, BIAS, f)
    n, nphi = 24, 48
    x, w = np.polynomial.legendre.leggauss(n)
    ph = 2 * np.pi * (np.arange(nphi) + 0.5) / nphi
    m1, m2, p = np.meshgrid(x, x, ph, indexing='ij')
    W = np.einsum('i,j->ij', w / 2, w / 2)[..., None] / nphi
    s1, s2 = np.sqrt(1 - m1 ** 2), np.sqrt(1 - m2 ** 2)
    e1, e2 = np.stack([s1, 0 * s1, m1]), np.stack([s2 * np.cos(p), s2 * np.sin(p), m2])
    T = parallelogram(k1 * e1, k2 * e2, P, BIAS, f)
    L = lambda l, m: np.polynomial.legendre.Legendre.basis(l)(m)
    for part in T:
        for (l1, l2) in [(0, 0), (0, 2), (2, 0), (2, 4), (4, 2)]:
            b = (2 * l1 + 1) * (2 * l2 + 1) * np.sum(W * L(l1, m1) * L(l2, m2) * T[part])
            assert np.isclose(a[part][(l1, l2)], b, rtol=1e-6, atol=1e-6 * abs(a[part][(0, 0)]))


@pytest.fixture(scope='module')
def setup():
    rng = np.random.default_rng(0)
    R, nbar = 300.0, 1e-3
    x = rng.uniform(-R, R, size=(60000, 3))
    pos = x[np.sum(x * x, 1) < R * R][:20000] + [0, 0, 1e5]
    V = 4 / 3 * np.pi * R ** 3
    tr = Tracer('T', {'POSITION': pos, 'WEIGHT': np.ones(len(pos)), 'NZ': np.full(len(pos), nbar)}, nbar * V / len(pos))
    cov = GaussianCovariance([tr], np.array([0.05, 0.06, 0.15, 0.16]), ells=(0, 2), L_max=0)
    return cov, V


def test_covariance(setup):
    cov, V = setup
    t = TrispectrumCovariance(cov, p_lin(), BIAS, f=0.7, n_mid=32, n_end=16)
    assert np.isclose(t.window_integral('T'), 1 / V, rtol=0.02)
    comps = t.components([('T', 'T')])
    for C in comps.values():
        assert np.allclose(C, C.T)                                   # C_{l1 l2}(k1, k2) = C_{l2 l1}(k2, k1)
    C, index = t.covariance([('T', 'T')])
    assert C.shape == (6, 6) and len(index) == 6 and np.allclose(C, comps['snake'] + comps['star'])
    tp = TrispectrumCovariance(cov, p_lin(), BIAS, f=0.7, n_mid=32, n_end=16, n_workers=2)
    assert np.allclose(tp.covariance([('T', 'T')])[0], C, rtol=1e-12)
    # bin (k, k) of the diagonal block equals the multipoles at the bin's k node / V
    m = multipoles(0.055, 0.055, t.P, BIAS, 0.7, ells=(0, 2), mu12=t.mu12)
    assert np.isclose(C[0, 0], t.window_integral('T') * (m['snake'][(0, 0)] + m['star'][(0, 0)]), rtol=1e-10)


def test_templates_and_discreteness_components(setup, tmp_path):
    cov, V = setup
    d = DiscretenessCovariance(cov, p_lin(), b1=2.0, f=0.5, n_mu=8, n_phi=12, n_k=1)
    comps = d.components([('T', 'T')])
    C, _ = d.covariance([('T', 'T')])
    assert np.allclose(C, comps['B'] + comps['P'])
    G = np.eye(6)
    tpl = CovarianceTemplates(G).update(comps, prefix='disc_')
    assert np.allclose(tpl(), G + C) and np.allclose(tpl(disc_B=0.0), G + comps['P'])
    tpl.save(tmp_path / 't.npz')
    back = CovarianceTemplates.load(tmp_path / 't.npz')
    assert back.names == tpl.names and np.allclose(back(disc_P=2.0), tpl(disc_P=2.0))
    with pytest.raises(KeyError):
        tpl(snake=1.0)


def test_response_split_and_completion():
    rng = np.random.default_rng(3)
    k = np.linspace(0.01, 0.3, 12)
    n = 3 * len(k)
    G = np.diag(rng.uniform(1, 2, n))
    U = rng.normal(size=(n, 2)) * 0.3
    T = U @ U.T - np.diag(np.diag(U @ U.T))                    # a coupling with zero diagonal (not PSD with G)
    T *= 3.0
    parts = response_split(T, k, 0.08, G)
    assert np.allclose(parts['LL'] + parts['LH'] + parts['HH'], T)
    M = G + parts['LH']                                        # long x hard couplings on top of G
    assert np.linalg.eigvalsh(M).min() < 0                     # not a covariance on its own here
    assert np.linalg.eigvalsh(M + parts['completion']).min() > -1e-10   # Schur complement: PSD


def test_template_fit_recovers_amplitudes():
    rng = np.random.default_rng(5)
    n, N = 20, 4000
    G = np.diag(rng.uniform(1, 2, n))
    u = rng.normal(size=n); T1 = 0.3 * np.outer(u, u)
    v = np.linspace(-1, 1, n); T2 = 0.5 * np.outer(v, v)
    tpl = CovarianceTemplates(G).add('a', T1).add('b', T2)
    X = rng.multivariate_normal(np.zeros(n), tpl(a=1.5, b=0.5), size=N)
    amps, m2l = tpl.fit(np.cov(X.T), N)
    assert abs(amps['a'] - 1.5) < 0.25 and abs(amps['b'] - 0.5) < 0.15
    assert m2l <= tpl.loglike(np.cov(X.T), N, a=1.0, b=1.0)
