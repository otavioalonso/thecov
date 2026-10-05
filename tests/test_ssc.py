"""Super-sample covariance (thecov.ssc): responses, long-mode variances, assembly."""
import numpy as np
import pytest
from scipy.special import spherical_jn

from thecov import Tracer, PowerSpectrumModel, GaussianCovariance, SuperSampleCovariance, response_coefficients


def test_real_space_responses():
    """matter, real space: isotropic 47/21 - n/3, tidal (2/3)(8/7 - n) in ell = n = 2, nothing else"""
    a, c = response_coefficients(1.0)
    exp_a, exp_c = np.zeros((3, 3)), np.zeros((3, 3))
    exp_a[0, 0], exp_c[0, 0] = 47 / 21, -1 / 3
    exp_a[1, 1], exp_c[1, 1] = 2 / 3 * 8 / 7, -2 / 3
    assert np.allclose(a, exp_a, atol=1e-6) and np.allclose(c, exp_c, atol=1e-6)


def test_bias_terms_and_no_nu4():
    b1, b2, bs2 = 2.0, 0.7, -0.4
    a0, _ = response_coefficients(b1)
    a, c = response_coefficients(b1, b2=b2, bs2=bs2)
    assert np.isclose(a0[0, 0], b1 ** 2 * 47 / 21, atol=1e-6)
    assert np.isclose(a[0, 0] - a0[0, 0], 2 * b1 * b2, atol=1e-6)            # 2 b1 b2 (isotropic)
    assert np.isclose(a[1, 1] - a0[1, 1], 4 / 3 * b1 * bs2, atol=1e-6)       # 2 b1 bs2 (mu_kq^2 - 1/3)
    a, c = response_coefficients(b1, f=0.8, b2=b2, bs2=bs2)
    assert np.allclose(a[:, 2], 0, atol=1e-6) and np.allclose(c[:, 2], 0, atol=1e-6)   # no nu^4 dependence
    assert a[0, 0] > a0[0, 0] and a[1, 0] > 0                                  # RSD adds isotropic response


def p_lin():
    k = np.logspace(-4, 1, 2000)
    return k, 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)


def sphere(R, n, center, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-R, R, size=(3 * n, 3))
    x = x[np.sum(x * x, 1) < R * R][:n]
    return x + np.asarray(center)


@pytest.fixture(scope='module')
def sphere_ssc():
    R, nbar, D = 300.0, 1e-3, 1e5          # distant observer: x^ ~ z^ everywhere
    pos = sphere(R, 30_000, [0.0, 0.0, D], seed=1)
    V = 4 / 3 * np.pi * R ** 3
    tr = Tracer('T', {'POSITION': pos, 'WEIGHT': np.ones(len(pos)), 'NZ': np.full(len(pos), nbar)}, nbar * V / len(pos))
    k, P = p_lin()
    cov = GaussianCovariance([tr], np.arange(0.02, 0.2001, 0.02), ells=(0, 2), L_max=2, n_sub=6000, n_near=30000,
                             ds_pair=10.0)
    model = PowerSpectrumModel()
    model.add(('T', 'T'), {0: (k, 4.0 * P), 2: (k, 1.0 * P)})
    cov.set_model(model)
    ssc = SuperSampleCovariance(cov, (k, P), b1=2.0, f=0.0, n_near=30000)
    return ssc, R


def test_sigma2_uniform_sphere(sphere_ssc):
    ssc, R = sphere_ssc
    sig = ssc.sigma2('T')
    k, P = p_lin()
    q = np.linspace(1e-5, 0.5, 200001)
    Pq = np.interp(q, k, P) * np.exp(-q ** 2)
    Wq = 3 * spherical_jn(1, q * R) / (q * R)
    s00 = np.trapezoid(q ** 2 * Pq * Wq ** 2, q) / (2 * np.pi ** 2)
    for X in ('W', 'M'):
        assert np.isclose(sig[((X, 0), (X, 0))], s00, rtol=0.03)
        assert np.isclose(sig[((X, 2), (X, 2))], s00 / 5, rtol=0.05)
        assert abs(sig[((X, 0), (X, 2))]) < 0.02 * s00
    assert np.isclose(sig[(('W', 0), ('M', 0))], s00, rtol=0.03)               # uniform: m^2 and m same shape


def test_ssc_assembly(sphere_ssc):
    """isotropic real-space tracer, uniform window, unit normalisation: P0-P0 block is
    (d R_0 - 2 P0 b1)(...) sigma_b^2 with the LA of alpha and norm (b1 delta_L each)"""
    ssc, R = sphere_ssc
    C, labels = ssc.covariance([('T', 'T')], ells=(0,))
    sig = ssc.sigma2('T')[(('W', 0), ('W', 0))]
    resp = ssc.responses()[(0, 0)]
    cov = ssc.cov
    P0 = ssc._bin_average(lambda kk: cov.model('T', 'T', 0, kk))       # unmasked model, d = I_k / I = 1
    v = resp - 2 * ssc.b1 * P0
    tr = cov._tracer('T')
    N = tr.alpha * len(tr.w)                                             # galaxies (unit weights)
    var, J, J3 = ssc.discreteness_integrals('T')
    assert np.isclose(var, 4.0 / N, rtol=0.03)                           # (1/N)(1 + 1)^2: alpha and the data in norm
    assert np.isclose(J / cov.I('T', 'T'), 2.0 / N, rtol=0.03)
    # Poisson part alone (no collapsed bispectrum): -2 (2/N + 2/N) P P + (4/N) P P = -4/N P P
    lin = SuperSampleCovariance(cov, p_lin(), b1=2.0, f=0.0, b2=0.0, bs2=0.0)
    assert np.allclose(lin.discreteness_covariance('T', (0,)), -4.0 / N * np.outer(P0, P0), rtol=0.05)
    noD = SuperSampleCovariance(cov, p_lin(), b1=2.0, f=0.0, discreteness=False)
    noD.windows = ssc.windows
    C0, _ = noD.covariance([('T', 'T')], ells=(0,))
    assert np.allclose(C0, np.outer(v, v) * sig, rtol=0.05)
    assert np.allclose(C - C0, ssc.discreteness_covariance('T', (0,)), rtol=1e-8)
    assert np.all(np.diag(C) > 0)
    ssc_noLA = SuperSampleCovariance(cov, p_lin(), b1=2.0, f=0.0, local_average=False)
    ssc_noLA.windows = ssc.windows                                       # same pair counts
    C2, _ = ssc_noLA.covariance([('T', 'T')], ells=(0,))
    assert np.allclose(C2, np.outer(resp, resp) * ssc_noLA.sigma2('T')[(('W', 0), ('W', 0))], rtol=1e-8)
