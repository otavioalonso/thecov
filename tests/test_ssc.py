"""Super-sample covariance (thecov.ssc): responses, long-mode variances, assembly."""
import numpy as np
import pytest
from scipy.special import spherical_jn

from thecov import Tracer, PowerSpectrum, GaussianCovariance, SuperSampleCovariance, response_coefficients


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
    power = PowerSpectrum({0: (k, 4.0 * P), 2: (k, 1.0 * P)}, kedges=np.arange(0.02, 0.2001, 0.02))
    cov = GaussianCovariance([tr], power, n_sub=6000, n_near=30000, ds_pair=10.0, smoothing=None)
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
    assert np.isclose(sig[(('W', 0), ('M', 0))], s00, rtol=0.03)               # uniform: W^2 and W same shape


def test_ssc_assembly(sphere_ssc):
    """isotropic real-space tracer, uniform window, unit normalisation: P0-P0 block is
    (d R_0 - 2 P0 b1)(...) sigma_b^2 with the LA of alpha and norm (b1 delta_L each)"""
    ssc, R = sphere_ssc
    C, labels = ssc.covariance([('T', 'T')], ells=(0,))
    sig = ssc.sigma2('T')[(('W', 0), ('W', 0))]
    resp = ssc.responses()[(0, 0)]
    cov = ssc.cov
    P0 = ssc._bin_average(cov.power_of('T', 'T').function(0))       # not window-convolved, d = I_k / I = 1
    v = resp - 2 * ssc.b1 * P0
    tr = cov._tracer('T')
    N = tr.alpha * len(tr.weights)                                             # galaxies (unit weights)
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


def test_dressed_responses_reduce_to_tree_level():
    """response_multipoles with the undamped P_lin equals a P + c dP/dlnk; FoG damping lowers the high-k
    quadrupole response and keeps the nu^4 response zero"""
    from scipy.interpolate import CubicSpline
    from thecov.tree.responses import response_multipoles, response_coefficients
    from thecov.tree.power import Dressed
    k = np.logspace(-4, 1, 3000)
    P = 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)
    b1, f, b2, bs2 = 2.2, 0.76, 0.3, -0.7
    a, c = response_coefficients(b1, f, b2, bs2)
    spl = CubicSpline(np.log(k), P)
    ks = np.array([0.03, 0.1, 0.25])
    R = response_multipoles(ks, Dressed(k, P), b1, f, b2, bs2)
    for i, l in enumerate((0, 2, 4)):
        for j, n in enumerate((0, 2)):
            assert np.allclose(R[(l, n)], a[i, j] * spl(np.log(ks)) + c[i, j] * spl(np.log(ks), 1), rtol=1e-5)
    Rd = response_multipoles(ks, Dressed(k, P, 4.0), b1, f, b2, bs2)
    assert Rd[(2, 2)][-1] < 0.8 * R[(2, 2)][-1] and np.isclose(Rd[(0, 0)][0], R[(0, 0)][0], rtol=0.02)
    assert np.max(np.abs(Rd[(0, 4)])) < 1e-6 * np.max(np.abs(Rd[(0, 0)]))


def test_no_wiggle():
    from thecov.tree.power import no_wiggle, eh_nowiggle, ir_damped
    k = np.logspace(-4, 1, 3000)
    h, om, fb = 0.6766, 0.1424, 0.157
    smooth = eh_nowiggle(k, h, om, fb, n_s=0.965)
    P = smooth * (1 + 0.05 * np.sin(k * 105.0) * np.exp(-(k / 0.3) ** 2) * (k > 0.02))
    Pnw = no_wiggle(k, P, h, om, fb, n_s=0.965)
    sel = (k > 0.4) | (k < 0.005)
    assert np.allclose(Pnw[sel], smooth[sel], rtol=2e-3)                   # broadband unbiased
    assert np.allclose(ir_damped(k, P, Pnw, 0.0), P) and np.allclose(ir_damped(k, P, Pnw, 50.0)[k > 0.1], Pnw[k > 0.1])


def test_local_average_statistics_and_norm_kinds(sphere_ssc):
    """uniform sphere, real space, b1 = 2: delta_norm = 2 b1 D (data-randoms) has variance 4 b1^2 sigma_00 + Poisson,
    the P0-delta_norm covariance is (R - 2 b1 P0) 2 b1 sigma_00 - P0 Var_P + T; 'alpha' uses one factor and
    'randoms' two factors of the M average only"""
    ssc, R = sphere_ssc
    cov, b1 = ssc.cov, ssc.b1
    st = ssc.local_average_statistics('T', ells=(0,))
    sig = ssc.sigma2('T')
    s00 = sig[(('W', 0), ('W', 0))]
    assert np.isclose(st['sigma_norm_clustering'] ** 2, 4 * b1 ** 2 * s00, rtol=0.05)        # D^W ~ D^M for a uniform sphere
    var_p, J, J3 = ssc.discreteness_integrals('T')
    assert np.isclose(st['sigma_norm'] ** 2, st['sigma_norm_clustering'] ** 2 + var_p, rtol=1e-8)
    P0 = ssc._bin_average(cov.power_of('T', 'T').function(0))
    resp = ssc.responses()[(0, 0)]
    from scipy.interpolate import CubicSpline
    kl, Pl = p_lin()
    P2 = ssc._bin_average(lambda kk: CubicSpline(np.log(kl), Pl)(np.log(kk)) ** 2)
    T = 2 * P0 * J / cov.I('T', 'T') + (ssc.b2 + 2 / 3 * ssc.bs2) * b1 ** 2 * P2 * J3 / cov.I('T', 'T')   # <P_raw eps_N>
    d = ssc._dilution('T')                                                              # I_k / I (~1 for the sphere)
    c = {('W', 0): d * (resp - b1 * P0), ('M', 0): -b1 * d * P0}                        # the LA weights g_0 = b1
    expect = sum(c[x] * b1 * sig[(x, y)] for x in c for y in c) + T - d * P0 * var_p
    assert np.allclose(st['cov'], expect, rtol=1e-6)
    assert np.allclose(st['cov'], (resp - 2 * b1 * P0) * 2 * b1 * s00 + T - P0 * var_p, rtol=0.25)  # sphere: all sigma^2 ~ s00, d ~ 1
    assert np.allclose(st['slope'] * st['sigma_norm'] ** 2 * P0, st['cov'])
    assert np.all(np.abs(st['corr']) <= 1 + 1e-9) and np.all(st['corr'] < -0.9)       # one long mode: fully (anti)correlated
    G = np.diag(np.full(len(P0), 1e6))
    st2 = ssc.local_average_statistics('T', ells=(0,), C_total=G)
    assert np.allclose(st2['corr_total'], st['cov'] / np.sqrt(1e6 * st['sigma_norm'] ** 2))
    for kind, lam_M, lam_W in (('alpha', 1.0, 0.0), ('randoms', 2.0, 0.0), ('data-randoms', 1.0, 1.0)):
        s = SuperSampleCovariance(cov, p_lin(), b1=2.0, f=0.0, discreteness=False, norm_kind=kind)
        s.windows = ssc.windows
        cf = s.coefficients('T', (0,))
        assert np.allclose(cf[('W', 0)], d * (resp - lam_W * b1 * P0)) and (('M', 0) not in cf or np.allclose(cf[('M', 0)], -lam_M * b1 * d * P0))
        assert (('M', 0) in cf) == (lam_M > 0)
        st = s.local_average_statistics('T', ells=(0,))
        assert np.isclose(st['sigma_norm_clustering'] ** 2, (lam_M + lam_W) ** 2 * b1 ** 2 * s00, rtol=0.05)
    with pytest.raises(ValueError):
        SuperSampleCovariance(cov, p_lin(), b1=2.0, f=0.0, norm_kind='mesh')


def test_sigma2_thin_shell_against_fourier_space():
    """Thin shell segment seen from the observer (line of sight varies by 50 deg): all long-mode variances
    sigma^2_{(X n)(Y n')}, X, Y in {W, M}, n, n' in {0, 2}, against the exact Fourier-space expression
    int d^3q/(2pi)^3 P(q) W_X(q) W_Y^*(q) / (I_X I_Y) with W_{(X,n)}(q) = sum_r w_X,r L_n(q^.x^_r) e^{-iq.x_r} (type-1 NUFFTs,
    Riemann sum on a box 8x the survey). Independent of the pair counts and of the tripolar expansion; checks the
    lam = 2, 4 channels, the cross channels and the (W, M) ordering. 2026-10-27: all ten agree to 1 % (far-pair sampling)."""
    finufft = pytest.importorskip('finufft')
    from itertools import combinations_with_replacement
    rng = np.random.default_rng(11)
    Nr = 40_000
    r = rng.uniform(800 ** 3, 1000 ** 3, 4 * Nr) ** (1 / 3)
    ct = rng.uniform(np.cos(np.radians(50)), 1, 4 * Nr)
    ph = rng.uniform(0, 2 * np.pi, 4 * Nr)
    st = np.sqrt(1 - ct ** 2)
    pos = np.stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct], 1)[:Nr]
    r = r[:Nr]
    wgt = 1.0 + (r - 900.0) / 200.0                                       # radial weights: W (w^2) and M (w) windows differ
    nbar = 1e-3
    Vs = 4 / 3 * np.pi * (1000 ** 3 - 800 ** 3) * (1 - np.cos(np.radians(50))) / 2
    tr = Tracer('T', {'POSITION': pos, 'WEIGHT': wgt, 'NZ': np.full(Nr, nbar)}, nbar * Vs / Nr)
    k, P = p_lin()
    qmax = 0.06
    Pcut = P * np.exp(-(k / qmax) ** 8)                                     # band limit shared by both sides
    power = PowerSpectrum({0: (k, 4 * P), 2: (k, P)}, kedges=np.arange(0.02, 0.2001, 0.02))
    cov = GaussianCovariance([tr], power, n_sub=8000, n_near=40000, ds_pair=10.0, smoothing=None)
    ssc = SuperSampleCovariance(cov, (k, Pcut), b1=2.0, f=0.6, n_near=40000)
    sig = ssc.sigma2('T')
    # Fourier side
    Lb = 12000.0
    kf = 2 * np.pi / Lb
    nq = 2 * int(np.ceil(0.075 / kf)) + 2
    x0 = pos.min(0) - 10
    th = [np.ascontiguousarray(2 * np.pi * (pos[:, i] - x0[i]) / Lb) for i in range(3)]
    xh = pos / np.linalg.norm(pos, axis=1)[:, None]
    qo = 2 * np.pi * np.fft.fftfreq(nq, d=Lb / nq)
    QO = np.meshgrid(qo, qo, qo, indexing='ij')
    qm = np.sqrt(sum(Q ** 2 for Q in QO))
    with np.errstate(invalid='ignore', divide='ignore'):
        qh = [np.where(qm > 0, Q / qm, 0) for Q in QO]
    Pq = np.interp(qm.ravel(), k, Pcut).reshape(qm.shape)
    Pq[0, 0, 0] = 0.0
    opts = dict(eps=1e-9, modeord=1)
    wX = {'M': wgt, 'W': wgt * (nbar * wgt)}
    Wq, I = {}, {}
    for X, wx in wX.items():
        F0 = finufft.nufft3d1(*th, wx.astype(complex), (nq,) * 3, isign=-1, **opts)
        Q2 = np.zeros_like(F0)
        for (i, j) in combinations_with_replacement(range(3), 2):
            Fij = finufft.nufft3d1(*th, (wx * xh[:, i] * xh[:, j]).astype(complex), (nq,) * 3, isign=-1, **opts)
            Q2 += (1 if i == j else 2) * qh[i] * qh[j] * Fij
        Wq[(X, 0)], Wq[(X, 2)], I[X] = F0, 1.5 * Q2 - 0.5 * F0, wx.sum()
    for key, v in sig.items():
        (X, n), (Y, n2) = key
        ref = np.sum(Pq * (Wq[(X, n)] * np.conj(Wq[(Y, n2)])).real) / Lb ** 3 / (I[X] * I[Y])
        assert abs(v / ref - 1) < 0.04, (key, v, ref)


@pytest.mark.slow
def test_poisson_local_average_term_box():
    """Poisson-sampled Gaussian field in a periodic box (sigma_delta = 0.26, no clipping), P_hat normalised by the realised
    N^2 (pypower's norm: alpha and the data, lam_total = 2) versus the mean N: Cov_real - Cov_fixed = (-8 + 4) P_i P_j / N.
    6000 realisations give the mean over all (i, j) of the ratio to -4/N as 1.06 +- 0.20 (2026-10-27); 3000 here."""
    rng = np.random.default_rng(9)
    n, L = 48, 1000.0
    V = L ** 3
    kfb = 2 * np.pi / L
    kx = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    kk = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    Pk = 2e4 * (np.maximum(kk, 1e-4) / 0.05) ** -1.3 * np.exp(-(kk / 0.05) ** 8)
    amp = np.sqrt(Pk / V) * n ** 3
    amp[0, 0, 0] = 0
    Nbar = 30000
    edges = kfb * np.array([2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    ib = np.digitize(kk.ravel(), edges) - 1
    nbins = len(edges) - 1
    ib[(ib < 0) | (ib >= nbins)] = nbins
    nsh = np.bincount(ib, minlength=nbins + 1)[:nbins]
    Ns, raw = [], []
    for _ in range(3000):
        d = np.fft.ifftn(np.fft.fftn(rng.normal(size=(n, n, n))) / n ** 1.5 * amp).real
        cnt = rng.poisson(np.clip(Nbar / n ** 3 * (1 + d), 0, None)).astype(float)
        F = np.fft.fftn(cnt - cnt.mean())
        Ns.append(cnt.sum())
        raw.append((np.bincount(ib, np.abs(F.ravel()) ** 2, nbins + 1)[:nbins] / nsh - cnt.sum()) * V)
    Ns, raw = np.array(Ns), np.array(raw)
    Nm = Ns.mean()
    Pf, Pr = raw / Nm ** 2, raw / Ns[:, None] ** 2
    Pb = Pf.mean(0)
    Cf, Cr = np.cov(Pf.T), np.cov(Pr.T)
    rel = np.mean([(Cr[i, j] - Cf[i, j]) / (Pb[i] * Pb[j]) for i in range(nbins) for j in range(nbins)])
    assert abs(rel / (-4 / Nm) - 1) < 0.6
