"""Multi-tracer tests. None of them needs mocks; each is an exact identity or an independent
brute-force reference.

1. Split-sample identity (any geometry): split one catalogue into halves A and B. The full-sample
   estimator is linear in the halves, P^TT = (P^AA + P^AB + P^BA + P^BB)/4 (the four ordered
   spectra, Legendre weight on the first label), so
       Cov(P^TT) = (1/16) sum over the 4 x 4 ordered-spectrum blocks
   must reproduce the single-tracer covariance of the full sample, window, leakage, multipoles and
   shot noise included (halves carry nbar/2 each, the cross has none).
1b. As a by-product of 1 and 2, the cross-window normalisation I_AB (which requires interpolating
   one tracer's nbar w at the other catalogue's positions) is checked against analytic values.
2. Periodic-box ratios (same cube, independent random catalogues, different nbar and spectra):
       C[P^AB_l1, P^CD_l2] = (2l1+1)(2l2+1)/N int dmu/2 L_l1 L_l2 [P^AD(mu) P^CB(mu) + P^AC(mu) P^DB(mu)],
   P^AA -> P^AA + (1+alpha_A)/nbar_A, times the common leakage factor of the cube (Monte-Carlo).
3. Two different footprints (top-hat sphere A, Gaussian profile B, independent randoms), l = 0,
   against the brute-force shell average of P P W~ W~ with analytic window transforms. This is
   the case where the cross window W^AB has to be interpolated from a different random catalogue.
"""
import numpy as np
import pytest
from scipy.special import spherical_jn, eval_legendre

from thecov import Tracer, PowerSpectrumModel, GaussianCovariance
from tests.test_box import leakage_fractions


def kaiser(P, b, f):
    beta = f / b
    return {0: b ** 2 * P * (1 + 2 * beta / 3 + beta ** 2 / 5),
            2: b ** 2 * P * (4 * beta / 3 + 4 * beta ** 2 / 7),
            4: b ** 2 * P * (8 * beta ** 2 / 35)}


def kaiser_cross(P, b1, b2, f):
    return {0: P * (b1 * b2 + (b1 + b2) * f / 3 + f ** 2 / 5),
            2: P * (2 * (b1 + b2) * f / 3 + 4 * f ** 2 / 7),
            4: P * (8 * f ** 2 / 35)}


# =============================================================================== 1. split sample
def test_split_sample_identity():
    R, N, nbar, dist = 500.0, 40000, 3e-4, 1500.0
    rng = np.random.default_rng(5)
    pts = rng.uniform(-R, R, size=(3 * N, 3))
    pts = pts[np.sum(pts ** 2, 1) < R * R][:N] + np.array([0, 0, dist])
    V = 4 * np.pi / 3 * R ** 3
    alpha = nbar * V / N
    half = N // 2
    make = lambda p, nb: {'POSITION': p, 'WEIGHT': np.ones(len(p)), 'NZ': np.full(len(p), nb)}
    T = Tracer('T', make(pts, nbar), alpha)
    A = Tracer('A', make(pts[:half], nbar / 2), alpha)      # N_gal/2 over N_ran/2: same alpha
    B = Tracer('B', make(pts[half:], nbar / 2), alpha)

    k = np.linspace(0.0, 1.0, 200)
    P = 2e4 * (k / 0.05 + 1e-3) / (1 + (k / 0.05) ** 2.2)
    mult = {L: (k, Pl) for L, Pl in kaiser(P, 1.8, 0.7).items()}
    model = PowerSpectrumModel()
    for pair in [('T', 'T'), ('A', 'A'), ('A', 'B'), ('B', 'B')]:
        model.add(pair, mult)

    k_edges = np.arange(0.02, 0.13, 0.02)
    opts = dict(ells=(0, 2), L_max=2, s_max=2 * R, ds=2.0, ds_pair=10.0, shot_noise=True,
                n_sub=2000, n_near=40000, s_split=80.0, seed=0)
    single = GaussianCovariance([T], k_edges, **opts).set_model(model)
    multi = GaussianCovariance([A, B], k_edges, **opts).set_model(model)
    spectra = [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]
    for l1 in (0, 2):
        for l2 in (0, 2):
            ref = single.block(('T', 'T'), ('T', 'T'), l1, l2)
            tot = sum(multi.block(sp1, sp2, l1, l2) for sp1 in spectra for sp2 in spectra) / 16.0
            d = np.diag(tot) / np.diag(ref) - 1
            assert np.all(np.abs(d) < 0.03), (l1, l2, d)
            assert np.linalg.norm(tot - ref) / np.linalg.norm(ref) < 0.05, (l1, l2)


# =============================================================================== 2. periodic box
def _legendre_int(l1, l2, PL_a, PL_b):
    """int dmu/2 L_l1 L_l2 P_a(mu) P_b(mu) for multipole dicts {L: value}."""
    x, w = np.polynomial.legendre.leggauss(40)
    Pa = sum(v * eval_legendre(L, x) for L, v in PL_a.items())
    Pb = sum(v * eval_legendre(L, x) for L, v in PL_b.items())
    return 0.5 * np.sum(w * eval_legendre(l1, x) * eval_legendre(l2, x) * Pa * Pb)


def box_multitracer_analytic(k_edges, V, spectra, AB, CD, l1, l2):
    """Periodic-box Gaussian covariance diagonal for ordered spectra AB, CD; spectra[(X,Y)] = {L: P_L}."""
    A, B = AB
    C, D = CD
    S = lambda X, Y: spectra[tuple(sorted((X, Y)))]
    Vi = (k_edges[1:] ** 3 - k_edges[:-1] ** 3) / 3.0 / (2 * np.pi ** 2)
    val = _legendre_int(l1, l2, S(A, D), S(C, B)) + _legendre_int(l1, l2, S(A, C), S(D, B))
    return (2 * l1 + 1) * (2 * l2 + 1) / (V * Vi) * val


def test_box_multitracer_ratios():
    L_box, dist = 2000.0, 2e5
    rng = np.random.default_rng(7)
    nb = {'A': 3e-5, 'B': 6e-5}
    alpha = {'A': 0.5, 'B': 0.25}          # randoms sample nbar/alpha: N_ran = nbar V / alpha
    tracers = []
    for name in ('A', 'B'):
        N = int(round(nb[name] * L_box ** 3 / alpha[name]))
        pos = rng.uniform(-L_box / 2, L_box / 2, size=(N, 3))
        pos[:, 0] += dist
        tracers.append(Tracer(name, {'POSITION': pos, 'WEIGHT': np.ones(N), 'NZ': np.full(N, nb[name])},
                              alpha=alpha[name]))
    k = np.linspace(0.0, 1.0, 20)
    Pflat = np.full_like(k, 1.5e4)
    bA, bB, f = 2.0, 1.2, 0.8
    spectra_L = {('A', 'A'): kaiser(1.5e4, bA, f), ('A', 'B'): kaiser_cross(1.5e4, bA, bB, f), ('B', 'B'): kaiser(1.5e4, bB, f)}
    model = PowerSpectrumModel()
    for pair, mult in spectra_L.items():
        model.add(pair, {L: (k, np.full_like(k, v)) for L, v in mult.items()})
    k_edges = np.arange(0.0, 0.205, 0.01)
    cov = GaussianCovariance(tracers, k_edges, ells=(0, 2), L_max=4, ds=2.0, ds_pair=10.0, shot_noise=True,
                             n_sub=3000, n_near=100000, s_split=80.0, seed=1).set_model(model)
    # shot noise (autos only) enters the periodic reference through the monopole
    ref_spectra = {pair: dict(m) for pair, m in spectra_L.items()}
    for name in ('A', 'B'):
        ref_spectra[(name, name)][0] += (1 + alpha[name]) / nb[name]
    retained, _ = leakage_fractions(L_box, k_edges)
    sel = slice(3, -1)
    spectra = [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]
    ratios = {}
    for sp1 in spectra:
        for sp2 in spectra:
            for l1 in (0, 2):
                for l2 in (0, 2):
                    C = cov.block(sp1, sp2, l1, l2)
                    A = box_multitracer_analytic(k_edges, L_box ** 3, ref_spectra, sp1, sp2, l1, l2)
                    r = np.diag(C) / (A * retained)
                    ratios[(sp1, sp2, l1, l2)] = r
                    if np.max(np.abs(A)) > 0:
                        assert np.allclose(r[sel], 1.0, atol=0.04), (sp1, sp2, l1, l2, r)
    # At fixed (l1, l2) the leakage is a property of the geometry, not of the tracer combination:
    # every tracer combination must share the same ratio to the periodic value. (Across different
    # (l1, l2) the leakage does differ -- the mu-structure of the integrand is not the same -- so the
    # comparison is made within each (l1, l2) group. The l1 != l2 blocks are small differences of
    # larger terms, so they carry more pair-count noise.)
    groups = {}
    for (sp1, sp2, l1, l2), r in ratios.items():
        groups.setdefault((l1, l2), []).append(((sp1, sp2), r[sel]))
    for (l1, l2), entries in groups.items():
        mean = np.mean([r for _, r in entries], axis=0)
        atol = 0.03 if l1 == l2 else 0.05
        for key, r in entries:
            assert np.allclose(r / mean, 1.0, atol=atol), (l1, l2, key, r / mean)


# =============================================================================== 3. two footprints
def _sphere_pts(rng, R, n):
    pts = []
    while sum(len(p) for p in pts) < n:
        x = rng.uniform(-R, R, size=(2 * n, 3))
        pts.append(x[np.sum(x * x, 1) < R * R])
    return np.concatenate(pts)[:n]


def bruteforce_l0(k_edges, windows, spectra, AB, CD, nq=24):
    """Brute-force C^{ABCD}_00(i,j) for isotropic spectra and spherically symmetric windows.

    windows[(X,Y)] -> list of (omega~(q) callable, p(k) callable) spectrum-window pairs (incl. shot noise);
    the pair sets follow the note: term 1 uses P^{AD} x P^{CB}, term 2 P^{AC} x P^{DB}.
    """
    A, B = AB
    C, D = CD
    key = lambda X, Y: tuple(sorted((X, Y)))
    I = lambda X, Y: windows[key(X, Y)][0][0](0.0)        # int omega = omega~(0) of the clustering window
    q = np.linspace(1e-6, 2 * k_edges[-1] + 0.05, 200001)
    x, wgl = np.polynomial.legendre.leggauss(nq)
    nb = len(k_edges) - 1
    C_out = np.zeros((nb, nb))
    for (pairs_x, pairs_xp) in [(windows[key(A, D)], windows[key(C, B)]), (windows[key(A, C)], windows[key(D, B)])]:
        for (wq, p) in pairs_x:              # at x-hat, momentum k2 (bin j)
            for (wqp, pp) in pairs_xp:       # at x'-hat, momentum k1 (bin i)
                f = q * wq(q) * wqp(q)
                cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(q))])
                G = lambda y: np.interp(y, q, cum)
                for i in range(nb):
                    k1 = 0.5 * (k_edges[i + 1] - k_edges[i]) * x + 0.5 * (k_edges[i + 1] + k_edges[i])
                    w1 = 0.5 * (k_edges[i + 1] - k_edges[i]) * wgl * k1 ** 2 * pp(k1)
                    n1 = (k_edges[i + 1] ** 3 - k_edges[i] ** 3) / 3
                    for j in range(nb):
                        k2 = 0.5 * (k_edges[j + 1] - k_edges[j]) * x + 0.5 * (k_edges[j + 1] + k_edges[j])
                        w2 = 0.5 * (k_edges[j + 1] - k_edges[j]) * wgl * k2 ** 2 * p(k2)
                        n2 = (k_edges[j + 1] ** 3 - k_edges[j] ** 3) / 3
                        K1, K2 = k1[:, None], k2[None, :]
                        F = (G(K1 + K2) - G(np.abs(K1 - K2))) / (2 * K1 * K2)
                        C_out[i, j] += np.sum(w1[:, None] * w2[None, :] * F) / (n1 * n2)
    return C_out / (I(A, B) * I(C, D))


def test_two_footprints_vs_bruteforce():
    RA, nA, NA = 600.0, 2e-4, 60000
    sig, n0, NB = 150.0, 8e-4, 200000
    rng = np.random.default_rng(11)
    posA = _sphere_pts(rng, RA, NA)
    posB = rng.normal(scale=sig, size=(NB, 3))
    posB = posB[np.sum(posB ** 2, 1) < RA * RA]              # truncate at R_A (mass beyond 4 sigma ~ 1e-4)
    nbB = n0 * np.exp(-np.sum(posB ** 2, 1) / (2 * sig ** 2))
    VA = 4 * np.pi / 3 * RA ** 3
    MB = n0 * (2 * np.pi * sig ** 2) ** 1.5
    alphaA, alphaB = nA * VA / NA, MB / NB
    A = Tracer('A', {'POSITION': posA, 'WEIGHT': np.ones(NA), 'NZ': np.full(NA, nA)}, alphaA)
    B = Tracer('B', {'POSITION': posB, 'WEIGHT': np.ones(len(posB)), 'NZ': nbB}, alphaB)

    k = np.linspace(0.0, 1.0, 300)
    shape = (k / 0.05 + 1e-3) / (1 + (k / 0.05) ** 2.2)
    amp = {('A', 'A'): 2.0e4, ('A', 'B'): 1.3e4, ('B', 'B'): 0.9e4}
    model = PowerSpectrumModel()
    for pair, a in amp.items():
        model.add(pair, {0: (k, a * shape)})
    k_edges = np.arange(0.02, 0.13, 0.02)
    cov = GaussianCovariance([A, B], k_edges, ells=(0,), L_max=0, s_max=2 * RA, ds=2.0, ds_pair=10.0,
                             shot_noise=True, n_sub=3000, n_near=200000, s_split=80.0, seed=2).set_model(model)

    # analytic window transforms (real, isotropic)
    sph = lambda R: (lambda q: 4 * np.pi * R ** 3 * np.where(q > 0, spherical_jn(1, q * R) / np.where(q > 0, q * R, 1), 1 / 3))
    gauss = lambda s2: (lambda q: (2 * np.pi * s2) ** 1.5 * np.exp(-q ** 2 * s2 / 2))
    pk = lambda pair: (lambda kk: np.interp(kk, k, amp[pair] * shape))
    one = lambda kk: np.ones_like(kk)
    windows = {
        ('A', 'A'): [(lambda q: nA ** 2 * sph(RA)(q), pk(('A', 'A'))), (lambda q: (1 + alphaA) * nA * sph(RA)(q), one)],
        ('A', 'B'): [(lambda q: nA * n0 * gauss(sig ** 2)(q), pk(('A', 'B')))],
        ('B', 'B'): [(lambda q: n0 ** 2 * gauss(sig ** 2 / 2)(q), pk(('B', 'B'))), (lambda q: (1 + alphaB) * n0 * gauss(sig ** 2)(q), one)],
    }
    # normalisations from the randoms vs analytic (checks the cross-window interpolation directly)
    assert np.isclose(cov.I('A', 'A'), nA ** 2 * VA, rtol=0.01)
    assert np.isclose(cov.I('B', 'B'), n0 ** 2 * (np.pi * sig ** 2) ** 1.5, rtol=0.03)
    assert np.isclose(cov.I('A', 'B'), nA * MB, rtol=0.02), (cov.I('A', 'B') / (nA * MB))

    spectra = [('A', 'A'), ('A', 'B'), ('B', 'B')]
    for a, sp1 in enumerate(spectra):
        for sp2 in spectra[a:]:
            C = cov.block(sp1, sp2, 0, 0)
            Cb = bruteforce_l0(k_edges, windows, amp, sp1, sp2)
            d = np.diag(C) / np.diag(Cb) - 1
            assert np.all(np.abs(d) < 0.04), (sp1, sp2, d)
            # The off-diagonals are judged on the scale of the diagonal, not relatively: these
            # windows are narrow in configuration space (sigma = 150, R = 600), so the bin-to-bin
            # leakage is a per-cent-level effect and a relative tolerance on it would only be
            # measuring pair-count noise on a small number.
            scale = np.sqrt(np.outer(np.diag(Cb), np.diag(Cb)))
            off = np.abs(C - Cb) / scale
            np.fill_diagonal(off, 0.0)
            assert np.max(off) < 0.03, (sp1, sp2, np.max(off), np.max(np.abs(np.diag(Cb, 1)) / np.diag(scale, 1)))
            # transpose symmetry of the ordered blocks
            assert np.allclose(cov.block(sp2, sp1, 0, 0), C.T, rtol=1e-10, atol=1e-12 * np.max(np.abs(C)))


def test_full_matrix_positive_definite():
    rng = np.random.default_rng(3)
    R, dist = 400.0, 1200.0
    make = lambda n, nb: {'POSITION': _sphere_pts(rng, R, n) + np.array([0, 0, dist]), 'WEIGHT': np.ones(n), 'NZ': np.full(n, nb)}
    V = 4 * np.pi / 3 * R ** 3
    A = Tracer('A', make(20000, 3e-4), 3e-4 * V / 20000)
    B = Tracer('B', make(20000, 6e-4), 6e-4 * V / 20000)
    k = np.linspace(0.0, 1.0, 200)
    P = 2e4 * (k / 0.05 + 1e-3) / (1 + (k / 0.05) ** 2.2)
    model = PowerSpectrumModel()
    model.add(('A', 'A'), {L: (k, v) for L, v in kaiser(P, 2.0, 0.8).items()})
    model.add(('B', 'B'), {L: (k, v) for L, v in kaiser(P, 1.2, 0.8).items()})
    model.add(('A', 'B'), {L: (k, v) for L, v in kaiser_cross(P, 2.0, 1.2, 0.8).items()})
    k_edges = np.arange(0.02, 0.17, 0.02)
    cov = GaussianCovariance([A, B], k_edges, ells=(0, 2), L_max=4, s_max=2 * R, ds=2.0, ds_pair=10.0,
                             n_sub=2500, n_near=20000, seed=4).set_model(model)
    C, labels = cov.covariance([('A', 'A'), ('A', 'B'), ('B', 'B')])
    ev = np.linalg.eigvalsh(C)
    assert ev[0] > 0 and len(labels) == C.shape[0]
