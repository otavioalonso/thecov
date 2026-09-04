"""Box test (after box_test.ipynb): uniform randoms in a cube, flat P_0, P_2 = P_4 = 0, shot noise.

The reference is the periodic-box Gaussian covariance

    C_{l1 l2}(k_i) = 2 (2 l1 + 1)(2 l2 + 1) / N_i  sum_{L1 L2} P_L1 P_L2 int dmu/2 L_l1 L_l2 L_L1 L_L2,
    N_i = V * V_i,  P_0 -> P_0 + (1 + alpha) / nbar,

whose explicit coefficients are the ones written in the notebook (checked in test_notebook_coefficients).

What to expect from a *windowed* (non-periodic) cube: the estimator picks up modes k2 = k1 + q with q
drawn from |W~(q)|^2, whose width is ~1/L. For L dk = 40 about 10 % of the variance of a bin leaks
into the two neighbouring bins, so the diagonal is ~0.9 of the periodic value and the neighbours
carry ~5 % each. The retained/leaked fractions are predicted independently here by Monte-Carlo
sampling of q from the cube's |W~|^2 (leakage_fractions), and the code must reproduce them.
"""
import numpy as np
import pytest
from scipy.special import eval_legendre

from pkcov import Tracer, PowerSpectrumModel, GaussianCovariance


# --------------------------------------------------------------------------- reference formulae
def legendre4(l1, l2, L1, L2):
    x, w = np.polynomial.legendre.leggauss(40)
    return 0.5 * np.sum(w * eval_legendre(l1, x) * eval_legendre(l2, x) * eval_legendre(L1, x) * eval_legendre(L2, x))


def analytic_box(k_edges, V, PL, ell1, ell2):
    """Periodic-box Gaussian covariance diagonal; PL = {L: P_L(k_i) array}."""
    Vi = (k_edges[1:] ** 3 - k_edges[:-1] ** 3) / 3.0 / (2 * np.pi ** 2)
    out = np.zeros(len(Vi))
    for L1, P1 in PL.items():
        for L2, P2 in PL.items():
            out += P1 * P2 * legendre4(ell1, ell2, L1, L2)
    return 2.0 * (2 * ell1 + 1) * (2 * ell2 + 1) / (V * Vi) * out


def notebook_analytic(P0, P2, P4):
    """The explicit coefficients from box_test.ipynb (times 2/N)."""
    return {
        (0, 0): P0 ** 2 + P2 ** 2 / 5 + P4 ** 2 / 9,
        (0, 2): 2 * P0 * P2 + 2 / 7 * P2 ** 2 + 4 / 7 * P2 * P4 + 100 / 693 * P4 ** 2,
        (2, 2): 5 * P0 ** 2 + 20 / 7 * P0 * P2 + 20 / 7 * P0 * P4 + 15 / 7 * P2 ** 2 + 120 / 77 * P2 * P4 + 8945 / 9009 * P4 ** 2,
        (0, 4): 2 * P0 * P4 + 18 / 35 * P2 ** 2 + 40 / 77 * P2 * P4 + 162 / 1001 * P4 ** 2,
        (2, 4): 36 / 7 * P0 * P2 + 200 / 77 * P0 * P4 + 108 / 77 * P2 ** 2 + 3578 / 1001 * P2 * P4 + 900 / 1001 * P4 ** 2,
        (4, 4): 9 * P0 ** 2 + 360 / 77 * P0 * P2 + 2916 / 1001 * P0 * P4 + 16101 / 5005 * P2 ** 2 + 3240 / 1001 * P2 * P4 + 42849 / 17017 * P4 ** 2,
    }


def leakage_fractions(L, k_edges, n_mc=300000, seed=1):
    """Monte-Carlo: for k1 uniform in each shell and q ~ |W~_cube(q)|^2, the fractions of k2 = k1 + q
    landing in the same bin and in the two neighbouring bins."""
    rng = np.random.default_rng(seed)
    qmax = 0.2
    qg = np.linspace(-qmax, qmax, 400001)
    cdf = np.cumsum(np.sinc(qg * L / 2 / np.pi) ** 2)
    cdf /= cdf[-1]
    draw = lambda n: np.interp(rng.uniform(0, 1, n), cdf, qg)
    same, neigh = [], []
    for lo, hi in zip(k_edges[:-1], k_edges[1:]):
        dk = hi - lo
        k1 = np.cbrt(rng.uniform(lo ** 3, hi ** 3, n_mc))
        u = rng.uniform(-1, 1, n_mc)
        ph = rng.uniform(0, 2 * np.pi, n_mc)
        kv = np.c_[k1 * np.sqrt(1 - u ** 2) * np.cos(ph), k1 * np.sqrt(1 - u ** 2) * np.sin(ph), k1 * u]
        k2 = np.linalg.norm(kv + np.c_[draw(n_mc), draw(n_mc), draw(n_mc)], axis=1)
        same.append(np.mean((k2 >= lo) & (k2 < hi)))
        neigh.append(np.mean(((k2 >= hi) & (k2 < hi + dk)) | ((k2 >= lo - dk) & (k2 < lo))))
    return np.array(same), np.array(neigh)


# --------------------------------------------------------------------------- set-up
def box_covariance(boxsize=4000.0, nrandoms=400000, alpha=1.0, P0_flat=1e4, kmax=0.2, dk=0.01,
                   distance=0.0, ells=(0, 2), n_sub=3000, n_near=100000, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-boxsize / 2, boxsize / 2, size=(nrandoms, 3))
    pos[:, 0] += distance
    nbar = alpha * nrandoms / boxsize ** 3
    randoms = {'POSITION': pos, 'WEIGHT': np.ones(nrandoms), 'NZ': np.full(nrandoms, nbar)}
    tr = Tracer('A', randoms, alpha)
    k_edges = np.arange(0.0, kmax + dk / 2, dk)
    kk = np.linspace(0.0, 1.0, 50)
    model = PowerSpectrumModel()
    model.add(('A', 'A'), {0: (kk, np.full_like(kk, P0_flat))})
    cov = GaussianCovariance([tr], k_edges, ells=ells, L_max=2, ds=2.0, ds_pair=10.0, shot_noise=True,
                             n_sub=n_sub, n_near=n_near, s_split=80.0, seed=seed)
    cov.compute_windows([('A', 'A')]).set_model(model)
    blocks = {(l1, l2): cov.block(('A', 'A'), ('A', 'A'), l1, l2) for l1 in ells for l2 in ells}
    Ptot = P0_flat + (1 + alpha) / nbar
    PL = {0: np.full(len(k_edges) - 1, Ptot)}
    analytic = {(l1, l2): analytic_box(k_edges, boxsize ** 3, PL, l1, l2) for l1 in ells for l2 in ells}
    return k_edges, blocks, analytic


# --------------------------------------------------------------------------- tests
def test_notebook_coefficients():
    """The general Legendre formula reproduces the explicit coefficients written in the notebook."""
    P0, P2, P4 = 1.0, 0.37, 0.11
    nb = notebook_analytic(P0, P2, P4)
    k_edges = np.array([0.1, 0.11])
    V = 1.0
    Vi = (k_edges[1:] ** 3 - k_edges[:-1] ** 3) / 3 / (2 * np.pi ** 2)
    PL = {0: np.array([P0]), 2: np.array([P2]), 4: np.array([P4])}
    for (l1, l2), val in nb.items():
        gen = analytic_box(k_edges, V, PL, l1, l2)[0] * (V * Vi[0]) / 2
        assert np.isclose(gen, val, rtol=1e-10), (l1, l2, gen, val)


@pytest.fixture(scope="module")
def box0():
    return box_covariance(distance=0.0)


@pytest.fixture(scope="module")
def box_far():
    return box_covariance(distance=2e5, seed=3)


def test_box_monopole_leakage(box0):
    k_edges, C, A = box0
    same, neigh = leakage_fractions(4000.0, k_edges)
    ratio = np.diag(C[(0, 0)]) / A[(0, 0)]
    sel = slice(2, -1)      # skip the k -> 0 bin and the last bin (leaks out of the k range)
    # diagonal = retained fraction of the periodic-box value
    assert np.allclose(ratio[sel], same[sel], rtol=0.03), np.c_[ratio, same]
    # the two neighbouring bins carry the leaked variance: C(i,j) V_j / (V_i A_i) = fraction(i -> j)
    Vi = k_edges[1:] ** 3 - k_edges[:-1] ** 3
    M = C[(0, 0)] * Vi[None, :] / Vi[:, None] / A[(0, 0)][:, None]
    nb = np.diag(M, 1)[1:] + np.diag(M, -1)[:-1]
    assert np.allclose(nb[1:-1], neigh[2:-2], atol=0.025), np.c_[nb[1:-1], neigh[2:-2]]
    assert abs(np.mean(nb[1:-1] - neigh[2:-2])) < 0.012
    # and the periodic-box value is NOT reproduced by the diagonal alone (~10 % low for L dk = 40)
    assert np.all(ratio[sel] < 0.95)


def test_box_far_multipoles(box_far):
    """Distant box (fixed line of sight): ell = 2 has the same leakage factor as ell = 0, and C_02 = 0."""
    k_edges, C, A = box_far
    r00 = np.diag(C[(0, 0)]) / A[(0, 0)]
    r22 = np.diag(C[(2, 2)]) / A[(2, 2)]
    sel = slice(3, -1)
    assert np.allclose(r22[sel], r00[sel], rtol=0.03), np.c_[r00, r22]
    corr02 = np.diag(C[(0, 2)]) / np.sqrt(np.diag(C[(0, 0)]) * np.diag(C[(2, 2)]))
    assert np.all(np.abs(corr02[sel]) < 0.03), corr02


def test_box_origin_varying_los(box0, box_far):
    """Box centred on the observer: the varying line of sight adds leakage for ell = 2 and a small
    (negative) C_02, both absent for the distant box. The monopole is unchanged."""
    k_edges, C0, A0 = box0
    _, Cf, Af = box_far
    sel = slice(3, -1)
    r00_0 = np.diag(C0[(0, 0)]) / A0[(0, 0)]
    r22_0 = np.diag(C0[(2, 2)]) / A0[(2, 2)]
    r22_f = np.diag(Cf[(2, 2)]) / Af[(2, 2)]
    assert np.all(r22_0[sel] < r22_f[sel])                       # extra leakage
    assert np.allclose(r22_0[sel], r22_f[sel], rtol=0.08)         # but a few per cent only
    corr02 = np.diag(C0[(0, 2)]) / np.sqrt(np.diag(C0[(0, 0)]) * np.diag(C0[(2, 2)]))
    assert np.all(np.abs(corr02[sel]) < 0.05)


def test_shot_noise_factor(box0):
    """With alpha = 1 the FKP shot noise is (1 + alpha)/nbar, not 1/nbar: comparing with the
    notebook's analytic expression (1/nbar) would give a ratio ~ 2.7 instead of ~0.9."""
    k_edges, C, A = box0
    nbar = 1.0 * 400000 / 4000.0 ** 3
    P_nb = 1e4 + 1.0 / nbar
    A_nb = analytic_box(k_edges, 4000.0 ** 3, {0: np.full(len(k_edges) - 1, P_nb)}, 0, 0)
    ratio_nb = np.diag(C[(0, 0)]) / A_nb
    assert np.all(ratio_nb[3:-1] > 2.0)
