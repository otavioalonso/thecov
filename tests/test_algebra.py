import math

import numpy as np
import pytest
from scipy.special import sph_harm_y, eval_legendre

from pkcov.wigner import wigner_3j, w3j_tensor, gaunt_tensor, tri, tri_multi, CouplingCoefficients, FOUR_PI
from pkcov.harmonics import sph_harm_table, tripolar_direct, tripolar_frame, unit_vectors
from pkcov.covariance import GaussianCovariance


def test_3j_known_values():
    assert np.isclose(wigner_3j(1, 1, 0, 0, 0, 0), -1 / math.sqrt(3))
    assert np.isclose(wigner_3j(2, 2, 0, 0, 0, 0), 1 / math.sqrt(5))
    assert np.isclose(wigner_3j(1, 1, 2, 0, 0, 0), math.sqrt(2 / 15))
    assert np.isclose(wigner_3j(2, 2, 2, 0, 0, 0), -math.sqrt(2 / 35))


def test_3j_vs_sympy():
    sympy = pytest.importorskip("sympy")
    from sympy.physics.wigner import wigner_3j as w3
    rng = np.random.default_rng(0)
    for _ in range(200):
        j1, j2 = (int(x) for x in rng.integers(0, 9, 2))
        j3 = int(rng.integers(abs(j1 - j2), j1 + j2 + 1))
        m1, m2 = int(rng.integers(-j1, j1 + 1)), int(rng.integers(-j2, j2 + 1))
        m3 = -m1 - m2
        if abs(m3) > j3:
            continue
        assert np.isclose(wigner_3j(j1, j2, j3, m1, m2, m3), float(w3(j1, j2, j3, m1, m2, m3)), atol=1e-12)


def test_3j_orthogonality():
    for j1, j2 in [(2, 3), (4, 4), (6, 8)]:
        for j3 in range(abs(j1 - j2), j1 + j2 + 1):
            for j3p in range(abs(j1 - j2), j1 + j2 + 1):
                s = sum((2 * j3 + 1) * wigner_3j(j1, j2, j3, m1, m2, -m1 - m2) * wigner_3j(j1, j2, j3p, m1, m2, -m1 - m2)
                        for m1 in range(-j1, j1 + 1) for m2 in range(-j2, j2 + 1) if abs(m1 + m2) <= min(j3, j3p))
                assert np.isclose(s, (2 * j3 + 1) if j3 == j3p else 0.0, atol=1e-12)


def test_sph_harm_table_vs_scipy():
    rng = np.random.default_rng(1)
    ct = rng.uniform(-1, 1, 50)
    ph = rng.uniform(0, 2 * np.pi, 50)
    lmax = 12
    Y = sph_harm_table(ct, ph, lmax)
    theta = np.arccos(ct)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ref = sph_harm_y(l, m, theta, ph)
            assert np.allclose(Y[l, m + lmax], ref, atol=1e-12)


def _quad_sphere(n=60):
    x, w = np.polynomial.legendre.leggauss(n)
    phi = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    ct = np.repeat(x, len(phi))
    ph = np.tile(phi, n)
    wt = np.repeat(w, len(phi)) * (2 * np.pi / len(phi))
    return ct, ph, wt


def test_gaunt_vs_numerical_integration():
    ct, ph, wt = _quad_sphere()
    lmax = 6
    Y = sph_harm_table(ct, ph, lmax)
    for (l1, l2, l3) in [(2, 2, 4), (4, 2, 6), (2, 4, 2), (0, 4, 4), (6, 6, 4)]:
        G = gaunt_tensor((l1, l2, l3))
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m3 = -m1 - m2
                if abs(m3) > l3:
                    continue
                num = np.sum(wt * Y[l1, m1 + lmax] * Y[l2, m2 + lmax] * Y[l3, m3 + lmax])
                assert np.isclose(num.real, G[m1 + l1, m2 + l2, m3 + l3], atol=1e-10)
                assert abs(num.imag) < 1e-10


def test_gaunt4_merge_rule():
    """prod Y = sum_{Lam mu} G^{(4)} conj(Y_{Lam mu}) at random points."""
    rng = np.random.default_rng(2)
    ct = rng.uniform(-1, 1, 20)
    ph = rng.uniform(0, 2 * np.pi, 20)
    l1, l2, l3 = 2, 4, 2
    lmax = l1 + l2 + l3
    Y = sph_harm_table(ct, ph, lmax)
    for (m1, m2, m3) in [(1, -3, 0), (0, 0, 0), (2, 2, -1), (-2, 4, 2)]:
        lhs = Y[l1, m1 + lmax] * Y[l2, m2 + lmax] * Y[l3, m3 + lmax]
        rhs = np.zeros_like(lhs)
        for Lam in tri_multi((l1, l2, l3)):
            G = gaunt_tensor((l1, l2, l3, Lam))
            for mu in range(-Lam, Lam + 1):
                rhs += G[m1 + l1, m2 + l2, m3 + l3, mu + Lam] * np.conj(Y[Lam, mu + lmax])
        assert np.allclose(lhs, rhs, atol=1e-12)


def test_tripolar_frame_equals_direct():
    rng = np.random.default_rng(3)
    x = unit_vectors(rng.normal(size=(200, 3)))
    xp = unit_vectors(rng.normal(size=(200, 3)))
    s = unit_vectors(rng.normal(size=(200, 3)))
    for (L1, L2, L) in [(0, 0, 0), (2, 0, 2), (2, 2, 0), (2, 2, 4), (4, 2, 2), (4, 4, 8), (6, 2, 4), (8, 4, 12)]:
        d = tripolar_direct(L1, L2, L, x, xp, s)
        f = tripolar_frame(L1, L2, L, x, xp, s)
        assert np.max(np.abs(d.imag)) < 1e-12
        assert np.allclose(d.real, f, atol=1e-12)


def test_coupling_3j_proportionality():
    cc = CouplingCoefficients()
    worst = 0.0
    for term in (1, 2):
        for key in GaussianCovariance.index_tuples(term, 2, 4, 2, 4):
            lam, lamp, Lam1, Lam2, Lam = key
            worst = max(worst, cc.check_3j_proportional(term, 2, 4, 2, 4, lam, lamp, Lam1, Lam2, Lam))
    assert worst < 1e-12


def _legendre4(l1, l2, L1, L2):
    x, w = np.polynomial.legendre.leggauss(40)
    return 0.5 * np.sum(w * eval_legendre(l1, x) * eval_legendre(l2, x) * eval_legendre(L1, x) * eval_legendre(L2, x))


@pytest.mark.parametrize("term", [1, 2])
def test_box_limit_identity(term):
    """Distant-observer periodic box: for each (l1, l2, L1, L2),

       (4 pi)^{5/2} / ((2L1+1)(2L2+1)) sum_{lam, Lam1} sqrt(2 Lam1 + 1) t(term; lam, lam, Lam1, Lam1, 0)
         = (2 l1 + 1)(2 l2 + 1) int dmu/2 L_l1 L_l2 L_L1 L_L2 .
    """
    cc = CouplingCoefficients()
    for l1 in (0, 2, 4):
        for l2 in (0, 2, 4):
            for L1 in (0, 2, 4):
                for L2 in (0, 2, 4):
                    lhs = 0.0
                    for (lam, lamp, Lam1, Lam2, Lam) in GaussianCovariance.index_tuples(term, l1, l2, L1, L2):
                        if Lam != 0:
                            continue
                        assert lam == lamp and Lam1 == Lam2
                        lhs += math.sqrt(2 * Lam1 + 1) * cc.t(term, l1, l2, L1, L2, lam, lam, Lam1, Lam1, 0)
                    lhs *= FOUR_PI ** 2.5 / ((2 * L1 + 1) * (2 * L2 + 1))
                    rhs = (2 * l1 + 1) * (2 * l2 + 1) * _legendre4(l1, l2, L1, L2)
                    assert np.isclose(lhs, rhs, atol=1e-10), (term, l1, l2, L1, L2, lhs, rhs)
