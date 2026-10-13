r"""Angular algebra of the bispectrum covariance (thecov.bispectrum_algebra) against direct quadrature.

The arbiter of every sign and normalisation is brute-force angular integration:

* Sugiyama basis: H_202 = +1/sqrt(5), S_202 = L_2(a . n), S_220 = L_2(a . b), orthogonality <conj(S) S'> = delta / (N H^2).
* Bipolar vertex coefficients: conj(Y_{L3 g})(p3^) P_L3(|p1 + p2|) is reconstructed at random directions from c^{L3}_{ab}
  (exactly for a polynomial P, to truncation for a BAO-like P); the L3 = 0 reduction to Legendre coefficients.
* Coupling numbers T_n: with FIVE INDEPENDENT directions (r1^, y1^, r2^, y2^, x^) the series sum_n T_n Phi_n with the
  radial factors at fixed magnitudes must equal (-i)^{l1+l2} conj(S~_ell)(r1^, r2^, x^) times the direct quadrature of
  << e^{i p1.r1 + i p2.r2} S~_ell' P(p1; y1) P(p2; y2) P(p1 + p2; x) >> over the two momentum directions.  With a vertex
  spectrum of finite bipolar rank the series is finite and the comparison is exact.
* Box identity: sum_n T_n Q_n^box reproduces << conj(S~_ell) S~_ell' P P P >> with a global line of sight, i.e. the
  class-1 term of Sugiyama et al. (2020) eq. 49, with Q_n^box = V sqrt(N/4pi) H delta_{lam1 0} delta_{lam2 0}.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import eval_legendre

from thecov.bispectrum_algebra import (H, N, sugiyama_S, sugiyama_S_tilde, cg_tensor, bipolar_coefficients,
                                       legendre_coefficients_opening, BispectrumCoupling, box_window_value, phase,
                                       _ylm_all)
from thecov.wigner import FOUR_PI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'desi_validation' / 'bispectrum_dev'))
from angular_check import unit, direct_V, direct_box_average, series_F  # noqa: E402

K0 = 0.1
PL_POLY = {0: lambda k: 1.0 + (k / K0) ** 2, 2: lambda k: 0.6 * (k / K0) ** 2, 4: lambda k: 0.2 * (k / K0) ** 4}
PL_ISO = {0: PL_POLY[0]}
P_BAO = lambda k: 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6) * (1 + 0.1 * np.sin(k * 105))  # noqa: E731
P1, P2, R1, R2 = 0.08, 0.12, 30.0, 50.0


# --------------------------------------------------------------------------------------------- basis
def test_H_and_S():
    rng = np.random.default_rng(1)
    a, b, n = unit(5, rng), unit(5, rng), unit(5, rng)
    assert np.isclose(H(2, 0, 2), 1 / np.sqrt(5))
    assert np.allclose(sugiyama_S((0, 0, 0), a, b, n), 1.0)
    assert np.allclose(sugiyama_S((2, 0, 2), a, b, n), eval_legendre(2, np.sum(a * n, 1)), atol=1e-12)
    assert np.allclose(sugiyama_S((2, 2, 0), a, b, n), eval_legendre(2, np.sum(a * b, 1)), atol=1e-12)


def test_S_orthogonality():
    rng = np.random.default_rng(2)
    A, B = unit(200000, rng), unit(200000, rng)
    n = np.broadcast_to(unit(1, rng)[0], A.shape)
    ells = [(0, 0, 0), (2, 0, 2), (2, 2, 0)]
    for e1 in ells:
        for e2 in ells:
            v = np.mean(np.conj(sugiyama_S(e1, A, B, n)) * sugiyama_S(e2, A, B, n)).real
            expect = 1 / (N(*e1) * H(*e1) ** 2) if e1 == e2 else 0.0
            assert abs(v - expect) < 0.01


# --------------------------------------------------------------------------------------------- bipolar vertex
def _reconstruct(L3, g, P, p1, p2, d1, d2, a_max):
    Y1, Y2, LM = _ylm_all(a_max + L3, d1), _ylm_all(a_max + L3, d2), a_max + L3
    rhs = 0.0
    for a in range(a_max + 1):
        for b in range(abs(a - L3), a + L3 + 1):
            if (a + b + L3) % 2:
                continue
            c = bipolar_coefficients(L3, a, b, p1, p2, P)
            C = cg_tensor(a, b, L3)
            for al in range(-a, a + 1):
                be = g - al
                if abs(be) <= b and C[al + a, be + b, g + L3] != 0.0:
                    rhs = rhs + c * C[al + a, be + b, g + L3] * np.conj(Y1[a, al + LM]) * np.conj(Y2[b, be + LM])
    return rhs


@pytest.mark.parametrize('L3', [0, 2, 4])
def test_bipolar_reconstruction_polynomial(L3):
    """Y_{L3}(p3^) p3^{L3} (1 + p3^2/k0^2) has bipolar ranks <= L3 + 2: the expansion is finite and exact."""
    rng = np.random.default_rng(3)
    d1, d2 = unit(6, rng), unit(6, rng)
    P = lambda k: (k / K0) ** L3 * (1.0 + (k / K0) ** 2)  # noqa: E731
    p3v = P1 * d1 + P2 * d2
    p3 = np.linalg.norm(p3v, axis=1)
    for g in range(-L3, L3 + 1):
        lhs = np.conj(_ylm_all(L3, p3v / p3[:, None])[L3, g + L3]) * P(p3)
        rhs = _reconstruct(L3, g, P, P1, P2, d1, d2, a_max=L3 + 2)
        assert np.max(np.abs(lhs - rhs)) < 1e-10 * np.max(np.abs(lhs))


def test_bipolar_reconstruction_bao():
    rng = np.random.default_rng(4)
    d1, d2 = unit(6, rng), unit(6, rng)
    p1, p2 = 0.11, 0.17
    p3v = p1 * d1 + p2 * d2
    p3 = np.linalg.norm(p3v, axis=1)
    lhs = P_BAO(p3) / np.sqrt(FOUR_PI)
    rhs = _reconstruct(0, 0, P_BAO, p1, p2, d1, d2, a_max=30)
    assert np.max(np.abs(lhs - rhs)) < 1e-4 * np.max(np.abs(lhs))


def test_bipolar_L3_zero_is_legendre():
    p1, p2 = 0.11, 0.17
    pa = legendre_coefficients_opening(P_BAO, p1, p2, 6)
    for a in range(7):
        c = float(bipolar_coefficients(0, a, a, p1, p2, P_BAO))
        assert np.isclose(c, (-1) ** a * pa[a] * np.sqrt(FOUR_PI / (2 * a + 1)), rtol=1e-10)


# --------------------------------------------------------------------------------------------- coupling numbers
COUP = BispectrumCoupling()


def _dirs():
    rng = np.random.default_rng(3)
    return tuple(unit(1, rng)[0] for _ in range(5))


def _check_five_directions(ell, ellp, PL, a_max):
    r1h, y1, r2h, y2, x = _dirs()
    V = direct_V(ellp, PL, P1, P2, R1 * r1h, R2 * r2h, y1, y2, x)
    Fd = (-1j) ** (ell[0] + ell[1]) * np.conj(sugiyama_S_tilde(ell, r1h, r2h, x)) * V
    Fs, nterms = series_F(ell, ellp, PL, P1, P2, R1, R2, (r1h, y1, r2h, y2, x), lam_w=40, a_max=a_max, coup=COUP)
    assert nterms > 0
    assert abs(Fs.imag) < 1e-12 * abs(Fd) + 1e-15
    assert abs(Fs - Fd) < 1e-8 * abs(Fd), (ell, ellp, Fs, Fd)


@pytest.mark.parametrize('ell,ellp', [((0, 0, 0), (0, 0, 0)), ((2, 0, 2), (0, 0, 0)), ((0, 0, 0), (2, 0, 2)),
                                      ((2, 0, 2), (2, 0, 2)), ((2, 2, 0), (2, 0, 2))])
def test_coupling_isotropic(ell, ellp):
    _check_five_directions(ell, ellp, PL_ISO, a_max=2)


@pytest.mark.parametrize('ell,ellp', [((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (2, 0, 2))])
def test_coupling_anisotropic(ell, ellp):
    _check_five_directions(ell, ellp, PL_POLY, a_max=4)


@pytest.mark.slow
@pytest.mark.parametrize('ell,ellp', [((2, 0, 2), (0, 0, 0)), ((2, 0, 2), (2, 0, 2)), ((2, 2, 0), (2, 0, 2))])
def test_coupling_anisotropic_slow(ell, ellp):
    _check_five_directions(ell, ellp, PL_POLY, a_max=4)


def _box_series(ell, ellp, PL, p1, p2, a_max=4):
    tot = 0.0
    for (L1, L2, L3, a, b, Lam, Lamp, n) in BispectrumCoupling.index_set(ell, ellp, tuple(sorted(PL)), 0, a_max):
        h = box_window_value(n, 1.0)
        if h == 0.0:
            continue
        T = COUP.t(ell, ellp, L1, L2, L3, a, b, Lam, Lamp, n)
        if abs(T) < 1e-14:
            continue
        c = float(bipolar_coefficients(L3, a, b, p1, p2, PL[L3]))
        tot += FOUR_PI ** 3 / ((2 * L1 + 1) * (2 * L2 + 1) * (2 * L3 + 1)) * T * h * c * PL[L1](p1) * PL[L2](p2) \
            * phase(ell, Lam, Lamp)
    return tot / FOUR_PI ** 2


@pytest.mark.parametrize('ell,ellp', [((0, 0, 0), (0, 0, 0)), ((2, 0, 2), (0, 0, 0)), ((2, 0, 2), (2, 0, 2)),
                                      ((2, 2, 0), (2, 2, 0)), ((2, 2, 0), (2, 0, 2))])
def test_box_identity(ell, ellp):
    """<< conj(S~) S~' P P P >> (global line of sight) = (4pi)^-2 sum_n T_n Q_n^box (4pi)^3 c P P / prod(2L+1)."""
    rng = np.random.default_rng(5)
    nhat = unit(1, rng)[0]
    d = direct_box_average(ell, ellp, PL_POLY, P1, P2, nhat)
    s = _box_series(ell, ellp, PL_POLY, P1, P2)
    assert abs(d.imag) < 1e-12 * abs(d)
    assert abs(s - d.real) < 1e-8 * abs(d)
