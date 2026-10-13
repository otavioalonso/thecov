r"""Angular algebra of the bispectrum covariance (desi_validation/report/bispectrum_covariance_guide.typ).

Conventions
-----------
* Orthonormal complex spherical harmonics Y_{lm} with the Condon-Shortley phase, conj(Y_{lm}) = (-1)^m Y_{l,-m}.
* Gaunt integrals G^{m_1..m_n}_{l_1..l_n} = int dOmega prod_a Y_{l_a m_a} (wigner.gaunt_tensor, indexed by m + l).
* Clebsch-Gordan <a alpha b beta | L gamma> = (-1)^{a-b+gamma} sqrt(2L+1) (a b L; alpha beta -gamma).
* Sugiyama basis (Sugiyama, Saito, Beutler & Seo 2019):
      H_{l1 l2 L} = (l1 l2 L; 0 0 0),   N_{l1 l2 L} = (2l1+1)(2l2+1)(2L+1),
      S~_{l1 l2 L}(a, b, n) = sum_{m1 m2 M} (l1 l2 L; m1 m2 M) Y_{l1 m1}(a) Y_{l2 m2}(b) Y_{L M}(n),
      S_{l1 l2 L} = (4 pi)^{3/2} / (H sqrt(N)) S~    (so that S_000 = 1 and S_202 = L_2(a . n)).

The class-1 Gaussian term is built from three kinds of objects, which this module provides the first two of:

* the bipolar vertex coefficients c^{L3}_{ab}(p1, p2) of the third side of the triangle,
      conj(Y_{L3 g})(p3^) P_{L3}(|p1 + p2|) = sum_{ab} c^{L3}_{ab} sum_{alpha beta} <a alpha b beta | L3 g>
                                              conj(Y_{a alpha})(p1^) conj(Y_{b beta})(p2^);
* the coupling numbers T_n (pure 3j algebra), the projection of the angular kernel on the orthonormal invariant basis
      Phi_n(r1^, y1^, r2^, y2^, x^) = sum (K1 K2 Lam3; k1 k2 m3) [Y_lam1(r1^) x Y_L1(y1^)]_{K1 k1}
                                       [Y_lam2(r2^) x Y_L2(y2^)]_{K2 k2} Y_{Lam3 m3}(x^),
  n = (lam1, L1, K1, lam2, L2, K2, Lam3).

The radial kernels (third kind) live in bispectrum.py.  Every formula here is checked numerically in
tests/test_bispectrum_algebra.py against direct angular quadrature, which is the arbiter of the signs.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import spherical_jn

from .harmonics import sph_harm_table, angles
from .wigner import wigner_3j, w3j_tensor, gaunt_tensor, tri, FOUR_PI

# ----------------------------------------------------------------------------------------------- basis
def H(l1: int, l2: int, L: int) -> float:
    """H_{l1 l2 L} = (l1 l2 L; 0 0 0).  H_{l 0 l} = (-1)^l / sqrt(2l+1), so H_202 = +1/sqrt(5)."""
    return wigner_3j(l1, l2, L, 0, 0, 0)


def N(l1: int, l2: int, L: int) -> int:
    return (2 * l1 + 1) * (2 * l2 + 1) * (2 * L + 1)


def _Y(l: int, m: int, v: np.ndarray) -> np.ndarray:
    """Y_{lm} at unit vectors v (..., 3)."""
    ct, ph = angles(v)
    return sph_harm_table(ct, ph, l)[l, m + l]


def _ylm_all(lmax: int, v: np.ndarray) -> np.ndarray:
    """Y[l, m + lmax] at unit vectors v (..., 3): shape (lmax+1, 2 lmax+1, ...)."""
    ct, ph = angles(v)
    return sph_harm_table(ct, ph, lmax)


def sugiyama_S_tilde(ell, a, b, n) -> np.ndarray:
    """S~_{l1 l2 L}(a^, b^, n^) for unit vectors a, b, n (broadcastable (..., 3))."""
    l1, l2, L = ell
    lmax = max(ell)
    Ya, Yb, Yn = _ylm_all(lmax, a), _ylm_all(lmax, b), _ylm_all(lmax, n)
    W = w3j_tensor(l1, l2, L)
    out = 0.0
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            M = -m1 - m2
            if abs(M) > L or W[m1 + l1, m2 + l2, M + L] == 0.0:
                continue
            out = out + W[m1 + l1, m2 + l2, M + L] * Ya[l1, m1 + lmax] * Yb[l2, m2 + lmax] * Yn[L, M + lmax]
    return out


def sugiyama_S(ell, a, b, n) -> np.ndarray:
    """The normalised S_{l1 l2 L} of Sugiyama et al.: S_000 = 1, S_202 = L_2(a . n)."""
    return FOUR_PI ** 1.5 / (H(*ell) * math.sqrt(N(*ell))) * sugiyama_S_tilde(ell, a, b, n)


# ----------------------------------------------------------------------------------------------- tensors
@lru_cache(maxsize=None)
def cg_tensor(a: int, b: int, L: int) -> np.ndarray:
    """C[alpha + a, beta + b, gamma + L] = <a alpha b beta | L gamma>."""
    W = w3j_tensor(a, b, L)                                     # (a b L; alpha beta m3), m3 = -gamma
    gam = np.arange(-L, L + 1)
    C = W[:, :, ::-1] * ((-1.0) ** (a - b + gam))[None, None, :] * math.sqrt(2 * L + 1)
    return C


def _negate(T: np.ndarray, axes, ls) -> np.ndarray:
    """T'[..., m, ...] = (-1)^m T[..., -m, ...] along the given axes (ls: the l of each axis)."""
    out = T
    for ax, l in zip(axes, ls):
        out = np.flip(out, axis=ax)
        ph = (-1.0) ** np.arange(-l, l + 1)
        shape = [1] * out.ndim
        shape[ax] = 2 * l + 1
        out = out * ph.reshape(shape)
    return out


@lru_cache(maxsize=None)
def leg_tensor(ellp: int, L: int, a: int, Lam: int) -> np.ndarray:
    """g[m' + ell', M + L, alpha + a, nu + Lam] = int dOmega Y_{ell' m'} conj(Y_{L M}) conj(Y_{a alpha}) conj(Y_{Lam nu}),
    the re-expansion of the three harmonics a primed momentum carries (estimator, leg spectrum, vertex) in Y_{Lam nu}."""
    G = gaunt_tensor((ellp, L, a, Lam))
    return _negate(G, (1, 2, 3), (L, a, Lam))


@lru_cache(maxsize=None)
def leg_projection_tensor(lam: int, ell: int, Lam: int) -> np.ndarray:
    """R[rho + lam, m + ell, nu + Lam] = int dOmega conj(Y_{lam rho}) conj(Y_{ell m}) Y_{Lam nu}  (at r^)."""
    G = gaunt_tensor((lam, ell, Lam))
    return _negate(G, (2,), (Lam,))


@lru_cache(maxsize=None)
def anchor_tensor(Lam3: int, L: int, Lp: int, L3: int) -> np.ndarray:
    """X[mu3 + Lam3, M + L, M' + L', gamma + L3] = int dOmega conj(Y_{Lam3 mu3}) conj(Y_{L M}) Y_{L' M'} Y_{L3 gamma} (at x^)."""
    G = gaunt_tensor((Lam3, L, Lp, L3))
    return _negate(G, (0, 1), (Lam3, L))


# class 2 (pairing (1 1')(2 3')(3 2') in the variables p1, p3 = p1 + p2; guide sec. 'Class 2')
@lru_cache(maxsize=None)
def vertex2_tensor(ellp2: int, L2: int, lpp: int) -> np.ndarray:
    """g2[m'2 + ell'2, M2 + L2, m'' + l''] = int dOmega Y_{ell'2 m'2} conj(Y_{L2 M2}) conj(Y_{l'' m''}): the estimator
    harmonic and the anchor-spectrum harmonic of p2^ merged into Y_{l'' m''}(p2^)."""
    return _negate(gaunt_tensor((ellp2, L2, lpp)), (1, 2), (L2, lpp))


@lru_cache(maxsize=None)
def leg3_tensor(L3: int, b: int, Lamp: int) -> np.ndarray:
    """g3[M3 + L3, beta + b, nu + Lam'] = int dOmega conj(Y_{L3 M3}) conj(Y_{b beta}) conj(Y_{Lam' nu}): the harmonics of
    p3^ (leg-2 spectrum at y2^, vertex) re-expanded in Y_{Lam' nu}(p3^)."""
    return _negate(gaunt_tensor((L3, b, Lamp)), (0, 1, 2), (L3, b, Lamp))


@lru_cache(maxsize=None)
def y2_tensor(Ly: int, L3: int, Lp: int) -> np.ndarray:
    """Py[mu_y + Ly, M3 + L3, M' + L'] = int dOmega conj(Y_{Ly mu_y}) Y_{L3 M3} Y_{L' M'} (at y2^: leg-2 spectrum and the
    primed line of sight)."""
    return _negate(gaunt_tensor((Ly, L3, Lp)), (0,), (Ly,))


@lru_cache(maxsize=None)
def anchor2_tensor(Lam3: int, L: int, L2: int) -> np.ndarray:
    """X2[mu3 + Lam3, M + L, M2 + L2] = int dOmega conj(Y_{Lam3 mu3}) conj(Y_{L M}) Y_{L2 M2} (at x^: unprimed line of
    sight and the anchor spectrum P(p2; x^))."""
    return _negate(gaunt_tensor((Lam3, L, L2)), (0, 1), (Lam3, L))


# ----------------------------------------------------------------------------------------------- bipolar vertex
def _p3_nodes(p1, p2, n_quad):
    """Gauss-Legendre nodes in p3 on [|p1 - p2|, p1 + p2] (the opening angle through p3^2 = p1^2 + p2^2 + 2 p1 p2 mu;
    integrating in p3 regularises the sqrt(1 + mu) behaviour at p1 = p2) with dmu = p3 dp3 / (p1 p2)."""
    x, w = np.polynomial.legendre.leggauss(n_quad)
    lo, hi = abs(p1 - p2), p1 + p2
    p3 = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    wmu = 0.5 * (hi - lo) * w * p3 / (p1 * p2)
    mu = np.clip((p3 ** 2 - p1 ** 2 - p2 ** 2) / (2 * p1 * p2), -1.0, 1.0)
    return p3, mu, wmu


def legendre_coefficients_opening(P, p1: float, p2: float, a_max: int, n_quad: int = 96) -> np.ndarray:
    """p_a(p1, p2), a = 0..a_max, with P(|p1 + p2|) = sum_a p_a L_a(p1^ . p2^)."""
    from scipy.special import eval_legendre
    p3, mu, wmu = _p3_nodes(p1, p2, n_quad)
    f = P(p3)
    return np.array([(2 * a + 1) / 2.0 * np.sum(wmu * f * eval_legendre(a, mu)) for a in range(a_max + 1)])


def bipolar_coefficients(L3: int, a: int, b: int, p1, p2, P_L3, n_quad: int = 96) -> np.ndarray:
    """c^{L3}_{ab}(p1, p2) for arrays p1, p2 (broadcast together), P_L3 a callable of the magnitude.

        c = 8 pi^2 sqrt((2a+1)/4pi) / (2 L3 + 1) int dmu P_L3(p3) sum_g <a 0 b g | L3 g> Y_{b g}(theta_2, 0) Y_{L3 g}(theta_3, 0),
    with p1^ = z^, p2^ in the xz-plane at cos theta_2 = mu, and p3^ at cos theta_3 = (p1 + p2 mu) / p3.  Zero unless
    a + b + L3 is even and |a - b| <= L3 <= a + b.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    shape = np.broadcast(p1, p2).shape
    if (a + b + L3) % 2 or L3 < abs(a - b) or L3 > a + b:
        return np.zeros(shape)
    p1b, p2b = np.broadcast_to(p1, shape).ravel(), np.broadcast_to(p2, shape).ravel()
    out = np.zeros(len(p1b))
    C = cg_tensor(a, b, L3)                                      # [alpha, beta, gamma]
    lmax = max(b, L3)
    x, w = np.polynomial.legendre.leggauss(n_quad)
    for n, (q1, q2) in enumerate(zip(p1b, p2b)):
        lo, hi = abs(q1 - q2), q1 + q2
        p3 = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        wmu = 0.5 * (hi - lo) * w * p3 / (q1 * q2)
        mu = np.clip((p3 ** 2 - q1 ** 2 - q2 ** 2) / (2 * q1 * q2), -1.0, 1.0)
        ct3 = np.clip((q1 + q2 * mu) / p3, -1.0, 1.0)
        Y2 = sph_harm_table(mu, np.zeros_like(mu), lmax).real              # [l, m + lmax, node]
        Y3 = sph_harm_table(ct3, np.zeros_like(mu), lmax).real
        f = np.asarray(P_L3(p3), dtype=float)
        acc = 0.0
        for g in range(-min(b, L3), min(b, L3) + 1):
            acc = acc + C[a, g + b, g + L3] * np.sum(wmu * f * Y2[b, g + lmax] * Y3[L3, g + lmax])
        out[n] = acc
    out *= 8 * math.pi ** 2 * math.sqrt((2 * a + 1) / FOUR_PI) / (2 * L3 + 1)
    return out.reshape(shape)


def bipolar_harmonic(a: int, b: int, L: int, g: int, u, v) -> np.ndarray:
    """[Y_a(u^) x Y_b(v^)]_{L g} = sum <a alpha b beta | L g> Y_{a alpha}(u^) Y_{b beta}(v^)."""
    C = cg_tensor(a, b, L)
    lmax = max(a, b)
    Yu, Yv = _ylm_all(lmax, u), _ylm_all(lmax, v)
    out = 0.0
    for al in range(-a, a + 1):
        be = g - al
        if abs(be) <= b and C[al + a, be + b, g + L] != 0.0:
            out = out + C[al + a, be + b, g + L] * Yu[a, al + lmax] * Yv[b, be + lmax]
    return out


def invariant_basis(n, r1, y1, r2, y2, x) -> np.ndarray:
    """Phi_n at unit vectors (broadcastable), n = (lam1, L1, K1, lam2, L2, K2, Lam3)."""
    lam1, L1, K1, lam2, L2, K2, Lam3 = n
    W = w3j_tensor(K1, K2, Lam3)
    Yx = _ylm_all(Lam3, x)
    out = 0.0
    for k1 in range(-K1, K1 + 1):
        B1 = bipolar_harmonic(lam1, L1, K1, k1, r1, y1)
        for k2 in range(-K2, K2 + 1):
            m3 = -k1 - k2
            if abs(m3) > Lam3 or W[k1 + K1, k2 + K2, m3 + Lam3] == 0.0:
                continue
            out = out + W[k1 + K1, k2 + K2, m3 + Lam3] * B1 * bipolar_harmonic(lam2, L2, K2, k2, r2, y2) * Yx[Lam3, m3 + Lam3]
    return out


# ----------------------------------------------------------------------------------------------- coupling numbers
def all_tri(a: int, b: int) -> list:
    """All integers in the triangle {|a - b|, ..., a + b} (no parity restriction)."""
    return list(range(abs(a - b), a + b + 1))


class BispectrumCoupling:
    """T_n for the class-1 term: the projection of conj(S~_ell)(r1^, r2^, x^) A on Phi_n, where A is the angular
    polynomial of the primed estimator after its momenta are averaged over the shells (guide, sec. 'General
    multipoles').  Pure 3j algebra; cached.

    Index letters of the contraction (all magnetic numbers, arrays indexed by m + l):
        leg 1: rho1 (lam1), m1 (ell1), m'1 (ell'1), M1 (L1), alpha (a), nu1 (Lam), kappa1 (K1)
        leg 2: rho2 (lam2), m2 (ell2), m'2 (ell'2), M2 (L2), beta (b), nu2 (Lam'), kappa2 (K2)
        anchor: M (L), M' (L'), gamma (L3), mu3 (Lam3)
    """

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _leg(lam, ell, ellp, L, a, Lam):
        """B[rho, M, m, m', alpha] = sum_nu R[rho, m, nu] g[m', M, alpha, nu]: leg harmonics before the coupling to K."""
        R = leg_projection_tensor(lam, ell, Lam)          # rho m nu
        g = leg_tensor(ellp, L, a, Lam)                   # m' M alpha nu
        return np.einsum('rmn,pMan->rMmpa', R, g)

    def t_all(self, ell, ellp, L1, L2, L3, a, b, Lam, Lamp, lam1, lam2) -> dict:
        """{(K1, K2, Lam3): T_n} for all couplings of the given leg ranks (one tensor chain, many projections)."""
        from .wigner import tri_multi
        key = (tuple(ell), tuple(ellp), L1, L2, L3, a, b, Lam, Lamp, lam1, lam2)
        if key in self._cache:
            return self._cache[key]
        l1, l2, L = ell
        lp1, lp2, Lp = ellp
        B1 = self._leg(lam1, l1, lp1, L1, a, Lam)         # r A m p a
        B2 = self._leg(lam2, l2, lp2, L2, b, Lamp)        # s B n q b
        CGv = cg_tensor(a, b, L3)                         # a b g
        W = w3j_tensor(l1, l2, L)                         # m n M
        Wp = w3j_tensor(lp1, lp2, Lp)                     # p q N
        # the five-direction magnetic tensor before the anchor integral; order keeps intermediates ~1e6
        D = np.einsum('rAmpa,mnM->rApanM', B1, W)
        D = np.einsum('rApanM,abg->rApnMbg', D, CGv)
        D = np.einsum('rApnMbg,sBnqb->rApMgsBq', D, B2)
        D = np.einsum('rApMgsBq,pqN->rAMgsBN', D, Wp)
        out = {}
        for Lam3 in tri_multi((L, Lp, L3)):
            X = anchor_tensor(Lam3, L, Lp, L3)            # u M N g
            F = np.einsum('rAMgsBN,uMNg->rAsBu', D, X)
            for K1 in all_tri(lam1, L1):
                CG1 = cg_tensor(lam1, L1, K1)             # r A k
                for K2 in all_tri(lam2, L2):
                    if Lam3 not in all_tri(K1, K2):
                        continue
                    CG2 = cg_tensor(lam2, L2, K2)         # s B l
                    W3 = w3j_tensor(K1, K2, Lam3)         # k l u
                    out[(K1, K2, Lam3)] = float(np.einsum('rAk,sBl,klu,rAsBu->', CG1, CG2, W3, F, optimize=True))
        self._cache[key] = out
        return out

    def t2_all(self, ell, ellp, L1, L2, L3, lpp, a, b, Lam, Lamp, Ly, lam1, lam2) -> dict:
        """Class-2 coupling numbers {(K1, K2, Lam3): T} for the pairing (1 1')(2 3')(3 2') in the (p1, p3) variables.

        Labels: L1 = rank of P(p1; y1^) (leg 1), L2 = rank of P(p2; x^) (the anchor spectrum, inside the vertex),
        L3 = rank of P(p3; y2^) (leg 2), l'' = rank of Y_{ell'2}(p2^) conj(Y_{L2})(p2^) merged, (a, b) = bipolar ranks
        of the vertex on (p1^, p3^), Lam / Lam' = ranks on p1^ / p3^ after merging, Ly = rank at y2^ (L3 with the
        primed line of sight L').  The Phi_n basis label at y2^ is Ly (in place of class 1's L2).
        """
        from .wigner import tri_multi
        key = ('c2', tuple(ell), tuple(ellp), L1, L2, L3, lpp, a, b, Lam, Lamp, Ly, lam1, lam2)
        if key in self._cache:
            return self._cache[key]
        l1, l2, L = ell
        lp1, lp2, Lp = ellp
        B1 = self._leg(lam1, l1, lp1, L1, a, Lam)                       # r A m p a   (rho1, M1, m1, m'1, alpha)
        R2 = leg_projection_tensor(lam2, l2, Lamp)                      # s n v       (rho2, m2, nu2)
        g3 = leg3_tensor(L3, b, Lamp)                                   # C b v       (M3, beta, nu2)
        B2 = np.einsum('snv,Cbv->sCnb', R2, g3)                         # s C n b
        g2 = vertex2_tensor(lp2, L2, lpp)                               # q B e       (m'2, M2, m'')
        V2 = _negate(cg_tensor(a, b, lpp), (2,), (lpp,))                # a b e : (-1)^{m''} <a alpha b beta | l'', -m''>
        W = w3j_tensor(l1, l2, L)                                       # m n M
        Wp = w3j_tensor(lp1, lp2, Lp)                                   # p q N
        Py = y2_tensor(Ly, L3, Lp)                                      # y C N       (mu_y, M3, M')
        out = {}
        # chain: contract the estimator 3j symbols and the vertex first, then the two legs
        D = np.einsum('rAmpa,mnM->rApanM', B1, W)                        # rho1 M1 m'1 alpha m2 M
        D = np.einsum('rApanM,pqN->rAaNnMq', D, Wp)                      # rho1 M1 alpha M' m2 M m'2
        D = np.einsum('rAaNnMq,qBe->rAaNnMBe', D, g2)                     # ... M2 m''
        D = np.einsum('rAaNnMBe,abe->rANnMBb', D, V2)                     # rho1 M1 M' m2 M M2 beta
        D = np.einsum('rANnMBb,sCnb->rANMBsC', D, B2)                     # rho1 M1 M' M M2 rho2 M3
        D = np.einsum('rANMBsC,yCN->rAMBsy', D, Py)                       # rho1 M1 M M2 rho2 mu_y
        for Lam3 in tri_multi((L, L2)):
            X2 = anchor2_tensor(Lam3, L, L2)                              # u M B
            F = np.einsum('rAMBsy,uMB->rAsyu', D, X2)                      # rho1 M1 rho2 mu_y mu3
            for K1 in all_tri(lam1, L1):
                CG1 = cg_tensor(lam1, L1, K1)
                for K2 in all_tri(lam2, Ly):
                    if Lam3 not in all_tri(K1, K2):
                        continue
                    CG2 = cg_tensor(lam2, Ly, K2)
                    W3 = w3j_tensor(K1, K2, Lam3)
                    out[(K1, K2, Lam3)] = float(np.einsum('rAk,syl,klu,rAsyu->', CG1, CG2, W3, F, optimize=True))
        self._cache[key] = out
        return out

    def t(self, ell, ellp, L1, L2, L3, a, b, Lam, Lamp, n) -> float:
        lam1, nL1, K1, lam2, nL2, K2, Lam3 = n
        if nL1 != L1 or nL2 != L2:
            return 0.0
        return self.t_all(ell, ellp, L1, L2, L3, a, b, Lam, Lamp, lam1, lam2).get((K1, K2, Lam3), 0.0)

    # ------------------------------------------------------------------------------------------- index sets
    @staticmethod
    def index_set(ell, ellp, Ls, lam_w: int, a_max: int | None = None):
        """All (L1, L2, L3, a, b, Lam, Lam', n) allowed by the selection rules with lam_a <= lam_w (guide, sec.
        'Truncation'): Lam in tri(ell'_1, L1, a) (same parity, through the Gaunt g), lam1 in tri(ell1, Lam),
        K1 in all_tri(lam1, L1); likewise leg 2; Lam3 in tri(L, L', L3) (even) and all_tri(K1, K2).
        a <= lam_w + ell1 + ell'1 + L1 (and b likewise) unless a_max is given."""
        from .wigner import tri_multi
        l1, l2, L = ell
        lp1, lp2, Lp = ellp
        out = []
        for L1 in Ls:
            for L2 in Ls:
                for L3 in Ls:
                    amax = lam_w + l1 + lp1 + L1 if a_max is None else a_max
                    bmax = lam_w + l2 + lp2 + L2 if a_max is None else a_max
                    for a in range(0, amax + 1):
                        for b in range(abs(a - L3), min(a + L3, bmax) + 1):
                            if (a + b + L3) % 2:
                                continue
                            for Lam in tri_multi((lp1, L1, a)):
                                for Lamp in tri_multi((lp2, L2, b)):
                                    for lam1 in tri(l1, Lam):
                                        if lam1 > lam_w:
                                            continue
                                        for lam2 in tri(l2, Lamp):
                                            if lam2 > lam_w:
                                                continue
                                            for K1 in all_tri(lam1, L1):
                                                for K2 in all_tri(lam2, L2):
                                                    for Lam3 in sorted(set(tri_multi((L, Lp, L3))) & set(all_tri(K1, K2))):
                                                        out.append((L1, L2, L3, a, b, Lam, Lamp,
                                                                    (lam1, L1, K1, lam2, L2, K2, Lam3)))
        return out


def phase(ell, Lam: int, Lamp: int) -> float:
    """(-i)^{ell1 + ell2} i^{Lam + Lam'}, real by parity."""
    e = Lam + Lamp - ell[0] - ell[1]
    assert e % 2 == 0, (ell, Lam, Lamp)
    return (-1.0) ** (e // 2)


def phase2(ell, Lam: int, Lamp: int) -> float:
    """Class 2: (-i)^{ell1 + ell2} i^{Lam} (-i)^{Lam'}, real by parity (Lam + Lam' even)."""
    e = Lam - Lamp - ell[0] - ell[1]
    assert e % 2 == 0, (ell, Lam, Lamp)
    return (-1.0) ** (e // 2)


def box_window_value(n, V: float) -> float:
    """Q_n in a periodic box of volume V with a global line of sight (guide, box check of sec. 'Rotational invariance'):

        Q_n^box = V int dOmega_r1 dOmega_r2 Phi_n(r1^, n^, r2^, n^, n^) = V sqrt(N_{L1 L2 Lam3} / 4 pi) H_{L1 L2 Lam3}

    for lam1 = lam2 = 0 (then K_a = L_a), zero otherwise: with lam = 0 the leg couplings are Y_00 Y_{L kappa}, each
    leg integral gives sqrt(4 pi), and sum_m (L1 L2 Lam3; m) Y Y Y(n^) = H sqrt(N) / (4 pi)^{3/2}.  Verified against
    the direct angular average in tests/test_bispectrum_algebra.py (box identity)."""
    lam1, L1, K1, lam2, L2, K2, Lam3 = n
    if lam1 != 0 or lam2 != 0 or K1 != L1 or K2 != L2:
        return 0.0
    return V * H(L1, L2, Lam3) * math.sqrt(N(L1, L2, Lam3) / FOUR_PI)
