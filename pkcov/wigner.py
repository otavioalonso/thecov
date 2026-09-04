r"""Wigner 3j symbols, Gaunt tensors and the coupling coefficients t^(1), t^(2).

Conventions (see the accompanying derivation, Section 2):

    G^{m_1 ... m_n}_{l_1 ... l_n} = \int dOmega  prod_a Y_{l_a m_a}(n)

with orthonormal complex spherical harmonics including the Condon-Shortley phase.
The "merge rule" reads  prod_a Y_{l_a m_a} = sum_{Lambda mu} G^{m_1..m_n mu}_{l_1..l_n Lambda} conj(Y_{Lambda mu}).
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np

FOUR_PI = 4.0 * math.pi


# ---------------------------------------------------------------------------
# Wigner 3j (exact Racah formula, cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Wigner 3j symbol (j1 j2 j3; m1 m2 m3) for integer arguments."""
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    f = math.factorial
    delta = Fraction(f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3), f(j1 + j2 + j3 + 1))
    pref = delta * (f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3))
    kmin = max(0, j2 - j3 - m1, j1 - j3 + m2)
    kmax = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    s = Fraction(0)
    for k in range(kmin, kmax + 1):
        den = (f(k) * f(j1 + j2 - j3 - k) * f(j1 - m1 - k) * f(j2 + m2 - k)
               * f(j3 - j2 + m1 + k) * f(j3 - j1 - m2 + k))
        s += Fraction((-1) ** k, den)
    sign = -1.0 if (j1 - j2 - m3) % 2 else 1.0
    return sign * float(s) * math.sqrt(float(pref))


def tri(a: int, b: int) -> list:
    """Same-parity triangle set {|a-b|, |a-b|+2, ..., a+b}."""
    return list(range(abs(a - b), a + b + 1, 2))


@lru_cache(maxsize=None)
def tri_multi(ls: tuple) -> tuple:
    """Set of total angular momenta reachable by coupling the list ls (same parity)."""
    if len(ls) == 1:
        return (ls[0],)
    out = set()
    for a in tri_multi(ls[:-1]):
        out.update(tri(a, ls[-1]))
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Tensors over magnetic numbers (index = m + l)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def w3j_tensor(l1: int, l2: int, l3: int) -> np.ndarray:
    """Array W[m1+l1, m2+l2, m3+l3] = (l1 l2 l3; m1 m2 m3)."""
    W = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1))
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            m3 = -m1 - m2
            if abs(m3) <= l3:
                W[m1 + l1, m2 + l2, m3 + l3] = wigner_3j(l1, l2, l3, m1, m2, m3)
    return W


@lru_cache(maxsize=None)
def gaunt_tensor(ls: tuple) -> np.ndarray:
    """n-harmonic integral G^{m_1..m_n}_{l_1..l_n} as an array indexed by (m_a + l_a).

    For n = 2:  (-1)^{m_1} delta_{l_1 l_2} delta_{m_1,-m_2}.
    For n = 3:  the ordinary Gaunt coefficient.
    For n >= 4: recursion (eq. gaunt-rec of the note), coupling the first n-1 harmonics
                to an intermediate Lambda'.
    """
    ls = tuple(int(l) for l in ls)
    n = len(ls)
    if n == 2:
        l1, l2 = ls
        G = np.zeros((2 * l1 + 1, 2 * l2 + 1))
        if l1 == l2:
            for m in range(-l1, l1 + 1):
                G[m + l1, -m + l2] = (-1) ** m
        return G
    if n == 3:
        l1, l2, l3 = ls
        pref = math.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / FOUR_PI) * wigner_3j(l1, l2, l3, 0, 0, 0)
        if pref == 0.0:
            return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1))
        return pref * w3j_tensor(l1, l2, l3)
    # recursion: G(l_1..l_{n-1}, l_n, Lam)[..., m_n, mu]
    head, ln, Lam = ls[:-2], ls[-2], ls[-1]
    shape = tuple(2 * l + 1 for l in ls)
    G = np.zeros(shape)
    for Lp in tri_multi(head):
        if Lam not in tri(Lp, ln):
            continue
        A = gaunt_tensor(head + (Lp,))                      # (..., 2Lp+1)  index mu'
        B = gaunt_tensor((Lp, ln, Lam))                     # (2Lp+1, 2ln+1, 2Lam+1) index (-mu', m_n, mu)
        mup = np.arange(-Lp, Lp + 1)
        Bflip = B[::-1] * ((-1.0) ** mup)[:, None, None]    # Bflip[mu'] = (-1)^{mu'} B[-mu']
        G += np.tensordot(A, Bflip, axes=([A.ndim - 1], [0]))
    return G


# ---------------------------------------------------------------------------
# Coupling coefficients
# ---------------------------------------------------------------------------
class CouplingCoefficients:
    """t^(1) and t^(2) of the note (eqs. t1, t2), computed by full contraction.

    Index letters used in the einsum:
        a = m_1 (ell_1)   b = M_1 (L_1)   c = nu  (lambda)
        d = m_2 (ell_2)   e = M_2 (L_2)   f = nu' (lambda')
        g = mu_1 (Lambda_1)   h = mu_2 (Lambda_2)   i = mu (Lambda)
    """

    def __init__(self):
        self._cache = {}

    def unprojected(self, term, ell1, ell2, L1, L2, lam, lamp, Lam1, Lam2, Lam) -> np.ndarray:
        """T^{mu_1 mu_2 mu} before contraction with the 3j symbol."""
        K1 = gaunt_tensor((ell1, L1, lam))      # abc
        K2 = gaunt_tensor((ell2, L2, lamp))     # def
        S = gaunt_tensor((lam, lamp, Lam))      # cfi
        if term == 1:
            X = gaunt_tensor((ell1, L2, Lam1))   # aeg   harmonics at x-hat : (ell_1, L_2)
            Xp = gaunt_tensor((L1, ell2, Lam2))  # bdh   harmonics at x'-hat: (L_1, ell_2)
            A1 = np.einsum('abc,aeg->bceg', K1, X)
            A2 = np.einsum('def,bdh->efbh', K2, Xp)
            A3 = np.einsum('bceg,efbh->cgfh', A1, A2)
            return np.einsum('cgfh,cfi->ghi', A3, S)
        elif term == 2:
            X = gaunt_tensor((ell1, ell2, L2, Lam1))  # adeg  harmonics at x-hat : (ell_1, ell_2, L_2)
            Xp = gaunt_tensor((L1, Lam2))             # bh    harmonics at x'-hat: (L_1)
            A1 = np.einsum('abc,bh->ach', K1, Xp)
            A2 = np.einsum('def,adeg->afg', K2, X)
            A3 = np.einsum('ach,afg->chfg', A1, A2)
            return np.einsum('chfg,cfi->ghi', A3, S)
        raise ValueError("term must be 1 or 2")

    def t(self, term, ell1, ell2, L1, L2, lam, lamp, Lam1, Lam2, Lam) -> float:
        key = (term, ell1, ell2, L1, L2, lam, lamp, Lam1, Lam2, Lam)
        if key not in self._cache:
            T = self.unprojected(*key)
            self._cache[key] = float(np.sum(T * w3j_tensor(Lam1, Lam2, Lam)))
        return self._cache[key]

    def check_3j_proportional(self, *key) -> float:
        """Return max |T - t * 3j| / max|T| ; should be ~1e-15 (see Appendix A)."""
        T = self.unprojected(*key)
        W = w3j_tensor(*key[-3:])
        t = float(np.sum(T * W))
        scale = np.max(np.abs(T))
        if scale < 1e-13:          # numerically zero tensor: nothing to check
            return 0.0
        return float(np.max(np.abs(T - t * W)) / scale)
