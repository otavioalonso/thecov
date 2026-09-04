"""Normalised associated Legendre functions, spherical-harmonic tables and the
tripolar spherical harmonic S_{Lambda1 Lambda2 Lambda}(x, x', s) of the note (eq. S).
"""
from __future__ import annotations

import math

import numpy as np

from .wigner import wigner_3j, FOUR_PI


def normalized_legendre(x, lmax: int) -> np.ndarray:
    """Fully normalised associated Legendre functions Pbar[l, m](x), m >= 0.

    Y_{l m}(theta, phi) = Pbar[l, m](cos theta) exp(i m phi)   (Condon-Shortley phase included).
    Returns an array of shape (lmax+1, lmax+1) + x.shape ; entries with m > l are zero.
    """
    x = np.asarray(x, dtype=float)
    P = np.zeros((lmax + 1, lmax + 1) + x.shape)
    sx = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    P[0, 0] = 1.0 / math.sqrt(FOUR_PI)
    for m in range(1, lmax + 1):
        P[m, m] = -math.sqrt((2 * m + 1) / (2 * m)) * sx * P[m - 1, m - 1]
    for m in range(0, lmax):
        P[m + 1, m] = math.sqrt(2 * m + 3) * x * P[m, m]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            a = math.sqrt((4 * l * l - 1) / (l * l - m * m))
            b = math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
            P[l, m] = a * (x * P[l - 1, m] - b * P[l - 2, m])
    return P


def sph_harm_table(cos_theta, phi, lmax: int) -> np.ndarray:
    """Y[l, m + lmax](theta, phi) for all 0 <= l <= lmax, -l <= m <= l.

    Shape (lmax+1, 2*lmax+1) + cos_theta.shape, complex.
    """
    cos_theta = np.asarray(cos_theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    P = normalized_legendre(cos_theta, lmax)
    Y = np.zeros((lmax + 1, 2 * lmax + 1) + cos_theta.shape, dtype=complex)
    for m in range(0, lmax + 1):
        e = np.exp(1j * m * phi)
        for l in range(m, lmax + 1):
            Y[l, m + lmax] = P[l, m] * e
            if m > 0:
                Y[l, -m + lmax] = (-1) ** m * np.conj(P[l, m] * e)
    return Y


def unit_vectors(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def angles(nhat: np.ndarray):
    """(cos theta, phi) of unit vectors, shape (..., 3)."""
    ct = np.clip(nhat[..., 2], -1.0, 1.0)
    phi = np.arctan2(nhat[..., 1], nhat[..., 0])
    return ct, phi


def tripolar_direct(Lam1: int, Lam2: int, Lam: int, xhat, xphat, shat) -> np.ndarray:
    """S_{Lam1 Lam2 Lam}(x, x', s) = sum (3j) conj(Y)(x) conj(Y)(x') conj(Y)(s), evaluated directly.

    Slow (used for tests); xhat, xphat, shat are arrays of unit vectors with a common shape (..., 3).
    """
    ct1, ph1 = angles(xhat)
    ct2, ph2 = angles(xphat)
    ct3, ph3 = angles(shat)
    Y1 = np.conj(sph_harm_table(ct1, ph1, Lam1))
    Y2 = np.conj(sph_harm_table(ct2, ph2, Lam2))
    Y3 = np.conj(sph_harm_table(ct3, ph3, Lam))
    out = np.zeros(np.broadcast(ct1, ct2, ct3).shape, dtype=complex)
    for mu1 in range(-Lam1, Lam1 + 1):
        for mu2 in range(-Lam2, Lam2 + 1):
            mu = -mu1 - mu2
            if abs(mu) > Lam:
                continue
            w = wigner_3j(Lam1, Lam2, Lam, mu1, mu2, mu)
            if w == 0.0:
                continue
            out += w * Y1[Lam1, mu1 + Lam1] * Y2[Lam2, mu2 + Lam2] * Y3[Lam, mu + Lam]
    return out


def tripolar_frame_weights(Lam1: int, Lam2: int, Lams) -> np.ndarray:
    """Weights w[mu, j] such that, in the frame where s-hat = z-hat,

        S_{Lam1 Lam2 Lam_j}(x, x', s) = sum_{mu=0}^{min(Lam1,Lam2)} w[mu, j] *
              Pbar[Lam1, mu](c1) Pbar[Lam2, mu](c2) cos(mu * dphi),

    with c1 = x.s, c2 = x'.s and dphi the azimuthal angle between x and x' about s.
    Valid for even Lam1 + Lam2 + Lam (the function is then real and parity even).
    """
    mumax = min(Lam1, Lam2)
    W = np.zeros((mumax + 1, len(Lams)))
    for j, Lam in enumerate(Lams):
        pref = math.sqrt((2 * Lam + 1) / FOUR_PI)
        for mu in range(0, mumax + 1):
            w3 = wigner_3j(Lam1, Lam2, Lam, mu, -mu, 0)
            W[mu, j] = pref * (w3 if mu == 0 else 2.0 * (-1) ** mu * w3)
    return W


def tripolar_frame(Lam1: int, Lam2: int, Lam: int, xhat, xphat, shat) -> np.ndarray:
    """Same as tripolar_direct but via the rotated-frame formula (used in the pair counts)."""
    c1 = np.einsum('...i,...i->...', xhat, shat)
    c2 = np.einsum('...i,...i->...', xphat, shat)
    cx = np.einsum('...i,...i->...', xhat, xphat)
    cosd = cos_dphi(c1, c2, cx)
    P1 = normalized_legendre(c1, Lam1)
    P2 = normalized_legendre(c2, Lam2)
    W = tripolar_frame_weights(Lam1, Lam2, [Lam])[:, 0]
    mumax = min(Lam1, Lam2)
    cm = cos_multiples(cosd, mumax)
    out = np.zeros_like(c1)
    for mu in range(mumax + 1):
        out += W[mu] * P1[Lam1, mu] * P2[Lam2, mu] * cm[mu]
    return out


def cos_dphi(c1, c2, cx):
    """cos of the azimuthal separation of x and x' about s, from the three dot products."""
    den = np.sqrt(np.clip(1 - c1 * c1, 0, None)) * np.sqrt(np.clip(1 - c2 * c2, 0, None))
    safe = den > 1e-12
    out = np.ones_like(den)
    out[safe] = (cx[safe] - c1[safe] * c2[safe]) / den[safe]
    return np.clip(out, -1.0, 1.0)


def cos_multiples(cosd, mumax: int) -> np.ndarray:
    """cos(mu * dphi) for mu = 0..mumax by Chebyshev recursion; shape (mumax+1,) + cosd.shape."""
    cm = np.empty((mumax + 1,) + cosd.shape)
    cm[0] = 1.0
    if mumax >= 1:
        cm[1] = cosd
    for mu in range(2, mumax + 1):
        cm[mu] = 2.0 * cosd * cm[mu - 1] - cm[mu - 2]
    return cm
