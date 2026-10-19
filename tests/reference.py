"""Slow reference implementations, kept only to test the fast ones in thecov."""
import numpy as np

from thecov.harmonics import angles, sph_harm_table, normalized_legendre, cos_dphi, cos_multiples, tripolar_frame_weights
from thecov.wigner import wigner_3j, w3j_tensor


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


def check_3j_proportional(cc, *key) -> float:
    """Return max |T - t * 3j| / max|T| ; should be ~1e-15 (see Appendix A)."""
    T = cc.unprojected(*key)
    W = w3j_tensor(*key[-3:])
    t = float(np.sum(T * W))
    scale = np.max(np.abs(T))
    if scale < 1e-13:          # numerically zero tensor: nothing to check
        return 0.0
    return float(np.max(np.abs(T - t * W)) / scale)
