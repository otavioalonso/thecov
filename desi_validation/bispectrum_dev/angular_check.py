"""Direct angular quadrature of the class-1 primed average, used by tests/test_bispectrum_algebra.py."""
import numpy as np
from scipy.special import spherical_jn, eval_legendre
from thecov.bispectrum_algebra import (BispectrumCoupling, bipolar_coefficients, invariant_basis, phase,
                                       sugiyama_S_tilde)
from thecov.wigner import FOUR_PI


def unit(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def sphere_grid(nth, nph):
    """Product Gauss-Legendre x uniform-azimuth grid with weights normalised to a mean over the sphere."""
    x, w = np.polynomial.legendre.leggauss(nth)
    ph = np.arange(nph) * 2 * np.pi / nph
    ct = np.repeat(x, nph)
    phi = np.tile(ph, nth)
    wt = np.repeat(w, nph) * (2 * np.pi / nph) / FOUR_PI
    st = np.sqrt(1 - ct ** 2)
    return np.stack([st * np.cos(phi), st * np.sin(phi), ct], 1), wt


def direct_V(ellp, PL, p1, p2, r1v, r2v, y1, y2, x, nth=28, nph=56):
    """<< e^{i p1.r1 + i p2.r2} S~_ell'(p1^, p2^, x^) P(p1; y1^) P(p2; y2^) P(p1 + p2; x^) >>_{Omega1 Omega2}."""
    d, w = sphere_grid(nth, nph)
    n = len(d)
    P1 = sum(PL[L](p1) * eval_legendre(L, d @ y1) for L in PL)
    P2 = sum(PL[L](p2) * eval_legendre(L, d @ y2) for L in PL)
    e1 = np.exp(1j * p1 * d @ r1v) * P1 * w
    e2 = np.exp(1j * p2 * d @ r2v) * P2 * w
    out = 0.0
    for i0 in range(0, n, 128):
        D1 = d[i0:i0 + 128]
        p3v = p1 * D1[:, None, :] + p2 * d[None, :, :]
        p3 = np.linalg.norm(p3v, axis=-1)
        p3h = p3v / p3[..., None]
        P3 = sum(PL[L](p3) * eval_legendre(L, p3h @ x) for L in PL)
        S = sugiyama_S_tilde(ellp, np.broadcast_to(D1[:, None, :], p3v.shape),
                             np.broadcast_to(d[None, :, :], p3v.shape), np.broadcast_to(x, p3v.shape))
        out = out + np.sum(e1[i0:i0 + 128, None] * e2[None, :] * S * P3)
    return out


def direct_box_average(ell, ellp, PL, p1, p2, nhat, nth=28, nph=56):
    """<< conj(S~_ell) S~_ell' P(p1; n) P(p2; n) P(p1 + p2; n) >>_{Omega1 Omega2} with a global line of sight."""
    d, w = sphere_grid(nth, nph)
    n = len(d)
    P1 = sum(PL[L](p1) * eval_legendre(L, d @ nhat) for L in PL) * w
    P2 = sum(PL[L](p2) * eval_legendre(L, d @ nhat) for L in PL) * w
    out = 0.0
    for i0 in range(0, n, 128):
        D1 = d[i0:i0 + 128]
        p3v = p1 * D1[:, None, :] + p2 * d[None, :, :]
        p3 = np.linalg.norm(p3v, axis=-1)
        p3h = np.where(p3[..., None] > 0, p3v / np.where(p3 > 0, p3, 1.0)[..., None], nhat)   # p3 = 0 at antipodal nodes
        P3 = sum(PL[L](p3) * eval_legendre(L, p3h @ nhat) for L in PL)
        A = np.broadcast_to(D1[:, None, :], p3v.shape)
        B = np.broadcast_to(d[None, :, :], p3v.shape)
        Nn = np.broadcast_to(nhat, p3v.shape)
        S = np.conj(sugiyama_S_tilde(ell, A, B, Nn)) * sugiyama_S_tilde(ellp, A, B, Nn)
        out = out + np.sum(P1[i0:i0 + 128, None] * P2[None, :] * S * P3)
    return out


def series_F(ell, ellp, PL, p1, p2, r1, r2, dirs, lam_w, a_max, coup):
    """sum_n T_n Phi_n with the radial factors at fixed magnitudes (no shell averages): the angular kernel
    (-i)^{l1+l2} conj(S~_ell)(r1^, r2^, x^) V at independent directions."""
    r1h, y1, r2h, y2, x = dirs
    Ls = tuple(sorted(PL))
    tot, nterms, cache_c, cache_phi = 0.0, 0, {}, {}
    for (L1, L2, L3, a, b, Lam, Lamp, n) in BispectrumCoupling.index_set(ell, ellp, Ls, lam_w, a_max):
        T = coup.t(ell, ellp, L1, L2, L3, a, b, Lam, Lamp, n)
        if abs(T) < 1e-14:
            continue
        key = (L3, a, b)
        if key not in cache_c:
            cache_c[key] = float(bipolar_coefficients(L3, a, b, p1, p2, PL[L3]))
        c = cache_c[key]
        if c == 0.0:
            continue
        if n not in cache_phi:
            cache_phi[n] = invariant_basis(n, r1h, y1, r2h, y2, x)
        pref = FOUR_PI ** 3 / ((2 * L1 + 1) * (2 * L2 + 1) * (2 * L3 + 1)) * PL[L1](p1) * PL[L2](p2) * c
        rad = spherical_jn(Lam, p1 * r1) * spherical_jn(Lamp, p2 * r2) * phase(ell, Lam, Lamp)
        tot = tot + pref * rad * T * cache_phi[n]
        nterms += 1
    return tot, nterms
