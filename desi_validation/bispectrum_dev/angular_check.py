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


# ----------------------------------------------------------------------------------------------- class 2
def direct_V2(ellp, PL_leg1, PL_anchor, PL_leg2, p1, p3, r1v, r2v, y1, y2, x, nth=28, nph=56, k_range=None):
    """<< e^{i p1.r1 - i p3.r2} S~_ell'(p1^, p2^, y2^) P(p1; y1^) P(p3; y2^) P(p2; x^) [1(|p2| in k_range)] >>_{Omega1 Omega3},
    p2 = p3 - p1: the primed average of the class-2 pairing (1 1')(2 3')(3 2') in the (p1, p3) variables."""
    d, w = sphere_grid(nth, nph)
    n = len(d)
    P1 = sum(PL_leg1[L](p1) * eval_legendre(L, d @ y1) for L in PL_leg1)
    P3 = sum(PL_leg2[L](p3) * eval_legendre(L, d @ y2) for L in PL_leg2)
    e1 = np.exp(1j * p1 * d @ r1v) * P1 * w
    e3 = np.exp(-1j * p3 * d @ r2v) * P3 * w
    out = 0.0
    for i0 in range(0, n, 128):
        D1 = d[i0:i0 + 128]
        p2v = p3 * d[None, :, :] - p1 * D1[:, None, :]
        p2 = np.linalg.norm(p2v, axis=-1)
        p2h = np.where(p2[..., None] > 0, p2v / np.where(p2 > 0, p2, 1.0)[..., None], x)
        P2 = sum(PL_anchor[L](p2) * eval_legendre(L, p2h @ x) for L in PL_anchor)
        if k_range is not None:
            P2 = P2 * ((p2 >= k_range[0]) & (p2 < k_range[1]))
        S = sugiyama_S_tilde(ellp, np.broadcast_to(D1[:, None, :], p2v.shape), p2h, np.broadcast_to(y2, p2v.shape))
        out = out + np.sum(e1[i0:i0 + 128, None] * e3[None, :] * S * P2)
    return out


def class2_labels(ell, ellp, Ls, lam_w, a_max=None):
    """(L1, L2, L3, l'', a, b, Lam, Lam', Ly, n) of the class-2 pairing with lam_a <= lam_w."""
    from thecov.wigner import tri, tri_multi
    from thecov.bispectrum_algebra import all_tri
    l1, l2, L = ell
    lp1, lp2, Lp = ellp
    out = []
    for L1 in Ls:
        for L2 in Ls:
            for L3 in Ls:
                for lpp in tri(lp2, L2):
                    amax = lam_w + l1 + lp1 + L1 if a_max is None else a_max
                    bmax = lam_w + l2 + L3 if a_max is None else a_max
                    for a in range(0, amax + 1):
                        for b in range(abs(a - lpp), min(a + lpp, bmax) + 1):
                            if (a + b + lpp) % 2:
                                continue
                            for Lam in tri_multi((lp1, L1, a)):
                                for Lamp in tri(L3, b):
                                    for Ly in tri(L3, Lp):
                                        for lam1 in tri(l1, Lam):
                                            if lam1 > lam_w:
                                                continue
                                            for lam2 in tri(l2, Lamp):
                                                if lam2 > lam_w:
                                                    continue
                                                for K1 in all_tri(lam1, L1):
                                                    for K2 in all_tri(lam2, Ly):
                                                        for Lam3 in sorted(set(tri(L, L2)) & set(all_tri(K1, K2))):
                                                            out.append((L1, L2, L3, lpp, a, b, Lam, Lamp, Ly,
                                                                        (lam1, L1, K1, lam2, Ly, K2, Lam3)))
    return out


def series_F2(ell, ellp, PL_leg1, PL_anchor, PL_leg2, p1, p3, r1, r2, dirs, lam_w, a_max, coup, k_range=None):
    """sum_n T2_n Phi_n with the radial factors at fixed (p1, p3): the class-2 angular kernel."""
    from thecov.bispectrum_algebra import phase2
    r1h, y1, r2h, y2, x = dirs
    Ls = tuple(sorted(set(PL_leg1) | set(PL_anchor) | set(PL_leg2)))
    tot, nterms, cache_c, cache_phi = 0.0, 0, {}, {}
    for (L1, L2, L3, lpp, a, b, Lam, Lamp, Ly, n) in class2_labels(ell, ellp, Ls, lam_w, a_max):
        if L1 not in PL_leg1 or L2 not in PL_anchor or L3 not in PL_leg2:
            continue
        T = coup.t2_all(ell, ellp, L1, L2, L3, lpp, a, b, Lam, Lamp, Ly, n[0], n[3]).get((n[2], n[5], n[6]), 0.0)
        if abs(T) < 1e-14:
            continue
        key = (L2, lpp, a, b)
        if key not in cache_c:
            if k_range is None:
                cache_c[key] = (-1) ** a * float(bipolar_coefficients(lpp, a, b, p1, p3, PL_anchor[L2]))
            else:
                from thecov.bispectrum import vertex_coefficients
                cache_c[key] = (-1) ** a * float(vertex_coefficients(lpp, a, b, [p1], [p3], PL_anchor[L2], k_range=k_range)[0])
        c = cache_c[key]
        if c == 0.0:
            continue
        if n not in cache_phi:
            cache_phi[n] = invariant_basis(n, r1h, y1, r2h, y2, x)
        pref = FOUR_PI ** 3 / ((2 * L1 + 1) * (2 * L2 + 1) * (2 * L3 + 1)) * PL_leg1[L1](p1) * PL_leg2[L3](p3) * c
        rad = spherical_jn(Lam, p1 * r1) * spherical_jn(Lamp, p3 * r2) * phase2(ell, Lam, Lamp)
        tot = tot + pref * rad * T * cache_phi[n]
        nterms += 1
    return tot, nterms
