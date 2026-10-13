"""Tapered-box reference for the radial assembly (tests/test_bispectrum_algebra.py::test_radial_assembly_tapered_box).

With Q_n(r1, r2) = Q_n^box exp(-(r1^2 + r2^2) / 2 sigma^2) only lam = 0 contributes, the leg integrals become the
smoothed shell indicators I_i(p) = 4 pi int r^2 jbar^{(i)}_l(r) j_l(p r) e^{-r^2/2 sigma^2} dr (1D, fast convergence),
and the assembled C[i, j, i', j'] (without prefactor) must equal
    V << I_i(p1) I_j(p2) << conj(S~_ell) S~_ell' P(p1) P(p2) P(p1 + p2) >>_Omega >>_{i' j'} .
"""
import numpy as np
from scipy.special import spherical_jn
from angular_check import direct_box_average


def tapered_reference(K, ell, ellp, PL, sigma, nhat, V=1.0, nq_ref=6, nth=16, nph=32):
    nb, r = K.nbins, K.r
    x, w = np.polynomial.legendre.leggauss(nq_ref)
    lo, hi = K.k_edges[:-1], K.k_edges[1:]
    kq = 0.5 * (hi - lo)[:, None] * x + 0.5 * (hi + lo)[:, None]
    wq = 0.5 * (hi - lo)[:, None] * w * kq ** 2
    tap = np.exp(-r ** 2 / (2 * sigma ** 2))
    kf = kq.ravel()

    def smoothed_indicator(l):
        J = spherical_jn(l, kf[:, None] * r[None, :])
        return 4 * np.pi * np.einsum('ir,Qr->iQ', K.jbar(l) * K.r_weights * tap, J).reshape(nb, nb, nq_ref)
    I1, I2 = smoothed_indicator(ell[0]), smoothed_indicator(ell[1])
    ang = np.zeros((len(kf), len(kf)))
    for a in range(len(kf)):
        for b in range(len(kf)):
            ang[a, b] = direct_box_average(ell, ellp, PL, kf[a], kf[b], nhat, nth, nph).real
    ang = ang.reshape(nb, nq_ref, nb, nq_ref)
    wn = wq / K.norm[:, None]
    return V * np.einsum('iIq,jJp,Iq,Jp,IqJp->ijIJ', I1, I2, wn, wn, ang)


def box_class2_direct(ell, ellp, PL, k_edges, V, Vi, n_mc=2_000_000, seed=0, nhat=None):
    """Analytic class-2 box term (pairing (1 1')(2 3')(3 2')) by Monte Carlo over the two shells (exact indicator):
    C[i, j, i', j'] = (1/(V V_i V_j)) delta_ii' << 1(|p1+p2| in j) conj(S~_ell)(p1^, -p3^, n) S~_ell'(p1^, p2^, n) P P P >>_{i' j'}
    WITHOUT the prefactor (4pi)^3 H H' sqrt(N N')."""
    from thecov.bispectrum_algebra import sugiyama_S_tilde
    from scipy.special import eval_legendre
    rng = np.random.default_rng(seed)
    nb = len(k_edges) - 1
    if nhat is None:
        nhat = np.array([0.0, 0.0, 1.0])

    def unit(n):
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1)[:, None]
    out = np.zeros((nb, nb, nb, nb))
    for I in range(nb):
        for J in range(nb):
            p1 = rng.uniform(k_edges[I] ** 3, k_edges[I + 1] ** 3, n_mc) ** (1 / 3)
            p2 = rng.uniform(k_edges[J] ** 3, k_edges[J + 1] ** 3, n_mc) ** (1 / 3)
            d1, d2 = unit(n_mc), unit(n_mc)
            p3v = p1[:, None] * d1 + p2[:, None] * d2
            p3 = np.linalg.norm(p3v, axis=1)
            p3h = p3v / np.where(p3 > 0, p3, 1.0)[:, None]
            n3 = np.broadcast_to(nhat, d1.shape)
            P = (sum(PL[L](p1) * eval_legendre(L, d1 @ nhat) for L in PL) * sum(PL[L](p2) * eval_legendre(L, d2 @ nhat) for L in PL)
                 * sum(PL[L](p3) * eval_legendre(L, p3h @ nhat) for L in PL))
            S = (np.conj(sugiyama_S_tilde(ell, d1, -p3h, n3)) * sugiyama_S_tilde(ellp, d1, d2, n3)).real
            jb = np.digitize(p3, k_edges) - 1
            ok = (jb >= 0) & (jb < nb)
            acc = np.bincount(jb[ok], weights=(P * S)[ok], minlength=nb) / n_mc
            for j in range(nb):
                out[I, j, I, J] += acc[j] / (V * Vi[I] * Vi[j])
    return out
