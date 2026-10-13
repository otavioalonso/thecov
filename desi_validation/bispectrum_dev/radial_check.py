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
