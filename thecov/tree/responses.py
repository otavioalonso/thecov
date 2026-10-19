"""Tree-level (and dressed) responses of the power-spectrum multipoles to a long mode, for the SSC.

R_ell^(n)(k): response of P_ell at k to delta_n = the long mode weighted by L_n(q^.x^) (see thecov.ssc).
Kernels: SCF99 redshift-space Z1, Z2 of a biased tracer (delta_g contains b2/2 delta^2 + bs2/2 s^2).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

ELLS = (0, 2, 4)


# ----------------------------------------------------------------------------- responses
def _F2(a, b):
    ab = np.sum(a * b, 0)
    ma, mb = np.sum(a * a, 0), np.sum(b * b, 0)
    return 5 / 7 + 0.5 * ab * (1 / ma + 1 / mb) + 2 / 7 * ab ** 2 / (ma * mb)


def _G2(a, b):
    ab = np.sum(a * b, 0)
    ma, mb = np.sum(a * a, 0), np.sum(b * b, 0)
    return 3 / 7 + 0.5 * ab * (1 / ma + 1 / mb) + 4 / 7 * ab ** 2 / (ma * mb)


def _S2(a, b):
    ab = np.sum(a * b, 0)
    return ab ** 2 / (np.sum(a * a, 0) * np.sum(b * b, 0)) - 1 / 3


def _Z1(a, b1, f):
    return b1 + f * a[2] ** 2 / np.sum(a * a, 0)


def _Z2(a, b, b1, b2, bs2, f):
    """SCF99 second-order redshift-space kernel (delta_g contains b2/2 delta^2 + bs2/2 s^2); n^ = z^."""
    k = a + b
    mk2 = np.sum(k * k, 0)
    ma, mb = np.sum(a * a, 0), np.sum(b * b, 0)
    rsd = 0.5 * f * k[2] * (a[2] / ma * _Z1(b, b1, f) + b[2] / mb * _Z1(a, b1, f))
    return b1 * _F2(a, b) + f * k[2] ** 2 / mk2 * _G2(a, b) + rsd + 0.5 * b2 + 0.5 * bs2 * _S2(a, b)


def response_coefficients(b1, f=0.0, b2=0.0, bs2=0.0, ells=ELLS, ns=ELLS, eps=1e-4, n_gl=16, n_phi=24):
    """(a, c): arrays (len(ells), len(ns)) with R_ell^(n)(k) = a P(k) + c dP/dlnk (tree level).

    The squeezed limit is the average over +-eps (the 1/eps bulk-flow terms are odd and cancel; the
    error is O(eps^2)); the angular integrals are exact (Gauss-Legendre in mu and nu, uniform in phi,
    for polynomials of the degrees that occur)."""
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    mu, nu, ph = np.meshgrid(xg, xg, phi, indexing='ij')
    st, sn = np.sqrt(1 - mu ** 2), np.sqrt(1 - nu ** 2)
    k = np.stack([st * np.cos(ph), st * np.sin(ph), mu])
    qh = np.stack([sn, np.zeros_like(nu), nu])

    def R(neff):
        out = 0.0
        for e in (eps, -eps):
            q = e * qh
            kb = -k + q
            ratio = 1 + neff * (np.sqrt(np.sum(kb * kb, 0)) - 1)          # P(|kb|)/P(k) to first order
            out = out + 2 * (_Z2(k, -q, b1, b2, bs2, f) * _Z1(k, b1, f)
                             + _Z2(kb, -q, b1, b2, bs2, f) * _Z1(kb, b1, f) * ratio)
        return 0.5 * out.mean(axis=2)                                          # azimuthal average

    R0, R1 = R(0.0), R(1.0)
    a, c = np.zeros((len(ells), len(ns))), np.zeros((len(ells), len(ns)))
    for i, l in enumerate(ells):
        Ll = np.polynomial.legendre.Legendre.basis(l)(xg)
        for j, n in enumerate(ns):
            Ln = np.polynomial.legendre.Legendre.basis(n)(xg)
            wgt = (2 * l + 1) / 2 * (2 * n + 1) / 2 * np.outer(wg * Ll, wg * Ln)
            a[i, j] = np.sum(wgt * R0)
            c[i, j] = np.sum(wgt * (R1 - R0))
    return a, c


def response_multipoles(k, p_dressed, b1, f=0.0, b2=0.0, bs2=0.0, ells=ELLS, ns=ELLS, eps=1e-4, n_gl=16, n_phi=24,
                        chunk=64):
    """{(ell, n): R_ell^(n)(k)} for a 'dressed' power p_dressed(v) of a wavevector v (3, ...) with n^ = z^,
    e.g. an IR-damped P_lin(|v|) times fingers-of-God damping exp(-(v_z sigma_v)^2).

    Same squeezed limit as response_coefficients (2 Z2(k, -q) Z1(k) P(k) + 2 Z2(k_b, -q) Z1(k_b) P(k_b),
    k_b = -k + q, averaged over q -> +-eps q), with P evaluated at the actual vectors instead of being
    linearised as a P + c dP/dlnk: this captures the response of the damped, anisotropic power (dilation of
    the damping included). For p_dressed = P_lin(|v|) it reduces to a P + c dP/dlnk."""
    k = np.atleast_1d(np.asarray(k, float))
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    mu, nu, ph = np.meshgrid(xg, xg, phi, indexing='ij')
    st, sn = np.sqrt(1 - mu ** 2), np.sqrt(1 - nu ** 2)
    kh = np.stack([st * np.cos(ph), st * np.sin(ph), mu])[..., None]
    qh = np.stack([sn, np.zeros_like(nu), nu])[..., None]
    out = {(l, n): np.zeros(len(k)) for l in ells for n in ns}
    Lm = {l: np.polynomial.legendre.Legendre.basis(l)(xg) for l in ells}
    Ln = {n: np.polynomial.legendre.Legendre.basis(n)(xg) for n in ns}
    for s0 in range(0, len(k), chunk):
        K = k[s0:s0 + chunk]
        kv = kh * K
        R = 0.0
        for e in (eps, -eps):
            q = e * K * qh
            kb = -kv + q
            R = R + (_Z2(kv, -q, b1, b2, bs2, f) * _Z1(kv, b1, f) * p_dressed(kv)
                     + _Z2(kb, -q, b1, b2, bs2, f) * _Z1(kb, b1, f) * p_dressed(kb))
        R = R.mean(axis=2)                                                    # (mu, nu, k): +-eps and phi averages
        for l in ells:
            for n in ns:
                wgt = (2 * l + 1) / 2 * (2 * n + 1) / 2 * np.outer(wg * Lm[l], wg * Ln[n])
                out[(l, n)][s0:s0 + chunk] = np.einsum('ij,ijk->k', wgt, R)
    return out
