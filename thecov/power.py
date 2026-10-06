r"""Linear power spectrum helpers for the non-Gaussian terms: no-wiggle spectrum, BAO (IR) damping and a
"dressed" redshift-space power with Gaussian fingers-of-God damping.

  P_nw   : Eisenstein & Hu (1998) zero-baryon shape times the Gaussian-smoothed (in ln k) ratio P / P_EH,
           so that the smoothing only acts on the (nearly flat) wiggle ratio and does not bias the broadband;
  P_IR   : P_nw + (P - P_nw) exp(-k^2 Sigma^2)   (leading-order IR resummation, isotropic Sigma);
  D_FoG  : exp(-(k mu sigma_v)^2) on the power (exp(-(k mu sigma_v)^2 / 2) per field).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def eh_nowiggle(k, h, omega_m, f_baryon, T_cmb=2.7255, n_s=1.0):
    """EH98 zero-baryon ('no-wiggle') shape T(k)^2 k^n_s (arbitrary normalisation); k in h/Mpc."""
    k = np.asarray(k, float)
    om = omega_m                                    # Omega_m h^2
    ob = f_baryon * om
    theta = T_cmb / 2.7
    s = 44.5 * np.log(9.83 / om) / np.sqrt(1 + 10 * ob ** 0.75)            # Mpc
    alpha = 1 - 0.328 * np.log(431 * om) * f_baryon + 0.38 * np.log(22.3 * om) * f_baryon ** 2
    kk = k * h                                                              # 1/Mpc
    gamma = om / h * (alpha + (1 - alpha) / (1 + (0.43 * kk * s) ** 4))     # Gamma_eff (h/Mpc units)
    q = k * theta ** 2 / gamma
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    T = L0 / (L0 + C0 * q ** 2)
    return T ** 2 * k ** n_s


def no_wiggle(k, P, h, omega_m, f_baryon, n_s=0.9649, width=0.25):
    """smooth (no-BAO) version of P on the grid k (log-spaced preferred): P_EH x smooth(P / P_EH)."""
    k, P = np.asarray(k, float), np.asarray(P, float)
    ref = eh_nowiggle(k, h, omega_m, f_baryon, n_s=n_s)
    lk = np.log(k)
    dl = np.median(np.diff(lk))
    ratio = np.log(P / ref)
    sm = gaussian_filter1d(ratio, width / dl, mode='nearest')
    return ref * np.exp(sm)


def ir_damped(k, P, P_nw, sigma):
    """P_nw + (P - P_nw) exp(-k^2 sigma^2)"""
    k = np.asarray(k, float)
    return P_nw + (P - P_nw) * np.exp(-(k * sigma) ** 2)


def fog(k_z, sigma_v):
    """Gaussian fingers-of-God damping of the power, exp(-(k_z sigma_v)^2)"""
    return np.exp(-(np.asarray(k_z) * sigma_v) ** 2)


class Dressed:
    """picklable callable P(v) = P(|v|) exp(-(v_z sigma_v)^2) for wavevectors v (3, ...) (n^ = z^); P from a
    table (cubic spline in ln k, 0 outside)."""

    def __init__(self, k, P, sigma_v=0.0):
        from scipy.interpolate import CubicSpline
        k, P = np.asarray(k, float), np.asarray(P, float)
        self._spl = CubicSpline(np.log(k), P)
        self.kmin, self.kmax = k[0], k[-1]
        self.sigma_v = float(sigma_v)

    def __call__(self, v):
        m = np.sqrt(np.sum(v * v, 0))
        out = np.zeros_like(m)
        ok = (m > self.kmin) & (m < self.kmax)
        out[ok] = self._spl(np.log(m[ok]))
        return out * fog(v[2], self.sigma_v) if self.sigma_v else out
