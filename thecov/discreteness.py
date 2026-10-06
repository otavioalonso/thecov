r"""Poisson (discreteness) non-Gaussian terms of the covariance of power-spectrum multipoles.

For a weighted galaxy field sampled by points, the connected four-point function of the counts contains,
besides the clustering trispectrum, terms in which galaxies coincide. The estimator subtracts its own
REALISED shot noise (sum_g w^2 + alpha^2 sum_r w^2, as pypower / jaxpower do), i.e. each P_hat is a sum over
DISTINCT pairs: coincidences inside one P_hat are excluded (no sum w^4 / n^3 constant, no triple terms).
What remains are coincidences between the two spectra:

  one shared galaxy (4 ways)          (1/n) 2 [B(k1, k2, -k1-k2) + B(k1, -k2, k2-k1)],   window int S m^2,
  two shared galaxies (2 ways)        (1/n^2) [P(|k1 + k2|) + P(|k1 - k2|)],             window int S^2,

with S = nbar <w^2> the data's own shot-noise density (randoms do not cluster with the data, so their
coincidences do not contribute) and m = nbar <w>. In the narrow-window (local) approximation, with the local
line of sight n^ for both spectra,

  C^{disc}_{l1 l2}(k1, k2) = (2 l1 + 1)(2 l2 + 1) < L_l1(k1^.n^) L_l2(k2^.n^)
        [ 2 J_Smm (B(k1, k2, -k1-k2) + B(k1, -k2, k2-k1)) + J_SS (P_s(k1+k2) + P_s(k1-k2)) ] >_{k1^, k2^, shells},
  J_Smm = int S m^2 / norm^2,   J_SS = int S^2 / norm^2,

which for a uniform window and unit weights is (1/V)[(2/n)(B + B) + (1/n^2)(P + P)] (Smith 2009; Meiksin &
White 1999). B and P_s are tree-level redshift-space (SCF99 Z1, Z2; b1, b2, bs2, f) from P_lin; the angular
average is done with n^ = z^ and k1^ at zero azimuth (Gauss-Legendre in mu1, mu2, uniform in the azimuth
difference), the shells with 2 Gauss-Legendre nodes in |k| per bin. The integrals use the local window (the
clustering in these terms is on scales ~ 1/k).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .ssc import _Z1, _Z2


class DiscretenessCovariance:
    """Poisson non-Gaussian terms for the set-up of a GaussianCovariance (auto-spectra of one tracer).

    Parameters: cov (its k bins, normalisation and tracers), p_lin = (k, P) linear matter power at z_eff,
    b1, f, b2 = 0, bs2 = -4/7 (b1 - 1); n_mu, n_phi, n_k: angular / shell quadrature; sigma_fog: Gaussian
    fingers-of-God damping [Mpc/h] of the galaxy fields (exp(-(k_z sigma)^2 / 2) each; raise n_mu, n_phi with it).
    For BAO damping pass an IR-damped p_lin (thecov.power.ir_damped).
    """

    def __init__(self, cov, p_lin, b1, f, b2=0.0, bs2=None, n_mu=12, n_phi=16, n_k=2, sigma_fog=0.0):
        self.cov = cov
        k, P = (np.asarray(x, float) for x in p_lin)
        self._spl = CubicSpline(np.log(k), P)
        self._kmin, self._kmax = k[0], k[-1]
        self.b1, self.f, self.b2 = float(b1), float(f), float(b2)
        self.bs2 = -4.0 / 7.0 * (self.b1 - 1.0) if bs2 is None else float(bs2)
        self.n_mu, self.n_phi, self.n_k = int(n_mu), int(n_phi), int(n_k)
        self.sigma_fog = float(sigma_fog)

    def P(self, k):
        k = np.asarray(k, float)
        out = np.zeros_like(k)
        ok = (k > self._kmin) & (k < self._kmax)
        out[ok] = self._spl(np.log(k[ok]))
        return out

    def window_integrals(self, A):
        """(J_Smm, J_SS) = (int S m^2, int S^2) / norm^2 from the host randoms (S = data's nbar <w^2>)."""
        T = self.cov._tracer(A)
        a, w, m = T.alpha, np.asarray(T.w, float), np.asarray(T.mw, float)
        # sum_g w^2 g(x_g) = alpha sum_r scale w_r^2 g(x_r): the density S = nbar <w^2> sampled by the randoms
        # as alpha * scale * w_r^2 / (random density) -> int S g = alpha sum_r scale w_r^2 g(x_r) / ... with
        # g = m^2: int S m^2 = alpha sum_r scale w_r^2 m_r^2 / ... (one random per random-density volume);
        # for S^2 the second factor is the local value S(x_r) = scale w_r m_r (nbar <w^2> = m <w^2>/<w>).
        sc = T.shotnoise_scale
        norm = self.cov.I(A, A)
        # int S g = alpha sum_r scale w_r^2 g(x_r) (the data's sum w^2 g, as for the shot-noise window);
        # the local value S(x) = scale nbar <w^2> = scale m <w^2>/<w> (global weight ratio, so that per-object
        # weight scatter does not turn <w^2>^2 into <w^3><w>)
        r = np.sum(w ** 2) / np.sum(w)
        J_Smm = a * np.sum(sc * w ** 2 * m ** 2) / norm ** 2
        J_SS = a * np.sum(sc * w ** 2 * sc * r * m) / norm ** 2
        return J_Smm, J_SS

    def _B(self, k1, k2):
        """tree-level redshift-space galaxy bispectrum B(k1, k2, -k1-k2); vectors (3, ...), n^ = z^"""
        b1, b2, bs2, f = self.b1, self.b2, self.bs2, self.f
        k3 = -k1 - k2
        m1, m2, m3 = (np.sqrt(np.sum(v * v, 0)) for v in (k1, k2, k3))
        P1, P2, P3 = self.P(m1), self.P(m2), self.P(m3)
        Z1 = lambda v: _Z1(v, b1, f)
        out = 2 * _Z2(k1, k2, b1, b2, bs2, f) * Z1(k1) * Z1(k2) * P1 * P2
        good = m3 > 1e-8 * np.maximum(m1, m2)                 # k3 -> 0: P(k3) -> 0 (and the k=0 mode is removed)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (2 * _Z2(k2, k3, b1, b2, bs2, f) * Z1(k2) * Z1(k3) * P2 * P3
                 + 2 * _Z2(k3, k1, b1, b2, bs2, f) * Z1(k3) * Z1(k1) * P3 * P1)
        B = out + np.where(good, np.nan_to_num(t), 0.0)
        if self.sigma_fog:                                     # exp(-(k_z sigma)^2 / 2) per field
            B = B * np.exp(-0.5 * self.sigma_fog ** 2 * (k1[2] ** 2 + k2[2] ** 2 + k3[2] ** 2))
        return B

    def _Ps(self, K):
        m = np.sqrt(np.sum(K * K, 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            z1 = np.where(m > 0, _Z1(K, self.b1, self.f), 0.0)
        out = z1 ** 2 * self.P(m)
        return out * np.exp(-(K[2] * self.sigma_fog) ** 2) if self.sigma_fog else out

    def covariance(self, spectra, ells=None):
        """C^disc for [P^{AA}_ell(k_i)] ordered by ell, bin (as GaussianCovariance.covariance)."""
        comps, index = self.components(spectra, ells=ells, return_index=True)
        return comps['B'] + comps['P'], index

    def components(self, spectra, ells=None, return_index=False):
        """{'B': one shared galaxy (bispectrum, 1/n), 'P': two shared galaxies (power spectrum, 1/n^2)}"""
        cov = self.cov
        ells = cov.ells if ells is None else tuple(ells)
        spectra = [tuple(str(x) for x in sp) for sp in spectra]
        if len(spectra) != 1 or spectra[0][0] != spectra[0][1]:
            raise NotImplementedError('discreteness terms: auto-spectrum of one tracer')
        A = spectra[0][0]
        J_Smm, J_SS = self.window_integrals(A)
        edges = cov.k_edges
        xk, wk = np.polynomial.legendre.leggauss(self.n_k)
        xm, wm = np.polynomial.legendre.leggauss(self.n_mu)
        phi = 2 * np.pi * (np.arange(self.n_phi) + 0.5) / self.n_phi
        nb = len(edges) - 1
        # shell nodes (k^2-weighted)
        kn, kw = [], []
        for i in range(nb):
            lo, hi = edges[i], edges[i + 1]
            kk = 0.5 * (hi - lo) * xk + 0.5 * (hi + lo)
            ww = 0.5 * (hi - lo) * wk * kk ** 2
            kn.append(kk)
            kw.append(ww / ww.sum())
        mu1, mu2, ph = np.meshgrid(xm, xm, phi, indexing='ij')
        s1, s2 = np.sqrt(1 - mu1 ** 2), np.sqrt(1 - mu2 ** 2)
        u1 = np.stack([s1, np.zeros_like(s1), mu1])
        u2 = np.stack([s2 * np.cos(ph), s2 * np.sin(ph), mu2])
        wang = (wm[:, None, None] * wm[None, :, None] / 4.0 / self.n_phi) * np.ones_like(mu1)   # <.> over k1^, k2^
        L = {l: (np.polynomial.legendre.Legendre.basis(l)(mu1), np.polynomial.legendre.Legendre.basis(l)(mu2)) for l in ells}
        out = {p: np.zeros((len(ells) * nb, len(ells) * nb)) for p in ('B', 'P')}
        for i in range(nb):
            for j in range(i, nb):
                acc = {(t, l1, l2): 0.0 for t in out for l1 in ells for l2 in ells}
                for a_, ka in zip(kw[i], kn[i]):
                    for b_, kb in zip(kw[j], kn[j]):
                        k1, k2 = ka * u1, kb * u2
                        F = {'B': 2 * J_Smm * (self._B(k1, k2) + self._B(k1, -k2)),
                             'P': J_SS * (self._Ps(k1 + k2) + self._Ps(k1 - k2))}
                        for t in out:
                            for l1 in ells:
                                for l2 in ells:
                                    acc[(t, l1, l2)] += (a_ * b_ * (2 * l1 + 1) * (2 * l2 + 1)
                                                         * np.sum(wang * L[l1][0] * L[l2][1] * F[t]))
                for t in out:
                    for p, l1 in enumerate(ells):
                        for q, l2 in enumerate(ells):
                            out[t][p * nb + i, q * nb + j] = acc[(t, l1, l2)]
                            out[t][q * nb + j, p * nb + i] = acc[(t, l1, l2)]
        if return_index:
            return out, [(A, A, l, i) for l in ells for i in range(nb)]
        return out
