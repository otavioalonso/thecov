r"""Super-sample covariance (SSC) of windowed power-spectrum multipoles in redshift space.

A long matter mode delta_L, larger than the scales the multipoles resolve, changes the local power
spectrum everywhere in the survey (beat coupling, BC) and the estimator's normalisation (local
average, LA). With a local line of sight n^ = x^, the long mode's effect on the multipole ell at
position x depends on q^ only through nu = q^.x^ after the average over k^:

    Delta P_ell(k; x) = sum_n R_ell^(n)(k) delta_n(x),
    delta_n(x) = int d^3q/(2 pi)^3 delta_L(q) e^{i q.x} L_n(q^.x^).

Responses (tree level, Scoccimarro, Couchman & Frieman 1999 kernels Z1, Z2 with b1, b2, bs2, f): the
squeezed limit
    R(k, k^; q^) = lim_{q->0} 2 [ Z2(k, -q) Z1(k) P(k) + Z2(-k + q, -q) Z1(k - q) P(|k - q|) ],
which contains growth, dilation (d P / d ln k), bias and redshift-space (Kaiser, velocity-gradient)
terms; the 1/q bulk-flow terms cancel. Projected: R_ell^(n)(k) = (2ell+1)/2 (2n+1)/2 int dmu dnu
L_ell(mu) L_n(nu) <R>_phi = a_{ell n} P(k) + c_{ell n} dP/dlnk. Real space: R_0^(0) = 47/21 - n_eff/3
and R_2^(2) = (2/3)(8/7 - n_eff) for b1 = 1 (isotropic and tidal responses).

Estimator (pypower / jaxpower conventions): F = w (n_g - alpha n_r), P_hat = |F|^2 / norm with
alpha = sum_g w / sum_r w and norm = alpha sum_cells D_c R_c / V_c computed for each realisation from
its own data. A long mode then gives, to first order,

    Delta P_hat_ell(k) = d_k sum_n R_ell^(n)(k) D^W_n  -  P_hat_ell(k) sum_n g_n (D^W_n + D^M_n),

    D^w_n = int w(x) delta_n(x) d^3x / int w,     W = m_A m_B (pairs),  M = m_A (alpha's average),
    g_0 = b1 + f/3, g_2 = 2 f / 3  (the observed long mode (b1 + f nu^2) delta_L in Legendre components),
    d_k = I_k / norm (the window's dilution of the measured spectrum; I_k = int m^2 without smoothing).
The first term is BC (weighted by the clustering window); the second is LA: alpha absorbs the m-weighted
mean density and the data in `norm` the m^2-weighted one (for a uniform window and an isotropic long
mode: R - 2, the usual local-average effect). Set local_average=False for a normalisation fixed in
advance. P_hat_ell is the measured (window-diluted) spectrum: the model itself when the covariance's
model is masked, model x d_k otherwise.

Long-mode variances from thecov's window pair counts (exact for the local line of sight):

    sigma^2_{(w n)(w' n')} = <D^w_n D^w'_n'>
       = (4 pi)^{3/2} / (I_w I_w') sum_lam (-1)^{lam/2} sqrt((2 lam + 1) / ((2n+1)(2n'+1))) (n n' lam; 0 0 0)
         int s^2 ds xi_lam(s) Q^{w w'}_{n n' lam}(s),
    xi_lam(s) = int q^2 dq / (2 pi^2) P_lin(q) j_lam(q s),

with Q the tripolar window function of windows.py (S_{n n' lam}(x^, x'^, s^), s = x' - x). For n = n' = 0
this is sigma_b^2 = int d^3s xi(s) Q_WW(s) / I^2. The covariance is

    C^SSC_{l1 l2}(i, j) = sum_{X, Y} c^X_{l1}(k_i) c^Y_{l2}(k_j) sigma^2_{XY},

with c^{(W, n)}_l = d_k R_l^(n) - P_hat_l g_n and c^{(M, n)}_l = - P_hat_l g_n.

Discreteness of the local average. alpha and norm are sums over galaxies, so they also fluctuate by
Poisson noise, eps_N = eps_alpha + eps_D with weights u_alpha = w / int m and u_D = w m / int m^2 per galaxy.
With P_hat = P_raw (1 - eps_N), the covariance gains
    - P_hat_l2 T_l1 - P_hat_l1 T_l2 + P_hat_l1 P_hat_l2 Var_P(eps_N),
    Var_P(eps_N) = sum_g w^2 (u_alpha + u_D)^2 / w^2,
    T_l(k) = <P_raw_l eps_N> = 2 P_l^true(k) [J_alpha + J_D] / norm + B_c,l(k) [J3_alpha + J3_D] / norm,
J_alpha = int nbar<w^2> m / int m, J_D = int nbar<w^2> m^2 / int m^2 (a galaxy counted in eps_N and in the
pair), J3_alpha = int m^3 / int m, J3_D = int m^4 / int m^2 (the collapsed tree-level bispectrum
B_c = 2 Z2(k, -k) Z1^2 P_lin^2, Z2(k, -k) = b2/2 + bs2/3). For a uniform window, unit weights and alpha
only this is WS20's -2/N, -2/N, +1/N (their eq. 44 discreteness terms); with jaxpower's norm, -8/N + 4/N.
Galaxy sums use the randoms: sum_g w^2 g = alpha sum_r scale w_r^2 g (scale = Tracer.shotnoise_scale).

Scope: auto-spectra of one tracer (cross spectra need the responses of two tracers; not implemented).
The local window m_A m_B is used for the long-mode shape (long modes do not resolve veto holes); its
normalisation enters through d_k = I_k / norm, so with a pair-averaged window (WindowSmoothing) the
amplitude is that of the smoothed window.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from .tracers import Window
from .windows import WindowLibrary
from .wigner import wigner_3j, tri, FOUR_PI

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


# ----------------------------------------------------------------------------- the covariance
class SuperSampleCovariance:
    """SSC of the multipoles of a GaussianCovariance's set-up (same tracers, k bins, window library).

    Parameters
    ----------
    cov    : GaussianCovariance (its windows, normalisation and model are used; set_model and, for
             DESI-like spectra, set_normalization first)
    p_lin  : (k, P) linear matter power spectrum at the effective redshift [(Mpc/h)^3], wide k range
             (e.g. 1e-4 - 10 h/Mpc)
    b1, f  : linear bias and growth rate; b2 (default 0) and bs2 (default -4/7 (b1 - 1), local
             Lagrangian) enter the responses
    local_average : include the LA term of a per-realisation alpha and norm (pypower / jaxpower)
    discreteness  : include the Poisson (and collapsed-bispectrum) part of the LA (needs local_average)
    damping: Gaussian damping [Mpc/h] of P_lin in xi_lam (regularises xi at s -> 0; irrelevant for SSC)
    n_mu, n_near : the SSC's own pair counts (same s bins, far-pair sampling and seed as cov). The
             long-mode variances weight the Lam = 4 component of Q_{22 Lam} more than the Gaussian
             terms do, and evaluating S at the cell-mean mu biases it at second order in the mu cell
             (-1% of Q_000 with 24 cells, i.e. -8% in sigma^2_22 for a sphere); 96 cells remove it.
             Small separations matter little for long modes, so fewer near pairs suffice.
    dilution: d_k = I_k / norm of the BC term, scalar or per k-bin (default cov.I_k / cov.I, i.e.
             int m^2 / norm for a local window: for footprints with fine veto masks pass the
             pair-averaged value, e.g. WindowSmoothing.I_k / norm)
    """

    def __init__(self, cov, p_lin, b1, f, b2=0.0, bs2=None, local_average=True, damping=1.0, n_mu=96,
                 n_near=300000, dilution=None, discreteness=True):
        self.cov = cov
        self.dilution = dilution
        opts = dict(cov.windows.opts)
        opts.update(n_mu=int(n_mu), n_near=min(int(n_near), int(opts['n_near'])))
        self.windows = WindowLibrary(cov.windows.s_edges, **opts)
        self.k_lin, self.p_lin = (np.asarray(x, float) for x in p_lin)
        self.b1, self.f, self.b2 = float(b1), float(f), float(b2)
        self.bs2 = -4.0 / 7.0 * (self.b1 - 1.0) if bs2 is None else float(bs2)
        self.local_average = bool(local_average)
        self.discreteness = bool(discreteness) and self.local_average
        self.damping = float(damping)
        self.a, self.c = response_coefficients(self.b1, self.f, self.b2, self.bs2)
        scale = np.max(np.abs(self.a)) + np.max(np.abs(self.c))
        self.ns_bc = tuple(n for j, n in enumerate(ELLS)                   # nu^4 vanishes (to O(eps^2))
                           if np.max(np.abs(self.a[:, j]) + np.abs(self.c[:, j])) > 1e-6 * scale)
        self.g = {0: self.b1 + self.f / 3.0, 2: 2.0 * self.f / 3.0}
        self._sigma2 = {}

    # -- windows
    def _windows(self, A):
        T = self.cov._tracer(A)
        return {'W': Window('W', T, T), 'M': Window('M', T)}

    def _modes(self):
        """the long-mode projections X = (window, n) that enter"""
        modes = [('W', n) for n in sorted(set(self.ns_bc) | (set(self.g) if self.local_average else set()))]
        if self.local_average:
            modes += [('M', n) for n in sorted(self.g)]
        return modes

    def request_windows(self, A):
        win = self._windows(A)
        modes = self._modes()
        for i, (x, n) in enumerate(modes):
            for (y, n2) in modes[i:]:
                self.windows.request(win[x], win[y], [(n, n2, lam) for lam in tri(n, n2)])
        return win

    def compute_windows(self, A, verbose=False):
        self.request_windows(A)
        if self.windows.missing():
            self.windows.compute_all(verbose=verbose)
        return self

    def save_windows(self, path):
        """the SSC's pair counts (geometry only; reusable for any cosmology and bias)"""
        self.windows.save(path)

    def load_windows(self, path):
        self.windows.load(path)
        return self

    # -- xi_lam
    def xi(self, lam, s):
        """xi_lam(s) = int q^2 dq / (2 pi^2) P_lin(q) j_lam(q s), on a linear q grid fine enough for s."""
        s = np.asarray(s, float)
        qmax = min(self.k_lin[-1], 5.0 / max(self.damping, 1e-3))
        dq = min(1e-3, 0.25 / s.max())
        q = np.arange(max(self.k_lin[0], 1e-5), qmax, dq)
        pq = np.exp(np.interp(np.log(q), np.log(self.k_lin), np.log(np.maximum(self.p_lin, 1e-300))))
        pq *= np.exp(-(q * self.damping) ** 2) * q ** 2 * dq / (2 * np.pi ** 2)
        out = np.empty(len(s))
        for i0 in range(0, len(s), 64):
            out[i0:i0 + 64] = spherical_jn(lam, np.outer(s[i0:i0 + 64], q)) @ pq
        return out

    # -- long-mode variances
    def sigma2(self, A):
        """{(X, Y): sigma^2_XY} for the projections X = (window, n) of _modes()."""
        if A in self._sigma2:
            return self._sigma2[A]
        win = self.compute_windows(A)._windows(A)
        s, sw = self.cov.s, self.cov.s_weights
        I = {name: w.integral() for name, w in win.items()}
        xis = {}
        out = {}
        modes = self._modes()
        for i, (x, n) in enumerate(modes):
            for (y, n2) in modes[i:]:
                tot = 0.0
                for lam in tri(n, n2):
                    w3 = wigner_3j(n, n2, lam, 0, 0, 0)
                    if w3 == 0.0:
                        continue
                    if lam not in xis:
                        xis[lam] = self.xi(lam, s)
                    Q = self.windows.get(win[x], win[y], n, n2, lam, s)
                    tot += ((-1) ** (lam // 2) * math.sqrt((2 * lam + 1) / ((2 * n + 1) * (2 * n2 + 1))) * w3
                            * np.sum(sw * xis[lam] * Q))
                val = FOUR_PI ** 1.5 * tot / (I[x] * I[y])
                out[((x, n), (y, n2))] = out[((y, n2), (x, n))] = val
        self._sigma2[A] = out
        return out

    # -- coefficients per k-bin
    def _bin_average(self, func):
        K = self.cov.kernels
        return np.sum(K.wq * func(K.kq), axis=1) / K.norm

    def responses(self):
        """{(ell, n): R_ell^(n) averaged over each k bin} (tree level, true-power units)."""
        lk = np.log(self.k_lin)
        spl = CubicSpline(lk, self.p_lin)
        P = lambda k: spl(np.log(k))
        dP = lambda k: spl(np.log(k), 1)
        Pb, dPb = self._bin_average(P), self._bin_average(dP)
        return {(l, n): self.a[i, j] * Pb + self.c[i, j] * dPb
                for i, l in enumerate(ELLS) for j, n in enumerate(ELLS)}

    def _dilution(self, A):
        if self.dilution is not None:
            return np.broadcast_to(np.asarray(self.dilution, float), (self.cov.nbins,)).copy()
        return np.asarray(self.cov.I_k(A, A), float) / self.cov.I(A, A)

    def coefficients(self, A, ells):
        """{X: (len(ells) * nbins,) vector c^X} in the data-vector order (ell, bin)."""
        cov = self.cov
        d = self._dilution(A)
        R = self.responses()
        Pmeas = {}
        for l in ells:
            Pm = self._bin_average(lambda k: cov.model(A, A, l, k))
            Pmeas[l] = Pm if cov.masked else Pm * d
        out = {}
        for (x, n) in self._modes():
            vec = []
            for l in ells:
                v = np.zeros(cov.nbins)
                if x == 'W' and (l, n) in R:
                    v += d * R[(l, n)]
                if self.local_average and n in self.g:
                    v -= Pmeas[l] * self.g[n]
                vec.append(v)
            out[(x, n)] = np.concatenate(vec)
        return out

    def discreteness_integrals(self, A):
        """Var_P(eps_N) and (J_alpha + J_D, J3_alpha + J3_D) from the host randoms (see module docstring)."""
        T = self.cov._tracer(A)
        a, w, m = T.alpha, np.asarray(T.w, float), np.asarray(T.mw, float)
        w2 = a * T.shotnoise_scale * w ** 2                    # sum_g w^2 g  ->  sum_r w2_r g(x_r)
        I_M, I_W = a * np.sum(w), a * np.sum(w * m)
        var = np.sum(w2 * (1.0 / I_M + m / I_W) ** 2)
        J = np.sum(w2 * m) / I_M + np.sum(w2 * m ** 2) / I_W
        J3 = a * np.sum(w * m ** 2) / I_M + a * np.sum(w * m ** 3) / I_W
        return var, J, J3

    def discreteness_covariance(self, A, ells):
        """the Poisson / collapsed-bispectrum part of the local average, (len(ells) nbins)^2"""
        cov = self.cov
        d = self._dilution(A)
        norm = cov.I(A, A)
        var, J, J3 = self.discreteness_integrals(A)
        lk = np.log(self.k_lin)
        spl = CubicSpline(lk, self.p_lin)
        P2 = self._bin_average(lambda k: spl(np.log(k)) ** 2)
        b1, f = self.b1, self.f
        kais = {0: b1 ** 2 + 2 / 3 * b1 * f + f ** 2 / 5, 2: 4 / 3 * b1 * f + 4 / 7 * f ** 2, 4: 8 / 35 * f ** 2}
        zc = self.b2 + 2.0 / 3.0 * self.bs2                    # 2 Z2(k, -k) = b2 + 2 bs2 / 3
        Pm, Tv = [], []
        for l in ells:
            p = self._bin_average(lambda k: cov.model(A, A, l, k))
            p = p if cov.masked else p * d
            Pm.append(p)
            Tv.append(2.0 * p / d * J / norm + zc * kais.get(l, 0.0) * P2 * J3 / norm)
        Pm, Tv = np.concatenate(Pm), np.concatenate(Tv)
        return -np.outer(Tv, Pm) - np.outer(Pm, Tv) + var * np.outer(Pm, Pm)

    def covariance(self, spectra, ells=None):
        """C^SSC for the data vector [P^{AA}_ell(k_i)] ordered by spectrum, ell, bin (as
        GaussianCovariance.covariance); returns (matrix, labels)."""
        cov = self.cov
        if cov.model is None:
            raise RuntimeError("set the covariance's model first (cov.set_model)")
        ells = cov.ells if ells is None else tuple(ells)
        spectra = [tuple(str(x) for x in sp) for sp in spectra]
        for sp in spectra:
            if sp[0] != sp[1]:
                raise NotImplementedError('SSC is implemented for auto-spectra only')
        if len(set(sp[0] for sp in spectra)) > 1:
            raise NotImplementedError('SSC is implemented for a single tracer')
        A = spectra[0][0]
        sig = self.sigma2(A)
        cf = self.coefficients(A, ells)
        X = list(cf)
        V = np.stack([cf[x] for x in X])                        # (nX, n)
        S = np.array([[sig[(x, y)] for y in X] for x in X])
        C = V.T @ S @ V
        if self.discreteness:
            C = C + self.discreteness_covariance(A, ells)
        labels = [(A, A, l, i) for l in ells for i in range(cov.nbins)]
        return C, labels

    def amplitude_scatter(self, A, ells=None):
        """Diagnostics: per (ell) the SSC standard deviation of P_ell at each bin in units of the
        measured P_0 (comparable with sqrt(A_ll) of desi_validation.rank1_excess)."""
        ells = self.cov.ells if ells is None else tuple(ells)
        C, _ = self.covariance([(A, A)], ells)
        nb = self.cov.nbins
        d = self._dilution(A)
        P0 = self._bin_average(lambda k: self.cov.model(A, A, 0, k))
        P0 = P0 if self.cov.masked else P0 * d
        return {l: np.sqrt(np.clip(np.diag(C)[i * nb:(i + 1) * nb], 0, None)) / np.abs(P0) for i, l in enumerate(ells)}
