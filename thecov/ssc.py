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

    D^w_n = int w(x) delta_n(x) d^3x / int w,     W = W_A W_B (pairs),  M = W_A (alpha's average),
    g_0 = b1 + f/3, g_2 = 2 f / 3  (the observed long mode (b1 + f nu^2) delta_L in Legendre components),
    d_k = I_k / norm (the window's dilution of the measured spectrum; I_k = int W_A^2 without smoothing).
The first term is BC (weighted by the clustering window); the second is LA: alpha absorbs the W-weighted
mean density and the data in `norm` the W^2-weighted one (for a uniform window and an isotropic long
mode: R - 2, the usual local-average effect). Set local_average=False for a normalisation fixed in
advance. P_hat_ell is the measured (window-diluted) spectrum: the covariance's power spectrum itself when it
is window-convolved, the spectrum x d_k otherwise.

Long-mode variances from thecov's window pair counts (exact for the local line of sight):

    sigma^2_{(w n)(w' n')} = <D^w_n D^w'_n'>
       = (4 pi)^{3/2} / (I_w I_w') sum_lam (-1)^{lam/2} sqrt((2 lam + 1) / ((2n+1)(2n'+1))) (n n' lam; 0 0 0)
         int s^2 ds xi_lam(s) Q^{w w'}_{n n' lam}(s),
    xi_lam(s) = int q^2 dq / (2 pi^2) P_lin(q) j_lam(q s),

with Q the tripolar window function of pair_counts.py (S_{n n' lam}(x^, x'^, s^), s = x' - x). For n = n' = 0
this is sigma_b^2 = int d^3s xi(s) Q_WW(s) / I^2. The covariance is

    C^SSC_{l1 l2}(i, j) = sum_{X, Y} c^X_{l1}(k_i) c^Y_{l2}(k_j) sigma^2_{XY},

with c^{(W, n)}_l = d_k R_l^(n) - P_hat_l g_n and c^{(M, n)}_l = - P_hat_l g_n.

Discreteness of the local average. alpha and norm are sums over galaxies, so they also fluctuate by
Poisson noise, eps_N = eps_alpha + eps_D with weights u_alpha = w / int W and u_D = w W / int W^2 per galaxy.
With P_hat = P_raw (1 - eps_N), the covariance gains
    - P_hat_l2 T_l1 - P_hat_l1 T_l2 + P_hat_l1 P_hat_l2 Var_P(eps_N),
    Var_P(eps_N) = sum_g w^2 (u_alpha + u_D)^2 / w^2,
    T_l(k) = <P_raw_l eps_N> = 2 P_l^true(k) [J_alpha + J_D] / norm + B_c,l(k) [J3_alpha + J3_D] / norm,
J_alpha = int nbar<w^2> W / int W, J_D = int nbar<w^2> W^2 / int W^2 (a galaxy counted in eps_N and in the
pair), J3_alpha = int W^3 / int W, J3_D = int W^4 / int W^2 (the collapsed tree-level bispectrum
B_c = 2 Z2(k, -k) Z1^2 P_lin^2, Z2(k, -k) = b2/2 + bs2/3). For a uniform window, unit weights and alpha
only this is WS20's -2/N, -2/N, +1/N (their eq. 44 discreteness terms); with jaxpower's norm, -8/N + 4/N.
Galaxy sums use the randoms: sum_g w^2 g = alpha sum_r scale w_r^2 g (scale = GaussianCovariance.shotnoise_scale).

Scope: auto-spectra of one tracer (cross spectra need the responses of two tracers; not implemented).
The local window W_A W_B is used for the long-mode shape (long modes do not resolve veto holes); its
normalisation enters through d_k = I_k / norm, so with a pair-averaged window (WindowSmoothing) the
amplitude is that of the smoothed window.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from . import defaults as D
from .tracers import Window
from .pair_counts import WindowLibrary
from .wigner import wigner_3j, tri, FOUR_PI
from .tree.responses import ELLS, response_coefficients, response_multipoles


# ----------------------------------------------------------------------------- the covariance
class SuperSampleCovariance:
    """SSC of the multipoles of a GaussianCovariance's set-up (same tracers, k bins, window library).

    Parameters
    ----------
    cov    : GaussianCovariance (its windows, normalisation and power spectrum are used: set its power first, with
             the estimator's norm for DESI-like spectra)
    p_lin  : (k, P) linear matter power spectrum at the effective redshift [(Mpc/h)^3], wide k range
             (e.g. 1e-4 - 10 h/Mpc)
    b1, f  : linear bias and growth rate; b2 (default 0) and bs2 (default -4/7 (b1 - 1), local
             Lagrangian) enter the responses
    local_average : include the LA term of a per-realisation alpha and norm (pypower / jaxpower)
    norm_kind : which averages of the long mode the realised normalisation contains: 'data-randoms'
             (norm = alpha sum_c D_c R_c: alpha's W-weighted mean AND the data's W^2-weighted mean, the
             module docstring), 'randoms' (norm = alpha^2 sum_c R_c^2: alpha twice), 'alpha' (norm = alpha x a
             fixed randoms integral: alpha once). The net P_0 response changes sign between the first and the
             last for b1 ~ 2 (BC - 2 g_0 P vs BC - g_0 P); `local_average_statistics` gives the sigma(delta_norm)
             and the P_hat-delta_norm regression slope that each implies, to be compared with the mocks.
    discreteness  : include the Poisson (and collapsed-bispectrum) part of the LA (needs local_average)
    damping: Gaussian damping [Mpc/h] of P_lin in xi_lam (regularises xi at s -> 0; irrelevant for SSC)
    n_mu, n_near : the SSC's own pair counts (same s bins, far-pair sampling and seed as cov). The
             long-mode variances weight the Lam = 4 component of Q_{22 Lam} more than the Gaussian
             terms do, and evaluating S at the cell-mean mu biases it at second order in the mu cell
             (-1% of Q_000 with 24 cells, i.e. -8% in sigma^2_22 for a sphere); 96 cells remove it.
             Small separations matter little for long modes, so fewer near pairs suffice.
    p_dressed: optional callable P(v) of a wavevector v (3, ...), n^ = z^: the responses are then computed from
             this dressed power (e.g. thecov.tree.power.Dressed: IR-damped P_lin and fingers-of-God damping)
             instead of the tree-level a P_lin + c dP_lin/dlnk
    dilution: d_k = I_k / norm of the BC term, scalar or per k-bin (default cov.I_k / cov.I, i.e.
             int W^2 / norm for a local window: for footprints with fine veto masks pass the
             pair-averaged value, e.g. WindowSmoothing.I_k / norm)
    """

    # how many times the W-weighted (M, alpha) and W^2-weighted (W, the data in norm) means of the long mode enter
    # ln norm: delta_norm = sum_n g_n (lam_W D^W_n + lam_M D^M_n) + eps_N
    NORM_KINDS = {'data-randoms': {'M': 1.0, 'W': 1.0},   # norm = alpha sum_c D_c R_c: alpha and the data realised
                  'randoms': {'M': 2.0, 'W': 0.0},        # norm = alpha^2 sum_c R_c^2: realised through alpha only, twice
                  'alpha': {'M': 1.0, 'W': 0.0}}          # norm = alpha x (fixed randoms integral, e.g. sum_r nbar w^2)

    def __init__(self, cov, p_lin, b1, f, b2=0.0, bs2=None, local_average=True, damping=1.0, n_mu=D.SSC_N_MU,
                 n_near=D.SSC_N_NEAR, dilution=None, discreteness=True, p_dressed=None, norm_kind='data-randoms'):
        if norm_kind not in self.NORM_KINDS:
            raise ValueError(f'norm_kind must be one of {sorted(self.NORM_KINDS)}')
        self.cov, self.p_dressed, self.dilution, self.norm_kind = cov, p_dressed, dilution, norm_kind
        opts = dict(cov.windows.opts)
        opts.update(n_mu=int(n_mu), n_near=min(int(n_near), int(opts['n_near'])))
        self.windows = WindowLibrary(cov.windows.s_edges, **opts)
        self.k_lin, self.p_lin = (np.asarray(x, float) for x in p_lin)
        self.b1, self.f, self.b2 = float(b1), float(f), float(b2)
        self.bs2 = -4.0 / 7.0 * (self.b1 - 1.0) if bs2 is None else float(bs2)
        self.local_average = bool(local_average)
        self.discreteness = bool(discreteness) and self.local_average
        # The convention of the estimator's norm decides the sign of the net P_0 response (see
        # local_average_statistics); it must be checked on the mocks.
        self.lam = dict(self.NORM_KINDS[norm_kind]) if self.local_average else {'M': 0.0, 'W': 0.0}
        self.damping = float(damping)
        self.a, self.c = response_coefficients(self.b1, self.f, self.b2, self.bs2)    # R = a P + c dP/dlnk (tree level)
        scale = np.max(np.abs(self.a)) + np.max(np.abs(self.c))
        self.ns_bc = tuple(n for j, n in enumerate(ELLS)                   # nu^4 vanishes (to O(eps^2))
                           if np.max(np.abs(self.a[:, j]) + np.abs(self.c[:, j])) > 1e-6 * scale)
        self.g = {0: self.b1 + self.f / 3.0, 2: 2.0 * self.f / 3.0}         # observed long mode, Legendre components
        self._sigma2 = {}

    # -- long-mode projections X = (window, n) and their windows
    def _long_mode_windows(self, A):
        T = self.cov._tracer(A)
        return {'W': Window('W', T, T), 'M': Window('M', T)}

    def _projections(self):
        """the long-mode projections X = (window, n) that enter: (W, n) for beat coupling and the data's share of the
        local average, (M, n) for alpha's"""
        ns_W = sorted(set(self.ns_bc) | (set(self.g) if self.lam['W'] else set()))
        return [('W', n) for n in ns_W] + ([('M', n) for n in sorted(self.g)] if self.lam['M'] else [])

    def _projection_pairs(self):
        X = self._projections()
        return [(x, y) for i, x in enumerate(X) for y in X[i:]]

    def request_windows(self, A):
        windows = self._long_mode_windows(A)
        for (x, n), (y, n2) in self._projection_pairs():
            self.windows.request(windows[x], windows[y], [(n, n2, lam) for lam in tri(n, n2)])
        return windows

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

    # -- xi_lam and the long-mode variances
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

    def sigma2(self, A):
        """{(X, Y): sigma^2_XY} for the projections X = (window, n) (both orders)."""
        if A not in self._sigma2:
            windows = self.compute_windows(A)._long_mode_windows(A)
            s, s_weights = self.cov.s, self.cov.s_weights
            integral = {name: w.integral() for name, w in windows.items()}
            xi = {}
            out = {}
            for (x, n), (y, n2) in self._projection_pairs():
                total = 0.0
                for lam in tri(n, n2):
                    w3 = wigner_3j(n, n2, lam, 0, 0, 0)
                    if w3 == 0.0:
                        continue
                    if lam not in xi:
                        xi[lam] = self.xi(lam, s)
                    Q = self.windows.get(windows[x], windows[y], n, n2, lam, s)
                    total += ((-1) ** (lam // 2) * math.sqrt((2 * lam + 1) / ((2 * n + 1) * (2 * n2 + 1))) * w3
                              * np.sum(s_weights * xi[lam] * Q))
                out[((x, n), (y, n2))] = out[((y, n2), (x, n))] = FOUR_PI ** 1.5 * total / (integral[x] * integral[y])
            self._sigma2[A] = out
        return self._sigma2[A]

    # -- per-k-bin quantities
    def _bin_average(self, func):
        return self.cov.hankel.bin_average(func)

    def _linear_power(self):
        spl = CubicSpline(np.log(self.k_lin), self.p_lin)
        return (lambda k: spl(np.log(k))), (lambda k: spl(np.log(k), 1))

    def responses(self):
        """{(ell, n): R_ell^(n) averaged over each k bin} (tree level, true-power units); with p_dressed, from the
        dressed (damped) power at each k node (response_multipoles)."""
        if self.p_dressed is not None:
            K = self.cov.hankel
            kq = np.asarray(K.kq, float)
            R = response_multipoles(kq.ravel(), self.p_dressed, self.b1, self.f, self.b2, self.bs2)
            return {key: np.sum(K.wq * v.reshape(kq.shape), axis=1) / K.norm for key, v in R.items()}
        P, dP = self._linear_power()
        Pb, dPb = self._bin_average(P), self._bin_average(dP)
        return {(l, n): self.a[i, j] * Pb + self.c[i, j] * dPb for i, l in enumerate(ELLS) for j, n in enumerate(ELLS)}

    def _dilution(self, A):
        """d_k = I_k / norm per bin (the `dilution` given, or the covariance's I_k / I)."""
        if self.dilution is not None:
            return np.broadcast_to(np.asarray(self.dilution, float), (self.cov.nbins,)).copy()
        return np.asarray(self.cov.I_k(A, A), float) / self.cov.I(A, A)

    def _measured(self, A, l):
        """P_hat_l per bin: the window-diluted spectrum the estimator measures (the window-convolved spectrum itself,
        or the spectrum x d_k)."""
        power = self.cov.power_of(A, A)
        p = power.binned(l, s=self.cov.s)
        return p if power.window_convolved else p * self._dilution(A)

    def coefficients(self, A, ells):
        """{X: (len(ells) * nbins,) vector c^X} in the data-vector order (ell, bin):
        c^{(W, n)}_l = d_k R_l^(n) - lam_W P_hat_l g_n,  c^{(M, n)}_l = - lam_M P_hat_l g_n."""
        d, R = self._dilution(A), self.responses()
        measured = {l: self._measured(A, l) for l in ells}

        def one(x, n, l):
            v = np.zeros(self.cov.nbins)
            if x == 'W' and (l, n) in R:
                v += d * R[(l, n)]
            if self.lam[x] and n in self.g:
                v -= self.lam[x] * measured[l] * self.g[n]
            return v
        return {(x, n): np.concatenate([one(x, n, l) for l in ells]) for (x, n) in self._projections()}

    # -- Poisson noise of the local average
    def discreteness_integrals(self, A):
        """Var_P(eps_N) and (J_alpha + J_D, J3_alpha + J3_D) from the host randoms (see module docstring)."""
        T = self.cov._tracer(A)
        a, w, W = T.alpha, np.asarray(T.weights, float), np.asarray(T.weighted_density, float)
        w2 = a * self.cov.shotnoise_scale(A) * w ** 2                    # sum_g w^2 g  ->  sum_r w2_r g(x_r)
        I_M, I_W = a * np.sum(w), a * np.sum(w * W)
        lM, lW = self.lam['M'], self.lam['W']
        var = np.sum(w2 * (lM / I_M + lW * W / I_W) ** 2)
        J = lM * np.sum(w2 * W) / I_M + lW * np.sum(w2 * W ** 2) / I_W
        J3 = lM * a * np.sum(w * W ** 2) / I_M + lW * a * np.sum(w * W ** 3) / I_W
        return var, J, J3

    def _poisson_terms(self, A, ells):
        """(P_hat, T, Var_P(eps_N)): measured multipoles and T_l = <P_raw_l eps_N> as (len(ells) nbins,) vectors."""
        d, norm = self._dilution(A), self.cov.I(A, A)
        var, J, J3 = self.discreteness_integrals(A)
        P, _ = self._linear_power()
        P2 = self._bin_average(lambda k: P(k) ** 2)
        b1, f = self.b1, self.f
        kaiser = {0: b1 ** 2 + 2 / 3 * b1 * f + f ** 2 / 5, 2: 4 / 3 * b1 * f + 4 / 7 * f ** 2, 4: 8 / 35 * f ** 2}
        collapsed = self.b2 + 2.0 / 3.0 * self.bs2                    # 2 Z2(k, -k) = b2 + 2 bs2 / 3
        measured = [self._measured(A, l) for l in ells]
        T = [2.0 * p / d * J / norm + collapsed * kaiser.get(l, 0.0) * P2 * J3 / norm for p, l in zip(measured, ells)]
        return np.concatenate(measured), np.concatenate(T), var

    def discreteness_covariance(self, A, ells):
        """the Poisson / collapsed-bispectrum part of the local average, (len(ells) nbins)^2"""
        Pm, T, var = self._poisson_terms(A, ells)
        return -np.outer(T, Pm) - np.outer(Pm, T) + var * np.outer(Pm, Pm)

    # -- results
    def covariance(self, spectra, ells=None):
        """C^SSC for the data vector [P^{AA}_ell(k_i)] ordered by spectrum, ell, bin (as GaussianCovariance.covariance);
        returns (matrix, labels)."""
        cov = self.cov
        if not cov.power:
            raise RuntimeError("set the covariance's power spectrum first (cov.set_power)")
        ells = cov.ells if ells is None else tuple(ells)
        spectra = [tuple(str(x) for x in sp) for sp in spectra]
        if any(sp[0] != sp[1] for sp in spectra):
            raise NotImplementedError('SSC is implemented for auto-spectra only')
        if len(set(sp[0] for sp in spectra)) > 1:
            raise NotImplementedError('SSC is implemented for a single tracer')
        A = spectra[0][0]
        sigma2, coefficients = self.sigma2(A), self.coefficients(A, ells)
        X = list(coefficients)
        V = np.stack([coefficients[x] for x in X])                        # (nX, n)
        C = V.T @ np.array([[sigma2[(x, y)] for y in X] for x in X]) @ V
        if self.discreteness:
            C = C + self.discreteness_covariance(A, ells)
        return C, [(A, A, l, i) for l in ells for i in range(cov.nbins)]

    # -- diagnostics
    def local_average_statistics(self, A, ells=None, C_total=None):
        """What the per-realisation normalisation implies, for a direct test on mocks (no fitting):
        delta_norm = norm / <norm> - 1 = sum_n g_n (lam_W D^W_n + lam_M D^M_n) + eps_N, so
            Var(delta_norm) = sum_XY lam_X lam_Y g_X g_Y sigma^2_XY + Var_P(eps_N),
            Cov(P_hat_l(k), delta_norm) = sum_XY c^X_l(k) lam_Y g_Y sigma^2_XY + T_l(k) - P_hat_l(k) Var_P(eps_N).
        Returns dict(sigma_norm, sigma_norm_clustering, sigma_norm_poisson, cov, slope, corr, P_hat) with cov the
        (len(ells) nbins,) covariance, slope = cov / Var(delta_norm) / P_hat (the regression slope of P_hat_l on
        delta_norm in units of the measured multipole), corr the correlation of the long-mode (clustering) parts of
        P_hat_l(k) and delta_norm (|corr| <= 1; ~ +-1 when one long-mode component dominates both) and, with C_total
        (the full covariance of the data vector, Gaussian + non-Gaussian), corr_total = cov / sqrt(C_total_ii Var).
        On the mocks: std(norm) / mean(norm), and the regression of each P_hat_l(k) on delta_norm, are the same
        quantities; they fix the normalisation convention (`norm_kind`) and test the long-mode variances and the
        net (beat coupling minus local average) response at once. The Poisson parts (Var_P, T_l) are included whatever
        `discreteness` is, since the realised norm carries them."""
        ells = self.cov.ells if ells is None else tuple(ells)
        if not self.local_average:
            raise RuntimeError('local_average_statistics needs local_average=True')
        sigma2, coefficients = self.sigma2(A), self.coefficients(A, ells)
        weight = {x: self.lam[x[0]] * self.g.get(x[1], 0.0) for x in coefficients}       # the weight of D^X in delta_norm
        pairs = [(x, y) for x in coefficients for y in coefficients]
        var_clustering = sum(weight[x] * weight[y] * sigma2[(x, y)] for x, y in pairs)
        var_bin = sum(coefficients[x] * coefficients[y] * sigma2[(x, y)] for x, y in pairs)   # clustering SSC per bin
        cov_clustering = sum(coefficients[x] * weight[y] * sigma2[(x, y)] for x, y in pairs)
        Pm, T, var_poisson = self._poisson_terms(A, ells)
        var = var_clustering + var_poisson
        cov_total = cov_clustering + T - Pm * var_poisson
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.where(var_bin * var_clustering > 0, cov_clustering / np.sqrt(np.abs(var_bin * var_clustering)), 0.0)
        out = dict(sigma_norm=math.sqrt(var), sigma_norm_clustering=math.sqrt(max(var_clustering, 0.0)),
                   sigma_norm_poisson=math.sqrt(var_poisson), cov=cov_total, slope=cov_total / var / Pm, corr=corr, P_hat=Pm)
        if C_total is not None:
            out['corr_total'] = cov_total / np.sqrt(np.diag(np.asarray(C_total, float)) * var)
        return out

    def amplitude_scatter(self, A, ells=None):
        """Diagnostics: per (ell) the SSC standard deviation of P_ell at each bin in units of the measured P_0
        (comparable with sqrt(A_ll) of desi_validation.rank1_excess)."""
        ells = self.cov.ells if ells is None else tuple(ells)
        C, _ = self.covariance([(A, A)], ells)
        nb = self.cov.nbins
        P0 = self._measured(A, 0)
        return {l: np.sqrt(np.clip(np.diag(C)[i * nb:(i + 1) * nb], 0, None)) / np.abs(P0) for i, l in enumerate(ells)}
