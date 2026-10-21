"""A power spectrum P^{AB}_L(k) with its multipoles and k-binning

    power = PowerSpectrum({0: (k, P0), 2: (k, P2), 4: (k, P4)}, kedges=kedges)
    power = PowerSpectrum({...}, kedges=kedges, window_convolved=True)

Entries are arrays (k, P) or callables P(k). Multipoles not provided are zero.
Several tracers: one PowerSpectrum per pair, {('A', 'A'): ..., ('A', 'B'): ...}.

What the covariance uses:
    power.binned(L)                  P_L per k bin: the given values themselves for binned input,
                                     otherwise the k^2-weighted bin average
    power.hankel(s).transform(...)   int_i k^2 dk P_L(k) j_lam(k s) / int_i k^2 dk
    power(k, L)                      P_L at arbitrary k
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn


EXTRAPOLATIONS = (False, True, 'zero', 'power-law')
POWER_LAW_FIT_POINTS = 8          # points at each end used to fit an unspecified power-law slope


def fitted_slope(k, P):
    """d ln|P| / d ln k by a least-squares line through the given points (k > 0)."""
    ok = k > 0
    return float(np.polyfit(np.log(k[ok]), np.log(np.abs(P[ok]) + 1e-300), 1)[0])


def interpolate(k, P, in_log_k=False, extrapolate=False, derivative=0, low_slope=None, high_slope=None):
    """P(k) between the given points by a cubic spline, in k or in ln k (in_log_k, for spectra over decades such as the
    linear power); derivative=1 gives dP/dk, or dP/dlnk with in_log_k. Beyond the given k (extrapolate): False raises,
    'zero' gives zero, True continues the spline, 'power-law' continues P(k) ~ k^n from the end values, with
    n = low_slope / high_slope at the two ends (None: fitted to the POWER_LAW_FIT_POINTS end points)."""
    k, P = np.asarray(k, float), np.asarray(P, float)
    x = np.log(k) if in_log_k else k
    spl = CubicSpline(x, P, extrapolate=extrapolate is True)
    if extrapolate == 'power-law':
        n = POWER_LAW_FIT_POINTS
        low = fitted_slope(k[:n], P[:n]) if low_slope is None else float(low_slope)
        high = fitted_slope(k[-n:], P[-n:]) if high_slope is None else float(high_slope)

    def P_at(kk):
        kk = np.asarray(kk, float)
        out = spl(np.log(kk) if in_log_k else kk, derivative)
        outside_range = np.isnan(out)
        if np.any(outside_range):
            if extrapolate is False:
                raise ValueError(f"P requested outside its k range [{k[0]:.4g}, {k[-1]:.4g}]")
            if extrapolate == 'power-law':
                for edge, (k_end, P_end, slope) in ((kk < k[0], (k[0], P[0], low)), (kk > k[-1], (k[-1], P[-1], high))):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        value = P_end * (kk[edge] / k_end) ** slope
                    out[edge] = value if derivative == 0 else slope * value / (1.0 if in_log_k else kk[edge])
            else:
                out = np.where(outside_range, 0.0, out)
        return out
    return P_at


def as_function(entry, **options):
    return entry if callable(entry) else interpolate(*entry, **options)


class PowerSpectrum:
    """Multipoles P_L(k) of one power spectrum (arrays or callables).

    kedges           : the k binning of the data vector
    window_convolved : the multipoles are measured ones (window-convolved, normalised by the estimator's norm)
    norm             : the estimator's normalisation (P_hat = |F|^2 / norm); None: the FKP integral over the randoms
    shotnoise        : the estimator's shot-noise numerator, int S = sum_g w_g^2 + alpha^2 sum_r w_r^2 (pypower /
                       jaxpower `num_shotnoise`, so P_shot = shotnoise / norm); auto spectra only. None: estimated
                       from the randoms
    ells             : the multipoles of the data vector (default: the multipoles given)
    in_log_k         : interpolate arrays in ln k (wide-range spectra, e.g. the linear power)
    extrapolate      : beyond the k range of the arrays, False (default) raises, 'zero' gives zero, True continues
                       the spline, 'power-law' continues P ~ k^n with n = low_slope / high_slope at the two ends
                       (None: fitted to the end points)
    """

    def __init__(self, multipoles, kedges=None, window_convolved=False, norm=None, shotnoise=None, ells=None,
                 in_log_k=False, extrapolate=False, low_slope=None, high_slope=None):
        self._entries = {int(L): self._checked(entry) for L, entry in multipoles.items()}
        self.kedges = None if kedges is None else np.asarray(kedges, dtype=float)
        self.window_convolved = bool(window_convolved)
        self.norm = None if norm is None else float(norm)
        self.shotnoise = None if shotnoise is None else float(shotnoise)
        self._ells = None if ells is None else tuple(int(ell) for ell in ells)
        if not any(extrapolate is e or (isinstance(e, str) and extrapolate == e) for e in EXTRAPOLATIONS):
            raise ValueError(f"extrapolate must be one of {EXTRAPOLATIONS}")
        self._interpolation = dict(in_log_k=bool(in_log_k), extrapolate=extrapolate, low_slope=low_slope,
                                   high_slope=high_slope)
        self._hankel = {}
        self._functions = {}

    @classmethod
    def from_estimator(cls, spectra, ells=(0, 2, 4)) -> 'PowerSpectrum':
        """A measured power spectrum from estimator output (lsstypes power-spectrum multipoles, or files to read with
        lsstypes; already selected in k): one realisation (e.g. the data) as it is, or several (e.g. mocks) averaged,
        together with norm and num_shotnoise. Window-convolved, with the estimator's binning. The values are placed at
        kmid, so binned(L) returns them as they are; the arrays are extended flat to k = 0 and to the last
        edge for the in-bin Hankel integrals."""
        import lsstypes
        spectra = [spectra] if isinstance(spectra, (str, bytes)) or not isinstance(spectra, (list, tuple)) else spectra
        spectra = [lsstypes.read(x) if isinstance(x, (str, bytes)) or hasattr(x, '__fspath__') else x for x in spectra]
        ells = tuple(int(ell) for ell in ells)
        first = spectra[0].get(ells[0])
        e = np.asarray(first.edges('k'))
        kedges = np.concatenate([e[:, 0], e[-1:, 1]])
        kmid = 0.5 * (kedges[1:] + kedges[:-1])
        k_ext = np.concatenate([[0.0], kmid, [kedges[-1]]])
        multipoles = {}
        for ell in ells:
            P = np.mean([np.real(np.asarray(sp.get(ell).value())) for sp in spectra], axis=0)
            multipoles[ell] = (k_ext, np.concatenate([P[:1], P, P[-1:]]))

        def mean_of(name):
            return float(np.mean([np.ravel(sp.get(ells[0]).values(name))[0] for sp in spectra]))
        return cls(multipoles, kedges=kedges, window_convolved=True, norm=mean_of('norm'),
                   shotnoise=mean_of('num_shotnoise'), ells=ells)

    @staticmethod
    def _checked(entry):
        if callable(entry):
            return entry
        k, P = np.asarray(entry[0], dtype=float), np.asarray(entry[1], dtype=float)
        if np.any(np.diff(k) <= 0):
            raise ValueError("k must be strictly increasing")
        return k, P

    def _copy(self, **changes):
        current = dict(kedges=self.kedges, window_convolved=self.window_convolved, norm=self.norm,
                       shotnoise=self.shotnoise, ells=self._ells, **self._interpolation)
        return PowerSpectrum(self._entries, **{**current, **changes})

    def windowed(self, window_convolved=True) -> 'PowerSpectrum':
        """The same multipoles, flagged as window-convolved (measured, normalised like the estimator) or not."""
        return self._copy(window_convolved=bool(window_convolved))

    def binned_on(self, kedges) -> 'PowerSpectrum':
        """The same multipoles on the k binning kedges."""
        return self._copy(kedges=np.asarray(kedges, dtype=float))

    def extrapolated(self, extrapolate='power-law', low_slope=None, high_slope=None) -> 'PowerSpectrum':
        """The same multipoles, continued beyond the k range of their arrays as `extrapolate` says."""
        return self._copy(extrapolate=extrapolate, low_slope=low_slope, high_slope=high_slope)

    @property
    def extrapolation(self):
        return self._interpolation['extrapolate']

    # -- access
    @property
    def nbins(self) -> int:
        return len(self.kedges) - 1

    @property
    def kmid(self) -> np.ndarray:
        return 0.5 * (self.kedges[1:] + self.kedges[:-1])

    def multipoles(self):
        """The multipoles given (the L of the covariance's input power)."""
        return sorted(self._entries)

    @property
    def ells(self) -> tuple:
        """The multipoles of the data vector (default: the multipoles given)."""
        return tuple(self.multipoles()) if self._ells is None else self._ells

    def entry(self, L):
        """The multipole as given (arrays or callable), or None if not provided."""
        return self._entries.get(int(L))

    def __call__(self, k, L=0):
        """P_L(k) at arbitrary k (zero for multipoles not provided)."""
        return self.function(L)(k)

    def derivative(self, k, L=0):
        """dP_L/dlnk at k (spline derivative for arrays; centred difference in ln k for callables)."""
        entry, k = self.entry(L), np.asarray(k, dtype=float)
        if entry is None:
            return np.zeros_like(k)
        if callable(entry):
            h = 1e-4
            return (entry(k * np.exp(h)) - entry(k * np.exp(-h))) / (2 * h)
        dP = interpolate(*entry, **self._interpolation, derivative=1)(k)
        return dP if self._interpolation['in_log_k'] else k * dP

    def function(self, L):
        """P_L as a callable of k (zero for multipoles not provided), interpolated as set for this spectrum (cached)."""
        L = int(L)
        if L not in self._functions:
            entry = self.entry(L)
            self._functions[L] = ((lambda k: np.zeros_like(np.asarray(k, dtype=float))) if entry is None
                                  else as_function(entry, **self._interpolation))
        return self._functions[L]

    # -- binned quantities
    def hankel(self, s) -> 'BinnedHankel':
        """The bin-averaged Hankel transforms of this binning for the separation grid s (cached)."""
        if self.kedges is None:
            raise ValueError("the power spectrum has no k binning (kedges)")
        key = (len(s), float(s[0]), float(s[-1]))
        if key not in self._hankel:
            self._hankel[key] = BinnedHankel(self.kedges, s)
        return self._hankel[key]

    def binned(self, L, s=None) -> np.ndarray:
        """P_L per k bin: the given values themselves when the arrays are given at the bin kmid (binned spectra,
        e.g. the measured data or the mean of mocks), otherwise the k^2-weighted bin average (quadrature of hankel(s); s only sets the
        number of nodes)."""
        entry = self.entry(L)
        if entry is None:
            return np.zeros(self.nbins)
        if not callable(entry):
            k, P = entry
            kmid = self.kmid
            idx = np.clip(np.searchsorted(k, kmid), 0, len(k) - 1)
            below = np.maximum(idx - 1, 0)
            idx = np.where(np.abs(k[below] - kmid) < np.abs(k[idx] - kmid), below, idx)      # nearest given k
            if np.all(np.abs(k[idx] - kmid) <= 1e-9 * np.maximum(kmid, 1e-12)):
                return P[idx].copy()
        return self.hankel(np.array([1.0]) if s is None else s).bin_average(self.function(L))


class BinnedHankel:
    """Hankel transforms averaged over k bins, pbar^{(i)}_lam(s) = int_i k^2 dk p(k) j_lam(k s) / int_i k^2 dk, on a grid
    of s, by Gauss-Legendre quadrature in |k| inside each bin (also used for plain bin averages)."""

    def __init__(self, kedges, s, n_quad: int | None = None):
        self.kedges = np.asarray(kedges, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.nbins = len(self.kedges) - 1
        if n_quad is None:
            dk = np.max(np.diff(self.kedges))
            osc = dk * self.s.max() / (2 * np.pi)          # oscillations of j(ks) across a bin
            n_quad = int(10 * osc) + 16
        x, w = np.polynomial.legendre.leggauss(n_quad)
        lo, hi = self.kedges[:-1], self.kedges[1:]
        self.kq = 0.5 * (hi - lo)[:, None] * x[None, :] + 0.5 * (hi + lo)[:, None]   # (nbins, nq)
        self.wq = 0.5 * (hi - lo)[:, None] * w[None, :] * self.kq ** 2                # includes k^2
        self.norm = (hi ** 3 - lo ** 3) / 3.0
        self._bessel = {}
        self._cache = {}

    def bin_average(self, func) -> np.ndarray:
        """int_i k^2 dk f(k) / int_i k^2 dk per bin."""
        return np.sum(self.wq * func(self.kq), axis=1) / self.norm

    def _jl(self, lam: int) -> np.ndarray:
        if lam not in self._bessel:
            self._bessel[lam] = spherical_jn(lam, self.kq[:, :, None] * self.s[None, None, :])  # (nbins, nq, ns)
        return self._bessel[lam]

    def transform(self, entry, lam: int, tag=None, factor=None) -> np.ndarray:
        """pbar^{(i)}_{lam}(s) for p = entry (callable, e.g. power.function(L), or arrays (k, P); None -> p = 1, shot noise), times factor(k) if
        given. Shape (nbins, ns); cached under `tag`."""
        key = (tag, lam)
        if tag is not None and key in self._cache:
            return self._cache[key]
        pk = np.ones_like(self.kq) if entry is None else as_function(entry)(self.kq)
        if factor is not None:
            pk = pk * factor(self.kq)
        out = np.einsum('iq,iqs->is', self.wq * pk, self._jl(lam)) / self.norm[:, None]
        if tag is not None:
            self._cache[key] = out
        return out


def hankel_integral(func, x_edges, y, lam=0, factor=None) -> np.ndarray:
    """int x^2 f(x) j_lam(x y) dx over [x_edges[0], x_edges[-1]] for every y, by the quadrature of BinnedHankel on the
    intervals x_edges (more intervals for an f with structure: each gets nodes for the oscillations across it).
    E.g. xi_0(r) = hankel_integral(P, kedges, r) / (2 pi^2)."""
    quadrature = BinnedHankel(x_edges, y)
    return quadrature.norm @ quadrature.transform(func, lam, factor=factor)
