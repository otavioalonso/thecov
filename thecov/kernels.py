"""Model power spectra and the shell kernels pbar^{(i)}_{L lambda}(s) (eq. pbar of the note):

    pbar^{(i)}_{L lambda}(s) = int_i k^2 dk p_L(k) j_lambda(k s) / int_i k^2 dk .
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn


class PowerSpectrumModel:
    """Container for model multipoles P_L^{AB}(k) tabulated on arbitrary k grids.

    Usage:
        model = PowerSpectrumModel()
        model.add(('LRG', 'LRG'), {0: (k, P0), 2: (k, P2), 4: (k, P4)})
    Multipoles not provided are taken to be zero. Cross spectra are symmetric in the tracers.
    """

    def __init__(self):
        self._spl = {}

    @staticmethod
    def _key(A, B):
        return tuple(sorted((str(A), str(B))))

    def add(self, tracers, multipoles: dict):
        key = self._key(*tracers)
        for L, (k, P) in multipoles.items():
            k = np.asarray(k, dtype=float)
            P = np.asarray(P, dtype=float)
            if np.any(np.diff(k) <= 0):
                raise ValueError("k must be strictly increasing")
            self._spl[key + (int(L),)] = CubicSpline(k, P, extrapolate=False)

    def has(self, A, B, L) -> bool:
        return self._key(A, B) + (int(L),) in self._spl

    def multipoles(self, A, B):
        key = self._key(A, B)
        return sorted(k[2] for k in self._spl if k[:2] == key)

    def __call__(self, A, B, L, k):
        spl = self._spl.get(self._key(A, B) + (int(L),))
        if spl is None:
            return np.zeros_like(np.asarray(k, dtype=float))
        out = spl(k)
        if np.any(np.isnan(out)):
            raise ValueError(f"P_{L}^{A}{B} requested outside its tabulated k range "
                             f"[{spl.x[0]:.4g}, {spl.x[-1]:.4g}]")
        return out


class ShellKernels:
    """Bin-averaged Bessel transforms of a function p_L(k) for all k bins, on a grid of s."""

    def __init__(self, k_edges, s, n_quad: int | None = None):
        self.k_edges = np.asarray(k_edges, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.nbins = len(self.k_edges) - 1
        if n_quad is None:
            dk = np.max(np.diff(self.k_edges))
            osc = dk * self.s.max() / (2 * np.pi)          # oscillations of j(ks) across a bin
            n_quad = int(10 * osc) + 16
        x, w = np.polynomial.legendre.leggauss(n_quad)
        lo, hi = self.k_edges[:-1], self.k_edges[1:]
        self.kq = 0.5 * (hi - lo)[:, None] * x[None, :] + 0.5 * (hi + lo)[:, None]   # (nbins, nq)
        self.wq = 0.5 * (hi - lo)[:, None] * w[None, :] * self.kq ** 2                # includes k^2
        self.norm = (hi ** 3 - lo ** 3) / 3.0
        self._bessel = {}
        self._cache = {}

    def _jl(self, lam: int) -> np.ndarray:
        if lam not in self._bessel:
            self._bessel[lam] = spherical_jn(lam, self.kq[:, :, None] * self.s[None, None, :])  # (nbins, nq, ns)
        return self._bessel[lam]

    def average(self, pfunc, lam: int, tag=None) -> np.ndarray:
        """pbar^{(i)}_{lambda}(s) for the function pfunc(k) (None -> 1, i.e. shot noise). Shape (nbins, ns)."""
        key = (tag, lam)
        if tag is not None and key in self._cache:
            return self._cache[key]
        pk = np.ones_like(self.kq) if pfunc is None else pfunc(self.kq)
        out = np.einsum('iq,iqs->is', self.wq * pk, self._jl(lam)) / self.norm[:, None]
        if tag is not None:
            self._cache[key] = out
        return out
