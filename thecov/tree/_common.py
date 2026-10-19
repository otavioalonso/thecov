"""Helpers shared by the model-based covariance terms (trispectrum, discreteness, response approach)."""
import numpy as np
from scipy.interpolate import CubicSpline


class LinearPower:
    """P_lin(k) from a table (cubic spline in ln k; 0 outside the table, in particular at k = 0); picklable"""

    def __init__(self, k, P):
        k, P = np.asarray(k, float), np.asarray(P, float)
        self._spl = CubicSpline(np.log(k), P)
        self.kmin, self.kmax = k[0], k[-1]

    def __call__(self, k):
        k = np.asarray(k, float)
        out = np.zeros_like(k)
        ok = (k > self.kmin) & (k < self.kmax)
        out[ok] = self._spl(np.log(k[ok]))
        return out


def shell_nodes(k_edges, n):
    """Gauss-Legendre nodes in |k| per bin with k^2-weighted, normalised weights: ([nodes per bin], [weights per bin])."""
    x, w = np.polynomial.legendre.leggauss(n)
    nodes, weights = [], []
    for lo, hi in zip(k_edges[:-1], k_edges[1:]):
        k = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        wk = w * k ** 2
        nodes.append(k)
        weights.append(wk / wk.sum())
    return nodes, weights


def fill_symmetric(C, i, j, values, ells, nbins, scale=1.0):
    """C[(l1, i), (l2, j)] = C[(l2, j), (l1, i)] = scale * values[(l1, l2)] in the (ell, bin) data-vector order."""
    for a, l1 in enumerate(ells):
        for b, l2 in enumerate(ells):
            C[a * nbins + i, b * nbins + j] = C[b * nbins + j, a * nbins + i] = scale * values[(l1, l2)]
