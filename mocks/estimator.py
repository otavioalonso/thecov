r"""Yamamoto-FKP power-spectrum multipole estimator, evaluated with FFTs.

The estimator is the one the covariance is derived for,

    F^A_l(k) = int d^3x e^{-i k.x} L_l(k^ . x^) [n^A_g(x) - alpha_A n^A_r(x)] w_A(x),
    P^AB_l(k_i) = (2l+1) / I_AB  <F^A_l(k) F^B_0(-k)>_shell,

with the line of sight at the galaxy carrying the Legendre weight. The Legendre polynomial is
expanded in Cartesian moments of x^ (Bianchi et al. 2015; Scoccimarro 2015), so that

    F_0 = FFT[F],
    F_2 = (3/2) k^_i k^_j FFT[x^_i x^_j F] - (1/2) F_0                       (6 extra FFTs),
    F_4 = (35/8) k^_i k^_j k^_k k^_l FFT[x^_i x^_j x^_k x^_l F]
          - (30/8) k^_i k^_j FFT[x^_i x^_j F] + (3/8) F_0                    (15 extra FFTs).

Mass assignment is CIC (or TSC) with the window deconvolved, optionally interlaced (Sefusatti et al.
2016): a second grid shifted by half a cell along every axis is averaged in with the phase
exp(-i k.h), which cancels exactly every alias image k + 2 k_Nyq n with odd n_x + n_y + n_z. That
matters here beyond the usual shot-noise argument. The mocks are drawn from an intensity that is
piecewise constant on the SAME grid, so the galaxy field carries coherent images of every mode at
k + 2 k_Nyq n; without interlacing these fold back as a deterministic, direction-dependent
multiplicative bias on P(k) (about 6 % from n = (1,0,0) at k / k_Nyq = 0.56 with CIC), which
biases the mock variance by the same factor. With interlacing the leading residual comes from
n = (1,1,0) and friends.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .field import Grid

ORDER = {'ngp': 1, 'cic': 2, 'tsc': 3}


def assign(grid: Grid, pos: np.ndarray, weights: np.ndarray, scheme: str = 'cic') -> np.ndarray:
    """Mass assignment of weighted points onto the grid (periodic), by np.bincount."""
    p = ORDER[scheme]
    N = grid.N
    out = np.zeros(N ** 3)
    if len(pos) == 0:
        return out.reshape((N,) * 3)
    u = (pos - grid.box_min) / grid.cell - 0.5          # cell-centre coordinates
    if p == 1:
        i0 = np.rint(u).astype(np.int64)
        idx = ((i0[:, 0] % N) * N + i0[:, 1] % N) * N + i0[:, 2] % N
        return np.bincount(idx, weights=weights, minlength=N ** 3).reshape((N,) * 3)
    if p == 2:
        i0 = np.floor(u).astype(np.int64)
        d = u - i0
        offs = (0, 1)
        wfun = lambda o, d: d if o else 1.0 - d
    else:
        i0 = np.rint(u).astype(np.int64) - 1
        d = u - (i0 + 1)                                  # in [-1/2, 1/2]
        offs = (0, 1, 2)
        wfun = lambda o, d: (0.5 * (0.5 - d) ** 2 if o == 0 else
                             0.75 - d ** 2 if o == 1 else 0.5 * (0.5 + d) ** 2)
    wax = [[wfun(o, d[:, a]) for o in offs] for a in range(3)]
    iax = [[(i0[:, a] + o) % N for o in offs] for a in range(3)]
    for ox, oy, oz in itertools.product(range(len(offs)), repeat=3):
        idx = (iax[0][ox] * N + iax[1][oy]) * N + iax[2][oz]
        out += np.bincount(idx, weights=weights * wax[0][ox] * wax[1][oy] * wax[2][oz],
                           minlength=N ** 3)
    return out.reshape((N,) * 3)


def cic_assign(grid: Grid, pos: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Cloud-in-cell assignment (kept for backwards compatibility)."""
    return assign(grid, pos, weights, 'cic')


def assignment_correction(grid: Grid, scheme: str = 'cic') -> np.ndarray:
    """1 / W(k) for deconvolving the assignment window, W = prod_i sinc(n_i / N)^p."""
    p = ORDER[scheme]
    n1 = np.fft.fftfreq(grid.N, d=1.0 / grid.N)
    n3 = np.fft.rfftfreq(grid.N, d=1.0 / grid.N)
    s = lambda n: np.sinc(n / grid.N)          # np.sinc(x) = sin(pi x)/(pi x)
    w = (s(n1)[:, None, None] * s(n1)[None, :, None] * s(n3)[None, None, :]) ** p
    return 1.0 / w


def cic_correction(grid: Grid) -> np.ndarray:
    """1 / W_CIC(k) (kept for backwards compatibility)."""
    return assignment_correction(grid, 'cic')


def _raw_multipoles(grid: Grid, F: np.ndarray, ells, corr, kh):
    """F_l(k) of one real-space field on one grid (no interlacing)."""
    out = {}
    F0 = np.fft.rfftn(F) * corr
    out[0] = F0
    if 2 in ells or 4 in ells:
        r = grid.radius()
        xh = [grid.unit_vector(i, r) for i in range(3)]
        A2 = np.zeros_like(F0)
        for i, j in itertools.combinations_with_replacement(range(3), 2):
            m = 1.0 if i == j else 2.0
            A2 += m * kh[i] * kh[j] * (np.fft.rfftn(xh[i] * xh[j] * F) * corr)
        if 2 in ells:
            out[2] = 1.5 * A2 - 0.5 * F0
        if 4 in ells:
            A4 = np.zeros_like(F0)
            for c in itertools.combinations_with_replacement(range(3), 4):
                m = 24.0 / np.prod([math.factorial(c.count(a)) for a in range(3)])
                prod_x = xh[c[0]] * xh[c[1]] * xh[c[2]] * xh[c[3]]
                A4 += m * kh[c[0]] * kh[c[1]] * kh[c[2]] * kh[c[3]] * (np.fft.rfftn(prod_x * F) * corr)
            out[4] = (35 * A4 - 30 * A2 + 3 * F0) / 8.0
    return out


class MultipoleFields:
    """F_l(k) for one tracer, computed once and reused for every spectrum it enters.

    `scheme` is the mass assignment ('cic' or 'tsc'); `interlace` averages in a second grid shifted
    by half a cell (see the module docstring).
    """

    def __init__(self, grid: Grid, gal_pos, gal_w, ran_pos, ran_w, alpha, ells=(0, 2),
                 scheme='cic', interlace=False):
        self.grid = grid
        self.ells = tuple(ells)
        corr = assignment_correction(grid, scheme)
        kx, ky, kz = grid.kvec()
        k = grid.knorm()
        safe = np.where(k > 0, k, 1.0)
        kh = [kx / safe, ky / safe, kz / safe]
        grids = [grid]
        if interlace:
            h = 0.5 * grid.cell
            grids.append(Grid(grid.box_min + h, grid.L, grid.N, dtype=grid.dtype))
        self.F = None
        for n, g in enumerate(grids):
            F = assign(g, gal_pos, gal_w, scheme) - alpha * assign(g, ran_pos, ran_w, scheme)
            Fl = _raw_multipoles(g, F, self.ells, corr, kh)
            del F
            if n == 0:
                self.F = Fl
            else:
                # g's cell centres sit at +h relative to `grid`: F_true = exp(-i k.h) FFT[F_g]
                phase = np.exp(-1j * 0.5 * grid.cell * (kx + ky + kz))
                for ell in self.F:
                    self.F[ell] = 0.5 * (self.F[ell] + phase * Fl[ell])
            del Fl


class ShellBinner:
    """Bins the half-grid of modes into |k| shells, with the Hermitian multiplicity."""

    def __init__(self, grid: Grid, k_edges):
        self.k_edges = np.asarray(k_edges, dtype=float)
        k = grid.knorm().ravel()
        nz = grid.N // 2 + 1
        mult = np.full((grid.N, grid.N, nz), 2.0)
        mult[:, :, 0] = 1.0
        if grid.N % 2 == 0:
            mult[:, :, -1] = 1.0
        self.mult = mult.ravel()
        idx = np.digitize(k, self.k_edges) - 1
        ok = (idx >= 0) & (idx < len(self.k_edges) - 1) & (k > 0)
        self.idx, self.ok = idx, ok
        self.norm = np.bincount(idx[ok], weights=self.mult[ok], minlength=len(self.k_edges) - 1)
        self.k_eff = (np.bincount(idx[ok], weights=self.mult[ok] * k[ok], minlength=len(self.k_edges) - 1)
                      / np.where(self.norm > 0, self.norm, 1.0))

    def average(self, arr: np.ndarray) -> np.ndarray:
        v = arr.ravel()
        num = np.bincount(self.idx[self.ok], weights=(self.mult * v)[self.ok],
                          minlength=len(self.k_edges) - 1)
        return num / np.where(self.norm > 0, self.norm, 1.0)


def cross_multipole(fields_A: MultipoleFields, fields_B: MultipoleFields, ell: int,
                    binner: ShellBinner, I_AB: float) -> np.ndarray:
    """P^AB_l(k_i): (2l+1)/I_AB times the shell average of Re[F^A_l F^B_0*].

    No cell-volume factor appears: CIC assignment accumulates weighted *counts* per cell, i.e.
    already the integral of the density over the cell, so the FFT of that array is the continuum
    F(k) = int d^3x e^{-ikx} F(x) directly.
    """
    prod = (fields_A.F[ell] * np.conj(fields_B.F[0])).real
    return (2 * ell + 1) / I_AB * binner.average(prod)


def shot_noise(alpha_A, ran_w_A, I_AA, same_tracer=True, gal_w_A=None):
    """The FKP shot-noise constant, zero for a cross spectrum.

    With `gal_w_A` (the pypower / jaxpower convention, and the default of the validation) it is the
    REALISED sum_g w_g^2 + alpha^2 sum_r w_r^2, divided by I: every self-pair is removed. Without it,
    the expected value (1 + alpha) alpha sum_r w_r^2 / I is used; then the realised self-pair sum
    stays in P and its Poisson fluctuation, variance int nbar w^4, adds a fully correlated term to
    the monopole-auto covariance that is not part of the Gaussian formula.
    """
    if not same_tracer:
        return 0.0
    if gal_w_A is not None:
        return (float(np.sum(np.asarray(gal_w_A) ** 2)) + alpha_A ** 2 * float(np.sum(ran_w_A ** 2))) / I_AA
    return (1.0 + alpha_A) * alpha_A * float(np.sum(ran_w_A ** 2)) / I_AA


def realised_alpha(gal_w, ran_w):
    """sum w_data / sum w_randoms, as pypower / jaxpower compute it for each catalogue."""
    return float(np.sum(gal_w)) / float(np.sum(ran_w))
