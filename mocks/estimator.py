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
    no = len(offs)
    # one bincount per block of points over all no^3 offsets at once: a bincount per offset would
    # allocate and add a full N^3 output every time (27 of them for TSC), which dominated the cost
    block = max(1, 4_000_000 // no ** 3)
    weights = np.asarray(weights, dtype=float)
    for b0 in range(0, len(pos), block):
        sl = slice(b0, b0 + block)
        wax = [np.stack([wfun(o, d[sl, a]) for o in offs]) for a in range(3)]          # (no, n)
        iax = [np.stack([(i0[sl, a] + o) % N for o in offs]) for a in range(3)]
        idx = ((iax[0][:, None, None, :] * N + iax[1][None, :, None, :]) * N
               + iax[2][None, None, :, :]).ravel()
        ww = (weights[sl][None, None, None, :] * wax[0][:, None, None, :] * wax[1][None, :, None, :]
              * wax[2][None, None, :, :]).ravel()
        out += np.bincount(idx, weights=ww, minlength=N ** 3)
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


class MultipoleEstimator:
    """The same estimator as MultipoleFields, set up once and reused for every realisation.

    For a validation run everything but the galaxies is fixed, so per worker it caches: the painted
    random catalogues (the FKP field is linear, F = paint(gal) - alpha paint(ran), so the ~15x larger
    random catalogue is painted once instead of per mock and per interlaced grid), the line-of-sight
    geometry (1/r^2 and the cell-centre coordinates, so x^_i x^_j = x_i x_j / r^2 needs no per-mock
    unit-vector arrays), the k-space factors and the interlacing phase. Grid fields and FFTs are in
    single precision (scipy.fft, complex64): on the production mocks the multipoles agree with the
    float64 MultipoleFields to < 1e-4 of P0 (rounding; ~1e-3 of the statistical error per bin), and
    the memory traffic that bounds these operations halves. Shell averages accumulate in float64.
    About 4x faster per realisation than MultipoleFields at N = 256 with interlacing.
    """

    def __init__(self, grid: Grid, ells=(0, 2), scheme='cic', interlace=False, fft_workers=1):
        if any(ell not in (0, 2) for ell in ells):
            raise ValueError("MultipoleEstimator supports ells 0 and 2 (use MultipoleFields for 4)")
        self.grid, self.ells, self.scheme = grid, tuple(ells), scheme
        self.fft_workers = int(fft_workers)
        f32 = np.float32
        self.grids = [grid]
        if interlace:
            h = 0.5 * grid.cell
            self.grids.append(Grid(grid.box_min + h, grid.L, grid.N, dtype=grid.dtype))
        self.corr = assignment_correction(grid, scheme).astype(f32)
        kx, ky, kz = grid.kvec()
        k2 = (kx ** 2 + ky ** 2 + kz ** 2)
        inv_k2 = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
        kv = (kx, ky, kz)
        # m k^_i k^_j / W(k) for the six (i <= j): half-size k-space arrays, set up once
        self.kfac = {(i, j): ((1.0 if i == j else 2.0) * kv[i] * kv[j] * inv_k2 * self.corr).astype(f32)
                     for i, j in itertools.combinations_with_replacement(range(3), 2)} if 2 in self.ells else {}
        self.phase = (np.exp(-1j * 0.5 * grid.cell * (kx + ky + kz)).astype(np.complex64)
                      if interlace else None)
        self.geom = []
        for g in self.grids:
            x, y, z = g.coords()
            r2 = x ** 2 + y ** 2 + z ** 2
            self.geom.append(([x.astype(f32), y.astype(f32), z.astype(f32)],
                              np.where(r2 > 0, 1.0 / np.where(r2 > 0, r2, 1.0), 0.0).astype(f32)))
        self._ran = {}

    def _rfftn(self, a):
        import scipy.fft
        return scipy.fft.rfftn(a, workers=self.fft_workers)

    def set_randoms(self, name, ran_pos, ran_w):
        """Paint a random catalogue once on every (interlaced) grid."""
        self._ran[name] = [assign(g, ran_pos, ran_w, self.scheme).astype(np.float32) for g in self.grids]

    def fields(self, name, gal_pos, gal_w, alpha):
        """F_l(k) of one tracer (an object with the .F dict that cross_multipole reads)."""
        out = None
        for n, g in enumerate(self.grids):
            F = assign(g, gal_pos, gal_w, self.scheme).astype(np.float32)
            F -= np.float32(alpha) * self._ran[name][n]
            Fl = self._multipoles(F, n)
            del F
            if n == 0:
                out = Fl
            else:              # g's cell centres sit at +h: F_true = exp(-i k.h) FFT[F_g]
                for ell in out:
                    out[ell] = 0.5 * (out[ell] + self.phase * Fl[ell])
        res = type('Fields', (), {})()
        res.F, res.ells, res.grid = out, self.ells, self.grid
        return res

    def _multipoles(self, F, n):
        out = {}
        F0 = self._rfftn(F)
        F0 *= self.corr
        out[0] = F0
        if 2 in self.ells:
            c, inv_r2 = self.geom[n]
            G = F * inv_r2                                   # F / r^2
            buf = np.empty_like(G)
            A2 = np.zeros_like(F0)
            for (i, j), kf in self.kfac.items():
                np.multiply(G, c[i], out=buf)
                buf *= c[j]
                Fij = self._rfftn(buf)                       # FFT[x^_i x^_j F]
                Fij *= kf                                    # m k^_i k^_j / W(k)
                A2 += Fij
                del Fij
            del G, buf
            A2 *= np.float32(1.5)
            A2 -= np.float32(0.5) * F0
            out[2] = A2
        return out


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
