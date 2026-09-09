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

Mass assignment is CIC with the standard window deconvolution; keep k_max below about half the
Nyquist frequency so that aliasing stays negligible.
"""
from __future__ import annotations

import itertools

import numpy as np

from .field import Grid


def cic_assign(grid: Grid, pos: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Cloud-in-cell assignment of weighted points onto the grid (periodic)."""
    out = np.zeros((grid.N,) * 3)
    if len(pos) == 0:
        return out
    u = (pos - grid.box_min) / grid.cell - 0.5
    i0 = np.floor(u).astype(np.int64)
    d = u - i0
    N = grid.N
    for dx in (0, 1):
        wx = d[:, 0] if dx else 1 - d[:, 0]
        ix = (i0[:, 0] + dx) % N
        for dy in (0, 1):
            wy = d[:, 1] if dy else 1 - d[:, 1]
            iy = (i0[:, 1] + dy) % N
            for dz in (0, 1):
                wz = d[:, 2] if dz else 1 - d[:, 2]
                iz = (i0[:, 2] + dz) % N
                np.add.at(out, (ix, iy, iz), weights * wx * wy * wz)
    return out


def cic_correction(grid: Grid) -> np.ndarray:
    """1 / W_CIC(k)^2 for deconvolving the assignment window."""
    kf = np.pi / grid.N
    n1 = np.fft.fftfreq(grid.N, d=1.0 / grid.N)
    n3 = np.fft.rfftfreq(grid.N, d=1.0 / grid.N)
    s = lambda n: np.sinc(n / grid.N)          # np.sinc(x) = sin(pi x)/(pi x)
    w = (s(n1)[:, None, None] * s(n1)[None, :, None] * s(n3)[None, None, :]) ** 2
    return 1.0 / w


class MultipoleFields:
    """F_l(k) for one tracer, computed once and reused for every spectrum it enters."""

    def __init__(self, grid: Grid, gal_pos, gal_w, ran_pos, ran_w, alpha, ells=(0, 2)):
        self.grid = grid
        self.ells = tuple(ells)
        F = cic_assign(grid, gal_pos, gal_w) - alpha * cic_assign(grid, ran_pos, ran_w)
        corr = cic_correction(grid)
        r = grid.radius()
        xh = [grid.unit_vector(i, r) for i in range(3)]
        kx, ky, kz = grid.kvec()
        k = grid.knorm()
        safe = np.where(k > 0, k, 1.0)
        kh = [kx / safe, ky / safe, kz / safe]
        self.F = {}
        F0 = np.fft.rfftn(F) * corr
        self.F[0] = F0
        if 2 in self.ells or 4 in self.ells:
            A2 = np.zeros_like(F0)
            for i, j in itertools.product(range(3), repeat=2):
                A2 += kh[i] * kh[j] * (np.fft.rfftn(xh[i] * xh[j] * F) * corr)
            if 2 in self.ells:
                self.F[2] = 1.5 * A2 - 0.5 * F0
        if 4 in self.ells:
            A4 = np.zeros_like(F0)
            for i, j, m, n in itertools.product(range(3), repeat=4):
                A4 += kh[i] * kh[j] * kh[m] * kh[n] * (np.fft.rfftn(xh[i] * xh[j] * xh[m] * xh[n] * F) * corr)
            self.F[4] = (35 * A4 - 30 * A2 + 3 * F0) / 8.0
        del F


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


def shot_noise(alpha_A, ran_w_A, I_AA, same_tracer=True):
    """The FKP shot-noise constant (1 + alpha) int nbar w^2 / I, zero for a cross spectrum.

    int nbar w^2 is estimated from the randoms as alpha * sum_r w^2, which is why alpha must be the
    ratio of the *expected* galaxy count to the random count: using a realisation-dependent alpha
    would propagate the realisation's own number fluctuation into the subtracted constant.
    """
    if not same_tracer:
        return 0.0
    return (1.0 + alpha_A) * alpha_A * float(np.sum(ran_w_A ** 2)) / I_AA
