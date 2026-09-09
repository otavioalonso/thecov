r"""Gaussian density and displacement fields on a cubic grid.

Conventions
-----------
The grid has N cells per side, cell volume V_c = (L/N)^3. With numpy's unnormalised transforms,

    delta_k^grid = rfftn(white) * sqrt(P(k) / V_c),      delta(x) = irfftn(delta_k^grid),

which gives  <|delta_k^cont|^2> = V P(k)  with delta_k^cont = V_c delta_k^grid, i.e. the usual
continuum normalisation, and Var[delta(x)] = int d^3k/(2pi)^3 P(k).

The linear (Zel'dovich) displacement is Psi(k) = i k / k^2 delta_m(k); the redshift-space
displacement of a tracer moving with the matter is s = f (Psi . r^) r^, applied to each particle
along ITS OWN line of sight. That is what makes these mocks a genuine test of the local
plane-parallel approximation: nothing in the generation assumes a global line of sight.

Amplitudes are meant to be kept low (sigma_delta <~ 0.3). These mocks validate the *Gaussian*
covariance formula, so a nearly-Gaussian field is what is wanted; it also keeps 1 + delta positive
so that Poisson sampling needs no clipping.
"""
from __future__ import annotations

import numpy as np


class Grid:
    """Cubic grid with an observer at the physical origin."""

    def __init__(self, box_min, L: float, N: int, dtype=np.float32):
        self.box_min = np.asarray(box_min, dtype=float)
        self.L = float(L)
        self.N = int(N)
        self.cell = self.L / self.N
        self.V_cell = self.cell ** 3
        self.V_box = self.L ** 3
        self.dtype = dtype

    # -- coordinates ---------------------------------------------------------
    def axis(self, i: int) -> np.ndarray:
        return self.box_min[i] + (np.arange(self.N) + 0.5) * self.cell

    def coords(self):
        """Cell-centre coordinates as three broadcastable arrays (x, y, z)."""
        return np.meshgrid(self.axis(0), self.axis(1), self.axis(2), indexing='ij', sparse=True)

    def radius(self) -> np.ndarray:
        x, y, z = self.coords()
        return np.sqrt(x ** 2 + y ** 2 + z ** 2).astype(self.dtype)

    def unit_vector(self, i: int, r: np.ndarray | None = None) -> np.ndarray:
        """x^_i on the grid (0 where r = 0)."""
        x, y, z = self.coords()
        c = (x, y, z)[i]
        r = self.radius() if r is None else r
        out = np.broadcast_to(c, (self.N,) * 3) / np.where(r > 0, r, 1.0)
        return np.where(r > 0, out, 0.0).astype(self.dtype)

    # -- Fourier -------------------------------------------------------------
    def kvec(self):
        kf = 2 * np.pi / self.L
        k1 = np.fft.fftfreq(self.N, d=1.0 / self.N) * kf
        k3 = np.fft.rfftfreq(self.N, d=1.0 / self.N) * kf
        return k1[:, None, None], k1[None, :, None], k3[None, None, :]

    def knorm(self) -> np.ndarray:
        kx, ky, kz = self.kvec()
        return np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    @property
    def k_nyquist(self) -> float:
        return np.pi / self.cell


def cell_window(grid: Grid) -> np.ndarray:
    """T(k) = prod_i sinc(n_i / N): the Fourier transform of one grid cell.

    Galaxies drawn cell by cell and then placed uniformly inside the cell realise the CELL-AVERAGED
    density, i.e. the clustering is multiplied by T(k). Generating the field with an extra 1/T
    cancels this exactly, so the point process has the intended spectrum. T never vanishes on the
    grid (its smallest value is sinc(1/2)^3 = 0.258), so the division is safe.
    """
    n1 = np.fft.fftfreq(grid.N, d=1.0 / grid.N)
    n3 = np.fft.rfftfreq(grid.N, d=1.0 / grid.N)
    return (np.sinc(n1 / grid.N)[:, None, None] * np.sinc(n1 / grid.N)[None, :, None]
            * np.sinc(n3 / grid.N)[None, None, :])


class GaussianField:
    """delta_m and the linear displacement field for one realisation.

    `deconvolve_cell` pre-divides by the cell window so that the sampled point process, whose
    intensity is piecewise constant over cells, has exactly the requested spectrum. The same factor
    is inherited by the displacement field, which is why particles must take their displacement from
    their OWN cell (nearest-cell, not trilinear): both the density and the RSD term then pick up the
    same single power of T and both are cancelled.

    `k_cut` zeroes the spectrum above a fraction of the Nyquist frequency. With no power above
    k_cut, aliasing of the clustering signal into k < 2 k_Nyq - k_cut vanishes identically; only the
    (much smaller) shot-noise aliasing survives, so keep k_max <~ 0.4 k_Nyq.
    """

    def __init__(self, grid: Grid, pk_lin, rng: np.random.Generator, deconvolve_cell=True,
                 k_cut=0.8):
        self.grid = grid
        self.pk = pk_lin
        self.rng = rng
        self.deconvolve_cell = bool(deconvolve_cell)
        self.k_cut = float(k_cut) * grid.k_nyquist
        self._deltak = None

    def _generate_k(self):
        g = self.grid
        white = self.rng.standard_normal((g.N, g.N, g.N))
        dk = np.fft.rfftn(white)
        del white
        k = g.knorm()
        amp = np.zeros_like(k)
        nz = (k > 0) & (k <= self.k_cut)
        amp[nz] = np.sqrt(np.maximum(self.pk(k[nz]), 0.0) / g.V_cell)
        if self.deconvolve_cell:
            amp /= cell_window(g)
        dk *= amp
        dk[0, 0, 0] = 0.0                       # no mean mode
        self._deltak = dk
        return dk

    @property
    def deltak(self):
        return self._deltak if self._deltak is not None else self._generate_k()

    def delta(self) -> np.ndarray:
        return np.fft.irfftn(self.deltak, s=(self.grid.N,) * 3).astype(self.grid.dtype)

    def displacement(self, i: int) -> np.ndarray:
        """Psi_i(x) = irfftn(i k_i / k^2 delta_k) -- the Zel'dovich displacement component."""
        g = self.grid
        kv = g.kvec()[i]
        k2 = g.knorm() ** 2
        fac = np.zeros_like(k2)
        nz = k2 > 0
        fac[nz] = 1.0 / k2[nz]
        return np.fft.irfftn(1j * kv * fac * self.deltak, s=(g.N,) * 3).astype(g.dtype)

    def free(self):
        self._deltak = None


def white_noise_field(grid: Grid, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """A field with a constant power spectrum P(k) = amplitude (the tracer stochasticity)."""
    if amplitude <= 0:
        return np.zeros((grid.N,) * 3, dtype=grid.dtype)
    return (rng.standard_normal((grid.N,) * 3) * np.sqrt(amplitude / grid.V_cell)).astype(grid.dtype)


def trilinear(field: np.ndarray, grid: Grid, pos: np.ndarray) -> np.ndarray:
    """Interpolate a periodic grid field at arbitrary positions."""
    u = (pos - grid.box_min) / grid.cell - 0.5
    i0 = np.floor(u).astype(np.int64)
    d = (u - i0).astype(np.float64)
    N = grid.N
    out = np.zeros(len(pos))
    for dx in (0, 1):
        wx = d[:, 0] if dx else 1 - d[:, 0]
        ix = (i0[:, 0] + dx) % N
        for dy in (0, 1):
            wy = d[:, 1] if dy else 1 - d[:, 1]
            iy = (i0[:, 1] + dy) % N
            for dz in (0, 1):
                wz = d[:, 2] if dz else 1 - d[:, 2]
                iz = (i0[:, 2] + dz) % N
                out += wx * wy * wz * field[ix, iy, iz]
    return out
