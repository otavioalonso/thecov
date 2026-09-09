r"""A realistic (if synthetic) survey geometry and the mock catalogues drawn in it.

The footprint is a spherical cap between two comoving distances, with

  * a smooth radial selection nbar(r) (so the window is not a top hat in the radial direction),
  * a few circular holes in the angular mask (bright stars / missing tiles),
  * a smooth large-scale completeness gradient across the cap,
  * FKP weights w = 1 / (1 + nbar P0).

The two tracers share the angular mask but have different nbar(r), bias and stochasticity, which is
the realistic multi-tracer situation: strongly overlapping but not identical windows.

Galaxies are Poisson-sampled cell by cell from  nbar(x) C(x) [1 + delta_X(x)]  and then displaced
radially by the linear velocity field, so redshift-space distortions are built with each galaxy's
own line of sight.
"""
from __future__ import annotations

import numpy as np

from .field import Grid


class Footprint:
    """Angular mask plus radial selection, evaluated on a grid or at points."""

    def __init__(self, r_min=400.0, r_max=900.0, cap_deg=30.0, holes=((8.0, 12.0, 3.0),
                                                                     (-14.0, 5.0, 2.5),
                                                                     (3.0, -16.0, 4.0)),
                 gradient=0.25, axis=(0.0, 0.0, 1.0)):
        self.r_min, self.r_max = float(r_min), float(r_max)
        self.cos_cap = np.cos(np.radians(cap_deg))
        self.cap_deg = float(cap_deg)
        self.holes = [(np.radians(a), np.radians(b), np.radians(c)) for a, b, c in holes]
        self.gradient = float(gradient)
        self.axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)

    def completeness(self, x, y, z, r):
        """Angular completeness in [0, 1]; zero outside the cap and inside the holes."""
        with np.errstate(invalid='ignore', divide='ignore'):
            ct = np.where(r > 0, (x * self.axis[0] + y * self.axis[1] + z * self.axis[2]) / np.where(r > 0, r, 1.0), 0.0)
        inside = ct >= self.cos_cap
        # local tangent-plane coordinates about the cap axis (axis assumed close to z)
        with np.errstate(invalid='ignore', divide='ignore'):
            tx = np.where(r > 0, x / np.where(r > 0, r, 1.0), 0.0)
            ty = np.where(r > 0, y / np.where(r > 0, r, 1.0), 0.0)
        comp = np.where(inside, 1.0 + self.gradient * tx / np.sin(np.radians(self.cap_deg)), 0.0)
        for (hx, hy, hr) in self.holes:
            comp = np.where((tx - hx) ** 2 + (ty - hy) ** 2 < hr ** 2, 0.0, comp)
        return np.clip(comp, 0.0, None)

    def radial(self, r, kind='A'):
        """nbar(r) for the two tracers (units (h/Mpc)^3)."""
        if kind == 'A':
            n0, rc, sig = 6.0e-4, 550.0, 180.0
        else:
            n0, rc, sig = 3.0e-4, 750.0, 260.0
        out = n0 * np.exp(-0.5 * ((r - rc) / sig) ** 2)
        return np.where((r >= self.r_min) & (r <= self.r_max), out, 0.0)

    def nbar_grid(self, grid: Grid, kind='A'):
        x, y, z = grid.coords()
        r = grid.radius()
        xb = np.broadcast_to(x, r.shape)
        yb = np.broadcast_to(y, r.shape)
        zb = np.broadcast_to(z, r.shape)
        return (self.radial(r, kind) * self.completeness(xb, yb, zb, r)).astype(np.float64)

    def extent(self):
        """Bounding box (min, max) of the footprint."""
        s = np.sin(np.radians(self.cap_deg))
        c = np.cos(np.radians(self.cap_deg))
        lo = np.array([-self.r_max * s, -self.r_max * s, self.r_min * c])
        hi = np.array([self.r_max * s, self.r_max * s, self.r_max])
        return lo, hi


def make_grid(fp: Footprint, N: int, box_factor: float = 2.5) -> Grid:
    """A cubic grid centred on the footprint, `box_factor` times its largest extent.

    The box must be comfortably larger than the survey: the covariance couples modes separated by
    |q| ~ 1/R_survey, and that structure is sampled at the box spacing 2 pi / L. A small box
    therefore biases the *mock* covariance (a Riemann-sum error of the test set-up, not of pkcov).
    Run the validation at two values of box_factor to check it.
    """
    lo, hi = fp.extent()
    centre = 0.5 * (lo + hi)
    L = box_factor * float(np.max(hi - lo))
    return Grid(centre - 0.5 * L, L, N)


def sample_points(intensity: np.ndarray, grid: Grid, rng: np.random.Generator,
                  cells=None, return_cells=False):
    """Poisson-sample points with the given mean number per cell, placed uniformly inside cells."""
    lam = np.clip(intensity, 0.0, None)
    if cells is None:
        cells = np.flatnonzero(lam.ravel() > 0)
    counts = rng.poisson(lam.ravel()[cells])
    keep = counts > 0
    idx = np.repeat(cells[keep], counts[keep])
    if len(idx) == 0:
        empty = np.zeros((0, 3))
        return (empty, (np.zeros(0, int),) * 3) if return_cells else empty
    i, j, k = np.unravel_index(idx, (grid.N,) * 3)
    base = grid.box_min + np.stack([i, j, k], axis=1) * grid.cell
    pos = base + rng.random((len(idx), 3)) * grid.cell
    return (pos, (i, j, k)) if return_cells else pos


def radial_rsd(pos: np.ndarray, psi, grid: Grid, f: float, cells=None) -> np.ndarray:
    """Displace each galaxy by f (Psi . r^) r^ along its OWN line of sight.

    The displacement is taken from the galaxy's own cell (not interpolated) so that it carries the
    same single power of the cell window as the density it was drawn from -- see GaussianField.
    """
    r = np.linalg.norm(pos, axis=1)
    rhat = pos / np.where(r > 0, r, 1.0)[:, None]
    if cells is None:
        ijk = np.floor((pos - grid.box_min) / grid.cell).astype(np.int64) % grid.N
        cells = (ijk[:, 0], ijk[:, 1], ijk[:, 2])
    psi_r = sum(psi[i][cells] * rhat[:, i] for i in range(3))
    return pos + f * psi_r[:, None] * rhat


class Catalogues:
    """Random catalogues and FKP weights, built once and shared by every realisation."""

    def __init__(self, fp: Footprint, grid: Grid, n_random_factor=15.0, P0_fkp=1e4,
                 rng: np.random.Generator | None = None, tracers=('A', 'B')):
        self.fp, self.grid = fp, grid
        self.tracers = tuple(tracers)
        self.P0_fkp = float(P0_fkp)
        rng = np.random.default_rng(1234) if rng is None else rng
        self.nbar = {t: fp.nbar_grid(grid, t) for t in self.tracers}
        self.cells = {t: np.flatnonzero(self.nbar[t].ravel() > 0) for t in self.tracers}
        self.n_gal_expected = {t: float(self.nbar[t].sum() * grid.V_cell) for t in self.tracers}
        self.randoms, self.alpha, self.w_ran, self.nbar_ran = {}, {}, {}, {}
        for t in self.tracers:
            lam = self.nbar[t] * grid.V_cell * n_random_factor
            pos = sample_points(lam, grid, rng, self.cells[t])
            self.randoms[t] = pos
            self.nbar_ran[t] = self._nbar_at(t, pos)
            self.w_ran[t] = 1.0 / (1.0 + self.nbar_ran[t] * self.P0_fkp)
            self.alpha[t] = self.n_gal_expected[t] / len(pos)

    def _nbar_at(self, t, pos):
        r = np.linalg.norm(pos, axis=1)
        comp = self.fp.completeness(pos[:, 0], pos[:, 1], pos[:, 2], r)
        return self.fp.radial(r, t) * comp

    def weights_at(self, t, pos):
        return 1.0 / (1.0 + self._nbar_at(t, pos) * self.P0_fkp)

    def pkcov_randoms(self, t):
        """The dict expected by pkcov.Tracer, plus its alpha."""
        return ({'POSITION': self.randoms[t], 'WEIGHT': self.w_ran[t], 'NZ': self.nbar_ran[t]},
                self.alpha[t])

    def I(self, t1, t2):
        """int nbar_1 nbar_2 w_1 w_2 evaluated on the grid (exact, for cross-checking pkcov's I)."""
        n1, n2 = self.nbar[t1], self.nbar[t2]
        w1 = 1.0 / (1.0 + n1 * self.P0_fkp)
        w2 = 1.0 / (1.0 + n2 * self.P0_fkp)
        return float((n1 * n2 * w1 * w2).sum() * self.grid.V_cell)


def make_mock(cat: Catalogues, field, bias: dict, stoch: dict, f: float,
              rng: np.random.Generator, rsd=True):
    """One realisation: Poisson-sample each tracer from the shared field and apply radial RSD."""
    grid = cat.grid
    delta_m = field.delta()
    psi = [field.displacement(i) for i in range(3)] if rsd else None
    field.free()
    out = {}
    for t in cat.tracers:
        noise = None
        if stoch.get(t, 0.0) > 0:
            from .field import white_noise_field
            noise = white_noise_field(grid, stoch[t], rng)
        d = bias[t] * delta_m if noise is None else bias[t] * delta_m + noise
        lam = cat.nbar[t] * grid.V_cell * (1.0 + d)
        pos, cells = sample_points(lam, grid, rng, cat.cells[t], return_cells=True)
        if rsd:
            pos = radial_rsd(pos, psi, grid, f, cells=cells)
        out[t] = pos
    return out


def model_multipoles(k, pk_lin, bias: dict, stoch: dict, f: float, tracers=('A', 'B')):
    """The exact multipoles of the mocks: P_XY(k, mu) = K_X K_Y P_lin + delta_XY N_X, K = b + f mu^2."""
    P = pk_lin(k)
    out = {}
    for i, X in enumerate(tracers):
        for Y in tracers[i:]:
            bX, bY = bias[X], bias[Y]
            m0 = (bX * bY + f * (bX + bY) / 3.0 + f ** 2 / 5.0) * P
            if X == Y:
                m0 = m0 + stoch.get(X, 0.0)
            out[(X, Y)] = {0: m0,
                           2: (2 * f * (bX + bY) / 3.0 + 4 * f ** 2 / 7.0) * P,
                           4: (8 * f ** 2 / 35.0) * P}
    return out
