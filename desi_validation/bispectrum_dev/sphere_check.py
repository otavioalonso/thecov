"""Uniform sphere centred on the observer: Q_n from anchored pair sums vs Monte-Carlo quadrature."""
import numpy as np
from thecov import Tracer
from thecov.tracers import Window
from thecov.bispectrum_algebra import invariant_basis, box_window_value
from thecov.bispectrum_windows import AnchoredMoments

R_SPHERE, ALPHA = 500.0, 0.1


def sphere_tracer(n_ran=200000, seed=0, R=R_SPHERE):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-R, R, size=(3 * n_ran, 3))
    x = x[np.sum(x * x, 1) < R ** 2][:n_ran]
    V = 4 / 3 * np.pi * R ** 3
    nbar = ALPHA * len(x) / V
    T = Tracer('S', {'POSITION': x, 'WEIGHT': np.ones(len(x)), 'NZ': np.full(len(x), nbar)}, alpha=ALPHA)
    return T, nbar, V


def mc_reference(n, r1, r2, nbar, n_mc=400000, seed=1, kinds=('W', 'W', 'W'), alpha=ALPHA):
    """int d^3x dOmega1 dOmega2 W_a(x) W_1(x + r1) W_2(x + r2) Phi_n by Monte Carlo (x uniform in the sphere)."""
    rng = np.random.default_rng(seed)
    V = 4 / 3 * np.pi * R_SPHERE ** 3
    x = rng.uniform(-R_SPHERE, R_SPHERE, size=(2 * n_mc, 3))
    x = x[np.sum(x * x, 1) < R_SPHERE ** 2][:n_mc]
    def unit(m):
        v = rng.normal(size=(m, 3)); return v / np.linalg.norm(v, axis=1)[:, None]
    r1h, r2h = unit(len(x)), unit(len(x))
    y1, y2 = x + r1 * r1h, x + r2 * r2h
    inside = (np.sum(y1 * y1, 1) < R_SPHERE ** 2) & (np.sum(y2 * y2, 1) < R_SPHERE ** 2)
    wk = {'W': nbar ** 2, 'S': (1 + alpha) * nbar}
    weight = wk[kinds[0]] * wk[kinds[1]] * wk[kinds[2]]
    xh = x / np.linalg.norm(x, axis=1)[:, None]
    y1h = y1 / np.linalg.norm(y1, axis=1)[:, None]
    y2h = y2 / np.linalg.norm(y2, axis=1)[:, None]
    phi = invariant_basis(n, r1h, y1h, r2h, y2h, xh)
    vals = inside * phi
    est = weight * V * (4 * np.pi) ** 2 * np.mean(vals)
    err = weight * V * (4 * np.pi) ** 2 * np.std(vals) / np.sqrt(len(vals))
    return est, err
