r"""Estimator aliasing on the mock set-up, measured without cosmic variance.

    python -m diagnostics.aliasing --grid 512 --box-factor 2.5 --kmax 0.4 --ref-grid 768

One mock catalogue is generated on the production grid and its multipoles are estimated with
several assignment schemes on that grid, and with TSC + interlacing on a finer reference grid of
the same box. Because the catalogue is common, the ratio to the reference is the deterministic
aliasing bias of each scheme (plus a small incoherent part from aliased shot noise); no cosmic
variance enters. The mock covariance is biased by the square of this ratio, so it must be well
below the per-bin statistical error of the validation, 1 / sqrt(2 (N_mock - 1)) in sigma.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from mocks.estimator import MultipoleFields, ShellBinner, cross_multipole, shot_noise
from mocks.field import GaussianField, Grid
from mocks.run_validation import BIAS, GROWTH, STOCH, pk_lin
from mocks.survey import Catalogues, Footprint, make_grid, make_mock


def measure(cat, cats, grid, k_edges, ells, scheme, interlace, spectra):
    binner = ShellBinner(grid, k_edges)
    t0 = time.time()
    f = {t: MultipoleFields(grid, cats[t], cat.weights_at(t, cats[t]), cat.randoms[t], cat.w_ran[t],
                            cat.alpha[t], ells=ells, scheme=scheme, interlace=interlace)
         for t in cat.tracers}
    out = {}
    for (X, Y) in spectra:
        I = cat.I(X, Y)
        for ell in ells:
            P = cross_multipole(f[X], f[Y], ell, binner, I)
            if X == Y and ell == 0:
                P = P - shot_noise(cat.alpha[X], cat.w_ran[X], I)
            out[(X, Y, ell)] = P
    return out, binner.k_eff, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid', type=int, default=512)
    ap.add_argument('--ref-grid', type=int, default=768)
    ap.add_argument('--box-factor', type=float, default=2.5)
    ap.add_argument('--kmin', type=float, default=0.01)
    ap.add_argument('--kmax', type=float, default=0.4)
    ap.add_argument('--dk', type=float, default=0.02)
    ap.add_argument('--ells', type=int, nargs='+', default=[0, 2])
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--n-random-factor', type=float, default=15.0)
    args = ap.parse_args()

    fp = Footprint()
    grid = make_grid(fp, args.grid, args.box_factor)
    ref = Grid(grid.box_min, grid.L, args.ref_grid, dtype=grid.dtype)
    print(f"L = {grid.L:.0f}, k_Nyq = {grid.k_nyquist:.3f} (ref {ref.k_nyquist:.3f}), "
          f"k_max/k_Nyq = {args.kmax / grid.k_nyquist:.2f} (ref {args.kmax / ref.k_nyquist:.2f})")
    cat = Catalogues(fp, grid, n_random_factor=args.n_random_factor)
    rng = np.random.default_rng(args.seed)
    cats = make_mock(cat, GaussianField(grid, pk_lin, rng), BIAS, STOCH, GROWTH, rng, rsd=True)
    spectra = [('A', 'A'), ('A', 'B'), ('B', 'B')]
    k_edges = np.arange(args.kmin, args.kmax + args.dk / 2, args.dk)
    ells = tuple(args.ells)

    P_ref, k_eff, t = measure(cat, cats, ref, k_edges, ells, 'tsc', True, spectra)
    print(f"reference N={args.ref_grid} tsc+interlaced: {t:.0f} s")
    print("k:          " + " ".join(f"{k:6.3f}" for k in k_eff))
    for scheme, inter in (('cic', False), ('cic', True), ('tsc', True)):
        P, _, t = measure(cat, cats, grid, k_edges, ells, scheme, inter, spectra)
        print(f"\nN={args.grid} {scheme}{'+interlaced' if inter else ''}  ({t:.0f} s)   "
              f"P / P_ref - 1  [%]  (l=2: difference in units of |P_0^ref|)")
        for key in P:
            X, Y, ell = key
            if ell == 0:
                r = 100 * (P[key] / P_ref[key] - 1)
            else:
                r = 100 * (P[key] - P_ref[key]) / np.abs(P_ref[(X, Y, 0)])
            print(f"  {X}{Y} l={ell}: " + " ".join(f"{v:+6.2f}" for v in r))


if __name__ == '__main__':
    main()
