"""Convergence report for the numerical control parameters.

`ds`, `ds_pair`, `n_sub`, `n_near`, `s_split` and `min_pairs` are user-facing knobs for which the
README only gives rules of thumb. This script sweeps them one at a time on a synthetic footprint and
prints the drift of the covariance diagonal and of the first two off-diagonals relative to a
reference (converged) setting, so you can choose values for your own survey from measured numbers.

    python -m diagnostics.convergence [--footprint sphere|shell] [--quick]

Interpretation:
  * the diagonal should be flat in every parameter well before the reference setting;
  * the first off-diagonal converges last in n_sub / n_near -- it is the quantity limited by
    pair-count noise, not by the method;
  * a residual drift in ds means the s grid is too coarse for your k_max (need ds <~ pi / 2 k_max);
  * a residual drift in ds_pair means Q is being interpolated too coarsely.
"""
import argparse
import time

import numpy as np

from pkcov import Tracer, PowerSpectrumModel, GaussianCovariance

REFERENCE = dict(ds=1.0, ds_pair=5.0, n_sub=8000, n_near=200000, s_split=80.0, min_pairs=20)
SWEEPS = {
    'ds': [8.0, 4.0, 2.0, 1.0],
    'ds_pair': [40.0, 20.0, 10.0, 5.0],
    'n_sub': [1000, 2000, 4000, 8000],
    'n_near': [20000, 50000, 100000, 200000],
    's_split': [0.0, 40.0, 80.0, 160.0],
    'min_pairs': [0, 20, 100],
}


def make_tracer(footprint, seed=0):
    rng = np.random.default_rng(seed)
    if footprint == 'sphere':
        R, n, nbar, dist = 500.0, 60000, 2e-4, 1500.0
        pts = []
        while sum(len(p) for p in pts) < n:
            x = rng.uniform(-R, R, size=(2 * n, 3))
            pts.append(x[np.sum(x * x, 1) < R * R])
        pos = np.concatenate(pts)[:n] + np.array([0, 0, dist])
        V = 4 * np.pi / 3 * R ** 3
        nz = np.full(n, nbar)
        alpha = nbar * V / n
    else:                       # a spherical-cap shell, closer to a real footprint
        n = 80000
        rmin, rmax, half = 1000.0, 2000.0, 35.0
        r = np.cbrt(rng.uniform(rmin ** 3, rmax ** 3, n))
        cmin = np.cos(np.radians(half))
        ct = rng.uniform(cmin, 1.0, n)
        ph = rng.uniform(0, 2 * np.pi, n)
        st = np.sqrt(1 - ct ** 2)
        pos = np.c_[r * st * np.cos(ph), r * st * np.sin(ph), r * ct]
        nz = 3e-4 * np.exp(-((r - 1400.0) / 500.0) ** 2)
        V = 2 * np.pi * (1 - cmin) * (rmax ** 3 - rmin ** 3) / 3
        alpha = float(np.mean(nz)) * V / n
    return Tracer('A', {'POSITION': pos, 'WEIGHT': np.ones(n), 'NZ': nz}, alpha)


def make_model():
    k = np.linspace(0.0, 1.0, 400)
    P = 2e4 * (k / 0.05 + 1e-3) / (1 + (k / 0.05) ** 2.2)
    model = PowerSpectrumModel()
    model.add(('A', 'A'), {0: (k, P), 2: (k, 0.5 * P), 4: (k, 0.1 * P)})
    return model


def summary(cov, k_edges, ells):
    """Diagonal and first two off-diagonals of each (ell, ell) block, as flat vectors."""
    out = {}
    for ell in ells:
        C = cov.block(('A', 'A'), ('A', 'A'), ell, ell)
        d = np.diag(C)
        out[(ell, 'diag')] = d
        out[(ell, 'off1')] = np.diag(C, 1) / np.sqrt(d[:-1] * d[1:])
        out[(ell, 'off2')] = np.diag(C, 2) / np.sqrt(d[:-2] * d[2:])
    return out


def run(tracer, model, k_edges, ells, **opts):
    cov = GaussianCovariance([tracer], k_edges, ells=ells, L_max=4, shot_noise=True, seed=0,
                             **opts).set_model(model)
    t0 = time.time()
    s = summary(cov, k_edges, ells)
    return s, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--footprint', default='sphere', choices=['sphere', 'shell'])
    ap.add_argument('--quick', action='store_true', help='cheaper reference and shorter sweeps')
    args = ap.parse_args()

    ref_opts = dict(REFERENCE)
    sweeps = {k: v for k, v in SWEEPS.items()}
    if args.quick:
        ref_opts.update(ds=2.0, ds_pair=10.0, n_sub=4000, n_near=100000)
        sweeps = {k: v[-3:] for k, v in sweeps.items()}

    tracer = make_tracer(args.footprint)
    model = make_model()
    k_edges = np.arange(0.02, 0.22, 0.02)
    ells = (0, 2)
    print(f"footprint: {args.footprint}, {tracer.size} randoms, k bins {len(k_edges) - 1}")
    print(f"reference: {ref_opts}\n")

    t0 = time.time()
    ref, _ = run(tracer, model, k_edges, ells, **ref_opts)
    print(f"reference computed in {time.time() - t0:.1f} s\n")

    hdr = f"{'parameter':>10} {'value':>10} {'time/s':>7} " + " ".join(
        f"{f'l={e} {w}':>12}" for e in ells for w in ('diag', 'off1', 'off2'))
    print(hdr)
    print("-" * len(hdr))
    for name, values in sweeps.items():
        for v in values:
            opts = dict(ref_opts)
            opts[name] = v
            got, dt = run(tracer, model, k_edges, ells, **opts)
            cells = []
            for e in ells:
                for w in ('diag', 'off1', 'off2'):
                    a, b = got[(e, w)], ref[(e, w)]
                    if w == 'diag':
                        cells.append(f"{np.max(np.abs(a / b - 1)):12.4f}")
                    else:                      # already normalised by the diagonal
                        cells.append(f"{np.max(np.abs(a - b)):12.4f}")
            print(f"{name:>10} {str(v):>10} {dt:7.1f} " + " ".join(cells))
        print()
    print("diag entries: max |ratio - 1|.  off1/off2: max absolute change of the correlation "
          "coefficient.  Both relative to the reference setting.")


if __name__ == '__main__':
    main()
