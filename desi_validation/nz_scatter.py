"""The radial long-mode content of the mocks, measured: n(z) of every mock from the DATA files only (no randoms, seconds
per mock), its mock-to-mock covariance, and the comparison with linear theory at every radial scale. NERSC, debug queue,
~15 min for 859 mocks with 32 processes:

    cd ~/thecov && sbatch -N 1 -C cpu -q debug -t 00:30:00 -J nz -o $OUT/nz_scatter_LRG1.log --wrap "... python -u -m desi_validation.nz_scatter --bin LRG1 --workers 32"

Why: the super-sample term of a thin shell is dominated by radial modes, and the mocks' delta_norm (= 2 x the shell-mean
galaxy overdensity) scatters 2-3x less than linear theory predicts for the window (HANDOFF 8.2). This measures directly,
per radial scale, how much of the long-mode power the mocks contain, whether NGC and SGC share modes (same parent box),
and gives the covariance C_nz(z, z') that a mock-calibrated SSC can use instead of P_lin for the radial modes.

Output: <--out>/nz_scatter_<bin>.npz with NZ[region] (n_mock, n_bins) weighted counts, mock ids, z edges, and the
printed analysis: std/mean of the counts in merged bins (500, 250, 100, 50, 25 Mpc/h) against Poisson and against the
linear-theory slab prediction (A P_lin x (b1 + f)^2 from the ssc npz of --label when found), the NGC-SGC correlation of
the shell counts, and the eigenvalues of the normalised covariance.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc  # noqa: E402

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def one_mock(args):
    mock, bin_, regions, n_bins, cs_version = args
    tracer, zr = dc.TRACER_SPECS[bin_]
    paths = dc.Paths(kind='holi_v3', mock=mock)
    paths.cs_version = cs_version
    edges = np.linspace(zr[0], zr[1], n_bins + 1)
    out = {}
    for r in regions:
        fn = paths.data_fn(tracer, r)
        try:
            d = dc.read_catalog(fn, ['Z', 'WEIGHT', 'WEIGHT_FKP', 'RA', 'DEC'], paths.h5_group, required=('Z',))
        except Exception as e:
            out[r] = (None, f'{type(e).__name__}: {e}')
            continue
        z = np.asarray(d['Z'], float)
        w = np.asarray(d.get('WEIGHT', np.ones_like(z)), float)      # completeness weights only: the number density
        out[r] = (np.histogram(z, edges, weights=w)[0], None)
    return mock, out


def slab_sigma(dr, A, kl, P):
    """rms of the mean matter density of a slab of thickness dr [Mpc/h] and area A [(Mpc/h)^2] (radial modes)"""
    q = np.linspace(1e-4, 2.0, 200000)
    Pq = np.interp(q, kl, P)
    W = np.sinc(q * dr / 2 / np.pi) ** 2
    return np.sqrt(np.trapezoid(Pq * W, q) / np.pi / A)


def analyse(NZ, ids, edges, region, theory, area):
    n, nb = NZ.shape
    zc = 0.5 * (edges[1:] + edges[:-1])
    chi = dc.comoving_distance(edges)
    Lr = chi[-1] - chi[0]
    print(f'\n{region}: {n} mocks, {nb} z bins, shell {chi[0]:.0f}-{chi[-1]:.0f} Mpc/h ({Lr:.0f} thick), area {area / (np.pi / 180) ** 2:.0f} deg^2')
    print('  merged bins: measured std/mean of the weighted count, its clustering part (Poisson removed), linear theory for the slab')
    for m in (1, 2, 5, 10, 20):
        if nb % m:
            continue
        merged = NZ.reshape(n, m, -1).sum(2)
        meas = merged.std(0, ddof=1) / merged.mean(0)
        pois = 1 / np.sqrt(merged.mean(0))
        clus = np.sqrt(np.maximum(meas ** 2 - pois ** 2, 0))
        line = f'  {m:2d} bins ({Lr / m:5.0f} Mpc/h): measured {meas.mean() * 100:.2f}%  clustering {clus.mean() * 100:.2f}%  Poisson {pois.mean() * 100:.2f}%'
        if theory is not None:
            kl, P, g = theory
            th = g * slab_sigma(Lr / m, area, kl, P)
            line += f'  theory {th * 100:.2f}%  ratio {clus.mean() / th:.2f} (+- {clus.mean() / th / np.sqrt(2 * (n - 1)) * np.sqrt(m):.2f})'
        print(line)
    d = NZ / NZ.mean(0) - 1
    C = np.cov(d.T)
    w = np.linalg.eigvalsh(C)[::-1]
    print('  eigenvalues of Cov(delta n(z)) x 1e4 (largest first): ' + ' '.join(f'{x * 1e4:.2f}' for x in w[:6]))
    print('  correlation of adjacent bins: ' + ' '.join(f'{C[i, i + 1] / np.sqrt(C[i, i] * C[i + 1, i + 1]):+.2f}' for i in range(min(nb - 1, 10))))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--label', default=None, help='ssc_<label>.npz in --out for P_lin and b1, f (default ssc_holi-kcore2-<bin>)')
    ap.add_argument('--out', default='/global/cfs/cdirs/desicollab/users/oalves/thecov_validation')
    ap.add_argument('--mocks', type=int, nargs='*', default=None, help='default: every mock directory found')
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--n-bins', type=int, default=20)
    ap.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument('--cs-version', default='holi-v3-altmtl')
    ap.add_argument('--analyse-only', action='store_true', help='reuse <out>/nz_scatter_<bin>.npz')
    args = ap.parse_args()
    b = args.bin
    tracer, zr = dc.TRACER_SPECS[b]
    out_fn = os.path.join(args.out, f'nz_scatter_{b}.npz')
    edges = np.linspace(zr[0], zr[1], args.n_bins + 1)
    if args.analyse_only and os.path.exists(out_fn):
        z = np.load(out_fn)
        NZ = {r: z[f'{r}/NZ'] for r in args.regions if f'{r}/NZ' in z.files}
        ids = {r: z[f'{r}/mocks'] for r in NZ}
        areas = {r: float(z[f'{r}/area']) for r in NZ}
    else:
        mocks = args.mocks
        if mocks is None:
            pat = dc.Paths(kind='holi_v3', mock=0)._dir().replace('altmtl0', 'altmtl*').replace('mock0', 'mock*')
            mocks = sorted({int(os.path.basename(os.path.dirname(p))[4:]) for p in glob.glob(pat)
                            if os.path.basename(os.path.dirname(p))[4:].isdigit()})
        log(f'{b}: {len(mocks)} mocks, {args.workers} processes')
        tasks = [(m, b, args.regions, args.n_bins, args.cs_version) for m in mocks]
        if args.workers > 1:
            from multiprocessing import Pool
            with Pool(args.workers) as pool:
                res = pool.map(one_mock, tasks, chunksize=4)
        else:
            res = [one_mock(t) for t in tasks]
        NZ, ids, areas = {}, {}, {}
        for r in args.regions:
            rows = [(m, o[r][0]) for m, o in res if o[r][0] is not None]
            bad = [(m, o[r][1]) for m, o in res if o[r][0] is None]
            if bad:
                log(f'{r}: {len(bad)} mocks failed, e.g. mock {bad[0][0]}: {bad[0][1]}')
            if not rows:
                continue
            ids[r] = np.array([m for m, _ in rows])
            NZ[r] = np.array([v for _, v in rows], float)
            # footprint area from one mock's data positions
            paths = dc.Paths(kind='holi_v3', mock=int(ids[r][0]))
            d = dc.read_catalog(paths.data_fn(tracer, r), ['RA', 'DEC'], paths.h5_group, required=('RA', 'DEC'))
            areas[r] = float(dc.healpix_area(d['RA'], d['DEC']))
        np.savez(out_fn, edges=edges, **{f'{r}/NZ': NZ[r] for r in NZ}, **{f'{r}/mocks': ids[r] for r in NZ},
                 **{f'{r}/area': areas[r] for r in NZ})
        log(f'saved {out_fn}')
    theory = None
    lab = args.label or f'holi-kcore2-{b}'
    fn = os.path.join(args.out, f'ssc_{lab}.npz')
    if os.path.exists(fn):
        s = np.load(fn)
        r0 = next(r for r in args.regions if f'{r}/plin_k' in s.files)
        b1, f = s[f'{r0}/params'][0], s[f'{r0}/params'][3]
        theory = (s[f'{r0}/plin_k'], s[f'{r0}/plin_P_damped_normalised'], b1 + f)
        print(f'theory: A P_lin (damped, normalised) of {fn} [{r0}], radial long mode (b1 + f) delta with b1 {b1:.2f} f {f:.2f}')
    chi = dc.comoving_distance(edges)
    deltas = {}
    for r in NZ:
        area = areas[r] * (chi[-1] ** 3 - chi[0] ** 3) / 3 / (chi[-1] - chi[0]) / 1.0   # sr x mean r^2 -> (Mpc/h)^2
        deltas[r] = analyse(NZ[r], ids[r], edges, r, theory, area)
    if len(deltas) == 2:
        (ra, da), (rb, db) = deltas.items()
        common = np.intersect1d(ids[ra], ids[rb])
        ia = np.searchsorted(ids[ra], common)
        ib = np.searchsorted(ids[rb], common)
        ta, tb = NZ[ra][ia].sum(1), NZ[rb][ib].sum(1)
        print(f'\n{ra}-{rb} correlation of the shell counts over {len(common)} common mocks: {np.corrcoef(ta, tb)[0, 1]:+.3f} '
              '(~0 for independent volumes; >> 0 if the caps share parent-box modes)')
        cz = np.array([np.corrcoef(da[ia, j], db[ib, j])[0, 1] for j in range(da.shape[1])])
        print('  per z bin: ' + ' '.join(f'{x:+.2f}' for x in cz))


if __name__ == '__main__':
    main()
