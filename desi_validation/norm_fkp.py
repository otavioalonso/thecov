"""What drives each mock's estimator normalisation? A test of per-realisation FKP weights and randoms that copy each
mock's redshifts, from the data catalogues only. NERSC, debug queue (~10-15 min for all mocks of one tracer):

    cd ~/thecov && sbatch -N 1 -C cpu -q debug -t 00:30:00 -J normfkp -o $OUT/norm_fkp_LRG1.log --wrap "source \\
      /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\\$HOME/thecov:\\$PYTHONPATH \\
      && cd ~/thecov && python -u -m desi_validation.norm_fkp --bin LRG1 --workers 32"

Background. The pipeline's norm is alpha sum_c D_c R_c / V_c (jaxpower, cellsize 10). The holi randoms take their
redshifts (and NX) from the data of their own mock, so alpha R(x) follows each mock's radial profile, and the FKP weights
1 / (1 + NX P0) follow each mock's n(z). Then a density change delta of the shell changes n w_FKP = n / (1 + n P0) by
delta / (1 + n P0) only: for LRG1 (n P0 ~ 5) norm hardly responds to the long modes, for QSO (n P0 ~ 0.2) almost fully.
thecov's local-average statistics assume fixed weights and fixed randoms.

For each mock and cap, norm is predicted from its data as  sum_d w_d n_R(z_d) A(theta_d)  with A the angular density of
the randoms (from the randoms mock, healpix) and n_R the randoms' radial profile, under four assumptions:
    own-FKP / own-randoms     w = WEIGHT x WEIGHT_FKP of this mock; n_R = this mock's weighted n(z)      (the pipeline?)
    own-FKP / fixed-randoms   n_R = the mean weighted n(z) over mocks; alpha = sum_d w / const
    fixed-FKP / own-randoms   w = WEIGHT / (1 + NX_ref(z) P0) with NX_ref the mean over mocks
    fixed-FKP / fixed-randoms (thecov's norm_kind='data-randoms' with fixed weights)
and compared mock by mock with the norm in the spectrum files: the right model has corr ~ 1 and slope ~ 1. Also printed:
whether NX follows each mock's own n(z), the FKP P0, and the m^2-weighted mean of WEIGHT_FKP (the response factor of
norm to the shell-mean density with per-mock FKP weights). Output: <--out>/norm_fkp_<bin>.npz.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc  # noqa: E402

T0 = time.time()
_G = {}


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def one_mock(args):
    """per-z-bin sums of one mock's data and its per-galaxy (z bin, angular pixel, weights)"""
    mock, cap = args
    g = _G
    paths = dc.Paths(kind='holi_v3', mock=mock, catalog_dir=g['catalog_dir'])
    try:
        d = dc.read_catalog(paths.data_fn(g['tracer'], cap), ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP', 'NX'],
                            paths.h5_group, required=('RA', 'DEC', 'Z', 'WEIGHT'))
    except Exception as e:
        return mock, cap, f'{type(e).__name__}: {e}'
    import healpy as hp
    z = np.asarray(d['Z'], float)
    keep = (z > g['edges'][0]) & (z < g['edges'][-1])
    z = z[keep]
    W = np.asarray(d['WEIGHT'], float)[keep]
    fkp = np.asarray(d['WEIGHT_FKP'], float)[keep] if 'WEIGHT_FKP' in d else None
    nx = np.asarray(d['NX'], float)[keep] if 'NX' in d else None
    iz = np.clip(np.searchsorted(g['edges'], z) - 1, 0, len(g['edges']) - 2)
    pix = hp.ang2pix(g['nside'], np.asarray(d['RA'], float)[keep], np.asarray(d['DEC'], float)[keep], lonlat=True)
    nb = len(g['edges']) - 1
    out = dict(N=np.bincount(iz, minlength=nb).astype(float), SW=np.bincount(iz, W, nb))
    if fkp is not None:
        w = W * fkp
        out['S1'] = np.bincount(iz, w, nb)
        A = g['A'][pix]
        out['SA'] = np.bincount(iz, w * A, nb)                       # sum_d w_d A(theta_d) per z bin
        out['SA_W'] = np.bincount(iz, W * A, nb)                     # same with WEIGHT only (for the fixed-FKP models)
        out['fkp_m'] = float(np.sum(w * fkp) / np.sum(w))            # w-weighted mean of WEIGHT_FKP
    if nx is not None:
        out['NXsum'] = np.bincount(iz, nx, nb)
        if fkp is not None:
            ok = (nx > 0) & (fkp > 0) & (fkp < 1)
            out['P0'] = float(np.median((1 / fkp[ok] - 1) / nx[ok])) if ok.any() else np.nan
    return mock, cap, out


def file_norms(args, cap):
    fn = os.path.join(args.dir, f'report_data_{args.label}.npz')
    if os.path.exists(fn):
        z = np.load(fn)
        key = f'{args.bin}/{cap}/x1/'
        if key + 'norm' in z.files:
            ids = [int(str(s)[4:]) if str(s).startswith('mock') else int(s) for s in z[key + 'mock_ids']]
            return dict(zip(ids, np.asarray(z[key + 'norm'], float))), fn
    tracer, zr = dc.TRACER_SPECS[args.bin]
    paths = dc.Paths(kind='holi_v3', mock=0)
    fns = paths.spectra_fns(tracer, zr, cap)
    s = dc.read_spectra(fns, kmin=0.02, kmax=0.3, rebin=5)
    ids = [int(os.path.basename(os.path.dirname(f))[4:]) for f in fns]
    return dict(zip(ids, s['norm'])), paths.spectra_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--label', default=None, help='dump with per-mock norm (default holi-kcore2-<bin>); else the spectrum files')
    ap.add_argument('--dir', default=os.path.expanduser('~/thecov_desi/holi_v3_mock173'))
    ap.add_argument('--out', default='/global/cfs/cdirs/desicollab/users/oalves/thecov_validation')
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--mocks', type=int, nargs='*', default=None, help='default: every mock with a norm')
    ap.add_argument('--randoms-mock', type=int, default=173, help='mock whose randoms give the angular footprint A')
    ap.add_argument('--nside', type=int, default=128)
    ap.add_argument('--dz', type=float, default=0.002)
    ap.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument('--catalog-dir', default=None, help=argparse.SUPPRESS)       # tests
    ap.add_argument('--loader', default='auto', help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.label = args.label or f'holi-kcore2-{args.bin}'
    import healpy as hp
    tracer, zr = dc.TRACER_SPECS[args.bin]
    edges = np.arange(zr[0], zr[1] + 0.5 * args.dz, args.dz)
    chi = dc.comoving_distance(edges)
    vol = (chi[1:] ** 3 - chi[:-1] ** 3) / 3.0                          # per sr
    catalog_dir = args.catalog_dir or dc.Paths().catalog_dir
    out = {}
    for cap in args.regions:
        norms, src = file_norms(args, cap)
        mocks = sorted(norms) if args.mocks is None else [m for m in args.mocks if m in norms]
        log(f'{args.bin} {cap}: {len(mocks)} mocks with a norm ({src})')
        # angular density of the randoms (footprint, veto holes at the pixel scale, completeness of the weights)
        paths = dc.Paths(kind='holi_v3', mock=args.randoms_mock, catalog_dir=catalog_dir)
        paths.loader, paths.cs_version, paths.cs_parent_version = args.loader, 'holi-v3-altmtl', 'data-dr2-v2'
        rc = dc.load_region(paths, args.bin, cap, n_random_files=1)
        A = np.bincount(hp.ang2pix(args.nside, np.asarray(rc.randoms['RA'], float), np.asarray(rc.randoms['DEC'], float),
                                   lonlat=True), minlength=hp.nside2npix(args.nside)).astype(float)
        A /= A[A > 0].mean()
        del rc
        _G.update(tracer=tracer, edges=edges, nside=args.nside, A=A, catalog_dir=catalog_dir)
        tasks = [(m, cap) for m in mocks]
        if args.workers > 1:
            from multiprocessing import get_context
            with get_context('fork').Pool(args.workers) as pool:
                res = pool.map(one_mock, tasks, chunksize=4)
        else:
            res = [one_mock(t) for t in tasks]
        bad = [(m, r) for m, _, r in res if isinstance(r, str)]
        if bad:
            log(f'{cap}: {len(bad)} mocks failed, e.g. {bad[0]}')
        res = [(m, r) for m, _, r in res if not isinstance(r, str)]
        ids = np.array([m for m, _ in res])
        nf = np.array([norms[m] for m in ids])
        stack = lambda key: np.array([r[key] for _, r in res]) if key in res[0][1] else None   # noqa: E731
        N, SW, S1, SA, SA_W, NXs = (stack(k) for k in ('N', 'SW', 'S1', 'SA', 'SA_W', 'NXsum'))
        d_file = np.log(nf) - np.log(nf).mean()
        print(f'\n{args.bin} {cap}: {len(ids)} mocks; sigma(delta_norm) in the files {d_file.std(ddof=1) * 100:.3f}%')
        if NXs is not None:
            nxbar = np.where(N > 0, NXs / np.maximum(N, 1), np.nan)               # mean NX per z bin, per mock
            own = SW / vol                                                     # weighted counts per volume (x const)
            good = np.all(np.isfinite(nxbar), axis=0) & (N.min(0) > 50)
        if NXs is not None and good.any():
            r_own = np.log(nxbar[:, good] / own[:, good])
            r_mean = np.log(nxbar[:, good] / own[:, good].mean(0))
            print(f'  NX vs the mock\'s own weighted n(z): rms over mocks of ln(NX / own n(z)) {np.mean(r_own.std(0)) * 100:.2f}%, '
                  f'of ln(NX / mean n(z)) {np.mean(r_mean.std(0)) * 100:.2f}%  (first << second: NX follows each mock\'s n(z))')
        P0s = [r.get('P0', np.nan) for _, r in res]
        if np.isfinite(P0s).any():
            print(f'  FKP P0 from (1/WEIGHT_FKP - 1)/NX: median over mocks {np.nanmedian(P0s):.4g}')
        if S1 is None:
            print('  no WEIGHT_FKP in the data files: the FKP models cannot be built')
            continue
        fkp_m = np.array([r['fkp_m'] for _, r in res])
        print(f'  m^2-weighted mean of WEIGHT_FKP (response of norm to the shell density with per-mock FKP): {fkp_m.mean():.3f}')
        nR_own = S1 / vol                                                      # randoms copy each mock's weighted n(z)
        nR_fix = nR_own.mean(0)
        P0 = np.nanmedian(P0s) if np.isfinite(P0s).any() else None
        models = {'own-FKP / own-randoms': np.sum(SA * nR_own, 1),
                  'own-FKP / fixed-randoms': S1.sum(1) * np.sum(SA * nR_fix, 1)}
        if P0 is not None and NXs is not None:
            nx_ref = np.nanmean(np.where(N > 0, NXs / np.maximum(N, 1), np.nan), 0)
            ffix = 1 / (1 + np.nan_to_num(nx_ref) * P0)                       # fixed FKP weight per z bin
            S1f, SAf = SW * ffix, SA_W * ffix
            nRf_own = S1f / vol
            models['fixed-FKP / own-randoms'] = np.sum(SAf * nRf_own, 1)
            models['fixed-FKP / fixed-randoms'] = S1f.sum(1) * np.sum(SAf * nRf_own.mean(0), 1)
        print(f'  {"model of norm":28s} {"corr":>6s} {"slope":>6s} {"sigma":>8s}   (file vs model, mock by mock)')
        for name, M in models.items():
            dm = np.log(M) - np.log(M).mean()
            c = np.corrcoef(dm, d_file)[0, 1]
            slope = np.cov(dm, d_file)[0, 1] / dm.var(ddof=1)
            print(f'  {name:28s} {c:6.3f} {slope:6.3f} {dm.std(ddof=1) * 100:7.3f}%')
            out[f'{cap}/{name}'] = M
        out.update({f'{cap}/mocks': ids, f'{cap}/norm_file': nf, f'{cap}/fkp_m': fkp_m, f'{cap}/N': N, f'{cap}/S1': S1})
        print('  reading: the model with corr ~ 1 and slope ~ 1 is how the pipeline normalises; the fixed-FKP / fixed-randoms '
              'sigma is what thecov\'s local-average statistics assume (times the long-mode content of the mocks)')
    fn = os.path.join(args.out, f'norm_fkp_{args.bin}.npz')
    os.makedirs(args.out, exist_ok=True)
    np.savez(fn, edges=edges, **out)
    log(f'saved {fn}')


if __name__ == '__main__':
    main()
