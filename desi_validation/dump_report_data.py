"""Compact data for the validation report: mock data vectors, thecov matrices and window/weight numbers,
so that every figure and table of the report can be made offline from a few MB.

    python -m desi_validation.dump_report_data                                    # holi v3 altmtl, LRG1 + QSO
    python -m desi_validation.dump_report_data --label abacus-complete --bins LRG1 \\
        --cs-version abacus-2ndgen-dr2-complete --spectra-dir <dir with mock*/> --mock 0
    python -m desi_validation.dump_report_data --label abacus-altmtl --bins LRG1 \\
        --cs-version abacus-2ndgen-dr2-altmtl --spectra-dir <dir with mock*/> --mock 0

Same environment and allocation as run_window_shape. Windows and 0.005 covariances come from the
caches of the earlier runs (same cache directories); the wider-bin covariances (x2, x4) reuse the cached
windows, and the naive window (each random's own weight, --naive) needs new pair counts (~80 s per region).

Output: OUT/report_data_<label>.npz (float32 mock vectors, float64 matrices) + .json (numbers). Upload both.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='holi-altmtl')
    ap.add_argument('--bins', nargs='+', default=['LRG1', 'QSO'])
    ap.add_argument('--modes', nargs='+', default=['random-density', 'nx'])
    ap.add_argument('--factors', nargs='+', type=int, default=[2, 4], help='wider bins: 0.005 x factor')
    ap.add_argument('--no-naive', action='store_true', help='skip the naive (own-weight) window covariance')
    ap.add_argument('--cs-version', default=None)
    ap.add_argument('--spectra-dir', default=None)
    ap.add_argument('--mock', type=int, default=None)
    ap.add_argument('--synthetic', default=None, help=argparse.SUPPRESS)     # testing on the synthetic set
    args = ap.parse_args()

    paths = dc.Paths(kind='holi_v3', mock=173)
    paths.catalog_names.update({'ELG_LOPnotqso': 'ELGnotqso', 'LRG+ELG_LOPnotqso': 'LRG+ELGnotqso'})
    paths.loader = 'auto'
    paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
    OUT = os.path.expanduser(f'~/thecov_desi/{paths.kind}_mock{paths.mock}')
    if args.cs_version or args.spectra_dir or args.mock is not None:     # same cache layout as run_window_shape
        paths.cs_version = args.cs_version or paths.cs_version
        paths.spectra_dir = args.spectra_dir or paths.spectra_dir
        paths.mock = paths.mock if args.mock is None else args.mock
        OUT = os.path.expanduser(f'~/thecov_desi/{paths.cs_version}_mock{paths.mock}')
    cfg = dataclasses.replace(pl.Config(), nw_modes=tuple(args.modes), naive=not args.no_naive,
                              coarse_check=True, coarse_factors=tuple(args.factors))
    if args.synthetic:
        paths = dc.Paths(catalog_dir=args.synthetic + '/catalogs', spectra_dir=args.synthetic + '/spectra', loader='files')
        dc.TRACER_SPECS.clear(); dc.TRACER_SPECS['TEST'] = ('LRG', (0.4, 0.6))
        cfg = dataclasses.replace(cfg, surface_density=150.0, kmax=0.1, kmin=0.02, target_near_pairs=3e8, n_sub_far=5000,
                                  fill_random_files=1, fill_nside=128)
        OUT, args.bins = args.synthetic + '/results_dump', ['TEST']
    log(f'{args.label}: catalogues {paths.cs_version} mock {paths.mock}, spectra {paths.spectra_dir}, caches {OUT}')

    arrays, meta = {}, dict(label=args.label, cs_version=paths.cs_version, mock=paths.mock, config=dataclasses.asdict(cfg),
                            bins={})

    def put_spec(prefix, s):
        arrays[f'{prefix}/V'] = s['vectors'].astype(np.float32)
        for key in ('k', 'k_edges', 'nmodes', 'norm', 'num_shotnoise'):
            if s.get(key) is not None:
                arrays[f'{prefix}/{key}'] = np.asarray(s[key])
        arrays[f'{prefix}/mock_ids'] = np.asarray(pl.mock_ids(s))

    for b in args.bins:
        D = pl.run_bin(paths, b, cfg, OUT, log=log, keep=True)
        if 'validation' not in D:
            log(f'{b}: no spectra, skipped'); continue
        mb = meta['bins'][b] = dict(ells=list(cfg.ells), gccomb=D['gccomb'], cases=[])
        for r, s in D['spectra'].items():
            put_spec(f'{b}/{r}/x1', s)
        for (r, mode), C in D['covariances'].items():
            arrays[f'{b}/{r}/x1/C/{mode}'] = C
            mb['cases'].append([r, mode, 1])
        for (r, mode, f), (s2, C2) in D['coarse'].items():
            if f'{b}/{r}/x{f}/V' not in arrays:
                put_spec(f'{b}/{r}/x{f}', s2)
            arrays[f'{b}/{r}/x{f}/C/{mode}'] = C2
            mb['cases'].append([r, mode, f])
        # weights and windows (item: <w^2> vs <w>^2)
        mb['weights'], mb['veff'], mb['tracer_info'] = {}, {}, {}
        for r, rc in D['regions'].items():
            wd = dc.weight_diagnostics(rc, b)
            mb['weights'][r] = {k: v for k, v in wd.items() if not k.endswith('columns')}
            mb['veff'][r] = dc.window_moments(rc, modes=tuple(m for m in args.modes if m != 'fill'),     # + 'own-weight' = naive
                                              surface_density_deg2=cfg.surface_density)
            log(f"{b} {r}: <w^2>/<w>^2 data {wd['data_w2_over_wmean2']:.4f} randoms {wd['randoms_w2_over_wmean2']:.4f}; "
                f"1/V_eff " + ', '.join(f'{m}: {v:.4g}' for m, v in mb['veff'][r].items()))
        mb['tracer_info'] = D.get('tracer_info', {})
        mb['catalogue_info'] = {r: {k: v for k, v in i.items() if np.size(v) < 100}
                                for r, i in D.get('catalogue_info', {}).items()}
        mb['chi2'] = {f'{r} | {m}': [v['chi2_ratio'], v['chi2_sigma']] for (r, m), v in D['validation'].items()}
        for key, v in mb['chi2'].items():
            log(f'{b} {key}: <chi2>/n {v[0]:.4f} ({v[1]:+.1f} sigma)')
        del D

    fn = os.path.join(OUT, f'report_data_{args.label}')
    np.savez_compressed(fn + '.npz', **arrays)
    pl.to_json([meta], fn + '.json')
    size = (os.path.getsize(fn + '.npz') + os.path.getsize(fn + '.json')) / 1e6
    log(f'saved {fn}.npz/.json ({size:.1f} MB); upload both')


if __name__ == '__main__':
    main()
