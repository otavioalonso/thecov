"""Recompute the candidate normalisations from the catalogues of a few mocks and compare, mock by mock, with the
`norm` stored in the spectrum files: the candidate whose ratio to the file's norm does not scatter across mocks is the
convention the pipeline used. NERSC, debug queue (~2 min per mock and cap with one random file):

    cd ~/thecov && sbatch -N 1 -C cpu -q debug -t 00:30:00 -J normcat \\
      -o /global/cfs/cdirs/desicollab/users/oalves/thecov_validation/norm_catalogs_LRG1.log \\
      --wrap "source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\\$HOME/thecov:\\$PYTHONPATH && cd ~/thecov && python -u -m desi_validation.check_norm_catalogs --bin LRG1 --label holi-kcore2-LRG1 --mocks 0 1 2 3 4 5 6 7"

Candidates (cells of `--cellsize` Mpc/h, as jaxpower's compute_fkp2_normalization with cellsize=10; CIC there, NGP here,
which changes the value by ~1e-3 but not its mock-to-mock fluctuation):
    DR      = alpha sum_c D_c R_c / V_c                        jaxpower split=None ('data-randoms': alpha and the data)
    RRsplit = alpha sum_c R1_c R2_c / V_c x (N/N1)(N/N2)      jaxpower split=seed  (randoms only: alpha once; with
                                                              redshifts shuffled from the data, the randoms also carry
                                                              the mock's radial n(z))
    NXr     = alpha sum_r w_r^2 NX_r                            nbar-based, randoms  ('alpha')
    NXd     = sum_d w_d^2 NX_d                                  nbar-based, data
and num_shotnoise = sum_d w_d^2 + alpha^2 sum_r w_r^2 as a check that these are the pipeline's catalogues and weights.
Also reports whether the randoms (positions, redshifts, NX) differ from mock to mock.
Output: table per mock, then per candidate the mean and rms over mocks of ln(candidate / norm_file) and the correlation
of its fluctuation with the file's; saved to <--out>/norm_catalogs_<label>.npz.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def cell_sum(pa, wa, pb, wb, cs, lo):
    """sum_c A_c B_c / cs^3 for weighted points painted with NGP on cells of side cs"""
    n = np.int64(1 << 20)

    def keys(p):
        i = np.floor((p - lo) / cs).astype(np.int64)
        return (i[:, 0] * n + i[:, 1]) * n + i[:, 2]
    ua, ia = np.unique(keys(pa), return_inverse=True)
    ub, ib = np.unique(keys(pb), return_inverse=True)
    A, B = np.bincount(ia, weights=wa), np.bincount(ib, weights=wb)
    _, ja, jb = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    return float(np.sum(A[ja] * B[jb]) / cs ** 3)


def fingerprint(cat):
    return {c: (len(cat[c]), float(np.sum(np.asarray(cat[c][:20000], float)))) for c in ('RA', 'DEC', 'Z', 'NX') if c in cat}


def file_norm(z, b, r, mock, paths, tracer, zr):
    """(norm, num_shotnoise) of this mock from the dump (by mock id) or from its spectrum file"""
    key = f'{b}/{r}/x1/mock_ids'
    if z is not None and key in z.files:
        ids = [str(x) for x in z[key]]
        tag = f'mock{mock}'
        if tag in ids:
            i = ids.index(tag)
            return float(z[f'{b}/{r}/x1/norm'][i]), float(z[f'{b}/{r}/x1/num_shotnoise'][i]), 'dump'
    fns = [fn for fn in paths.spectra_fns(tracer, zr, r) if f'mock{mock}' in fn.split(os.sep)]
    if not fns:
        return np.nan, np.nan, 'missing'
    s = dc.read_spectra(fns[:1], kmax=0.4, rebin=5)
    return float(s['norm'][0]), float(s['num_shotnoise'][0]), os.path.basename(os.path.dirname(fns[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--label', default='holi-kcore2-LRG1', help='dump with per-mock norm and mock_ids (optional)')
    ap.add_argument('--dir', default=os.path.expanduser('~/thecov_desi/holi_v3_mock173'))
    ap.add_argument('--out', default='/global/cfs/cdirs/desicollab/users/oalves/thecov_validation')
    ap.add_argument('--mocks', type=int, nargs='+', default=list(range(8)))
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--n-random-files', type=int, default=1)
    ap.add_argument('--cellsize', type=float, default=10.0)
    ap.add_argument('--cs-version', default='holi-v3-altmtl')
    args = ap.parse_args()

    b = args.bin
    tracer, zr = dc.TRACER_SPECS[b]
    fn = os.path.join(args.dir, f'report_data_{args.label}.npz')
    z = np.load(fn) if os.path.exists(fn) else None
    log(f'{b}: dump {"found" if z is not None else "not found"} ({fn}); mocks {args.mocks}')
    out_fn = os.path.join(args.out, f'norm_catalogs_{args.label}.npz')
    os.makedirs(args.out, exist_ok=True)
    rows = {r: [] for r in args.regions}
    prints = {r: [] for r in args.regions}
    names = ['norm_file', 'nsn_file', 'alpha', 'DR', 'RRsplit', 'NXr', 'NXd', 'nsn_recomputed']
    rng = np.random.default_rng(0)
    for r in args.regions:
        for mock in args.mocks:
            paths = dc.Paths(kind='holi_v3', mock=mock)
            paths.loader = 'auto'
            paths.cs_version, paths.cs_parent_version = args.cs_version, 'data-dr2-v2'
            try:
                rc = dc.load_region(paths, b, r, n_random_files=args.n_random_files)
            except Exception as e:
                log(f'{r} mock {mock}: catalogue load failed: {type(e).__name__}: {e}')
                continue
            d, ra = rc.data, rc.randoms
            wd, wr = dc.total_weight(d), dc.total_weight(ra)
            alpha = float(wd.sum() / wr.sum())
            pd = dc.sky_to_cartesian(d['RA'], d['DEC'], d['Z'])
            pr = dc.sky_to_cartesian(ra['RA'], ra['DEC'], ra['Z'])
            lo = np.minimum(pd.min(0), pr.min(0)) - 1.0
            cs = args.cellsize
            DR = alpha * cell_sum(pd, wd, pr, wr, cs, lo)
            half = rng.random(len(wr)) < 0.5
            RRsplit = alpha * cell_sum(pr[half], wr[half], pr[~half], wr[~half], cs, lo) * (wr.sum() / wr[half].sum()) * (wr.sum() / wr[~half].sum())
            nx_d = np.asarray(d.get('NX', np.full(len(wd), np.nan)), float)
            nx_r = np.asarray(ra.get('NX', np.full(len(wr), np.nan)), float)
            NXd, NXr = float(np.sum(wd ** 2 * nx_d)), float(alpha * np.sum(wr ** 2 * nx_r))
            nsn_re = float(np.sum(wd ** 2) + alpha ** 2 * np.sum(wr ** 2))
            nf, sf, src = file_norm(z, b, r, mock, paths, tracer, zr)
            row = [nf, sf, alpha, DR, RRsplit, NXr, NXd, nsn_re]
            rows[r].append(row)
            prints[r].append(fingerprint(ra))
            log(f'{r} mock {mock} ({src}): norm_file {nf:.6g} | DR/norm {DR / nf:.5f} RRsplit/norm {RRsplit / nf:.5f} '
                f'NXr/norm {NXr / nf:.5f} NXd/norm {NXd / nf:.5f} | nsn_recomputed/nsn_file {nsn_re / sf:.5f} | alpha {alpha:.5g}; '
                f'{len(wd)} data, {len(wr)} randoms')
            np.savez(out_fn, **{f'{rr}/rows': np.array(v, float) for rr, v in rows.items() if v}, names=np.array(names),
                     mocks=np.array(args.mocks))
            del rc, d, ra, pd, pr
        R = np.array(rows[r], float)
        if len(R) < 2:
            continue
        print(f'\n{b} {r}: {len(R)} mocks. sigma(delta_norm_file) {np.std(np.log(R[:, 0]), ddof=1) * 100:.3f}%')
        for j, name in enumerate(names[3:], 3):
            lr = np.log(R[:, j] / R[:, 0])
            dc_, dn = np.log(R[:, j]) - np.log(R[:, j]).mean(), np.log(R[:, 0]) - np.log(R[:, 0]).mean()
            corr = np.sum(dc_ * dn) / np.sqrt(np.sum(dc_ ** 2) * np.sum(dn ** 2))
            print(f'  {name:14s}: mean ratio to norm_file {np.exp(lr.mean()):.5f}, rms of ln ratio {lr.std(ddof=1) * 100:.3f}%, '
                  f'corr of fluctuations {corr:+.3f}, sigma(delta) {np.std(np.log(R[:, j]), ddof=1) * 100:.3f}%')
        lr = np.log(R[:, 7] / R[:, 1])
        print(f'  num_shotnoise recomputed / file: mean {np.exp(lr.mean()):.5f}, rms {lr.std(ddof=1) * 100:.3f}% '
              '(~0 means these are the pipeline catalogues and weights)')
        fp = prints[r]
        same = {c: all(f.get(c) == fp[0].get(c) for f in fp) for c in fp[0]}
        print(f'  randoms identical across mocks? {same}  (False for Z: redshifts shuffled from each mock\'s data)')
        print('  reading: the candidate with rms of ln ratio << sigma(delta_norm_file) and corr ~ 1 is the convention; '
              'DR -> norm_kind data-randoms; RRsplit or NXr -> alpha (randoms only); NXd -> data (not implemented in thecov.ssc)')


if __name__ == '__main__':
    main()
