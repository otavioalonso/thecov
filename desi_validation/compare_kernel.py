"""Compare covariance variants from dump_report_data outputs: chi2/n per k range, variance ratios per k
range, and the bin-width decomposition  var(mocks)/var(thecov) - 1 = a + b (dk / 0.005).

    python -m desi_validation.compare_kernel holi-kcore2-LRG1:kernel holi-fillprod-LRG1:fill holi-kcore-LRG1:random-density

Each argument is LABEL:MODE, read from DIR/report_data_LABEL.npz (DIR: --dir, default
~/thecov_desi/holi_v3_mock173). MODE is the per-region mode name ('kernel', 'fill', 'random-density');
for GCcomb the matching 'combined-regions [MODE]' (or 'combined-regions' for random-density) is used.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

K_RANGES = [(0.02, 0.2), (0.02, 0.3), (0.02, 0.06), (0.06, 0.12), (0.12, 0.2), (0.2, 0.3)]
V_RANGES = [(0.02, 0.06), (0.06, 0.12), (0.12, 0.2), (0.2, 0.3)]
FACTORS = (1, 2, 4)


def get(z, b, r, f, mode):
    p = f'{b}/{r}/x{f}'
    if r == 'GCcomb':
        m = 'combined-regions' if mode == 'random-density' else f'combined-regions [{mode}]'
        mode = m if p + f'/C/{m}' in z else 'combined-regions'     # a single-mode run has no [mode] suffix
    return z[p + '/V'].astype(float), z[p + f'/C/{mode}'], z[p + '/k']


def chi2(V, C, idx):
    L = np.linalg.cholesky(C[np.ix_(idx, idx)])
    zz = np.linalg.solve(L, (V[:, idx] - V[:, idx].mean(0)).T)
    return (zz ** 2).sum(0).mean() / (len(idx) * (1 - 1 / len(V)))


def kidx(k, lo, hi, nl=3):
    s = np.flatnonzero((k >= lo - 1e-9) & (k <= hi + 1e-9))
    return np.concatenate([s + j * len(k) for j in range(nl)])


def ratios(z, b, r, mode):
    out = []
    for f in FACTORS:
        V, C, k = get(z, b, r, f, mode)
        rr = V.var(0, ddof=1) / np.diag(C)
        per = 4 // f
        out.append(rr.reshape(3, len(k) // per, per).mean(-1))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cases', nargs='+', help='LABEL:MODE')
    ap.add_argument('--dir', default=os.path.expanduser('~/thecov_desi/holi_v3_mock173'))
    ap.add_argument('--bin', default='LRG1')
    args = ap.parse_args()
    cases = []
    for c in args.cases:
        label, mode = c.rsplit(':', 1)
        cases.append((label, mode, np.load(os.path.join(args.dir, f'report_data_{label}.npz'))))
    b = args.bin
    print('chi2/n (x1) for k ranges', ' '.join(f'{lo}-{hi}' for lo, hi in K_RANGES),
          '| var ratio P0, P2 for', ' '.join(f'{lo}-{hi}' for lo, hi in V_RANGES))
    for r in ('NGC', 'SGC', 'GCcomb'):
        for label, mode, z in cases:
            try:
                V, C, k = get(z, b, r, 1, mode)
            except KeyError:
                continue
            c2 = [chi2(V, C, kidx(k, lo, hi)) for lo, hi in K_RANGES]
            rv, nb = V.var(0, ddof=1) / np.diag(C), len(k)
            v0 = [rv[:nb][(k >= lo) & (k < hi)].mean() for lo, hi in V_RANGES]
            v2 = [rv[nb:2 * nb][(k >= lo) & (k < hi)].mean() for lo, hi in V_RANGES]
            print(f'{r:6s} {label:22s} {mode:15s}', ' '.join(f'{x:.3f}' for x in c2), '| P0',
                  ' '.join(f'{x:.3f}' for x in v0), '| P2', ' '.join(f'{x:.3f}' for x in v2))
    print('\na (Gaussian-like) and b (per unit bin width), mean of NGC and SGC, '
          'k < 0.2 / 0.2-0.3, ell = 0, 2, 4')
    A = np.vstack([np.ones(len(FACTORS)), FACTORS]).T
    for label, mode, z in cases:
        try:
            R = np.mean([ratios(z, b, r, mode) for r in ('NGC', 'SGC')], 0)
        except KeyError:
            continue
        coef, *_ = np.linalg.lstsq(A, (R - 1).reshape(len(FACTORS), -1), rcond=None)
        ab = coef.reshape(2, *R.shape[1:])
        lo = get(z, b, 'NGC', 4, mode)[2] < 0.2
        print(f'{label:22s} {mode:15s} a:', ' '.join(f'{ab[0, l, lo].mean():+.3f}/{ab[0, l, ~lo].mean():+.3f}' for l in range(3)),
              ' b:', ' '.join(f'{ab[1, l, lo].mean():+.3f}/{ab[1, l, ~lo].mean():+.3f}' for l in range(3)))


if __name__ == '__main__':
    main()
