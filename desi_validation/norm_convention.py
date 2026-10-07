"""Which normalisation does the estimator realise per mock? Answered from the dump alone (no catalogues, seconds):

    cd ~/thecov && python -m desi_validation.norm_convention --bin LRG1 --label holi-kcore2-LRG1
    cd ~/thecov && python -m desi_validation.norm_convention --bin QSO  --label holi-kcore2-QSO

Per mock i the files store norm_i and num_shotnoise_i = sum_d w_d^2 + alpha_i^2 sum_r w_r^2. The second is dominated by
the data term, so delta_nsn = num_shotnoise_i / <num_shotnoise> - 1 ~ the realised fluctuation of the (w^2-weighted)
number of galaxies: it traces alpha_i (delta^M) with ~0.1 % Poisson noise. The candidate normalisations then predict,
for the regression ln norm = a + s ln num_shotnoise + residual:

    norm = alpha x fixed randoms integral  ('alpha')         : s = 1, residual ~ Poisson only (<~ 0.1 %)
    norm = alpha^2 sum_c R_c^2             ('randoms')       : s = 2, residual ~ 0 (alpha^2 term of num_shotnoise)
    norm = alpha sum_c D_c R_c             ('data-randoms')  : s ~ 2, residual = sigma(delta^W - delta^M) + Poisson of
                                                                the data in norm (~0.1-0.3 %)

and the regression of P_hat_l(k) on delta_norm and on delta_nsn separates the m- and m^2-weighted long modes.
Also prints sigma(delta_norm), sigma(delta_nsn), the correlation, and what thecov.ssc predicts for each convention is
in ssc_check's 'normalisation per mock' block (this script needs no pair counts).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation.compare_kernel import get  # noqa: E402

K_SHOW = (0.03, 0.05, 0.1, 0.15, 0.2, 0.3)


def regress(y, x):
    """slope, intercept, residual rms and correlation of y on x (1d arrays over mocks)"""
    x0, y0 = x - x.mean(), y - y.mean()
    s = np.sum(x0 * y0) / np.sum(x0 ** 2)
    res = y0 - s * x0
    return s, y.mean() - s * x.mean(), res.std(ddof=2), np.corrcoef(x, y)[0, 1]


def analyse(tag, V, k, norm, nsn, nb):
    dn = norm / norm.mean() - 1.0
    ds = nsn / nsn.mean() - 1.0
    print(f'\n{tag}: {len(V)} mocks')
    print(f'  sigma(delta_norm) {dn.std(ddof=1) * 100:.3f}%   sigma(delta_nsn) {ds.std(ddof=1) * 100:.3f}%   '
          f'corr {np.corrcoef(dn, ds)[0, 1]:+.3f}')
    s, _, rms, r = regress(np.log(norm), np.log(nsn))
    print(f'  ln norm on ln num_shotnoise: slope {s:.3f}, residual rms {rms * 100:.3f}%  '
          f'(alpha-only: slope 1, rms <~ 0.1%; randoms^2: slope 2, rms ~ 0; data x randoms: slope ~2, rms 0.1-0.3%)')
    s2, _, rms2, _ = regress(np.log(norm) - 2 * np.log(nsn), np.log(nsn))
    print(f'  rms of ln(norm / num_shotnoise^2) {np.std(np.log(norm) - 2 * np.log(nsn), ddof=1) * 100:.3f}%, '
          f'of ln(norm / num_shotnoise) {np.std(np.log(norm) - np.log(nsn), ddof=1) * 100:.3f}%')
    idx = [int(np.argmin(np.abs(k - kk))) for kk in K_SHOW]
    Pm = V.mean(0)
    for l, lab in ((0, 'P0'), (1, 'P2')):
        for name, x in (('delta_norm', dn), ('delta_nsn ', ds)):
            sl = []
            for i in idx:
                y = V[:, l * nb + i] / Pm[l * nb + i]
                sl.append(regress(y, x)[0])
            print(f'  {lab} on {name}: slope at k ~ {K_SHOW}: ' + ' '.join(f'{v:+.2f}' for v in sl))
        # joint regression on both (separates the m- and m^2-weighted means if they differ)
        X = np.stack([dn - dn.mean(), ds - ds.mean()], 1)
        sl = []
        for i in idx:
            y = V[:, l * nb + i] / Pm[l * nb + i]
            b = np.linalg.lstsq(X, y - y.mean(), rcond=None)[0]
            sl.append(b)
        print(f'  {lab} on both (norm, nsn):            ' + ' '.join(f'({b[0]:+.2f},{b[1]:+.2f})' for b in sl))
    # amplitude mode (template P0 over 0.02 < k < 0.3) vs delta_norm, as in the report
    sel = (k > 0.02) & (k < 0.3)
    T = Pm[:nb][sel]
    A = ((V[:, :nb][:, sel] - Pm[:nb][sel]) @ T) / (T @ T)
    print(f'  P0 amplitude mode: rms {A.std(ddof=1) * 100:.3f}%, corr with delta_norm {np.corrcoef(A, dn)[0, 1]:+.3f}, '
          f'with delta_nsn {np.corrcoef(A, ds)[0, 1]:+.3f}; slope on delta_norm {regress(A, dn)[0]:+.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--label', default='holi-kcore2-LRG1')
    ap.add_argument('--mode', default='kernel')
    ap.add_argument('--dir', default=os.path.expanduser('~/thecov_desi/holi_v3_mock173'))
    args = ap.parse_args()
    z = np.load(os.path.join(args.dir, f'report_data_{args.label}.npz'))
    b = args.bin
    for r in ('NGC', 'SGC'):
        V, C, k = get(z, b, r, 1, args.mode)
        nb = len(k)
        norm = np.asarray(z[f'{b}/{r}/x1/norm'], float).ravel()
        key = f'{b}/{r}/x1/num_shotnoise'
        if key not in z.files or norm.size != len(V):
            print(f'{b} {r}: per-mock norm / num_shotnoise not in the dump ({norm.size} norms, {len(V)} mocks): '
                  're-run dump_report_data (it saves both) or use check_norm_catalogs.py')
            continue
        nsn = np.asarray(z[key], float).ravel()
        analyse(f'{b} {r}', V, k, norm, nsn, nb)


if __name__ == '__main__':
    main()
