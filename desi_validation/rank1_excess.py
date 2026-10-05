"""Decompose mocks - thecov into a low-rank response term fitted on the OFF-diagonal, and look at what
is left on the diagonal (the clean test of the Gaussian/window term).

    python -m desi_validation.rank1_excess holi-kcore2-LRG1:kernel holi-fillprod-LRG1:fill holi-kcore-LRG1:random-density
    python -m desi_validation.rank1_excess --bin QSO holi-kcore2-QSO:kernel

Why. A term fully correlated across k (super-sample response R(k)R(k') sigma_b^2, or the discreteness
constant sum w^4 / norm^2) contributes to the variance of every bin whatever its width, while the Gaussian
variance of a bin of width dk scales as 1/dk only if adjacent bins are uncorrelated. thecov's adjacent
0.005 bins are correlated at ~0.4 by the window, so in the a + b (dk/0.005) split of the report a fully
correlated term shows up ~2/3 as "a" (Gaussian-like). That split cannot separate window errors from
super-sample terms. The off-diagonal elements far from the diagonal (|dk| >= 0.03, where the windowed
Gaussian covariance is ~0) can: there the excess is read directly.

Model fitted on the off-diagonal of each (ell, ell') block, elements |i - j| >= sep (weights 1/(C_ii C_jj)):
    Cm - C = A_ll' P0(k) P0(k')  [+ const for (0, 0)],
then the diagonal residual var(mocks) / diag(C + fitted) per k range and ell, chi2/n before and after,
and the joint template-parameter variance ratios (A, A2, alpha, SN; as in the report) before and after.
sqrt(A_00) is the coherent fractional P0 amplitude scatter, sqrt(A_22) the coherent P2 scatter in units of P0.

Reads DIR/report_data_LABEL.npz (--dir, default ~/thecov_desi/holi_v3_mock173); LABEL:MODE as compare_kernel.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from desi_validation.compare_kernel import get, chi2

K_RANGES = [(0.02, 0.06), (0.06, 0.12), (0.12, 0.2), (0.2, 0.3)]


def fit_rank1(Cm, C, P0, nb, sep=6):
    """A_ll' (and the (0,0) constant) from the off-diagonal; returns (A, C + fitted term)."""
    D = Cm - C
    idx = np.arange(nb)
    far = np.abs(idx[:, None] - idx[None, :]) >= sep
    T = np.outer(P0, P0)[far]
    A, M = {}, C.copy()
    for a in range(3):
        for b in range(a, 3):
            sa, sb = slice(a * nb, (a + 1) * nb), slice(b * nb, (b + 1) * nb)
            Db = D[sa, sb][far]
            w = 1.0 / np.outer(np.diag(C)[sa], np.diag(C)[sb])[far]
            X = np.vstack([T, np.ones_like(T)]).T if (a, b) == (0, 0) else T[:, None]
            v = np.linalg.lstsq(X * np.sqrt(w)[:, None], Db * np.sqrt(w), rcond=None)[0]
            A[(a, b)] = v
            blk = v[0] * np.outer(P0, P0) + (v[1] if len(v) > 1 else 0.0)
            M[sa, sb] += blk
            if a != b:
                M[sb, sa] += blk.T
    return A, M


def param_ratios(V, C, k, nb, kmax):
    """var(mocks) / thecov prediction of the joint GLS estimates of A, A2, alpha, SN (report Sec. 9)."""
    Pm = V.mean(0)
    P0, P2, P4 = Pm[:nb], Pm[nb:2 * nb], Pm[2 * nb:]
    I = np.concatenate([np.flatnonzero(k <= kmax + 1e-9) + j * nb for j in range(3)])
    lnk = np.log(k)
    dP = lambda P: -np.gradient(P, lnk)
    z = np.zeros(nb)
    T = np.stack([np.concatenate([P0, P2, P4]), np.concatenate([z, P2, z]),
                  np.concatenate([dP(P0), dP(P2), dP(P4)]), np.concatenate([np.ones(nb), z, z])], 1)[I]
    Ci = np.linalg.inv(C[np.ix_(I, I)])
    Fi = np.linalg.inv(T.T @ Ci @ T)
    est = (Fi @ T.T @ Ci @ (V[:, I] - V[:, I].mean(0)).T).T
    return est.var(0, ddof=1) / np.diag(Fi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cases', nargs='+', help='LABEL:MODE')
    ap.add_argument('--dir', default=os.path.expanduser('~/thecov_desi/holi_v3_mock173'))
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--sep', type=int, default=6, help='fit the off-diagonal |i - j| >= sep bins (0.005 each)')
    args = ap.parse_args()
    cases = [(c.rsplit(':', 1)[0], c.rsplit(':', 1)[1]) for c in args.cases]
    for r in ('NGC', 'SGC', 'GCcomb'):
        print(f'\n{args.bin} {r}: residual var(mocks)/diag(C + rank-1 term) for k in',
              ' '.join(f'{lo}-{hi}' for lo, hi in K_RANGES))
        for label, mode in cases:
            try:
                z = np.load(os.path.join(args.dir, f'report_data_{label}.npz'))
                V, C, k = get(z, args.bin, r, 1, mode)
            except (KeyError, FileNotFoundError):
                continue
            nb = len(k)
            Cm = np.cov(V.T)
            P0 = V[:, :nb].mean(0)
            A, M = fit_rank1(Cm, C, P0, nb, sep=args.sep)
            res = []
            for l in range(3):
                s = slice(l * nb, (l + 1) * nb)
                rr = np.diag(Cm)[s] / np.diag(M)[s]
                res.append('[' + ' '.join(f'{rr[(k >= lo) & (k < hi)].mean():.3f}' for lo, hi in K_RANGES) + ']')
            n = V.shape[1]
            allidx = np.arange(n)
            print(f'  {label:22s} {mode:15s} chi2/n {chi2(V, C, allidx):.4f} -> {chi2(V, M, allidx):.4f} | '
                  f'l=0 {res[0]} l=2 {res[1]} l=4 {res[2]} | sigma_P0 {np.sqrt(max(A[(0, 0)][0], 0)) * 100:.2f}% '
                  f'sigma_P2/P0 {np.sqrt(max(A[(1, 1)][0], 0)) * 100:.2f}% r02 '
                  f'{A[(0, 1)][0] / np.sqrt(max(A[(0, 0)][0] * A[(1, 1)][0], 1e-30)):+.2f} const00 {A[(0, 0)][1]:.0f}')
            for kmax in (0.2, 0.3):
                print(f'      kmax {kmax}: joint var ratios (A, A2, alpha, SN) {np.round(param_ratios(V, C, k, nb, kmax), 2)}'
                      f' -> with rank-1 term {np.round(param_ratios(V, M, k, nb, kmax), 2)}')


if __name__ == '__main__':
    main()
