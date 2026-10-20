"""Three window treatments against Gaussian fields on the DESI randoms: reduction to a small summary file.

    python -m desi_validation.grf_compare --dir ~/thecov/grfr_v2 --sets LRG1:NGC:_rsd QSO:NGC:_rsd --out grf_compare.npz

Reads the fields (fields_<bin>_<cap><tag>_task*.npz) and the thecov outputs of the three window modes
(thecov_<bin>_<cap><tag>_{none,random-density,kernel}.npz, each with C_true and C_masked), and saves per set:
  k, n_fields, ells;
  fields/<l1l2>         diagonal of the fields' sample covariance block (00, 22, 02, ...);
  fields/00x22          diag(S00) diag(S22) (for the error of the cross-correlation coefficient);
  <mode>/<input>/<l1l2> the same diagonals of thecov's covariance (input: true or masked);
  <mode>/<input>/chi2n  mean chi2/n of the fields under the full covariance (fields' own mean subtracted);
  <mode>/<input>/wishart  Wishart deviance per dof, (tr(C^-1 S) - ln det(C^-1 S) - n) / n;
  <mode>/I_k_over_norm, <mode>/I_local_over_norm.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from desi_validation.grf_randoms import load_fields

MODES = ('none', 'random-density', 'kernel')
INPUTS = ('true', 'masked')


def reduce_set(directory, b, r, tag):
    g = load_fields(directory, b, r, tag)
    ells = tuple(int(e) for e in g['ells']) if 'ells' in g else (0,)
    V = np.concatenate([g[f'P_mask_{ell}'] for ell in ells], axis=1)
    nb, N = len(g['k']), len(V)
    S = np.cov(V.T)
    d = V - V.mean(0)
    blk = lambda M, a, b_: np.diag(M[a * nb:(a + 1) * nb, b_ * nb:(b_ + 1) * nb])
    pairs = [(a, b_) for a in range(len(ells)) for b_ in range(a, len(ells))]
    out = {'k': g['k'], 'n_fields': np.array(N), 'ells': np.array(ells)}
    for a, b_ in pairs:
        out[f'fields/{ells[a]}{ells[b_]}'] = blk(S, a, b_)
    if len(ells) > 1:
        out['fields/00x22'] = blk(S, 0, 0) * blk(S, 1, 1)
    n = S.shape[0]
    for mode in MODES:
        fn = os.path.join(directory, f'thecov_{b}_{r}{tag}_{mode}.npz')
        if not os.path.exists(fn):
            print(f'missing {fn}')
            continue
        z = np.load(fn, allow_pickle=True)
        assert int(z['n_fields']) <= N
        for key in ('I_k_over_norm', 'I_local_over_norm'):
            if key in z.files:
                out[f'{mode}/{key}'] = np.asarray(z[key])
        out[f'{mode}/n_fields_model'] = np.array(int(z['n_fields']))
        for inp in INPUTS:
            C = np.asarray(z[f'C_{inp}'], float)
            assert C.shape == S.shape, (fn, C.shape, S.shape)
            for a, b_ in pairs:
                out[f'{mode}/{inp}/{ells[a]}{ells[b_]}'] = blk(C, a, b_)
            Ci = np.linalg.inv(C)
            out[f'{mode}/{inp}/chi2n'] = np.array(np.einsum('mi,ij,mj->m', d, Ci, d).mean() / n)
            M = Ci @ S
            out[f'{mode}/{inp}/wishart'] = np.array((np.trace(M) - np.linalg.slogdet(M)[1] - n) / n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.expanduser('~/thecov/grfr_v2'))
    ap.add_argument('--sets', nargs='+', default=['LRG1:NGC:_rsd', 'QSO:NGC:_rsd'])
    ap.add_argument('--out', default='grf_compare.npz')
    args = ap.parse_args()
    res = {}
    for st in args.sets:
        b, r, tag = (st.split(':') + [''])[:3]
        for key, v in reduce_set(args.dir, b, r, tag).items():
            res[f'{b}_{r}{tag}/{key}'] = v
        print(st, 'fields', int(res[f'{b}_{r}{tag}/n_fields']))
    np.savez(args.out, **res)
    print('saved', args.out, len(res))


if __name__ == '__main__':
    main()
