"""Multipole-block diagnostics of a grf_randoms variant against thecov: chi2/n per block and the conditional
variance of l = 4 given l = 0, 2 (the LRG1 excess of HANDOFF 7.9 / review 3.E, here on exactly Gaussian fields).

    python -m desi_validation.grf_randoms_blocks --dir ~/thecov/grfr_v2 --bin LRG1 --region NGC --tag _rsd --mode kernel

For realisations V (N x 3nb) and a model C (thecov, 'true' input power), with mean subtracted:
  chi2/n per block S:  mean_n r_S^T C_SS^-1 r_S / (|S| (1 - 1/N))
  conditional l = 4:   r_4|02 = V_4 - C_4a C_aa^-1 (V_a - mean), C_4|02 = C_44 - C_4a C_aa^-1 C_a4,
                       ratio = mean_n r^T C_4|02^-1 r / (nb (1 - 1/N));  1 if thecov's (4|0,2) structure is right.
Bootstrap errors over realisations.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from desi_validation.grf_randoms_plot import load


def chi2n(V, C):
    N = len(V)
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, (V - V.mean(0)).T)
    return (z ** 2).sum(0).mean() / (V.shape[1] * (1 - 1 / N))


def conditional(V, C, nb, ells):
    i4 = ells.index(4)
    a = np.r_[0:i4 * nb, (i4 + 1) * nb:len(ells) * nb]
    s4 = np.arange(i4 * nb, (i4 + 1) * nb)
    Caa_inv = np.linalg.inv(C[np.ix_(a, a)])
    B = C[np.ix_(s4, a)] @ Caa_inv
    R = V[:, s4] - (V[:, a] - V[:, a].mean(0)) @ B.T
    Cc = C[np.ix_(s4, s4)] - B @ C[np.ix_(a, s4)]
    return chi2n(R, Cc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--tag', default='')
    ap.add_argument('--mode', default='kernel')
    ap.add_argument('--which', default='true', choices=['true', 'masked'])
    ap.add_argument('--kmax', type=float, default=None)
    ap.add_argument('--n-boot', type=int, default=200)
    args = ap.parse_args()
    g, th = load(args.dir, args.bin, args.region, args.tag)
    t = th[args.mode]
    V, k, ells = g['V'], g['k'], g['ells']
    nb = len(k)
    C = t[f'C_{args.which}']
    if args.kmax:
        keep = np.concatenate([np.flatnonzero(k <= args.kmax) + e * nb for e in range(len(ells))])
        V, C = V[:, keep], C[np.ix_(keep, keep)]
        nb = int(np.sum(k <= args.kmax))
    N = len(V)
    rng = np.random.default_rng(1)
    boots = [rng.integers(0, N, N) for _ in range(args.n_boot)]
    print(f'{args.bin} {args.region}{args.tag} [{args.mode}, {args.which} P]: N = {N}, nb = {nb}, ells {ells}')
    blocks = {'all': np.arange(len(ells) * nb)}
    for i, e in enumerate(ells):
        blocks[f'l={e}'] = np.arange(i * nb, (i + 1) * nb)
    for i, e in enumerate(ells):
        for j, e2 in enumerate(ells):
            if j > i:
                blocks[f'l={e},{e2}'] = np.r_[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb]
    for name, idx in blocks.items():
        c0 = chi2n(V[:, idx], C[np.ix_(idx, idx)])
        bs = np.std([chi2n(V[b][:, idx], C[np.ix_(idx, idx)]) for b in boots])
        print(f'  chi2/n {name:8s} {c0:.3f} +- {bs:.3f}')
    if 4 in ells and len(ells) == 3:
        c0 = conditional(V, C, nb, ells)
        bs = np.std([conditional(V[b], C, nb, ells) for b in boots])
        print(f'  var(l=4 | l=0,2) / thecov: {c0:.3f} +- {bs:.3f}')
        for lo, hi in ((0.02, 0.1), (0.1, 0.2), (0.2, 0.3)):
            m = (k[:nb] >= lo) & (k[:nb] < hi)
            if m.sum() < 3:
                continue
            idx = np.concatenate([np.flatnonzero(m) + e * nb for e in range(len(ells))])
            print(f'    k {lo}-{hi}: var(l=4 | l=0,2) / thecov '
                  f'{conditional(V[:, idx], C[np.ix_(idx, idx)], int(m.sum()), ells):.3f}')


if __name__ == '__main__':
    main()
