"""Var_GRF / Var_thecov from grf_randoms.py outputs, with and without the box control variate.

    python -m desi_validation.grf_randoms_plot --dir ~/thecov_desi/holi_v3_mock173/LRG1/grf_randoms --out grf_randoms.png

Control variate: the same fields measured in the full periodic box, P_box(k), have an exactly known variance
sigma2_box = 2 sum P^2 / N_shell^2. For bin i, with Z_n = (X_n - mean X)^2 N/(N-1) (X: the windowed P0) and
U_nj = (Y_nj - <Y_j>_exact)^2 (Y: P_box in bins j = i-2..i+2, whose mean and variance are known exactly),
    Var_cv = mean(Z) - beta . (mean(U) - sigma2_box),   beta from the least-squares fit of Z on U.
mean(U) - sigma2_box has expectation zero, so the correction is unbiased for any fixed beta (beta estimated
from the same sample: bias O(1/N)). Errors: bootstrap over realisations (beta refitted in each resample).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

BIN = 'LRG1'
GROUP = 4                       # bins per plotted point (4 x 0.005 = 0.02 h/Mpc)


def load(d, b, r, tag=''):
    fns = sorted(glob.glob(os.path.join(d, f'fields_{b}_{r}{tag}_task*.npz')))
    zs = [np.load(f) for f in fns]
    g = {key: zs[0][key] for key in ('k', 'box_mean', 'box_var', 'norm')}
    g['ells'] = [int(e) for e in zs[0]['ells']] if 'ells' in zs[0] else [0]
    for key in ['P_box'] + [k_ for k_ in zs[0].files if k_.startswith('P_mask')]:
        g[key] = np.concatenate([z[key] for z in zs])
    g['V'] = np.concatenate([g[f'P_mask_{e}'] if f'P_mask_{e}' in g else g['P_mask'] for e in g['ells']], axis=1)
    th = {}
    for mode in ('random-density', 'kernel'):
        fn = os.path.join(d, f'thecov_{b}_{r}{tag}_{mode}.npz')
        if os.path.exists(fn):
            th[mode] = dict(np.load(fn))
    return g, th


def variances(X, Y, mu, s2, half=2):
    """per-bin sample variance of X and its control-variate version"""
    N, nb = X.shape
    Z = (X - X.mean(0)) ** 2 * N / (N - 1)
    U = (Y - mu) ** 2
    plain, cv = Z.mean(0), np.empty(nb)
    for i in range(nb):
        j = np.arange(max(0, i - half), min(nb, i + half + 1))
        A = U[:, j] - U[:, j].mean(0)
        beta = np.linalg.lstsq(A, Z[:, i] - Z[:, i].mean(), rcond=None)[0]
        cv[i] = Z[:, i].mean() - beta @ (U[:, j].mean(0) - s2[j])
    return plain, cv


def grouped(v, ref):
    n = len(v) // GROUP * GROUP
    return (v[:n] / ref[:n]).reshape(-1, GROUP).mean(1)


def analyse(g, th, n_boot=300, seed=1):
    V, Y, mu, s2 = g['V'], g['P_box'], g['box_mean'], g['box_var']
    N, nb, ells = len(V), len(g['k']), g['ells']
    rng = np.random.default_rng(seed)
    out = dict(N=N, ells=ells, k=grouped(g['k'], np.ones_like(g['k'])), box_check=grouped(Y.var(0, ddof=1), s2))
    boots = [rng.integers(0, N, N) for _ in range(n_boot)]
    for mode, t in th.items():
        for which in ('true', 'masked'):
            ref = t[f'var_{which}']
            for ei, ell in enumerate(ells):
                sl = slice(ei * nb, (ei + 1) * nb)
                v = V[:, sl].var(0, ddof=1)
                bp = np.array([grouped(V[i][:, sl].var(0, ddof=1), ref[sl]) for i in boots])
                out[(mode, which, ell)] = dict(plain=grouped(v, ref[sl]), plain_err=bp.std(0))
    return out


def plot(results, fn, tag=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    caps = list(results)
    ells = results[caps[0]]['ells']
    fig, axes = plt.subplots(len(ells), len(caps), figsize=(5.2 * len(caps), 3.6 * len(ells)), squeeze=False, sharex=True)
    colors = {'random-density': '#c0392b', 'kernel': '#1f5fa8'}
    labels = {'random-density': r'local window $m^2$', 'kernel': r'$\xi$-kernel window'}
    for col, cap in enumerate(caps):
        res = results[cap]
        k, dk = res['k'], 0.0012
        for row, ell in enumerate(ells):
            ax = axes[row, col]
            for mode in ('random-density', 'kernel'):
                if (mode, 'true', ell) not in res:
                    continue
                d = res[(mode, 'true', ell)]
                ax.errorbar(k + (dk if mode == 'kernel' else -dk), d['plain'], d['plain_err'], fmt='o', color=colors[mode],
                            ms=4.5, capsize=0, lw=1, label=labels[mode])
            ax.axhline(1, color='0.4', lw=0.8, zorder=0)
            ax.set_ylabel(rf'Var$_{{\rm GRF}}$ / Var$_{{\rm thecov}}$  ($P_{ell}$)')
            if row == 0:
                ax.set_title(f'{BIN} {cap}{tag}: {res["N"]} Gaussian fields on the DESI randoms', fontsize=10)
            if row == len(ells) - 1:
                ax.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(fn, dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=os.path.expanduser('~/thecov_desi/holi_v3_mock173/LRG1/grf_randoms'))
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--out', default='grf_randoms.png')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    global BIN
    BIN = args.bin
    results = {}
    for cap in ('NGC', 'SGC'):
        if glob.glob(os.path.join(args.dir, f'fields_{args.bin}_{cap}{args.tag}_task*.npz')):
            results[cap] = analyse(*load(args.dir, args.bin, cap, args.tag))
    for cap, res in results.items():
        print(f'{cap}: N = {res["N"]}; box check Var(P_box)/exact (0.02-wide groups):',
              ' '.join(f'{x:.3f}' for x in res['box_check']))
        for key, d in res.items():
            if isinstance(key, tuple):
                print(f'  {key[0]:15s} {key[1]:6s} ell={key[2]}', ' '.join(f'{x:.3f}' for x in d['plain']),
                      '| err', ' '.join(f'{x:.3f}' for x in d['plain_err']),
                      f'| mean {np.mean(d["plain"]):.3f} +- {np.sqrt(np.sum(d["plain_err"] ** 2)) / len(d["plain"]):.3f}')
    plot(results, args.out, args.tag)
    print('saved', args.out)


if __name__ == '__main__':
    main()
