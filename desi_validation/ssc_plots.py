"""Figures for the SSC + discreteness prediction (output of ssc_check.py) against the mocks.

    python -m desi_validation.ssc_plots --bin LRG1 --label holi-kcore2-LRG1

Reads <--dir>/report_data_<label>.npz (mocks, Gaussian covariance) and <--out>/ssc_<label>.npz and writes
to <--out>/ssc_plots_<label>/:

  variance.png       var(mocks) / diag(model) per multipole and region, for each covariance model
  offdiag_<r>.png    off-diagonal excess (C_mocks - C_Gauss)_ij / sqrt(C_ii C_jj), averaged over k_j in a
                     band, against the predicted non-Gaussian terms; blocks 00, 02, 22
  eigen.png          eigenvalues of C_model^-1/2 C_mocks C_model^-1/2 (largest first) against the
                     Wishart (Marchenko-Pastur) expectation for N mocks
  params.png         joint variance ratios of (A, A2, alpha, SN) at kmax 0.2 and 0.3
  budget.png         diag(C_SSC) / diag(C) and diag(C_disc) / diag(C) per multipole
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from desi_validation.compare_kernel import get  # noqa: E402
from desi_validation.rank1_excess import param_ratios  # noqa: E402

INK, INK2, GRID, BAND = '#0b0b0b', '#52514e', '#e4e3df', '#f0efec'
GAUSS = '#8a8984'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#4a3aa7']           # fixed categorical order
REGIONS = ('NGC', 'SGC', 'GCcomb')
ELLS = (0, 2, 4)

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'lines.linewidth': 1.6, 'figure.dpi': 130, 'savefig.bbox': 'tight'})


def models(s, r, nola=True):
    """{name: non-Gaussian covariance added to the Gaussian one} present for region r (fixed order)"""
    g = lambda key: s[f'{r}/{key}'] if f'{r}/{key}' in s.files else None
    ssc, ssc_nola, disc, disc_loc = g('C_ssc'), g('C_ssc_noLA'), g('C_disc'), g('C_disc_local')
    out = {}
    if ssc is not None:
        out['SSC'] = ssc
    if ssc is not None and disc_loc is not None:
        out['SSC + disc (local)'] = ssc + disc_loc
    if ssc is not None and disc is not None:
        out['SSC + disc'] = ssc + disc                      # the recommended model (window-convolved discreteness)
    t0 = g('C_T0_response')
    if t0 is None and g('C_T0_snake') is not None:
        t0 = g('C_T0_snake') + g('C_T0_star')
    if ssc is not None and disc is not None and t0 is not None:
        out['SSC + disc + T0 response'] = ssc + disc + t0
    if nola and ssc_nola is not None:
        out['SSC (no LA)'] = ssc_nola
    return out


COLOR = {'SSC': 0, 'SSC + disc (local)': 2, 'SSC + disc': 1, 'SSC + disc + T0 response': 3, 'SSC (no LA)': 4,
         'disc 4-pt': 1, 'T0 response': 3}                # colour follows the model, not its rank


def color(i, name):
    return SERIES[COLOR.get(name, i) % len(SERIES)]


def style(i, name):
    return dict(color=color(i, name), ls='--' if 'no LA' in name else '-', label=name)


def fig_variance(data, fn):
    fig, axes = plt.subplots(len(data), 3, figsize=(10, 2.5 * len(data)), sharex=True, squeeze=False)
    for row, (r, (V, C, k, M)) in enumerate(data.items()):
        nb, N = len(k), len(V)
        vm = V.var(0, ddof=1)
        for col, l in enumerate(ELLS):
            ax = axes[row, col]
            s = slice(col * nb, (col + 1) * nb)
            e = np.sqrt(2 / (N - 1))
            ax.fill_between(k, 1 - e, 1 + e, color=BAND, lw=0, label=f'mock noise (N={N})')
            ax.axhline(1, color=INK2, lw=0.8)
            ax.plot(k, vm[s] / np.diag(C)[s], color=GAUSS, lw=1.2, label='Gaussian (kernel)')
            for i, (name, X) in enumerate(M.items()):
                ax.plot(k, vm[s] / np.diag(C + X)[s], **style(i, name))
            ax.set_title(f'{r}  ell={l}', loc='left', fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel('var(mocks) / diag(model)')
            if row == len(data) - 1:
                ax.set_xlabel('k [h/Mpc]')
            ax.set_ylim(0.5, 1.6)
    axes[0, 0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Diagonal: mock variance over model variance (1 = right)', x=0.01, y=1.02, ha='left', color=INK)
    fig.savefig(fn)
    plt.close(fig)


def fig_offdiag(r, V, C, k, M, fn, bands=((0.08, 0.12), (0.18, 0.22), (0.28, 0.32)), blocks=((0, 0), (0, 1), (1, 1))):
    nb, N = len(k), len(V)
    Cm = np.cov(V.T)
    d = np.sqrt(np.diag(C))
    norm = np.outer(d, d)
    fig, axes = plt.subplots(len(bands), len(blocks), figsize=(10, 2.6 * len(bands)), sharex=True, squeeze=False)
    for row, (lo, hi) in enumerate(bands):
        js = np.flatnonzero((k >= lo) & (k <= hi))
        for col, (p, q) in enumerate(blocks):
            ax = axes[row, col]
            I = p * nb + np.arange(nb)
            J = q * nb + js
            mask = np.ones((nb, len(js)), bool)
            if p == q:                                              # drop the diagonal and first neighbours
                mask = np.abs(np.arange(nb)[:, None] - js[None, :]) > 1
            avg = lambda X: np.array([X[i, J][mask[ii]].mean() if mask[ii].any() else np.nan for ii, i in enumerate(I)])
            ex = avg((Cm - C) / norm)
            nj = mask.sum(1).clip(1)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.errorbar(k, ex, yerr=1 / np.sqrt(N * nj), fmt='o', ms=2.5, color=INK, elinewidth=0.6,
                        label='mocks - Gaussian')
            for i, (name, X) in enumerate(M.items()):
                ax.plot(k, avg(X / norm), **style(i, name))
            ax.axvspan(lo, hi, color=BAND, lw=0)
            ax.set_title(f'block {2 * p}{2 * q}, k_j in [{lo}, {hi}]', loc='left', fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel('excess corr. coefficient')
            if row == len(bands) - 1:
                ax.set_xlabel('k_i [h/Mpc]')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{r}: off-diagonal excess over the Gaussian covariance, '
                 '(C_ij - C^G_ij) / sqrt(C^G_ii C^G_jj), averaged over k_j in the shaded band',
                 x=0.01, y=1.02, ha='left', color=INK)
    fig.savefig(fn)
    plt.close(fig)


def wishart_quantiles(n, N, n_top, n_sim=20, seed=0):
    rng = np.random.default_rng(seed)
    ev = np.zeros((n_sim, n))
    for t in range(n_sim):
        X = rng.standard_normal((N, n))
        ev[t] = np.sort(np.linalg.eigvalsh(np.cov(X.T)))[::-1]
    return np.percentile(ev[:, :n_top], [16, 50, 84], axis=0)


def whitened_eigs(Cm, M):
    L = np.linalg.cholesky(M)
    Li = np.linalg.inv(L)
    return np.sort(np.linalg.eigvalsh(Li @ Cm @ Li.T))[::-1]


def fig_eigen(data, fn, n_top=12):
    fig, axes = plt.subplots(1, len(data), figsize=(3.6 * len(data), 3.2), squeeze=False)
    for col, (r, (V, C, k, M)) in enumerate(data.items()):
        ax = axes[0, col]
        Cm = np.cov(V.T)
        q = wishart_quantiles(C.shape[0], len(V), n_top)
        x = np.arange(1, n_top + 1)
        ax.fill_between(x, q[0], q[2], color=BAND, lw=0, label='Wishart 16-84%')
        ax.plot(x, q[1], color=INK2, lw=0.8)
        ax.plot(x, whitened_eigs(Cm, C)[:n_top], 'o-', color=GAUSS, ms=3, lw=1, label='Gaussian (kernel)')
        for i, (name, X) in enumerate(M.items()):
            try:
                ev = whitened_eigs(Cm, C + X)[:n_top]
            except np.linalg.LinAlgError:
                continue                                            # not positive definite
            ax.plot(x, ev, marker='o', ms=3, lw=1, **style(i, name))
        ax.set_title(r, loc='left', fontsize=9, color=INK)
        ax.set_xlabel('eigenvalue rank')
        if col == 0:
            ax.set_ylabel('eig of C_model^-1/2 C_mocks C_model^-1/2')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Largest whitened eigenvalues (a rank-1 excess shows up as an outlier at rank 1)',
                 x=0.01, y=1.02, ha='left', color=INK)
    fig.savefig(fn)
    plt.close(fig)


def fig_params(data, fn, kmaxs=(0.2, 0.3)):
    names = ['A', 'A2', 'alpha', 'SN']
    fig, axes = plt.subplots(len(kmaxs), len(data), figsize=(3.6 * len(data), 2.6 * len(kmaxs)), sharey=True, squeeze=False)
    for col, (r, (V, C, k, M)) in enumerate(data.items()):
        nb, N = len(k), len(V)
        for row, kmax in enumerate(kmaxs):
            ax = axes[row, col]
            entries = [('Gaussian (kernel)', C, GAUSS)]
            for i, (name, X) in enumerate(M.items()):
                if np.linalg.eigvalsh(C + X).min() > 0:
                    entries.append((name, C + X, color(i, name)))
            w = 0.8 / len(entries)
            e = np.sqrt(2 / (N - 1))
            ax.axhspan(1 - e, 1 + e, color=BAND, lw=0)
            ax.axhline(1, color=INK2, lw=0.8)
            for j, (name, Mx, c) in enumerate(entries):
                ax.bar(np.arange(4) + (j - (len(entries) - 1) / 2) * w, param_ratios(V, Mx, k, nb, kmax), width=w * 0.9,
                       color=c, label=name)
            ax.set_xticks(np.arange(4), names)
            ax.set_title(f'{r}, kmax {kmax}', loc='left', fontsize=9, color=INK)
            ax.grid(axis='x', visible=False)
            if col == 0:
                ax.set_ylabel('var(mocks) / var(model)')
    axes[0, 0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Joint parameter variance ratios (fit of A, A2, alpha, SN to each mock)', x=0.01, y=1.02, ha='left', color=INK)
    fig.savefig(fn)
    plt.close(fig)


def fig_budget(data, s, fn):
    fig, axes = plt.subplots(len(data), 3, figsize=(10, 2.4 * len(data)), sharex=True, squeeze=False)
    for row, (r, (V, C, k, M)) in enumerate(data.items()):
        nb = len(k)
        dC = np.diag(C)
        terms = [(name, s[f'{r}/{key}']) for name, key in (('SSC', 'C_ssc'),
                                                         ('disc 4-pt', 'C_disc'), ('T0 response', 'C_T0_response'))
                 if f'{r}/{key}' in s.files]
        vm = V.var(0, ddof=1)
        for col, l in enumerate(ELLS):
            ax = axes[row, col]
            sl = slice(col * nb, (col + 1) * nb)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.plot(k, vm[sl] / dC[sl] - 1, color=INK, lw=0.8, alpha=0.6, label='mocks / Gaussian - 1')
            for i, (name, X) in enumerate(terms):
                ax.plot(k, np.diag(X)[sl] / dC[sl], color=color(i, name), label=name)
            ax.set_title(f'{r}  ell={l}', loc='left', fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel('fraction of Gaussian diag')
            if row == len(data) - 1:
                ax.set_xlabel('k [h/Mpc]')
    axes[0, 0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Diagonal budget: each non-Gaussian term relative to the Gaussian variance', x=0.01, y=1.02, ha='left', color=INK)
    fig.savefig(fn)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--label', default='holi-kcore2-LRG1')
    ap.add_argument('--mode', default='kernel')
    ap.add_argument('--dir', default=os.path.expanduser('~/thecov_desi/holi_v3_mock173'))
    ap.add_argument('--out', default='/global/cfs/cdirs/desicollab/users/oalves/thecov_validation')
    ap.add_argument('--ssc', default=None, help='ssc npz (default <out>/ssc_<label>.npz)')
    args = ap.parse_args()

    z = np.load(os.path.join(args.dir, f'report_data_{args.label}.npz'))
    s = np.load(args.ssc or os.path.join(args.out, f'ssc_{args.label}.npz'))
    od = os.path.join(args.out, f'ssc_plots_{args.label}')
    os.makedirs(od, exist_ok=True)
    data = {}
    for r in REGIONS:
        M = models(s, r)
        if not M:
            continue
        V, C, k = get(z, args.bin, r, 1, args.mode)
        data[r] = (V, C, k, M)
    fig_variance(data, os.path.join(od, 'variance.png'))
    for r, (V, C, k, M) in data.items():
        fig_offdiag(r, V, C, k, M, os.path.join(od, f'offdiag_{r}.png'))
    fig_eigen(data, os.path.join(od, 'eigen.png'))
    fig_params(data, os.path.join(od, 'params.png'))
    fig_budget(data, s, os.path.join(od, 'budget.png'))
    print(f'figures in {od}')


if __name__ == '__main__':
    main()
