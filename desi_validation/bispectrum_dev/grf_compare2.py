"""GRF sample variances against class 1 + class 2 on the sphere (and the box), for one or two sphere sizes.

usage: grf_compare2.py  measurements.npz theory_c12.npz [measurements2.npz theory2.npz]  -> bispectrum_grf_c12.png
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grf_sphere import labels  # noqa: E402


def ratios(mfile, tfile):
    M, T = np.load(mfile), np.load(tfile)
    labs = [tuple(int(v) for v in l) for l in M['labels']]
    assert labs == labels()
    sph = M['sphere']
    vs = sph.var(0, ddof=1)
    e = np.sqrt(2 / (len(sph) - 1))
    C1, C2 = T['C_class1'], T['C_class2']
    d1, d2 = C1.diagonal(), C2.diagonal()
    r12 = vs / (d1 + d2)
    r1 = vs / d1
    Cg = np.cov(sph, rowvar=False)
    Ct = C1 + C2
    cg = Cg / np.sqrt(np.outer(Cg.diagonal(), Cg.diagonal()))
    ct = Ct / np.sqrt(np.outer(Ct.diagonal(), Ct.diagonal()))
    return dict(labels=labs, r12=r12, r1=r1, err=e * r12, c2_over_c1=d2 / d1, corr_grf=cg, corr_theory=ct,
                n_real=len(sph), R=float(M['R']), L=float(M['L']), N=int(M['N']))


def summarize(res, name):
    r, e = res['r12'], res['err']
    labs = np.array(res['labels'])
    is000 = np.flatnonzero(labs[:, 0] == 0)  # label = (L, i, j): B000 elements first, then B202
    n000 = len(is000)
    b000, b202 = slice(0, n000), slice(n000, len(r))
    print(f'{name}: R = {res["R"]:.0f}, {res["n_real"]} realisations')
    for sl, lab in ((b000, 'B000'), (b202, 'B202')):
        chi2 = np.sum(((r[sl] - 1) / e[sl]) ** 2)
        print(f'   {lab}: var/(C1+C2) mean {r[sl].mean():.3f} +- {e[sl].mean() / np.sqrt(sl.stop - sl.start):.3f}, '
              f'min {r[sl].min():.3f} max {r[sl].max():.3f}, chi2 {chi2:.1f} / {sl.stop - sl.start}')
    tri = np.triu_indices(len(r), 1)
    big = np.abs(res['corr_theory'][tri]) > 0.1
    print(f'   off-diagonal |corr| > 0.1 (n = {big.sum()}): GRF {res["corr_grf"][tri][big].mean():.3f} vs theory '
          f'{res["corr_theory"][tri][big].mean():.3f}, rms diff {np.std(res["corr_grf"][tri][big] - res["corr_theory"][tri][big]):.3f}')


def main(args):
    sets = [(ratios(args[0], args[1]), None)]
    if len(args) >= 4:
        sets.append((ratios(args[2], args[3]), None))
    sets = [(res, f'R = {res["R"]:.0f} Mpc/h') for res, _ in sets]
    for res, name in sets:
        summarize(res, name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]
    labs = sets[0][0]['labels']
    x = np.arange(len(labs))
    cols = ['C3', 'C0']
    for k, (res, name) in enumerate(sets):
        off = (k - 0.5 * (len(sets) - 1)) * 0.3
        ax.errorbar(x + off, res['r12'], res['err'], fmt='o' if k == 0 else 's', color=cols[k], ms=5,
                    label=f'{name}: var(GRF) / (class 1 + class 2)')
        ax.plot(x + off, res['r1'], 'x', color=cols[k], alpha=0.5, label=f'{name}: var(GRF) / class 1 only' if k == 0 else None)
    ax.axhline(1, color='gray', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l[0]}{l[1]}{l[2]}\n({l[3]},{l[4]})' for l in labs], fontsize=7)
    ax.set_ylabel('var(GRF) / var(theory)')
    ax.set_title('Gaussian fields on a sphere centred on the observer, isotropic P, dk = 0.04')
    ax.legend(fontsize=8, loc='upper left')
    ax = axes[1]
    for k, (res, name) in enumerate(sets):
        tri = np.triu_indices(len(labs), 1)
        ax.plot(res['corr_theory'][tri], res['corr_grf'][tri], '.', ms=4, color=cols[k], label=name)
    lim = 0.7
    ax.plot([-lim, lim], [-lim, lim], color='gray', lw=0.8)
    ax.set_xlabel('theory correlation (class 1 + 2)')
    ax.set_ylabel('GRF correlation')
    ax.set_title('off-diagonal correlations')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / 'bispectrum_grf_c12.png', dpi=130)
    np.savez(HERE / 'grf_results_c12.npz', **{f'{k}_{n}': v for k, (res, n) in enumerate(sets) for n, v in res.items()
                                                 if not isinstance(v, (list, tuple))})


if __name__ == '__main__':
    main(sys.argv[1:])
