"""Figure: variance of the mocks over the model variance, per multipole, for the Gaussian term with the local and the
xi-kernel windows and for the recommended model. Ratios averaged over groups of 4 bins (Delta k = 0.02); the error bars
are the Wishart scatter of that average under the model, including the correlation of neighbouring bins."""
import sys
from common import *

TRACERS = [a for a in sys.argv[1:] if a in ('LRG1', 'QSO')] or ['LRG1', 'QSO']   # make passes one; a notebook does both
GROUP = 4
SHOW = ('G_local', 'G', 'rec')
for b in TRACERS:
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.6), sharex=True, sharey=True)
    for row, cap in enumerate(('NGC', 'SGC')):
        d = holi(b, cap)
        V, k, nb, N = d['V'], d['k'], len(d['k']), len(d['V'])
        S = np.cov(V.T)
        for j, m in enumerate(SHOW):
            C = d['C'][m]
            r = np.diag(S) / np.diag(C)
            rho2 = corr(C) ** 2
            for col in range(3):
                ax = axes[row, col]
                xs, ys, es = [], [], []
                for g in range(nb // GROUP):
                    idx = col * nb + g * GROUP + np.arange(GROUP)
                    xs.append(k[idx - col * nb].mean())
                    ys.append(r[idx].mean())
                    es.append(np.sqrt(2 * rho2[np.ix_(idx, idx)].sum() / (N - 1)) / GROUP)
                off = (j - 1) * 0.0025
                ax.errorbar(np.array(xs) + off, ys, es, fmt='o', color=MODEL_COLOR[m], ms=3, lw=0.9, capsize=0,
                            label=MODEL_LABEL[m])
        for col in range(3):
            ax = axes[row, col]
            ax.axhline(1, color=INK2, lw=0.7)
            if row == 0:
                ax.set_title(rf'$\ell = {2 * col}$')
            if col == 0:
                ax.set_ylabel(f'{b} {cap}\n' + r'$\hat\sigma^2_{\rm mocks} / \sigma^2_{\rm model}$')
            if row == 1:
                ax.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$')
    axes[0, 0].set_ylim(0.82, 1.32)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    save(fig, f'diagonals_{b}')
