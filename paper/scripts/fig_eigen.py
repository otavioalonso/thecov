"""Figure: sorted eigenvalues of the whitened sample covariance C^{-1/2} S C^{-1/2} (LRG1 and QSO NGC, k < 0.3), for three covariance
models, divided by the median of the same statistic for Gaussian ensembles of N vectors drawn from the model itself
(null test, whose large-n limit is the Marchenko-Pastur law), with its 95% envelope. Against the 95% envelope of the same statistic for Gaussian ensembles of N vectors drawn from
the model itself (null test; the Marchenko-Pastur law is its large-n limit)."""
from common import *

rng = np.random.default_rng(1)
NNULL = 200
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4), sharey=True)
for ax, b in zip(axes, ('LRG1', 'QSO')):
    d = holi(b, 'NGC')
    V, N, n = d['V'], len(d['V']), d['V'].shape[1]
    null = np.array([np.sort(np.linalg.eigvalsh(np.cov(rng.standard_normal((N, n)).T)))[::-1] for _ in range(NNULL)])
    lo, med, hi = np.percentile(null, [2.5, 50, 97.5], axis=0)
    r = np.arange(1, n + 1)
    ax.fill_between(r, lo / med, hi / med, color=GRID, lw=0, label='null ensembles, 95%')
    ax.axhline(1, color=INK2, lw=0.7)
    for m in ('G_local', 'G', 'rec'):
        W, _ = whitened_cov(V, d['C'][m])
        lam = np.sort(np.linalg.eigvalsh(W))[::-1]
        ax.plot(r, lam / med, color=MODEL_COLOR[m], lw=1.2, label=MODEL_LABEL[m])
    ax.set_xscale('log')
    ax.set_xlabel('rank')
    ax.set_title(f'{b} NGC')
axes[0].set_ylabel(r'$\lambda_i(C^{-1/2} S\, C^{-1/2}) / \lambda_i^{\rm null}$')
axes[0].set_yscale('log')
axes[0].set_ylim(0.85, 3.2)
axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_yticks([0.9, 1, 1.5, 2, 3], ['0.9', '1', '1.5', '2', '3'])
axes[1].legend(loc='upper right', fontsize=6.5)
fig.tight_layout(w_pad=0.6)
save(fig, 'eigen')
