"""Figure: (a) distribution of chi^2 of the 859 LRG1 NGC mocks (k < 0.3, n = 168) under three covariance models, against
the distribution expected if the model were exact; (b, c) mean chi^2 / n as a function of k_max, LRG1 and QSO, both caps,
with the +-1 sigma band of the mean for N mocks under the model."""
from scipy.stats import chi2 as chi2dist
from common import *

MODELS = ('G_local', 'G', 'rec')
fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.3), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
d = holi('LRG1', 'NGC')
V, N, n = d['V'], len(d['V']), d['V'].shape[1]
ax = axes[0]
bins = np.linspace(110, 330, 45)
for m in MODELS:
    c2 = chi2(V, d['C'][m]) * N / (N - 1)            # about the sample mean: rescale to n degrees of freedom
    ax.hist(c2, bins=bins, histtype='step', color=MODEL_COLOR[m], lw=1.2, label=MODEL_LABEL[m])
    ax.axvline(c2.mean(), color=MODEL_COLOR[m], lw=0.9, ymax=0.12)
x = np.linspace(bins[0], bins[-1], 300)
ax.plot(x, chi2dist.pdf(x, n) * N * (bins[1] - bins[0]), color=INK, lw=0.9, ls='--', label=rf'$\chi^2_{{{n}}}$')
ax.set_xlabel(r'$\chi^2$ (LRG1 NGC, $k < 0.3$)')
ax.set_ylabel('mocks per bin')
ax.set_title('(a)')
h, l = ax.get_legend_handles_labels()

KMAX = np.arange(0.06, 0.301, 0.02)
for ax, b in zip(axes[1:], ('LRG1', 'QSO')):
    for cap, ls in (('NGC', '-'), ('SGC', '--')):
        d = holi(b, cap)
        V, k, N = d['V'], d['k'], len(d['V'])
        for m in MODELS:
            y = [chi2(V, d['C'][m], kidx(k, km)).mean() / (len(kidx(k, km)) * (N - 1) / N) for km in KMAX]
            ax.plot(KMAX, y, color=MODEL_COLOR[m], ls=ls, lw=1.2)
    nn = np.array([len(kidx(k, km)) for km in KMAX])
    ax.fill_between(KMAX, 1 - np.sqrt(2 / (nn * N)), 1 + np.sqrt(2 / (nn * N)), color=GRID, lw=0)
    ax.axhline(1, color=INK2, lw=0.7)
    ax.set_xlabel(r'$k_{\max}\ [h\,{\rm Mpc}^{-1}]$')
    ax.set_title(f'({"bc"[b == "QSO"]}) {b}: ' + r'$\langle\chi^2\rangle / n$')
    ax.set_ylim(0.95, 1.2)
axes[1].plot([], [], color=INK2, ls='-', label='NGC')
axes[1].plot([], [], color=INK2, ls='--', label='SGC')
axes[1].legend(loc='upper left', fontsize=6.8)
fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=7)
fig.tight_layout(w_pad=0.6)
save(fig, 'chi2')
