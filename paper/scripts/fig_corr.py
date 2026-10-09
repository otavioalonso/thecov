"""Figure: (a) correlation matrix of the LRG1 NGC mocks (upper left) and of the recommended model (lower right);
(b, c) sample-minus-model covariance in units of its Wishart standard deviation under the model, (S - C) / sigma_W,
for the Gaussian term alone (upper left) and for the recommended model (lower right), LRG1 and QSO NGC."""
from common import *


def frame(ax, nb):
    for j in (1, 2):
        ax.axhline(j * nb - 0.5, color=INK2, lw=0.5)
        ax.axvline(j * nb - 0.5, color=INK2, lw=0.5)
    ticks = [j * nb + nb / 2 for j in range(3)]
    ax.set_xticks(ticks, [r'$\ell=0$', r'$\ell=2$', r'$\ell=4$'])
    ax.set_yticks(ticks, [r'$\ell=0$', r'$\ell=2$', r'$\ell=4$'], rotation=90, va='center')
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def split(upper_left, lower_right):
    """image (origin lower) with `upper_left` above the diagonal of the picture and `lower_right` below it"""
    M = lower_right.copy()
    il = np.tril_indices(len(M), -1)       # row > col: drawn above the diagonal with origin='lower'
    M[il] = upper_left[il]
    np.fill_diagonal(M, np.nan)
    return M


fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.45))
d = holi('LRG1', 'NGC')
nb = len(d['k'])
S = np.cov(d['V'].T)
im = axes[0].imshow(split(corr(S), corr(d['C']['rec'])), cmap=DIVERGING, vmin=-0.6, vmax=0.6, origin='lower',
                     interpolation='nearest')
axes[0].set_title('(a) LRG1 NGC correlation', fontsize=8)
frame(axes[0], nb)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03, ticks=[-0.5, 0, 0.5])
for ax, b, t in ((axes[1], 'LRG1', r'(b) LRG1 NGC, $(S - C)/\sigma_W$'), (axes[2], 'QSO', r'(c) QSO NGC, $(S - C)/\sigma_W$')):
    d = holi(b, 'NGC')
    S, N = np.cov(d['V'].T), len(d['V'])
    R = {m: (S - d['C'][m]) / wishart_sigma(d['C'][m], N) for m in ('G', 'rec')}
    im2 = ax.imshow(split(R['G'], R['rec']), cmap=DIVERGING, vmin=-4, vmax=4, origin='lower', interpolation='nearest')
    ax.set_title(t, fontsize=8)
    frame(ax, len(d['k']))
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03, ticks=[-4, -2, 0, 2, 4])
for ax, (ul, lr) in zip(axes, (('mocks', 'model'), ('Gaussian', 'recommended'), ('Gaussian', 'recommended'))):
    ax.text(0.03, 0.97, ul, transform=ax.transAxes, ha='left', va='top', fontsize=6.8, color=INK,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=1))
    ax.text(0.97, 0.03, lr, transform=ax.transAxes, ha='right', va='bottom', fontsize=6.8, color=INK,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=1))
fig.tight_layout(w_pad=0.8)
save(fig, 'corr')
