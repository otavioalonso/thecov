"""Figure: size of each covariance term on the diagonal, relative to the Gaussian term (LRG1 and QSO, NGC)."""
from common import *

fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.7), sharex=True, sharey=True)
TERMS = [('ssc', 'super-sample (SSC)', ORANGE, '-'), ('ssc_noLA', 'SSC without local average', ORANGE, (0, (1, 1.5))), ('disc', 'discreteness', AQUA, '-'),
         ('T0_tree', r'tree-level $T_0$ (rejected)', RED, '--'), ('T0_response', r'response $T_0$ (rejected)', VIOLET, ':')]
for row, b in enumerate(('LRG1', 'QSO')):
    d = holi(b, 'NGC')
    k, nb, dg = d['k'], len(d['k']), np.diag(d['C']['G'])
    for col, ell in enumerate((0, 2, 4)):
        ax = axes[row, col]
        s = slice(col * nb, (col + 1) * nb)
        for key, lab, c, ls in TERMS:
            if key in d['terms']:
                y = np.diag(d['terms'][key])[s] / dg[s]
                lw = 0.9 if key == 'ssc_noLA' else 1.4
                ax.plot(k, np.where(y > 0, y, np.nan), color=c, ls=ls, lw=lw, label=lab)
                if (y < 0).any():   # negative diagonal (local-average term dominates): |y| in grey
                    ax.plot(k, np.where(y < 0, -y, np.nan), color=c, ls='-', lw=3.0, alpha=0.25,
                            label=(r'negative values ($|\cdot|$ shown)' if row == 0 and col == 0 else None))
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1.0)
        if row == 0:
            ax.set_title(rf'$\ell = {ell}$')
        if col == 0:
            ax.set_ylabel(f'{b} NGC\n' + r'$C^X_{ii} / C^{\rm G}_{ii}$')
        if row == 1:
            ax.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.07), handlelength=2.4)
fig.tight_layout(h_pad=0.4, w_pad=0.4)
save(fig, 'budget')
