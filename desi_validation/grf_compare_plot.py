"""Figures of the three-window comparison against Gaussian fields (input: the npz of desi_validation/grf_compare.py).

    from desi_validation.grf_compare_plot import figure
    figure(dict(np.load("grf_compare.npz")), [("LRG1 NGC", "LRG1_NGC_rsd/"), ("QSO NGC", "QSO_NGC_rsd/")], "true", "grf_compare_trueP")
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

MODELS = [('none', 'own weight $w^2$', '0.55', '--'),
          ('random-density', 'nearest-random weights', 'k', '-'),
          ('kernel', 'kernel smoothing', 'tab:red', '-')]
BLOCKS = [('00', r'$P_0$'), ('22', r'$P_2$'), ('02', r'$P_0 \times P_2$')]


def figure(R, sets, inp, fn, ylim=(0.6, 1.5)):
    mpl.rcdefaults()
    fig, axs = plt.subplots(len(sets), 3, figsize=(12, 3.1 * len(sets)), sharex=True, squeeze=False,
                            gridspec_kw=dict(wspace=0, hspace=0))
    for r_, (label, pre) in enumerate(sets):
        k, N = R[pre + 'k'], int(R[pre + 'n_fields'])
        for c_, (b, lab) in enumerate(BLOCKS):
            ax = axs[r_, c_]
            if b != '02':
                S = R[f'{pre}fields/{b}']
                sig = 1 / np.sqrt(2 * (N - 1))
                ax.axhspan(1 - sig, 1 + sig, color='0.85', lw=0, zorder=0, label=r'fields, 1$\sigma$')
                ax.axhline(1, color='0.4', lw=0.8, zorder=1)
                for m, mlab, col, ls in MODELS:
                    ax.plot(k, np.sqrt(R[f'{pre}{m}/{inp}/{b}'] / S), color=col, ls=ls, label=mlab)
                ax.set_ylim(*ylim)
                if c_ == 0:
                    ax.set_ylabel(f'{label}  ({N} fields)\n' + r'$\sqrt{C_{ii}/C^{\rm fields}_{ii}}$')
                else:
                    ax.tick_params(labelleft=False)
            else:
                rho = R[f'{pre}fields/02'] / np.sqrt(R[f'{pre}fields/00x22'])
                e = (1 - rho ** 2) / np.sqrt(N - 1)
                ax.fill_between(k, rho - e, rho + e, color='0.75', lw=0)
                for m, mlab, col, ls in MODELS:
                    C02, C00, C22 = (R[f'{pre}{m}/{inp}/{x}'] for x in ('02', '00', '22'))
                    ax.plot(k, C02 / np.sqrt(C00 * C22), color=col, ls=ls)
                ax.axhline(0, color='0.4', lw=0.8)
                ax.yaxis.tick_right(); ax.yaxis.set_label_position('right')
                ax.set_ylabel(r'$C^{02}_{ii}/\sqrt{C^{00}_{ii}\,C^{22}_{ii}}$')
            if r_ == 0:
                ax.set_title(lab)
            if r_ == len(sets) - 1:
                ax.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
            ax.set_xlim(0.015, 0.305)
    for ax in axs[-1]:
        ax.set_xticks([0.05, 0.1, 0.15, 0.2, 0.25])
    h, l = axs[0, 0].get_legend_handles_labels()
    title = {'true': 'model input: true (windowless) power spectrum',
             'masked': "model input: fields' mean (window-convolved), as with mocks"}[inp]
    top = 1 - 1.05 / fig.get_figheight()
    fig.subplots_adjust(top=top)
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, top + 0.33 / fig.get_figheight()), fontsize=10)
    fig.text(0.5, top + 0.75 / fig.get_figheight(), title, ha='center', va='bottom', fontsize=11)
    fig.savefig(fn + '.pdf', bbox_inches='tight')
    fig.savefig(fn + '.png', dpi=150, bbox_inches='tight')
    return fig, axs
