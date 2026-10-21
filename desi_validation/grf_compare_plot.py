"""Three window treatments of the Gaussian covariance against Gaussian fields at the DESI randoms.

Everything is shown relative to the kernel-smoothed model, so the models are smooth curves and
the fields (the exact answer, up to sampling noise) are the points.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED, RULE = '#1d1d1f', '#6e6e73', '#d2d2d7'
MODELS = (  # key, label, colour, line style, width, zorder
    ('none', r'own weight $w^2$', '#9a9a9f', (0, (5, 2.2)), 1.3, 3),
    ('random-density', 'nearest-random weights', INK, '-', 1.3, 4),
    ('kernel', 'kernel smoothing', '#c8102e', '-', 1.9, 5),
)

def style():
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
        'font.size': 9.5, 'axes.labelsize': 10.5, 'axes.titlesize': 11,
        'axes.edgecolor': INK, 'axes.linewidth': 0.7, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 3.2, 'ytick.major.size': 3.2, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.45, 'ytick.minor.width': 0.45,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'legend.frameon': False, 'savefig.dpi': 300, 'pdf.fonttype': 42,
    })

def points(ax, k, y, e):
    return ax.errorbar(k, y, e, fmt='o', ms=2.9, mfc='white', mec=INK, mew=0.6, ecolor=MUTED,
                       elinewidth=0.55, capsize=0, zorder=6)

def figure(R, sets, inp, fn, ylim, yticks, title):
    style()
    fig, axs = plt.subplots(len(sets), 3, figsize=(7.4, 2.35 * len(sets) + 0.95), sharex=True,
                            gridspec_kw=dict(wspace=0, hspace=0))
    for row, (name, pre) in zip(axs, sets):
        k, N = R[pre + 'k'], int(R[pre + 'n_fields'])
        ref = lambda b: R[f'{pre}kernel/{inp}/{b}']
        for ax, b, lab in ((row[0], '00', 'P_0'), (row[1], '22', 'P_2')):
            ax.axhline(1, color=RULE, lw=0.8, zorder=1)
            for m, _, col, ls, lw, z in MODELS:
                ax.plot(k, np.sqrt(R[f'{pre}{m}/{inp}/{b}'] / ref(b)), color=col, ls=ls, lw=lw, zorder=z,
                        solid_capstyle='round')
            y = np.sqrt(R[f'{pre}fields/{b}'] / ref(b))
            points(ax, k, y, y / np.sqrt(2 * (N - 1)))  # sigma of a sample standard deviation
            lo, hi = ylim; pad = 0.025 * (hi - lo)
            for sel, yy, mk in ((y < lo, lo + pad, 'v'), (y > hi, hi - pad, '^')):
                ax.plot(k[sel], np.full(sel.sum(), yy), mk, ms=4, mfc=INK, mec=INK, zorder=7, clip_on=False)
            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)
        ax = row[2]
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        for m, _, col, ls, lw, z in MODELS:
            C02, C00, C22 = (R[f'{pre}{m}/{inp}/{x}'] for x in ('02', '00', '22'))
            ax.plot(k, C02 / np.sqrt(C00 * C22), color=col, ls=ls, lw=lw, zorder=z, solid_capstyle='round')
        rho = R[f'{pre}fields/02'] / np.sqrt(R[f'{pre}fields/00x22'])
        pts = points(ax, k, rho, (1 - rho**2) / np.sqrt(N - 1))
        ax.yaxis.tick_right(); ax.yaxis.set_label_position('right'); ax.tick_params(axis='y', which='both', left=True)
        ax.set_ylim(-0.14, 0.64)
        ax.set_yticks([0, 0.2, 0.4])
        ax.set_ylabel(r'$\rho_{02}(k)$', rotation=270, labelpad=13)
        row[1].tick_params(axis='y', labelleft=False)
        row[0].set_ylabel(r'$\sigma(k)\,/\,\sigma_\mathrm{kernel}(k)$')
        row[0].text(-0.385, 0.5, name, transform=row[0].transAxes, rotation=90, ha='center', va='center',
                    fontsize=12, color=INK)
        row[0].text(-0.305, 0.5, f'{N} fields', transform=row[0].transAxes, rotation=90, ha='center', va='center',
                    fontsize=8.5, color=MUTED, style='italic')
    for ax, t in zip(axs[0], (r'monopole  $P_0$', r'quadrupole  $P_2$', r'correlation  $P_0 \times P_2$')):
        ax.set_title(t, pad=6, color=INK)
    for ax in axs[-1]:
        ax.set_xlabel(r'$k\ \ [h\,\mathrm{Mpc}^{-1}]$')
        ax.set_xlim(0.012, 0.308)
        ax.set_xticks([0.05, 0.1, 0.15, 0.2, 0.25])
        ax.set_xticklabels(['0.05', '0.10', '0.15', '0.20', '0.25'])
    handles = [Line2D([], [], color=col, ls=ls, lw=lw) for _, _, col, ls, lw, _ in MODELS]
    handles.append(pts)
    labels = [l for _, l, *_ in MODELS] + [r'Gaussian fields, $\pm1\sigma$']
    h = fig.get_figheight()
    top = 1 - 0.95 / h
    fig.subplots_adjust(left=0.135, right=0.915, bottom=0.42 / h * 1.6, top=top)
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, top + 0.30 / h),
               handlelength=2.4, columnspacing=1.6, fontsize=9)
    fig.text(0.5, top + 0.66 / h, title, ha='center', va='bottom', fontsize=11.5, color=INK)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{fn}.{ext}')
    return fig, axs

if __name__ == '__main__':
    import sys
    R = dict(np.load(sys.argv[1] if len(sys.argv) > 1 else 'grf_compare.npz'))
    sets = [('LRG1  NGC', 'LRG1_NGC_rsd/'), ('QSO  NGC', 'QSO_NGC_rsd/')]
    figure(R, sets, 'true', 'grf_compare_trueP', (0.78, 1.64), [1.0, 1.2, 1.4],
           'Gaussian covariance, model fed the true (windowless) power spectrum')
    figure(R, sets, 'masked', 'grf_compare_masked', (0.84, 1.16), [0.90, 0.95, 1.00, 1.05, 1.10],
           "Gaussian covariance, model fed the fields' mean (as with mocks)")
