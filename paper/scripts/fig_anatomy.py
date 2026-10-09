"""Figure: the structure of each non-Gaussian term. (a) eigenvalues of the term whitened by the Gaussian covariance,
C_G^{-1/2} C_X C_G^{-1/2}: how many directions it acts on and how strongly; (b-c) the leading directions of SSC and of
the discreteness term, as patterns in the data vector in units of the Gaussian error of each bin."""
from common import *

d = holi('LRG1', 'NGC')
k, nb = d['k'], len(d['k'])
CG = d['C']['G']
w, U = np.linalg.eigh(CG)
Wh, Wi = (U * np.sqrt(w)) @ U.T, (U / np.sqrt(w)) @ U.T
sg = np.sqrt(np.diag(CG))
TERMS = [('ssc', 'SSC', ORANGE), ('disc', 'discreteness', AQUA), ('T0_tree', r'tree $T_0$', RED),
         ('T0_response', r'response $T_0$', VIOLET)]

fig = plt.figure(figsize=(TEXTWIDTH, 2.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.25], wspace=0.35)
ax = fig.add_subplot(gs[0])
eig = {}
for key, lab, c in TERMS:
    lam, vec = np.linalg.eigh(Wi @ d['terms'][key] @ Wi)
    order = np.argsort(-np.abs(lam))
    eig[key] = (lam[order], vec[:, order])
    ax.plot(np.arange(1, 31), np.abs(lam[order][:30]), 'o-', color=c, ms=2.5, lw=1.0, label=lab)
ax.set_yscale('log')
ax.set_ylim(1e-4, 30)
ax.set_xlabel('rank')
ax.set_ylabel(r'$|\lambda|$ of $C_{\rm G}^{-1/2} C_X C_{\rm G}^{-1/2}$')
ax.set_title('(a) eigenvalues')
ax.legend(loc='lower right', fontsize=6.5)

for j, (key, title) in enumerate((('ssc', '(b) SSC: leading directions'), ('disc', '(c) discreteness: leading directions'))):
    ax = fig.add_subplot(gs[j + 1])
    lam, vec = eig[key]
    for m, ls in zip(range(2), ('-', '--')):
        pat = (Wh @ vec[:, m]) * np.sqrt(abs(lam[m])) / sg        # data-space pattern, in Gaussian sigma per bin
        pat *= np.sign(pat[np.argmax(np.abs(pat))])
        for e, c in ELL_COLOR.items():
            s = slice((e // 2) * nb, (e // 2 + 1) * nb)
            ax.plot(k, pat[s], color=c, ls=ls, lw=1.1, label=(rf'$\ell={e}$' if m == 0 else None))
    ax.axhline(0, color=BASE, lw=0.6)
    ax.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$')
    ax.set_title(title)
    if j == 0:
        ax.set_ylabel(r'$\sqrt{\lambda}\,v / \sigma_{\rm G}$')
        ax.legend(loc='lower left', fontsize=6.8, title='solid: 1st, dashed: 2nd', title_fontsize=6.5)
save(fig, 'anatomy')
