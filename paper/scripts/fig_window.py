"""Figure: the window of the Gaussian term. (a) the xi-kernel K_k(r) = xi_0(r) j_0(kr) / P_0(k) that pair-averages the
window, r^2 K_k(r) for three k: it extends over a correlation length, so window structure on <~ 20 Mpc/h scales (veto
holes, edges) is not captured by the local window m^2(x); (b) the resulting change of the Gaussian variance,
C_G(xi-kernel) / C_G(local), and the change from evaluating the window with each random's own weight instead of the
locally averaged weight, per multipole, for LRG1 and QSO (NGC solid, SGC dashed)."""
from scipy.special import spherical_jn
from scipy.integrate import simpson
from common import *

d = holi('LRG1', 'NGC')
b1, b2, bs2, f, z = d['params']
kl, Pl = d['plin_k'], d['plin_P_damped_normalised']
P0 = (b1 ** 2 + 2 / 3 * b1 * f + f ** 2 / 5) * Pl
kk = np.linspace(1e-4, 3.0, 30000)
pp = np.interp(kk, kl, P0) * np.exp(-(kk * 1.0) ** 2)
r = np.linspace(0.5, 150, 600)
xi = np.array([simpson(kk ** 2 * pp * spherical_jn(0, kk * ri), x=kk) for ri in r]) / (2 * np.pi ** 2)

fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.3), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
ax = axes[0]
for kv, c in ((0.05, BLUE), (0.1, ORANGE), (0.2, VIOLET)):
    K = xi * spherical_jn(0, kv * r) / np.interp(kv, kl, P0)
    ax.plot(r, 4 * np.pi * r ** 2 * K, color=c, label=rf'$k = {kv}$')
ax.axhline(0, color=BASE, lw=0.6)
ax.set_xlabel(r'$r\ [h^{-1}{\rm Mpc}]$')
ax.set_ylabel(r'$4\pi r^2 K_k(r)\ [h\,{\rm Mpc}^{-1}]$')
ax.set_title(r'(a) $\xi$-kernel', fontsize=8)
ax.legend(fontsize=6.8)
ax.set_xlim(0, 150)

for ax, key, t in ((axes[1], 'G', r'(b) $\xi$-kernel / local window'), (axes[2], 'own', '(c) own weight / mean weight')):
    for b, lw in (('LRG1', 1.4), ('QSO', 0.9)):
        for cap, ls in (('NGC', '-'), ('SGC', '--')):
            dd = holi(b, cap)
            zz = load(f'holi_{b}')
            num = dd['C']['G'] if key == 'G' else zz[f'{cap}/x1/C_G_ownweight']
            rr = np.diag(num) / np.diag(dd['C']['G_local'])
            nb = len(dd['k'])
            for j, (e, c) in enumerate(ELL_COLOR.items()):
                ax.plot(dd['k'], rr[j * nb:(j + 1) * nb], color=c, ls=ls, lw=lw,
                        label=(rf'$\ell={e}$' if b == 'LRG1' and cap == 'NGC' else None))
            ax.text(0.298, rr[:nb][-3:].mean() + 0.006, b if cap == 'NGC' else '', ha='right', fontsize=6.8, color=INK, bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))
    ax.axhline(1, color=INK2, lw=0.7)
    ax.set_ylim(0.985, 1.13)
    ax.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$')
    ax.set_title(t, fontsize=8)
axes[1].set_ylabel('ratio of Gaussian variances')
axes[1].legend(fontsize=6.5, loc='upper right', ncol=1)
fig.tight_layout(w_pad=0.6)
save(fig, 'window')
