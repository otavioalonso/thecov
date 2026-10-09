"""Figure: what changes with the bin width. A Gaussian-like error of the model (wrong window, wrong shot noise) gives a
mock/model variance ratio independent of Delta k, while a missing smooth non-Gaussian term (SSC, connected trispectrum)
grows relative to the Gaussian term in proportion to Delta k. The same 859 LRG1 mocks binned at Delta k = 0.005, 0.01,
0.02 (the non-Gaussian terms are rebinned exactly with the mode-weighted rebinning matrix).
(a) whitened trace tr(C^{-1} S)/n, k < 0.3; (b) mean diagonal ratio for 0.1 < k < 0.3 per multipole;
(c) the leading direction of C_G^{-1/2} S C_G^{-1/2} at each bin width, as a pattern in the data vector (units of the
Gaussian error per bin, normalised to its maximum), with its eigenvalue. In (b) solid: Gaussian, dotted: recommended."""
from common import *

F = (1, 2, 4)
DK = np.array(F) * 0.005
fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
out = {}
base = {cap: holi('LRG1', cap, 1) for cap in ('NGC', 'SGC')}


def model_at(cap, f, m):
    d1, df = base[cap], holi('LRG1', cap, f)
    if m in ('G', 'G_local'):
        return df['C'][m], df
    R = rebin_matrix(d1['nmodes'], f)
    return df['C']['G'] + R @ (d1['C'][m] - d1['C']['G']) @ R.T, df


for cap, ls, mk in (('NGC', '-', 'o'), ('SGC', '--', 's')):
    for m in ('G_local', 'G', 'rec'):
        tr, dg = [], []
        for f in F:
            C, df = model_at(cap, f, m)
            S = np.cov(df['V'].T)
            tr.append(np.trace(np.linalg.solve(C, S)) / len(S))
            k, nb = df['k'], len(df['k'])
            sel = np.flatnonzero((k > 0.1) & (k < 0.3))
            dg.append([np.mean(np.diag(S)[sel + j * nb] / np.diag(C)[sel + j * nb]) for j in range(3)])
        out[f'{cap}/{m}'] = dict(trace=tr, diag=dg)
        axes[0].plot(DK, tr, ls=ls, marker=mk, color=MODEL_COLOR[m], ms=3.5, mfc='white' if cap == 'SGC' else None,
                     label=MODEL_LABEL[m] if cap == 'NGC' else None)
        if m != 'G_local':
            dg = np.array(dg)
            for j, (e, c) in enumerate(ELL_COLOR.items()):
                axes[1].plot(DK, dg[:, j], ls=ls if m == 'G' else ':', marker=mk, color=c, ms=3,
                             mfc='white' if cap == 'SGC' else None, lw=1.1 if m == 'G' else 1.4,
                             label=(rf'$\ell={e}$' if cap == 'NGC' and m == 'G' else None))
axes[0].set_ylabel(r'${\rm tr}(C^{-1}S)/n$')
axes[0].set_title('(a) whitened trace', fontsize=8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=7)
axes[1].set_ylabel(r'$\langle \hat\sigma^2 / \sigma^2_{\rm model}\rangle$')
axes[1].set_title(r'(b) diagonal, $0.1<k<0.3$', fontsize=8)
axes[1].legend(fontsize=6.3, loc='upper left', ncol=3, handlelength=1.4, columnspacing=0.6)
for ax in axes[:2]:
    ax.set_xlabel(r'$\Delta k\ [h\,{\rm Mpc}^{-1}]$')
    ax.set_xticks(DK)
    ax.set_xlim(0.003, 0.022)
    ax.axhline(1, color=INK2, lw=0.7)

ax = axes[2]
for f, lw, al in zip(F, (0.8, 1.1, 1.6), (0.6, 0.8, 1.0)):
    df = holi('LRG1', 'NGC', f)
    C = df['C']['G']
    W, Wi = whitened_cov(df['V'], C)
    lam, U = np.linalg.eigh(W)
    pat = np.linalg.solve(Wi, U[:, -1]) / np.sqrt(np.diag(C))
    pat /= pat[np.argmax(np.abs(pat))]
    k, nb = df['k'], len(df['k'])
    for j, (e, c) in enumerate(ELL_COLOR.items()):
        ax.plot(k, pat[j * nb:(j + 1) * nb], color=c, lw=lw, alpha=al,
                label=(rf'$\Delta k = {0.005 * f:.3f}$, $\lambda_1 = {lam[-1]:.1f}$' if j == 0 else None))
ax.axhline(0, color=BASE, lw=0.6)
ax.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$')
ax.set_ylabel(r'$v_1 / \max|v_1|$')
ax.set_title(r'(c) leading direction', fontsize=8)
ax.legend(fontsize=6.0, loc='upper left')
fig.tight_layout(w_pad=0.5)
save(fig, 'binning')
write_numbers('binning', out)
