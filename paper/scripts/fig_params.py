"""Figure: what the covariance does to parameter errors. Each mock is fitted for the amplitude of the multipoles (A), an
extra quadrupole amplitude (A2), an isotropic dilation (alpha) and a constant in the monopole (SN), by generalised least
squares with the model covariance (templates from the mean of the mocks). Plotted: variance of the estimates over the
mocks divided by the variance predicted by the model, as a function of k_max, with the +-1 sigma band for N mocks."""
from common import *

NAMES = (r'$A$', r'$A_2$', r'$\alpha$', r'$N_{\rm SN}$')


def param_ratios(V, C, k, kmax):
    nb = len(k)
    Pm = V.mean(0)
    P0, P2, P4 = Pm[:nb], Pm[nb:2 * nb], Pm[2 * nb:]
    I = kidx(k, kmax)
    dP = lambda P: -np.gradient(P, np.log(k))
    z = np.zeros(nb)
    T = np.stack([np.concatenate([P0, P2, P4]), np.concatenate([z, P2, z]),
                  np.concatenate([dP(P0), dP(P2), dP(P4)]), np.concatenate([np.ones(nb), z, z])], 1)[I]
    Ci = np.linalg.inv(C[np.ix_(I, I)])
    Fi = np.linalg.inv(T.T @ Ci @ T)
    est = (Fi @ T.T @ Ci @ (V[:, I] - V[:, I].mean(0)).T).T
    return est.var(0, ddof=1) / np.diag(Fi)


KMAX = np.arange(0.1, 0.301, 0.025)
fig, axes = plt.subplots(3, 4, figsize=(TEXTWIDTH, 4.6), sharex=True, sharey=True)
out = {}
for row, (b, cap) in enumerate((('LRG1', 'NGC'), ('LRG1', 'SGC'), ('QSO', 'NGC'))):
    d = holi(b, cap)
    V, k, N = d['V'], d['k'], len(d['V'])
    for m in ('G_local', 'G', 'rec'):
        R = np.array([param_ratios(V, d['C'][m], k, km) for km in KMAX])
        out[f'{b}/{cap}/{m}'] = {f'{km:.3f}': r.tolist() for km, r in zip(KMAX, R)}
        for j in range(4):
            axes[row, j].plot(KMAX, R[:, j], color=MODEL_COLOR[m], marker='o', ms=2.5, lw=1.2,
                              label=MODEL_LABEL[m] if row == 0 and j == 0 else None)
    for j in range(4):
        ax = axes[row, j]
        e = np.sqrt(2 / (N - 1))
        ax.axhspan(1 - e, 1 + e, color=GRID, lw=0)
        ax.axhline(1, color=INK2, lw=0.7)
        if row == 0:
            ax.set_title(NAMES[j])
        if j == 0:
            ax.set_ylabel(f'{b} {cap}\n' + r'$\hat\sigma^2 / \sigma^2_{\rm model}$')
        if row == 2:
            ax.set_xlabel(r'$k_{\max}$')
axes[0, 0].set_ylim(0.75, 2.0)
axes[0, 0].set_yscale('log')
axes[0, 0].set_yticks([0.8, 1, 1.25, 1.5, 2], ['0.8', '1', '1.25', '1.5', '2'])
axes[0, 0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=7)
fig.tight_layout(h_pad=0.3, w_pad=0.3)
save(fig, 'params')
write_numbers('params', out)
