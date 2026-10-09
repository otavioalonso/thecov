"""Figure: does fibre assignment change the covariance beyond what the model predicts? The 25 AbacusSummit LRG1 mocks
with complete and with altmtl (fibre-assigned) targeting share their initial conditions, so each statistic can be
compared pairwise, each version against its own Gaussian model (local window; the altmtl window has the lower,
fibre-assigned density). (a) chi^2_i / n (k < 0.3) of every mock, altmtl against complete; (b) the paired double ratio
of mock to model variance, altmtl over complete, per multipole, geometric mean over three k ranges, paired-bootstrap
68% intervals."""
from common import *

z = load('abacus_LRG1')
rng = np.random.default_rng(3)
NBOOT = 2000
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5), gridspec_kw=dict(width_ratios=[1, 1.7]))
out = {}
ax = axes[0]
for cap, c in (('NGC', BLUE), ('SGC', ORANGE)):
    Vc, Va = z[f'complete/{cap}/x1/V'], z[f'altmtl/{cap}/x1/V']
    n, N = Vc.shape[1], len(Vc)
    xc = chi2(Vc, z[f'complete/{cap}/x1/C_G_local']) / (n * (N - 1) / N)
    xa = chi2(Va, z[f'altmtl/{cap}/x1/C_G_local']) / (n * (N - 1) / N)
    dlt = xa - xc
    out[cap] = dict(chi2_complete=float(xc.mean()), chi2_altmtl=float(xa.mean()), diff=float(dlt.mean()),
                    diff_err=float(dlt.std(ddof=1) / np.sqrt(N)), corr=float(np.corrcoef(xc, xa)[0, 1]))
    ax.scatter(xc, xa, s=12, color=c, edgecolor='white', lw=0.5, zorder=3,
               label=rf'{cap}: $\Delta = {dlt.mean():+.3f} \pm {dlt.std(ddof=1) / np.sqrt(N):.3f}$')
lim = (0.7, 1.75)
ax.plot(lim, lim, color=INK2, lw=0.7, ls='--')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.set_xlabel(r'$\chi^2_i / n$, complete')
ax.set_ylabel(r'$\chi^2_i / n$, altmtl')
ax.set_title('(a) per-mock $\\chi^2$', fontsize=8)
ax.legend(loc='upper left', fontsize=6.3, handletextpad=0.2)

ax = axes[1]
KR = ((0.02, 0.1), (0.1, 0.2), (0.2, 0.3))
for cap, mk, off in (('NGC', 'o', -0.06), ('SGC', 's', 0.06)):
    Vc, Va = z[f'complete/{cap}/x1/V'], z[f'altmtl/{cap}/x1/V']
    k, N = z[f'complete/{cap}/x1/k'], len(Vc)
    nb = len(k)
    mr = np.diag(z[f'altmtl/{cap}/x1/C_G_local']) / np.diag(z[f'complete/{cap}/x1/C_G_local'])

    def stat(s):
        lr = np.log(Va[s].var(0, ddof=1) / Vc[s].var(0, ddof=1) / mr)
        return np.array([[np.exp(lr[(e // 2) * nb + np.flatnonzero((k >= a) & (k < b))].mean()) for (a, b) in KR]
                         for e in (0, 2, 4)])
    r = stat(np.arange(N))
    boot = np.array([stat(rng.integers(0, N, N)) for _ in range(NBOOT)])
    lo, hi = np.percentile(boot, [16, 84], axis=0)
    out[cap]['double_ratio'] = r.tolist()
    for j, (e, c) in enumerate(ELL_COLOR.items()):
        x = np.arange(3) + (j - 1) * 0.22 + off
        ax.errorbar(x, r[j], [r[j] - lo[j], hi[j] - r[j]], fmt=mk, color=c, ms=3.2, lw=0.9,
                    mfc=c if cap == 'NGC' else 'white', label=(rf'$\ell={e}$' if cap == 'NGC' else None))
ax.set_xticks(range(3), [f'${a}<k<{b}$' for a, b in KR])
ax.axhline(1, color=INK2, lw=0.7)
ax.set_ylim(0.8, 1.2)
ax.set_ylabel(r'$\dfrac{\hat\sigma^2_{\rm altmtl} / \sigma^2_{\rm altmtl,\,model}}{\hat\sigma^2_{\rm complete} / \sigma^2_{\rm complete,\,model}}$')
ax.set_title('(b) paired double ratio (filled: NGC, open: SGC)', fontsize=8)
ax.legend(loc='upper left', ncol=3, fontsize=6.8)
fig.tight_layout(w_pad=1.0)
save(fig, 'fiber')
write_numbers('fiber', out)
print(out)
