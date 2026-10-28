"""Vector-PDF figures for the mock bispectrum covariance report.  usage: python make_report_figures.py <data dir>"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as an

D = sys.argv[1] if len(sys.argv) > 1 else '.'
an.D = D
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'legend.frameon': False, 'axes.titlelocation': 'left', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
COL = {'holi': '#222222', 'glam': '#2b5797', 'abc': '#c0392b', 'aba': '#e67e22', 'a2c': '#7d3c98', 'a2a': '#4a9c6d'}
LAB = {'holi': 'holi v3 altmtl', 'glam': 'GLAM-Uchuu v2 altmtl', 'abc': 'Abacus-HF v2 complete', 'aba': 'Abacus-HF v2 altmtl',
       'a2c': 'Abacus 2nd-gen complete', 'a2a': 'Abacus 2nd-gen altmtl'}
BINS = list(an.BINS)
data = an.load()
k = data['holi'][('LRG1', 'NGC')]['k_b']; sel = (k >= an.KMIN) & (k < an.KMAX); ks = k[sel]; n = sel.sum(); p = 2 * n
res = {}
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter, NullLocator


def logticks(ax, ticks):
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def kticks(ax):
    ax.set_xticks([0.05, 0.1, 0.15, 0.2]); ax.set_xticklabels(['', '0.1', '', '0.2'])


def top_legend(fig, suites, y=1.0):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[s], lw=1.2) for s in suites]
    fig.legend(h, [LAB[s] for s in suites], loc='lower center', ncol=len(suites), bbox_to_anchor=(0.5, y), fontsize=6.5, handlelength=1.5)


def Ptot(g):
    return np.interp(ks, g['k_p'], g['P0'].mean(0) + np.mean(g['snP'] / g['normP']))


def band_ratio(Xs, Xh, m, scale=1.0, nb=40):
    """sum var_s / sum var_h over the elements m (scaled), with delete-block jackknife errors over both suites"""
    f = lambda A, B: np.sum(A[:, m].var(0, ddof=1) / scale[m]) / np.sum(B[:, m].var(0, ddof=1))
    r0 = f(Xs, Xh)
    def jk(X, other, first):
        bl = np.array_split(np.arange(len(X)), min(nb, len(X)))
        v = np.array([f(np.delete(X, b, 0), other) if first else f(other, np.delete(X, b, 0)) for b in bl])
        return (len(bl) - 1) / len(bl) * np.sum((v - v.mean()) ** 2)
    return r0, np.sqrt(jk(Xs, Xh, True) + jk(Xh, Xs, False))


# ---------------------------------------------------------------- fig 1: relative errors
fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.3), sharex=True)
for j, b in enumerate(BINS):
    for s in ('holi', 'glam', 'abc', 'aba', 'a2a'):
        if (b, 'NGC') not in data[s]:
            continue
        X = an.vec(data[s][(b, 'NGC')], sel); mu0 = np.abs(X[:, :n].mean(0))
        axes[0, j].plot(ks, X[:, :n].std(0) / mu0, color=COL[s], lw=1.0, label=LAB[s])
        axes[1, j].plot(ks, X[:, n:].std(0) / mu0, color=COL[s], lw=1.0)
    axes[0, j].set_title(b)
    for row in (0, 1):
        axes[row, j].set_yscale('log'); axes[row, j].set_ylim(0.025, 1.2); logticks(axes[row, j], [0.03, 0.1, 0.3, 1]); kticks(axes[row, j])
        if j: axes[row, j].set_yticklabels([])
    axes[1, j].set_xlabel('$k$ [$h$/Mpc]')
axes[0, 0].set_ylabel(r'$\sigma(B_{000}) / |\langle B_{000}\rangle|$'); axes[1, 0].set_ylabel(r'$\sigma(B_{202}) / |\langle B_{000}\rangle|$')
fig.tight_layout(w_pad=0.3, h_pad=0.6, rect=(0, 0, 1, 0.93)); top_legend(fig, ('holi', 'glam', 'abc', 'aba', 'a2a'), 0.93)
fig.savefig(os.path.join(OUT, 'relerr.pdf')); plt.close(fig)

# ---------------------------------------------------------------- fig 2: variance ratios to holi (P_tot^3-scaled), per element
fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.4), sharex=True, sharey='row')
for j, b in enumerate(BINS):
    gh = data['holi'][(b, 'NGC')]; Xh = an.vec(gh, sel)
    for s in ('glam', 'abc', 'aba', 'a2a'):
        if (b, 'NGC') not in data[s]:
            continue
        gs = data[s][(b, 'NGC')]; Xs = an.vec(gs, sel)
        scale = np.concatenate([(Ptot(gs) / Ptot(gh)) ** 3] * 2)
        r = Xs.var(0, ddof=1) / Xh.var(0, ddof=1) / scale
        for row, sl in ((0, slice(0, n)), (1, slice(n, p))):
            axes[row, j].plot(ks, r[sl], color=COL[s], lw=0.9, label=LAB[s])
    for row in (0, 1):
        axes[row, j].axhline(1, color='0.6', lw=0.6, zorder=0); axes[row, j].set_yscale('log'); axes[row, j].set_ylim(0.15, 3)
        logticks(axes[row, j], [0.2, 0.5, 1, 2]); kticks(axes[row, j])
    axes[0, j].set_title(b); axes[1, j].set_xlabel('$k$ [$h$/Mpc]')
axes[0, 0].set_ylabel('var / var(holi), $B_{000}$'); axes[1, 0].set_ylabel('var / var(holi), $B_{202}$')
fig.tight_layout(w_pad=0.3, h_pad=0.6, rect=(0, 0, 1, 0.93)); top_legend(fig, ('glam', 'abc', 'aba', 'a2a'), 0.93)
fig.savefig(os.path.join(OUT, 'varratio_scaled.pdf')); plt.close(fig)

# band ratios (raw and scaled) with jackknife errors, all regions -> table
bands = {}
for s in ('glam', 'abc', 'aba', 'a2c', 'a2a'):
    for b in BINS:
        for r in an.REGIONS:
            if (b, r) not in data[s] or (b, r) not in data['holi']:
                continue
            gs, gh = data[s][(b, r)], data['holi'][(b, r)]
            Xs, Xh = an.vec(gs, sel), an.vec(gh, sel)
            scale = np.concatenate([(Ptot(gs) / Ptot(gh)) ** 3] * 2)
            for name, off in (('B000', 0), ('B202', n)):
                for lo, hi in an.BANDS:
                    m = np.zeros(p, bool); m[off:off + n] = (ks >= lo) & (ks < hi)
                    r0, e0 = band_ratio(Xs, Xh, m, np.ones(p)); r1, e1 = band_ratio(Xs, Xh, m, scale)
                    bands[f'{s}|{b}|{r}|{name}|{lo:g}-{hi:g}'] = dict(raw=r0, raw_err=e0, scaled=r1, scaled_err=e1)
res['band_ratios'] = bands

# ---------------------------------------------------------------- fig 3: correlation matrices, LRG1 NGC
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15))
for ax, s in zip(axes, ('holi', 'glam', 'abc', 'aba')):
    X = an.vec(data[s][('LRG1', 'NGC')], sel); Rm = an.corr(np.cov(X, rowvar=False))
    im = ax.imshow(Rm, vmin=-1, vmax=1, cmap='RdBu_r', origin='lower', interpolation='nearest')
    ax.set_title(f'{LAB[s]} ({len(X)})', fontsize=6.5)
    ax.set_xticks([n / 2, 1.5 * n]); ax.set_xticklabels(['$B_{000}$', '$B_{202}$']); ax.set_yticks([n / 2, 1.5 * n]); ax.set_yticklabels(['$B_{000}$', '$B_{202}$'])
    ax.axhline(n - 0.5, color='k', lw=0.4); ax.axvline(n - 0.5, color='k', lw=0.4)
    for sp in ax.spines.values(): sp.set_visible(True)
cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01); cb.set_label('correlation', fontsize=7); cb.set_ticks([-1, 0, 1])
fig.savefig(os.path.join(OUT, 'corr_LRG1_NGC.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- fig 4: correlation structure summary
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharex=True)
x = np.arange(len(BINS)); suites = ('holi', 'glam', 'abc', 'aba', 'a2a')
summ = {}
for i_s, s in enumerate(suites):
    nn, fl, cr, nne, fle = [], [], [], [], []
    for b in BINS:
        X = an.vec(data[s][(b, 'NGC')], sel); N = len(X); Rm = an.corr(np.cov(X, rowvar=False))
        cs = an.corr_summary(Rm, n)
        nn.append(cs['B000']['nn1']); fl.append(cs['B000']['floor']); cr.append(cs['B202']['nn1'])
        nne.append((1 - cs['B000']['nn1'] ** 2) / np.sqrt(N)); fle.append(1 / np.sqrt(N) / np.sqrt(len(ks) * 0.4))
    summ[s] = dict(nn1_B000=nn, floor_B000=fl, nn1_B202=cr)
    dx = (i_s - 2) * 0.13
    axes[0].errorbar(x + dx, nn, yerr=nne, fmt='o', ms=3, color=COL[s], lw=0.8, label=LAB[s])
    axes[1].errorbar(x + dx, fl, yerr=fle, fmt='o', ms=3, color=COL[s], lw=0.8)
    axes[2].errorbar(x + dx, cr, yerr=nne, fmt='o', ms=3, color=COL[s], lw=0.8)
for ax, t in zip(axes, (r'$B_{000}$: $\rho(k_i, k_{i+1})$ (median)', r'$B_{000}$: mean $\rho$, $|i - j| \geq 10$', r'$B_{202}$: $\rho(k_i, k_{i+1})$ (median)')):
    ax.set_title(t); ax.set_xticks(x); ax.set_xticklabels(BINS); ax.axhline(0, color='0.7', lw=0.6, zorder=0)
axes[0].legend(fontsize=5.5, loc='upper right')
fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(OUT, 'corr_summary.pdf')); plt.close(fig)
res['corr_summary_NGC'] = summ

# ---------------------------------------------------------------- fig 5: chi2 consistency
R = json.load(open(os.path.join(D, 'results.json'))) if os.path.exists(os.path.join(D, 'results.json')) else json.load(open('results.json'))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
rows = ('holi', 'glam', 'abc', 'aba', 'a2c', 'a2a')
for ax, ref in zip(axes, ('holi', 'glam')):
    M = np.full((len(rows), len(BINS)), np.nan)
    for i, s in enumerate(rows):
        for j, b in enumerate(BINS):
            v = R['chi2'].get(f'{s}|{ref}|{b}|NGC')
            if v: M[i, j] = v['mean_over_p']
    im = ax.imshow(np.log2(M), vmin=-2, vmax=2, cmap='RdBu_r', aspect='auto')
    for i in range(len(rows)):
        for j in range(len(BINS)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6, color='k' if 0.5 < M[i, j] < 2 else 'w')
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels(BINS); ax.set_yticks(range(len(rows))); ax.set_yticklabels([LAB[s] for s in rows], fontsize=6)
    ax.set_title(f'mean $\\chi^2/p$ under the {LAB[ref]} covariance (NGC)', fontsize=7)
    for sp in ax.spines.values(): sp.set_visible(True)
fig.tight_layout(w_pad=1.0); fig.savefig(os.path.join(OUT, 'chi2_consistency.pdf')); plt.close(fig)

# ---------------------------------------------------------------- fig 6: fibre assignment, paired boxes
fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.3), sharex=True, sharey='row')
fa = {}
for j, b in enumerate(BINS):
    for alt, comp, ls in (('aba', 'abc', '-'),):
        ga, gc = data[alt][(b, 'NGC')], data[comp][(b, 'NGC')]
        common = np.intersect1d(ga['num'], gc['num'])
        Xa = an.vec(ga, sel)[np.isin(ga['num'], common)]; Xc = an.vec(gc, sel)[np.isin(gc['num'], common)]
        ra = Xa.var(0, ddof=1) / Xc.var(0, ddof=1)
        scale = np.concatenate([(Ptot(ga) / Ptot(gc)) ** 3] * 2)
        for row, sl in ((0, slice(0, n)), (1, slice(n, p))):
            axes[row, j].plot(ks, ra[sl], color=COL[alt], lw=0.9, label='raw')
            axes[row, j].plot(ks, (ra / scale)[sl], color=COL[alt], lw=0.9, ls='--', label='$P_{\\rm tot}^3$-scaled')
        fa[b] = dict(raw_B000=float(np.median(ra[:n])), scaled_B000=float(np.median((ra / scale)[:n])), raw_B202=float(np.median(ra[n:])), scaled_B202=float(np.median((ra / scale)[n:])),
                     Ptot_ratio_at_0p1=float((Ptot(ga) / Ptot(gc))[np.argmin(abs(ks - 0.1))]))
    for row in (0, 1):
        axes[row, j].axhline(1, color='0.6', lw=0.6, zorder=0); axes[row, j].set_yscale('log'); axes[row, j].set_ylim(0.25, 8)
        logticks(axes[row, j], [0.3, 1, 3]); kticks(axes[row, j])
    axes[0, j].set_title(b); axes[1, j].set_xlabel('$k$ [$h$/Mpc]')
axes[0, 0].set_ylabel('var(altmtl) / var(complete)\n$B_{000}$'); axes[1, 0].set_ylabel('$B_{202}$')
axes[0, 0].legend(fontsize=5.5, loc='upper left')
fig.tight_layout(w_pad=0.3, h_pad=0.6); fig.savefig(os.path.join(OUT, 'fibre_assignment.pdf')); plt.close(fig)
res['fibre_assignment_hf_NGC'] = fa

# ---------------------------------------------------------------- fig 7: holi QSO coherent mode
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
for s in ('holi', 'glam', 'abc'):
    for r, ls in (('NGC', '-'), ('SGC', ':')):
        g = data[s][('QSO', r)]; X = an.vec(g, sel); Z = (X - X.mean(0)) / X.std(0); cm = Z[:, :n].mean(1)
        axes[0].hist(cm / cm.std() if False else cm, bins=40, range=(-2.5, 2.5), histtype='step', color=COL[s], ls=ls, density=True,
                     label=f'{LAB[s]} {r}')
axes[0].set_xlabel(r'QSO $B_{000}$ common mode'); axes[0].set_ylabel('density'); axes[0].legend(fontsize=5, loc='upper left', bbox_to_anchor=(0.0, 1.0)); axes[0].set_xlim(-1.5, 2.5); axes[0].set_ylim(0, 2.6)
for s in ('holi', 'glam', 'abc', 'aba'):
    sds = []
    for b in BINS:
        g = data[s][(b, 'NGC')]; X = an.vec(g, sel); Z = (X - X.mean(0)) / X.std(0); sds.append(Z[:, :n].mean(1).std())
    axes[1].plot(BINS, sds, 'o-', color=COL[s], ms=3, lw=0.8, label=LAB[s])
axes[1].set_ylabel(r'sd of the $B_{000}$ common mode'); axes[1].set_title('coherent amplitude fluctuation (NGC)'); axes[1].legend(fontsize=5, loc='upper left')
A = np.load(os.path.join(D, 'results_arrays.npz')) if os.path.exists(os.path.join(D, 'results_arrays.npz')) else np.load('results_arrays.npz')
for s in ('holi', 'glam'):
    for r, ls in (('NGC', '-'), ('SGC', ':')):
        c2 = A[f'{s}|QSO|{r}|chi2']
        axes[2].hist(c2, bins=50, range=(30, 160), histtype='step', color=COL[s], ls=ls, density=True, label=f'{LAB[s]} {r}')
from scipy.stats import chi2 as _c2
xx = np.linspace(30, 160, 200); axes[2].plot(xx, _c2.pdf(xx, p), color='0.5', lw=0.7, label=f'$\\chi^2_{{{p}}}$')
axes[2].set_xlabel(r'QSO realisation $\chi^2$ (own covariance)'); axes[2].legend(fontsize=5, loc='upper right')
fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(OUT, 'holi_qso.pdf')); plt.close(fig)

# ---------------------------------------------------------------- fig 8: mean bispectra
fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.2), sharex=True)
for j, b in enumerate(BINS):
    for s in ('holi', 'glam', 'abc', 'aba', 'a2a'):
        if (b, 'NGC') not in data[s]:
            continue
        X = an.vec(data[s][(b, 'NGC')], sel)
        axes[0, j].plot(ks, ks ** 2 * X[:, :n].mean(0) / 1e6, color=COL[s], lw=0.9, label=LAB[s])
        axes[1, j].plot(ks, ks ** 2 * X[:, n:].mean(0) / 1e6, color=COL[s], lw=0.9)
    kticks(axes[0, j]); kticks(axes[1, j])
    axes[0, j].set_title(b); axes[1, j].set_xlabel('$k$ [$h$/Mpc]')
axes[0, 0].set_ylabel(r'$k^2 \langle B_{000} \rangle$ [$10^6$]'); axes[1, 0].set_ylabel(r'$k^2 \langle B_{202} \rangle$ [$10^6$]')
fig.tight_layout(w_pad=0.3, h_pad=0.6, rect=(0, 0, 1, 0.93)); top_legend(fig, ('holi', 'glam', 'abc', 'aba', 'a2a'), 0.93)
fig.savefig(os.path.join(OUT, 'mean_bispectra.pdf')); plt.close(fig)

# signal-to-noise with each large suite's own covariance
sn = {}
for s in ('holi', 'glam'):
    for b in BINS:
        for r in ('NGC', 'SGC', 'GCcomb'):
            X = an.vec(data[s][(b, r)], sel); N = len(X)
            Ci = np.linalg.inv(np.cov(X, rowvar=False)) * (N - p - 2) / (N - 1); mu = X.mean(0)
            sn[f'{s}|{b}|{r}'] = float(np.sqrt(mu @ Ci @ mu))
res['snr'] = sn
json.dump(res, open(os.path.join(OUT, 'figure_results.json'), 'w'), indent=1)
print('figures done')
