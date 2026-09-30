"""Figures and numbers for the mock-validation report (run from the repository root).

    python reports/validation_v5/make_figures.py <run_dir> <control_run_dir>

run_dir holds comparison.npz, mocks.npz and windows.npz of the mock_version-4 run; control_run_dir
the comparison.npz of the mock_version-2 run (expected shot noise), used as a control.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.getcwd())
from mocks.survey import Footprint, make_grid, Catalogues, model_multipoles
from mocks.run_validation import pk_lin, BIAS, GROWTH

run, ctrl = sys.argv[1], sys.argv[2]
OUT = os.path.dirname(os.path.abspath(__file__))
f = np.load(os.path.join(run, 'comparison.npz'), allow_pickle=True)
V, C, Ch, k = f['vectors'], f['C_analytic'], f['C_mock'], f['k_eff']
mk = np.load(os.path.join(run, 'mocks.npz'))
N, nd = V.shape
nb = len(k)
SPEC = [('A', 'A'), ('A', 'B'), ('B', 'B')]
ELLS = (0, 2)
COL = {'AA': '#2a78d6', 'AB': '#eb6834', 'BB': '#1baf7a'}      # validated categorical slots 1-3
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4, 'legend.frameon': False,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'serif',
                     'mathtext.fontset': 'cm', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
nums = {}


def blk(i, j=None):
    return slice(i * nb, (i + 1) * nb)


def corr(M):
    d = np.sqrt(np.diag(M))
    return M / np.outer(d, d)


# ------------------------------------------------------------------ geometry
fp = Footprint()
grid = make_grid(fp, 256, 2.0)
cat = Catalogues(fp, grid, n_random_factor=15.0)
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={'width_ratios': [1.25, 1]})
r = np.linspace(380, 920, 400)
for t, c in (('A', COL['AA']), ('B', COL['BB'])):
    ax[0].plot(r, fp.radial(r, t) * 1e4, color=c)
    ax[0].text(r[np.argmax(fp.radial(r, t))], fp.radial(r, t).max() * 1e4 * 1.04, f'tracer {t}', color=INK,
               ha='center', va='bottom', fontsize=8)
ax[0].set_xlabel(r'$r$ [$h^{-1}$Mpc]'); ax[0].set_ylabel(r'$\bar n(r)$ [$10^{-4}\,h^3$Mpc$^{-3}$]')
ax[0].set_ylim(0, 7.2)
pos = cat.randoms['A'][::40]
rr = np.linalg.norm(pos, axis=1)
w = cat.w_ran['A'][::40]
sc = ax[1].scatter(np.degrees(pos[:, 0] / rr), np.degrees(pos[:, 1] / rr), s=0.3, c=cat.nbar_ran['A'][::40] * 1e4,
                   cmap='Blues', rasterized=True, vmin=0)
ax[1].set_aspect('equal'); ax[1].set_xlabel(r'$\hat x_1$ [deg]'); ax[1].set_ylabel(r'$\hat x_2$ [deg]')
cb = fig.colorbar(sc, ax=ax[1], shrink=0.85); cb.set_label(r'$\bar n_A$ [$10^{-4}$]', fontsize=7.5)
fig.savefig(os.path.join(OUT, 'fig_geometry.pdf')); plt.close(fig)
nums['N_ran'] = {t: int(len(cat.randoms[t])) for t in cat.tracers}
nums['N_gal'] = {t: float(cat.n_gal_expected[t]) for t in cat.tracers}
nums['alpha'] = {t: float(cat.alpha[t]) for t in cat.tracers}

# ------------------------------------------------------------------ mean multipoles
mean = V.mean(0)
sig_t = np.sqrt(np.diag(C))
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), sharex=True)
for s_i, (X, Y) in enumerate(SPEC):
    for e_i, ell in enumerate(ELLS):
        b = blk(2 * s_i + e_i)
        a = ax[e_i]
        a.fill_between(k, k * (mean[b] - sig_t[b]), k * (mean[b] + sig_t[b]), color=COL[X + Y], alpha=0.18, lw=0)
        a.plot(k, k * mean[b], color=COL[X + Y], label=X + Y)
for e_i, ell in enumerate(ELLS):
    ax[e_i].set_xlabel(r'$k$ [$h$Mpc$^{-1}$]')
    ax[e_i].set_ylabel(rf'$k\,\langle \hat P_{ell}\rangle$ [$h^{{-2}}$Mpc$^2$]')
    ax[e_i].axhline(0, color=GRID, lw=0.8, zorder=0)
ax[0].legend(loc='upper right')
fig.savefig(os.path.join(OUT, 'fig_multipoles.pdf')); plt.close(fig)

# ------------------------------------------------------------------ sigma ratio
ratio = np.sqrt(np.diag(Ch) / np.diag(C))
band = 1 / np.sqrt(2 * (N - 1))
fig, ax = plt.subplots(2, 3, figsize=(6.8, 3.4), sharex=True, sharey=True)
for s_i, (X, Y) in enumerate(SPEC):
    for e_i, ell in enumerate(ELLS):
        a = ax[e_i, s_i]
        a.axhspan(1 - band, 1 + band, color=GRID, lw=0, zorder=0)
        a.axhspan(1 - 2 * band, 1 + 2 * band, color=GRID, alpha=0.45, lw=0, zorder=0)
        a.axhline(1, color=INK2, lw=0.6)
        a.plot(k, ratio[blk(2 * s_i + e_i)], 'o-', color=COL[X + Y], ms=2.5, lw=0.9)
        a.text(0.97, 0.92, rf'{X}{Y}, $\ell={ell}$', transform=a.transAxes, ha='right', va='top', color=INK)
        a.set_ylim(0.86, 1.14)
for a in ax[1]:
    a.set_xlabel(r'$k$ [$h$Mpc$^{-1}$]')
for a in ax[:, 0]:
    a.set_ylabel(r'$\sigma_{\rm mock}/\sigma_{\rm thecov}$')
fig.savefig(os.path.join(OUT, 'fig_sigma_ratio.pdf')); plt.close(fig)
nums['sigma_ratio_rms_dev'] = float(np.std(ratio - 1))
nums['sigma_ratio_band'] = float(band)
nums['var_ratio_mean'] = float(np.mean(ratio ** 2))

# low-k: mean variance ratio over the first three bins of all blocks, bootstrap over mocks
rng = np.random.default_rng(0)
lowk = np.concatenate([np.arange(b * nb, b * nb + 3) for b in range(6)])
rest = np.concatenate([np.arange(b * nb + 3, (b + 1) * nb) for b in range(6)])
def mv(idx, Vs):
    return np.mean(np.var(Vs[:, idx], axis=0, ddof=1) / np.diag(C)[idx])
boot = np.array([[mv(lowk, V[s]), mv(rest, V[s])] for s in (rng.integers(0, N, N) for _ in range(400))])
nums['lowk_var_ratio'] = [float(mv(lowk, V)), float(boot[:, 0].std())]
nums['rest_var_ratio'] = [float(mv(rest, V)), float(boot[:, 1].std())]
nums['lowk_per_bin'] = [[float(np.mean(ratio[[b * nb + j for b in range(6)]] ** 2)) for j in range(3)]]

# ------------------------------------------------------------------ correlation matrices
cm, ca = corr(Ch), corr(C)
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.1))
M = np.tril(cm, -1) + np.triu(ca, 1) + np.eye(nd)
im = ax[0].imshow(M, cmap='RdBu_r', vmin=-0.6, vmax=0.6, interpolation='nearest')
ax[0].set_title('mocks (lower) vs. thecov (upper)', fontsize=8.5, color=INK)
cb = fig.colorbar(im, ax=ax[0], shrink=0.8)
with np.errstate(divide='ignore', invalid='ignore'):
    z = (cm - ca) / ((1 - ca ** 2) / np.sqrt(N - 1))
iu = np.triu_indices(nd, 1)
im2 = ax[1].imshow(np.where(np.eye(nd, dtype=bool), 0, z), cmap='RdBu_r', vmin=-4, vmax=4, interpolation='nearest')
ax[1].set_title(r'$(\hat\rho_{ij}-\rho_{ij})\,/\,\sigma_\rho$', fontsize=8.5, color=INK)
cb2 = fig.colorbar(im2, ax=ax[1], shrink=0.8)
labels = [f'{X}{Y}{ell}' for (X, Y) in SPEC for ell in ELLS]
for a in ax:
    a.set_xticks(np.arange(6) * nb + nb / 2 - 0.5); a.set_xticklabels(labels, fontsize=7)
    a.set_yticks(np.arange(6) * nb + nb / 2 - 0.5); a.set_yticklabels(labels, fontsize=7)
    for b in range(1, 6):
        a.axhline(b * nb - 0.5, color='w', lw=0.5); a.axvline(b * nb - 0.5, color='w', lw=0.5)
fig.savefig(os.path.join(OUT, 'fig_correlation.pdf'), dpi=300); plt.close(fig)
nums['corr_resid_std'] = float(z[iu].std())
nums['corr_resid_mean'] = float(z[iu].mean())
nums['corr_resid_frac_gt3'] = float(np.mean(np.abs(z[iu]) > 3))
nums['corr_resid_expected_gt3'] = float(2 * stats.norm.sf(3))

# ------------------------------------------------------------------ chi2 and eigenvalues
L = np.linalg.cholesky(C)
Z = np.linalg.solve(L, (V - mean).T)
chi2 = np.sum(Z ** 2, axis=0)
exp = nd * (1 - 1 / N)
Mw = np.linalg.solve(L, np.linalg.solve(L, Ch).T).T
ev = np.linalg.eigvalsh(0.5 * (Mw + Mw.T))
q = nd / (N - 1)
lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
ax[0].hist(chi2, bins=40, density=True, color=COL['AA'], alpha=0.55, lw=0)
x = np.linspace(chi2.min(), chi2.max(), 300)
ax[0].plot(x, stats.chi2.pdf(x * nd / exp, df=nd) * nd / exp, color=INK, lw=1.2)
ax[0].axvline(chi2.mean(), color=COL['AB'], lw=1.2)
ax[0].text(0.03, 0.92, rf'$\langle\chi^2\rangle={chi2.mean():.1f}$' + '\n' + rf'expected ${exp:.1f}\pm{exp*np.sqrt(2/(nd*N)):.1f}$', transform=ax[0].transAxes, ha='left', va='top', color=INK)
ax[0].set_xlabel(r'$\chi^2_i=(d_i-\bar d)^{\rm T}C^{-1}(d_i-\bar d)$')
ax[0].set_ylabel('density')
ax[1].hist(ev, bins=35, density=True, color=COL['BB'], alpha=0.55, lw=0)
xe = np.linspace(lo, hi, 400)
ax[1].plot(xe, np.sqrt(np.maximum((hi - xe) * (xe - lo), 0)) / (2 * np.pi * q * xe), color=INK, lw=1.2)
ax[1].set_xlabel(r'eigenvalues of $C^{-1/2}\hat C\,C^{-1/2}$'); ax[1].set_ylabel('density')
ax[1].text(0.97, 0.92, 'Marchenko–Pastur\n(exact $C$)', transform=ax[1].transAxes, ha='right', va='top', color=INK)
fig.savefig(os.path.join(OUT, 'fig_chi2_eigen.pdf')); plt.close(fig)
nums.update(chi2_mean=float(chi2.mean()), chi2_exp=float(exp), chi2_err=float(exp * np.sqrt(2 / (nd * N))),
            chi2_var=float(chi2.var()), chi2_var_exp=float(2 * exp), ev_min=float(ev.min()), ev_max=float(ev.max()),
            mp_lo=float(lo), mp_hi=float(hi), ks_p=float(stats.kstest(chi2 * nd / exp, stats.chi2(df=nd).cdf).pvalue))

# per-block chi2
half = nb // 2
pb = {}
for b, name in enumerate(labels):
    row = []
    for sl in (slice(b * nb, (b + 1) * nb), slice(b * nb, b * nb + half), slice(b * nb + half, (b + 1) * nb)):
        Lb = np.linalg.cholesky(C[sl, sl]); zb = np.linalg.solve(Lb, (V[:, sl] - mean[sl]).T)
        dim = Lb.shape[0]; e = dim * (1 - 1 / N); m = np.sum(zb ** 2, axis=0).mean()
        row.append([float(m / e), float((m - e) / (e * np.sqrt(2 / (dim * N))))])
    pb[name] = row
nums['per_block'] = pb
nums['k_split'] = float(k[half])

# ------------------------------------------------------------------ control: expected shot noise
fc = np.load(os.path.join(ctrl, 'comparison.npz'), allow_pickle=True)
Vc, Cc, Chc = fc['vectors'], fc['C_analytic'], fc['C_mock']
P = model_multipoles(k, lambda kk: pk_lin(kk, 0.6), BIAS, {}, GROWTH, ('A', 'B'))
Cs = Cc.copy()
for t, b in (('A', 0), ('B', 4)):
    a_ = cat.alpha[t]; wr = cat.w_ran[t]; nr = cat.nbar_ran[t]; I = cat.I(t, t); p = P[(t, t)][0]
    Cs[blk(b), blk(b)] += (a_ * np.sum(wr ** 4) + 2 * (p[:, None] + p[None, :]) * a_ * np.sum(nr * wr ** 4)) / I ** 2
def chi2m(Vx, Cx):
    Lx = np.linalg.cholesky(Cx); zx = np.linalg.solve(Lx, (Vx - Vx.mean(0)).T); return float(np.sum(zx ** 2, 0).mean())
nums['control'] = dict(gauss=chi2m(Vc, Cc), selfpair=chi2m(Vc, Cs))
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
for a, (t, b) in zip(ax, (('A', 0), ('B', 4))):
    a.axhspan(1 - band, 1 + band, color=GRID, lw=0, zorder=0)
    a.axhline(1, color=INK2, lw=0.6)
    rc = np.sqrt(np.diag(Chc)[blk(b)] / np.diag(Cc)[blk(b)])
    rs = np.sqrt(np.diag(Chc)[blk(b)] / np.diag(Cs)[blk(b)])
    a.plot(k, rc, 'o-', ms=2.5, lw=0.9, color=COL['AB'], label='expected shot noise, Gaussian $C$')
    a.plot(k, rs, 's-', ms=2.5, lw=0.9, color=COL['AA'], label=r'same mocks, $C$ + self-pair terms')
    a.plot(k, ratio[blk(b)], '^-', ms=2.5, lw=0.9, color=COL['BB'], label='realised shot noise (this run)')
    a.text(0.03, 0.92, f'{t}{t}, $\\ell=0$', transform=a.transAxes, color=INK)
    a.set_xlabel(r'$k$ [$h$Mpc$^{-1}$]')
ax[0].set_ylabel(r'$\sigma_{\rm mock}/\sigma_{\rm thecov}$')
h, l = ax[1].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.08))
fig.savefig(os.path.join(OUT, 'fig_control.pdf')); plt.close(fig)

# ------------------------------------------------------------------ sum_g w^2 check
k3 = grid.knorm(); sw = {}
for j, t in enumerate(('A', 'B')):
    nbar = cat.nbar[t]; wg = np.where(nbar > 0, 1 / (1 + nbar * cat.P0_fkp), 0); bb = BIAS[t]
    Pg = (bb * bb + 2 * bb * GROWTH / 3 + GROWTH ** 2 / 5) * np.where(k3 > 0, pk_lin(np.where(k3 > 0, k3, 1), 0.6), 0)
    Wk = np.fft.rfftn(nbar * wg * wg) * grid.V_cell
    mult = np.full(Wk.shape, 2.0); mult[..., 0] = 1; mult[..., -1] = 1
    clus = float(np.sum(mult * Pg * np.abs(Wk) ** 2) / grid.V_box)
    pois = float(cat.alpha[t] * np.sum(cat.w_ran[t] ** 4))
    sw[t] = dict(meas=float(mk['sumw2'][:, j].var()), pois=pois, clus=clus)
nums['sumw2'] = sw
nums['selfpair_frac'] = {t: [float((cat.alpha[t] * np.sum(cat.w_ran[t] ** 4) / cat.I(t, t) ** 2) / np.diag(C)[b * nb + j])
                             for j in (10, 20, 28)] for t, b in (('A', 0), ('B', 4))}

# ------------------------------------------------------------------ windows
w = np.load(os.path.join(run, 'windows.npz'))
meta = json.loads(str(w['meta'].item()))
se = w['s_edges']; sc_ = 0.5 * (se[1:] + se[:-1])
fig, ax = plt.subplots(figsize=(3.3, 2.4))
want = [((('W', 'A', 'A'), ('W', 'A', 'A')), 'AA', r'$W^{AA}\!\times W^{AA}$'),
        ((('W', 'A', 'B'), ('W', 'A', 'B')), 'AB', r'$W^{AB}\!\times W^{AB}$'),
        ((('W', 'B', 'B'), ('W', 'B', 'B')), 'BB', r'$W^{BB}\!\times W^{BB}$')]
for key, c, lab in want:
    for rec in meta:
        kk = tuple(tuple(x) for x in rec['key'])
        if kk == key and tuple(rec['trip']) == (0, 0, 0):
            Q = w[rec['q']]
            ax.plot(sc_, Q / Q[0], color=COL[c], label=lab)
        if kk == key and tuple(rec['trip']) == (2, 0, 2):
            Q2 = w[rec['q']]
for key, c, lab in want[:1]:
    for rec in meta:
        if tuple(tuple(x) for x in rec['key']) == key and tuple(rec['trip']) == (2, 0, 2):
            Q00 = [w[r['q']] for r in meta if tuple(tuple(x) for x in r['key']) == key and tuple(r['trip']) == (0, 0, 0)][0]
            ax.plot(sc_, w[rec['q']] / Q00[0], color=COL[c], ls='--', label=r'$Q_{202}$, $W^{AA}\!\times W^{AA}$')
ax.axhline(0, color=GRID, lw=0.8, zorder=0)
ax.set_xlabel(r'$s$ [$h^{-1}$Mpc]'); ax.set_ylabel(r'$Q_{\Lambda_1\Lambda_2\Lambda}(s)/Q_{000}(s_1)$')
ax.legend(fontsize=7)
fig.savefig(os.path.join(OUT, 'fig_windows.pdf')); plt.close(fig)

json.dump(nums, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1)
print(json.dumps(nums, indent=1))
