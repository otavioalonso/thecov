import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import *
from binwidth import run_decomp, ratios, F

C1, C2, C3, C4, C5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 1.6, 'legend.frameon': False,
                     'savefig.bbox': 'tight', 'font.family': 'serif', 'mathtext.fontset': 'cm'})
ELL_C = {0: C1, 2: C2, 4: C3}
RES = {}

def coarse_ratio(name, b, r, mode=None, f=1):
    d = get(name, b, r, f, mode)
    rr = d['V'].var(0, ddof=1) / np.diag(d['C'])
    nb = len(d['k']); per = 4 // f
    return get(name, b, r, 4)['k'], rr.reshape(3, nb // per, per).mean(-1), np.sqrt(2 / (len(d['V']) - 1) / per)

# ---------------------------------------------------------------- Fig 1: altmtl vs complete
fig, ax = plt.subplots(1, 4, figsize=(10.5, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 1, 1]))
RES['paired'] = {}
for i, r in enumerate(['NGC', 'SGC', 'GCcomb']):
    a, b_ = get('complete', 'LRG1', r), get('altmtl', 'LRG1', r)
    assert (a['ids'] == b_['ids']).all()
    xa, xb = chi2_i(a['V'], a['C']), chi2_i(b_['V'], b_['C'])
    d = xa - xb
    RES['paired'][r] = dict(complete=float(xa.mean()), altmtl=float(xb.mean()), diff=float(d.mean()),
                            diff_err=float(d.std(ddof=1) / np.sqrt(len(d))), corr=float(np.corrcoef(xa, xb)[0, 1]))
    lo, hi = min(xa.min(), xb.min()) - 0.05, max(xa.max(), xb.max()) + 0.05
    ax[i].plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls='--')
    ax[i].scatter(xa, xb, s=16, color=C1, edgecolor='white', linewidth=0.6, zorder=3)
    ax[i].set_xlim(lo, hi); ax[i].set_ylim(lo, hi); ax[i].set_aspect('equal')
    ax[i].set_xlabel(r'$\chi^2_i/n$, complete'); ax[i].set_title(f'{r}')
    ax[i].text(0.04, 0.96, f"$r={RES['paired'][r]['corr']:.2f}$\n$\\Delta={d.mean():+.3f}\\pm{d.std(ddof=1)/np.sqrt(len(d)):.3f}$",
               transform=ax[i].transAxes, va='top', fontsize=7.5, color=INK)
ax[0].set_ylabel(r'$\chi^2_i/n$, altmtl')
a3 = ax[3]
for name, ls, lab in (('holi', '-', 'holi altmtl (859)'), ('complete', '--', 'Abacus complete (25)'), ('altmtl', ':', 'Abacus altmtl (25)')):
    rs = [coarse_ratio(name, 'LRG1', r) for r in ('NGC', 'SGC')]
    kc, R = rs[0][0], np.mean([x[1] for x in rs], 0)
    for j, l in enumerate((0, 2)):
        a3.plot(kc, R[j], ls=ls, color=ELL_C[l], label=lab if j == 0 else None)
a3.axhline(1, color=INK2, lw=0.8)
a3.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$'); a3.set_ylabel(r'$\sigma^2_{\rm mock}/\sigma^2_{\rm thecov}$')
a3.set_title(r'LRG1 caps: $P_0$ (blue), $P_2$ (orange)'); a3.legend(loc='upper left', handlelength=2.2)
fig.tight_layout(); fig.savefig('fig1_altmtl_complete.pdf'); fig.savefig('fig1_altmtl_complete.png', dpi=110); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: bin width
fig, ax = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
RES['binwidth'] = {}
for b in ('LRG1', 'QSO'):
    R, ab, err, kc = run_decomp('holi', b, nboot=200)
    RES['binwidth'][b] = dict(k=kc.tolist(), R=R.tolist(), a=ab[0].tolist(), b=ab[1].tolist(), a_err=err[0].tolist(), b_err=err[1].tolist())
R = np.array(RES['binwidth']['LRG1']['R'])
for j, l in enumerate((0, 2, 4)):
    for fi, (f, ls) in enumerate(zip(F, ('-', '--', ':'))):
        ax[0, j].plot(kc, R[fi, j] - 1, ls=ls, color=ELL_C[l], label=rf'$\Delta k={0.005*f:.3f}$')
    ax[0, j].set_title(rf'LRG1 $P_{l}$: $\sigma^2_{{\rm mock}}/\sigma^2_{{\rm thecov}}-1$')
    ax[0, j].axhline(0, color=INK2, lw=0.8)
    for b, mk, alpha in (('LRG1', 'o', 1.0), ('QSO', 's', 0.45)):
        rb = RES['binwidth'][b]
        A, B, eA, eB = (np.array(rb[x])[j] for x in ('a', 'b', 'a_err', 'b_err'))
        ax[1, j].errorbar(kc - 0.002, A, eA, fmt=mk + '-', color=C5, ms=3.5, lw=1.1, alpha=alpha, capsize=0, label=f'{b}: $a$ (Gaussian-like)')
        ax[1, j].errorbar(kc + 0.002, B, eB, fmt=mk + '-', color=C4, ms=3.5, lw=1.1, alpha=alpha, capsize=0, label=f'{b}: $b$ (per $\\Delta k=0.005$)')
    ax[1, j].axhline(0, color=INK2, lw=0.8)
    ax[1, j].set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$'); ax[1, j].set_title(rf'$P_{l}$: excess $= a(k) + b(k)\,\Delta k/0.005$')
ax[0, 0].legend(loc='upper left'); ax[1, 0].legend(loc='upper left', fontsize=6.8)
fig.tight_layout(); fig.savefig('fig2_binwidth.pdf'); fig.savefig('fig2_binwidth.png', dpi=110); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: weights (naive vs smoothed m)
fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), sharey=False)
for i, (name, b) in enumerate((('holi', 'LRG1'), ('holi', 'QSO'), ('complete', 'LRG1'))):
    for mode, ls, lab in (('random-density', '-', r'smoothed $m$'), ('none', '--', r'naive (own weight)')):
        rs = [coarse_ratio(name, b, r, mode) for r in ('NGC', 'SGC')]
        kc, Rm = rs[0][0], np.mean([x[1] for x in rs], 0)
        for j, l in enumerate((0, 2)):
            ax[i].plot(kc, Rm[j], ls=ls, color=ELL_C[l], label=f'{lab}' if j == 0 else None)
    ax[i].axhline(1, color=INK2, lw=0.8)
    ax[i].set_title({'holi': 'holi', 'complete': 'Abacus complete'}[name] + f' {b}: $P_0$ (blue), $P_2$ (orange)')
    ax[i].set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$')
ax[0].set_ylabel(r'$\sigma^2_{\rm mock}/\sigma^2_{\rm thecov}$'); ax[0].legend(loc='upper left')
fig.tight_layout(); fig.savefig('fig3_weights.pdf'); fig.savefig('fig3_weights.png', dpi=110); plt.close(fig)
RES['weights'] = {}
for name in ('holi', 'complete'):
    z, m = run(name)
    for b, mb in m['bins'].items():
        for r in ('NGC', 'SGC'):
            w, ve, ti = mb['weights'][r], mb['veff'][r], mb['tracer_info']
            RES['weights'][f'{name} {b} {r}'] = dict(
                w2_data=w['data_w2_over_wmean2'], w2_rand=w['randoms_w2_over_wmean2'],
                veff_naive_over_smooth=ve['own-weight'] / ve['random-density'], veff_nx_over_smooth=ve['nx'] / ve['random-density'],
                I_over_norm_smooth=ti[f'{r}__random-density']['I_over_norm'], I_over_norm_naive=ti[f'{r}__none']['I_over_norm'],
                chi2_smooth=mb['chi2'][f'{r} | random-density'][0], chi2_naive=mb['chi2'][f'{r} | none'][0],
                chi2_nx=mb['chi2'][f'{r} | nx'][0], N_data=w['N_data'])

# ---------------------------------------------------------------- Fig 4: directions
fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
RES['directions'] = {}
for b, col in (('LRG1', C1), ('QSO', C2)):
    d = get('holi', b, 'NGC'); V, C = d['V'], d['C']; n = V.shape[1]; N = len(V)
    L = np.linalg.cholesky(C); Li = np.linalg.inv(L)
    E = Li @ np.cov(V.T) @ Li.T
    ev, evec = np.linalg.eigh(E)
    lo, hi = mp_edges(n, N)
    ax[0].plot(np.arange(1, n + 1), ev[::-1], color=col, label=b)
    if b == 'LRG1':
        ax[0].axhspan(lo, hi, color=GRID, alpha=0.6, lw=0, label='Marchenko–Pastur')
        # leading excess directions in data space, against templates
        P = V.mean(0); k = d['k']; nb = len(k)
        tmpl = {'amplitude': P, 'dilation': -np.concatenate([k * np.gradient(P[j*nb:(j+1)*nb], k) for j in range(3)]),
                'P2 amplitude': np.concatenate([0 * k, P[nb:2*nb], 0 * k])}
        RES['directions']['LRG1 NGC top eigenvalues'] = ev[::-1][:6].tolist()
        RES['directions']['MP edges'] = [lo, hi]
        ov = {}
        for t, vec in tmpl.items():
            u = Li @ vec; u /= np.linalg.norm(u)
            ov[t] = [float((evec[:, -i - 1] @ u) ** 2) for i in range(3)]
            ov[t + ' rayleigh'] = float(u @ E @ u)
        RES['directions']['template overlaps (top 3) and u^T E u'] = ov
        for i, ls in zip(range(3), ('-', '--', ':')):
            vd = L @ evec[:, -i - 1]
            vd = vd / np.sqrt(np.diag(C))           # in units of thecov sigma per element
            s = np.sign(vd[:nb].sum()) or 1
            for j, l in enumerate((0, 2, 4)):
                ax[1].plot(k, s * vd[j*nb:(j+1)*nb], ls=ls, color=ELL_C[l], lw=1.2,
                           label=(rf'$P_{l}$' if i == 0 else None))
        ax[1].text(0.98, 0.04, 'solid/dashed/dotted: 1st/2nd/3rd', transform=ax[1].transAxes, ha='right', fontsize=7, color=INK2)
ax[0].set_xscale('log'); ax[0].set_xlabel('rank'); ax[0].set_ylabel(r'eigenvalue of $C^{-1/2}\hat C C^{-1/2}$')
ax[0].set_title('whitened mock covariance (NGC)'); ax[0].legend()
ax[1].axhline(0, color=INK2, lw=0.8); ax[1].set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$'); ax[1].set_ylabel(r'$\delta P/\sigma_{\rm thecov}$')
ax[1].set_title('LRG1 NGC: leading excess directions'); ax[1].legend(loc='upper left')
for f, mk, col in ((1, 'o', C1), (2, 's', C2)):
    d = get('holi', 'LRG1', 'NGC', f); C = d['C']; D = np.sqrt(np.diag(C))
    e, v = np.linalg.eigh(C / np.outer(D, D))
    Cm = np.cov(d['V'].T) / np.outer(D, D)
    ratio = np.einsum('ij,ik,kj->j', v, Cm, v) / e
    ax[2].scatter(e, ratio, s=9, color=col, label=rf'$\Delta k={0.005*f:.3f}$', edgecolor='white', linewidth=0.4)
    RES['directions'][f'near-null x{f}'] = dict(eig=e[:6].tolist(), ratio=ratio[:6].tolist())
ax[2].set_xscale('log'); ax[2].axhline(1, color=INK2, lw=0.8)
ax[2].set_xlabel('eigenvalue of thecov correlation matrix'); ax[2].set_ylabel('mock / thecov variance along it')
ax[2].set_title('LRG1 NGC: near-null directions'); ax[2].legend()
fig.tight_layout(); fig.savefig('fig4_directions.pdf'); fig.savefig('fig4_directions.png', dpi=110); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: amplitude scatter and norm regression
def amp(V, C, P, idx):
    Ci = np.linalg.inv(C[np.ix_(idx, idx)]); t = P[idx]
    return (V[:, idx] - t) @ Ci @ t / (t @ Ci @ t), 1 / np.sqrt(t @ Ci @ t)
fig, ax = plt.subplots(1, 2, figsize=(8.4, 2.9))
RES['amplitude'] = {}
ranges = ((0.02, 0.1), (0.1, 0.2), (0.2, 0.3))
for b, col in (('LRG1', C1), ('QSO', C2)):
    out = {}
    for r in ('NGC', 'SGC'):
        d = get('holi', b, r); V, C, k = d['V'], d['C'], d['k']; P = V.mean(0); nb = len(k)
        for l, off in ((0, 0), (2, nb)):
            for lo, hi in ranges:
                idx = off + np.flatnonzero((k >= lo) & (k < hi))
                A, g = amp(V, C, P, idx)
                ex = A.var() - 1.0 * g ** 2
                out.setdefault(f'P{l} {lo}-{hi}', []).append(dict(sig=float(A.std()), gauss=float(g), excess=float(np.sign(ex) * np.sqrt(abs(ex)))))
        dn = d['norm'] / d['norm'].mean() - 1
        A, g = amp(V, C, P, np.arange(V.shape[1]))
        out.setdefault('norm', []).append(dict(sigma_dnorm=float(dn.std()), corr_A_dnorm=float(np.corrcoef(A, dn)[0, 1]),
                                               sigma_A=float(A.std()), gauss=float(g)))
    RES['amplitude'][b] = out
    x = np.arange(3)
    for l, mk, ls in ((0, 'o', '-'),):
        y = [np.mean([o['excess'] for o in out[f'P{l} {lo}-{hi}']]) for lo, hi in ranges]
        g = [np.mean([o['gauss'] for o in out[f'P{l} {lo}-{hi}']]) for lo, hi in ranges]
        ax[0].plot(x, 100 * np.array(y), mk + ls, color=col, label=f'{b}: excess')
        ax[0].plot(x, 100 * np.array(g), mk + ':', color=col, mfc='white', label=f'{b}: Gaussian')
ax[0].set_xticks(x, [f'{lo}–{hi}' for lo, hi in ranges]); ax[0].set_xlabel(r'$k$ range of the $P_0$ amplitude fit')
ax[0].set_ylabel(r'rms of $A_0$ [%]'); ax[0].set_title(r'$P_0$ amplitude scatter (mean of caps)'); ax[0].legend(fontsize=7)
RES['la_regression'] = {}
for b, col in (('LRG1', C1), ('QSO', C2)):
    betas = []
    for r in ('NGC', 'SGC'):
        d = get('holi', b, r, 4); V = d['V']; P = V.mean(0); nb = len(d['k'])
        dn = d['norm'] / d['norm'].mean() - 1
        X = np.vstack([np.ones(len(V)), dn]).T
        coef, res, *_ = np.linalg.lstsq(X, V, rcond=None)
        resid = V - X @ coef
        se = np.sqrt(resid.var(0, ddof=2) / ((dn - dn.mean()) ** 2).sum())
        betas.append((coef[1][:nb] / P[:nb], se[:nb] / P[:nb]))
    bm = np.mean([x[0] for x in betas], 0); be = np.sqrt(np.sum([x[1] ** 2 for x in betas], 0)) / 2
    RES['la_regression'][b] = dict(k=d['k'].tolist(), beta_over_P0=bm.tolist(), err=be.tolist())
    ax[1].errorbar(d['k'], bm, be, fmt='o-', color=col, ms=3.5, lw=1.1, label=b, capsize=0)
ax[1].axhline(0, color=INK2, lw=0.8)
ax[1].axhline(-1, color=INK2, lw=0.8, ls='--'); ax[1].text(0.29, -0.93, r'$\hat P\propto 1/{\rm norm}$', ha='right', fontsize=7, color=INK2)
ax[1].set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$'); ax[1].set_ylabel(r'$\partial \hat P_0/\partial\,\delta_{\rm norm}\ /\ \bar P_0$')
ax[1].set_title(r'response of $\hat P_0$ to the realised norm'); ax[1].legend()
fig.tight_layout(); fig.savefig('fig5_amplitude.pdf'); fig.savefig('fig5_amplitude.png', dpi=110); plt.close(fig)

# ---------------------------------------------------------------- parameter projections
def params(name, b, r, kmax):
    d = get(name, b, r); V, C, k = d['V'], d['C'], d['k']; nb = len(k)
    P = V.mean(0)
    dil = -np.concatenate([k * np.gradient(P[j*nb:(j+1)*nb], k) for j in range(3)])
    T = np.vstack([P, np.concatenate([0 * k, P[nb:2*nb], 0 * k]), dil, np.concatenate([np.ones(nb), 0 * k, 0 * k])]).T
    idx = kidx(k, kmax)
    T, Vs, Cs = T[idx], V[:, idx], C[np.ix_(idx, idx)]
    Ci = np.linalg.inv(Cs); Fm = T.T @ Ci @ T; Fi = np.linalg.inv(Fm)
    th = (Vs - Vs.mean(0)) @ Ci @ T @ Fi
    ratio_joint = th.var(0, ddof=1) / np.diag(Fi)
    single = []
    for p in range(T.shape[1]):
        t = T[:, p]; s2 = 1 / (t @ Ci @ t); est = (Vs - Vs.mean(0)) @ Ci @ t * s2
        single.append(est.var(ddof=1) / s2)
    return ratio_joint.tolist(), single, len(V)
RES['params'] = {}
for name in ('holi', 'complete', 'altmtl'):
    for b in (('LRG1', 'QSO') if name == 'holi' else ('LRG1',)):
        for r in ('NGC', 'SGC', 'GCcomb'):
            for kmax in (0.2, 0.3):
                j, s, N = params(name, b, r, kmax)
                RES['params'][f'{name}|{b}|{r}|{kmax}'] = dict(joint=j, single=s, N=N)
json.dump(RES, open('results.json', 'w'), indent=1)
print('done')
