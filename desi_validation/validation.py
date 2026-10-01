"""The statistical tests of the mock-validation report, for one data vector (one tracer bin, one
region): numbers in a dict, figures on request.

    out = validate(V, C, k, ells)               # V (N_mock, n), C (n, n), k (nb,), ells
    figures(out, V, C, k, ells, title=...)      # the report's figures for this case

With an exact C and N mocks of an n-element vector (q = n / (N - 1)):

* chi^2_i = (d_i - dbar)^T C^-1 (d_i - dbar):  <chi^2> = n (1 - 1/N) +- n (1 - 1/N) sqrt(2 / (n N)),
  variance ~ 2n, distribution chi^2_n rescaled by 1 - 1/N (Kolmogorov-Smirnov p-value);
* eigenvalues of C^-1/2 Chat C^-1/2 fill the Marchenko-Pastur interval [(1-sqrt q)^2, (1+sqrt q)^2];
* sigma_mock / sigma_thecov per element scatters by 1/sqrt(2 (N-1));
* correlation residuals z_ij = (rhohat_ij - rho_ij) sqrt(N-1) / (1 - rho_ij^2) are unit Gaussians.

The chi^2 is evaluated on the whole vector, per multipole, per multipole and k half, and as a
function of k_max. The mocks' own mean is used (the model mean is not needed; no Hartlap factor is
needed either, since C is the analytic matrix).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def corr(M):
    d = np.sqrt(np.diag(M))
    return M / np.outer(d, d)


def chi2_stats(V, C, idx=None):
    """(<chi2>/expected, deviation in sigma, chi2_i) for the elements idx."""
    if idx is not None:
        V, C = V[:, idx], C[np.ix_(idx, idx)]
    N, n = V.shape
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        e = np.linalg.eigvalsh(C)
        raise ValueError(f'covariance not positive definite (min/max eigenvalue {e[0] / e[-1]:.2g})') from None
    z = np.linalg.solve(L, (V - V.mean(0)).T)
    chi2 = np.sum(z ** 2, axis=0)
    e = n * (1 - 1 / N)
    return chi2.mean() / e, (chi2.mean() - e) / (e * np.sqrt(2 / (n * N))), chi2


def whitened_eigenvalues(V, C):
    L = np.linalg.cholesky(C)
    Ch = np.cov(V, rowvar=False)
    M = np.linalg.solve(L, np.linalg.solve(L, Ch).T).T
    return np.linalg.eigvalsh(0.5 * (M + M.T))


def mp_interval(n, N):
    q = n / (N - 1)
    return ((1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2) if q < 1 else None


def eigen_directions(C, C_mock, n=6):
    """Smallest eigenvalues of thecov's correlation matrix and the mock/thecov variance ratio along
    their eigenvectors (1 for an exact C; sampling scatter sqrt(2/N))."""
    D = np.sqrt(np.diag(C))
    e, v = np.linalg.eigh(C / np.outer(D, D))
    ratio = np.einsum('ij,ik,kj->j', v, C_mock / np.outer(D, D), v) / e
    return e[:n], ratio[:n], v[:, :n]


def validate(V, C, k, ells, kmax_list=None, n_lowk=3, n_boot=300, seed=0):
    V = np.asarray(V, float)
    N, n = V.shape
    nb = len(k)
    ells = tuple(ells)
    Ch = np.cov(V, rowvar=False)
    out = dict(N=N, n=n, nb=nb, ells=ells, k=np.asarray(k).tolist())
    blk = lambda i: np.arange(i * nb, (i + 1) * nb)

    # whole vector
    r, s, chi2 = chi2_stats(V, C)
    e = n * (1 - 1 / N)
    out.update(chi2_ratio=r, chi2_sigma=s, chi2_mean=chi2.mean(), chi2_exp=e, chi2_err=e * np.sqrt(2 / (n * N)),
               chi2_var=chi2.var(), chi2_var_exp=2 * e,
               ks_p=float(stats.kstest(chi2 * n / e, stats.chi2(df=n).cdf).pvalue))
    out['_chi2'] = chi2

    # eigenvalues
    ev = whitened_eigenvalues(V, C)
    mp = mp_interval(n, N)
    out.update(ev_min=ev.min(), ev_max=ev.max(), mp=mp,
               n_ev_above_mp=int(np.sum(ev > mp[1])) if mp else None,
               n_ev_below_mp=int(np.sum(ev < mp[0])) if mp else None)
    out['_ev'] = ev

    # per multipole: all k, low half, high half
    half = nb // 2
    out['k_split'] = float(k[half])
    out['per_block'] = {}
    for i, ell in enumerate(ells):
        b = blk(i)
        out['per_block'][ell] = [chi2_stats(V, C, idx)[:2] for idx in (b, b[:half], b[half:])]

    # chi^2 as a function of k_max (whole vector and per multipole)
    if kmax_list is None:
        kmax_list = [km for km in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
                     if k[0] < km <= k[-1] + 0.5 * (k[-1] - k[-2]) + 1e-9]   # up to the last bin's upper edge
    kk = np.asarray(k)
    out['kmax_list'] = list(kmax_list)
    out['chi2_kmax'] = {}
    for km in kmax_list:
        sel = kk <= km + 1e-9
        idx = np.concatenate([blk(i)[sel] for i in range(len(ells))])
        row = {'all': chi2_stats(V, C, idx)[:2]}
        for i, ell in enumerate(ells):
            row[ell] = chi2_stats(V, C, blk(i)[sel])[:2]
        out['chi2_kmax'][km] = row

    # diagonal
    ratio = np.sqrt(np.diag(Ch) / np.diag(C))
    band = 1 / np.sqrt(2 * (N - 1))
    out.update(sigma_ratio_rms_dev=float(np.std(ratio - 1)), sigma_ratio_band=band,
               var_ratio_mean=float(np.mean(ratio ** 2)))
    out['var_ratio_per_ell'] = {ell: float(np.mean(ratio[blk(i)] ** 2)) for i, ell in enumerate(ells)}
    # trend with k: least-squares slope of the variance ratio per unit k, per multipole
    out['var_ratio_slope'] = {ell: float(np.polyfit(kk, ratio[blk(i)] ** 2, 1)[0]) for i, ell in enumerate(ells)}
    out['_sigma_ratio'] = ratio

    # low k: mean variance ratio over the first n_lowk bins of every multipole, bootstrap over mocks
    rng = np.random.default_rng(seed)
    lowk = np.concatenate([blk(i)[:n_lowk] for i in range(len(ells))])
    rest = np.concatenate([blk(i)[n_lowk:] for i in range(len(ells))])
    dC = np.diag(C)
    mv = lambda idx, X: np.mean(np.var(X[:, idx], axis=0, ddof=1) / dC[idx])
    boot = np.array([[mv(lowk, V[b]), mv(rest, V[b])] for b in (rng.integers(0, N, N) for _ in range(n_boot))])
    out['lowk_var_ratio'] = (float(mv(lowk, V)), float(boot[:, 0].std()))
    out['rest_var_ratio'] = (float(mv(rest, V)), float(boot[:, 1].std()))

    # correlation residuals
    cm, ca = corr(Ch), corr(C)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (cm - ca) / ((1 - ca ** 2) / np.sqrt(N - 1))
    iu = np.triu_indices(n, 1)
    out.update(corr_resid_mean=float(z[iu].mean()), corr_resid_std=float(z[iu].std()),
               corr_resid_frac_gt3=float(np.mean(np.abs(z[iu]) > 3)), corr_resid_expected_gt3=float(2 * stats.norm.sf(3)))
    out['_z'] = z

    # near-null directions of thecov's correlation matrix
    e0, rat, _ = eigen_directions(C, Ch)
    out['null_eig'], out['null_ratio'] = e0.tolist(), rat.tolist()
    return out


def excess_structure(V, C, templates=None, n_top=10):
    """Structure of the excess Chat - C, whitened by C: its eigenvalues (sampling noise alone keeps
    them below ~(1 + sqrt(q))^2 - 1, q = n / (N - 1): 'noise_edge'), the fraction of its trace in the top 1, 3 and 10 modes, and fits of
    rank-1 terms s^2 t t^T for templates t (default: the mean vector, i.e. a common fluctuation of the
    amplitude of each mock, as from super-sample modes or mock-to-mock changes of the selection).
    Returns a dict; 'sigma' are the fitted fractional rms of each template (NaN if the fit is < 0),
    'var_ratio_after' the mean variance ratio once that term is added to C."""
    N, n = V.shape
    Ch = np.cov(V, rowvar=False)
    L = np.linalg.cholesky(C)
    Li = np.linalg.inv(L)
    E = Li @ (Ch - C) @ Li.T
    ev = np.linalg.eigvalsh(0.5 * (E + E.T))[::-1]
    tr = np.trace(E)
    q = n / (N - 1)
    out = dict(top_eigenvalues=ev[:n_top].tolist(), trace=float(tr), noise_edge=float((1 + np.sqrt(q)) ** 2 - 1),
               frac_top={m: float(ev[:m].sum() / tr) if tr != 0 else float('nan') for m in (1, 3, 10)})
    mean = V.mean(0)
    templates = templates or {'amplitude (mean vector)': mean}
    out['templates'] = {}
    for name, t in templates.items():
        u = Li @ t                                 # whitened template
        s2 = float(u @ E @ u / (u @ u) ** 2)        # least squares for s^2 in E ~ s^2 u u^T along u
        C2 = C + max(s2, 0) * np.outer(t, t)
        out['templates'][name] = dict(sigma=float(np.sqrt(s2)) if s2 > 0 else float('nan'),
                                      var_ratio_after=float(np.mean(np.diag(Ch) / np.diag(C2))),
                                      chi2_after=float(chi2_stats(V, C2)[0]))
    return out


def summary_row(out):
    """One line of numbers for the cross-tracer table."""
    row = dict(N=out['N'], n=out['n'], chi2=out['chi2_ratio'], chi2_sig=out['chi2_sigma'], ks_p=out['ks_p'],
               ev=(out['ev_min'], out['ev_max']), mp=out['mp'], var_ratio=out['var_ratio_mean'],
               z_std=out['corr_resid_std'], lowk=out['lowk_var_ratio'][0], null_ratio=out['null_ratio'][0])
    for km, r in out['chi2_kmax'].items():
        row[f'chi2_k{km}'] = r['all']
    return row


def print_report(out, label=''):
    f = lambda t: f'{t[0]:.3f} ({t[1]:+.1f})'
    print(f"== {label}  N_mock = {out['N']}, n = {out['n']}")
    print(f"  <chi2> = {out['chi2_mean']:.2f}, expected {out['chi2_exp']:.2f} +- {out['chi2_err']:.2f}  "
          f"(ratio {out['chi2_ratio']:.4f}, {out['chi2_sigma']:+.1f} sigma); var {out['chi2_var']:.0f} vs "
          f"{out['chi2_var_exp']:.0f}; KS p = {out['ks_p']:.3f}")
    mp = out['mp']
    print(f"  eigenvalues of C^-1/2 Chat C^-1/2 in [{out['ev_min']:.3f}, {out['ev_max']:.3f}]; Marchenko-Pastur "
          + (f"[{mp[0]:.3f}, {mp[1]:.3f}], {out['n_ev_below_mp']} below / {out['n_ev_above_mp']} above" if mp else 'n/a (N < n)'))
    print(f"  per multipole <chi2>/expected (sigma):   all k | k < {out['k_split']:.3f} | k > {out['k_split']:.3f}")
    for ell, rows in out['per_block'].items():
        print(f"    ell = {ell}:  " + ' | '.join(f(r) for r in rows))
    print('  chi2 vs k_max:  ' + '; '.join(f"{km}: {f(r['all'])}" for km, r in out['chi2_kmax'].items()))
    print(f"  sigma_mock/sigma_thecov: rms deviation {out['sigma_ratio_rms_dev']:.3f} (sampling {out['sigma_ratio_band']:.3f}); "
          f"mean variance ratio {out['var_ratio_mean']:.3f}; per ell "
          + ', '.join(f'{ell}: {v:.3f}' for ell, v in out['var_ratio_per_ell'].items()))
    print(f"  variance ratio, first bins {out['lowk_var_ratio'][0]:.3f} +- {out['lowk_var_ratio'][1]:.3f}; "
          f"rest {out['rest_var_ratio'][0]:.3f} +- {out['rest_var_ratio'][1]:.3f}")
    print(f"  correlation residuals z: mean {out['corr_resid_mean']:+.3f}, std {out['corr_resid_std']:.3f}, "
          f"|z|>3 {100 * out['corr_resid_frac_gt3']:.2f}% (Gaussian {100 * out['corr_resid_expected_gt3']:.2f}%)")
    print(f"  near-null directions: corr. eigenvalues {np.round(out['null_eig'][:4], 4)}, "
          f"mock/thecov variance {np.round(out['null_ratio'][:4], 2)} (sampling +-{np.sqrt(2 / out['N']):.2f})")


# --------------------------------------------------------------------------- figures
COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#8a5cd6')


def figures(out, V, C, k, ells, title=''):
    import matplotlib.pyplot as plt
    k = np.asarray(k)
    nb, N, n = len(k), out['N'], out['n']
    blk = lambda i: slice(i * nb, (i + 1) * nb)
    figs = {}

    # 1. mean multipoles with thecov errors
    mean, sig = V.mean(0), np.sqrt(np.diag(C))
    fig, ax = plt.subplots(1, len(ells), figsize=(4.2 * len(ells), 3.0))
    for i, ell in enumerate(ells):
        a = ax[i]
        a.fill_between(k, k * (mean[blk(i)] - sig[blk(i)]), k * (mean[blk(i)] + sig[blk(i)]), color=COLORS[i], alpha=0.2, lw=0,
                       label=r'$\pm\sigma_{\rm thecov}$')
        a.errorbar(k, k * mean[blk(i)], k * V.std(0)[blk(i)] / np.sqrt(N), color=COLORS[i], lw=1, label='mock mean')
        a.axhline(0, color='0.85', lw=0.8, zorder=0)
        a.set_xlabel(r'$k$ [$h$/Mpc]'); a.set_ylabel(rf'$k\,P_{ell}(k)$')
    ax[0].legend(fontsize=8); fig.suptitle(title, fontsize=10); fig.tight_layout()
    figs['multipoles'] = fig

    # 2. sigma ratio per element
    ratio, band = out['_sigma_ratio'], out['sigma_ratio_band']
    fig, ax = plt.subplots(1, len(ells), figsize=(4.2 * len(ells), 2.8), sharey=True)
    for i, ell in enumerate(ells):
        a = ax[i]
        a.axhspan(1 - band, 1 + band, color='0.88', lw=0); a.axhspan(1 - 2 * band, 1 + 2 * band, color='0.94', lw=0, zorder=0)
        a.axhline(1, color='0.4', lw=0.6)
        a.plot(k, ratio[blk(i)], 'o-', color=COLORS[i], ms=2.5, lw=0.9)
        a.set_title(rf'$\ell={ell}$', fontsize=9); a.set_xlabel(r'$k$ [$h$/Mpc]')
    ax[0].set_ylabel(r'$\sigma_{\rm mock}/\sigma_{\rm thecov}$'); fig.suptitle(title, fontsize=10); fig.tight_layout()
    figs['sigma_ratio'] = fig

    # 3. chi2 distribution and eigenvalue density
    chi2, ev, mp = out['_chi2'], out['_ev'], out['mp']
    e = out['chi2_exp']
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
    ax[0].hist(chi2, bins=40, density=True, color=COLORS[0], alpha=0.55, lw=0)
    x = np.linspace(chi2.min(), chi2.max(), 300)
    ax[0].plot(x, stats.chi2.pdf(x * n / e, df=n) * n / e, color='k', lw=1.2)
    ax[0].axvline(chi2.mean(), color=COLORS[1], lw=1.2)
    ax[0].text(0.03, 0.95, rf"$\langle\chi^2\rangle={chi2.mean():.1f}$" + '\n' + rf"expected ${e:.1f}\pm{out['chi2_err']:.1f}$"
               + '\n' + f"KS p = {out['ks_p']:.2f}", transform=ax[0].transAxes, va='top', fontsize=8)
    ax[0].set_xlabel(r'$\chi^2_i$'); ax[0].set_ylabel('density')
    ax[1].hist(ev, bins=40, density=True, color=COLORS[2], alpha=0.55, lw=0)
    if mp:
        q = n / (N - 1)
        xe = np.linspace(*mp, 400)
        ax[1].plot(xe, np.sqrt(np.maximum((mp[1] - xe) * (xe - mp[0]), 0)) / (2 * np.pi * q * xe), color='k', lw=1.2)
        ax[1].text(0.97, 0.95, 'Marchenko-Pastur\n(exact C)', transform=ax[1].transAxes, ha='right', va='top', fontsize=8)
    ax[1].set_xlabel(r'eigenvalues of $C^{-1/2}\hat C C^{-1/2}$'); ax[1].set_ylabel('density')
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    figs['chi2_eigen'] = fig

    # 4. correlation matrices and residuals
    Ch = np.cov(V, rowvar=False)
    cm, ca = corr(Ch), corr(C)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    im = ax[0].imshow(np.tril(cm, -1) + np.triu(ca, 1) + np.eye(n), cmap='RdBu_r', vmin=-0.6, vmax=0.6, interpolation='nearest')
    ax[0].set_title('mocks (lower) / thecov (upper)', fontsize=9); fig.colorbar(im, ax=ax[0], shrink=0.8)
    im2 = ax[1].imshow(np.where(np.eye(n, dtype=bool), 0, out['_z']), cmap='RdBu_r', vmin=-4, vmax=4, interpolation='nearest')
    ax[1].set_title(r'$z_{ij}=(\hat\rho_{ij}-\rho_{ij})/\sigma_\rho$', fontsize=9); fig.colorbar(im2, ax=ax[1], shrink=0.8)
    for a in ax:
        a.set_xticks(np.arange(len(ells)) * nb + nb / 2 - 0.5); a.set_xticklabels([f'P{l}' for l in ells])
        a.set_yticks(np.arange(len(ells)) * nb + nb / 2 - 0.5); a.set_yticklabels([f'P{l}' for l in ells])
        for b in range(1, len(ells)):
            a.axhline(b * nb - 0.5, color='w', lw=0.6); a.axvline(b * nb - 0.5, color='w', lw=0.6)
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    figs['correlation'] = fig

    # 5. chi2 vs k_max, and the near-null directions
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
    kms = out['kmax_list']
    for j, key in enumerate(['all'] + list(ells)):
        r = np.array([out['chi2_kmax'][km][key][0] for km in kms])
        ax[0].plot(kms, r, 'o-', color='k' if key == 'all' else COLORS[j - 1], ms=3, lw=1,
                   label='all' if key == 'all' else rf'$\ell={key}$')
    ns = [sum(int(np.sum(k <= km + 1e-9)) for _ in ells) for km in kms]
    errs = [np.sqrt(2 / (m * N)) for m in ns]
    ax[0].fill_between(kms, 1 - np.array(errs), 1 + np.array(errs), color='0.88', lw=0, zorder=0)
    ax[0].axhline(1, color='0.4', lw=0.6)
    ax[0].set_xlabel(r'$k_{\max}$ [$h$/Mpc]'); ax[0].set_ylabel(r'$\langle\chi^2\rangle$ / expected'); ax[0].legend(fontsize=8)
    e0, rat = np.array(out['null_eig']), np.array(out['null_ratio'])
    ax[1].semilogx(e0, rat, 'o', color=COLORS[1])
    ax[1].axhspan(1 - np.sqrt(2 / N), 1 + np.sqrt(2 / N), color='0.88', lw=0); ax[1].axhline(1, color='0.4', lw=0.6)
    ax[1].set_xlabel('eigenvalue of thecov correlation matrix'); ax[1].set_ylabel('mock / thecov variance along it')
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    figs['kmax_null'] = fig
    return figs


def window_figure(windows_file, title='', triples=((0, 0, 0), (2, 0, 2), (0, 2, 2), (2, 2, 0), (4, 0, 4))):
    """Tripolar window functions Q(s) stored by thecov (windows_<name>.npz), normalised to Q_000 at the
    first separation bin."""
    import json
    import matplotlib.pyplot as plt
    with np.load(windows_file) as f:
        se = f['s_edges']
        meta = json.loads(str(f['meta'].item()))
        Q = {(tuple(tuple(k) for k in rec['key']), tuple(rec['trip'])): f[rec['q']] for rec in meta}
    sc = 0.5 * (se[1:] + se[:-1])
    keys = sorted({k for k, _ in Q})
    fig, ax = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.0), squeeze=False)
    for a, key in zip(ax[0], keys):
        Q0 = Q.get((key, (0, 0, 0)))
        if Q0 is None:
            continue
        for j, trip in enumerate(triples):
            if (key, trip) in Q:
                a.plot(sc, Q[(key, trip)] / Q0[0], color=COLORS[j % 4], ls='-' if j < 4 else '--', lw=1,
                       label='Q' + ''.join(map(str, trip)))
        a.axhline(0, color='0.85', lw=0.8, zorder=0)
        a.set_xscale('log'); a.set_xlabel(r'$s$ [Mpc/$h$]'); a.set_ylabel(r'$Q(s)/Q_{000}(s_1)$')
        a.set_title(' x '.join(str(k[0]) for k in key), fontsize=8); a.legend(fontsize=7)
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    return fig
