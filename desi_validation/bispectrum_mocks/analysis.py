"""Mock-based covariance of the diagonal Sugiyama bispectrum multipoles B000(k, k), B202(k, k) across mock suites.

usage: python analysis.py <dir with bispec_<suite>.npz>  -> results.json, results_arrays.npz
"""
import sys, json, os
import numpy as np
from scipy.stats import kurtosis, skew, chi2 as chi2dist

D = sys.argv[1] if len(sys.argv) > 1 else '.'
SUITES = {  # key: (file stem, label, role)
    'holi': ('holi-v3-altmtl', 'holi v3 altmtl', 'reference'),
    'glam': ('glam-uchuu-v2-altmtl', 'GLAM-Uchuu v2 altmtl', 'large'),
    'abc': ('abacus-hf-dr2-v2-rotated-complete-claude', 'Abacus-HF v2 complete (rotated)', 'abacus'),
    'aba': ('abacus-hf-dr2-v2-altmtl', 'Abacus-HF v2 altmtl', 'abacus'),
    'a2c': ('abacus-2ndgen-dr2-complete', 'Abacus 2nd-gen complete', 'abacus'),
    'a2a': ('abacus-2ndgen-dr2-altmtl', 'Abacus 2nd-gen altmtl', 'abacus'),
}
BINS = {  # label: (tracer, z, alias for the 2nd-gen complete ELG naming)
    'LRG1': ('LRG', 'z0.4-0.6'), 'LRG2': ('LRG', 'z0.6-0.8'), 'LRG3': ('LRG', 'z0.8-1.1'),
    'ELG1': ('ELG_LOPnotqso', 'z0.8-1.1'), 'ELG2': ('ELG_LOPnotqso', 'z1.1-1.6'), 'QSO': ('QSO', 'z0.8-2.1'),
}
REGIONS = ('NGC', 'SGC', 'GCcomb')
KMIN, KMAX = 0.02, 0.205
LRG1_EXCLUDE = {122, 131, 166, 218, 470, 573, 664, 667, 746, 793, 880}   # holi LRG1, defective n(z)
BANDS = [(0.02, 0.08), (0.08, 0.14), (0.14, 0.205)]


def load():
    data = {}
    for s, (stem, label, role) in SUITES.items():
        f = os.path.join(D, f'bispec_{stem}.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        groups = {}
        for b, (tr, zr) in BINS.items():
            for r in REGIONS:
                key = f'{tr}_{zr}_{r}'
                alt = key.replace('ELG_LOPnotqso', 'ELG_LOP')
                use = key if f'{key}__B000' in z.files else (alt if f'{alt}__B000' in z.files else None)
                if use is None:
                    continue
                g = {k.split('__')[1]: z[k] for k in z.files if k.startswith(use + '__')}
                ids = np.array([int(str(i)[4:]) if str(i)[4:].isdigit() else -1 for i in g['ids']])
                keep = ids >= 0
                if s == 'holi' and b == 'LRG1':
                    keep &= ~np.isin(ids, list(LRG1_EXCLUDE))
                o = np.argsort(ids[keep])
                for k_ in ('B000', 'B202', 'sn000', 'sn202', 'norm000', 'norm202', 'P0', 'P2', 'P4', 'normP', 'snP', 'wsum_data', 'wsum_randoms', 'ids'):
                    if k_ in g:
                        g[k_] = g[k_][keep][o]
                g['num'] = ids[keep][o]
                g['tracer_name'] = use.rsplit('_', 1)[0]
                groups[(b, r)] = g
        data[s] = groups
    return data


def vec(g, sel):
    return np.concatenate([g['B000'][:, sel], g['B202'][:, sel]], axis=1)


def corr(C):
    d = np.sqrt(np.diag(C)); return C / np.outer(d, d)


def corr_summary(R, n):
    """median rho(i,i+1), rho(i,i+2), mean rho over |i-j| >= 10 for each multipole block, and the B000-B202 same-k cross"""
    out = {}
    for name, sl in (('B000', slice(0, n)), ('B202', slice(n, 2 * n))):
        Rb = R[sl, sl]
        out[name] = dict(nn1=float(np.median(np.diag(Rb, 1))), nn2=float(np.median(np.diag(Rb, 2))),
                         floor=float(Rb[np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= 10].mean()))
    out['cross_same_k'] = float(np.median(np.diag(R[:n, n:])))
    return out


def jackknife(fn, X, nblocks=40):
    n = len(X); blocks = np.array_split(np.random.default_rng(0).permutation(n), min(nblocks, n))
    vals = np.array([fn(np.delete(X, b, axis=0)) for b in blocks])
    m = len(blocks)
    return np.sqrt((m - 1) / m * np.sum((vals - vals.mean(0)) ** 2, axis=0))


def main():
    data = load()
    k = data['holi'][('LRG1', 'NGC')]['k_b']
    sel = (k >= KMIN) & (k < KMAX)
    ks = k[sel]; n = sel.sum(); p = 2 * n
    res = dict(k=ks.tolist(), n_k=int(n), suites={s: dict(stem=v[0], label=v[1]) for s, v in SUITES.items() if s in data},
               groups={}, pairs={}, chi2={}, gccomb={}, outliers={}, hartlap={})
    arrays = {'k': ks}
    for s, groups in data.items():
        for (b, r), g in groups.items():
            X = vec(g, sel); N = len(X)
            mean, var = X.mean(0), X.var(0, ddof=1)
            C = np.cov(X, rowvar=False); R = corr(C)
            kurt = kurtosis(X, axis=0, bias=False); sk = skew(X, axis=0, bias=False)
            tag = f'{s}|{b}|{r}'
            arrays[f'{tag}|mean'] = mean; arrays[f'{tag}|var'] = var; arrays[f'{tag}|R'] = R; arrays[f'{tag}|kurt'] = kurt
            if 'P0' in g:
                kp = g['k_p'] if 'k_p' in g else None
                Ptot = g['P0'].mean(0) + np.mean(g['snP'] / g['normP'])
                arrays[f'{tag}|Ptot_at_kb'] = np.interp(ks, kp, Ptot) if kp is not None else np.full(n, np.nan)
            arrays[f'{tag}|sn000'] = (g['sn000'] / g['norm000']).mean(0)[sel]
            arrays[f'{tag}|norm'] = np.array([g['norm000'][:, 0].mean()])
            res['groups'][tag] = dict(N=N, tracer_name=str(g['tracer_name']), norm=float(g['norm000'][:, 0].mean()),
                                      corr=corr_summary(R, n), kurt_median=float(np.median(kurt)), kurt_max=float(np.max(kurt)),
                                      skew_median=float(np.median(sk)), rel_err_B000_at_0p1=float(np.sqrt(var[np.argmin(abs(ks - 0.1))]) / abs(mean[np.argmin(abs(ks - 0.1))])))
            if N > p + 2:
                res['hartlap'][tag] = (N - p - 2) / (N - 1)
    # variance ratios against holi (and GLAM against holi), per element, with errors
    for s in data:
        if s == 'holi':
            continue
        for (b, r) in data[s]:
            if (b, r) not in data['holi']:
                continue
            Xs, Xh = vec(data[s][(b, r)], sel), vec(data['holi'][(b, r)], sel)
            ratio = Xs.var(0, ddof=1) / Xh.var(0, ddof=1)
            ks_, kh_ = kurtosis(Xs, axis=0), kurtosis(Xh, axis=0)
            err = ratio * np.sqrt((2 + np.clip(ks_, -1.5, None)) / (len(Xs) - 1) + (2 + np.clip(kh_, -1.5, None)) / (len(Xh) - 1))
            tag = f'{s}/holi|{b}|{r}'
            arrays[f'{tag}|ratio'] = ratio; arrays[f'{tag}|err'] = err
            gs, gh = data[s][(b, r)], data['holi'][(b, r)]
            if 'P0' in gs and 'P0' in gh:
                Ps = np.interp(ks, gs['k_p'], gs['P0'].mean(0) + np.mean(gs['snP'] / gs['normP']))
                Ph = np.interp(ks, gh['k_p'], gh['P0'].mean(0) + np.mean(gh['snP'] / gh['normP']))
                scale = np.concatenate([(Ps / Ph) ** 3] * 2)
                arrays[f'{tag}|ratio_scaled'] = ratio / scale
            bands = {}
            for lo, hi in BANDS:
                m = np.concatenate([(ks >= lo) & (ks < hi)] * 2)
                w = 1 / err[m] ** 2
                for name, mm in (('B000', np.r_[np.ones(n, bool), np.zeros(n, bool)][m]), ('B202', np.r_[np.zeros(n, bool), np.ones(n, bool)][m])):
                    rr, ww = ratio[m][mm], w[mm]
                    bands[f'{name}_{lo:g}-{hi:g}'] = dict(mean=float(np.sum(rr * ww) / np.sum(ww)), err=float(1 / np.sqrt(np.sum(ww))),
                                                          scaled=float(np.median((arrays.get(f'{tag}|ratio_scaled', ratio * np.nan))[m][mm])))
            res['pairs'][tag] = bands
    # paired fibre-assignment comparison: altmtl vs complete of the same boxes
    for alt, comp in (('aba', 'abc'), ('a2a', 'a2c')):
        if alt not in data or comp not in data:
            continue
        for (b, r) in data[alt]:
            if (b, r) not in data[comp]:
                continue
            ga, gc = data[alt][(b, r)], data[comp][(b, r)]
            common = np.intersect1d(ga['num'], gc['num'])
            if len(common) < 10:
                continue
            Xa = vec(ga, sel)[np.isin(ga['num'], common)]; Xc = vec(gc, sel)[np.isin(gc['num'], common)]
            XX = np.concatenate([Xa, Xc], axis=1)
            fn = lambda Y: Y[:, :p].var(0, ddof=1) / Y[:, p:].var(0, ddof=1)
            ratio = fn(XX); err = jackknife(fn, XX, nblocks=len(common))
            dmean = (Xa.mean(0) - Xc.mean(0)) / np.sqrt(Xc.var(0, ddof=1))
            rho = np.array([np.corrcoef(Xa[:, i], Xc[:, i])[0, 1] for i in range(p)])
            tag = f'{alt}/{comp}|{b}|{r}'
            arrays[f'{tag}|ratio'] = ratio; arrays[f'{tag}|err'] = err; arrays[f'{tag}|dmean'] = dmean; arrays[f'{tag}|rho'] = rho
            res['pairs'][tag] = dict(n_pairs=int(len(common)), B000_ratio_median=float(np.median(ratio[:n])), B202_ratio_median=float(np.median(ratio[n:])),
                                     B000_dmean_median=float(np.median(dmean[:n])), B202_dmean_median=float(np.median(dmean[n:])),
                                     rho_median=float(np.median(rho)))
    # global consistency: mean chi2/p of suite-s realisations (about their own mean) under the inverse covariance of a large suite
    for ref in ('holi', 'glam'):
        for s in data:
            for (b, r) in data[s]:
                if (b, r) not in data[ref] or r == 'GCcomb':
                    continue
                Xr = vec(data[ref][(b, r)], sel); Nr = len(Xr)
                if Nr <= p + 2:
                    continue
                Cinv = np.linalg.inv(np.cov(Xr, rowvar=False)) * (Nr - p - 2) / (Nr - 1)
                Xs = vec(data[s][(b, r)], sel); Ns = len(Xs)
                if s == ref:   # half split: one half's chi2 under the other half's covariance
                    h = Nr // 2; A, B_ = Xr[:h], Xr[h:]
                    if h <= p + 2:
                        continue
                    Cinv_h = np.linalg.inv(np.cov(A, rowvar=False)) * (h - p - 2) / (h - 1)
                    Z = B_ - B_.mean(0); c2 = np.einsum('ij,jk,ik->i', Z, Cinv_h, Z) * len(B_) / (len(B_) - 1)
                else:
                    Z = Xs - Xs.mean(0); c2 = np.einsum('ij,jk,ik->i', Z, Cinv, Z) * Ns / (Ns - 1)
                # the trace ratio tr(C_ref^-1 C_s)/p and its scatter
                res['chi2'][f'{s}|{ref}|{b}|{r}'] = dict(mean_over_p=float(c2.mean() / p), err=float(c2.std() / np.sqrt(len(c2)) / p), N=int(len(c2)))
    # outliers: holi and GLAM realisations with an extreme chi2 under their own (leave-in) covariance
    for s in ('holi', 'glam'):
        for (b, r), g in data[s].items():
            if r == 'GCcomb':
                continue
            X = vec(g, sel); N = len(X)
            Cinv = np.linalg.inv(np.cov(X, rowvar=False)) * (N - p - 2) / (N - 1)
            Z = X - X.mean(0); c2 = np.einsum('ij,jk,ik->i', Z, Cinv, Z)
            pv = chi2dist.sf(c2, p)
            # common-mode amplitude: projection on the all-ones direction of the standardised residual
            Zs = Z / X.std(0); cm = Zs.mean(1)
            flagged = g['num'][pv < 1e-4]
            res['outliers'][f'{s}|{b}|{r}'] = dict(n_flagged=int(len(flagged)), flagged=flagged.tolist()[:20],
                                                   cm_sd=float(cm.std()), cm_max=float(np.abs(cm).max()), cm_argmax=int(g['num'][np.argmax(np.abs(cm))]))
            arrays[f'{s}|{b}|{r}|chi2'] = c2; arrays[f'{s}|{b}|{r}|cm'] = cm
    # rotated-complete independence: mean correlation of standardised residuals between mock i and i + 25 m
    if 'abc' in data:
        lag = {}
        for L in (25, 50, 75):
            vals = []
            for (b, r), g in data['abc'].items():
                if r == 'GCcomb':
                    continue
                X = vec(g, sel); Z = (X - X.mean(0)) / X.std(0); num = g['num']
                idx = {m: i for i, m in enumerate(num)}
                vals += [float(Z[idx[m]] @ Z[idx[m + L]] / p) for m in num if m + L in idx]
            lag[L] = dict(mean=float(np.mean(vals)), err=float(np.std(vals) / np.sqrt(len(vals))))
        vals = []
        for (b, r), g in data['abc'].items():
            if r == 'GCcomb':
                continue
            X = vec(g, sel); Z = (X - X.mean(0)) / X.std(0); num = g['num']
            idx = {m: i for i, m in enumerate(num)}
            vals += [float(Z[idx[m]] @ Z[idx[m2]] / p) for m in num for m2 in num if m2 > m and (m2 - m) % 25 != 0]
        lag['other'] = dict(mean=float(np.mean(vals)), err=float(np.std(vals) / np.sqrt(len(vals))))
        res['rotation_lags'] = {str(k_): v for k_, v in lag.items()}
    # GCcomb definition
    for s in data:
        g = data[s]
        for b in BINS:
            if all((b, r) in g for r in REGIONS):
                gn, gs_, gc = g[(b, 'NGC')], g[(b, 'SGC')], g[(b, 'GCcomb')]
                if len(gn['num']) != len(gc['num']) or len(gs_['num']) != len(gc['num']) or not (np.all(gn['num'] == gc['num']) and np.all(gs_['num'] == gc['num'])):
                    continue
                avg = (gn['norm000'] * gn['B000'] + gs_['norm000'] * gs_['B000']) / (gn['norm000'] + gs_['norm000'])
                res['gccomb'][f'{s}|{b}'] = float(np.nanmax(np.abs(gc['B000'][:, sel] - avg[:, sel]) / gc['B000'][:, sel].std(0)))
    json.dump(res, open('results.json', 'w'), indent=1)
    np.savez_compressed('results_arrays.npz', **arrays)
    print('done', len(res['groups']), 'groups')


if __name__ == '__main__':
    main()
