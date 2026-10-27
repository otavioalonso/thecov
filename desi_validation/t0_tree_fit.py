"""Amplitude of the tree-level T0 in the holi mocks (HANDOFF section 12).

The template is the tree-level connected trispectrum term, snake + star at b3 = 0 (`C_T0_snake + C_T0_star` of the
ssc_check output). Not star_b3 (= d C_star / d b3, a template for b3), and not the response-approach pieces
(`C_T0_completion`, `C_T0_response`: an alternative model of the same T0). The base model is the production Gaussian
term (kernel smoothing, RSD) + the discreteness terms measured from the thinning closed form on one catalogue for the
P0-P0 block (|i-j| <= 1 replaced by continuity from |i-j| = 2, 3); the other blocks use the tree discreteness terms of
ssc_check scaled by the measured/tree ratio of the P0-P0 off-diagonals (a stopgap; with P4 it makes the base model
non-PD). Data-vector blocks are selected by position (ell // 2), k <= k_max.

For each multipole set: the Wishart likelihood scan of G + P/n^2 + B/n + a T0, its minimum and the curvature error, the
smallest G-whitened eigenvalue (PD check) at a = 0, best, 1; a two-amplitude fit (a_hard for both bins >= k_split,
a_sq for the rest); mock vs model correlations of the squeezed (k 0.02-0.05 x 0.2-0.3) and hard (both 0.15-0.3,
|i-j| >= 4) couplings.

    python -m desi_validation.t0_tree_fit --tracers LRG1 QSO --region NGC --out <dir> [--plot]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize

VALIDATION = '/global/cfs/cdirs/desicollab/users/oalves/thecov_validation'
HOME = os.path.expanduser('~/thecov')
SETS = ((0,), (2,), (0, 2))


# ----------------------------------------------------------------------------- inputs
def cont_diag(M):
    """replace |i-j| <= 1 of a measured term by continuity from |i-j| = 2, 3 (q ~ 0 ridge removed)"""
    M = M.copy(); n = len(M)
    for i in range(n):
        v = np.mean([M[i, j] for j in (i - 3, i - 2, i + 2, i + 3) if 0 <= j < n])
        for j in (i - 1, i, i + 1):
            if 0 <= j < n:
                M[i, j] = v
    return 0.5 * (M + M.T)


def ratio_off(meas, tree, k, kmin=0.1, offs=range(3, 9)):
    """measured / tree ratio of the off-diagonals |i-j| in offs at k > kmin (P0-P0 block)"""
    n = len(k); r = np.flatnonzero(k > kmin)
    pairs = [(i, i + o) for i in r for o in offs if i + o < n]
    return np.mean([meas[p] for p in pairs]) / np.mean([tree[p] for p in pairs])


def load(tracer, region, ssc, thinning, mock_vectors, excluded):
    z = np.load(ssc.format(tracer=tracer))
    F = lambda key: z[f'{region}/{key}'].astype(float)
    th = np.load(os.path.join(thinning.format(tracer=tracer, region=region), 'thinning_closed_form.npz'))
    k, Pn = th['k'], th['Pn2']
    Bn = th['lin'] - th['g1'] - 2 * Pn                                        # B/n with the corrected bookkeeping
    nb = len(k)
    TB, TP = F('C_disc_B'), F('C_disc_P')
    rB, rP = ratio_off(Bn, TB[:nb, :nb], k), ratio_off(Pn, TP[:nb, :nb], k)
    B, P = rB * TB, rP * TP
    B[:nb, :nb], P[:nb, :nb] = cont_diag(Bn), cont_diag(Pn)
    mv = np.load(mock_vectors)
    V = mv[f'{tracer}/{region}/vectors'].astype(float); ids = mv[f'{tracer}/{region}/mocks']
    if tracer.startswith('LRG'):
        V = V[~np.isin(ids, np.load(excluded)['excluded'])]
    assert np.allclose(mv[f'{tracer}/{region}/k'], k, rtol=1e-3)
    return dict(k=k, nb=nb, G=F('C'), P=P, B=B, rB=rB, rP=rP, V=V,
                T0=F('C_T0_snake') + F('C_T0_star'),
                pieces={p: F(f'C_T0_{p}') for p in ('snake', 'star', 'star_b3', 'completion', 'response')})


# ----------------------------------------------------------------------------- statistics
def select(d, ells, kmax):
    return np.concatenate([l // 2 * d['nb'] + np.flatnonzero(d['k'] <= kmax + 1e-9) for l in ells])


def wishart_D(C, S, n):
    """[tr(C^-1 S) - ln det(C^-1 S) - n] / n; (N_mock - 1) n D = -2 ln L_Wishart + const (inf if C is not PD)"""
    M = np.linalg.solve(C, S)
    sign, logdet = np.linalg.slogdet(M)
    return np.inf if sign <= 0 else (np.trace(M) - logdet - n) / n


def min_eig_vs_G(G, C):
    return float(sla.eigh(C, G, eigvals_only=True)[0])


def parts(d, ells, kmax):
    idx = select(d, ells, kmax); sub = lambda M: M[np.ix_(idx, idx)]
    x = d['V'][:, idx] - d['V'][:, idx].mean(0)
    return idx, sub, x, np.cov(x.T)


def amplitude_scan(d, ells, kmax, amps):
    idx, sub, x, S = parts(d, ells, kmax)
    n, N = len(idx), len(x)
    G = sub(d['G']); C0 = G + sub(d['P']) + sub(d['B']); T = sub(d['T0'])
    D = np.array([wishart_D(C0 + a * T, S, n) for a in amps])
    i0 = int(np.nanargmin(D)); best = float(amps[i0])
    dchi = (N - 1) * n * (D - D[i0])
    ok = np.isfinite(dchi) & (dchi < 9)
    sigma = np.nan
    if ok.sum() >= 3:
        c2 = np.polyfit(amps[ok] - best, dchi[ok], 2)
        sigma = float(1 / np.sqrt(c2[0])) if c2[0] > 0 else np.nan
    at = lambda a: float(np.interp(a, amps, dchi))
    return dict(amps=amps, dchi=dchi, best=best, sigma=sigma, dchi_a0=at(0.0), dchi_a1=at(1.0),
                min_eig=[min_eig_vs_G(G, C0 + a * T) for a in (0.0, best, 1.0)], n=n, N=N)


def two_amplitudes(d, ells, kmax, k_split):
    """a_hard (both bins >= k_split) and a_sq (the rest) of the T0 template, with Fisher errors"""
    idx, sub, x, S = parts(d, ells, kmax)
    n, N = len(idx), len(x); kk = d['k'][idx % d['nb']]
    hard = (kk[:, None] >= k_split) & (kk[None, :] >= k_split)
    T = sub(d['T0']); TH = np.where(hard, T, 0.0); TS = T - TH
    C0 = sub(d['G']) + sub(d['P']) + sub(d['B'])
    f = lambda p: (N - 1) * n * wishart_D(C0 + p[0] * TH + p[1] * TS, S, n)
    p0 = minimize(f, [0.5, 0.5], method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-3)).x
    h, H = 0.02, np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
            H[i, j] = (f(p0 + ei + ej) - f(p0 + ei - ej) - f(p0 - ei + ej) + f(p0 - ei - ej)) / (4 * h * h)
    cov = np.linalg.inv(H / 2)
    return dict(a_hard=float(p0[0]), a_sq=float(p0[1]), err=np.sqrt(np.diag(cov)).tolist(),
                corr=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])), dchi_both1=float(f([1.0, 1.0]) - f(p0)))


def couplings(d, ells, kmax, a_best):
    """mean correlation, squeezed (k 0.02-0.05 x 0.2-0.3) and hard (both 0.15-0.3, |i-j| >= 4): mocks and models"""
    idx, sub, x, S = parts(d, ells, kmax)
    kk = d['k'][idx % d['nb']]
    C0 = sub(d['G']) + sub(d['P']) + sub(d['B']); T = sub(d['T0'])
    corr = lambda C: C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    lo = np.flatnonzero((kk >= 0.02) & (kk < 0.05)); hi = np.flatnonzero((kk >= 0.2) & (kk < 0.3))
    hh = np.flatnonzero((kk >= 0.15) & (kk < 0.3))
    stats = {'squeezed': lambda C: float(np.mean(corr(C)[np.ix_(lo, hi)])),
             'hard': lambda C: float(np.mean([corr(C)[i, j] for i in hh for j in hh if abs(i - j) >= 4]))}
    out = {}
    for name, fn in stats.items():
        jack = [fn(np.cov(np.delete(x, p, 0).T)) for p in np.array_split(np.arange(len(x)), 8)]
        out[name] = dict(mocks=fn(S), mocks_err=float(np.std(jack, ddof=1) * 7 / np.sqrt(8)), a0=fn(C0),
                         best=fn(C0 + a_best * T), a1=fn(C0 + T))
    return out


def ng_direction(d, ells, kmax, a_best):
    """mock/model variance along the top generalized eigenvector of (P/n^2 + B/n) vs G, and the template projections"""
    idx, sub, x, S = parts(d, ells, kmax)
    G = sub(d['G']); NG = sub(d['P']) + sub(d['B'])
    v = sla.eigh(NG, G)[1][:, -1]
    ratio = lambda C: float(np.mean((x @ v) ** 2) / (v @ C @ v))
    return dict(base=ratio(G + NG), best=ratio(G + NG + a_best * sub(d['T0'])), a1=ratio(G + NG + sub(d['T0'])),
                missing=float((v @ S @ v - v @ (G + NG) @ v) / (v @ G @ v)),
                projections={p: float((v @ sub(M) @ v) / (v @ G @ v)) for p, M in d['pieces'].items()})


def block_amplitudes(d, T, rebin=5, ell_pairs=((0, 0), (2, 2), (0, 2))):
    """Amplitude of a T0 template T (full 168x168) per k-block, after rebinning to rebin*dk.

    Rebinning to dk = 0.025 brings the window-induced neighbour correlation of the Gaussian term down
    to ~0.35 and makes the Wishart element errors meaningful, so each block is fitted by weighted least
    squares in correlation units (a = sum w R T / sum w T^2, w = 1/err^2, err from Wishart). The base
    model is G + P/n^2 + B/n (measured discreteness). Blocks: diagonal, adjacent, low-low (<0.1),
    squeezed (<0.1 x >=0.1), hard (>=0.1 and >=0.15, off-diagonal). Returns a list of dict rows.
    """
    nb = len(d['k']); m = nb // rebin
    Rb = np.zeros((m, nb))
    for i in range(m):
        Rb[i, i * rebin:(i + 1) * rebin] = 1 / rebin
    kb = Rb @ d['k']; C = d['G'] + d['P'] + d['B']; rows = []
    for l1, l2 in ell_pairs:
        s1 = slice(l1 // 2 * nb, l1 // 2 * nb + nb); s2 = slice(l2 // 2 * nb, l2 // 2 * nb + nb)
        V1, V2 = d['V'][:, s1] @ Rb.T, d['V'][:, s2] @ Rb.T; N = len(V1)
        S12 = np.cov(V1.T, V2.T)[:m, m:]
        d1 = np.sqrt(np.diag(Rb @ C[s1, s1] @ Rb.T)); d2 = np.sqrt(np.diag(Rb @ C[s2, s2] @ Rb.T)); nrm = np.outer(d1, d2)
        Tc = Rb @ T[s1, s2] @ Rb.T / nrm; Rc = (S12 - Rb @ C[s1, s2] @ Rb.T) / nrm; Sc = S12 / nrm
        S11 = np.cov(V1.T) / np.outer(d1, d1); S22 = np.cov(V2.T) / np.outer(d2, d2)
        err = np.sqrt((Sc ** 2 + np.outer(np.diag(S11), np.diag(S22))) / (N - 1))
        I, J = np.meshgrid(range(m), range(m), indexing='ij'); K1, K2 = kb[I], kb[J]
        blocks = {'diag': I == J, 'adjacent': np.abs(I - J) == 1, 'low-low <0.1 off': (K1 < 0.1) & (K2 < 0.1) & (I != J),
                  'squeezed <0.1 x >=0.1': ((K1 < 0.1) & (K2 >= 0.1)) | ((K2 < 0.1) & (K1 >= 0.1)),
                  'hard >=0.1 off': (K1 >= 0.1) & (K2 >= 0.1) & (I != J),
                  'hard >=0.15 off': (K1 >= 0.15) & (K2 >= 0.15) & (I != J)}
        for nm, msk in blocks.items():
            if l1 == l2:
                msk = msk & (J >= I)
            w = (Tc / err ** 2)[msk]
            rows.append(dict(block_ells=f'P{l1}xP{l2}', block=nm, amplitude=np.sum(w * Rc[msk]) / np.sum(w * Tc[msk]),
                             err=1 / np.sqrt(np.sum((Tc ** 2 / err ** 2)[msk])), mean_T0_corr=Tc[msk].mean(),
                             mean_mock_excess_corr=Rc[msk].mean(), n_elements=int(msk.sum())))
    return rows


# ----------------------------------------------------------------------------- output
def plot(results, fn):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    col = {'(0,)': 'C0', '(2,)': 'C2', '(0, 2)': 'C4'}
    for (tr, key), r in results.items():
        if not tr.startswith('LRG'):
            continue
        s = r['scan']
        axs[0].plot(s['amps'], s['dchi'], color=col[key], label=f"{tr} {key}: a = {s['best']:.2f} +- {s['sigma']:.2f}")
        for ks, m in (('0.07', 'o'), ('0.1', 's')):
            t = r['two'][ks]
            axs[2].errorbar(t['a_sq'], t['a_hard'], xerr=t['err'][1], yerr=t['err'][0], fmt=m, color=col[key],
                            mfc=col[key] if m == 'o' else 'none')
        for j, (nm, c) in enumerate(r['couplings'].items()):
            y = j + 0.3 * list(col).index(key)
            axs[1].errorbar(c['mocks'], y, xerr=c['mocks_err'], fmt='kD', ms=4)
            axs[1].plot([c['a0'], c['best'], c['a1']], [y - 0.08] * 3, 'o', color=col[key], ms=3)
            axs[1].plot(c['a1'], y - 0.08, 'x', color='C3')
    axs[0].axvline(1, color='k', ls=':'); axs[0].set(xlabel='amplitude of tree T0', ylabel='Delta(-2 ln L)', ylim=(0, 600))
    axs[0].legend(fontsize=7)
    axs[1].set_yticks([0, 0.3, 1, 1.3]); axs[1].set_yticklabels(['P0 squeezed', 'P2 squeezed', 'P0 hard', 'P2 hard'])
    axs[1].set(xlabel='mean correlation (black: mocks; dots: a = 0, best, 1; x: a = 1)')
    axs[2].plot(1, 1, 'r*'); axs[2].set(xlabel='a_sq', ylabel='a_hard (o: k_split 0.07, s: 0.10)')
    fig.tight_layout(); fig.savefig(fn, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tracers', nargs='+', default=['LRG1', 'QSO'])
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--kmax', type=float, default=0.3)
    ap.add_argument('--ssc', default=f'{VALIDATION}/ssc_holi-kcore2-{{tracer}}.npz')
    ap.add_argument('--thinning', default=f'{HOME}/thinning/{{tracer}}_{{region}}_mock173')
    ap.add_argument('--mock-vectors', default=f'{HOME}/responses/mock_vectors.npz')
    ap.add_argument('--excluded', default=f'{HOME}/responses/survey_mean_regression_LRG1.npz')
    ap.add_argument('--out', default='.')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--blocks', action='store_true', help='per-k-block amplitudes after rebinning to 5 dk (section 13)')
    ap.add_argument('--t0-variants', default=None,
                    help='npz with <variant>/snake, <variant>/star (t0_run.py output) to use as extra T0 templates in --blocks')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    amps = np.linspace(-0.5, 2.0, 251)
    results, rows = {}, []
    for tr in args.tracers:
        d = load(tr, args.region, args.ssc, args.thinning, args.mock_vectors, args.excluded)
        print(f'{tr} {args.region}: {len(d["V"])} mocks; disc scaling measured/tree B {d["rB"]:.2f}, P {d["rP"]:.2f}')
        if args.blocks:
            templates = {'prod': d['T0']}
            if args.t0_variants:
                tv = np.load(args.t0_variants)
                templates.update({v: tv[f'{v}/snake'] + tv[f'{v}/star'] for v in {k.split('/')[0] for k in tv if '/' in k}})
            brows = [dict(tracer=tr, variant=v, **r) for v, T in templates.items() for r in block_amplitudes(d, T)]
            with open(os.path.join(args.out, f't0_blocks_{tr}_{args.region}.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(brows[0])); w.writeheader(); w.writerows(brows)
            for r in brows:
                if r['block'] in ('diag', 'squeezed <0.1 x >=0.1', 'hard >=0.15 off'):
                    print(f'  blocks {r["variant"]:7s} {r["block_ells"]} {r["block"]:22s} a = {r["amplitude"]:+.2f} +- {r["err"]:.2f}'
                          f'  (T0 corr {r["mean_T0_corr"]:.3f}, mock excess {r["mean_mock_excess_corr"]:.3f})')
        for ells in SETS:
            s = amplitude_scan(d, ells, args.kmax, amps)
            two = {str(ks): two_amplitudes(d, ells, args.kmax, ks) for ks in (0.07, 0.1)} if tr.startswith('LRG') else {}
            cp = couplings(d, ells, args.kmax, s['best']) if len(ells) == 1 else {}
            ng = ng_direction(d, ells, args.kmax, s['best'])
            results[tr, str(ells)] = dict(scan=s, two=two, couplings=cp, ng=ng)
            print(f'  ells {str(ells):7s} a = {s["best"]:+.2f} +- {s["sigma"]:.2f}; Delta(-2lnL) a=0 {s["dchi_a0"]:7.1f}, '
                  f'a=1 {s["dchi_a1"]:7.1f}; min eig vs G (0, best, 1) {np.round(s["min_eig"], 3)}; NG direction '
                  f'{ng["base"]:.2f} -> {ng["best"]:.2f} -> {ng["a1"]:.2f}'
                  + ''.join(f'; k_split {ks}: a_hard {t["a_hard"]:.2f} +- {t["err"][0]:.2f}, a_sq {t["a_sq"]:.2f} +- '
                            f'{t["err"][1]:.2f}' for ks, t in two.items()))
            rows.append(dict(tracer=tr, region=args.region, ells='+'.join(map(str, ells)), a=s['best'], sigma=s['sigma'],
                             dchi_a0=s['dchi_a0'], dchi_a1=s['dchi_a1'], min_eig_a0=s['min_eig'][0],
                             min_eig_best=s['min_eig'][1], min_eig_a1=s['min_eig'][2],
                             **{f'{nm}_ksplit{ks}': t[nm] for ks, t in two.items() for nm in ('a_hard', 'a_sq')}))
    with open(os.path.join(args.out, f't0_tree_fit_{args.region}.csv'), 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump({f'{tr} {e}': {k: v for k, v in r.items() if k != 'scan'} | {'scan': {k: v for k, v in r['scan'].items()
               if k not in ('amps', 'dchi')}} for (tr, e), r in results.items()},
              open(os.path.join(args.out, f't0_tree_fit_{args.region}.json'), 'w'), indent=1)
    if args.plot:
        plot(results, os.path.join(args.out, f't0_tree_fit_{args.region}.png'))


if __name__ == '__main__':
    main()
