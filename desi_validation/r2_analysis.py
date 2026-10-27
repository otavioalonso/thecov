"""Second-order response R_2 from single catalogues (response_ibisp.py --soft-band outputs) against the ensemble (HANDOFF 14).

    python -m desi_validation.r2_analysis responses/r2/response_LRG1_NGC_mocks17*_r2_kL0.02.npz --bundle t0_inputs.npz

Per catalogue, band and k_q: regress (Q_0 - QS)(k) = a P_0(k) + b over k in [0.12, 0.3]; R_2 = 2 a / <delta_s^2 g>_m2 x
(1 + SN/P_s)^2. Ensemble reference from the bundle (mock vectors, Gaussian + measured discreteness + SSC base, tree T0):
R_2 = 2 Cov_NG(P_s, P_h) / (P_h V_band Var(P_s)) (1 + SN/P_s)^2, SN/P_s from the estimator's sn / Ps_band_raw.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

HARD = ((0.1, 0.2), (0.2, 0.3))


def single(fn, kmin=0.12, kmax=0.3):
    """Slope-fit R_2 per (band, k_q) for l = 0 and l = 2; the shared-galaxy terms Qh (if present) are removed."""
    R = np.load(fn); k = R['k']; rows = []
    mock = int(re.search(r'mocks(\d+)', fn).group(1)) if re.search(r'mocks(\d+)', fn) else -1
    for run in range(R['P0'].shape[0]):
        d2g, sn, ps = R['d2g_m2'][run], float(R['sn'][run]), R['Ps_band_raw'][run]
        sel = (k >= kmin) & (k <= kmax)
        for ell in (0, 2):
            P = R[f'P{ell}'][run]; Q = R[f'Q{ell}'][run].copy()
            QS = R['QS'][run].copy() if ell == 0 else np.zeros_like(R['QS'][run])
            if f'Qh{ell}' in R.files:                                   # shared-galaxy (B/n-type) terms, partner side
                Q -= R[f'Qh{ell}'][run]; QS -= R['QhS'][run] if ell == 0 else 0.0
            if f'Qc{ell}' in R.files:                                   # long-mode self/partner term (times -beta in g)
                bq = R['beta'][run][..., None]; Q += bq * R[f'Qc{ell}'][run]; QS += R['beta'][run] * R['QcS'][run] if ell == 0 else 0.0
            A = np.vstack([P[sel], np.ones(sel.sum())]).T
            for iB, (k1, k2) in enumerate(R['soft_band'][run]):
                for iQ, kq in enumerate(R['k_q'][run]):
                    y = Q[iB, iQ] - QS[iB, iQ]
                    (a, b), *_ = np.linalg.lstsq(A, y[sel], rcond=None)
                    err = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * (y[sel] - A @ (a, b)).var())
                    snf = (1 + sn / ps[iB, iQ]) ** 2
                    hterm = 2 * np.mean(R[f'Qh{ell}'][run][iB, iQ][sel] / P[sel]) / d2g[iB, iQ] * snf if f'Qh{ell}' in R.files else np.nan
                    rows.append(dict(mock=mock, ell=ell, band=f'{k1:g}-{k2:g}', k1=k1, k2=k2, k_q=kq, SN_over_Ps=sn / ps[iB, iQ],
                                     R2=2 * a / d2g[iB, iQ] * snf, R2_fit_err=2 * err[0] / d2g[iB, iQ] * snf, const_over_sn=b / sn,
                                     shared_galaxy_term=hterm))
    return pd.DataFrame(rows)


def ensemble(bundle, tracer, region, bands, snr, t0_variants=None, ell_h=0):
    """R_2 implied by the 848-mock covariance and by the tree T0 block, per (soft P0 band, hard P_ell_h range).

    R_2 = 2 Cov_NG(P0_s, Pl_h) / (Pl_h V_band Var(P0_s)) (1 + SN/P_s)^2, Cov_NG = mocks - (G + discreteness + SSC); the
    P0-P0 discreteness is the thinning closed form, the P0-P2 cross block the scaled tree terms (stopgap, section 12).
    """
    from desi_validation.t0_tree_fit import cont_diag, ratio_off
    Z = np.load(bundle); F = lambda key: Z[f'F/{tracer}/{region}/{key}'].astype(float)
    k = F('k'); nb = len(k); Pm = F('mock_mean'); h0 = ell_h // 2 * nb
    V = Z[f'MV/{tracer}/{region}/vectors'].astype(float); ids = Z[f'MV/{tracer}/{region}/mocks']
    if tracer.startswith('LRG'):
        V = V[~np.isin(ids, Z['excluded_LRG1'])]
    S = np.cov(V.T); N = len(V)
    th = {key.split('/')[-1]: Z[key].astype(float) for key in Z.files if key.startswith(f'thin/{tracer}/{region}')}
    Pn = th['Pn2']; Bn = th['lin'] - th['g1'] - 2 * Pn
    TB, TP = F('C_disc_B'), F('C_disc_P')
    B, P_ = ratio_off(Bn, TB[:nb, :nb], k) * TB, ratio_off(Pn, TP[:nb, :nb], k) * TP
    B[:nb, :nb], P_[:nb, :nb] = cont_diag(Bn), cont_diag(Pn)
    G = F('C'); base = G + P_ + B + F('C_ssc')
    trees = {'tree_prod': F('C_T0_snake') + F('C_T0_star')}
    if t0_variants:
        TV = np.load(t0_variants)
        trees.update({f'tree_{v}': TV[f'{v}/snake'] + TV[f'{v}/star'] for v in {key.split('/')[0] for key in TV.files if '/' in key}})
    sel = lambda b, off=0: off + np.flatnonzero((k >= b[0]) & (k < b[1]))
    bc = lambda M, i, j: M[np.ix_(i, j)].mean()
    rows = []
    for bs in bands:
        i = sel(bs); Vband = (bs[1] ** 3 - bs[0] ** 3) / (6 * np.pi ** 2); f_sn = (1 + snr[bs]) ** 2
        for bh in HARD:
            j = sel(bh, h0); Ph = Pm[j].mean(); x = V - V.mean(0)
            err_cov = (x[:, i].mean(1) * x[:, j].mean(1)).std() / np.sqrt(N)
            r = dict(ell_h=ell_h, band=f'{bs[0]:g}-{bs[1]:g}', hard=f'{bh[0]:g}-{bh[1]:g}',
                     R2_mocks=2 * bc(S - base, i, j) / (Ph * Vband * bc(S, i, i)) * f_sn, err=2 * err_cov / (Ph * Vband * bc(S, i, i)) * f_sn)
            for nm, T in trees.items():
                r[f'R2_{nm}'] = 2 * bc(T, i, j) / (Ph * Vband * bc(G, i, i)) * f_sn
            rows.append(r)
    return pd.DataFrame(rows)


def tree_expectation(bundle, tracer, region, bands, k_qs, k_hs=(0.15, 0.25), N=40000, seed=3):
    """Tree-level expectation of the estimator, R_2^tree = sum T / (P0_h sum P_g(p) P_g(q-p)) over the estimator's own
    configurations (p in the band, q ~ W_q, k on the hard shell), with the snake terms whose internal momentum is the
    long mode q removed (the delta_q projection of the estimator kills them). k_q = 0 gives the collapsed limit."""
    import itertools
    from thecov.tree.trispectrum import trispectrum, galileon_bias, Z1, Z2, Z3, _dot
    from thecov.tree._common import LinearPower
    Z = np.load(bundle); b1, b2, bs2, f, _ = Z[f'F/{tracer}/{region}/params']
    Plin = LinearPower(Z[f'F/{tracer}/{region}/plin_k'], Z[f'F/{tracer}/{region}/plin_P']); bias = galileon_bias(b1, b2, bs2)
    rng = np.random.default_rng(seed)

    def randdir(n):
        u = rng.uniform(-1, 1, n); ph = rng.uniform(0, 2 * np.pi, n); s_ = np.sqrt(1 - u ** 2)
        return np.vstack([s_ * np.cos(ph), s_ * np.sin(ph), u])

    def pg(v):
        m = np.sqrt((v ** 2).sum(0)); return (b1 + f * (v[2] / m) ** 2) ** 2 * Plin(m)

    def tri_noIR(ks, qvec):
        mags = [np.sqrt(_dot(k, k)) for k in ks]; Pk = [Plin(m) for m in mags]; z1 = [Z1(k, bias, f) for k in ks]
        qm = np.sqrt(_dot(qvec, qvec)); T = 0.0
        for a, b in itertools.combinations(range(4), 2):
            c, d = (i for i in range(4) if i not in (a, b)); pre = 4 * z1[a] * z1[b] * Pk[a] * Pk[b]
            for cc, dd in ((c, d), (d, c)):
                s_ = ks[a] + ks[cc]; sm = np.sqrt(_dot(s_, s_))
                T = T + np.where(np.isclose(sm, qm, rtol=1e-6, atol=1e-12), 0.0,
                                 pre * Plin(sm) * Z2(-ks[a], s_, bias, f) * Z2(-ks[b], ks[b] + ks[dd], bias, f))
        for d in range(4):
            a, b, c = (i for i in range(4) if i != d)
            T = T + 6 * Z3(-ks[a], -ks[b], -ks[c], bias, f) * z1[a] * z1[b] * z1[c] * Pk[a] * Pk[b] * Pk[c]
        return T

    rows = []
    for band in bands:
        for kh in k_hs:
            P0h = np.mean(pg(kh * randdir(N))); row = dict(band=f'{band[0]:g}-{band[1]:g}', k_h=kh)
            for kq in (0,) + tuple(k_qs):
                k = kh * randdir(N); p = (rng.uniform(band[0] ** 3, band[1] ** 3, N)) ** (1 / 3) * randdir(N)
                q = np.zeros((3, N)) if kq == 0 else rng.normal(0, kq, (3, N)); p2 = q - p
                m2 = np.sqrt((p2 ** 2).sum(0)); keep = (m2 >= band[0]) & (m2 < band[1])
                k, p, q, p2 = k[:, keep], p[:, keep], q[:, keep], p2[:, keep]
                if kq == 0:
                    T = trispectrum([k, -k, p, -p], Plin, bias, f); Tt = T['snake'] + T['star']
                else:
                    Tt = tri_noIR([k, -k - q, p, p2], q)
                row['R2_tree_collapsed' if kq == 0 else f'R2_tree_kq{kq:g}'] = np.sum(Tt) / (P0h * np.sum(pg(p) * pg(p2)))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--bundle', default=None, help='t0_inputs bundle for the ensemble reference')
    ap.add_argument('--t0-variants', default=None)
    ap.add_argument('--tracer', default='LRG1'); ap.add_argument('--region', default='NGC')
    ap.add_argument('--out', default='r2_single_catalogue.csv')
    args = ap.parse_args()
    df = pd.concat([single(f) for f in args.files]); df.to_csv(args.out, index=False)
    piv = df.pivot_table(index=['ell', 'band', 'k_q'], columns='mock', values='R2')
    n = piv.notna().sum(1); piv['mean'] = piv.mean(1); piv['scatter'] = piv.std(1, ddof=1); piv['err_mean'] = piv['scatter'] / np.sqrt(n)
    print(piv.round(2).to_string())
    if args.bundle:
        bands = sorted({(r.k1, r.k2) for r in df.itertuples()})
        snr = {b: df[(df.k1 == b[0]) & (df.k2 == b[1])].SN_over_Ps.mean() for b in bands}
        ens = pd.concat([ensemble(args.bundle, args.tracer, args.region, bands, snr, args.t0_variants, ell_h=l) for l in (0, 2)])
        print(ens.round(2).to_string(index=False)); ens.to_csv(args.out.replace('.csv', '_ensemble.csv'), index=False)
        tr = tree_expectation(args.bundle, args.tracer, args.region, bands, sorted(df.k_q.unique()))
        print(tr.round(3).to_string(index=False)); tr.to_csv(args.out.replace('.csv', '_tree.csv'), index=False)


if __name__ == '__main__':
    main()
