"""Assemble the recommended ("best") covariance of one tracer bin from the catalogue_covariance outputs of its two caps,
combine the caps into GCcomb, and validate all three against the mocks.

    python -m desi_validation.assemble_best --bin LRG2 [--catcov-dir ~/thecov/catcov] [--label holi-kcore2-LRG2]
        [--out /global/cfs/cdirs/desicollab/users/oalves/thecov_validation/best_covariances]

Per cap (catcov/best_<bin>_<cap>/catalogue_covariance.npz, decomposed with disc_multipoles.decompose):
    G      = G_PP + s G_PS + s^2 G_SS, s = <num_shotnoise of the files> / S of the catalogue run (the files' shot noise)
    disc   = measured B/n + P/n^2 in every block (P/n^2 outside P0xP0: tree x measured/tree ratio; P0xP0 |i-j|<=1 by
             continuity). The hybrid (measured P0xP0, tree elsewhere; disc_hybrid) is stored too: for LRG2 it is not
             positive definite with P4 (min eig of G^-1/2 C G^-1/2 -0.8 NGC, -3.5 SGC), while the measured terms are
             (0.92 / 0.88) and fit the mocks best (P0+P2+P4 <chi2>/n 0.98 / 0.99, lowest Wishart deviance)
    SSC    = clustering SSC + measured Poisson local average, C_LA = -T P^T - P T^T + Var P P^T with T = T_M + T_W and
             Var = var_MM + 2 var_MW + var_WW from the thinning (HANDOFF 22, v5), P = the SSC's P_hat
    C_best = G + disc + SSC
    T0_tree (snake + star) and T0_box (P0xP0 squeezed, k < 0.1 <= k', x0; rest of P0xP0 x0.3; l > 0 blocks x1) are stored
    separately. GCcomb: dc.combine_regions with the files' mean norms (the GCcomb spectra are the norm-weighted average of the
    caps); caps treated as independent (no NGC x SGC super-sample term).
Validation against the mocks of report_data_<label>.npz (LRG1: the 12 defective-n(z) mocks removed): <chi2>/n and
Var(chi2)/2n for P0, P0+P2, P0+P2+P4 at kmax 0.2 and 0.3, and the P0 diagonal ratio mock/model in k bands, for G only,
G + disc, C_best and C_best + T0_box.
Output: <out>/<bin>_<region>.npz (+ .json with the numbers), and rows appended to <out>/summary.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import disc_multipoles as dm  # noqa: E402

CAPS = ('NGC', 'SGC')
ELLS = (0, 2, 4)
EXCLUDED_LRG1 = [122, 131, 166, 218, 470, 573, 664, 667, 699, 746, 793, 880]   # HANDOFF 8.4 (defective n(z), LRG1 slice)
KB = ((0.02, 0.1), (0.1, 0.2), (0.2, 0.3))


def continuity(M, nb):
    """fill |i - j| <= 1 of the P0xP0 block of a measured term from the neighbouring off-diagonals (|i-j| = 2, 3)"""
    M = M.copy()
    B = M[:nb, :nb].copy()
    for i in range(nb):
        B[i, max(i - 1, 0):i + 2] = np.mean([B[i, j] for j in (i - 3, i - 2, i + 2, i + 3) if 0 <= j < nb])
    M[:nb, :nb] = (B + B.T) / 2
    return M


def t0_box(T0, k, nb):
    """box-calibrated T0 (HANDOFF 22): P0xP0 squeezed x0, rest of P0xP0 x0.3, l > 0 blocks x1"""
    T = T0.copy()
    lo = k < 0.1
    sq = np.logical_xor.outer(lo, lo)
    T[:nb, :nb] = np.where(sq, 0.0, 0.3 * T0[:nb, :nb])
    return T


def cap_terms(d_cat, s_files):
    d = dm.decompose(d_cat)
    z = np.load(os.path.join(d_cat, 'catalogue_covariance.npz'))
    nb, k = d['nb'], d['k']
    s = s_files / d['sn']
    G = d['GPP'] + s * d['GPS'] + s ** 2 * d['GSS']
    meas = dict(d)
    meas['Bn'] = continuity(d['Bn'], nb)
    meas['Pn2'] = continuity(d['Pn2'], nb)
    hyb = dm.hybrid(meas)
    disc = meas['Bn'] + meas['Pn2']
    P = z['ssc_P_hat']
    T = z['la_T_M'] + z['la_T_W']
    var = float(z['la_var_MM'] + 2 * z['la_var_MW'] + z['la_var_WW'])
    C_la = -np.outer(T, P) - np.outer(P, T) + var * np.outer(P, P)
    ssc = z['C_ssc_clustering'] + C_la
    T0 = z['C_T0_snake'] + z['C_T0_star']
    info = dict(shotnoise_files=float(s_files), shotnoise_catalogue=float(d['sn']), shotnoise_scale=float(s),
                ratio_Pn2_meas_tree=float(d['ratio_P']), la_T_meas_over_tree=float(np.median((T / z['ssc_T_tree'])[:nb][k > 0.05])),
                la_var_meas_over_tree=var / float(z['ssc_var_tree']), J4=float(z['J4']),
                params_b1_b2_bs2_f=[float(x) for x in z['params']], damping_ir_sigmav_A=[float(x) for x in z['damping']],
                timings={q[5:]: float(z[q]) for q in z.files if q.startswith('time_')})
    terms = dict(G=G, disc=disc, disc_hybrid=hyb, disc_B_meas=meas['Bn'], disc_P_meas=meas['Pn2'], disc_B_tree=d['tree_B'], disc_P_tree=d['tree_P'],
                 SSC=ssc, SSC_clustering=z['C_ssc_clustering'], SSC_LA_measured=C_la, SSC_LA_tree=z['C_ssc_poisson'],
                 T0_tree=T0, T0_box=t0_box(T0, k, nb))
    return k, nb, terms, info


def chi2_stats(V, C, idx):
    N = len(V)
    D = V[:, idx] - V[:, idx].mean(0)
    Ci = np.linalg.inv(C[np.ix_(idx, idx)])
    chi2 = np.einsum('ij,jk,ik->i', D, Ci, D) * N / (N - 1)
    n = len(idx)
    return float(chi2.mean() / n), float(chi2.var(ddof=1) / (2 * n))


def validate(V, k, nb, models):
    out = {}
    S = np.cov(V.T)
    for nm, C in models.items():
        row = {'min_eig_over_G': float(np.linalg.eigvalsh(C).min())}
        for kmax in (0.2, 0.3):
            sel = np.flatnonzero(k <= kmax + 1e-9)
            for ells in ((0,), (0, 2), (0, 2, 4)):
                idx = np.concatenate([l // 2 * nb + sel for l in ells])
                m, v = chi2_stats(V, C, idx)
                tag = 'P' + '+P'.join(map(str, ells)) + f'_k{kmax}'
                row[f'chi2n_{tag}'], row[f'varchi2_{tag}'] = round(m, 4), round(v, 4)
        x = np.diag(S)[:nb] / np.diag(C)[:nb]
        row['P0diag'] = [round(float(np.mean(x[(k >= lo) & (k < hi + 1e-9)])), 4) for lo, hi in KB]
        out[nm] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--catcov-dir', default=os.path.expanduser('~/thecov/catcov'))
    ap.add_argument('--report-dir', default=os.path.expanduser('~/thecov_desi/holi_v3_mock173'))
    ap.add_argument('--label', default=None)
    ap.add_argument('--out', default='/global/cfs/cdirs/desicollab/users/oalves/thecov_validation/best_covariances')
    args = ap.parse_args()
    b = args.bin
    label = args.label or f'holi-kcore2-{b}'
    from desi_validation import desi_compare as dc
    R = np.load(os.path.join(args.report_dir, f'report_data_{label}.npz'))
    commit = subprocess.run(['git', '-C', os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    per, norms, vecs = {}, {}, {}
    for r in CAPS + ('GCcomb',):
        V, ids = R[f'{b}/{r}/x1/V'].astype(float), R[f'{b}/{r}/x1/mock_ids']
        ids = np.array([int(str(i).replace('mock', '')) for i in ids])          # ids are 'mock<N>' strings
        keep = ~np.isin(ids, EXCLUDED_LRG1) if b == 'LRG1' else np.ones(len(ids), bool)
        vecs[r] = V[keep]
        norms[r] = float(R[f'{b}/{r}/x1/norm'][keep].mean())
        if r in CAPS:
            s_files = float(R[f'{b}/{r}/x1/num_shotnoise'][keep].mean())
            k, nb, terms, info = cap_terms(os.path.join(args.catcov_dir, f'best_{b}_{r}'), s_files)
            assert np.allclose(k, R[f'{b}/{r}/x1/k'], rtol=1e-3), 'k binning differs between the catalogue run and the files'
            per[r] = (terms, info)
    w = {r: norms[r] / (norms['NGC'] + norms['SGC']) for r in CAPS}
    Pc = w['NGC'] * vecs['NGC'].mean(0) + w['SGC'] * vecs['SGC'].mean(0)
    comb_dev = float(np.max(np.abs(Pc - vecs['GCcomb'].mean(0)) / np.abs(vecs['GCcomb'].mean(0)).max()))
    comb = {t: dc.combine_regions([per[r][0][t] for r in CAPS], [norms[r] for r in CAPS]) for t in per['NGC'][0]}
    per['GCcomb'] = (comb, dict(weights=w, mean_spectrum_check=comb_dev, note='caps independent: no NGC x SGC super-sample term'))
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for r, (terms, info) in per.items():
        best = terms['G'] + terms['disc'] + terms['SSC']
        V = vecs[r]
        Gd = np.sqrt(np.diag(terms['G']))
        unit = lambda C: C / np.outer(Gd, Gd)                           # for the min-eigenvalue check in G units
        models = {'G': terms['G'], 'G+disc': terms['G'] + terms['disc'], 'best': best, 'best_hybrid_disc': terms['G'] + terms['disc_hybrid'] + terms['SSC'],
                  'best+T0_box': best + terms['T0_box']}
        val = validate(V, k, nb, {nm: C for nm, C in models.items()})
        for nm, C in models.items():
            Gi = np.linalg.cholesky(terms['G']); Li = np.linalg.inv(Gi)
            val[nm]['min_eig_over_G'] = round(float(np.linalg.eigvalsh(Li @ C @ Li.T).min()), 4)
        meta = dict(bin=b, region=r, n_mocks=int(len(V)), ells=list(ELLS), k=k.tolist(), commit=commit, label=label,
                    mock_catalogue=173, power='mock mean', **info, validation=val)
        np.savez(os.path.join(args.out, f'{b}_{r}.npz'), C_best=best, k=k, ells=np.array(ELLS), mock_mean=V.mean(0),
                 n_mocks=len(V), **{f'C_{t}': v for t, v in terms.items()})
        json.dump(meta, open(os.path.join(args.out, f'{b}_{r}.json'), 'w'), indent=1)
        for nm, row in val.items():
            rows.append(dict(bin=b, region=r, model=nm, n_mocks=len(V), **{q: v for q, v in row.items() if q != 'P0diag'},
                             **{f'P0diag_{lo}-{hi}': v for (lo, hi), v in zip(KB, row['P0diag'])}))
        print(f'{b} {r} ({len(V)} mocks): ' + ' | '.join(f'{nm} <chi2>/n P0+P2 k0.3 {val[nm]["chi2n_P0+P2_k0.3"]:.3f} '
                                                       f'P0+P2+P4 {val[nm]["chi2n_P0+P2+P4_k0.3"]:.3f} min eig {val[nm]["min_eig_over_G"]:+.3f}'
                                                       for nm in models), flush=True)
    fn = os.path.join(args.out, 'summary.csv')
    import csv
    new = not os.path.exists(fn)
    old = [] if new else [row for row in csv.DictReader(open(fn)) if row['bin'] != b]
    keys = list(rows[0].keys())
    with open(fn, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        wr.writeheader()
        for row in old + rows:
            wr.writerow(row)
    print(f'GCcomb mean-spectrum check (norm-weighted caps vs files): max rel dev {comb_dev:.2e}; weights {w}')
    print(f'saved {args.out}/{b}_{{NGC,SGC,GCcomb}}.npz/.json, summary.csv')


if __name__ == '__main__':
    main()
