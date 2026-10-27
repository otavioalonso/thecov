"""Measured discreteness terms for P0, P2, P4 from one catalogue, and the resulting covariance against the mocks.

    python -m desi_validation.disc_multipoles --dir <thinning dir with thinning_closed_form_ell024.npz, thecov_p{1,0.5,0.25}.npz>
           [--bundle t0_inputs.npz --tracer LRG1 --region NGC]

Inputs (thinning_closed_form --ells 0 2 4; thinning_cov thecov --frac p at p = 1, 0.5, 0.25 with the same ells and
--tree-params): lin_ell = sum_g G^l_g G^l'_g (exact, all blocks), quad (exact, l = l' = 0), thecov Gaussian C_G(p) and the
tree terms disc_B, disc_P. thecov's Gaussian is quadratic in the shot noise, G(a) = G_1 + a g1 + a^2 g2 (a = (1-p)/p,
from the realised S_p), so
    P/n^2 = quad - g2,   B/n = lin - g1 - 2 P/n^2.
P/n^2 outside the (0, 0) block (the two-shared-galaxy term, 1-2 per cent of G) is the tree term scaled by the measured /
tree ratio of the (0, 0) off-diagonals (ratio_off); B/n is exact in every block. The Gaussian splits as G_PP + G_PS + G_SS.
With --bundle: G + B/n + P/n^2 (+ SSC from the bundle) for P0+P2+P4 against the clean mock covariance (PD, Wishart
deviance per multipole set, diagonal ratios in k bands).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

KR = ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3))
FIT_THREE = False


def decompose(d):
    one = os.path.join(d, 'catalogue_covariance.npz')
    if os.path.exists(one):                              # catalogue_covariance output: everything in one file
        cf = t1 = np.load(one)
        lin, quad00 = cf['lin_ell'], cf['quad']
        nb = len(quad00)
    else:
        cf = np.load(os.path.join(d, 'thinning_closed_form_ell024.npz'))
        lin, quad00 = cf['lin_ell'], cf['quad']
        nb = len(cf['k'])
        t1 = np.load(os.path.join(d, 'thecov_p1.npz'))
    G = {1: t1['C_G']}
    n = G[1].shape[0]
    assert lin.shape == (n, n), (lin.shape, n)
    if 'C_PP' in t1.files and not FIT_THREE:            # exact split from one run (GaussianCovariance.covariance_parts)
        GPP, GPS, GSS = t1['C_PP'], t1['C_PS'], t1['C_SS']
        sw2, sn1 = float(t1['sum_w2']), float(t1['sn'])
        r = sn1 / sw2                                   # S(a) = S_1 + a sum w^2: G(a) = PP + (1 + a/r) PS + (1 + a/r)^2 SS
        g1 = GPS / r + 2 * GSS / r ** 2
        g2 = GSS / r ** 2
        a = {'mode': 'one-run split'}
    else:                                               # three runs at p = 1, 0.5, 0.25: quadratic fit in a
        tc = {p: np.load(os.path.join(d, f'thecov_p{p:g}.npz')) for p in (1, 0.5, 0.25)}
        G = {p: tc[p]['C_G'] for p in tc}
        sn = {p: float(tc[p]['sn']) for p in tc}
        sw2 = (sn[0.25] - sn[1]) / 3.0                  # sum w^2 of the galaxies (S_p = sw2 / p + randoms)
        a = {p: (sn[p] - sn[1]) / sw2 for p in (0.5, 0.25)}
        A = np.array([[a[p], a[p] ** 2] for p in (0.5, 0.25)])
        g1, g2 = np.linalg.solve(A, np.array([(G[p] - G[1]).ravel() for p in (0.5, 0.25)])).reshape(2, n, n)
        r = sn[1] / sw2
        GSS = g2 * r ** 2
        GPS = g1 * r - 2 * g2 * r ** 2
        GPP = G[1] - GPS - GSS
    tc = {1: t1}
    TB, TP = tc[1]['disc_B'], tc[1]['disc_P']
    k = cf['k']
    Pn2_00 = quad00 - g2[:nb, :nb]
    ratio = ratio_off(Pn2_00, TP[:nb, :nb], k)
    Pn2 = ratio * TP
    Pn2[:nb, :nb] = Pn2_00
    Bn = lin - g1 - 2 * Pn2
    return dict(k=k, nb=nb, G=G[1], GPP=GPP, GPS=GPS, GSS=GSS, Bn=Bn, Pn2=Pn2, tree_B=TB, tree_P=TP, ratio_P=ratio,
                a=a, sn=float(t1["sn"]), sw2=sw2)


def ratio_off(meas, tree, k, kmin=0.1, offs=range(3, 9)):
    """measured / tree from the off-diagonals |i-j| in offs at k > kmin (the near-diagonal is window-dominated)"""
    num = den = 0.0
    for o in offs:
        i = np.flatnonzero(k[:-o] > kmin)
        num += np.sum(meas[i, i + o]); den += np.sum(tree[i, i + o])
    return num / den


def band(d, M, ell1, ell2=None):
    ell2 = ell1 if ell2 is None else ell2
    nb, k = d['nb'], d['k']
    i1, i2 = ell1 // 2 * nb, ell2 // 2 * nb
    ref = np.sqrt(np.outer(np.diag(d['G'])[i1:i1 + nb], np.diag(d['G'])[i2:i2 + nb]))
    x = np.diag(M[i1:i1 + nb, i2:i2 + nb] / ref)
    return [float(np.mean(x[(k >= lo) & (k < hi)])) for lo, hi in KR]


def wishart_D(C, S):
    n = len(C)
    M = np.linalg.solve(C, S)
    s, ld = np.linalg.slogdet(M)
    return 0.5 * (np.trace(M) - ld - n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--bundle', default=None, help='t0_inputs.npz (mock vectors, production SSC, excluded mocks)')
    ap.add_argument('--tracer', default='LRG1'); ap.add_argument('--region', default='NGC')
    ap.add_argument('--kmax', type=float, default=0.3)
    ap.add_argument('--out', default=None)
    ap.add_argument('--fit-three', action='store_true', help='use the three-run quadratic fit even if the one-run split is present')
    args = ap.parse_args()
    global FIT_THREE
    FIT_THREE = args.fit_three
    d = decompose(args.dir)
    print(f'measured/tree P/n^2 ratio (l = 0 off-diagonals, k > 0.1): {d["ratio_P"]:.3f}; {d["a"]}')
    print('diagonal / G, band means', KR)
    for l1, l2 in ((0, 0), (2, 2), (4, 4), (0, 2), (0, 4), (2, 4)):
        for nm in ('GPP', 'GPS', 'GSS', 'Bn', 'Pn2', 'tree_B', 'tree_P'):
            print(f'  P{l1}xP{l2} {nm:7s} ' + ' '.join(f'{v:+.4f}' for v in band(d, d[nm], l1, l2)))
    if args.out:
        np.savez(args.out, **{key: v for key, v in d.items() if isinstance(v, np.ndarray)})
    if not args.bundle:
        return
    Z = np.load(args.bundle)
    F = lambda key: Z[f'F/{args.tracer}/{args.region}/{key}'].astype(float)
    V = Z[f'MV/{args.tracer}/{args.region}/vectors'].astype(float); ids = Z[f'MV/{args.tracer}/{args.region}/mocks']
    if args.tracer.startswith('LRG'):
        V = V[~np.isin(ids, Z['excluded_LRG1'])]
    S = np.cov(V.T); N = len(V)
    nb, k = d['nb'], d['k']
    assert np.allclose(F('k'), k, rtol=1e-3)
    ssc = F('C_ssc')
    models = {'G (catalogue Gaussian)': d['G'],
              'G + tree disc': d['G'] + d['tree_B'] + d['tree_P'],
              'G + measured disc': d['G'] + d['Bn'] + d['Pn2'],
              'G + measured disc + SSC': d['G'] + d['Bn'] + d['Pn2'] + ssc}
    sel_k = np.flatnonzero(k <= args.kmax + 1e-9)
    print(f'\nagainst {N} mocks (k <= {args.kmax}); Wishart deviance per d.o.f.-scaled: (N-1) * D, and min eigenvalue of G^-1/2 C G^-1/2')
    for ells in ((0,), (2,), (4,), (0, 2), (0, 2, 4)):
        idx = np.concatenate([l // 2 * nb + sel_k for l in ells])
        sub = lambda M: M[np.ix_(idx, idx)]
        Gs = sub(d['G']); w, U = np.linalg.eigh(Gs); Wh = U @ np.diag(w ** -0.5) @ U.T
        row = []
        for nm, C in models.items():
            ev = np.linalg.eigvalsh(Wh @ sub(C) @ Wh).min()
            row.append(f'{nm}: {(N - 1) * wishart_D(sub(C), sub(S)):8.1f} (min eig {ev:+.3f})')
        print(f'  ells {str(ells):10s} ' + ' | '.join(row))
    print('\nmock diagonal / model diagonal, band means')
    for l in (0, 2, 4):
        sl = slice(l // 2 * nb, l // 2 * nb + nb)
        for nm, C in models.items():
            x = np.diag(S[sl, sl]) / np.diag(C[sl, sl])
            print(f'  P{l} {nm:26s} ' + ' '.join(f'{np.mean(x[(k >= lo) & (k < hi)]):.3f}' for lo, hi in KR))


if __name__ == '__main__':
    main()
