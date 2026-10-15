"""Closed-form thinning covariance of P0 from the data catalogue alone (no thinnings), tested against the
Monte-Carlo thinning of thinning_cov.py on the same catalogue.

    python -u -m desi_validation.thinning_closed_form --bin LRG1 --region NGC --mock 173

For a random thinning eps_g ~ Bernoulli(p) with weights w_g / p, write eta_g = eps_g / p - 1 (<eta> = 0,
<eta^2> = a = (1 - p) / p). With F_1 the full-density field (galaxies minus alpha randoms) and the realised
shot noise sum_g (w_g eps_g / p)^2 subtracted,
    P_p(k) - P_1(k) = sum_g eta_g G_g(k) + sum_{g != h} w_g w_h eta_g eta_h Re[e_g(k) e_h^*(k)],
    G_g(k) = 2 Re[F_1^*(k) w_g e_g(k)] - 2 w_g^2,      e_g(k) = exp(-i k.x_g),
(the eta_g^2 diagonal of |D|^2 cancels against the shot-noise term exactly), hence
    Cov_thin[P_p(k), P_p(k')] = a sum_g G_g(k) G_g(k') + a^2/2 [ |M_1(k,k')|^2 + |M_2(k,k')|^2 - 2 sum_g w_g^4 ],
    M_1(k,k') = sum_g w_g^2 e_g(k) e_g^*(k'),   M_2(k,k') = sum_g w_g^2 e_g(k) e_g(k').
Shell-averaged for P0: every piece is a sum over the galaxies of products of the data field and the w^2-weighted
catalogue's transform, i.e. mesh-level quantities. Here they are computed exactly with NUFFTs on the output grid of
thinning_cov.py (same box, bins, norm, LOS-free monopole), by looping over galaxy chunks for the linear term
(G_g needs F_1 at all modes, N_g x N_modes) and by two more transforms for M_1, M_2 (|M|^2 shell averages are
convolutions of the w^2-catalogue power; done on the same grid with an FFT over the mode difference).
The a term is the collapsed bispectrum "B/n" of the galaxies (one shared particle), the a^2 term the "P/n^2"
(two shared), both as the data realise them.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc  # noqa: E402
from desi_validation.grf_randoms import NTH, log  # noqa: E402
from desi_validation.thinning_cov import KR, setup_mock  # noqa: E402


def main():
    import finufft
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--mock', type=int, default=173)
    ap.add_argument('--random-files', type=int, default=4)
    ap.add_argument('--pad', type=float, default=1.1)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    paths, cfg, od = setup_mock(args)
    b, r = args.bin, args.region
    g = dict(np.load(os.path.join(od, 'thinning.npz')))
    k, edges, nb, norm = g['k'], g['k_edges'], len(g['k']), float(g['norm'])
    rc = dc.load_region(paths, b, r, n_random_files=args.random_files)
    wd, wr = dc.total_weight(rc.data), dc.total_weight(rc.randoms)
    xd = dc.sky_to_cartesian(rc.data['RA'], rc.data['DEC'], rc.data['Z'])
    xr = dc.sky_to_cartesian(rc.randoms['RA'], rc.randoms['DEC'], rc.randoms['Z'])
    del rc
    alpha = float(np.sum(wd) / np.sum(wr))
    lo, hi = np.minimum(xd.min(0), xr.min(0)), np.maximum(xd.max(0), xr.max(0))
    L = args.pad * float(np.max(hi - lo))
    lo = 0.5 * (lo + hi) - 0.5 * L
    kf = 2 * np.pi / L
    n = 2 * int(np.ceil(edges[-1] / kf)) + 4
    log(f'{b} {r} mock {args.mock}: {len(wd)} galaxies, box {L:.0f}, grid {n}^3 ({n ** 3 / 1e6:.0f}M modes), norm {norm:.4g}')
    ko = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KO = np.meshgrid(ko, ko, ko, indexing='ij')
    kom = np.sqrt(KO[0] ** 2 + KO[1] ** 2 + KO[2] ** 2).ravel()
    del KO
    io = np.digitize(kom, edges) - 1
    io[(io < 0) | (io >= nb)] = nb
    nsh = np.bincount(io, minlength=nb + 1)[:nb]
    opts = dict(eps=args.eps, nthreads=NTH, modeord=1)
    th = lambda x: [np.ascontiguousarray(2 * np.pi * (x[:, i] - lo[i]) / L) for i in range(3)]
    t0 = time.time()
    F1 = finufft.nufft3d1(*th(xd), wd.astype(complex), (n,) * 3, isign=-1, **opts).ravel() \
        - alpha * finufft.nufft3d1(*th(xr), wr.astype(complex), (n,) * 3, isign=-1, **opts).ravel()
    P1 = (np.bincount(io, np.abs(F1) ** 2, nb + 1)[:nb] / nsh - float(np.sum(wd ** 2)) - alpha ** 2 * float(np.sum(wr ** 2))) / norm
    log(f'  F_1 done ({time.time() - t0:.0f} s); P_1 vs thinning_cov P1: max rel dev {np.max(np.abs(P1 / g["P1"][:nb] - 1)):.2e}')

    # linear term: Cov_lin(k,k') = a sum_g Gbar_g(k) Gbar_g(k'), Gbar_g(i) = [2 Re(w_g sum_{k in i} F_1^*(k) e_g(k)) / nsh_i - 2 w_g^2] / norm.
    # sum_{k in i} F_1^*(k) e_g(k) is a type-2 NUFFT of F_1^* restricted to shell i, evaluated at the galaxies: one transform per shell,
    # all galaxies at once (N_g x nb doubles = 56 x 0.7M x 8 B = 0.3 GB), no N_g x N_modes array.
    t0 = time.time()
    thd = th(xd)
    F1c = np.conj(F1).reshape(n, n, n)
    io3 = io.reshape(n, n, n)
    Gb = np.empty((nb, len(wd)))
    for i in range(nb):
        v = finufft.nufft3d2(*thd, np.where(io3 == i, F1c, 0), isign=-1, **opts)
        Gb[i] = (2 * v.real * wd / nsh[i] - 2 * wd ** 2) / norm
        if i in (0, nb - 1):
            log(f'    shell {i}: {time.time() - t0:.0f} s')
    Gsum = Gb @ Gb.T
    del F1c, Gb
    log(f'  linear (B/n-like) term done ({time.time() - t0:.0f} s)')

    # quadratic term: a^2 [ <|W2(k' - k)|^2>_{k in i, k' in j} + <|W2(k' + k)|^2> - 2 sum w^4 ] / norm^2, W2 = transform of the w^2 galaxies.
    # Shell-pair averages are correlations of shell indicators with |W2|^2: corr_i(d) = sum_k ind_i(k) pw(k + d) by FFT, then
    # Q_ij = sum_d corr_i(d) ind_j(d) (k' = d, i.e. pw(k' - k) summed over k in i, k' in j; and pw(k' + k) with ind_j reflected).
    t0 = time.time()
    W2 = finufft.nufft3d1(*thd, (wd ** 2).astype(complex), (n,) * 3, isign=-1, **opts)
    PW = sfft.fftn(np.abs(W2) ** 2, workers=NTH)
    del W2
    Q = np.zeros((nb, nb))
    for i in range(nb):
        INDi = sfft.fftn((io3 == i).astype(np.float32), workers=NTH)
        corr = sfft.ifftn(np.conj(INDi) * PW, workers=NTH).real.ravel()
        Q[i] = np.bincount(io, corr, nb + 1)[:nb]
    Q = Q / np.outer(nsh, nsh)
    del PW
    quad = (Q + Q - 2 * float(np.sum(wd ** 4))) / norm ** 2          # M_1 and M_2 pieces (shells are symmetric under k -> -k), each minus g = h
    log(f'  quadratic (P/n^2-like) term done ({time.time() - t0:.0f} s)')

    out = dict(k=k, P1=P1, lin=Gsum, quad=quad, alpha=alpha, norm=norm, n_gal=len(wd))
    fn = os.path.join(od, 'thinning_closed_form.npz')
    np.savez(fn, **out)
    G1 = np.load(os.path.join(od, 'thecov_p1.npz'))['C_G']
    d1 = np.diag(G1)[:nb]
    print(f'closed form vs Monte Carlo thinning ({b} {r} mock {args.mock}); units: diag G_thecov(p=1); bands {KR}')
    for p in g['fracs']:
        a = (1 - p) / p
        Cc = a * Gsum + a ** 2 * quad
        if f'P_p{p:g}' not in g:
            continue
        Ct = np.cov(g[f'P_p{p:g}'][:, :nb].T)
        N = len(g[f'P_p{p:g}'])
        row = []
        for lo_, hi_ in KR:
            m = (k >= lo_) & (k < hi_)
            mc, cf = np.mean(np.diag(Ct)[m] / d1[m]), np.mean(np.diag(Cc)[m] / d1[m])
            row.append(f'MC {mc:.3f} (+-{mc * np.sqrt(2 / N) / np.sqrt(m.sum() / 2.5):.3f}) closed {cf:.3f} '
                       f'[lin {np.mean(a * np.diag(Gsum)[m] / d1[m]):.3f} quad {np.mean(a ** 2 * np.diag(quad)[m] / d1[m]):.3f}]')
        print(f'  p = {p}: ' + ' | '.join(row))
        m = k > 0.15
        od_mc = [np.mean([Ct[i, i + d] / np.sqrt(d1[i] * d1[i + d]) for i in np.flatnonzero(m) if i + d < nb]) for d in (1, 2, 3)]
        od_cf = [np.mean([Cc[i, i + d] / np.sqrt(d1[i] * d1[i + d]) for i in np.flatnonzero(m) if i + d < nb]) for d in (1, 2, 3)]
        print(f'         off-diagonal |i-j| = 1..3, k > 0.15: MC ' + ' '.join(f'{x:+.3f}' for x in od_mc) + '  closed ' + ' '.join(f'{x:+.3f}' for x in od_cf))
        # noise-free decomposition at full density. thecov's Gaussian difference dG(p) = G(S_p) - G(S_1) is a quadratic in a = (1-p)/p
    # (PS and SS terms); fit g1, g2 from the two thinned thecov runs. Then the galaxies' own terms at p = 1 are
    #   B/n  = lin  - g1   (scales as 1/p = 1 + a)         P/n^2 = quad - g2   (scales as 1/p^2 = 1 + 2a + a^2, the a^2 part)
    # both in units of diag G_thecov(p = 1), per bin; the 2a P/n^2 piece is absorbed consistently (it is in g1 + lin at the same order).
    import glob as _glob
    fns = {float(os.path.basename(f)[8:-4]): f for f in _glob.glob(os.path.join(od, 'thecov_p*.npz'))}
    fns = {p: f for p, f in fns.items() if p != 1}
    if len(fns) >= 2:
        ps = sorted(fns)
        A = np.array([[(1 - p) / p, ((1 - p) / p) ** 2] for p in ps])
        dG = np.array([np.load(fns[p])['C_G'][:nb, :nb] - G1[:nb, :nb] for p in ps])
        coef, *_ = np.linalg.lstsq(A, dG.reshape(len(ps), -1), rcond=None)
        g1, g2 = coef.reshape(2, nb, nb)
        if len(ps) > 2:
            resid = dG.reshape(len(ps), -1) - A @ coef
            print(f'\n  quadratic-in-a fit of thecov dG over p = {ps}: rms residual / diag G = '
                  f'{np.sqrt(np.mean(resid.reshape(len(ps), nb, nb)[:, np.arange(nb), np.arange(nb)] ** 2 / d1 ** 2)):.4f} (pair-count noise of the thecov runs)')
        Bn, Pn2 = Gsum - g1, quad - g2
        print('\nfull-density discreteness terms of the catalogue (closed form minus thecov Gaussian S-dependence), units diag G(p=1):')
        print('  band        B/n diag    P/n^2 diag   | thecov Gaussian S-terms g1, g2 (same units) | tree level (HANDOFF 9.4): B/n 0.07-0.08, P/n^2 0.015 at k 0.15-0.3')
        for lo_, hi_ in KR:
            m = (k >= lo_) & (k < hi_)
            print(f'  {lo_}-{hi_}:  {np.mean(np.diag(Bn)[m] / d1[m]):+.3f}      {np.mean(np.diag(Pn2)[m] / d1[m]):+.3f}      | '
                  f'{np.mean(np.diag(g1)[m] / d1[m]):.3f}, {np.mean(np.diag(g2)[m] / d1[m]):.3f}')
        m = k > 0.15
        for name, M_ in (('B/n', Bn), ('P/n^2', Pn2)):
            od_ = [np.mean([M_[i, i + d] / np.sqrt(d1[i] * d1[i + d]) for i in np.flatnonzero(m) if i + d < nb]) for d in (0, 1, 2, 3, 5, 8)]
            print(f'  {name:6s} (i,j) at k > 0.15, |i-j| = 0,1,2,3,5,8: ' + ' '.join(f'{x:+.3f}' for x in od_))
        out.update(Bn=Bn, Pn2=Pn2, g1=g1, g2=g2)
        np.savez(fn, **out)
    log(f'saved {fn}')


if __name__ == '__main__':
    main()
