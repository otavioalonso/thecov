"""Long-mode responses of P_l from the catalogue itself (integrated bispectrum in Fourier space, no sub-volumes),
for the SSC and squeezed-T0 terms.

    python -u -m desi_validation.response_ibisp --bin LRG1 --region NGC --mock 173 --mocks 173 174 175 176
    python -u -m desi_validation.response_ibisp --bin LRG1 --region NGC --mocks 173 174 175 --k-L 0.03 0.02 0.015 0.01 0.007 0.005   (scale dependence R(k_L))
    python -u -m desi_validation.response_ibisp --bin LRG1 --region NGC --data        (the DESI data catalogue)

Estimator. With F the FKP field (galaxies minus alpha randoms) and delta_L(x) = [F / (alpha n_r)]_lowpass(k < k_L) the
long-mode overdensity inside the survey (window-weighted, so it vanishes outside), the cross-spectrum
    C_l(k) = <F_l^*(k) FT[delta_L F](k)>_shell / norm
is the bispectrum collapsed on one long mode (k, -k - q, q, q < k_L), i.e. the position-dependent power spectrum
without the sub-volume binning (Chiang et al. 2014 integrated bispectrum, done with the filter instead of boxes).
The response is  dlnP_l / d delta_L (k) = C_l(k) / [P_l(k) sigma_L^2],  sigma_L^2 = <delta_L^2>_survey (weighted by m).
For an isotropic P the cross spectrum of an RSD field also carries the dipole-free l = 2 response; we measure l = 0, 2
with the first-point LOS on both legs, which is the quantity SuperSampleCovariance.responses() provides at tree level
(R_l^(0) / P_l, HANDOFF 9.2 measured it as a regression on delta_shell: P0 -0.06..-0.11, P2 +0.8..+2.5; tree level
~-0.25 and +0.5..+2). Shot noise: the three-point function of the discrete field has a (1/n) P term when the long and one
short leg share a galaxy; it is removed by using the galaxy field for one leg and the shot-free cross with the
w^2-weighted catalogue subtracted (the same collapsed term as thinning_closed_form: C_l^SN = <F_l^* FT[delta_L S]>,
S = sum w^2 delta_D - alpha^2 sum w_r^2 delta_D). On mocks: average over the mock ids given (cosmic variance of a single
realisation is the limit, not the estimator noise); the same call on the data gives the data's own response.
Second-order (squeezed T0 "completion"): the regression of P_hat_l(k) on the band power of delta_L, i.e.
    J_l(k) = <F_l^*(k) FT[delta_L^2 F](k)> / (P_l(k) <delta_L^4> - ...)   is also printed (leading term only).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402
from desi_validation.grf_randoms import NTH, log, tensor_index  # noqa: E402

KR = ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3))


def paths_for(args, mock):
    if args.data:
        paths = dc.Paths(kind='data', mock=0)
        paths.catalog_dir = args.data_dir
        paths.cs_version = paths.cs_parent_version = 'data-dr2-v2'
    else:
        paths = dc.Paths(kind='holi_v3', mock=mock)
        paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
    paths.catalog_names.update({'ELG_LOPnotqso': 'ELGnotqso', 'LRG+ELG_LOPnotqso': 'LRG+ELGnotqso'})
    paths.loader = 'auto'
    return paths


def one_catalogue(args, mock, grid_state):
    import finufft
    paths = paths_for(args, mock)
    cfg = pl.Config()
    b, r = args.bin, args.region
    rc = dc.load_region(paths, b, r, n_random_files=args.random_files)
    wd, wr = dc.total_weight(rc.data), dc.total_weight(rc.randoms)
    xd = dc.sky_to_cartesian(rc.data['RA'], rc.data['DEC'], rc.data['Z'])
    xr = dc.sky_to_cartesian(rc.randoms['RA'], rc.randoms['DEC'], rc.randoms['Z'])
    if args.lnpoisson is not None:
        # Poisson-sampled lognormal catalogue with DISTINCT randoms (the data's own bookkeeping): G = A_L dL + A_S ds on the
        # randoms, delta_g = exp(G - var G / 2) - 1, galaxies drawn from one half of the randoms with p = p0 (1 + delta_g), the
        # other half are the randoms. Local power is exactly (1 + delta_L)^2 P_s: R = 2 per unit galaxy long mode, any filter.
        from desi_validation.grf_randoms import p_input, kgrid
        A_L, A_S, p0, seed = args.lnpoisson
        s1 = dc.read_spectra(paths.spectra_fns(*dc.TRACER_SPECS[b], r), kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=cfg.ells)
        P0 = s1['vectors'][:, :len(s1['k'])].mean(0) * 1.22
        lo_, hi_ = xr.min(0), xr.max(0)
        Lg = 1.2 * float(np.max(hi_ - lo_))
        lo_ = 0.5 * (lo_ + hi_) - 0.5 * Lg
        ng = sfft.next_fast_len(2 * int(np.ceil(0.32 / (2 * np.pi / Lg))) + 4)
        kg = kgrid(ng, Lg)
        Pg = p_input(s1['k'], P0, kg, 0.32)
        rng = np.random.default_rng(int(seed) + 1000 * mock)
        Wm = sfft.fftn(rng.standard_normal((ng,) * 3), workers=NTH) * (np.sqrt(Pg / (Lg / ng) ** 3) / ng ** 3)
        WLg = np.exp(-0.5 * (kg / args.k_L[0]) ** 2) if args.filter == 'gauss' else (kg < args.k_L[0]).astype(float)
        thg = [np.ascontiguousarray(2 * np.pi * (xr[:, i] - lo_[i]) / Lg) for i in range(3)]
        opts0 = dict(eps=args.eps, nthreads=NTH, modeord=1)
        G = A_L * finufft.nufft3d2(*thg, Wm * WLg, isign=1, **opts0).real + A_S * finufft.nufft3d2(*thg, Wm * (1 - WLg), isign=1, **opts0).real
        del Wm, WLg, kg, Pg
        dg = np.exp(G - 0.5 * G.var()) - 1.0
        perm = rng.permutation(len(wr))
        par, est = perm[: len(wr) // 2], perm[len(wr) // 2:]
        sel = par[rng.random(len(par)) < np.clip(p0 * (1 + dg[par]), 0, 1)]
        xd, wd = xr[sel], wr[sel]
        xr, wr = xr[est], wr[est]
        log(f'  lognormal-Poisson catalogue: sigma_G = {G.std():.3f} {len(sel)} galaxies from {len(par)} parents '
            f'(p0 = {p0}), {len(est)} randoms; expected dlnP0/d delta_gL = 2, dlnP2 = 0')
        del G, dg, perm, par, est, sel
    elif args.synthetic is not None:
        # known-response field on the real randoms: delta_g = g delta_L + delta_s (1 + R_m delta_L), delta_L = W_L delta_m,
        # delta_s = (1 - W_L) delta_m, W_L the estimator's own filter. Expected dlnP/d delta_{g,L} = 2 R_m / g; P2 response = 0.
        from desi_validation.grf_randoms import p_input, kgrid
        R_m, g_b, seed = args.synthetic
        s1 = dc.read_spectra(paths.spectra_fns(*dc.TRACER_SPECS[b], r), kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=cfg.ells)
        P0 = s1['vectors'][:, :len(s1['k'])].mean(0) * 1.22
        lo_, hi_ = xr.min(0), xr.max(0)
        Lg = 1.2 * float(np.max(hi_ - lo_))
        lo_ = 0.5 * (lo_ + hi_) - 0.5 * Lg
        ng = sfft.next_fast_len(2 * int(np.ceil(0.32 / (2 * np.pi / Lg))) + 4)
        kg = kgrid(ng, Lg)
        Pg = p_input(s1['k'], P0, kg, 0.32)
        rng = np.random.default_rng(int(seed) + 1000 * mock)
        Wm = sfft.fftn(rng.standard_normal((ng,) * 3), workers=NTH) * (np.sqrt(Pg / (Lg / ng) ** 3) / ng ** 3)
        WLg = np.exp(-0.5 * (kg / args.k_L[0]) ** 2) if args.filter == 'gauss' else (kg < args.k_L[0]).astype(float)
        thg = [np.ascontiguousarray(2 * np.pi * (xr[:, i] - lo_[i]) / Lg) for i in range(3)]
        opts0 = dict(eps=args.eps, nthreads=NTH, modeord=1)
        dL_r = finufft.nufft3d2(*thg, Wm * WLg, isign=1, **opts0).real
        ds_r = finufft.nufft3d2(*thg, Wm * (1 - WLg), isign=1, **opts0).real
        del Wm, WLg, kg, Pg
        if g_b < 0:                               # lognormal: delta_g = exp(sG - sG^2/2) - 1 with sG = |g| x (dL + ds)/sigma: R = 2 exactly for every q
            dG = dL_r + ds_r
            dG *= abs(g_b) / dG.std()
            dg = np.exp(dG - 0.5 * dG.var()) - 1.0
            log(f'  lognormal field on the randoms: sigma_G = {abs(g_b)}; expected dlnP0/d delta_gL = 2 (all q), dlnP2 = 0')
        else:
            dg = g_b * dL_r + ds_r * (1 + R_m * dL_r)
            log(f'  synthetic field on the randoms: R_m = {R_m}, g = {g_b}; sigma(delta_L) = {dL_r.std() * 100:.2f} %, '
                f'sigma(delta_s) = {ds_r.std():.3f}; expected dlnP0/d delta_gL = {2 * R_m / g_b:.3f} (modulation only at q < k_L), dlnP2 = 0')
        xd, wd = xr, wr * (1 + dg)
        del dL_r, ds_r, dg
    del rc
    alpha = float(np.sum(wd) / np.sum(wr))
    if grid_state is None:
        lo, hi = np.minimum(xd.min(0), xr.min(0)), np.maximum(xd.max(0), xr.max(0))
        L = args.pad * float(np.max(hi - lo))
        lo = 0.5 * (lo + hi) - 0.5 * L
        n = sfft.next_fast_len(2 * int(np.ceil(args.kmax / (2 * np.pi / L))) + 4)
        grid_state = dict(lo=lo, L=L, n=n)
        log(f'{b} {r}: box {L:.0f}, grid {n}^3 (k_f = {2 * np.pi / L:.5f}); k_L = {args.k_L}')
    lo, L, n = grid_state['lo'], grid_state['L'], grid_state['n']
    th = lambda x: [np.ascontiguousarray(2 * np.pi * (x[:, i] - lo[i]) / L) for i in range(3)]
    opts = dict(eps=args.eps, nthreads=NTH, modeord=1)
    ko = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KO = np.meshgrid(ko, ko, ko, indexing='ij')
    km = np.sqrt(KO[0] ** 2 + KO[1] ** 2 + KO[2] ** 2)
    edges = np.arange(cfg.kmin, args.kmax + 1e-9, 0.005)
    nb = len(edges) - 1
    io = np.digitize(km.ravel(), edges) - 1
    io[(io < 0) | (io >= nb)] = nb
    nsh = np.bincount(io, minlength=nb + 1)[:nb]
    with np.errstate(invalid='ignore', divide='ignore'):
        khat = [np.where(km > 0, K_ / km, 0.0).ravel() for K_ in KO]
    del KO
    # fields on the grid: F (data - alpha randoms), N_r (alpha randoms, the mean density m), S (shot-noise density)
    t0 = time.time()
    Fd = finufft.nufft3d1(*th(xd), wd.astype(complex), (n,) * 3, isign=-1, **opts)
    Fr = finufft.nufft3d1(*th(xr), (alpha * wr).astype(complex), (n,) * 3, isign=-1, **opts)
    F = Fd - Fr
    orders = {0: ([()], np.array([1.0])), 2: tensor_index(2)}
    tens = [t for o in sorted(orders) for t in orders[o][0]]

    def moments(x, w):
        xh = x / np.linalg.norm(x, axis=1)[:, None]
        cs = np.empty((len(tens), len(w)), complex)
        for n_, t in enumerate(tens):
            v = w.astype(float).copy()
            for i in t:
                v *= xh[:, i]
            cs[n_] = v
        return finufft.nufft3d1(*th(x), cs, (n,) * 3, isign=-1, **opts).reshape(len(tens), -1)

    def poles(Fm):
        F0 = Fm[0]
        Q2 = np.zeros(F0.shape, complex)
        for pos, ((i, j), mu) in enumerate(zip(*orders[2]), start=1):
            Q2 += mu * khat[i] * khat[j] * Fm[pos]
        return {0: F0, 2: 1.5 * Q2 - 0.5 * F0}

    def shell(v):
        return np.bincount(io, v, nb + 1)[:nb] / nsh
    M = poles(moments(xd, wd) - alpha * moments(xr, wr))                       # F_l (once per catalogue)
    F0 = M[0]
    sn = float(np.sum(wd ** 2) + alpha ** 2 * np.sum(wr ** 2))
    if args.synthetic is not None and args.lnpoisson is None:   # data = randoms (same points): the self pairs are those of ONE weighted point set
        wq = wd - alpha * wr                      #   F = sum_r (w_d - alpha w_r) e,  self term sum_r wq^2 (and wq^2 delta_L for C)
        sn = float(np.sum(wq ** 2))
    nL = len(args.k_L)
    out = dict(k=0.5 * (edges[1:] + edges[:-1]), k_L=np.array(args.k_L), alpha=alpha, n_gal=len(wd),
               sig2=np.zeros(nL), c_sn=np.zeros(nL), dL_at_gal=np.zeros(nL))
    for ell in (0, 2):
        out[f'P{ell}'] = (2 * ell + 1) * shell((F0 * np.conj(M[ell])).real) - (sn if ell == 0 else 0.0)
        for key in ('C', 'CS', 'C2_'):
            out[f'{key}{ell}'] = np.zeros((nL, nb))
    log(f'  mock {mock}: {len(wd)} galaxies, alpha {alpha:.4f}; F_l done ({time.time() - t0:.0f} s)')
    for iL, k_L in enumerate(args.k_L):
        t1 = time.time()
        # long-mode overdensity: smooth F and m with the same filter W_L(k) = exp(-(k/k_L)^2/2), delta_L = F_L / m_L on the grid where m_L > 0
        WL = np.exp(-0.5 * (km / k_L) ** 2) if args.filter == 'gauss' else (km < k_L).astype(float)
        F_L = sfft.ifftn(F * WL, workers=NTH).real
        m_L = sfft.ifftn(Fr * WL, workers=NTH).real
        thr = 0.05 * m_L.max()
        dL = np.where(m_L > thr, F_L / np.maximum(m_L, thr), 0.0)
        sig2 = float(np.sum(m_L * dL ** 2) / np.sum(m_L))                      # m-weighted variance of the long mode
        del WL, F_L, m_L
        # the modulated fields: delta_L(x) F(x) and delta_L(x)^2 F(x), as particle sums (exact, no aliasing): evaluate dL at the particles
        dLk = sfft.fftn(dL, workers=NTH).astype(complex) / n ** 3
        del dL
        dLd = finufft.nufft3d2(*th(xd), dLk, isign=1, **opts).real
        dLr = finufft.nufft3d2(*th(xr), dLk, isign=1, **opts).real
        del dLk
        ML = poles(moments(xd, wd * dLd) - alpha * moments(xr, wr * dLr))         # FT[delta_L F]_l
        ML2 = poles(moments(xd, wd * dLd ** 2) - alpha * moments(xr, wr * dLr ** 2))
        # discreteness of the three-point estimator: the same galaxy in F^* and in the F of FT[delta_L F] gives the self term
        # sum_g w_g^2 delta_L(x_g) (+ alpha^2 sum_r w_r^2 delta_L(x_r)), one scalar; <delta_L at galaxies> = <delta_g delta_L> = int P W_L (> sigma_L^2 = int P W_L^2)
        c_sn = float(np.sum(wd ** 2 * dLd) + alpha ** 2 * np.sum(wr ** 2 * dLr))
        if args.synthetic is not None and args.lnpoisson is None:
            c_sn = float(np.sum(wq ** 2 * dLd))
        out['sig2'][iL], out['c_sn'][iL] = sig2, c_sn
        out['dL_at_gal'][iL] = float(np.sum(wd * dLd) / np.sum(wd))
        for ell in (0, 2):
            out[f'C{ell}'][iL] = (2 * ell + 1) * shell((np.conj(F0) * ML[ell]).real)                 # raw cross-spectrum (SN part reported separately)
            out[f'CS{ell}'][iL] = c_sn if ell == 0 else 0.0                                          # self term: monopole only, k-independent
            out[f'C2_{ell}'][iL] = (2 * ell + 1) * shell((np.conj(F0) * ML2[ell]).real)
        log(f'    k_L = {k_L:g}: sigma_L = {np.sqrt(sig2) * 100:.3f} %, <delta_L at gal> = {out["dL_at_gal"][iL] * 1e4:.2f}e-4 ({time.time() - t1:.0f} s)')
        del dLd, dLr, ML, ML2
    log(f'  responses done ({time.time() - t0:.0f} s)')
    return out, grid_state


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--mock', type=int, default=173)
    ap.add_argument('--mocks', type=int, nargs='+', default=None, help='average the responses over these mock ids')
    ap.add_argument('--data', action='store_true')
    ap.add_argument('--data-dir', default='/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2')
    ap.add_argument('--random-files', type=int, default=4)
    ap.add_argument('--k-L', type=float, nargs='+', default=[0.01], help='Gaussian low-pass(es) of the long mode [h/Mpc]; several values loop inside each catalogue (F_l computed once)')
    ap.add_argument('--kmax', type=float, default=0.3)
    ap.add_argument('--filter', default='gauss', choices=['gauss', 'tophat'], help='long-mode filter W_L(k)')
    ap.add_argument('--pad', type=float, default=1.1)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--synthetic', type=float, nargs=3, default=None, metavar=('R_M', 'G', 'SEED'),
                    help='known-response field on the randoms instead of the data (validation); G < 0: lognormal with sigma_G = |G| (R = 2 at all q)')
    ap.add_argument('--lnpoisson', type=float, nargs=4, default=None, metavar=('A_L', 'A_S', 'P0', 'SEED'),
                    help='Poisson-sampled lognormal catalogue with distinct randoms (validation, R = 2)')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    od = args.out_dir or os.path.expanduser(f'~/thecov/responses')
    os.makedirs(od, exist_ok=True)
    mocks = [0] if args.data else (args.mocks or [args.mock])
    grid_state, runs = None, []
    for m in mocks:
        try:
            o, grid_state = one_catalogue(args, m, grid_state)
            runs.append(o)
        except FileNotFoundError as e:
            log(f'  mock {m}: skipped ({e})')
    k = runs[0]['k']
    tag = ('data' if args.data else 'mocks' + '-'.join(str(m) for m in mocks)) + (f'_synthR{args.synthetic[0]:g}g{args.synthetic[1]:g}' if args.synthetic else '') + (f'_lnpoisson{args.lnpoisson[0]:g}_{args.lnpoisson[1]:g}' if args.lnpoisson else '') + (f'_{args.filter}' if args.filter != 'gauss' else '')
    kLtag = f'{args.k_L[0]:g}' if len(args.k_L) == 1 else f'{min(args.k_L):g}-{max(args.k_L):g}x{len(args.k_L)}'
    fn = os.path.join(od, f'response_{args.bin}_{args.region}_{tag}_kL{kLtag}.npz')
    agg = {key: np.array([o[key] for o in runs]) for key in runs[0] if key not in ('k', 'k_L')}
    np.savez(fn, k=k, k_L=np.array(args.k_L), mocks=np.array(mocks), **agg)
    print(f'{args.bin} {args.region} [{tag}]: {len(runs)} catalogues; bands {KR}')
    print('  dlnP_l / d delta_L = (C - SN) / (P <delta_g delta_L>), <delta_g delta_L> = int P W (the mean of delta_L at the galaxies; the "SN" self-pair term '
          'sum_g w^2 delta_L(x_g) is removed); error = scatter over catalogues / sqrt(N)')
    for iL, k_L in enumerate(args.k_L):
        print(f'  k_L = {k_L:g}: sigma_L = {np.sqrt(agg["sig2"][:, iL]).mean() * 100:.3f} %, <delta_L at gal> / sigma_L^2 = {(agg["dL_at_gal"][:, iL] / agg["sig2"][:, iL]).mean():.2f}')
        for ell in (0, 2):
            P, C, CS = agg[f'P{ell}'], agg[f'C{ell}'][:, iL], agg[f'CS{ell}'][:, iL]
            s2 = agg['dL_at_gal'][:, iL][:, None]
            resp = ((C - CS) / (P * s2)).mean(0)
            err = ((C - CS) / (P * s2)).std(0) / np.sqrt(max(len(runs), 1)) if len(runs) > 1 else np.full(len(k), np.nan)
            snp = (CS / (P * s2)).mean(0)
            row = ' | '.join(f'{np.mean(resp[(k >= lo) & (k < hi)]):+.3f} +- {np.nanmean(err[(k >= lo) & (k < hi)]) / np.sqrt(np.sum((k >= lo) & (k < hi)) / 2.5):.3f} (SN {np.mean(snp[(k >= lo) & (k < hi)]):+.3f})'
                            for lo, hi in KR)
            print(f'    l = {ell}: {row}')
    print('  tree level per unit GALAXY long mode (review 1.2 / g0 = b1 + f/3 = 2.45, LRG1): P0 1.2-1.6 (global-mean), P2 1.8-2.7;'
          ' net (after local average) per unit galaxy mode = this - 2. Bands with k < ~3 k_L are outside the squeezed limit.')
    log(f'saved {fn}')


if __name__ == '__main__':
    main()
