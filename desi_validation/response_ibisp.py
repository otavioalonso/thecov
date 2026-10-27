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
                f'sigma(delta_s) = {ds_r.std():.3f}; expected dlnP0/d delta_gL = {2 * R_m / g_b:.3f} (modulation only at q < k_L), dlnP2 = 0; '
                f'R_2 = {2 * R_m ** 2 / g_b ** 2:.3f} per unit galaxy soft power')
        if args.synthetic_poisson is not None:
            # Poisson sampling with DISTINCT randoms: galaxies drawn from one half of the randoms with p = p0 (1 + delta_g),
            # the other half are the randoms -> realistic shot noise and bookkeeping (as --lnpoisson)
            p0 = args.synthetic_poisson
            perm = rng.permutation(len(wr)); par, est = perm[: len(wr) // 2], perm[len(wr) // 2:]
            sel = par[rng.random(len(par)) < np.clip(p0 * (1 + dg[par]), 0, 1)]
            xd, wd = xr[sel], wr[sel]; xr, wr = xr[est], wr[est]
            log(f'  Poisson-sampled synthetic catalogue: {len(sel)} galaxies from {len(par)} parents (p0 = {p0}), {len(est)} randoms')
            del perm, par, est, sel
        else:
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
    if args.synthetic is not None and args.lnpoisson is None and args.synthetic_poisson is None:   # data = randoms (same points): the self pairs are those of ONE weighted point set
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
        if args.synthetic is not None and args.lnpoisson is None and args.synthetic_poisson is None:
            c_sn = float(np.sum(wq ** 2 * dLd))
        out['sig2'][iL], out['c_sn'][iL] = sig2, c_sn
        out['dL_at_gal'][iL] = float(np.sum(wd * dLd) / np.sum(wd))
        for ell in (0, 2):
            out[f'C{ell}'][iL] = (2 * ell + 1) * shell((np.conj(F0) * ML[ell]).real)                 # raw cross-spectrum (SN part reported separately)
            out[f'CS{ell}'][iL] = c_sn if ell == 0 else 0.0                                          # self term: monopole only, k-independent
            out[f'C2_{ell}'][iL] = (2 * ell + 1) * shell((np.conj(F0) * ML2[ell]).real)
        log(f'    k_L = {k_L:g}: sigma_L = {np.sqrt(sig2) * 100:.3f} %, <delta_L at gal> = {out["dL_at_gal"][iL] * 1e4:.2f}e-4 ({time.time() - t1:.0f} s)')
        del dLd, dLr, ML, ML2
    if args.soft_band:
        # ---- second-order response from the integrated trispectrum (HANDOFF section 14) -----------------------------
        # Regressor: the local power of the soft band [k1, k2), S_q(x) = [m_w^2 delta_s^2]_lowpass(k_q) / [m_w^2]_lowpass(k_q),
        # delta_s = (F W_band) / m_w, m_w = m smoothed at k_w (fixed), minus its m^2-weighted mean and its m^2-weighted
        # projection on the long mode delta_q = (F W_q) / m_q (removes the first-order coupling R_1 <delta_q S_q>).
        # Fourier form: Q_l(k) = <F_l^*(k) FT[g F](k)>_shell = sum_q W_q(q) sum_p W_b(p) W_b(|q-p|) <F*(k) F(k+q) F(p) F(q-p)>.
        # Disconnected (Gaussian) part: sum_q g(q) <F*(k) F(k+q)> = P_l(k) int m^2 g = P_l(k) <g>_{m^2}, which is why the
        # mean (and the delta_q projection) must be m^2-weighted: with <g>_{m^2} = 0 it vanishes and only the connected part
        #     Q_l(k) - QS = R_2 P_l(k) <delta_s^2 g>_{m^2} / 2
        # remains (collapsed trispectrum T -> R_2 P(k) P(p) P(|q-p|); for a top-hat band the Gaussian covariance of
        # delta_s^2 with g is exactly twice the same mode sum, hence the 1/2). QS = sum_g w_g^2 g(x_g) (+ alpha^2 randoms)
        # is the self-pair term. R_2 is per unit galaxy soft-band power, in global coordinates (what the squeezed T0 block
        # needs: Cov(P_s, P_h) = R_2 P_h V_band Var_clust(P_s) / 2, V_band = (k2^3 - k1^3) / 6 pi^2). The band shot noise
        # enters <delta_s^2 g> but not the response: multiply R_2 by (1 + SN / P_s)^2 afterwards (sn, Ps_band_raw saved).
        # Discreteness terms removed: self pairs (QS), the galaxy's own contribution to S_q (h, both pair members: Qh) and to
        # delta_q (c, both members: beta Qc). Neglected: both pair members inside the soft legs (P/n^2-type, ~2 c_bb w^2/den
        # relative to <delta_s^2 g>: 1-5 per cent of R_2).
        bands = [(args.soft_band[i], args.soft_band[i + 1]) for i in range(0, len(args.soft_band), 2)]
        nB, nQ = len(bands), len(args.k_q)
        out['soft_band'] = np.array(bands); out['k_q'] = np.array(args.k_q); out['sn'] = sn; out['k_w'] = args.k_w
        for key in ('Q0', 'Q2', 'Qh0', 'Qh2', 'Qc0', 'Qc2'):
            out[key] = np.zeros((nB, nQ, nb))
        for key in ('QS', 'g_gal', 'g_m', 'g_m2', 'd2g_gal', 'd2g_m', 'd2g_m2', 'var_g_m2', 'd2_gal', 'd2_m2', 'beta', 'sig2_q', 'Ps_band_raw', 'QhS', 'h_gal', 'c_bb', 'QcS', 'c_gal'):
            out[key] = np.zeros((nB, nQ))
        wra = alpha * wr
        Ww = np.exp(-0.5 * (km / args.k_w) ** 2)
        m_w = sfft.ifftn(Fr * Ww, workers=NTH).real
        thr_w = 0.1 * m_w.max(); mask_w = m_w > thr_w; mw = np.maximum(m_w, thr_w)
        mw2k = sfft.fftn(np.where(mask_w, m_w, 0.0) ** 2, workers=NTH)                 # for the local normalisation
        mwk = sfft.fftn(np.where(mask_w, m_w, 0.0), workers=NTH).astype(complex) / n ** 3
        mw_r = finufft.nufft3d2(*th(xr), mwk, isign=1, **opts).real                   # m_w at the randoms: m^2 weights
        om_r = wra * np.maximum(mw_r, 0.0)
        del Ww, mwk, mw_r
        for iB, (k1, k2) in enumerate(bands):
            tb = time.time()
            Wb = ((km >= k1) & (km < k2)).astype(float)
            Fb = sfft.ifftn(F * Wb, workers=NTH).real                                 # m delta_s (+ window leakage)
            kb_r = sfft.ifftn(Wb, workers=NTH).real                                   # band kernel K_b(r) (unit-weight galaxy -> Fb)
            Fb_k = sfft.fftn(np.where(mask_w, Fb, 0.0), workers=NTH)
            del Wb
            ds2 = np.where(mask_w, (Fb / mw) ** 2, 0.0)                                # soft-band overdensity squared
            loc_k = sfft.fftn(np.where(mask_w, Fb, 0.0) ** 2, workers=NTH)             # m_w^2 delta_s^2
            inb = (out['k'] >= k1) & (out['k'] < k2)
            ps_raw = float(np.mean(out['P0'][inb])) if inb.any() else np.nan           # band monopole, raw units, shot noise removed
            ak = sfft.fftn(ds2, workers=NTH).astype(complex) / n ** 3
            ds2_d = finufft.nufft3d2(*th(xd), ak, isign=1, **opts).real
            ds2_r = finufft.nufft3d2(*th(xr), ak, isign=1, **opts).real
            del ak
            for iQ, k_q in enumerate(args.k_q):
                Wq = np.exp(-0.5 * (km / k_q) ** 2) if args.filter == 'gauss' else (km < k_q).astype(float)
                m_q = sfft.ifftn(Fr * Wq, workers=NTH).real
                thr = 0.05 * m_q.max(); mask = m_q > thr; mq = np.maximum(m_q, thr)
                dq = np.where(mask, sfft.ifftn(F * Wq, workers=NTH).real / mq, 0.0)   # long mode at k_q
                den = sfft.ifftn(mw2k * Wq, workers=NTH).real
                den = np.maximum(den, 0.05 * den.max())
                Sq = np.where(mask, sfft.ifftn(loc_k * Wq, workers=NTH).real / den, 0.0)
                sig2_q = float(np.sum(m_q * dq ** 2) / np.sum(m_q))
                # a galaxy's own contribution to the local soft power at its position (the B/n-type discreteness term of
                # the four-point estimator, shared galaxy between a hard leg and a soft leg): with kappa(r) = w_q(r) K_b(r),
                # h(x_g) = [2 w_g (Fb * kappa)(x_g) + w_g^2 c_bb] / den(x_g), c_bb = sum_r w_q(r) K_b(r)^2
                wq_r = sfft.ifftn(Wq, workers=NTH).real
                kap_k = sfft.fftn(wq_r * kb_r, workers=NTH)
                c_bb = float(np.sum(wq_r * kb_r ** 2))
                Hg = sfft.ifftn(Fb_k * kap_k, workers=NTH).real
                Vq_cell = float(np.mean(Wq))                                          # w_q(r = 0): a unit weight's own contribution to (F W_q)
                mq_k = sfft.fftn(mq, workers=NTH).astype(complex) / n ** 3
                mq_d = finufft.nufft3d2(*th(xd), mq_k, isign=1, **opts).real
                del Wq, m_q, mq, mask, wq_r, mq_k
                at = {}
                for nm, arr in (('Sq', Sq), ('dq', dq)):
                    ak = sfft.fftn(arr, workers=NTH).astype(complex) / n ** 3
                    at[nm + '_d'] = finufft.nufft3d2(*th(xd), ak, isign=1, **opts).real
                    at[nm + '_r'] = finufft.nufft3d2(*th(xr), ak, isign=1, **opts).real
                    del ak
                for nm, arr in (('H', Hg), ('den', den)):
                    ak = sfft.fftn(arr, workers=NTH).astype(complex) / n ** 3
                    at[nm + '_d'] = finufft.nufft3d2(*th(xd), ak, isign=1, **opts).real
                    del ak
                del Sq, dq, Hg, den
                h_d = (2 * wd * at['H_d'] + wd ** 2 * c_bb) / np.maximum(at['den_d'], 1e-30)
                c_d = wd * Vq_cell / np.maximum(mq_d, 1e-30)                           # own contribution to delta_q at the galaxy
                Sbar = float(np.sum(om_r * at['Sq_r']) / np.sum(om_r))
                dqbar = float(np.sum(om_r * at['dq_r']) / np.sum(om_r))
                beta = float(np.sum(om_r * (at['dq_r'] - dqbar) * (at['Sq_r'] - Sbar)) / np.sum(om_r * (at['dq_r'] - dqbar) ** 2))
                # galaxy side: own contributions to S_q (h_d) and to delta_q (c_d) removed; the pair partner's contributions
                # (galaxy a in F^*) are the same quantities on the F^* leg: Qh (to subtract) and Qc (times -beta, to add back)
                g_d = at['Sq_d'] - Sbar - beta * (at['dq_d'] - c_d - dqbar) - h_d
                g_r = at['Sq_r'] - Sbar - beta * (at['dq_r'] - dqbar)
                MQ = poles(moments(xd, wd * g_d) - alpha * moments(xr, wr * g_r))
                Mh = moments(xd, wd * h_d)[0]
                Mc = moments(xd, wd * c_d)[0]
                for ell in (0, 2):
                    out[f'Q{ell}'][iB, iQ] = (2 * ell + 1) * shell((np.conj(F0) * MQ[ell]).real)
                    out[f'Qh{ell}'][iB, iQ] = (2 * ell + 1) * shell((np.conj(Mh) * M[ell]).real)
                    out[f'Qc{ell}'][iB, iQ] = (2 * ell + 1) * shell((np.conj(Mc) * M[ell]).real)
                out['QS'][iB, iQ] = float(np.sum(wd ** 2 * g_d) + alpha ** 2 * np.sum(wr ** 2 * g_r))
                out['QhS'][iB, iQ] = float(np.sum(wd ** 2 * h_d)); out['QcS'][iB, iQ] = float(np.sum(wd ** 2 * c_d))
                out['h_gal'][iB, iQ] = float(np.sum(wd * h_d) / np.sum(wd)); out['c_bb'][iB, iQ] = c_bb
                out['c_gal'][iB, iQ] = float(np.sum(wd * c_d) / np.sum(wd))
                if args.synthetic is not None and args.lnpoisson is None and args.synthetic_poisson is None:
                    out['QS'][iB, iQ] = float(np.sum(wq ** 2 * g_d)); out['QhS'][iB, iQ] = float(np.sum(wq ** 2 * h_d)); out['QcS'][iB, iQ] = float(np.sum(wq ** 2 * c_d))
                out['g_gal'][iB, iQ] = float(np.sum(wd * g_d) / np.sum(wd))
                out['g_m'][iB, iQ] = float(np.sum(wra * g_r) / np.sum(wra))
                out['g_m2'][iB, iQ] = float(np.sum(om_r * g_r) / np.sum(om_r))
                out['d2g_gal'][iB, iQ] = float(np.sum(wd * ds2_d * g_d) / np.sum(wd))
                out['d2g_m'][iB, iQ] = float(np.sum(wra * ds2_r * g_r) / np.sum(wra))
                out['d2g_m2'][iB, iQ] = float(np.sum(om_r * ds2_r * g_r) / np.sum(om_r))
                out['var_g_m2'][iB, iQ] = float(np.sum(om_r * g_r ** 2) / np.sum(om_r))
                out['d2_gal'][iB, iQ] = float(np.sum(wd * ds2_d) / np.sum(wd))
                out['d2_m2'][iB, iQ] = float(np.sum(om_r * ds2_r) / np.sum(om_r))
                out['beta'][iB, iQ] = beta; out['sig2_q'][iB, iQ] = sig2_q; out['Ps_band_raw'][iB, iQ] = ps_raw
                qc = out['Q0'][iB, iQ] - out['QS'][iB, iQ] - (out['Qh0'][iB, iQ] - out['QhS'][iB, iQ]) + beta * (out['Qc0'][iB, iQ] - out['QcS'][iB, iQ])
                r2 = 2 * qc / out['P0'] / out['d2g_m2'][iB, iQ]
                hb = (out['Qh0'][iB, iQ] - out['QhS'][iB, iQ]) / out['P0'] / out['d2g_m2'][iB, iQ]
                log(f'    band [{k1:g}, {k2:g}) k_q = {k_q:g}: <delta_s^2>_m2 = {out["d2_m2"][iB, iQ]:.4f}, rms g = {np.sqrt(out["var_g_m2"][iB, iQ]):.4f}, '
                    f'<g>_gal = {out["g_gal"][iB, iQ]:.2e}, <h>_gal = {out["h_gal"][iB, iQ]:.2e}, beta = {beta:.3f}; R_2(P0) in bands {KR[2:]}: '
                    + ', '.join(f'{np.mean(r2[(out["k"] >= lo) & (out["k"] < hi)]):+.2f}' for lo, hi in KR[2:])
                    + '; shared-galaxy term (each side, R_2 units): ' + ', '.join(f'{np.mean(hb[(out["k"] >= lo) & (out["k"] < hi)]):+.2f}' for lo, hi in KR[2:])
                    + f'; long-mode self term beta*(Qc-QcS)/(P d2g) each side: {np.mean(beta * (out["Qc0"][iB, iQ] - out["QcS"][iB, iQ]) / out["P0"] / out["d2g_m2"][iB, iQ]):+.3f} ({time.time() - tb:.0f} s)')
                del at, g_d, g_r, MQ, Mh, Mc, h_d, c_d, mq_d, kap_k
            del ds2, loc_k, ds2_d, ds2_r, kb_r, Fb_k
        del m_w, mw, mask_w, mw2k, om_r
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
    ap.add_argument('--synthetic-poisson', type=float, default=None, metavar='P0',
                    help='with --synthetic: Poisson-sample the field from half of the randoms with p = P0 (1 + delta_g) (realistic shot noise)')
    ap.add_argument('--lnpoisson', type=float, nargs=4, default=None, metavar=('A_L', 'A_S', 'P0', 'SEED'),
                    help='Poisson-sampled lognormal catalogue with distinct randoms (validation, R = 2)')
    ap.add_argument('--soft-band', type=float, nargs='+', default=None, metavar='K',
                    help='second-order response: soft bands as k1 k2 [k1 k2 ...]; regress the hard power on the local band power')
    ap.add_argument('--k-q', type=float, nargs='+', default=[0.02], help='smoothing scale(s) of the local soft power (squeezedness q/k_s)')
    ap.add_argument('--k-w', type=float, default=0.05, help='Gaussian smoothing of the mean density used for the soft overdensity and the m^2 weights')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    if args.soft_band is not None and len(args.soft_band) % 2:
        ap.error('--soft-band takes pairs k1 k2')
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
    tag = ('data' if args.data else 'mocks' + '-'.join(str(m) for m in mocks)) + (f'_synthR{args.synthetic[0]:g}g{args.synthetic[1]:g}' if args.synthetic else '') + (f'_lnpoisson{args.lnpoisson[0]:g}_{args.lnpoisson[1]:g}' if args.lnpoisson else '') + (f'_poisson{args.synthetic_poisson:g}' if args.synthetic_poisson is not None else '') + (f'_{args.filter}' if args.filter != 'gauss' else '')
    tag += '_r2' if args.soft_band else ''
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
    if args.soft_band:
        print('  second-order response R_2 = 2 (Q_l - QS) / (P_l <delta_s^2 g>_m2) x (1 + SN/P_s)^2 per unit galaxy soft-band power '
              '(hard bands 0.1-0.2, 0.2-0.3; error = scatter over catalogues / sqrt(N))')
        for iB, (k1, k2) in enumerate(agg['soft_band'][0]):
            for iQ, k_q in enumerate(agg['k_q'][0]):
                snf = (1 + agg['sn'] / agg['Ps_band_raw'][:, iB, iQ]) ** 2
                line = f'    band [{k1:g}, {k2:g}) k_q = {k_q:g}: <delta_s^2>_m2 = {agg["d2_m2"][:, iB, iQ].mean():.4f}, rms g / <delta_s^2> = ' \
                       f'{(np.sqrt(agg["var_g_m2"][:, iB, iQ]) / agg["d2_m2"][:, iB, iQ]).mean():.3f}, (1+SN/P_s)^2 = {snf.mean():.2f}, ' \
                       f'<g>_gal / rms g = {(agg["g_gal"][:, iB, iQ] / np.sqrt(agg["var_g_m2"][:, iB, iQ])).mean():+.3f}'
                for ell in (0, 2):
                    bq = agg['beta'][:, iB, iQ][:, None]
                    P = agg[f'P{ell}']; Q = agg[f'Q{ell}'][:, iB, iQ] - agg[f'Qh{ell}'][:, iB, iQ] + bq * agg[f'Qc{ell}'][:, iB, iQ]
                    QS = (agg['QS'][:, iB, iQ] - agg['QhS'][:, iB, iQ] + bq[:, 0] * agg['QcS'][:, iB, iQ])[:, None] if ell == 0 else 0.0
                    r2 = 2 * (Q - QS) / P / agg['d2g_m2'][:, iB, iQ][:, None] * snf[:, None]
                    cells = []
                    for lo, hi in KR[2:]:
                        m_ = (k >= lo) & (k < hi); v = r2[:, m_].mean(1)
                        cells.append(f'{v.mean():+.2f} +- {v.std() / np.sqrt(len(v)):.2f}' if len(v) > 1 else f'{v.mean():+.2f}')
                    line += f' | l={ell}: ' + ', '.join(cells)
                print(line)
    log(f'saved {fn}')


if __name__ == '__main__':
    main()
