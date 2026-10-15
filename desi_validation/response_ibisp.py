"""Long-mode responses of P_l from the catalogue itself (integrated bispectrum in Fourier space, no sub-volumes),
for the SSC and squeezed-T0 terms.

    python -u -m desi_validation.response_ibisp --bin LRG1 --region NGC --mock 173 --mocks 173 174 175 176
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
    del rc
    alpha = float(np.sum(wd) / np.sum(wr))
    if grid_state is None:
        lo, hi = np.minimum(xd.min(0), xr.min(0)), np.maximum(xd.max(0), xr.max(0))
        L = args.pad * float(np.max(hi - lo))
        lo = 0.5 * (lo + hi) - 0.5 * L
        n = sfft.next_fast_len(2 * int(np.ceil(args.kmax / (2 * np.pi / L))) + 4)
        grid_state = dict(lo=lo, L=L, n=n)
        log(f'{b} {r}: box {L:.0f}, grid {n}^3; k_L = {args.k_L}')
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
    # long-mode overdensity: smooth F and m with the same filter W_L(k) = exp(-(k/k_L)^2/2), delta_L = F_L / m_L on the grid where m_L > 0
    WL = np.exp(-0.5 * (km / args.k_L) ** 2)
    F_L = sfft.ifftn(F * WL, workers=NTH).real
    m_L = sfft.ifftn(Fr * WL, workers=NTH).real
    thr = 0.05 * m_L.max()
    dL = np.where(m_L > thr, F_L / np.maximum(m_L, thr), 0.0)
    sig2 = float(np.sum(m_L * dL ** 2) / np.sum(m_L))                      # m-weighted variance of the long mode
    log(f'  mock {mock}: {len(wd)} galaxies, alpha {alpha:.4f}; sigma_L = {np.sqrt(sig2) * 100:.3f} % ({time.time() - t0:.0f} s)')
    # the modulated fields: delta_L(x) F(x) and delta_L(x)^2 F(x), as particle sums (exact, no aliasing): evaluate dL at the particles
    dLd = finufft.nufft3d2(*th(xd), sfft.fftn(dL, workers=NTH).astype(complex) / n ** 3, isign=1, **opts).real
    dLr = finufft.nufft3d2(*th(xr), sfft.fftn(dL, workers=NTH).astype(complex) / n ** 3, isign=1, **opts).real
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
    M = poles(moments(xd, wd) - alpha * moments(xr, wr))                       # F_l
    ML = poles(moments(xd, wd * dLd) - alpha * moments(xr, wr * dLr))         # FT[delta_L F]_l
    ML2 = poles(moments(xd, wd * dLd ** 2) - alpha * moments(xr, wr * dLr ** 2))
    MS = poles(moments(xd, wd ** 2 * dLd) - alpha ** 2 * moments(xr, wr ** 2 * dLr))   # shot-noise collapsed term
    F0 = M[0]
    out = dict(k=0.5 * (edges[1:] + edges[:-1]), sig2=sig2, alpha=alpha, n_gal=len(wd))
    sn = float(np.sum(wd ** 2) + alpha ** 2 * np.sum(wr ** 2))
    for ell in (0, 2):
        P = (2 * ell + 1) * shell((F0 * np.conj(M[ell])).real) - (sn if ell == 0 else 0.0)
        C = (2 * ell + 1) * shell((np.conj(F0) * ML[ell]).real)                 # raw cross-spectrum (SN part reported separately)
        CS = (2 * ell + 1) * shell((np.conj(F0) * MS[ell]).real)
        C2 = (2 * ell + 1) * shell((np.conj(F0) * ML2[ell]).real)
        out[f'P{ell}'], out[f'C{ell}'], out[f'CS{ell}'], out[f'C2_{ell}'] = P, C, CS, C2
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
    ap.add_argument('--k-L', type=float, default=0.01, help='Gaussian low-pass of the long mode [h/Mpc]')
    ap.add_argument('--kmax', type=float, default=0.3)
    ap.add_argument('--pad', type=float, default=1.1)
    ap.add_argument('--eps', type=float, default=1e-4)
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
    tag = 'data' if args.data else 'mocks' + '-'.join(str(m) for m in mocks)
    fn = os.path.join(od, f'response_{args.bin}_{args.region}_{tag}_kL{args.k_L:g}.npz')
    agg = {key: np.array([o[key] for o in runs]) for key in runs[0]}
    np.savez(fn, k=k, mocks=np.array(mocks), **agg)
    print(f'{args.bin} {args.region} [{tag}]: {len(runs)} catalogues, sigma_L = {np.sqrt(agg["sig2"]).mean() * 100:.3f} % (k_L {args.k_L}); bands {KR}')
    print('  dlnP_l / d delta_L  (C_l / (P_l sigma_L^2); "SN" = shot-noise collapsed part, already removed from the first number)')
    for ell in (0, 2):
        P, C, CS, C2 = agg[f'P{ell}'], agg[f'C{ell}'], agg[f'CS{ell}'], agg[f'C2_{ell}']
        s2 = agg['sig2'][:, None]
        resp = ((C - CS) / (P * s2)).mean(0)
        err = ((C - CS) / (P * s2)).std(0) / np.sqrt(max(len(runs), 1)) if len(runs) > 1 else np.full(len(k), np.nan)
        snp = (CS / (P * s2)).mean(0)
        row = ' | '.join(f'{np.mean(resp[(k >= lo) & (k < hi)]):+.3f} +- {np.nanmean(err[(k >= lo) & (k < hi)]) / np.sqrt(np.sum((k >= lo) & (k < hi)) / 2.5):.3f} (SN {np.mean(snp[(k >= lo) & (k < hi)]):+.3f})'
                        for lo, hi in KR)
        print(f'  l = {ell}: {row}')
        if ell == 0:
            j = (C2 / (P * 3 * s2 ** 2)).mean(0)                      # <dL^4> = 3 sig^4 for Gaussian long modes: leading second-order response / 2
            print('  second order (C2_0 / (3 P sigma_L^4), ~ d^2 lnP / d delta_L^2 / 2):', ' | '.join(f'{np.mean(j[(k >= lo) & (k < hi)]):+.2f}' for lo, hi in KR))
    print('  tree level (HANDOFF 9.2 / review 1.2, LRG1): P0 ~ -0.25 .. -0.3 net (local-mean), R_0^(0)/P_0 = 2.9-4.0 global; P2 +0.5 .. +2.2;'
          ' mocks vs delta_shell: P0 -0.06 .. -0.11, P2 +0.8 .. +2.5')
    log(f'saved {fn}')


if __name__ == '__main__':
    main()
