"""End-to-end test of the survey-mean (local-average) response bookkeeping with a KNOWN answer.

    python -u -m desi_validation.survey_mean_synthetic --bin LRG1 --region NGC --seeds 8 --deltas -0.06 -0.03 0 0.03 0.06

Catalogue: the Poisson-sampled lognormal of response_ibisp --lnpoisson (galaxies drawn from one half of the DESI randoms
with p = p0 (1 + delta_u) (1 + delta_g), the other half are the randoms), with an explicit, exactly known survey-mean mode
delta_u multiplying the sampling probability. For a lognormal field a uniform multiplier is a long mode of infinite
wavelength: the local power is (1 + delta_u)^2 P_s, i.e. R_global = 2 per unit galaxy mode, by construction.
For each seed the same field and the same uniform draws are used for every delta_u (common random numbers), so the slope
d ln P_hat / d delta_u is measured almost free of cosmic variance; the scatter of the slope across seeds is the error.

P_hat_l is measured as the DESI pipeline does, with three normalisation conventions:
    matched  : NX(z) re-estimated from the realised data n(z) (randoms' NX follows the data, as the holi pipeline does,
               HANDOFF 9.3), per-realisation FKP weights 1 / (1 + NX P_fkp), norm = alpha sum_r w_r^2 NX(z_r)
    matched0 : NX follows the data, no FKP weights
    fixed    : NX(z) and FKP weights frozen to the reference n(z) of the delta_u = 0 realisation of seed 0
Expected net responses d ln P_hat_0 / d delta_u:  matched 0 (= 2 - 2),  matched0 0,  fixed +1 (= 2 - 1: only alpha
follows the density). thecov's local-average term with data-matched randoms predicts R - 2; this is the check that the
subtraction, and the FKP cancellation, hold in the estimator as implemented. A second, independent estimate regresses
ln P_hat at delta_u = 0 across seeds on the realised survey mean of the box's own long modes (noisier).
Output: ~/thecov/responses/survey_mean_synthetic_<bin>_<region>.npz
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
from desi_validation.grf_randoms import NTH, log, tensor_index, p_input, kgrid  # noqa: E402
from desi_validation.response_ibisp import paths_for  # noqa: E402

KR = ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3))


def main():
    import finufft
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--mock', type=int, default=173, help='mock whose randoms (and n(z), FKP P0) are used')
    ap.add_argument('--random-files', type=int, default=4)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--deltas', type=float, nargs='+', default=[-0.06, -0.03, 0.0, 0.03, 0.06])
    ap.add_argument('--A-L', type=float, default=3.0)
    ap.add_argument('--A-S', type=float, default=0.4)
    ap.add_argument('--p0', type=float, default=0.08, help='sampling probability of the parent randoms at delta = 0')
    ap.add_argument('--k-L', type=float, default=0.01, help='split between the boosted long modes and the short modes')
    ap.add_argument('--P-fkp', type=float, default=1e4)
    ap.add_argument('--kmax', type=float, default=0.3)
    ap.add_argument('--pad', type=float, default=1.1)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--n-zbins', type=int, default=20)
    ap.add_argument('--out-dir', default=os.path.expanduser('~/thecov/responses'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    b, r = args.bin, args.region
    cfg = pl.Config()
    args.data = False
    paths = paths_for(args, args.mock)
    rc = dc.load_region(paths, b, r, n_random_files=args.random_files)
    tracer, zr = dc.TRACER_SPECS[b]
    zd_real = np.asarray(rc.data['Z'], float)
    if 'NX' in rc.data:
        nx_real = np.asarray(rc.data['NX'], float)
    else:                                                      # NX from the FKP weight: w = 1 / (1 + NX P_fkp)
        nx_real = (1 / np.asarray(rc.data['WEIGHT_FKP'], float) - 1) / args.P_fkp
    try:
        wr_all = dc.total_weight(rc.randoms, 'default')
    except Exception:  # noqa: BLE001
        wr_all = np.ones(len(rc.randoms['Z']))
    zr_all = np.asarray(rc.randoms['Z'], float)
    xr_all = dc.sky_to_cartesian(rc.randoms['RA'], rc.randoms['DEC'], zr_all)
    # reference n(z) in absolute units from the real data's NX, per z bin; the synthetic n(z) = NX_real x N_syn(z) / N_real(z)
    zedges = np.linspace(zr[0], zr[1], args.n_zbins + 1)
    iz_real = np.clip(np.digitize(zd_real, zedges) - 1, 0, args.n_zbins - 1)
    nx_bin = np.bincount(iz_real, nx_real, args.n_zbins) / np.maximum(np.bincount(iz_real, minlength=args.n_zbins), 1)
    n_real = np.bincount(iz_real, minlength=args.n_zbins).astype(float)
    s1 = dc.read_spectra(paths.spectra_fns(tracer, zr, r), kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=cfg.ells)
    P0shape = s1['vectors'][:, :len(s1['k'])].mean(0) * 1.22
    del rc
    # parents / estimator randoms (fixed split)
    rng0 = np.random.default_rng(12345)
    perm = rng0.permutation(len(wr_all))
    par, est = perm[: len(wr_all) // 2], perm[len(wr_all) // 2:]
    xr, wr_base, zr_ = xr_all[est], wr_all[est], zr_all[est]
    xp, wp, zp = xr_all[par], wr_all[par], zr_all[par]
    iz_r = np.clip(np.digitize(zr_, zedges) - 1, 0, args.n_zbins - 1)
    # lognormal generator (as response_ibisp --lnpoisson)
    lo_, hi_ = xr_all.min(0), xr_all.max(0)
    Lg = 1.2 * float(np.max(hi_ - lo_))
    lo_ = 0.5 * (lo_ + hi_) - 0.5 * Lg
    ng = sfft.next_fast_len(2 * int(np.ceil(0.32 / (2 * np.pi / Lg))) + 4)
    kg = kgrid(ng, Lg)
    Pg = p_input(s1['k'], P0shape, kg, 0.32)
    WLg = np.exp(-0.5 * (kg / args.k_L) ** 2)
    thg = [np.ascontiguousarray(2 * np.pi * (xp[:, i] - lo_[i]) / Lg) for i in range(3)]
    opts0 = dict(eps=args.eps, nthreads=NTH, modeord=1)
    # estimator grid
    lo, hi = xr_all.min(0), xr_all.max(0)
    L = args.pad * float(np.max(hi - lo))
    lo = 0.5 * (lo + hi) - 0.5 * L
    n = sfft.next_fast_len(2 * int(np.ceil(args.kmax / (2 * np.pi / L))) + 4)
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
    del KO, km
    orders = {0: ([()], np.array([1.0])), 2: tensor_index(2)}
    tens = [t for o in sorted(orders) for t in orders[o][0]]
    log(f'{b} {r}: {len(par)} parents, {len(est)} randoms; generator box {Lg:.0f} grid {ng}^3; estimator box {L:.0f} grid {n}^3')

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

    mom_cache = {}

    def measure(xd, wd, wr, key=None):
        """P_hat_0, P_hat_2 (numerator only: shell averages minus shot noise); the randoms' moments are cached by key
        (the randoms' weights of 'matched0' and 'fixed' never change)."""
        alpha = float(np.sum(wd) / np.sum(wr))
        if key is not None and key in mom_cache:
            Mr = mom_cache[key]
        else:
            Mr = moments(xr, wr)
            if key is not None:
                mom_cache[key] = Mr
        M = poles(moments(xd, wd) - alpha * Mr)
        sn = float(np.sum(wd ** 2) + alpha ** 2 * np.sum(wr ** 2))
        num0 = shell((M[0] * np.conj(M[0])).real) - sn
        num2 = 5 * shell((M[0] * np.conj(M[2])).real)
        return alpha, num0, num2

    conv = ('matched', 'matched0', 'fixed')
    res = {c: np.zeros((args.seeds, len(args.deltas), 2, nb)) for c in conv}
    norms = {c: np.zeros((args.seeds, len(args.deltas))) for c in conv}
    d_true = np.zeros((args.seeds, len(args.deltas)))
    d_real = np.zeros((args.seeds, len(args.deltas)))
    ngal = np.zeros((args.seeds, len(args.deltas)))
    nx_ref = fkp_ref = None
    for s in range(args.seeds):
        t0 = time.time()
        rng = np.random.default_rng(1000 + s)
        Wm = sfft.fftn(rng.standard_normal((ng,) * 3), workers=NTH) * (np.sqrt(Pg / (Lg / ng) ** 3) / ng ** 3)
        G = args.A_L * finufft.nufft3d2(*thg, Wm * WLg, isign=1, **opts0).real + args.A_S * finufft.nufft3d2(*thg, Wm * (1 - WLg), isign=1, **opts0).real
        del Wm
        dg = np.exp(G - 0.5 * G.var()) - 1.0
        u = rng.random(len(par))
        mean_dg = float(np.sum(wp * dg) / np.sum(wp))                      # realised survey mean of the box's long modes
        log(f'seed {s}: sigma_G = {G.std():.3f}, survey mean of (1 + delta_g) - 1 = {mean_dg * 100:+.3f} % ({time.time() - t0:.0f} s)')
        for j, du in enumerate(args.deltas):
            t1 = time.time()
            prob = np.clip(args.p0 * (1 + du) * (1 + dg), 0, 1)
            sel = u < prob
            xd, zd = xp[sel], zp[sel]
            wd0 = wp[sel]
            d_true[s, j] = (1 + du) * (1 + mean_dg) - 1
            d_real[s, j] = float(np.sum(prob * wp) / (args.p0 * np.sum(wp))) - 1
            ngal[s, j] = sel.sum()
            # realised n(z) of this catalogue, in absolute units
            iz_d = np.clip(np.digitize(zd, zedges) - 1, 0, args.n_zbins - 1)
            n_syn = np.bincount(iz_d, minlength=args.n_zbins).astype(float)
            nx_syn = nx_bin * n_syn / np.maximum(n_real, 1)
            fkp_syn = 1 / (1 + nx_syn * args.P_fkp)
            if nx_ref is None:
                nx_ref, fkp_ref = nx_syn.copy(), fkp_syn.copy()
                log(f'  reference n(z): n P_fkp at the peak = {nx_ref.max() * args.P_fkp:.2f}, <w_fkp> = {fkp_ref.mean():.3f} '
                    f'({sel.sum()} galaxies, clipped fraction {np.mean(prob >= 1) * 100:.3f} %)')
            for c, (nx_c, fkp_c) in {'matched': (nx_syn, fkp_syn), 'matched0': (nx_syn, np.ones_like(fkp_syn)),
                                     'fixed': (nx_ref, fkp_ref)}.items():
                wd, wr = wd0 * fkp_c[iz_d], wr_base * fkp_c[iz_r]
                alpha, num0, num2 = measure(xd, wd, wr, key=None if c == 'matched' else c)
                norm = alpha * float(np.sum(wr ** 2 * nx_c[iz_r]))
                res[c][s, j, 0], res[c][s, j, 1], norms[c][s, j] = num0 / norm, num2 / norm, norm
            log(f'  delta_u = {du:+.2f}: {sel.sum()} galaxies, delta_true = {d_true[s, j] * 100:+.3f} %; '
                f'P0(0.1-0.2) matched {np.mean(res["matched"][s, j, 0][(edges[:-1] >= 0.1) & (edges[:-1] < 0.2)]):.0f} ({time.time() - t1:.0f} s)')
    k = 0.5 * (edges[1:] + edges[:-1])
    du = np.array(args.deltas)
    duc = du - du.mean()
    print(f'\n{b} {r}: {args.seeds} seeds x {len(du)} delta_u; R_global = 2 by construction. Expected d ln P_hat_0 / d delta_u: '
          f'matched 0, matched0 0, fixed +1. Bands {KR}')
    out = dict(k=k, deltas=du, d_true=d_true, d_real=d_real, ngal=ngal)
    for c in conv:
        lnP0 = np.log(res[c][:, :, 0, :])                                  # (seed, delta, k)
        p2 = res[c][:, :, 1, :] / res[c][:, 0, 0, :].mean(0)              # P2 / <P0> at delta = 0
        sl0 = np.einsum('j,sjk->sk', duc, lnP0) / (duc @ duc)             # per-seed slope (common random numbers)
        sl2 = np.einsum('j,sjk->sk', duc, p2) / (duc @ duc)
        sln = np.einsum('j,sj->s', duc, np.log(norms[c])) / (duc @ duc)
        m0, e0 = sl0.mean(0), sl0.std(0, ddof=1) / np.sqrt(args.seeds)
        m2, e2 = sl2.mean(0), sl2.std(0, ddof=1) / np.sqrt(args.seeds)
        row = lambda m, e: ' | '.join(f'{np.mean(m[(k >= lo) & (k < hi)]):+.3f} +- {np.sqrt(np.mean(e[(k >= lo) & (k < hi)] ** 2)) / np.sqrt(np.sum((k >= lo) & (k < hi)) / 2.5):.3f}' for lo, hi in KR)
        print(f'  {c:9s}: d ln norm / d delta_u = {sln.mean():+.3f} +- {sln.std(ddof=1) / np.sqrt(args.seeds):.3f}')
        print(f'             d ln P0 / d delta_u      : {row(m0, e0)}')
        print(f'             d (P2 / <P0>) / d delta_u: {row(m2, e2)}')
        out[f'{c}/P'] = res[c]; out[f'{c}/norm'] = norms[c]; out[f'{c}/slope0'] = sl0; out[f'{c}/slope2'] = sl2
    # independent estimate: across seeds at delta_u = 0, regress on the realised survey mean of the box's long modes
    j0 = int(np.argmin(np.abs(du)))
    x = d_true[:, j0] - d_true[:, j0].mean()
    if args.seeds > 2 and x.std() > 0:
        print(f'  across seeds at delta_u = {du[j0]:+.2f}: sigma(realised survey mean) = {d_true[:, j0].std(ddof=1) * 100:.3f} %')
        for c in conv:
            lnP0 = np.log(res[c][:, j0, 0, :])
            bb = x @ (lnP0 - lnP0.mean(0)) / (x @ x)
            print(f'    {c:9s}: d ln P0 / d delta_survey = ' + ' | '.join(f'{np.mean(bb[(k >= lo) & (k < hi)]):+.2f}' for lo, hi in KR) + '  (no error bar: one regression over the seeds)')
    fn = os.path.join(args.out_dir, f'survey_mean_synthetic_{b}_{r}.npz')
    np.savez(fn, **out)
    log(f'saved {fn}')


if __name__ == '__main__':
    main()
