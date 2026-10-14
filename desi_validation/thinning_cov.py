"""Dilution test on a real mock catalogue without new spectra: the covariance of P_l over random thinnings of the
galaxies, against thecov's Gaussian difference. Isolates the discreteness (SN-NG) terms model-independently.

    python -u -m desi_validation.thinning_cov thin   --bin LRG1 --region NGC --mock 173 --fracs 0.5 0.25 --nthin 200
    python -u -m desi_validation.thinning_cov thecov --bin LRG1 --region NGC --mock 173 --frac 1
    python -u -m desi_validation.thinning_cov thecov --bin LRG1 --region NGC --mock 173 --frac 0.5   (and 0.25)
    python -u -m desi_validation.thinning_cov report --bin LRG1 --region NGC --mock 173

Identity (law of total covariance): for a catalogue thinned at random to a fraction p, with kept weights w / p (so
that alpha, norm and <P_hat> are unchanged),  E_mocks[ Cov_thin(P_hat_p) ] = Cov(P_hat_p) - Cov(P_hat_1).
Under dilution the Gaussian term changes as thecov predicts (shot noise S -> S_p), B/n scales as 1/p, P/n^2 as
1/p^2, SSC and T0 (clustering statistics of the parent) do not change. So
    Cov_thin(p) - [G_thecov(S_p) - G_thecov(S_1)] = disc(p) - disc(1)
is the discreteness covariance of the galaxies' own bispectrum and sampling, with no shape prior and no new spectra.
The thinning covariance of one catalogue has small cosmic variance (it is a sum over ~1e6 galaxies); a few mocks
average it. Estimator: exact (NUFFT) Yamamoto multipoles, first-point line of sight, realised shot noise subtracted,
the files' k bins and norm; the randoms' transform is computed once per catalogue.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402
from desi_validation.grf_randoms import NTH, T0, log, tensor_index  # noqa: E402

KR = ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3))


def setup_mock(args):
    paths = dc.Paths(kind='holi_v3', mock=args.mock)
    paths.catalog_names.update({'ELG_LOPnotqso': 'ELGnotqso', 'LRG+ELG_LOPnotqso': 'LRG+ELGnotqso'})
    paths.loader = 'auto'
    paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
    cfg = pl.Config()
    od = args.out_dir or os.path.expanduser(f'~/thecov/thinning/{args.bin}_{args.region}_mock{args.mock}')
    os.makedirs(od, exist_ok=True)
    return paths, cfg, od


def thin(args):
    import finufft
    paths, cfg, od = setup_mock(args)
    b, r = args.bin, args.region
    tracer, zr = dc.TRACER_SPECS[b]
    s1 = dc.read_spectra(paths.spectra_fns(tracer, zr, r), kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=cfg.ells)
    k, edges, nb = s1['k'], np.asarray(s1['k_edges']), len(s1['k'])
    norm = float(s1['norm'].mean())
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
    n_out = 2 * int(np.ceil(edges[-1] / kf)) + 4
    log(f'{b} {r} mock {args.mock}: {len(wd)} galaxies, {len(wr)} randoms ({args.random_files} files), alpha {alpha:.4f}, '
        f'box {L:.0f}, output grid {n_out}^3, norm {norm:.4g}, files num_shotnoise {s1["num_shotnoise"].mean():.4g}')
    ko = [2 * np.pi * np.fft.fftfreq(n_out, d=L / n_out)] * 3
    KO = np.meshgrid(*ko, indexing='ij')
    kom = np.sqrt(KO[0] ** 2 + KO[1] ** 2 + KO[2] ** 2)
    io = np.digitize(kom.ravel(), edges) - 1
    io[(io < 0) | (io >= nb)] = nb
    nsh = np.bincount(io, minlength=nb + 1)[:nb]
    with np.errstate(invalid='ignore', divide='ignore'):
        khat = [np.where(kom > 0, K_ / kom, 0.0).ravel() for K_ in KO]
    del KO, kom
    opts = dict(eps=args.eps, nthreads=NTH, modeord=1)
    ells = tuple(args.ells)
    orders = {0: ([()], np.array([1.0]))}
    if max(ells) >= 2:
        orders[2] = tensor_index(2)
    if max(ells) >= 4:
        orders[4] = tensor_index(4)
    tens = [t for o in sorted(orders) for t in orders[o][0]]

    def moments(x, w):
        th = [np.ascontiguousarray(2 * np.pi * (x[:, i] - lo[i]) / L) for i in range(3)]
        xh = x / np.linalg.norm(x, axis=1)[:, None]
        cs = np.empty((len(tens), len(w)), complex)
        for n_, t in enumerate(tens):
            v = w.astype(float).copy()
            for i in t:
                v *= xh[:, i]
            cs[n_] = v
        return finufft.nufft3d1(*th, cs, (n_out,) * 3, isign=-1, **opts).reshape(len(tens), -1)

    def multipoles(Fm, sn):
        F0 = Fm[0]
        pos, Fl_ = 1, {0: F0}
        if 2 in orders:
            Q2 = np.zeros(F0.shape, complex)
            for (i, j), mu in zip(*orders[2]):
                Q2 += mu * khat[i] * khat[j] * Fm[pos]; pos += 1
            Fl_[2] = 1.5 * Q2 - 0.5 * F0
        if 4 in orders:
            Q4 = np.zeros(F0.shape, complex)
            for (i, j, k_, l), mu in zip(*orders[4]):
                Q4 += mu * khat[i] * khat[j] * khat[k_] * khat[l] * Fm[pos]; pos += 1
            Fl_[4] = 35 / 8 * Q4 - 15 / 4 * Q2 + 3 / 8 * F0
        out = []
        for ell, Fl in Fl_.items():
            pm = (F0 * Fl.conj()).real
            p = (2 * ell + 1) * np.bincount(io, pm, nb + 1)[:nb] / nsh / norm
            out.append(p - (sn / norm if ell == 0 else 0.0))
        return np.concatenate(out)

    t0 = time.time()
    Fr = alpha * moments(xr, wr)
    snr = alpha ** 2 * float(np.sum(wr ** 2))
    log(f'  randoms transformed ({time.time() - t0:.0f} s)')
    Fd = moments(xd, wd)
    P1 = multipoles(Fd - Fr, float(np.sum(wd ** 2)) + snr)
    log(f'  full catalogue: P0 (first, middle, last) {P1[0]:.1f} {P1[nb // 2]:.1f} {P1[nb - 1]:.1f}; '
        f'files mean {s1["vectors"][:, 0].mean():.1f} {s1["vectors"][:, nb // 2].mean():.1f} {s1["vectors"][:, nb - 1].mean():.1f}')
    out = dict(k=k, k_edges=edges, ells=np.array(ells), chunk_note='P0 only unless --ells', norm=norm, alpha=alpha, n_gal=len(wd), sum_w2=float(np.sum(wd ** 2)),
               sn_randoms=snr, P1=P1, mean_files=s1['vectors'].mean(0), fracs=np.array(args.fracs), random_files=args.random_files)
    rng = np.random.default_rng(args.seed)
    for p in args.fracs:
        Pp, SNp = np.zeros((args.nthin, len(ells) * nb)), np.zeros(args.nthin)
        t0 = time.time()
        for j in range(args.nthin):
            keep = rng.random(len(wd)) < p
            SNp[j] = float(np.sum((wd[keep] / p) ** 2)) + snr
            Pp[j] = multipoles(moments(xd[keep], wd[keep] / p) - Fr, SNp[j])
            if j == 0 or (j + 1) % 20 == 0:
                log(f'  p = {p}: thinning {j + 1}/{args.nthin} ({time.time() - t0:.0f} s)')
                out[f'P_p{p:g}'], out[f'SN_p{p:g}'] = Pp[:j + 1], SNp[:j + 1]
                np.savez(os.path.join(od, 'thinning.npz'), **out)             # incremental: survives the time limit
            if time.time() - T0 > args.max_time:
                log(f'  time limit: p = {p} stopped after {j + 1} thinnings'); Pp, SNp = Pp[:j + 1], SNp[:j + 1]
                break
        out[f'P_p{p:g}'], out[f'SN_p{p:g}'] = Pp, SNp
        d = Pp.mean(0) - P1
        log(f'  p = {p}: mean(P_p) - P_1 over sigma_thin: max |{np.max(np.abs(d) / Pp.std(0)) * np.sqrt(args.nthin):.2f}| '
            f'(should be ~< 3); SN_p / norm {SNp.mean() / norm:.1f} (p = 1: {(out["sum_w2"] + snr) / norm:.1f})')
    fn = os.path.join(od, 'thinning.npz')
    np.savez(fn, **out)
    log(f'saved {fn}')


def thecov_stage(args):
    paths, cfg, od = setup_mock(args)
    b, r, p = args.bin, args.region, args.frac
    g = dict(np.load(os.path.join(od, 'thinning.npz')))
    tracer, zr = dc.TRACER_SPECS[b]
    ells = tuple(int(e) for e in g['ells'])
    cfg = dataclasses.replace(cfg, ells=ells)
    spec = dc.read_spectra(paths.spectra_fns(tracer, zr, r), kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=ells)
    sn = float(g['sum_w2'] / p + g['sn_randoms']) if p == 1 else float(g[f'SN_p{p:g}'].mean())
    spec = dict(spec, num_shotnoise=np.full(len(spec['norm']), sn))          # the files' norm; the thinned shot noise
    cfg = dataclasses.replace(cfg, n_randoms_max=int(args.n_randoms_max), target_near_pairs=args.target_near_pairs)
    rc = dc.load_region(paths, b, r, n_random_files=cfg.n_random_files)
    name = f'THIN_{b}_{r}_p{p:g}_kernel'
    tr, _ = dc.build_tracer(name, [rc], nw='random-density', n_randoms_max=cfg.n_randoms_max // 2,
                            surface_density_deg2=cfg.surface_density, verbose=False, shotnoise_target=sn)
    sm = pl.kernel_smoothing(paths, b, r, name, spec, cfg, log=log)
    del rc
    n_near, _ = dc.suggest_n_near(tr, target_pairs=cfg.target_near_pairs)
    log(f'{b} {r} p = {p}: thecov kernel Gaussian with SN / norm = {sn / g["norm"]:.1f}, n_near {n_near}')
    t0 = time.time()
    C, cov = dc.thecov_covariance(tr, spec, n_sub=cfg.n_sub_far, n_near=n_near, verbose=False,
                                  windows_file=os.path.join(od, f'windows_{name}.npz'), power_input='masked', smoothing=sm)
    log(f'  done in {time.time() - t0:.0f} s')
    res = dict(C_G=C, sn=sn, n_near=n_near)
    try:                                                                    # tree-level discreteness terms (proxy inputs)
        from thecov.discreteness import DiscretenessCovariance
        nb = len(spec['k'])
        P0 = spec['vectors'][:, :nb].mean(0) * g['norm'] / cov.I_randoms
        kk = np.concatenate([[spec['k_edges'][0] / 4], spec['k'], [spec['k_edges'][-1]]])
        D = DiscretenessCovariance(cov, (kk, np.concatenate([[P0[0]], P0, [P0[-1]]])), b1=1.0, f=0.0)
        comps, index = D.components(spec, ells=spec['ells'], return_index=True)
        res['disc_components'], res['disc_index'] = np.asarray(comps), np.asarray(index, dtype=object)
        log(f'  discreteness components (b1 = 1, f = 0, measured P): {list(index)}')
    except Exception as e:  # noqa: BLE001
        log(f'  discreteness terms not computed: {e!r}')
    fn = os.path.join(od, f'thecov_p{p:g}.npz')
    np.savez(fn, **res)
    log(f'saved {fn}')


def report(args):
    _, _, od = setup_mock(args)
    g = dict(np.load(os.path.join(od, 'thinning.npz')))
    k, nb = g['k'], len(g['k'])
    G1 = np.load(os.path.join(od, 'thecov_p1.npz'))['C_G']
    print(f'{args.bin} {args.region} mock {args.mock}: {g["n_gal"]} galaxies; thinning covariance vs thecov Gaussian difference')
    print('units: diagonal of G_thecov(p = 1); bands', KR)
    for p in g['fracs']:
        fn = os.path.join(od, f'thecov_p{p:g}.npz')
        if not os.path.exists(fn):
            print(f'  p = {p}: thecov missing'); continue
        Gp = np.load(fn)['C_G']
        Pp = g[f'P_p{p:g}']
        N = len(Pp)
        Ct = np.cov(Pp.T)
        dG = Gp - G1
        for ei, ell in enumerate(int(e) for e in g['ells']):
            sl = slice(ei * nb, (ei + 1) * nb)
            d1, dt, dg = np.diag(G1)[sl], np.diag(Ct)[sl], np.diag(dG)[sl]
            row = []
            for lo, hi in KR:
                m = (k >= lo) & (k < hi)
                thin, gauss = np.mean(dt[m] / d1[m]), np.mean(dg[m] / d1[m])
                err = thin * np.sqrt(2 / N) / np.sqrt(m.sum() / 2.5)
                row.append(f'{thin:.3f} - {gauss:.3f} = {thin - gauss:+.3f} +- {err:.3f}')
            print(f'  p = {p} l = {ell}: ' + ' | '.join(row))
        # off-diagonal: mean correlation-like excess over |i - j| = 1..3 in the P0 block at k > 0.15
        s0 = slice(0, nb)
        m = k > 0.15
        num = np.array([np.mean([(Ct[s0, s0] - dG[s0, s0])[i, i + d] / np.sqrt(G1[i, i] * G1[i + d, i + d])
                                 for i in np.flatnonzero(m) if i + d < nb]) for d in (0, 1, 2, 3)])
        print(f'  p = {p} P0 block, k > 0.15, (thin - dG)_ij / sqrt(G_ii G_jj) for |i-j| = 0..3: ' + ' '.join(f'{x:+.3f}' for x in num))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['thin', 'thecov', 'report'])
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--mock', type=int, default=173)
    ap.add_argument('--random-files', type=int, default=4)
    ap.add_argument('--fracs', type=float, nargs='+', default=[0.5, 0.25])
    ap.add_argument('--frac', type=float, default=1.0)
    ap.add_argument('--nthin', type=int, default=200)
    ap.add_argument('--ells', type=int, nargs='+', default=[0], help='multipoles (0 only is ~20x cheaper)')
    ap.add_argument('--pad', type=float, default=1.1)
    ap.add_argument('--max-time', type=float, default=1500.0)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--n-randoms-max', type=float, default=8e6)
    ap.add_argument('--target-near-pairs', type=float, default=5e9)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    {'thin': thin, 'thecov': thecov_stage, 'report': report}[args.stage](args)


if __name__ == '__main__':
    main()
