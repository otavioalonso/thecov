"""Exact Gaussian covariance of P0 on the real DESI randoms (holes and edges resolved), against thecov.

    # 1. fields (one debug job per task; tasks add realisations)
    python -u -m desi_validation.grf_randoms fields --bin LRG1 --region NGC --task 0 --nreal 40
    # 2. thecov for the same window (one job per window mode; reads the field outputs)
    python -u -m desi_validation.grf_randoms thecov --bin LRG1 --region NGC --mode random-density
    python -u -m desi_validation.grf_randoms thecov --bin LRG1 --region NGC --mode kernel

Why. run_gaussian_footprint.py multiplies the field by a window painted on a healpix-512 map with fill-fraction
dilution, i.e. the veto holes enter as partial coverage of ~2.5 Mpc/h pixels, the same approximation as the
xi-kernel window for sub-kernel holes: that test cannot arbitrate the holes. Here the mask is the randoms
themselves: no random falls in a hole, so the holes are resolved to the randoms' own resolution.

Construction (exact, no painting, no aliasing):
  - delta: Gaussian, band-limited at |k| < k_c on a periodic box (cube, pad x footprint), power P_in;
  - delta(x_r) at every random by exact trigonometric interpolation (type-2 NUFFT);
  - F(k) = sum_r q_r delta(x_r) exp(-i k.x_r), q_r = s w_r (s = sum w_data / sum w_randoms), by type-1 NUFFT on the
    box's modes: F is LINEAR in delta, hence exactly Gaussian, and its covariance is the exact Gaussian covariance
    of the window defined by the randoms (pairs of distinct randoms) plus a white self-pair term
    sum_r q_r^2 delta_r^2, which is subtracted like a shot noise and given to thecov as its S amplitude;
  - P0_hat(k) = <|F|^2>_shell / norm - SN / norm (norm: the spectra files').
thecov is then given the same randoms' window, P_in exactly (unmasked) or the fields' mean (masked, as for mocks).

Control variate: the same delta measured in the full box (no window) has an exactly known mean and variance
(2 sum P^2 / N_shell^2 over the full-grid modes of the shell); it is saved per realisation.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import sys
import time
import types

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402
from desi_validation.run_gaussian_footprint import setup  # noqa: E402

T0 = time.time()
NTH = max(1, (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()) // 2)   # physical cores (SMT 2)


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def out_dir(args, OUT):
    d = args.out_dir or os.path.join(OUT, args.bin, 'grf_randoms')
    os.makedirs(d, exist_ok=True)
    return d


def files_spectra(paths, cfg, b, r):
    tracer, zr = dc.TRACER_SPECS[b]
    return dc.read_spectra(paths.spectra_fns(tracer, zr, r), kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=cfg.ells)


def p_input(k, P0, kk, k_c):
    """isotropic input power: P0 on [k0, k1], held below, power law above, zero beyond k_c"""
    slope = np.polyfit(np.log(k[-8:]), np.log(np.abs(P0[-8:])), 1)[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(kk < k[0], P0[0], np.where(kk > k[-1], P0[-1] * (kk / k[-1]) ** slope, np.interp(kk, k, P0)))
    return np.where(kk < k_c, p, 0.0)


def kgrid(n, L):
    f = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    return np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2)


def fields(args):
    import finufft
    paths, cfg, OUT = setup(args)
    od = out_dir(args, OUT)
    b, r = args.bin, args.region
    s1 = files_spectra(paths, cfg, b, r)
    k, edges, nb = s1['k'], np.asarray(s1['k_edges']), len(s1['k'])
    norm = float(s1['norm'].mean())
    rc = dc.load_region(paths, b, r, n_random_files=args.random_files)
    w = dc.total_weight(rc.randoms)
    s = float(np.sum(dc.total_weight(rc.data)) / np.sum(w))
    x = dc.sky_to_cartesian(rc.randoms['RA'], rc.randoms['DEC'], rc.randoms['Z'])
    del rc
    if len(w) > args.max_randoms:          # the mask is defined by any dense subset; the self-pair term is modelled
        keep = np.random.default_rng(12345).random(len(w)) < args.max_randoms / len(w)
        s, w, x = s * np.sum(w) / np.sum(w[keep]), w[keep], x[keep]
    q = (s * w).astype(float)
    lo, hi = x.min(0), x.max(0)
    L = args.pad * float(np.max(hi - lo))
    lo = 0.5 * (lo + hi) - 0.5 * L
    th = [np.ascontiguousarray(2 * np.pi * (x[:, i] - lo[i]) / L) for i in range(3)]
    del x
    kf = 2 * np.pi / L
    n_gen = sfft.next_fast_len(2 * int(np.ceil(args.k_c / kf)) + 4)
    n_out = 2 * int(np.ceil(edges[-1] / kf)) + 4
    vc = (L / n_gen) ** 3
    log(f'{b} {r}: {len(q)} randoms ({args.random_files} files), s = {s:.4g}, box L = {L:.0f} Mpc/h, '
        f'kf = {kf:.5f}, generation grid {n_gen}^3 (k_c {args.k_c}), output grid {n_out}^3, norm {norm:.4g}')

    # input power: a realistic amplitude (the test is exact for any P_in): the mocks' mean P0 x int m^2 / norm
    P0 = s1['vectors'][:, :nb].mean(0) * args.amp
    kg = kgrid(n_gen, L)
    Pg = p_input(k, P0, kg, args.k_c)
    amp = (np.sqrt(Pg / vc) / n_gen ** 3)
    ig = np.digitize(kg.ravel(), edges) - 1
    ig[(ig < 0) | (ig >= nb)] = nb
    Pr = Pg.ravel()
    nsh_g = np.bincount(ig, minlength=nb + 1)[:nb]
    box_mean = np.bincount(ig, Pr, nb + 1)[:nb] / nsh_g
    box_var = 2 * np.bincount(ig, Pr ** 2, nb + 1)[:nb] / nsh_g ** 2
    del kg, Pg
    ko = kgrid(n_out, L).ravel()
    io = np.digitize(ko, edges) - 1
    io[(io < 0) | (io >= nb)] = nb
    nsh_o = np.bincount(io, minlength=nb + 1)[:nb]
    del ko
    opts = dict(eps=args.eps, nthreads=NTH, modeord=1)

    first = args.task * args.nreal
    P_mask, P_box, SN = np.zeros((args.nreal, nb)), np.zeros((args.nreal, nb)), np.zeros(args.nreal)
    for j, i in enumerate(range(first, first + args.nreal)):
        t0 = time.time()
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, ['NGC', 'SGC'].index(r), i]))
        W = sfft.fftn(rng.standard_normal((n_gen,) * 3), workers=NTH)
        P_box[j] = np.bincount(ig, (W.real ** 2 + W.imag ** 2).ravel() * Pr, nb + 1)[:nb] / nsh_g / n_gen ** 3
        W *= amp
        d = finufft.nufft3d2(*th, W, isign=1, **opts).real            # delta at the randoms
        del W
        SN[j] = float(np.sum((q * d) ** 2))
        F = finufft.nufft3d1(*th, (q * d).astype(complex), (n_out,) * 3, isign=-1, **opts)
        del d
        P_mask[j] = (np.bincount(io, (F.real ** 2 + F.imag ** 2).ravel(), nb + 1)[:nb] / nsh_o - SN[j]) / norm
        del F
        if j < 2 or (j + 1) % 10 == 0:
            log(f'  realisation {i}: {time.time() - t0:.1f} s; SN/norm {SN[j] / norm:.1f}; '
                f'P_box/P_in (bin 0, last) {P_box[j, 0] / box_mean[0]:.3f}, {P_box[j, -1] / box_mean[-1]:.3f}')
        if time.time() - T0 > args.max_time:
            log(f'  time limit: stopping after {j + 1} realisations')
            P_mask, P_box, SN = P_mask[:j + 1], P_box[:j + 1], SN[:j + 1]
            break
    fn = os.path.join(od, f'fields_{b}_{r}_task{args.task}.npz')
    np.savez(fn, k=k, k_edges=edges, norm=norm, s=s, L=L, n_gen=n_gen, n_out=n_out, k_c=args.k_c, amp=args.amp,
             P_mask=P_mask, P_box=P_box, SN=SN, box_mean=box_mean, box_var=box_var, nsh_g=nsh_g, nsh_o=nsh_o,
             indices=np.arange(first, first + len(SN)), random_files=args.random_files)
    log(f'saved {fn} ({len(SN)} realisations)')


def load_fields(od, b, r):
    fns = sorted(glob.glob(os.path.join(od, f'fields_{b}_{r}_task*.npz')))
    zs = [dict(np.load(f)) for f in fns]
    out = dict(zs[0])
    for key in ('P_mask', 'P_box', 'SN', 'indices'):
        out[key] = np.concatenate([z[key] for z in zs])
    assert len(np.unique(out['indices'])) == len(out['indices']), 'duplicate realisations across tasks'
    return out


def run_thecov(args):
    paths, cfg, OUT = setup(args)
    od = out_dir(args, OUT)
    b, r, mode = args.bin, args.region, args.mode
    g = load_fields(od, b, r)
    nb, n = len(g['k']), len(g['SN'])
    sn = float(g['SN'].mean())
    log(f'{b} {r} [{mode}]: {n} realisations; SN (self pairs) / norm = {sn / g["norm"]:.2f}')
    cfg = dataclasses.replace(cfg, n_randoms_max=int(args.n_randoms_max), target_near_pairs=args.target_near_pairs)
    base = files_spectra(paths, cfg, b, r)

    def spec(vec0):
        v = np.zeros((len(vec0), 3 * nb))
        v[:, :nb] = vec0
        return dict(base, vectors=v, norm=np.full(len(vec0), g['norm']), num_shotnoise=np.full(len(vec0), sn))
    spec_masked = spec(g['P_mask'])                        # model = the fields' mean (as with mocks)
    spec_true = spec(g['box_mean'][None, :])               # P_in exactly (shell average on the box modes)
    rc = dc.load_region(paths, b, r, n_random_files=cfg.n_random_files)
    name = f'GRFR_{b}_{r}_{mode}'
    tr, _ = dc.build_tracer(name, [rc], nw='random-density', n_randoms_max=cfg.n_randoms_max // 2,
                            surface_density_deg2=cfg.surface_density, verbose=False, shotnoise_target=sn)
    sm = pl.kernel_smoothing(paths, b, r, name, spec_true, cfg, log=log) if mode == 'kernel' else None
    del rc
    n_near, _ = dc.suggest_n_near(tr, target_pairs=cfg.target_near_pairs)
    wfile = os.path.join(od, f'windows_{name}.npz')
    res = {}
    for label, sp, pin in (('masked', spec_masked, 'masked'), ('true', spec_true, 'unmasked')):
        t0 = time.time()
        C, cov = dc.thecov_covariance(tr, sp, n_sub=cfg.n_sub_far, n_near=n_near, verbose=False, windows_file=wfile,
                                      power_input=pin, smoothing=sm)
        res[f'var_{label}'] = np.diag(C)[:nb]
        res['I_local_over_norm'] = cov.I_randoms / g['norm']
        if cov.I_k_bins is not None:
            res['I_k_over_norm'] = np.asarray(cov.I_k_bins) / g['norm']
        log(f'  thecov [{label}] {time.time() - t0:.0f} s; int m^2 / norm = {res["I_local_over_norm"]:.4f}' +
            ('' if cov.I_k_bins is None else f'; I_k / norm first, last = {res["I_k_over_norm"][0]:.4f}, '
                                              f'{res["I_k_over_norm"][-1]:.4f}'))
    fn = os.path.join(od, f'thecov_{b}_{r}_{mode}.npz')
    np.savez(fn, k=g['k'], n_fields=n, sn=sn, n_near=n_near, **res)
    v = g['P_mask'].var(0, ddof=1)
    for lo, hi in ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3)):
        m = (g['k'] >= lo) & (g['k'] < hi)
        log(f'  Var_GRF / Var_thecov {lo}-{hi}: true P {np.mean(v[m] / res["var_true"][m]):.3f}, '
            f'masked {np.mean(v[m] / res["var_masked"][m]):.3f}')
    log(f'saved {fn}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['fields', 'thecov'])
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--region', default='NGC', choices=['NGC', 'SGC'])
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--nreal', type=int, default=40)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--random-files', type=int, default=10, help='random files defining the mask (fields)')
    ap.add_argument('--max-randoms', type=float, default=4e7, help='random subsample defining the mask (fields)')
    ap.add_argument('--pad', type=float, default=1.5, help='periodic box side / largest footprint extent')
    ap.add_argument('--k-c', type=float, default=0.4, help='band limit of the fields [h/Mpc]')
    ap.add_argument('--amp', type=float, default=1.22, help='P_in = mocks mean P0 x amp (~ int m^2 / norm)')
    ap.add_argument('--eps', type=float, default=1e-6, help='NUFFT tolerance')
    ap.add_argument('--max-time', type=float, default=1500.0, help='stop adding realisations after this [s]')
    ap.add_argument('--mode', default='random-density', choices=['random-density', 'kernel'])
    ap.add_argument('--n-randoms-max', type=float, default=8e6)
    ap.add_argument('--target-near-pairs', type=float, default=1e10)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    args.cs_version = args.spectra_dir = args.mock = args.synthetic = None
    (fields if args.stage == 'fields' else run_thecov)(args)


if __name__ == '__main__':
    main()
