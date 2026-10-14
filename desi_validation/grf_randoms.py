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


def tensor_index(order):
    """index tuples i<=j(<=k<=l) of a symmetric tensor of rank `order` in 3D and their multiplicities"""
    import itertools
    from math import factorial
    idx = list(itertools.combinations_with_replacement(range(3), order))
    mult = []
    for t in idx:
        m = factorial(order)
        for v in range(3):
            m //= factorial(t.count(v))
        mult.append(m)
    return idx, np.array(mult, float)


def fields(args):
    import finufft
    paths, cfg, OUT = setup(args)
    od = out_dir(args, OUT)
    b, r, tag = args.bin, args.region, args.tag
    s1 = files_spectra(paths, cfg, b, r)
    k, edges, nb = s1['k'], np.asarray(s1['k_edges']), len(s1['k'])
    norm, num_sn = float(s1['norm'].mean()), float(s1['num_shotnoise'].mean())
    ells = tuple(args.ells)
    rc = dc.load_region(paths, b, r, n_random_files=args.random_files)
    w = dc.total_weight(rc.randoms)
    s = float(np.sum(dc.total_weight(rc.data)) / np.sum(w))
    x = dc.sky_to_cartesian(rc.randoms['RA'], rc.randoms['DEC'], rc.randoms['Z'])
    del rc
    rng0 = np.random.default_rng(12345)
    if args.poisson:            # galaxies are sampled from the first half of the files, randoms are the second half
        nf = args.random_files
        half = len(w) // 2    # files are concatenated in order: first half of the rows ~ first nf/2 files
        par, ran = slice(0, half), slice(half, None)
        xg_all, wg_all = x[par], w[par]
        x, w = x[ran], w[ran]
        rho = args.poisson * np.sum(wg_all) / (s * (np.sum(wg_all) + np.sum(w)))      # sum_g w_g / sum_data w
        log(f'poisson mode: {len(wg_all)} parent randoms, {len(w)} estimator randoms, p0 = {args.poisson}, rho = {rho:.4f}')
    if len(w) > args.max_randoms:          # the mask is defined by any dense subset; the self-pair term is modelled
        keep = rng0.random(len(w)) < args.max_randoms / len(w)
        s, w, x = s * np.sum(w) / np.sum(w[keep]), w[keep], x[keep]
    q = (s * w).astype(float)
    allx = np.vstack([x, xg_all]) if args.poisson else x
    lo, hi = allx.min(0), allx.max(0)
    L = args.pad * float(np.max(hi - lo))
    lo = 0.5 * (lo + hi) - 0.5 * L
    th = [np.ascontiguousarray(2 * np.pi * (x[:, i] - lo[i]) / L) for i in range(3)]
    xhat = x / np.linalg.norm(x, axis=1)[:, None]
    if args.poisson:
        thg = [np.ascontiguousarray(2 * np.pi * (xg_all[:, i] - lo[i]) / L) for i in range(3)]
        xhg = xg_all / np.linalg.norm(xg_all, axis=1)[:, None]
    del allx
    kf = 2 * np.pi / L
    n_gen = sfft.next_fast_len(2 * int(np.ceil(args.k_c / kf)) + 4)
    n_out = 2 * int(np.ceil(edges[-1] / kf)) + 4
    vc = (L / n_gen) ** 3
    log(f'{b} {r}{tag}: {len(q)} randoms ({args.random_files} files), s = {s:.4g}, box L = {L:.0f} Mpc/h, '
        f'kf = {kf:.5f}, generation grid {n_gen}^3 (k_c {args.k_c}), output grid {n_out}^3, norm {norm:.4g}, ells {ells}')

    P0 = s1['vectors'][:, :nb].mean(0) * args.amp
    kg = kgrid(n_gen, L)
    Pg = p_input(k, P0, kg, args.k_c)
    sigma2 = float(Pg.sum() / (vc * n_gen ** 3))                    # Var delta of the band-limited field
    if args.sigma:                                                  # rescale to a target rms (poisson mode)
        Pg *= args.sigma ** 2 / sigma2
        sigma2 = args.sigma ** 2
    amp = (np.sqrt(Pg / vc) / n_gen ** 3)
    ig = np.digitize(kg.ravel(), edges) - 1
    ig[(ig < 0) | (ig >= nb)] = nb
    Pr = Pg.ravel()
    nsh_g = np.bincount(ig, minlength=nb + 1)[:nb]
    box_mean = np.bincount(ig, Pr, nb + 1)[:nb] / nsh_g
    box_var = 2 * np.bincount(ig, Pr ** 2, nb + 1)[:nb] / nsh_g ** 2
    f = args.rsd
    if f:
        kk = [2 * np.pi * np.fft.fftfreq(n_gen, d=L / n_gen)] * 3
        KX, KY, KZ = np.meshgrid(*kk, indexing='ij')
        with np.errstate(invalid='ignore', divide='ignore'):
            inv_k2 = np.where(kg > 0, 1.0 / kg ** 2, 0.0)
        Kc = (KX, KY, KZ)
        del KX, KY, KZ
        t2, _ = tensor_index(2)
    del kg, Pg
    ko = [2 * np.pi * np.fft.fftfreq(n_out, d=L / n_out)] * 3
    KO = np.meshgrid(*ko, indexing='ij')
    kom = np.sqrt(KO[0] ** 2 + KO[1] ** 2 + KO[2] ** 2)
    io = np.digitize(kom.ravel(), edges) - 1
    io[(io < 0) | (io >= nb)] = nb
    nsh_o = np.bincount(io, minlength=nb + 1)[:nb]
    with np.errstate(invalid='ignore', divide='ignore'):
        khat = [np.where(kom > 0, K_ / kom, 0.0).ravel() for K_ in KO]
    del KO, kom
    opts = dict(eps=args.eps, nthreads=NTH, modeord=1)
    orders = {0: ([()], np.array([1.0]))}
    if max(ells) >= 2:
        orders[2] = tensor_index(2)
    if max(ells) >= 4:
        orders[4] = tensor_index(4)
    tens = [t for o in sorted(orders) for t in orders[o][0]]          # all tensor components, order 0, 2, 4

    def moments(theta, xh, d):
        """type-1 NUFFTs of d * xhat_i xhat_j ... for every component in `tens`; shape (len(tens), n_out^3)"""
        cs = np.empty((len(tens), len(d)), complex)
        for n_, t in enumerate(tens):
            v = d.copy()
            for i in t:
                v *= xh[:, i]
            cs[n_] = v
        return finufft.nufft3d1(*theta, cs, (n_out,) * 3, isign=-1, **opts).reshape(len(tens), -1)

    def multipoles(Fm):
        """P_ell(k) from the moments: Yamamoto, one line of sight on the ell leg"""
        F0 = Fm[0]
        out = {0: (F0.real ** 2 + F0.imag ** 2)}
        pos = 1
        if 2 in orders:
            idx2, m2 = orders[2]
            Q2 = np.zeros(F0.shape, complex)
            for (i, j), mu in zip(idx2, m2):
                Q2 += mu * khat[i] * khat[j] * Fm[pos]; pos += 1
            F2 = 1.5 * Q2 - 0.5 * F0
            out[2] = (F0 * F2.conj()).real
        if 4 in orders:
            idx4, m4 = orders[4]
            Q4 = np.zeros(F0.shape, complex)
            for (i, j, k_, l), mu in zip(idx4, m4):
                Q4 += mu * khat[i] * khat[j] * khat[k_] * khat[l] * Fm[pos]; pos += 1
            F4 = 35 / 8 * Q4 - 15 / 4 * Q2 + 3 / 8 * F0
            out[4] = (F0 * F4.conj()).real
        return out

    first = args.task * args.nreal
    P_mask = {ell: np.zeros((args.nreal, nb)) for ell in ells}
    P_box, SN, SIG = np.zeros((args.nreal, nb)), np.zeros(args.nreal), np.zeros(args.nreal)
    NG = np.zeros(args.nreal)
    for j, i in enumerate(range(first, first + args.nreal)):
        t0 = time.time()
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, ['NGC', 'SGC'].index(r), i]))
        W = sfft.fftn(rng.standard_normal((n_gen,) * 3), workers=NTH)
        P_box[j] = np.bincount(ig, (W.real ** 2 + W.imag ** 2).ravel() * Pr, nb + 1)[:nb] / nsh_g / n_gen ** 3
        W *= amp
        if f:
            stack = np.empty((7,) + W.shape, complex)
            stack[0] = W
            for n_, (a_, b_) in enumerate(t2):
                stack[n_ + 1] = W * Kc[a_] * Kc[b_] * inv_k2
            del W
            def evaluate(theta, xh):
                g = finufft.nufft3d2(*theta, stack, isign=1, **opts).real
                d = g[0].copy()
                for n_, (a_, b_) in enumerate(t2):
                    d += f * (2.0 if a_ != b_ else 1.0) * xh[:, a_] * xh[:, b_] * g[n_ + 1]
                return d
        else:
            def evaluate(theta, xh):
                return finufft.nufft3d2(*theta, W, isign=1, **opts).real
        if args.poisson:
            dg = evaluate(thg, xhg)
            p = np.clip(args.poisson * (1 + dg), 0, 1)
            sel = rng.random(len(p)) < p
            wg = wg_all[sel]
            alpha = np.sum(wg) / np.sum(w)
            NG[j] = sel.sum(); SIG[j] = dg.var()
            Fm = moments(tuple(t_[sel] for t_ in thg), xhg[sel], wg) - alpha * moments(th, xhat, w)
            SN[j] = float(np.sum(wg ** 2) + alpha ** 2 * np.sum(w ** 2))
            nrm = norm * rho ** 2
            del dg, p, sel
        else:
            d = evaluate(th, xhat)
            SIG[j] = d.var()
            if args.shot_noise == 'files':                             # white weight noise up to the files' shot noise
                s_eta = np.sqrt(max(num_sn / np.sum(q ** 2) - SIG[j], 0.0))
                d += s_eta * rng.standard_normal(len(d))
            dd = q * d
            SN[j] = float(np.sum(dd ** 2))
            Fm = moments(th, xhat, dd)
            nrm = norm
            del d, dd
        if f:
            del stack
        else:
            del W
        pm = multipoles(Fm)
        del Fm
        for ell in ells:
            P_mask[ell][j] = (2 * ell + 1) * np.bincount(io, pm[ell], nb + 1)[:nb] / nsh_o / nrm
        P_mask[0][j] -= SN[j] / nrm
        if j < 2 or (j + 1) % 10 == 0:
            log(f'  realisation {i}: {time.time() - t0:.1f} s; SN/norm {SN[j] / nrm:.1f}; sigma(d) {np.sqrt(SIG[j]):.3f}; '
                f'P_box/P_in (bin 0, last) {P_box[j, 0] / box_mean[0]:.3f}, {P_box[j, -1] / box_mean[-1]:.3f}' +
                (f'; N_gal {int(NG[j])}' if args.poisson else ''))
        if time.time() - T0 > args.max_time:
            log(f'  time limit: stopping after {j + 1} realisations')
            n_done = j + 1
            P_mask = {ell: v[:n_done] for ell, v in P_mask.items()}
            P_box, SN, SIG, NG = P_box[:n_done], SN[:n_done], SIG[:n_done], NG[:n_done]
            break
    fn = os.path.join(od, f'fields_{b}_{r}{tag}_task{args.task}.npz')
    np.savez(fn, k=k, k_edges=edges, norm=norm, s=s, L=L, n_gen=n_gen, n_out=n_out, k_c=args.k_c, amp=args.amp,
             ells=np.array(ells), rsd=f, shot_noise=args.shot_noise, poisson=args.poisson or 0.0, sigma=np.sqrt(sigma2),
             rho=(rho if args.poisson else 1.0), num_shotnoise_files=num_sn,
             P_mask=P_mask[0], **{f'P_mask_{ell}': P_mask[ell] for ell in ells}, P_box=P_box, SN=SN, SIG=SIG, NG=NG,
             box_mean=box_mean, box_var=box_var, nsh_g=nsh_g, nsh_o=nsh_o,
             indices=np.arange(first, first + len(SN)), random_files=args.random_files)
    log(f'saved {fn} ({len(SN)} realisations)')


def load_fields(od, b, r, tag=''):
    fns = sorted(glob.glob(os.path.join(od, f'fields_{b}_{r}{tag}_task*.npz')))
    zs = [dict(np.load(f)) for f in fns]
    out = dict(zs[0])
    for key in [k_ for k_ in zs[0] if k_.startswith('P_mask')] + ['P_box', 'SN', 'SIG', 'NG', 'indices']:
        if key in zs[0]:
            out[key] = np.concatenate([z[key] for z in zs])
    assert len(np.unique(out['indices'])) == len(out['indices']), 'duplicate realisations across tasks'
    return out


def run_thecov(args):
    paths, cfg, OUT = setup(args)
    od = out_dir(args, OUT)
    b, r, mode, tag = args.bin, args.region, args.mode, args.tag
    g = load_fields(od, b, r, tag)
    nb, n = len(g['k']), len(g['SN'])
    ells = tuple(int(e) for e in g['ells']) if 'ells' in g else (0,)
    rho = float(g['rho']) if 'rho' in g else 1.0
    nrm = float(g['norm']) * rho ** 2
    sn = float(g['SN'].mean())
    fr = float(g['rsd']) if 'rsd' in g else 0.0
    log(f'{b} {r}{tag} [{mode}]: {n} realisations, ells {ells}, rsd f = {fr}, rho = {rho:.4f}; SN / norm = {sn / nrm:.2f}')
    cfg = dataclasses.replace(cfg, n_randoms_max=int(args.n_randoms_max), target_near_pairs=args.target_near_pairs, ells=ells)
    base = files_spectra(paths, cfg, b, r)
    V = np.concatenate([g[f'P_mask_{ell}'] if f'P_mask_{ell}' in g else g['P_mask'] for ell in ells], axis=1)

    def spec(vec):
        return dict(base, ells=ells, vectors=vec, norm=np.full(len(vec), nrm), num_shotnoise=np.full(len(vec), sn))
    spec_masked = spec(V)                                             # model = the fields' mean (as with mocks)
    kais = {0: 1 + 2 * fr / 3 + fr ** 2 / 5, 2: 4 * fr / 3 + 4 * fr ** 2 / 7, 4: 8 * fr ** 2 / 35}
    spec_true = spec(np.concatenate([kais[ell] * g['box_mean'] for ell in ells])[None, :] * rho ** 2)
    rc = dc.load_region(paths, b, r, n_random_files=cfg.n_random_files)
    name = f'GRFR_{b}_{r}{tag}_{mode}'
    tr, _ = dc.build_tracer(name, [rc], nw='random-density', n_randoms_max=cfg.n_randoms_max // 2,
                            surface_density_deg2=cfg.surface_density, verbose=False, shotnoise_target=sn)
    sm = pl.kernel_smoothing(paths, b, r, name, spec_true, cfg, log=log) if mode == 'kernel' else None
    del rc
    n_near, _ = dc.suggest_n_near(tr, target_pairs=cfg.target_near_pairs)
    wfile = os.path.join(od, f'windows_{name}.npz')
    res = dict(ells=np.array(ells), rho=rho, rsd=fr)
    for label, sp, pin in (('masked', spec_masked, 'masked'), ('true', spec_true, 'unmasked')):
        t0 = time.time()
        C, cov = dc.thecov_covariance(tr, sp, n_sub=cfg.n_sub_far, n_near=n_near, verbose=False, windows_file=wfile,
                                      power_input=pin, smoothing=sm)
        res[f'C_{label}'] = C
        res[f'var_{label}'] = np.diag(C)
        res['I_local_over_norm'] = cov.I_randoms / nrm
        if cov.I_k_bins is not None:
            res['I_k_over_norm'] = np.asarray(cov.I_k_bins) / nrm
        log(f'  thecov [{label}] {time.time() - t0:.0f} s; int m^2 / norm = {res["I_local_over_norm"]:.4f}' +
            ('' if cov.I_k_bins is None else f'; I_k / norm first, last = {res["I_k_over_norm"][0]:.4f}, '
                                              f'{res["I_k_over_norm"][-1]:.4f}'))
        if label == 'true' and g.get('poisson', 0) and float(g['poisson']) > 0:
            try:                                                      # discreteness (SN-NG) terms for the Poisson variant
                from thecov.discreteness import DiscretenessCovariance
                kk = np.concatenate([[g['k_edges'][0] / 4], g['k'], [g['k_edges'][-1]]])
                pl_ = kk, np.concatenate([[g['box_mean'][0]], g['box_mean'], [g['box_mean'][-1]]]) * rho ** 2
                D = DiscretenessCovariance(cov, pl_, b1=1.0, f=0.0)
                comps, index = D.components(sp, ells=ells, return_index=True)
                res['disc_components'] = np.asarray(comps)
                res['disc_index'] = np.asarray(index, dtype=object)
                log(f'  discreteness components: {index}')
            except Exception as e:                                    # noqa: BLE001
                log(f'  discreteness terms not computed: {e!r}')
    fn = os.path.join(od, f'thecov_{b}_{r}{tag}_{mode}.npz')
    np.savez(fn, k=g['k'], n_fields=n, sn=sn, n_near=n_near, **{k_: v for k_, v in res.items() if k_ != 'disc_index'},
             **({'disc_index': res['disc_index']} if 'disc_index' in res else {}))
    v = V.var(0, ddof=1)
    for ei, ell in enumerate(ells):
        sl = slice(ei * nb, (ei + 1) * nb)
        for lo, hi in ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3)):
            m = (g['k'] >= lo) & (g['k'] < hi)
            log(f'  ell={ell} Var_GRF / Var_thecov {lo}-{hi}: true P {np.mean(v[sl][m] / res["var_true"][sl][m]):.3f}, '
                f'masked {np.mean(v[sl][m] / res["var_masked"][sl][m]):.3f}')
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
    ap.add_argument('--tag', default='', help='variant tag in the file names (e.g. _snL, _rsd, _poi)')
    ap.add_argument('--ells', type=int, nargs='+', default=[0], help='multipoles to measure (fields)')
    ap.add_argument('--shot-noise', default='none', choices=['none', 'files'],
                    help="fields: add white weight noise so that the self-pair term equals the files' num_shotnoise")
    ap.add_argument('--rsd', type=float, default=0.0, help='fields: linear RSD with this f at the true line of sight')
    ap.add_argument('--poisson', type=float, default=0.0,
                    help='fields: sample galaxies from the first half of the random files with p = p0 (1 + delta)')
    ap.add_argument('--sigma', type=float, default=0.0, help='fields: rescale the field to this rms (poisson mode)')
    args = ap.parse_args()
    args.cs_version = args.spectra_dir = args.mock = args.synthetic = None
    (fields if args.stage == 'fields' else run_thecov)(args)


if __name__ == '__main__':
    main()
