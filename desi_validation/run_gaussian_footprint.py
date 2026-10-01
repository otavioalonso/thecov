"""Exact Gaussian covariance on the real footprint from Gaussian random fields, against thecov.

    python -m desi_validation.run_gaussian_footprint                       # LRG1 NGC + SGC, holi v3 window
    python -m desi_validation.run_gaussian_footprint --nreal 400 --cell 5  # more realisations, finer mesh

Why. With wider k bins the LRG1 excess splits into a part proportional to the bin width (non-Gaussian)
and a part that is not (a ~ 0.10 in P0 and P2, ~0 for QSO). Two Gaussian-level suspects remain:
  1. the local approximation W(x) = m(x)^2 of thecov's clustering window (m(x) m(x+r) at r within a
     correlation length is what the covariance really contains; fine veto masks and completeness
     patterns break the approximation), and
  2. the model normalisation P_model = <P_hat> norm / int m^2 (one number for all k).
Here both are bypassed: Gaussian fields delta with power P_in are multiplied by m(x) on a mesh, white
noise with the shot-noise density S(x) is added, and P0 is estimated as in the files (|F|^2 / norm minus
the shot noise). The variance over realisations is the exact Gaussian covariance of the windowed
estimator (no local approximation, no assumption on the normalisation), for an isotropic P
(P2 = P4 = 0, so that the line of sight plays no role), compared with thecov's for the same model.

m(x) and S(x) are built from many random files as (weighted angular density on a healpix map) x
(weighted radial distribution), which resolves the veto masks down to the pixel size (nside 512 ~ 7'
~ 2.5 Mpc/h at z = 0.5) without the Poisson noise of painting randoms in small cells, and assumes the
angular and radial parts of the weighted density factorise (the n(z) variation across the sky was
measured to be at the Poisson level).

Reports, per region and bin width (x1, x2, x4 = 0.005, 0.01, 0.02 h/Mpc), in k ranges:
  - <P0_hat>_GRF / <P0>_mocks       (test of the model normalisation: 1 if consistent)
  - Var_GRF / Var_thecov            (test of thecov's Gaussian term: 1 if the local approximation holds)
  - Var_mocks / Var_GRF             (what is left for the mocks: the part that is not Gaussian)
Caveat: the fields have no integral constraint (m is fixed, alpha is not re-estimated per
realisation), so the first bins (k < ~0.03) of <P>grf/<P>mocks and of the variances are not
comparable; the window convolution of the input P (held constant below k = 0.02) also matters there.
Output: OUT/LRG1/gaussian_footprint_<region>.json and a summary on stdout.
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import dataclasses
import json
import os
import sys
import time

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402

T0 = time.time()
KRANGES = ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3))


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


# ----------------------------------------------------------------------------- window maps
class SeparableWindow:
    """m(x) = alpha n_w(theta) p_w(z) / (chi^2 dchi/dz), S(x) = num_sn x (same with w^2) / sum w^2."""

    def __init__(self, rc, num_sn, nside=512, dz=0.002, cosmo=None, n_files=1, surface_density=2500.0):
        import healpy as hp
        self.hp, self.nside = hp, nside
        r, d = rc.randoms, rc.data
        w = dc.total_weight(r)
        self.alpha = dc.total_weight(d).sum() / w.sum()
        self.int_m = self.alpha * w.sum()
        npix = hp.nside2npix(nside)
        om = 4 * np.pi / npix
        pix = hp.ang2pix(nside, np.asarray(r['RA'], float), np.asarray(r['DEC'], float), lonlat=True)
        self.aw = np.bincount(pix, w, npix) / om / w.sum()            # angular pdf of the weights [1/sr]
        self.aw2 = np.bincount(pix, w ** 2, npix) / om / np.sum(w ** 2)
        # fraction of each pixel inside the footprint (veto holes below the pixel size dilute it): the
        # z-range randoms only sample it, so use all of them (the redshift cut does not depend on angle)
        expected = n_files * surface_density * om * (180 / np.pi) ** 2 * len(r['Z']) / max(rc.n_randoms_all_z, len(r['Z']))
        self.fill = np.bincount(pix, None, npix) / expected
        z = np.asarray(r['Z'], float)
        self.zedges = np.arange(z.min() - 1e-9, z.max() + dz, dz)
        chi = dc.comoving_distance(self.zedges, cosmo)
        dv = (chi[1:] ** 3 - chi[:-1] ** 3) / 3.0                     # volume per sr of each z shell
        pw = np.histogram(z, self.zedges, weights=w)[0] / w.sum()
        pw2 = np.histogram(z, self.zedges, weights=w ** 2)[0] / np.sum(w ** 2)
        self.rad_m = self.alpha * w.sum() * pw / dv                   # m = aw(theta) * rad_m(z)
        self.rad_s = num_sn * pw2 / dv                                # S = aw2(theta) * rad_s(z)
        zz = np.linspace(self.zedges[0], self.zedges[-1], 4001)
        self.chi_grid, self.z_grid = dc.comoving_distance(zz, cosmo), zz
        self.chi_min, self.chi_max = chi[0], chi[-1]
        self.npix_occupied = int(np.sum(self.aw > 0))

    def at(self, x):
        """m, S and the pixel fill fraction at cartesian positions x (n, 3)."""
        rr = np.sqrt(np.sum(x ** 2, axis=1))
        out_m, out_s, out_f = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
        ok = (rr > self.chi_min) & (rr < self.chi_max)
        if not ok.any():
            return out_m, out_s, out_f
        xo, ro = x[ok], rr[ok]
        z = np.interp(ro, self.chi_grid, self.z_grid)
        iz = np.clip(np.searchsorted(self.zedges, z) - 1, 0, len(self.rad_m) - 1)
        theta = np.arccos(np.clip(xo[:, 2] / ro, -1, 1))
        phi = np.arctan2(xo[:, 1], xo[:, 0])
        pix = self.hp.ang2pix(self.nside, theta, phi)
        out_m[ok] = self.aw[pix] * self.rad_m[iz]
        out_s[ok] = self.aw2[pix] * self.rad_s[iz]
        out_f[ok] = self.fill[pix]
        return out_m, out_s, out_f


class Grid:
    def __init__(self, pos, cell, pad, workers=None):
        lo, hi = pos.min(0), pos.max(0)
        ext = hi - lo
        self.cell, self.vc = cell, cell ** 3
        self.shape = tuple(sfft.next_fast_len(int(np.ceil(pad * e / cell)) + 2) for e in ext)
        self.lo = 0.5 * (lo + hi) - 0.5 * np.array(self.shape) * cell
        self.workers = workers or os.cpu_count()
        kx = 2 * np.pi * np.fft.fftfreq(self.shape[0], d=cell)[:, None, None]
        ky = 2 * np.pi * np.fft.fftfreq(self.shape[1], d=cell)[None, :, None]
        kz = 2 * np.pi * np.fft.rfftfreq(self.shape[2], d=cell)[None, None, :]
        self.kmag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2).astype(np.float32)
        nz = self.shape[2]
        herm = np.full(kz.shape[2], 2.0, np.float32)                  # rfft: count the conjugate modes
        herm[0] = 1.0
        if nz % 2 == 0:
            herm[-1] = 1.0
        self.herm = np.broadcast_to(herm[None, None, :], self.kmag.shape)
        log(f'mesh {self.shape} ({np.prod(self.shape) / 1e6:.0f}M cells, cell {cell} Mpc/h, '
            f'kNyq {np.pi / cell:.3f}, kf {2 * np.pi / (max(self.shape) * cell):.4f})')

    def fill(self, win, sub=2):
        """m, S and the footprint fill fraction, averaged over sub^3 points per cell."""
        m = np.zeros(self.shape, np.float32)
        s = np.zeros(self.shape, np.float32)
        fr = np.zeros(self.shape, np.float32)
        offs = (np.arange(sub) + 0.5) / sub
        ny, nz = self.shape[1], self.shape[2]
        jy, jz = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
        for i in range(self.shape[0]):
            acc_m = np.zeros(ny * nz)
            acc_s = np.zeros(ny * nz)
            acc_f = np.zeros(ny * nz)
            for ox in offs:
                for oy in offs:
                    for oz in offs:
                        x = np.stack([np.full(ny * nz, self.lo[0] + (i + ox) * self.cell),
                                      self.lo[1] + (jy.ravel() + oy) * self.cell,
                                      self.lo[2] + (jz.ravel() + oz) * self.cell], axis=1)
                        a, b, c = win.at(x)
                        acc_m += a
                        acc_s += b
                        acc_f += c
            m[i] = (acc_m / sub ** 3).reshape(ny, nz)
            s[i] = (acc_s / sub ** 3).reshape(ny, nz)
            fr[i] = (acc_f / sub ** 3).reshape(ny, nz)
        return m, s, fr

    def centers(self, flat_idx):
        ijk = np.stack(np.unravel_index(flat_idx, self.shape), axis=1)
        return self.lo + (ijk + 0.5) * self.cell

    def rfft(self, f):
        return sfft.rfftn(f, workers=self.workers)

    def irfft(self, F):
        return sfft.irfftn(F, s=self.shape, workers=self.workers)


_G = {}          # read-only state shared with the worker processes (inherited through fork)


def _realisation(i):
    """one Gaussian realisation, single-threaded; returns {window: {f: P0_hat(k)}}"""
    g = _G
    grid = g['grid']
    rng = np.random.default_rng(np.random.SeedSequence([g['seed'], g['region_id'], i]))   # pilot: i >= 10**6
    fft_kw = dict(workers=1)
    white = sfft.rfftn(rng.standard_normal(grid.shape, dtype=np.float32), **fft_kw)
    noise = g['noise_amp'] * rng.standard_normal(grid.shape, dtype=np.float32)
    out = {}
    for name, mm in g['windows'].items():             # same white noise for every window: paired
        delta = sfft.irfftn(white * g['amp'][name], s=grid.shape, **fft_kw).astype(np.float32)
        F = sfft.rfftn(mm * delta + noise, **fft_kw)
        del delta
        p = (F.real ** 2 + F.imag ** 2).ravel() * g['herm']
        del F
        sum1 = np.bincount(g['idx1'], p, g['n1'] + 1)[:-1]
        out[name] = {f: (coarse_sum(sum1, cm) if cm is not None else
                         np.bincount(g['idx'][f], p, g['nedges'][f])[:-1]) * g['scale'] / g['nmodes'][f] - g['sn']
                     for f, cm in g['cmap'].items()}
    return i, out


def n_workers(cells, requested=None, bytes_per_cell=56, mem_fraction=0.7):
    """worker processes that fit in memory (~56 bytes per cell per realisation in flight)"""
    if requested:
        return requested
    try:
        mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
    except (ValueError, OSError):
        mem = 64e9
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    return max(1, min(cpus, int(mem_fraction * mem / (bytes_per_cell * cells))))


def coarse_sum(x, cm):
    """sums of the fine-bin values x over the coarse bins whose edges are the fine edges cm"""
    c = np.concatenate([[0.0], np.cumsum(x)])
    return c[cm[1:]] - c[cm[:-1]]


def coarse_map(e1, ef):
    """indices of the fine edges e1 at the coarse edges ef, if they nest (else None)."""
    pos = [int(np.argmin(np.abs(e1 - e))) for e in ef]
    return pos if np.allclose(e1[pos], ef, rtol=0, atol=1e-6) else None


# ----------------------------------------------------------------------------- main
def setup(args):
    paths = dc.Paths(kind='holi_v3', mock=173)
    paths.catalog_names.update({'ELG_LOPnotqso': 'ELGnotqso', 'LRG+ELG_LOPnotqso': 'LRG+ELGnotqso'})
    paths.loader = 'auto'
    paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
    out = os.path.expanduser(f'~/thecov_desi/{paths.kind}_mock{paths.mock}')
    if args.cs_version or args.spectra_dir or args.mock is not None:
        paths.cs_version = args.cs_version or paths.cs_version
        paths.spectra_dir = args.spectra_dir or paths.spectra_dir
        paths.mock = paths.mock if args.mock is None else args.mock
        out = os.path.expanduser(f'~/thecov_desi/{paths.cs_version}_mock{paths.mock}')
    cfg = pl.Config()
    if args.synthetic:
        paths = dc.Paths(catalog_dir=args.synthetic + '/catalogs', spectra_dir=args.synthetic + '/spectra', loader='files')
        dc.TRACER_SPECS.clear(); dc.TRACER_SPECS['TEST'] = ('LRG', (0.4, 0.6))
        cfg = dataclasses.replace(cfg, surface_density=150.0, kmax=0.1, target_near_pairs=3e8, n_sub_far=5000)
        out = args.synthetic + '/results_script'
    return paths, cfg, out


def shell_index(grid, edges):
    idx = np.digitize(grid.kmag, edges) - 1
    idx[(idx < 0) | (idx >= len(edges) - 1)] = len(edges) - 1               # overflow bin
    return idx.ravel()


def run_region(paths, cfg, OUT, b, r, args):
    tracer, zr = dc.TRACER_SPECS[b]
    fns = paths.spectra_fns(tracer, zr, r)
    specs = {f: dc.read_spectra(fns, kmin=cfg.kmin, kmax=cfg.kmax, rebin=f * cfg.rebin, ells=cfg.ells) for f in (1, 2, 4)}
    s1 = specs[1]
    norm, num_sn = float(s1['norm'].mean()), float(s1['num_shotnoise'].mean())
    log(f'{b} {r}: {len(s1["vectors"])} mock spectra, norm {norm:.4g}, num_shotnoise {num_sn:.4g}')

    # thecov, isotropic model (P2 = P4 = 0), with the cached windows of the validation runs
    rc1 = dc.load_region(paths, b, r, n_random_files=cfg.n_random_files)
    tr, _ = dc.build_tracer(f'{b}_{r}', [rc1], nw='random-density', n_randoms_max=cfg.n_randoms_max // 2,
                            surface_density_deg2=cfg.surface_density, verbose=False, shotnoise_target=num_sn)
    wfile = pl.windows_path(OUT, b, tr.name)
    if not os.path.exists(wfile):
        log(f'  no cached windows {wfile}: computing them (pair counts)')
    var_thecov, I_thecov = {}, None
    for f, s in specs.items():
        nb = len(s['k'])
        s_iso = copy.deepcopy(s)
        s_iso['vectors'][:, nb:] = 0.0
        n_near, _ = dc.suggest_n_near(tr, target_pairs=cfg.target_near_pairs)
        C, cov = dc.thecov_covariance(tr, s_iso, n_sub=cfg.n_sub_far, n_near=n_near, verbose=False, windows_file=wfile,
                                      model_norm_correction=cfg.model_norm_correction)
        var_thecov[f] = np.diag(C)[:nb]
        I_thecov = cov.I_randoms
    pos_tr = tr.pos
    del rc1
    log(f'  thecov (isotropic model) done; int m^2 (thecov) / norm = {I_thecov / norm:.4f}')

    # window maps from many random files
    rcK = dc.load_region(paths, b, r, n_random_files=args.random_files)
    win = SeparableWindow(rcK, num_sn, nside=args.nside, dz=args.dz, n_files=args.random_files,
                          surface_density=cfg.surface_density)
    occ = win.fill > 0
    log(f'  pixel fill fraction: median {np.median(win.fill[occ]):.3f}, mean {np.mean(win.fill[occ]):.3f} over occupied pixels')
    log(f'  maps: {len(rcK.randoms["Z"])} randoms ({args.random_files} files), nside {args.nside} '
        f'({win.npix_occupied} occupied pixels), {len(win.rad_m)} z shells')
    del rcK
    grid = Grid(pos_tr, args.cell, args.pad)
    m, S, frac = grid.fill(win, sub=args.sub)
    vc = grid.vc
    windows = {'maps': m}
    if not args.maps_only:
        # thecov's own m (smoothed: 32 nearest randoms in 3D, one n(z) per cap), diluted by the same
        # sub-pixel fill fraction as the maps (veto holes smaller than a pixel)
        inside = np.flatnonzero(frac.ravel() > 0)
        _, nn = tr.tree.query(grid.centers(inside), k=1, workers=-1)
        mt = np.zeros(m.size, np.float32)
        mt[inside] = tr.mw[nn] * frac.ravel()[inside]
        windows['thecov_m'] = mt.reshape(m.shape)
        del mt, nn, inside
    del frac, tr
    wstats = {}
    for name, mm in windows.items():
        I1 = float(np.sum(mm, dtype=float) * vc)
        I2, I4 = float(np.sum(mm.astype(float) ** 2) * vc), float(np.sum(mm.astype(float) ** 4) * vc)
        wstats[name] = dict(int_m=I1, I2_over_norm=I2 / norm, veff=I4 / I2 ** 2)
        log(f'  mesh window [{name}]: int m = {I1:.4g} (alpha sum w_r = {win.int_m:.4g}), '
            f'int m^2 / norm = {I2 / norm:.4f} (thecov {I_thecov / norm:.4f}), 1/V_eff = {I4 / I2 ** 2:.4g}')
    log(f'  int S = {np.sum(S, dtype=float) * vc:.4g} (num_shotnoise {num_sn:.4g})')

    # input power: thecov's model, P0 only (isotropic), extrapolated beyond the measured range
    k1, P0 = s1['k'], s1['vectors'][:, :len(s1['k'])].mean(0) * norm / I_thecov
    kg = grid.kmag
    lo_k, hi_k = k1[0], k1[-1]
    slope = np.polyfit(np.log(k1[-8:]), np.log(np.abs(P0[-8:])), 1)[0]
    with np.errstate(divide='ignore'):
        Pin = np.where(kg < lo_k, P0[0], np.where(kg > hi_k, P0[-1] * (kg / hi_k) ** slope, np.interp(kg, k1, P0)))
    Pin[0, 0, 0] = 0.0
    amp = np.sqrt(np.maximum(Pin, 0) / vc).astype(np.float32)
    del Pin
    noise_amp = np.sqrt(S / vc).astype(np.float32)
    e1 = np.asarray(specs[1]['k_edges'])
    cmap = {f: coarse_map(e1, np.asarray(specs[f]['k_edges'])) for f in specs}
    cmap = {f: (None if c is None else np.asarray(c)) for f, c in cmap.items()}
    own = [f for f in specs if cmap[f] is None]            # binnings that do not nest in the 0.005 bins
    idx = {f: shell_index(grid, specs[f]['k_edges']) for f in [1] + own}
    herm = grid.herm.ravel()
    nm1 = np.bincount(idx[1], herm, len(e1))[:-1]
    nmodes = {f: (coarse_sum(nm1, np.asarray(cmap[f])) if cmap[f] is not None else
                  np.bincount(idx[f], herm, len(specs[f]['k_edges']))[:-1]) for f in specs}
    nproc = n_workers(int(np.prod(grid.shape)), args.workers)
    _G.update(grid=grid, amp={w: amp for w in windows}, noise_amp=noise_amp, windows=windows, herm=herm, idx1=idx[1], n1=len(e1) - 1,
              idx=idx, cmap=cmap, nedges={f: len(specs[f]['k_edges']) for f in specs}, nmodes=nmodes,
              scale=vc ** 2 / norm, sn=num_sn / norm, seed=args.seed, region_id=args.regions_all.index(r))
    log(f'  {args.nreal} realisations (+ {args.pilot} pilot) on {nproc} worker processes')

    def run(indices, label):
        P = {w: {f: np.zeros((len(indices), len(specs[f]['k']))) for f in specs} for w in windows}
        pos = {i: j for j, i in enumerate(indices)}
        t0 = time.time()
        with mp.get_context('fork').Pool(nproc) as pool:
            for n, (i, out) in enumerate(pool.imap_unordered(_realisation, indices)):
                for w in out:
                    for f in out[w]:
                        P[w][f][pos[i]] = out[w][f]
                if n in (0, nproc - 1) or (n + 1) % 100 == 0:
                    log(f'  {label}: {n + 1}/{len(indices)} done ({time.time() - t0:.0f} s)')
        return P

    # pilot: calibrate each window's input power so that <P_hat> matches the mocks' mean (the variance
    # is only comparable at the same mean power); the factor c(k) is smoothed with a cubic in k
    calib = {}
    if args.pilot:
        Pp = run(list(range(10 ** 6, 10 ** 6 + args.pilot)), 'pilot')
        k1m, mean1 = s1['k'], s1['vectors'][:, :len(s1['k'])].mean(0)
        use = k1m >= 0.03
        amps = {}
        for w in windows:
            c = mean1 / Pp[w][1].mean(0)
            coef = np.polyfit(k1m[use], c[use], 3)
            c_fit = np.polyval(coef, k1m)
            c_grid = np.interp(np.clip(grid.kmag, k1m[use][0], k1m[-1]), k1m, c_fit).astype(np.float32)
            amps[w] = amp * np.sqrt(np.maximum(c_grid, 0))
            calib[w] = dict(k=k1m.tolist(), c=c.tolist(), c_fit=c_fit.tolist())
            log(f'  calibration [{w}]: <P>mocks/<P>grf = ' + ', '.join(
                f'{np.mean(c[(k1m >= lo) & (k1m < hi)]):.3f} ({lo}-{hi})' for lo, hi in KRANGES
                if np.any((k1m >= lo) & (k1m < hi))))
        _G['amp'] = amps
        del Pp
    P_hat = run(list(range(args.nreal)), 'realisations')
    _G.clear()

    static = dict(norm=norm, num_sn=num_sn, I_thecov_over_norm=I_thecov / norm, windows=wstats,
                  cell=args.cell, nside=args.nside, pilot=args.pilot, calibration=calib, binnings={})
    for f, s in specs.items():
        nb = len(s['k'])
        Vm = s['vectors'][:, :nb]
        static['binnings'][f'x{f}'] = dict(
            k=s['k'].tolist(), mean_mocks=Vm.mean(0).tolist(), var_mocks=Vm.var(0, ddof=1).tolist(),
            var_thecov=var_thecov[f].tolist(), nmodes_grid=nmodes[f].tolist(),
            nmodes_files=None if s.get('nmodes') is None else np.asarray(s['nmodes']).tolist())
    return static, {f'{w}|x{f}': P_hat[w][f] for w in windows for f in specs}


def finalize(static, P):
    """results from the static part and the realisations P {'window|xf': (nreal, nbins)}"""
    res = copy.deepcopy(static)
    windows = sorted({key.split('|')[0] for key in P}, key=lambda w: w != 'maps')
    res['nreal'] = len(next(iter(P.values())))
    for key_f, d in res['binnings'].items():
        nb = len(d['k'])
        P_hat = {w: {key_f: P[f'{w}|{key_f}']} for w in windows}
        f = key_f
        for name in windows:
            d[f'mean_grf_{name}'] = P_hat[name][f].mean(0).tolist()
            d[f'var_grf_{name}'] = P_hat[name][f].var(0, ddof=1).tolist()
        if len(windows) == 2:                         # paired ratio maps / thecov_m
            a, b_ = P_hat['maps'][f], P_hat['thecov_m'][f]
            d['var_ratio_maps_over_thecov_m'] = (a.var(0, ddof=1) / b_.var(0, ddof=1)).tolist()
            d['corr_maps_thecov_m'] = [float(np.corrcoef(a[:, j], b_[:, j])[0, 1]) for j in range(nb)]
    return res


def summary(r, res):
    ws = res['windows']
    print(f'\n{r}: {res["nreal"]} realisations, cell {res["cell"]} Mpc/h, nside {res["nside"]}; int m^2/norm: thecov '
          f'{res["I_thecov_over_norm"]:.4f}, ' + ', '.join(f'mesh[{w}] {v["I2_over_norm"]:.4f}' for w, v in ws.items()))
    print('  maps     = m at high resolution (healpix x weighted n(z)): exact Gaussian with the fine window')
    print('  thecov_m = thecov\'s own smoothed m on the same footprint: exact Gaussian with thecov\'s window')
    names = list(ws)
    head = f"{'bins':>5s} {'k range':>11s}"
    for w in names:
        head += f" {'<P>grf/mocks':>13s} {'Var grf/thecov':>15s} {'mocks/grf':>10s}"
    if len(names) == 2:
        head += f" {'maps/thecov_m':>14s}"
    print(head + '   [' + ' | '.join(names) + ']   (+-: grf var)')
    for key, d in res['binnings'].items():
        k = np.asarray(d['k'])
        mo, t, mm = (np.asarray(d[x]) for x in ('var_mocks', 'var_thecov', 'mean_mocks'))
        for lo, hi in KRANGES:
            sel = (k >= lo) & (k < hi)
            if not sel.any():
                continue
            line = f'{key:>5s} {lo:5.2f}-{hi:<5.2f}'
            for w in names:
                g, mg_ = np.asarray(d[f'var_grf_{w}']), np.asarray(d[f'mean_grf_{w}'])
                line += f' {np.mean(mg_[sel] / mm[sel]):13.4f} {np.mean(g[sel] / t[sel]):15.4f} {np.mean(mo[sel] / g[sel]):10.4f}'
            if len(names) == 2:
                line += f" {np.mean(np.asarray(d['var_ratio_maps_over_thecov_m'])[sel]):14.4f}"
            print(line + f'   ({np.sqrt(2 / (res["nreal"] - 1) / sel.sum()):.3f})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--nreal', type=int, default=300, help='realisations (per task with --array)')
    ap.add_argument('--cell', type=float, default=6.0, help='mesh cell [Mpc/h]; Nyquist pi/cell')
    ap.add_argument('--pad', type=float, default=1.5, help='box / footprint extent (avoids periodic wrap)')
    ap.add_argument('--sub', type=int, default=2, help='sub-points per cell side for m and S')
    ap.add_argument('--nside', type=int, default=512)
    ap.add_argument('--dz', type=float, default=0.002)
    ap.add_argument('--random-files', type=int, default=10, help='random files for the window maps')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=None,
                    help='parallel realisations (default: as many as fit in 70%% of the free memory, at most the cores)')
    ap.add_argument('--pilot', type=int, default=120,
                    help='pilot realisations to calibrate each window\'s input power to the mocks\' mean (0: none)')
    ap.add_argument('--maps-only', action='store_true', help="skip the run with thecov's own m")
    ap.add_argument('--cs-version', default=None)
    ap.add_argument('--spectra-dir', default=None)
    ap.add_argument('--mock', type=int, default=None)
    ap.add_argument('--synthetic', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    paths, cfg, OUT = setup(args)
    if args.synthetic:
        args.bin = 'TEST'
    args.regions_all = list(args.regions)
    for r in args.regions:
        static, P = run_region(paths, cfg, OUT, args.bin, r, args)
        res = finalize(static, P)
        fn = os.path.join(OUT, args.bin, f'gaussian_footprint_{r}.json')
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        json.dump(res, open(fn, 'w'))
        summary(r, res)
        log(f'saved {fn}')
    log('done')


if __name__ == '__main__':
    main()
