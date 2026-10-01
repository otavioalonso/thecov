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

    def __init__(self, rc, num_sn, nside=512, dz=0.002, cosmo=None):
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
        """m and S at cartesian positions x (n, 3)."""
        rr = np.sqrt(np.sum(x ** 2, axis=1))
        out_m, out_s = np.zeros(len(x)), np.zeros(len(x))
        ok = (rr > self.chi_min) & (rr < self.chi_max)
        if not ok.any():
            return out_m, out_s
        xo, ro = x[ok], rr[ok]
        z = np.interp(ro, self.chi_grid, self.z_grid)
        iz = np.clip(np.searchsorted(self.zedges, z) - 1, 0, len(self.rad_m) - 1)
        theta = np.arccos(np.clip(xo[:, 2] / ro, -1, 1))
        phi = np.arctan2(xo[:, 1], xo[:, 0])
        pix = self.hp.ang2pix(self.nside, theta, phi)
        out_m[ok] = self.aw[pix] * self.rad_m[iz]
        out_s[ok] = self.aw2[pix] * self.rad_s[iz]
        return out_m, out_s


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
        """m and S averaged over sub^3 points per cell."""
        m = np.zeros(self.shape, np.float32)
        s = np.zeros(self.shape, np.float32)
        offs = (np.arange(sub) + 0.5) / sub
        ny, nz = self.shape[1], self.shape[2]
        jy, jz = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
        for i in range(self.shape[0]):
            acc_m = np.zeros(ny * nz)
            acc_s = np.zeros(ny * nz)
            for ox in offs:
                for oy in offs:
                    for oz in offs:
                        x = np.stack([np.full(ny * nz, self.lo[0] + (i + ox) * self.cell),
                                      self.lo[1] + (jy.ravel() + oy) * self.cell,
                                      self.lo[2] + (jz.ravel() + oz) * self.cell], axis=1)
                        a, b = win.at(x)
                        acc_m += a
                        acc_s += b
            m[i] = (acc_m / sub ** 3).reshape(ny, nz)
            s[i] = (acc_s / sub ** 3).reshape(ny, nz)
        return m, s

    def rfft(self, f):
        return sfft.rfftn(f, workers=self.workers)

    def irfft(self, F):
        return sfft.irfftn(F, s=self.shape, workers=self.workers)


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
    del rc1, tr
    log(f'  thecov (isotropic model) done; int m^2 (thecov) / norm = {I_thecov / norm:.4f}')

    # window maps from many random files
    rcK = dc.load_region(paths, b, r, n_random_files=args.random_files)
    win = SeparableWindow(rcK, num_sn, nside=args.nside, dz=args.dz)
    log(f'  maps: {len(rcK.randoms["Z"])} randoms ({args.random_files} files), nside {args.nside} '
        f'({win.npix_occupied} occupied pixels), {len(win.rad_m)} z shells')
    del rcK
    grid = Grid(pos_tr, args.cell, args.pad)
    m, S = grid.fill(win, sub=args.sub)
    vc = grid.vc
    I2, I4 = float(np.sum(m.astype(float) ** 2) * vc), float(np.sum(m.astype(float) ** 4) * vc)
    log(f'  mesh window: int m = {np.sum(m) * vc:.4g} (alpha sum w_r = {win.int_m:.4g}), '
        f'int m^2 / norm = {I2 / norm:.4f} (thecov {I_thecov / norm:.4f}), int S = {np.sum(S) * vc:.4g} '
        f'(num_shotnoise {num_sn:.4g}); 1/V_eff = {I4 / I2 ** 2:.4g}')

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
    idx = {f: shell_index(grid, specs[f]['k_edges']) for f in specs}
    herm = grid.herm.ravel()
    nmodes = {f: np.bincount(idx[f], herm, len(specs[f]['k_edges']))[:-1] for f in specs}

    rng = np.random.default_rng(args.seed + (0 if r == 'NGC' else 1000))
    P_hat = {f: np.zeros((args.nreal, len(specs[f]['k']))) for f in specs}
    t0 = time.time()
    for i in range(args.nreal):
        delta = grid.irfft(grid.rfft(rng.standard_normal(grid.shape, dtype=np.float32)) * amp).astype(np.float32)
        F = m * delta
        F += noise_amp * rng.standard_normal(grid.shape, dtype=np.float32)
        p = np.abs(grid.rfft(F)) ** 2 * (vc ** 2 / norm)
        p = p.ravel() * herm
        for f in specs:
            P_hat[f][i] = np.bincount(idx[f], p, len(specs[f]['k_edges']))[:-1] / nmodes[f] - num_sn / norm
        if i in (0, 4) or (i + 1) % 50 == 0:
            log(f'  realisation {i + 1}/{args.nreal} ({(time.time() - t0) / (i + 1):.1f} s each)')

    res = dict(norm=norm, num_sn=num_sn, I_thecov_over_norm=I_thecov / norm, I_mesh_over_norm=I2 / norm,
               veff_mesh=I4 / I2 ** 2, nreal=args.nreal, cell=args.cell, nside=args.nside, binnings={})
    for f, s in specs.items():
        nb = len(s['k'])
        Vm = s['vectors'][:, :nb]
        res['binnings'][f'x{f}'] = dict(k=s['k'].tolist(), mean_grf=P_hat[f].mean(0).tolist(), mean_mocks=Vm.mean(0).tolist(),
                                        var_grf=P_hat[f].var(0, ddof=1).tolist(), var_mocks=Vm.var(0, ddof=1).tolist(),
                                        var_thecov=var_thecov[f].tolist(), nmodes_grid=nmodes[f].tolist(),
                                        nmodes_files=None if s.get('nmodes') is None else np.asarray(s['nmodes']).tolist())
    return res


def summary(r, res):
    print(f'\n{r}: int m^2/norm thecov {res["I_thecov_over_norm"]:.4f}, mesh {res["I_mesh_over_norm"]:.4f}; '
          f'{res["nreal"]} realisations, cell {res["cell"]} Mpc/h, nside {res["nside"]}')
    print(f"{'bins':>5s} {'k range':>11s} {'<P>grf/<P>mocks':>16s} {'Var grf/thecov':>15s} {'Var mocks/grf':>14s} "
          f"{'Var mocks/thecov':>17s}  (+-, grf var)")
    for key, d in res['binnings'].items():
        k = np.asarray(d['k'])
        g, mo, t = (np.asarray(d[x]) for x in ('var_grf', 'var_mocks', 'var_thecov'))
        mg_, mm = np.asarray(d['mean_grf']), np.asarray(d['mean_mocks'])
        for lo, hi in KRANGES:
            s = (k >= lo) & (k < hi)
            if not s.any():
                continue
            err = np.sqrt(2 / (res['nreal'] - 1) / s.sum())
            print(f'{key:>5s} {lo:5.2f}-{hi:<5.2f} {np.mean(mg_[s] / mm[s]):16.4f} {np.mean(g[s] / t[s]):15.4f} '
                  f'{np.mean(mo[s] / g[s]):14.4f} {np.mean(mo[s] / t[s]):17.4f}  ({err:.3f})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--nreal', type=int, default=300)
    ap.add_argument('--cell', type=float, default=6.0, help='mesh cell [Mpc/h]; Nyquist pi/cell')
    ap.add_argument('--pad', type=float, default=1.5, help='box / footprint extent (avoids periodic wrap)')
    ap.add_argument('--sub', type=int, default=2, help='sub-points per cell side for m and S')
    ap.add_argument('--nside', type=int, default=512)
    ap.add_argument('--dz', type=float, default=0.002)
    ap.add_argument('--random-files', type=int, default=10, help='random files for the window maps')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cs-version', default=None)
    ap.add_argument('--spectra-dir', default=None)
    ap.add_argument('--mock', type=int, default=None)
    ap.add_argument('--synthetic', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    paths, cfg, OUT = setup(args)
    if args.synthetic:
        args.bin = 'TEST'
    out = {}
    for r in args.regions:
        res = run_region(paths, cfg, OUT, args.bin, r, args)
        out[r] = res
        fn = os.path.join(OUT, args.bin, f'gaussian_footprint_{r}.json')
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        json.dump(res, open(fn, 'w'))
        summary(r, res)
        log(f'saved {fn}')
    log('done')


if __name__ == '__main__':
    main()
