"""Predict the super-sample covariance of the DESI mocks with thecov.SuperSampleCovariance and test it
against the low-rank excess measured by rank1_excess.py (no fitting of the SSC amplitude).

    python -m desi_validation.ssc_check --bin LRG1 --label holi-kcore2-LRG1 --mode kernel
    python -m desi_validation.ssc_check --bin QSO  --label holi-kcore2-QSO  --mode kernel

Inputs: the dumped mock vectors and covariance of `--label` (report_data_<label>.npz in --dir), one
random file per cap for the window shape, the DESI fiducial cosmology (cosmoprimo) for P_lin and f at
z_eff. b1 is fitted to P2/P0 of the mock mean at 0.02 <= k <= 0.06 (Kaiser), b2 from the Lazeyras et al.
(2016) b2(b1) relation, bs2 = -4/7 (b1 - 1) (all overridable). The BC dilution d_k = I_k / norm is that
of the pair-averaged window (WindowSmoothing cache of the kernel runs) when found, else the Kaiser fit's
amplitude at low k.

Reports per cap and for GCcomb: the predicted coherent amplitudes in the form of rank1_excess
(sigma_P0, sigma_P2/P0, their correlation, from the same off-diagonal fit applied to C_SSC) against the
mocks'; chi2/n, residual diagonal variance ratios and joint parameter variance ratios with C + C_SSC
(with and without the local-average term). Output: <--out>/ssc_<label>.npz (C_SSC per region, sigma^2,
responses, parameters) and the window pair counts in --dir (reused on reruns).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402
from desi_validation.compare_kernel import get, chi2  # noqa: E402
from desi_validation.rank1_excess import fit_rank1, param_ratios, K_RANGES  # noqa: E402

T0 = time.time()
Z_EFF = {'LRG1': 0.510, 'LRG2': 0.706, 'LRG3': 0.919, 'ELG1': 0.955, 'ELG2': 1.317, 'QSO': 1.491, 'BGS': 0.295}


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def kaiser(b1, f):
    b1 = np.asarray(b1, float)
    return np.array([b1 ** 2 + 2 / 3 * b1 * f + f ** 2 / 5, 4 / 3 * b1 * f + 4 / 7 * f ** 2 + 0 * b1, 8 / 35 * f ** 2 + 0 * b1])


def fit_b1(k, P0, P2, plin, f, kmax=0.06):
    """b1 from P2/P0 (Kaiser) and the amplitude d = P0 / (Kaiser_0 P_lin) at k <= kmax"""
    sel = k <= kmax + 1e-9
    ratio = np.mean(P2[sel]) / np.mean(P0[sel])
    bs = np.linspace(0.5, 5, 4501)
    K = kaiser(bs, f)
    b1 = float(bs[np.argmin(np.abs(K[1] / K[0] - ratio))])
    d = float(np.mean(P0[sel] / (kaiser(b1, f)[0] * plin(k[sel]))))
    return b1, d


def kernel_dilution(out_dir, b, r, k_edges, norm, cache_tag):
    """I_k / norm of the pair-averaged window from the kernel runs' cached smoothing, or None"""
    from thecov import WindowSmoothing
    fns = sorted(glob.glob(os.path.join(out_dir, b, f'*{b}_{r}_kernel_*{cache_tag}*.smoothing.npz')) +
                 glob.glob(os.path.join(out_dir, f'*{b}_{r}_kernel_*{cache_tag}*.smoothing.npz')), key=os.path.getmtime)
    if not fns:
        return None, None
    sm = WindowSmoothing().load(fns[-1])
    key = next(iter(sm._pairs))
    return np.asarray(sm.I_k(*key, k_edges), float) / norm, fns[-1]


def summarize(tag, V, C, Cssc, k, nb):
    Cm = np.cov(V.T)
    P0 = V[:, :nb].mean(0)
    A_m, _ = fit_rank1(Cm, C, P0, nb)
    A_p, _ = fit_rank1(C + Cssc, C, P0, nb)
    fmt = lambda A: (f"sigma_P0 {np.sqrt(max(A[(0, 0)][0], 0)) * 100:.2f}%  sigma_P2/P0 {np.sqrt(max(A[(1, 1)][0], 0)) * 100:.2f}%  "
                     f"r02 {A[(0, 1)][0] / np.sqrt(max(A[(0, 0)][0] * A[(1, 1)][0], 1e-30)):+.2f}  "
                     f"A44 {A[(2, 2)][0] * 1e5:.2f}e-5")
    print(f'  {tag}: mocks     {fmt(A_m)}')
    print(f'  {tag}: predicted {fmt(A_p)}')
    allidx = np.arange(V.shape[1])
    M = C + Cssc
    res = []
    for l in range(3):
        s = slice(l * nb, (l + 1) * nb)
        rr = np.diag(Cm)[s] / np.diag(M)[s]
        res.append('[' + ' '.join(f'{rr[(k >= lo) & (k < hi)].mean():.3f}' for lo, hi in K_RANGES) + ']')
    print(f'  {tag}: chi2/n {chi2(V, C, allidx):.4f} -> {chi2(V, M, allidx):.4f} | var ratio vs C+C_SSC '
          f'l=0 {res[0]} l=2 {res[1]} l=4 {res[2]}')
    for kmax in (0.2, 0.3):
        print(f'      kmax {kmax}: joint var ratios (A, A2, alpha, SN) {np.round(param_ratios(V, C, k, nb, kmax), 2)}'
              f' -> {np.round(param_ratios(V, M, k, nb, kmax), 2)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--label', default='holi-kcore2-LRG1')
    ap.add_argument('--mode', default='kernel', help='covariance in the dump the SSC is added to')
    ap.add_argument('--dir', default=os.path.expanduser('~/thecov_desi/holi_v3_mock173'))
    ap.add_argument('--out', default='/global/cfs/cdirs/desicollab/users/oalves/thecov_validation')
    ap.add_argument('--cache-tag', default='_r8p10', help="cache tag of the kernel runs (for I_k)")
    ap.add_argument('--b1', type=float, default=None)
    ap.add_argument('--b2', type=float, default=None)
    ap.add_argument('--bs2', type=float, default=None)
    ap.add_argument('--f', type=float, default=None)
    ap.add_argument('--zeff', type=float, default=None)
    ap.add_argument('--n-near', type=int, default=300000)
    args = ap.parse_args()

    from cosmoprimo.fiducial import DESI
    from scipy.interpolate import CubicSpline
    from thecov import GaussianCovariance, PowerSpectrumModel, SuperSampleCovariance

    paths = dc.Paths(kind='holi_v3', mock=173)
    paths.loader = 'auto'
    paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
    cfg = pl.Config()
    z = np.load(os.path.join(args.dir, f'report_data_{args.label}.npz'))
    b = args.bin
    zeff = args.zeff or Z_EFF[b]
    cosmo = DESI()
    kl = np.logspace(-4.5, 1.0, 3000)
    pl_ = cosmo.get_fourier().pk_interpolator().to_1d(z=zeff)
    Pl = np.asarray(pl_(kl))
    plin = CubicSpline(np.log(kl), Pl)
    plin_k = lambda k: plin(np.log(k))
    f = args.f if args.f is not None else float(cosmo.growth_rate(zeff))
    log(f'{b}: z_eff {zeff}, f {f:.3f}, P_lin from cosmoprimo DESI fiducial')

    out, C_ssc, norms = {}, {}, {}
    for r in ('NGC', 'SGC'):
        V, C, k = get(z, b, r, 1, args.mode)
        nb = len(k)
        e = np.asarray(z[f'{b}/{r}/x1/k_edges'])
        k_edges = np.r_[e[:, 0], e[-1, 1]] if e.ndim == 2 else e
        norm = float(np.mean(z[f'{b}/{r}/x1/norm']))
        norms[r] = norm
        spec = dict(vectors=V, k=k, k_edges=k_edges, ells=(0, 2, 4))
        Pm = V.mean(0)
        b1_fit, d_fit = fit_b1(k, Pm[:nb], Pm[nb:2 * nb], plin_k, f)
        b1 = args.b1 if args.b1 is not None else b1_fit
        b2 = args.b2 if args.b2 is not None else 0.412 - 2.143 * b1 + 0.929 * b1 ** 2 + 0.008 * b1 ** 3
        bs2 = args.bs2 if args.bs2 is not None else -4 / 7 * (b1 - 1)
        d_ker, fn = kernel_dilution(args.dir, b, r, k_edges, norm, args.cache_tag)
        dil = d_ker if d_ker is not None else d_fit
        log(f'{r}: b1 {b1:.3f} (P2/P0 fit {b1_fit:.3f}), b2 {b2:.3f}, bs2 {bs2:.3f}; dilution '
            + (f'from {os.path.basename(fn)}: {d_ker[0]:.3f} ... {d_ker[-1]:.3f}' if d_ker is not None else 'not cached')
            + f'; Kaiser-fit amplitude {d_fit:.3f}')

        rc = dc.load_region(paths, b, r, n_random_files=cfg.n_random_files)
        tr, _ = dc.build_tracer(f'{b}_{r}_ssc', [rc], nw='random-density', n_randoms_max=cfg.n_randoms_max // 2,
                                surface_density_deg2=cfg.surface_density, verbose=False)
        del rc
        cov = GaussianCovariance([tr], k_edges, ells=(0, 2, 4), L_max=4, n_sub=cfg.n_sub_far, n_near=args.n_near,
                                 ds=2.0, ds_pair=10.0, s_split=80.0)
        cov.set_normalization(tr.name, tr.name, norm)
        model = PowerSpectrumModel()
        model.add((tr.name, tr.name), dc.model_from_mocks(spec))
        cov.set_model(model, masked=True)
        res = {}
        for la in (True, False):
            ssc = SuperSampleCovariance(cov, (kl, Pl), b1=b1, f=f, b2=b2, bs2=bs2, local_average=la,
                                        n_near=args.n_near, dilution=dil)
            wfile = os.path.join(args.dir, f'ssc_windows_{b}_{r}.npz')
            if la and os.path.exists(wfile):
                ssc.load_windows(wfile)
            if la:
                ssc.compute_windows(tr.name, verbose=True)
                ssc.save_windows(wfile)
                windows = ssc.windows
            else:
                ssc.windows = windows
            Cs, _ = ssc.covariance([(tr.name, tr.name)], ells=(0, 2, 4))
            res[la] = (ssc, Cs)
        ssc, Cs = res[True]
        sig = ssc.sigma2(tr.name)
        log(f'{r}: sigma^2 ' + ', '.join(f'{x[0]}{x[1]}-{y[0]}{y[1]} {v:.3e}' for (x, y), v in sig.items() if x <= y))
        print(f'\n{b} {r}  (a00 {ssc.a[0, 0]:.3f}, c00 {ssc.c[0, 0]:.3f}, a02 {ssc.a[0, 1]:.3f}, a20 {ssc.a[1, 0]:.3f}, '
              f'a22 {ssc.a[1, 1]:.3f}; R/P_lin)')
        summarize('with LA   ', V, C, Cs, k, nb)
        summarize('without LA', V, C, res[False][1], k, nb)
        C_ssc[r] = Cs
        out[f'{r}/C_ssc'], out[f'{r}/C_ssc_noLA'] = Cs, res[False][1]
        out[f'{r}/sigma2_keys'] = np.array([f'{x}|{y}' for (x, y) in sig])
        out[f'{r}/sigma2'] = np.array(list(sig.values()))
        out[f'{r}/params'] = np.array([b1, b2, bs2, f, zeff])
        out[f'{r}/dilution'] = np.asarray(dil, float) * np.ones(nb)
        del tr, cov

    V, C, k = get(z, b, 'GCcomb', 1, args.mode)
    Cg = dc.combine_regions([C_ssc['NGC'], C_ssc['SGC']], [norms['NGC'], norms['SGC']])
    print(f'\n{b} GCcomb')
    summarize('with LA   ', V, C, Cg, k, len(k))
    out['GCcomb/C_ssc'] = Cg
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f'ssc_{args.label}.npz')
    np.savez(fn, **out)
    log(f'saved {fn}')


if __name__ == '__main__':
    main()
