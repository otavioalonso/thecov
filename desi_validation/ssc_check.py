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

Also the discreteness 4-point terms (thecov.DiscretenessCovariance) and the tree-level trispectrum
(thecov.TrispectrumCovariance; Galileon biases from b1, b2, bs2 above, b3 from Lazeyras et al. b3(b1),
bGamma3 = 23/42 (b1 - 1); --no-t0 to skip; --workers processes), with their components saved separately
(C_disc_B, C_disc_P, C_T0_snake, C_T0_star, C_T0_star_b3 = d C_T0 / d b3) for amplitude fits.

Recommended non-Gaussian covariance (saved as <cap>/C_nongauss, nothing fitted): SSC (with the local average) +
discreteness 4-point terms, both built from A P_lin (A: the mock amplitude, fit_amplitude), BAO- and FoG-damped,
the discreteness terms window-convolved (thecov.covariance_tools.window_convolve; --no-window-local to skip).
The response-based T0 (tree level for k < --k-split, squeezed couplings + their completion, IR-safe collapsed term)
is computed and reported as a diagnostic (<cap>/C_T0_*); -2 dlnL tables against the mocks for every variant;
--fit-templates adds maximum-likelihood template amplitudes (diagnostic). All inputs are saved for local rebuilds
(desi_validation/ssc_dev/local_rebuild.py); bundle with desi_validation/export_bundle.sh.

Damping (default; --no-damping for the old tree level): BAO damping of P_lin (thecov.power.ir_damped,
--ir-sigma) and Gaussian fingers-of-God sigma_v fitted to the mock-mean P2/P0 (--sigma-fog to fix it), applied to
the SSC responses (p_dressed), the discreteness B and P and T0.

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


def fit_sigma_v(k, P0, P2, b1, f, kmin=0.05, kmax=0.3):
    """Gaussian fingers-of-God sigma_v [Mpc/h] from the mock-mean P2/P0 (Kaiser x exp(-(k mu sigma_v)^2), b1 fixed)"""
    from scipy.optimize import minimize_scalar
    x, w = np.polynomial.legendre.leggauss(32)
    L2 = 2.5 * (3 * x ** 2 - 1)
    sel = (k >= kmin) & (k <= kmax)
    ratio = P2[sel] / P0[sel]

    def model(sig):
        K = (b1 + f * x ** 2) ** 2 * np.exp(-(k[sel, None] * x * sig) ** 2)
        return np.sum(w * K * L2, 1) / np.sum(w * K, 1)
    return float(minimize_scalar(lambda sg: np.sum((model(sg) - ratio) ** 2), bounds=(0, 15), method='bounded').x)


def fit_amplitude(k, P0, dil, b1, f, sigma_v, plin, kmin=0.02, kmax=0.08):
    """A = P0_mocks / (dilution x Kaiser-FoG(b1, f, sigma_v) x P_lin), median over kmin <= k <= kmax: the factor that
    normalises the linear galaxy power of the model to the mocks (b1 is fixed by P2/P0; A absorbs sigma_8^2 or b1^2
    offsets). All non-Gaussian terms are built from A P_lin."""
    x, w = np.polynomial.legendre.leggauss(32)
    sel = (k >= kmin) & (k <= kmax)
    K = (b1 + f * x ** 2) ** 2 * np.exp(-(k[sel, None] * x * sigma_v) ** 2)
    model = np.atleast_1d(dil * np.ones_like(k))[sel] * plin(k[sel]) * np.sum(w * K, 1) / 2
    return float(np.median(P0[sel] / model))


TEMPLATE_CASES = [  # (name, templates held at 1, templates fitted); the rest are 0 (diagnostic only)
    ('fit: SSC, disc', [], ['ssc', 'disc']),
    ('fit: SSC, disc, T0 tree', [], ['ssc', 'disc', 'T0tree']),
    ('fit: SSC, disc_B, disc_P, T0 tree', [], ['ssc', 'disc_B', 'disc_P', 'T0tree']),
    ('fit: SSC, disc, T0 response', [], ['ssc', 'disc', 'T0resp']),
    ('fit: SSC, disc_B, disc_P, T0 pieces', [],
     ['ssc', 'disc_B', 'disc_P', 'T0_LL', 'T0_LH', 'T0_completion', 'T0_collapsed', 'T0_HH']),
]


def loglike_table(tag, V, C, variants):
    """-2 Delta ln L (Wishart, mocks) of each covariance variant against the Gaussian covariance alone; nothing fitted"""
    from thecov import CovarianceTemplates
    S, N = np.cov(V.T), len(V)
    ref = CovarianceTemplates(C).loglike(S, N)
    print(f'  {tag}: -2 dlnL vs the Gaussian covariance (mocks, nothing fitted; lower is better)')
    for name, Cx in variants:
        v = CovarianceTemplates(C + Cx).loglike(S, N)
        print(f'    {name:44s} ' + (f'{v - ref:+9.1f}' if np.isfinite(v) else '   not PD'))


def template_table(tag, V, C, comps, k, nb):
    """Wishart maximum-likelihood amplitudes of the non-Gaussian templates for the mock covariance, and
    -2 Delta ln L against the Gaussian covariance alone (what the sub-volume fit will do on the data)."""
    from thecov import CovarianceTemplates
    comps = dict(comps)
    if 'disc_B' in comps and 'disc' not in comps:
        comps['disc'] = comps['disc_B'] + comps['disc_P']
    if 'T0_LL' in comps and 'T0resp' not in comps:
        comps['T0resp'] = comps['T0_LL'] + comps['T0_LH'] + comps['T0_completion'] + comps.get('T0_collapsed', 0.0)
    S, N = np.cov(V.T), len(V)
    tpl = CovarianceTemplates(C).update(comps)
    zero = {n: 0.0 for n in tpl.names}
    ref = tpl.loglike(S, N, **zero)
    print(f'  {tag}: template amplitudes (Wishart ML on the mocks); -2 dlnL vs Gaussian, chi2/n')
    out = {}
    for name, ones, free in TEMPLATE_CASES:
        if any(t not in tpl.names for t in ones + free):
            continue
        fixed = dict(zero, **{t: 1.0 for t in ones})
        if free:
            amps, m2l = tpl.fit(S, N, free=free, fixed=fixed, start={t: 1.0 for t in free})
        else:
            amps, m2l = fixed, tpl.loglike(S, N, **fixed)
        M = tpl(**amps)
        try:
            c2 = f'{chi2(V, M, np.arange(V.shape[1])):.4f}'
        except np.linalg.LinAlgError:
            c2 = ' n/PD '
        out[name] = {t: amps[t] for t in ones + free}
        val = f'{m2l - ref:+9.1f}' if m2l < 1e299 else '   not PD'
        err = {}
        if free and m2l < 1e299:                             # Fisher errors at the best fit (Wishart, N - 1 dof)
            Mi = np.linalg.inv(M)
            X = [Mi @ tpl.templates[t] for t in free]
            F = np.array([[0.5 * (N - 1) * np.sum(a * b.T) for b in X] for a in X])
            try:
                err = dict(zip(free, np.sqrt(np.diag(np.linalg.inv(F)))))
            except np.linalg.LinAlgError:
                pass
        out[name + ' (errors)'] = err
        print(f'    {name:52s} {val}  {c2}  ' + ', '.join(f'{t}={amps[t]:.2f}' + (f'+-{err[t]:.2f}' if t in err else '')
                                                       for t in ones + free))
    return out


def kernel_dilution(out_dir, b, r, k_edges, norm, cache_tag):
    """I_k / norm of the pair-averaged window of the kernel runs, or (None, None): read from the cached
    kernel covariance (cov_<r>_<b>_<r>_kernel_*<cache_tag>_*.npz, which stores I_k per bin), else from
    the WindowSmoothing cache"""
    nb = len(k_edges) - 1
    fns = sorted(glob.glob(os.path.join(out_dir, b, f'cov_{r}_{b}_{r}_kernel_*{cache_tag}_*.npz')), key=os.path.getmtime)
    for fn in fns[::-1]:
        with np.load(fn) as f:
            if 'I_k' in f.files and np.size(f['I_k']) == nb:
                return np.asarray(f['I_k'], float) / norm, fn
    from thecov import WindowSmoothing
    fns = sorted(glob.glob(os.path.join(out_dir, b, f'*{b}_{r}_kernel_*{cache_tag}*.smoothing.npz')) +
                 glob.glob(os.path.join(out_dir, f'*{b}_{r}_kernel_*{cache_tag}*.smoothing.npz')), key=os.path.getmtime)
    for fn in fns[::-1]:
        try:
            sm = WindowSmoothing().load(fn)
            key = next(iter(sm._pairs))
            return np.asarray(sm.I_k(*key, k_edges), float) / norm, fn
        except Exception:
            continue
    return None, None


def min_whitened_eig(C, M):
    """smallest eigenvalue of C^-1/2 M C^-1/2 (M positive definite iff > 0)"""
    L = np.linalg.cholesky(C)
    Li = np.linalg.inv(L)
    return float(np.linalg.eigvalsh(Li @ M @ Li.T).min())


def summarize(tag, V, C, Cssc, k, nb):
    Cm = np.cov(V.T)
    P0 = V[:, :nb].mean(0)
    A_m, _ = fit_rank1(Cm, C, P0, nb)
    A_p, _ = fit_rank1(C + Cssc, C, P0, nb)

    def fmt(A):
        a, b = A[(0, 0)][0], A[(1, 1)][0]
        r = f'{A[(0, 1)][0] / np.sqrt(a * b):+.2f}' if min(a, b) > 1e-3 * max(abs(a), abs(b), 1e-12) else '  n/a'
        return (f"sigma_P0 {np.sign(a) * np.sqrt(abs(a)) * 100:+.2f}%  sigma_P2/P0 {np.sign(b) * np.sqrt(abs(b)) * 100:+.2f}%  "
                f"r02 {r}  A44 {A[(2, 2)][0] * 1e5:.2f}e-5")
    print(f'  {tag}: mocks     {fmt(A_m)}')
    print(f'  {tag}: predicted {fmt(A_p)}   (negative sigma = negative fitted amplitude)')
    M = C + Cssc
    lam = min_whitened_eig(C, M)
    res = []
    for l in range(3):
        s = slice(l * nb, (l + 1) * nb)
        rr = np.diag(Cm)[s] / np.diag(M)[s]
        res.append('[' + ' '.join(f'{rr[(k >= lo) & (k < hi)].mean():.3f}' for lo, hi in K_RANGES) + ']')
    print(f'  {tag}: min eig of C^-1/2 (C+C_x) C^-1/2 = {lam:.4f} | var ratio vs C+C_x '
          f'l=0 {res[0]} l=2 {res[1]} l=4 {res[2]}')
    if lam <= 0:
        print(f'  {tag}: C + C_x NOT positive definite -> chi2 / parameter ratios skipped')
        return
    allidx = np.arange(V.shape[1])
    print(f'  {tag}: chi2/n {chi2(V, C, allidx):.4f} -> {chi2(V, M, allidx):.4f}')
    for kmax in (0.2, 0.3):
        print(f'      kmax {kmax}: joint var ratios (A, A2, alpha, SN) {np.round(param_ratios(V, C, k, nb, kmax), 2)}'
              f' -> {np.round(param_ratios(V, M, k, nb, kmax), 2)}')


NORM_K = (0.05, 0.1, 0.2)


def norm_statistics(tag, V, norm_arr, k, nb, variants, C_total, A):
    """The direct, fit-free test of the local-average term: sigma(delta_norm) and the regression slope of each
    P_hat_l(k) on delta_norm = norm_i / <norm> - 1 over the mocks, against SuperSampleCovariance.local_average_statistics
    for each normalisation convention (`variants`: {norm_kind: ssc}). The slope's sign is the net (beat coupling minus
    local average) response: tree level gives ~ -0.2 P_0 for 'data-randoms' and ~ +0.5 P_0 for 'alpha' (b1 ~ 2)."""
    dn = norm_arr / norm_arr.mean() - 1.0
    Vc = V - V.mean(0)
    cov_m = (Vc * dn[:, None]).mean(0) * len(V) / (len(V) - 1)
    slope_m = cov_m / dn.var(ddof=1) / V.mean(0)
    corr_m = cov_m / (V.std(0, ddof=1) * dn.std(ddof=1))
    idx = [int(np.argmin(np.abs(k - kk))) for kk in NORM_K]
    print(f'  {tag}: normalisation per mock: sigma(delta_norm) mocks {dn.std(ddof=1) * 100:.3f}%')
    for l, lab in ((0, 'l=0'), (1, 'l=2')):
        print(f'      mocks      {lab}: slope dP/dnorm / P at k ~ {NORM_K}: '
              + ' '.join(f'{slope_m[l * nb + i]:+.2f}' for i in idx)
              + ' | corr(P_hat, delta_norm): ' + ' '.join(f'{corr_m[l * nb + i]:+.2f}' for i in idx))
    for kind, ssc in variants.items():
        try:
            st = ssc.local_average_statistics(A, ells=(0, 2, 4), C_total=C_total)
        except Exception as e:                                  # diagnostic only: never stop the run
            print(f'      {kind:12s}: local_average_statistics failed: {e}')
            continue
        print(f'      {kind:12s}: sigma(delta_norm) {st["sigma_norm"] * 100:.3f}% (clustering {st["sigma_norm_clustering"] * 100:.3f}%,'
              f' Poisson {st["sigma_norm_poisson"] * 100:.3f}%)')
        for l, lab in ((0, 'l=0'), (1, 'l=2')):
            print(f'                    {lab}: slope ' + ' '.join(f'{st["slope"][l * nb + i]:+.2f}' for i in idx)
                  + ' | corr ' + ' '.join(f'{st["corr_total"][l * nb + i]:+.2f}' for i in idx)
                  + ' | corr of the long-mode parts ' + ' '.join(f'{st["corr"][l * nb + i]:+.2f}' for i in idx))
    return dict(sigma_norm_mocks=dn.std(ddof=1), slope_mocks=slope_m, corr_mocks=corr_m)


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
    ap.add_argument('--norm-kind', default='data-randoms', choices=('data-randoms', 'randoms', 'alpha'),
                    help="what jaxpower's realised norm contains (see thecov.ssc.SuperSampleCovariance); all three are "
                         "compared with the mocks' sigma(delta_norm) and P_hat-delta_norm slopes in the log")
    ap.add_argument('--b3', type=float, default=None, help='T0: Galileon-basis b3 (default 0; the Lazeyras b3(b1) is in another basis)')
    ap.add_argument('--workers', type=int, default=min(64, os.cpu_count() or 1), help='T0: processes')
    ap.add_argument('--no-t0', action='store_true', help='skip the tree-level trispectrum')
    ap.add_argument('--sigma-fog', type=float, default=None, help='FoG sigma_v [Mpc/h] (default: fit to the mock P2/P0)')
    ap.add_argument('--ir-sigma', type=float, default=6.0, help='BAO damping scale of P_lin [Mpc/h]')
    ap.add_argument('--no-damping', action='store_true', help='tree-level responses with the undamped P_lin (old)')
    ap.add_argument('--no-normalize', action='store_true', help='do not normalise A P_lin to the mock amplitude')
    ap.add_argument('--k-split', type=float, default=0.06, help='T0 response split: long bins k < k_split')
    ap.add_argument('--no-window-local', action='store_true', help='do not window-convolve the local-approximation terms')
    ap.add_argument('--fit-templates', action='store_true', help='also fit template amplitudes to the mocks (diagnostic)')
    ap.add_argument('--cs-version', default='holi-v3-altmtl', help='mock catalogues (e.g. abacus-2ndgen-dr2-altmtl)')
    ap.add_argument('--mock', type=int, default=173, help='mock whose randoms / footprint are used (abacus: 0)')
    args = ap.parse_args()

    from cosmoprimo.fiducial import DESI
    from scipy.interpolate import CubicSpline
    from thecov import (GaussianCovariance, PowerSpectrumModel, SuperSampleCovariance, DiscretenessCovariance,
                        TrispectrumCovariance, galileon_bias, response_split, response_covariance)
    from thecov.covariance_tools import window_convolve

    paths = dc.Paths(kind='holi_v3', mock=args.mock)
    paths.loader = 'auto'
    paths.cs_version, paths.cs_parent_version = args.cs_version, 'data-dr2-v2'
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
    from thecov.power import no_wiggle, ir_damped, Dressed
    try:
        h, om, fb, ns = cosmo.h, cosmo.Omega0_m * cosmo.h ** 2, cosmo.Omega0_b / cosmo.Omega0_m, cosmo.n_s
    except Exception:                                           # Planck 2018 / DESI fiducial
        h, om, fb, ns = 0.6766, 0.14239, 0.15745, 0.9665
    Pir = Pl if args.no_damping else ir_damped(kl, Pl, no_wiggle(kl, Pl, h, om, fb, n_s=ns), args.ir_sigma)

    out, C_ssc, C_disc, C_t0, norms = {}, {}, {}, {}, {}
    os.makedirs(args.out, exist_ok=True)
    out_fn = os.path.join(args.out, f'ssc_{args.label}.npz')

    def save(d):
        np.savez(out_fn, **d)
        log(f'saved {out_fn}')
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
        sigma_v = 0.0 if args.no_damping else (args.sigma_fog if args.sigma_fog is not None
                                                  else fit_sigma_v(k, Pm[:nb], Pm[nb:2 * nb], b1, f))
        d_ker, dil_fn = kernel_dilution(args.dir, b, r, k_edges, norm, args.cache_tag)
        dil = d_ker if d_ker is not None else d_fit
        A_norm = 1.0 if args.no_normalize else fit_amplitude(k, Pm[:nb], dil, b1, f, sigma_v, plin_k)
        PlA, PirA = A_norm * Pl, A_norm * Pir
        damp = dict(p_dressed=None if args.no_damping else Dressed(kl, PirA, sigma_v))
        log(f'{r}: damping: ' + ('none (tree level, undamped P_lin)' if args.no_damping else
                                 f'BAO Sigma {args.ir_sigma} Mpc/h, FoG sigma_v {sigma_v:.2f} Mpc/h (fit to the mock P2/P0)')
            + f'; linear amplitude normalised to the mocks: A = {A_norm:.3f}' + (' (off)' if args.no_normalize else ''))
        log(f'{r}: b1 {b1:.3f} (P2/P0 fit {b1_fit:.3f}), b2 {b2:.3f}, bs2 {bs2:.3f}; dilution '
            + (f'from {os.path.basename(dil_fn)}: {d_ker[0]:.3f} ... {d_ker[-1]:.3f}' if d_ker is not None else 'not cached')
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
        # variants: LA with its Poisson self-calibration term, LA without it, no LA
        for key, la, pois in (('LA', True, True), ('LA_noPoisson', True, False), ('noLA', False, False)):
            ssc = SuperSampleCovariance(cov, (kl, PlA), b1=b1, f=f, b2=b2, bs2=bs2, local_average=la, **damp,
                                        n_near=args.n_near, dilution=dil, discreteness=pois, norm_kind=args.norm_kind)
            wfile = os.path.join(args.dir, f'ssc_windows_{b}_{r}.npz')
            if key == 'LA':
                if os.path.exists(wfile):
                    ssc.load_windows(wfile)
                ssc.compute_windows(tr.name, verbose=True)
                ssc.save_windows(wfile)
                windows = ssc.windows
            else:
                ssc.windows = windows
            Cs, _ = ssc.covariance([(tr.name, tr.name)], ells=(0, 2, 4))
            res[key] = (ssc, Cs)
        ssc, Cs = res['LA']
        sig = ssc.sigma2(tr.name)
        var, J, J3 = ssc.discreteness_integrals(tr.name)
        log(f'{r}: discreteness: sigma_P(eps_norm) = {np.sqrt(var) * 100:.3f}% (alpha + data in norm), '
            f'2 J / norm = {2 * J / norm:.3e}, J3 / norm = {J3 / norm:.3e}')
        log(f'{r}: sigma^2 ' + ', '.join(f'{x[0]}{x[1]}-{y[0]}{y[1]} {v:.3e}' for (x, y), v in sig.items() if x <= y))
        print(f'\n{b} {r}  (a00 {ssc.a[0, 0]:.3f}, c00 {ssc.c[0, 0]:.3f}, a02 {ssc.a[0, 1]:.3f}, a20 {ssc.a[1, 0]:.3f}, '
              f'a22 {ssc.a[1, 1]:.3f}; R/P_lin)')
        norm_arr = np.asarray(z[f'{b}/{r}/x1/norm'], float).ravel()
        if norm_arr.size == len(V):
            kinds = {}
            for kind in ('data-randoms', 'randoms', 'alpha'):
                sk = SuperSampleCovariance(cov, (kl, PlA), b1=b1, f=f, b2=b2, bs2=bs2, local_average=True, **damp,
                                           n_near=args.n_near, dilution=dil, discreteness=True, norm_kind=kind)
                sk.windows = windows
                kinds[kind] = sk
            ns = norm_statistics(f'{b} {r}', V, norm_arr, k, nb, kinds, C + Cs, tr.name)
            out[f'{r}/norm_stats_mocks'] = np.array([ns['sigma_norm_mocks']])
            out[f'{r}/norm_slope_mocks'], out[f'{r}/norm_corr_mocks'] = ns['slope_mocks'], ns['corr_mocks']
        else:
            log(f'{r}: no per-mock norm array in the dump ({norm_arr.size} values for {len(V)} mocks): norm statistics skipped')
        disc = DiscretenessCovariance(cov, (kl, PirA), b1=b1, f=f, b2=b2, bs2=bs2, sigma_fog=sigma_v,
                                      **(dict(n_mu=16, n_phi=24) if sigma_v else {}))
        dcomp = disc.components([(tr.name, tr.name)], ells=(0, 2, 4))
        Cd_loc = dcomp['B'] + dcomp['P']
        wconv = (lambda X: X) if args.no_window_local else (lambda X: window_convolve(X, C))
        Cd = wconv(Cd_loc)                                   # the discreteness terms with the window's mode mixing
        out[f'{r}/C_disc_B'], out[f'{r}/C_disc_P'] = dcomp['B'], dcomp['P']
        out[f'{r}/C_disc_local'], out[f'{r}/C_disc'] = Cd_loc, Cd
        J_Smm, J_SS = disc.window_integrals(tr.name)
        log(f'{r}: discreteness 4-point: J_Smm {J_Smm:.3e}, J_SS {J_SS:.3e}; diag(C_disc)/diag(C) l=0 at k~0.05, 0.15, 0.25: '
            + ', '.join(f'{Cd[i, i] / C[i, i]:.3f}' for i in (6, 26, 46))
            + ('' if args.no_window_local else ' (window-convolved)'))
        C_disc[r] = Cd
        C_ssc[r] = Cs
        out[f'{r}/C_ssc'], out[f'{r}/C_ssc_LA_noPoisson'], out[f'{r}/C_ssc_noLA'] = Cs, res['LA_noPoisson'][1], res['noLA'][1]
        out[f'{r}/C'] = C
        out[f'{r}/C_nongauss'] = Cs + Cd                    # the recommended non-Gaussian covariance (nothing fitted)
        out[f'{r}/sigma2_keys'] = np.array([f'{x}|{y}' for (x, y) in sig])
        out[f'{r}/sigma2'] = np.array(list(sig.values()))
        out[f'{r}/params'] = np.array([b1, b2, bs2, f, zeff])
        out[f'{r}/damping'] = np.array([0.0 if args.no_damping else args.ir_sigma, sigma_v, A_norm])
        out[f'{r}/dilution'] = np.asarray(dil, float) * np.ones(nb)
        # inputs, so that every term can be rebuilt locally (desi_validation/ssc_dev/local_rebuild.py)
        out[f'{r}/plin_k'], out[f'{r}/plin_P'], out[f'{r}/plin_P_damped_normalised'] = kl, Pl, PirA
        out[f'{r}/J_disc'] = np.array([J_Smm, J_SS])
        out[f'{r}/LA_integrals'] = np.array([var, J, J3, norm])
        out[f'{r}/k'], out[f'{r}/k_edges'], out[f'{r}/mock_mean'] = k, k_edges, V.mean(0)
        save(out)
        variants = [('SSC (LA)', Cs), ('SSC (LA, no Poisson)', res['LA_noPoisson'][1]), ('SSC (no LA)', res['noLA'][1]),
                    ('SSC (LA) + disc 4-pt (local)', Cs + Cd_loc), ('SSC (LA) + disc 4-pt', Cs + Cd)]
        if not args.no_t0:
            b3 = args.b3 if args.b3 is not None else 0.0
            gb = galileon_bias(b1, b2, bs2, b3=b3)
            t0 = TrispectrumCovariance(cov, (kl, PirA), gb, f=f, n_workers=args.workers, sigma_fog=sigma_v,
                                       **(dict(n_mu=20, n_psi=40) if sigma_v else {}))
            tt = time.time()
            tc = t0.components([(tr.name, tr.name)], ells=(0, 2, 4))
            Ccoll = t0.collapsed([(tr.name, tr.name)], args.k_split, ells=(0, 2, 4))
            J4 = t0.window_integral(tr.name)
            Ct = tc['snake'] + tc['star']
            log(f'{r}: T0 ({gb}, J4 {J4:.3e}, {time.time() - tt:.0f} s with {args.workers} processes); '
                'diag(C_T0)/diag(C) l=0 at k~0.05, 0.15, 0.25: ' + ', '.join(f'{Ct[i, i] / C[i, i]:.3f}' for i in (6, 26, 46))
                + '; collapsed (k_split %.3g): ' % args.k_split + ', '.join(f'{Ccoll[i, i] / C[i, i]:.3f}' for i in (6, 26, 46)))
            for key, M in tc.items():
                out[f'{r}/C_T0_{key}'] = M
            out[f'{r}/C_T0_collapsed_local'] = Ccoll
            out[f'{r}/bias_galileon'] = np.array(list(gb.as_dict().values()))
            out[f'{r}/J4'] = np.array(J4)
            sp = response_split(wconv(Ct), k, args.k_split, C + Cs + Cd, C_collapsed=wconv(Ccoll))
            for key, M in sp.items():
                out[f'{r}/C_T0_{key}'] = M
            Cresp = response_covariance(sp)
            out[f'{r}/C_T0_response'] = Cresp
            C_t0[r] = Cresp
            save(out)
            variants += [('SSC + disc + T0 tree', Cs + Cd + wconv(Ct)),
                         ('SSC + disc + T0 LL + squeezed + compl.', Cs + Cd + sp['LL'] + sp['LH'] + sp['completion']),
                         ('SSC + disc + T0 response (+ collapsed)', Cs + Cd + Cresp)]
        for tag, Cx in variants:
            summarize(f'{tag:40s}', V, C, Cx, k, nb)
        loglike_table(f'{b} {r}', V, C, variants)
        if args.fit_templates:
            comps = {'ssc': Cs, 'disc_B': wconv(dcomp['B']), 'disc_P': wconv(dcomp['P'])}
            if not args.no_t0:
                comps['T0tree'] = wconv(Ct)
                comps.update({f'T0_{key}': sp[key] for key in ('LL', 'LH', 'HH', 'completion')})
                comps['T0_collapsed'] = sp['HH_collapsed']
            amps = template_table(f'{b} {r}', V, C, comps, k, nb)
            out[f'{r}/template_fits'] = np.array(repr(amps))
        save(out)
        C_ssc[r + '_noP'] = res['LA_noPoisson'][1]
        del tr, cov

    V, C, k = get(z, b, 'GCcomb', 1, args.mode)
    Cg = dc.combine_regions([C_ssc['NGC'], C_ssc['SGC']], [norms['NGC'], norms['SGC']])
    Cgn = dc.combine_regions([C_ssc['NGC_noP'], C_ssc['SGC_noP']], [norms['NGC'], norms['SGC']])
    Cgd = dc.combine_regions([C_disc['NGC'], C_disc['SGC']], [norms['NGC'], norms['SGC']])
    out['GCcomb/C_ssc'], out['GCcomb/C_ssc_LA_noPoisson'], out['GCcomb/C_disc'] = Cg, Cgn, Cgd
    out['GCcomb/C_nongauss'] = Cg + Cgd
    variants = [('SSC (LA)', Cg), ('SSC (LA) + disc 4-pt', Cg + Cgd), ('SSC (LA, no Poisson) + disc 4-pt', Cgn + Cgd)]
    if C_t0:
        Cgt = dc.combine_regions([C_t0['NGC'], C_t0['SGC']], [norms['NGC'], norms['SGC']])
        out['GCcomb/C_T0_response'] = Cgt
        variants.append(('SSC + disc + T0 response (+ collapsed)', Cg + Cgd + Cgt))
    save(out)
    print(f'\n{b} GCcomb')
    for tag, Cx in variants:
        summarize(f'{tag:40s}', V, C, Cx, k, len(k))
    loglike_table(f'{b} GCcomb', V, C, variants)
    if args.fit_templates:
        comb = lambda key: dc.combine_regions([out[f'NGC/{key}'], out[f'SGC/{key}']], [norms['NGC'], norms['SGC']])
        comps = {'ssc': Cg, 'disc': Cgd}
        if C_t0:
            comps['T0resp'] = Cgt
        amps = template_table(f'{b} GCcomb', V, C, comps, k, len(k))
        out['GCcomb/template_fits'] = np.array(repr(amps))
    save(out)

if __name__ == '__main__':
    main()
