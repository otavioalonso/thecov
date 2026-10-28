"""All covariance terms of one catalogue in one process (one job): Gaussian split, measured and tree discreteness,
tree T0, SSC.

    python -u -m desi_validation.catalogue_covariance --bin LRG1 --region NGC --mock 173 --params <ssc_check npz> \
        [--cs-version abacus-2ndgen-dr2-complete] [--ells 0 2 4] [--single] [--no-t0] [--no-ssc] [--out-dir DIR]

Stages (the existing code, called in-process; the catalogue loader is memoised so data and randoms are read once per
random-file count):
  1. thinning stub (P_l, sum w^2, randoms' shot noise; thinning_cov thin with 2 thinnings);
  2. measured discreteness: thinning_closed_form --ells (exact B/n for all blocks, P/n^2 for l = l' = 0);
  3. Gaussian (kernel) with its PP/PS/SS split from one set of windows, and the tree B/n, P/n^2 (thinning_cov thecov,
     p = 1, --tree-params);
  4. tree T0 (snake + star, b3 = 0) on the same GaussianCovariance, with the production bias/damping inputs;
  5. SSC (local average, Poisson self-calibration term), as ssc_check's 'LA' variant.
Bias, f, damping, linear power and dilution are read from --params (an ssc_check output; <region>/params, damping,
plin_k, plin_P, plin_P_damped_normalised, dilution). Output: <out-dir>/catalogue_covariance.npz with C_PP, C_PS, C_SS,
C_G, lin_ell, quad, disc_B, disc_P, C_T0_snake, C_T0_star, C_ssc and the timings; decompose with disc_multipoles.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc  # noqa: E402
from desi_validation.grf_randoms import log  # noqa: E402


def memoise_loader():
    """dc.load_region cached on (catalogue version, mock, bin, region, random files) for the life of the process."""
    if getattr(dc.load_region, '_memoised', False):
        return
    orig = dc.load_region
    cache = {}

    @functools.wraps(orig)
    def load_region(paths, tracer_bin, region, n_random_files=1, **kw):
        key = (paths.cs_version, paths.mock, tracer_bin, region, n_random_files, tuple(sorted(kw.items())))
        if key not in cache:
            cache[key] = orig(paths, tracer_bin, region, n_random_files=n_random_files, **kw)
        return cache[key]
    load_region._memoised = True
    dc.load_region = load_region


def poisson_loader(seed):
    """Replace every catalogue the stages load by an unclustered one in the same footprint: galaxies drawn from one half
    of the randoms with p0 = N_data / N_half (same count, weights and n(z) as the real catalogue), the other half as the
    randoms. Its discreteness terms vanish in expectation, so measured lin, quad must equal thecov's 2 G_SS / r^2,
    G_SS / r^2 (the null test of thecov's shot-noise covariance, HANDOFF 16)."""
    import dataclasses
    orig = dc.load_region

    @functools.wraps(orig)
    def load_region(paths, tracer_bin, region, n_random_files=1, **kw):
        rc = orig(paths, tracer_bin, region, n_random_files=n_random_files, **kw)
        rng = np.random.default_rng(seed + 1000 * n_random_files)
        R = rc.randoms; nr = len(R['Z'])
        perm = rng.permutation(nr); par, est = perm[: nr // 2], perm[nr // 2:]
        p0 = len(rc.data['Z']) / len(par)
        sel = par[rng.random(len(par)) < p0]
        data = {c: np.asarray(v)[sel] for c, v in R.items()}
        rand = {c: np.asarray(v)[est] for c, v in R.items()}
        log(f'  Poisson catalogue (seed {seed}): {len(sel)} galaxies from {len(par)} randoms (p0 = {p0:.4f}; real catalogue '
            f'{len(rc.data["Z"])}), {len(est)} randoms')
        return dataclasses.replace(rc, data=data, randoms=rand, n_randoms_all_z=rc.n_randoms_all_z // 2,
                                   info=dict(rc.info, poisson_seed=seed))
    dc.load_region = load_region


GRF = {}


def grf_loader(args, paths):
    """Gaussian-field null test of thecov's P x S term. Galaxies = Poisson sample, p = p0 (1 + delta_s), of one half of the randoms;
    delta_s is a band-limited (k_c) Gaussian field of rms sigma with the shape of b1^2 P_lin (damped, from --params) and linear
    RSD with f at the true line of sight (as grf_randoms); the other half are the randoms. A Gaussian field has B = 0 (up to the
    clipping of p < 0: 5 % of the points at sigma = 0.6, skewness ~0.1 of tree level), so in every (l, l') block
        lin - 2 P/n^2(P_in) = (G_PS(P_in) + 2 G_SS) / r
    to the accuracy of thecov's P x S and S x S terms (the S x S part is known from the Poisson test, HANDOFF 17). thecov receives
    the Kaiser multipoles of P_in unmasked, the files' norm scaled by rho^2 (rho = sum w_gal / sum w_data), and the tree P/n^2
    is recomputed with (P_in, b1 = 1, f) after the thecov stage."""
    import dataclasses
    import scipy.fft as sfft
    import finufft
    from desi_validation.grf_randoms import NTH, tensor_index
    sigma, p0, seed = args.grf
    f, k_c = args.grf_f, args.grf_kc
    zp = np.load(os.path.expanduser(args.params)); key = lambda q: zp[f'{args.region}/{q}']
    kl, PirA = key('plin_k'), key('plin_P_damped_normalised')
    orig = dc.load_region

    @functools.wraps(orig)
    def load_region(paths_, tracer_bin, region, n_random_files=1, **kw):
        rc = orig(paths_, tracer_bin, region, n_random_files=n_random_files, **kw)
        if GRF.get('rho') is not None and n_random_files != GRF['n_random_files']:
            return rc                                                   # other loads (kernel smoothing) use the randoms only
        t0 = time.time()
        rng = np.random.default_rng(int(seed))
        R = rc.randoms; nr = len(R['Z'])
        perm = rng.permutation(nr); par, est = perm[: nr // 2], perm[nr // 2:]
        xr = dc.sky_to_cartesian(R['RA'], R['DEC'], R['Z'])
        lo, hi = xr.min(0), xr.max(0)
        L = 1.5 * float(np.max(hi - lo)); lo = 0.5 * (lo + hi) - 0.5 * L
        kf = 2 * np.pi / L
        n_gen = sfft.next_fast_len(2 * int(np.ceil(k_c / kf)) + 4)
        kk = 2 * np.pi * np.fft.fftfreq(n_gen, d=L / n_gen)
        KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing='ij')
        kg = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
        Pg = np.where(kg < k_c, np.interp(kg, kl, PirA, left=PirA[0], right=0.0), 0.0)
        vc = (L / n_gen) ** 3
        s2 = float(Pg.sum() / (vc * n_gen ** 3))
        kaiser0 = 1 + 2 * f / 3 + f ** 2 / 5                                # sigma is the REDSHIFT-SPACE rms (clipping is set by it)
        A = sigma ** 2 / s2 / kaiser0                                     # P_in = A * PirA (band-limited): real-space rms sigma / sqrt(kaiser0)
        Pg *= A
        W = sfft.fftn(rng.standard_normal((n_gen,) * 3), workers=NTH) * (np.sqrt(Pg / vc) / n_gen ** 3)
        del Pg
        xp = xr[par]
        thg = [np.ascontiguousarray(2 * np.pi * (xp[:, i] - lo[i]) / L) for i in range(3)]
        xh = xp / np.linalg.norm(xp, axis=1)[:, None]
        opts = dict(eps=1e-6, nthreads=NTH, modeord=1)
        if f:
            with np.errstate(invalid='ignore', divide='ignore'):
                inv_k2 = np.where(kg > 0, 1.0 / kg ** 2, 0.0)
            Kc = (KX, KY, KZ)
            t2, _ = tensor_index(2)
            stack = np.empty((7,) + W.shape, complex); stack[0] = W
            for n_, (a_, b_) in enumerate(t2):
                stack[n_ + 1] = W * Kc[a_] * Kc[b_] * inv_k2
            del W, KX, KY, KZ, kg, inv_k2
            g = finufft.nufft3d2(*thg, stack, isign=1, **opts).real
            del stack
            d = g[0].copy()
            for n_, (a_, b_) in enumerate(t2):
                d += f * (2.0 if a_ != b_ else 1.0) * xh[:, a_] * xh[:, b_] * g[n_ + 1]
            del g
        else:
            del KX, KY, KZ, kg
            d = finufft.nufft3d2(*thg, W, isign=1, **opts).real
            del W
        p = np.clip(p0 * (1 + d), 0, 1)
        sel = par[rng.random(len(p)) < p]
        data = {c: np.asarray(v)[sel] for c, v in R.items()}
        rand = {c: np.asarray(v)[est] for c, v in R.items()}
        rho = float(np.sum(dc.total_weight(data)) / np.sum(dc.total_weight(rc.data)))
        GRF.update(rho=rho, A=A, f=f, k_c=k_c, kl=kl, P_in=np.where(kl < k_c, A * PirA, 0.0), sigma_realised=float(d.std()),
                   clipped=float(np.mean(p0 * (1 + d) < 0)), n_random_files=n_random_files, n_gal=len(sel), L=L, n_gen=n_gen)
        log(f'  GRF catalogue (sigma {sigma}, realised {d.std():.3f}; f {f}; k_c {k_c}; p0 {p0}; seed {seed}): {len(sel)} galaxies from '
            f'{len(par)} randoms (clipped {GRF["clipped"]:.4f}), {len(est)} randoms; rho {rho:.4f}; box {L:.0f}, grid {n_gen}^3; {time.time() - t0:.0f} s')
        return dataclasses.replace(rc, data=data, randoms=rand, n_randoms_all_z=rc.n_randoms_all_z // 2, info=dict(rc.info, grf=dict(GRF)))
    dc.load_region = load_region

    orig_rs = dc.read_spectra

    @functools.wraps(orig_rs)
    def read_spectra(fns, **kw):
        spec = orig_rs(fns, **kw)
        if GRF.get('rho') is None:
            dc.load_region(paths, args.bin, args.region, n_random_files=4)
        k = spec['k']; P = np.interp(k, GRF['kl'], GRF['P_in'])
        kaiser = {0: 1 + 2 * f / 3 + f ** 2 / 5, 2: 4 * f / 3 + 4 * f ** 2 / 7, 4: 8 * f ** 2 / 35}
        vec = np.concatenate([kaiser[l] * P for l in spec['ells']])
        if args.grf_power == 'realised' and GRF.get('out_dir'):          # the catalogue's own P_l from the closed-form stage (masked)
            cf = os.path.join(GRF['out_dir'], 'thinning_closed_form_ell024.npz')
            if os.path.exists(cf):
                vec = np.load(cf)['P1_ell']
                log('  thecov power: the GRF catalogue\'s realised P_l (masked input)')
        return dict(spec, vectors=vec[None, :], norm=np.asarray(spec['norm']) * GRF['rho'] ** 2)
    dc.read_spectra = read_spectra

    orig_tc = dc.thecov_covariance

    @functools.wraps(orig_tc)
    def thecov_covariance(*a, **kw):
        kw['power_input'] = 'masked' if args.grf_power == 'realised' else 'unmasked'   # P_in is the true (unconvolved) spectrum
        return orig_tc(*a, **kw)
    dc.thecov_covariance = thecov_covariance


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1'); ap.add_argument('--region', default='NGC')
    ap.add_argument('--mock', type=int, default=173)
    ap.add_argument('--cs-version', default=None)
    ap.add_argument('--params', required=True, help='ssc_check npz with the bias, damping, linear power and dilution')
    ap.add_argument('--ells', type=int, nargs='+', default=[0, 2, 4])
    ap.add_argument('--single', action='store_true', help='single-precision NUFFTs in the measured discreteness')
    ap.add_argument('--no-t0', action='store_true'); ap.add_argument('--no-ssc', action='store_true')
    ap.add_argument('--workers', type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument('--ssc-n-near', type=int, default=300000)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--poisson', type=int, default=None, metavar='SEED', help='null test: unclustered catalogue drawn from the randoms')
    ap.add_argument('--grf', type=float, nargs=3, default=None, metavar=('SIGMA', 'P0', 'SEED'), help='null test of the P x S term: Poisson sample of a Gaussian field (see grf_loader)')
    ap.add_argument('--own-power', action='store_true', help="thecov Gaussian with the catalogue's own P_l (Paper 2 rule; removes the realised-P leakage into B/n)")
    ap.add_argument('--grf-f', type=float, default=0.762); ap.add_argument('--grf-kc', type=float, default=0.4)
    ap.add_argument('--grf-power', default='input', choices=['input', 'realised'], help='thecov P for the GRF test: P_in unmasked, or the catalogue\'s realised P_l masked')
    args = ap.parse_args(argv)
    from desi_validation import thinning_closed_form as tcf, thinning_cov as tcv
    if args.poisson is not None:
        poisson_loader(args.poisson)                # wrapped first, so the memoised loader caches the Poisson catalogue
    if args.grf is not None:
        grf_loader(args, tcv.setup_mock(args)[0])
    memoise_loader()
    tag = '' if not args.cs_version else f'_{args.cs_version}'
    od = args.out_dir or os.path.expanduser(f'~/thecov/catcov/{args.bin}_{args.region}_mock{args.mock}{tag}')
    os.makedirs(od, exist_ok=True)
    GRF['out_dir'] = od
    common = ['--bin', args.bin, '--region', args.region, '--mock', str(args.mock), '--out-dir', od]
    if args.cs_version:
        common += ['--cs-version', args.cs_version]
    ells = [str(l) for l in args.ells]
    T, t00 = {}, time.time()

    t0 = time.time()
    tcv.main(['thin'] + common + ['--ells'] + ells + ['--fracs', '0.5', '--nthin', '2'])
    T['thin_stub'] = time.time() - t0
    t0 = time.time()
    tcf.main(common + ['--ells'] + ells + (['--single'] if args.single else []))
    T['measured_discreteness'] = time.time() - t0
    t0 = time.time()
    tcv.main(['thecov'] + common + ['--frac', '1', '--tree-params', args.params, '--thecov-random-files', '4']
             + (['--own-power'] if args.own_power else []))   # shares the 4-file load
    T['gaussian_split_tree_disc'] = time.time() - t0
    cov, tr, res = tcv.LAST['cov'], tcv.LAST['tracer'], tcv.LAST['res']
    if args.grf is not None:                                          # tree P/n^2 (B = 0) of the Gaussian field: P_in, b1 = 1, f
        from thecov.tree.discreteness import DiscretenessCovariance
        D = DiscretenessCovariance(cov, (GRF['kl'], GRF['P_in']), b1=1.0, f=GRF['f'], b2=0.0, bs2=0.0)
        comp = D.components([(tr.name, tr.name)], ells=tuple(sorted(set(args.ells) | {0})))
        res = dict(res, disc_B=np.asarray(comp['B']), disc_P=np.asarray(comp['P']), **{f'grf_{q}': v for q, v in GRF.items()})
        log('  tree P/n^2 recomputed with the Gaussian field inputs')

    zp = np.load(os.path.expanduser(args.params)); key = lambda q: zp[f'{args.region}/{q}']
    b1, b2, bs2, f, _ = (float(x) for x in key('params'))
    ir_sigma, sigma_v, A_norm = (float(x) for x in key('damping'))
    kl, Pl, PirA = key('plin_k'), key('plin_P'), key('plin_P_damped_normalised')
    out = dict(res)
    cf_name = 'thinning_closed_form.npz' if max(args.ells) == 0 else 'thinning_closed_form_ell' + ''.join(map(str, sorted(set(args.ells) | {0}))) + '.npz'
    cf = np.load(os.path.join(od, cf_name))
    out.update(k=cf['k'], lin_ell=cf['lin_ell'] if 'lin_ell' in cf.files else cf['lin'], quad=cf['quad'],
               P1_ell=cf['P1_ell'] if 'P1_ell' in cf.files else cf['P1'])
    spectra = [(tr.name, tr.name)]
    if not args.no_t0:
        from thecov import TrispectrumCovariance
        from thecov.tree.trispectrum import galileon_bias
        t0 = time.time()
        t0c = TrispectrumCovariance(cov, (kl, PirA), galileon_bias(b1, b2, bs2, b3=0.0), f=f, n_workers=args.workers,
                                    sigma_fog=sigma_v, n_k=2, **(dict(n_mu=20, n_psi=40) if sigma_v else {}))
        tc = t0c.components(spectra, ells=tuple(args.ells))
        out.update(C_T0_snake=tc['snake'], C_T0_star=tc['star'], J4=t0c.window_integral(tr.name))
        T['tree_T0'] = time.time() - t0
        log(f'  tree T0 done ({T["tree_T0"]:.0f} s, {args.workers} processes)')
    if not args.no_ssc:
        from thecov import SuperSampleCovariance
        from thecov.tree.power import Dressed
        t0 = time.time()
        dil = key('dilution') if f'{args.region}/dilution' in zp.files else None
        ssc = SuperSampleCovariance(cov, (kl, A_norm * Pl), b1=b1, f=f, b2=b2, bs2=bs2, local_average=True,
                                    p_dressed=Dressed(kl, PirA, sigma_v), n_near=args.ssc_n_near, dilution=dil,
                                    discreteness=True, norm_kind='data-randoms',
                                    p_lin_long=(kl, Pl))         # long-mode variance from the fiducial P_lin, not A_norm P_lin
        ssc.compute_windows(tr.name, verbose=False)
        Cs, _ = ssc.covariance(spectra, ells=tuple(args.ells))
        Cpois = ssc.discreteness_covariance(tr.name, tuple(args.ells))          # tree Poisson local-average part (rank ~1 along P)
        Pm_, T_, var_ = ssc._poisson_terms(tr.name, tuple(args.ells))
        out.update(C_ssc=Cs, C_ssc_poisson=Cpois, C_ssc_clustering=Cs - Cpois, ssc_T_tree=T_, ssc_var_tree=var_, ssc_P_hat=Pm_)
        T['ssc'] = time.time() - t0
        log(f'  SSC done ({T["ssc"]:.0f} s)')
    T['total'] = time.time() - t00
    out.update({f'time_{k_}': v for k_, v in T.items()}, params=np.array([b1, b2, bs2, f]), damping=np.array([ir_sigma, sigma_v, A_norm]))
    fn = os.path.join(od, 'catalogue_covariance.npz')
    np.savez(fn, **{k_: v for k_, v in out.items() if v is not None})
    log('timings [s]: ' + ', '.join(f'{k_} {v:.0f}' for k_, v in T.items()))
    log(f'saved {fn}')


if __name__ == '__main__':
    main()
