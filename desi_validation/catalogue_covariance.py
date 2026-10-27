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
    args = ap.parse_args(argv)
    if args.poisson is not None:
        poisson_loader(args.poisson)                # wrapped first, so the memoised loader caches the Poisson catalogue
    memoise_loader()
    from desi_validation import thinning_closed_form as tcf, thinning_cov as tcv
    tag = '' if not args.cs_version else f'_{args.cs_version}'
    od = args.out_dir or os.path.expanduser(f'~/thecov/catcov/{args.bin}_{args.region}_mock{args.mock}{tag}')
    os.makedirs(od, exist_ok=True)
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
    tcv.main(['thecov'] + common + ['--frac', '1', '--tree-params', args.params, '--thecov-random-files', '4'])   # shares the 4-file load
    T['gaussian_split_tree_disc'] = time.time() - t0
    cov, tr, res = tcv.LAST['cov'], tcv.LAST['tracer'], tcv.LAST['res']

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
                                    sigma_fog=sigma_v, **(dict(n_mu=20, n_psi=40) if sigma_v else {}))
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
                                    discreteness=True, norm_kind='data-randoms')
        ssc.compute_windows(tr.name, verbose=False)
        Cs, _ = ssc.covariance(spectra, ells=tuple(args.ells))
        out.update(C_ssc=Cs)
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
