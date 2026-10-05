"""How much Gaussian variance can the veto holes add, and at which angular scales? (no pair counts)

    python -m desi_validation.hole_fraction --bins LRG1 QSO --regions NGC SGC

Idea. Inside the footprint, the pair-averaged clustering window is W = m(x) (K * m)(x) ~ m^2 f(x), with
f the fraction of the area around x (over a pair separation) that is not vetoed. A uniform f cancels
against the masked model normalisation (P_model ~ norm / int W): only the VARIATION of f across the
footprint changes the covariance. With m ~ rho w and the shot-noise density S ~ rho w^2:

    R_PP = [int m^4 f^2 / int m^4] / [int m^2 f / int m^2]^2     (clustering x clustering term)
    R_PS = [int m^2 f S / int m^2 S] / [int m^2 f / int m^2]     (clustering x shot-noise term)
    (shot noise x shot noise: unchanged)

relative to the local window m^2 that thecov uses by default. Here f = f_R is the fill fraction of a
healpix pixel of scale R, for R from ~2' to ~7 deg. R_PP(R) shows how much variance the hole-fraction
inhomogeneity can add at each scale, whether it saturates (then any reasonable pair-averaging gives the
same answer) or keeps growing, and <f>_R = int m^2 f / int m^2 maps the kernel's I_k / int m^2 (logged
by the kernel runs) to an effective angular scale.

Estimators. The fill fraction is counted from the randoms of `--random-files` files (z-cut, as in
fill_map), split into a map subset and an independent evaluation subset (`--n-eval` randoms), so that a
random never counts itself. f = c / E and f^2 = c (c - 1) / E^2 are unbiased for Poisson counts c with
expectation E per full pixel (E is printed: below ~10 the finest scales are noisy, not biased). The
integrals are sums over the evaluation randoms of g / rho (they sample rho_r); the weight of each random
stands in for the local mean weight (it only sets the averaging measure of a ratio).

Caveat: at degree scales (nside <~ 32) pixels straddling the footprint edge also count as partly empty,
so the coarsest rows mix edges with holes (a hole-free 30 x 60 deg cap gives R_PP = 1.008 at nside 64,
1.017 at 32, 1.032 at 16). Synthetic checks: uniform 20% holes give <f> = 0.80 but R_PP = 1.005 (they
cancel); 10% / 40% holes in two halves give R_PP = 1.038 against the analytic 1.036.

Output: a table per tracer and region on stdout, and OUT/hole_fraction_<tracer>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc  # noqa: E402

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def hole_statistics(ra, dec, rho, w, expected_full, nside_max=2048, nside_min=8, n_eval=2_000_000, seed=0):
    """R_PP, R_PS and <f> per healpix scale. `expected_full`: expected randoms (of all of ra, dec) in a
    full pixel at nside_max. Returns a list of dicts, finest scale first."""
    import healpy as hp
    rng = np.random.default_rng(seed)
    n = len(ra)
    ev = np.zeros(n, bool)
    ev[rng.choice(n, size=min(n_eval, n // 10), replace=False)] = True
    pix = hp.ang2pix(nside_max, np.asarray(ra, float), np.asarray(dec, float), lonlat=True, nest=True)
    counts = np.bincount(pix[~ev], minlength=hp.nside2npix(nside_max)).astype(float)
    E0 = expected_full * (~ev).sum() / n                        # map subset only
    pe, rho_e, w_e = pix[ev], np.asarray(rho, float)[ev], np.asarray(w, float)[ev]
    a_pp, a_i, a_ps = rho_e ** 3 * w_e ** 4, rho_e * w_e ** 2, rho_e ** 2 * w_e ** 4   # m^4/rho, m^2/rho, m^2 S/rho
    out, nside, c, E, p = [], nside_max, counts, E0, pe
    while nside >= nside_min:
        ce = c[p]
        f, f2 = ce / E, ce * (ce - 1.0) / E ** 2
        mean_f = np.sum(a_i * f) / a_i.sum()
        r_pp = (np.sum(a_pp * f2) / a_pp.sum()) / mean_f ** 2
        r_ps = (np.sum(a_ps * f) / a_ps.sum()) / mean_f
        occ = c > 0
        out.append(dict(nside=int(nside), scale_arcmin=float(np.degrees(np.sqrt(hp.nside2pixarea(nside))) * 60),
                        expected_per_pixel=float(E), mean_f=float(mean_f), R_PP=float(r_pp), R_PS=float(r_ps),
                        rms_f_over_f=float(np.sqrt(max(r_pp - 1.0, 0.0))),
                        frac_pixels_partial=float(np.mean(c[occ] < 0.9 * E))))
        c, E, p, nside = c.reshape(-1, 4).sum(1), 4 * E, p >> 2, nside // 2   # NEST: 4 children per parent
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bins', nargs='+', default=['LRG1', 'QSO'])
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--random-files', type=int, default=10)
    ap.add_argument('--nside-max', type=int, default=2048)
    ap.add_argument('--nside-min', type=int, default=8)
    ap.add_argument('--n-eval', type=float, default=2e6)
    ap.add_argument('--surface-density', type=float, default=2500.0, help='randoms per deg^2 per file')
    args = ap.parse_args()

    paths = dc.Paths(kind='holi_v3', mock=173)
    paths.loader = 'auto'
    paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
    OUT = os.path.expanduser(f'~/thecov_desi/{paths.kind}_mock{paths.mock}')
    os.makedirs(OUT, exist_ok=True)
    import healpy as hp

    for b in args.bins:
        res = {}
        for r in args.regions:
            rc = dc.load_region(paths, b, r, n_random_files=args.random_files)
            ran = rc.randoms
            nz = len(ran['Z'])
            log(f'{b} {r}: {nz} randoms ({args.random_files} files, z-cut)')
            expected = (args.surface_density * rc.n_random_files * hp.nside2pixarea(args.nside_max, degrees=True)
                        * nz / max(rc.n_randoms_all_z, nz))
            rho, _ = dc.random_density(rc, args.surface_density)
            stats = hole_statistics(ran['RA'], ran['DEC'], rho, dc.total_weight(ran), expected,
                                    nside_max=args.nside_max, nside_min=args.nside_min, n_eval=int(args.n_eval))
            chi = float(dc.comoving_distance(np.array([np.median(ran['Z'])]))[0])
            for s in stats:
                s['scale_mpch'] = chi * np.radians(s['scale_arcmin'] / 60)
            res[r] = dict(chi_median=chi, stats=stats)
            print(f'\n{b} {r} (median chi {chi:.0f} Mpc/h)\n'
                  f"{'nside':>6} {'scale':>8} {'Mpc/h':>7} {'E/pix':>8} {'<f>':>7} {'R_PP':>7} {'rms f/f':>8} "
                  f"{'R_PS':>7} {'partial':>8}")
            for s in stats:
                print(f"{s['nside']:6d} {s['scale_arcmin']:7.1f}' {s['scale_mpch']:7.1f} {s['expected_per_pixel']:8.1f} "
                      f"{s['mean_f']:7.4f} {s['R_PP']:7.4f} {s['rms_f_over_f']:8.3f} {s['R_PS']:7.4f} "
                      f"{s['frac_pixels_partial']:8.3f}", flush=True)
            del rc, ran
        fn = os.path.join(OUT, f'hole_fraction_{b}.json')
        with open(fn, 'w') as fh:
            json.dump(res, fh, indent=1)
        log(f'saved {fn}')


if __name__ == '__main__':
    main()
