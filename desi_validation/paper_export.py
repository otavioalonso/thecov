"""Inputs for the covariance-validation paper that only exist at NERSC: sky maps of the veto-hole fill fraction and of the
mean weight, for the figures on the window. One debug job (~15 min for LRG1 + QSO, both caps):

    bash ~/thecov/desi_validation/paper_export.sh

Per tracer and region the maps are counts of the randoms on a NEST healpix grid (nside 2048 and the coarser levels follow
by summing children), stored sparse (occupied pixels only), with the expected count of a full pixel E, so that the fill
fraction is f = counts / E, exactly the estimator of `hole_fraction.py`. Also the weighted counts (sum of the total
weight and of its square) for the mean-weight and <w^2>/<w>^2 maps.

Output: <--out>/paper_maps_<tracer>.npz with, per region R: R/pix (int64), R/n (counts), R/sw, R/sw2, R/E (scalar),
R/nside, R/chi_median.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc  # noqa: E402

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bins', nargs='+', default=['LRG1', 'QSO'])
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--random-files', type=int, default=10)
    ap.add_argument('--nside', type=int, default=2048)
    ap.add_argument('--surface-density', type=float, default=2500.0, help='randoms per deg^2 per file')
    ap.add_argument('--out', default='/global/cfs/cdirs/desicollab/users/oalves/thecov_validation')
    args = ap.parse_args()
    import healpy as hp

    paths = dc.Paths(kind='holi_v3', mock=173)
    paths.loader = 'auto'
    paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
    os.makedirs(args.out, exist_ok=True)
    for b in args.bins:
        out = {}
        for r in args.regions:
            rc = dc.load_region(paths, b, r, n_random_files=args.random_files)
            ran = rc.randoms
            nz = len(ran['Z'])
            E = (args.surface_density * rc.n_random_files * hp.nside2pixarea(args.nside, degrees=True)
                 * nz / max(rc.n_randoms_all_z, nz))
            pix = hp.ang2pix(args.nside, np.asarray(ran['RA'], float), np.asarray(ran['DEC'], float), lonlat=True, nest=True)
            w = np.asarray(dc.total_weight(ran), float)
            u, inv = np.unique(pix, return_inverse=True)
            out[f'{r}/pix'] = u.astype(np.int64)
            out[f'{r}/n'] = np.bincount(inv).astype(np.int32)
            out[f'{r}/sw'] = np.bincount(inv, weights=w).astype(np.float32)
            out[f'{r}/sw2'] = np.bincount(inv, weights=w * w).astype(np.float32)
            out[f'{r}/E'] = float(E)
            out[f'{r}/nside'] = int(args.nside)
            out[f'{r}/chi_median'] = float(dc.comoving_distance(np.array([np.median(ran['Z'])]))[0])
            log(f'{b} {r}: {nz} randoms, {len(u)} occupied pixels at nside {args.nside}, E = {E:.1f} per full pixel')
            del rc, ran
        fn = os.path.join(args.out, f'paper_maps_{b}.npz')
        np.savez_compressed(fn, **out)
        log(f'saved {fn}')


if __name__ == '__main__':
    main()
