"""Inputs for the covariance-validation paper that only exist at NERSC: sky maps of the veto-hole fill fraction and of the
mean weight, for the figures on the window. One debug job (~15 min for LRG1 + QSO, both caps):

    bash ~/thecov/desi_validation/paper_export.sh

Per tracer and region the maps are counts of the randoms on a NEST healpix grid, stored sparse (occupied pixels only),
with the expected count of a full nside-2048 pixel E, so that the fill fraction is f = counts / (E 4^(11 - log2 nside)),
exactly the estimator of `hole_fraction.py`.

Output: <--out>/paper_maps_<tracer>.npz, slim (a few MB): per region R the counts at nside 256 (R/pix256, R/n256), the
full-resolution counts in a 4 x 4 deg patch (R/zoom_pix, R/zoom_n at nside 2048, centred on R/zoom_center), R/E (expected
randoms in a full nside-2048 pixel), R/nside, R/chi_median. `--slim` converts an existing full-resolution file in place
(seconds, no catalogue reading):

    python -m desi_validation.paper_export --slim
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


def slim(full, nside_low=256, zoom_deg=4.0):
    """reduce a full-resolution map dict (R/pix, R/n at nside, NEST) to what the paper figure needs"""
    import healpy as hp
    out = {}
    for r in sorted({k.split('/')[0] for k in full}):
        nside = int(full[f'{r}/nside'])
        pix, n = np.asarray(full[f'{r}/pix'], np.int64), np.asarray(full[f'{r}/n'], np.int64)
        shift = 2 * int(np.log2(nside // nside_low))
        lo, inv = np.unique(pix >> shift, return_inverse=True)
        out[f'{r}/pix256'] = lo.astype(np.int32)
        out[f'{r}/n256'] = np.bincount(inv, weights=n).astype(np.int32)
        ra, dec = hp.pix2ang(nside, pix, nest=True, lonlat=True)
        ra0 = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(ra))))) % 360
        dec0 = float(np.median(dec))
        sel = (np.abs(((ra - ra0 + 180) % 360) - 180) * np.cos(np.radians(dec0)) < 0.75 * zoom_deg) & \
              (np.abs(dec - dec0) < 0.75 * zoom_deg)
        out[f'{r}/zoom_pix'], out[f'{r}/zoom_n'] = pix[sel], n[sel].astype(np.int32)
        out[f'{r}/zoom_center'] = np.array([ra0, dec0])
        for key in ('E', 'nside', 'chi_median'):
            out[f'{r}/{key}'] = full[f'{r}/{key}']
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bins', nargs='+', default=['LRG1', 'QSO'])
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--random-files', type=int, default=10)
    ap.add_argument('--nside', type=int, default=2048)
    ap.add_argument('--surface-density', type=float, default=2500.0, help='randoms per deg^2 per file')
    ap.add_argument('--out', default='/global/cfs/cdirs/desicollab/users/oalves/thecov_validation')
    ap.add_argument('--slim', action='store_true', help='only convert existing full-resolution paper_maps_*.npz in place')
    args = ap.parse_args()
    import healpy as hp
    if args.slim:
        for b in args.bins:
            fn = os.path.join(args.out, f'paper_maps_{b}.npz')
            z = np.load(fn)
            if f'{args.regions[0]}/pix256' in z.files:
                log(f'{fn} is already slim')
                continue
            out = slim({k: z[k] for k in z.files})
            np.savez_compressed(fn, **out)
            log(f'slimmed {fn}: {os.path.getsize(fn) / 1e6:.1f} MB')
        return

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
            u, inv = np.unique(pix, return_inverse=True)
            out[f'{r}/pix'] = u.astype(np.int64)
            out[f'{r}/n'] = np.bincount(inv).astype(np.int32)
            out[f'{r}/E'] = float(E)
            out[f'{r}/nside'] = int(args.nside)
            out[f'{r}/chi_median'] = float(dc.comoving_distance(np.array([np.median(ran['Z'])]))[0])
            log(f'{b} {r}: {nz} randoms, {len(u)} occupied pixels at nside {args.nside}, E = {E:.1f} per full pixel')
            del rc, ran
        fn = os.path.join(args.out, f'paper_maps_{b}.npz')
        np.savez_compressed(fn, **slim(out))
        log(f'saved {fn}')


if __name__ == '__main__':
    main()
