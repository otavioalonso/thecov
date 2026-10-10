"""Reduce the raw NERSC bundles to the small data products the paper is built from (paper/products/, not in git).

    python -I paper/scripts/make_products.py --raw <dir> [<dir> ...]

Every <dir> is searched (recursively) for the files below; the newest copy wins. Each product records, in
products/manifest.json, the raw files it came from with their sha256, the git commit of the repository and the date,
so every figure can be traced back to the NERSC run that produced its inputs.

Raw inputs (all written by desi_validation/ at NERSC):
  report_data_holi-kcore2-<b>.npz   859 holi mocks, Gaussian covariance with the xi-kernel window (dump_report_data)
  ssc_holi-kcore2-<b>.npz           non-Gaussian terms for the same mocks (ssc_check)
  report_data_holi-altmtl.npz       the same mocks, Gaussian covariance with the local window m^2 (random-density),
                                    own-weight window (none), binnings x1, x2, x4
  report_data_abacus-complete.npz,  25 AbacusSummit mocks, complete and altmtl (paired), local window
  report_data_abacus-altmtl.npz
  report_data_abacus-kcore-complete-LRG1.npz, ssc_abacus-kcore-complete-LRG1.npz   Abacus complete, kernel window + terms
  paper_maps_<b>.npz, hole_fraction_<b>.json, nz_scatter_<b>.npz   (paper_export.sh; optional)
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(os.path.dirname(HERE), 'products')
CAPS = ('NGC', 'SGC', 'GCcomb')


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def find(dirs, name, require=()):
    """newest copy of `name` under `dirs` that contains every key in `require` (older ssc_check versions did not save
    e.g. the linear power spectrum); copies without them are reported and skipped"""
    hits = sorted({os.path.realpath(p) for d in dirs for p in glob.glob(os.path.join(d, '**', name), recursive=True)},
                  key=os.path.getmtime, reverse=True)
    for p in hits:
        if require:
            files = set(np.load(p, allow_pickle=False).files)
            missing = [k for k in require if k not in files]
            if missing:
                print(f'  skipping {p}: missing {missing} (written by an older version of the NERSC script)')
                continue
        if len(hits) > 1:
            print(f'  {name}: using {p} (newest of {len(hits)} copies with the required keys)')
        return p
    return None


def window_convolve(C_local, C_gauss):
    """thecov.covariance_tools.window_convolve, inlined so that this script has no package dependency"""
    d = np.sqrt(np.diag(C_gauss))
    w, v = np.linalg.eigh(C_gauss / np.outer(d, d))
    K = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    return np.outer(d, d) * (K @ (C_local / np.outer(d, d)) @ K.T)


def holi(dirs, b, manifest):
    rk = find(dirs, f'report_data_holi-kcore2-{b}.npz', require=(f'{b}/NGC/x1/V', f'{b}/NGC/x1/C/kernel'))
    sk = find(dirs, f'ssc_holi-kcore2-{b}.npz', require=('NGC/C_ssc', 'NGC/C_disc', 'NGC/params', 'NGC/plin_k',
                                                         'NGC/plin_P_damped_normalised'))
    rl = find(dirs, 'report_data_holi-altmtl.npz')
    if not (rk and sk):
        print(f'holi {b}: no usable kernel dump / ssc file (see above): skipped. The ssc file must come from ssc_check.py '
              'at or after commit 5fd84d4 (2026-10-06, the final model run)')
        return
    zk, zs = np.load(rk, allow_pickle=False), np.load(sk, allow_pickle=False)
    zl = np.load(rl, allow_pickle=False) if rl else None
    out = {}
    for r in CAPS:
        for f in (1, 2, 4):
            p = f'{b}/{r}/x{f}'
            if p + '/V' not in zk.files:
                continue
            g = 'combined-regions' if r == 'GCcomb' else 'kernel'
            out[f'{r}/x{f}/V'] = zk[p + '/V'].astype(np.float64)
            for key in ('k', 'k_edges', 'nmodes', 'norm', 'num_shotnoise'):
                out[f'{r}/x{f}/{key}'] = zk[f'{p}/{key}']
            out[f'{r}/x{f}/mock_ids'] = np.array([int(str(s)[4:]) for s in zk[p + '/mock_ids']])
            out[f'{r}/x{f}/C_G'] = zk[f'{p}/C/{g}']
            if zl is not None and p + '/V' in zl.files:
                assert (zl[p + '/mock_ids'] == zk[p + '/mock_ids']).all(), 'mock sets differ'
                gl = 'combined-regions' if r == 'GCcomb' else 'random-density'
                out[f'{r}/x{f}/C_G_local'] = zl[f'{p}/C/{gl}']
                if f'{p}/C/none' in zl.files:
                    out[f'{r}/x{f}/C_G_ownweight'] = zl[f'{p}/C/none']
        C = out[f'{r}/x1/C_G']
        keys = ['C_ssc', 'C_ssc_noLA', 'C_ssc_LA_noPoisson', 'C_disc', 'C_disc_local', 'C_disc_B', 'C_disc_P',
                'C_T0_response', 'C_T0_LL', 'C_T0_LH', 'C_T0_completion', 'C_T0_HH_collapsed', 'mock_mean', 'params',
                'damping', 'sigma2', 'sigma2_keys', 'template_fits', 'plin_k', 'plin_P', 'plin_P_damped_normalised']
        for key in keys:
            if f'{r}/{key}' in zs.files:
                out[f'{r}/x1/{key}'] = zs[f'{r}/{key}']
        if f'{r}/C_T0_snake' in zs.files:   # tree-level T0 with the window's mode mixing, as in ssc_check
            out[f'{r}/x1/C_T0_tree'] = window_convolve(zs[f'{r}/C_T0_snake'] + zs[f'{r}/C_T0_star'], C)
    fn = os.path.join(PROD, f'holi_{b}.npz')
    np.savez_compressed(fn, **out)
    manifest[os.path.basename(fn)] = {os.path.basename(x): sha256(x) for x in (rk, sk, rl) if x}
    print(f'wrote {fn}')


def abacus(dirs, manifest):
    out, src = {}, []
    for kind in ('complete', 'altmtl'):
        fn = find(dirs, f'report_data_abacus-{kind}.npz')
        if not fn:
            continue
        z = np.load(fn, allow_pickle=False)
        src.append(fn)
        for r in CAPS:
            for f in (1, 2, 4):
                p = f'LRG1/{r}/x{f}'
                if p + '/V' not in z.files:
                    continue
                out[f'{kind}/{r}/x{f}/V'] = z[p + '/V'].astype(np.float64)
                out[f'{kind}/{r}/x{f}/mock_ids'] = z[p + '/mock_ids']
                for key in ('k', 'k_edges', 'nmodes', 'norm'):
                    out[f'{kind}/{r}/x{f}/{key}'] = z[f'{p}/{key}']
                out[f'{kind}/{r}/x{f}/C_G_local'] = z[f'{p}/C/' + ('combined-regions' if r == 'GCcomb' else 'random-density')]
    rk, sk = find(dirs, 'report_data_abacus-kcore-complete-LRG1.npz'), find(dirs, 'ssc_abacus-kcore-complete-LRG1.npz')
    if rk and sk:
        zk, zs = np.load(rk, allow_pickle=False), np.load(sk, allow_pickle=False)
        src += [rk, sk]
        for r in CAPS:
            p = f'LRG1/{r}/x1'
            out[f'kernel/{r}/x1/V'] = zk[p + '/V'].astype(np.float64)
            out[f'kernel/{r}/x1/C_G'] = zk[f'{p}/C/' + ('combined-regions' if r == 'GCcomb' else 'kernel')]
            for key in ('C_ssc', 'C_disc', 'C_T0_response', 'template_fits'):
                if f'{r}/{key}' in zs.files:
                    out[f'kernel/{r}/x1/{key}'] = zs[f'{r}/{key}']
            if f'{r}/C_T0_snake' in zs.files:
                out[f'kernel/{r}/x1/C_T0_tree'] = window_convolve(zs[f'{r}/C_T0_snake'] + zs[f'{r}/C_T0_star'],
                                                                  out[f'kernel/{r}/x1/C_G'])
    if out:
        fn = os.path.join(PROD, 'abacus_LRG1.npz')
        np.savez_compressed(fn, **out)
        manifest[os.path.basename(fn)] = {os.path.basename(x): sha256(x) for x in src}
        print(f'wrote {fn}')


def passthrough(dirs, manifest):
    for name in ('paper_maps_LRG1.npz', 'paper_maps_QSO.npz', 'hole_fraction_LRG1.json', 'hole_fraction_QSO.json',
                 'nz_scatter_LRG1.npz', 'nz_scatter_QSO.npz'):
        fn = find(dirs, name)
        if fn:
            shutil.copy(fn, os.path.join(PROD, name))
            manifest[name] = {name: sha256(fn)}
            print(f'copied {name}')


def defective_mocks(manifest, nsig=6.0):
    """mocks whose n(z) has a bin more than nsig robust standard deviations from the median, in either cap (holi v3:
    12 LRG mocks with +55-65% galaxies in 0.46 < z < 0.47). Written to products/excluded_mocks.json and removed from
    every holi statistic by common.holi()."""
    out = {}
    for b in ('LRG1', 'QSO'):
        fn = os.path.join(PROD, f'nz_scatter_{b}.npz')
        if not os.path.exists(fn):
            continue
        z = np.load(fn, allow_pickle=False)
        bad, detail = set(), {}
        for r in ('NGC', 'SGC'):
            NZ, ids = z[f'{r}/NZ'], z[f'{r}/mocks']
            d = NZ / np.median(NZ, 0) - 1
            mad = 1.4826 * np.median(np.abs(d - np.median(d, 0)), 0)
            hit = np.abs(d - np.median(d, 0)) > nsig * mad
            for i in np.flatnonzero(hit.any(1)):
                bad.add(int(ids[i]))
                j = int(np.argmax(np.abs(d[i]) * hit[i]))
                detail.setdefault(str(int(ids[i])), {})[r] = dict(z_bin=[float(z['edges'][j]), float(z['edges'][j + 1])],
                                                                  excess=float(d[i, j]))
        out[b] = dict(mocks=sorted(bad), detail=detail, criterion=f'|delta n(z)| > {nsig} MAD in any bin')
        print(f'{b}: {len(bad)} defective mocks {sorted(bad)}')
    json.dump(out, open(os.path.join(PROD, 'excluded_mocks.json'), 'w'), indent=1)
    manifest['excluded_mocks.json'] = {'from': 'nz_scatter_<b>.npz'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', nargs='+', required=True, help='directories holding the unpacked NERSC bundles')
    args = ap.parse_args()
    os.makedirs(PROD, exist_ok=True)
    mfn = os.path.join(PROD, 'manifest.json')
    manifest = json.load(open(mfn)) if os.path.exists(mfn) else {}
    for b in ('LRG1', 'QSO'):
        holi(args.raw, b, manifest)
    abacus(args.raw, manifest)
    passthrough(args.raw, manifest)
    defective_mocks(manifest)
    try:
        commit = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = 'unknown'
    manifest['_built'] = dict(date=datetime.datetime.now().isoformat(timespec='seconds'), git=commit)
    json.dump(manifest, open(mfn, 'w'), indent=1)
    print(f'wrote {mfn}')


if __name__ == '__main__':
    main()
