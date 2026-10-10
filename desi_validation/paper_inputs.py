"""Check (and bundle) every raw NERSC input of the validation paper (paper/scripts/make_products.py).

    python -m desi_validation.paper_inputs --check     prints one line per input: ok / MISSING / STALE (+ what to run);
                                                       writes the list of actions to <out>/paper_inputs_actions.txt
    python -m desi_validation.paper_inputs --bundle    packs every input that is ok into <out>/paper_raw_<date>.tgz

Driven by desi_validation/paper_update.sh. An input is STALE when it exists but lacks keys the paper needs (e.g. an
ssc_holi-kcore2 file written by an ssc_check run with --no-t0 or by an old version).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import tarfile

import numpy as np

OUT = os.environ.get('PAPER_OUT', '/global/cfs/cdirs/desicollab/users/oalves/thecov_validation')
HOME = os.environ.get('PAPER_HOME', os.path.expanduser('~/thecov_desi'))
HOLI = os.path.join(HOME, 'holi_v3_mock173')
ABC = os.path.join(HOME, 'abacus-2ndgen-dr2-complete_mock0')
ABA = os.path.join(HOME, 'abacus-2ndgen-dr2-altmtl_mock0')

# (path, required keys, action if missing or stale)
SSC = 'ssc'           # rerun ssc_check (paper_update.sh submits it)
EXPORT = 'export'     # paper_export.sh (maps, hole statistics, n(z))
ABACUS = 'abacus'     # bash ~/thecov/desi_validation/run_abacus_t0_test.sh complete
DUMP = 'dump'         # Gaussian-covariance dumps: expensive, never resubmitted automatically


def inputs():
    L = []
    for b in ('LRG1', 'QSO'):
        L += [(os.path.join(HOLI, f'report_data_holi-kcore2-{b}.npz'),
               [f'{b}/NGC/x1/V', f'{b}/NGC/x1/C/kernel', f'{b}/SGC/x4/C/kernel', f'{b}/GCcomb/x1/C/combined-regions'], DUMP),
              (os.path.join(OUT, f'ssc_holi-kcore2-{b}.npz'),
               ['NGC/C_ssc', 'NGC/C_disc', 'NGC/params', 'NGC/plin_k', 'NGC/plin_P_damped_normalised', 'NGC/C_T0_snake',
                'NGC/C_T0_response', 'SGC/C_T0_response', 'GCcomb/C_ssc'], SSC),
              (os.path.join(OUT, f'paper_maps_{b}.npz'), ['NGC/pix256', 'SGC/zoom_pix'], EXPORT),
              (os.path.join(OUT, f'hole_fraction_{b}.json'), ['NGC', 'SGC'], EXPORT),
              (os.path.join(OUT, f'nz_scatter_{b}.npz'), ['NGC/NZ', 'SGC/NZ', 'edges'], EXPORT)]
    L += [(os.path.join(HOLI, 'report_data_holi-altmtl.npz'),
           ['LRG1/NGC/x1/C/random-density', 'LRG1/NGC/x1/C/none', 'QSO/SGC/x4/C/random-density'], DUMP),
          (os.path.join(ABC, 'report_data_abacus-complete.npz'), ['LRG1/NGC/x1/V', 'LRG1/SGC/x4/C/random-density'], DUMP),
          (os.path.join(ABA, 'report_data_abacus-altmtl.npz'), ['LRG1/NGC/x1/V', 'LRG1/SGC/x4/C/random-density'], DUMP),
          (os.path.join(ABC, 'report_data_abacus-kcore-complete-LRG1.npz'), ['LRG1/NGC/x1/C/kernel'], ABACUS),
          (os.path.join(OUT, 'ssc_abacus-kcore-complete-LRG1.npz'),
           ['NGC/C_ssc', 'NGC/C_disc', 'NGC/C_T0_snake', 'NGC/C_T0_response'], ABACUS)]
    return L


def status(path, keys):
    if not os.path.exists(path):
        return 'MISSING', keys
    if path.endswith('.json'):
        have = set(json.load(open(path)))
    else:
        have = set(np.load(path, allow_pickle=False).files)
    miss = [k for k in keys if k not in have]
    return ('STALE' if miss else 'ok'), miss


def check():
    actions = set()
    for path, keys, act in inputs():
        st, miss = status(path, keys)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M') if st != 'MISSING' else ''
        print(f'{st:7s} {mtime:16s} {path}' + (f'   (lacks {miss[:3]}{"..." if len(miss) > 3 else ""}) -> {act}' if st != 'ok' else ''))
        if st != 'ok':
            actions.add(act if act != SSC else f'{SSC}:{os.path.basename(path).split("-")[-1][:-4]}')
    with open(os.path.join(OUT, 'paper_inputs_actions.txt'), 'w') as fh:
        fh.write('\n'.join(sorted(actions)) + ('\n' if actions else ''))
    if DUMP in actions:
        print('\nA Gaussian-covariance dump is missing: these take several jobs and are not resubmitted automatically. '
              'See HANDOFF.md (sections 6-7) for the dump_report_data commands.')
    if ABACUS in actions:
        print('\nAbacus inputs missing: bash ~/thecov/desi_validation/run_abacus_t0_test.sh complete')
    print(f'\nactions: {sorted(actions) or "none"}')


def bundle():
    fn = os.path.join(OUT, f'paper_raw_{datetime.datetime.now():%Y%m%d_%H%M}.tgz')
    n, skipped = 0, []
    with tarfile.open(fn, 'w:gz') as tar:
        for path, keys, _ in inputs():
            st, _ = status(path, keys)
            if st != 'ok':
                skipped.append(f'{st} {path}')
                continue
            # one directory per source so that equal names (none here) cannot collide
            tar.add(path, arcname=os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)))
            n += 1
    for s in skipped:
        print(f'not bundled: {s}')
    print(f'{fn}: {n} files, {os.path.getsize(fn) / 1e6:.0f} MB')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--bundle', action='store_true')
    a = ap.parse_args()
    check() if a.check else bundle()
