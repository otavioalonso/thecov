"""T0 in a periodic box: the Abacus c000 LRG z = 0.5 cubic mocks (25 boxes x 3 lines of sight, mock_challenge_box/abacus-hf-v2).

    python -u -m desi_validation.box_t0 --stage read [--tracer LRG --z 0.500 --cosmo c000 --hod hod-base_B]

read: every mesh2 power-spectrum file (P0, P2, P4, k, nmodes, shotnoise, norm, attrs) and every mesh3 sugiyama-diagonal
bispectrum file into one npz (<out>/box_<tracer>_<z>_<cosmo>.npz) with the metadata printed, for the offline analysis
(window-free: the Gaussian covariance is diagonal in k, J4 = 1/V, no SSC; the off-diagonal covariance is discreteness + T0).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation.grf_randoms import log  # noqa: E402

BOX_DIR = '/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/mock_challenge_box/abacus-hf-v2'


def _attrs(obj):
    out = {}
    for name in ('attrs', 'meta'):
        try:
            a = getattr(obj, name)
            a = a() if callable(a) else a
            for kk, vv in dict(a).items():
                try:
                    out[f'{name}/{kk}'] = np.asarray(vv)
                except Exception:  # noqa: BLE001
                    out[f'{name}/{kk}'] = np.array(str(vv))
        except Exception:  # noqa: BLE001
            continue
    return out


def stage_read(args):
    import lsstypes as types
    out_dir = args.out_dir or os.path.expanduser('~/thecov/box')
    os.makedirs(out_dir, exist_ok=True)
    tag = f'{args.tracer}_z{args.z}_{args.cosmo}_{args.hod}'
    out = {}
    for los in ('x', 'y', 'z'):
        fns = sorted(glob.glob(os.path.join(BOX_DIR, f'mesh2_spectrum_poles_{tag}_los-{los}_*.h5')), key=lambda f: int(f.rsplit('_', 1)[1][:-3]))
        log(f'los {los}: {len(fns)} power spectrum files')
        for j, fn in enumerate(fns):
            s = types.read(fn)
            i = int(fn.rsplit('_', 1)[1][:-3])
            ells = [int(e) for e in getattr(s, 'ells', (0, 2, 4))] if hasattr(s, 'ells') else [0, 2, 4]
            p0 = s.get(0)
            if j == 0 and los == 'x':
                log(f'  type {type(s).__name__}; dir: ' + ' '.join(a for a in dir(s) if not a.startswith('_'))[:500])
                at = _attrs(s); at.update({f'p0/{kk}': vv for kk, vv in _attrs(p0).items()})
                for kk, vv in at.items():
                    log(f'  attr {kk}: {str(vv)[:120]}')
                    out[f'attrs/{kk}'] = vv
                out['k'] = np.asarray(p0.coords('k')); out['k_edges'] = np.asarray(p0.edges('k')); out['nmodes'] = np.asarray(p0.values('nmodes'))
                log(f'  k: {len(out["k"])} bins {out["k"][:3]} ... {out["k"][-1]:.4f}; ells {ells}')
                try:
                    log(f'  p0 values keys: {list(p0.values().keys()) if hasattr(p0.values(), "keys") else p0.values()}')
                except Exception as e:  # noqa: BLE001
                    log(f'  p0 values(): {e!r}')
            for l in (0, 2, 4):
                out[f'P/{los}/{i}/{l}'] = np.real(np.asarray(s.get(l).value()))
            for q in ('shotnoise', 'num_shotnoise', 'norm'):
                try:
                    out[f'{q}/{los}/{i}'] = np.ravel(np.asarray(p0.values(q)))[0]
                except Exception:  # noqa: BLE001
                    pass
        fns3 = sorted(glob.glob(os.path.join(BOX_DIR, f'mesh3_spectrum_sugiyama-diagonal_poles_{tag}_los-{los}_*.h5')), key=lambda f: int(f.rsplit('_', 1)[1][:-3]))
        log(f'los {los}: {len(fns3)} bispectrum files')
        for j, fn in enumerate(fns3):
            b = types.read(fn)
            i = int(fn.rsplit('_', 1)[1][:-3])
            if j == 0 and los == 'x':
                log(f'  bispectrum type {type(b).__name__}; dir: ' + ' '.join(a for a in dir(b) if not a.startswith('_'))[:500])
                for kk, vv in _attrs(b).items():
                    log(f'  battr {kk}: {str(vv)[:120]}'); out[f'battrs/{kk}'] = vv
                try:
                    labs = [str(lab) for lab, _ in b.items()]
                    log(f'  bispectrum leaves: {labs}')
                except Exception as e:  # noqa: BLE001
                    log(f'  bispectrum items(): {e!r}')
            try:
                for lab, leaf in b.items():
                    key = str(lab).replace(' ', '').replace("'", '').replace('{', '').replace('}', '').replace(':', '-').replace(',', '_')
                    if j == 0 and los == 'x':
                        out[f'B/coords/{key}'] = np.asarray(leaf.coords('k')) if 'k' in getattr(leaf, 'coords', lambda: {})() else np.zeros(0)
                        try:
                            log(f'  leaf {key}: value shape {np.shape(leaf.value())}; coords {list(leaf.coords().keys()) if hasattr(leaf.coords(), "keys") else leaf.coords()}')
                        except Exception as e:  # noqa: BLE001
                            log(f'  leaf {key}: {e!r}')
                    out[f'B/{los}/{i}/{key}'] = np.real(np.asarray(leaf.value()))
                    for q in ('shotnoise', 'num_shotnoise', 'norm', 'nmodes'):
                        try:
                            out[f'B/{q}/{los}/{i}/{key}'] = np.asarray(leaf.values(q))
                        except Exception:  # noqa: BLE001
                            pass
            except Exception as e:  # noqa: BLE001
                if j == 0:
                    log(f'  bispectrum read: {e!r}')
    fn = os.path.join(out_dir, f'box_{tag}.npz')
    np.savez_compressed(fn, **out)
    log(f'saved {fn} ({os.path.getsize(fn) / 1e6:.1f} MB, {len(out)} arrays)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['read'], default='read')
    ap.add_argument('--tracer', default='LRG'); ap.add_argument('--z', default='0.500'); ap.add_argument('--cosmo', default='c000')
    ap.add_argument('--hod', default='hod-base_B')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args(argv)
    stage_read(args)


if __name__ == '__main__':
    main()
