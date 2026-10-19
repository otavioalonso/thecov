"""Mock power-spectrum multipole vectors (holi-v3-altmtl) as a compact npz, for covariance tests away from NERSC.

    python -u -m desi_validation.mock_vectors --bins LRG1 QSO --regions NGC SGC

Output: <--out>/mock_vectors.npz with <bin>/<region>/{vectors (N_mock, 3 n_k), mocks, k, norm}. Mock catalogues only.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bins', nargs='+', default=['LRG1', 'QSO'])
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--out', default=os.path.expanduser('~/thecov/responses'))
    args = ap.parse_args()
    cfg = pl.Config()
    paths = dc.Paths(kind='holi_v3', mock=173)
    out = {}
    for b in args.bins:
        tracer, zr = dc.TRACER_SPECS[b]
        for r in args.regions:
            fns = paths.spectra_fns(tracer, zr, r)
            s = dc.read_spectra(fns, kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=(0, 2, 4))
            out[f'{b}/{r}/vectors'] = s['vectors'].astype(np.float64)
            out[f'{b}/{r}/mocks'] = np.array([int(re.search(r'mock(\d+)', f).group(1)) for f in fns])
            out[f'{b}/{r}/k'] = s['k']
            out[f'{b}/{r}/norm'] = s['norm']
            print(b, r, s['vectors'].shape, flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'mock_vectors.npz'), **out)


if __name__ == '__main__':
    main()
