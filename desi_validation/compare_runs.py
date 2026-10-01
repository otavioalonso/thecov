"""Paired comparison of two run_window_shape reports on the SAME realisations (e.g. complete vs altmtl
versions of the same Abacus mocks): per region and construction of m, <chi2>/n of each run and of
the mock-by-mock difference, which cancels most of the sample variance shared by the two versions.

    python -m desi_validation.compare_runs A/window_shape_report.json B/window_shape_report.json [--labels complete altmtl]
"""
import argparse
import json

import numpy as np


def load(fn):
    rep = json.load(open(fn))
    return rep[0] if isinstance(rep, list) else rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('reports', nargs=2)
    ap.add_argument('--labels', nargs=2, default=['A', 'B'])
    a = ap.parse_args()
    ra, rb = (load(fn) for fn in a.reports)
    la, lb = a.labels
    for b in ra.get('test2', {}):
        if b not in rb.get('test2', {}):
            continue
        print(f'\n== {b}: <chi2>/n per run (deviation in sigma) and of the paired difference {la} - {lb}')
        print(f"{'case':38s}{'N':>5s}{la:>18s}{lb:>18s}{'paired diff / n':>22s}")
        for case, ca in ra['test2'][b].items():
            cb = rb['test2'][b].get(case)
            if cb is None or 'chi2_i' not in ca or 'chi2_i' not in cb:
                continue
            ia = dict(zip(ca['mock_ids'], ca['chi2_i'])); ib = dict(zip(cb['mock_ids'], cb['chi2_i']))
            common = sorted(set(ia) & set(ib))
            n = ca['n']
            d = np.array([ia[m] - ib[m] for m in common]) / n
            err = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            print(f"{case:38s}{len(common):5d}{ca['chi2']:10.4f} ({ca['chi2_sigma']:+4.1f}){cb['chi2']:10.4f} ({cb['chi2_sigma']:+4.1f})"
                  f"{d.mean():+12.4f} +- {err:.4f}")
            if not common:
                print('   (no mock ids in common: not the same realisations)')


if __name__ == '__main__':
    main()
