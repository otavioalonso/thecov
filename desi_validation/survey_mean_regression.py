"""Net survey-mean response of the files' P_hat_l, measured across the holi mocks with an explicit convention.

    python -u -m desi_validation.survey_mean_regression --bin LRG1 --workers 32

For every mock: the weighted galaxy count N_m of each cap (data file, WEIGHT only, z-cut) and the files' P_hat_l(k)
(the holi-v3-altmtl spectra, kmin 0.02, kmax 0.3, dk 0.005). Regressor: the survey-mean GALAXY overdensity
    delta_L = N_m / median(N) - 1          (NOT delta ln int n^2, which is 2 delta_L for a uniform change)
Slopes d ln P_hat_0 / d delta_L and d(P_hat_2 / <P_0>) / d delta_L per k and per band, bootstrap errors over mocks.
Also printed: the slope against delta ln sum_z N(z)^2 (HANDOFF 9.2's regressor) to fix the factor 2 between the two
conventions, sigma(delta_L), and the Poisson part of it. Defective mocks (an n(z) bin > 6 robust sigma from the median,
make_products.defective_mocks) are excluded. Expected (section 9 of grf_randoms_results.md): with data-matched randoms
P_hat responds as R_0(q -> 0) - 2 = -0.09 / -0.10 / -0.19 (0.05-0.1 / 0.1-0.2 / 0.2-0.3, LRG1 NGC) per unit delta_L.
Output: <--out>/survey_mean_regression_<bin>.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402

T0 = time.time()
KR = ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3))


def log(msg):
    print(f'[{time.time() - T0:6.0f} s] {msg}', flush=True)


def one_mock(args):
    mock, bin_, regions, n_bins = args
    tracer, zr = dc.TRACER_SPECS[bin_]
    paths = dc.Paths(kind='holi_v3', mock=mock)
    edges = np.linspace(zr[0], zr[1], n_bins + 1)
    out = {}
    for r in regions:
        try:
            d = dc.read_catalog(paths.data_fn(tracer, r), ['Z', 'WEIGHT'], paths.h5_group, required=('Z',))
            z = np.asarray(d['Z'], float)
            w = np.asarray(d.get('WEIGHT', np.ones_like(z)), float)
            out[r] = np.histogram(z, edges, weights=w)[0]
        except Exception as e:  # noqa: BLE001
            out[r] = None
    return mock, out


def slopes(y, x, nboot=400, seed=1):
    """OLS slope of every column of y (n_mock, n) on x (n_mock,), with bootstrap errors over mocks."""
    xc = x - x.mean()
    b = xc @ (y - y.mean(0)) / (xc @ xc)
    rng = np.random.default_rng(seed)
    bs = np.empty((nboot, y.shape[1]))
    for i in range(nboot):
        s = rng.integers(0, len(x), len(x))
        xs = x[s] - x[s].mean()
        bs[i] = xs @ (y[s] - y[s].mean(0)) / (xs @ xs)
    return b, bs.std(0, ddof=1)


def band_rows(k, b, e):
    return ' | '.join(f'{np.mean(b[(k >= lo) & (k < hi)]):+.3f} +- {np.sqrt(np.mean(e[(k >= lo) & (k < hi)] ** 2)) / np.sqrt(np.sum((k >= lo) & (k < hi)) / 2.5):.3f}'
                      for lo, hi in KR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--n-bins', type=int, default=20)
    ap.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument('--out', default=os.path.expanduser('~/thecov/responses'))
    ap.add_argument('--nsig', type=float, default=6.0, help='defective-mock cut on any n(z) bin (robust sigma)')
    args = ap.parse_args()
    b = args.bin
    tracer, zr = dc.TRACER_SPECS[b]
    cfg = pl.Config()
    os.makedirs(args.out, exist_ok=True)
    paths = dc.Paths(kind='holi_v3', mock=173)
    paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
    # the mocks with spectra, per region
    spec, ids = {}, {}
    for r in args.regions:
        fns = paths.spectra_fns(tracer, zr, r)
        ids[r] = np.array([int(re.search(r'mock(\d+)', fn).group(1)) for fn in fns])
        spec[r] = dc.read_spectra(fns, kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=(0, 2, 4))
        log(f'{r}: {len(fns)} spectra, {len(spec[r]["k"])} k bins')
    mocks = sorted(set().union(*[set(v.tolist()) for v in ids.values()]))
    tasks = [(m, b, args.regions, args.n_bins) for m in mocks]
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            res = dict(pool.map(one_mock, tasks, chunksize=4))
    else:
        res = dict(one_mock(t) for t in tasks)
    log(f'counts read for {len(res)} mocks')
    out = {}
    # defective mocks: any n(z) bin > nsig robust sigma from the median, in either cap
    bad = set()
    NZ = {}
    for r in args.regions:
        rows = [(m, res[m][r]) for m in ids[r] if res.get(m, {}).get(r) is not None]
        NZ[r] = (np.array([m for m, _ in rows]), np.array([h for _, h in rows], float))
        d = NZ[r][1] / np.median(NZ[r][1], 0) - 1
        mad = 1.4826 * np.median(np.abs(d - np.median(d, 0)), 0)
        hit = (np.abs(d - np.median(d, 0)) > args.nsig * mad).any(1)
        bad |= set(NZ[r][0][hit].tolist())
    log(f'{len(bad)} defective mocks excluded: {sorted(bad)}')
    for r in args.regions:
        m_nz, H = NZ[r]
        keep = ~np.isin(m_nz, list(bad))
        m_nz, H = m_nz[keep], H[keep]
        # align spectra and counts on mock id
        pos = {m: i for i, m in enumerate(ids[r])}
        sel = np.array([m in pos for m in m_nz])
        m_nz, H = m_nz[sel], H[sel]
        isp = np.array([pos[m] for m in m_nz])
        k = spec[r]['k']
        nk = len(k)
        V = spec[r]['vectors'][isp]
        P0, P2, P4 = V[:, :nk], V[:, nk:2 * nk], V[:, 2 * nk:3 * nk]
        N = H.sum(1)
        dL = N / np.median(N) - 1
        d_n2 = np.log(np.sum(H ** 2, 1)) - np.log(np.median(np.sum(H ** 2, 1)))          # HANDOFF 9.2's regressor
        d_norm = np.log(spec[r]['norm'][isp]) - np.log(np.median(spec[r]['norm'][isp]))
        n = len(N)
        pois = 1 / np.sqrt(N.mean())
        print(f'\n{r}: {n} mocks. sigma(delta_L) = {dL.std(ddof=1) * 100:.3f} % (Poisson {pois * 100:.3f} %, clustering '
              f'{np.sqrt(max(dL.var(ddof=1) - pois ** 2, 0)) * 100:.3f} %); sigma(delta ln sum N^2) = {d_n2.std(ddof=1) * 100:.3f} % '
              f'(= {d_n2.std(ddof=1) / dL.std(ddof=1):.2f} x sigma(delta_L)); sigma(delta_norm) = {d_norm.std(ddof=1) * 100:.3f} %, '
              f'corr(delta_norm, delta_L) = {np.corrcoef(d_norm, dL)[0, 1]:+.2f}')
        lnP0 = np.log(P0)
        p2 = P2 / P0.mean(0)
        p4 = P4 / P0.mean(0)
        res_r = {}
        for name, x in (('delta_L', dL), ('delta_ln_sumN2', d_n2), ('delta_norm', d_norm)):
            b0, e0 = slopes(lnP0, x)
            b2, e2 = slopes(p2, x)
            b4, e4 = slopes(p4, x)
            res_r[name] = dict(b0=b0, e0=e0, b2=b2, e2=e2, b4=b4, e4=e4)
            print(f'  regressor {name:15s} bands {KR}')
            print(f'    d ln P0 / dx      : {band_rows(k, b0, e0)}')
            print(f'    d (P2 / <P0>) / dx: {band_rows(k, b2, e2)}')
            print(f'    d (P4 / <P0>) / dx: {band_rows(k, b4, e4)}')
        print('  expected for delta_L with data-matched randoms: R_0(q->0) - 2 = -0.09 / -0.10 / -0.19 (bands 2-4, LRG1 NGC in-survey '
              'response); tree level -0.4 .. -0.8; per unit delta ln sum N^2 everything is halved.')
        out[f'{r}/mocks'], out[f'{r}/N'], out[f'{r}/NZ'], out[f'{r}/k'] = m_nz, N, H, k
        out[f'{r}/delta_L'], out[f'{r}/delta_ln_sumN2'], out[f'{r}/delta_norm'] = dL, d_n2, d_norm
        for name, dd in res_r.items():
            for key, v in dd.items():
                out[f'{r}/{name}/{key}'] = v
    fn = os.path.join(args.out, f'survey_mean_regression_{b}.npz')
    np.savez(fn, edges=np.linspace(zr[0], zr[1], args.n_bins + 1), excluded=np.array(sorted(bad)), **out)
    log(f'saved {fn}')


if __name__ == '__main__':
    main()
