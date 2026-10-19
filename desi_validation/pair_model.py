"""Is the clustered pair ("P/n^2") term of a catalogue more than its own two-point clustering? Exact model vs measurement.

    python -u -m desi_validation.pair_model --bin QSO --region NGC --mocks 173 --data

The clustered part of the pair term (pair_diagnostics.py) is 2 sum_{g != h} w_g^2 w_h^2 <e^{i q.(x_g - x_h)}> minus its
smooth-density expectation. If the objects are a Poisson sampling of a field with power P, its expectation is EXACTLY
    E[|W2(q)|^2]_clustered = (1 / V) sum_p P(p) |W~(q - p)|^2,     W~ = transform of n w^2 (the randoms, alpha-scaled),
the full window convolution (no local approximation of P(|k - k'|)). Anything above it is non-Poisson (one-halo pairs,
duplicates, exclusion) or weights correlated with the density. P(p) is the mean shot-noise-subtracted P0 of the holi
spectra at dk = 0.001 (window-convolved: an underestimate of the true power at p < ~0.01), isotropic, zero below the
first bin. The model goes through the same shell-pair averaging as the measurement. Output: npz next to pairs_*.npz.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402
from desi_validation.grf_randoms import NTH, log  # noqa: E402
from desi_validation.pair_diagnostics import paths_for, OFFS  # noqa: E402

KR = ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3))


def main():
    import finufft
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='QSO')
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--mocks', type=int, nargs='*', default=[173])
    ap.add_argument('--data', action='store_true')
    ap.add_argument('--data-dir', default='/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2')
    ap.add_argument('--random-files', type=int, default=2)
    ap.add_argument('--n-spectra', type=int, default=100, help='mock spectra averaged for P(p)')
    ap.add_argument('--kmin-rows', type=float, default=0.05)
    ap.add_argument('--pad', type=float, default=1.1)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--out', default=os.path.expanduser('~/thecov/thinning'))
    args = ap.parse_args()
    b, r = args.bin, args.region
    tracer, zr = dc.TRACER_SPECS[b]
    cfg = pl.Config()
    p0 = paths_for(False, 173, args.data_dir)
    fns = p0.spectra_fns(tracer, zr, r)
    s1 = dc.read_spectra(fns[:1], kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=(0,))
    k, edges = s1['k'], np.asarray(s1['k_edges'])
    nb = len(k)
    sf = dc.read_spectra(fns[:args.n_spectra], kmin=0.0, kmax=0.6, rebin=1, ells=(0,))
    kf_, Pf = sf['k'], sf['vectors'].mean(0)[:len(sf['k'])]
    log(f'P(p) from {len(sf["vectors"])} spectra, k {kf_[0]:.4f}-{kf_[-1]:.3f}: P(0.005) = {np.interp(0.005, kf_, Pf):.4g}, P(0.02) = '
        f'{np.interp(0.02, kf_, Pf):.4g}, P(0.1) = {np.interp(0.1, kf_, Pf):.4g}, P(0.3) = {np.interp(0.3, kf_, Pf):.4g}')
    rc0 = dc.load_region(p0, b, r, n_random_files=1)
    xr0 = dc.sky_to_cartesian(rc0.randoms['RA'], rc0.randoms['DEC'], rc0.randoms['Z'])
    del rc0
    lo, hi = xr0.min(0) - 50, xr0.max(0) + 50
    L = args.pad * float(np.max(hi - lo))
    lo = 0.5 * (lo + hi) - 0.5 * L
    n = 2 * int(np.ceil(edges[-1] / (2 * np.pi / L))) + 4
    n += n % 2
    del xr0
    ko = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KO = np.meshgrid(ko, ko, ko, indexing='ij')
    kom = np.sqrt(KO[0] ** 2 + KO[1] ** 2 + KO[2] ** 2)
    del KO
    Pgrid = np.where(kom >= kf_[0] - 0.0005, np.interp(kom, kf_, Pf, left=0.0, right=Pf[-1]), 0.0).astype(np.float32)
    io = np.digitize(kom.ravel(), edges) - 1
    del kom
    io[(io < 0) | (io >= nb)] = nb
    nsh = np.bincount(io, minlength=nb + 1)[:nb]
    io3 = io.reshape(n, n, n)
    rows = np.flatnonzero(k >= args.kmin_rows)
    Pk_r = sfft.rfftn(Pgrid, workers=NTH)
    del Pgrid
    opts = dict(eps=args.eps, nthreads=NTH, modeord=1)
    th = lambda x: [np.ascontiguousarray(2 * np.pi * (x[:, i] - lo[i]) / L) for i in range(3)]
    log(f'{b} {r}: box {L:.0f}, grid {n}^3')
    od173 = os.path.join(args.out, f'{b}_{r}_mock173')
    dG = None
    if os.path.exists(os.path.join(od173, 'thecov_p1.npz')):
        norm173 = float(np.load(os.path.join(od173, 'thinning.npz'))['norm'])
        dG = np.diag(np.load(os.path.join(od173, 'thecov_p1.npz'))['C_G'])[:nb]
    cats = [(m, False) for m in args.mocks] + ([(0, True)] if args.data else [])
    out = {}
    for m, data in cats:
        name = 'data' if data else f'mock{m}'
        t0 = time.time()
        rc = dc.load_region(paths_for(data, m, args.data_dir), b, r, n_random_files=args.random_files)
        wd, wr = dc.total_weight(rc.data), dc.total_weight(rc.randoms)
        xd = dc.sky_to_cartesian(rc.data['RA'], rc.data['DEC'], rc.data['Z'])
        xr = dc.sky_to_cartesian(rc.randoms['RA'], rc.randoms['DEC'], rc.randoms['Z'])
        del rc
        a2 = float(np.sum(wd ** 2) / np.sum(wr ** 2))
        PW = []
        selfs = [float(np.sum(wd ** 4)), float(np.sum((a2 * wr ** 2) ** 2))]
        for x, v, s_ in ((xd, wd ** 2, selfs[0]), (xr, a2 * wr ** 2, selfs[1])):
            W2 = finufft.nufft3d1(*th(x), v.astype(complex), (n,) * 3, isign=-1, **opts)
            pw = (np.abs(W2) ** 2 - s_).astype(np.float32)          # pair part only (self pairs removed)
            del W2
            PW.append(sfft.rfftn(pw, workers=NTH))
            if len(PW) == 2:                                        # model: (1/V) sum_p P(p) pw_R(q - p)
                model = sfft.irfftn(Pk_r * PW[1], s=(n,) * 3, workers=NTH).astype(np.float32) / L ** 3
                PW.append(sfft.rfftn(model, workers=NTH))
                del model
            del pw
        log(f'  {name}: transforms done ({time.time() - t0:.0f} s)')
        Q = np.zeros((3, nb, nb))
        for i in rows:
            IND = sfft.rfftn((io3 == i).astype(np.float32), workers=NTH)
            for f in range(3):
                corr = sfft.irfftn(np.conj(IND) * PW[f], s=(n,) * 3, workers=NTH).ravel()
                Q[f, i] = np.bincount(io, corr, nb + 1)[:nb]
        Q = 2 * Q / np.maximum(np.outer(nsh, nsh), 1)[None]          # M_1 + M_2 (equal by shell symmetry)
        D, R, M = Q
        ex = D - R                                                   # clustered pair part, measured
        out[name] = Q
        log(f'  {name}: shell loop done ({time.time() - t0:.0f} s)')
        rr = np.flatnonzero(k > 0.15)
        sd = np.mean(np.diag(R)[rr])
        print(f'\n== {name} ({b} {r}): clustered pair term, measured vs exact clustering model, units of the smooth (SS) diagonal, k > 0.15')
        print(f'   |i-j| = {OFFS}')
        for lab, A in (('measured', ex), ('model', M)):
            print(f'   {lab:9s}' + ' '.join(f'{np.mean([A[i, i + d] for i in rr if i + d < nb]) / sd:+.4f}' for d in OFFS))
        print('   ratio    ' + ' '.join(f'{np.mean([ex[i, i + d] for i in rr if i + d < nb]) / np.mean([M[i, i + d] for i in rr if i + d < nb]):7.3f}' for d in OFFS))
        print('   diag by band (0.05-0.1 | 0.1-0.2 | 0.2-0.3): measured ' + ' | '.join(f'{np.mean(np.diag(ex)[(k >= lo_) & (k < hi_)]) / np.mean(np.diag(R)[(k >= lo_) & (k < hi_)]):+.4f}' for lo_, hi_ in KR)
              + '   model ' + ' | '.join(f'{np.mean(np.diag(M)[(k >= lo_) & (k < hi_)]) / np.mean(np.diag(R)[(k >= lo_) & (k < hi_)]):+.4f}' for lo_, hi_ in KR))
        if dG is not None and not data and m == 173:
            print(f'   in units of diag G (thecov, mock 173): measured diag {np.mean(np.diag(ex)[rr]) / norm173 ** 2 / np.mean(dG[rr]):+.4f}, model diag '
                  f'{np.mean(np.diag(M)[rr]) / norm173 ** 2 / np.mean(dG[rr]):+.4f}; |i-j| = 3: measured '
                  f'{np.mean([ex[i, i + 3] / np.sqrt(dG[i] * dG[i + 3]) for i in rr if i + 3 < nb]) / norm173 ** 2:+.4f}, model '
                  f'{np.mean([M[i, i + 3] / np.sqrt(dG[i] * dG[i + 3]) for i in rr if i + 3 < nb]) / norm173 ** 2:+.4f}')
        fn = os.path.join(args.out, f'pairmodel_{b}_{r}.npz')
        np.savez(fn, k=k, offs=np.array(OFFS), **{f'{nm}/Q': v for nm, v in out.items()})
        log(f'saved {fn}')


if __name__ == '__main__':
    main()
