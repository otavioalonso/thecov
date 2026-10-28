"""Exact (closed-form thinning) discreteness terms of one Abacus cubic HOD box, plane-parallel multipoles l = 0, 2, 4.

    python -u -m desi_validation.box_thinning --phase 0 --los z [--single]

Reads the abacus_HF_cubic DR2_v1.0 LRG z0.5 catalogue of AbacusSummit_base_c000_ph<phase>, applies the redshift-space
displacement along the line of sight (v_los / (a H), c000 cosmology), measures P_l on the exact k_f = 2 pi / L grid (NUFFT,
unit weights, norm = N^2 / V, shot noise N subtracted for l = 0) and the thinning closed form of thinning_closed_form.py
(lin_ll' for all blocks, quad for (0, 0), zero-padded). The plane-parallel line of sight is obtained by placing the observer
at 1e8 Mpc/h along -los (x^_g = los^ to 1e-8). In a box the Gaussian covariance is diagonal in k, so for k != k'
    Cov_box = [lin - 2 P/n^2] (B/n) + P/n^2 + T0 = lin - quad + T0   (exact in the (0, 0) block),
which isolates T0 with no window, no SSC and no model for the discreteness. Output <out>/box_thinning_ph<phase>_<los>.npz.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation.grf_randoms import NTH, log  # noqa: E402
from desi_validation.thinning_closed_form import multipole_linear  # noqa: E402

CAT = ('/global/cfs/cdirs/desi/mocks/cai/abacus_HF_cubic/DR2_v1.0/AbacusSummit_base_c000_ph{ph:03d}_z0.500/'
       'abacus_HF_cubic_HODDIES_LRG_z0.500_DR2_v1.0_AbacusSummit_base_c000_ph{ph:03d}_clustering.dat.fits')
BOX_NPZ = os.path.expanduser('~/thecov/box/box_LRG_z0.500_c000_hod-base_B.npz')
L = 2000.0
# c000: h 0.6736, omega_b 0.02237, omega_c 0.1200, m_nu 0.06 eV -> Omega_m 0.3152; a H(z = 0.5) in (km/s) / (Mpc/h)
OM = (0.02237 + 0.1200 + 0.06 / 93.14) / 0.6736 ** 2
AH = 100.0 * np.sqrt(OM * 1.5 ** 3 + 1 - OM) / 1.5


def read_catalogue(fn):
    try:
        import fitsio
        d = fitsio.read(fn)
    except ImportError:
        from astropy.io import fits
        d = fits.getdata(fn)
    names = list(d.dtype.names)
    log(f'{os.path.basename(fn)}: {len(d)} rows; columns {names}')
    low = {n.lower(): n for n in names}
    pos = np.stack([np.asarray(d[low[c]], float) for c in ('x', 'y', 'z')], 1)
    vel = np.stack([np.asarray(d[low[c]], float) for c in ('vx', 'vy', 'vz')], 1) if 'vz' in low else None
    return pos, vel


def main(argv=None):
    import finufft
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--phase', type=int, default=0)
    ap.add_argument('--los', default='z', choices=['x', 'y', 'z'])
    ap.add_argument('--single', action='store_true')
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--quad-pad', type=int, default=2)
    ap.add_argument('--ells', type=int, nargs='+', default=[0, 2, 4])
    ap.add_argument('--out-dir', default=os.path.expanduser('~/thecov/box'))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    t_all = time.time()
    pos, vel = read_catalogue(CAT.format(ph=args.phase))
    ax = 'xyz'.index(args.los)
    if vel is not None:
        pos[:, ax] += vel[:, ax] / AH
    pos = (pos + L / 2) % L - L / 2                                       # box centred at 0, periodic
    N = len(pos)
    V = L ** 3
    norm = N ** 2 / V
    log(f'{N} galaxies, RSD along {args.los} (aH = {AH:.2f} km/s per Mpc/h), nbar {N / V:.4e}, norm {norm:.4g}')
    # observer far away along -los: x^_g -> los^ (plane-parallel multipoles through the end-point machinery)
    D = 1e8
    xd = pos.copy(); xd[:, ax] += D
    wd = np.ones(N)
    xr = np.zeros((1, 3)); xr[0, ax] = D; wr = np.zeros(1); alpha = 0.0
    lo = np.array([-L / 2, -L / 2, -L / 2]); lo[ax] += D
    edges = np.arange(0.02, 0.3001, 0.005); nb = len(edges) - 1; k = 0.5 * (edges[1:] + edges[:-1])
    kf = 2 * np.pi / L
    n = 2 * int(np.ceil(edges[-1] / kf)) + 4
    ko = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KO = np.meshgrid(ko, ko, ko, indexing='ij')
    kom = np.sqrt(KO[0] ** 2 + KO[1] ** 2 + KO[2] ** 2).ravel()
    del KO
    io = np.digitize(kom, edges) - 1
    io[(io < 0) | (io >= nb)] = nb
    nsh = np.bincount(io, minlength=nb + 1)[:nb]
    opts = dict(eps=args.eps, nthreads=NTH, modeord=1)
    th = lambda x: [np.ascontiguousarray(2 * np.pi * (x[:, i] - lo[i]) / L) for i in range(3)]
    log(f'grid {n}^3 (k_f {kf:.5f}), {nb} bins, modes per bin {nsh[0]} ... {nsh[-1]}')
    t0 = time.time()
    thd = th(xd)
    F1 = finufft.nufft3d1(*thd, wd.astype(complex), (n,) * 3, isign=-1, **opts).ravel()
    P1 = (np.bincount(io, np.abs(F1) ** 2, nb + 1)[:nb] / nsh - N) / norm
    log(f'  F_1 done ({time.time() - t0:.0f} s); P0 (first, mid, last) {P1[0]:.1f} {P1[nb // 2]:.1f} {P1[-1]:.1f}')
    # compare with the pipeline's P0 of this box / los (dk = 0.001 bins rebinned by 5 from k = 0.02)
    cmp = {}
    if os.path.exists(BOX_NPZ):
        bz = np.load(BOX_NPZ, allow_pickle=True)
        key = f'P/{args.los}/{args.phase}/0'
        if key in bz.files:
            nm = bz['nmodes']; p = bz[key]
            groups = [np.arange(20 + 5 * i, 25 + 5 * i) for i in range(nb)]
            p5 = np.array([np.sum(p[g] * nm[g]) / np.sum(nm[g]) for g in groups])
            cmp['P0_pipeline'] = p5
            log(f'  pipeline P0 (first, mid, last) {p5[0]:.1f} {p5[nb // 2]:.1f} {p5[-1]:.1f}; ratio NUFFT/pipeline mean {np.mean(P1 / p5):.4f} rms {np.std(P1 / p5):.4f}')
    # l = 0 linear term (one shell per type-2 transform)
    t0 = time.time()
    F1c = np.conj(F1).reshape(n, n, n); io3 = io.reshape(n, n, n)
    Gb = np.empty((nb, N))
    for i in range(nb):
        v = finufft.nufft3d2(*thd, np.where(io3 == i, F1c, 0), isign=-1, **opts)
        Gb[i] = (2 * v.real * wd / nsh[i] - 2 * wd ** 2) / norm
        if i in (0, nb - 1):
            log(f'    shell {i}: {time.time() - t0:.0f} s')
    Gsum = Gb @ Gb.T
    del F1c, Gb
    log(f'  linear (B/n-like) term done ({time.time() - t0:.0f} s)')
    # quadratic term on the padded grid
    t0 = time.time()
    qp = args.quad_pad; n2 = qp * n
    W2 = finufft.nufft3d1(*thd, (wd ** 2).astype(complex), (n2,) * 3, isign=-1, **opts)
    PW = sfft.rfftn((np.abs(W2) ** 2).astype(np.float64), workers=NTH)
    del W2
    ko2 = 2 * np.pi * np.fft.fftfreq(n2, d=L / n2)
    KO2 = np.meshgrid(ko2, ko2, ko2, indexing='ij')
    kom2 = np.sqrt(KO2[0] ** 2 + KO2[1] ** 2 + KO2[2] ** 2).ravel()
    del KO2
    io2 = np.digitize(kom2, edges) - 1
    io2[(io2 < 0) | (io2 >= nb)] = nb
    assert np.array_equal(np.bincount(io2, minlength=nb + 1)[:nb], nsh)
    io23 = io2.reshape(n2, n2, n2)
    Q = np.zeros((nb, nb))
    for i in range(nb):
        INDi = sfft.rfftn((io23 == i).astype(np.float64), workers=NTH)
        corr = sfft.irfftn(np.conj(INDi) * PW, s=(n2,) * 3, workers=NTH).ravel()
        Q[i] = np.bincount(io2, corr, nb + 1)[:nb]
    Q = Q / np.outer(nsh, nsh)
    quad = (Q + Q - 2 * float(np.sum(wd ** 4))) / norm ** 2
    del PW, io2, io23
    log(f'  quadratic (P/n^2-like) term done ({time.time() - t0:.0f} s)')
    out = dict(k=k, k_edges=edges, P1=P1, lin=Gsum, quad=quad, N=N, norm=norm, nbar=N / V, boxsize=L, nsh=nsh, los=args.los,
               phase=args.phase, aH=AH, **cmp)
    ells = tuple(sorted(set(args.ells) | {0}))
    if max(ells) > 0:
        out.update(multipole_linear(args, ells, xd, wd, xr, wr, alpha, lo, L, n, io, nsh, nb, norm, opts, th, Gsum))
    fn = os.path.join(args.out_dir, f'box_thinning_ph{args.phase:03d}_{args.los}.npz')
    np.savez(fn, **out)
    log(f'saved {fn} ({time.time() - t_all:.0f} s total)')


if __name__ == '__main__':
    main()
