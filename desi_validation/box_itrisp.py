"""Squeezed / hard trispectrum T(k1, q - k1, k2, -q - k2) measured INSIDE each Abacus cubic box (no box-to-box scatter), q -> 0.

    python -u -m desi_validation.box_itrisp --phases 0 1 ... 24 --los z x y [--n 256 --max-time 6600]

Per box and line of sight (redshift-space columns x_rsd, y_rsd, z_rsd of the abacus_HF_cubic DR2_v1.0 LRG z0.5 catalogues):
galaxies split at random into halves A, B; delta^X(k) = (1/N_X) sum_{g in X} exp(-i k.x_g) on the n^3 grid modes (type-1 NUFFT,
exact). For bands b (width 0.025 from k = 0.02) and leg weights w_0 = 1, w_2 = 5 L_2(mu_k):
    D_b^{X,l}(q) = sum_{k in b, q-k in b} delta^X(k) delta^X(q - k) w_l(q - k)  -  n_b^l(q) delta^X(q) / N_X     (self pairs removed)
    n_b^l(q)    = sum_{k in b, q-k in b} w_l(q - k)
computed as n^3 FFT[ifft(delta 1_b) ifft(delta 1_b w_l)] and kept only at the small |q| <= q_max (no aliasing: 2 k_max < k_Ny + ...).
For non-adjacent bands s, h and q != 0 every disconnected pairing vanishes, so with the halves on different legs
    V^3 <D_s^{A,l} D_h^{B,l'}(q)^*> / (n_s(q) n_h(q))  =  < w_l(k1) w_l'(k2) T(k1, q-k1, k2, -q-k2) >     (clustering only, distinct galaxies)
which at q -> 0 is the trispectrum entering the box covariance, C^T0_{sh} = T / V, EXCEPT for the beat-coupling term
proportional to P(q) (internal momentum q), absent at q = 0: it is removed offline with the in-box integrated bispectra
<D_b^{A,l}(q) delta^B(q)^*> and the long-mode power <delta^A(q) delta^B(q)^*> (all measured; no model).
Output per box/LOS: D (2 halves x n_band x 2 ells x n_q), delta^A, delta^B at the q modes, n_b^l(q), q vectors, N_A, N_B.
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
from desi_validation.box_thinning import CAT, read_catalogue  # noqa: E402

L = 2000.0


def read_rsd(fn, los):
    try:
        import fitsio
        d = fitsio.read(fn, columns=['x', 'y', 'z', f'{los}_rsd'])
    except ImportError:
        from astropy.io import fits
        d = fits.getdata(fn)
    pos = np.stack([np.asarray(d[c], float) for c in ('x', 'y', 'z')], 1)
    pos[:, 'xyz'.index(los)] = np.asarray(d[f'{los}_rsd'], float)
    return (pos + L / 2) % L


def main(argv=None):
    import finufft
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--phases', type=int, nargs='+', default=list(range(25)))
    ap.add_argument('--los', nargs='+', default=['z', 'x', 'y'])
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--qmax', type=float, default=2.5, help='|q| <= qmax k_f')
    ap.add_argument('--eps', type=float, default=1e-6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-time', type=float, default=6600.0)
    ap.add_argument('--out-dir', default=os.path.expanduser('~/thecov/box/itrisp'))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    n = args.n; kf = 2 * np.pi / L
    edges = 0.02 + 0.025 * np.arange(12); nband = len(edges) - 1
    m = np.fft.fftfreq(n, d=1.0 / n)                                   # integer mode numbers, FFT order
    M = np.meshgrid(m, m, m, indexing='ij')
    kmag = kf * np.sqrt(M[0] ** 2 + M[1] ** 2 + M[2] ** 2)
    assert 2 * edges[-1] + args.qmax * kf < 2 * np.pi * n / L and edges[-1] < np.pi * n / L, 'grid too small: products alias into small q'
    qsel = (kmag > 0) & (kmag <= args.qmax * kf + 1e-12)
    qidx = np.flatnonzero(qsel.ravel())
    qvec = np.stack([M[i].ravel()[qidx] for i in range(3)], 1)
    log(f'grid {n}^3, k_f {kf:.5f}, k_Ny {np.pi * n / L:.3f}; {nband} bands {edges[0]:.3f}..{edges[-1]:.3f}; {len(qidx)} q modes (|q| <= {args.qmax} k_f)')
    ind = [((kmag >= edges[b]) & (kmag < edges[b + 1])) for b in range(nband)]
    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    for los in args.los:
        ax = 'xyz'.index(los)
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = np.where(kmag > 0, kf * M[ax] / np.where(kmag > 0, kmag, 1), 0.0)
        w2 = 5 * (1.5 * mu ** 2 - 0.5)
        # normalisation counts n_b^l(q) (geometry only)
        nq = np.zeros((nband, 2, len(qidx)))
        for b in range(nband):
            fb = sfft.ifftn(ind[b].astype(np.complex128), workers=NTH)
            for il, wl in enumerate((None, w2)):
                g = fb if wl is None else sfft.ifftn(ind[b] * wl.astype(np.complex128), workers=NTH)
                nq[b, il] = (n ** 3 * sfft.fftn(fb * g, workers=NTH)).ravel()[qidx].real
        for ph in args.phases:
            fn_out = os.path.join(args.out_dir, f'itrisp_ph{ph:03d}_{los}.npz')
            if os.path.exists(fn_out):
                continue
            if time.time() - t_start > args.max_time:
                log('max time reached'); return
            t0 = time.time()
            pos = read_rsd(CAT.format(ph=ph), los)
            half = rng.random(len(pos)) < 0.5
            th = [np.ascontiguousarray(2 * np.pi * pos[:, i] / L) for i in range(3)]
            dl = {}
            for X, sel in (('A', half), ('B', ~half)):
                NX = int(sel.sum())
                F = finufft.nufft3d1(*[t[sel] for t in th], np.ones(NX, complex), (n,) * 3, isign=-1, eps=args.eps, modeord=1) / NX
                F[0, 0, 0] = 0.0
                dl[X] = (F, NX)
            D = np.zeros((2, nband, 2, len(qidx)), complex)
            for ix, X in enumerate('AB'):
                F, NX = dl[X]
                dq = F.ravel()[qidx]
                for b in range(nband):
                    fb = sfft.ifftn(F * ind[b], workers=NTH)
                    for il, wl in enumerate((None, w2)):
                        g = fb if wl is None else sfft.ifftn(F * ind[b] * wl, workers=NTH)
                        D[ix, b, il] = (n ** 3 * sfft.fftn(fb * g, workers=NTH)).ravel()[qidx] - nq[b, il] * dq / NX
            np.savez(fn_out, D=D, dA=dl['A'][0].ravel()[qidx], dB=dl['B'][0].ravel()[qidx], nq=nq, qvec=qvec, edges=edges,
                     NA=dl['A'][1], NB=dl['B'][1], L=L, los=los, phase=ph)
            log(f'  ph{ph:03d} los {los}: N {len(pos)} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
