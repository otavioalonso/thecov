"""Toy validation of the random-based subtraction in thinning_closed_form (lin_ell_R, quad_R).

    python -m desi_validation.rand_sub_toy --nreal 8 --sigma 0.0      # Poisson null: lin - lin_R and quad_R vanish
    python -m desi_validation.rand_sub_toy --nreal 8 --sigma 0.15     # Gaussian field: lin - lin_R - 2 quad_R vanishes (B = 0)

Survey: a spherical-shell sector seen from the origin (end-point LOS), radial n(r) gradient and a cubic hole, FKP-like weights
varying with r; randoms N_r, galaxies a Bernoulli sample p0 (1 + delta) of a disjoint random set, delta a band-limited Gaussian
field (sigma) evaluated exactly (type-2 NUFFT). The closed-form code paths (multipole_linear, quadratic_terms) are called as in
production with ells (0, 2, 4).
"""
from __future__ import annotations

import argparse
import types

import numpy as np

from desi_validation import thinning_closed_form as tcf


def survey(rng, n):
    """points in the sector: r in [400, 700] with n(r) ~ r^-1, cos(theta) > 0.6 around z, minus a cube hole"""
    out = np.empty((0, 3))
    while len(out) < n:
        m = 3 * n
        u = rng.random(m)
        r = (400 ** 2 + u * (700 ** 2 - 400 ** 2)) ** 0.5          # density ~ 1/r in volume
        ct = 0.6 + 0.4 * rng.random(m); ph = 2 * np.pi * rng.random(m); st = np.sqrt(1 - ct ** 2)
        x = np.stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct], 1)
        hole = np.all(np.abs(x - np.array([0, 0, 520])) < 60, axis=1)
        out = np.vstack([out, x[~hole]])
    return out[:n]


def weight(x):
    r = np.linalg.norm(x, axis=1)
    return 1.0 / (1.0 + 0.6 * (550 / r))                          # FKP-like, varies with the density


def gaussian_field(rng, xs, L, lo, sigma, kc=0.08):
    import finufft
    import scipy.fft as sfft
    if sigma == 0:
        return [np.zeros(len(x)) for x in xs]
    ng = 64
    kk = 2 * np.pi * np.fft.fftfreq(ng, d=L / ng)
    K = np.sqrt(sum(a ** 2 for a in np.meshgrid(kk, kk, kk, indexing='ij')))
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where((K > 0) & (K < kc), (K / 0.02) ** -1.5 / (1 + (K / 0.02) ** -2), 0.0)
    W = sfft.fftn(rng.standard_normal((ng,) * 3)) * np.sqrt(P)
    out = []
    for x in xs:
        th = [np.ascontiguousarray(2 * np.pi * (x[:, i] - lo[i]) / L) for i in range(3)]
        out.append(finufft.nufft3d2(*th, np.ascontiguousarray(W), isign=1, eps=1e-9, modeord=1).real)
    s = np.std(out[0]) if np.std(out[0]) > 0 else 1.0
    return [d * sigma / s for d in out]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--nreal', type=int, default=6)
    ap.add_argument('--sigma', type=float, default=0.0)
    ap.add_argument('--nr', type=int, default=400000)
    ap.add_argument('--p0', type=float, default=0.06)
    ap.add_argument('--rand-sub', type=float, default=4.0)
    ap.add_argument('--out', default='rand_sub_toy.npz')
    a = ap.parse_args(argv)
    rng = np.random.default_rng(1)
    xr = survey(rng, a.nr); wr = weight(xr)
    xpar = survey(rng, a.nr)                                       # parent set for the galaxies (disjoint from the randoms)
    lo, hi = xr.min(0), xr.max(0)
    L = 1.15 * float(np.max(hi - lo)); lo = 0.5 * (lo + hi) - 0.5 * L
    kf = 2 * np.pi / L
    edges = np.arange(2.5, 2.5 + 2.0 * 12, 2.0) * kf                 # 11 bins, 2 k_f wide (as production: 2.5 k_f)
    nb = len(edges) - 1
    n = 2 * int(np.ceil(edges[-1] / kf)) + 4
    ko = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    kom = np.sqrt(sum(q ** 2 for q in np.meshgrid(ko, ko, ko, indexing='ij'))).ravel()
    io = np.digitize(kom, edges) - 1; io[(io < 0) | (io >= nb)] = nb
    nsh = np.bincount(io, minlength=nb + 1)[:nb]
    opts = dict(eps=1e-9, nthreads=8, modeord=1)
    th = lambda x: [np.ascontiguousarray(2 * np.pi * (x[:, i] - lo[i]) / L) for i in range(3)]
    args = types.SimpleNamespace(single=False)
    ells = (0, 2, 4)
    res = {q: [] for q in ('lin', 'linR', 'quad', 'quadR', 'P', 'ngal')}
    for j in range(a.nreal):
        rj = np.random.default_rng(100 + j)
        d = gaussian_field(rj, [xpar], L, lo, a.sigma)[0]
        sel = rj.random(len(xpar)) < np.clip(a.p0 * (1 + d), 0, 1)
        xd = xpar[sel]; wd = weight(xd)
        alpha = float(np.sum(wd) / np.sum(wr))
        norm = alpha * float(np.sum(wr * wr * 1.0)) * (len(wd) / L ** 3)    # any fixed positive normalisation: ratios only
        s_all = wr ** 2
        nsub = min(len(wr), int(a.rand_sub * len(wd)))
        sub = rj.choice(len(wr), nsub, replace=False)
        rsub = dict(x=xr[sub], w=wr[sub], s=s_all[sub], weff=wr[sub])
        G0 = np.zeros((nb, nb))
        o = tcf.multipole_linear(args, ells, xd, wd, xr, wr, alpha, lo, L, n, io, nsh, nb, norm, opts, th, G0, rsub=rsub)
        q, qR = tcf.quadratic_terms(th(xd), th(xr), wd, s_all, 2 * n, L, edges, nb, nsh, norm, opts)
        res['lin'].append(o['lin_ell']); res['linR'].append(o['lin_ell_R']); res['quad'].append(q); res['quadR'].append(qR)
        res['P'].append(o['P1_ell']); res['ngal'].append(len(wd))
        print(f'realisation {j}: {len(wd)} galaxies, {nsub} randoms in the subtraction', flush=True)
    np.savez(a.out, **{q: np.array(v) for q, v in res.items()}, k=0.5 * (edges[1:] + edges[:-1]), nsh=nsh, sigma=a.sigma)
    lin, linR, quad, quadR = (np.array(res[q]) for q in ('lin', 'linR', 'quad', 'quadR'))
    D = np.sqrt(np.diag(lin.mean(0)))
    U = np.outer(D, D)
    print(f'\nsigma = {a.sigma}; units: sqrt(lin_ii lin_jj) (the unclustered linear term ~ g1); mean over {a.nreal} realisations +- error')
    blocks = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for (bi, bj) in blocks:
        s = np.s_[bi * nb:(bi + 1) * nb, bj * nb:(bj + 1) * nb]
        X = (lin - linR)[:, s[0], s[1]] / U[s]
        for lab, off in (('diag', 0), ('|i-j|=1', 1), ('|i-j|>=2', 2)):
            if off < 2:
                v = np.array([np.mean(np.diagonal(x, off)) for x in X])
            else:
                mask = np.abs(np.subtract.outer(np.arange(nb), np.arange(nb))) >= 2
                v = np.array([np.mean(x[mask]) for x in X])
            print(f'  (lin - lin_R) block ({2 * bi},{2 * bj}) {lab:9s}: {v.mean():+.4f} +- {v.std() / np.sqrt(len(v)):.4f}'
                  + (f'   [galaxy-only quad: {np.mean([np.mean(np.diagonal(qq, off)) for qq in quad / U[:nb, :nb]]):+.4f}]' if (bi, bj) == (0, 0) and off < 2 else ''))
    for lab, off in (('diag', 0), ('|i-j|=1', 1)):
        v = np.array([np.mean(np.diagonal(qq, off)) for qq in quadR / U[:nb, :nb]])
        w = np.array([np.mean(np.diagonal(x[:nb, :nb] - 2 * qq, off)) for x, qq in zip((lin - linR) / U, quadR / U[:nb, :nb])])
        print(f'  quad_R (0,0) {lab:9s}: {v.mean():+.4f} +- {v.std() / np.sqrt(len(v)):.4f};   (lin - lin_R - 2 quad_R)(0,0): {w.mean():+.4f} +- {w.std() / np.sqrt(len(w)):.4f}')


if __name__ == '__main__':
    main()
