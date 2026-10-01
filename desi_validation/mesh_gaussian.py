"""Exact Gaussian covariance of P0 on an FFT mesh, for a window given by randoms with weights
(adapted from the scratch script lrg1_mesh_gaussian.py run at NERSC).

For a Gaussian field with window W = m^2 and shot-noise density S, the estimator
P0_i = 1/(N_i norm) sum_{k in shell i} |F(k)|^2 has

    Var(P0_i) = 2 / (N_i^2 norm^2) sum_{k, k' in i} |P(k) W(k - k') + S(k - k')|^2

(P(k) -> sqrt(P(k) P(k')) in the cross term), done exactly on the grid in configuration space.
P(k, mu) is the mocks' mean multipoles x norm / int W (thecov's model_norm_correction), with a
global line of sight. Every |W|^2, W S*, |S|^2 is a cross-spectrum of fields painted from DISJOINT
random subsets, so random shot noise does not bias it.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.fft as sfft

WORKERS = None


def _workers():
    import os
    return WORKERS or os.cpu_count()


class Mesh:
    def __init__(self, pos, cell=12.0, pad=2.0, log=print):
        lo, hi = pos.min(0), pos.max(0)
        ext = hi - lo
        self.cell = cell
        self.shape = tuple(sfft.next_fast_len(int(np.ceil(pad * e / cell)) + 2) for e in ext)
        self.lo = lo - cell
        self.size = int(np.prod(self.shape))
        k1 = [2 * np.pi * np.fft.fftfreq(n, d=cell) for n in self.shape]
        self.k1 = [k.reshape([-1 if a == b else 1 for b in range(3)]) for a, k in enumerate(k1)]
        kx, ky, kz = self.k1
        self.kmag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2).astype(np.float32)
        s = [np.sinc(k * cell / (2 * np.pi)) for k in self.k1]
        self.sinc = (s[0] * s[1] * s[2]).astype(np.float32)
        log(f'mesh {self.shape} ({self.size / 1e6:.0f}M cells, cell {cell} Mpc/h, kNyq {np.pi / cell:.3f})')

    def paint(self, pos, w):
        i = np.floor((pos - self.lo) / self.cell).astype(np.int64)
        flat = np.ravel_multi_index(i.T, self.shape)
        return (np.bincount(flat, weights=w, minlength=self.size).reshape(self.shape) / self.cell ** 3).astype(np.float32)

    def ft(self, field):
        return (sfft.fftn(field, workers=_workers()) * (self.cell ** 3) / self.sinc).astype(np.complex64)

    def corr(self, F1, F2, qmax):
        f = np.real(F1 * np.conj(F2))
        f[self.kmag > qmax] = 0.0
        return np.real(sfft.ifftn(f, workers=_workers())).astype(np.float32)

    def shell_fft(self, weight, k_lo, k_hi):
        m = (self.kmag >= k_lo) & (self.kmag < k_hi)
        a = np.where(m, weight, 0.0)
        return np.real(sfft.fftn(a, workers=_workers())).astype(np.float32), int(m.sum())


def self_test(log=print):
    """Uniform periodic box: Var P0 = 2 P^2 / N_i."""
    class Box(Mesh):
        def __init__(self, n=48, L=480.0):
            self.cell = L / n; self.shape = (n,) * 3; self.size = n ** 3
            k1 = 2 * np.pi * np.fft.fftfreq(n, d=self.cell)
            self.k1 = [k1.reshape([-1 if a == b else 1 for b in range(3)]) for a in range(3)]
            kx, ky, kz = self.k1
            self.kmag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
            self.sinc = np.ones(self.shape)
    b = Box()
    nbar, P = 3e-4, 1e4
    Wk = b.ft(np.full(b.shape, nbar ** 2))
    g = b.corr(Wk, Wk, qmax=np.inf)
    norm = nbar ** 2 * 480.0 ** 3
    A1, N1 = b.shell_fft(np.full(b.shape, P), 0.1, 0.11)
    v = 2 * np.sum(g * A1 * A1, dtype=np.float64) / (N1 * N1 * norm ** 2)
    ok = abs(v / (2 * P ** 2 / N1) - 1) < 1e-4
    log(f'mesh self-test: Var / (2 P^2 / N) = {v / (2 * P ** 2 / N1):.6f} ({"ok" if ok else "FAILED"})')
    return ok


def windows_from_tracer(mesh, tr, num_sn, n_split=4, seed=1):
    """W_s = alpha_s sum_{r in s} w_r m_r delta (thecov's m = tr.mw) and S_s with integral num_sn,
    for n_split disjoint random subsets. Returns (W list, S list, int W)."""
    lab = np.random.default_rng(seed).integers(0, n_split, len(tr.w))
    W, S = [], []
    for s in range(n_split):
        sel = lab == s
        a_s = tr.alpha * tr.w.sum() / tr.w[sel].sum()
        W.append(mesh.ft(mesh.paint(tr.pos[sel], a_s * tr.w[sel] * tr.mw[sel])))
        S.append(mesh.ft(mesh.paint(tr.pos[sel], num_sn * tr.w[sel] ** 2 / np.sum(tr.w[sel] ** 2))))
    return W, S, float(tr.alpha * np.sum(tr.w * tr.mw))


def p0_variance(mesh, W, S, intW, spec, norm, shells, los, qcut=0.1, log=print, label=''):
    """Var(P0) per shell index in `shells` (indices into spec['k']), two independent estimates
    (subset pairs (0,1) and (2,3)). Returns array (2, len(shells))."""
    terms = []
    for i, j in ((0, 1), (2, 3)):
        terms.append(dict(WW=mesh.corr(W[i], W[j], qcut),
                          WS=0.5 * (mesh.corr(W[i], S[j], qcut) + mesh.corr(W[j], S[i], qcut)),
                          SS=mesh.corr(S[i], S[j], qcut)))
    nb, edges = len(spec['k']), spec['k_edges']
    mean = spec['vectors'].mean(0)
    kx, ky, kz = mesh.k1
    los = np.asarray(los, float) / np.linalg.norm(los)     # global line of sight (small effect on P0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = np.nan_to_num((kx * los[0] + ky * los[1] + kz * los[2]) / mesh.kmag).astype(np.float32)
    leg = {0: 1.0, 2: 0.5 * (3 * mu ** 2 - 1), 4: (35 * mu ** 4 - 30 * mu ** 2 + 3) / 8}
    km = np.clip(mesh.kmag, spec['k'][0], spec['k'][-1])
    P = sum(np.interp(km, spec['k'], mean[i * nb:(i + 1) * nb]) * leg[ell] for i, ell in enumerate(spec['ells']))
    P = (P * norm / intW).astype(np.float32)
    del km, mu, leg
    out = np.zeros((2, len(shells)))
    t0 = time.time()
    for j, i in enumerate(shells):
        AP, N_i = mesh.shell_fft(P, edges[i], edges[i + 1])
        AR, _ = mesh.shell_fft(np.sqrt(np.maximum(P, 0)), edges[i], edges[i + 1])
        A1, _ = mesh.shell_fft(np.ones(1, np.float32), edges[i], edges[i + 1])
        for e, c in enumerate(terms):
            pp = 2 * np.sum(c['WW'] * AP * AP, dtype=np.float64)
            ps = 4 * np.sum(c['WS'] * AR * AR, dtype=np.float64)
            ss = 2 * np.sum(c['SS'] * A1 * A1, dtype=np.float64)
            out[e, j] = (pp + ps + ss) / (N_i * N_i * norm ** 2)
    log(f'   mesh {label}: {len(shells)} shells in {time.time() - t0:.0f} s')
    return out
