"""Gaussian random fields on a spherical footprint measured with the Sugiyama FFT estimator, against
BispectrumGaussianCovariance (class 1).

Design.  Isotropic (real-space) P(k) on an N^3 mesh in a periodic box of side L; the observer at the box centre; the
field F = nbar 1_sphere (delta - mean_sphere(delta)) (the alpha step of the estimator: int F = 0, which with uniform
omega kills the k3 ~ 0 term).  Estimators (guide eq. est, S_000 = 1, S_202 = L_2(q1^ . x^)):

    B_000(i, j) = (1 / I3) int d^3x F(x) F_i(x) F_j(x),           F_i(x) = << e^{i q.x} F(q) >>_i  (shell-filtered field),
    B_202(i, j) = (4 pi / I3) sum_m int d^3x F(x) Y_2m(x^) F_{i,2m}(x) F_j(x),   F_{i,2m} = << e^{i q.x} conj(Y_2m(q^)) F(q) >>_i,

with << >>_i the mean over the lattice modes of shell i.  Because the omitted class-2 pairings are ~ dk/k of the
diagonal (20-40 % here), the same estimator is also run on the unmasked periodic box: there the Gaussian covariance is
known analytically for class 1 (first term of Sugiyama et al. 2020 eq. 49) and the ratio GRF/class-1 measures the
class-2 fraction (plus lattice effects); the window test is then ratio(sphere) = ratio(box) element by element.  A
quadratic field delta + eps (delta^2 - sigma^2), with B = 2 eps [P P + 2 perms], checks the estimator normalisation
through its mean.

Run:  python desi_validation/bispectrum_dev/grf_sphere.py [n_real]   (writes grf_sphere_results.npz + .png next to it)
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.fft as sfft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

L_BOX, N_MESH = 1300.0, 128
K_EDGES = np.arange(0.04, 0.2001, 0.04)
if os.environ.get('BISPEC_K_EDGES'):
    K_EDGES = np.array([float(v) for v in os.environ['BISPEC_K_EDGES'].split(',')])
ELLS = ((0, 0, 0), (2, 0, 2))


def P_model(k):
    return 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)


class Mesh:
    def __init__(self, L=L_BOX, N=N_MESH, k_edges=K_EDGES):
        self.L, self.N = float(L), int(N)
        self.dV = (L / N) ** 3
        kf = 2 * np.pi / L
        k1 = sfft.fftfreq(N, d=1.0 / N) * kf
        kx, ky, kz = np.meshgrid(k1, k1, k1, indexing='ij')
        self.k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
        kk = np.where(self.k > 0, self.k, 1.0)
        qhat = np.stack([kx / kk, ky / kk, kz / kk], axis=-1)
        self.Y2q = self._y2(qhat)                                   # (5, N, N, N) complex, Y_2m(q^), m = -2..2
        self.k_edges = np.asarray(k_edges)
        self.shells = [(self.k >= lo) & (self.k < hi) for lo, hi in zip(k_edges[:-1], k_edges[1:])]
        self.nmodes = np.array([s.sum() for s in self.shells])
        # positions relative to the observer at the box centre
        x1 = (np.arange(N) + 0.5) * (L / N) - L / 2
        X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
        self.r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        xhat = np.stack([X, Y, Z], axis=-1) / self.r[..., None]
        self.Y2x = self._y2(xhat)

    @staticmethod
    def _y2(nhat):
        from thecov.harmonics import sph_harm_table
        ct = np.clip(nhat[..., 2], -1, 1)
        ph = np.arctan2(nhat[..., 1], nhat[..., 0])
        Y = sph_harm_table(ct, ph, 2)
        return np.stack([Y[2, m + 2] for m in range(-2, 3)])

    def grf(self, seed):
        """Real Gaussian field delta(x) with power P (variance per mode P/L^3 in the dV-normalised FT convention)."""
        rng = np.random.default_rng(seed)
        white = rng.normal(size=(self.N,) * 3)
        wk = sfft.fftn(white, workers=2) / self.N ** 1.5                 # unit variance per mode
        # convention: delta_k = dV sum_x delta e^{-ikx} = dV fftn(delta), <|delta_k|^2> = L^3 P(k); with
        # delta = ifftn(d_k): <|d_k|^2> = P N^6 / L^3, and sigma^2 = sum_k P / L^3 = int d^3k/(2pi)^3 P
        dk = wk * np.sqrt(P_model(np.where(self.k > 0, self.k, 1e-10))) * self.N ** 3 / self.L ** 1.5
        dk[self.k == 0] = 0.0
        return sfft.ifftn(dk, workers=2).real

    def measure(self, F, I3):
        """B_000 and B_202 for all (i, j) (i <= j for 000, all for 202); returns dict ell -> array."""
        Fk = sfft.fftn(F, workers=2) * self.dV                             # F(q)
        nb = len(self.shells)
        # F_i(x) = (1/V_i) (1/L^3) sum_{q in i} e^{iqx} F(q);  ifftn = (1/N^3) sum -> (1/L^3) sum = ifftn * N^3 / L^3;
        # V_i = nmodes / L^3  ->  F_i = ifftn * N^3 / nmodes
        Fi = [sfft.ifftn(Fk * s, workers=2).real * self.N ** 3 / self.nmodes[i] for i, s in enumerate(self.shells)]
        out = {}
        b000 = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(i, nb):
                b000[i, j] = b000[j, i] = self.dV * np.sum(F * Fi[i] * Fi[j]) / I3
        out[(0, 0, 0)] = np.array([b000[i, j] for i in range(nb) for j in range(i, nb)])
        b202 = np.zeros((nb, nb))
        for i, s in enumerate(self.shells):
            G = np.zeros(F.shape)
            for m in range(5):
                Fim = sfft.ifftn(Fk * s * np.conj(self.Y2q[m]), workers=2) * self.N ** 3 / self.nmodes[i]
                G += (self.Y2x[m] * Fim).real
            for j in range(nb):
                b202[i, j] = 4 * np.pi * self.dV * np.sum(F * G * Fi[j]) / I3
        out[(2, 0, 2)] = b202.ravel()
        return out


_MESH = None


def _init(L=L_BOX, N=N_MESH):
    global _MESH
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    _MESH = Mesh(L=L, N=N)


def _one(args):
    seed, kind, nbar, R, eps = args
    m = _MESH
    d = m.grf(seed)
    if eps:
        d = d + eps * (d ** 2 - np.mean(d ** 2))
    if kind == 'box':
        F = nbar * d
        I3 = nbar ** 3 * m.L ** 3
    else:
        mask = m.r < R
        d = d - d[mask].mean()
        F = nbar * mask * d
        I3 = nbar ** 3 * mask.sum() * m.dV
    res = m.measure(F, I3)
    return np.concatenate([res[e] for e in ELLS])


def run(kind, n_real, nbar, R, eps=0.0, seed0=0, workers=None, L=L_BOX, N=N_MESH):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with Pool(workers, initializer=_init, initargs=(L, N)) as pool:
        rows = pool.map(_one, [(seed0 + s, kind, nbar, R, eps) for s in range(n_real)], chunksize=4)
    return np.array(rows)


def labels():
    nb = len(K_EDGES) - 1
    out = [(0, 0, 0, i, j) for i in range(nb) for j in range(i, nb)]
    out += [(2, 0, 2, i, j) for i in range(nb) for j in range(nb)]
    return out


if __name__ == '__main__':
    # usage: grf_sphere.py n_real [L N R tag]   (default: L = 1300, N = 128, R = 500 -> grf_sphere_measurements.npz)
    n_real = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    L = float(sys.argv[2]) if len(sys.argv) > 2 else L_BOX
    N = int(sys.argv[3]) if len(sys.argv) > 3 else N_MESH
    from sphere_check import sphere_tracer, R_SPHERE
    R = float(sys.argv[4]) if len(sys.argv) > 4 else R_SPHERE
    tag = sys.argv[5] if len(sys.argv) > 5 else ''
    T, nbar, V = sphere_tracer(R=R)
    t0 = time.time()
    box = run('box', n_real, nbar, R, L=L, N=N)
    print(f'box: {n_real} realisations in {time.time() - t0:.0f}s', flush=True)
    t0 = time.time()
    sph = run('sphere', n_real, nbar, R, L=L, N=N)
    print(f'sphere: {n_real} realisations in {time.time() - t0:.0f}s', flush=True)
    t0 = time.time()
    quad = run('box', 100, nbar, R, eps=0.3, seed0=10 ** 6, L=L, N=N)
    print(f'quadratic box: 100 realisations in {time.time() - t0:.0f}s', flush=True)
    np.savez(HERE / f'grf_sphere_measurements{tag}.npz', box=box, sphere=sph, quad=quad, labels=np.array(labels()),
             k_edges=K_EDGES, nbar=nbar, R=R, L=L, N=N, nmodes=Mesh(L=L, N=N).nmodes)
