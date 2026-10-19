"""B_000(i, j) restricted to anchor magnitude |q1 + q2| in bins (smooth window, N given): var per bin vs exact2 Cq1 + Cq2."""
import sys, time
sys.path.insert(0, 'desi_validation/bispectrum_dev')
import numpy as np, scipy.fft as sfft
from multiprocessing import Pool
from grf_sphere import Mesh, P_model, K_EDGES
L = 1300.0; N = int(sys.argv[1]); n_real = int(sys.argv[2]); seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
m = Mesh(L=L, N=N)
mask = np.where(m.r < 600, np.exp(-m.r ** 2 / (2 * 200.0 ** 2)), 0.0)
I3 = np.sum(mask ** 3) * m.dV
QB = np.arange(0, 0.4001, 0.04)
qbins = [((m.k >= lo) & (m.k < hi)).astype(float) for lo, hi in zip(QB[:-1], QB[1:])]
nb = len(K_EDGES) - 1
el = [(i, j) for i in range(nb) for j in range(i, nb)]


def one(seed):
    d = m.grf(seed); d = d - np.sum(mask * d) / np.sum(mask)
    F = mask * d
    Fk = sfft.fftn(F, workers=1) * m.dV
    Fi = [sfft.ifftn(Fk * s, workers=1).real * N ** 3 / s.sum() for s in m.shells]
    Fb = [sfft.ifftn(Fk * q, workers=1).real / m.dV for q in qbins]
    out = np.zeros((len(el), len(qbins)))
    for e, (i, j) in enumerate(el):
        FF = Fi[i] * Fi[j]
        for b in range(len(qbins)):
            out[e, b] = m.dV * np.sum(Fb[b] * FF) / I3
    return out


if __name__ == '__main__':
    t0 = time.time()
    with Pool(3) as pool:
        rows = np.array(pool.map(one, range(seed0, seed0 + n_real), chunksize=10))
    print(f'{n_real} realisations {time.time() - t0:.0f}s', flush=True)
    np.savez(f'desi_validation/bispectrum_dev/grf_qbins_N{N}_seed{seed0}.npz', B=rows, el=np.array(el), Q_bins=QB)
