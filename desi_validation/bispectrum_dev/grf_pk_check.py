"""Power-spectrum control: var(P_0) of Gaussian fields on the sphere vs thecov.GaussianCovariance (same local
approximation A1, same randoms), same mesh and shells as the bispectrum test.  usage: grf_pk_check.py n_real L N R"""
import sys, time
from pathlib import Path
import numpy as np
import scipy.fft as sfft
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from grf_sphere import Mesh, P_model, K_EDGES
from sphere_check import sphere_tracer

_M = None


def _init(L, N):
    global _M
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    _M = Mesh(L=L, N=N)


def _one(args):
    seed, kind, nbar, R = args
    m = _M
    d = m.grf(seed)
    if kind == 'box':
        F, I2 = nbar * d, nbar ** 2 * m.L ** 3
    else:
        mask = m.r < R
        F, I2 = nbar * mask * (d - d[mask].mean()), nbar ** 2 * mask.sum() * m.dV
    Fk = sfft.fftn(F, workers=2) * m.dV
    P2 = np.abs(Fk) ** 2
    return np.array([P2[s].mean() for s in m.shells]) / I2


if __name__ == '__main__':
    n_real, L, N, R = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    T, nbar, V = sphere_tracer(R=R)
    out = {}
    for kind in ('box', 'sphere'):
        t0 = time.time()
        with Pool(7, initializer=_init, initargs=(L, N)) as pool:
            out[kind] = np.array(pool.map(_one, [(s, kind, nbar, R) for s in range(n_real)], chunksize=4))
        print(f'{kind}: {time.time() - t0:.0f}s', flush=True)
    # theory: thecov Gaussian covariance of P_0 on the sphere (no shot noise), and the box value 2 P^2 / N_modes
    from thecov import PowerSpectrumModel, GaussianCovariance
    kk = np.logspace(-4, 0.7, 4000)
    model = PowerSpectrumModel(); model.add(('S', 'S'), {0: (kk, P_model(kk))})
    cov = GaussianCovariance([T], K_EDGES, ells=(0,), L_max=0, shot_noise=False, n_sub=5000, n_near=200000)
    t0 = time.time(); cov.compute_windows([('S', 'S')]); cov.set_model(model)
    C = cov.block(('S', 'S'), ('S', 'S'), 0, 0)
    print(f'thecov P(k) covariance: {time.time() - t0:.0f}s', flush=True)
    m = Mesh(L=L, N=N)
    Vd = m.nmodes / L ** 3
    kc = 0.5 * (K_EDGES[1:] + K_EDGES[:-1])
    Pbar = np.array([np.mean(P_model(m.k[s])) for s in m.shells])
    box_th = 2 * Pbar ** 2 / (L ** 3 * Vd)
    vb, vs = out['box'].var(0, ddof=1), out['sphere'].var(0, ddof=1)
    e = np.sqrt(2 / (n_real - 1))
    Vc = (K_EDGES[1:] ** 3 - K_EDGES[:-1] ** 3) / 3 / (2 * np.pi ** 2)
    print('shell   var_box/box_theory   var_sphere/thecov (x V_c/V_d)   GRF leakage var_s/var_b*Vs/Vb   thecov leakage C/box')
    for i in range(len(kc)):
        print(f'  {kc[i]:.2f}   {vb[i] / box_th[i]:.3f} +- {e:.3f}        {vs[i] / (C[i, i] * Vc[i] / Vd[i]):.3f} +- {e:.3f}'
              f'                 {vs[i] / vb[i] * V / L ** 3:.3f}                     {C[i, i] / (2 * Pbar[i] ** 2 / (L ** 3 * Vc[i])) * V / L ** 3:.3f}')
    np.savez(HERE / f'grf_pk_check_R{int(R)}.npz', box=out['box'], sphere=out['sphere'], C=C, box_th=box_th, k_edges=K_EDGES)
