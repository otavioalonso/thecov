"""Sphere-only GRF measurement (no box/quadratic stages): usage n_real L N R tag [workers] [seed0]"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grf_sphere import run, labels, K_EDGES, Mesh
from sphere_check import sphere_tracer
n_real, L, N, R, tag = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
workers = int(sys.argv[6]) if len(sys.argv) > 6 else 3
seed0 = int(sys.argv[7]) if len(sys.argv) > 7 else 0
T, nbar, V = sphere_tracer(R=R)
t0 = time.time()
sph = run('sphere', n_real, nbar, R, L=L, N=N, workers=workers, seed0=seed0)
print(f'sphere: {n_real} realisations in {time.time() - t0:.0f}s', flush=True)
np.savez(HERE / f'grf_sphere_measurements{tag}.npz', sphere=sph, labels=np.array(labels()), k_edges=K_EDGES, nbar=nbar, R=R, L=L, N=N,
         nmodes=Mesh(L=L, N=N).nmodes, seed0=seed0)
