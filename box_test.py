"""Box test after box_test.ipynb, for pkcov.

Uniform randoms in a cube of side `boxsize` (observer at `distance` along x from the centre),
flat P_0, P_2 = P_4 = 0, shot noise included, compared with the periodic-box formula
2/N_modes (...) as in the notebook, plus the expectation for a finite (non-periodic) window:
the retained fraction of each bin's variance predicted by sampling q from |W~_cube(q)|^2.

Run:  PYTHONPATH=. python box_test.py [distance]
"""
import sys
import time

import numpy as np

from pkcov import Tracer, PowerSpectrumModel, GaussianCovariance
from tests.test_box import analytic_box, leakage_fractions

boxsize = 4e3
nrandoms = 2000000
alpha = 1.0
distance = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
P0_flat = 1e4
kmin, kmax, dk = 0.0, 0.4, 0.01
ellmax = 2

rng = np.random.default_rng(0)
positions = rng.uniform(-boxsize / 2, boxsize / 2, size=(nrandoms, 3))
positions[:, 0] += distance
nbar = alpha * nrandoms / boxsize ** 3
randoms = {'POSITION': positions, 'WEIGHT': np.ones(nrandoms), 'NZ': np.full(nrandoms, nbar)}
tracer = Tracer('A', randoms, alpha=alpha)

k_edges = np.arange(kmin, kmax + dk / 2, dk)
kk = np.linspace(0.0, 1.0, 50)
model = PowerSpectrumModel()
model.add(('A', 'A'), {0: (kk, np.full_like(kk, P0_flat))})     # P2 = P4 = 0

ells = tuple(range(0, ellmax + 1, 2))
t0 = time.time()
cov = GaussianCovariance([tracer], k_edges, ells=ells, L_max=ellmax, ds=2.0, ds_pair=10.0,
                         shot_noise=True, n_sub=5000, n_near=200000, s_split=80.0)
cov.compute_windows([('A', 'A')])
print(f"window functions: {time.time() - t0:.1f} s")
cov.set_model(model)
block = {(l1, l2): cov.block(('A', 'A'), ('A', 'A'), l1, l2) for l1 in ells for l2 in ells}

# periodic-box reference (notebook), with the FKP shot noise (1 + alpha)/nbar
P0 = P0_flat + (1 + alpha) / nbar
PL = {0: np.full(len(k_edges) - 1, P0)}
analytic = {(l1, l2): analytic_box(k_edges, boxsize ** 3, PL, l1, l2) for l1 in ells for l2 in ells}
retained, neighbours = leakage_fractions(boxsize, k_edges)

kc = 0.5 * (k_edges[1:] + k_edges[:-1])
np.set_printoptions(precision=3, linewidth=150)
print("diag(C00) / periodic :", np.diag(block[(0, 0)]) / analytic[(0, 0)])
print("retained fraction (MC):", retained)
print("diag(C22) / periodic :", np.diag(block[(2, 2)]) / analytic[(2, 2)])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(kc, np.sqrt(analytic[(0, 0)]), c='gray', ls='dashed', label='periodic box, $2/N_{modes}$')
ax[0].semilogy(kc, np.sqrt(analytic[(0, 0)] * retained), c='k', ls='dotted', label='periodic x retained fraction')
ax[0].semilogy(kc, np.sqrt(np.diag(block[(0, 0)])), c='red', label='pkcov, $\\ell=0$')
ax[0].semilogy(kc, np.sqrt(np.diag(block[(2, 2)])), c='blue', label='pkcov, $\\ell=2$')
ax[0].semilogy(kc, np.sqrt(analytic[(2, 2)]), c='blue', ls='dashed', alpha=0.5)
ax[0].set_xlabel('k [h/Mpc]'); ax[0].set_ylabel(r'$\sigma(P_\ell)$'); ax[0].legend(fontsize=8)
ax[1].plot(kc, np.sqrt(np.diag(block[(0, 0)]) / analytic[(0, 0)]), c='red', label=r'$\ell=0$')
ax[1].plot(kc, np.sqrt(np.diag(block[(2, 2)]) / analytic[(2, 2)]), c='blue', label=r'$\ell=2$')
ax[1].plot(kc, np.sqrt(retained), c='k', ls='dotted', label='sqrt(retained fraction), MC')
ax[1].axhline(1, c='gray', ls='dashed')
ax[1].set_ylim(0.85, 1.02); ax[1].set_xlabel('k [h/Mpc]'); ax[1].set_ylabel(r'$\sigma / \sigma_{periodic}$'); ax[1].legend(fontsize=8)
fig.suptitle(f'box test: L={boxsize:.0f}, distance={distance:.0f}, alpha={alpha}')
fig.tight_layout()
fig.savefig(f'box_test_d{int(distance)}.png', dpi=120)
print(f"saved box_test_d{int(distance)}.png")
