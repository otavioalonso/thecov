"""Diagnostic for tests/test_multitracer.py::test_box_multitracer_ratios.

Prints, for every ordered spectrum pair and every (l1, l2), the ratio of the computed diagonal to
the periodic-box reference times the Monte-Carlo retained fraction. The interesting quantities are:

* the overall level (should be ~1: the absolute check),
* the spread WITHIN each (l1, l2) group (should be ~0: the leakage cannot depend on which tracers
  are involved -- this is the multi-tracer statement),
* the differences BETWEEN (l1, l2) groups (genuinely non-zero: the leakage depends on the
  mu-structure of the integrand, which the isotropic Monte-Carlo does not capture).

Use it to set the tolerances in the test for your own n_sub / box size.

    python -m diagnostics.box_multitracer_ratios [n_sub]
"""
import collections
import sys
import time

import numpy as np

from thecov import Tracer, PowerSpectrumModel, GaussianCovariance
from tests.test_box import leakage_fractions
from tests.test_multitracer import kaiser, kaiser_cross, box_multitracer_analytic

n_sub = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
L_box, dist = 2000.0, 2e5
rng = np.random.default_rng(7)
nb = {'A': 3e-5, 'B': 6e-5}
alpha = {'A': 0.5, 'B': 0.25}
tracers = []
for name in ('A', 'B'):
    N = int(round(nb[name] * L_box ** 3 / alpha[name]))
    pos = rng.uniform(-L_box / 2, L_box / 2, size=(N, 3))
    pos[:, 0] += dist
    tracers.append(Tracer(name, {'POSITION': pos, 'WEIGHT': np.ones(N), 'NZ': np.full(N, nb[name])},
                          alpha=alpha[name]))

k = np.linspace(0.0, 1.0, 20)
bA, bB, f = 2.0, 1.2, 0.8
spectra_L = {('A', 'A'): kaiser(1.5e4, bA, f), ('A', 'B'): kaiser_cross(1.5e4, bA, bB, f),
             ('B', 'B'): kaiser(1.5e4, bB, f)}
model = PowerSpectrumModel()
for pair, mult in spectra_L.items():
    model.add(pair, {L: (k, np.full_like(k, v)) for L, v in mult.items()})

k_edges = np.arange(0.0, 0.205, 0.01)
t0 = time.time()
cov = GaussianCovariance(tracers, k_edges, ells=(0, 2), L_max=4, ds=2.0, ds_pair=10.0, shot_noise=True,
                         n_sub=n_sub, n_near=100000, s_split=80.0, seed=1).set_model(model)
ref = {pair: dict(m) for pair, m in spectra_L.items()}
for name in ('A', 'B'):
    ref[(name, name)][0] += (1 + alpha[name]) / nb[name]
retained, _ = leakage_fractions(L_box, k_edges)
sel = slice(3, -1)

spectra = [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]
res = {}
for sp1 in spectra:
    for sp2 in spectra:
        for l1 in (0, 2):
            for l2 in (0, 2):
                C = cov.block(sp1, sp2, l1, l2)
                A = box_multitracer_analytic(k_edges, L_box ** 3, ref, sp1, sp2, l1, l2)
                res[(sp1, sp2, l1, l2)] = np.diag(C) / (A * retained)
print(f"computed in {time.time() - t0:.1f} s with n_sub = {n_sub}\n")

groups = collections.defaultdict(list)
for (sp1, sp2, l1, l2), r in res.items():
    groups[(l1, l2)].append(((sp1, sp2), np.mean(r[sel])))
print(f"{'(l1,l2)':>8}  {'mean':>7}  {'min':>7}  {'max':>7}  {'spread':>7}")
for key in sorted(groups):
    v = np.array([x for _, x in groups[key]])
    print(f"{str(key):>8}  {v.mean():7.4f}  {v.min():7.4f}  {v.max():7.4f}  {np.ptp(v):7.4f}")

print("\nper-block means (rows: first spectrum, cols: second), l1 = l2 = 0:")
print("          " + "".join(f"{str(s):>12}" for s in spectra))
for sp1 in spectra:
    row = "".join(f"{np.mean(res[(sp1, sp2, 0, 0)][sel]):12.4f}" for sp2 in spectra)
    print(f"{str(sp1):>10}" + row)

worst = max(res.items(), key=lambda kv: abs(np.mean(kv[1][sel]) - 1))
print(f"\nfarthest block from 1: {worst[0]} -> {np.mean(worst[1][sel]):.4f}")
