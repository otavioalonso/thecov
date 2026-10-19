"""Timing of the stages of the standard Gaussian run (same synthetic survey as standard_run.py).

    python tests/regression/benchmark.py     # prints seconds per stage; compare before / after a refactoring
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)

from standard_run import survey, p_lin  # noqa: E402
from thecov import Tracer, PowerSpectrumModel, GaussianCovariance, WindowSmoothing  # noqa: E402


def main(repeat=3):
    R, nbar, b1, f = 250.0, 5e-4, 2.0, 0.7
    rng = np.random.default_rng(1)
    holes = [(rng.uniform(-200, 200, 3), rng.uniform(4, 12)) for _ in range(200)]
    pos, ref = survey(R, 20_000, holes, seed=2), survey(R, 200_000, holes, seed=3)
    wfun = lambda x: 1.0 / (1.0 + nbar * 1e4) * (1 + 0.1 * np.sin(x[:, 0] / 50.0))
    alpha = nbar * 4 / 3 * np.pi * R ** 3 * 0.9 / len(pos)
    tr = Tracer('T', {'POSITION': pos, 'WEIGHT': wfun(pos), 'NZ': np.full(len(pos), nbar), 'NW': nbar * wfun(pos)}, alpha)
    kl, Pl = p_lin()
    k_edges = np.arange(0.02, 0.1601, 0.02)
    kc = 0.5 * (k_edges[1:] + k_edges[:-1])
    model = PowerSpectrumModel()
    model.add(('T', 'T'), {0: (kl, 4.7 * Pl), 2: (kl, 2.3 * Pl), 4: (kl, 0.1 * Pl)})
    out = {}
    t = time.perf_counter()
    sm = WindowSmoothing(cell=6.0, r_max=100.0, tol=5e-3)
    sm.add_density('T', ref, wfun(ref), alpha * len(pos) / len(ref))
    sm.set_power('T', 'T', kc, np.interp(kc, kl, 6 * Pl))
    cov = GaussianCovariance([tr], k_edges, ells=(0, 2, 4), L_max=4, n_sub=1500, n_near=8000, smoothing=sm)
    cov.build_smoothing([('T', 'T')], log=lambda *a: None)
    out['smoothing'] = time.perf_counter() - t
    t = time.perf_counter()
    cov.compute_windows([('T', 'T')])
    out['pair counts'] = time.perf_counter() - t
    cov.set_model(model, masked=True)
    ts = []
    for _ in range(repeat):
        cov.set_model(model, masked=True)              # clears the kernel cache: full assembly each time
        t = time.perf_counter()
        cov.covariance([('T', 'T')])
        ts.append(time.perf_counter() - t)
    out['assembly (min of %d)' % repeat] = min(ts)
    for k, v in out.items():
        print(f'{k:24s} {v:8.3f} s')


if __name__ == '__main__':
    main()
