"""Periodic-box class 1 + class 2 theory for a GRF measurement file (continuous shells, lattice mode-count correction).

usage: grf_box_theory.py measurements.npz out.npz      (BISPEC_K_EDGES must match the measurement file)
"""
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from thecov import PowerSpectrumModel  # noqa: E402
from thecov.bispectrum import BispectrumGaussianCovariance  # noqa: E402
from thecov.bispectrum_algebra import box_window_value  # noqa: E402
from grf_sphere import P_model, K_EDGES, ELLS, Mesh  # noqa: E402


class FakeTracer:
    name = 'S'

    def s_max(self):
        return 3000.0


class BoxLib:
    def __init__(self, r, Vbox, nbar):
        self.nr, self.Vbox, self.nbar = len(r), Vbox, nbar

    def request(self, *a):
        pass

    def missing(self):
        return []

    def compute_all(self, verbose=False):
        pass

    def get(self, name, kinds, n, r):
        h = box_window_value(n, self.Vbox)
        return None if h == 0 else np.full((self.nr, self.nr), h * self.nbar ** 6)


def main(mfile, out):
    M = np.load(mfile)
    labs = [tuple(int(v) for v in l) for l in M['labels']]
    box, nbar, L, N = M['box'], float(M['nbar']), float(M['L']), int(M['N'])
    assert np.allclose(M['k_edges'], K_EDGES), 'set BISPEC_K_EDGES to the measurement shells'
    Vbox = L ** 3
    kk = np.logspace(-4, 0.7, 4000)
    model = PowerSpectrumModel()
    model.add(('S', 'S'), {0: (kk, P_model(kk))})
    cov = BispectrumGaussianCovariance(FakeTracer(), K_EDGES, multipoles=ELLS, L_max=0, lam_w=0, r_max=3000.0, dr=4.0,
                                       shot_noise=False)
    cov.windows = BoxLib(cov.r, Vbox, nbar)
    cov.set_normalization(nbar ** 3 * Vbox).set_model(model)
    t0 = time.time()
    C1, labs_t = cov.covariance(terms=('class1',))
    C2, _ = cov.covariance(terms=('class2',))
    print(f'assembly {time.time() - t0:.0f}s', flush=True)
    assert [tuple(l) for l in labs_t] == labs
    m = Mesh(L=L, N=N)
    Vd = m.nmodes / Vbox
    Vc = (K_EDGES[1:] ** 3 - K_EDGES[:-1] ** 3) / 3 / (2 * np.pi ** 2)
    corr = np.array([Vc[l[3]] / Vd[l[3]] * Vc[l[4]] / Vd[l[4]] for l in labs])
    vb, eb = box.var(0, ddof=1), np.sqrt(2 / (len(box) - 1))
    d = (C1.diagonal() + C2.diagonal()) * corr
    r = vb / d
    is000 = np.array([l[0] == 0 for l in labs])
    for sel, name in ((is000, 'B000'), (~is000, 'B202')):
        print(f'box {name}: var/(C1+C2) mean {r[sel].mean():.3f} +- {eb / np.sqrt(sel.sum()):.3f}, '
              f'chi2 {np.sum(((r[sel] - 1) / eb / r[sel]) ** 2):.1f} / {sel.sum()}')
    for l, rr, c1, c2 in zip(labs, r, C1.diagonal(), C2.diagonal()):
        print(f'  {l}  {rr:.3f} +- {eb * rr:.3f}   C2/C1 {c2 / c1:.3f}')
    np.savez(out, C1=C1, C2=C2, corr=corr, labels=np.array(labs))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
