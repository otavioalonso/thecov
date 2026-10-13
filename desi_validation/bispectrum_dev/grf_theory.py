"""Class-1 theory for the GRF test: BispectrumGaussianCovariance on the sphere (lam_w = 0, 2, 4) and the analytic
class-1 box covariance (first term of Sugiyama et al. 2020 eq. 49, both pairings, discrete mode counts)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from thecov import PowerSpectrumModel, BispectrumGaussianCovariance
from thecov.bispectrum_algebra import H, N
from thecov.wigner import FOUR_PI
from sphere_check import sphere_tracer, R_SPHERE
from angular_check import direct_box_average, unit
from grf_sphere import P_model, K_EDGES, ELLS, L_BOX, Mesh, labels


def box_class1(ell, ellp, PL, k_edges, V, Vi, nq=6, nth=16, nph=32):
    """C^{(1)}_box[i, j, i', j'] = pref/(V V_i V_j) [delta_ii' delta_jj' <<conj S S' PPP>>_ij + (i<->j in the primed)]."""
    nb = len(k_edges) - 1
    x, w = np.polynomial.legendre.leggauss(nq)
    lo, hi = k_edges[:-1], k_edges[1:]
    kq = 0.5 * (hi - lo)[:, None] * x + 0.5 * (hi + lo)[:, None]
    wq = 0.5 * (hi - lo)[:, None] * w * kq ** 2
    wq /= wq.sum(axis=1)[:, None]
    nhat = unit(1, np.random.default_rng(0))[0]
    out = np.zeros((nb, nb, nb, nb))
    sw = (ellp[1], ellp[0], ellp[2])
    for e, swap in ((ellp, False), (sw, True)):
        pref = FOUR_PI ** 3 * H(*ell) * H(*e) * np.sqrt(N(*ell) * N(*e))
        for i in range(nb):
            for j in range(nb):
                ang = sum(wq[i, a] * wq[j, b] * direct_box_average(ell, e, PL, kq[i, a], kq[j, b], nhat, nth, nph).real
                          for a in range(nq) for b in range(nq))
                val = pref / (V * Vi[i] * Vi[j]) * ang
                if swap:
                    out[i, j, j, i] += val
                else:
                    out[i, j, i, j] += val
    return out


if __name__ == '__main__':
    T, nbar, V = sphere_tracer()
    kk = np.logspace(-4, 0.7, 4000)
    model = PowerSpectrumModel(); model.add(('S', 'S'), {0: (kk, P_model(kk))})
    cov = BispectrumGaussianCovariance(T, K_EDGES, multipoles=ELLS, L_max=0, lam_w=4, r_max=1000.0, dr=4.0, dr_pair=20.0,
                                       n_anchor=3000, n_sub=6000, n_near=200000, shot_noise=False)
    wfile = HERE / 'grf_sphere_windows.npz'
    if wfile.exists():
        cov.load_windows(wfile)
        print('windows loaded from', wfile, flush=True)
    else:
        t0 = time.time(); cov.compute_windows(verbose=True); print(f'windows {time.time() - t0:.0f}s', flush=True)
        cov.save_windows(wfile)
    cov.set_model(model)
    res = {}
    for lam_w in (0, 2, 4):
        cov.assembler.lam_w = lam_w; cov._blocks = {}
        t0 = time.time(); C, labs = cov.covariance(symmetrize=True); print(f'lam_w {lam_w}: assembly {time.time() - t0:.0f}s', flush=True)
        res[f'C_sphere_lam{lam_w}'] = C
    assert [tuple(l) for l in labs] == labels()
    np.savez(HERE / 'grf_sphere_theory.npz', labels=np.array(labels()), **res)
    # analytic class-1 box with the lattice mode counts
    m = Mesh(); Vi = m.nmodes / L_BOX ** 3; Vbox = L_BOX ** 3
    PL = {0: P_model}
    nb = len(K_EDGES) - 1
    el = {e: [(i, j) for i in range(nb) for j in range(i, nb)] if e[0] == e[1] else [(i, j) for i in range(nb) for j in range(nb)] for e in ELLS}
    blocks = {}
    t0 = time.time()
    for ell in ELLS:
        for ellp in ELLS:
            C4 = box_class1(ell, ellp, PL, K_EDGES, Vbox, Vi, nq=5, nth=12, nph=24)
            blocks[(ell, ellp)] = np.array([[C4[i, j, I, J] for (I, J) in el[ellp]] for (i, j) in el[ell]])
    print(f'box class 1: {time.time() - t0:.0f}s', flush=True)
    Cbox = np.block([[blocks[(a, b)] for b in ELLS] for a in ELLS])
    # the box estimator divides by I3 = nbar^3 Vbox and has W = nbar^2: the formula above is for nbar = 1 -> scale
    # C ~ (W^3 / I3^2) = nbar^6 / nbar^6 = 1: no change
    np.savez(HERE / 'grf_sphere_theory.npz', C_box_class1=Cbox, labels=np.array(labels()), **res)
    print('saved')
