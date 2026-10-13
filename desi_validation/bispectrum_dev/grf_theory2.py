"""Class 1 + class 2 theory on the sphere for the GRF test (lam_w = 2), with the pair sums recomputed for the class-2
invariants; saves grf_sphere_theory_c12.npz (C_class1, C_class2, labels) and the windows."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from thecov import PowerSpectrumModel, BispectrumGaussianCovariance
from sphere_check import sphere_tracer
from grf_sphere import P_model, K_EDGES, ELLS, labels

if __name__ == '__main__':
    lam_w = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    R = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0
    tag = sys.argv[3] if len(sys.argv) > 3 else ''
    T, nbar, V = sphere_tracer(R=R)
    kk = np.logspace(-4, 0.7, 4000)
    model = PowerSpectrumModel(); model.add(('S', 'S'), {0: (kk, P_model(kk))})
    cov = BispectrumGaussianCovariance(T, K_EDGES, multipoles=ELLS, L_max=0, lam_w=lam_w, r_max=2 * R, dr=4.0, dr_pair=20.0,
                                       n_anchor=3000, n_sub=6000, n_near=200000, shot_noise=False, class2=True)
    wfile = HERE / f'grf_sphere_windows_c12_lam{lam_w}{tag}.npz'
    if wfile.exists():
        cov.load_windows(wfile); print('windows loaded', flush=True)
    else:
        t0 = time.time(); cov.compute_windows(verbose=True); print(f'windows {time.time() - t0:.0f}s', flush=True)
        cov.save_windows(wfile)
    cov.set_model(model)
    t0 = time.time(); C1, labs = cov.covariance(terms=('class1',)); print(f'class 1 {time.time() - t0:.0f}s', flush=True)
    t0 = time.time(); C2, _ = cov.covariance(terms=('class2',)); print(f'class 2 {time.time() - t0:.0f}s', flush=True)
    assert [tuple(l) for l in labs] == labels()
    np.savez(HERE / f'grf_sphere_theory_c12_lam{lam_w}{tag}.npz', C_class1=C1, C_class2=C2, labels=np.array(labels()))
    print('saved', flush=True)
