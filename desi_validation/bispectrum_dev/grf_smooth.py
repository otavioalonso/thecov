"""GRF test with a smooth Gaussian window (sigma = R / 3, cut at R) centred on the observer: measurement or theory.

usage: grf_smooth.py measure n_real L N R tag     -> grf_smooth_measurements{tag}.npz (kind 'smooth' only)
       grf_smooth.py theory lam_w R tag            -> grf_smooth_theory_c12_lam{lam_w}{tag}.npz
The i != j elements are free of the k3 ~ 0 term, which a non-uniform window switches on.
"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from grf_sphere import P_model, K_EDGES, ELLS, labels, run, Mesh  # noqa: E402
from sphere_check import smooth_tracer  # noqa: E402

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'measure':
        n_real, L, N, R = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
        tag = sys.argv[6] if len(sys.argv) > 6 else ''
        T, nbar, V = smooth_tracer(R=R)
        t0 = time.time()
        seed0 = int(sys.argv[8]) if len(sys.argv) > 8 else 0
        sm = run('smooth', n_real, nbar, R, L=L, N=N, workers=int(sys.argv[7]) if len(sys.argv) > 7 else None, seed0=seed0)
        print(f'smooth: {n_real} realisations in {time.time() - t0:.0f}s', flush=True)
        np.savez(HERE / f'grf_smooth_measurements{tag}.npz', smooth=sm, labels=np.array(labels()), k_edges=K_EDGES,
                 nbar=nbar, R=R, L=L, N=N, nmodes=Mesh(L=L, N=N).nmodes, seed0=seed0)
    else:
        from thecov import PowerSpectrumModel, BispectrumGaussianCovariance
        lam_w, R = int(sys.argv[2]), float(sys.argv[3])
        tag = sys.argv[4] if len(sys.argv) > 4 else ''
        n_anchor = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
        seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        n_ran = int(sys.argv[7]) if len(sys.argv) > 7 else 200000
        T, nbar, V = smooth_tracer(n_ran=n_ran, R=R)
        kk = np.logspace(-4, 0.7, 4000)
        model = PowerSpectrumModel(); model.add(('S', 'S'), {0: (kk, P_model(kk))})
        cov = BispectrumGaussianCovariance(T, K_EDGES, multipoles=ELLS, L_max=0, lam_w=lam_w, r_max=2 * R, dr=4.0,
                                           dr_pair=20.0, n_anchor=n_anchor, n_sub=2 * n_anchor, n_near=n_ran, shot_noise=False,
                                           class2=True, seed=seed)
        wfile = HERE / f'grf_smooth_windows_c12_lam{lam_w}{tag}.npz'
        if wfile.exists():
            cov.load_windows(wfile)
        else:
            t0 = time.time(); cov.compute_windows(verbose=True); print(f'windows {time.time() - t0:.0f}s', flush=True)
            cov.save_windows(wfile)
        cov.set_model(model)
        print(f'I3 randoms / analytic {cov.I3() / (nbar ** 3 * (2 * np.pi * (R / 3.0) ** 2 / 3) ** 1.5):.4f}', flush=True)
        cov.set_normalization(nbar ** 3 * (2 * np.pi * (R / 3.0) ** 2 / 3) ** 1.5)   # what the mesh estimator divides by (1e-5)
        t0 = time.time(); C1, labs = cov.covariance(terms=('class1',)); print(f'class 1 {time.time() - t0:.0f}s', flush=True)
        t0 = time.time(); C2, _ = cov.covariance(terms=('class2',)); print(f'class 2 {time.time() - t0:.0f}s', flush=True)
        assert [tuple(l) for l in labs] == labels()
        np.savez(HERE / f'grf_smooth_theory_c12_lam{lam_w}{tag}.npz', C_class1=C1, C_class2=C2, labels=np.array(labels()))
        print('saved', flush=True)
