"""Example: Gaussian covariance of P_ell for two tracers sharing a synthetic footprint.

Synthetic survey: a spherical cap (half-opening angle 35 deg) between comoving distances
1200 and 2200 Mpc/h, observer at the origin, smooth n(z) for each tracer, FKP weights.
Replace `make_randoms` by your own random catalogues ({'POSITION', 'WEIGHT'[, 'NZ']}).
"""
import time

import numpy as np

from thecov import Tracer, PowerSpectrumModel, GaussianCovariance


# ----------------------------------------------------------------------------- synthetic survey
def make_randoms(n, nbar_of_r, P0_fkp, rmin=1200.0, rmax=2200.0, half_angle_deg=35.0, seed=0):
    rng = np.random.default_rng(seed)
    # sample r with pdf ∝ r^2 nbar(r) (uniform in volume, then thin by nbar)
    r_grid = np.linspace(rmin, rmax, 2000)
    pdf = r_grid ** 2 * nbar_of_r(r_grid)
    cdf = np.cumsum(pdf) / np.sum(pdf)
    r = np.interp(rng.uniform(0, 1, n), cdf, r_grid)
    cmin = np.cos(np.radians(half_angle_deg))
    ct = rng.uniform(cmin, 1.0, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - ct ** 2)
    pos = np.c_[r * st * np.cos(ph), r * st * np.sin(ph), r * ct]
    nbar = nbar_of_r(r)
    w = 1.0 / (1.0 + nbar * P0_fkp)                    # FKP weights
    # alpha = N_gal / N_ran with N_gal = int nbar dV
    omega = 2 * np.pi * (1 - cmin)
    Ngal = omega * np.trapezoid(r_grid ** 2 * nbar_of_r(r_grid), r_grid)
    alpha = Ngal / n
    return {'POSITION': pos, 'WEIGHT': w, 'NZ': nbar}, alpha


def kaiser_multipoles(k, P_lin, b, f):
    beta = f / b
    P0 = b ** 2 * P_lin * (1 + 2 * beta / 3 + beta ** 2 / 5)
    P2 = b ** 2 * P_lin * (4 * beta / 3 + 4 * beta ** 2 / 7)
    P4 = b ** 2 * P_lin * (8 * beta ** 2 / 35)
    return {0: (k, P0), 2: (k, P2), 4: (k, P4)}


def kaiser_cross(k, P_lin, b1, b2, f):
    P0 = P_lin * (b1 * b2 + (b1 + b2) * f / 3 + f ** 2 / 5)
    P2 = P_lin * (2 * (b1 + b2) * f / 3 + 4 * f ** 2 / 7)
    P4 = P_lin * (8 * f ** 2 / 35)
    return {0: (k, P0), 2: (k, P2), 4: (k, P4)}


if __name__ == "__main__":
    # --- randoms for two tracers (LRG-like and ELG-like) on the same footprint
    nbar_A = lambda r: 3e-4 * np.exp(-((r - 1500.0) / 600.0) ** 2)
    nbar_B = lambda r: 8e-4 * np.exp(-((r - 1900.0) / 500.0) ** 2)
    randoms_A, alpha_A = make_randoms(40000, nbar_A, P0_fkp=1e4, seed=1)
    randoms_B, alpha_B = make_randoms(40000, nbar_B, P0_fkp=4e3, seed=2)
    A = Tracer('A', randoms_A, alpha=alpha_A)
    B = Tracer('B', randoms_B, alpha=alpha_B)

    # --- model multipoles on an arbitrary k grid (here: a smooth toy linear spectrum)
    k = np.geomspace(5e-4, 1.5, 600)
    P_lin = 2.5e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.4)
    f = 0.8
    model = PowerSpectrumModel()
    model.add(('A', 'A'), kaiser_multipoles(k, P_lin, b=2.0, f=f))
    model.add(('B', 'B'), kaiser_multipoles(k, P_lin, b=1.2, f=f))
    model.add(('A', 'B'), kaiser_cross(k, P_lin, 2.0, 1.2, f))

    # --- covariance setup
    k_edges = np.arange(0.01, 0.205, 0.01)      # 19 bins
    cov = GaussianCovariance([A, B], k_edges, ells=(0, 2, 4), L_max=4,
                             ds=2.0, ds_pair=10.0, shot_noise=True, n_sub=2500, seed=0)
    spectra = [('A', 'A'), ('A', 'B'), ('B', 'B')]

    t0 = time.time()
    cov.compute_windows(spectra, verbose=False)            # pair counts: geometry only
    print(f"window functions: {time.time() - t0:.1f} s")
    cov.save_windows("windows_example.npz")                  # reusable for any model

    cov.set_model(model)
    t0 = time.time()
    C, labels = cov.covariance(spectra, verbose=False)
    print(f"covariance ({C.shape[0]}x{C.shape[0]}): {time.time() - t0:.1f} s")

    # --- a few numbers
    kc = 0.5 * (k_edges[1:] + k_edges[:-1])
    nb = len(kc)
    idx = {(sp, l): i for i, (sp, l) in enumerate((tuple(lab[:2]), lab[2]) for lab in labels[::nb])}
    diag = np.sqrt(np.diag(C))
    print("\nfractional errors sigma(P_ell)/P_ell at a few k for ('A','A'):")
    for l in (0, 2, 4):
        blk = idx[(('A', 'A'), l)]
        Pl = model('A', 'A', l, kc)
        sig = diag[blk * nb:(blk + 1) * nb]
        print(f"  ell={l}: " + " ".join(f"{s / p:+.3f}" for s, p in zip(sig[::4], Pl[::4])))
    corr = C / np.outer(diag, diag)
    print("\ncorrelation between P0^AA and P0^AB at the same k:", np.round(np.diag(corr[idx[(('A','A'),0)]*nb:(idx[(('A','A'),0)]+1)*nb, idx[(('A','B'),0)]*nb:(idx[(('A','B'),0)]+1)*nb])[::4], 3))
    print("correlation between P0^AA(k_i) and P0^AA(k_{i+1}):", np.round(np.diag(corr[:nb, :nb], 1)[::4], 3))
    ev = np.linalg.eigvalsh(C)
    print(f"\nsmallest / largest eigenvalue: {ev[0]:.3e} / {ev[-1]:.3e}")
    np.save("cov_example.npy", C)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ticks = [i * nb for i in range(len(idx))]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        names = [f"{sp[0]}{sp[1]} l={l}" for (sp, l) in idx]
        ax.set_xticklabels(names, rotation=90, fontsize=7); ax.set_yticklabels(names, fontsize=7)
        fig.colorbar(im, label="correlation")
        fig.tight_layout(); fig.savefig("corr_example.png", dpi=130)
        print("saved corr_example.png")
    except ImportError:
        pass
