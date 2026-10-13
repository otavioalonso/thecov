"""Compare the GRF sample covariances (grf_sphere.py) with the class-1 theory (grf_theory.py); figure + summary."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grf_sphere import P_model, K_EDGES, labels  # noqa: E402


def sample_cov(X):
    return np.cov(X, rowvar=False)


def var_err(X, n_boot=400, seed=0):
    """Bootstrap error of the variances."""
    rng = np.random.default_rng(seed)
    n = len(X)
    vs = np.array([np.var(X[rng.integers(0, n, n)], axis=0, ddof=1) for _ in range(n_boot)])
    return vs.std(axis=0)


def main(make_plot=True):
    M = np.load(HERE / 'grf_sphere_measurements.npz')
    T = np.load(HERE / 'grf_sphere_theory.npz')
    labs = [tuple(l) for l in M['labels']]
    assert labs == labels() and [tuple(l) for l in T['labels']] == labs
    box, sph, quad = M['box'], M['sphere'], M['quad']
    nbar = float(M['nbar'])
    out = {}
    # ---- estimator normalisation: quadratic field, <B_000> = 2 eps [P P + 2 perms] shell-averaged
    kc = 0.5 * (K_EDGES[1:] + K_EDGES[:-1])
    eps = 0.3
    nb = len(kc)
    mean_q = quad.mean(axis=0)
    err_q = quad.std(axis=0) / np.sqrt(len(quad))
    print('estimator normalisation (quadratic field, box):')
    for n, (l1, l2, L, i, j) in enumerate(labs):
        if (l1, l2, L) != (0, 0, 0):
            continue
        # third side averaged over the opening angle at the bin centres (crude: enough for a normalisation check)
        mu = np.linspace(-1, 1, 2001)
        k3 = np.sqrt(kc[i] ** 2 + kc[j] ** 2 + 2 * kc[i] * kc[j] * mu)
        Pi, Pj, P3 = P_model(kc[i]), P_model(kc[j]), np.mean(P_model(k3))
        pred = 2 * eps * (Pi * Pj + Pi * P3 + Pj * P3) * nbar ** 3 / nbar ** 3
        print(f'  B000({i},{j}): measured {mean_q[n]:.4e} +- {err_q[n]:.1e}   tree-level {pred:.4e}   ratio {mean_q[n] / pred:.3f}')
        out.setdefault('norm_ratio', []).append(mean_q[n] / pred)
    # ---- variances
    vb, vs = box.var(axis=0, ddof=1), sph.var(axis=0, ddof=1)
    eb, es = var_err(box), var_err(sph)
    Cb1 = T['C_box_class1']
    rows = []
    print('\nvariance ratios  (GRF / class-1 theory)   box: analytic class 1;  sphere: thecov lam_w = 0, 2, 4')
    for n, lab in enumerate(labs):
        rb = vb[n] / Cb1[n, n]
        r0, r2, r4 = (vs[n] / T[f'C_sphere_lam{w}'][n, n] for w in (0, 2, 4))
        rows.append((lab, rb, eb[n] / Cb1[n, n], r0, r2, r4, es[n] / T['C_sphere_lam4'][n, n]))
        print(f'  {lab}: box {rb:.3f} +- {eb[n] / Cb1[n, n]:.3f}   sphere {r0:.3f} / {r2:.3f} / {r4:.3f} +- {es[n] / T["C_sphere_lam4"][n, n]:.3f}'
              f'   sphere/box {r4 / rb:.3f}')
    out['rows'] = rows
    # ---- correlation of neighbouring elements (sphere): GRF vs theory
    Cs = sample_cov(sph)
    Ct = T['C_sphere_lam4']
    corr_g = Cs / np.sqrt(np.outer(np.diag(Cs), np.diag(Cs)))
    corr_t = Ct / np.sqrt(np.outer(np.diag(Ct), np.diag(Ct)))
    out['corr'] = (corr_g, corr_t)
    if make_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2, 1]))
        ax = axes[0]
        x = np.arange(len(labs))
        ax.errorbar(x - 0.15, [r[1] for r in rows], [r[2] for r in rows], fmt='s', color='k', label='periodic box: GRF / analytic class 1')
        ax.errorbar(x + 0.15, [r[5] for r in rows], [r[6] for r in rows], fmt='o', color='C3', label=r'sphere: GRF / thecov class 1 ($\lambda_w = 4$)')
        ax.plot(x + 0.15, [r[3] for r in rows], 'x', color='C0', label=r'sphere, $\lambda_w = 0$')
        ax.plot(x + 0.15, [r[4] for r in rows], '+', color='C2', label=r'sphere, $\lambda_w = 2$')
        ax.axhline(1, color='gray', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{l[0]}{l[1]}{l[2]}\n({l[3]},{l[4]})' for l in labs], fontsize=7)
        ax.set_ylabel('var(GRF) / var(class-1 theory)')
        ax.set_title('Gaussian fields, isotropic P, R = 500 Mpc/h sphere; excess over 1 = class-2 pairings (~dk/k)')
        ax.legend(fontsize=8, loc='upper left')
        ax = axes[1]
        tri = np.triu_indices(len(labs), 1)
        ax.plot(corr_t[tri], corr_g[tri], '.', ms=4)
        lim = max(abs(corr_t[tri]).max(), abs(corr_g[tri]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], color='gray', lw=0.8)
        ax.set_xlabel('theory correlation (class 1, sphere)')
        ax.set_ylabel('GRF correlation')
        ax.set_title(f'off-diagonal correlations ({len(sph)} realisations)')
        fig.tight_layout()
        fig.savefig(HERE / 'bispectrum_grf_sphere.png', dpi=130)
        np.savez(HERE / 'grf_sphere_results.npz', labels=np.array(labs), ratio_box=[r[1] for r in rows],
                 ratio_box_err=[r[2] for r in rows], ratio_sphere_lam0=[r[3] for r in rows],
                 ratio_sphere_lam2=[r[4] for r in rows], ratio_sphere_lam4=[r[5] for r in rows],
                 ratio_sphere_err=[r[6] for r in rows], corr_grf=corr_g, corr_theory=corr_t,
                 norm_ratio=out.get('norm_ratio', []))
    return out


if __name__ == '__main__':
    main()
