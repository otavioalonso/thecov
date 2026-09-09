r"""End-to-end validation: multi-tracer Gaussian mocks in a realistic window vs the pkcov prediction.

    python -m mocks.run_validation --n-mocks 400 --grid 256 --nproc 8 --out results/

What this tests that nothing else does
--------------------------------------
Every other test in the suite validates the *implementation* against exact identities or independent
references. This one probes the two physical APPROXIMATIONS: the local plane-parallel treatment of
each correlated pair, and the assumption that the window varies slowly over a correlation length.
The mocks make no plane-parallel assumption -- each galaxy is displaced along its own line of sight
-- and the footprint is a wide-angle cap with a hole-punched angular mask and a smooth n(z).

Statistics
----------
Comparing the covariance element by element needs ~2/eps^2 mocks for a fractional accuracy eps on
the diagonal (800 for 5 %). The whole matrix, however, can be tested far more cheaply with

    chi^2_i = (d_i - dbar)^T C_analytic^-1 (d_i - dbar),   <chi^2> = n_dim (1 - 1/N_mock),

whose mean has a relative error of sqrt(2 / (n_dim N_mock)) -- 1.2 % for n_dim = 48 and 300 mocks.
This is sensitive to the off-diagonal structure as well, because a wrong correlation pattern shows
up as a shifted mean and a distorted chi^2 distribution. The eigenvalues of
C^-1/2 Chat C^-1/2 give the same information resolved by direction.

Systematics OF THE TEST (not of pkcov)
--------------------------------------
* Box size. The covariance couples modes separated by |q| ~ 1/R_survey, and the mocks sample that
  structure at the box spacing 2 pi / L: a small box biases the *mock* covariance. Run at two values
  of --box-factor to check convergence.
* Aliasing. Keep k_max <~ 0.4 k_Nyquist (reported at start-up).
* Non-Gaussianity. The field amplitude is deliberately low so that 1 + delta stays positive and the
  Gaussian covariance is the right answer; --sigma8-like rescales it if you want to probe that.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial

import numpy as np

from pkcov import Tracer, PowerSpectrumModel, GaussianCovariance

from .estimator import MultipoleFields, ShellBinner, cross_multipole, shot_noise
from .field import GaussianField
from .survey import Catalogues, Footprint, make_grid, make_mock, model_multipoles

BIAS = {'A': 1.9, 'B': 1.2}
STOCH = {'A': 300.0, 'B': 800.0}        # white "clustering" stochasticity, part of P^XX
GROWTH = 0.78


def pk_lin(k, amplitude=1.0):
    """A smooth, broadly CDM-like linear spectrum with a mild BAO-like wiggle."""
    k = np.asarray(k, dtype=float)
    k0 = 0.05
    shape = (k / k0) / (1.0 + (k / k0) ** 2) ** 2
    wig = 1.0 + 0.05 * np.sin(k * 100.0) * np.exp(-(k / 0.25) ** 2)
    return amplitude * 2.2e4 * shape * wig


# --------------------------------------------------------------------------- one realisation
def one_mock(seed, cat, binner, spectra, ells, I_tab, amplitude):
    rng = np.random.default_rng(seed)
    field = GaussianField(cat.grid, lambda k: pk_lin(k, amplitude), rng)
    cats = make_mock(cat, field, BIAS, STOCH, GROWTH, rng, rsd=True)
    fields = {}
    for t in cat.tracers:
        pos = cats[t]
        fields[t] = MultipoleFields(cat.grid, pos, cat.weights_at(t, pos),
                                    cat.randoms[t], cat.w_ran[t], cat.alpha[t], ells=ells)
    out = []
    for (X, Y) in spectra:
        for ell in ells:
            P = cross_multipole(fields[X], fields[Y], ell, binner, I_tab[(X, Y)])
            if X == Y and ell == 0:
                P = P - shot_noise(cat.alpha[X], cat.w_ran[X], I_tab[(X, Y)])
            out.append(P)
    return np.concatenate(out), {t: len(cats[t]) for t in cat.tracers}


def _worker(seed, cat, binner, spectra, ells, I_tab, amplitude):
    v, n = one_mock(seed, cat, binner, spectra, ells, I_tab, amplitude)
    return v, n


# --------------------------------------------------------------------------- analytic side
def analytic_covariance(cat, k_edges, spectra, ells, amplitude, n_sub, n_near, ds, ds_pair,
                        verbose=True):
    tracers = []
    for t in cat.tracers:
        rnd, alpha = cat.pkcov_randoms(t)
        tracers.append(Tracer(t, rnd, alpha))
    k = np.linspace(0.0, 1.2 * k_edges[-1], 400)
    mult = model_multipoles(k, lambda kk: pk_lin(kk, amplitude), BIAS, STOCH, GROWTH, cat.tracers)
    model = PowerSpectrumModel()
    for pair, m in mult.items():
        model.add(pair, {L: (k, v) for L, v in m.items()})
    cov = GaussianCovariance(tracers, k_edges, ells=ells, L_max=4, ds=ds, ds_pair=ds_pair,
                             shot_noise=True, n_sub=n_sub, n_near=n_near, s_split=80.0, seed=0)
    t0 = time.time()
    cov.compute_windows(spectra, verbose=verbose).set_model(model)
    if verbose:
        print(f"[pkcov] window functions: {time.time() - t0:.1f} s")
    t0 = time.time()
    C, labels = cov.covariance(spectra, ells=ells)
    if verbose:
        print(f"[pkcov] covariance {C.shape[0]}x{C.shape[0]}: {time.time() - t0:.1f} s")
    return C, labels, cov


# --------------------------------------------------------------------------- report
def report(vectors, C, labels, k_eff, spectra, ells, out_dir):
    n_mock, n_dim = vectors.shape
    mean = vectors.mean(axis=0)
    Chat = np.cov(vectors, rowvar=False)
    d = np.diag(C)
    dhat = np.diag(Chat)
    print("\n" + "=" * 78)
    print(f"{n_mock} mocks, data vector of {n_dim} elements")
    print("=" * 78)

    # 1. chi^2 with the analytic matrix -- the whole-matrix test
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, (vectors - mean).T)
    chi2 = np.sum(z ** 2, axis=0)
    expect = n_dim * (1.0 - 1.0 / n_mock)
    err = expect * np.sqrt(2.0 / (n_dim * n_mock))
    print(f"\n<chi^2> = {chi2.mean():.2f}   expected {expect:.2f} +- {err:.2f}"
          f"   ({(chi2.mean() - expect) / err:+.1f} sigma)")
    print(f"  var(chi^2) = {chi2.var():.1f}   expected ~ {2 * expect:.1f}")

    # 2. eigenvalues of C^-1/2 Chat C^-1/2
    M = np.linalg.solve(L, np.linalg.solve(L, Chat).T).T
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    print(f"\neigenvalues of C^-1/2 Chat C^-1/2: mean {ev.mean():.3f}, "
          f"range [{ev.min():.3f}, {ev.max():.3f}] "
          f"(sampling width ~ {np.sqrt(2.0 / n_mock):.3f} per mode)")

    # 3. diagonal, per block
    print("\ndiagonal ratio  sigma_mock / sigma_analytic  (1 +- "
          f"{1 / np.sqrt(2 * (n_mock - 1)):.3f} statistical):")
    nb = len(k_eff)
    i = 0
    for (X, Y) in spectra:
        for ell in ells:
            r = np.sqrt(dhat[i:i + nb] / d[i:i + nb])
            print(f"  {X}{Y} l={ell}: " + " ".join(f"{v:5.3f}" for v in r))
            i += nb

    # 4. first off-diagonal correlation
    print("\nfirst off-diagonal correlation (mock vs analytic), auto blocks:")
    i = 0
    for (X, Y) in spectra:
        for ell in ells:
            s = slice(i, i + nb)
            cm = np.diag(Chat[s, s], 1) / np.sqrt(dhat[s][:-1] * dhat[s][1:])
            ca = np.diag(C[s, s], 1) / np.sqrt(d[s][:-1] * d[s][1:])
            print(f"  {X}{Y} l={ell}: mock " + " ".join(f"{v:+5.2f}" for v in cm))
            print(f"  {' ' * len(f'{X}{Y} l={ell}')}  pkcov " + " ".join(f"{v:+5.2f}" for v in ca))
            i += nb

    np.savez(os.path.join(out_dir, "comparison.npz"), vectors=vectors, C_analytic=C,
             C_mock=Chat, chi2=chi2, k_eff=k_eff, labels=np.array(labels, dtype=object))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(np.sqrt(dhat / d), 'o', ms=3)
        ax[0].axhline(1, c='k', lw=0.8)
        ax[0].fill_between([0, n_dim], 1 - 1 / np.sqrt(2 * (n_mock - 1)), 1 + 1 / np.sqrt(2 * (n_mock - 1)),
                           color='0.85', zorder=0)
        ax[0].set_xlabel('data vector element'); ax[0].set_ylabel(r'$\sigma_{mock}/\sigma_{pkcov}$')
        corr_m = Chat / np.sqrt(np.outer(dhat, dhat))
        corr_a = C / np.sqrt(np.outer(d, d))
        im = ax[1].imshow(np.tril(corr_m) + np.triu(corr_a, 1), vmin=-1, vmax=1, cmap='RdBu_r')
        ax[1].set_title('mock (lower) vs pkcov (upper)')
        fig.colorbar(im, ax=ax[1])
        ax[2].hist(chi2, bins=30, density=True, alpha=0.6)
        xs = np.linspace(chi2.min(), chi2.max(), 200)
        from scipy.stats import chi2 as chi2dist
        ax[2].plot(xs, chi2dist.pdf(xs, df=expect), 'k-')
        ax[2].set_xlabel(r'$\chi^2$ with the pkcov matrix')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "validation.png"), dpi=130)
        print(f"\nwrote {out_dir}/validation.png and comparison.npz")
    except ImportError:
        print(f"\nwrote {out_dir}/comparison.npz (matplotlib unavailable)")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-mocks', type=int, default=300)
    ap.add_argument('--grid', type=int, default=256)
    ap.add_argument('--box-factor', type=float, default=2.5)
    ap.add_argument('--kmin', type=float, default=0.01)
    ap.add_argument('--kmax', type=float, default=0.15)
    ap.add_argument('--dk', type=float, default=0.01)
    ap.add_argument('--ells', type=int, nargs='+', default=[0, 2])
    ap.add_argument('--tracers', nargs='+', default=['A', 'B'])
    ap.add_argument('--amplitude', type=float, default=1.0, help='rescale P_lin (Gaussianity check)')
    ap.add_argument('--n-random-factor', type=float, default=15.0)
    ap.add_argument('--nproc', type=int, default=1)
    ap.add_argument('--seed0', type=int, default=10000)
    ap.add_argument('--n-sub', type=int, default=5000)
    ap.add_argument('--n-near', type=int, default=200000)
    ap.add_argument('--ds', type=float, default=2.0)
    ap.add_argument('--ds-pair', type=float, default=10.0)
    ap.add_argument('--out', default='mock_validation')
    ap.add_argument('--resume', action='store_true', help='append to an existing vectors.npy')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'config.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=2)

    fp = Footprint()
    grid = make_grid(fp, args.grid, args.box_factor)
    cat = Catalogues(fp, grid, n_random_factor=args.n_random_factor, tracers=tuple(args.tracers))
    k_edges = np.arange(args.kmin, args.kmax + args.dk / 2, args.dk)
    binner = ShellBinner(grid, k_edges)
    ells = tuple(args.ells)
    spectra = [(X, Y) for i, X in enumerate(args.tracers) for Y in args.tracers[i:]]
    I_tab = {(X, Y): cat.I(X, Y) for (X, Y) in spectra}

    print(f"box L = {grid.L:.0f} Mpc/h, N = {grid.N}, cell = {grid.cell:.2f}, "
          f"k_Nyquist = {grid.k_nyquist:.3f}")
    print(f"k_max / k_Nyquist = {args.kmax / grid.k_nyquist:.2f} "
          f"({'ok' if args.kmax < 0.45 * grid.k_nyquist else 'TOO HIGH: aliasing'})")
    for t in args.tracers:
        print(f"  tracer {t}: <N_gal> = {cat.n_gal_expected[t]:.0f}, N_ran = {len(cat.randoms[t])}, "
              f"alpha = {cat.alpha[t]:.4f}")
    lo, hi = fp.extent()
    print(f"  survey extent {np.round(hi - lo, 0)}, box/survey = "
          f"{grid.L / np.max(hi - lo):.2f}  (raise --box-factor to check convergence)")

    C, labels, _ = analytic_covariance(cat, k_edges, spectra, ells, args.amplitude,
                                       args.n_sub, args.n_near, args.ds, args.ds_pair)

    path = os.path.join(args.out, 'vectors.npy')
    done = np.load(path) if (args.resume and os.path.exists(path)) else np.zeros((0, C.shape[0]))
    todo = [args.seed0 + i for i in range(len(done), args.n_mocks)]
    print(f"\nrunning {len(todo)} mocks ({len(done)} already stored) on {args.nproc} process(es)")

    fn = partial(_worker, cat=cat, binner=binner, spectra=spectra, ells=ells, I_tab=I_tab,
                 amplitude=args.amplitude)
    results, t0 = [], time.time()
    if args.nproc > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(args.nproc) as pool:
            for i, (v, n) in enumerate(pool.imap_unordered(fn, todo, chunksize=1)):
                results.append(v)
                if (i + 1) % 10 == 0:
                    el = time.time() - t0
                    print(f"  {i + 1}/{len(todo)}  {el:.0f} s  (eta {el / (i + 1) * (len(todo) - i - 1):.0f} s)")
                    np.save(path, np.vstack([done] + results) if len(done) else np.array(results))
    else:
        for i, s in enumerate(todo):
            v, n = fn(s)
            results.append(v)
            if (i + 1) % 5 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(todo)}  {el:.0f} s  N_gal {n}")
                np.save(path, np.vstack([done] + results) if len(done) else np.array(results))
    vectors = np.vstack([done] + results) if len(done) else np.array(results)
    np.save(path, vectors)
    print(f"mocks done in {time.time() - t0:.0f} s")

    report(vectors, C, labels, binner.k_eff, spectra, ells, args.out)


if __name__ == '__main__':
    main()
