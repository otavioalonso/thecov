r"""End-to-end validation: multi-tracer Gaussian mocks in a realistic window vs the thecov prediction.

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

Systematics OF THE TEST (not of thecov)
--------------------------------------
* Box size. The covariance couples modes separated by |q| ~ 1/R_survey, and the mocks sample that
  structure at the box spacing 2 pi / L: a small box biases the *mock* covariance. Run at two values
  of --box-factor to check convergence.
* Aliasing. The mocks are drawn on the estimator's own grid, so without interlacing the grid images
  fold back as a direction-dependent multiplicative bias (+6 % in P at 0.56 k_Nyquist with CIC)
  that shows up as a k-only sawtooth in sigma_mock / sigma_thecov. Use --interlace --scheme tsc,
  which is clean to 0.7 k_Nyquist. k_max must also stay below the field's k_cut = 0.8 k_Nyquist.
  The driver refuses to run otherwise unless --allow-aliasing is given.
* Clipping. Poisson sampling needs 1 + b delta + noise >= 0; clipping suppresses the clustering by
  P(x > -1)^2. sigma(x) is computed at start-up INCLUDING the white stochasticity field, and the
  driver refuses sigma > 0.35 (0.4 % power loss). sigma(b delta) grows as the cell shrinks: at
  7 Mpc/h cells use --amplitude 0.6. The stochasticity is off by default (see STOCH).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from thecov import Tracer, PowerSpectrumModel, GaussianCovariance

from .estimator import MultipoleEstimator, MultipoleFields, ShellBinner, cross_multipole, realised_alpha, shot_noise
from .field import GaussianField
from .survey import Catalogues, Footprint, make_grid, make_mock, model_multipoles

BIAS = {'A': 1.9, 'B': 1.2}
# White "clustering" stochasticity, part of P^XX; multiplied by --stoch-scale, which defaults to 0.
# It cannot be realised with Gaussian cells: its per-cell sigma is sqrt(STOCH / V_cell) ~ 1 at
# 7 Mpc/h cells, so 1 + x gets clipped in 16-26 % of cells and the mocks lose 25-45 % of their
# clustering power (measured). In thecov it enters only as an additive term in P^XX with the
# W^AA window -- the same code path as the clustering -- so leaving it out costs no coverage.
STOCH = {'A': 300.0, 'B': 800.0}
MAX_SIGMA_CLIP = 0.35                 # sigma(1 + x) above this: clipping removes > 0.4 % of P
GROWTH = 0.78
# Largest k_max / k_Nyquist at which the estimator's aliasing bias on P is negligible (<~0.4 %),
# keyed by (scheme, interlace). Measured with diagnostics/aliasing.py (N=256 vs an N=512
# TSC-interlaced reference, same catalogue): at 0.56 k_Nyq plain CIC is off by +6 % (AA) and -14 %
# (AB), CIC+interlacing by <=0.3 %, TSC+interlacing by <=0.2 %; TSC+interlacing stays <=0.4 %
# (0.9 % for AB) to 0.73 k_Nyq. Plain CIC is already 2-3 % off at 0.45.
MAX_KMAX_OVER_KNYQ = {('cic', False): 0.45, ('tsc', False): 0.45,
                      ('cic', True): 0.6, ('tsc', True): 0.7}
K_CUT = 0.8                           # GaussianField default: no power above K_CUT * k_Nyquist


def pk_lin(k, amplitude=1.0):
    """A smooth, broadly CDM-like linear spectrum with a mild BAO-like wiggle."""
    k = np.asarray(k, dtype=float)
    k0 = 0.05
    shape = (k / k0) / (1.0 + (k / k0) ** 2) ** 2
    wig = 1.0 + 0.05 * np.sin(k * 100.0) * np.exp(-(k / 0.25) ** 2)
    # The overall amplitude is deliberately low. Poisson sampling clips 1 + b delta at zero, and
    # with sigma(b delta) ~ 0.9 the mocks lose ~12 % of their power -- which would be misread as a
    # covariance failure. run_validation prints sigma(b delta) and the clipped fraction at start-up.
    return amplitude * 4.0e3 * shape * wig


# --------------------------------------------------------------------------- one realisation
class MockState:
    """Everything a worker needs to make and measure one realisation (built once per process)."""

    def __init__(self, cfg):
        self.cfg = cfg
        fp = Footprint()
        self.grid = make_grid(fp, cfg['grid'], cfg['box_factor'])
        self.cat = Catalogues(fp, self.grid, n_random_factor=cfg['n_random_factor'],
                              tracers=tuple(cfg['tracers']))
        self.k_edges = np.arange(cfg['kmin'], cfg['kmax'] + cfg['dk'] / 2, cfg['dk'])
        self.ells = tuple(cfg['ells'])
        self.spectra = [(X, Y) for i, X in enumerate(cfg['tracers']) for Y in cfg['tracers'][i:]]
        self.I_tab = {sp: self.cat.I(*sp) for sp in self.spectra}
        self.stoch = {t: cfg['stoch_scale'] * STOCH[t] for t in cfg['tracers']}
        self.binner = ShellBinner(self.grid, self.k_edges)
        self.est = None
        if cfg['estimator'] == 'native' and set(self.ells) <= {0, 2}:
            # set up once per worker: randoms painted once, geometry cached, single precision
            self.est = MultipoleEstimator(self.grid, self.ells, cfg['scheme'], cfg['interlace'])
            for t in self.cat.tracers:
                self.est.set_randoms(t, self.cat.randoms[t], self.cat.w_ran[t])
        self.jax = None
        if cfg['estimator'] == 'jaxpower':
            from .jaxpower_estimator import JaxpowerEstimator
            self.jax = JaxpowerEstimator(self.grid, self.k_edges, self.ells, scheme=cfg['scheme'],
                                         interlace=cfg['interlace'])

    def measure(self, seed):
        """(data vector, normalisation per spectrum, realised sum_g w^2 per tracer)."""
        cfg, cat = self.cfg, self.cat
        rng = np.random.default_rng(seed)
        field = GaussianField(self.grid, lambda k: pk_lin(k, cfg['amplitude']), rng)
        cats = make_mock(cat, field, BIAS, self.stoch, GROWTH, rng, rsd=True)
        gal, gw = {}, {}
        for t in cat.tracers:
            w = cat.weights_at(t, cats[t])
            keep = w > 0                      # moved out of the survey by RSD: not observed
            gal[t], gw[t] = cats[t][keep], w[keep]
        sumw2 = np.array([np.sum(gw[t] ** 2) for t in cat.tracers])
        if self.jax is not None:
            fixed = self.I_tab if cfg['norm'] == 'fixed' else None
            v, norms = self.jax(gal, gw, cat.randoms, cat.w_ran, self.spectra, fixed_norm=fixed)
            return v, np.array([norms[sp] for sp in self.spectra]), sumw2
        # native estimator, pypower / jaxpower conventions: realised alpha and shot noise
        alpha = {t: realised_alpha(gw[t], cat.w_ran[t]) for t in cat.tracers}
        if self.est is not None:
            fields = {t: self.est.fields(t, gal[t], gw[t], alpha[t]) for t in cat.tracers}
        else:
            fields = {t: MultipoleFields(self.grid, gal[t], gw[t], cat.randoms[t], cat.w_ran[t], alpha[t],
                                         ells=self.ells, scheme=cfg['scheme'], interlace=cfg['interlace'])
                      for t in cat.tracers}
        out = []
        for (X, Y) in self.spectra:
            for ell in self.ells:
                P = cross_multipole(fields[X], fields[Y], ell, self.binner, self.I_tab[(X, Y)])
                if X == Y and ell == 0:
                    P = P - shot_noise(alpha[X], cat.w_ran[X], self.I_tab[(X, Y)], gal_w_A=gw[X])
                out.append(P)
        return np.concatenate(out), np.array([self.I_tab[sp] for sp in self.spectra]), sumw2


_STATE = None


def _init_worker(cfg):
    global _STATE
    _STATE = MockState(cfg)


def _worker(seed):
    v, norms, sumw2 = _STATE.measure(seed)
    return seed, v, norms, sumw2


# --------------------------------------------------------------------------- analytic side
def analytic_covariance(cat, k_edges, spectra, ells, amplitude, n_sub, n_near, ds, ds_pair, stoch,
                        windows_path=None, verbose=True):
    """thecov's matrix for the mock set-up. With `windows_path`, the pair counts are loaded from it
    when it exists and saved to it otherwise (the caller checks that the geometry matches)."""
    tracers = []
    for t in cat.tracers:
        rnd, alpha = cat.thecov_randoms(t)
        tracers.append(Tracer(t, rnd, alpha))
    k = np.linspace(0.0, 1.2 * k_edges[-1], 400)
    mult = model_multipoles(k, lambda kk: pk_lin(kk, amplitude), BIAS, stoch, GROWTH, cat.tracers)
    model = PowerSpectrumModel()
    for pair, m in mult.items():
        model.add(pair, {L: (k, v) for L, v in m.items()})
    cov = GaussianCovariance(tracers, k_edges, ells=ells, L_max=4, ds=ds, ds_pair=ds_pair,
                             shot_noise=True, n_sub=n_sub, n_near=n_near, s_split=80.0, seed=0)
    t0 = time.time()
    if windows_path is not None and os.path.exists(windows_path):
        cov.load_windows(windows_path)
        if verbose:
            print(f"[thecov] window functions loaded from {windows_path}")
    else:
        cov.compute_windows(spectra, verbose=verbose)
        if verbose:
            print(f"[thecov] window functions: {time.time() - t0:.1f} s")
        if windows_path is not None:
            _atomic(windows_path, lambda f: cov.save_windows(f))
    cov.set_model(model)
    t0 = time.time()
    C, labels = cov.covariance(spectra, ells=ells)
    if verbose:
        print(f"[thecov] covariance {C.shape[0]}x{C.shape[0]}: {time.time() - t0:.1f} s")
    return C, labels, cov


# --------------------------------------------------------------------------- caching
# Everything in <out>/ is tagged with the options it depends on; a mismatch is an error, never a
# silent mix. Mocks are stored with their seeds, so resuming runs exactly the missing seeds (with
# several processes the mocks finish out of order, and counting them was not enough).
# Bumped whenever the mock survey itself changes (catalogues, nbar, weights), so that caches made
# with an older definition are refused. 2: NZ and weights from the grid cells (see Catalogues).
# 3: pypower / jaxpower conventions -- realised alpha and shot noise, galaxies with w = 0 dropped.
# 4: field generated in single precision (same statistics, different realisation per seed).
MOCK_VERSION = 4
KEYS_WINDOWS = ('mock_version', 'cell_means', 'grid', 'box_factor', 'n_random_factor', 'tracers', 'ells',
                'n_sub', 'n_near', 'ds', 'ds_pair')
KEYS_ANALYTIC = KEYS_WINDOWS + ('kmin', 'kmax', 'dk', 'amplitude', 'stoch_scale')
KEYS_MOCKS = ('mock_version', 'grid', 'box_factor', 'n_random_factor', 'tracers', 'ells', 'kmin', 'kmax', 'dk',
              'amplitude', 'stoch_scale', 'scheme', 'interlace', 'estimator', 'norm')


def _atomic(path, write):
    """Write through a temporary file and rename, so an interrupted run never leaves a torn file."""
    base, ext = os.path.splitext(path)
    tmp = base + '.tmp' + ext
    write(tmp)
    os.replace(tmp, path)


def _subset(cfg, keys):
    return {k: cfg.get(k) for k in keys}


def _check_config(stored, current, keys, what):
    bad = [f"{k}: stored {stored.get(k)!r}, now {current.get(k)!r}" for k in keys
           if stored.get(k) != current.get(k)]
    if bad:
        raise SystemExit(f"error: {what} in the output directory were made with different options:\n  "
                         + "\n  ".join(bad) + "\nUse a new --out, or the original options.")


def _load_mocks(out, cfg, n_dim, n_spec, n_tr, prior=None):
    """What is already stored in `out`: dict of vectors, seeds, norms (per spectrum, the
    normalisation each mock was divided by) and sumw2 (realised sum_g w^2 per tracer)."""
    path = os.path.join(out, 'mocks.npz')
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as f:
            stored = json.loads(str(f['config']))
            _check_config(stored, cfg, KEYS_MOCKS, "stored mocks")
            return {k: f[k] for k in ('vectors', 'seeds', 'norms', 'sumw2')}
    legacy = os.path.join(out, 'vectors.npy')
    if os.path.exists(legacy):
        raise SystemExit(f"error: {legacy} was made by an older version of the mock survey "
                         f"(mock_version < {MOCK_VERSION}); use a new --out")
    return {'vectors': np.zeros((0, n_dim)), 'seeds': np.zeros(0, dtype=np.int64),
            'norms': np.zeros((0, n_spec)), 'sumw2': np.zeros((0, n_tr))}


def _save_mocks(out, cfg, store):
    path = os.path.join(out, 'mocks.npz')
    _atomic(path, lambda f: np.savez(f, vectors=np.asarray(store['vectors']),
                                     seeds=np.asarray(store['seeds'], dtype=np.int64),
                                     norms=np.asarray(store['norms']), sumw2=np.asarray(store['sumw2']),
                                     config=np.array(json.dumps(_subset(cfg, KEYS_MOCKS)))))


def _rescale_to_norms(C, spectra, ells, nbins, I_tab, norms):
    """thecov's matrix is normalised by I_AB = int W^AB; an estimator that divides by another
    normalisation n_AB has covariance C I_AB I_CD / (n_AB n_CD). jaxpower's and pypower's default
    (data x randoms on a 10 Mpc/h mesh) is ~2 % below I_AB here, i.e. ~4 % in the variance."""
    f = np.repeat([I_tab[sp] / norms[i] for i, sp in enumerate(spectra)], len(ells) * nbins)
    return C * np.outer(f, f)


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
    # a sample covariance of N draws in n dimensions spreads the eigenvalues of C^-1 Chat over the
    # Marchenko-Pastur range even when C is exact -- not over +- sqrt(2/N)
    q = n_dim / (n_mock - 1)
    print(f"\neigenvalues of C^-1/2 Chat C^-1/2: mean {ev.mean():.3f}, "
          f"range [{ev.min():.3f}, {ev.max():.3f}]  "
          f"(exact C: Marchenko-Pastur [{(1 - np.sqrt(q)) ** 2:.3f}, {(1 + np.sqrt(q)) ** 2:.3f}])")

    # the chi^2 test restricted to each block, low and high k: localises a failure
    nb = len(k_eff)
    blocks = [f"{X}{Y}{ell}" for (X, Y) in spectra for ell in ells]
    half = nb // 2
    print(f"\nper-block <chi^2> / expected (k split at {k_eff[half]:.3f}; sigma in brackets):")
    for b, name in enumerate(blocks):
        cells = []
        for sl in (slice(b * nb, (b + 1) * nb), slice(b * nb, b * nb + half), slice(b * nb + half, (b + 1) * nb)):
            Lb = np.linalg.cholesky(C[sl, sl])
            zb = np.linalg.solve(Lb, (vectors[:, sl] - mean[sl]).T)
            dim = Lb.shape[0]
            e = dim * (1.0 - 1.0 / n_mock)
            m = np.sum(zb ** 2, axis=0).mean()
            cells.append(f"{m / e:.3f} ({(m - e) / (e * np.sqrt(2.0 / (dim * n_mock))):+.1f})")
        print(f"  {name:6s} all {cells[0]}   low-k {cells[1]}   high-k {cells[2]}")

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
            print(f"  {' ' * len(f'{X}{Y} l={ell}')}  thecov " + " ".join(f"{v:+5.2f}" for v in ca))
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
        ax[0].set_xlabel('data vector element'); ax[0].set_ylabel(r'$\sigma_{mock}/\sigma_{thecov}$')
        corr_m = Chat / np.sqrt(np.outer(dhat, dhat))
        corr_a = C / np.sqrt(np.outer(d, d))
        im = ax[1].imshow(np.tril(corr_m) + np.triu(corr_a, 1), vmin=-1, vmax=1, cmap='RdBu_r')
        ax[1].set_title('mock (lower) vs thecov (upper)')
        fig.colorbar(im, ax=ax[1])
        ax[2].hist(chi2, bins=30, density=True, alpha=0.6)
        xs = np.linspace(chi2.min(), chi2.max(), 200)
        from scipy.stats import chi2 as chi2dist
        ax[2].plot(xs, chi2dist.pdf(xs, df=expect), 'k-')
        ax[2].set_xlabel(r'$\chi^2$ with the thecov matrix')
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
    ap.add_argument('--scheme', choices=['cic', 'tsc'], default='cic', help='mass assignment')
    ap.add_argument('--interlace', action='store_true', help='interlaced estimator (see estimator.py)')
    ap.add_argument('--estimator', choices=['native', 'jaxpower'], default='native',
                    help='native: mocks/estimator.py with the pypower / jaxpower conventions; '
                         'jaxpower: measure the mocks with jaxpower itself')
    ap.add_argument('--norm', choices=['fixed', 'mesh'], default='fixed',
                    help='fixed: divide by int nbar^2 w^2 (thecov I_AB); mesh: jaxpower default '
                         '(data x randoms on a 10 Mpc/h mesh), thecov rescaled to its mean')
    ap.add_argument('--stoch-scale', type=float, default=0.0,
                    help='multiplies the white stochasticity STOCH (default 0: see STOCH)')
    ap.add_argument('--allow-clipping', action='store_true',
                    help='run even if Poisson sampling clips the field (diagnostics only)')
    ap.add_argument('--allow-aliasing', action='store_true',
                    help='run even if k_max is too close to k_Nyquist (diagnostics only)')
    ap.add_argument('--resume', action='store_true',
                    help='continue in an existing --out: run only the missing seeds, reuse the analytic side')
    ap.add_argument('--report-only', action='store_true',
                    help='no new mocks: rebuild the report from what is stored in --out')
    args = ap.parse_args()

    if args.norm == 'mesh' and args.estimator != 'jaxpower':
        raise SystemExit("error: --norm mesh needs --estimator jaxpower")
    if args.estimator == 'jaxpower':
        from .jaxpower_estimator import available
        if not available():
            raise SystemExit("error: --estimator jaxpower needs jaxpower (pip install git+https://github.com/adematti/jax-power)")
    os.makedirs(args.out, exist_ok=True)
    cfg = vars(args).copy()
    cfg['tracers'], cfg['ells'] = list(args.tracers), list(args.ells)
    cfg['mock_version'] = MOCK_VERSION
    cfg['cell_means'] = 'shared'           # thecov's default since the scatter-bound binning rewrite
    reuse = args.resume or args.report_only
    stored_any = any(os.path.exists(os.path.join(args.out, f))
                     for f in ('mocks.npz', 'vectors.npy', 'analytic.npz', 'windows.npz'))
    if stored_any and not reuse:
        raise SystemExit(f"error: {args.out} already holds results; pass --resume or --report-only, "
                         f"or choose a new --out")
    cpath = os.path.join(args.out, 'config.json')
    prior = None
    if os.path.exists(cpath):
        with open(cpath) as fh:
            prior = json.load(fh)
        # A key missing from an older config.json means the behaviour of the version that wrote it,
        # not today's default: stochasticity was always on (scale 1), CIC without interlacing.
        prior.setdefault('stoch_scale', 1.0)
        prior.setdefault('mock_version', 1)
        for k in KEYS_MOCKS + KEYS_ANALYTIC:
            if k != 'mock_version':
                prior.setdefault(k, ap.get_default(k))

    fp = Footprint()
    grid = make_grid(fp, args.grid, args.box_factor)
    stoch = {t: args.stoch_scale * STOCH[t] for t in args.tracers}
    cat = Catalogues(fp, grid, n_random_factor=args.n_random_factor, tracers=tuple(args.tracers))
    k_edges = np.arange(args.kmin, args.kmax + args.dk / 2, args.dk)
    binner = ShellBinner(grid, k_edges)
    ells = tuple(args.ells)
    spectra = [(X, Y) for i, X in enumerate(args.tracers) for Y in args.tracers[i:]]
    I_tab = {(X, Y): cat.I(X, Y) for (X, Y) in spectra}

    print(f"box L = {grid.L:.0f} Mpc/h, N = {grid.N}, cell = {grid.cell:.2f}, "
          f"k_Nyquist = {grid.k_nyquist:.3f}")
    ratio = args.kmax / grid.k_nyquist
    print(f"k_max / k_Nyquist = {ratio:.2f}")
    # Both failure modes produce a k-only sawtooth in sigma_mock / sigma_thecov that is identical
    # across blocks and is easily misread as a covariance failure, so refuse to run them.
    problems = []
    limit = MAX_KMAX_OVER_KNYQ[(args.scheme, args.interlace)]
    if ratio > limit:
        problems.append(f"k_max / k_Nyquist = {ratio:.2f} > {limit} for {args.scheme}"
                        f"{'+interlacing' if args.interlace else ''}: aliasing")
    if args.kmax > K_CUT * grid.k_nyquist:
        problems.append(f"k_max = {args.kmax:.3f} > k_cut = {K_CUT * grid.k_nyquist:.3f}: "
                        f"the mock field has no clustering power in the top bins")
    if problems:
        msg = "; ".join(problems) + ". Raise --grid, lower --box-factor or --kmax."
        if not args.allow_aliasing:
            raise SystemExit("error: " + msg + " (--allow-aliasing overrides)")
        print("WARNING: " + msg)
    for t in args.tracers:
        print(f"  tracer {t}: <N_gal> = {cat.n_gal_expected[t]:.0f}, N_ran = {len(cat.randoms[t])}, "
              f"alpha = {cat.alpha[t]:.4f}")
    lo, hi = fp.extent()
    print(f"  survey extent {np.round(hi - lo, 0)}, box/survey = "
          f"{grid.L / np.max(hi - lo):.2f}  (raise --box-factor to check convergence)")

    from .field import GaussianField as _GF
    from .survey import clipping_diagnostics
    _d = _GF(grid, lambda k: pk_lin(k, args.amplitude), np.random.default_rng(0)).delta()
    diag = clipping_diagnostics(_d, BIAS, stoch, grid)
    del _d
    bad = []
    for t, (sig, frac) in diag.items():
        ok = sig < MAX_SIGMA_CLIP
        print(f"  sigma(b_{t} delta + noise_{t}) = {sig:.3f}, clipped cells = {frac:.2e}  "
              f"({'ok' if ok else 'TOO HIGH: clipping removes power'})")
        if not ok:
            bad.append(t)
    if bad:
        msg = (f"sigma(1 + x) > {MAX_SIGMA_CLIP} for tracer(s) {', '.join(bad)}: Poisson sampling clips "
               f"the field and the mocks lose power. Lower --amplitude (sigma scales as its square "
               f"root) or --stoch-scale, or raise the cell size.")
        if not args.allow_clipping:
            raise SystemExit("error: " + msg + " (--allow-clipping overrides)")
        print("WARNING: " + msg)

    n_dim = len(spectra) * len(ells) * (len(k_edges) - 1)
    store = _load_mocks(args.out, cfg, n_dim, len(spectra), len(args.tracers), prior)
    if store['vectors'].shape[1:] != (n_dim,):
        raise SystemExit(f"error: stored mocks have {store['vectors'].shape[1]} elements, expected {n_dim}")
    if args.report_only and len(store['vectors']) < 2:
        raise SystemExit(f"error: --report-only needs stored mocks in {args.out}")
    have = set(int(x) for x in store['seeds'])
    if len(have) != len(store['seeds']):
        raise SystemExit("error: stored mocks contain duplicated seeds")

    apath = os.path.join(args.out, 'analytic.npz')
    wpath = os.path.join(args.out, 'windows.npz')
    if os.path.exists(wpath):
        with open(wpath + '.json') as fh:
            _check_config(json.load(fh), cfg, KEYS_WINDOWS, "stored window functions")
    if os.path.exists(apath):
        with np.load(apath, allow_pickle=False) as f:
            _check_config(json.loads(str(f['config'])), cfg, KEYS_ANALYTIC, "the stored analytic covariance")
            C, labels = f['C'], list(f['labels'])
        print(f"[thecov] covariance {C.shape[0]}x{C.shape[0]} loaded from {apath}")
    else:
        if not os.path.exists(wpath):
            with open(wpath + '.json', 'w') as fh:
                json.dump(_subset(cfg, KEYS_WINDOWS), fh, indent=2)
        C, labels, _ = analytic_covariance(cat, k_edges, spectra, ells, args.amplitude,
                                           args.n_sub, args.n_near, args.ds, args.ds_pair, stoch,
                                           windows_path=wpath)
        _atomic(apath, lambda f: np.savez(f, C=C, labels=np.array([str(l) for l in labels]),
                                          config=np.array(json.dumps(_subset(cfg, KEYS_ANALYTIC)))))
        print(f"[thecov] saved {apath} and {wpath}")

    if C.shape[0] != n_dim:
        raise SystemExit(f"error: the analytic covariance is {C.shape[0]}x{C.shape[0]}, expected {n_dim}")
    with open(cpath, 'w') as fh:                    # only once every stored file has been checked
        json.dump(cfg, fh, indent=2)

    def finish(store):
        v = store['vectors']
        mean_norm = store['norms'].mean(axis=0)
        print("\nnormalisation used / thecov I_AB: " + ", ".join(
            f"{X}{Y} {mean_norm[i] / I_tab[(X, Y)]:.4f} (scatter {store['norms'][:, i].std() / mean_norm[i]:.1e})"
            for i, (X, Y) in enumerate(spectra)))
        Cn = _rescale_to_norms(C, spectra, ells, len(k_edges) - 1, I_tab, mean_norm)
        for j, t in enumerate(args.tracers):
            a = cat.alpha[t]
            pred = a * np.sum(cat.w_ran[t] ** 4)            # Poisson: Var(sum_g w^2) = int nbar w^4
            print(f"realised sum_g w^2 of {t}: var / int nbar w^4 = {store['sumw2'][:, j].var() / pred:.3f} "
                  f"(Poisson: 1; fluctuations no longer enter P -- the shot noise is realised)")
        report(v, Cn, labels, binner.k_eff, spectra, ells, args.out)

    if args.report_only:
        print(f"\nreport from {len(store['vectors'])} stored mocks")
        finish(store)
        return
    todo = [s for s in range(args.seed0, args.seed0 + args.n_mocks) if s not in have]
    print(f"\nrunning {len(todo)} mocks ({len(store['vectors'])} already stored) on {args.nproc} process(es)"
          f", estimator {args.estimator}, normalisation {args.norm}")

    new = {'vectors': [], 'seeds': [], 'norms': [], 'sumw2': []}

    def checkpoint():
        if not new['seeds']:
            return
        merged = {k: np.concatenate([store[k], np.asarray(new[k]).reshape((-1,) + store[k].shape[1:])])
                  for k in store}
        _save_mocks(args.out, cfg, merged)
        return merged

    def collect(i, res):
        sd, v, norms, sumw2 = res
        for k, x in zip(('seeds', 'vectors', 'norms', 'sumw2'), (sd, v, norms, sumw2)):
            new[k].append(x)
        every = 10 if args.nproc > 1 else 5
        if (i + 1) % every == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(todo)}  {el:.0f} s  (eta {el / (i + 1) * (len(todo) - i - 1):.0f} s)")
            checkpoint()

    t0 = time.time()
    if args.nproc > 1:
        import multiprocessing as mp
        if args.estimator == 'jaxpower':
            # JAX does not survive fork: spawn fresh workers, each rebuilding the (deterministic) mock
            # survey, and keep XLA to one thread per worker so that nproc workers do not oversubscribe.
            ctx = mp.get_context('spawn')
            os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false '
                                               'intra_op_parallelism_threads=1')
        else:
            ctx = mp.get_context('fork')
        with ctx.Pool(args.nproc, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, todo, chunksize=1)):
                collect(i, res)
    else:
        _init_worker(cfg)
        for i, sd in enumerate(todo):
            collect(i, _worker(sd))
    merged = checkpoint() or store
    print(f"mocks done in {time.time() - t0:.0f} s")
    finish(merged)

if __name__ == '__main__':
    main()
