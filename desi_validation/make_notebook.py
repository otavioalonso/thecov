"""Writes desi_covariance_comparison.ipynb (python desi_validation/make_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# thecov vs. the covariance of DESI mocks

Compares the Gaussian covariance from `thecov` with the sample covariance of the DESI DR2 mock power
spectra (jaxpower, `weight-default-FKP`), for one tracer bin, per region (NGC, SGC) and combined.

**What the covariance needs from the catalogues — and what is checked, not assumed.** The spectra are
measured with the total weight $w = $ `WEIGHT` $\times$ `WEIGHT_FKP` on data and randoms, and
$\alpha = \sum_d w / \sum_r w$. `WEIGHT` is *per object* (completeness × imaging systematics ×
redshift failures; each random inherits the `WEIGHT` of a random data object). The Gaussian
covariance then needs

* the clustering window $W = m(\mathbf x)^2$, where $m(\mathbf x) = E[\sum_g w_g\,\delta_D(\mathbf x-\mathbf x_g)]$
  is the **mean weighted density — a smooth field**. Building it as `NZ × WEIGHT` at each random with
  the random's *own* weight gives $\langle w^2\rangle$ instead of $\langle w\rangle^2$, i.e. a window
  too large by $1+\mathrm{var}(w)/\langle w\rangle^2$;
* the shot-noise window, whose integral must be the realised $\sum_d w^2 + \alpha^2\sum_r w^2$
  (the `num_shotnoise` of each spectrum);
* the estimator's normalisation (`norm`), since $C \propto 1/\mathrm{norm}^2$; jaxpower/pypower
  use a mesh-based normalisation a few per cent away from $\int m^2$.

Section 2 measures every one of these on the catalogues; section 4 checks them against the numbers
stored in the spectrum files. `thecov`'s `Tracer` takes $m$ as the column `NW` (see
`desi_compare.build_tracer` for how it is built without relying on what `NX` means).

Set `SYNTHETIC = True` to run the whole notebook on a small synthetic data set in the same formats
(`python -m desi_validation.synthetic <dir> 200`), where the answer is known.
""")

code(r"""
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt

THECOV_DIR = os.environ.get('THECOV_DIR', os.path.expanduser('~/thecov'))   # the thecov repository (branch thecov2)
sys.path.insert(0, THECOV_DIR)
from desi_validation import desi_compare as dc

SYNTHETIC = bool(int(os.environ.get('THECOV_SYNTHETIC', '0')))   # True: run on the synthetic test set
TRACER_BIN = 'LRG1'                                   # one of dc.TRACER_SPECS
N_RANDOM_FILES = 1                                    # random files per region (18 exist)
N_RANDOMS_MAX = 4_000_000                             # randoms kept for thecov (all regions)
SURFACE_DENSITY = 2500.0                              # randoms per deg^2 per file (DESI)
KMAX, REBIN = 0.4, 5                                  # as in the example: every 5th bin, k <= 0.4
KMIN = 0.02                                           # as the DESI fits; the example keeps k >= 0 (see section 3)
TARGET_NEAR_PAIRS = 2e9                               # sets n_near (pair-count Monte-Carlo noise)
N_SUB_FAR = 20000
OUT = os.path.expanduser(f'~/thecov_desi/{TRACER_BIN}')

if SYNTHETIC:
    SYN = os.environ.get('THECOV_SYNTHETIC_DIR', os.path.expanduser('~/thecov_desi/synthetic'))
    paths = dc.Paths(catalog_dir=SYN + '/catalogs', spectra_dir=SYN + '/spectra')
    dc.TRACER_SPECS['TEST'] = ('LRG', (0.4, 0.6))
    TRACER_BIN, SURFACE_DENSITY, KMAX, TARGET_NEAR_PAIRS, N_SUB_FAR = 'TEST', 150.0, 0.1, 3e8, 5000
    OUT = SYN + '/results'
else:
    paths = dc.Paths()        # CHECK: kind/mock of the catalogues vs the mock set of the spectra
os.makedirs(OUT, exist_ok=True)
TRACER, ZRANGE = dc.TRACER_SPECS[TRACER_BIN]
print(TRACER, ZRANGE)
print(paths.data_fn(TRACER, 'NGC'))
print(paths.randoms_fn(TRACER, 'NGC', 0))
print(len(paths.spectra_fns(TRACER, ZRANGE, 'NGC')), 'NGC spectra, e.g.', (paths.spectra_fns(TRACER, ZRANGE, 'NGC') or [None])[0])
""")

md(r"""
## 1. Catalogues

One mock's data and randoms per region, cut to the redshift range. The region comes from the file
(no RA/DEC cut: the example's `select_region` is redundant on per-region files, and its SGC branch
`not (array) & (array)` raises for arrays).

**Check the mock versions.** In the example the geometry comes from `holi_v1/altmtl201` while the
spectra are `holi-v3-altmtl`: the covariance must be computed for the footprint, $n(z)$ and weights
of the mocks whose spectra you compare with.
""")

code(r"""
t0 = time.time()
regions = {r: dc.load_region(paths, TRACER_BIN, r, n_random_files=N_RANDOM_FILES) for r in ('NGC', 'SGC')}
for r, rc in regions.items():
    print(f"{r}: {len(rc.data['Z'])} data, {len(rc.randoms['Z'])} randoms "
          f"({rc.n_randoms_all_z} before the z cut); columns: {sorted(rc.data)}")
print(f'{time.time() - t0:.0f} s')
""")

md(r"""
## 2. Weight diagnostics

What to look for:

* **`WEIGHT_over_product_of_...`** — is `WEIGHT` the product of its components (median 1, spread 0)?
* **`P0_implied_*`** — `WEIGHT_FKP` $= 1/(1+\mathrm{NX}\,P_0)$ with the expected $P_0$?
* **`alpha_*`** — unweighted vs. weighted $\alpha$. The estimator (and `thecov`) use `alpha_total`.
  The example's `get_alpha` for a single region uses `len(data)/len(randoms)`, which is only right if
  the mean weights of data and randoms agree.
* **`*_w2_over_wmean2`** — $\langle w^2\rangle/\langle w\rangle^2$: the size of the bias of the naive
  (`NZ × WEIGHT`) window. The data and random values should agree if the randoms inherit the data
  weights.
* **`shotnoise_scale`** — realised $(\sum_d w^2 + \alpha^2\sum_r w^2)$ over the randoms' prediction
  $(1+\alpha)\alpha\sum_r w^2$. It differs from 1 if the random weights do not reproduce the
  distribution of the data weights; `thecov` rescales the shot-noise window by it.
""")

code(r"""
diag = {r: dc.weight_diagnostics(rc, TRACER_BIN) for r, rc in regions.items()}
skip = ('per_z', 'data_columns', 'random_columns')
keys = [k for k in diag['NGC'] if k not in skip]
fmt = lambda v: (f'{v:.5g}' if isinstance(v, float) else str(tuple(round(x, 4) for x in v)) if isinstance(v, tuple) else str(v))
print(f"{'':42s}" + ''.join(f'{r:>28s}' for r in diag))
for k in keys:
    print(f'{k:42s}' + ''.join(f'{fmt(diag[r].get(k)):>28s}' for r in diag))
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for r, dg in diag.items():
    t = dg['per_z']; z = [row['z'] for row in t]
    ax[0].plot(z, [row['wd'] for row in t], '-', label=f'{r} data')
    ax[0].plot(z, [row['wr'] for row in t], '--', label=f'{r} randoms')
    ax[1].plot(z, [row['w2d'] / row['wd'] ** 2 for row in t], '-', label=f'{r} data')
    ax[1].plot(z, [row['w2r'] / row['wr'] ** 2 for row in t], '--', label=f'{r} randoms')
    ax[2].plot(z, [row['ratio_counts'] for row in t], '-', label=r)
ax[0].set_ylabel(r'$\langle w\rangle$'); ax[1].set_ylabel(r'$\langle w^2\rangle/\langle w\rangle^2$')
ax[2].set_ylabel(r'$\sum_d w\,/\,\alpha\sum_r w$ per z bin'); ax[2].axhline(1, color='k', lw=0.5)
for a in ax: a.set_xlabel('z'); a.legend(fontsize=8)
plt.tight_layout()
""")

md(r"""
The right panel should be 1 at every $z$ (the weighted randoms follow the weighted data in
redshift); a trend means the randoms' redshifts or weights do not track the data.

**What NX is.** `build_tracer` builds $m(\mathbf x) = \alpha\,\rho_r(z)\,\langle w\rangle_{\rm local}$ from
the random density (uniform on the sky at `SURFACE_DENSITY` per deg² per file) and the local mean
random weight. The ratio $m/(\mathrm{NX}\cdot\mathrm{WEIGHT\_FKP})$ printed below tells what NX
represents: $\approx\langle\mathrm{WEIGHT}\rangle\sim 1/\mathrm{completeness}$ if NX is the
*observed* density, $\approx 1$ if NX already includes the completeness weighting. If `healpy` is
available the footprint area is also measured directly, which checks `SURFACE_DENSITY`.
""")

code(r"""
for r, rc in regions.items():
    rho, omega = dc.random_density(rc, SURFACE_DENSITY)
    print(f'{r}: footprint area from the random count {omega * (180 / np.pi) ** 2:.1f} deg^2')
    try:
        import healpy as hp
        nside = 256
        pix = hp.ang2pix(nside, rc.randoms['RA'], rc.randoms['DEC'], lonlat=True)
        occ = np.bincount(pix, minlength=hp.nside2npix(nside))
        full = occ > 0.5 * np.median(occ[occ > 0])          # pixels mostly inside
        print(f'    healpy (nside {nside}): ~{full.sum() * hp.nside2pixarea(nside, degrees=True):.1f} deg^2 '
              f'(pixels at least half-full; edges make this approximate)')
    except ImportError:
        pass
""")

md(r"""
## 3. Mock spectra

Read every mock, keep every `REBIN`-th bin in `[KMIN, KMAX]`. The example script keeps $k \ge 0$; the
first bins ($k \lesssim 2\pi/L$) are outside the regime of any windowed Gaussian covariance, which
assumes $k \gg 1/L$ — there it can even be non-positive-definite (the lowest hexadecapole bin is the
usual culprit), so the comparison starts at `KMIN` = 0.02 as the DESI fits do. The mean is the
model for `thecov` (window-convolved and shot-noise subtracted — an approximation at the lowest $k$;
replace by a theory model if available).

**How are the GCcomb spectra combined?** If GCcomb is the norm-weighted average of NGC and SGC, its
covariance is $\sum_r n_r^2 C_r/(\sum_r n_r)^2$; if it is one estimate on the concatenated
catalogues, `thecov` should be run on the combined geometry. For well-separated caps the two agree to
leading order; both are computed below.
""")

code(r"""
spec = {}
for r in ('NGC', 'SGC', 'GCcomb'):
    fns = paths.spectra_fns(TRACER, ZRANGE, r)
    if fns:
        spec[r] = dc.read_spectra(fns, kmin=KMIN, kmax=KMAX, rebin=REBIN)
        s = spec[r]
        print(f"{r}: {len(fns)} mocks, {len(s['k'])} k bins [{s['k_edges'][0]:.3f}, {s['k_edges'][-1]:.3f}], "
              f"norm {s['norm'].mean():.5g} +- {s['norm'].std():.2g}, shot noise {s['shotnoise'].mean():.5g}")
nb, ells = len(spec['NGC']['k']), spec['NGC']['ells']
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for r, s in spec.items():
    m, e = s['vectors'].mean(0), s['vectors'].std(0)
    for i, ell in enumerate(ells):
        ax[i].errorbar(s['k'], s['k'] * m[i * nb:(i + 1) * nb], s['k'] * e[i * nb:(i + 1) * nb], label=r, capsize=0, lw=1)
for i, ell in enumerate(ells):
    ax[i].set_xlabel('k [h/Mpc]'); ax[i].set_ylabel(f'k P_{ell}'); ax[i].legend(fontsize=8)
plt.tight_layout()
if 'GCcomb' in spec:
    nN, nS = spec['NGC']['norm'].mean(), spec['SGC']['norm'].mean()
    avg = (nN * spec['NGC']['vectors'].mean(0) + nS * spec['SGC']['vectors'].mean(0)) / (nN + nS)
    err = spec['GCcomb']['vectors'].std(0) / np.sqrt(len(spec['GCcomb']['vectors']))
    print('GCcomb mean vs norm-weighted NGC/SGC average, in units of its error: rms',
          np.sqrt(np.mean(((spec['GCcomb']['vectors'].mean(0) - avg) / err) ** 2)).round(2),
          '; norm GCcomb / (norm NGC + norm SGC) =', (spec['GCcomb']['norm'].mean() / (nN + nS)).round(4))
""")

md(r"""
## 4. Tracers, and the checks against the estimator

For each region a `thecov` tracer with the clustering windows built from the smooth $m$ (`NW`), and,
for comparison, the naive set-up (`NZ = NX`, each random's own weight). Two numbers must match the
spectrum files:

* $\int m^2$ against the estimator's `norm` — equal up to the few-per-cent smoothing of the mesh
  normalisation (the covariance is normalised by the files' `norm` in any case);
* the predicted shot noise $\int S / \mathrm{norm}$ against the files' mean `shotnoise`.
""")

code(r"""
from thecov.tracers import Window
tracers, infos = {}, {}
for r in ('NGC', 'SGC'):
    for mode in ('random-density', 'none'):
        tr, info = dc.build_tracer(f'{TRACER_BIN}_{r}' + ('' if mode != 'none' else '_naive'), [regions[r]],
                                   nw=mode, n_randoms_max=N_RANDOMS_MAX // 2, surface_density_deg2=SURFACE_DENSITY)
        tracers[(r, mode)], infos[(r, mode)] = tr, info
        IW, IS = Window('W', tr, tr).integral(), Window('S', tr).integral()
        s = spec[r]
        print(f"  {r} {mode:15s}: int m^2 / norm = {IW / s['norm'].mean():.4f};  "
              f"shot noise predicted/files = {IS / s['norm'].mean() / s['shotnoise'].mean():.4f};  "
              f"median m/(NX WEIGHT_FKP) = {info['regions'][r]['median_m_over_NXwFKP']}")
# the combined NGC+SGC geometry (one estimate on the concatenated catalogues)
tracers[('GCcomb', 'random-density')], infos[('GCcomb', 'random-density')] = dc.build_tracer(
    f'{TRACER_BIN}_GCcomb', [regions['NGC'], regions['SGC']], nw='random-density',
    n_randoms_max=N_RANDOMS_MAX, surface_density_deg2=SURFACE_DENSITY)
""")

md(r"""
## 5. thecov covariances

Pair counts are the expensive part (cached in `OUT/windows_*.npz`). `n_near` is set from the survey
volume so that ~`TARGET_NEAR_PAIRS` near pairs are counted; the far pairs use all pairs of
`N_SUB_FAR` randoms.
""")

code(r"""
covs, objs = {}, {}
for key in [('NGC', 'random-density'), ('SGC', 'random-density'), ('NGC', 'none'), ('SGC', 'none'),
            ('GCcomb', 'random-density')]:
    r, mode = key
    if r not in spec and r != 'GCcomb':
        continue
    tr = tracers[key]
    n_near, V = dc.suggest_n_near(tr, target_pairs=TARGET_NEAR_PAIRS)
    s = spec[r] if r in spec else spec['NGC']
    t0 = time.time()
    C, cov = dc.thecov_covariance(tr, s, n_sub=N_SUB_FAR, n_near=n_near,
                                  windows_file=os.path.join(OUT, f'windows_{tr.name}.npz'), verbose=False)
    covs[key], objs[key] = C, cov
    print(f'{r} {mode}: V ~ {V:.3g} (Mpc/h)^3, n_near {n_near}, {time.time() - t0:.0f} s; '
          f'thecov I from randoms / estimator norm = {cov.I_randoms / s["norm"].mean():.4f}')
if ('NGC', 'random-density') in covs and 'GCcomb' in spec:
    covs[('GCcomb', 'combined-regions')] = dc.combine_regions(
        [covs[('NGC', 'random-density')], covs[('SGC', 'random-density')]],
        [spec['NGC']['norm'].mean(), spec['SGC']['norm'].mean()])
np.savez(os.path.join(OUT, 'thecov_covariances.npz'), **{f'{r}__{m}': C for (r, m), C in covs.items()})
""")

md(r"""
## 6. Comparison with the mocks

Real mocks are not Gaussian: the connected trispectrum and super-sample covariance add to the
diagonal and correlate bins, increasingly with $k$. A Gaussian covariance should match at low $k$ and
fall short at high $k$; the $\chi^2$ per $k_{\max}$ quantifies where. Below, $\chi^2$ and the
variance ratio use the mocks' own mean (the Hartlap factor is not needed: the analytic matrix is
inverted, not the sample one).
""")

code(r"""
res = {}
for (r, mode), C in covs.items():
    if r not in spec:
        continue
    s = spec[r]
    Cm = np.cov(s['vectors'], rowvar=False)
    res[(r, mode)] = dc.compare(C, Cm, s['vectors'], nb, ells, k=s['k'],
                                kmax_list=[km for km in (0.1, 0.2, 0.3, 0.4) if km <= s['k_edges'][-1] + 1e-9])
    o = res[(r, mode)]
    print(f"\n{r} [{mode}]  N_mock={o['N_mock']}, n={o['n']}")
    for k_, v in o.items():
        if k_.startswith('chi2'):
            print(f'   {k_:22s} <chi2>/n = {v[0]:.3f}  ({v[1]:+.1f} sigma)')
    print(f"   eigenvalues of C^-1/2 Cmock C^-1/2 in [{o['eig'][0]:.3f}, {o['eig'][1]:.3f}]; "
          f"exact C: Marchenko-Pastur {tuple(round(x, 3) for x in o['mp']) if o['mp'] else 'n/a (N_mock < n)'}")
""")

code(r"""
fig, ax = plt.subplots(len(spec), len(ells), figsize=(4.4 * len(ells), 3.0 * len(spec)), squeeze=False, sharex=True)
for i, r in enumerate(spec):
    N = len(spec[r]['vectors']); band = np.sqrt(2 / (N - 1))
    for j, ell in enumerate(ells):
        a = ax[i, j]
        a.axhspan(1 - band, 1 + band, color='0.9')
        a.axhline(1, color='k', lw=0.6)
        for mode, ls in (('random-density', '-o'), ('none', '--'), ('combined-regions', ':')):
            if (r, mode) in res:
                a.plot(spec[r]['k'], res[(r, mode)]['var_ratio'][j * nb:(j + 1) * nb], ls, ms=2.5, lw=1, label=mode)
        a.set_title(f'{r}, ell={ell}', fontsize=9)
        a.set_ylabel(r'$\sigma^2_{\rm mock}/\sigma^2_{\rm thecov}$')
    ax[i, 0].legend(fontsize=7)
for a in ax[-1]:
    a.set_xlabel('k [h/Mpc]')
plt.tight_layout()
""")

md(r"""
**Near-null directions.** With bins narrower than $2\pi/D$ ($D$ the survey size, radially the depth)
the window correlates neighbouring bins almost perfectly, and thecov's correlation matrix has
eigenvalues $\ll 1$ — directions that alternate in sign from bin to bin. The full-vector $\chi^2$ is
dominated by them, and any effect outside the Gaussian model shows up there first. Such effects include
the radial integral constraint and extra noise from randoms whose redshifts are taken from the data,
as in DESI. Below: the mock/thecov variance ratio along thecov's smallest-eigenvalue directions (1 is
perfect; Marchenko–Pastur scatter applies), and the whole comparison repeated with bins twice as wide.
""")

code(r"""
for (r, mode), C in covs.items():
    if r not in spec or mode == 'none':
        continue
    e, ratio, _ = dc.eigen_directions(C, np.cov(spec[r]['vectors'], rowvar=False))
    print(f'{r:7s} {mode:17s} smallest corr. eigenvalues {np.round(e, 4)}\n{"":26s}mock/thecov variance  {np.round(ratio, 2)}')

# the same comparison with bins twice as wide (windows are cached: only the kernels are recomputed)
for r in spec:
    key = (r, 'random-density')
    if key not in tracers:
        continue
    s2 = dc.read_spectra(paths.spectra_fns(TRACER, ZRANGE, r), kmin=KMIN, kmax=KMAX, rebin=2 * REBIN)
    C2, _ = dc.thecov_covariance(tracers[key], s2, n_sub=N_SUB_FAR, n_near=10 ** 6, verbose=False,
                                 windows_file=os.path.join(OUT, f'windows_{tracers[key].name}.npz'))
    o = dc.compare(C2, np.cov(s2['vectors'], rowvar=False), s2['vectors'], len(s2['k']), ells, k=s2['k'], kmax_list=[])
    res[(r, 'random-density, bins x2')] = o
    print(f"{r} bins x2: <chi2>/n = {o['chi2_all'][0]:.3f} ({o['chi2_all'][1]:+.1f} sigma), "
          f"eigenvalues [{o['eig'][0]:.2f}, {o['eig'][1]:.2f}] vs Marchenko-Pastur {np.round(o['mp'], 2) if o['mp'] else 'n/a'}")
""")

code(r"""
def corr(M):
    d = np.sqrt(np.diag(M)); return M / np.outer(d, d)
for r in spec:
    if (r, 'random-density') not in covs:
        continue
    Cm = np.cov(spec[r]['vectors'], rowvar=False); C = covs[(r, 'random-density')]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    M = np.tril(corr(Cm), -1) + np.triu(corr(C), 1) + np.eye(len(C))
    im = ax[0].imshow(M, vmin=-0.5, vmax=0.5, cmap='RdBu_r'); ax[0].set_title(f'{r}: mocks (lower) / thecov (upper)')
    plt.colorbar(im, ax=ax[0])
    for b in range(1, len(ells)):
        for a in ax:
            a.axhline(b * nb - 0.5, color='k', lw=0.4); a.axvline(b * nb - 0.5, color='k', lw=0.4)
    i0 = np.argmin(np.abs(spec[r]['k'] - 0.1))
    for j, ell in enumerate(ells):
        row = j * nb + i0
        ax[1].plot(corr(Cm)[row], '-', lw=0.8, label=f'mocks, row P{ell}(k={spec[r]["k"][i0]:.2f})')
        ax[1].plot(corr(C)[row], '--', lw=0.8)
    ax[1].set_title('rows of the correlation matrix (dashed: thecov)'); ax[1].legend(fontsize=7)
    plt.tight_layout()
""")

md(r"""
## 7. Optional: previous thecov covariances

If the covariances from the earlier version exist (the example's `load_covariance`), compare their
diagonals with this one.
""")

code(r"""
OLD = f'/dvs_ro/cfs/cdirs/desi/users/oalves/thecovs/cai/holi/cov_{TRACER_BIN}_{{region}}.txt'
for r in spec:
    fn = OLD.format(region=r)
    if os.path.exists(fn) and (r, 'random-density') in covs:
        Cold = np.loadtxt(fn)
        if Cold.shape == covs[(r, 'random-density')].shape:
            rat = np.diag(Cold) / np.diag(covs[(r, 'random-density')])
            print(r, 'old/new diagonal ratio per ell (mean):', [rat[j * nb:(j + 1) * nb].mean().round(3) for j in range(len(ells))])
        else:
            print(r, 'old covariance has shape', Cold.shape, '(different binning)')
""")

code(r"""
summary = {f'{r}__{m}': {k_: (list(v) if isinstance(v, tuple) else v) for k_, v in o.items() if k_ != 'var_ratio'}
           for (r, m), o in res.items()}
summary['diagnostics'] = {r: {k_: v for k_, v in d.items() if k_ not in ('data_columns', 'random_columns')} for r, d in diag.items()}
summary['tracers'] = {f'{r}__{m}': {k_: v for k_, v in i.items() if k_ != 'regions'} for (r, m), i in infos.items()}
json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=float)
print('wrote', OUT)
""")

nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}}
nbf.write(nb, 'desi_validation/desi_covariance_comparison.ipynb')
print('wrote desi_validation/desi_covariance_comparison.ipynb with', len(cells), 'cells')
