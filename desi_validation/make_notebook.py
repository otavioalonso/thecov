"""Writes desi_covariance_comparison.ipynb (python desi_validation/make_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Validation of `thecov` against the DESI DR2 mocks

Gaussian covariance from `thecov` vs. the sample covariance of the DR2 mock power spectra (jaxpower,
`weight-default-FKP`), for every tracer bin and region (NGC, SGC, GCcomb), $P_{0,2,4}$ in
$0.005\,h\,{\rm Mpc}^{-1}$ bins from $k_{\min}=0.02$ to $k_{\max}=0.3$ (the fits use $k<0.2$).

**Part I** looks at one bin in detail: catalogue weights, the inputs the covariance needs, checks
against the numbers stored in the spectrum files, and the full set of tests of the validation
report. **Part II** runs the same pipeline on every bin and tabulates it.

### What the covariance needs from the catalogues (checked, not assumed)
The spectra use $w=$ `WEIGHT`$\times$`WEIGHT_FKP` on data and randoms and
$\alpha=\sum_d w/\sum_r w$. `WEIGHT` is per object (completeness, imaging, redshift failures) and each
random inherits the `WEIGHT` of a random data object. The Gaussian covariance then needs

* the clustering window $W=m(\mathbf x)^2$, with $m=E[\sum_g w_g\delta_D(\mathbf x-\mathbf x_g)]$ the
  **smooth** mean weighted density. `NZ`$\times$`WEIGHT` with each random's own weight gives
  $\langle w^2\rangle$ instead of $\langle w\rangle^2$, a window too large by $1+{\rm var}(w)/\langle w\rangle^2$.
  `desi_compare.build_tracer` builds $m=\alpha\,\rho_r(z)\,\langle w\rangle_{\rm local}$ from the random
  density and passes it to `thecov.Tracer` as `NW`;
* the shot-noise window, scaled to the realised $\sum_d w^2+\alpha^2\sum_r w^2$ (`num_shotnoise`);
* the estimator's normalisation (`norm`, mesh-based in jaxpower/pypower), by which the covariance is
  divided instead of thecov's own $\int m^2$.

### Statistics (as in the validation report)
With $N$ mocks of an $n$-element vector and an exact $C$:
$\langle\chi^2\rangle=n(1-1/N)\pm n(1-1/N)\sqrt{2/(nN)}$ with $\chi^2_i=(d_i-\bar d)^TC^{-1}(d_i-\bar d)$,
variance $\simeq 2n$ and a rescaled $\chi^2_n$ distribution (KS test); the eigenvalues of
$C^{-1/2}\hat CC^{-1/2}$ fill the Marchenko–Pastur interval $[(1-\sqrt q)^2,(1+\sqrt q)^2]$,
$q=n/(N-1)$; $\sigma_{\rm mock}/\sigma_{\rm thecov}$ scatters by $1/\sqrt{2(N-1)}$; correlation residuals
$z_{ij}=(\hat\rho_{ij}-\rho_{ij})\sqrt{N-1}/(1-\rho_{ij}^2)$ are unit Gaussians. These are evaluated on the
whole vector, per multipole and $k$ half, and as a function of $k_{\max}$.

**What to expect.** The mocks are not Gaussian: the connected trispectrum and super-sample
covariance add variance and correlations growing with $k$ (mostly in $P_0$), so a Gaussian
covariance should pass at low $k$ and fall short increasingly at high $k$. The $\chi^2(k_{\max})$
curves show where; the per-element and per-multipole numbers show how.
""")

code(r"""
import os, sys, time, json, dataclasses
import numpy as np
import matplotlib.pyplot as plt

THECOV_DIR = os.environ.get('THECOV_DIR', os.path.expanduser('~/thecov'))   # thecov repository (branch thecov2)
sys.path.insert(0, THECOV_DIR)
from desi_validation import desi_compare as dc, validation as va, pipeline as pl

SYNTHETIC = bool(int(os.environ.get('THECOV_SYNTHETIC', '0')))   # True: the synthetic test set

cfg = pl.Config(
    kmin=0.02, kmax=0.3, rebin=5,          # 0.005 h/Mpc bins (spectra are stored in 0.001 bins)
    n_random_files=1,                      # random files per region read for thecov (18 exist)
    n_randoms_max=4_000_000,               # randoms kept for the pair counts (all regions)
    surface_density=2500.0,                # randoms per deg^2 per file
    target_near_pairs=2e9,                 # sets n_near: Monte-Carlo noise of the windows at s < 80
    n_sub_far=20000,                       # all pairs of n_sub_far randoms for s > 80
    nw_modes=('random-density',),          # Part II construction of m; ('angular',) if Part I favours it
    naive=False,                           # Part II: the naive covariance is built only in Part I
    coarse_check=True)                     # also validate with 0.01 bins
DETAIL_BIN = 'LRG1'                         # Part I
BINS = list(dc.TRACER_SPECS)                # Part II: every bin
paths = dc.Paths(kind='holi_v3', mock=173)  # catalogues of one mock of the spectra's release
# spectra name -> catalogue name where they differ (edit if a bin reports MISSING below)
paths.catalog_names.update({'ELG_LOPnotqso': 'ELGnotqso', 'LRG+ELG_LOPnotqso': 'LRG+ELGnotqso'})
# Catalogue reader. 'auto': clustering_statistics.tools.read_clustering_catalog if importable -- the
# spectra pipeline's own reader (INDWEIGHT = WEIGHT x WEIGHT_FKP, slim randoms completed from the
# parent randoms of cs_parent_version) -- else the h5 files directly ('files').
paths.loader = 'auto'
paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
paths.cs_extra = {}                          # further catalog options (override propose_fiducial's)
# Only for loader='files': holi v3 randoms are slim (TARGETID, TARGETID_DATA, WEIGHT, NX); RA/DEC come from
# parent random files with TARGETID, RA, DEC ({tracer}, {region}, {i}, {kind}, {mock}).
paths.random_positions = None
OUT = os.path.expanduser(f'~/thecov_desi/{paths.kind}_mock{paths.mock}')   # caches depend on the catalogues
if SYNTHETIC:
    SYN = os.environ.get('THECOV_SYNTHETIC_DIR', os.path.expanduser('~/thecov_desi/synthetic'))
    paths = dc.Paths(catalog_dir=SYN + '/catalogs', spectra_dir=SYN + '/spectra')
    dc.TRACER_SPECS.clear(); dc.TRACER_SPECS['TEST'] = ('LRG', (0.4, 0.6))
    DETAIL_BIN, BINS, OUT = 'TEST', ['TEST'], SYN + '/results_nb'
    cfg.surface_density, cfg.kmax, cfg.target_near_pairs, cfg.n_sub_far = 150.0, 0.1, 3e8, 5000
os.makedirs(OUT, exist_ok=True)
for b in BINS:
    t, zr = dc.TRACER_SPECS[b]
    print(f'{b:8s} {t:22s} z={zr}  spectra: ' + ', '.join(f'{r} {len(paths.spectra_fns(t, zr, r))}' for r in cfg.regions))
print('catalogues, e.g.', paths.data_fn(dc.TRACER_SPECS[DETAIL_BIN][0], 'NGC'))
paths.check(BINS)
print('catalogue reader:', 'clustering_statistics' if (paths.loader == 'clustering_statistics' or
      (paths.loader == 'auto' and dc._have_clustering_statistics())) else 'files')
if paths.random_positions is None and not SYNTHETIC and not dc._have_clustering_statistics():
    print('\npaths.random_positions not set; candidate parent random files (OK = has TARGETID, RA, DEC):')
    dc.find_random_sources(paths, dc.TRACER_SPECS[DETAIL_BIN][0])
""")

md(r"""
# Part I — one bin in detail

The whole pipeline for `DETAIL_BIN` (catalogues, diagnostics, spectra, tracers, covariances,
validation), keeping everything for inspection. The naive covariance is built too, for comparison.
Pair counts and covariances are cached in `OUT/<bin>/`.

**Mock versions.** The catalogues (`holi_v3/altmtl173`) must be of the release whose spectra are
compared (`holi-v3-altmtl`): the covariance depends on their footprint, $n(z)$ and weights. Caches
are kept per catalogue release and mock in `OUT`.
""")

code(r"""
cfg_detail = dataclasses.replace(cfg, naive=True, nw_modes=('random-density', 'angular'))   # compare constructions of m
D = pl.run_bin(paths, DETAIL_BIN, cfg_detail, OUT, keep=True)
spec, regs, ells = D['spectra'], D['regions'], cfg.ells
nb = len(next(iter(spec.values()))['k'])
""")

md(r"""
## I.1 Weights

* `WEIGHT_over_product_of_...` — is `WEIGHT` the product of its components (median 1, spread 0)?
* `P0_implied_*` — `WEIGHT_FKP` $=1/(1+{\rm NX}\,P_0)$ with the expected $P_0$?
* `alpha_*` — unweighted vs weighted $\alpha$; the estimator and `thecov` use `alpha_total` (the
  example's `get_alpha` used `len(data)/len(randoms)` for single regions).
* `*_w2_over_wmean2` — $\langle w^2\rangle/\langle w\rangle^2$, the bias of the naive window; data and
  randoms should agree if the randoms inherit the data weights.
* `shotnoise_scale` — realised $\sum_d w^2+\alpha^2\sum_rw^2$ over the randoms' $(1+\alpha)\alpha\sum_rw^2$.
""")

code(r"""
diag = D['diagnostics']
print('catalogue info:', D.get('catalogue_info'))
fmt = lambda v: (f'{v:.5g}' if isinstance(v, float) else str(tuple(round(x, 4) for x in v)) if isinstance(v, (tuple, list)) else str(v))
print(f"{'':42s}" + ''.join(f'{r:>28s}' for r in diag))
for k_ in [k_ for k_ in next(iter(diag.values())) if k_ != 'per_z']:
    print(f'{k_:42s}' + ''.join(f'{fmt(diag[r].get(k_)):>28s}' for r in diag))

fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
for r, dg in diag.items():
    t = dg['per_z']; z = [row['z'] for row in t]
    ax[0].plot(z, [row['wd'] for row in t], '-', label=f'{r} data'); ax[0].plot(z, [row['wr'] for row in t], '--', label=f'{r} randoms')
    ax[1].plot(z, [row['w2d'] / row['wd'] ** 2 for row in t], '-', label=f'{r} data')
    ax[1].plot(z, [row['w2r'] / row['wr'] ** 2 for row in t], '--', label=f'{r} randoms')
    ax[2].plot(z, [row['ratio_counts'] for row in t], '-', label=r)
ax[0].set_ylabel(r'$\langle w\rangle$'); ax[1].set_ylabel(r'$\langle w^2\rangle/\langle w\rangle^2$')
ax[2].set_ylabel(r'$\sum_d w\,/\,\alpha\sum_r w$ per z bin'); ax[2].axhline(1, color='k', lw=0.5)
for a in ax: a.set_xlabel('z'); a.legend(fontsize=7)
plt.tight_layout()
""")

md(r"""
The right panel should be 1 at every $z$ (weighted randoms follow the weighted data in redshift).

**What NX is, and the footprint area.** $m/({\rm NX}\cdot$`WEIGHT_FKP`$)\approx\langle$`WEIGHT`$\rangle$
($\sim$1/completeness) if NX is the observed density, $\approx1$ if it already includes the completeness
weighting. The area from the random count checks `surface_density` (healpy, if available, measures it
independently).
""")

code(r"""
for r, rc in regs.items():
    rho, omega = dc.random_density(rc, cfg.surface_density)
    ti = D['tracer_info'][f'{r}__random-density']
    print(f"{r}: area from the random count {omega * (180 / np.pi) ** 2:.1f} deg^2; "
          f"median m/(NX WEIGHT_FKP) = {ti['regions'][r]['median_m_over_NXwFKP']}")
    try:
        import healpy as hp
        nside = 256
        occ = np.bincount(hp.ang2pix(nside, rc.randoms['RA'], rc.randoms['DEC'], lonlat=True), minlength=hp.nside2npix(nside))
        full = occ > 0.5 * np.median(occ[occ > 0])
        print(f'    healpy (nside {nside}): ~{full.sum() * hp.nside2pixarea(nside, degrees=True):.1f} deg^2 (approximate at edges)')
    except ImportError:
        pass
""")

md(r"""
## I.2 Spectra and what they store

* **Shot-noise convention.** If every file's `shotnoise` equals `num_shotnoise / norm`, the realised
  shot noise is subtracted and the self-pair term of the expected-shot-noise convention (validation
  report, control run) does not arise.
* **GCcomb.** If each GCcomb vector equals the norm-weighted average of the same mock's NGC and SGC,
  its covariance is $\sum_r n_r^2C_r/(\sum n_r)^2$ exactly (`combined-regions`); otherwise it is one
  estimate on the concatenated catalogues and `thecov` is run on the combined geometry.
* **Var(`num_shotnoise`)** over the mocks against its Poisson part $\sum_d w^4+\alpha^4\sum_r w^4$
  (from this mock's catalogues); clustering of the weighted counts adds to it (+10–40% in the report's
  Gaussian mocks).
""")

code(r"""
for r, c in D['spectra_checks'].items():
    pv = D['poisson_var_num_shotnoise'].get(r)
    pv_s = f'; Var/Poisson part = {c["num_shotnoise_var"] / sum(pv):.2f}' if pv else ''
    print(f"{r:7s} N={c['N']:4d}  realised shot noise subtracted: {c['shotnoise_is_realised']} (max rel dev {c['max_rel_dev']:.1e}); "
          f"norm {c['norm_mean']:.5g} (scatter {100 * c['norm_rel_std']:.2f}%){pv_s}")
print('GCcomb definition:', D['gccomb'])

fig, ax = plt.subplots(1, len(ells), figsize=(14, 3.4))
for r, s in spec.items():
    m_, e_ = s['vectors'].mean(0), s['vectors'].std(0)
    for i, ell in enumerate(ells):
        ax[i].errorbar(s['k'], s['k'] * m_[i * nb:(i + 1) * nb], s['k'] * e_[i * nb:(i + 1) * nb], label=r, lw=1)
for i, ell in enumerate(ells):
    ax[i].set_xlabel(r'$k$ [$h$/Mpc]'); ax[i].set_ylabel(rf'$k P_{ell}$'); ax[i].legend(fontsize=8)
plt.tight_layout()
""")

md(r"""
## I.3 Tracers: checks against the estimator

* $\int m^2/$`norm` — the mesh normalisation of jaxpower (data × randoms on 10 Mpc/$h$ cells) is
  smoothed over the fine veto masks, so it falls short of $\int m^2$ by up to ~20% on DESI footprints
  (1.23 for LRG1 holi). The estimator's mean is then $(\int m^2/{\rm norm})\times P$; with
  `model_norm_correction` (default) the model fed to thecov is the mocks' mean times
  ${\rm norm}/\int m^2$, and the covariance is normalised by `norm` (both needed: without the first,
  the clustering terms are too large by $(\int m^2/{\rm norm})^2$). The naive window is in addition high
  by $\sim\langle w^2\rangle/\langle w\rangle^2$.
* catalogue shot-noise numerator / the files' mean `num_shotnoise` — differs from 1 by $\simeq\alpha$ if
  fewer random files are loaded here than the spectra used ($\alpha^2\sum_r w^2\approx\alpha\sum_d w^2$). With
  `shotnoise_from_files` (default) the shot-noise window is scaled to the files' value instead.
* the warning "alpha ... implied ... ratio ~1.2" from `thecov.Tracer` compares NW with a kNN estimate
  of the random density, which the fine veto masks bias low by the same mechanism; `median
  m/(NX WEIGHT_FKP)` ≈ 1 and the area from the random count are the reliable checks of $m$.
""")

code(r"""
for key, ti in D['tracer_info'].items():
    print(f"{key:28s} alpha {ti['alpha']:.4g}, shot-noise scale {ti['shotnoise_scale']:.4f}, "
          f"int m^2 / norm = {ti.get('I_over_norm', float('nan')):.4f}, "
          f"catalogue shot noise / files = {ti.get('shotnoise_catalogue_over_files', float('nan')):.4f}")
""")

md(r"""
**Why $\int m^2\neq$ norm.** jaxpower's `norm` is $\alpha\sum_{\rm cells}D_cR_c/V_c$ on 10 Mpc/$h$ cells.
Recomputed here on this mock's catalogue for shrinking cells: at 10 Mpc/$h$ it should reproduce the
files' `norm` (which also confirms that the catalogue and weights are those of the spectra); as the
cells shrink it should approach thecov's $\int m^2$, if the difference is the dilution of cells that
straddle the footprint edges and veto holes. This is what justifies the model correction
${\rm norm}/\int m^2$.
""")

code(r"""
for r, rc in regs.items():
    mn = dc.mesh_normalization(rc, (10.0, 5.0, 2.5, 1.25))
    I_ = D['tracer_info'][f'{r}__random-density']['I_over_norm'] * spec[r]['norm'].mean()
    print(f"{r}: files' norm {spec[r]['norm'].mean():.5g}; thecov int m^2 {I_:.5g}; "
          + ', '.join(f'{cs:g} Mpc/h: {v:.5g} ({v / I_:.3f} of int m^2)' for cs, v in mn.items()))
""")

md(r"""
**Effective volume of the window.** The Gaussian clustering covariance scales as
$1/V_{\rm eff}=\int m^4/(\int m^2)^2$ (the amplitude of $m$ cancels). Completeness weights vary on the
sky with sharp boundaries (number of overlapping tiles); a 3D 32-neighbour average of the weights
(`random-density`, radius ~15 Mpc/$h$) smooths them and underestimates $1/V_{\rm eff}$, an angular
128-neighbour average (`angular`, ~0.1 deg) resolves them. `own-weight` (each random's own weight) is
biased high by the weight scatter and bounds it from above. A difference of $x$% here is a difference
of up to $x$% in the clustering part of the covariance.
""")

code(r"""
for r, rc in regs.items():
    vm = dc.window_moments(rc, surface_density_deg2=cfg.surface_density)
    ref = vm['random-density']
    print(f"{r}: 1/V_eff relative to 'random-density': " + ', '.join(f'{k_} {v / ref:.4f}' for k_, v in vm.items()))
""")

md(r"""
## I.4 Validation

For each region: the report's numbers, then its figures — mean multipoles with the thecov
$\pm\sigma$ band; $\sigma_{\rm mock}/\sigma_{\rm thecov}$ per element ($\pm1,2\sigma$ sampling bands); the
$\chi^2_i$ distribution and the eigenvalue density against Marchenko–Pastur; the correlation matrices
and residuals $z_{ij}$; $\langle\chi^2\rangle(k_{\max})$ per multipole; and the mock/thecov variance along
the near-null directions of thecov's correlation matrix (see I.6).
""")

code(r"""
V_ = D['validation']
for r in spec:
    key = (r, 'random-density') if (r, 'random-density') in V_ else (r, 'combined-regions')
    if key not in V_:
        continue
    out = V_[key]
    va.print_report(out, f'{DETAIL_BIN} {r} [{key[1]}]')
    figs = va.figures(out, spec[r]['vectors'], D['covariances'][key], spec[r]['k'], ells, title=f'{DETAIL_BIN} {r} [{key[1]}]')
    plt.show()
""")

md(r"""
## I.5 Variants: naive window, and how GCcomb is combined

$\langle\chi^2\rangle$/expected (deviation in $\sigma$) for every covariance built: the `NW` (smooth $m$)
set-up, the naive `NZ`$\times$`WEIGHT` one, and for GCcomb both the combined geometry (if the files
are a joint estimate) and the combination of the regions' covariances.
""")

code(r"""
f2 = lambda t: f'{t[0]:.3f} ({t[1]:+.1f})'
print(f"{'case':42s}{'all':>16s}" + ''.join(f'{f"P{l}":>16s}' for l in ells) + ''.join(f'{f"kmax {km}":>16s}' for km in (0.1, 0.2, 0.3)))
for (r, m), out in V_.items():
    pb = out['per_block']
    print(f'{r + " [" + m + "]":42s}{f2((out["chi2_ratio"], out["chi2_sigma"])):>16s}'
          + ''.join(f'{f2(pb[l][0]):>16s}' for l in ells)
          + ''.join(f'{f2(out["chi2_kmax"][km]["all"]) if km in out["chi2_kmax"] else "":>16s}' for km in (0.1, 0.2, 0.3)))

fig, ax = plt.subplots(len(spec), len(ells), figsize=(4.4 * len(ells), 2.8 * len(spec)), squeeze=False, sharex=True)
for i, r in enumerate(spec):
    N = len(spec[r]['vectors']); band = np.sqrt(2 / (N - 1))
    for j, ell in enumerate(ells):
        a = ax[i, j]; a.axhspan(1 - band, 1 + band, color='0.9'); a.axhline(1, color='k', lw=0.6)
        for m, ls in (('random-density', '-o'), ('angular', '-s'), ('none', '--'), ('combined-regions', ':'),
                      ('combined-regions [angular]', ':')):
            if (r, m) in V_:
                a.plot(spec[r]['k'], V_[(r, m)]['_sigma_ratio'][j * nb:(j + 1) * nb] ** 2, ls, ms=2, lw=1, label=m)
        a.set_title(f'{r}, ell={ell}', fontsize=9); a.set_ylabel(r'$\sigma^2_{\rm mock}/\sigma^2_{\rm thecov}$')
    ax[i, 0].legend(fontsize=7)
for a in ax[-1]: a.set_xlabel(r'$k$ [$h$/Mpc]')
plt.tight_layout()
""")

md(r"""
## I.6 Near-null directions and bin width

With bins narrower than $2\pi/D$ ($D$ the survey extent, radially its depth) the window correlates
neighbouring bins almost perfectly, and thecov's correlation matrix has eigenvalues $\ll1$ along
directions that alternate in sign from bin to bin. The full-vector $\chi^2$ is dominated by them, and
anything outside the Gaussian model shows up there first. On the synthetic test set (Gaussian
mocks, 0.005 bins, $k<0.1$) the mocks had 4–12× thecov's variance along the one or two smallest
directions, for reasons not yet identified; with 0.01 bins the same mocks passed every test. Below:
the same numbers here, and the whole validation repeated with bins twice as wide.
""")

code(r"""
for (r, m), out in V_.items():
    if m != 'none':
        print(f"{r:7s} {m:26s} corr. eigenvalues {np.round(out['null_eig'][:4], 4)}  mock/thecov variance {np.round(out['null_ratio'][:4], 2)}"
              f"   <chi2> {out['chi2_ratio']:.3f} ({out['chi2_sigma']:+.1f})")
""")

md(r"""
## I.7 Window functions
Tripolar window functions $Q_{\Lambda_1\Lambda_2\Lambda}(s)$ of the clustering ($W\times W$), mixed
($W\times S$) and shot-noise ($S\times S$) terms, normalised to $Q_{000}$ at the first bin.
""")

code(r"""
for r in ('NGC', 'SGC'):
    fn = pl.windows_path(OUT, DETAIL_BIN, f'{DETAIL_BIN}_{r}')
    if os.path.exists(fn):
        va.window_figure(fn, title=f'{DETAIL_BIN} {r}'); plt.show()
""")

md(r"""
## I.8 Optional: previous thecov covariances
If the covariances of the earlier version exist (the example's `load_covariance`), compare diagonals.
""")

code(r"""
OLD = f'/dvs_ro/cfs/cdirs/desi/users/oalves/thecovs/cai/holi/cov_{DETAIL_BIN}_{{region}}.txt'
for r in spec:
    fn = OLD.format(region=r)
    key = (r, 'random-density') if (r, 'random-density') in D['covariances'] else (r, 'combined-regions')
    if os.path.exists(fn) and key in D['covariances']:
        Cold, Cnew = np.loadtxt(fn), D['covariances'][key]
        if Cold.shape != Cnew.shape:
            # old binning: every 5th 0.001 bin from k = 0 -> 0.005-wide bins from 0; pick ours
            nb_old = Cold.shape[0] // len(ells)
            k_old = (np.arange(nb_old) + 0.5) * 0.005
            sel = np.array([np.argmin(np.abs(k_old - k_)) for k_ in spec[r]['k']])
            if np.max(np.abs(k_old[sel] - spec[r]['k'])) > 1e-3:
                print(r, 'old covariance', Cold.shape, ': k bins do not match'); continue
            idx = np.concatenate([j * nb_old + sel for j in range(len(ells))])
            Cold = Cold[np.ix_(idx, idx)]
        rat = np.diag(Cold) / np.diag(Cnew)
        print(r, 'old/new diagonal, mean per ell:', [rat[j * nb:(j + 1) * nb].mean().round(3) for j in range(len(ells))])
        try:
            o = va.validate(spec[r]['vectors'], Cold, spec[r]['k'], ells)
            print(f"   old covariance vs mocks: <chi2>/n {o['chi2_ratio']:.3f} ({o['chi2_sigma']:+.1f}); variance ratio per ell "
                  + ', '.join(f'{l}: {v:.3f}' for l, v in o['var_ratio_per_ell'].items())
                  + f"; z mean {o['corr_resid_mean']:+.2f}")
        except ValueError as ex:
            print('   old covariance:', ex)
""")

md(r"""
# Part II — all tracer bins

The same pipeline (`NW` set-up only) for every bin in `BINS`. Figures for each bin and region are
written to `OUT/<bin>/fig_<region>_*.pdf`; the numbers to `OUT/summary.json`. A bin that fails (e.g.
missing files) is reported and skipped. Cached pair counts and covariances are reused, so the loop
can be interrupted and rerun.
""")

code(r"""
results = []
for b in BINS:
    if b == DETAIL_BIN:
        res = D
    else:
        try:
            res = pl.run_bin(paths, b, cfg, OUT)
        except Exception as ex:                       # keep going: report and skip
            print(f'{b}: FAILED: {type(ex).__name__}: {ex}')
            continue
    results.append(res)
    for (r, m), out in res.get('validation', {}).items():
        if m == 'none' or 'bins x2' in m:
            continue
        s = res['spectra'][r]
        figs = va.figures(out, s['vectors'], res['covariances'][(r, m)], s['k'], cfg.ells, title=f'{b} {r} [{m}]')
        for name, f in figs.items():
            f.savefig(os.path.join(OUT, b, f'fig_{r}_{m}_{name}.pdf')); plt.close(f)
    pl.to_json(results, os.path.join(OUT, 'summary.json'))
""")

code(r"""
f2 = lambda t: f'{t[0]:.3f} ({t[1]:+.1f})'
for mode in ('random-density', 'combined-regions', 'random-density, bins x2'):
    rows = pl.summary_table(results, mode)
    if not rows:
        continue
    print(f'\n[{mode}]  <chi2>/expected (sigma): whole vector and per k_max; eigenvalue range vs Marchenko-Pastur; '
          'mean variance ratio; std of z_ij; first-bins variance ratio; mock/thecov variance along the smallest direction')
    kms = [k_ for k_ in rows[0] if k_.startswith('chi2_k')]
    print(f"{'bin':9s}{'reg':7s}{'N':>5s}{'n':>5s}{'all':>16s}" + ''.join(f'{k_[6:]:>16s}' for k_ in kms)
          + f"{'eig':>14s}{'MP':>14s}{'var':>7s}{'z std':>7s}{'lowk':>7s}{'null':>7s}")
    for rw in rows:
        ev, mp = rw['ev'], rw['mp']
        print(f"{rw['bin']:9s}{rw['region']:7s}{rw['N']:5d}{rw['n']:5d}{f2((rw['chi2'], rw['chi2_sig'])):>16s}"
              + ''.join(f'{f2(rw[k_]):>16s}' for k_ in kms)
              + f"{f'[{ev[0]:.2f},{ev[1]:.2f}]':>14s}" + f"{(f'[{mp[0]:.2f},{mp[1]:.2f}]' if mp else 'n/a'):>14s}"
              + f"{rw['var_ratio']:7.3f}{rw['z_std']:7.3f}{rw['lowk']:7.3f}{rw['null_ratio']:7.2f}")
""")

md(r"""
### Across bins
Left: $\langle\chi^2\rangle$/expected of the whole vector as a function of $k_{\max}$ (grey: $\pm1\sigma$ for
the median $N$). Others: the variance ratio $\sigma^2_{\rm mock}/\sigma^2_{\rm thecov}$ of $P_0$, $P_2$,
$P_4$ averaged in $k$ bins of width 0.05. Colour: tracer bin; line style: region.
""")

code(r"""
regions_style = {'NGC': '-o', 'SGC': '--s', 'GCcomb': ':^'}
fig, ax = plt.subplots(1, 1 + len(cfg.ells), figsize=(4.6 * (1 + len(cfg.ells)), 3.6))
cmap = plt.get_cmap('tab10')
Ns, kms = [], None
for ib, res in enumerate(results):
    for (r, m), out in res.get('validation', {}).items():
        if m not in ('random-density', 'combined-regions') or (m == 'combined-regions' and (r, 'random-density') in res['validation']):
            continue
        Ns.append(out['N']); kms = out['kmax_list']
        ax[0].plot(kms, [out['chi2_kmax'][km]['all'][0] for km in kms], regions_style.get(r, '-'), color=cmap(ib % 10), ms=3, lw=1,
                   label=f"{res['bin']} {r}")
        k_ = np.asarray(out['k']); nb_ = len(k_); edges = np.arange(cfg.kmin, cfg.kmax + 0.05, 0.05)
        idx = np.digitize(k_, edges) - 1
        for j, ell in enumerate(cfg.ells):
            v = out['_sigma_ratio'][j * nb_:(j + 1) * nb_] ** 2
            sel = [i for i in range(len(edges) - 1) if np.any(idx == i)]
            ax[1 + j].plot([k_[idx == i].mean() for i in sel], [v[idx == i].mean() for i in sel],
                           regions_style.get(r, '-'), color=cmap(ib % 10), ms=3, lw=1)
if Ns:
    Nm = int(np.median(Ns)); kk = np.array(kms)
    n_k = np.array([len(cfg.ells) * max(1, int(round((km - cfg.kmin) / (0.001 * cfg.rebin)))) for km in kk])
    ax[0].fill_between(kk, 1 - np.sqrt(2 / (n_k * Nm)), 1 + np.sqrt(2 / (n_k * Nm)), color='0.88', zorder=0)
ax[0].axhline(1, color='k', lw=0.6); ax[0].set_xlabel(r'$k_{\max}$'); ax[0].set_ylabel(r'$\langle\chi^2\rangle$ / expected')
ax[0].legend(fontsize=6, ncol=2)
for j, ell in enumerate(cfg.ells):
    ax[1 + j].axhline(1, color='k', lw=0.6); ax[1 + j].set_xlabel(r'$k$ [$h$/Mpc]')
    ax[1 + j].set_ylabel(rf'$\sigma^2_{{\rm mock}}/\sigma^2_{{\rm thecov}}$, $P_{ell}$')
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_all_bins.pdf'))
""")

md(r"""
### Reading the results
* **Set-up checks first** (Part I, and `summary.json` → `tracer_info`, `spectra_checks`):
  `median m/(NX WEIGHT_FKP)` ≈ 1, the area, and a catalogue shot noise within ≈ α of the files'.
  A uniform offset of the variance ratio at all $k$ and $\ell$ points to an amplitude input (model
  normalisation, shot noise), not to the window.
* **Low $k$** ($k_{\max}\lesssim0.1$): Gaussian terms dominate; $\langle\chi^2\rangle$ should be close to 1
  and the eigenvalues inside Marchenko–Pastur, up to the near-null directions of I.6 (compare with
  `bins x2`).
* **High $k$**: an excess growing with $k$, mostly in $P_0$ and in the off-diagonal correlations, is
  the non-Gaussian (trispectrum, super-sample) covariance that `thecov` does not include. Its size per
  tracer bin is what this notebook measures.
* **Low-$k$ deficit**: a first-bins variance ratio slightly below 1 is the expected integral
  constraint (realised $\alpha$), not modelled.
""")

nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}}
nbf.write(nb, 'desi_validation/desi_covariance_comparison.ipynb')
print('wrote desi_validation/desi_covariance_comparison.ipynb with', len(cells), 'cells')
