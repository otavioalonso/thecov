# thecov vs. DESI mock covariances

* `desi_covariance_comparison.ipynb`: the validation. Part I looks at one bin in detail (weights,
  the checks against the spectrum files, every test of the validation report, figures). Part II
  loops over every tracer bin and region and writes the tables, `summary.json` and figures to `OUT`.
  Default range: P0,2,4 in 0.005 bins, k = 0.02-0.3; also repeated with 0.01 bins. Regenerate it with
  `python desi_validation/make_notebook.py` after editing that script.
* `pipeline.py`: one tracer bin end to end (`run_bin`). It caches window pair counts and
  covariances in `OUT/<bin>/`, detects whether GCcomb is the norm-weighted average of NGC and SGC or a
  joint estimate, and checks the shot-noise convention of the files.
* `validation.py`: the report's statistics (`validate`, `print_report`) and figures (`figures`,
  `window_figure`).
* `desi_compare.py`: loading (mpytools / h5py / fitsio), weight diagnostics, `Tracer` construction
  with `NW`, and the thecov covariance.
* `synthetic.py`: a small data set in the same formats with a known answer
  (`python -m desi_validation.synthetic <dir> <n_mocks> [--random-z data|nz] [--fresh-randoms]
  [--N 128] [--box-factor 1] [--regions NGC,SGC,GCcomb] [--amplitude 1]`). Its clustering is cut
  at k ~ 0.1. Run the notebook on it with `THECOV_SYNTHETIC=1 THECOV_SYNTHETIC_DIR=<dir>`.

## Weights: what the covariance needs

The spectra (`weight-default-FKP`) use `w = WEIGHT * WEIGHT_FKP` on data and randoms, and
`alpha = sum_d w / sum_r w`. `WEIGHT` is per object and the randoms inherit it from random data
objects, so:

1. **Clustering window.** It needs the smooth mean weighted density
   `m(x) = E[sum_g w_g delta_D(x - x_g)]`, not `NX * w` evaluated with each random's own weight: the
   latter gives `<w^2>` where `<w>^2` is needed, i.e. a window too large by
   `1 + var(w)/<w>^2` (~7% on the synthetic set). `build_tracer` builds
   `m = alpha * rho_randoms(z) * <w>_local` (column `NW` of `thecov.Tracer`) from the random density,
   so it does not depend on what `NX` means (it prints `m / (NX WEIGHT_FKP)`, which answers that).
2. **Shot noise.** Rescaled to the realised `sum_d w^2 + alpha^2 sum_r w^2` (`num_shotnoise`).
3. **Normalisation.** The covariance is normalised by the spectrum files' `norm` (the mesh
   normalisation of jaxpower/pypower), not by thecov's `int m^2`.
4. **alpha.** Weighted. The old `get_alpha` for a single region used `len(data)/len(randoms)`.

## Caveats found in the example script

* The example took the geometry from `holi_v1/altmtl201` while the spectra are `holi-v3-altmtl`; the
  default is now `holi_v3/altmtl173` (catalogue tracer names: `ELGnotqso` for `ELG_LOPnotqso`).
* `select_region('SGC')` uses `not array`, which raises; per-region files need no RA/DEC cut anyway.
* The first k bins (`k L <~ 1`) are outside the regime of any windowed Gaussian covariance and can
  make it non-positive-definite; the notebook starts at `KMIN = 0.02`.
* The model is the window-convolved mean of the mocks (an approximation at the lowest k).
* Real mocks are non-Gaussian: expect an excess of mock variance and correlations growing with k.

## Result on the synthetic set (LRG-like, z 0.4-0.6, two 22-degree caps, ~100 mocks, k 0.02-0.1)

* Set-up checks: the spectra's `num_shotnoise` equals the catalogue value; thecov's `int m^2` with `NW`
  is within 0.8-1.7% of the mesh `norm`; with the naive `NZ * WEIGHT` it is 7-8% high. The predicted
  shot noise matches the files to 0.4%.
* Per-multipole blocks: `<chi2>/n` is within ~1-2 sigma of 1 with `NW` (NGC, SGC, GCcomb). With the naive
  window the covariance is 5-15% too large (`<chi2>/n` 0.84-0.87, -4 sigma, on NGC P0 and P2).
* Full vector with 0.01-wide bins: `<chi2>/n` = 0.99-1.04 and eigenvalues inside Marchenko-Pastur.
* Full vector with 0.005-wide bins: thecov's correlation matrix has 1-2 near-null directions
  (eigenvalue ~1e-3). They alternate in sign from bin to bin and combine P0+P2+P4, i.e. radial
  modes. The mocks put 4-12 times more variance there, which gives `<chi2>/n` = 1.03-1.15 per cap
  and 1.3-1.6 for GCcomb.
  Not the cause: pair-count sampling, `L_max` (4 or 8), the s resolution, randoms taking the
  data's redshifts vs a smooth n(z), and one random catalogue for all mocks vs a fresh one per mock.
  Open. The notebook prints this diagnostic (`dc.eigen_directions`) and repeats the comparison with
  bins twice as wide, so it can be checked on the DESI mocks.

