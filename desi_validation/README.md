# thecov vs. DESI mock covariances

* `desi_covariance_comparison.ipynb` — the comparison, one tracer bin per run (NGC, SGC, GCcomb).
  Regenerate it with `python desi_validation/make_notebook.py` after editing that script.
* `desi_compare.py` — loading (mpytools / h5py / fitsio), weight diagnostics, `Tracer` construction,
  thecov covariance, comparison statistics. Reimplements what is needed from the old `catalogs.py`.
* `synthetic.py` — a small data set in the same formats with a known answer
  (`python -m desi_validation.synthetic <dir> 200`, ~30 min on 4 cores); run the notebook on it with
  `THECOV_SYNTHETIC=1 THECOV_SYNTHETIC_DIR=<dir>`.

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

* The geometry path is `holi_v1/altmtl201` while the spectra are `holi-v3-altmtl`: use the
  catalogues of the mocks whose spectra are compared.
* `select_region('SGC')` uses `not array`, which raises; per-region files need no RA/DEC cut anyway.
* The first k bins (`k L <~ 1`) are outside the regime of any windowed Gaussian covariance and can
  make it non-positive-definite; the notebook starts at `KMIN = 0.02`.
* The model is the window-convolved mean of the mocks (an approximation at the lowest k).
* Real mocks are non-Gaussian: expect an excess of mock variance and correlations growing with k.
