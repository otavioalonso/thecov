# Task for Claude Code (run at NERSC): why does the LRG1 covariance fall short of the mocks?

You are working in the `thecov` repository, branch `thecov2`, on NERSC (Perlmutter). Read
`desi_validation/README.md` first. Do not modify `thecov/` (the library). You may add scripts under
`desi_validation/scratch/` (create it). Run anything heavier than a few minutes inside an interactive
allocation (`salloc -N 1 -C cpu -q interactive -t 01:00:00 -A <account>`; ask the user for the
account), not on a login node. Use the Python environment the user's notebook uses (cosmodesiconda,
with `clustering_statistics`, `jaxpower`, `lsstypes`, `mpytools` importable); check with
`python -c "import clustering_statistics.tools, jaxpower, lsstypes"`.

## Context

`desi_validation/desi_covariance_comparison.ipynb` compares the Gaussian covariance from `thecov`
with the sample covariance of 859 DESI DR2 holi-v3 mock power spectra (P0, P2, P4; 0.005 h/Mpc bins,
k = 0.02-0.3). The pipeline is `desi_validation/pipeline.py:run_bin`; catalogues are read with
`clustering_statistics.tools.read_clustering_catalog` (`desi_compare.load_region_cs`).

Results so far (mock variance / thecov variance; <chi2>/n with the mocks' own mean):

| bin  | region | <chi2>/n | var ratio all / P0 / P2 / P4 | corr. residual z mean | eig. above Marchenko-Pastur |
|------|--------|----------|------------------------------|-----------------------|-----------------------------|
| QSO  | NGC    | 1.009    | 1.015 / 1.036 / 0.999 / 1.009 | +0.12                 | 1                           |
| QSO  | SGC    | 1.000    | 1.005 / 1.035 / 0.999 / 0.980 | +0.16                 | 1                           |
| LRG1 | NGC    | 1.096    | 1.131 / 1.205 / 1.120 / 1.069 | +1.15                 | 7                           |
| LRG1 | SGC    | 1.099    | 1.133 / 1.188 / 1.126 / 1.084 | +0.85                 | 8                           |

QSO (shot-noise dominated) passes, so shot noise, normalisation and the model correction are right.
LRG1 has a ~13% excess, roughly flat in k (already 1.12 in the first three bins), with strongly
positive correlation residuals: a correlated, low-rank extra component, not just a wrong effective
volume. Already ruled out: angular vs 3D smoothing of the weights in m (changes 1/V_eff by 2%),
catalogue/weight mismatch (our DR at 10 Mpc/h reproduces the files' `norm` to 0.6%), the old thecov
covariances (2-3% lower still). Open puzzle: DR (jaxpower's norm) reaches only 0.84-0.88 of thecov's
int m^2 at 1.25 Mpc/h cells.

## Steps

1. `git pull origin thecov2`. Confirm `desi_compare.mesh_normalization` returns `{'DR', 'RR'}` per
   cell size and `validation.excess_structure` exists.

2. Write `desi_validation/scratch/lrg1_excess.py` that, for `TRACER_BIN='LRG1'` and both NGC and SGC:
   - sets up `paths` and `cfg` exactly as in the notebook's setup cell (holi_v3, mock 173,
     `loader='auto'`, `cs_version='holi-v3-altmtl'`, `cs_parent_version='data-dr2-v2'`,
     `cfg = pipeline.Config()` defaults, `OUT = ~/thecov_desi/holi_v3_mock173`), so cached windows
     and covariances in `OUT/LRG1/` are reused (it should not recompute pair counts; if it starts to,
     stop and report the cache file names it looked for);
   - runs `D = pipeline.run_bin(paths, 'LRG1', cfg, OUT, keep=True)`;
   - **Test A (footprint consistency):** for each region, `dc.mesh_normalization(rc, (10, 5, 2.5,
     1.25, 0.6, 0.3))`; print DR, RR, DR/RR and both relative to thecov's int m^2
     (`D['tracer_info'][f'{r}__random-density']['I_over_norm'] * spec[r]['norm'].mean()`). Also run it
     with 4 random files (`dataclasses.replace(cfg, n_random_files=4)` and `dc.load_region(...)`
     directly) to see whether the small-cell values are noise-limited.
   - **Test B (structure of the excess):** for each region and for GCcomb (`combined-regions`),
     `va.excess_structure(V, C, templates=...)` with templates: the mean vector (all multipoles), and
     the mean restricted to each multipole (zeros elsewhere), and additionally
     `k dP/dk` of the mean per multipole (finite differences, a dilation-like template). Print the top
     10 whitened eigenvalues, `noise_edge`, `frac_top`, and per template sigma, chi2_after,
     var_ratio_after.
   - **Test C (is the excess the same in every mock?):** split the 859 mocks into halves (even/odd
     index) and into thirds; recompute the variance ratio per multipole and <chi2>/n for each subset.
     Also list the mocks with the largest chi2_i (top 10, from `va.chi2_stats`) and check whether they
     are clustered in mock id (print mock ids from `pipeline.mock_ids(spec[r])`).
   - **Test D (per-mock normalisation):** correlate each mock's chi2_i and its projection on the mean
     vector (`(V - V.mean(0)) @ C^-1 @ mean`) with the mock's `norm` and `num_shotnoise`
     (`spec[r]['norm']`, `spec[r]['num_shotnoise']`); print Pearson r and its 3-sigma threshold
     (3/sqrt(N)).
   - saves all numbers to `OUT/LRG1/excess_tests.json`.

3. Run it (in the allocation). If it fails, fix the scratch script (not the library) and rerun.

4. Report back, concisely, in this format:
   - Test A table: cell size | DR/RR NGC | DR/RR SGC | DR/I NGC | RR/I NGC (1 and 4 random files).
   - Test B: top eigenvalues vs noise edge; frac_top; table of templates with sigma, chi2_after,
     var_ratio_after.
   - Test C: subset table; top-chi2 mock ids.
   - Test D: correlations.
   - A short interpretation using the guide below, and any errors you hit.

## Interpretation guide

- **DR/RR < 1 at small cells (and decreasing as cells shrink):** randoms occupy regions the data cannot
  (a veto applied to the randoms but not the data, or vice versa); thecov's window, built from the
  randoms, is then too uniform. This would explain both the int m^2 / norm gap and part of the excess.
  Report the cell size where DR/RR departs from 1.
- **DR/RR ~ 1 at all cells, DR and RR still rising at 0.3 Mpc/h:** the norm/int m^2 gap is just fine
  veto structure; not a problem.
- **One template (amplitude) with sigma ~ few % bringing chi2_after to ~1.00-1.02:** each mock carries a
  common amplitude fluctuation (super-sample modes, per-realisation fibre assignment, or box
  replication in the mocks). Not a thecov bug; report sigma per region.
- **Excess spread over many eigenmodes (frac_top[10] small, many eigenvalues just above the noise
  edge):** a broadband shortfall of the Gaussian clustering term; report it and stop.
- **Excess concentrated in a few mocks (Test C):** a problem with those mock spectra; list them.
- **Strong correlation with norm or num_shotnoise (Test D):** the per-mock normalisation or the
  shot-noise subtraction contributes; report r.

Do not change defaults in `desi_validation/` or push anything; leave the scratch script and the JSON
for the user.
