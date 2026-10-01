# Task for Claude Code (run at NERSC): does thecov's window miss an n(z) that varies across the sky?

Same rules as `TASK_lrg1_excess.md`: branch `thecov2`, do not modify `thecov/`, scripts go in
`desi_validation/scratch/`, heavy work in an `salloc` allocation, same environment, same `paths`/`cfg`
set-up as the notebook (holi_v3, mock 173, `loader='auto'`, `OUT=~/thecov_desi/holi_v3_mock173`).

## Context (from your previous runs)

`scratch/lrg1_mesh_gaussian.py` computed the exact Gaussian P0 covariance on an FFT mesh:
- window A (thecov's own m) agrees with thecov to +2-3% at all k;
- window B (m straight from local random counts, 4 files, cell by cell) gives +6-9% more variance than
  thecov, converged in the cell size (`lrg1_mesh_resolution.py`: 1.089, 1.080, 1.072, 1.062, 1.063 at
  h = 12, 6, 3, 1.5, 0.75 Mpc/h; 0.375 is noise-dominated);
- mocks / B is ~1.01-1.02 at k < 0.05, i.e. B explains the low-k excess; the remainder grows with k.

Since the amplitude of m cancels in the covariance, A and B differ in the SHAPE of m on large scales.
thecov's m = alpha rho_r(z) <w>_local uses ONE rho_r(z) per cap; if the randoms' n(z) varies across
the sky (e.g. redshifts drawn from the data of their own imaging region, BASS/MzLS vs DECaLS in NGC),
it cannot follow it. `desi_compare` now has `nw='patch'`: rho_r(z) measured per healpix patch
(`random_density_patches`, nside 8 by default), and `nz_variation` to measure how much n(z) varies.

## Steps

1. `git pull origin thecov2`. Check that `dc.random_density_patches`, `dc.nz_variation` exist and
   that `dc.load_region(...)` puts `pix_counts_all` in `rc.info` (healpy must be importable).

2. **n(z) variation.** For LRG1 NGC and SGC (and QSO NGC, which validated, as a control), with
   1 random file: `dc.nz_variation(rc, nside=8)` and `nside=4`. Report the rms deviation vs the
   Poisson expectation, `mean_z_north_minus_south`, and the north/south n(z) ratio per z (NGC).

3. **1/V_eff.** `dc.window_moments(rc, modes=('random-density', 'patch'), nside=s)` for s in 4, 8, 16
   (pass `nside=s`), LRG1 NGC/SGC and QSO NGC. Report patch / random-density.

4. **Mesh test with the patch window.** Extend `scratch/lrg1_mesh_gaussian.py` (copy it to
   `lrg1_mesh_patch.py`): add window `A_patch`, built exactly like A but from
   `tr_p, _ = dc.build_tracer(f'LRG1_{r}_patch', [rc1], nw='patch', n_randoms_max=cfg.n_randoms_max // 2,
   surface_density_deg2=cfg.surface_density, verbose=False)` with `rc1` the 1-file region from
   `run_bin`'s `D['regions'][r]`. Report A_patch / thecov, A_patch / B (B at its 12 Mpc/h value, which
   your previous run saved in `mesh_gaussian_<r>.json`; also compare with the h = 1.5 Mpc/h value in
   `mesh_resolution_<r>.json`), and mocks / A_patch, per k range (0.02-0.05, 0.05-0.1, 0.1-0.2).
   Reuse the cached thecov covariance for the "thecov" column.

5. **thecov with the patch window.** Run
   `pl.run_bin(paths, 'LRG1', dataclasses.replace(cfg, nw_modes=('random-density', 'patch')), OUT, keep=True)`
   (new pair counts for the patch tracers, ~80 s per region; the random-density ones are cached). For
   NGC, SGC and GCcomb (`combined-regions [patch]`) print `va.print_report` for both modes and the
   mean variance ratio per multipole in k ranges 0.02-0.05, 0.05-0.1, 0.1-0.2, 0.2-0.3.

6. **Optional, the 2-3% A vs thecov offset.** Repeat the mesh A computation and a thecov covariance
   with an isotropic model (P2 = P4 = 0: zero those rows of the mean vector in a copy of `spec` before
   `dc.thecov_covariance`, windows cached; in the mesh script set `Pl[2] = Pl[4] = 0`). If A / thecov
   becomes ~1.000, the offset was the mesh's global line of sight, not thecov.

7. Save numbers to `OUT/LRG1/window_shape_tests.json`. Report:
   - step 2 table (rms vs Poisson, <z> N-S), and the N/S n(z) ratio vs z for LRG1 NGC;
   - step 3 table;
   - step 4 table: k range | A/thecov | A_patch/thecov | B/thecov | mocks/A_patch | mocks/B;
   - step 5: <chi2>/n and variance ratios per k range, random-density vs patch;
   - step 6 if run.

## Interpretation guide

- n(z) rms >> Poisson (LRG1) but ~Poisson (QSO), and A_patch ~ B: the cap-wide n(z) was the missing
  window structure; `nw='patch'` should become the default and the remaining excess (growing with k)
  is non-Gaussian (SSC + trispectrum).
- n(z) rms ~ Poisson and A_patch ~ A: the missing structure is elsewhere (3D, at scales between the
  patch size and ~15 Mpc/h); report and stop.
- A_patch between A and B: partly; try nside 16 / 32 in step 4 and report the trend.

Do not change defaults or push; leave the scripts and the JSON for the user.
