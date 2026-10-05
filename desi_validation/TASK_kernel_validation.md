# Task for Claude Code (run at NERSC): validate the pair-averaged clustering window (WindowSmoothing)

Handoff from the cloud session that wrote the code. Everything you need is in this file and in the
repository; read `desi_validation/report/window_kernel.typ` for the derivation if needed.

## Rules

- Branch `thecov2`. `git pull origin thecov2` first. Commit and push your changes there (no PR). Do not put
  model names in commits. Commit trailer: `Co-Authored-By: Claude <noreply@anthropic.com>`.
- **Debug queue only** (`-q debug`, 30 min, 1 CPU node). No regular queue. At most 2 debug jobs at once.
  One job per cap (NGC, SGC), then a combine job that only reads caches.
- Jobs inherit the login environment (`--export=ALL`); run them from `~/thecov`.
- Write commands with real paths, no `<placeholders>`.
- `thecov/` (core) may be changed if a run shows a bug; then run `python -m pytest -q tests/test_smoothing.py`
  (5 tests, ~3 min) before pushing. DESI-specific code goes in `desi_validation/` only.
- Report numbers, not impressions. When something fails, show the log lines.

## Paths

- Repository: `~/thecov`. Caches and outputs: `~/thecov_desi/holi_v3_mock173` (`OUT`).
- Mocks: holi-v3-altmtl, 859 realisations; catalogues for mock 173 (`dc.Paths(kind='holi_v3', mock=173)`,
  set up inside `dump_report_data.py`).
- **Files for the user** (logs, summaries, small outputs): put them, or copies, in
  `/global/cfs/cdirs/desicollab/users/oalves/thecov_validation` (easier for the user to reach than `~`).
- Logs: `~/kcore2_NGC.log`, `~/kcore2_SGC.log`, `~/kcore2_all.log` (names used below).

## Physics in brief

Gaussian covariance of P_ell(k) for DESI LRG1 (0.4 < z < 0.6). Fits use k = 0.02-0.2 (dk = 0.005);
validation goes to k = 0.3, ells 0, 2, 4. Metric: <chi2>/n of the mock vectors with the thecov
covariance (1 = perfect), and the variance ratio var(mocks)/var(thecov) per k range.

- thecov's default clustering window is local, W = m^2. DESI LRG1 has many sub-arcmin veto holes, so
  int m^2 / norm = 1.2254 (NGC), 1.2104 (SGC): m^2 at a point overweights the hole edges.
- The exact Gaussian answer pair-averages the window over the correlation length:
  W_k(x) = m(x) (K_k * m)(x), K_k(r) = xi_0(r) j_0(kr) / P_0(k) averaged over the bin, int K_k = 1.
  The mean of the estimator becomes <P_hat(k)> = P(k) I_k / norm, I_k = int W_k.
- `thecov.WindowSmoothing` implements it per k-bin with a two-stage basis:
  1. kernel basis (SVD of the per-bin kernels, metric 4 pi r^2 Q_mm); converged to tol = 2e-3; about 17-20
     kernels are needed for 56 bins of 0.005 up to k = 0.3. I_k comes from this stage;
  2. window basis: SVD of the smoothed windows (K * m)(x) at the randoms in the metric int m^2 v v'.
     Only these enter the pair counts (cost ~ B^2). Synthetic test: 17 kernels -> 6 windows.
- `GaussianCovariance.set_model(model, masked=True)`: the model is the window-convolved mock mean
  (multiplied by norm / I_k); `masked=False`: a theory P, used as is.
- Pipeline mode `nw='kernel'` (`desi_validation/pipeline.py`, `kernel_*` fields of `Config`).

## Results so far (LRG1; the reference numbers for this task)

| chi2/n, k = 0.02-0.2 (x1)  | NGC   | SGC   | GCcomb |
|---|---|---|---|
| default m^2 (random-density)      | 1.090 | 1.095 | 1.095 |
| kernel, single-stage, 12 kernels (not converged, I_k error 6.6e-2), default sampling | 1.048 | 1.044 | 1.049 |
| fill (m x healpix fill fraction, nside 128, hand-tuned), default sampling | 1.034 | 1.029 | - |
| fill, production sampling (`--target-near-pairs 1e10 --n-randoms-max 8e6`) | 1.014 | 1.023 | 1.019 |

Over k = 0.02-0.3: default 1.096 / 1.099 / 1.098, kernel 1.059 / 1.050 / 1.056, fill-production 1.024 / 1.031 / 1.026.

Bin-width split var ratio - 1 = a + b (dk/0.005), NGC+SGC mean, k < 0.2, ell = 0/2/4:
default a = 0.106/0.096/0.079; kernel a = 0.036/0.025/0.008; fill-production a = 0.035/0.021/0.004;
b = 0.04/0.02/0.01 for all (non-Gaussian, not a window effect).

Gaussian-random-field test on the real footprint (`run_gaussian_footprint.py`): the exact Gaussian
variance needs the input power calibrated by c(k) = norm / I_k: NGC 1.276, 1.242, 1.214, 1.202;
SGC 1.262, 1.220, 1.193, 1.179 in four k bands (0.02-0.3). The single-stage kernel gave
int m (K_k * m) / norm = 0.9258 (first bin) ... 1.0196 (middle) ... 1.0246 (last) NGC and
0.9249 ... 1.0249 ... 1.0333 SGC, i.e. c = 1.324 (first bin), 1.202, 1.196 NGC. The first bin is too
high, consistent with the unconverged basis underestimating I_k at low k.

Timing of the single-stage run (NGC): setup 420 s, smoothing 145 s, pair counts 476 s (12 windows,
n_near 1.3e6), x2/x4 binnings 60 s each. Fill-production pair counts: 215 s with 1 window, n_near 3.0e6.

## Steps

1. **Converged kernel run at production sampling** (two-stage basis, commit 1a6794b or later):
   ```
   cd ~/thecov && git pull origin thecov2 && for r in NGC SGC; do sbatch -N 1 -C cpu -q debug -t 00:30:00 -J kc$r -o ~/kcore2_$r.log --export=ALL --wrap "cd ~/thecov && python -u -m desi_validation.dump_report_data --label holi-kcore2-$r-LRG1 --bins LRG1 --modes kernel --regions $r --no-naive --target-near-pairs 1e10 --n-randoms-max 8e6 --cache-tag _r8p10"; done
   ```
   Check in each log: the line `N kernels -> B windows (errors: I_k ..., windows ...)` (both errors below
   2e-3; report N and B), the pair-count time on the `cov_..._r5_...` line, and
   `int m (K_k * m) / norm`. If a job times out, resubmit the same line: finished smoothings and pair
   counts are cached (`OUT/*.smoothing.npz` next to the windows) and reused.

2. **Combine** (after both finish; reads the caches):
   ```
   cd ~/thecov && sbatch -N 1 -C cpu -q debug -t 00:30:00 -J kcall -o ~/kcore2_all.log --export=ALL --wrap "cd ~/thecov && python -u -m desi_validation.dump_report_data --label holi-kcore2-LRG1 --bins LRG1 --modes kernel --no-naive --target-near-pairs 1e10 --n-randoms-max 8e6 --cache-tag _r8p10"
   ```

3. **Analyse** against the references (all files are in `OUT`):
   ```
   cd ~/thecov && python -m desi_validation.compare_kernel holi-kcore2-LRG1:kernel holi-fillprod-LRG1:fill holi-kcore-LRG1:kernel holi-kcore-LRG1:random-density
   ```
   Report: chi2/n per k range and region, the variance ratios at 0.02-0.06 (where the converged basis
   should differ from the single-stage run), a and b. Also the predicted calibration
   1.2254 / [int m (K_k * m)/norm] (NGC), 1.2104 / [...] (SGC) for the first, middle and last bins vs
   the GRF c(k) above.

   Expected: chi2/n close to fill-production (~1.02) without a hand-tuned scale; first-bin calibration
   moving from 1.324 towards ~1.28. If chi2/n is clearly worse than fill-production, look at the
   variance ratios per k range to find where, then at the basis errors and sizes in the log.

4. **If B (window basis) is large** (> ~10) or the pair counts do not fit 30 min: try
   `--kernel-tol 5e-3` (another label and cache tag, e.g. `holi-kcore2t5-...`, `--cache-tag _r8p10t5`),
   and compare chi2/n with step 1 to show the tolerance does not matter at the 0.005 level.

5. **QSO control** (it validated with the default window, chi2/n ~ 1; the kernel must not spoil it):
   same as steps 1-3 with `--bins QSO`, labels `holi-kcore2-$r-QSO` / `holi-kcore2-QSO`, then
   `python -m desi_validation.compare_kernel --bin QSO holi-kcore2-QSO:kernel`.

6. Write a short summary (numbers and the table above, updated) to
   `desi_validation/report/kernel_validation_results.md`, commit, push.

## Later (only if steps 1-6 are done and the user agrees)

- All tracers (BGS, ELG, LRG+ELG need their catalogue names checked in `dc.TRACER_SPECS`).
- Make `kernel` the default window mode for DESI if it matches fill-production.
- Update the report (`desi_validation/report/report.tex`) with a kernel section.
- SSC module; redshift-space (anisotropic) kernel, tested with the GRF in redshift space.

## Known harmless messages

- `WEIGHT not in catalog: ...dark_N_full_noveto.ran.h5`: the noveto randoms have no WEIGHT column; expected.
- `UserWarning: Tracer ...: alpha=0.3331 but ... (ratio 1.61)`: present in every mode, also default; known.
- `Jax plugin configuration error ... CUDA_ERROR_NO_DEVICE`: CPU node; jax falls back to CPU.
