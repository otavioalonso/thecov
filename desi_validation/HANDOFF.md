# Handoff: state of the DESI validation of thecov and what to do next

Start here. Branch `thecov2`. First written 2026-10-05; sections 1-5 rewritten 2026-10-12 to the current state
(the dated sections 6-9 are the history, with notes where a later result superseded them; section 9 is the newest).
Everything is reproducible from files in the repository and the NERSC outputs in
`/global/cfs/cdirs/desicollab/users/oalves/thecov_validation` (logs, ssc_*.npz, report_data_*.npz, bundles) and
`~/thecov_desi/<sim>/` (pair counts, covariances, kernel caches).

Workflow from 2026-10-12: the data are being copied to the user's machine and the analysis continues with a local
Claude Code session (section 10: what is needed locally). NERSC remains the place for pair counts of new windows.

Reading order: `report/validation_report_tutorial.md` (the whole programme at introductory level, 2026-10-15); this file (sections 1, 3, 9); `report/ssc_t0_review.md`; `report/ssc_formalism.pdf`;
`report/window_kernel.typ` (the pair-averaged window); `paper/README.md` (the JCAP draft and its notebooks:
`paper/paper_nersc.ipynb`, figures from the original files; `paper/thecov_walkthrough.ipynb`, every term computed with
explicit thecov calls and compared with the mocks). Superseded: `report/report.tex` section 6, `report/proposal_lowrank.md`
(the low-rank term is now the SSC term).

## Publication plan (agreed 2026-10-22)

Two methods papers on mocks alone, plus a DESI collaboration supporting document that only quotes them.

- **Paper 1 — the covariance model and code (thecov).** Gaussian term with the window via pair counts and the
  smoothing kernels (BinnedHankel quadrature, a section of this paper, not its own paper); the tree-level non-Gaussian
  terms in the same framework (discreteness B/n, P/n^2; SSC; T0 snake + star); validation against the holi and
  Abacus ensembles. Pitch: an analytic P_l covariance for a realistic DESI footprint that reproduces N-body mocks to
  X per cent without tuning. The base-model paper; it comes first.
- **Paper 2 — non-Gaussian covariance from the catalogue itself.** One principle applied three times: every NG
  ingredient the tree level gets wrong is a response or moment of the galaxy field that the single data realisation
  measures itself — (i) discreteness via the thinning closed form (measured/tree 1.26, 1.23 for LRG1; section 9.4),
  (ii) SSC with the first-order response from the position-dependent power spectrum (R_0 = 1.9 vs tree; survey-mean
  response cancellation; updates 2026-10-15), (iii) the squeezed T0 via the second-order response R_2 from the
  integrated-trispectrum estimator (tree overpredicts the squeezed block by 2, section 13; the hard block is the tree
  with the measured P, 1.00 +- 0.08). Headline rule: no number from a mock ensemble enters the model; mock 173 plays
  the data and the 848-mock ensemble is the referee. Derivations (survey-mean cancellation, R_2 Legendre structure,
  the incompleteness of the isotropic separate-universe expansion) go in appendices. `ng_covariance_calibration.typ`
  is the seed of this paper.
- **DESI DR2 application** (data covariance, blinding-dependent numbers): a separate collaboration document; keep
  those numbers out of Papers 1 and 2 so they can go out on mocks alone.
- Caveats. (a) The R_2 estimator (section 14) must not hold Paper 2 back: if the prototype on mocks 173-178 takes
  long, discreteness + SSC from the catalogue is a complete story and the squeezed T0 becomes a short follow-up
  letter; decide after the first prototype. (b) Rejected: one combined paper (its second half would contradict the
  tree-level model of the first half) and one paper per term (hides that it is one principle).
- Material sorting: `report/` and `report_figures/` keep the validation record for Paper 1; the catalogue-calibration
  derivations (`report/catalogue_calibration_plan.md`, `ng_covariance_calibration.typ`, sections 12-14 here) are
  Paper 2 material.

## 1. Current conclusions (2026-10-12), in order of confidence

1. **thecov's Gaussian covariance is right for the window it is given**, and the **xi-kernel window
   (`thecov.WindowSmoothing`) is the right window.** Random-field arbiter on the real LRG1 NGC footprint (400 Gaussian
   realisations, healpix 512 x weighted n(z), 6 Mpc/h mesh, section 9.1): Var_GRF / Var_thecov of P0 = 1.004 / 1.025 /
   1.019 (xi kernel) vs 1.084 / 1.070 / 1.039 (local window m^2) at k = 0.03-0.1 / 0.1-0.2 / 0.2-0.3. Own weight and local
   mean weight agree to 0.5% once both are built as alpha rho_r w (the old own-weight variant double-counted completeness).
   QSO validates to ~1% with any window. Open: the same test with CELL=4 and for SGC (convergence).

2. **The super-sample term (SSC with the local average, thecov/ssc.py) has the right sign and shape.** Regressing each
   mock's P_hat on its realised shell density (delta ln int n(z)^2 at fixed WEIGHT, 848 clean mocks, section 9.2):
   dlnP/d delta_shell = P0 -0.06 to -0.11, P2 +0.8 to +2.5 (growing with k); tree-level prediction ~-0.25 (P0) and
   +0.5 to +2 (P2). P2 agrees; the P0 net response (a near-cancellation of beat coupling and local average) is ~3x weaker in
   the mocks. The mocks DO contain the long modes (shell-count scatter 0.8-1.1%); sections 8.2-8.3 were wrong on this.

3. **The files' normalisation is fully explained by per-mock FKP weights** (`norm_fkp.py`, section 9.3): holi randoms copy
   each mock's redshifts and NX, so w_FKP = 1/(1 + NX P0) follows each mock's n(z). With own-FKP weights the recomputed norm
   tracks the files' norm with corr 0.99-1.00, slope 1.00, residual 0.4-0.75%; with fixed FKP weights it does not.
   Consequence: norm responds to the shell density only through w_FKP (corr(delta_norm, delta_shell) 0.41 NGC, 0.25 SGC for
   LRG1; 0.98 QSO), so the delta_norm regression tests of sections 8-8.2 measured noise. thecov's
   `local_average_statistics` still assumes fixed weights (sigma(delta_norm) predicted 2.5x too large for LRG1): to fix.

4. **The Poisson non-Gaussian (discreteness, SN-NG) terms are too small for the mocks, and the window convolution was
   hiding it** (section 9.4). `DiscretenessCovariance` = Wadekar & Scoccimarro (2020) SN-NG terms (their eqs. 89-92: same
   B/n and P/n^2 terms, same window integrals I34/I22^2 = J_Smm, I24/I22^2 = J_SS, same true-shot-noise bookkeeping).
   `covariance_tools.window_convolve` (K = corr(C_G)^1/2, rows of unit L2 norm) multiplies any k-smooth term by
   (sum_j K_ij)^2 = 2.1-2.3 (LRG1, 0.005 bins) / 1.3-1.4 (QSO); with K row-normalised to unit sum it is ~the identity
   (0.86-1.0). The earlier "disc amplitude 1.13-1.2, close to 1" was that inflation. Against the local terms the mocks want
   B/n x 1.8-2.3 (LRG1) and P/n^2 x 2.2-4.1 (QSO); SSC fits to ~1 (LRG1). The mocks' residual (mocks - G - SSC) is flat
   across neighbouring bins: no near-diagonal structure for a convolution to fix. Do not use `window_convolve` (or fix its
   normalisation) and find the missing SN-NG variance: window integrals with per-mock weights, non-Poisson (one-halo)
   pairs, local approximation of P(|k1 - k2|). `sn_diagnostics.py` gathers the inputs.

5. **Tree-level T0 is right block by block except in the squeezed block** (holi and Abacus complete N-body, sections
   7.1-7.8, 9.4, 12, 13). A single amplitude fits at 0.38 +- 0.05 (LRG1 P0), 0.61 +- 0.05 (P2) (section 12), but that
   is a compromise: per k-block at dk = 0.025 (section 13) the P0-P0 diagonal, adjacent, low-low and hard (k >= 0.15)
   blocks are at amplitude 1 (hard: 1.00 +- 0.08 once the tree kernels are fed the measured nonlinear P), and only
   the squeezed block (k < 0.1 x k >= 0.1) is at 0.50 +- 0.03 (P2: 0.68; P0-P2 cross: 0.27, with the hard cross block
   ~0). PD at amplitude 1 in local form once b3 = 0 (the "not PD" of 7.1 used the Lazeyras b3 in the wrong basis).
   The squeezed block is the second-order response R_2 of the galaxy power spectrum, which the tree kernels
   overpredict by 2; it needs its own catalogue estimator (section 13).

6. **12 defective LRG holi mocks** (n(z) spike at 0.46 < z < 0.47; section 8.4) are excluded everywhere (848 LRG1 mocks).

7. Residual of the recommended model (kernel Gaussian + SSC + window-convolved disc, LRG1 NGC, walkthrough): chi2/dof 1.02
   (k_max 0.12) / 1.03 (0.3); after SSC + disc the worst whitened directions carry +25-40% (out of sample); A2 variance
   still 1.10-1.17x at k_max 0.2-0.3, constant shot-noise parameter 5-15% too large around 0.2. Item 4 changes this
   model: redo the budget with the local (or correctly normalised) SN-NG terms at their measured amplitude.

8. Still valid from the 2026-10-05 analysis: the bin-width (a + b dk) split of `report/report.tex` is contaminated
   (adjacent 0.005 bins correlate at 0.4, so a fully correlated term shows up ~2/3 as "a"); the low-rank off-diagonal
   excess of `report/proposal_lowrank.md` (sigma_P0 0.51 / 0.66%, P2 mode 0.94 / 1.22%, LRG1 NGC / SGC) is the SSC term.

> **Update 2026-10-14 (`report/grf_randoms_results.md`, `report/full_budget_plan.md`):** the window question is closed
> with the holes resolved exactly: Gaussian fields evaluated at the DESI randoms themselves (no painting) give
> Var_GRF / Var_thecov = 0.97-1.01 for the xi-kernel window and 0.65-0.78 (true P) / 1.077 (fields' mean) for m^2,
> LRG1 both caps, and the same for LRG2, ELG2 NGC, QSO NGC. PS/SS terms at the files' shot noise, P2/P4 and linear
> RSD at the true line of sight agree within the 5-8 % the field counts allow; thecov's SS term is checked to a few %
> at S/P up to 5. Item 2's "converging the kernel basis: chi2/n does not move" is confounded by pair-count sampling:
> at matched sampling convergence raised chi2/n by 0.02 (the unconverged kernel over-diluted by accident). Queued:
> the dilution-by-thinning decomposition of the discreteness terms (`thinning_cov.py`) and the clean P/n^2 field test.
>
> **2026-10-15:** the discreteness terms are now measured from the catalogue alone, in closed form, no thinnings, no templates
> (`thinning_closed_form.py`, `grf_randoms_results.md` section 7): LRG1 B/n = 2x tree level, P/n^2 ~ 1.5x; QSO P/n^2 = 2.6x. Item 4 of
> section 1 is thereby answered: thecov's discreteness module is right at tree level; the mocks' galaxies have a larger
> collapsed bispectrum (one-halo) and QSO non-Poisson pairs. The response estimator (`response_ibisp.py`, integrated
> bispectrum in Fourier space, normalised by int P W) is validated on the footprint with a Poisson-sampled lognormal catalogue
> (R = 2 recovered to 5 %): mock LRG1 responses per unit galaxy long mode are 1.8-2.0 (P0) and 3.0-3.2 (P2) at k = 0.05-0.3,
> 1.3-1.5x tree level. Input for step 3.B (measured responses in the SSC).

## 2. What was tried and should not be repeated

- Tuning the fill scale (nside 512 -> 64): chi2/n improves for the wrong reason (item 4).
- Converging the kernel basis to fix chi2/n: it fixes I_k (6.6e-2 -> 1e-5) but chi2/n does not move,
  because the residual is not the window.
- Explaining the residual by redshift-space anisotropy of the kernel: it cannot explain a gap with the
  isotropic GRF test, and the residual is not Gaussian-level anyway.
- The hole-fraction statistic predicting less than the fill window changes thecov: resolved, the effect
  is on the diagonal only to the extent the hole density varies on scales >~ 1/dk; it does (Galactic).
- Weight resolution (7' maps vs 32-neighbour mean): the GRF showed no difference (0.2%).
- Regressing P_hat or the coherent P0 mode on the files' delta_norm to test the local average: with per-mock FKP weights
  norm barely follows the shell density (corr 0.25-0.41 for LRG1). Regress on the shell density itself (section 9.2).
- `window_convolve` for smooth terms (disc, T0): it inflates them by (sum_j K_ij)^2; any agreement it produces is
  that factor (section 9.4).
- Tree-level T0 in any form (full, LL only, response split, collapsed): rejected on holi and Abacus (7.1-7.8, 9.4).
- An "own weight" window built as NX x WEIGHT: NX already contains the completeness (double counting, +10%).
- Comparing only the leading eigenvectors of the SSC term and of the mocks' excess (the walkthrough's old "shapes"
  panel): SSC's first vector is P2-dominated, the mocks' worst Gaussian direction is a P0 amplitude; compare subspaces
  or projections.

## 3. Next steps, in order (2026-10-12)

A. **The missing SN-NG variance (conclusion 4).** Run `desi_validation/sn_diagnostics.py` (one debug job per bin and cap,
   command in section 9.5) or the equivalent locally. It saves: J_Smm, J_SS from the randoms (global <w^2>/<w>, as thecov)
   and from the data (local <w^2>/<w>); weight moments; data 1 + xibar(<r) at 0.25-16 Mpc/h (non-Poisson pairs); every
   mock's multipoles to k = 0.6 with thecov's local G, SSC, B/n, P/n^2 on those bins (the SN-NG terms grow with k: the
   strongest amplitude test). Decide between (i) window integrals, (ii) non-Poisson sampling, (iii) the local
   approximation of the P(|k1 - k2|) term; then fix `DiscretenessCovariance` and drop or renormalise `window_convolve`
   (rows of K summing to 1).

B. **Local average with per-mock FKP weights** (conclusion 3): `SuperSampleCovariance.local_average_statistics` and the LA
   term should use the response of norm with w_FKP = 1/(1 + n P0) recomputed per realisation (delta(n w_FKP) =
   w_FKP^2 delta n). Check the predicted sigma(delta_norm) against the mocks (0.39-0.43% NGC / 0.75% SGC for LRG1).

C. **The P0 SSC response** (conclusion 2): measured slope vs shell density ~3x weaker than tree level. Put the shell-density
   regression in the walkthrough next to thecov's fixed-weight prediction (beat coupling and local average separately).

D. **Random-field convergence**: `bash desi_validation/run_grf_tasks.sh LRG1 NGC 3 80 4` (cell 4) and SGC, before the paper
   claims the 2% agreement of the xi kernel.

E. **Paper**: correct the text on long modes and normalisation (per-mock FKP), the SN-NG result, replace the walkthrough's
   "shapes" panel by a subspace / projection comparison, then the budget with the corrected SN-NG term.

F. Later: `Config.nw_modes = ('kernel',)` as the pipeline default (the holi-kcore2 dumps already use it); all tracers.

## 4. Tools

`desi_validation/`:
- `dump_report_data.py`: runs the pipeline, dumps mock vectors + covariances per window mode and binning.
- `compare_kernel.py`, `rank1_excess.py`: chi2/n, variance ratios, the off-diagonal low-rank fit, parameter ratios.
- `ssc_check.py`: SSC, discreteness and T0 for the holi / Abacus dumps (`ssc_<label>.npz`: every term, local and
  window-convolved, with the parameters); `ssc_plots.py`; `ssc_dev/` (designs, local rebuilds, T0 diagnostics).
- `norm_fkp.py`: the files' norm from the data catalogues per mock, own vs fixed FKP weights and randoms.
- `nz_scatter.py`: n(z) of all mocks (defective-mock detection, shell-density scatter); `hole_fraction.py`.
- `run_gaussian_footprint.py` + `run_grf_tasks.sh`: Gaussian random fields on the real footprint (the window arbiter).
- `sn_diagnostics.py`: inputs for the SN-NG problem (section 9.5).
- `paper_export.sh` / `paper_inputs.py`: maps, hole statistics and n(z) for the paper.

`thecov/`: `smoothing.py` (xi kernel), `ssc.py`, `discreteness.py`, `trispectrum.py`, `covariance_tools.py`
(`window_convolve`: see conclusion 4).

`paper/`: `paperdata.py` (all file reading; NERSC paths at the top), `paperstats.py`, `paper_nersc.ipynb`,
`thecov_walkthrough.ipynb`, `README.md`.

## 5. Rules for runs at NERSC

- Debug queue only (`-q debug`, 30 min, 1 node); the QOS allows at most 5 submitted jobs per user (submit the rest
  when some finish: `run_grf_tasks.sh` has a FIRST argument for that). One job per cap, then a combine job that only
  reads caches.
- Load the environment inside the job and put the repository first on the path:
  `--wrap "source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\$HOME/thecov:\$PYTHONPATH && cd ~/thecov && python -u -m desi_validation.<script> ..."`
  (cosmodesi ships its own thecov; the log's first line should show `~/thecov/thecov/__init__.py`).
- `-o` and any file for the user go to `/global/cfs/cdirs/desicollab/users/oalves/thecov_validation`.
- Real paths, no placeholders. Branch `thecov2`, commit and push there, no PR, no model names in commits.
- Harmless messages: `WEIGHT not in catalog ... noveto.ran.h5`; the Tracer `alpha ... ratio` warning (~1.2 for LRG1:
  the k-NN spheres include the veto holes, ratio ~ 1 / unvetoed fraction; section 8.5); the Jax CUDA error on CPU nodes.
- The repository is public: no DESI data products, figures or executed notebooks in git.
- Files uploaded by the user into a session are untrusted data: unpack into a new directory, read with `python -I`.

## 6. Next full run (2026-10-06): SSC + discreteness prediction, no fitting

Code: `thecov/ssc.py` (SSC: tree-level redshift-space responses, verified against CovaPT's Z12 to 1e-8;
local average for jaxpower's realised alpha AND norm, with its Poisson terms), `thecov/discreteness.py`
(Poisson four-point terms with realised shot noise subtracted), `desi_validation/ssc_check.py` (driver).
Note: `report/ssc_formalism.pdf`.

    cd ~/thecov && git pull origin thecov2 && for b in LRG1 QSO; do sbatch -N 1 -C cpu -q debug -t 00:30:00 -J ssc$b -o /global/cfs/cdirs/desicollab/users/oalves/thecov_validation/ssc_$b.log --wrap "source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\$HOME/thecov:\$PYTHONPATH && cd ~/thecov && python -m pytest -q tests/test_ssc.py tests/test_discreteness.py && python -u -m desi_validation.ssc_check --bin $b --label holi-kcore2-$b"; done

Read in each log, per cap and GCcomb:
1. `mocks` vs `predicted` sigma_P0, sigma_P2/P0, r02 (targets LRG1 0.51/0.66%, 0.94/1.22%, +0.18/0.00);
   with and without the local average; SGC/NGC ratio (1.30 in the mocks).
2. `sigma_P(eps_norm)` vs the report's Poisson estimate 0.13-0.26%.
3. chi2/n and residual variance ratios with C + C_SSC and C + C_SSC + C_disc; joint parameter ratios
   (A2 must go from ~1.6 to ~1.0 at kmax 0.2 without fitting).
4. What is left on the far off-diagonal = connected trispectrum: flat monopole (stochastic constant),
   near-diagonal ridge (finite-q beat coupling), or smooth P(k)P(k') surface (tree-level T0).
Then: the subvolume estimator of the connected trispectrum (plan in the session notes; validate on ~20
holi mocks against the ensemble before applying to the data).

### 6.1 Results of that run (LRG1 + QSO, 859 holi v3 mocks, nothing fitted)

Covariance = kernel Gaussian + SSC (LA) + discreteness 4-pt:

| | sigma_P0 pred/mocks | sigma_P2/P0 pred/mocks | r02 pred/mocks | chi2/n Gauss -> total |
|---|---|---|---|---|
| LRG1 NGC | 0.43 / 0.51 % | 1.10 / 0.94 % | +0.35 / +0.17 | 1.0625 -> 1.0286 |
| LRG1 SGC | 0.72 / 0.66 % | 1.85 / 1.22 % | +0.56 / 0.00 | 1.0705 -> 1.0456 |
| LRG1 GCcomb | 0.37 / 0.41 % | 0.95 / 0.74 % | +0.43 / +0.16 | 1.0652 -> 1.0326 |
| QSO NGC | 0.31 / 0.30 % | 0.46 / 0.46 % | +0.49 / -0.67 | 1.0068 -> 0.9945 |
| QSO SGC | 0.65 / 0.58 % | 0.88 / 0.90 % | +0.49 / -0.49 | 1.0000 -> 0.9832 |

- SSC with the LA Poisson self-calibration term ALONE is not positive definite for LRG1 (min eig of
  C^-1/2 M C^-1/2 = -0.05 NGC, -0.015 SGC); it must always be used with the discreteness 4-pt (then 0.92 / 0.82).
  With vs without the LA Poisson term (both + disc 4-pt) differ by ~0.05-0.1% in sigma_P0, within the noise.
- LRG1 joint parameter variance ratios at kmax 0.3 (A, A2, alpha, SN): NGC [1.25 2.11 1.09 1.22] -> [1.10 1.07 0.93 1.01].
- Remaining: (i) l=2 high-k variance overpredicted (ratio 0.80-0.87 at 0.3<k<0.4) and sigma_P2/P0 too large
  for LRG: tree-level redshift-space responses and B/P in the 4-pt have no FoG damping; (ii) l=0 residual
  1.05-1.1 at high k and chi2/n 1.03-1.05 for LRG: the connected T0; (iii) r02 sign for QSO (mocks
  anticorrelate P0 and P2/P0; tree level gives +) -- candidate: velocity-dispersion (FoG) fluctuations
  that raise P0 and lower P2 together, a non-perturbative T0 piece the template fit should absorb.
- Bug fixed after this run (84f463c): the npz save overwrote the kernel *.smoothing.npz caches of LRG1 and QSO
  (both caps); delete them (they now hold SSC output); the dilution now comes from the cached kernel covariance's I_k.

## 7. Tree-level T0 (`thecov/trispectrum.py`, 2026-10-06)

- `TrispectrumCovariance(cov, p_lin, bias, f, n_workers=...)`: Kobayashi's (PowerSpecCovFFT) decomposition,
  snake (T2211) + star (T3111), SCF99 redshift-space Z1..Z3 from the exact mapping, Galileon bias basis
  (`Bias`; `galileon_bias(b1, b2, bs2, b3=...)` converts from the ssc (b2, bs2) basis), window factor
  J4 = int m^4 / norm^2. Components: snake, star, star_b3 (= d star / d b3) for templates.
- Validation: F3 angle average = the P13 kernel (1e-10); Z2 = ssc._Z2; l1 = l2 blocks equal PowerSpecCovFFT's
  to 4 digits for every bias parameter (b1, b2, bG2, b3, bG3, bdG2, bGamma3 switched on one at a time:
  scratch xcheck_t0.py); all blocks against an independent (mu1, mu2, phi) quadrature (tests).
- PowerSpecCovFFT is wrong for l1 != l2 where its k1 <-> k2 swapped terms matter (k1 >= k2): e.g. C_02 at
  k1 = k2 is 0.77x (snake) / 1.24x (star) of the correct value; at k1 < k2 it agrees to 1e-3 (the swapped
  term is P(k2)^2-suppressed there). Same issue as found earlier for its shot-noise terms.
- Cost ~0.13 s per pair of k nodes and process (Gram-matrix kernels), spawn-based process pool.
- `CovarianceTemplates(C_fixed).update(components, prefix=...)`: C(A) = C_fixed + sum A_i C_i, save/load;
  the hook for the subvolume amplitude fit (templates: T0 snake, star, star_b3, disc B, disc P, SSC).
- ssc_check now adds T0 (b3 from Lazeyras b3(b1)) and reports 'SSC (LA) + disc 4-pt + T0' (and b3 = 0);
  ssc_plots shows it.

### 7.1 First T0 run against the mocks (LRG1 + QSO, 2026-10-06) -- tree-level T0 FAILS for LRG

- QSO: T0 is negligible (diag ~0.1% of C); nothing changes (chi2/n 0.9945 -> 0.9947).
- LRG1: C + C_SSC + C_disc + C_T0 is NOT positive definite (min whitened eig -0.96 NGC, -1.09 SGC).
  Per-block amplitude of T0 fitted to the mocks' off-diagonal excess over Gauss+SSC+disc (|i-j|>1):
  block 00: 0.35-0.42 (NGC), -0.1 to -0.24 (SGC); blocks 02, 22: -0.1 to -0.65 (i.e. none wanted);
  04, 24, 44: 0-0.5. Squeezed coupling (k_i ~ 0.02-0.05 x k_j ~ 0.2-0.3): T0 predicts corr 0.10-0.25, mocks
  0.00-0.09 (+-0.011). Figure: report/t0_offdiag_LRG1_NGC.png.
- The implementation is not the problem (matches PowerSpecCovFFT; squeezed limit cancels the (k/q)^2 IR pieces
  as it should, leaving T -> R2 P(q)^2 P(k)). Tested locally (ssc_dev/t0_diagnostics, CAMB P_lin):
  BAO damping (no-wiggle / IR-resummed P_lin) removes the oscillations but not the amplitude; Gaussian FoG
  damping of the external legs (sigma_v = 2.1 Mpc/h from the mocks' P2/P0) only moves min eig -0.96 -> -0.64;
  T0 restricted to the 00 block is still not PD (its low-k diagonal is negative: star with Lazeyras b3 = -4.6,
  which is in a different basis than the Galileon b3 -- the conversion is not done).
- Reading: tree-level redshift-space T0 is far outside its validity at k ~ 0.2-0.3 for LRG (as for the tree-level
  RSD bispectrum, valid to k ~ 0.1), especially the quadrupole/f-dependent terms. Not usable as a fixed term, and its
  shapes are not good templates for 02/22. Candidates: response-based T0 with measured (nonlinear, FoG-damped)
  multipoles and their derivatives (Barreira & Schmidt style), shared with the SSC response fix; or generic smooth
  templates fitted to subvolumes. Current best remains Gauss + SSC(LA) + disc 4-pt.
- `sigma_fog` option added to TrispectrumCovariance (Gaussian damping of the four external fields).

### 7.2 BAO + fingers-of-God damping of SSC and discreteness (option 3, 2026-10-06)

- `thecov/power.py`: `no_wiggle` (EH98 shape x smoothed ratio), `ir_damped`, `fog`, `Dressed` (P(|v|) exp(-(v_z s)^2)).
- `SuperSampleCovariance(p_dressed=...)`: responses from the dressed power at each k node (`response_multipoles`,
  same squeezed limit, reduces to a P + c dP/dlnk when undamped; nu^4 response stays 0). `DiscretenessCovariance(sigma_fog=)`.
- Local rebuild from the NERSC npz (ssc_dev/t0_diagnostics/sd_damped.py; rebuild reproduces chi2 1.0285 vs 1.0286):
  LRG1, sigma_v fitted to the mock P2/P0 with Gaussian damping (NGC 2.09, SGC 2.75 Mpc/h), BAO Sigma = 6:
    NGC sigma_P2/P0 1.10 -> 0.95 % (mocks 0.94), r02 0.35 -> 0.22 (0.17), l=2 var ratio k 0.2-0.3 0.868 -> 0.930,
        sigma_P0 0.43 -> 0.41 (0.51), chi2/n 1.0286 -> 1.0317, kmax 0.3 ratios [1.11 1.06 0.96 0.99];
    SGC sigma_P2/P0 1.85 -> 1.46 % (1.22), r02 0.56 -> 0.33 (0.00), l=2 var ratio 0.800 -> 0.922, chi2/n 1.0456 -> 1.0514.
  BAO damping alone changes ~nothing; sigma_v = 4 overcorrects (sigma_P2/P0 0.65 %). The remaining chi2 excess is the
  monopole coupling at high k (the part T0 should supply; tree level fails, see 7.1).
- ssc_check now applies both by default (sigma_v fitted automatically; --no-damping for the old tree level).

### 7.3 Response-based T0 and the template likelihood (2026-10-06)

> **Note 2026-10-12:** the "disc ~2x" found here (before the window convolution) was the real signal; the window
> convolution of 7.4 hid it by inflating smooth terms (section 9.4).

- Normalisation (the main finding): the mock P0 at 0.02 < k < 0.08 is A = 0.73 (NGC) / 0.70 (SGC) times
  dilution x Kaiser-FoG(b1 = 2.2 from P2/P0) x P_lin (cosmoprimo DESI). Every non-Gaussian term built from b1 and
  P_lin was therefore normalised to a galaxy power ~1.4x too high (T0 ~ A^-3 too high). `ssc_check` now fits A and
  builds SSC, discreteness and T0 from A P_lin (`--no-normalize` for the old behaviour). With it the free SSC
  amplitude in the template fit is 1.00 (vs 0.4-0.5 without): SSC is predicted from first principles.
- `response_split(C_T0, k, k_split, C_long)` (thecov.trispectrum): LL / LH (squeezed) / HH blocks and the
  completion C_HL C_LL^-1 C_LH (variance the hard modes inherit from the long-mode power they respond to; O(P^4),
  makes the squeezed couplings positive-definite by construction).
- `CovarianceTemplates.loglike / fit`: Wishart -2 ln L and ML amplitudes (the sub-volume fitter).
- LRG1 NGC, local (ssc_dev/t0_diagnostics/tpl_fit.py, k_split 0.06), -2 dlnL vs Gaussian / chi2/n:
    SSC + disc fixed            -2691  1.035
    + tree T0 (all pieces) =1   -1325  1.065   (rejected)
    fit SSC, disc               -2954  1.031   SSC 1.01, disc 1.96
    fit SSC, disc, T0           -2969  1.029   T0 -0.12
    all pieces free             -3148  1.020   SSC 1.17, disc 1.7, T0_LL -0.4, T0_LH -0.04, T0_HH -0.3, completion 2.7
  i.e. the mocks reject the tree-level T0 shapes (even normalised and FoG-damped) but want ~2x the discreteness
  shapes (nonlinear B, non-Poisson pairs) and the response completion (hard modes inheriting the long-mode power
  variance). ssc_check prints this table per cap and GCcomb (`template_table`).
- Next: the NERSC run with normalisation (both tracers), then the sub-volume version of the same fit with the
  templates {SSC (fixed or 1 amp), disc, completion}.

### 7.4 Final model (implemented; validated locally on LRG1 + QSO, both caps, 2026-10-06)

> **Superseded in part (2026-10-12):** item (iii) is wrong. `window_convolve` multiplies k-smooth terms by
> (sum_j K_ij)^2 = 2.1-2.3 (LRG1) / 1.3-1.4 (QSO); its likelihood gain is that factor, not window physics (section 9.4).

Recommended, nothing fitted: Gaussian (kernel) + SSC (LA) + discreteness 4-pt, with
(i) the linear galaxy power normalised to the mocks (A = 0.73 / 0.70 LRG1, 0.81 / 0.67 QSO NGC / SGC),
(ii) BAO (Sigma 6) and FoG damping (sigma_v fitted to P2/P0: 2.1 / 2.7 LRG1, 3.3 / 4.0 QSO),
(iii) the discreteness terms window-convolved: `thecov.covariance_tools.window_convolve` (mixing kernel
     K = corr(C_G)^1/2, exact for the Gaussian term by construction; removes the local approximation's spurious
     bin-to-bin structure near the diagonal).
Local rebuild from the earlier NERSC bundle (ssc_dev/local_rebuild.py), -2 dlnL vs Gaussian / chi2/n:
    LRG1 NGC  NERSC run -2945 / 1.029  ->  -3320 / 1.023   (sigma_P0 0.52 vs 0.51 %, sigma_P2/P0 0.80 vs 0.94 %, r02 0.16 vs 0.17)
    LRG1 SGC            -1560 / 1.046  ->  -2594 / 1.034   (0.77 vs 0.66 %, 1.18 vs 1.22 %, 0.12 vs 0.00)
    QSO  NGC             -698 / 0.995  ->   -676 / 0.995
    QSO  SGC             -692 / 0.983  ->   -702 / 0.985
Response-based T0 (thecov.trispectrum): tree for k < k_split (LL), squeezed couplings (LH) + completion
(X^T A^-1 X + X^T A^-1 B0 + B0^T A^-1 X: PSD by construction), IR-safe collapsed term
(`collapsed_multipoles`: T = 2 [R(k1, p) P]^2 P_L(p) for p = |k1 -+ k2| < k_split; the masked tree is not IR safe),
hard block dropped. On LRG1 NGC every T0 piece makes the likelihood worse once the discreteness term is
window-convolved (LL: +370 to +450; LL + LH + compl: +1100; collapsed: not PD at -0.03 whitened eigenvalue);
reported as diagnostic only. The Lazeyras b3 is not in the Galileon basis; default b3 = 0.
Template fits (--fit-templates, diagnostic) earlier wanted disc ~2x before the window convolution.
Next: the NERSC run of this ssc_check (saves every part), then the sub-volume pipeline if needed.

### 7.5 NERSC run of the final model (2026-10-06, log ssc_final_LRG1_QSO.log; figures report/final_run/)

> **Note 2026-10-12:** the disc amplitudes ~1 quoted here are for the inflated (window-convolved) term; against the
> local term the mocks want 1.1-2.5 (section 9.4). The chi2/n and parameter ratios of this model stand as measured.

Recommended C_nongauss = SSC (LA) + window-convolved discreteness, normalised + damped, nothing fitted. Reproduces
the local rebuild exactly (LRG1 NGC -3321 vs -3320). chi2/n (Gaussian -> model), kmax 0.3 joint variance ratios
(A, A2, alpha, SN):
    LRG1 NGC     1.0625 -> 1.0231   [1.25 2.11 1.09 1.22] -> [1.04 1.18 0.97 1.01]
    LRG1 SGC     1.0705 -> 1.0342   [1.60 1.80 1.46 1.43] -> [1.07 1.15 1.05 1.03]
    LRG1 GCcomb  1.0652 -> 1.0262   [1.31 1.95 1.15 1.26] -> [1.04 1.12 0.98 1.00]
    QSO  NGC     1.0068 -> 0.9948   [1.00 1.03 0.96 1.41] -> [0.95 0.97 0.92 1.20]
    QSO  SGC     1.0000 -> 0.9854   [1.08 0.99 1.05 1.39] -> [1.00 0.94 0.97 1.09]
    QSO  GCcomb  1.0027 -> 0.9899   [1.07 1.00 1.07 1.43] -> [1.02 0.95 1.02 1.18]
Coherent amplitudes LRG1 GCcomb: sigma_P0 0.42 % (mocks 0.41), sigma_P2/P0 0.67 % (0.74), r02 0.14 (0.16).
Largest whitened mock eigenvalue ~2.3 (Wishart edge 2.1): no remaining rank-1 excess; diagonal variance ratios
within ~2-5 % at all k and l. Template fits (diagnostic) gain little: LRG1 NGC SSC 1.23 / disc 1.22 for -22 in
-2 lnL, SGC 1.36 / 0.90 for -12, GCcomb 1.18 / 1.13 for -9; QSO amplitudes are degenerate.
Response T0 (diagnostic): its pieces get ML amplitudes 0.02-0.3; with the windowed collapsed term the model has a
whitened eigenvalue of 34 (badly wrong) -> not used. Open: residual chi2/n 1.02-1.03 for LRG (diffuse, not a
low-rank term), the QSO r02 sign, the QSO shot-noise parameter variance (1.2 at kmax 0.3, NGC).

### 7.6 Abacus T0 test (prepared 2026-10-07)

`bash desi_validation/run_abacus_t0_test.sh [altmtl|complete]`: kernel Gaussian dumps for the 25 AbacusSummit DR2 mocks
(LRG1, same options as holi-kcore2), then ssc_check (--cs-version abacus-2ndgen-dr2-<v> --mock 0 --fit-templates) and
the bundle. Question: is the tree-level T0 amplitude (template fit with SSC and disc free; Fisher errors printed)
~1 in N-body (Abacus) while ~0.07 +- 0.02 in holi (NGC)? Forecast sigma(A_T0) ~ 0.07-0.09 with 25 mocks, both caps;
control: holi with 25 mocks gives 0.17 +- 0.08. Caveats: SSC differs (2 Gpc/h periodic boxes; left free); caps may be
correlated (compare per cap); model mean from 25 mocks.

### 7.7 Abacus complete T0 test: tree-level T0 rejected by N-body too (2026-10-07)

25 AbacusSummit DR2 complete mocks, LRG1 (log abacus_t0_complete_ssc.log). Template fit (Wishart ML, Fisher errors),
SSC and disc free: T0 tree amplitude NGC 0.09 +- 0.11, SGC 0.21 +- 0.06, combined 0.18 +- 0.05 (holi NGC: 0.07 +- 0.02);
T0 response 0.05 +- 0.06 / 0.04 +- 0.07. T0 = 1 is excluded at > 8 sigma in each cap; with T0 fixed at 1 the model is
not PD. So the rejection is not a holi artefact: tree-level redshift-space T0 overpredicts the connected coupling at
k ~ 0.1-0.3 also in N-body; at most ~20 % of it is present beyond SSC + window-convolved discreteness.
Recommended model on Abacus: chi2/n 1.102 -> 1.065 (NGC), 1.096 -> 1.069 (SGC); -2 dlnL -91 / -40 (unfitted).
Abacus caveats: the Gaussian model is worse than for holi (x4 bins chi2/n 1.14-1.22 in the dump; model mean from 25
mocks); NGC and SGC share box modes (same-bin cap correlation +0.15 at k < 0.12, ~2-3 sigma), so GCcomb (chi2 1.12)
is not valid as an independent combination; SSC amplitude unconstrained (+-0.6). The altmtl run would only test the
fiber-assignment interplay (lower priority).

### 7.8 k_split scan of the tree (low k) + response (high k) T0 (holi LRG1, 2026-10-07)

ssc_dev/t0_diagnostics/ksplit_scan.py (bundle npz; unfitted, -2 dlnL vs Gaussian; recommended SSC + disc: -3321 NGC, -2594 SGC):
  tree LL only (all k < k_split): k_split 0.04 -3283 / 0.06 -2957 / 0.08 -1777 / 0.10 not PD / 0.12 not PD (NGC; SGC alike)
  LL + squeezed + completion: always worse (-381 to -2097 NGC), chi2/n can drop (1.009) while ln L worsens: too much variance
  collapsed (response) term: worse at 0.04, not PD (window-convolved) for k_split >= 0.06
=> the tree-level T0 fails already inside its nominal regime (k < 0.1): its low-k block is not compatible with
   Gaussian + SSC + disc. Likely cause: for the thin LRG1 shell the low k bins are near the window scale, where the local
   (narrow-window) T0 is invalid and the long-mode coupling is what SSC + local average already describe (with the LA
   cancellation that the in-survey squeezed T0 lacks). The response pieces at high k come out 4-10x too large
   (template amplitudes 0.02-0.25 on holi and Abacus): tree-level responses with Gaussian FoG overestimate the galaxy
   redshift-space responses at k ~ 0.1-0.3. A proper version needs (i) a window-convolved (non-local) low-k T0 and
   (ii) nonlinear responses, e.g. measured from sub-volumes (position-dependent power spectrum).

### 7.9 What the recommended model still misses (residual analysis, 2026-10-07)

Whitened residuals of the mocks under C + C_nongauss (holi LRG1/QSO, 859 mocks; Abacus complete LRG1, 25):
- QSO: consistent with noise after a uniform rescale (chi2/n 0.985-0.995, i.e. 0.5-1.5 % too much variance); one
  6.8 sigma eigenvalue in NGC. Open: r02 sign, shot-noise parameter variance 1.2 (NGC, kmax 0.3).
- LRG1: every single multipole and every pair is at chi2/n 0.99-1.005, but all three together give 1.023 / 1.034:
  the whole excess is var(l=4 | l=0, l=2) = 1.07 (NGC) / 1.11 (SGC), peaking at 0.1 < k < 0.2 (1.08-1.09), already
  present in the Gaussian part alone and unchanged by SSC / disc; the top whitened eigenvalues (2.2-3.0 vs noise
  2.05 +- 0.02) are l=4-dominated, bin-to-bin oscillating directions. Abacus COMPLETE shows the same (1.16 / 1.12
  +- 0.04) -> not fiber assignment. Not P6+ truncation of the model (P6/P0 < 0.2 % for the fitted damping; box test
  changes var(l4|l0,l2) by < 0.1 %). Not seen for QSO (shot-noise dominated, where window/LOS anisotropy couplings
  matter little). Candidates: the anisotropic (l-mixing) structure of the Gaussian term at nP > 1 (window
  multipoles / kernel smoothing anisotropy), the estimator's line-of-sight convention vs thecov's, non-Gaussian
  l=4 couplings. Parameter-level: quadrupole-amplitude variance still 1.15-1.18x the model at kmax 0.3 (LRG1).
- T0 is not needed (sec. 7.7-7.8).

## 8. Review of the SSC and T0 implementations (2026-10-07): `report/ssc_t0_review.md`

> **Superseded in part (2026-10-12):** the delta_norm-based tests of the local average in this section and in 8.2 are
> not meaningful for holi: per-mock FKP weights decouple norm from the shell density (section 9.3). The convention
> (8.1, data x randoms) stands.

Read it before the next `ssc_check` run. In short: the algebra of `thecov/ssc.py`, `discreteness.py` and `trispectrum.py`
is verified (independent kernel paths, IR limit, from-scratch T0 assembly; `ssc_dev/review_checks.py`), but the
local-average term as implemented (norm = alpha sum D R, `norm_kind='data-randoms'`) predicts a NEGATIVE regression
slope of P_hat_0 on delta_norm (-0.2 to -0.4 P_0, net response R_0 - 2 g_0 = -1.1 P_0 for LRG1) and |corr| ~ 1 between
the coherent P_0 mode and delta_norm, while the report measured +0.1 to +0.25 P_0 and r = +0.14 / +0.17; the measured
sigma(delta_norm) = 0.43 % also bounds the isotropic long mode to sigma_0 <~ 0.08 % under that convention, i.e. an
isotropic SSC of <~ 0.1 % in P_0, not 0.43 %. A normalisation that realises alpha only (`norm_kind='alpha'`) reproduces
the slope, the correlation and sigma(delta_norm) at tree level. `ssc_check` now prints, per cap, the mocks' sigma(delta_norm),
slopes and correlations next to the predictions of all three conventions (`--norm-kind` selects the one used for C_SSC):
read that block first in the next run and settle the convention in jaxpower. Then: measured responses (sub-volumes),
the sub-sampling test that separates T0 from the discreteness terms, and the Gaussian-field test of var(l=4 | l=0, 2).

### 8.1 How jaxpower computes norm (read from github.com/adematti/jax-power, jaxpower/mesh2.py, 2026-10-07)

`compute_fkp2_normalization(fkp, cellsize=10, split=None)`: with `split=None` ("the pypower normalization")
norm = alpha x sum_cells D_c R_c / V_c (CIC on 10 Mpc/h cells, data x randoms, alpha = sum w_d / sum w_r) -- thecov's
`norm_kind='data-randoms'`; with `split=<seed>` norm = alpha x sum_cells R1_c R2_c / V_c over two disjoint random
subsamples (randoms only: alpha realised once; but the mock randoms take their redshifts from the mock's data, so the
randoms' integral still carries the realised radial n(z)). num_shotnoise = sum_d w_d^2 + alpha^2 sum_r w_r^2. Which call
the DESI spectra pipeline (clustering_statistics, not public) made is settled at NERSC by three scripts, in this order:

    # 1. the code itself (no job, ~1 min): prints the 'norm' / 'split' functions of jaxpower and clustering_statistics
    #    and the attributes of one spectrum file
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && cd ~/thecov && git pull origin thecov2 && \
      PYTHONPATH=$HOME/thecov:$PYTHONPATH python -m desi_validation.inspect_norm_code --bin LRG1 --region NGC --mock 173 \
      > /global/cfs/cdirs/desicollab/users/oalves/thecov_validation/norm_code.txt 2>&1
    # 2. the dump alone (seconds): slope of ln norm on ln num_shotnoise over the 859 mocks (1: alpha only, ~2: data x randoms
    #    or randoms^2), residual rms, and the P_hat regressions on delta_norm and delta_nsn
    PYTHONPATH=$HOME/thecov:$PYTHONPATH python -m desi_validation.norm_convention --bin LRG1 --label holi-kcore2-LRG1 \
      > /global/cfs/cdirs/desicollab/users/oalves/thecov_validation/norm_convention_LRG1.txt
    PYTHONPATH=$HOME/thecov:$PYTHONPATH python -m desi_validation.norm_convention --bin QSO --label holi-kcore2-QSO \
      > /global/cfs/cdirs/desicollab/users/oalves/thecov_validation/norm_convention_QSO.txt
    # 3. the catalogues of 8 mocks (debug queue, ~20 min): DR, RRsplit, NX-based candidates vs the files' norm, mock by mock
    cd ~/thecov && sbatch -N 1 -C cpu -q debug -t 00:30:00 -J normcat -o /global/cfs/cdirs/desicollab/users/oalves/thecov_validation/norm_catalogs_LRG1.log \
      --wrap "source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\$HOME/thecov:\$PYTHONPATH && cd ~/thecov && python -u -m desi_validation.check_norm_catalogs --bin LRG1 --label holi-kcore2-LRG1 --mocks 0 1 2 3 4 5 6 7"
    # 4. then the SSC with the convention found (and the predicted vs measured norm statistics in the log)
    cd ~/thecov && sbatch -N 1 -C cpu -q debug -t 00:30:00 -J sscLRG1 -o /global/cfs/cdirs/desicollab/users/oalves/thecov_validation/ssc_norm_LRG1.log \
      --wrap "source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\$HOME/thecov:\$PYTHONPATH && cd ~/thecov && python -u -m desi_validation.ssc_check --bin LRG1 --label holi-kcore2-LRG1 --no-t0 --norm-kind data-randoms"

Read: (1) the `split` argument of the pipeline's call; (2) the slope (1 vs 2) and the residual; (3) which candidate's
ratio to the file's norm has rms << 0.4 %. If it is DR, the tree-level net response is wrong in sign for these galaxies
and the responses must be measured (review sec. 3.B); if it is RRsplit / NXr, rerun ssc_check with `--norm-kind alpha`.

**Confirmed (2026-10-07, from the pipeline code): split=None, the pypower convention, norm = alpha sum_c D_c R_c / V_c.**
So `norm_kind='data-randoms'` (the default) is the right bookkeeping, and with randoms that take the mock's own redshifts
alpha n_r(x) is the realised radial profile n_rad(z): norm ∝ ∫ n_rad^2, delta_norm = 2 D^W for every long mode, and the
net monopole response is R - 2 g = R_phys, the local-mean-referenced response, = -1.1 P_0 for b1 = 2.2 (bias response
2 (b1 + b2 - b1^2) / b1 = -2.15 dominates). The measured slope +0.1 to +0.25 P_0 therefore cannot be a tree-level SSC +
LA effect (it would need R ~ 8-9 P_0, b2 ~ 5). What the scripts now decide: whether the stored P_hat are divided by the
per-mock norm at all (`norm_convention.py`: the P_0 amplitude-mode slope on delta_norm is ~ -0.3 if they are, ~ +0.7 for a
fixed normalisation; check also in lsstypes how `.value()` applies `norm`), and the residual of ln norm on ln num_shotnoise
(the weighting difference between the m^2-weighted norm and the w^2-weighted count plus Poisson, expected 0.1-0.2 %).
Whatever the answer, the P_0 amplitude mode of the mocks (0.5 %, r = 0.14 with delta_norm) is not the isotropic SSC of this
model, and the responses should be measured from the mocks (review sec. 3.B) before the SSC term is trusted for LRG.

### 8.2 Results of the normalisation checks (2026-10-07, bundle norm_checks_20261007.tgz)

> **Superseded (2026-10-12):** the mocks do contain the long modes (8.4, 9.2). The small sigma(delta_norm) and the
> ~0 slopes are the per-mock FKP weights (9.3), not missing modes; the model's "2.3x too large" sigma(delta_norm) is the
> fixed-weight assumption of `local_average_statistics`.

- **Convention settled**: the pipeline calls `compute_fkp2_normalization(fkp, cellsize=10)` (spectrum2_tools.py:474-501,
  `norm = {'cellsize': 10.}`, no split): norm = alpha sum_c D_c R_c / V_c. Recomputed DR tracks the files' norm mock by
  mock (corr 0.95 / 0.97 NGC / SGC over 6 mocks, residual 0.14-0.17 % from NGP vs CIC and one random file). The randoms
  differ between mocks (positions, redshifts, NX). `norm_kind='data-randoms'` is right.
- **The model's long-mode variance is 2.3x too large for LRG1 in sigma (5x in variance)**: predicted sigma(delta_norm)
  0.99 % NGC / 1.39 % SGC (clustering 0.96 / 1.34 %, pair-count sigma^2_W0W0 = 3.1e-6 / 6.2e-6) against the mocks'
  0.43 / 0.75 % (clustering part ~0.36 / 0.66 % after the Poisson 0.24 / 0.36 %). For QSO it is consistent (0.22 vs
  0.25 %, 0.33 vs 0.31 %). The pair-count sigma_b (0.18 % NGC) is what linear theory gives for a 1.2e9 (Mpc/h)^3 shell
  (sigma_b^2 ~ P(q ~ 1/L_r)/V), so the holi LRG1 mocks have ~2.7x less rms in the shell's mean density than a
  linear-theory realisation of the DESI fiducial cosmology. Candidates: the mocks' n(z) is (partly) fixed by construction
  (galaxies downsampled to a target n(z) per mock, which removes the radial long modes), or the parent box lacks the
  600-1200 Mpc/h radial modes. `check_norm_catalogs.py` now prints the mock-to-mock n(z) scatter per z bin against
  Poisson (linear theory: 2-3 % per 50 Mpc/h slab for LRG; Poisson ~0.5 %): rerun it. If the n(z) is fixed per mock, the
  mocks cannot validate the LRG super-sample term (the real data keep those modes) and the 6.1 / 7.5 agreement of
  sigma_P0 (0.43 vs 0.51 %) was reached with a long-mode variance 5x larger than the mocks contain.
- **The P_hat-delta_norm slope**: mocks l = 0 at k ~ 0.1, 0.2: +0.12, +0.19 (NGC), +0.02, +0.24 (SGC); correlations
  0.03-0.10; the P_0 amplitude mode has r = +0.03 with delta_norm (NGC, SGC). The model (data-randoms) gives -0.25 to
  -0.18 and r(long-mode parts) = -1. Note the Poisson part of delta_norm gives slope ~0, not -1: the extra pairs of a
  Poisson-high realisation cancel its normalisation (T - P Var_P ~ 0 for a uniform window). So the mocks' P_0 mode is
  not driven by the realised mean density, consistent with the radial modes being largely absent from them.
- `norm_convention.py`: num_shotnoise is not a clean density tracer for LRG1 (sigma 0.68 % vs 0.43 % for norm, corr
  0.3; QSO 0.13 % and corr 0.84): the alpha^2 sum_r w_r^2 term with per-mock randoms dominates its scatter. Ignore its
  slope test; the catalogue DR test is the one that settled the convention.

### 8.3 n(z) scatter of 8 mocks (nz_check_20261007_0351.tgz) and the 859-mock measurement to run

> **Superseded** by 8.4 (8 mocks were noise).

8 mocks, 10 z bins: the radial density scatter per 50-100 Mpc/h bin is 1.3-2.2 % (NGC), 1.3-3.4 % (SGC), well above Poisson
(0.4-0.6 %) and 0.75-0.8 of the linear slab prediction; but the whole-shell (500 Mpc/h) count scatters by 0.19 % (NGC,
clustering part) against 0.59 % predicted (ratio 0.32), consistent with the 859-mock delta_norm ratio 0.37 (NGC) / 0.49
(SGC). So the mocks have the 50-100 Mpc/h radial modes and lack the >~ 250 Mpc/h ones: a parent-box / replication or a
wide-bin n(z) rescaling in the mock construction, not a fixed n(z). Consequences: (i) the pair-count sigma^2 and the
tree-level LA are not falsified by these mocks; the mocks simply do not contain the modes; (ii) the LRG super-sample term
cannot be validated on holi at the 500 Mpc/h scale, and the model's sigma_P0 "agreement" was accidental; (iii) a
mock-calibrated validation can use the measured radial covariance. `desi_validation/nz_scatter.py` measures n(z) of all
859 mocks from the data files (seconds per mock), the scatter per radial scale against theory, the NGC-SGC correlation
of the shell counts (shared parent-box modes), and saves C_nz(z, z').

### 8.4 n(z) of all 1000 holi mocks (paper_inputs_20261009_0214.tgz): 12 defective LRG mocks; 8.3 superseded

- **12 defective LRG mocks**: 122, 131, 166, 218, 470, 573, 664, 667, 699, 746, 793, 880 have +55-65% galaxies in the
  single slice 0.46 < z < 0.47, in both caps (QSO: none). 11 of them are in the 859 used so far. They are now excluded by
  the paper pipeline (`paper/products/excluded_mocks.json`, written by `make_products.py` from nz_scatter, 6-MAD
  criterion; `PAPER_KEEP_ALL_MOCKS=1` keeps them). Effect: LRG1 NGC Gaussian-model lambda_1 5.57 -> 5.15, P0-amplitude
  variance ratio (recommended model) 1.14 -> 1.05; chi2/n of the recommended model changes by < 0.002. Report them to the
  mock makers.
- **8.3 was noise.** With 988 clean mocks the clustering rms of the LRG1 slab counts over linear theory, (b1 + f)
  sigma_slab, is 0.86-0.88 for L <~ 100 Mpc/h and 0.75-0.78 (NGC) / 0.69-0.72 (SGC) at 230-450 Mpc/h: the mocks
  contain ~60% of the linear variance of the shell-scale mode, not ~10%. QSO: 0.9-0.95 up to ~900 Mpc/h (the
  whole-shell value is Poisson-dominated; nz_scatter's Poisson term ignores the weights).
- So the mocks' sigma(delta_norm) = 0.43 / 0.75% is ~half what their shell counts imply (~0.8% for delta_norm = 2
  delta_shell): the normalisation responds only partly to the shell-mean density (the per-mock randoms follow the mock's
  own n(z)). The local-average term should be re-derived for that normalisation before the next SSC run.
- The caps share long modes: NGC-SGC correlation of the total counts 0.34 (LRG1, clean) and 0.45 (QSO), although the
  P(k) bins are uncorrelated. A GCcomb SSC that assumes independent caps underestimates the mocks' shell-mean mode.

### 8.5 Walkthrough results (LRG1 NGC, 2026-10-12): the alpha warning, the normalisation, the window arbiter

- **thecov's alpha warning is the veto holes, not an error in m.** The check compares m with a 64-NN density over
  ~20 Mpc/h spheres (~0.9 deg), which include the holes; m is the density between them. Expected ratio 1/<f> at 0.92 deg
  = 1/0.816 = 1.22 (hole_fraction_LRG1.json); measured 1.21 (mean weight) and 1.20 (kernel tracer). The warning now says so.
- **The old "own weight" variant multiplied NX (already completeness-weighted) by WEIGHT** (m_own / m_mean bimodal, +10% in
  amplitude). The walkthrough now builds it as alpha rho_r w_own, the same density as the mean-weight window.
- **Per-mock FKP weights suppress the response of norm to the long modes.** The holi randoms copy each mock's redshifts and
  NX, so w_FKP = 1 / (1 + NX P0) follows each mock's n(z) and n w_FKP responds to a density change by w_FKP only.
  Regression of the files' delta_norm on delta int n(z)^2 (WEIGHT-weighted counts of all mocks): slope 0.19, corr 0.41
  (LRG1 NGC; nP0 ~ 5 -> 1/(1+nP0) ~ 0.17), 0.85 and 0.98 (QSO; nP0 ~ 0.2 -> 0.83). thecov's local_average_statistics
  assume fixed weights and randoms, hence its 2.5x too large sigma(delta_norm). `norm_fkp.py` tests the mechanism mock by
  mock from the data catalogues (four models: own / fixed FKP x own / fixed randoms); results in 9.3 (both questions
  left open here are answered there).
- **Window arbiter**: `run_gaussian_footprint.py --no-thecov --save-realisations --task i` (submit with
  `run_grf_tasks.sh <bin> <cap>`) saves the random-field realisations; `paper/thecov_walkthrough.ipynb` section 3d compares
  them with own-weight, mean-weight and xi-kernel thecov (isotropic model): variance per k and adjacent-bin correlations.
  Results in 9.1.

## 9. Results of 2026-10-12 (walkthrough runs, random fields, norm_fkp, SN-NG analysis)

Inputs: `thecov_walkthrough.ipynb` executed at NERSC (LRG1 NGC; two runs), the GRF realisations, `norm_fkp_<bin>.log`, the
Oct 6 bundle (`thecov_bundle_20261006_2145.tgz`: report_data_holi-kcore2-*, ssc_holi-kcore2-*), `paper_inputs`
(nz_scatter). The local analysis scripts are not in the repository (they read DESI data); their logic is described here.

### 9.1 Random-field window arbiter (LRG1 NGC, 400 realisations, cell 6 Mpc/h, healpix 512)

Var_GRF / Var_thecov of P0 (isotropic model, both from the same footprint):

| k range | xi kernel | local window, mean weight | local window, own weight |
|---|---|---|---|
| 0.03-0.1 | 1.004 | 1.084 | 1.083 |
| 0.1-0.2 | 1.025 | 1.070 | 1.073 |
| 0.2-0.3 | 1.019 | 1.039 | 1.031 |

The xi kernel is right to ~2%; the weight construction does not matter. The mocks agree independently: with SSC + disc,
mock / model ~ 1 at k < 0.1 in all multipoles with the kernel, while the local-window models are 5-13% low. To do:
cell 4 and SGC.

### 9.2 Long-mode response measured against the shell density (848 clean mocks)

Regressor: delta_shell = delta ln int n(z)^2 with fixed WEIGHT (from nz_scatter), not delta_norm. dlnP_hat / d delta_shell:

| | P0 0.02-0.1 | P0 0.1-0.2 | P0 0.2-0.3 | P2 0.02-0.1 | P2 0.1-0.2 | P2 0.2-0.3 |
|---|---|---|---|---|---|---|
| LRG1 NGC | +0.035 +- 0.027 | -0.064 +- 0.013 | -0.090 +- 0.011 | +0.80 +- 0.15 | +0.89 +- 0.085 | +1.03 +- 0.10 |
| LRG1 SGC | +0.048 +- 0.029 | -0.069 +- 0.014 | -0.113 +- 0.012 | +0.93 +- 0.17 | +1.38 +- 0.12 | +2.46 +- 0.24 |

QSO: consistent with 0 (shell scatter 0.3%, too small). thecov's fixed-weight prediction (walkthrough): P0 ~ -0.25 to
-0.3, P2 +0.5 to +2.2. Signs and the P2 size agree; the P0 net response is ~3x weaker in the mocks. Shell-density scatter
0.8-1.1% (LRG1): the long modes are present. Against delta_norm the slopes are ~0 (P0) and negative (P2): not meaningful.

### 9.3 The files' norm: per-mock FKP weights (`norm_fkp.py`)

Own-FKP models reproduce the files' norm mock by mock: LRG1 NGC corr 0.987, slope 0.99, residual 0.43%; LRG1 SGC 0.997,
1.00, 0.75%; QSO corr 0.99, slope ~1. With fixed FKP weights the residual is 1.48% (NGC) / 1.62% (SGC) for LRG1.
<w_FKP>_m^2 = 0.186 (LRG1) and 0.85 (QSO) sets how much norm responds to a density change. corr(delta_norm, delta_shell)
= 0.41 / 0.25 (LRG1 NGC / SGC), 0.98 (QSO). The walkthrough's per-mock FKP prediction of sigma(delta_norm) (0.18%) is still
half the mocks' 0.39% (LRG1 NGC): the rest of the scatter is not yet modelled (randoms per mock, Poisson of the data).

### 9.4 SN-NG terms: Wadekar & Scoccimarro equivalence, the window convolution, amplitudes

- `DiscretenessCovariance` is WS20's SN-NG: one shared galaxy (4 cases) -> tree-level B, weight int S m^2 / norm^2
  (their I34 / I22^2, eq. 91); two shared galaxies (2 cases) -> P(|k1 +- k2|), weight int S^2 / norm^2 (I24 / I22^2,
  eq. 92); no 1/n^3 or within-spectrum coincidences (true shot noise subtracted, eq. 79). WS20's local-average shot-noise
  terms (their eqs. 44, 86) are in `SuperSampleCovariance(discreteness=True)`. WS20 evaluate SN-NG in the local
  approximation; the window convolution is thecov's own addition.
- `window_convolve(C_local, C_G)` = D^1/2 K (C_local / D) K^T D^1/2 with K = corr(C_G)^1/2. Rows of K have unit L2 norm
  (right for white noise); a real window's rows sum to 1 (preserves a smooth function). Measured on the P0 diagonal at
  k = 0.15-0.3, convolved / local = 2.14 / 2.26 (LRG1 NGC / SGC), 1.33 / 1.38 (QSO), equal to (sum_j K_ij)^2 to 2%; with
  K / K.sum(1) it is 0.86-1.00.
- Wishart ML amplitude a in G + SSC + a disc (848 mocks): local / convolved disc

  | k_max | LRG1 NGC | LRG1 SGC | QSO NGC | QSO SGC |
  |---|---|---|---|---|
  | 0.12 | 2.10 / 1.25 | 1.40 / 1.00 | 1.53 / 1.10 | 1.40 / 1.00 |
  | 0.2 | 1.93 / 1.20 | 1.15 / 0.90 | 1.83 / 1.35 | 1.70 / 1.23 |
  | 0.3 | 1.98 / 1.23 | 1.10 / 0.90 | 2.48 / 1.83 | 1.85 / 1.33 |

  Errors 0.05-0.25. At a = 1 the convolved term wins (-2 dlnL 76-712 LRG1, 2-95 QSO); with a free, QSO has no preference,
  LRG1 still prefers the convolved shape. Residual (mocks - G - SSC) on the P0 block at k = 0.15-0.3, |i-j| = 0..3, units
  sqrt(G_ii G_jj): LRG1 NGC 0.234 / 0.235 / 0.244 / 0.253 (+-0.01; local disc 0.097, convolved 0.206), QSO NGC
  0.060 / 0.065 / 0.094 / 0.081 (local 0.025): flat, no near-diagonal structure.
- Separate amplitudes at k_max 0.3, B / P / T0 window-mixed with the row-normalised K:

  | | SSC | B/n | P/n^2 | T0 (tree) |
  |---|---|---|---|---|
  | LRG1 NGC | 1.21 +- 0.15 | 2.26 +- 0.24 | 6.4 +- 1.0 | 0.39 +- 0.06 |
  | LRG1 SGC | 1.36 +- 0.16 | 1.76 +- 0.19 | 3.9 +- 0.7 | 0.36 +- 0.06 |
  | QSO NGC | degenerate | degenerate | 4.15 +- 0.33 | degenerate |
  | QSO SGC | degenerate | degenerate | 2.17 +- 0.24 | degenerate |

  P0 diagonal / Gaussian at k 0.15-0.3: B/n 0.07-0.08, P/n^2 0.014-0.02, T0 0.03 (LRG1); P/n^2 0.02-0.03 (QSO); mocks - G
  0.20-0.23 (LRG1), 0.06 (QSO). LRG1 P/n^2 is unstable at k_max 0.2. Local (unmixed) T0 is rejected outright (negative
  amplitude); `C_T0_response` (P0 diagonal 34% of G) is not usable as a template.
- Conclusion: the mocks contain 1.8-2.3x (LRG1, B/n) and 2.2-4.1x (QSO, P/n^2) the predicted Poisson non-Gaussian
  covariance, smooth in k. Candidates: (i) J_Smm, J_SS with per-mock FKP weights or the global <w^2>/<w>
  (`window_integrals` uses sum w^2 / sum w of the randoms everywhere); (ii) non-Poisson sampling (one-halo pairs raise the
  effective 1/n; would hit QSO's P/n^2 most); (iii) the local approximation of P(|k1 - k2|) near k1 ~ k2.

### 9.5 `sn_diagnostics.py` (written 2026-10-12, not yet run)

    cd ~/thecov && git pull origin thecov2
    OUT=/global/cfs/cdirs/desicollab/users/oalves/thecov_validation
    ENV="source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\$HOME/thecov:\$PYTHONPATH && cd \$HOME/thecov"
    for B in LRG1 QSO; do for C in NGC SGC; do
      sbatch -N 1 -C cpu -q debug -t 00:30:00 -J sn$B$C -o $OUT/sn_diag_${B}_${C}.log \
        --wrap "$ENV && python -u -m desi_validation.sn_diagnostics --bin $B --cap $C"
    done; done

Output `sn_diag_<bin>_<cap>.npz` (saved after each stage: spectra, integrals, pairs, disc, SSC, Gaussian). Only the
pair-excess helper has been tested (uniform points -> 1).

## 10. Working locally (from 2026-10-12)

What a local session needs (keep it outside the repository or in a git-ignored directory; the repository is public):
- Spectra: `/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base/{holi-v3-altmtl,
  abacus-2ndgen-dr2-complete,abacus-2ndgen-dr2-altmtl}/mock*/mesh2_spectrum_poles_*_weight-default-FKP.h5`
  (read with `lsstypes`; `paperdata._read_one_spectrum`).
- Catalogues: the holi random files are slim (TARGETID, WEIGHT, NX); RA, DEC, Z come from the parent data-dr2-v2 randoms
  through `clustering_statistics` (NERSC only). Export completed catalogues at NERSC with `desi_compare.load_region`
  (RA, DEC, Z, WEIGHT, WEIGHT_FKP, NX, cut to the bin) for mock 173 (data + the random files needed) and the data of a few
  other mocks, rather than copying the raw files.
- Products: `/global/cfs/cdirs/desicollab/users/oalves/thecov_validation/` and `~/thecov_desi/` (pair counts, covariances,
  kernel caches, ssc windows: avoids recomputing the expensive windows).
- Paths are hard-coded for NERSC in `paper/paperdata.py` (top of the file) and `desi_compare.Paths`: point them at the
  local copies (a `DESI_DATA` root is the planned change).
- Environment: this repository's `thecov`, numpy, scipy, `lsstypes`, `cosmoprimo`, optionally `jax`.

## 11. Bispectrum covariance: class-1 Gaussian term implemented (2026-10-13)

`thecov/bispectrum_algebra.py`, `thecov/bispectrum.py`, `thecov/bispectrum_windows.py`; `BispectrumGaussianCovariance`
exported from `thecov`. Derivation `report/bispectrum_covariance_guide.typ`; implementation notes, conventions,
validation and the remaining terms in `report/bispectrum_implementation.md`. Tests `tests/test_bispectrum_*.py`
(fast: 26 tests, ~4 min; slow ones deselected with -m "not slow"). Development scripts and the GRF validation in
`desi_validation/bispectrum_dev/` (grf_sphere.py -> grf_theory.py -> grf_compare.py; windows cached in
grf_sphere_windows.npz). Status (2026-10-14): classes 1 AND 2 implemented (`class2=True`, `terms=` separates them). Coupling numbers,
vertices, radial assembly exact against independent quadrature; window invariants on a sphere agree with Monte Carlo to
3 %; periodic-box Gaussian fields reproduce class 1 + class 2 to 1.00 +- 0.02 including the off-diagonal (pure class-2)
structure. On a sphere centred on the observer (R = 500 and 1000, dk = 0.04) class 2 fixes the off-diagonal
correlations and most of the diagonal, but a +8-10 % B000 offset that does not scale with R remains (numerics and
pair-sum noise excluded), and a B202 excess on the low-k quadrupole elements that halves from R = 500 to 1000 (the local
line-of-sight approximation). The B000 offset is now understood (report section 4.1): (a) the mesh did not resolve the
sharp window (k_f = 2.4 k_w; the same sphere in a 2000 Mpc/h box agrees to 1 +- 3 %), (b) the pair sums at 3000 anchors
carry a coherent 2.5-4 % Q0 noise (use >= 10 000), (c) uniform anchor sampling is badly biased for non-uniform
weights (now weight-proportional). A Gaussian-window GRF (4000+4000 realisations) agrees with the theory to 2 %
including the new k3 ~ 0 term (k3_zero_block, validated against an exact lattice Wick sum: exact_mesh_gaussian2.py).
Corrections to the notes found on the way: none to the physics; Q_n^box = V sqrt(N/4pi) H as the guide says.

### Update 2026-10-15 (c): R_l(k_L), survey-mean response resolved as a cancellation

`response_ibisp.py` now takes several `--k-L` (F_l once per catalogue). LRG1 NGC, mocks 173-178: R_0 flat at
1.89-1.94 / 1.89-1.90 / 1.80-1.82 (bands 0.05-0.1 / 0.1-0.2 / 0.2-0.3) for k_L = 0.015-0.03; R_2 rises from 2.8 to
3.9 as k_L drops from 0.03 to 0.007 (onset at 2 pi / shell thickness: transverse long modes). k_L = 0.005 breaks down
(<delta_L>_gal -> 0: the survey-mean mode is invisible in-survey). Net survey-mean P0 response R_0 - 2 = -0.09 /
-0.10 / -0.19 vs 9.2's -0.064 / -0.090 (x2 if 9.2's slopes are per delta ln int n^2 - to confirm). Section 9 of
report/grf_randoms_results.md. Next: use R_0(q->0) in the local-average term; R_2 with the window's long-mode
anisotropy; test 3 (lognormal-Poisson from a larger box) for the end-to-end check.

### Update 2026-10-15 (d): survey-mean response closed (report section 10)

`survey_mean_regression.py` (848 mocks, regressor delta_L = N/median - 1): 9.2's regressor was 2 delta_L (sigma ratio
2.05, slopes reproduced). Per unit delta_L: d ln P0 = +0.04 / -0.12 / -0.17 (NGC), +0.10 / -0.13 / -0.24 (SGC), vs
in-survey R_0 - 2 = -0.09 / -0.10 / -0.19: agreement within 1 sigma. d ln P2 = 1.4-2.1 (NGC), 1.8-2.7 (SGC) -> R_2(q->0)
= 3.4-4.1, the transverse (small-k_L) limit. `survey_mean_synthetic.py` (known uniform mode delta_u, lognormal-Poisson,
common random numbers): d ln P0 / d delta_u = 0.00 +- 0.02 with data-matched NX (+-FKP, expected 2 - 2) and
1.00 +- 0.02 with fixed NX (expected 2 - 1); d ln norm / d delta_u = 0.40 (own FKP) / 2.00 (no FKP) / 1.00 (fixed).
Next for thecov: R_0(q->0) = 1.9 (LRG1) in the local-average term; R_2 with the window's long-mode anisotropy; the
super-survey sigma^2 with the same anisotropy (radial from nz_scatter, transverse from |W(q)|^2).

### Update 2026-10-19: B/n bookkeeping corrected; QSO pair term is ordinary clustering

- thinning closed form: B/n = lin - g1 - 2 P/n^2 (the 2a part of the 1/p^2 scaling was missed). LRG1 NGC B/n 0.090 diag
  (1.2x tree), 0.12 off-diag; QSO off-diag 0.03-0.04. P/n^2 unchanged. QSO "2.6x" was against LRG1's tree level (units error).
- `pair_diagnostics.py`, `pair_model.py`: QSO mocks have no duplicates, few close pairs (3-7 % of the pair term), and an
  off-diagonal pair term equal to the exact clustering expectation (1.01-1.04).
  Nothing to report to the mock builders; the HANDOFF 9.4 QSO P/n^2 excess is a thecov (local approximation) or fit
  question. report/grf_randoms_results.md sections 7 (correction) and 11; tutorial report Part B rewritten.

## 12. Tree-level T0 against the mocks with the measured discreteness terms (2026-10-21; `t0_tree_fit.py`)

Template: T0 = snake + star at b3 = 0 (`C_T0_snake + C_T0_star` of `ssc_holi-kcore2-<tracer>.npz`). Not star_b3
(= d C_star / d b3) and not `C_T0_completion` / `C_T0_response` (the response-approach rebuild of the same T0, which
also contains the completion and the window-convolution inflation of 9.4). Base model: production Gaussian (kernel,
RSD) + measured P/n^2 and B/n from the thinning closed form (mock 173) in the P0-P0 block, tree disc terms scaled by the
P0 measured/tree off-diagonal ratio (B 1.26, P 1.23) elsewhere. Wishart likelihood, 848 clean LRG1 / 859 QSO NGC mocks,
k <= 0.3. Data-vector blocks are selected by position (ell // 2): an interim session analysis selected rows 112-167 for
"ell = 2" (i.e. P4) and summed all five T0 pieces, which gave a spurious "T0 ~10x too large, a ~ 0.1"; neither error is
in the repository code (every block loop here runs over positions).

| LRG1 NGC | a (tree T0) | Delta(-2 ln L) a = 0 / a = 1 | min eig G^-1/2 C G^-1/2 at a = 1 | NG direction mock/model, a = 0 -> 1 |
|---|---|---|---|---|
| P0 | 0.38 +- 0.05 | +69 / +223 | 0.62 | 1.38 -> 0.99 |
| P2 | 0.61 +- 0.05 | +209 / +62 | 0.69 | 1.97 -> 1.19 |
| P0+P2 | 0.43 +- 0.03 | +223 / +567 | 0.42 | 1.38 -> 0.99 |

- Where a = 1 fails: squeezed coupling, mean corr(k 0.02-0.05, k 0.2-0.3) mocks 0.067 (P0) / 0.039 (P2) vs model 0.122 /
  0.077 at a = 1 (0.061 / 0.049 at the best a); hard coupling, mean corr(|i-j| >= 4, k 0.15-0.3) mocks 0.184 / 0.133 vs
  0.170 / 0.090 at a = 1 (tree about right for P0, too small for P2). This is 7.1's squeezed excess.
- Two amplitudes (a_hard: both bins >= k_split; a_sq: the rest): a_sq = 0.37-0.39 (P0), 0.48-0.51 (P2), 0.39 (P0+P2) for
  k_split 0.07 and 0.10; a_hard = 0.35 / 0.51 (P0), 0.74 / 0.95 (P2), 0.47 / 0.63 (P0+P2). (1, 1) is excluded by
  Delta(-2 ln L) ~ 220 (P0) to 600 (P0+P2).
- QSO: T0 negligible, amplitude unconstrained (sigma 1.5-2.4).
- P0+P2+P4 cannot be tested: the base model is not PD there (min eig -0.065 at a = 0), from the scaled tree disc blocks.
- Reading: the in-survey first-order response is 1.2-1.5x tree level (R_0 = 1.9, update 2026-10-15 (c)), so a weak
  R_1 does not explain the squeezed overshoot; the squeezed T0 (k1 << k2) is set by the second-order response
  (T -> R_2 P(k1)^2 P(k2), 7.1), not measured yet. Superseded by section 13: the single amplitude is a compromise
  between blocks that want a = 1 and the squeezed block that wants 0.5; a fitted a_sq is not an option (it cannot be set
  from one catalogue).
- Reproduce: `python -m desi_validation.t0_tree_fit --tracers LRG1 QSO --region NGC --out <dir> --plot` (Perlmutter
  paths by default; CSV, JSON and figure).

## 13. Tree-level T0 block by block: only the squeezed block is wrong (2026-10-22; `t0_tree_fit.py --blocks`)

Why the single amplitude of section 12 is misleading. In the production binning (dk = 0.005) the Gaussian term has
a 0.46 correlation between adjacent bins (window convolution), the whitened T0 has one large positive and one large
negative eigen-direction, both concentrated on the lowest k bins, and 80 per cent of the Wishart weight sits on the
squeezed block. The tree T0 is also not positive on its own: its (i, i+1) element exceeds its diagonal (ratio 1.05-1.3)
and the P2 diagonal is negative at low k, so a one-parameter scan mixes blocks that disagree. Rebinning to dk = 0.025
(5 bins; adjacent correlation 0.39 -> errors on individual elements become meaningful) and fitting an amplitude per
block in correlation units (weighted least squares, Wishart element errors; `block_amplitudes`) separates them.
Base model as in section 12. LRG1 NGC, 848 mocks; `prod` = production inputs (P_ir A, sigma_v 2.088), `meas` = the
same tree kernels fed with the mock-mean nonlinear P0 / dilution divided by the Kaiser-FoG monopole (ratio to P_ir A:
1.02, 1.02, 1.15, 1.31 at k = 0.03, 0.1, 0.2, 0.3), run locally with `t0_run.py` (single process, ~7 min; the `prod`
run reproduces the production `C_T0_snake + C_T0_star` to 2e-7).

| block (dk = 0.025) | P0xP0 prod | P0xP0 meas | P2xP2 prod | P2xP2 meas | P0xP2 prod | P0xP2 meas |
|---|---|---|---|---|---|---|
| diagonal (T0 = 7 / 5 / 2 per cent of the variance) | 1.26 +- 0.22 | 1.11 +- 0.18 | 2.27 +- 0.30 | 2.00 +- 0.25 | 0.4 +- 0.5 | 0.3 +- 0.5 |
| adjacent | 0.92 +- 0.11 | 0.86 +- 0.10 | 1.60 +- 0.13 | 1.50 +- 0.12 | | |
| low-low (< 0.1, off-diagonal) | 0.84 +- 0.21 | 0.83 +- 0.20 | 0.68 +- 0.30 | 0.67 +- 0.29 | | |
| squeezed (< 0.1 x >= 0.1) | 0.53 +- 0.03 | 0.50 +- 0.03 | 0.68 +- 0.04 | 0.68 +- 0.04 | 0.28 +- 0.05 | 0.27 +- 0.04 |
| hard (>= 0.1, off-diagonal) | 0.84 +- 0.05 | 0.75 +- 0.05 | 1.36 +- 0.05 | 1.26 +- 0.05 | 0.13 +- 0.07 | 0.11 +- 0.06 |
| hard (>= 0.15, off-diagonal) | 1.24 +- 0.10 | 1.00 +- 0.08 | 1.84 +- 0.09 | 1.57 +- 0.08 | 0.08 +- 0.14 | 0.06 +- 0.12 |

Figure: `report_figures/t0_block_ratio_LRG1_NGC.png` (mock excess, tree T0 `meas`, and their ratio, per block, for the
three multipole blocks). Full table: `t0_block_amplitudes_LRG1_NGC.csv` (also sigma_v = 3.5 and 5.0 variants, which
only shrink T0 and move every amplitude up: the single-amplitude a goes 0.38 -> 0.49 -> 0.65 for P0; FoG is not the
explanation of anything here).

- **P0xP0.** Diagonal, adjacent, low-low and hard blocks are consistent with amplitude 1; with the measured nonlinear
  P as input the hard block (k >= 0.15) is 1.00 +- 0.08. The only deficient block is the squeezed one, 0.50 +- 0.03.
  So the tree T0 with a dressed (measured) power spectrum is right at the 10 per cent level wherever both k are
  hard, and the hard-block "amplitude 0.35-0.5" of section 12 was an artefact of the one-parameter fit.
- **P2xP2.** Squeezed 0.68; hard 1.6-1.8 and diagonal 2.0-2.3: the mocks have more P2-P2 non-Gaussian covariance at
  k >= 0.15 than the tree T0 gives (this is the 7.x "P2 excess", 1.97 in the NG direction). Part of it may be the
  ell = 2 discreteness blocks, which are still the scaled tree terms (stopgap; the thinning closed form for ell = 2 is
  on the work list). Not a T0 amplitude problem in the squeezed sense.
- **P0xP2.** The mocks show almost no T0-like coupling between P0 and P2: squeezed 0.27 +- 0.04, hard 0.1 +- 0.1
  (mock excess 0.035 / 0.006 in correlation units against tree 0.12 / 0.07). The tree angular structure of the P0-P2
  coupling is not what the mocks do.
- **Reading.** In the squeezed limit the tree T0 is the second-order response, T(k_s, -k_s, k_h, -k_h) -> R_2(mu)
  P(k_s)^2 P(k_h) (IR-safe; for matter in real space the coefficient is exactly R_1 = 47/21 - n/3 at first order and
  the Legendre coefficients of R_2 are (4.19, 0, 7.86, 0, 2.82) at k_h = 0.2 for the test spectrum). The mocks say
  the second-order response of the nonlinear redshift-space galaxy power spectrum at k_h = 0.1-0.3 is half the tree
  value for P0, 0.7 for P2 and ~0.3 for the P0-P2 cross term, while the first-order response is 1.2-1.5x tree
  (update 2026-10-15 (c)). Second-order responses of the nonlinear matter P are known to fall below tree level at
  these k (separate-universe measurements); FoG and halo exclusion act in the same direction for LRGs.
- **Consequence for the catalogue-only plan.** The hard block needs no calibration beyond feeding the tree kernels
  the measured power spectrum (`meas`), which is catalogue data. The squeezed block needs R_2 of the catalogue, which
  cannot be taken from the tree kernels. Candidate single-realisation estimators, both at the 2-3 sigma level per
  catalogue by a mode-counting estimate: (i) the collapsed four-point function, mean over soft-hard mode pairs of
  (|delta_s|^2 - P_s)(|delta_h|^2 - P_h) in the catalogue, S/N ~ sqrt(N_s N_h) R_2 P_s / V per bin pair; (ii) the
  quadratic term of the position-dependent power spectrum, P_h(x) = P_h [1 + R_1 delta_L(x) + R_2 delta_L(x)^2 / 2]
  over ~100 sub-volumes. Neither is implemented. The alternative that stays within the rule is to take the matter
  second-order response from separate-universe theory (not the mocks) times the measured first-order galaxy
  response squared over tree, which needs a derivation before it is used.
- Reproduce: `python -m desi_validation.t0_tree_fit --tracers LRG1 --region NGC --blocks --t0-variants
  t0_variants_LRG1_NGC.npz --out <dir>`; variants from `python desi_validation/t0_run.py LRG1 NGC meas out.npz`
  (needs the `t0_inputs` bundle: production matrices, thinning closed form, mock vectors, mock-mean P0).


## 14. The second-order response R_2 from one catalogue: feasibility (2026-10-23; `response_ibisp.py --soft-band`, `r2_analysis.py`)

**Verdict: feasible in principle, not at the precision Paper 2 needs. The squeezed T0 goes to the follow-up letter.**

Why it is possible at all. The squeezed T0 block is the collapsed trispectrum T(k_s, -k_s, k_h, -k_h) = R_2 P_s^2 P_h, and one
realisation holds one sample of each collapsed pair (block amplitude error ~1/rho ~ 0.9). The same R_2 governs every nearly
collapsed configuration T(k_h, -k_h + q, p, q - p), |q| << k_h, of which there are ~(k_q/k_f)^3 per collapsed one: regress the
local hard power on the local soft-band power. Toy check (periodic box, amplitude-modulated hard field): ~20x the
information of the collapsed covariance per realisation.

Estimator (same F_l / filter machinery as the first-order response; `--soft-band k1 k2 ... --k-q ...`):
  delta_s = (F W_band) / m_w (sharp band, m_w = mean density smoothed at k_w = 0.05);
  S_q(x) = [m_w^2 delta_s^2]_q / [m_w^2]_q  (local band power smoothed at k_q, m^2-weighted);
  g = S_q - <S_q>_{m^2} - beta (delta_q - <delta_q>_{m^2}),  delta_q = (F W_q)/m_q;
  Q_l(k) = <F_l^*(k) FT[g F](k)>_shell;   R_2 = 2 Q_conn / (P_l <delta_s^2 g>_{m^2}) x (1 + SN/P_s)^2,
with Q_conn = Q - QS - (Qh - QhS) + beta (Qc - QcS): self pairs; the galaxy's own contribution to the local soft power
(h = [2 w (Fb * kappa) + w^2 c_bb]/den, kappa = w_q K_b; both pair members -> Qh on the F^* leg) — the B/n-type term of the
four-point estimator; the galaxy's own contribution to delta_q (c = w V_q/m_q; both members -> Qc). Normalisation: for a
sharp band the Gaussian covariance of delta_s^2 with g is twice the connected mode sum (filter once, as int P W for R_1),
hence the 2; (1 + SN/P_s)^2 removes the band shot noise from the normaliser. R_2 is read off as the slope of
(Q_conn)(k) = a P_l(k) + b over k = 0.12-0.3 (the constant absorbs the remaining k-independent discreteness; it is small once
Qh and Qc are removed). Tree expectation in the SAME units (`r2_analysis.tree_expectation`, Monte Carlo over the estimator's
configurations with the long-mode snake terms removed, as the beta projection does): 1.3-1.6 for LRG1, collapsed; x1.05-1.5
at k_q = 0.02-0.03 for the 0.03-0.05 band (non-collapsed trispectrum is larger, not smaller).

Lessons (each cost a round of six 12-min runs): (i) the disconnected term sum_q g(q) <F^*(k) F(k+q)> = P(k) int m^2 g is the
m^2-weighted mean of the regressor (m- or galaxy-weighting gave R_2 = -0.4 to -2.4). (ii) The shared-galaxy term h is as
large as the whole tree signal (0.7-1.9 in R_2 units per pair member); a- and b-sides agree to all digits, as they must.
(iii) The delta_q self/partner term (0.4-1.1, opposite sign) comes from the beta projection. Neglected: both pair members
inside the soft legs (P/n^2-type, 1-5 per cent). The weighted-randoms synthetic test (`--synthetic R g SEED`) is blind to
(ii)-(iii) because its "galaxies" are the 16M randoms; the Poisson-sampled version (`--synthetic-poisson P0`, galaxies from
half the randoms at the LRG1 density, the other half as randoms, exact R_2 = 2 R^2/g^2) is the right test.

Results (LRG1 NGC; `responses/r2/`, `r2_*.csv`):
- Poisson-sampled synthetic, exact R_2 = 1.183, two seeds, bands 0.02-0.035 / 0.035-0.05: k_q = 0.02: 1.53, 1.68 / 1.70,
  0.31 (mean 1.61, 1.00) — unbiased within the single-realisation noise (+-0.6); k_q = 0.01: 3.7, -7.3 / 6.6, -2.7 and
  k_q = 0.005: 9, -30 / 17, 2 — noise-dominated (few q modes; the per-k fit errors are 3-10x too small there). R_1 from
  the same catalogues: 1.55-1.66 vs 1.538 exact.
- Holi mocks 173-177 one at a time, all corrections, k_q = 0.02 (per-catalogue scatter in brackets; tree in the same units):
  0.03-0.05: 1.24 +- 0.24 (0.54), tree 1.70 -> ratio 0.73;  0.05-0.075: 0.79 +- 0.26 (0.57), tree 1.57 -> 0.51;
  0.075-0.1: 0.37 +- 0.24 (0.54), tree 1.56 -> 0.24.  Ensemble (848 mocks, section 13): 0.50-0.68 of tree.
  k_q = 0.03 gives 0.47 / 0.41 / 0.03 of tree; k_q = 0.01 is noise-dominated. The 0.075-0.1 band falls with k_q to zero,
  a trend absent from the Poisson synthetic and opposite to the tree's q-dependence: real-catalogue physics (the
  response to a smoothed local soft power dies as the smoothing scale shrinks; non-Poisson discreteness is another
  candidate) that is not understood.
- Precision: +-0.55 per band per catalogue = 70 per cent of the expected signal (0.55 x tree ~ 0.8); ~40 per cent pooling
  the bands, ~30 per cent with both caps — against the +-35 per cent that keeps the squeezed block's effect on sigma(f)
  below 1-2 per cent (section 13). With the unresolved k_q dependence on top, this is a research project, not an
  ingredient of Paper 2.

Decision (publication plan, caveat a): Paper 2 = discreteness + SSC from the catalogue; the hard T0 block = the tree with
the measured P (1.00 +- 0.08, section 13); the squeezed block = the tree with the mock-validated statement that it is
overpredicted by ~2 and that this moves sigma(f) by 2-5 per cent — stated as a limitation, not tuned. The estimator, its
discreteness terms, the Poisson synthetic test and the tree expectation are the material for the letter.

Reproduce: mocks `python -u -m desi_validation.response_ibisp --bin LRG1 --region NGC --mock 173 --k-L 0.02 --soft-band 0.03 0.05
0.05 0.075 0.075 0.1 --k-q 0.01 0.02 0.03`; synthetic `... --mock 173 --synthetic 1.0 1.3 4 --filter tophat --k-L 0.05
--synthetic-poisson 0.085 --soft-band 0.02 0.035 0.035 0.05 --k-q 0.005 0.01 0.02` (one debug node, 8-12 min each; 5-job QOS
limit); analysis `python -m desi_validation.r2_analysis responses/r2/corrected_*.npz --bundle <t0_inputs.npz>` (slope fit,
ensemble reference for l = 0 and 2, tree expectation).

### Update 2026-10-23 (b): theory cannot supply the squeezed block; the recipe that remains

- Barreira & Schmidt 2017 (JCAP06(2017)053, eqs. 3.22, 4.7, 5.5-5.8) + Wagner et al. 2015 growth-only responses computed
  here with CAMB halofit (Takahashi and Mead2020) at z = 0.51 (CAMB linear P = bundle plin to 0.1 per cent): G_1 rises
  1.24 -> 1.51 and G_2 2.27 -> 3.4 over k = 0.1-0.3; with the nonlinear dilation terms the collapsed angle-averaged
  coefficient Cbar = R_2/2 + 2/3 R_K2 + 2/9 R_K.K + 4/45 R_KK (tree: 2 Cbar = 4.19 for the test spectrum, = the
  parallelogram Monte Carlo of section 13) is 0.9-1.3 of tree. For matter the tree squeezed covariance is right to ~20
  per cent at these k; the mocks' factor 0.5 is galaxy-sector physics (bias/velocity responses, satellites).
- In the collapsed limit (mu_12 = -1, f_12 = 1) the R_1 and R_K terms of eq. 3.22 vanish identically: the catalogue's
  precise first-order response carries no information about this block. b_3 rescales squeezed and hard blocks alike
  (dT0/db3 / T0 = 0.115 vs 0.126) and is excluded as the explanation.
- Recipe if the block is to come from the catalogue (section 14 estimator, unbiased on the Poisson synthetic at
  k_q = 0.02): a_sq = pooled slope amplitude over soft bands k_s <= 0.075, k_q = 0.02 and 0.03, both caps, l = 0,
  divided by the tree expectation in estimator units (`r2_analysis.tree_expectation`); one 12-min debug-node run per
  catalogue. Six-mock values: 0.73 / 0.51 of tree (k_q 0.02), 0.47 / 0.41 (0.03) for the two bands, ensemble 0.50-0.68.
  Expected precision +-0.2 of tree per cap pair (= +-35 per cent of the true 0.55; 1-2 per cent on sigma(f));
  systematic +-0.1 of tree from the k_q trend; the 0.075-0.1 band (0.24 -> 0.03 with k_q) is not used.
- Zero-cost alternative: tree squeezed block at amplitude 1 with the 2x overprediction and its 2-5 per cent (sigma f) /
  5-8 per cent (sigma sigma_v) effect stated as a limitation.

## 15. Fibre assignment and the discreteness terms: Abacus complete vs altmtl (2026-10-24)

Abacus 2nd-gen DR2 mock 0, LRG1 NGC, P0. Measured B/n and P/n^2 from the thinning closed form on each catalogue;
thecov kernel Gaussian at shot-noise levels p = 1, 0.5, 0.25 splits G exactly into G_PP + G_PS + G_SS (G is quadratic in
S = sum w^2 / p + randoms term); tree-level B/n, P/n^2 from thecov with the complete ssc_check inputs (b1 2.376, f 0.762,
sigma_v 2.22) for both windows. All runs on the Perlmutter checkout 725d5ba (it cannot fast-forward: uncommitted edit in
paper/thecov_walkthrough.ipynb there), identical for both versions. `thinning_cov.py` / `thinning_closed_form.py` now take
`--cs-version`; `thinning_cov.py thecov --tree-params <ssc npz>`. Outputs: `responses/abacus_disc/`, figure
`report_figures/abacus_fibre_discreteness_LRG1_NGC.png`.

Galaxies 941k -> 704k (75 per cent); shot noise / norm 2290 -> 3359 (x1.47, more than 1/0.75 from the completeness-weight
scatter). Band means of the diagonal, units of G(complete), k = 0.1-0.2 / 0.2-0.3:

| term | complete | altmtl | ratio | measured / tree, complete | altmtl |
|---|---|---|---|---|---|
| G total | 1 / 1 | 1.27 / 1.42 | 1.27 / 1.42 | | |
| G_PP | 0.63 / 0.40 | 0.68 / 0.43 | 1.08 | | |
| G_PS | 0.33 / 0.46 | 0.48 / 0.67 | 1.46 | | |
| G_SS | 0.044 / 0.135 | 0.107 / 0.316 | 2.4 | | |
| B/n | 0.047 / 0.087 | 0.067 / 0.118 | 1.40 / 1.35 | 1.06 / 1.35 | 0.84 / 1.03 |
| P/n^2 | 0.006 / 0.023 | 0.001 / 0.025 | 0.2 / 1.07 | 1.9 / 2.1 | 0.13 / 0.87 |

- The Gaussian absorbs most of the fibre-assignment change and thecov models it (it is built from the altmtl weights):
  G_PS scales as the shot noise (1.46 vs 1.47), G_SS as its square plus the w^4 window (2.4), G_PP rises 8 per cent
  (weight scatter in the W^4 normalisation).
- B/n rises 1.35-1.4x, the tree level predicts 1.76x (w^3 window): fibre collisions remove close pairs, the realised
  one-shared-galaxy term grows less than the weights alone imply. Measured/tree drops from 1.06-1.35 (complete) to
  0.84-1.03 (altmtl).
- P/n^2 is suppressed in altmtl below k ~ 0.25 (0.001 vs tree 0.008 at 0.1-0.2), the signature of the collision
  exclusion at small separation (sub-Poisson pairs); in complete it is 2x tree, as for holi (section 9.4).
- Net: the discreteness NG term relative to the own Gaussian is unchanged (B/n + P/n^2 = 5.3 / 11 per cent of G complete,
  5.3 / 10 per cent altmtl), but its ratio to the tree level is not: on fibre-assigned data the tree overpredicts B/n by
  up to 20 per cent and P/n^2 by up to 8x at k < 0.2 — another reason to take these terms from the catalogue.
- Caveats: one mock, one cap; G_SS and the B/n-P/n^2 split inherit the pair-count noise of the three thecov runs
  (low-k P/n^2 below ~3e-3 of G is noise, hence the negative altmtl values at k < 0.12).

## 16. Measured discreteness for P0, P2, P4 and the optimised one-job pipeline (2026-10-27)

**Method.** `thinning_closed_form --ells 0 2 4`: the one-shared-galaxy (B/n) thinning term is exact for all nine
(l, l') blocks (end-point line of sight, L_l(k.x_g) expanded in k-hat tensors, 24 batched type-2 NUFFTs per shell);
checked against finite differences of the estimator over every galaxy of a toy survey with a varying line of sight
(all blocks to 1e-8). The two-shared-galaxy term (P/n^2, 1-2 per cent of G) stays exact for (0, 0) only; elsewhere it is
the tree term times the measured/tree ratio of (0, 0). thecov's shot-noise dependence g1, g2 now comes from one run:
`GaussianCovariance.covariance_parts` splits G into PP, PS, SS from the same windows (PP + PS + SS = covariance(); a
spectrum with shot noise S' gives PP + (S'/S) PS + (S'/S)^2 SS exactly; test_covariance_parts_sum_and_shotnoise_scaling).

**A bias found on the way.** The earlier three-run fit (thecov at p = 1, 0.5, 0.25) used separately computed pair
counts per run; G_PP differed by 0.5-0.7 per cent between runs, which the fit amplified into g1 and gave a spurious
negative B/n for l = 2, 4. The Abacus split of section 15 has the same construction: its G_PS/G_SS and B/n vs P/n^2 split
carry this bias (the measured lin and quad themselves, and B/n + 3 P/n^2 at p = 0.5, do not).

**Result, holi mock 173 LRG1 NGC (one-run split), band means of the diagonal / G, k = 0.1-0.2 / 0.2-0.3:**
P0xP0 B/n 0.071 / 0.135 (tree 0.063 / 0.082), P/n^2 0.0046 / 0.024 (tree 0.0063 / 0.021). Mock 174 reproduces the
(0,0) block to a few per cent (B/n + 2P/n^2: 0.080 / 0.183 vs 0.079 / 0.174). For l > 0 the measured B/n is below the tree
and negative at k < 0.1 (P2xP2 -0.006, P4xP4 -0.003 to -0.01 of G) in BOTH mocks: a systematic, not realisation noise,
of the size of thecov's own O(1/kL) line-of-sight ambiguity of the shot-noise term (`block()` docstring). Cross blocks
scatter by ~0.005 of G between the two mocks.

Against the 848 mocks (k <= 0.3, with SSC; Wishart (N-1) D, lower is better):

| model | P0 | P2 | P4 | P0+P2 | P0+P2+P4 |
|---|---|---|---|---|---|
| G + tree disc everywhere | 937 | 915 | 830 | 3614 | 8498 |
| G + measured disc everywhere | 896 | 923 | 851 | 3672 | 9890 |
| **G + measured (0,0), tree elsewhere** | **896** | 915 | 830 | **3581** | **8206** |

P0+P2+P4 is positive definite in all of these (min eigenvalue of G^-1/2 C G^-1/2 0.94; the scaled-tree stopgap of
section 12 gave -0.065). Mock/model diagonal for P0 at k = 0.1-0.2 / 0.2-0.3: 1.003 / 1.035 with the measured terms
(Gaussian alone 1.079 / 1.199). Recommendation: measured discreteness in the P0 block, tree for l > 0 until the
l > 0 shot-noise term of thecov is checked against the exact estimator (the thinning linear term can serve as that
check: lin - g1 should vanish for an unclustered catalogue).

**Optimisations (no GPU):** one Gaussian run instead of three (split); `--single` NUFFTs (multipole term 671 -> 368 s,
9e-7 from double); tree T0 on one octant of the angular grid (exact to 2e-16, 8.0x; T0 for 56 bins x 3 ells in 8 s on
64 processes); `catalogue_covariance.py`: all terms of one catalogue in one process with a memoised loader, the
Gaussian computed once through its parts. Timings: see the end of this section.

## 17. Poisson-catalogue null test of the shot-noise term; g1 bug; full timings (2026-10-27)

**Test** (`catalogue_covariance --poisson 1 --no-t0 --no-ssc --ells 0 2 4 --single`, holi mock 173 LRG1 NGC footprint):
galaxies = Poisson draw (p0 = 0.090) from one half of the 4-file randoms (706,984 vs 706,162 real; same weights, n(z)),
the other half as randoms; measured P0 ~ 0 (-27 at k_mid vs 7504). Expected: exact thinning lin_ll' = d/da G(a) = 2 G_SS / r
(P-independent), quad00 = G_SS / r^2, r = S_1 / sum w^2.

**Bug found**: `disc_multipoles.decompose` one-run split used g1 = G_PS/r + 2 G_SS/r^2; the a-linear coefficient of
(1 + a/r)^2 G_SS is 2 G_SS / r (fixed). With the old coefficient lin/pred = 1.09-1.11 = r in every block (Poisson r = 1.090);
with the fix:
- diagonal blocks: lin / (2 G_SS/r) - 1 = +0.4 / +1.8 / +1.5 % (P0, k 0.02-0.1 / 0.1-0.2 / 0.2-0.3), P2 -0.3/+0.8/+1.7 %, P4 -0.2/+1.3/+1.3 %;
  quad00 / (G_SS/r^2) = 1.014 / 1.021 / 1.014;
- residual in units of the real-catalogue G: <= 0.009 (diagonal), <= 0.002 (cross blocks) (`responses/poisson_null/`,
  `report_figures/poisson_null_test_LRG1_NGC.png`).
=> thecov's shot-noise covariance is right for l = 2, 4 (no O(1/kL) LOS problem at this level); a common ~1.5 % excess of the
exact term at k > 0.1, the same at every l (normalisation of the SS window / kernel smoothing; small, state in Paper 1).
The low / negative l > 0 measured B/n of section 16 is therefore NOT a thecov shot-noise artefact.

**Section 16 numbers redone with the fix** (mock 173, end-to-end run catcov/LRG1_NGC_mock173_v2): P0xP0 B/n 0.0022/0.0190/0.0689/0.1249
of G (k bands 0.02-0.05/0.05-0.1/0.1-0.2/0.2-0.3; tree 0.0063/0.0254/0.0637/0.0829), P/n^2 0.0005/0.0057/0.0266 (last three);
P2xP2 B/n -0.0044/-0.0063/-0.0025/+0.0062 (tree +0.0006/0.0039/0.0148/0.0253); P4xP4 -0.0016/-0.0043/-0.0079/-0.0067.
The "three-run fit bias" of section 16 should be re-examined: part of the one-run vs three-run difference was this bug.
Wishart (N-1)D vs 848 mocks, P0 / P2 / P4 / P0+P2 / P0+P2+P4: tree+SSC 929/907/826/3595/8397; measured everywhere+SSC
886/915/846/3647/9846; **hybrid (measured P0xP0, tree elsewhere) + SSC 886/907/826/3560/8131 (best; min eig 0.944)**.
Recommendation unchanged. (`disc_multipoles` now prints tree+SSC and hybrid+SSC.)

**Cold-cache end-to-end timings** (one CPU node, all terms, ells 0 2 4, --single): thin stub 184 s, measured discreteness 441 s,
Gaussian (smoothing 280 s + pair counts) + split + tree disc 765 s, tree T0 8 s, SSC 28 s; total 1426 s (24 min) -> fits the
30-min debug queue. (A harmless jax CUDA_ERROR_NO_DEVICE message is logged on CPU nodes.)

## 18. Code review of the discreteness computation (2026-10-27; `report/discreteness_code_review.md`)

Bugs: (1) quad (exact P/n^2) aliased by the circular FFT correlation for |k + k'| > grid Nyquist (0.305): clustering
part overestimated +6 % at k = 0.22, +26 % at 0.27, +42 % at 0.30 -> measured P/n^2 at 0.2-0.3 (0.027 vs tree 0.021) is
mostly this; B/n low by 2x the excess, hybrid P0 block low by 1x, ratio_off high. Fixed (5e9f97b, --quad-pad 2, brute-force
checked); mock 173 rerun in catcov/LRG1_NGC_mock173_v3. (2) thecov G in the mock comparison uses the device shot noise
(4 random files, 28311) not the files' (27650): G 2.2 % high at k = 0.2-0.3; assemble G(files' S) + B/n + P/n^2 (to do in
disc_multipoles). (3) Structural: G + B/n + P/n^2 = G_PP + (1-1/r) G_PS + (1-1/r)^2 G_SS + lin - quad, so the SUM is
insensitive to thecov's shot-noise terms, but the SPLIT puts thecov's P x S error (untested by the Poisson test) into
"B/n" at full strength: the negative l > 0 B/n is a few-% test of G_PS, not a measurement. (4) Grid shells +-2 % vs
continuum in the first bins (~0.002 of G leak). Block-wise Wishart with T0 and next steps (GRF-Poisson test of G_PS,
MC thinning for l > 0 quad, catalogue P in thecov) in the report.

## 19. Code review of the tree-level T0 (2026-10-27; `report/t0_code_review.md`)

Kernels and quadratures verified (F3, G3 vs the P13 kernels to 1e-6; Z1, Z2 vs tree.responses to 1e-16; LOS and mu12 rules
converged < 1e-5 with FoG; n_k = 1 biases the diagonal low by 1-6 %, use n_k = 2). **The T0 validation is parameter-limited:**
b1 = 2.202 comes from a pure-Kaiser fit to the window-convolved P2/P0 at k <= 0.06 and A = 0.739 absorbs the mismatch
(sqrt(A) b1 = 1.893; flagged 2026-10-07, never changed). With b1 = 1.893, A = 1 the T0 changes by x0.6 (P0 hard), x1.15 (P2 hard),
x1.3-1.4 (squeezed P2, P0xP2) with Lazeyras b2 = -0.262, or x1.5 everywhere with the production b2, bs2
(`responses/t0_review/t0_parameter_split_sensitivity.csv`): the section 13 amplitudes (1.00 / 1.6-1.8 / 0.5) are conditional on
the split, and the Lazeyras-consistent one turns the ell-dependent tension into a common ~1.5-1.65 (nonlinear P in the kernels
can do that). J4 = int W^4 / norm^2 uses the local window (likely 20-40 % high; the Gaussian's I_k / int W^2 is 0.77-0.85).
Next: window-convolved (b1, sigma_v) fit to the catalogue with A = 1 as check, b2 from R1; J4 cell-size convergence; n_k = 2;
redo sections 13 and 16 tree terms.

## 20. Code review of the SSC (2026-10-27; `report/ssc_code_review.md`)

Algebra already verified (2026-10-07 review). New: (1) the production C_ssc = clustering SSC (rank 3, P2 up to 3.4 % of
G at k 0.2-0.3, P0 <= 0.4 % except 2.6 % at 0.2-0.3) + Poisson local-average term (rank ~1 along P_hat, -0.6 to -2.3 % of
the P0 diagonal, -5 % of the coherent P0 mode). Wishart vs 848 mocks: the clustering part is wanted by P2 (1013 -> 914),
the Poisson LA part worsens every fit (P0 875 -> 896); a box simulation (3000 realisations) confirms the Poisson LA sign
and size (-1.9e-4 vs -4/N = -1.3e-4 P_i P_j, the excess being the collapsed-bispectrum part), so the coherent-P0 budget
(Poisson LA, SSC with tree vs measured responses, squeezed T0) is what is open. (2) sigma^2 uses A_norm x P_lin
(A = 0.74): the long-mode variance should use the fiducial P_lin -> clustering SSC x 1.35 (mildly preferred by the
mocks). (3) No hook for measured responses (R_0 is 1.27x tree, R_2 1.2-1.4x): add `responses=`. (4) Minor: B_c part of T
lacks d_k; no I_W consistency assert; lam = 2, 4 channels of sigma^2 and the far-pair sampling convergence untested; the
mocks lack radial long modes > 250 Mpc/h so the model SSC is not what the mocks contain. Next steps (report 4.1-4.6):
measured Poisson-LA term from the thinning vectors (no new job), p_lin_long + responses hook, per-mock long-mode
projections vs sigma2 (mock-fair SSC), GRF long-mode test of the lam > 0 channels, GRF-Poisson realised-norm test.

**Light checks (debug queue, same day).** (a) Padded vs aliased quad on the cached mock-173 inputs (`thinning/LRG1_NGC_mock173_quadpad`,
131 s for the padded term): clustering part aliased/padded = 1.00 / 1.08 / 1.31 / 1.51 at k = 0.17 / 0.22 / 0.27 / 0.30. P/n^2 (0,0)
diag at 0.2-0.3: 0.0266 -> 0.0212 of G (tree 0.0209); B/n 0.1249 -> 0.1357 (tree 0.0829; 0.1-0.2: 0.0690 vs 0.0637); B/n + P/n^2
0.1515 -> 0.1569; ratio_off 1.20 -> 1.14. So the measured two-shared-galaxy term agrees with tree to 1-2 % at k > 0.1 and the
excess over tree is entirely in the one-shared-galaxy term (1.08x at 0.1-0.2, 1.64x at 0.2-0.3). Hybrid + SSC with the files'
S: P0 mock/model 0.2-0.3 1.071 -> 1.066 (1.044 with tree T0); P0+P2+P4 Wishart 8119. (`responses/disc_multipoles/quad_padded_vs_aliased_*.csv`)
(b) Monte-Carlo thinning, p = 0.25 (a = 3), l = 0 2 4, 62 thinnings in 19 min (`thinning/LRG1_NGC_mock173_mc024`): MC covariance vs
a lin_exact + a^2 [quad_exact (0,0) | G_SS/r^2 + tree P/n^2 elsewhere]: diagonal blocks agree to (-16 +- 8, -6 +- 6, +4 +- 5, -1 +- 4) %
for P0, (+10 +- 10, -1 +- 6, -1 +- 5, -5 +- 4) % for P2, (-3 +- 8, -8 +- 6, -1 +- 5, -3 +- 4) % for P4; cross blocks within 1-3 sigma
(P0xP4 0.1-0.3 at -3.2, -2.4 sigma: noise or the tree P/n^2 proxy). The exact l > 0 linear term is validated end to end on the real
footprint at the ~5 % level of the thinning covariance. The MC cannot resolve the l > 0 P/n^2 (needs ~2000 thinnings at p = 0.25,
~450 at p = 0.1); the exact term for l > 0 remains the configuration-space route. (`responses/disc_multipoles/mc_thinning_vs_exact_*.csv`)

**Tests run (2026-10-27, local, `tests/test_ssc.py`)**: (a) `test_sigma2_thin_shell_against_fourier_space`: thin shell
segment with a 50-deg LOS spread and radial weights, all ten sigma^2_{(X n)(Y n')} (X, Y = W, M; n = 0, 2) against the exact
Fourier-space sum int P |W_X W_Y^*| with NUFFT windows: thecov / exact = 0.990-0.992 for every channel, the 1 % being the
far-pair sampling (changes by +-1 % with n_sub) -> lam = 2, 4 channels, cross channels and W/M ordering verified. (b) Poisson
local-average term in a box (realised N^2 normalisation vs mean, sigma_delta 0.26, 6000 realisations): (Cov_real - Cov_fixed)
/ (P_i P_j) = 1.06 +- 0.20 x (-4/N), Cov(P_hat, eps) = (0.9-1.2) x 2P/N -> the term is right (marked slow). (c)
`discreteness_integrals` on a uniform cube: var N = 4.000, J = 2 nbar, J3 = 2 nbar^2; alpha sum w W = int W^2 to 1e-5.
No bug found in ssc.py. Code changes: `p_lin_long=` (sigma^2 from the fiducial P_lin; catalogue_covariance now passes it:
clustering SSC x 1.35), `responses=` override ({(ell, n): per-bin}), d_k on the collapsed-bispectrum part of T, I_W
consistency check (RuntimeError). ssc_check.py still passes A_norm P_lin to sigma^2 (add p_lin_long there before the next
production run).
(c) Full one-job rerun with the padded quad (`catcov/LRG1_NGC_mock173_v3`, 1484 s total, measured-discreteness stage 441 -> 540 s)
reproduces the light check exactly: P0xP0 B/n 0.0022/0.0190/0.0690/0.1357, P/n^2 0.0000/0.0005/0.0056/0.0212; Wishart with the
files' S (P0 / P2 / P4 / P0+P2 / P0+P2+P4): tree + SSC 927/902/826/3590/8417, measured + SSC 881/910/843/3631/9827,
hybrid + SSC 881/902/826/3548/8119 (min eig 0.943). Use v3 as the reference mock-173 catalogue covariance from here on.

## 21. T0 validation redone with catalogue-consistent inputs (2026-10-27; `t0_inputs_fit.py`, `responses/t0_review/`)

**Inputs through the DESI window matrix** (`cutsky_window_matrix_validation/holi-v3-altmtl/mock0/window_mesh2_spectrum_poles_LRG_z0.4-0.6_NGC_..._smooth_mesh.h5`,
observable 3 x 418 bins, theory 3 x 501; `t0_inputs_fit.py --stage fit` dumps a slim copy and does the fit; refit locally at
k = 0.02-0.08 with a 2 % model-error floor, `window_fit_LRG1_NGC.csv`): mock mean, A free: b1 1.93, sigma_v 3.75, A 0.944
(chi2 2/24); A = 1: b1 1.87, sigma_v 4.25. Mock 173 alone, A = 1: b1 1.87, sigma_v 3.75 (A free is degenerate on one mock:
b1 2.18, A 0.75). The production b1 = 2.202 / A = 0.739 was the unconvolved-Kaiser artefact; the window LOWERS the model P2/P0
at k < 0.08 (0.39 -> 0.35-0.38) while the data sit at 0.42-0.47. **Catalogue recipe: A = 1 (fiducial P_lin), b1 from the
window-convolved P0, sigma_v from P2/P0: b1 1.87, sigma_v 3.75.** b2: Lazeyras(1.87) = -0.29 makes the tree T0 tiny/negative
(P0 diagonal -1.5 % of G at k < 0.1); b2 from the measured P0xP0 B/n against the tree B/n (linear in b2) gives 0.52 at
k = 0.1-0.2 but 2.4 at 0.2-0.3 and 1.8 from the off-diagonals: the collapsed bispectrum is beyond tree level at k > 0.2.
Adopted b2 = 0.52 (k = 0.1-0.2), bs2 = -4/7 (b1 - 1) = -0.50.

**J4 is inflated by the randoms' density noise** (`--stage j4`, 18 random files split in 4 disjoint quarters, sum_c W1 W2 W3 W4 V_c
unbiased): int W^4 = 4.70 / 4.30 / 4.10 / 3.85e-8 at cells 3 / 10 / 20 / 40 Mpc/h (naive single-field estimate 9.3e-7 at
3 Mpc/h: pure noise) vs 6.97e-8 implied by the production J4 = 9.70e-10 (norm 8.476). J4_eff = int W^4 / norm^2 = 5.7e-10 at
20 Mpc/h (0.588 x production; 0.62 at 10, 0.55 at 40). int W^2 cross = 8.83 -> 7.73 (3 -> 80 Mpc/h) against the estimator's
norm 8.476. `TrispectrumCovariance.window_integral` (alpha sum_r w_r W_r^3 with the per-random density) should be replaced by
a cell/quarter estimate or a kernel-smoothed W.

**Block amplitudes** (`t0_tree_fit.block_amplitudes`: rebinned to dk 0.025, correlation units, base = G + measured
B/n + P/n^2 of catcov v2 [one-run split, quad still aliased at k > 0.2], G at the device S; T0 with n_k = 2 and J4_eff;
`t0_block_amplitudes_newpar_LRG1_NGC.csv`, `report_figures/t0_block_amplitudes_inputs_LRG1_NGC.png`). Hard (k >= 0.15,
off-diagonal) / squeezed (k < 0.1 x k >= 0.1), base without / with the clustering SSC (positive eigen-part of C_ssc):
- production inputs: P0xP0 2.1 / 1.2, P2xP2 3.4 / 2.4, P0xP2 0.4 / 2.3; squeezed P0 0.9 / 0.7, P2 1.2 / 1.1, P0xP2 0.6 / 0.9
  (the section 13 "P0 hard = 1.0" was the inflated J4 compensating).
- b1 1.87, b2 0.52, sigma_v 3.75, A 1, measured P inside the kernels: P0xP0 0.95 / 0.54, P2xP2 2.7 / 1.9, P0xP2 0.75 / 2.6;
  squeezed P0 0.56 / 0.46, P2 1.07 / 0.92, P0xP2 0.55 / 0.78.
- b1 1.87, Lazeyras b2 -0.29: 3-5 everywhere in the hard blocks (the template nearly vanishes).
The hard-block result is a test of tree-level PT plus b2, not of the code (kernels and quadratures verified, section 19); the
mocks' off-diagonal correlation excess at k >= 0.15 is 0.14 (P0xP0) / 0.26 (P2xP2) without and 0.07 / 0.17 with the clustering
SSC in the base, against 0.04-0.11 from any tree T0 with J4_eff. The P2xP2 hard block wants 2-3x tree under every input choice;
the squeezed P0 block stays at 0.5 (section 13 conclusion unchanged); squeezed P2 and P0xP2 move to ~1 with the new inputs.
Section 16 tree discreteness terms and the SSC responses must also be recomputed with (b1, b2, bs2, sigma_v) = (1.87, 0.52,
-0.50, 3.75), A = 1.

**Decisive next step**: the 25 Abacus c000 LRG z = 0.5 cubic boxes (3 LOS each; `mock_challenge_box/abacus-hf-v2`, with the
sugiyama-diagonal bispectrum for b2): no window (J4 = 1/V exactly), no SSC, b1 from P0 with A = 1, b2 from the bispectrum, exact
Gaussian from the box mean P(k, mu), tree discreteness with the known nbar; what is left is T0 alone, hard and squeezed.

**Long-mode content of the mocks vs sigma2 (2026-10-27, `long_mode_check.py`, from nz_scatter_LRG1.npz + the ssc npz)**:
M-weighted survey-mean galaxy overdensity over 989 mocks: std 0.448 % (Poisson 0.125 %, clustering 0.430 %) vs thecov
g0^2 s_00 + 2 g0 g2 s_02 + g2^2 s_22 = 0.476 %: the mocks hold 82 % of the model's survey-mean variance (tree g_n, isotropic
P_lin). thecov's s_22 / s_00 = 0.65 (wide footprint: the LOS varies, so even radial modes project partly off L_2). An upper
bound treating all of it as locally radial (nu = 1, lam_W = 2) gives an SSC 1.3x (P0) - 1.8x (P2) thecov's clustering part
at k > 0.1 and a better Wishart (P2 914 -> 889; with responses x 1.3: 862) -- the n(z) data fix the variance, not its
n = 0/2 split, so this only says the mocks want the larger end. Measured Poisson-LA term and SSC pieces: job
catcov/LRG1_NGC_mock173_v4 (regular queue).

## 22. T0 in the Abacus cubic boxes, window-free (2026-10-28; `box_t0.py`, `box_thinning.py`, `responses/box_t0/`)

Data: 25 AbacusSummit base c000 LRG z = 0.5 HOD boxes (hod-base_B, L = 2000 Mpc/h, N = 7.27M, nbar 9.1e-4), 3 lines of sight
each, P0/P2/P4 at dk = 0.001 (rebinned to the 56 bins) and the Sugiyama (0,0,0), (2,0,2) diagonal bispectra
(`mock_challenge_box/abacus-hf-v2`; `responses/box_t0/box_LRG_z0.500_c000_hod-base_B.npz`). c000 P_lin at z = 0.5 from CAMB
(sigma8 0.6214, f 0.7623).
- Inputs from the boxes themselves (A = 1): b1 1.855, sigma_v 4.25 from P0, P2 at k 0.02-0.08 (chi2 9/24, 1 % floor);
  b2 = 0.38 from B_000(k) at k 0.03-0.08 with the tree RSD bispectrum (+FoG, bs2 = -4/7(b1-1)): data/tree 1.01 (k < 0.05),
  1.04 (0.05-0.1), 1.22 (0.1-0.15). Lazeyras(1.855) = -0.32: the HOD galaxies' b2 is positive, consistent with the cutsky
  B/n value 0.52 (section 21). The holi/Abacus LRG inputs agree (cutsky b1 1.87, sigma_v 3.75).
- Exact box Gaussian (diagonal in k; 2 (2l+1)(2l'+1)/N_k int L_l L_l' (P(k,mu) + 1/nbar)^2 with the box-mean multipoles,
  N_k = all modes): sample variance / Gaussian = 0.82-1.0 (P0), 0.97-1.09 (P2), 0.98-1.10 (P4) in k bands (75 correlated
  samples; the P0 k < 0.1 deficit is 2 sigma).
- Off-diagonal (k != k') covariance is non-Gaussian only: tree discreteness (J_SWW = 1/(nbar V), J_SS = 1/(nbar^2 V)) + T0
  (J4 = 1/V, exact). Rebinned to dk 0.025, correlation units, 75 samples (per-element noise 0.12), nonlinear P in the kernels
  (P0/K0 P_lin smoothed: 1.04 at k 0.1, 1.45 at 0.2, 1.82 at 0.3):
  P0xP0 hard (k >= 0.15): excess over Gaussian 0.25, tree disc 0.18, T0 template 0.22 -> T0 amplitude 0.29 +- 0.11 with disc at
  tree (joint fit: a_disc 1.03 +- 0.46, a_T0 0.17 +- 0.24, corr -0.94; with linear P in the kernels: a_disc 2.7, a_T0 ~ 0 or
  1.2 +- 0.2 with disc fixed). **P0xP0 squeezed (k < 0.1 x k >= 0.1): excess 0.05, disc 0.04, T0 template 0.23 -> amplitude
  0.05 +- 0.08: the tree squeezed T0 is rejected with no window and no SSC** (cutsky section 13: 0.5). [Corrected 2026-10-31: use the los-z-only
  value 0.12 +- 0.14 (~6 sigma); the three LOS of a box are 92 % correlated at k < 0.1, so '~10 sigma' from 75 samples overstated it.]
  P2xP2: hard 0.80 +- 0.30 (75) / 1.19 +- 0.44 (los-z only), squeezed 0.80 +- 0.22 / 1.1 +- 0.3, joint fit a_T0 1.3 +- 0.4
  (disc unconstrained). P0xP2 blocks: small negative excess (-0.02 to -0.05) against T0 0.01-0.07.
  (`box_t0_block_amplitudes_LRG_c000.csv`, `report_figures/box_offdiagonal_LRG_c000.png`.)
- Reading: the cutsky P2xP2 "2-3x tree" of section 21 is not reproduced in the boxes (0.8-1.3): it is cutsky-specific
  (window / LOS / SSC / the l = 2 Gaussian), not T0. The P0 hard block is degenerate between B/n (nonlinear bispectrum) and T0;
  the squeezed T0 is a tree-level failure for these galaxies, not a window or SSC effect.
- Pending: `box_thinning.py` (exact lin, quad of box ph000 los z; regular-queue job 75c8b6ca / SLURM 60414433, ~1 h) gives the
  measured discreteness in the box; Cov_box(k != k') - (lin - quad) = T0 exactly in the (0, 0) block, which breaks the
  P0 hard degeneracy. Then: all 25 boxes' measured disc is unnecessary (weak realisation dependence); the three LOS of a box
  are correlated, so quote the los-z-only errors.

**Mock 174 with the padded quad, and the realised-P leakage confirmed (2026-10-28).** `catcov/LRG1_NGC_mock174_v3` (its SSC stage
fails the ssc.py window-consistency check "alpha sum w W = 10.43 vs int W^2 = 8.49", ratio 1.228 = int m^2 / norm; mock 173 passes
in the same code: open, SSC side). Decomposed with the SAME thecov Gaussian as mock 173: P/n^2 (0,0) identical to 0.5 %
(0.0212 vs 0.0213 of G at 0.2-0.3); B/n (0,0) 0.0690/0.1357 (173) vs 0.0677/0.1267 (174) at 0.1-0.2/0.2-0.3; P2xP2 0.0010/0.0240 vs
-0.0002/0.0119; P4xP4 -0.0065/0.0021 vs -0.0103/-0.0130. The differences follow the realised spectra (174's P0 0.9 % and P2 16 %
below 173's at 0.2-0.3; lin(0,0) 0.8 % lower): finding 3 of section 18 at work, through P2/P4 as much as P0.
**`--own-power`** (thinning_cov thecov stage / catalogue_covariance): thecov's Gaussian with the mock's own measured P_l (file row
of that mock; the Paper 2 rule). Rerun of the thecov stage for both mocks (`catcov/LRG1_NGC_mock17{3,4}_own`): the cross blocks and the
low-k diagonal blocks now agree between the mocks to ~0.003 of G (thecov's pair-count sampling floor) and the cross blocks follow
tree level (P0xP2 0.0091/0.0081 vs 0.0087/0.0053, tree 0.0102/0.0091); (0,0) B/n 0.0701/0.1363 vs 0.0651/0.1276; the high-k
P2xP2 (0.0217 vs 0.0141; tree 0.0253) and P4xP4 (-0.0003 vs -0.0124; tree 0.0124) still differ by ~0.01 of G between the mocks.
Robust in both mocks and both P inputs: negative l > 0 B/n at k < 0.1 (P2xP2 -0.004/-0.005 to -0.006, P4xP4 -0.004 to -0.009 of G).
Candidates: thecov's P x S term for l > 0 (GRF-Poisson test queued) or fibre assignment (altmtl removes close pairs transverse
to the line of sight, an anisotropic one-shared-galaxy term: Abacus complete vs altmtl, queued, decides).
(`responses/disc_multipoles/bn_own_power_173_174.csv`)

**Measured Poisson local-average term, mock 173 (2026-10-28, catcov/LRG1_NGC_mock173_v5; W(x_g) = NX WEIGHT_FKP <WEIGHT>)**:
Var(eps) pieces x N_gal = 1.165 (alpha) / 1.155 (cross) / 1.146 (data in norm), total 4.62/N vs tree 4.18/N (the data's
<w^2>/<w>^2 = 1.165 vs the randoms' 1.05); T_0 measured / tree = 1.02 / 0.94 / 0.93 (k 0.05-0.1 / 0.1-0.2 / 0.2-0.3), T_2
1.02 / 0.92 / 0.97. Resulting LA covariance on the P0 diagonal: measured -0.013 / -0.035 / -0.054 of G vs tree -0.014 /
-0.043 / -0.068; coherent P0 mode -5.1 % (measured) vs -5.2 % (tree). => the tree Poisson-LA term is confirmed from the
catalogue (10-20 % high at k > 0.1 because of the randoms' W vs the galaxies' pair density; use the measured one). The
first attempt with W = NX w_g (the galaxy's own weight twice) gave var_WW 2.25/N: wrong, fixed.
The v4/v5 Gaussian uses sigma2 from the fiducial P_lin (clustering SSC x 1.35) and the padded P/n^2: (0,0) P/n^2 at
0.2-0.3 is now 0.0212 of G (tree 0.0209; the earlier 0.027 was the aliasing), ratio_off 1.12, B/n (0,0) 0.069 / 0.136
(tree 0.064 / 0.083). Wishart (N-1)D vs 848 mocks, G(files' S) + hybrid disc + ..., P0 / P2 / P4 / P0+P2 / P0+P2+P4:
no SSC 872 / 1003 / 827 / 3624 / 8193; + clustering SSC 844 / 882 / 826 / 3485 / 8057; + tree Poisson LA 866 / 884 / 826 /
3531 / 8102; + measured Poisson LA 860 / 883 / 826 / 3520 / 8092; + measured LA + tree T0 with the squeezed blocks halved
851 / 838 / 791 / 3489 / 8470 (min eig 0.36: the low-k T0 structure). P0 diagonal mock/model at k > 0.1: clustering SSC
only 0.986 / 0.986 (model high); + LA 1.018 / 1.032; + LA + T0 0.986 / 1.011; T0 without LA 0.956 / 0.967. So the P0
budget at k > 0.1 closes with G + measured disc + clustering SSC + Poisson LA + hard T0; the Wishart preference against
the LA term in LA-only models is the missing hard T0. Files: `catalogue_covariance_LRG1_NGC_mock173_v5.npz`,
`wishart_LRG1_NGC_mock173_v5.csv` (artifacts).

**Abacus complete vs altmtl with the padded quad and the one-run split (2026-10-28; `catcov/LRG1_NGC_mock0_abacus-{complete,altmtl}_v3`,
`responses/abacus_disc/abacus_fibre_discreteness_padded_LRG1_NGC.csv`; supersedes the section 15 split).** Units: the COMPLETE
catalogue's Gaussian diagonal; bands 0.02-0.05 / 0.05-0.1 / 0.1-0.2 / 0.2-0.3; tree with the complete-catalogue params for both.
- P0xP0: B/n complete 0.0039/0.0185/0.0603/0.1076 (tree 0.0042/0.0172/0.0464/0.0659: 1.3x / 1.6x at k > 0.1); altmtl 
  -0.0009/0.0169/0.0712/0.1301 (tree 0.0073/0.0304/0.0819/0.1164: 0.87x / 1.12x). P/n^2 complete 0.0179 vs tree 0.0111 (1.6x) at
  0.2-0.3; altmtl 0.0313 vs 0.0292 (1.07x; holi altmtl 1.01x).
- P2xP2: B/n complete +0.0023/+0.0032/+0.0170/+0.0433 (tree 0.0104/0.0202 at k > 0.1: 1.6-2.1x); altmtl -0.0022/-0.0087/0.0000/+0.0272
  (tree 0.0184/0.0356). P4xP4: complete -0.0017/+0.0022/+0.0117/+0.0273 (tree 0.0035/0.0101: ~3x); altmtl -0.0069/-0.0094/-0.0108/-0.0083.
=> **Fibre assignment flips the l > 0 one-shared-galaxy term from positive (1.6-3x tree, like the P0 excess) to zero / negative**, and
removes the P/n^2 excess. Same footprint, same code, same windows; the complete catalogue has MORE galaxies (941k vs 704k) and less
shot noise, so this is not a thecov P x S artefact: fibre collisions remove close pairs preferentially transverse to the line of
sight, which (i) removes the one-halo pairs that drive the l = 0 excesses over tree and (ii) makes the collapsed bispectrum
anisotropic with a negative L_2, L_4 weighting. The negative l > 0 B/n of the holi (altmtl) mocks 173/174 (section 18, own-power:
P2xP2 -0.004/-0.005, P4xP4 -0.004 to -0.012 of G) is the same signature. Consequence for the DESI covariance: the l > 0
discreteness terms of a fibre-assigned catalogue are NOT tree level (tree overpredicts P2xP2 by ~0.02-0.04 G at k > 0.1 and P4xP4
by 0.01-0.03 G); the measured terms carry it, but with the realisation scatter of section 18 (~0.01 G at k > 0.2 between mocks).
The mock comparison (hybrid best, measured l > 0 worse) then says the mocks' extra l > 0 variance is elsewhere (T0, P2 hard block
1.6-1.8x tree in section 13), not in B/n. GRF-Poisson test of thecov's P x S term still queued (two power inputs, sigma_s 0.3).

**Box thinning done (job 75c8b6ca, 302 s on one node; `responses/box_t0/box_thinning_ph000_z.npz`).** Caveat: the
`abacus_HF_cubic/DR2_v1.0` HODDIES LRG catalogue of ph000 has 5.60M galaxies (nbar 7.0e-4) while the pipeline's hod-base_B
spectra have 7.27M (nbar 9.1e-4) and P0 3.2 % higher: a different HOD realisation. lin was scaled by (nbar_cat/nbar) (P_pipe/P_cat)^2
= 0.82 and quad by (nbar_cat/nbar)^2 (P_pipe/P_cat) = 0.61; the NUFFT P0 of the catalogue matches the pipeline shape to 0.6 % rms.
- Measured discreteness = tree with nonlinear P inside the kernels: off-diagonal k >= 0.15 means in correlation units
  P0xP0 0.0352 vs 0.0361, P2xP2 0.0076 vs 0.0070; P/n^2 measured / tree 1.044; diagonal P0 0.046 vs 0.043 of G at 0.2-0.3
  (P2 0.014 vs 0.009, P4 0.009 vs 0.003). The HOD galaxies' discreteness terms are Poisson-like at the 5 % level, and the
  prescription "tree B/n, P/n^2 with the measured nonlinear P" is validated in the box.
- **T0 isolated (base = exact Gaussian + measured discreteness; `box_t0_block_amplitudes_measured_disc_LRG_c000.csv`):**
  P0xP0 hard (k >= 0.15, off-diagonal) 0.31 +- 0.11 of the tree T0 with nonlinear P (0.67 +- 0.23 of the tree with linear P);
  los-z only 0.15 +- 0.21 / 0.35 +- 0.42. P0xP0 squeezed 0.06 +- 0.08 (0.12 +- 0.14): excluded. P2xP2 hard 0.77 +- 0.30
  (1.16 +- 0.44 los-z), squeezed 0.79 +- 0.22, adjacent 1.3 +- 0.5. P0xP2 blocks consistent with zero (negative, 1 sigma).
- Diagonal: mocks / (G + measured disc) = 0.81 / 0.94 / 0.88 (P0, k bands), 0.97 / 1.00 / 1.07 (P2), 0.98 / 1.01 / 1.03 (P4):
  [Corrected 2026-10-31] per-LOS variance over the 25 independent boxes / exact Gaussian, bootstrap errors: P0 los z
  0.84 +- 0.05 / 0.99 +- 0.06 / 0.99 +- 0.07 (k 0.02-0.1 / 0.1-0.2 / 0.2-0.3; x, y similar), P2 and P4 consistent with 1.
  No P0 deficit at k > 0.1; at k < 0.1 a 2.3 sigma low fluctuation of these 25 ICs (P0 92 % correlated across LOS, so one
  detection, not three). The 'sub-Poisson' explanation is withdrawn (shot noise < 5 % of P at k < 0.1). The 848-mock cutsky
  P0 diagonal (+-0.5 %, section 16) remains the precise test and agrees.
- Conclusion for Paper 1: in a box with no window and no SSC, the tree-level T0 with nonlinear P is right for P2xP2 (0.8-1.2),
  too large by ~3x for the P0xP0 hard block and by >10x for the squeezed P0xP0 block; with linear P the P0 hard block is 0.7
  +- 0.2. The cutsky "P2xP2 hard wants 2-3x" (section 21) is therefore not T0. Next: a second box (ph001) to check the
  realisation dependence of the measured discreteness and the P0 diagonal deficit; a sub-Poisson S test (fit S_eff from the
  box P0 diagonal); then the cutsky P2 excess with the box-calibrated T0 (hard P0 x0.3-0.7, squeezed x0, P2 x1).

### b2 from the catalogue's own bispectrum (2026-10-31; `t0_inputs_fit --stage bispectrum`, `responses/t0_review/bispectrum_inputs_mock173.npz`)
Mock 173 LRG1 NGC Sugiyama diagonal B_000 (41 bins to k 0.2) fitted with the tree RSD bispectrum (b1 1.87, sigma_v 3.75, f 0.762,
undamped holi P_lin, bs2 = -4/7(b1-1)); B_202 is too weak to constrain b2 (box: -0.18 +- 0.17) and not used.
- Window: no bispectrum window matrix exists. The P window matrix row sums give d2 = 1.020 (k 0.05-0.2), reproduced by the
  unbiased cell int W^2 / P norm at 5-10 Mpc/h (1.028 / 1.009), i.e. the estimator's norm is within 2 % of the window at the mode
  scale. Cell int W^3 moves ~2x as much as int W^2 over the same range, so d3 = 1.00 +- 0.05 (systematic). (The file's B norm
  1.288 is in a different unit convention from the cell int W^3; the measured B_000 equals the Abacus box mean to 3-7 % at
  k 0.05-0.08, so the norm is physically consistent.)
- Errors: one Abacus box's B_000 scatter x sqrt(V_box / V_eff), V_eff = (int W^2)^3 / (int W^3)^2 = 1.75 (Gpc/h)^3 (x2.14).
  Validated on the boxes: B_000 fit recovers b2 = 0.38 +- 0.02 (mean of 25), single box 0.385 +- 0.063.
- **Result: b2 = 0.75 +- 0.11 (stat) +- 0.16 (d3) at k 0.04-0.1** (0.73 at 0.04-0.08, 0.76 at 0.03-0.1; 1.0 with chi2 16/14 to
  k 0.12, tree breaking). Strongly degenerate with b1 and sigma_v: b1 1.83 / 1.91 -> 0.92 / 0.59; sigma_v 2.5 / 5.0 -> 0.63 / 0.92.
  Total ~ +-0.3. Consistent with b2 from the measured B/n (0.52, k 0.1-0.2) and with the Abacus boxes (0.38-0.46 at the same k).
  All three exclude Lazeyras(b1) = -0.29 at > 3 sigma.
- Consequence for T0: the catalogue fixes b2 to 0.4-0.75; across that range the T0 template changes by ~ +-25 % (b2 0.28 -> 0.52
  raised the cutsky hard P0 template by 1.3x, section 21), and a larger b2 makes T0 larger, i.e. the box P0 hard block (wants
  0.3 of tree) gets worse, not better. No bias value inside the catalogue's allowed range removes the box pattern (P0 hard 0.3,
  P0 squeezed ~0, P2 ~1): this is a model failure, to be addressed by model changes tested in the box (internal-leg damping,
  measured-response squeezed limit, measured P throughout), not by bias tuning.
