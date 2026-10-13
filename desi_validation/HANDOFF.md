# Handoff: state of the DESI validation of thecov and what to do next

Start here. Branch `thecov2`. First written 2026-10-05; sections 1-5 rewritten 2026-10-12 to the current state
(the dated sections 6-9 are the history, with notes where a later result superseded them; section 9 is the newest).
Everything is reproducible from files in the repository and the NERSC outputs in
`/global/cfs/cdirs/desicollab/users/oalves/thecov_validation` (logs, ssc_*.npz, report_data_*.npz, bundles) and
`~/thecov_desi/<sim>/` (pair counts, covariances, kernel caches).

Workflow from 2026-10-12: the data are being copied to the user's machine and the analysis continues with a local
Claude Code session (section 10: what is needed locally). NERSC remains the place for pair counts of new windows.

Reading order: this file (sections 1, 3, 9); `report/ssc_t0_review.md`; `report/ssc_formalism.pdf`;
`report/window_kernel.typ` (the pair-averaged window); `paper/README.md` (the JCAP draft and its notebooks:
`paper/paper_nersc.ipynb`, figures from the original files; `paper/thecov_walkthrough.ipynb`, every term computed with
explicit thecov calls and compared with the mocks). Superseded: `report/report.tex` section 6, `report/proposal_lowrank.md`
(the low-rank term is now the SSC term).

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

5. **Tree-level T0 is rejected by the mocks** (holi and Abacus complete N-body, sections 7.1-7.8, 9.4): best amplitude
   0.26-0.5 (LRG1, row-normalised mixing), not PD at amplitude 1 in local form. It is not the missing LRG1 variance.

6. **12 defective LRG holi mocks** (n(z) spike at 0.46 < z < 0.47; section 8.4) are excluded everywhere (848 LRG1 mocks).

7. Residual of the recommended model (kernel Gaussian + SSC + window-convolved disc, LRG1 NGC, walkthrough): chi2/dof 1.02
   (k_max 0.12) / 1.03 (0.3); after SSC + disc the worst whitened directions carry +25-40% (out of sample); A2 variance
   still 1.10-1.17x at k_max 0.2-0.3, constant shot-noise parameter 5-15% too large around 0.2. Item 4 changes this
   model: redo the budget with the local (or correctly normalised) SN-NG terms at their measured amplitude.

8. Still valid from the 2026-10-05 analysis: the bin-width (a + b dk) split of `report/report.tex` is contaminated
   (adjacent 0.005 bins correlate at 0.4, so a fully correlated term shows up ~2/3 as "a"); the low-rank off-diagonal
   excess of `report/proposal_lowrank.md` (sigma_P0 0.51 / 0.66%, P2 mode 0.94 / 1.22%, LRG1 NGC / SGC) is the SSC term.

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
line-of-sight approximation). See report/bispectrum_implementation.md section 4 for the candidates and the next test.
Corrections to the notes found on the way: none to the physics; Q_n^box = V sqrt(N/4pi) H as the guide says.
