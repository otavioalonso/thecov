# Handoff: state of the DESI validation of thecov and what to do next

Start here. Written 2026-10-05 on branch `thecov2` by the cloud session that built `WindowSmoothing`
and did the low-rank analysis. Everything below is reproducible from files in the repository and the
`report_data_*.npz` dumps in `~/thecov_desi/holi_v3_mock173` at NERSC (copies of logs and small outputs
go to `/global/cfs/cdirs/desicollab/users/oalves/thecov_validation`).

Reading order: this file; `report/proposal_lowrank.md` (the current physics conclusion);
`report/kernel_validation_results.md` (the kernel runs); `report/report.tex` (the original report, whose
Section 6 interpretation is now superseded); `report/window_kernel.typ` (the derivation of the
pair-averaged window).

## 1. Conclusions, in order of confidence

1. **thecov's Gaussian covariance is right for the window it is given.** QSO validates to 1% in chi2/n,
   per-element variances and parameter errors. The GRF test on the real LRG1 footprint
   (`run_gaussian_footprint.py`) reproduces thecov to 0.3% with thecov's own window.

2. **The local window W = m^2 underestimates the LRG1 Gaussian variance by ~4-5%**, because the
   veto-hole fraction varies across the footprint (mostly on Galactic-latitude scales; `hole_fraction.py`
   shows R_PP = <f^2>/<f>^2 flat from 7' to 1 deg and rising beyond). Three independent estimates agree:
   GRF with a fill-resolved window (+5-7%, ~1% of it Poisson noise of the maps), thecov with the fill
   window at nside 512 (+4.5%), the hole-fraction statistic (+3-4%). Any hole-averaged window recovers it.

3. **The principled window is the xi-kernel (`thecov.WindowSmoothing`)**: W_k = m (K_k * m) with
   K_k = xi_0(r) j_0(kr)/P_0(k), per k-bin, two-stage basis (19-21 kernels -> 8-9 windows, errors < 2e-3),
   no free scale. After removing the low-rank term (item 4) it leaves the flattest residual of all windows:
   within +-3% of the mocks at k < 0.2 for all ell, both caps (`rank1_excess.py` on holi-kcore2-LRG1).
   The fill window at nside 64-128 gets a lower chi2/n only by absorbing part of item 4; do not tune it.
   The "I_k error" the earlier sessions chased (kernel 1-1.5% above the GRF's I_k at k > 0.1) is real but
   second order: the kernel's small-r mass comes from an extrapolated P_0 without fingers of God; it costs
   ~2% in variance at k > 0.15 and is within the Monte Carlo noise. Not worth more work now.

4. **Everything else in the LRG1 excess is one low-rank term**, read off the off-diagonal elements
   (|dk| >= 0.03) where the windowed Gaussian covariance is ~0: mocks - thecov = A_ll' P0(k) P0(k') in
   every (ell, ell') block, plus a constant in (0, 0). Fitted values (same for every window):

   | LRG1 | sigma of coherent P0 mode | P2 mode, units of P0 | corr | const (0,0) |
   |---|---|---|---|---|
   | NGC | 0.51% | 0.94% | +0.18 | 85 |
   | SGC | 0.66% | 1.22% | 0.00 | 120 |
   | QSO NGC / SGC | 0.30 / 0.58% | 0.46 / 0.90% | -0.66 / -0.49 | 380 / 570 |

   SGC/NGC = 1.30 for both modes vs sqrt(V_NGC/V_SGC) = 1.39: super-sample scaling. Two nearly
   uncorrelated modes, both proportional to P0: isotropic delta_b (P0) and tidal / line-of-sight (P2), as
   redshift-space SSC predicts. The constant is the discreteness term sum w^4 (1+alpha^3)/norm^2
   (monopole only). Adding the 7 fitted numbers to the kernel covariance: chi2/n 1.06 -> 1.02 (NGC),
   joint parameter variance ratios at kmax 0.2 become A 1.01/1.09, A2 0.96/1.05, alpha 0.90/1.08,
   SN 0.86/1.10 (NGC/SGC). The quadrupole-amplitude problem of the report (A2 1.6-1.7x) is this term.
   The QSO fit (7 free numbers) shows the method's floor: it absorbs 1-2% of noise.

5. **The bin-width (a + b dk) split of the report is contaminated.** thecov's adjacent 0.005 bins are
   correlated at 0.4, so a fully correlated term appears ~2/3 as "a" (Gaussian-like). The report's
   "a ~ 0.10 Gaussian-like excess" is ~0.05 window (item 2) + ~0.05 mislabelled low-rank term (item 4).
   Section 6 of `report.tex` needs rewriting on this point; the data and figures stay.

6. What is left at kmax = 0.3 (10-20% on parameters) is the bin-width-proportional part: connected and
   discreteness trispectrum. Expected; out of scope for kmax = 0.2.

## 2. What was tried and should not be repeated

- Tuning the fill scale (nside 512 -> 64): chi2/n improves for the wrong reason (item 4).
- Converging the kernel basis to fix chi2/n: it fixes I_k (6.6e-2 -> 1e-5) but chi2/n does not move,
  because the residual is not the window.
- Explaining the residual by redshift-space anisotropy of the kernel: it cannot explain a gap with the
  isotropic GRF test, and the residual is not Gaussian-level anyway.
- The hole-fraction statistic predicting less than the fill window changes thecov: resolved, the effect
  is on the diagonal only to the extent the hole density varies on scales >~ 1/dk; it does (Galactic).
- Weight resolution (7' maps vs 32-neighbour mean): the GRF showed no difference (0.2%).

## 3. Next steps, in order

A. **Make `kernel` the DESI default.** `desi_validation/pipeline.py`: `Config.nw_modes = ('kernel',)`
   with `kernel_random_files=10`, `kernel_cell=4`, `kernel_r_split=8`, `kernel_rmax=200`,
   `kernel_tol=2e-3`, production sampling (`target_near_pairs=1e10`, `n_randoms_max=8e6`). Cost: smoothing
   2-16 min + pair counts 10-20 min per cap, fits the 30-min debug queue. Keep `random-density` as the
   comparison. Re-run LRG1 + QSO once with the new defaults to refresh `report_data`.

B. **Super-sample module in thecov core** -- IMPLEMENTED (`thecov/ssc.py`, see `ssc_dev/DESIGN.md` STATUS);
   next: run `python -m desi_validation.ssc_check --bin LRG1 --label holi-kcore2-LRG1` (and QSO) at NERSC and
   compare the predicted sigma_P0, sigma_P2/P0, r02 with the mocks' (no fitting). Original plan: Target numbers are item 4, not
   chi2/n. Form: Cov_SSC[P_l(k), P_l'(k')] = sum_c R_l^c(k) R_l'^c(k') sigma_c^2 over long-mode
   components c; redshift-space responses from Wadekar & Scoccimarro (2020) / Li, Schmittfull & Seljak
   (2018) (growth, dilation, bias, Kaiser and tidal/LOS terms, minus the local-average term for
   jaxpower's realised normalisation, which the report measured to nearly cancel for P0 at low k).
   sigma_c^2 are window integrals of the linear power: sigma_b^2 = int d^3s xi_lin(s) Q_WW(s)/(int W)^2
   from thecov's cached pair counts (`WindowLibrary`, `Window('W', A, B)`), and the tidal / LOS moments
   from the Legendre-weighted Q_l(s) with the local line of sight that thecov already computes.
   First check, no new code (NERSC, minutes): sigma_b for the LRG1 caps from Q_WW(s) and a linear P(k)
   at z = 0.5; compare with sigma_0 / R_0 for R_0 ~ 1.5-2.5 and check the SGC/NGC ratio 1.30. If the
   isotropic mode alone gives ~0.5%, the P0 part is understood and only the P2 (tidal / LOS) responses
   need the full formalism; if it gives much less, the LOS velocity-gradient response is needed from the
   start. Validation: the module must predict sigma_0, sigma_2/P0, their ratio and ~zero correlation
   without fitting; then `rank1_excess.py` with the SSC term included should leave no off-diagonal excess
   and A2 -> 1.

C. **Discreteness terms** (cheap, exact): the constant sum_d w^4 + alpha^3 sum_r w^4 over norm^2 in
   the (0,0) block (targets 85 / 120 LRG1, 380 / 570 QSO; the catalogue sums are available in
   `desi_compare.weight_diagnostics`), and the sum w^3 x P(k) + P(k') terms. Add to
   `GaussianCovariance` as an optional non-Gaussian discreteness block.

D. **Report**: rewrite Section 6 (bin width) and the conclusions with items 4-5; add the off-diagonal
   decomposition as the primary diagnostic; keep a/b as a figure with the correlation caveat.

E. Later: all tracers (BGS, ELG, LRG+ELG: check names in `dc.TRACER_SPECS`); kmax = 0.3 (trispectrum).

## 4. Tools (all in `desi_validation/`)

- `dump_report_data.py`: runs the pipeline, dumps mock vectors + covariances per window mode and binning
  (`--modes`, `--regions`, `--cache-tag`, `--target-near-pairs`, `--n-randoms-max`, `--kernel-tol`).
- `compare_kernel.py`: chi2/n per k range, variance ratios, a/b split, for any set of dumps.
- `rank1_excess.py`: the off-diagonal low-rank fit, residual diagonal ratios, parameter ratios.
- `hole_fraction.py`: R_PP = <f^2>/<f>^2 of the veto-hole fraction vs angular scale (no pair counts).
- `run_gaussian_footprint.py`: exact Gaussian covariance on the real footprint from Gaussian fields.
- `thecov/smoothing.py` + `tests/test_smoothing.py`: the kernel window (core).

## 5. Rules for runs at NERSC

- Debug queue only (`-q debug`, 30 min, 1 node), at most 2 jobs at once; one job per cap, then a
  combine job that only reads caches.
- Load the environment inside the job and put the repository first on the path:
  `--wrap "source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\$HOME/thecov:\$PYTHONPATH && cd ~/thecov && python -u -m desi_validation.<script> ..."`
  (cosmodesi ships its own thecov; the log's first line should show `~/thecov/thecov/__init__.py`).
- `-o` and any file for the user go to `/global/cfs/cdirs/desicollab/users/oalves/thecov_validation`.
- Real paths, no placeholders. Branch `thecov2`, commit and push there, no PR, no model names in commits.
- Harmless messages: `WEIGHT not in catalog ... noveto.ran.h5`; the `alpha ... ratio 1.6` warning; the
  Jax CUDA error on CPU nodes.

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
