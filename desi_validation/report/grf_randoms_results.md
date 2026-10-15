# Gaussian fields on the DESI randoms: results (2026-10-13/14)

Tools: `grf_randoms.py` (fields / thecov stages, variants), `grf_randoms_plot.py`, `grf_randoms_blocks.py`,
`thinning_cov.py`, `run_grfr_chain.sh` (self-chaining debug jobs over `logs/grfr_queue.txt`), `plot_holes.py`.
Outputs: `~/thecov/grfr_v2/` (all runs after LRG1/LRG2), `~/thecov_desi/holi_v3_mock173/{LRG1,LRG2}/grf_randoms/`
(first runs), `~/thecov/thinning/`, logs in `~/thecov/logs/grfr_*.log`, `run_*.log`, plots `logs/grf_randoms_*.png`.
Construction: band-limited Gaussian field evaluated at the randoms (type-2 NUFFT), estimated by type-1 NUFFT,
realised self-pair term subtracted; the mask is the randoms themselves (holes resolved). See `full_budget_plan.md`.

## 1. The clustering window (P0, no shot noise beyond the self-pair term)

Var_GRF / Var_thecov, thecov given the exact input power; k bands 0.02-0.05 / 0.05-0.1 / 0.1-0.2 / 0.2-0.3.

| tracer, cap | fields | xi-kernel window | local window m^2 |
|---|---|---|---|
| LRG1 NGC | 250 | 0.972 0.998 0.993 1.009 | 0.651 0.705 0.731 0.765 |
| LRG1 SGC | 242 | 0.986 1.012 0.975 0.988 | 0.686 0.746 0.750 0.784 |
| LRG2 NGC | 146 | (queued) | 0.662 0.727 0.779 0.754 |
| LRG2 SGC | 170 | mean over k 0.991 | 0.688 0.775 0.762 0.769 |
| ELG2 NGC | 106 | 0.959 0.999 0.978 0.911 | 0.800 0.861 0.880 0.848 |
| QSO NGC | 64 | 0.967 0.983 1.065 0.977 | 0.806 0.845 0.966 0.924 |
| LRG3, ELG1 (both caps) | 200-264 | (queued, loader bug on the first pass) | |

With the fields' mean as the model (as with mocks) the kernel gives 1.00-1.01 and m^2 1.077 (LRG1 both caps),
1.073-1.093 (LRG2): the 7-9 % Gaussian deficit of the local window, from the holes. Per-point errors (0.02-wide
groups) are 5 % for 250 fields, 8 % for 100. The kernel's I_k misses the exact dilution by +1.5 % in the first bin
and -1.7 % in the last (LRG1; `kernel_validation_results.md`); convergence of the basis halved the first-bin error.

## 2. Shot noise, multipoles, redshift space (LRG1 NGC, xi-kernel; errors ~5-8 % per band)

- Shot noise at the files' level (white weight noise), P0/P2/P4, 124 fields:
  l=0 1.061 0.983 1.017 0.978; l=2 0.986 1.034 0.982 0.999; l=4 0.993 0.921 0.950 0.969.
- Same with linear RSD at the true line of sight (f = 0.75), 95 fields:
  l=0 1.147 0.974 1.048 0.924; l=2 1.056 1.043 1.005 0.988; l=4 1.026 0.926 0.995 0.973 (m^2: 0.67-0.81).
  No sign that the isotropic kernel fails for P2/P4 at this precision. `grf_randoms_blocks.py` gives the
  conditional var(l=4 | l=0,2) for HANDOFF 7.9 (not yet read).
- QSO NGC shot-noise variant: 18 fields only (2.5 min each), unusable.

## 3. Poisson sampling (P/n^2 test) and the frozen shot noise

First run (`_poi`, 400 realisations, galaxies from a FIXED parent half of the randoms, fixed estimator half):
Var / Var_thecov(Gaussian) = 0.985 0.910 0.866 0.818, i.e. BELOW Gaussian. Not the realised shot-noise
subtraction (Var(SN) is 3 % of the variance in the last band; expected-SN subtraction gives 0.850). It is the
fixed point sets: their shot-noise fields are a constant offset c(k) in F, and Var(|F + c|^2) =
(sigma_f^2 + |c|^2)^2 - |c|^4. Frozen fraction of S = 2 p0 / (1 + p0) = 0.51 (p0 = 0.34); predicted ratio
1 - (0.51 S / (P + S))^2 = 0.973 0.942 0.877 0.818 against 0.985 0.910 0.866 0.818 measured. This is a
parameter-free check of thecov's SS term at S/P up to 5 (the QSO regime). For the mocks the effect is negligible
(randoms' shot noise 1.7 % of S; galaxies independent per mock). `--resplit` (variant `_poi2`, queued) redraws
the halves every realisation; that run is the clean P/n^2 test.

## 4. Dilution by thinning (`thinning_cov.py`, queued)

E_mocks[Cov_thin(P_p)] = Cov(P_p) - Cov(P_1): thinning one catalogue (mock 173, kept weights w/p) measures the
dilution difference with cosmic variance removed. Minus thecov's Gaussian difference G(S_p) - G(S_1) it is
disc(p) - disc(1) of the galaxies' own B and sampling: B/n scales as 1/p, P/n^2 as 1/p^2, SSC and T0 cancel.
Compare with HANDOFF 9.4 (tree level: B/n 0.07-0.08 and P/n^2 0.015 of G on the P0 diagonal; mocks - G 0.20-0.23).
First attempt failed (22-moment NUFFTs too slow / 170 GB for QSO); rerun is P0 only with incremental saves.

## 5. Holes (`plot_holes.py`, LRG1 NGC, 1.5 deg patch)

Raw fill at 1 arcmin from 18 random files shows the bright-star masks (0.05-0.1 deg) and many single-pixel holes;
the kernel-smoothed fill has median 0.937 (6 % dilution) with dips to 0.85 (k = 0.163) / 0.75 (k = 0.298) on the
large masks. Figures: `holes_LRG1_compare.png`, `holes_LRG1_dense.png`, `holes_LRG1.png` (session artifacts).

## 6. Cost and bookkeeping

~26 CPU node-hours in the debug queue for everything above and the queued reruns. Known waste: 3 node-hours on the
loader bug (`KeyError: 'SIG'`, fixed), partial high-z fields of the first attempt (old settings, not mergeable), the
QSO shot-noise fields. The first LRG1/LRG2 outputs sit in `~/thecov_desi/...` (outside the `~/thecov` scope); copy
all outputs to `/global/cfs/cdirs/desicollab/users/oalves/thecov_validation/` when the queue is done.

## 7. Discreteness terms from the catalogue alone: closed form (2026-10-15)

`thinning_closed_form.py`: Cov_thin[P(k),P(k')] = a sum_g G_g(k) G_g(k') + a^2 [|W2(k-k')|^2 + |W2(k+k')|^2 - 2 sum w^4],
a = (1-p)/p, G_g = 2 Re[F_1^* w_g e_g] - 2 w_g^2, W2 the transform of the w^2-weighted galaxies. Verified on toys (2-4 %) and
against the 200-thinning Monte Carlo on mock 173 LRG1 NGC: every band within 1.5 sigma, off-diagonals to 5-10 %, at
p = 0.5 and 0.25; 60 s per catalogue instead of 400 thinnings. With thecov's Gaussian S-dependence fitted as
g1 a + g2 a^2 over p = 0.5, 0.25, 0.125 (rms residual 0.16 % LRG1 / 1.5 % QSO of G = the pair-count noise), the
galaxies' own terms at full density, units diag G(p=1):

| | B/n diag k 0.2-0.3 | B/n off-diag k > 0.15 | P/n^2 diag | P/n^2 off-diag | tree level (HANDOFF 9.4) |
|---|---|---|---|---|---|
| LRG1 NGC | +0.138 | +0.163..0.170 | +0.024 | +0.022 | B/n 0.07-0.08, P/n^2 0.015 |
| QSO NGC | +0.031 (noisy) | +0.111..0.114 | +0.044 | +0.039 | same |

LRG1: B/n = 2x tree level (the one-halo bispectrum of the mock galaxies), P/n^2 ~ 1.5x. QSO: P/n^2 = 2.6x (non-Poisson
pairs), B/n ~ 1.5x. Both match the template-fit amplitudes of HANDOFF 9.4 with no template and no mocks: this is the
"calibrate discreteness from the data" step, and it applies unchanged to the DESI catalogues (~1 min per cap).

## 8. Long-mode responses from the integrated bispectrum (`response_ibisp.py`): status

Estimator: C_l(k) = <F_l^*(k) FT[delta_L F](k)> with delta_L = W_L*(m delta)/W_L*m (Gaussian W_L, k_L = 0.01), minus the self
term sum_g w_g^2 delta_L(x_g); response R_l = (C_l - SN) / (P_l <delta_g delta_L>), with <delta_g delta_L> = int P W (NOT
Var(delta_L) = int P W^2: they differ by 2.2 for the Gaussian filter). Verified on periodic toys: sharp-split modulated field
2R to 5 %, lognormal field (R = 2 exactly) 2.04. On the real randoms the known-response constructions do NOT reproduce
their inputs (soft-split g = 2.45: 1.55 ok; g = 1: 1.9 instead of 3.8; top-hat g = 1: 1.5-2.8; lognormal: 3.9, then 8-14 +- 2-7
after correcting the data = randoms self-pair terms): the window's spectral leakage into delta_L and the self-pair
bookkeeping of a field painted on the randoms interact in a way not yet understood. Mock measurement (3 mocks, LRG1 NGC,
per unit galaxy long mode): l = 0: 2.11, 2.01, 1.93, 1.83 (+- 0.02-0.09); l = 2: 3.20, 3.22, 3.14, 2.99 (+- 0.1-0.9), against
tree level 1.2-1.6 and 1.8-2.7 (review 1.2 / g0). **Validated (2026-10-15, `--lnpoisson`):** a Poisson-sampled lognormal catalogue with DISTINCT randoms (the data's bookkeeping;
R = 2 exactly for any filter) on the LRG1 NGC footprint returns l = 0: 1.86 +- 0.06 / 1.98 +- 0.05 / 1.91 +- 0.06 (Gaussian filter,
bands 0.05-0.1 / 0.1-0.2 / 0.2-0.3, 3 catalogues) and 2.11 / 2.32 / 2.31 (+- 0.2-0.3, top-hat, 2 catalogues); l = 2 consistent
with 0. The earlier failures were artefacts of painting a continuous field on the random points (data = randoms). The first
band (k = 0.02-0.05, only 2-5 k_L) is outside the squeezed limit (2.8 for R = 2) and must not be used.
So the mock responses above (bands 2-4) stand: the mock LRG1 galaxies respond to the long mode 1.3-1.5x (P0) and
1.2-1.4x (P2) more strongly than tree level, the same physics as the B/n factor of 2 (one-halo). Next: a response table in
`SuperSampleCovariance.responses()` (HANDOFF 3.B/C) from this measurement; the survey-mean (local-average) response is a
separate quantity, measured by the 848-mock delta_shell regression (HANDOFF 9.2).

## 9. Scale dependence of the long-mode response, R_l(k_L) (2026-10-15)

`response_ibisp.py --k-L 0.03 0.02 0.015 0.01 0.007 0.005`, LRG1 NGC, holi mocks 173-178 (6 catalogues, F_l computed
once per catalogue, 12 min per 3 mocks on one debug node). Box 3103 Mpc/h (k_f = 0.0020). Responses per unit GALAXY
long mode, (C - SN) / (P <delta_g delta_L>), error = scatter over the 6 mocks / sqrt(6). Figure: response_kL_LRG1_NGC.png.

| k_L | sigma_L | <dL>_gal/sigma_L^2 | R_0 0.05-0.1 | R_0 0.1-0.2 | R_0 0.2-0.3 | R_2 0.05-0.1 | R_2 0.1-0.2 | R_2 0.2-0.3 |
|---|---|---|---|---|---|---|---|---|
| 0.030 | 18.8 % | 2.15 | 1.895 +- 0.017 | 1.899 +- 0.013 | 1.815 +- 0.008 | 2.42 +- 0.09 | 2.78 +- 0.06 | 2.77 +- 0.08 |
| 0.020 | 11.6 % | 2.19 | 1.910 +- 0.023 | 1.888 +- 0.017 | 1.803 +- 0.010 | 2.69 +- 0.12 | 2.85 +- 0.08 | 2.80 +- 0.10 |
| 0.015 | 8.1 % | 2.22 | 1.937 +- 0.034 | 1.898 +- 0.024 | 1.819 +- 0.018 | 2.89 +- 0.15 | 2.92 +- 0.10 | 2.89 +- 0.12 |
| 0.010 | 4.8 % | 2.16 | 2.011 +- 0.072 | 1.952 +- 0.058 | 1.900 +- 0.059 | 3.30 +- 0.22 | 3.15 +- 0.13 | 3.33 +- 0.26 |
| 0.007 | 3.0 % | 1.98 | 2.245 +- 0.185 | 2.161 +- 0.175 | 2.184 +- 0.214 | 4.33 +- 0.70 | 3.90 +- 0.40 | 4.84 +- 1.05 |
| 0.005 | 1.9 % | 1.78 | unusable | | | | | |

(The 0.02-0.05 band is outside the squeezed limit for k_L >= 0.01 and is not tabulated; at k_L = 0.03 the 0.05-0.1
band is marginal, 3 k_L = 0.09.)

Reading.
- R_0 is flat for k_L = 0.015-0.03 (a factor 2 in scale): 1.89-1.94 (0.05-0.1), 1.89-1.90 (0.1-0.2), 1.80-1.82
  (0.2-0.3), i.e. 1.2-1.5x the tree-level 1.2-1.6. Below k_L = 0.015 it rises (2.0 at 0.01, 2.2 at 0.007); at 0.005
  the estimator breaks down: <delta_L at the galaxies> is +-2e-4 and changes sign between mocks (the normaliser
  int P W -> 0), because a filter wider than the survey only sees the long mode's zero survey mean. This is the
  structural statement of the previous note made visible: the survey-mean mode cannot be measured in-survey.
- R_2 grows monotonically toward large scales: 2.8 -> 2.9 -> 3.2 -> 3.9 (band 0.1-0.2, k_L 0.03 -> 0.007), 2.5 sigma
  between 0.03 and 0.01 alone. The rise starts at k_L ~ 2 pi / 460 Mpc/h, the radial thickness of the LRG1 shell:
  below it the in-survey long modes are predominantly transverse, and the quadrupole response depends on the
  long mode's orientation to the line of sight (the tidal/RSD part of the response), while the monopole does not at
  leading order. The super-survey mode of a thin shell is also transverse-dominated, so the R_2 relevant to the SSC
  is the small-k_L value (3.2-3.9), not the isotropic 2.8. Part of the k_L = 0.007 rise (7 % in R_0) is the onset
  of the normaliser bias (<dL>_gal / sigma_L^2 drops from 2.2 to 2.0) and is common to both multipoles.
- Net survey-mean response. With data-matched randoms, norm follows (1 + delta_L)^2 and the per-mock FKP weights
  cancel between |F|^2 and norm for a uniform density change, so d ln P_hat / d delta_{g,L} = R_0(q -> 0) - 2.
  Taking the flat range (k_L 0.015-0.03): -0.09 +- 0.03 (0.05-0.1), -0.10 +- 0.02 (0.1-0.2), -0.19 +- 0.01
  (0.2-0.3) per unit galaxy mode, against -0.4 .. -0.8 at tree level and the 848-mock regression on delta_shell of
  HANDOFF 9.2: -0.064 +- 0.013 (0.1-0.2), -0.090 +- 0.011 (0.2-0.3), +0.035 (0.02-0.1). The "3x weaker than tree
  level" net response of 9.2 is therefore a near cancellation of a stronger-than-tree physical response (1.9) against
  the normalisation response (2), not a weak response. For P2 the same subtraction gives 0.7-0.9 (k_L 0.02-0.03) or
  1.2-1.9 (k_L <= 0.01, the transverse limit) against 9.2's +0.80 / +0.89 / +1.03.
- Open factor: 9.2's regressor is "delta ln int n(z)^2", which is 2 delta_L for a uniform density change; if the
  slopes there are per unit delta ln int n^2 they must be doubled before this comparison (P0 -0.13 / -0.18, P2 1.8 /
  2.1), which would match the 0.1-0.3 P0 values here to 1 sigma and favour the transverse-limit R_2. The regression
  code is not in the repository (walkthrough of the other session); to be confirmed from nz_scatter's NZ arrays
  (std/mean of the total shell count is sigma(delta_L) directly; 9.2 quotes 0.8-1.1 %).
- Consequence for thecov: the local-average term should use R_0(q -> 0) = 1.9 (LRG1) rather than the tree-level
  value, and the R_2 orientation dependence needs the window-weighted long-mode anisotropy (thin shell: transverse).
  Test 3 of the previous note (lognormal-Poisson catalogues cut from a larger box, known delta_survey) is the end-to-end
  check of this bookkeeping with a known answer.

## 10. The survey-mean response, closed (2026-10-15)

Two runs (one debug node each, 4 + 20 min). Figure: survey_mean_response_LRG1.png.

**10.1 848-mock regression with an explicit convention** (`survey_mean_regression.py`, LRG1, both caps, 11 defective
mocks excluded). Regressor delta_L = N_m / median(N) - 1 (weighted galaxy count of the cap). sigma(delta_L) = 0.417 %
NGC / 0.535 % SGC (Poisson 0.12 / 0.17 %). sigma(delta ln sum N(z)^2) = 2.05 x sigma(delta_L): HANDOFF 9.2's regressor
is 2 delta_L, and regressing on it reproduces 9.2 exactly (-0.062 +- 0.020, -0.091 +- 0.017 NGC; -0.061, -0.109 SGC).
Per unit delta_L (bands 0.05-0.1 / 0.1-0.2 / 0.2-0.3):

| | d ln P0 / d delta_L | d ln P2 / d delta_L (slope of P2/<P0> divided by <P2>/<P0> = 0.41 / 0.34 / 0.23) |
|---|---|---|
| NGC | +0.04 +- 0.10, -0.12 +- 0.04, -0.17 +- 0.03 | 1.4 +- 0.5, 1.8 +- 0.3, 2.1 +- 0.3 |
| SGC | +0.10 +- 0.10, -0.13 +- 0.05, -0.24 +- 0.04 | 1.8 +- 0.6, 2.2 +- 0.3, 2.7 +- 0.4 |
| predicted, in-survey R_0(k_L >= 0.015) - 2 (NGC) | -0.09 +- 0.03, -0.10 +- 0.02, -0.19 +- 0.01 | R_2 - 2: 0.4-0.9 (k_L 0.03, isotropic) or 1.3 +- 0.2, 1.2 +- 0.1, 1.3 +- 0.3 (k_L 0.01, transverse) |
| tree level | -0.4 .. -0.8 | |

P0: the mocks' net survey-mean response equals R_0(q -> 0) - 2 within 1 sigma in every band and both caps. P2: the
mocks imply R_2(q -> 0) = 3.4-4.1 (NGC), the small-k_L (transverse) limit of section 9, not the isotropic 2.8; the
0.2-0.3 band is somewhat above even that (2.1 vs 1.3). The delta_norm regressions are ~0 for P0 and negative for P2
(as 9.2 found): sigma(delta_norm) = 0.395 % with corr(delta_norm, delta_L) = 0.41 (NGC), norm is not a proxy for the mode.

**10.2 Synthetic catalogue with a known survey-mean mode** (`survey_mean_synthetic.py`, LRG1 NGC footprint, lognormal-
Poisson galaxies from one half of the mock-173 randoms, p0 = 0.08 -> 6.2e5 galaxies, n P_fkp = 4.0 at the peak,
<w_fkp> = 0.21; 8 seeds x delta_u in {-0.06, -0.03, 0, 0.03, 0.06} with common random numbers). R_global = 2 by
construction. d ln P_hat_0 / d delta_u (bands 0.05-0.1 / 0.1-0.2 / 0.2-0.3):

| normalisation | d ln norm / d delta_u | d ln P0 / d delta_u | expected |
|---|---|---|---|
| NX and FKP follow the data (holi pipeline, 9.3) | +0.397 +- 0.002 | -0.002 +- 0.019, +0.007 +- 0.014, -0.018 +- 0.013 | 0 = 2 - 2 |
| NX follows the data, no FKP | +1.999 +- 0.002 | -0.004, +0.007, -0.019 (same errors) | 0 |
| NX and FKP fixed | +1.000 +- 0.001 | +0.995 +- 0.019, +1.006 +- 0.014, +0.981 +- 0.013 | +1 = 2 - 1 |

d(P2/<P0>)/d delta_u = 0.00 +- 0.03 in all three (isotropic field). The bookkeeping is exact to 1-2 %: with
data-matched randoms the net response is R_global - 2 independently of the FKP weights (norm itself responds only by
0.40 = 2 <w_fkp>_m-weighted when the weights follow n(z), but |F|^2 carries the same weights and the ratio does not
see them), and with frozen NX only alpha follows the density (R_global - 1). The across-seed regression on the box's own
realised survey mean (8 points, no common random numbers) gives +0.3 (matched) / +1.1 (fixed): consistent, noisy.

**Conclusion.** The in-survey response extrapolated to q -> 0, the 848-mock survey-mean regression and the synthetic
end-to-end test agree: P_hat_0's net survey-mean response for LRG1 is R_0 - 2 = -0.1 to -0.2 per unit galaxy mode,
with R_0 = 1.9 (1.2-1.5x tree level), and the "3x weaker than tree level" of HANDOFF 9.2 was the factor 2 of its
regressor combined with the near cancellation. For thecov's local-average term: keep norm_kind = 'data-randoms'
(R - 2), use the measured R_0(q -> 0) = 1.9 for LRG1 instead of the tree-level 1.2-1.6, and for P2 the long-mode
anisotropy of the window (thin shell -> transverse modes, R_2 = 3.4-4) rather than the isotropic response. The SSC
amplitude then needs sigma^2 of the super-survey modes with the same anisotropy weighting (nz_scatter's radial
measurement plus the transverse |W(q)|^2), which is where the remaining uncertainty in the SSC term sits.
