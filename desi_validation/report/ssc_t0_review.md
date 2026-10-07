# Review of the SSC and T0 implementations (2026-10-07)

Scope: `thecov/ssc.py`, `thecov/discreteness.py`, `thecov/trispectrum.py`, `desi_validation/ssc_check.py`, read against
`report/ssc_formalism.typ`, HANDOFF sections 6-7 and the measurements in `report/report.tex` section 7. Numerical checks:
`ssc_dev/review_checks.py` (reproducible, seconds) and the test suites (`tests/test_ssc.py`, `test_discreteness.py`,
`test_trispectrum.py`: 17 tests pass, one added here).

## Summary

1. **The algebra of both modules is right.** Kernels, squeezed limit, long-mode variance formula, Poisson terms of the
   local average, the discreteness four-point counting, the T0 snake/star assembly, the Z3 exact-mapping terms and the
   IR cancellation were rederived or checked numerically against independent code paths (section 1.1, 2.1).
2. **The SSC's local-average (LA) term contradicts the mocks on the one quantity that tests it directly.** With the
   `data-randoms` convention implemented (norm = alpha sum_c D_c R_c, alpha and the data realised), the net monopole
   response is R_0 - 2 g_0 = -1.1 P_0 for LRG1, so the regression slope of P_hat_0 on delta_norm is predicted at
   -0.2 to -0.4 P_0 and the coherent P_0 mode is predicted to be anticorrelated with delta_norm at |r| ~ 1. The report
   measured +0.1 to +0.25 P_0 and r = +0.14 / +0.17. The agreement of the predicted sigma_P0 (0.43 vs 0.51 %) therefore
   does not validate the term. If instead only alpha is realised in the normalisation (`alpha` convention), tree level
   gives slope +0.2, r ~ +0.17 and sigma(delta_norm) = 0.43 % with sigma_0 = 0.16 % (section 1.2): all three measured
   numbers at once, but then the isotropic SSC supplies only ~0.2 % of the 0.5 % P_0 mode. Which convention jaxpower
   uses has to be settled on the mocks; `ssc_check` now prints the prediction of all three next to the mocks' values.
3. **T0 at tree level is implemented correctly; its rejection by the mocks is a physics result, but less sharp than
   the "> 8 sigma" suggests** because the T0 template is nearly degenerate with the discreteness B term and the Wishart
   fit with 859 mocks is sensitive at the 1 % level, below the accuracy of the fixed (Gaussian) part (section 2.2).
   A sub-sampling test separates the three terms model-independently (section 3.C).

## 1. Super-sample covariance

### 1.1 Verified

- **Z1, Z2** (`_Z1`, `_Z2`): the SCF99 form b1 F2 + f mu^2 G2 + (f mu k / 2)[mu1/k1 Z1(k2) + mu2/k2 Z1(k1)] + b2/2 +
  bs2/2 S2, checked term by term. The Galileon-basis kernels of `trispectrum.py` (`Z2`, `squeezed_response`) are an
  independent implementation; their squeezed response at finite q = 1e-3 k reproduces `response_coefficients`'
  a P + c dP/dlnk for every (ell, n) to 2e-6 of R_0^(0).
- **Squeezed limit**: the 1/q terms cancel between the two terms of the response (not only in the +-eps average), the
  +-eps average removes the O(eps) remainder; the O(eps^2) error is 1e-8. The absence of an L_4(nu) response is exact
  for these kernels: after the phi average every term is at most quadratic in nu.
- **Real-space limits** (47/21 - n/3, (2/3)(8/7 - n), 2 b1 b2, (4/3) b1 bs2): tests pass. The galaxy response
  47/21 - n/3 + 2 b2/b1 (global-mean referenced) is the right one; in local-mean terms it is growth + dilation (0.74 at
  n = -1.5) plus the bias response 2 (b2 + b1 - b1^2)/b1 = -2.15 for b1 = 2.2, which is why the net response after the
  local average is negative for a highly biased tracer.
- **sigma^2 formula** (eq. sigma2 of the note): rederived the prefactor (4 pi)^{3/2}, the sign (-1)^{lam/2} and the
  3j factor from the plane-wave and Legendre expansions; `sigma2()` matches. Caveat: the sphere test fixes only the
  lam = 0 channel (for a distant-observer sphere Q_{n n' lam} vanishes for lam > 0), so the lam = 2, 4 channels that
  dominate the n = 2 modes of a thin shell are unchecked except through the pair-count machinery shared with the
  Gaussian term.
- **Local average**: the (1 + delta^M)(1 + delta^W) structure of alpha sum_c D_c R_c, g_0 = b1 + f/3, g_2 = 2f/3, the
  Poisson variance sum_g w^2 (u_alpha + u_D)^2, the cross term 2 P J / norm and the collapsed bispectrum
  2 Z2(k, -k) = b2 + 2 bs2 / 3 with the Kaiser multipoles of Z1^2: all consistent. The realised shot noise is subtracted
  exactly, so the LA acts on the shot-noise-subtracted spectrum, as coded.
- **Discreteness four-point** (`discreteness.py`): one shared galaxy gives the connected three-point function only (a
  Gaussian field has no xi xi term in its three-point function, and the single-xi terms are the (1 + alpha) P / nbar
  pieces already in the Gaussian covariance); two shared galaxies give P(|k1 +- k2|); a shared random with two clustered
  galaxies only contributes at k1 + k2 = 0 within the window, i.e. it is the Gaussian (1 + alpha) shot-noise cross term.
  The counting 2 [B + B] / nbar and [P + P] / nbar^2 and the window integrals are right.

### 1.2 The local-average term against the mocks (the main point)

Tree-level numbers for LRG1 (b1 = 2.2, f = 0.76, b2 = 0.28 Lazeyras, bs2 = -4/7 (b1 - 1)), responses in units of the
measured multipole (`review_checks.py`):

| n_eff | R_0^(0)/P_0 | R_0^(2)/P_0 | R_2^(0)/P_2 | R_2^(2)/P_2 | net P_0, data-randoms (R - 2 g_0) | slope | net P_0, alpha (R - g_0) | slope |
|---|---|---|---|---|---|---|---|---|
| +0.5 | 2.92 | 0.49 | 4.42 | 0.92 | -1.99 | -0.41 | +0.46 | +0.19 |
| -1.5 | 3.78 | 1.00 | 6.16 | 6.63 | -1.13 | -0.23 | +1.33 | +0.54 |
| -2.0 | 4.00 | 1.13 | 6.59 | 8.06 | -0.91 | -0.19 | +1.54 | +0.63 |

(2 g_0 = 4.91; "slope" = d ln P_hat_0 / d delta_norm for the clustering part of delta_norm; the Poisson part of
delta_norm adds -1 times its variance fraction. The net is insensitive to b1 in 1.8-2.4: -0.94 to -1.20.)

What the report measured (section 7 of `report.tex`, 859 holi mocks): sigma(delta_norm) = 0.43 % NGC / 0.75 % SGC with a
Poisson part 0.13-0.26 % / 0.18-0.36 %; d P_hat_0 / d delta_norm = +0.1 to +0.25 P_0, rising with k; correlation of the
fitted P_0 amplitude with delta_norm +0.14 / +0.17.

Consequences, independent of the pair-count sigma^2 values of the NERSC run:

- In the implemented model the P_0 fluctuation and delta_norm are driven by the same long-mode projections (D^W_0,
  D^M_0; the n = 2 projections nearly cancel in P_0: R_0^(2)/P_0 - g_2 = +0.5 against -g_2 = -0.5), so the model
  predicts corr(P_0 mode, delta_norm) ~ -1 and a negative slope. The mocks give +0.14 and a positive slope. The model's
  sigma_P0 (0.43 % vs 0.51 %) is reached for the wrong reason.
- Conversely the measured norm scatter bounds the isotropic long mode: with data-randoms, sigma(delta_norm)_clust =
  2 g_0 sigma_0 <= 0.4 % gives sigma_0 <= 0.08 % (NGC) and an isotropic SSC contribution to sigma_P0 of at most
  |net| sigma_0 ~ 0.1 %. If the run's rank-4 term reached 0.43 % through the n = 0 modes, its sigma^2 implies
  sigma(delta_norm) ~ 2 %, five times the measured value. One of the two must be wrong; the run never printed the
  model's sigma(delta_norm).
- With the `alpha` convention (norm = alpha times a fixed randoms integral, e.g. sum_r nbar w_r^2; P_hat then scales
  with one realised factor) the tree level gives, with sigma_0 = 0.155 % and the measured Poisson part: sigma(delta_norm)
  = 0.43 %, slope = +1.33 x 0.78 - 0.22 = +0.2, r = +0.17. This matches all three measured numbers, which is a strong
  hint that the normalisation realises alpha only (or that the data mesh enters norm in a way that does not follow the
  survey-mean density). It must be checked in jaxpower (`desi_compare.mesh_normalization` reproduces the files' norm to
  0.6 % on one mock, which does not distinguish D x R from alpha^2 R x R or alpha x a fixed integral; the per-mock
  scatter, 0.43 %, does).
- Under the `alpha` convention the isotropic SSC explains ~0.2 % of the 0.5 % coherent P_0 mode and the rest is not a
  density-tracing long mode. The report's own conclusion ("the survey-mean density explains 2-3 % of the amplitude
  variance") stands, and the model would have to find the remaining 0.46 % elsewhere: the n = 2 channel in P_0 is
  nearly zero at tree level, so the candidates are the response of P_0 to the realised in-survey long-mode power
  (section 3.D) or a non-perturbative response.

jaxpower itself (github.com/adematti/jax-power, `jaxpower/mesh2.py`, read 2026-10-07) implements both: `compute_fkp2_normalization`
with `split=None` is alpha x sum_cells D_c R_c / V_c (data x randoms, `data-randoms`), with `split=<seed>` it is alpha x
sum_cells R1_c R2_c / V_c over two disjoint random subsamples (randoms only, `alpha`-like, except that the mock randoms carry
the mock's own shuffled redshifts). Which call the DESI pipeline made is decided by the three scripts of HANDOFF 8.1
(`inspect_norm_code.py`, `norm_convention.py`, `check_norm_catalogs.py`).

What was added:

- `SuperSampleCovariance(norm_kind=...)`: `'data-randoms'` (as before), `'randoms'` (alpha^2 sum R^2: alpha twice),
  `'alpha'` (alpha once); the LA coefficients, the Poisson variance and the J, J3 integrals follow the convention.
- `SuperSampleCovariance.local_average_statistics(A, ells, C_total=None)`: the predicted sigma(delta_norm) (clustering,
  Poisson, total), Cov(P_hat_l(k), delta_norm), the regression slope in units of P_hat and the correlations; tested on
  the sphere (`test_local_average_statistics_and_norm_kinds`).
- `ssc_check`: `--norm-kind`, and per cap a `normalisation per mock` block with the mocks' sigma(delta_norm), slopes and
  correlations at k ~ 0.05, 0.1, 0.2 for l = 0, 2 against the three conventions (same pair counts, so it costs seconds).
  Read this block first in the next run: the convention whose sigma(delta_norm) and slopes match is the one to use, and
  the matched sigma(delta_norm) is then a fit-free test of the pair-count sigma^2 (section 1.3, first bullet).

### 1.3 Other points on the SSC

- **b1 from the windowed P2/P0 at k <= 0.06 is biased high.** The window suppresses the measured quadrupole at low k
  for a thin shell, and the global and radial integral constraints lower the measured P_0 in the first bins. The fitted
  A = 0.73 / 0.70 (LRG1) absorbs the resulting mismatch (K_0(2.0)/K_0(2.2) = 0.85 already explains half of it). The net
  responses barely move with b1, but b2(b1) changes sign between b1 = 2.0 and 2.2, and the T0 and the collapsed-bispectrum
  LA term depend on it. Fit b1 (and A) with the window-convolved Kaiser model (thecov's Q_L(s) give the convolution as a
  Hankel transform), or take the DESI full-shape b1 for the tracer.
- **BC with a tree-level P_lin against an LA with the measured P_hat.** The near-cancellation makes the net response
  sensitive to the ratio of the two. The dressed power (BAO + FoG) is one answer; a cheaper and more consistent one is
  the ratio form R_l^(n)(k) = [(a_ln + c_ln n_eff(k)) / K_l] P_hat_l(k) with n_eff from the measured multipoles, so
  that both terms carry the same nonlinear, FoG-damped, window-diluted spectrum. For l = 2 the two forms differ by the
  FoG damping of the Kaiser quadrupole, which is what section 7.2 fitted sigma_v for.
- **Shuffled random redshifts.** If each mock's randoms take their redshifts from that mock's data, radial long modes
  are removed from F and the radial integral constraint changes the low-k Gaussian covariance and the mean of the first
  bins; it does not change the leading-order LA (norm still scales as the square of the realised density), but it does
  change which modes delta_norm traces. Worth confirming how the mock randoms were built before interpreting the norm
  statistics.
- `discreteness_covariance` uses the undamped P_lin for the collapsed bispectrum while the responses are dressed:
  harmless (zc = b2 + 2 bs2/3 = -0.18 for LRG1).
- The `xi(lam, s)` quadrature (linear q grid, dq <= 0.25/s_max, 1 Mpc/h damping) and the 96 mu cells are adequate; the
  lam = 4 channel is the one to watch if n_mu is ever lowered.

## 2. Tree-level trispectrum

### 2.1 Verified

- `parallelogram` (6 snake + 2 star terms) against `trispectrum` (12 + 4): tested; and both against a from-scratch
  T_2211 + T_3111 assembly with the standard F2 / symmetrised F3 in real space: 1.7e-15.
- `Z3`: the exact-mapping terms (f k_z)^m / m! with theta_2 = G2, the m = 1, 2, 3 products and their 1/3 symmetrisation
  weights, and `Fg3` in the Galileon basis (b2 F2, 2 bG2 sigma^2 F2, bdG2 sigma^2, 2 bGamma3 sigma^2 (F2 - G2), b3/6,
  bG3 G3) rechecked term by term. The q1 + q2 = 0 configurations are handled correctly: G2(q, -q + eps) = O(eps^2) while
  alpha, beta = O(1/eps), so the product vanishes and setting 1/|k12|^2 to 0 there is the right limit (the P13-kernel
  test exercises exactly this).
- **IR safety of the T0 covariance**: for k2 = 0.2 and k1 -> 0, snake and star each grow as 1/k1^2 (2e8 x P1^2 P2 at
  k1 = 2e-4) and cancel to a finite (snake + star)/(P1^2 P2) that converges from k1 = 0.02 to 0.005 (1428 -> 1395 for
  one orientation); the drift below k1 = 1e-3 is the O(1) x P1 P2^2 term (the P13-like response of P(k2) to the realised
  soft power), not round-off. The two-hard-one-soft Z3(k2, -k2, k1) is O(1), the one-hard-two-soft Z3(k1, -k1, k2) is
  O((k2/k1)^2) and cancels the snake's Z2(-k1, k1 + k2)^2 piece, as it must.
- Multipole projection, J4 = int m^4 / norm^2, k-bin averages, the mu12 end-point rule: consistent with the Kobayashi
  definition and the handoff's PowerSpecCovFFT comparison.

### 2.2 Reading of the rejection (sections 7.1-7.8)

The implementation is not the reason. Three things weaken the "T0 = 1 excluded at > 8 sigma" statement, though, and
one explains part of the "LL fails even at k < 0.04" result:

- **Degeneracy with the discreteness B term.** Both are broadband P(k) P(k')-like surfaces; their ratio is ~1/(nbar P)
  ~ 0.5-1 for LRG1 at k = 0.1-0.2. In 7.3 the free fit put 2x into disc and -0.1 into T0; after window-convolving disc
  (7.5) disc became 1.2 and T0 0.02-0.3. The split between the two is what the window convolution of a local-approximation
  term decides, and that convolution is itself a heuristic (K = corr(C_G)^{1/2}).
- **Sensitivity of the Wishart likelihood.** With 859 mocks and 228 elements, -2 dlnL changes by ~30 for a coherent 1 %
  error on a 24 x 24 block. The baseline (kernel Gaussian + SSC + disc) is only good to 2-3 % in chi2/n for LRG1
  (7.9: the l = 4 conditional variance 1.07-1.11 is in the Gaussian part). Amplitude fits of a 5-10 % term on top of a
  2-3 % baseline error are biased; the Abacus numbers (25 mocks, Gaussian chi2/n 1.14-1.22, shared box modes) more so.
- **Overlap with the SSC near the diagonal.** The SSC integral int P(q) |W(q)|^2 is the exact finite-volume form of
  the collapsed snake term for p = |k_a + k_c| below the window width, which for the 500 Mpc/h LRG1 shell extends to
  p ~ 0.01-0.02 in the radial direction. The tree-level T0 with sharp bins adds the same configurations again for
  |k1 - k2| in that range, and the squeezed LL/LH blocks for k_i < 0.02-0.03 are partly the long modes the SSC + LA
  already carry. This is the handoff's own reading in 7.8; a window-convolved low-k T0 (the s-space form: Q(s) against
  the trispectrum's Fourier transform, as the SSC does with xi) is the consistent object, not a k-split.
- The remaining fact, that the hard-hard tree-level T0 at k ~ 0.1-0.3 is too large by several times for LRG, is
  plausible physics: the Z2, Z3 kernels contain (f k mu)^n terms that Gaussian damping of the external legs does not
  tame, and the b2 of the mocks' HOD galaxies is unknown (section 1.3).

## 3. Suggested approaches, in order of cost

**A. Settle the normalisation convention and the long-mode variance on the mocks (minutes, no new pair counts).**
Run `ssc_check` with the new block. Expected outcomes: (i) `alpha` matches sigma(delta_norm) and the slopes: switch the
default, re-read HANDOFF 6.1-7.5 (the rank-4 term changes shape and amplitude); (ii) `data-randoms` matches
sigma(delta_norm) but not the slope: the tree-level response for these galaxies is wrong by ~+2 P_0 and must be
measured (B); (iii) nothing matches sigma(delta_norm): the pair-count sigma^2 (lam = 2, 4 channels, I_w normalisation)
or the mocks' super-survey content is the problem. In every case also regress on alpha_i alone if the files keep it: it
separates D^M from D^W.

**B. Measure the responses from the mocks instead of tree level.** The 859 mocks give, per bin, the regression of
P_hat_l(k) on delta_norm (now printed) and the position-dependent power spectrum: split each footprint into ~8-16
sub-volumes, measure P_hat_l and the sub-volume mean density; the slope over sub-volumes and mocks is R_l^(0)(k) with
nonlinearity, FoG, discreteness and the estimator's own normalisation included (Chiang et al. 2014 integrated
bispectrum). The n = 2 response follows from the sub-volumes' quadrupolar density moment. `SuperSampleCovariance` only
needs `responses()` replaced by a table; everything else (sigma^2 from the pair counts, the LA bookkeeping) stays.

**C. Separate T0, discreteness and SSC model-independently by sub-sampling.** Recompute the multipoles of each mock on
a random 50 % (and 25 %) subsample of its galaxies. The Gaussian term changes as thecov predicts, the discreteness B and
P terms scale as 2x and 4x, the SSC and T0 do not change. The off-diagonal excess of the diluted set minus the
undiluted set is the discreteness term alone; what is left after subtracting it and the rank-4 term is T0. This is a
few node-hours of jaxpower and removes the template degeneracy of section 2.2 entirely.

**D. The second-order response from the mocks.** Regress P_hat_l(k_high) on the band power P_hat_0(k < 0.05) of the
same mock (one scalar per mock): the slope is the "completion" J of `response_split`, measured. Its tree-level value
(the LH block) was 3-5x too large; the measured J, inserted in the completion formula, gives the hard-mode variance the
realised long-mode power induces, without the tree-level trispectrum.

**E. The l = 4 conditional-variance excess (7.9) is a Gaussian-level question; test it with the Gaussian-field
footprint run.** `run_gaussian_footprint.py` reproduces thecov's chi2/n to 0.3 %; compute var(l = 4 | l = 0, 2) on its
realisations. If it is 1.00 there and 1.07-1.11 in the mocks, the excess is a non-Gaussian (l = 4) coupling or the
estimator's line-of-sight convention; if it is 1.07 there too, it is thecov's window / LOS treatment of the (4, 4) and
(2, 4) blocks (the tripolar expansion for ell = 4 needs Lam up to 8 and the local-LOS assignment of jaxpower's
Yamamoto estimator for both pairs).

**F. Ratio-form responses** (section 1.3, second bullet): a one-line change in `responses()`, worth running alongside
the dressed power to see which keeps sigma_P2/P0 and r02 closer to the mocks once A is settled.
