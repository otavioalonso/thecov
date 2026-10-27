# Calibrating the non-Gaussian covariance from the catalogue itself

*Design note, 2026-10-21. Status of each piece as of thecov `thecov2` bdc247b; numbers from the holi-v3 LRG1/QSO NGC runs
recorded in HANDOFF.md (sections 7, 9, 12 and the 2026-10-15/19 updates).*

## 1. The rule

The covariance of the DESI power-spectrum multipoles will be built once, for one realisation of the data. Every input
must therefore be one of

1. a quantity measured on that catalogue (galaxies and randoms) — the spectrum itself, its fitted parameters, pair and
   quadruplet sums over the galaxies, the response of the small-scale power to the long modes present in the survey;
2. a quantity fixed by the window alone (pair counts of the randoms, long-mode variances of the footprint);
3. a quantity fixed by cosmology (the linear power spectrum, the growth-only separate-universe responses).

Nothing may be an amplitude fitted to a mock ensemble. The mocks are used only to validate: the whole covariance is built
from one mock catalogue as if it were the data and compared with the covariance of the 848 (LRG1) / 859 (QSO) mocks; the
spread of the result over a few single catalogues is the uncertainty we would carry on the data.

The decomposition is

    C = G[P_l, W] + C_disc[B/n, P/n^2] + C_SSC[sigma^2_L, R_l] + C_T0[squeezed: R_1, R_2; hard: tree],

and the question for each term is which of its ingredients is in classes 1-3 and how well it is measured on one
catalogue.

## 2. Inventory of single-catalogue measurables

| ingredient | estimator | status | per-catalogue precision |
|---|---|---|---|
| P_l(k), window-convolved, with norm and num_shotnoise | the estimator (pypower / jaxpower), `PowerSpectrum.from_estimator` | in production | the data's own noise (see section 5) |
| b1, f, sigma_v (FoG), b2 | the power-spectrum fit; sigma_v from P2/P0 | in production (ssc_check) | fit errors; b2 poorly constrained |
| B/n, P/n^2 (galaxies' own bispectrum and sampling, one-halo pairs included) | thinning closed form (`thinning_closed_form.py`): pair and quadruplet sums over the galaxies | implemented, validated against Monte-Carlo thinning, P0-P0 block | exact given the catalogue; its cosmic variance is what the mocks measure |
| R_l(k; k_L), first-order response of P_l to in-survey long modes | integrated bispectrum in Fourier space (`response_ibisp.py`) | validated on lognormal fields (R = 2 to 5 %); 6 mocks measured | 2-3 % (R_0), 5-10 % (R_2) per catalogue |
| R_0(q -> 0) - 2, net response to the survey-mean mode (local average) | identity with the in-survey R_0 at small k_L (update 2026-10-15 (d)) | verified on 848 mocks and on synthetic fields | inherits R_0's 2-3 % |
| sigma^2 of the long modes, radial and transverse | P_L x |W(q)|^2 from the randoms (thecov's window pair counts, SSC module) | in production (isotropic); anisotropic weighting pending | cosmology + window, no noise |
| J_l(k), second-order response | `response_ibisp` prints the leading term | exists, untested | too noisy on one survey (section 4.4) |
| dP_l / d ln k, d^2 P_l / d ln k^2 | smooth fit of the measured P_l | trivial | set by the smoothing of P_l |

Two things are *not* in the inventory: the second-order response R_2 as a measurement, and any amplitude of the hard
trispectrum. Section 4 says what replaces them.

## 3. Term by term

### 3.1 Gaussian term

G is a functional of the window (randoms) and of the spectrum. On the data the spectrum is the measured P_l itself,
flagged `window_convolved=True` with the estimator's norm and shot noise, so that thecov applies I / I_k (the kernel
window's dilution). Nothing is calibrated. Two choices remain and should be fixed now:

- **Which P_l goes in.** The measured multipoles carry their own noise (0.5 % per bin for P0, more for P2/P4), and the
  Gaussian term is quadratic in them. Options: the raw measurement (noise of ~1 % on the diagonal, correlated across the
  matrix), a smoothed version (spline with a smoothing scale of a few bins), or the best-fit model convolved with the
  window. The random-field tests used the mocks' mean, which is not available on the data. The smoothed measurement is
  the choice closest to what was validated; the model-convolved option ties the covariance to the fit. Recommendation:
  the smoothed measurement, with the raw one as a cross-check of the ~1 % effect.
- **Shot noise.** num_shotnoise of the estimator, from the catalogue (sum w^2 of the data plus alpha^2 sum w^2 of the
  randoms); already an input of `PowerSpectrum`.

### 3.2 Discreteness terms (B/n, P/n^2) — fully from the catalogue

The closed form of the thinning covariance. Thinning the galaxies with probability p and reweighting by 1/p,
F_p(k) = sum_g (w_g / p) eps_g e^{ik.x_g} - F_r, the covariance of P_p over the Bernoulli variables is a sum over pairs
and quadruplets of galaxies with known coefficients in a = 1/p - 1; its expectation over the clustering is

    E[Cov_thin(p)] = [G(S_p) - G(S_1)] + a B/n + (2a + a^2) P/n^2 ,

where G(S_p) - G(S_1) is thecov's Gaussian term at the thinned shot noise (known), the coefficient of a is the galaxies'
collapsed bispectrum term B/n (after removing the 2 P/n^2 piece: B/n = lin - g1 - 2 P/n^2), and the coefficient of a^2
is the pair term P/n^2. The sums are over the actual galaxies, so one-halo pairs, exclusion and any non-Poisson sampling
are in them; there is no model for the bispectrum. The computation is one pass of pair counts over the galaxies (about a
node-hour per cap), with no Monte-Carlo noise.

Results on mock 173 (LRG1 NGC): B/n 1.2x tree level on the P0 diagonal and 1.6x off the diagonal, P/n^2 1.2x; QSO:
P/n^2 1.9x, B/n consistent with ordinary clustering (no duplicates, 3-7 % close pairs). These replace the "mocks want
2x the discreteness terms" of HANDOFF 9.4 and are the model's values, measured rather than fitted.

What remains:

- **The other blocks.** The closed form has been run for the P0-P0 block. The l = 2, 4 blocks and the cross blocks need
  the Legendre weights in the galaxy sums (the same change as for the Gaussian term's RSD-aware windows); until then the
  tree discreteness terms scaled by the P0 off-diagonal ratio are used, which is a stopgap and makes P0+P2+P4 non-PD.
- **The q ~ 0 ridge.** At |i - j| <= 1 the measured terms contain the Gaussian-like ridge of nearly equal momenta, which
  belongs to the Gaussian term's bin-width-dependent part; it is replaced by continuity from |i - j| = 2, 3. Document it
  as part of the estimator.
- **Cosmic variance of the single catalogue.** The closed form is the realisation's own bispectrum and quadruplet
  statistics. Their scatter between catalogues is the irreducible error of this route; measure it on 5-10 mocks
  (thinning closed form on each) and carry it as the discreteness term's uncertainty.

### 3.3 Super-sample covariance — variances from the window, responses from the catalogue

SSC = (long-mode variances of the window) x (responses of P_l to a long mode), plus the local-average term for the
realised alpha and norm. The three ingredients:

1. **sigma^2_L of the super-survey and in-survey long modes**: P_L(q) |W(q)|^2 from the randoms' pair counts (thecov's
   SSC windows). Cosmology and window only; no calibration. Pending: the anisotropic weighting. For a thin shell the
   super-survey mode is transverse-dominated, so the quadrupole response that enters is the transverse one (R_2 = 3.4-4,
   not the isotropic 2.8); the radial variance comes from the same P_L |W|^2 with the shell's n(z), fixed by the data.
2. **The responses R_l(k)**: measured on the catalogue with the integrated-bispectrum estimator, at several k_L; the
   q -> 0 limit is the SSC response. On the holi LRG1 mocks R_0 = 1.9 (1.2-1.5x tree level) and R_2 -> 3.4-4; each
   single catalogue gives it to 2-3 %. The estimator subtracts the shot-noise three-point term with the w^2-weighted
   catalogue, so one-halo physics is in the measured response, as it should be.
3. **The local-average term**: the estimator's norm and alpha respond to the survey-mean mode, so the net P0 response
   is R_0 - 2 for the data-randoms norm convention (R_0 - g_0 for other conventions; `norm_kind`). The survey-mean
   mode itself is invisible in-survey, but the identity "net survey-mean response = in-survey R_0(q -> 0) - 2" was
   verified on the 848 mocks (regression on the shell density) and on synthetic fields with an imposed uniform mode.
   So the measured R_0 fixes this term too. Its Poisson part (the variance of alpha and norm from the finite galaxy
   number) is an integral over the randoms and data weights, already in `SuperSampleCovariance`.

Implementation step: `SuperSampleCovariance` takes R_l(k) as input (measured arrays per multipole and k band) instead of
the tree-level a P + c dP/dlnk; the tree level stays as the default for checks. SSC is subdominant above k ~ 0.1 for
these samples, so a 3 % error on R is harmless; the term matters for the low-k bins and the P0-P2 coupling.

### 3.4 Connected trispectrum T0 — the hard case

The tree-level T0 (snake + star, b3 = 0) against the mocks with the measured discreteness terms in place (HANDOFF 12):
amplitude 0.38 +- 0.05 (P0), 0.61 +- 0.05 (P2), 0.43 +- 0.03 (P0+P2); PD at amplitude 1. The overshoot is the squeezed
coupling (k 0.02-0.05 x 0.2-0.3: tree 0.12 vs mocks 0.067 in correlation, P0), while the hard-hard part is roughly right
for P0 and low for P2. A fitted amplitude is excluded by the rule, so the two blocks are treated separately.

**Squeezed block (k_1 << k_2): the response formalism.** For two soft and two hard modes the trispectrum is the response
of P(k_h) to the long modes: a first-order piece (R_1^2, the snake with soft momentum transfer) and a second-order piece
(R_2 P_L(k_s)^2 P(k_h), the star and the rest of the snake), each with its angle dependence (Barreira & Schmidt 2017).
In the response formalism R_n(k) = [growth-only coefficients G_n] + [dilation: d^n ln P / d ln k^n of the measured P]
+ [bias: b1, b2]. Of these, the dilation part is the data's own spectrum, the bias part is the fit, and the growth-only
coefficients are cosmology (26/21 at tree level; measured in separate-universe simulations to k ~ 1 h/Mpc, Wagner et
al. 2015, nearly independent of the sample). So the squeezed block becomes a functional of the measured P_l with no
amplitude. The first-order piece is directly checkable: the formalism's R_1 must reproduce the measured R_0 = 1.9 (and
the transverse R_2) on the same catalogue — if it does not, the measured R_1 replaces it and the discrepancy is carried
into R_2 as a systematic. The second-order estimator J_l of `response_ibisp` is a weak independent check (section 4.4).

**Hard block (both k >= k_split): the tree with the data's dressed power.** No response shortcut exists. The allowed
inputs are the tree kernels with the measured nonlinear P_l (IR-damped, FoG with the data's sigma_v) in place of P_L,
i.e. the `Dressed` power already used for the SSC responses. Whether this brings the hard P0 amplitude (0.35-0.5 against
the mocks) to 1 is to be tested; the hard block and B/n are both smooth at high k and partly degenerate, so the test must
hold B/n at its measured value. If the hard block stays low, that is a limitation of the tree level to be stated (and
bounded with the mocks), not tuned away.

## 4. Validation protocol (mocks as the judge, never as the input)

1. **One catalogue as the data.** Build G, B/n, P/n^2, SSC (measured R), T0 (response squeezed + dressed hard) from mock
   173 alone (LRG1 and QSO, NGC and SGC), with the pipeline that will run on the data.
2. **Compare with the ensemble covariance** of the 848 / 859 mocks: Wishart deviance per multipole set and k_max,
   per-block diagonal ratios in k bands, the mock/model variance along the leading non-Gaussian direction, the
   G-whitened eigenvalue spectrum against the Marchenko-Pastur edge, and the squeezed and hard mean correlations. The
   targets are the residuals of HANDOFF 7.9 / 9.4, which today are 1.02-1.03 in chi2/n for LRG1.
3. **Realisation scatter.** Repeat step 1 on mocks 174-178: the spread of the single-catalogue covariances (per term and
   in the parameter variances) is the uncertainty to quote for the data covariance. The thinning closed form and the
   responses are the only noisy inputs; the window pieces are not.
4. **Second-order response check.** On the 6 mocks, compare the formalism's R_2 with the leading-term J_l estimate
   averaged over the mocks (not a single-catalogue quantity, but a test of the formalism's R_2 for this sample).
5. **A different galaxy physics.** The 25 AbacusSummit complete mocks (section 7.7) as an independent ensemble for the
   same single-catalogue construction; the SSC differs (periodic boxes) and is left out of that comparison.
6. **Data.** Only after the blinding sign-off: the same pipeline on the DR2 catalogues; the measured B/n, P/n^2 and R_l
   of the data compared with the mocks' values is itself a test of the mocks' small-scale galaxy physics.

## 5. Noise and feedback of data-derived inputs

- **P_l in G and in the responses.** The ~1 % noise of the measured multipoles enters G quadratically and the dilation
  responses through derivatives; smoothing over a few bins removes most of it. Quantify on the mocks: G built from each
  mock's own P_l against G from the mocks' mean (a few hundred cases are cheap, the windows are shared).
- **Correlation between the covariance and the data.** The closed-form terms and the responses are statistics of the
  same realisation whose covariance they describe. The effect on parameter errors is of order the fractional cosmic
  variance of those statistics times the term's weight in C (a few per cent of a 10-20 % term), i.e. sub-per-cent on
  parameter errors; the realisation scatter of step 3 bounds it.
- **b2.** The bias term of the responses and the collapsed bispectrum in the local-average Poisson term depend on b2,
  which the fit constrains poorly. Use the fit's b2 with its error propagated, or the Lazeyras et al. b2(b1) relation
  as the prior; the sensitivity is small for SSC and should be quantified for the squeezed T0.

## 6. Status and work list

| step | status | cost (per tracer and cap) |
|---|---|---|
| Thinning closed form, P0-P0 block | done, validated | ~1 node-hour |
| Closed form for l = 2, 4 and cross blocks | to do (Legendre weights in the galaxy sums) | same |
| Response estimator R_l(k; k_L) | done, validated; measured on 6 mocks | ~0.5 node-hour |
| SSC with measured responses and anisotropic sigma^2 | to do (`SuperSampleCovariance(responses=...)`) | minutes |
| Squeezed T0 from the response formalism (R_1, R_2 of the measured P_l) | to do (new module in `thecov/tree`) | minutes |
| Hard T0 with the dressed measured power | to do (option of `TrispectrumCovariance`) | ~10 min, 64 processes |
| Single-catalogue build on mock 173 and the ensemble comparison | to do (`t0_tree_fit.py` has the comparison code) | hours |
| Realisation scatter on mocks 174-178 | to do | ~10 node-hours |
| Abacus complete cross-check | prepared (`run_abacus_t0_test.sh`) | ~5 node-hours |

Order: the l = 2, 4 closed form first (it removes the stopgap that breaks P0+P2+P4), then SSC with measured responses,
then the squeezed T0, then the hard-block test. Each step is validated against the ensemble before the next.
