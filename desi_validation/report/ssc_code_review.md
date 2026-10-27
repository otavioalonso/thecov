# Code review of the super-sample covariance (`thecov/ssc.py`, `tree/responses.py`, drivers), 2026-10-27

Builds on `report/ssc_t0_review.md` (2026-10-07), which verified the algebra (Z1/Z2, squeezed limit, real-space limits,
sigma^2 prefactor and sign, local-average structure, Poisson LA terms). This pass looks at what enters the production
matrix (`ssc_holi-kcore2-LRG1.npz`, `F/LRG1/NGC/C_ssc` of the t0 bundle) and at what is still untested.

## 1. What the production SSC is made of (LRG1 NGC, units of the Gaussian diagonal, bands 0.02-0.05 / 0.05-0.1 / 0.1-0.2 / 0.2-0.3)

The matrix has rank 4 with one negative eigenvalue (-0.31 of the leading one) whose eigenvector is 96 % aligned with
the mean P_hat vector: that is the Poisson local-average term (-T P^T - P T^T + Var P P^T, approximately rank 1 along
P_hat). Splitting it off by projection on P_hat:

| block | clustering SSC (BC - LA) | Poisson LA |
|---|---|---|
| P0 x P0 | +0.0024 +0.0013 +0.0043 +0.0264 | -0.0014 -0.0056 -0.0153 -0.0233 |
| P2 x P2 | +0.0006 +0.0056 +0.0186 +0.0336 | ~ -0.0003 |
| P4 x P4 | <= 0.001 | ~ 0 |

The Poisson LA term is -5 % of the coherent P0 mode (u^T C u, u = 1/sqrt(nb)); the P0 clustering part is small
because the net monopole response is a cancellation (tree: d R_0^(0) - 2 g_0 P_hat with R_0^(0) = 3.7 P0 at
n_eff = -1.3 and 2 g_0 = 4.9: net -1.2 P0; the mocks' in-survey measurement R_0 = 1.9 in galaxy units, HANDOFF 10,
makes the net -0.25 P0, i.e. the P0 clustering SSC would be ~25x smaller still). P2 carries the SSC that matters
(R_2^(0) = 2.5 P0, R_2^(2) = 2.6 P0 at tree level, little LA cancellation).

Wishart (N-1) D against 848 mocks, model G + hybrid discreteness (+ variant), P0 / P2 / P4 / P0+P2 / P0+P2+P4:

| SSC variant | | | | | |
|---|---|---|---|---|---|
| none | 886.8 | 1012.6 | 830.3 | 3654.0 | 8276.9 |
| as run (A = 0.74 in sigma^2) | 895.9 | 914.7 | 829.9 | 3580.5 | 8206.2 |
| clustering / 0.7387 + Poisson LA | 891.5 | 895.8 | 829.7 | 3583.6 | 8209.9 |
| clustering x 2 + Poisson LA | 886.3 | 874.0 | 829.4 | 3611.9 | 8238.8 |
| clustering only (no Poisson LA) | 874.8 | 913.7 | 829.8 | 3551.3 | 8176.9 |
| Poisson LA only | 915.5 | 1014.6 | 830.3 | 3682.8 | 8306.7 |

Reading: the clustering SSC is wanted by P2 (1013 -> 914, and a larger amplitude helps further); the Poisson LA term
makes every fit worse (P0 875 -> 896). Since the term's sign and size are confirmed by a direct simulation (section 3),
the mocks are telling us that the coherent P0 budget (Poisson LA -5 %, SSC clustering with tree vs measured responses,
squeezed T0 at 2x) is not yet closed, not that the term is absent. T0 is not in these models.

## 2. Findings in the code and the drivers

1. **sigma^2 is computed from A_norm x P_lin** (`ssc_check.py` 348/368, `catalogue_covariance.py` 139 pass `(kl, A_norm Pl)`
   as `p_lin`, and `xi()` uses `self.p_lin` for the long-mode variance). A_norm = 0.739 (LRG1 NGC) is the amplitude that
   matches the Kaiser-FoG model to the mocks' P0 at k <= 0.08 with b1 fixed by P2/P0; `ssc_t0_review` 1.3 argues that
   b1 is biased high by the window, so A absorbs a b1^2 offset, not sigma_8^2. The long-mode variance is a cosmology
   quantity and should use the fiducial P_lin: the clustering SSC is then 1/0.739 = 1.35x larger (P2: 3.4 -> 4.5 % of G
   at k = 0.2-0.3). The mocks mildly prefer the larger amplitude (table above). Fix: a separate `p_lin_long` (default
   `p_lin`) for `xi()`; keep A P_lin for the responses and the collapsed-bispectrum term. Also note that `fit_amplitude`
   reads the mocks' mean P0: for Paper 2 the catalogue's own P_hat_0 must be used.
2. **Responses are tree level** (or dressed tree level); the mocks' in-survey responses are 1.27x tree for R_0 (1.9 vs
   1.5 in galaxy units) and 1.2-1.4x for R_2 (3.4-4.1 transverse vs 2.8-2.9 isotropic tree), HANDOFF 10 / updates
   2026-10-15. There is no hook in `SuperSampleCovariance` for measured responses: `responses()` returns tree or dressed
   values only. Paper 2 ("SSC with measured R_0") needs `responses=` (dict {(ell, n): per-bin array}, overriding the
   measurable entries (0,0), (2,0); tree for the rest). Because the P0 term is a cancellation, this is the single
   largest lever on the P0 SSC; for P2 it is a 1.5-2x amplitude change.
3. **`_poisson_terms`**: the collapsed-bispectrum part of T (B_c J3 / norm) is not multiplied by the dilution d_k while
   the two-point part is (P_true = measured / d, then x J / norm). Both are pair sums through the same window, so the
   B_c term should carry d_k too. Harmless (B_c part is 10-25 % of T, d = 0.8: < 0.1 % of G), fix for consistency.
   `P2 = bin_average(P_lin^2)` with A P_lin and no FoG: consistent with the Kaiser factors used (tree), fine.
4. **No internal consistency asserts**: `discreteness_integrals` recomputes I_M = alpha sum w and I_W = alpha sum w W
   from the tracer while `sigma2` uses `Window('M').integral()`, `Window('W', T, T).integral()`; they must agree (W = the
   tracer's weighted density). Add `assert |I_W / cov.I(A, A) - 1| < 1e-3`. Same for the sign/ordering of the (W, 0)-(W, 2)
   cross channel: `out[(x,n),(y,n2)] = out[(y,n2),(x,n)]` is right (a covariance), but Q^{WM} vs Q^{MW} ordering in
   `windows.get` is only right if the library is symmetric in (window1, n) <-> (window2, n2); add a unit test on a toy.
5. **Mode-direction dependence of the LA bookkeeping**: with randoms that take the data's redshifts, a radial long mode
   is carried by the randoms too (alpha fixed, norm = int (alpha n_r w)^2 realised twice: lam = {M: 0, W: 2}), a transverse
   one is not (lam = {M: 1, W: 1}, the code's `data-randoms`). The net response is R - 2 g in both cases (and, as the
   2026-10-15 synthetic showed, independent of the FKP convention: the weights' response cancels between numerator and
   norm), so only the W- vs W^2-weighting of the same long mode differs: negligible for long modes. Not a bug; document.
6. **sigma^2 channels lam = 2, 4** (which carry the n = 2 modes that dominate the P2 SSC) are still only checked through
   the pair-count machinery shared with the Gaussian term (ssc_t0_review 1.1). Direct test in section 4.
7. **Convergence of the sampled far pairs**: `n_sub` far-pair sampling enters sigma^2 at large s where xi(s) Q(s) is
   integrated over the survey size; no convergence test on record. Run with n_sub x 2 and compare sigma^2 (seconds once
   the randoms are loaded).
8. **Mock comparison and the mocks' long-mode content**: the holi mocks lack radial modes >~ 250 Mpc/h (HANDOFF 8.3-8.4),
   while `xi()` integrates the full isotropic P_lin. The LRG1 shell is ~500 Mpc/h deep and 2-3 Gpc/h wide, so a sizeable
   part of sigma^2_{00} and most of the lam > 0 structure comes from radial q ~ 0.005-0.013: the model SSC is NOT what the
   mocks contain, in a direction that could be large. A mock-fair comparison needs sigma^2 evaluated with the mocks'
   long-mode content (section 4.3); for the data the full P_lin is right.
9. `local_average_statistics` and the delta_norm comparison are weak tests with own-FKP weights (norm responds by 0.4,
   not 2, to a radial mode); the direct long-mode projections (section 4.3) are the right statistics.

## 3. Direct simulation of the Poisson local-average term (new)

Periodic box 1000 Mpc/h, 48^3 cells, Gaussian field with P = 2e4 (k/0.05)^-1.3 (sigma_delta = 0.83 on the cells, clipped
at -1), N = 31 400 Poisson galaxies, 3000 realisations; P_hat = V (|F|^2 - N) / N^2 with the realised N (as pypower's
norm, lam_total = 2) versus the mean N. Prediction (module docstring, uniform window): Cov_real - Cov_fixed =
(-8 + 4) P_i P_j / N = -4 P_i P_j / N = -1.27e-4 P_i P_j. Measured: mean over the off-diagonal pairs -1.89 +- 0.21e-4 P_i P_j
(jackknife), diagonal -1.1 to -2.7e-4 P^2; Cov(P_hat, eps) / (2 P / N) = 1.3 (Var(eps) N = 1.01). The sign and the
order of magnitude are right; the 1.3-1.5x excess is the collapsed-bispectrum part (clipping a sigma = 0.83 Gaussian
gives a large positive skewness, i.e. an effective b2 > 0), which the code includes through J3 and 2 Z2(k, -k). A
small-sigma version (sigma_delta ~ 0.3, no clipping) should land on -4/N to a few per cent: worth running as a unit test
(`tests/`), 30 s locally.

## 4. Suggested next steps, in order of cost

4.1 **Measured Poisson-LA term from the catalogue** (one afternoon, no new job: piggyback on the next
`catalogue_covariance` run). The thinning closed form already holds the per-galaxy vectors G^l_g(i). With u_g =
w_g / sum w + w_g W_g / sum w W (W_g = NX_g w_g from the catalogue columns), T^meas_l(i) = sum_g G^l_g(i) u_g and
Var^meas = sum_g u_g^2 give the catalogue's own -T P^T - P T^T + Var P P^T (P = the catalogue's P1_ell), including the
collapsed bispectrum and the exact window integrals; compare with `discreteness_covariance` (tree). Cost: one extra
(nb x N_g) @ (N_g,) product per ell. This settles the -5 % coherent-P0 term from the catalogue alone.

4.2 **p_lin_long and the responses hook** (code, one hour): `SuperSampleCovariance(..., p_lin_long=None, responses=None)`;
drivers pass the fiducial P_lin for sigma^2 and, for Paper 2, the measured R_0^(0), R_2^(0) from `response_ibisp`.
Then rerun the SSC stage (28 s) and the Wishart table.

4.3 **sigma^2 against the long modes the mocks actually have** (one debug job + local analysis). Per mock, measure the
galaxy long-mode projections with fixed weights: D^M_g = sum_g w_g / <.> - 1, D^W_g = sum_g w_g W_g / <.> - 1, and the
n = 2 analogues sum_g w_g (W_g) L_2(<q^.x^>)... (for n = 2 use the sky/radial split: the shell-mean minus the
transverse-mean decomposition of nz_scatter is the practical proxy). Their 848-mock covariance is sum g_n g_n'
sigma^2_XY + Poisson, to compare with `sigma2()` (saved as `<region>/sigma2` in the ssc npz); the regression of each
P_hat_l(k) on them is c^X_l(k) / g directly (no fit). Then (i) the mock Wishart uses C_SSC built from the measured
Cov(D) (mock-fair), (ii) the data uses the full P_lin.

4.4 **Direct test of sigma^2's lam = 2, 4 channels** (one debug job): extend `grf_randoms fields` with `--long-modes`:
for each band-limited GRF realisation (modes above the box's k_f only), compute delta at the randoms (already done) and
delta_2 at the randoms (6 tensor fields, same NUFFTs as the P2 estimator), then D^W_0, D^W_2, D^M_0, D^M_2; 40-200
realisations give their covariance to 5-10 %, to compare with `sigma2()` run with the same band-limited P_in. This is
the first check of the n = 2 channels that is independent of the Gaussian term.

4.5 **GRF-Poisson realised-norm test of the whole LA bookkeeping** (one debug job): `grf_randoms --poisson` with the
estimator normalised by the realised alpha sum_c D_c R_c (pypower) versus fixed normalisation; the covariance difference
over ~100 realisations is the LA (clustering, in-box long modes + Poisson) prediction of `ssc.py` with known inputs. Tests
items 1-3 and section 3 together on the real footprint.

4.6 Unit tests to add: section 3's box test at small sigma (expects -4/N to 5 %); `sigma2` on a toy sphere for lam = 0
(exists) plus a toy thin shell against the 4.4 Monte Carlo; the I_W consistency assert.
