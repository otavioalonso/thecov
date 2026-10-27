# Review of the discreteness computation (2026-10-27)

Scope: `desi_validation/thinning_closed_form.py` (exact thinning terms lin, quad, multipole lin), `thinning_cov.py`
(thecov stage, Gaussian split, tree terms), `disc_multipoles.py` (decomposition and mock comparison),
`catalogue_covariance.py` (driver), `thecov/tree/discreteness.py` (tree B/n, P/n^2), `GaussianCovariance.covariance_parts`.

## Derivations re-checked (correct)

- Thinning algebra. eta_g = eps_g/p - 1, <eta^2> = a = (1-p)/p. With the realised shot noise subtracted,
  P_p - P_1 = sum_g eta_g G_g + sum_{g != h} w_g w_h eta_g eta_h Re(e_g e_h^*). Linear x quadratic cross moments vanish
  (<eta_g eta_h eta_l> with h != l needs g = h = l); the quadratic piece gives a^2 [|M_1|^2 + |M_2|^2 - 2 sum w^4]. Correct.
- Mapping to the ensemble terms. Law of total covariance + B/n ~ 1/p, P/n^2 ~ 1/p^2: lin = g1 + B/n + 2 P/n^2,
  quad = g2 + P/n^2 with G(a) = G_1 + a g1 + a^2 g2. Correct.
- One-run split. S(a) = S_1 + a sum w_g^2, so G(a) = G_PP + (1 + a/r) G_PS + (1 + a/r)^2 G_SS, r = S_1 / sum w_g^2:
  g1 = G_PS/r + 2 G_SS/r, g2 = G_SS/r^2. The code had 2 G_SS/r^2 in g1 (found by the Poisson null test, fixed in df4fc80).
- Multipole linear term. G^l_g = (2l+1) w_g Re<F_l^* e_g + L_l(k^.x^_g) F_0^* e_g>_i / norm (- 2 w_g^2/norm for l = 0):
  signs, NUFFT conventions (isign = -1 for both types, modeord = 1 with fftfreq), tensor expansion of L_2, L_4,
  (2l+1) once per G, the g = h term (zero for l > 0 up to shell anisotropy, as in the estimator), fixed alpha (as in the
  Monte-Carlo thinning). The l = 0 block reproduces the monopole loop to 9e-7. Nothing found.
- Tree terms (`thecov/tree/discreteness.py`). (1/V)[(2/n)(B(k1,k2,-k1-k2) + B(k1,-k2,k2-k1)) + (1/n^2)(P(k1+k2) + P(k1-k2))]
  re-derived from the four single and two double coincidences; window integrals J_SWW = int S W^2 / norm^2,
  J_SS = int S^2 / norm^2 with S the data's own n <w^2> (randoms do not cluster); `shotnoise_scale` ~ 1 corrects the
  realised / randoms-predicted ratio; FoG per field. Correct (local-window approximation as documented).

## Bugs and biases found

1. **quad (P/n^2, exact l = 0) was aliased.** The shell-pair average of |W2(k + k')|^2 was done as a circular FFT
   correlation on the output grid, whose Nyquist is k_max + 2 k_f = 0.305; |k + k'| reaches 0.6. Wrapped pairs:
   0 % for k <= 0.15, 4 % at 0.17, 30 % at 0.22, 53 % at 0.27, 61 % at 0.30. With the LRG1 P0 shape (slope -1.75 at
   k = 0.3) the clustering part of quad is overestimated by +0.2 % (0.17), +6.5 % (0.22), +26 % (0.27), +42 % (0.30):
   the measured P/n^2 of section 16/17 at k = 0.2-0.3 (0.027 of G vs tree 0.021) is largely this. Consequences:
   measured B/n = lin - g1 - 2 P/n^2 biased LOW by twice the P/n^2 excess (~0.005-0.01 of G at k > 0.2),
   B/n + P/n^2 (what enters the hybrid P0 block) low by once; `ratio_off` (measured / tree P/n^2 from the (0,0)
   off-diagonals at k > 0.1) biased high, hence P/n^2 in the l > 0 blocks too. The Poisson null test could not see it
   (|W2|^2 is flat for an unclustered catalogue). Fixed in 5e9f97b (`--quad-pad 2`: W2 on a 2x finer-index grid, shell
   indicators embedded; matches a brute-force double sum on a toy to 4e-15; `--quad-pad 1` reproduces the old numbers).
   Costs ~8 x the old 35 s. Rerun of mock 173 submitted (catcov/LRG1_NGC_mock173_v3).

2. **thecov's Gaussian uses the device's shot noise, not the files'.** `thecov_stage` sets num_shotnoise = sum w_g^2 +
   alpha^2 sum w_r^2 with the 4 random files loaded here (28311), while the spectra files used ~8 files (27650). That is right
   for g1, g2 (derivatives at the device catalogue) but the Gaussian added to the mocks' comparison is then 2.3 % high in S:
   +2.2 % of G on the P0 diagonal at k = 0.2-0.3 (G_PS dS + 2 G_SS dS). B/n and P/n^2 are galaxy-only and unaffected.
   Assemble C = G(files' S) + B/n + P/n^2; with it the P0 mock/model diagonal at 0.2-0.3 moves from 1.051 to 1.071
   (hybrid + SSC, no T0), i.e. more variance is missing at high k than section 17 states. Not yet changed in the code:
   `disc_multipoles` should rescale G_PS, G_SS by (S_files / S_device) before the comparison.

3. **The B/n vs P/n^2 split inherits thecov's P x S term at full strength; the sum does not.** Algebra:
   G + B/n_meas + P/n^2_meas = G_PP + (1 - 1/r) G_PS + (1 - 1/r)^2 G_SS + lin - quad. The assembled covariance depends on
   thecov's shot-noise terms only through the randoms' 4 % (1 - 1/r = 0.043); the measured terms replace the galaxies'
   shot-noise Gaussian entirely. But "B/n_meas" = B/n + (G_PS^true - G_PS^thecov)/r + (realised - model P leakage).
   For l > 0 the true B/n is 0.5-2.5 % of G while G_PS is 11-48 % of G: a 3 % error of thecov's P x S term (the Poisson
   test only checks S x S, to 1.5 %) is 0.003-0.015 of G, the size of the negative l > 0 "B/n" of section 16 at all k.
   So the l > 0 measured B/n is at present a few-per-cent test of thecov's P x S term, not a measurement of B/n;
   the (0,0) B/n (7-12 % of G) is robust. The realised-P leakage is negligible for mock 173 (P0 within 0.2 % of the
   mock mean; P2 +10 % at 0.2-0.3 enters G_PS(2,2) only through its sub-dominant P2 piece).

4. **Grid vs continuum shells.** The exact terms are shell sums on this box (k_f = 0.0020, 2.5 k_f per bin); thecov is
   continuum. Mode counts deviate from the bin volume by +2.0 / -0.1 / +0.1 / +1.3 / -1.1 % in the first five bins,
   < 1 % at k > 0.1. Leaks (count ratio - 1) x g1 ~ 0.002 of G into "B/n" at k < 0.05. Minor; note for the report.

5. **The Poisson test's 1.5 % excess** is common to all l and k > 0.1, i.e. a normalisation of thecov's S window
   (kernel smoothing of the shot-noise window, I_k / int W^2 = 0.77-0.85) rather than a line-of-sight effect. 0.1-0.3 %
   of G. State in Paper 1.

## Mock comparison with T0 and block-wise hybrids (mock 173, this run, 848 mocks, Wishart (N-1) D for P0+P2+P4)

Without T0: tree 8397, hybrid (0,0) 8131, (0,0)+(2,2) 8905, all diagonal blocks 9298, all measured 9846 (P2 alone:
tree 907 vs measured 915; the full-vector penalty of the measured (2,2) block, ~700, is far larger than its own-block
penalty, 8). With tree T0 added: 9605 / 8444 / 9085 / 9448 / 9950 and min eigenvalue 0.29-0.40 (the squeezed T0,
2x overpredicted, pushes the full vector towards non-PD; P2 and P4 alone improve, P0 worsens). Scaling the P2 hard
block of T0 by 1.7 (section 13 amplitude, diagnostic only) leaves measured (2,2) worse than tree (860 vs 853): the
compensation "tree B/n too high + tree T0 too low" explains part but not all of the l = 2 picture.
Recommendation unchanged (measured in (0,0), tree for l > 0) but the reason is now explicit: the l > 0 measured
terms are limited by thecov's P x S term, not by the thinning algebra.

## Next steps, in order

1. Re-decompose mock 173 with the padded quad (job submitted) and the files' shot noise in G; redo the section 16/17
   tables and the mock 174 / Abacus runs (3 debug jobs). Expect (0,0) B/n up by ~0.01 at k > 0.2, P/n^2 down to ~tree.
2. GRF-Poisson null test of the P x S term (1 debug job). `grf_randoms --poisson p0 --rsd f --sigma` already samples a
   band-limited Gaussian field with known P_in by the randoms in the footprint; save one realisation as a catalogue and
   run the closed form + thecov with P_in (power_input 'unmasked'). For a Gaussian field B = 0 (clipping aside), so
   lin - 2 P/n^2_tree(P_in) must equal G_PS(P_in)/r + 2 G_SS/r in every block: a direct per-cent test of G_PS for
   l = 0, 2, 4, and the arbiter of the negative l > 0 "B/n". Use sigma ~0.5 to keep the clipping bispectrum small.
3. Monte-Carlo thinning check of lin_ll' and quad_ll' on the real catalogue (1 debug job): `thinning_cov thin --ells 0 2 4
   --frac 0.25 --nthin 100 --single`; the covariance of the thinned P_l minus the exact a lin gives quad for l > 0 at
   ~15 % (good enough for a 1-2 % term) and replaces the tree x ratio_off approximation; validates the l > 0 lin
   end to end on the real footprint.
4. Use the catalogue's own P_l (lightly smoothed) in thecov's Gaussian for the decomposition (Paper 2 rule; removes
   the realised - model leakage identically) — `thecov_covariance` currently takes `model_from_mocks(spec)`.
5. Then revisit the l > 0 choice with the sum, not the split: Wishart of G(files S) + lin - quad (+ tree T0 hard, SSC).
