# Code review of the tree-level T0 (2026-10-27; complements report/ssc_t0_review.md section 2 of 2026-10-07)

Files: `thecov/tree/trispectrum.py` (kernels, parallelogram, multipoles, TrispectrumCovariance), inputs from `ssc_check.py`
(`fit_b1`, `fit_sigma_v`, `fit_amplitude`), `desi_validation/t0_run.py`, `t0_tree_fit.py`.

## Verified (no bug found in the kernels or the quadratures)

- F3, G3 (`FG3`) against the one-loop P13 kernels (Makino et al. 1992 / Scoccimarro 2004): angle-averaged F3(k, q, -q) and
  G3(k, q, -q) reproduce [12/r^2 - 158 + 100 r^2 - 42 r^4 + 3/r^3 (r^2-1)^3 (7 r^2+2) ln] / (3024 r^2) and
  [12/r^2 - 82 + 4 r^2 - 6 r^4 + 3/r^3 (r^2-1)^3 (r^2+2) ln] / (1008 r^2) to 1e-6 at r = 0.1 ... 4 (IR limit -1/(18 r^2) for both,
  Galilean invariance). F2 symmetric, F3 permutation-invariant (1e-34).
- `Z1`, `Z2` (Galileon basis) against `tree.responses._Z1, _Z2` ((b2, bs2) basis, independent code): 1e-16.
- Snake and star assembly of T(k1, -k1, k2, -k2) re-derived term by term (12 Z2 Z2 terms x 4, 4 Z3 terms x 6; the P(0) ones
  dropped as the SSC's); the q1 + q2 = 0 limits (alpha, beta ~ 1/eps against G2 = O(eps^2)) are right.
- Quadratures (production settings n_mu 20, n_psi 40, mu12 64 + 32 end nodes) converged to < 1e-5 with the FoG damping at
  k = 0.1 and 0.2 (diagonal and |i-j| = 3). The eightfold-symmetric grid (3698315) is exact.
- k-bin average: `n_k = 1` (production) vs 3: the diagonal is low by 1.9 / 5.2 / 5.8 % (P0xP0 / P2xP2 / P0xP2) at k = 0.1 and
  1.0 / 2.1 / 2.9 % at k = 0.2; |i-j| = 1 by 0.5-2 %; |i-j| >= 2 < 0.3 % (`responses/t0_review/t0_nk_convergence.csv`). Use
  n_k = 2 (4x the 8 s) for anything that reads the T0 diagonal.

## The dominant problem is the input parameters, not the code

`ssc_check.fit_b1` fits the pure Kaiser P2/P0 to the window-convolved mock-mean multipoles at k <= 0.06, then `fit_amplitude`
rescales P_lin by A = P0 / (Kaiser-FoG x P_lin) = 0.739 (LRG1 NGC). The window suppresses the convolved quadrupole at these k,
so b1 = 2.202 is high and A = 0.74 (a 26 % deficit of linear power for mocks built with the fiducial cosmology) absorbs it:
sqrt(A) b1 = 1.893 is the b1 the P0 amplitude wants. The 2026-10-07 review flagged this; the production inputs were not
changed. For T0 it is not a rescaling: T0 ~ Z1^4 Z2^2 P^3 and the (b1, A) split changes f/b1 (0.35 -> 0.40), b2/b1^2 and the
sign of the Lazeyras b2 (0.283 at 2.202, -0.262 at 1.893). Ratios of T0 (snake + star, with the production FoG) for the
alternative split, summed over the (k1, k2) pairs of each block (`t0_parameter_split_sensitivity.csv`):

| block (k pairs)            | ells  | b1 1.893, A 1, b2 -0.262, bs2 -0.510 | b1 1.893, A 1, production b2, bs2 |
|---------------------------|-------|--------------------------------------:|----------------------------------:|
| hard (both k >= 0.15)     | P0xP0 | 0.61 | 1.50 |
|                           | P2xP2 | 1.15 | 1.52 |
|                           | P4xP4 | 1.29 | 1.53 |
|                           | P0xP2 | 1.18 | 1.58 |
| squeezed (k1 < 0.1 <= k2) | P0xP0 | 0.88 | 1.50 |
|                           | P2xP2 | 1.36 | 1.53 |
|                           | P0xP2 | 1.32 | 1.54 |

So the block amplitudes of HANDOFF 13 (P0 hard 1.00 +- 0.08, P2 hard 1.6-1.8, squeezed 0.5 / 0.68 / 0.27) are conditional on
the production split. With the Lazeyras-consistent split the mocks would want ~1.65 (P0 hard) and ~1.4-1.6 (P2 hard): a common
factor, which nonlinear power inside the kernels can supply (P_nl/P_lin ~ 1.2-1.3 at k = 0.2-0.3, cubed), instead of an
ell-dependent one. The same inputs feed the tree discreteness terms (section 16, b2 and bs2 enter B/n at O(1)) and the SSC
responses (a_00 = 19.1 at b1 = 2.2 includes b2).

## J4 = int W^4 / norm^2 uses the local window

`window_integral` sums alpha w_r W_r^3 with W the randoms' local weighted density, which resolves the veto holes at the
randoms' resolution. The Gaussian term uses the pair-averaged window (I_k / int W^2 = 0.77-0.85 for LRG1 NGC: the local
int W^2 is 15-23 % above the one the k = 0.02-0.3 modes see); int W^4 weights the spikes more, so J4 is likely 20-40 % above the
effective value. This is a direct amplitude error of T0 of the same size as the parameter effects above. Cheap test: paint
the randoms on grids of cell 2, 5, 10, 20 Mpc/h and compute sum_c W_c^4 V_c / norm^2 (and sum_c W_c^2 V_c against I_k) to see
where it converges; or define J4 through the same kernel smoothing as the Gaussian.

## Minor

- The T0 near-diagonal (|k1 - k2| below the window width, ~0.01 radially for the LRG1 shell) adds configurations the SSC already
  carries; from int dq P(q)/q this is a few per cent of the diagonal T0 only (ssc_t0_review 2.2 third bullet).
- `sigma_fog` damps the four external legs only; the internal P(|k1 +- k2|) is the (IR-damped, A-scaled) linear power up to
  |k1 + k2| = 0.6, where P_lin is not the galaxy power (the 'meas' variant of t0_run.py is the test of this).

## Next steps, in order

1. **Fix the inputs from the catalogue (no mock numbers):** fit (b1, sigma_v) with the window-convolved Kaiser-FoG model
   (thecov's window / Q_L(s) Hankel transform, or the DESI window matrix for the mock) to mock 173's own P0, P2 at k = 0.02-0.15;
   require A = 1 with the fiducial P_lin as the consistency check. b2: Lazeyras(b1) as default; the catalogue's measured growth
   response R1 (section 14, 1-2 %) contains 2 b2/b1 and can fix b2 given b1 (then bs2 = -4/7 (b1-1)). One debug job for the fit
   plus the 8 s T0; redo the section 13 block analysis and the section 16 tree terms with the new inputs.
2. **J4 convergence with cell size** (10-minute node job), then adopt the effective J4 (or the kernel-smoothed definition).
3. Set n_k = 2 in `catalogue_covariance` for the T0 stage.
4. With 1-3 in place, repeat the hard-block comparison with the 'meas' (nonlinear P inside the kernels) variant: if the P0 and P2
   hard blocks come out at a common amplitude near 1, the tree T0 + measured P is the Paper 1 model and the squeezed block is the
   only stated limitation.
