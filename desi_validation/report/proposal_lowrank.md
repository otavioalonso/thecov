# Proposal: the LRG1 excess is a low-rank response term, not the window. Freeze the window, model the term.

Written after re-reading every test on `thecov2` (report, GRF test, kernel runs, hole-fraction diagnostic).
Supersedes the "next step" lists of `assessment_after_kernel.md` and `kernel_validation_results.md`.
Script: `desi_validation/rank1_excess.py`. Numbers below are from it, on the data already dumped.

## 1. The finding

The a + b(dk/0.005) split of the report cannot separate window errors from super-sample-type terms.
thecov's adjacent 0.005 bins are correlated at r = 0.4 (window width ~ 1/L), so a term that is fully
correlated across k (R(k)R(k') sigma_b^2, or the discreteness constant) adds the same amount to a bin's
variance at every bin width, while the Gaussian variance falls by (1+r)/2 rather than 1/2 per doubling:
such a term shows up ~2/3 as "a" (Gaussian-like). The "a ~ 0.035 that survives every window" is this.

Read the excess where the windowed Gaussian covariance is ~0 instead: the off-diagonal elements with
|dk| >= 0.03. There, in EVERY (ell, ell') block, mocks - thecov is proportional to P0(k) P0(k'):

| LRG1 | coherent P0 amplitude sigma_0 | coherent P2 amplitude, in units of P0 | corr(P0 mode, P2 mode) | const in (0,0) [(Mpc/h)^6] |
|---|---|---|---|---|
| NGC | 0.51% | 0.94% | +0.18 | 85 |
| SGC | 0.66% | 1.22% | 0.00 | 120 |
| SGC/NGC | 1.30 | 1.30 | | |
| QSO NGC / SGC | 0.30% / 0.58% | 0.46% / 0.90% | -0.66 / -0.49 | 380 / 570 |

- SGC/NGC = 1.30 for both modes, against sqrt(V_NGC/V_SGC) = sqrt(1.25e-9/6.49e-10) = 1.39: the
  volume scaling of a super-sample term (sigma_b ~ V^-1/2 times a weak P(1/L) factor).
- The P2 mode is ~2x the P0 mode and nearly uncorrelated with it: two long-mode components (isotropic
  delta_b for P0; tidal / line-of-sight for P2), as the redshift-space SSC formalism predicts. The P2
  response is proportional to P0, not P2, which is why a "P2 amplitude" template found 3-5% scatter.
- The (0,0) constant is the discreteness term sum w^4 (1 + alpha^3) / norm^2 (contributes only to the
  monopole: the Yamamoto Legendre factors average to delta_l0 delta_l'0). It is larger for QSO (sparser).
- The QSO numbers show the floor of the method: with 7 fitted numbers the fit absorbs ~1-2% of noise.

Adding the fitted term to a hole-corrected thecov covariance (the same 7 numbers for all k and ell):

| LRG1, k = 0.02-0.3 | chi2/n | residual var ratio l=0 (0.02-0.06 / 0.06-0.12 / 0.12-0.2 / 0.2-0.3) | joint var ratios A, A2, alpha, SN at kmax 0.2 | at kmax 0.3 |
|---|---|---|---|---|
| NGC fill128 production | 1.024 -> 0.981 | 1.04 / 0.98 / 0.95 / 0.98 | 1.07, 1.62, 0.91, 0.87 -> 1.02, 0.96, 0.91, 0.87 | 1.26, 2.12, 1.09, 1.22 -> 1.14, 1.02, 1.04, 1.11 |
| SGC fill128 production | 1.032 -> 0.994 | 1.01 / 0.98 / 0.99 / 0.95 | 1.16, 1.66, 1.09, 1.11 -> 1.09, 1.06, 1.07, 1.09 | 1.59, 1.81, 1.45, 1.42 -> 1.20, 1.12, 1.15, 1.19 |
| GCcomb | 1.026 -> 0.984 | 1.03 / 1.00 / 0.96 / 0.97 | 1.08, 1.62, 0.95, 0.99 -> 1.02, 0.99, 0.95, 0.97 | 1.31, 1.96, 1.15, 1.25 -> 1.15, 1.02, 1.06, 1.10 |

The quadrupole-amplitude error (A2: 1.6-1.7x the predicted variance, the worst number of the report) is
entirely this term. What remains at kmax = 0.3 (10-20%) is the bin-width-proportional part (connected +
discreteness trispectrum), expected and ignorable for kmax = 0.2.

The slight over-shoot on the diagonal at k > 0.12 (0.95) says the response is not exactly P0(k): the
dilation part (dP/dlnk) and the k-dependence of the discreteness terms are missing from the fit. That is
for the model, not the fit.

## 2. What this says about the window

With the low-rank term removed, the residual at k < 0.06 (where window matters most and the trispectrum
least) is: default +9-12%; fill512 +5-7% (NGC) / +4.5% (SGC); fill128 +2-4% / +1%; fill64 +1-4% / 0%;
kernel (12, default sampling) +1-3% / -2%. Monte Carlo noise of the pair counts at default sampling is
1-2%. The converged kernel at production sampling, from its ratios to fill128 in
`kernel_validation_results.md`, would sit at ~+2% / 0% at low k and ~0 at high k: the flattest of all.
Prediction to verify (step A below).

The physics is consistent with the three independent estimates of the hole effect (GRF +5-7% with ~1%
Poisson bias, thecov fill512 +4.5%, hole-fraction R_PP +3-4%): the veto holes add ~4-5% to the Gaussian
variance, and most of it comes from the large-scale (Galactic-latitude) variation of the hole density, not
from the holes themselves (R_PP is flat from 7' to 1 deg and rises only beyond). The k-independent part of
the window excess is done; what is left is at the 2% level, which is also the Monte Carlo noise.

Decision: the window term is finished. Use the xi-kernel (no free scale) or fill at nside 512 (holes <<
1/k_max, justified) as the DESI default; do not tune the fill scale: nside 64/128 fit chi2/n better only
by absorbing part of the low-rank term, which is the wrong physics.

## 3. Proposal

A. (NERSC, 1 min) Run `python -m desi_validation.rank1_excess holi-kcore2-LRG1:kernel holi-fillprod-LRG1:fill
   holi-kcore-LRG1:random-density` and `--bin QSO holi-kcore2-QSO:kernel`. Check that the converged kernel
   leaves the flattest residual (|ratio - 1| <~ 0.02 at k < 0.2 for all ell) and the same sigma_0, sigma_2.
   If so, make `kernel` (or fill512) the DESI default and close the window question.

B. (thecov core) Super-sample module with redshift-space responses, in the Wadekar & Scoccimarro (2020)
   / Li, Schmittfull & Seljak (2018) form: Cov_SSC(P_l(k), P_l'(k')) = sum over long-mode components c of
   R_l^c(k) R_l'^c(k') sigma_c^2, with c = isotropic delta_b (growth + dilation + bias response - 2 for the
   local average, which the report measured to nearly cancel for P0 at low k), and the tidal / line-of-sight
   components that drive P2. The sigma_c^2 are window integrals of the linear power that thecov already
   knows how to do: sigma_b^2 = int d^3s xi_lin(s) Q_WW(s) / (int W)^2 from the existing pair counts, and
   the tidal moments are the same with the Legendre-weighted Q_l(s) of the local line of sight. Validation
   targets are sigma_0 = 0.51% / 0.66% and sigma_2/P0 = 0.94% / 1.22% (NGC / SGC), their ratio 1.30, and
   their near-zero correlation; not chi2/n. A first check needs no code: compute sigma_b for the LRG1 caps
   from Q_WW(s) and xi_lin and compare with sigma_0 / R_0 for R_0 ~ 1.5-2.5.

C. (thecov core, cheap) Discreteness trispectrum terms from the catalogue: the constant
   sum_d w^4 + alpha^3 sum_r w^4 over norm^2 in the (0,0) block (target: 85 / 120 LRG1, 380 / 570 QSO),
   and the sum w^3 x P terms that give the k-dependence of the monopole's high-k excess. Exact given the
   catalogue and P(k); no new pair counts.

D. Report: replace Sec. 6 (bin width) interpretation with the off-diagonal decomposition; the "Gaussian-like
   a" is not Gaussian-level. Keep the a/b figure as a diagnostic, with the correlation caveat.

Not proposed: more window variants, the anisotropic kernel, tuning kernel_damping or the fill scale.
