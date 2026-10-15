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
