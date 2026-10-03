# Assessment after the converged kernel run: stop refining the window, test at the parameter level

Written by the cloud session that designed `WindowSmoothing`, after reading
`kernel_validation_results.md` (NERSC run on 1a6794b). Audience: the next Claude session working on
the DESI validation. This file supersedes the "next test" recommendation at the end of
`kernel_validation_results.md`. It discusses LRG1 throughout; QSO validates with every window.

## 1. Summary

1. **Averaging the window over the veto holes is real and large.** Any way of doing it removes about
   two thirds of the LRG1 excess:
   - the hand-tuned fill window;
   - the pair-averaged kernel window, converged or not.

   The physics is understood: W = m² at a point overweights hole edges (∫m²/norm = 1.225 for LRG1
   against 1.11 for QSO). The Gaussian-random-field test on the real footprint confirmed it
   independently.
2. **The remaining ~0.02–0.04 in χ²/n is not identified.** The window variants differ at that level in
   ways we cannot attribute. Tuning the window further to absorb it would be fitting, not physics.
3. **A larger part of the excess is not a window effect at all.** Both of these are the same for every
   window, so they are non-Gaussian:
   - the bin-width term b (≈ 0.04 / 0.02 / 0.01 at ℓ = 0 / 2 / 4);
   - most of the excess at k > 0.2 (a ≈ 0.10–0.13 at ℓ = 0).
4. **Decision:** freeze the window question. Keep `WindowSmoothing` in core as a documented option.
   Use a simple, fixed hole-averaged window for DESI. Then decide whether anything more is needed from
   the **parameter errors**, not from χ²/n.

## 2. Evidence

### 2.1 Every hole-averaging method recovers most of the excess

LRG1 GCcomb, χ²/n over k = 0.02–0.3, Δk = 0.005, ℓ = 0, 2, 4. All rows use default pair-count
sampling unless marked.

| window | χ²/n |
|---|---|
| default, W = m² | 1.098 |
| fill, healpix nside 512 (7′ ≈ 2.5 Mpc/h) | 1.067 |
| fill, nside 256 | 1.051 |
| fill, nside 128 (≈ 10 Mpc/h) | 1.046 |
| fill, nside 64 (≈ 20 Mpc/h) | 1.040 |
| kernel, single-stage, 12 kernels (unconverged) | 1.056 |
| kernel, two-stage converged, **production sampling** | 1.065 |
| fill nside 128, **production sampling** | 1.026 |

Production sampling (`--target-near-pairs 1e10 --n-randoms-max 8e6`) lowers χ²/n by about 0.02
relative to default sampling (fill128: 1.046 → 1.026). Most of that is Monte Carlo noise in the pair
counts inflating χ².

### 2.2 The fill window has no plateau

Each doubling of the healpix pixel scale lowers χ²/n by ~0.006 (1.067, 1.051, 1.046, 1.040). The
mocks keep preferring *more* dilution than any physically motivated scale gives.
- An earlier statement in this session ("fill converges at nside 128") was wrong. There is a trend,
  not convergence.
- So "fill at the scale that fits best" is a free parameter, and it should not be tuned on χ²/n.

### 2.3 The converged kernel is *worse* than the unconverged one at matched sampling

`kernel_validation_results.md` concludes that tolerance does not matter, because 12 kernels give 1.048
and 19 kernels give 1.049 (NGC, k = 0.02–0.2). Those two runs used *different* pair-count sampling:
- the 12-kernel run used default sampling;
- the 19-kernel run used production sampling, which lowers χ² by ~0.02 (see 2.1).

At matched sampling, converging the basis therefore *raised* χ²/n by about 0.02. The truncated basis
happened to over-dilute the high-k windows, which is the direction the mocks want.

The converged kernel is the better *calculation* of the pair-averaged Gaussian window: I_k is accurate
to 1e-5, and the first-bin calibration moved towards the Gaussian random field's value. It is the
worse *fit* to the mocks at k > 0.06.

### 2.4 Two candidate explanations in `kernel_validation_results.md` do not hold

**"The isotropic kernel ignores redshift-space anisotropy."**
- This cannot explain the 2–3% gap between the kernel's predicted calibration and the Gaussian random
  field's c(k) at the middle and last bins, because that test is isotropic by construction (P2 = P4 = 0;
  see the docstring of `run_gaussian_footprint.py`).
- Its sign for the mocks is also unclear:
  - on small scales, fingers of God concentrate the correlation along the line of sight, where angular
    holes do not dilute pairs (less dilution, the wrong direction);
  - on larger scales, the Kaiser effect does the opposite.

**"The Gaussian random field's c(k) is the target at high k."**
- The fields live on a 6 Mpc/h mesh, and the window on nside-512 pixels (≈ 2.5 Mpc/h) with
  fill-fraction dilution. The field therefore has almost no correlation below ~6 Mpc/h.
- That pushes its c(k) towards the broad-kernel limit (more dilution) exactly where the high-k kernels
  have most of their weight (r ≲ 10 Mpc/h).
- At high k, the field test and the kernel disagree mainly about ξ at small r. It is not ground truth
  there.

### 2.5 The high-k kernels depend on an uncertain input

For k ≳ 0.06 the kernel K_k(r) = ξ₀(r) j₀(kr)/P₀(k) takes much of its weight from r ≲ 10 Mpc/h.
There, ξ₀ is set by `extend_power`: the measured P0 continued as a power law from k = 0.3 to k = 5,
with 1 Mpc/h Gaussian damping.

The redshift-space monopole is suppressed by fingers of God well before k = 5, so this probably
overestimates small-r ξ₀. The kernel is then too narrow and under-dilutes, which is the observed sign.

The tempting next step is to vary the damping, cut the extrapolation, or use the mocks' measured
ξ₀(s). That brings back a free smoothing input in disguise, which is the same trap as tuning the fill
scale. **Do not pursue it unless the parameter-level test (Section 3) shows the window still matters.**

### 2.6 The non-window part is larger than the residual window ambiguity

The excess splits as var ratio − 1 = a + b·(Δk/0.005), using the NGC+SGC mean.
- b is identical for all windows: 0.04 / 0.02 / 0.01 at ℓ = 0 / 2 / 4. It is non-Gaussian, of the
  local-average / super-sample type.
- At k = 0.2–0.3, a ≈ 0.10–0.13 at ℓ = 0 for every hole-averaged window. That is also beyond the window.
- Earlier projections on amplitude-like modes (overall P0 and P2 amplitudes) found excess variance of
  the super-sample type. This is the most likely place where the parameter errors could be affected.

## 3. Recommended next step: parameter-level check (no new NERSC runs)

χ²/n tests all 168 modes equally. DESI cares about a few parameter directions. Test those directly with
the data already dumped (`report_data_*.npz` in `~/thecov_desi/holi_v3_mock173`).

1. **Templates** D = ∂(data vector)/∂θ, built from the mock mean P̄_ℓ(k) at k = 0.02–0.2:
   - A0, A2: the amplitudes of P̄_0 and P̄_2 separately (∂/∂A_ℓ = P̄_ℓ in multipole ℓ, 0 elsewhere).
     This also probes super-sample-like modes;
   - α_iso: −d P̄_ℓ / d ln k for all ℓ (BAO and shape dilation);
   - optionally ε, the Alcock–Paczynski warping, mixing ℓ via the standard first-order expressions;
   - optionally a Kaiser-like f direction (∂P0, ∂P2 at fixed b from the linear Kaiser ratios).
2. **For each covariance C:** default, fill128 (production), kernel two-stage, kernel single-stage.
   - predicted errors: F = Dᵀ C⁻¹ D, σ²_pred = diag(F⁻¹);
   - per-mock linear estimates: θ̂_n = F⁻¹ Dᵀ C⁻¹ (V_n − V̄);
   - report the ratio R_θ = Var_mocks(θ̂) / σ²_pred, each θ alone and jointly;
   - give bootstrap errors over the 859 mocks;
   - do it for NGC, SGC and GCcomb, and repeat for QSO as the control.
3. **Read-out:**
   - if R_θ is within ~1.00–1.05 (errors within ~2.5%) for the α's and f with the hole-averaged windows,
     the covariance is done for practical purposes. Freeze the window choice and write it up;
   - if the amplitude directions (A0, A2) are the outliers, the issue is super-sample / local-average
     variance. The next piece is the SSC module (Wadekar & Scoccimarro-type response), not window work;
   - the choice between fill128 and the kernel should be made on these numbers. If they agree to
     ~1–2% in σ, prefer the cheaper one with the stated physical scale.

A script for this belongs in `desi_validation/` (e.g. `param_projection.py`), reading the same npz
layout as `compare_kernel.py`. Note the Hartlap factor and the mock-mean subtraction when you compare
variances.

## 4. What not to do now

- Do not tune `kernel_damping`, `extend_power`, or the fill nside to minimise χ²/n.
- Do not build the anisotropic (redshift-space) kernel yet.
- Do not make `kernel` the DESI default. Fill at a fixed, stated scale is cheaper and fits at least as
  well. `WindowSmoothing` stays in core (tested, documented, harmless for QSO) for surveys where it
  matters.

## 5. Status of the code

- `thecov/smoothing.py`: two-stage basis. Stage 1 is the converged kernel basis, which gives I_k;
  stage 2 is the compressed window basis (B ≈ 8–9 for DESI), which goes into the pair counts.
  `GaussianCovariance(..., smoothing=...)` and `set_model(..., masked=True|False)` are in core. Tests are
  in `tests/test_smoothing.py`.
- `desi_validation/`:
  - `pipeline.py` modes `fill`, `kernel`;
  - `dump_report_data.py` (with `--kernel-tol` and `--kernel-max-basis`);
  - `compare_kernel.py` (χ²/n per k range, variance ratios, a/b split).
- Reference results: `desi_validation/report/kernel_validation_results.md`.
- Rules for runs at NERSC are unchanged:
  - debug queue only, at most 2 jobs at once, one job per cap;
  - real paths, no placeholders;
  - branch `thecov2`;
  - no model names in commits.
