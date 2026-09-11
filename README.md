# thecov — Gaussian covariance of windowed power-spectrum multipoles

Implementation of the separation-space (tripolar) formulation derived in `pk_covariance.typ`:

```
C^{ABCD}_{l1 l2}(i,j) = (4π)^4 / (I_AB I_CD)  Σ_{L1 L2} 1/((2L1+1)(2L2+1))  Σ_{λ λ'} (-1)^{(λ+λ')/2}  Σ_{Λ1 Λ2 Λ}
                        [ t^(1) I^(1)_ij + t^(2) I^(2)_ij ],
I^(n)_ij = Σ_{pairs} ∫ s² ds  p̄'^{(i)}_{L1 λ}(s)  p̄^{(j)}_{L2 λ'}(s)  Q^{ω ω'}_{Λ1 Λ2 Λ}(s)
```

* `p̄^{(i)}_{Lλ}(s)` — model `P_L(k)` averaged over k-bin *i* against `j_λ(ks)` (cosmology; 1D integrals)
* `Q^{ωω'}_{Λ1Λ2Λ}(s)` — tripolar window functions from pair counts of the randoms (geometry; once per survey)
* `t^(1), t^(2)` — Wigner-symbol coupling coefficients (pure numbers; cached)

Approximations: Gaussian field, local plane-parallel, window slowly varying over a correlation
length. Everything else (k-bin average, angular structure, shot noise, multi-tracer) is exact.

## Layout

```
thecov/
  wigner.py      3j symbols, n-harmonic Gaunt tensors, CouplingCoefficients (t^(1), t^(2))
  harmonics.py   normalised Legendre / spherical-harmonic tables, tripolar S in the s-hat frame
  tracers.py     Tracer (randoms, nbar, alpha), Window (W^AB, S^A), spectrum_window_pairs (P^AB)
  windows.py     TripolarWindow (pair counts -> Q), WindowLibrary (bookkeeping, symmetry, save/load)
  kernels.py     PowerSpectrumModel (arbitrary k grids), ShellKernels (p̄)
  covariance.py  GaussianCovariance (selection rules, both Wick terms, blocks, full matrix)
tests/           test suite (see below); `pytest -m "not slow"` skips the long validations
example.py       two tracers on a synthetic footprint, full 171x171 covariance
box_test.py      the uniform-cube test of box_test.ipynb, with the finite-window expectation
diagnostics/     scripts for setting tolerances / inspecting intermediate quantities
mocks/           Gaussian mock pipeline + end-to-end validation driver
```

Dependencies: numpy, scipy (sympy only for one optional test).

## Inputs

**Randoms**, one dict per tracer:

| key | shape | meaning |
|---|---|---|
| `POSITION` | (N,3) | comoving Cartesian coordinates, **observer at the origin** (the line of sight is x̂) |
| `WEIGHT`   | (N,)  | total weight w(x) applied to the density field (FKP × completeness × …) |
| `NZ`       | (N,)  | optional: mean density n̄(x) at each random. If absent, n̄ is estimated from the randoms with a k-nearest-neighbour density (warning issued) |

plus `alpha` = (weighted galaxies)/(weighted randoms): the randoms sample n̄/α. It normalises the
windows and enters the shot noise as (1+α).

**Model multipoles** `P_L^{AB}(k)` on any k grid (cubic-spline interpolated; the grid must cover the
covariance bins). Missing multipoles are zero; cross spectra are symmetric in the tracers.

## Usage

```python
import numpy as np
from thecov import Tracer, PowerSpectrumModel, GaussianCovariance

A = Tracer('A', randoms_A, alpha=alpha_A)
B = Tracer('B', randoms_B, alpha=alpha_B)

model = PowerSpectrumModel()
model.add(('A', 'A'), {0: (k, P0_AA), 2: (k, P2_AA), 4: (k, P4_AA)})
model.add(('A', 'B'), {0: (k, P0_AB), 2: (k, P2_AB), 4: (k, P4_AB)})
model.add(('B', 'B'), {0: (k, P0_BB), 2: (k, P2_BB), 4: (k, P4_BB)})

cov = GaussianCovariance([A, B], k_edges=np.arange(0.01, 0.205, 0.01), ells=(0, 2, 4), L_max=4,
                         ds=2.0, ds_pair=10.0, shot_noise=True, n_sub=5000, n_near=200000, s_split=80.0)

spectra = [('A', 'A'), ('A', 'B'), ('B', 'B')]
cov.compute_windows(spectra)            # pair counts (geometry) — expensive, once per survey
cov.save_windows('windows.npz')         # ... and reusable:  cov.load_windows('windows.npz')

cov.set_model(model)                    # cosmology — cheap, repeat at will
C, labels = cov.covariance(spectra)     # data vector ordered by spectrum, ell, k-bin
blk = cov.block(('A','A'), ('A','B'), 0, 2)   # a single (nbins x nbins) block
```

`labels[n] = (A, B, ell, i)` for row/column *n*. `covariance(..., symmetrize=True)` averages the
matrix with its transpose (exact for the first Wick term; the residual asymmetry measures the
x̂'→x̂ step in the second term and is ~1e-6 in the distant-observer limit).

Parameters: `ds` — step of the s grid for the radial integrals (≲ π/2k_max); `ds_pair` — radial
bin width of the pair counts (Q is smooth; 10 Mpc/h is fine); `n_sub`, `n_near`, `s_split` — see
below; `s_max` — defaults to the bounding box of the randoms. The model k grid must cover the
covariance bins (down to k = 0 if the first bin starts there).

## Accuracy and cost

* **Counting is split from contracting.** Because x' = x + s, the tripolar weight depends on a pair
  only through (r1, s, mu) with r1 = |x| and mu = x^ . s^ -- the azimuth about s^ vanishes
  identically, so there is no cos(m dphi) factor and no second unit-vector array. Pairs are
  histogrammed into cells of (r1, s, mu), and S is evaluated once per cell at the cell's WEIGHTED
  MEAN (first-order accurate, so the resolution is not critical in r1 or s). All triples share one
  set of counts, and the inner loop is one distance and one dot product. This is also the shape an
  external counter plugs into: Corrfunc/pycorr bin in exactly these variables (one call per radial
  shell), and a GPU kernel would need nothing more.
* `n_mu` must resolve oscillations of order max(Lam1, Lam2) in mu -- up to 12 for ells and L up to
  4 -- and defaults to 6x that. A fixed small value silently biases the high multipoles. Because it
  adapts to the multipoles actually requested, raising L_max changes the mu resolution and hence the
  discretisation of every triple; the effect is ~1e-3 on the off-diagonals, which is the size of the
  residual discretisation error. Pass an explicit `n_mu` when comparing two runs that differ in
  L_max or ells.
* **Backends.** `backend='auto'` (default) uses JAX if it imports, else numpy; `'jax'` and
  `'numpy'` force the choice. Only the counting step differs, and the two agree bitwise, so they can
  be compared directly. On CPU the JAX path runs the all-pairs kernel about 5x faster
  (27M vs 5M pairs/s for 1e8 pairs), because XLA fuses the chunk into one pass instead of
  materialising several (block x n2 x 3) temporaries, and it runs on a GPU unchanged. Two details
  matter for the speed: the pair arithmetic may run in float32 but the ~1e8 accumulations into
  ~1e5 cells are summed in float64, and equally-spaced bin edges (the usual case) are indexed
  arithmetically instead of by searchsorted. Chunks of ~2e5 pairs are optimal; larger ones are
  slower, so `chunk_pairs` should not be raised for JAX.
* `s_split` is snapped to the nearest pair-count bin edge. A bin that straddled the split would get
  near pairs only below it and far pairs only above it, while being normalised by its whole volume,
  and would come out low by the missing volume fraction (15 % for a bin of width 25 split at 80).
  With the defaults (`ds_pair=10`, `s_split=80`) the split already fell on an edge, so this only
  ever bit configurations with an incommensurate `ds_pair`.
* Pair counts are split in two regimes: all pairs of an `n_sub` subsample for `s ≥ s_split`
  (cost ∝ n_sub²) and a KD-tree neighbour search on a larger `n_near` subsample for `s < s_split`
  (cost ∝ n_near × local density), where the small subsample would have far too few pairs. The exact
  one-point value Q(s=0) ∝ ∫ω ω' anchors the radial spline, and bins with fewer than `min_pairs`
  pairs are dropped. About 4×10⁶ pairs/s per (Λ1,Λ2,Λ) triple in numpy; ~100 triples for
  ells ≤ 4, L ≤ 4.
* **The off-diagonal elements are limited by pair-count noise in Q(s)**, not by the method. For a
  spherical test window: diagonal accurate to ~1 % at `n_sub = 3000`; first off-diagonal (~7 % of the
  diagonal) to ~3 % at 3000 and ~1 % at 10 000; second off-diagonal (~0.3 %) to ~5 % at 10 000.
  With the exact Q injected all elements agree with a brute-force calculation to < 0.1 %, so the
  residual is entirely in Q. Far off-diagonals fluctuate around zero at the 0.1–0.4 % level of the
  diagonal; if you need them, increase `n_sub` or inject Q from an external pair counter
  (`TripolarWindow.Q[(Λ1,Λ2,Λ)]`).
* Blocks: a few seconds each for 20–40 bins; the model can be changed at no geometric cost.

## Multi-tracer

Tracers are independent `Tracer` objects; spectra are ordered pairs `(A, B)` where the Legendre
weight sits on the first label. `cov.covariance([('A','A'), ('A','B'), ('B','B')])` returns the
joint matrix. Points to be aware of:

* **Cross windows.** `W^{AB}` is sampled by A's randoms with `nbar_B w_B` taken from the nearest
  random of B, zeroed when that random is farther than `mask_factor` x the LOCAL spacing of B's
  randoms (estimated from the `mask_knn`-th neighbour, which has far less scatter than the first).
  Both refinements matter: a global threshold truncates the sparse outskirts of a tracer with a
  steep n(z), and a first-neighbour threshold masks interior points whose nearest random sits in a
  close pair. Always check `cov.I('A','B')` against an independent estimate of the overlap integral.
* **alpha and NZ must be consistent**: the randoms sample nbar/alpha. `Tracer` warns if the random
  density and NZ imply an alpha more than 15 % away from the one given; an inconsistent pair
  mis-normalises the exact Q(s=0) anchor relative to the pair counts.
* **Block symmetry.** In each correlator the derivation evaluates the power spectrum at one of the
  two momenta; the other choice differs by O(1/(k L_survey)) and is equally valid. The two agree on
  the diagonal but not off it, so the raw expression gives
  C^{ABCD}_{l1 l2}(i,j) != C^{CDAB}_{l2 l1}(j,i) at the level of ~0.3 % of the diagonal scale.
  `block()` returns the average of the two by default (`symmetrize=True`); `covariance()`
  symmetrizes the assembled matrix, so it is unaffected. `block(..., symmetrize=False)` gives the
  raw expression, and the difference between the two is a useful diagnostic of the approximation.

### Multi-tracer tests (`tests/test_multitracer.py`)

No mocks are needed: each test is an exact identity or an independent brute-force reference.

1. **Split-sample identity** (any geometry). Split one catalogue into halves A and B. The full-sample
   estimator is linear in the halves, so summing the 4 x 4 ordered-spectrum blocks and dividing by 16
   must reproduce the single-tracer covariance of the full sample -- window, leakage, multipoles and
   shot noise included (the halves carry nbar/2 each, the cross carries none, and the sum has to
   rebuild 1/nbar). Zero free parameters.
2. **Periodic-box ratios.** Two tracers in the same distant cube with different nbar, alpha and
   biases, against the multi-tracer Gaussian formula
   `C = (2l1+1)(2l2+1)/N int dmu/2 L_l1 L_l2 [P^AD P^CB + P^AC P^DB]` times the cube's leakage
   factor. Tests the term structure, shot-noise placement and I_AB normalisation with distinct
   spectra. Within each (l1, l2) the ratio must be common to all tracer combinations; across
   (l1, l2) it need not be, since the leakage depends on the mu-structure of the integrand.
3. **Two different footprints** (top-hat sphere and Gaussian profile, independent randoms), l = 0,
   against a brute-force shell average with analytic window transforms. This is the case where the
   cross window has to be interpolated from a foreign catalogue. Off-diagonals are judged on the
   scale of the diagonal, not relatively: these windows are narrow in configuration space, so the
   bin-to-bin leakage is a per-cent effect and a relative tolerance would only measure pair-count
   noise.
4. Positive-definiteness of the assembled multi-spectrum matrix.

`python -m diagnostics.box_multitracer_ratios [n_sub]` prints the ratios of test 2 grouped by
(l1, l2), for setting the tolerances at your own n_sub and box size.

## The box test (`box_test.py`, `tests/test_box.py`)

Uniform randoms in a 4 Gpc/h cube, flat P_0 = 10⁴, P_2 = P_4 = 0, α = 1, compared with the
periodic-box formula 2/N_modes × (…) with the coefficients of the classic Gaussian covariance
(reproduced by `tests/test_box.py::test_notebook_coefficients`). Two things must be kept in mind
when reading the ratio:

1. **A finite cube is not a periodic box.** The estimator couples k₁ to k₂ = k₁ + q with q drawn
   from |W̃(q)|², of width ~1/L. For L Δk = 40 about 10.6 % of each bin's variance moves to the two
   neighbouring bins (≈ 4.6 % each) and ~1.4 % further out, so the diagonal is 0.894 × the periodic
   value and σ/σ_periodic = 0.946. `leakage_fractions` predicts this by Monte-Carlo sampling of q
   from the cube's |W̃|² and the code reproduces it bin by bin (`test_box_monopole_leakage`).
   The variance is conserved: the volume-weighted row sums Σ_j (V_j/V_i) C(i,j) return to the
   periodic value up to pair-count noise.
2. **Shot noise.** For the FKP field δ = n_g − α n_r the noise is (1+α)/n̄, not 1/n̄: with α = 1
   (as many randoms as galaxies) it is twice the naive value and, for this configuration, dominates
   P_0 (6.4×10⁴ vs 10⁴). Comparing with an analytic expression that uses 1/n̄ gives a ratio ≈ 2.7
   instead of ≈ 0.9 (`test_shot_noise_factor`). Use α ≪ 1 (many randoms) or (1+α)/n̄ in the reference.
3. **Line of sight.** With the box centred on the observer the Yamamoto weights L_ℓ(k̂·x̂) vary
   across the box; for ℓ = 2 this spreads the window-multipole power to larger q and produces
   extra leakage (diagonal 0.857 instead of 0.894) and a small negative C_02 (−1.5 % correlation).
   Both vanish for a distant box (`test_box_far_multipoles`, `test_box_origin_varying_los`), where
   ℓ = 2 has the same leakage factor as ℓ = 0 and C_02 = 0.

`PYTHONPATH=. python box_test.py [distance]` reproduces the notebook's plots with the leakage-corrected
expectation overlaid (`box_test_d0.png`).

## Tests (`python -m pytest tests`)

* `test_algebra.py` — 3j vs sympy and orthogonality; Gaunt coefficients vs numerical integration;
  4-harmonic merge rule; harmonic tables vs scipy; tripolar frame formula vs direct evaluation;
  `T ∝ 3j` (Appendix A); **box-limit identity** of Section 7.1 for both Wick terms and all
  ℓ, L ≤ 4 (an independent derivation of the standard periodic-box covariance).
* `test_pipeline.py` — Q_000 of a sphere vs the analytic overlap volume; monopole covariance vs a
  brute-force `2/I² <P P |W̃(k1−k2)|²>` evaluation; distant sphere: multipole ratios
  `C_{ℓ1ℓ2}/C_00` vs the Kaiser box values (1–4 %) and block symmetry.
* `test_box.py` — the uniform-cube test described above.
* `test_invariance.py` — transformations under which the answer is known not to change, so no
  reference calculation is needed: w_A -> c w_A (exact); halving the randoms while doubling alpha
  (statistical — this is the alpha/NZ consistency that Tracer only warns about); a rigid rotation of
  every catalogue (exact, and the only end-to-end test of the tripolar machinery as a rotational
  scalar); padding L_max with a vanishing P_4 (exact). Plus the split-sample identity repeated with
  spatially varying weights — the only test with non-uniform w(x) — and the P = 0 limit, where the
  answer is pure Poisson noise and the S-S window family is isolated.
* `test_units.py` — pieces previously exercised only indirectly: `ShellKernels.average` against
  adaptive quadrature (including s = 0, the shot-noise branch and the automatic node count), the
  `PowerSpectrumModel` contract (symmetry, missing multipoles, out-of-range k), and a save/load
  round trip for the windows.
* `test_tripolar_window.py` (marked `slow`) — the anisotropic components of Q against direct 2D
  quadrature for a uniform sphere **centred on the observer**, where the line of sight sweeps the
  whole sky. Until now only Q_000 and Q_002 had ever been checked against anything. It also verifies
  the analytic s = 0 anchor, which had been derived by hand and never tested. Run the fast suite
  with `pytest -m "not slow"`.

## Diagnostics

* `python -m diagnostics.box_multitracer_ratios [n_sub]` — leakage ratios of the multi-tracer box
  test, grouped by (l1, l2), for setting tolerances.
* `python -m diagnostics.convergence [--footprint sphere|shell] [--quick]` — sweeps `ds`, `ds_pair`,
  `n_sub`, `n_near`, `s_split`, `min_pairs` one at a time and reports the drift of the diagonal and
  of the first two off-diagonal correlations against a converged reference. Use it to choose these
  parameters from measured numbers rather than from the rules of thumb above.

## End-to-end validation with mocks (`mocks/`)

    python -m mocks.run_validation --n-mocks 300 --grid 256 --nproc 8 --out results/

This is the only test that probes the **approximations** rather than the implementation. Multi-tracer
Gaussian mocks are generated in a realistic window and their sample covariance is compared with the
thecov prediction.

* **Geometry**: a spherical cap between 400 and 900 Mpc/h with a smooth n(z) per tracer, three
  circular holes in the angular mask and a completeness gradient. The two tracers share the angular
  mask but have different n(z), bias and stochasticity.
* **Mocks**: a Gaussian delta_m on a grid, tracers delta_X = b_X delta_m + n_X, Poisson-sampled cell
  by cell, then displaced by f (Psi . r^) r^ **along each galaxy's own line of sight**. Nothing in
  the generation assumes a global line of sight, so the local plane-parallel approximation is being
  tested, not assumed.
* **Estimator**: Yamamoto-FKP with the Cartesian-moment FFT decomposition (1 FFT for l = 0, 6 more
  for l = 2, 15 more for l = 4), CIC assignment with window deconvolution.
* **Model**: exact, because the mocks are built from a known spectrum --
  P_XY(k, mu) = (b_X + f mu^2)(b_Y + f mu^2) P_lin + delta_XY N_X.

### Statistics

Element-by-element agreement on the diagonal needs ~2/eps^2 mocks (800 for 5 %). The whole matrix is
tested far more cheaply by chi^2_i = (d_i - dbar)^T C_thecov^-1 (d_i - dbar), whose mean is
n_dim (1 - 1/N_mock) with a relative error sqrt(2 / (n_dim N_mock)) -- about 1.2 % for 48 elements
and 300 mocks -- and which is sensitive to the off-diagonal structure as well. The script also
reports the eigenvalues of C^-1/2 Chat C^-1/2 (mean 1, spread sqrt(2/N_mock)), the diagonal ratios
and the first off-diagonal correlations, and writes a plot plus `comparison.npz`. Runs can be
extended with `--resume`.

### Systematics of the test itself (not of thecov)

* **Box size.** The covariance couples modes separated by |q| ~ 1/R_survey and the mocks sample that
  structure at the box spacing 2 pi / L, so a small box biases the *mock* covariance. Run at two
  values of `--box-factor`.
* **Aliasing.** Keep k_max below about 0.4 k_Nyquist; the script prints the ratio and warns.
* **The cell window.** Galaxies placed uniformly inside their cell realise the *cell-averaged*
  density, which suppresses the clustering by T(k) = prod sinc(n_i/N) on top of the CIC assignment
  window. The field generator pre-divides by T to cancel it exactly, and the RSD displacement is
  taken from the galaxy's own cell so that it carries the same single power of T. Without this the
  measured P is ~10 % low at half-Nyquist and the covariance ~20 % low. `tests/test_mock_pipeline.py`
  verifies the cancellation.
* **Independent randoms per tracer.** Sharing one random catalogue between two tracers makes the
  -alpha n_r piece of the FKP field common to both, so its Poisson noise survives in the CROSS
  spectrum as a spurious constant alpha/nbar (5 % of P at k = 0.02 in the test set-up, 18 % by
  k = 0.1). `survey.Catalogues` gives each tracer its own randoms; the same care is needed with
  real catalogues.
* **Clipping.** Poisson sampling needs a non-negative intensity, so 1 + b delta is clipped at zero.
  This removes power: at sigma(b delta) ~ 0.9 the measured P is ~12 % low, which would be misread as
  a covariance failure. The default amplitude keeps sigma(b delta) ~ 0.3; the driver prints it and
  the clipped fraction at start-up and warns above 0.35. `--amplitude` rescales it.
* **Non-Gaussianity.** The low amplitude also keeps the field close to Gaussian, which is what the
  formula assumes; raise `--amplitude` if you want to see the departure (but watch the clipping).

`tests/test_mock_pipeline.py` (mostly `slow`) validates the mock machinery itself where the answer is
known exactly -- field variance, the shot-noise constant, monopole and Kaiser quadrupole recovery in
a periodic box, the absence of shot noise in the cross spectrum, and consistency between the random
catalogues and the grid integrals -- so that a failure of the end-to-end comparison can be attributed
to thecov rather than to the mocks.

## Not covered

Every test here validates the *implementation* against exact identities or independent references.
None of them probes the two physical **approximations**: the local plane-parallel treatment and the
assumption that the window varies slowly over a correlation length. That is what `mocks/` is for.
What remains uncovered after it: non-Gaussian (connected trispectrum and super-sample) contributions,
which are outside the scope of this formula altogether, and any effect of real survey systematics
(fibre collisions, imaging weights, redshift failures) on the window.

## Conventions

Orthonormal complex spherical harmonics with Condon–Shortley phase; Fourier convention
`f̃(k) = ∫ d³x e^{-ik·x} f(x)`; positions in Mpc/h, k in h/Mpc, n̄ in (h/Mpc)³, covariance in
(Mpc/h)⁶. All multipole indices are even.
