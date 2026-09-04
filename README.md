# pkcov — Gaussian covariance of windowed power-spectrum multipoles

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
pkcov/
  wigner.py      3j symbols, n-harmonic Gaunt tensors, CouplingCoefficients (t^(1), t^(2))
  harmonics.py   normalised Legendre / spherical-harmonic tables, tripolar S in the s-hat frame
  tracers.py     Tracer (randoms, nbar, alpha), Window (W^AB, S^A), spectrum_window_pairs (P^AB)
  windows.py     TripolarWindow (pair counts -> Q), WindowLibrary (bookkeeping, symmetry, save/load)
  kernels.py     PowerSpectrumModel (arbitrary k grids), ShellKernels (p̄)
  covariance.py  GaussianCovariance (selection rules, both Wick terms, blocks, full matrix)
tests/           13 tests (see below)
example.py       two tracers on a synthetic footprint, full 171x171 covariance
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
from pkcov import Tracer, PowerSpectrumModel, GaussianCovariance

A = Tracer('A', randoms_A, alpha=alpha_A)
B = Tracer('B', randoms_B, alpha=alpha_B)

model = PowerSpectrumModel()
model.add(('A', 'A'), {0: (k, P0_AA), 2: (k, P2_AA), 4: (k, P4_AA)})
model.add(('A', 'B'), {0: (k, P0_AB), 2: (k, P2_AB), 4: (k, P4_AB)})
model.add(('B', 'B'), {0: (k, P0_BB), 2: (k, P2_BB), 4: (k, P4_BB)})

cov = GaussianCovariance([A, B], k_edges=np.arange(0.01, 0.205, 0.01), ells=(0, 2, 4), L_max=4,
                         ds=2.0, ds_pair=10.0, shot_noise=True, n_sub=5000)

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
bin width of the pair counts (Q is smooth; 10 Mpc/h is fine); `n_sub` — randoms per tracer used in
the pair counts; `s_max` — defaults to the bounding box of the randoms.

## Accuracy and cost

* Pair counts scale as `n_sub²` × (number of (Λ1,Λ2,Λ) triples, ~100 for ells ≤ 4 and L ≤ 4);
  roughly 4×10⁶ pairs/s per triple in numpy.
* **The off-diagonal elements are limited by pair-count noise in Q(s)**, not by the method.
  For a spherical test window, the diagonal is accurate to 1–2 % already at `n_sub = 3000`; the
  first off-diagonal (~7 % of the diagonal) to ~10–20 % at 3000 and ~5 % at 10 000; the second
  off-diagonal (~0.3 %) needs more. With the exact Q injected all elements agree with a brute-force
  calculation to < 0.1 %, so the residual is entirely in Q. Use as many randoms as affordable, or
  compute Q with an external pair counter and set `TripolarWindow.Q[(Λ1,Λ2,Λ)]` directly.
* Blocks: a few seconds each for 20–30 bins.

## Tests (`python -m pytest tests`)

* `test_algebra.py` — 3j vs sympy and orthogonality; Gaunt coefficients vs numerical integration;
  4-harmonic merge rule; harmonic tables vs scipy; tripolar frame formula vs direct evaluation;
  `T ∝ 3j` (Appendix A); **box-limit identity** of Section 7.1 for both Wick terms and all
  ℓ, L ≤ 4 (an independent derivation of the standard periodic-box covariance).
* `test_pipeline.py` — Q_000 of a sphere vs the analytic overlap volume; monopole covariance vs a
  brute-force `2/I² <P P |W̃(k1−k2)|²>` evaluation; distant sphere: multipole ratios
  `C_{ℓ1ℓ2}/C_00` vs the Kaiser box values (1–4 %) and block symmetry.

## Conventions

Orthonormal complex spherical harmonics with Condon–Shortley phase; Fourier convention
`f̃(k) = ∫ d³x e^{-ik·x} f(x)`; positions in Mpc/h, k in h/Mpc, n̄ in (h/Mpc)³, covariance in
(Mpc/h)⁶. All multipole indices are even.
