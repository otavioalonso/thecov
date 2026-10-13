# Bispectrum covariance in thecov: implementation of the Gaussian class-1 term

Branch `thecov2`, 2026-10-13. Derivation: `report/bispectrum_covariance_guide.typ` (and `bispectrum_covariance.typ`).
This note records what is implemented, the conventions the code actually uses, the validation, the costs and what remains.

## 1. Module map

| module | content |
|---|---|
| `thecov/bispectrum_algebra.py` | Sugiyama basis (`H`, `N`, `sugiyama_S_tilde`, `sugiyama_S`); `cg_tensor`; the bipolar vertex coefficients `bipolar_coefficients` (and `legendre_coefficients_opening`); the leg/anchor Gaunt tensors; `BispectrumCoupling` (coupling numbers `T_n`, batched over `(K1, K2, Lam3)`, cached; `index_set`); `invariant_basis` (Φ_n, for tests); `box_window_value`; `phase`. |
| `thecov/bispectrum.py` | `extrapolated_power`; `BispectrumShellKernels` (shell nodes, `jbar`, node-wise Bessel tables, vectorised vertex matrices `vertex_coefficients`); `Class1Assembler` (radial labels and the double radial integral for all `(i j; i' j')`); `BispectrumGaussianCovariance` (user class). |
| `thecov/bispectrum_windows.py` | `rotation_to_z`; `MomentLabels` (bipolar harmonics of the pair geometry); `AnchoredMoments` (pair sums → `Q_n(r1, r2)`); `ThreePointWindowLibrary` (request / compute / interpolate / save / load). |
| `tests/test_bispectrum_algebra.py` | basis, bipolar coefficients, five-direction identity for `T_n`, box identity. |
| `tests/test_bispectrum_kernels.py` | vectorised vertex, tapered-box radial assembly, the full class in a tapered box (shot noise, second pairing). |
| `tests/test_bispectrum_windows.py` | uniform sphere vs Monte-Carlo quadrature, self-pair subtraction, shot-noise kinds, library round trip. |
| `desi_validation/bispectrum_dev/` | `angular_check.py`, `radial_check.py`, `sphere_check.py` (reference quadratures used by the tests); `grf_sphere.py`, `grf_theory.py`, `grf_compare.py` (Gaussian-field validation). |

Usage (single tracer):

```python
from thecov import Tracer, PowerSpectrumModel, BispectrumGaussianCovariance
cov = BispectrumGaussianCovariance(tracer, k_edges, multipoles=((0, 0, 0), (2, 0, 2)), L_max=4, lam_w=4)
cov.compute_windows(verbose=True)       # anchored pair sums (geometry), once per survey; save_windows / load_windows
cov.set_model(model)                    # PowerSpectrumModel with P_0, P_2, P_4 of the tracer (any k range: extrapolated)
C, labels = cov.covariance()            # labels (l1, l2, L, i, j); elements i <= j for l1 = l2
```

## 2. Conventions actually used

* **Harmonics.** Orthonormal complex `Y_lm` with the Condon–Shortley phase (`harmonics.sph_harm_table`), `conj(Y_lm) = (-1)^m Y_{l,-m}`; Gaunt integrals from `wigner.gaunt_tensor`; Clebsch–Gordan `<a α b β|L γ> = (-1)^{a-b+γ} sqrt(2L+1) (a b L; α β -γ)` (`cg_tensor`).
* **Sugiyama basis.** `H_{l1 l2 L} = (l1 l2 L; 0 0 0)` (so `H_202 = +1/√5`, status item (i)), `N = (2l1+1)(2l2+1)(2L+1)`, `S~ = Σ (3j) Y Y Y`, `S = (4π)^{3/2}/(H √N) S~`; checked `S_000 = 1`, `S_202 = L_2(a·n)`, `S_220 = L_2(a·b)`, orthogonality `<conj(S) S'> = δ/(N H²)`.
* **Which momentum carries the spectrum.** The primed estimator (shells `i', j'`) carries the three power spectra; the unprimed one (shells `i, j`) only bin-averaged Bessel functions. `block(symmetrize=True)` averages the direct block with the transposed mirror, as `GaussianCovariance.block` does.
* **Vertex.** `conj(Y_{L3 γ})(p̂3) P_{L3}(|p1+p2|) = Σ_ab c^{L3}_{ab}(p1,p2) Σ <aα bβ|L3 γ> conj(Y_aα)(p̂1) conj(Y_bβ)(p̂2)`; `c` by Gauss–Legendre in `p3 ∈ [|p1-p2|, p1+p2]` (the substitution removes the `√(1+μ)` behaviour at `p1 = p2`; 64 nodes). `L3 = 0` reduces to `c^0_aa = (-1)^a √(4π/(2a+1)) p_a`.
* **Invariant basis and T_n.** All five harmonics unconjugated (status item (ii)): `Φ_n = Σ (K1 K2 Λ3; κ1 κ2 μ3) [Y_λ1(r̂1)⊗Y_L1(ŷ1)]_{K1κ1} [Y_λ2(r̂2)⊗Y_L2(ŷ2)]_{K2κ2} Y_{Λ3μ3}(x̂)`, orthonormal; `T_n = ∫ conj(Φ_n) conj(S~_ℓ)(r̂1,r̂2,x̂) 𝒜`, built from `leg_tensor` (`∫ Y_{ℓ'm'} conj(Y_LM) conj(Y_aα) conj(Y_Λν)`), `leg_projection_tensor` (`∫ conj(Y_λρ) conj(Y_ℓm) Y_Λν`) and `anchor_tensor` (`∫ conj(Y_Λ3μ3) conj(Y_LM) Y_L'M' Y_L3γ`). `K_a` runs over all integers in `|λ_a − L_a| … λ_a + L_a`. All `T_n` are real; the `Φ_n` with `K1 + K2 + Λ3` odd are imaginary and their `Q_n` never enter (`Q_n` comes out real to 1e-16).
* **Box value.** `Q_n^box = V √(N_{L1 L2 Λ3}/4π) H_{L1 L2 Λ3} δ_{λ1 0} δ_{λ2 0}` — the guide's value (a first derivation that gave `(4π)^{3/2}` was wrong by `4π`; the box identity test settles it).
* **Phase.** `(-i)^{ℓ1+ℓ2} i^{Λ+Λ'} = (-1)^{(Λ+Λ'−ℓ1−ℓ2)/2}`, real by parity.
* **Second class-1 pairing.** `(1 2')(2 1')(3 3')` = the first with `ℓ' → (ℓ'2, ℓ'1, L')` and the primed shells transposed, `C^{(1b)}[i,j,i',j'] = C^{(1a)}_{ℓ,ℓ'_sw}[i,j,j',i']` (`S~` is symmetric under the swap for even `ℓ'1 + ℓ'2 + L'`).
* **Shot noise.** `W P_L + S δ_{L0}` at each of the three points: an `S` leg has `L = 0` and `P → 1`; an `S` anchor has `L3 = 0` and `c → √(4π) δ_a0 δ_b0`; eight `(anchor, leg1, leg2) ∈ {W, S}^3` combinations with the invariants `Q_n^{XYZ}` of the corresponding weights (`W = m²`, `S = (1+α) n̄ w²` as in `tracers.Window`).
* **Normalisation.** `I_3 = ∫ ω³ = α Σ_r w_r m_r²` from the randoms (`set_normalization` overrides it).
* **Model range.** `extrapolated_power` continues `P_L` below the tabulated `k_min` as a power law (the third side reaches `p3 → 0` on the `i' = j'` shells) and as `k^{-3}` above `k_max`.
* **Radial assembly.** `C[i,j,i',j'] = Σ_{q1∈i', q2∈j'} c[q1,q2] Σ_{r1 r2} (jbar_i A_q1)(r1) Q̃(r1,r2) (jbar_j B_q2)(r2)`, `Q̃ = Σ_n T_n Q_n`, done as one `(nb·nb·nq × nr) @ (nr × nr)` product per radial label `(L1 L2 L3 a b Λ Λ')`, then per primed shell `i'` an einsum over `q1` with the vertex matrix. Nothing larger than `nb² nq nr` is held.
* **Pair sums.** Anchors: `n_anchor` host randoms (seed offset 7919), weights `w~_a` (W: `w m`; S: `(1+α) w²`), `α_a = α N/n_anchor`; each anchor's frame rotated so `x̂ = ẑ` (`μ3 = 0`, `κ2 = −κ1`, `Y_{Λ3 0}(ẑ) = √((2Λ3+1)/4π)`). Far neighbours (`r ≥ s_split`): all pairs with the `n_sub` subsample; near (`r < s_split`): KD-tree of the `n_near` subsample; moments `A(x; b) = α_l Σ_{y∈b} w~_l(y) [Y_λ(r̂)⊗Y_L(ŷ)]_{Kκ} / ∫_b r² dr`. Self pairs (same neighbour in both factors, `b1 = b2`) subtracted exactly, grouped by moment-label pair. `Q_n` on the pair bins (`dr_pair`) is interpolated onto the `(r, r)` grid by a bicubic spline, constant beyond the first/last bin centre.

## 3. Validation

1. **Five-direction identity** (`test_coupling_*`): at fixed magnitudes and five independent random directions, `Σ_n T_n Φ_n` with the radial factors equals `(-i)^{ℓ1+ℓ2} conj(S~_ℓ)` × the direct product-quadrature of `<< e^{i p1·r1 + i p2·r2} S~_ℓ' P(p1;ŷ1) P(p2;ŷ2) P(p1+p2;x̂) >>` to 1e-8 (relative), for isotropic and anisotropic `P_L` (`L = 0, 2, 4`, polynomial so that the vertex expansion is finite) and all pairs of `000, 202, 220`. This is the decisive check of the Gaunt chain, the conjugation placement and the `λ > 0` ranks.
2. **Box identity** (`test_box_identity`): `Σ_n T_n Q_n^box` reproduces `<< conj(S~) S~' P P P >>` with a global line of sight (first term of Sugiyama et al. 2020 eq. 49) to 1e-8.
3. **Bipolar coefficients**: exact reconstruction of `conj(Y_{L3})(p̂3) P_{L3}(p3)` for polynomial `P` (1e-10), 1e-4 at `a_max = 30` for a BAO-like `P`; the `L3 = 0` Legendre reduction to 1e-10; the vectorised node-pair version equals the scalar one to 1e-16.
4. **Radial assembly in a tapered box** (`test_radial_assembly_*`, `test_class_tapered_box_*`): with `Q_n = Q_n^box exp(−(r1²+r2²)/2σ²)` the leg integrals become smoothed shell indicators (1D, fast convergence) and `Class1Assembler` / `BispectrumGaussianCovariance` (with shot noise and both pairings) agree with the independent reference to 1e-12. The un-tapered box identity converges only as `1/r_max` (3 % at 1000 Mpc/h, 0.8 % at 4000: adjacent top-hat shells' bin-averaged Bessel functions overlap logarithmically) — in a survey the window supplies the taper.
5. **Window invariants on a uniform sphere** (`test_sphere_invariants*`): against Monte-Carlo quadrature of `∫ d³x dΩ1 dΩ2 W W W Φ_n` (observer at the centre, `R = 500`), isotropic and anisotropic `n` including `λ > 0`, `L > 0`, `K ≠ λ`: within 3 % with 5000 anchors / 10000 far neighbours, within 8 % with 1500 / 4000. The estimator is unbiased; the scatter comes from the anchors (a 2000-anchor subsample has ~1.5 % radial-distribution fluctuations, which on this small sphere move the edge fraction by 2 %) and from the far neighbour catalogue shared by all anchors (its density fluctuations do not average over anchors). The `r1 = r2` bins follow the off-diagonal trend to 1 % (self-pair subtraction correct); the shot-noise kinds scale exactly by `(1+α)/n̄`; `Q_n` is real to 1e-16.
6. **Gaussian fields on the sphere** (`grf_sphere.py`, `grf_theory.py`, `grf_compare.py`): see section 4.

Fast tests (`-m "not slow"`): 26 tests in ~4 min. Slow ones: anisotropic five-direction checks, the converged sphere, the anisotropic tapered-box class check.

## 4. Gaussian-field test

Setup (`grf_sphere.py`): isotropic P(k) = 2e4 (k/0.02)/(1 + (k/0.02)^2.6) on a 128^3 mesh in a periodic box of
L = 1300 Mpc/h (cell 10.2, k_Nyq = 0.31), observer at the box centre, footprint a sphere of R = 500 (the `sphere_check`
tracer: n̄ = 3.8e-5, α = 0.1, 200k randoms), F = n̄ 1_sphere (δ − mean_sphere δ), shells 0.04–0.20 in steps of 0.04,
estimators B_000(i ≤ j) and B_202(i, j) (Sugiyama FFT estimator with the local line of sight), 800 realisations; the
same estimator on the unmasked periodic box (800 realisations); and a quadratic field δ + 0.3 (δ² − σ²) in the box
(200 realisations) whose mean B_000 = 2ε[PP + 2 perms] checks the estimator normalisation (ratio 1.00 ± 0.01 on the
k ≥ 0.1 elements; 0.90–0.97 at low k where the bin-centre evaluation of the third side is crude).
Theory (`grf_theory.py`): `BispectrumGaussianCovariance` on the sphere with λ_w = 0, 2, 4 (3000 anchors, 6000 far
neighbours, dr_pair = 20, dr = 4, r_max = 1000), and the analytic class-1 box covariance (first term of eq. 49, both
pairings, lattice mode counts; `box_class1`). Comparison: `grf_compare.py`; figure `bispectrum_grf_sphere.png`,
numbers `grf_sphere_results.npz`.

Results.

* The three truncations λ_w = 0, 2, 4 agree to 3 % on every element: the rank expansion of the window is converged
  at λ_w = 2 for this (smooth) footprint; the whole line-of-sight structure of B_202 is carried by the Λ3 ranks.
* Periodic box, var(GRF)/class 1 = 2.0 for B_000(0,0) and 1.2–1.6 elsewhere (B_202: 1.0–1.5). This is the omitted
  class-2 term, ≈ Δk/k with four pairings against two on i = j — as large as anticipated for Δk = 0.04 — plus a few
  per cent of lattice effects (the third side reaches beyond Nyquist on the k = 0.18 shell).
* Sphere, the leakage factor var_sphere/var_box × V_sphere/V_box (model independent) against the theory's
  C_sphere/C_box: theory 0.78–0.80 for every element (two legs each retaining ≈ 0.88 at R Δk = 20); GRF 0.76–0.92 on
  most elements, but 0.97, 1.07, 0.99 on B_000 (0,1), (0,3), (1,3) and 1.15, 1.32 on B_202 (0,2), (0,3). Class 2 has
  one leg coincidence instead of two and should retain ≈ √0.78 = 0.88; with the box-measured class-2 fraction f the
  prediction (0.78 + 0.88 f)/(1 + f) is 0.80–0.84, and GRF/prediction averages 1.085 ± 0.03 over the B_000 elements
  (element scatter 0.92–1.33 with ±0.09–0.12 errors, χ² ≈ 15 for 9 dof) and 1.15 ± 0.02 over B_202, with the
  B_202 elements that have the quadrupole leg in the lowest shell (k = 0.06, k R ≈ 30) and the other leg far away at
  1.47 and 1.69.
* Off-diagonal correlations: the theory's 5–7 % neighbour correlations (class 1 leakage) are present in the GRF
  (right panel), with the GRF showing additional 0.2–0.5 correlations between elements that share a shell — these are
  class-2 off-diagonal elements (the only Gaussian contribution off the diagonal), absent from class 1 by construction.

Reading. Class 1 under the window is right where it can be isolated (the (i, i) elements, where the class-1 share
is largest, agree to ≤ 6 %; the λ_w convergence and the exact algebraic tests exclude the window invariants and the
assembly as the source of the excess). The residual 8 ± 3 % (B_000) and the strongly element-dependent B_202 excess
are consistent with the class-2 term under the window differing from its box value: on the sphere the primed line of
sight of class 2 sits at ŷ2 instead of x̂ (the box ratio with a global line of sight cannot represent that for
B_202), and for the lowest shell k_w/k ≈ 0.1 so the local approximation is marginal. The test is therefore not yet
decisive at the 10 % level; it becomes so once class 2 is implemented (it reuses the same Q_n). Recommended rerun then:
thinner shells at higher k (Δk = 0.02, k ≥ 0.1) in a larger box (L = 2000, N = 192) to shrink both Δk/k and k_w/k,
≥ 2000 realisations, with OMP/BLAS threads set to 1 per worker (the pool here was oversubscribed: 1.9 s per
realisation effective on 8 cores against 0.3 s of FFT work).

![GRF test](bispectrum_dev/bispectrum_grf_sphere.png)

## 5. Costs (8 cores of this sandbox, numpy)

* Coupling numbers: `lam_w = 4`, `L ≤ 4`, `202 × 202`: 2046 radial labels, 33708 `(label, n)` terms, 2093 distinct invariants, 73 s (cached in memory; `000 × 000`: 926 labels, 1.6 s).
* Vertex matrices: 360k node pairs in 2.8 s per `(L3, a, b)`; ~120 labels for `L ≤ 4` → a few minutes per model.
* Radial assembly: per radial label one `(nb² nq × nr) @ (nr × nr)` product plus `nb` einsums; `nb = 9, nq = 63, nr = 375`: ~1 s per label → minutes per block at `lam_w = 4`.
* Pair sums: ~20 µs per anchor–neighbour pair (the self-pair products dominate: `n_groups × n_pairs × (2K+1)` complex multiplies), i.e. 2000 anchors × 5000 far neighbours ≈ 3–4 min plus the near pairs; the products over anchors are negligible. A jax/GPU backend for `_accumulate` (as `windows.py` has for the pair counts) would remove most of this.

## 6. What remains, in order

1. **Class 2** (`BispectrumGaussianCovariance.class2_block`): in the `(p1, p3)` variables it reuses `Q_n`; new pieces are the class-2 vertex (indicator × `Y_{ℓ'2}(p̂2) P(p2; x̂)` in bipolar harmonics of `(p̂1, p̂3)`), the leg-2 kernel integrating `j_Λ'(p3 r2)` over all `p3`, and the rank assignment (primed line of sight at `ŷ2`). It is `≈ dk/k` of class 1 on the diagonal — 20–40 % for `dk = 0.04`, 5 % for `dk = 0.005` at `k = 0.1` — and the only Gaussian term off the diagonal. The GRF test below measures it empirically.
2. **The k3 ≈ 0 term** (`k3_zero_block`): zero for uniform `ω`, up to tens of per cent of the `(i, i)` variance for a radially varying `ω`; needs the long-mode variance of the `W`-weighted field with the integral constraint (the SSC machinery has the ingredients).
3. **Non-Gaussian terms** `P B`, `B B`, `P T`: each reduces to the power-spectrum master formula with two-point windows of weights `(ω³, W)`, `(ω³, ω³)`, `(ω⁴, W)` and effective spectra `β`, `β^u β^p`, `τ` (guide sec. 'Non-Gaussian terms'); `WindowLibrary` can count the new weight pairs, the kernels need `B_{ℓ''}` on two-magnitude grids and `T` on general quadrilaterals.
4. **Pair-averaged (smoothed) windows** and `masked=True` models, as for the power spectrum.
5. **Performance**: jax backend for the pair sums; disk cache of the coupling numbers.
