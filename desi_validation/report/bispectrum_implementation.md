# Bispectrum covariance in thecov: implementation of the Gaussian term (classes 1 and 2)

Branch `thecov2`, 2026-10-13/14. Derivation: `report/bispectrum_covariance_guide.typ` (and `bispectrum_covariance.typ`).
This note records what is implemented, the conventions the code actually uses, the validation, the costs and what remains.

## 1. Module map

| module | content |
|---|---|
| `thecov/bispectrum_algebra.py` | Sugiyama basis (`H`, `N`, `sugiyama_S_tilde`, `sugiyama_S`); `cg_tensor`; the bipolar vertex coefficients `bipolar_coefficients` (and `legendre_coefficients_opening`); the leg/anchor Gaunt tensors of both classes; `BispectrumCoupling` (coupling numbers `t_all` for class 1 and `t2_all` for class 2, batched over `(K1, K2, Lam3)`, cached; `index_set`); `invariant_basis` (Φ_n, for tests); `box_window_value`; `phase`, `phase2`. |
| `thecov/bispectrum.py` | `extrapolated_power`; `BispectrumShellKernels` (shell nodes, `jbar`, node-wise Bessel tables, vectorised vertex matrices `vertex_coefficients` with an optional shell restriction of the third side, the class-2 `p3_nodes` / `vertex2` / `bessel_at`); `Class1Assembler` (radial labels and the double radial integral for all `(i j; i' j')`: `assemble` for class 1, `assemble2` for class 2); `BispectrumGaussianCovariance` (user class; `class2=True`, `terms=` to separate the classes). |
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
* **Class 2** (pairing `(1 1')(2 3')(3 2')`, guide sec. 'Class 2', status item iv). Variables `(p1, p3 = p1 + p2)`: leg 1 carries `e^{i p1·r1}` with `P(p1; ŷ1)` (rank `L1`) exactly as in class 1; leg 2 carries `e^{−i p3·r2}` with `P(p3; ŷ2)` (rank `L3`), the primed line of sight `Y_{L'}(ŷ2)` now sits at `ŷ2` (rank at `ŷ2`: `Ly ∈ tri(L3, L')`); the anchor carries `P(p2; x̂)` (rank `L2`, with `conj(Y_L)(x̂)` of the unprimed estimator: `Λ3 ∈ tri(L, L2)`). The vertex is `1[|p3 − p1| ∈ j'] Y_{ℓ'2}(p̂2) conj(Y_{L2})(p̂2) P_{L2}(p2)`, merged to `Y_{ℓ''}(p̂2)` (`ℓ'' ∈ tri(ℓ'2, L2)`, `vertex2_tensor`) and expanded in bipolar harmonics of `(p̂1, p̂3)`: `d^{ℓ''}_{ab}(p1, p3) = (−1)^a c^{ℓ''}_{ab}(p1, p3)[h]` with `h(p2) = 1[p2 ∈ j'] P_{L2}(p2)` (the sign from `p̂1 → −p̂1`; the indicator restricts the opening-angle quadrature to the intersection with the shell). Ranks on `p̂3`: `Λ' ∈ tri(L3, b)` (`leg3_tensor`); phase `(−i)^{ℓ1+ℓ2} i^Λ (−i)^{Λ'}`. The radial kernel integrates `p3` over `[max(klo − p1, p1 − khi, 0), p1 + khi]` with composite Gauss–Legendre pieces split at the indicator kinks `|klo ∓ p1|, khi ∓ p1` **and at every shell edge** (the `r2` integral selects the unprimed shell `j` of `p3` with a kernel that is a step on the scale of the window — a piece must not straddle it; forgetting this gave 20–30 % errors on specific elements), `j_Λ'(p3 r2)` from a fine table (`bessel_at`). The other three class-2 pairings are relabellings: primed swap `ℓ' → ℓ'_sw` with the primed shells transposed, unprimed swap `ℓ → ℓ_sw` with the unprimed shells transposed, and both. Same `Q_n` library (the `n` labels differ through `Ly`).
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
6. **Class 2** (`test_class2_*`): the five-direction identity with the class-2 chain (`direct_V2` / `series_F2`) is exact (1e-8) for isotropic and anisotropic `P_L` at the legs and the anchor and the `(000, 202)` pairs; `assemble2` with box windows at `r_max = 3000` agrees with a Monte-Carlo evaluation of the class-2 box term (exact indicator, `box_class2_direct`) to 0.5 % (mean) / 1.5 % (scatter, the `1/r_max` truncation).
7. **Gaussian fields** (`grf_sphere.py`, `grf_theory.py` / `grf_theory2.py`, `grf_compare.py` / `grf_compare2.py`): see section 4.

Fast tests (`-m "not slow"`): 32 tests in ~5 min. Slow ones: anisotropic five-direction checks (both classes), the converged sphere, the anisotropic tapered-box class check, the 202 class-2 box check, the GRF smoke test.

## 4. Gaussian-field test

Setup (`grf_sphere.py`): isotropic P(k) = 2e4 (k/0.02)/(1 + (k/0.02)^2.6) on a mesh in a periodic box, observer at the
box centre, footprint a sphere centred on the observer (`sphere_check` tracer, n̄ = 3.8e-5, α = 0.1, 200k randoms),
F = n̄ 1_sphere (δ − mean_sphere δ) (the α step: ∫F = 0, which with uniform ω kills the k3 ≈ 0 term), shells 0.04–0.20
in steps of 0.04, estimators B_000(i ≤ j) and B_202(i, j) (Sugiyama FFT estimator, local line of sight), and the same
estimator on the unmasked periodic box; a quadratic field δ + 0.3(δ² − σ²) with mean B_000 = 2ε[PP + 2 perms] fixes
the estimator normalisation (1.00 ± 0.01 on the k ≥ 0.1 elements). Two configurations: R = 500 in L = 1300, N = 128
(cell 10.2, k_Nyq = 0.31; 800 realisations) and R = 1000 in L = 2600, N = 192 (cell 13.5, k_Nyq = 0.23; 300
realisations). Theory: `grf_theory.py` (class 1, λ_w = 0, 2, 4), `grf_theory2.py` (class 1 + 2, λ_w = 2 and 4, pair
sums with 3000 anchors / 6000 far neighbours), the analytic box terms; comparison `grf_compare.py`, `grf_compare2.py`;
figures `bispectrum_grf_sphere.png` (class 1 only), `bispectrum_grf_c12.png` (class 1 + 2, both spheres).

**Periodic box (the arbiter for the formalism without a window).** var(GRF)/(class 1 + class 2) = 1.004 ± 0.016 (B_000)
and 1.017 ± 0.013 (B_202) on the 128³ mesh, element scatter consistent with the 5 % errors; the off-diagonal elements
(pure class 2) agree at 1.01 ± 0.04. On the 192³ mesh (k_Nyq = 0.23, so the third side of the k = 0.18 shell reaches
beyond Nyquist) 1.036 ± 0.016, dominated by the (3,3) element. Class 1 alone is low by the class-2 share: 1.2–2.0 at
Δk = 0.04 (C2/C1 = 0.84 on B_000(0,0), 0.2–0.7 elsewhere).

**Sphere, class 1 only** (`bispectrum_grf_sphere.png`): the rank expansion is converged (λ_w = 0/2/4 within 3 %); the
GRF/theory excess is the omitted class 2 plus a residual.

**Sphere, class 1 + 2** (`bispectrum_grf_c12.png`):

| | B_000 var/(C1+C2) | χ²/n | B_202 | χ²/n | off-diagonal corr (GRF vs theory, |corr| > 0.1) |
|---|---|---|---|---|---|
| R = 500 (800 real.) | 1.079 ± 0.017 (0.96–1.22) | 32/10 | 1.182 ± 0.015 (1.01–1.77) | 199/16 | 0.221 vs 0.205, rms diff 0.056 |
| R = 1000 (300 real.) | 1.103 ± 0.029 (0.92–1.24) | 22/10 | 1.129 ± 0.023 (0.97–1.40) | 46/16 | 0.211 vs 0.187, rms diff 0.062 |

* Class 2 reproduces the off-diagonal (shell-sharing) correlations, 0.2–0.5, which class 1 does not have at all.
* The B_202 excess is concentrated on the elements with the quadrupole leg in the lowest shell (k = 0.06) and the
  other leg far away — (0,1), (0,2), (0,3): 1.35, 1.55, 1.77 at R = 500 — and shrinks to 1.16, 1.15, 1.40 at
  R = 1000: the behaviour of the local line-of-sight approximation (A2: nearby points share the line of sight), whose
  error for the quadrupole weight at separation r and distance x is O((r/x)²) — the configurations of the bispectrum
  reach r ~ 1/Δk = 25–100 Mpc/h on a footprint where x ~ R.
* **Power-spectrum control** (`grf_pk_check.py`, 2000 realisations): on the same sphere, the same fields and the same
  randoms, var(P̂₀)/`GaussianCovariance` = 1.016 ± 0.016 per shell (1.04, 1.01, 1.02, 1.00) with the neighbour
  correlations matching (0.05 vs 0.05). The local approximation A1 is therefore fine for the two-point function at
  RΔk = 20; the bispectrum offset is specific to the three-point estimator.
* **Thin-shell run** (`grf_sphere.py 400 2000 192 800 _thin` with `BISPEC_K_EDGES=0.10,…,0.20`, i.e. Δk = 0.02,
  k = 0.10–0.20, R = 800 in a 2000 Mpc/h box, 192³; theory `grf_theory2.py 2 800 _thin`, box theory
  `grf_box_theory.py`; figure `bispectrum_grf_thin.png`): box B000 0.996 ± 0.018, B202 1.032 ± 0.014 (the formalism
  is exact at Δk = 0.02 too); sphere B000 1.069 ± 0.020 (χ² 33/15), B202 1.063 ± 0.015 (χ² 41/25), off-diagonal
  correlations 0.153 vs 0.149. Dividing out the box run on the same seeds, (var_sph/var_box)/(theory_sph/theory_box)
  = 1.052 ± 0.024 (B000) and 1.014 ± 0.018 (B202); the same paired ratio for R = 500, Δk = 0.04 is 1.058 ± 0.022 and
  1.120 ± 0.018. So: the B202 excess is gone once the low-k elements (k < 0.1) are out of the range and R is large
  (A2 confirmed), while the B000 offset is +5–8 % at R = 500 / 800 / 1000, Δk = 0.04 / 0.02 and k_w/Δk = 0.025–0.06
  alike — it does not scale with k_w/Δk, which rules out the k3 ≈ 0 long-mode term as its explanation. Per-element
  scatter beyond the nominal errors (χ²/dof ≈ 2) is present in the **box** ratios as well, where only lattice
  triangle-counting (beyond the mode-count correction) can produce it; the per-element errors should therefore be
  inflated by ~1.4 and the B000 offset read as a ~2σ effect per run, ~3.5σ combined.
* The B_000 offset, +8 ± 2 % (R = 500) and +10 ± 3 % (R = 1000), does **not** shrink with R and has element scatter
  above the noise (the B̂ samples are Gaussian to within kurtosis ±0.3, so the quoted errors are right). It is not the
  numerics: λ_w = 2 → 4 changes the theory by < 1.5 %, dr = 4 → 2 and n3 = 10 → 16 by < 0.2 %, and the pair-sum noise
  (two more anchor/neighbour seeds, up to 6000/12000) is a uniform 1–2 %. It is not the lattice (the box on the same
  mesh is exact). Candidates that remain, in order: the local approximation A1 at the level of the *three-point*
  structure (the window varies over the triangle; the pair-averaged windows of `smoothing.py` were needed for the same
  reason for P(k) in DESI, where the dilution I_k/norm was 0.80–0.85), and the pairings dropped as "beat modes". The
  next discriminating test is a GRF run with thinner shells at higher k (Δk = 0.02, k ≥ 0.1, L = 2600, N = 256), where
  both Δk/k and k_w/k are small and the class-2 share is ≈ 10 %.

![GRF test, class 1 + 2](bispectrum_dev/bispectrum_grf_c12.png)

## 5. Costs (8 cores of this sandbox, numpy)

* Coupling numbers: `lam_w = 4`, `L ≤ 4`, `202 × 202`: 2046 radial labels, 33708 `(label, n)` terms, 2093 distinct invariants, 73 s (cached in memory; `000 × 000`: 926 labels, 1.6 s).
* Vertex matrices: 360k node pairs in 2.8 s per `(L3, a, b)`; ~120 labels for `L ≤ 4` → a few minutes per model.
* Radial assembly: per radial label one `(nb² nq × nr) @ (nr × nr)` product plus `nb` einsums; `nb = 9, nq = 63, nr = 375`: ~1 s per label → minutes per block at `lam_w = 4`.
* Class 2: `assemble2` costs per radial label `nb²` × (one `(nq × n3) × nr` einsum); the sphere test (nb = 4, nr = 250, isotropic P) took 22 s at λ_w = 2 and 450 s at λ_w = 4 for the two multipoles; the box check with anisotropic P (202 × 202, nr = 750) 140 s. The class-2 `n` labels (with `Ly`) add ~3× to the window invariants (69 vs 24 at λ_w = 4 in the isotropic sphere test).
* Pair sums: ~20 µs per anchor–neighbour pair (the self-pair products dominate: `n_groups × n_pairs × (2K+1)` complex multiplies), i.e. 2000 anchors × 5000 far neighbours ≈ 3–4 min plus the near pairs; the products over anchors are negligible. A jax/GPU backend for `_accumulate` (as `windows.py` has for the pair counts) would remove most of this.

## 6. What remains, in order

1. **Understand the sphere residual** (section 4): thinner-shell GRF run; test the A1/A2 approximations by a GRF with
   a uniform-weight footprint far from the observer (global line of sight) to separate the two.
2. **The k3 ≈ 0 term** (`k3_zero_block`), pairing (1 2)(1' 2')(3 3'): zero for uniform `ω` only for the exact zero
   mode; the mean subtraction removes k3 = 0 but leaves the long modes |k3| ≲ k_w = 1/R, which the third-leg window
   W(k3) still admits. Design: (a) the (1 2)(1' 2') part is a two-point object on legs 1, 2 — the product of the
   two P-weighted leg kernels already in `BispectrumShellKernels.leg`, bin-averaged over shells (i, j) and (i', j')
   with the Y_{ℓ1}Y_{ℓ2} Sugiyama weights collapsing to the (ℓ1 = ℓ2, L = 0) or local-LOS equivalent; (b) the
   (3 3') part is the long-mode variance σ²_L = ∫ d³q/(2π)³ |W̃(q)|² P(q) of the W-weighted field with the integral
   constraint subtracted (|W̃(q) − W̃(0) ρ̃(q)/ρ̃(0)|² with ρ the mean-subtraction weight), i.e. the quantity the SSC
   module already computes for `W`; (c) the term is then an outer product of (a) over (i j) × (i' j') scaled by (b),
   diagonal in neither shell index, which is the signature to look for in the GRF: a floor common to all elements of
   the same multipole. Order of magnitude on the R = 500 sphere: σ²_L ∼ P(k_w)/V ∼ 10⁻² of the shell variance per
   unit k_w/Δk, so a few per cent at Δk = 0.04 and more for thinner shells — the thin-shell run (Δk = 0.02) tests
   exactly this scaling. For a radially varying `ω` the k3 = 0 mode itself survives and the term can reach tens of
   per cent of the (i, i) variance.
3. **Non-Gaussian terms** `P B`, `B B`, `P T`: each reduces to the power-spectrum master formula with two-point windows of weights `(ω³, W)`, `(ω³, ω³)`, `(ω⁴, W)` and effective spectra `β`, `β^u β^p`, `τ` (guide sec. 'Non-Gaussian terms'); `WindowLibrary` can count the new weight pairs, the kernels need `B_{ℓ''}` on two-magnitude grids and `T` on general quadrilaterals.
4. **Pair-averaged (smoothed) windows** and `masked=True` models, as for the power spectrum.
5. **Performance**: jax backend for the pair sums; disk cache of the coupling numbers.
