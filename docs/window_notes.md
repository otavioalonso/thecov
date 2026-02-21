
# Survey window effects in the disconnected covariance

## Overview

`thecov` computes analytical covariance matrices for galaxy power spectrum multipoles in arbitrary survey geometries. This document explains how survey window effects are modeled in the disconnected (Gaussian) covariance. 

The methodology is based on [Wadekar & Scoccimarro 2020](https://arxiv.org/abs/1910.02914).

---

## 1. Power Spectrum Multipoles

The $k$-binned redshift-space galaxy power spectrum is estimated as:

$$
P_\ell(k) \equiv \frac{2\ell+1}{I_{22}N_{k}} \int_{k} \frac{d^3 q}{(2\pi)^3} \; F_\ell(q)F_0(-q)
$$

where the density contrast multipoles in Fourier space are defined as:

$$
F_\ell(q) = \int_x d^3x\; e^{-i q \cdot x} \mathcal{L}_\ell( q \cdot x) \bar n(x) w(x) \delta(x)
$$

Also:
- $\bar{n}(x)$ is the mean number density (obtained from randoms),
- $w(x)$ is the weight field (e.g., FKP weights),
- $\mathcal{L}_\ell$ is the Legendre polynomial of order $\ell$,
- $I_{22} = \int d^3x \; \bar{n}^2(x) w^2(x)$ is the normalization of the power spectrum,
- $N_k$ is the number of modes in the $k$-shell.

---

## 2. Gaussian Covariance in a Survey Geometry

### 2.1 The Covariance Expression

The Gaussian covariance of power spectrum multipoles is:

$$
\text{Cov}[P_{\ell_1}(k_1), P_{\ell_2}(k_2)] = \text{Cov}^{\text{CV}} + \text{Cov}^{\text{mixed}} + \text{Cov}^{\text{SN}}
$$

Each term involves the window function convolved with power spectra.

### 2.2 Window Function Multipoles

The key quantities are the window function multipoles:

$$
W^{ij}_{\ell m}(q) = \int d^3x \; e^{i q\cdot x} \; \bar{n}^i(x) w^j(x) Y_{\ell m}(x)
$$

where:
- $\bar{n}(x)$ is the mean number density (obtained from randoms),
- $w(x)$ is the weight field (e.g., FKP weights),
- $Y_{\ell m}$ is the real spherical harmonic,
- $(i,j)$ indices indicate powers of $\bar{n}$ and $w$.

**In the code:** `compute_mesh(nbar_power, weight_power, ell, m)` computes window function multipoles by painting the randoms on meshes weighted by spherical harmonics (evaluated at each random position) and applying FFTs. In pseudo-code, it uses:

```python
positions = randoms['POSITION']

weights = randoms['NZ']**(nbar_power - 1)
        * randoms['WEIGHT']**(weight_power)
        * alpha
        * Ylm(ell, m, randoms['POSITION'])

catalog -> to_mesh(compensate=True).r2c().value * nmesh**3
```

### 2.3 Cosmic Variance Term

The cosmic variance contribution involves products of two power spectra:

$$
\text{Cov}^{\text{CV}}_{\ell_1 \ell_2}(k_1, k_2) = \sum_{\ell_3 \ell_4}  \mathcal{W}^{\text{CV}}_{\ell_1 \ell_2 \ell_3 \ell_4}(k_1, k_2) \frac{P_{\ell_3}(k_1) }{(2\ell_3 + 1)} \frac{P_{\ell_4}(k_2)}{(2\ell_4 + 1)}
$$

where $\mathcal{W}^{\text{CV}}$ is the **window coupling matrix** (which we call `window_matrix`).

The window matrix is computed as terms of the form:

$$
\begin{align}
\mathcal{W}^{\text{CV}}_{\ell_1 \ell_2 \ell_3 \ell_4}(k_1, k_2) &= \frac{(4\pi)^2}{I_{22}^2} \int_{k_1,k_2} \sum_{m_{i=1}}^{m_4}Y_{\ell_1}^{m_1}(k_{1|2})Y_{\ell_2}^{m_2}(k_{1|2})Y_{\ell_3}^{m_3}(k_{1|2})Y_{\ell_4}^{m_4}(k_{1|2})\\ &\times \sum_{\ell_a \ell_b m_a m_b} \mathcal{G}^{(\text{CV}1|2)\,\ell_a\ell_b m_a m_b}_{\ell_1\ell_2\ell_3\ell_4 m_1 m_2 m_3 m_4}  W^{22}_{\ell_a m_a}(\Delta k)  \bar W^{22}_{\ell_b m_b}(\Delta k) 
\end{align}
$$

where $\mathcal{G^{(i)}}$ are coefficients built as products of Gaunt coefficients encoding the information of how Legendre polynomials in the estimator couple. They are computed by the functions `get_*_gaunt_coefficients()` in `geometry.py` using `sympy` and then cached into files in `thecov/cache`.

Indices shown as $1|2$ in the expression indicate that they depend on the specific term.

#### Mode integration

The integral is performed by distributing chunks of $k_1$ modes to multiple processes (using `multiprocessing`). Around each mode, a cube of dimensions $\Delta k$ (set by the FFT resolution) is constructed. The contraction with the Gaunt coefficients is pre-computed and stored in shared memory. During integration, the covariance per mode pair is calculated by performing the final contraction with the Fourier-space spherical harmonics in the expression (done using `numba`). A sparse binning matrix is used to add each contribution to its corresponding $k$-bin.

### 2.4 Mixed Term (Power × Shot Noise)

The mixed term involves one instance of the power spectrum:

$$
\text{Cov}^{\text{mixed}}_{\ell_1 \ell_2}(k_1, k_2) = (1+\alpha) \sum_{\ell_3} \mathcal{W}^{\text{mixed}}_{\ell_1 \ell_2 \ell_3}(k_1, k_2) \, \frac{P_{\ell_3}(k_2)}{(2\ell_3 + 1)} + (k_1 \leftrightarrow k_2),
$$

and the window matrix is computed as:

$$
\begin{align}
\mathcal{W}^{\text{mixed}}_{\ell_1 \ell_2 \ell_3 }(k_1, k_2) &= \frac{(4\pi)^2}{I_{22}^2} \int_{k_1,k_2} Y_{\ell_1}^{m_1}(k_{1})Y_{\ell_2}^{m_2}(k_{1})Y_{\ell_3}^{m_3}(k_{2})\\ &\times \sum_{\ell_a \ell_b m_a m_b} \mathcal{G}^{(\text{mix})\,\ell_a\ell_b m_a m_b}_{\ell_1\ell_2\ell_3 m_1 m_2 m_3}  W^{22}_{\ell_a m_a}(\Delta k)  \bar W^{12}_{\ell_b m_b}(\Delta k) 
\end{align}
$$

Notice that it depends on powers $W^{22}$ and $W^{12}$ of the window.

### 2.5 Shot Noise Term

$$
\text{Cov}^{\text{SN}}_{\ell_1 \ell_2}(k_1, k_2) = (1+\alpha)^2 \, \mathcal{W}^{\text{SN}}_{\ell_1 \ell_2}(k_1, k_2)
$$


$$
\begin{align}
\mathcal{W}^{\text{SN}}_{\ell_1 \ell_2 }(k_1, k_2) &= \frac{(4\pi)^2}{I_{22}^2} \int_{k_1,k_2} Y_{\ell_1}^{m_1}(k_{1})Y_{\ell_2}^{m_2}(k_{2})\\ &\times \sum_{\ell_a \ell_b m_a m_b} \mathcal{G}^{(\text{SN})\,\ell_a\ell_b m_a m_b}_{\ell_1\ell_2m_1 m_2}  W^{12}_{\ell_a m_a}(\Delta k)  \bar W^{12}_{\ell_b m_b}(\Delta k) 
\end{align}
$$

which only involves $W^{12} \times \bar W^{12}$.

---

## 3. Implementation Architecture

### 3.1 Main Classes

| Class | File | Purpose |
|-------|------|---------|
| `SurveyGeometry` | `geometry.py` | Handles survey mask, computes window matrices |
| `BoxGeometry` | `geometry.py` | Simple periodic box (no window effects) |
| `GaussianCovariance` | `covariance.py` | Computes Gaussian covariance using window matrices |


### 3.2 Computational Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    SurveyGeometry.__init__                      │
│  • Load random catalog                                          │
│  • Set up FFT mesh (boxsize, nmesh from kmax)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                compute_window_matrix()                          │
│                                                                 │
│  PHASE 1: K-Mode Sampling                                       │
│    • Sample k-modes in each bin for Monte Carlo integration     │
│    • Function: math.sample_kmodes()                             │
│                                                                 │
│  PHASE 2: Window Mesh Computation                               │
│    • Compute W^{ij}_{ℓm}(q) via FFT for all needed (i,j,ℓ,m)    │
│    • Function: compute_mesh()                                   │
│    • Parallelized with ThreadPoolExecutor                       │
│                                                                 │
│  PHASE 3: Gaunt Coefficient Loading                             │
│    • Load/compute products of Gaunt coefficients                │
│    • Functions: get_*_gaunt_coefficients()                      │
│    • Cached to disk in thecov/cache/                            │
│                                                                 │
│  PHASE 4: Gaunt Contraction                                     │
│    • Contract Gaunt coefficients with window products           │
│    • Result: window_product[term] for each term                 │
│                                                                 │
│  PHASE 5: Shared Memory Setup                                   │
│    • Move arrays to shared memory for multiprocessing           │
│                                                                 │
│  PHASE 6: Mode Integration (Main Computation)                   │
│    • For each k₁-mode, integrate over k₂ grid                   │
│    • Numba-optimized kernels for inner loops                    │
│    • Parallelized with multiprocessing.Pool                     │
│                                                                 │
│  PHASE 7: Normalization                                         │
│    • Apply (4π)²/I₂₂² normalization                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              GaussianCovariance.compute_covariance()            │
│                                                                 │
│  • Contract window matrices with input power spectra:           │
│    Cov = W^CV · P ⊗ P + W^mixed · P + W^SN                      │
│                                                                 │
│  • einsum operations for efficient tensor contractions          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Functions Reference

### 4.1 `SurveyGeometry.__init__(randoms, alpha, nmesh, kmax, ...)`

**Purpose:** Initialize the geometry from a random catalog.

**Parameters:**
- `randoms`: Catalog with `POSITION`, `WEIGHT`, `NZ` columns
- `alpha`: $N_{\text{gal}} / N_{\text{ran}}$ ratio
- `kmax`: Maximum wavenumber for window FFTs (sets Nyquist)
- `boxpad`: Padding factor for FFT box

**What it does:**
1. Sets up the FFT mesh covering the survey footprint
2. Stores randoms for later window computation

### 4.2 `SurveyGeometry.compute_mesh(nbar_power, weight_power, ell, m)`

**Purpose:** Compute a single window multipole $W^{ij}_{\ell m}(\mathbf{q})$.

**Returns:** 3D complex array of shape `(nmesh, nmesh, nmesh)`

**Implementation:**
```python
positions = randoms['POSITION']

weights = randoms['NZ']**(nbar_power - 1)
        * randoms['WEIGHT']**(weight_power)
        * alpha
        * Ylm(ell, m, randoms['POSITION'])

catalog -> to_mesh(compensate=True).r2c().value * nmesh**3
```

### 4.3 `SurveyGeometry.get_*_gaunt_coefficients(mask_ellmax, pk_ellmax)`

**Purpose:** Compute or load cached Gaunt coefficient tensors.

**Functions:**
- `get_first_cosmic_variance_gaunt_coefficients()` — For $\langle P P \rangle$ term  1
- `get_second_cosmic_variance_gaunt_coefficients()` — For $\langle P P \rangle$ term  2  
- `get_mixed_gaunt_coefficients()` — For $\langle P \cdot \text{SN} \rangle$ term
- `get_shotnoise_gaunt_coefficients()` — For $\langle \text{SN} \cdot \text{SN} \rangle$ term

**Returns:** `SparseNDArray` with shape:
- Cosmic variance: `(ℓ₁, ℓ₂, ℓ₃, ℓ₄, m₁, m₂, m₃, m₄) → (ℓₐ, ℓᵦ, mₐ, mᵦ)`
- Mixed: `(ℓ₁, ℓ₂, ℓ₃, m₁, m₂, m₃) → (ℓₐ, ℓᵦ, mₐ, mᵦ)`
- Shot noise: `(ℓ₁, ℓ₂, m₁, m₂) → (ℓₐ, ℓᵦ, mₐ, mᵦ)`

### 4.4 `SurveyGeometry.compute_window_matrix(pk_ellmax, mask_ellmax, kmodes_sampled, n_workers)`

**Purpose:** Main function that computes all window matrices.

**Parameters:**
- `pk_ellmax`: Maximum ℓ for output power spectra (default: 4)
- `mask_ellmax`: Maximum ℓ for mask expansion (default: 4)
- `kmodes_sampled`: Number of k-modes to sample per bin
- `n_workers`: Number of parallel workers

**Returns:** Dictionary with keys:
- `'cosmic_variance'`: Array of shape `(ℓ₁, ℓ₂, ℓ₃, ℓ₄, k₁, k₂)`
- `'mixed_term'`: Array of shape `(ℓ₁, ℓ₂, ℓ₃, k₁, k₂)`
- `'shotnoise'`: Array of shape `(ℓ₁, ℓ₂, k₁, k₂)`

### 4.5 Numba Kernels: `_compute_cosmic_variance_first/second()`, `_compute_mixed_term_contribution()`, `_compute_shotnoise_contribution()`

**Purpose:** JIT-compiled inner loops for the mode integration.

**Why Numba?**
- Avoids creating large temporary arrays
- Fuses multiple operations into single loops
- ~2-5× faster than NumPy vectorization for this workload

**Signature:**
```python
@numba.njit(fastmath=True, cache=True)
def _compute_cosmic_variance_first(
    bin_indices,      # (n_grid,) - k₂ bin for each grid point
    valid_mask,       # (n_grid,) - mask for valid k₂ values
    window_values,    # (n_nonzero, n_grid) - window products
    coeff_indices,    # (n_nonzero, 8) - (ℓ,m) indices
    Yk1_flat,         # (n_lm,) - Yₗₘ(k̂₁) for this k₁
    Yk2_3d,           # (ℓmax+1, 2ℓmax+1, n_grid) - Yₗₘ(k̂₂) for all k₂
    pk_ellmax,        # Maximum ℓ
    kbins             # Number of k-bins
) -> result          # (n_ell, n_ell, n_ell, n_ell, kbins)
```

### 4.6 `GaussianCovariance._compute_covariance_survey()`

**Purpose:** Contract window matrices with power spectra to get final covariance.

**Implementation:**
```python
# Normalize power spectra
pks = [4π/(2ℓ+1) * P_ℓ(k) for ℓ in (0,2,4)]

# Cosmic variance: sum over ℓ₃, ℓ₄
cosmic_variance = einsum('ijklxy, kx, ly -> ijxy', W_CV, pks, pks)

# Mixed term: sum over ℓ₃
mixed_term = (1+α) * einsum('ijkxy, kx -> ijxy', W_mixed, pks)

# Shot noise
shotnoise = (1+α)² * W_SN

# Total
Cov[ℓ₁,ℓ₂](k₁,k₂) = cosmic_variance + mixed_term + shotnoise
```

---

## 5. How to Modify for Your Use Case

### 5.1 Performance Tuning

- **`kmodes_sampled`**: More samples = better accuracy when integrating modes per bin, slower computation. Default 2000 is usually sufficient.
- **`n_workers`**: Set to number of physical cores. Too many workers can cause memory bandwidth issues.
- **`nmesh`**: can be determined by `kmax` of the mask. Larger `kmax` = finer mesh = more memory. High number is likely not necessary for the Gaussian covariance, which decays very quickly away from the diagonal.

### 5.2 Other Estimators

To adapt the code for a different estimator, you would probably need to:
1. Modify the window function definitions in `compute_mesh()`
2. Update the Gaunt coefficient contractions if the angular structure changes
---

## 6. Mathematical Details

### 6.1 Real Spherical Harmonics

The code uses real spherical harmonics:

$$
Y_{\ell m}(\hat{x}) = \begin{cases}
\sqrt{2} \, \text{Re}[Y_\ell^m(\hat{x})] & m > 0 \\
Y_\ell^0(\hat{x}) & m = 0 \\
\sqrt{2} \, \text{Im}[Y_\ell^{|m|}(\hat{x})] & m < 0
\end{cases}
$$

Only even $\ell$ and even $m$ are used (parity constraint).

### 6.2 Gaunt Coefficients

$$
G_{\ell_1 \ell_2 \ell_3}^{m_1 m_2 m_3} = \int d\Omega \, Y_{\ell_1 m_1}(\hat{n}) \, Y_{\ell_2 m_2}(\hat{n}) \, Y_{\ell_3 m_3}(\hat{n})
$$

Selection rules:
- $|ℓ_1 - ℓ_2| ≤ ℓ_3 ≤ ℓ_1 + ℓ_2$
- $m_1 + m_2 + m_3 = 0$
- $ℓ_1 + ℓ_2 + ℓ_3$ even

### 6.3 Monte Carlo Integration Over k-Modes

The k-integration is done by Monte Carlo sampling:

$$
\langle f(\mathbf{k}_1, \mathbf{k}_2) \rangle_{k_1, k_2} \approx \frac{1}{N_1} \sum_{i=1}^{N_1} \frac{1}{N_2(k_2)} \sum_{\mathbf{k}_2 \in \text{bin}} f(\mathbf{k}_1^{(i)}, \mathbf{k}_2)
$$

where $N_1$ is `kmodes_sampled` and $N_2(k_2)$ is the number of modes in each $k_2$ bin (from the full grid).

---

## 7. Output Format

The covariance object has these useful methods:

```python
# Full covariance matrix (all multipoles concatenated)
cov.cov  # shape: (n_ell * n_k, n_ell * n_k)

# Correlation matrix
cov.cor  # normalized to unit diagonal

# Block for specific multipole pair
cov.get_ell_cov(ell1, ell2)  # shape: (n_k, n_k)

# Eigenvalues (for checking positive-definiteness)
cov.eigvals

# Add covariances
total_cov = gaussian + trispectrum + ssc
```

---

## 8. References

1. **Wadekar & Scoccimarro (2020)** — [arXiv:1910.02914](https://arxiv.org/abs/1910.02914)  
   "Galaxy power spectrum multipoles covariance in perturbation theory"
   
2. **Feldman, Kaiser & Peacock (1994)** — [ApJ 426, 23](https://ui.adsabs.harvard.edu/abs/1994ApJ...426...23F)  
   "Power-spectrum analysis of three-dimensional redshift surveys" (FKP estimator)

3. **Yamamoto et al. (2006)** — [PASJ 58, 93](https://ui.adsabs.harvard.edu/abs/2006PASJ...58...93Y)  
   "A Measurement of the Quadrupole Power Spectrum..." (Yamamoto estimator)
