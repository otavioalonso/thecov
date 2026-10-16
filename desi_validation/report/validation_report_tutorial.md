# Validating an analytic covariance for DESI power-spectrum multipoles, term by term

*A report for readers new to the problem. State of the work on 2026-10-15; branch `thecov2` of `thecov`,
directory `desi_validation/`. The specialist write-ups it summarises are `report/kernel_validation_results.md`,
`report/grf_randoms_results.md` (sections 1-10) and `HANDOFF.md`.*

## 1. The problem in one page

A galaxy survey like DESI measures the power spectrum multipoles $P_\ell(k)$, $\ell = 0, 2, 4$, of a tracer
(here LRGs in $0.4 < z < 0.6$, "LRG1", and QSOs as a control) in bins of width $\Delta k = 0.005\,h/{\rm Mpc}$
between $k = 0.02$ and $0.3$. To turn those 56 x 3 numbers into cosmological constraints one needs their
covariance matrix: how much each bin scatters from one realisation of the Universe to the next, and how the bins
scatter together.

There are two ways to get it. **Mocks**: run many simulated surveys (here the 859 holi-v3 "altmtl" mocks, 848 after
removing 11-12 with a defective $n(z)$) and take the sample covariance. **Analytic**: compute the covariance from
the survey geometry and a model of the clustering. `thecov` is the analytic code. The question of this work is
simple to state and hard to answer: *where and why does `thecov` disagree with the mocks, term by term?*

Why "term by term" matters. The covariance is a sum of physically distinct pieces, and a mismatch can hide in any
of them. Fitting one overall amplitude to the mocks tells you nothing about which piece is wrong, and several of the
pieces have nearly the same shape in $(k, k')$, so even fitting separate amplitudes gives one solution of a degenerate
system rather than a measurement. The approach taken here is therefore to build **test catalogues with known
statistics on the real DESI footprint**, so that one term at a time has a known answer, and to **measure the
inputs of the other terms from a single catalogue** (mock or data) instead of fitting them.

### 1.1 The four terms

Write the FKP field as $F(\mathbf k) = \sum_g w_g e^{i\mathbf k\cdot\mathbf x_g} - \alpha \sum_r w_r e^{i\mathbf k\cdot\mathbf x_r}$
(galaxies minus $\alpha$ times randoms), and the estimator $\hat P_\ell(k) = [\langle F_0^* F_\ell\rangle_{\rm shell} - S\,\delta_{\ell 0}] / {\rm norm}$,
with $S$ the shot noise and ${\rm norm} = \alpha \sum_r w_r^2 n(z_r)$ the normalisation (proportional to $\int n^2 w^2 \,d^3x$).

1. **Gaussian term (G).** If the density field were Gaussian, the covariance would be $2P^2/N_{\rm modes}$ in a
   periodic box. In a survey the window mixes modes, and the Gaussian covariance becomes a convolution of
   $P(k)P(k')$ with a *window function*. Everything about the survey geometry — footprint, $n(z)$, weights, holes —
   enters here. It is the largest term everywhere and the only one at low $k$.
2. **Super-sample covariance (SSC).** Density modes longer than the survey are not measured, but they modulate the
   measured power: in an overdense survey the small-scale power is higher (the *response* $R = d\ln P/d\delta_L$).
   Because the estimator normalises to the observed mean density, part of this effect cancels — the *local average*
   term. The net effect is a rank-one-like addition $\sigma_L^2 R_\ell(k) R_{\ell'}(k') P_\ell P_{\ell'}$.
3. **Discreteness (shot-noise non-Gaussian) terms.** Galaxies are points. Beyond the Gaussian shot noise $S$, the
   discreteness couples to clustering: a term $\propto B/\bar n$ (one galaxy shared between the two power estimates,
   with the bispectrum $B$) and a term $\propto P/\bar n^2$ (two galaxies shared). These are the Wadekar &
   Scoccimarro (2020) "SN-NG" terms, implemented in `thecov/discreteness.py`.
4. **Connected trispectrum (T0).** The non-Gaussianity of the continuous field itself, at tree level in perturbation
   theory (`thecov/trispectrum.py`).

The last three are all smooth, broadband surfaces in $(k, k')$ that grow toward high $k$; that is the degeneracy
mentioned above.

### 1.2 Units and conventions used throughout

- Bands: 0.02-0.05, 0.05-0.1, 0.1-0.2, 0.2-0.3 $h$/Mpc. Most comparisons quote the last three.
- "Var$_{\rm GRF}$/Var$_{\rm thecov}$" = variance measured from test fields divided by `thecov`'s prediction; 1 is perfect.
- "$\chi^2/n$" = the mocks' sample covariance tested against the analytic one (a likelihood-ratio statistic normalised
  to 1 for a perfect model, $k = 0.02$-$0.2$).
- Responses are quoted **per unit galaxy long mode**: $R = d\ln P / d\delta_{g,L}$ where $\delta_{g,L}$ is the
  galaxy (not matter) overdensity on large scales. The tree-level values for LRG1 ($b_1 = 2.2$, $f = 0.76$,
  $g_0 = b_1 + f/3 = 2.45$) are $R_0 = 1.2$-$1.6$ and $R_2 = 1.8$-$2.7$ in these units.
- "Mock 173" is the reference catalogue used whenever one catalogue suffices.

## 2. Part A — the Gaussian term and the window

### 2.1 What was at stake

`thecov` builds its Gaussian term from a window. The default DESI option used the *local* window $\int m^2$ (the
squared mean density of the randoms on a mesh). A newer option, `thecov.WindowSmoothing`, uses a *pair-averaged
$\xi$-kernel* window: the mean density is first smoothed with a kernel $K_k(r) = \xi_0(r)\, j_0(kr)/P_0(k)$ that
depends on $k$ — the idea being that what matters for the variance at scale $k$ is the density averaged over the
separations that contribute to $P(k)$, not the density at a point. The two differ by 7-9 % for LRG1 (the "$\int m^2/{\rm norm}$"
of 1.225 for NGC against a kernel integral $I_k$ that runs from 0.943 at low $k$ to 1.036 at high $k$) and by only
~1 % for QSO ($\int m^2/{\rm norm} = 1.11$).

The DESI footprint is full of small holes (bright-star masks of 0.05-0.1 deg, fibre-collision gaps, single-pixel
vetoes). A mesh cannot resolve them, and the question "is the kernel window right, or are the holes fooling both
windows?" could not be settled with mocks alone, because the mocks contain all four terms at once.

![LRG1 NGC: fraction of the sky filled, at 1 arcmin resolution and after smoothing with the k-dependent kernel](holes_LRG1_compare.png)

*Figure 1. A 1.5 deg patch of the LRG1 NGC footprint built from 18 random files. Left: raw fill at 1 arcmin — the
bright-star masks and many single-pixel holes. Right: the same fill smoothed with the $\xi$-kernel of a given $k$;
the median is 0.937 (a 6 % dilution of the density), with dips to 0.85 ($k = 0.163$) and 0.75 ($k = 0.298$) on the
largest masks.*

### 2.2 The decisive test: Gaussian fields on the DESI randoms

Instead of a mesh, take the actual random catalogue (which has every hole in it), evaluate a band-limited Gaussian
field with a known $P(k)$ *at the positions of the randoms* with a non-uniform FFT (type 2), use that as the
"galaxy" weight, and measure $\hat P_\ell$ exactly as the pipeline does (type-1 NUFFT, shot-noise self-pair term
subtracted). Repeat for a few hundred fields. Only the Gaussian term exists in such data, and its covariance is known
once the window is known, so this isolates the window question completely (`grf_randoms.py`).

![Var_GRF / Var_thecov for LRG1, both caps, kernel vs local window](grf_randoms_LRG1.png)

*Figure 2. Variance of $\hat P_0$ from 250 (NGC) / 242 (SGC) Gaussian fields on the LRG1 randoms, divided by
`thecov`'s Gaussian prediction with the true input power, for the $\xi$-kernel window and for the local $m^2$ window.*

Results (Var$_{\rm GRF}$/Var$_{\rm thecov}$, bands 0.02-0.05 / 0.05-0.1 / 0.1-0.2 / 0.2-0.3):

| tracer, cap | fields | $\xi$-kernel window | local window $m^2$ |
|---|---|---|---|
| LRG1 NGC | 250 | 0.972 0.998 0.993 1.009 | 0.651 0.705 0.731 0.765 |
| LRG1 SGC | 242 | 0.986 1.012 0.975 0.988 | 0.686 0.746 0.750 0.784 |
| LRG2 NGC | 146 | 0.973 1.014 1.040 0.975 | 0.662 0.727 0.779 0.754 |
| ELG2 NGC | 106 | 0.959 0.999 0.978 0.911 | 0.800 0.861 0.880 0.848 |
| QSO NGC | 64 | 0.967 0.983 1.065 0.977 | 0.806 0.845 0.966 0.924 |

The kernel window is right to a few per cent (the per-band error is 5-8 % with 100-250 fields). The local window
fails by 20-35 % when given the *true* power; when it is given the fields' mean measured power instead (as one does
with mocks, where the input power is not known) the failure shrinks to a 7-9 % deficit (1.077 for LRG1) — the
Gaussian-term error that the kernel was designed to remove. Variants with shot noise at the files' level and all three
multipoles (124 fields), and with linear redshift-space distortions ($f = 0.75$, 95 fields), agree within the same
5-8 % for $\ell = 0, 2, 4$: no sign that an isotropic kernel fails for the quadrupole and hexadecapole at this precision.

A by-product worth knowing about: a first Poisson-sampling variant with *fixed* random halves came out 2-18 % below
Gaussian. The reason is instructive — a fixed point set has a fixed shot-noise field, which acts as a constant offset
$c(\mathbf k)$ in $F$, and ${\rm Var}|F + c|^2 = (\sigma^2 + |c|^2)^2 - |c|^4$ is smaller than for a fluctuating one.
The prediction $1 - (0.51\,S/(P+S))^2 = 0.973, 0.942, 0.877, 0.818$ matched the measured 0.985, 0.910, 0.866, 0.818
with no free parameter; redrawing the halves every realisation removed the effect (1.060 1.002 1.014 1.012).

### 2.3 Against the mocks

With the converged kernel basis (19 -> 8 windows, $I_k$ error $10^{-5}$) the LRG1 $\chi^2/n$ is 1.049 / 1.061 / 1.055
(NGC / SGC / combined, $k = 0.02$-$0.2$), against 1.090-1.095 for the default window and 1.014-1.023 for a hand-tuned
"fill" window (density times a healpix fill fraction); QSO is 1.003 / 0.993 / 0.996 with any window. The residual of
the kernel against the mocks is in the bin-width-independent part of the variance, i.e. it behaves like a
Gaussian-term error of 1-2 % above $k = 0.06$ rather than non-Gaussianity; pair-count sampling of the kernel accounts for
part of it (converging the basis at matched sampling raised $\chi^2/n$ by 0.02). The kernel has no tuned scale, the fill
window has one; the kernel has not yet been made the DESI default on the $\chi^2/n$ criterion alone, but on the
random-field test it is the correct window.

**Conclusion of Part A.** The Gaussian term of `thecov` is right for the window it is given, and the $\xi$-kernel is
the right window, holes included. The window question is closed; whatever still disagrees with the mocks is in the
other three terms.

## 3. Part B — the discreteness terms, measured from one catalogue

### 3.1 The idea: thinning

Randomly keep each galaxy with probability $p$ (and reweight by $1/p$). Everything that depends on the *continuous*
field — SSC, T0, the clustering part of G — is unchanged, because a random subsample of a point process has the same
correlation functions as the parent. Only the discreteness changes, and in a known way: Gaussian shot noise $S \propto 1/p$
(predicted by `thecov`), $B/\bar n \propto 1/p$, $P/\bar n^2 \propto 1/p^2$. So the *extra* covariance created by
thinning, measured on one catalogue and averaged over thinnings, minus `thecov`'s Gaussian change, is a quadratic
polynomial in $a = (1-p)/p$ whose two coefficients are the two discreteness terms at full density. No mocks, no
template, no shape prior.

### 3.2 From Monte Carlo to a closed form

Averaging over 200-400 thinnings (`thinning_cov.py`) works but costs 400 power spectra per catalogue. The average
over thinnings can be done analytically (`thinning_closed_form.py`):

$$ {\rm Cov}_{\rm thin}[P(k), P(k')] = a \sum_g G_g(k) G_g(k') + a^2 \left[ |W_2(\mathbf k - \mathbf k')|^2 + |W_2(\mathbf k + \mathbf k')|^2 - 2 \sum_g w_g^4 \right], $$

with $G_g(k) = 2\,{\rm Re}[F^*(k)\, w_g e^{i\mathbf k\cdot\mathbf x_g}] - 2w_g^2$ (how much galaxy $g$ contributes to
the power at $k$) and $W_2$ the transform of the $w^2$-weighted catalogue. It agrees with the Monte Carlo within
1.5 $\sigma$ in every band and takes 60 s per catalogue.

### 3.3 Results

Galaxies' own discreteness terms at full density, in units of the Gaussian diagonal, from mock 173 (fit over
$p = 0.5, 0.25, 0.125$; rms residual of the quadratic fit 0.16 % of G for LRG1, 1.5 % for QSO):

| | $B/\bar n$ diagonal ($k$ 0.2-0.3) | $B/\bar n$ off-diagonal ($k > 0.15$) | $P/\bar n^2$ diagonal | $P/\bar n^2$ off-diagonal | tree level |
|---|---|---|---|---|---|
| LRG1 NGC | 0.138 | 0.163-0.170 | 0.024 | 0.022 | $B/\bar n$ 0.07-0.08, $P/\bar n^2$ 0.015 |
| QSO NGC | 0.031 (noisy) | 0.111-0.114 | 0.044 | 0.039 | same |

Reading: for LRGs the $B/\bar n$ term is **twice** its tree-level value and $P/\bar n^2$ about 1.5x; for QSOs
$P/\bar n^2$ is **2.6x** tree level and $B/\bar n$ about 1.5x. These are exactly the amplitudes the earlier Wishart
template fits to 848 mocks had asked for (HANDOFF 9.4: $B/\bar n$ x 1.8-2.3 for LRG1, $P/\bar n^2$ x 2.2-4.1 for
QSO), now obtained from one catalogue with no template. The physics is also legible: a factor 2 in $B/\bar n$ for
LRGs is the one-halo (satellite) bispectrum that tree-level perturbation theory does not contain; a factor 2.6 in
$P/\bar n^2$ for QSOs says the QSO point process is not Poisson at the pair level (an excess of close pairs or
duplicates — a question for the mock builders, testable with $p \approx 0.9$). Both measurements apply unchanged to
the DESI data catalogues (about a minute per cap).

**Conclusion of Part B.** The discreteness terms are *larger* than `thecov`'s tree-level implementation by
tracer-dependent factors that can be measured from the catalogue itself. This is the "calibrate from the data" step
for two of the three high-$k$ terms, and it breaks the shape degeneracy without using any mocks.

## 4. Part C — the long-mode response, measured from one catalogue

### 4.1 What a response is and why it is hard to measure

SSC needs $R_\ell(k) = d\ln P_\ell / d\delta_L$: by how much the small-scale power changes when the region sits in a
large-scale overdensity $\delta_L$. `thecov` uses the tree-level value. Measuring it directly in a survey sounds
impossible — the survey is one region, with one $\delta_L$ — but there is a classical trick (Chiang et al. 2014,
"position-dependent power spectrum"): long modes *inside* the survey vary from place to place, and the correlation of
the local power with the local long mode is a squeezed bispectrum, $\int B(\mathbf k, -\mathbf k - \mathbf q, \mathbf q)\,W_L(q)$,
which equals $R(k)\,P(k)\,\langle\delta_g\delta_L\rangle$ for $q \ll k$.

The estimator used here (`response_ibisp.py`) does this in Fourier space without sub-volumes: low-pass the FKP field
with a Gaussian filter of scale $k_L$ to get $\delta_L(\mathbf x)$ inside the survey, form the cross-spectrum
$C_\ell(k) = \langle F_\ell^*(k)\, \widetilde{[\delta_L F]}(k)\rangle$, subtract the self-pair term
$\sum_g w_g^2 \delta_L(\mathbf x_g)$ (the discreteness of a three-point estimator), and divide by $P_\ell(k)$ times
$\langle\delta_g\delta_L\rangle = \int P W_L$ — the mean of $\delta_L$ at the galaxies, *not* its variance
$\int P W_L^2$, which is 2.2 times smaller for a Gaussian filter. Getting that normalisation right was the single most
important detail.

### 4.2 Validation with a known answer

A lognormal field has an exactly known response: its local power is $(1 + \delta_L)^2 P_s$, so $R = 2$ per unit
galaxy long mode for any filter. Poisson-sampling a lognormal field on one half of the DESI randoms and using the other
half as randoms (so that galaxies and randoms are distinct point sets, as in the data) gives, on the LRG1 NGC footprint,
$R_0 = 1.86 \pm 0.06 / 1.98 \pm 0.05 / 1.91 \pm 0.06$ in the three usable bands and $R_2 \approx 0$. Earlier attempts
that painted a continuous field on the random points themselves (galaxies = randoms) had failed badly — a reminder
that the bookkeeping of self pairs must match the data's. The lowest band (0.02-0.05) is not in the squeezed limit for
$k_L = 0.01$ and is never used.

### 4.3 The mocks' response and its scale dependence

![Response of P0 and P2 versus the filter scale k_L, LRG1 NGC, six mocks](response_kL_LRG1_NGC.png)

*Figure 3. (a) Monopole and (b) quadrupole responses per unit galaxy long mode from six holi mocks, as a function of
the filter scale $k_L$; grey band = tree level, dotted line = 2. (c) The net survey-mean response derived from (a),
compared with the regression over 848 mocks (section 5).*

| $k_L$ | $\sigma_L$ | $R_0$ 0.05-0.1 | $R_0$ 0.1-0.2 | $R_0$ 0.2-0.3 | $R_2$ 0.05-0.1 | $R_2$ 0.1-0.2 | $R_2$ 0.2-0.3 |
|---|---|---|---|---|---|---|---|
| 0.030 | 18.8 % | 1.895 ± 0.017 | 1.899 ± 0.013 | 1.815 ± 0.008 | 2.42 ± 0.09 | 2.78 ± 0.06 | 2.77 ± 0.08 |
| 0.020 | 11.6 % | 1.910 ± 0.023 | 1.888 ± 0.017 | 1.803 ± 0.010 | 2.69 ± 0.12 | 2.85 ± 0.08 | 2.80 ± 0.10 |
| 0.015 | 8.1 % | 1.937 ± 0.034 | 1.898 ± 0.024 | 1.819 ± 0.018 | 2.89 ± 0.15 | 2.92 ± 0.10 | 2.89 ± 0.12 |
| 0.010 | 4.8 % | 2.011 ± 0.072 | 1.952 ± 0.058 | 1.900 ± 0.059 | 3.30 ± 0.22 | 3.15 ± 0.13 | 3.33 ± 0.26 |
| 0.007 | 3.0 % | 2.245 ± 0.185 | 2.161 ± 0.175 | 2.184 ± 0.214 | 4.33 ± 0.70 | 3.90 ± 0.40 | 4.84 ± 1.05 |
| 0.005 | 1.9 % | unusable (normaliser $\to 0$) | | | | | |

Three things to take from this table.

- **The monopole response is 1.9, flat over a factor 2 in $k_L$** (0.015-0.03), i.e. 1.2-1.5 times tree level. This
  is the same physics as the factor 2 in $B/\bar n$: the mock LRGs respond to a long mode more strongly than linear
  bias plus tree-level perturbation theory say, because of the one-halo term.
- **The quadrupole response is not flat**: it rises from 2.8 to 3.9 as $k_L$ decreases, and the rise begins at
  $k_L \approx 2\pi/460\,{\rm Mpc}/h$, the radial thickness of the LRG1 shell. Below that scale the long modes that
  fit inside the survey are predominantly *transverse* to the line of sight, and the quadrupole's response depends on
  the orientation of the long mode (its tidal/redshift-space part) while the monopole's does not at leading order.
  The super-survey mode of a thin shell is also transverse-dominated, so the $R_2$ relevant to SSC is the small-$k_L$
  value, 3.2-3.9, not the isotropic 2.8.
- **At $k_L = 0.005$ the estimator breaks down**, and the way it breaks is the key to section 5: the mean of
  $\delta_L$ at the galaxies becomes $\pm 2\times10^{-4}$ and changes sign between mocks. A filter wider than the
  survey can only see the long mode's survey mean, and that is zero by construction — the estimator normalises to the
  observed mean density.

## 5. Part D — the survey-mean mode and the "3x too weak" puzzle

### 5.1 The puzzle

The SSC term as implemented contains the *local-average* effect: the one long mode that is uniform across the survey
is absorbed by the normalisation, so its net effect on $\hat P$ is the physical response minus the normalisation's
response. With randoms whose $n(z)$ follows the data (the holi pipeline does this: the randoms copy each mock's
redshifts), ${\rm norm} \propto (1 + \delta_L)^2$, so the net response per unit galaxy mode is

$$ \frac{d\ln\hat P_0}{d\delta_L} = R_0 - 2 . $$

At tree level that is $1.2$-$1.6 - 2 = -0.4$ to $-0.8$. An earlier regression of each mock's $\hat P$ on its realised
shell density across 848 mocks (HANDOFF 9.2) found $-0.06$ to $-0.11$: "three times weaker than tree level", and
unexplained.

### 5.2 Resolution, from three directions

![Net survey-mean response: 848-mock regression, in-survey prediction and synthetic test](survey_mean_response_LRG1.png)

*Figure 4. (a) Slope of $\ln\hat P_0$ on the survey-mean galaxy overdensity $\delta_L$ across 848 mocks (points per
$k$ bin, diamonds per band), with the prediction $R_0 - 2$ from the in-survey response of Figure 3 (red bands) and
tree level (grey). (b) The same for the quadrupole, against the isotropic and transverse in-survey predictions.
(c) A synthetic catalogue with a known uniform mode: the measured slope per normalisation convention against its exact
expectation.*

1. **The regressor.** Redoing the regression with an explicit convention (`survey_mean_regression.py`; regressor
   $\delta_L = N_m/{\rm median}(N) - 1$, the weighted galaxy count of the cap) shows that 9.2's regressor,
   $\delta\ln\int n(z)^2$, is $2\delta_L$ (the measured ratio of scatters is 2.05) — and regressing on it reproduces
   9.2 to the last digit ($-0.062 \pm 0.020$, $-0.091 \pm 0.017$ NGC). Per unit $\delta_L$ the mocks give
   $+0.04 \pm 0.10$, $-0.12 \pm 0.04$, $-0.17 \pm 0.03$ (NGC) and $+0.10$, $-0.13$, $-0.24$ (SGC). The survey-mean
   scatter itself is $\sigma(\delta_L) = 0.42$ % (NGC) / 0.54 % (SGC), of which Poisson is 0.12 / 0.17 %.
2. **The prediction from one catalogue.** $R_0 - 2$ with the flat in-survey value gives $-0.09 \pm 0.03$,
   $-0.10 \pm 0.02$, $-0.19 \pm 0.01$: within 1 $\sigma$ of the mocks in every band and both caps. So the net response
   was never weak — it is a near cancellation of a *stronger*-than-tree physical response (1.9) against the
   normalisation's 2. For the quadrupole the mocks imply $R_2(q \to 0) = 3.4$-$4.1$, the transverse limit of Figure 3b,
   not the isotropic 2.8.
3. **The bookkeeping, with a known answer.** A lognormal-Poisson catalogue on the LRG1 NGC randoms
   ($6.2 \times 10^5$ galaxies, $\bar n P_{\rm fkp} = 4$) with an explicit uniform mode $\delta_u \in \{-0.06, \ldots, +0.06\}$
   multiplying the sampling probability, eight seeds, same random numbers for all $\delta_u$ so that cosmic variance
   cancels in the slope (`survey_mean_synthetic.py`). $R_{\rm global} = 2$ by construction. Measured
   $d\ln\hat P_0/d\delta_u$: $0.00 \pm 0.02$ when $n(z)$ and the FKP weights follow the data (expected $2 - 2$), the
   same without FKP weights, and $1.00 \pm 0.02$ when $n(z)$ is frozen (expected $2 - 1$: only $\alpha$ follows the
   density). The normalisation itself responds by 0.40 with per-realisation FKP weights (this is why the norm scatter
   is small and uninformative, HANDOFF 9.3), by 2.00 without them and by 1.00 when frozen — but the ratio
   $|F|^2/{\rm norm}$ does not see the weights.

**Conclusion of Part D.** For LRG1 the net survey-mean response of $\hat P_0$ is $R_0 - 2 = -0.1$ to $-0.2$ per unit
galaxy mode with $R_0 = 1.9$, established by the in-survey measurement, the 848-mock regression and a synthetic test
that agree. The "3x too weak" statement was a factor of 2 in the regressor combined with the cancellation.

## 6. What this means for `thecov`, and what is left

Status of the budget after this work:

| term | status | action for `thecov` |
|---|---|---|
| Gaussian | correct with the $\xi$-kernel window (random-field test, Figure 2); 1-2 % Gaussian-shaped residual against the mocks above $k = 0.06$ | make `kernel` the DESI default; remaining residual is pair-count sampling / mesh resolution, not the window concept |
| SSC + local average | sign, shape and normalisation bookkeeping validated; $R_0$ measured (1.9, LRG1) | use measured $R_0(q\to0)$ instead of tree level; give $R_2$ the window's long-mode anisotropy (thin shell $\Rightarrow$ transverse, $R_2 = 3.4$-$4$); compute $\sigma_L^2$ with the same anisotropy weighting |
| discreteness $B/\bar n$, $P/\bar n^2$ | measured from one catalogue; 2x / 1.5x (LRG1), 1.5x / 2.6x (QSO) tree level | port the closed-form thinning estimator into `DiscretenessCovariance` as the mesh-level calibration; do not use `window_convolve` |
| T0 | tree level rejected by mocks earlier (amplitude 0.2-0.4); not isolated here | bound as the remainder after the three measured terms; do not fit it |

Immediate next steps, in order of cost:
1. Run `thinning_closed_form.py` and `response_ibisp.py` on the real DESI LRG1/QSO catalogues (both caps, ~1-2 min
   each) and compare with the mock values — the first "calibrated from the data" numbers.
2. Insert the measured $R_0$, $R_2$ and discreteness amplitudes into the covariance and redo the mock comparison
   ($\chi^2/n$, variance ratios, whitened directions); the remainder is the T0 bound.
3. The super-survey variance with the correct anisotropy (radial from the mocks' $n(z)$ scatter, transverse from
   $|W(q)|^2$) — the last free piece of the SSC term.
4. Report the QSO $P/\bar n^2$ excess to the mock builders (a $p \approx 0.9$ thinning distinguishes duplicates from
   clustering).

## 7. Glossary

- **FKP field / weights.** $F = n_g - \alpha n_r$ with weights $w = w_{\rm comp}\,/(1 + \bar n(z) P_{\rm fkp})$,
  $P_{\rm fkp} = 10^4$ for LRGs; the FKP factor downweights dense regions to minimise the variance. $\alpha$ = total
  galaxy weight / total random weight.
- **norm.** The estimator's normalisation, $\alpha\sum_r w_r^2 n(z_r) \propto \int n^2 w^2$. Whether $n(z)$ in it
  follows the realised data or a fixed reference decides how the survey-mean mode cancels (section 5).
- **Window.** The function that convolves $P(k)P(k')$ into the Gaussian covariance; set by the survey density
  $m(\mathbf x)$ and weights. Local window $\int m^2$ vs pair-averaged $\xi$-kernel window.
- **Long mode / $\delta_L$ / $k_L$.** The density averaged over scales larger than $1/k_L$; in-survey long modes have
  zero mean over the survey by construction. The survey-mean mode is the single uniform one.
- **Response $R_\ell$.** $d\ln P_\ell/d\delta_L$, per unit galaxy long mode in this report.
- **Thinning.** Random subsampling of a point catalogue with probability $p$; leaves the continuous-field statistics
  unchanged and scales the discreteness terms by known powers of $(1-p)/p$.
- **Lognormal-Poisson catalogue.** Galaxies drawn as a Poisson process with rate $\propto \exp(G - \sigma_G^2/2)$, $G$
  Gaussian; the standard way to make a field with a known, exactly computable response ($R = 2$).
- **NUFFT.** Non-uniform FFT: evaluates a Fourier sum at arbitrary points (type 2) or transforms a weighted point set
  to a grid (type 1) without binning the points; `finufft` throughout.

## 8. Reproducibility

Scripts (all in `desi_validation/`, committed on `thecov2`): `grf_randoms.py`, `grf_randoms_plot.py`,
`grf_randoms_blocks.py`, `run_grfr_chain.sh` (Gaussian fields on the randoms and their variants), `plot_holes.py`,
`thinning_cov.py`, `thinning_closed_form.py` (discreteness), `response_ibisp.py` (responses, several `--k-L`),
`survey_mean_regression.py`, `survey_mean_synthetic.py` (survey-mean mode). Outputs on Perlmutter under
`~/thecov/{grfr_v2,thinning,responses,logs}/` (first LRG1/LRG2 field runs under `~/thecov_desi/holi_v3_mock173/`).
Every job ran on one CPU node of the debug queue in the `cosmodesi` environment; the whole programme used about
35 node-hours. The specialist reports listed at the top carry the full tables and the dead ends (control variates,
the data = randoms response constructions, the fixed-halves Poisson run), which are as informative as the results.
