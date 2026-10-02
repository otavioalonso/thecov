#set document(title: "The clustering window of the Gaussian covariance beyond the local approximation")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, it)

#let x = $bold(x)$
#let y = $bold(y)$
#let r = $bold(r)$
#let k = $bold(k)$
#let kp = $bold(k)'$
#let q = $bold(q)$
#let avg(body) = $lr(chevron.l body chevron.r)$
#let note(body) = block(fill: luma(242), inset: 8pt, radius: 3pt, width: 100%, body)

#align(center)[
  #text(size: 16pt, weight: "bold")[The clustering window of the Gaussian covariance\ beyond the local approximation]
  #v(0.3em)
  #text(size: 11pt)[From definitions to an implementation in `thecov`]
  #v(0.3em)
  #text(size: 9.5pt, fill: luma(80))[Validation note, branch `thecov2` --- October 2026]
]

#v(0.8em)
#note[
*Summary.* The Gaussian covariance of a windowed power spectrum contains, in its clustering terms, the product of the window at the two ends of a galaxy pair, $m(#x) m(#x + #r)$, weighted by the correlation function $xi(#r)$. `thecov` replaces this by $m^2(#x)$ (the _local approximation_). Near window structure smaller than a correlation length --- veto holes, footprint edges, completeness patches --- this is wrong, and for DESI LRG1 it biases the Gaussian variance low by $tilde 7%$. The correct clustering window is $W_k (#x) = m(#x) (K_k star m)(#x)$, with a kernel $K_k$ fixed by $xi$ itself, so no free smoothing scale is introduced. The same quantity sets the dilution of the measured power spectrum by the window, $I_k = integral W_k$, which replaces the single number $integral m^2$ in the model normalisation. We give the corrections for a theory and for a measured model power spectrum, the evidence from the DESI DR2 validation, and an implementation.
]

= Definitions

*Catalogues and weights.* Galaxies $g$ at $#x _g$ with total weights $w_g$ (`WEIGHT` $times$ `WEIGHT_FKP`), randoms $r$ with weights $w_r$, and $alpha = sum_g w_g \/ sum_r w_r$. The weighted overdensity field is
$ F(#x) = sum_g w_g delta_D (#x - #x _g) - alpha sum_r w_r delta_D (#x - #x _r). $

*Mean weighted density.* With weights treated as random variables, independent between objects and of the density field,
$ m(#x) = avg(sum_g w_g delta_D (#x - #x _g)) = macron(n)(#x) avg(w)(#x), $
where $avg(w)(#x)$ is the local mean weight. Clustering terms involve $avg(w)$; shot-noise terms involve $avg(w^2)$ through the shot-noise density
$ S(#x) = (1 + alpha) macron(n)(#x) avg(w^2)(#x), $
normalised so that $integral S = $ `num_shotnoise`, the realised shot noise the spectra subtract.

*Estimator.* With the normalisation `norm` of the spectrum files (computed by `jaxpower` on $10 h^(-1)$Mpc cells),
$ hat(P)(#k) = (|tilde(F)(#k)|^2 - integral S) / "norm", $
averaged over a $k$-shell and, for multipoles, weighted by Legendre polynomials of the line of sight (we keep the monopole notation; the multipole case follows with the usual $cal(L)_ell$ factors).

= The exact Gaussian covariance

For a Gaussian overdensity $delta$ with correlation function $xi$ and power spectrum $P$, the two-point function of $tilde(F)$ has a clustering and a shot-noise part,
$ avg(tilde(F)(#k) tilde(F)^*(#kp)) = G_P (#k, #kp) + G_S (#k - #kp), $ <eq:G>
$ G_P (#k, #kp) = integral d^3 #x d^3 #y thin m(#x) m(#y) xi(#x - #y) e^(-i #k dot #x + i #kp dot #y), quad G_S (#q) = integral d^3 #x thin S(#x) e^(-i #q dot #x). $
By Wick's theorem the covariance of the estimator is
$ "Cov"[hat(P)(k), hat(P)(k')] = 1/"norm"^2 lr(chevron.l |G_P (#k, #kp) + G_S (#k - #kp)|^2 + (#kp -> -#kp) chevron.r)_(k, k' "shells"), $ <eq:cov>
which expands into three terms: $P P$ ($|G_P|^2$), $P S$ ($2 "Re" G_P G_S^*$) and $S S$ ($|G_S|^2$). @eq:cov is exact for a Gaussian field; everything below concerns how $G_P$ is evaluated.

= The local approximation and its failure

Write $#y = #x + #r$ in $G_P$:
$ G_P (#k, #kp) = integral d^3 #x thin e^(-i (#k - #kp) dot #x) integral d^3 #r thin xi(#r) e^(i #kp dot #r) thin m(#x) m(#x + #r). $ <eq:Gr>
The $#r$ integral extends over a correlation length (tens of $h^(-1)$Mpc for the $k$ of interest). If $m$ is smooth on that scale, $m(#x + #r) approx m(#x)$, the $#r$ integral gives $P(k')$, and
$ G_P (#k, #kp) approx P(k) thin tilde(W)(#k - #kp), quad W(#x) = m^2(#x). $ <eq:local>
This is the local approximation used by `thecov` (and by FFT-mesh implementations of the Gaussian covariance). It fails wherever $m$ changes within a correlation length. DESI footprints contain many such structures: bright-star and other veto masks of arcminute size (below $1 h^(-1)$Mpc), completeness variations on tile scales, and footprint edges. Since the randoms sit only where $m > 0$, `thecov` evaluates $m^2$ between the holes, as if the partner of every pair were also between the holes.

= The pair-averaged window

Keep $m(#x + #r)$ inside the $#r$ integral of @eq:Gr and use only $k approx k'$ within a window width (the same order of approximation as @eq:local):
$ G_P (#k, #kp) approx P(k) thin tilde(W)_k (#k - #kp), quad W_k (#x) = m(#x) thin (K_k star m)(#x), $ <eq:Wk>
$ K_k (#r) = (xi(#r) e^(i #k dot #r)) / P(k), quad integral d^3 #r thin K_k (#r) = 1. $ <eq:kernel>
The correct clustering window is $m$ times $m$ _averaged over the partner's position_ with the kernel $K_k$. For an isotropic estimate one uses the angle average,
$ K_k (r) = (xi(r) j_0 (k r)) / P(k), quad integral_0^infinity 4 pi r^2 K_k (r) thin d r = 1. $
Two consequences:

+ *No free smoothing scale.* The averaging is set by $xi$; its radial weight $4 pi r^2 xi(r) j_0(k r)$ extends to $r tilde 10$--$30 h^(-1)$Mpc for $k = 0.1$--$0.3 h"Mpc"^(-1)$ and narrows with $k$. The local approximation is the limit $K_k -> delta_D$.
+ *Edges and holes are handled together.* Near an edge or a hole, $K_k star m$ is diluted because part of the kernel falls outside the survey. This is the physical loss of pairs straddling the edge, not an estimator bias.

= The mean and the normalisation

Setting $#kp = #k$ in @eq:G gives the mean of the estimator:
$ avg(hat(P)(k)) = G_P (#k, #k) \/ "norm" = P(k) thin I_k \/ "norm", quad I_k = integral d^3 #x thin m(#x) (K_k star m)(#x) = integral W_k. $ <eq:mean>
The measured power is diluted by the window by the $k$-dependent factor $I_k \/ "norm"$. The local approximation would give $integral m^2 \/ "norm"$ instead, larger by the same hole and edge effects (for DESI LRG1, $integral m^2 \/ "norm" = 1.22$, while a window resolved and diluted at $7'$ gives $1.03$).

= Corrections for the two choices of model power spectrum

The three terms of @eq:cov are evaluated as
$ P P &: quad P(k) P(k') thin |tilde(W)_k (#k - #kp)|^2, \
P S &: quad P(k) thin tilde(W)_k (#k - #kp) tilde(S)^*(#k - #kp) + "c.c.", \
S S &: quad |tilde(S)(#k - #kp)|^2, $
all divided by $"norm"^2$. $S$ is local (self-pairs) and needs no kernel.

== Case 1: theory power spectrum

Input: $P_"th" (k)$, the true redshift-space galaxy power, unconvolved.

#table(columns: (auto, 1fr), stroke: 0.4pt, inset: 6pt,
  [*clustering window*], [$W_k = m (K_k star m)$ in $P P$ and in the clustering leg of $P S$],
  [*shot-noise window*], [$S$ unchanged: each random's own $w^2$, $integral S = $ `num_shotnoise`],
  [*normalisation*], [the files' `norm` (the estimator's), not $integral m^2$ and not $I_k$],
  [*model power*], [$P_"th"$ as is, with no rescaling. It must be the full power the estimator sees: nonlinear, biased, redshift-space, *and* any non-Poisson stochasticity, since the files subtract only the realised Poisson shot noise.],
)

== Case 2: measured power spectrum

Input: the mean of the mocks, $macron(P)(k) approx P(k) I_k \/ "norm"$ (@eq:mean), shot noise subtracted.

#table(columns: (auto, 1fr), stroke: 0.4pt, inset: 6pt,
  [*clustering window*], [$W_k = m (K_k star m)$, as in case 1],
  [*shot-noise window*], [as in case 1],
  [*normalisation*], [the files' `norm`, as in case 1],
  [*model power*], [$P_"model" (k) = macron(P)(k) thin "norm" \/ I_k$, which recovers $P(k)$ and makes case 2 identical to case 1. In `thecov` this is automatic: its model correction divides by its own window integral $I = alpha sum_r w_r "NW"_r$, which equals $I_k$ once $"NW" = K_k star m$.],
  [*not undone*], [the window convolution also changes the _shape_ of $macron(P)$: smearing of the first bins and of the BAO feature, the integral constraint, and leakage between multipoles. Its effect on the covariance is about twice the relative error in $P$, small for $k gt.tilde 0.05 h"Mpc"^(-1)$. To remove it, deconvolve: fit the $P$ whose window-convolved multipoles equal $macron(P)_ell$.],
)

== What `thecov` did, and why the error was small

`thecov` used $macron(P) thin "norm" \/ integral m^2$ as the model and $W = m^2$ as the window. The first makes the model $P$ too low by $I_k \/ integral m^2 approx 0.80$--$0.85$; the second makes the $P P$ window too large. The two errors have opposite signs and nearly cancel, leaving the $tilde 7%$ deficit measured below.

= Evidence from the DESI DR2 validation (LRG1, holi v3, 859 mocks)

*Exact Gaussian reference.* Gaussian random fields $delta$ with the model power, multiplied by $m(#x)$ on a $6 h^(-1)$Mpc mesh (with $m$ from a healpix map at nside 512 times the weighted $n(z)$, from 10 random files), plus white noise of density $S$, estimated as in the files, 1000 realisations per cap: this evaluates @eq:cov with no local approximation. With the input power calibrated so that the mean equals the mocks', the calibration factor $avg(hat(P))_"mocks" \/ avg(hat(P))_"GRF"$ is the measured $k$-dependence of $I_k$ relative to `thecov`'s $integral m^2$:

#align(center, table(columns: 5, stroke: 0.4pt, inset: 5pt, align: center,
  [], [$0.02$--$0.05$], [$0.05$--$0.1$], [$0.1$--$0.2$], [$0.2$--$0.3$],
  [calibration NGC], [1.276], [1.242], [1.214], [1.202],
  [calibration SGC], [1.262], [1.220], [1.193], [1.179],
  [Var GRF / `thecov` NGC], [1.094], [1.038], [1.064], [1.050],
  [Var GRF / `thecov` SGC], [1.071], [1.099], [1.074], [1.059],
))

The variance ratio is independent of the bin width ($Delta k = 0.005, 0.01, 0.02$), i.e. a Gaussian-term error; the GRF with the high-resolution window and with `thecov`'s smooth mean weight (both diluted by holes) agree to $0.2%$, so fine structure of $avg(w)$ plays no role.

*Top-hat dilution.* Replacing `thecov`'s $m$ at each random by $m$ times the footprint fill fraction of a healpix pixel (a top-hat stand-in for $K_k$) raises the variance towards the exact result and converges at pixels of $tilde 10 h^(-1)$Mpc:

#align(center, table(columns: 6, stroke: 0.4pt, inset: 5pt, align: center,
  [pixel], [none], [$7'$], [$14'$], [$28'$], [$56'$],
  [P0 variance / default, NGC, $k < 0.1$], [1], [1.044--1.048], [1.056--1.061], [1.063--1.068], [1.068--1.075],
  [P0 variance / default, SGC, $k < 0.1$], [1], [1.041--1.043], [1.058--1.060], [1.073--1.076], [1.084--1.087],
  [$avg(chi^2)\/n$ GCcomb, $Delta k = 0.005$], [1.098], [1.067], [1.051], [1.046], [1.040],
))

With the $28'$ pixel and more pair counts (the sampling noise of the pair counts inflates $chi^2$ by $tilde 0.01$), $avg(chi^2)\/n = 1.024$ (NGC), $1.032$ (SGC), $1.026$ (GCcomb), down from $1.10$; the remaining excess is non-Gaussian (it grows with the bin width). The top-hat scale is the arbitrary element that the kernel $K_k$ of @eq:kernel removes.

= Implementation

Only the value of $m$ given to each random (`NW`) changes; pair counting, windows and covariance assembly are untouched.

+ *Kernel.* From the model monopole $P_0 (k)$ (theory, or $macron(P)_0 thin "norm"\/I$ iterated once), compute $xi_0 (r)$ by a Hankel transform and $K_k (r) = xi_0 (r) j_0(k r)\/P_0 (k)$, truncated at $R_"max" approx 150 h^(-1)$Mpc and renormalised to unit integral. Use the band-averaged kernel for each of $N_b = 3$--$4$ $k$-bands spanning $0.02$--$0.3 h"Mpc"^(-1)$.
+ *Smoothed density.* Paint $alpha sum_r w_r delta_D (#x - #x _r)$ from many random files (e.g. 10) on a mesh with cells of $3$--$5 h^(-1)$Mpc; convolve with $K_k$ by FFT; interpolate $(K_k star m)$ at `thecov`'s randoms. Painting is safe here: the smoothing is linear and much wider than a cell, so the painting noise averages out (nothing noisy is squared). The painted randoms must be independent of `thecov`'s randoms, so that no random averages with itself (the self-pair would bring back the $avg(w^2)$ bias).
+ *Windows.* For each band, set $"NW"_r = (K_k star m)(#x _r)$ and compute the window pair counts as now. The $S$ window is unchanged. The window integral is then $I_k$ and the model correction of case 2 is automatic; for case 1, switch it off and pass $P_"th"$.
+ *Covariance.* Assemble each band's rows and columns from its own windows and interpolate between bands in $(k, k')$; blocks with $k$ and $k'$ in different bands use the band of the geometric mean.
+ *Pair-count sampling.* Use enough randoms and near pairs ($10^10$, with the near sample not capped by the randoms kept) for the off-diagonal structure: its noise biases $chi^2$ high.

*Numerical parameters and their convergence tests.* Mesh cell (halve it), $R_"max"$ (increase it), number of $k$-bands (double it), random files for the painted density (double them), near pairs (double them). None of them is a physical choice.

= Validation plan and caveats

+ $I_k$ from the kernel must reproduce the GRF calibration curve above, with no mocks involved.
+ `thecov` with $"NW" = K_k star m$ must match the GRF variance at every $k$ and bin width, for LRG1 NGC and SGC, and for a second tracer with a different footprint and redshift (ELG1 or BGS).
+ Against the mocks, the bin-width-independent excess should vanish; the remainder is the non-Gaussian covariance (trispectrum, super-sample).

*Caveats.* (i) The kernel uses the isotropic $xi_0$ and $j_0$; the redshift-space $xi$ and the direction of $#k$ make it anisotropic for the multipole covariances. The top-hat test reduced the $P_2$ and $P_4$ excess as much as $P_0$'s, which suggests the isotropic kernel suffices; the GRF with an anisotropic $P$ can test it. (ii) $k approx k'$ within a window width, and $P$ smooth across it, as in the original approximation: this remains the low-$k$ limitation. (iii) Case 1 requires the stochastic part of the theory power to match the data's; case 2 inherits it from the mocks.
