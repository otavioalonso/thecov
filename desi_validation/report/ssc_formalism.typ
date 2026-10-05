#set document(title: "Super-sample covariance of windowed power-spectrum multipoles in thecov")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, it)

#let x = $bold(x)$
#let k = $bold(k)$
#let q = $bold(q)$
#let s = $bold(s)$
#let nh = $hat(bold(n))$
#let avg(body) = $lr(chevron.l body chevron.r)$
#let note(body) = block(fill: luma(242), inset: 8pt, radius: 3pt, width: 100%, body)

#align(center)[
  #text(size: 16pt, weight: "bold")[Super-sample covariance of windowed\ power-spectrum multipoles in `thecov`]
  #v(0.3em)
  #text(size: 11pt)[Redshift-space responses, the local average of a realised normalisation,\ and long-mode variances from tripolar window pair counts]
  #v(0.3em)
  #text(size: 9.5pt, fill: luma(80))[Formalism note, branch `thecov2` --- October 2026. Code: `thecov/ssc.py`; tests: `tests/test_ssc.py`; DESI check: `desi_validation/ssc_check.py`]
]

#v(0.8em)
#note[
*Summary.* Density fluctuations on scales comparable to or larger than the survey modulate the measured power spectrum coherently at all $k$. In the DESI DR2 LRG1 mocks this term is measured directly in the off-diagonal elements of the mock covariance, where the windowed Gaussian covariance vanishes: it is $A_(ell ell') P_0(k) P_0(k')$ with coherent amplitudes $sigma_(P_0) = 0.51%, 0.66%$ and $sigma_(P_2)\/P_0 = 0.94%, 1.22%$ (NGC, SGC), nearly uncorrelated, and it accounts for the quadrupole-amplitude error of the Gaussian covariance. This note derives the super-sample covariance (SSC) that `thecov` now computes: tree-level redshift-space responses of each multipole to a long mode, split by the long mode's orientation to the local line of sight; the local-average effect of an estimator whose $alpha$ and normalisation are computed from each realisation; and the long-mode variances as one-dimensional integrals of the linear correlation function against the tripolar window functions `thecov` already measures from pair counts. Nothing is fitted to the mocks.
]

= Estimator and long mode

*Estimator.* Galaxies and randoms with total weights $w$, $alpha = sum_g w_g \/ sum_r w_r$, the weighted field $F(#x) = w (n_g - alpha n_r)$ and, as in `pypower`/`jaxpower`,
$ hat(P)_ell (k) = (2 ell + 1) / "norm" integral (d Omega_k) / (4 pi) |tilde(F)(#k)|^2 cal(L)_ell (hat(k) dot #nh) - "shot noise", quad "norm" = alpha sum_c D_c R_c \/ V_c , $
with the local line of sight $#nh = hat(#x)$ and the normalisation computed _for each realisation_ from its own data $D_c$ and randoms $R_c$ in cells of $10 h^(-1)$Mpc. Write $m(#x) = macron(n)(#x) avg(w)(#x)$ for the mean weighted density; the clustering window of the Gaussian covariance is $W = m^2$ (or its pair-averaged version, see `window_kernel.typ`), with $integral W = I$, and the mean of the estimator is $P(k) I_k \/ "norm"$.

*Long mode.* Split the matter density into a long mode $delta_L$, with wavenumbers $q$ below the resolved $k$ and up to the survey scale, and short modes. To first order in $delta_L$ the short-mode power at position $#x$ is shifted by
$ Delta P(#k; #x) = integral (d^3 q) / (2 pi)^3 thin delta_L (#q) e^(i #q dot #x) thin R(#k, hat(q); #nh), $ <eq:dP>
where $R$ is the response of the redshift-space galaxy power spectrum to a long matter mode of unit amplitude along $hat(q)$.

= Responses

== Squeezed limit of the tree-level trispectrum

With the redshift-space perturbation-theory kernels $Z_1$, $Z_2$ (Scoccimarro, Couchman & Frieman 1999; local line of sight $#nh$),
$ Z_1(#k) = b_1 + f mu^2, $
$ Z_2(#k _1, #k _2) = b_1 F_2 + f mu^2 G_2 + (f mu k) / 2 [mu_1 / k_1 Z_1(#k _2) + mu_2 / k_2 Z_1(#k _1)] + b_2 / 2 + b_(s^2) / 2 S_2, $
with $#k = #k _1 + #k _2$, $mu = hat(k) dot #nh$ and $S_2 = (hat(k)_1 dot hat(k)_2)^2 - 1\/3$. The second-order field contains $2 Z_2(#k - #q, #q) delta(#k - #q) delta_L(#q)$, and its correlation with the linear field gives the response
$ R(#k, hat(q)) = lim_(q -> 0) 2 [ Z_2(#k, -#q) Z_1(#k) P(k) + Z_2(-#k + #q, -#q) Z_1(#k - #q) P(|#k - #q|) ]. $ <eq:R>
The two terms are each $cal(O)(1\/q)$: the bulk flow displaces short modes without changing their power. These pieces are odd in $#q$ and cancel; the finite remainder contains growth ($F_2$, $G_2$), dilation ($P(|#k - #q|)$ to first order in $q$, i.e. $d P \/ d ln k$), the bias responses $b_2$, $b_(s^2)$, and the redshift-space terms (the long mode's Kaiser and velocity-gradient effect on the short modes). $P$ is the linear matter power spectrum and $delta_L$ the linear matter long mode.

== Multipoles and the long-mode orientation

The estimator averages over $hat(k)$ with weight $cal(L)_ell (hat(k) dot #nh)$. After that average, $R$ depends on $hat(q)$ only through $nu = hat(q) dot #nh$, so
$ R_ell (k; nu) = (2 ell + 1) / 2 integral_(-1)^1 d mu thin cal(L)_ell (mu) avg(R)_phi = sum_n R_ell^((n))(k) cal(L)_n (nu), $
$ R_ell^((n))(k) = a_(ell n) P(k) + c_(ell n) (d P) / (d ln k), $ <eq:Rln>
with numbers $a_(ell n)$, $c_(ell n)$ that depend on $(b_1, b_2, b_(s^2), f)$. `response_coefficients` evaluates @eq:R on a quadrature grid in $(mu, nu, phi)$ and takes the squeezed limit as the average of $q = plus.minus epsilon$, which removes the odd $1\/epsilon$ terms exactly and leaves an $cal(O)(epsilon^2)$ error ($10^(-8)$ for $epsilon = 10^(-4)$). Gauss--Legendre in $mu$ and $nu$ and a uniform grid in $phi$ are exact for the polynomials that occur.

*Checks* (tested). Real space, $b_1 = 1$: $R_0^((0)) = (47\/21) P - (1\/3) d P \/ d ln k$ and $R_2^((2)) = (2\/3)(8\/7) P - (2\/3) d P\/ d ln k$, the isotropic and tidal (separate-universe) responses, and nothing else. Galaxy bias adds $2 b_1 b_2 P$ to $R_0^((0))$ and $(4\/3) b_1 b_(s^2) P$ to $R_2^((2))$. In redshift space $R_ell^((4)) = 0$: only $n = 0, 2$ occur. For $b_1 = 2.2$, $f = 0.76$, $b_2 = 0.28$ (Lazeyras et al. 2016), $b_(s^2) = -4\/7 (b_1 - 1)$:

#align(center, table(columns: 3, align: center, stroke: 0.4pt,
  [$ell$], [$a_(ell 0) \/ c_(ell 0)$], [$a_(ell 2) \/ c_(ell 2)$],
  [0], [$19.02 \/ -2.62$], [$3.76 \/ -1.54$],
  [2], [$12.43 \/ -2.22$], [$6.02 \/ -7.30$],
  [4], [$1.31 \/ -0.28$], [$2.46 \/ -1.38$]))

For comparison, the Kaiser monopole and quadrupole are $6.07 P$ and $2.56 P$. Two consequences: $P_2$ responds to the isotropic mode ($a_(20)$, mostly through the Kaiser term) as well as to the line-of-sight component $n = 2$, and its largest dilation coefficient sits in $c_(22)$. These are the two "amplitude" directions found in the mocks.

= Response of the estimator: beat coupling and local average

With the long mode, $n_g = macron(n)(1 + delta_g)$ where $delta_g$ is referenced to the global mean. The short-mode part of $|tilde(F)|^2$ is the local power of $delta_g$ weighted by the clustering window, so it gains $integral W(#x) Delta P(#k; #x)$: the _beat coupling_ (BC). The constant $alpha$ only shifts the $k approx 0$ modes. The normalisation, however, follows the realisation:
$ "norm" prop alpha sum_c D_c R_c prop (1 + macron(delta)^M)(1 + macron(delta)^W), quad macron(delta)^w = (integral w(#x) delta_(g,L)(#x)) / (integral w), $
with $M = m$ (the average in $alpha = sum_g w \/ sum_r w$) and $W = m^2$ (the average of the data in $D_c R_c$). $delta_(g,L) = (b_1 + f nu^2) delta_L$ is the observed long mode. Hence, to first order,
$ Delta hat(P)_ell (k) = d_k sum_n R_ell^((n))(k) D_n^W - hat(P)_ell (k) sum_n g_n (D_n^W + D_n^M), $ <eq:dPhat>
$ D_n^w = 1 / (integral w) integral d^3 #x thin w(#x) integral (d^3 q) / (2 pi)^3 delta_L (#q) e^(i #q dot #x) cal(L)_n (hat(q) dot hat(#x)), quad g_0 = b_1 + f / 3, quad g_2 = (2 f) / 3 . $
The second term is the _local average_ (LA). For a uniform window and an isotropic long mode in real space it is $-2 b_1 P$, and @eq:dPhat reduces to the familiar $(R - 2)$. Here it splits into an $m$-weighted part from $alpha$ and an $m^2$-weighted part from the data in `norm`. Those two parts differ when $m$ varies across the survey. The DESI LRG1 mocks show this cancellation: the measured slope $partial hat(P)_0 \/ partial delta_("norm") approx (0.1 "to" 0.25) macron(P)_0$ is small and positive, as $(R - 2)\/2$ predicts, not $-macron(P)_0$.

*Dilution.* $d_k = I_k \/ "norm"$ converts the response of the true power into that of the measured, window-diluted spectrum. For footprints with fine veto masks it must be the pair-averaged window's $I_k$, which is $approx 1$ for DESI LRG1. The local $integral m^2 \/ "norm" = 1.22$ would inflate the BC variance by $1.5$. In the LA term $hat(P)_ell$ is the measured spectrum itself. That is the model when the covariance's model is masked (the mock mean), and the model times $d_k$ otherwise. Setting `local_average=False` describes a normalisation fixed in advance.

= Long-mode variances from the window pair counts

The covariance of the projections in @eq:dPhat is
$ sigma^2_((w n)(w' n')) = avg(D_n^w D_(n')^(w')) = 1 / (I_w I_(w')) integral d^3 #x thin d^3 #x' thin w(#x) w'(#x') integral (d^3 q) / (2 pi)^3 P(q) thin e^(-i #q dot #s) cal(L)_n (hat(q) dot hat(#x)) cal(L)_(n') (hat(q) dot hat(#x)'), $
with $#s = #x' - #x$. Expanding the plane wave and both Legendre polynomials in spherical harmonics, the $hat(q)$ integral is a Gaunt coefficient. The remaining angular dependence on $(hat(#x), hat(#x)', hat(#s))$ is the real tripolar harmonic
$ S_(n n' lambda)(hat(#x), hat(#x)', hat(#s)) = sum_(m m' M) mat(n, n', lambda; m, m', M) Y^*_(n m)(hat(#x)) Y^*_(n' m')(hat(#x)') Y^*_(lambda M)(hat(#s)) $
used by `thecov`'s window pair counts,
$ Q^(w w')_(n n' lambda)(s) = integral d Omega_s integral d^3 #x thin w(#x) w'(#x + #s) S_(n n' lambda)(hat(#x), hat(#x)', hat(#s)). $
The result is a sum of one-dimensional integrals,
$ sigma^2_((w n)(w' n')) = (4 pi)^(3\/2) / (I_w I_(w')) sum_lambda (-1)^(lambda \/ 2) sqrt((2 lambda + 1) / ((2 n + 1)(2 n' + 1))) mat(n, n', lambda; 0, 0, 0) integral s^2 d s thin xi_lambda (s) Q^(w w')_(n n' lambda)(s), $ <eq:sigma2>
$ xi_lambda (s) = integral (q^2 d q) / (2 pi^2) P(q) j_lambda (q s). $
It is exact for the local line of sight: no plane-parallel or FFT line-of-sight approximation enters. For $n = n' = 0$ it is the usual $sigma_b^2 = integral d^3 s thin xi(s) Q_(w w')(s) \/ (I_w I_(w'))$. The pairs of windows needed are $(W, W)$, $(W, M)$ and $(M, M)$, with triples $(n, n', lambda) in {(0,0,0), (0,2,2), (2,0,2), (2,2,0), (2,2,2), (2,2,4)}$. The window $M$ (kind `'M'` in `thecov.Window`, tilde weight $w_r$) is new; $W$ is the local $m^2$. Long modes do not resolve veto holes, so the local window has the right shape, and the holes enter only through $d_k$.

*Check* (tested). For a uniform sphere of radius $R$ seen from far away, $hat(#x) approx #nh$, $Q_(n n' lambda)$ vanishes for $lambda > 0$, and @eq:sigma2 gives $sigma^2_(00) = integral q^2 d q \/ (2 pi^2) P(q) [3 j_1(q R) \/ (q R)]^2$, $sigma^2_(22) = sigma^2_(00)\/5$ and $sigma^2_(02) = 0$. The pair counts reproduce these to 1--3%.

*Resolution in $mu$.* The pair counts evaluate $S$ at the mean $mu = hat(x) dot hat(s)$ of each cell. This is first-order exact, and its second-order error biases $Q_(224)$ by $-1%$ of $Q_(000)$ with 24 cells. The $lambda = 4$ term of @eq:sigma2 carries a larger coefficient than $lambda = 0$, and $xi_4$ has no compensating negative tail, so the error became $-8%$ in $sigma^2_(22)$ for the sphere. With 96 cells it is gone. The SSC therefore keeps its own window library with 96 $mu$ cells. It also uses fewer near pairs, since small separations matter little for long modes.

= The covariance

Collecting @eq:dPhat for all multipoles and bins,
$ C^"SSC"_(ell_1 ell_2)(k_i, k_j) = sum_(X, Y) c^X_(ell_1)(k_i) thin c^Y_(ell_2)(k_j) thin sigma^2_(X Y), quad X, Y in {(W, 0), (W, 2), (M, 0), (M, 2)}, $
$ c^((W, n))_ell (k) = d_k R_ell^((n))(k) - hat(P)_ell (k) g_n, quad c^((M, n))_ell (k) = - hat(P)_ell (k) g_n, $
with responses and spectra averaged over each $k$ bin. The matrix has rank $<= 4$: it is the low-rank term seen in the mocks. Independent caps add with the weights of the combined estimator. For the norm-weighted mean of NGC and SGC, $C = sum_r "norm"_r^2 C_r \/ (sum_r "norm"_r)^2$.

= Numerics

- $xi_lambda (s)$ on `thecov`'s $s$ grid, by direct quadrature on a linear $q$ grid with $Delta q <= 0.25 \/ s_max$ and a $1 h^(-1)$Mpc Gaussian damping (irrelevant at the separations that matter).
- Bin averages of $R_ell^((n))$ and $hat(P)_ell$ with the Gauss--Legendre nodes of the shell kernels ($k^2$-weighted).
- Pair counts: the far pairs of `thecov`'s subsample ($n_"sub"$), $3 times 10^5$ near pairs, 96 $mu$ cells. They are saved and reloaded; they depend only on the geometry.
- Cost: a few minutes per cap. The responses and $xi_lambda$ take seconds and can be recomputed for any cosmology or bias without new pair counts.

= Validation targets and an estimate

The off-diagonal decomposition of the mock covariance (`rank1_excess.py`) gives quantities that the SSC must predict without fitting:

#align(center, table(columns: 5, align: center, stroke: 0.4pt,
  [LRG1], [$sigma_(P_0)$], [$sigma_(P_2) \/ P_0$], [$r(P_0, P_2)$], [ratio SGC/NGC],
  [NGC], [0.51%], [0.94%], [$+0.18$], [],
  [SGC], [0.66%], [1.22%], [$0.00$], [1.30 (both modes)]))

The ratio $1.30$ should follow from the $sigma^2$ of the two caps, against $sqrt(V_"NGC" \/ V_"SGC") = 1.39$ for a pure volume scaling. An order-of-magnitude estimate for the isotropic mode of LRG1 NGC: per unit long matter mode, BC gives about $+3.5$ times $P_0$ (including dilation at $n_"eff" approx -1.5$) and LA gives $-2 g_0 approx -4.9$. The net is $approx -1.4$, times $sigma_b approx 0.35%$ for a volume of $approx 1.5 times 10^9 (h^(-1)"Mpc")^3$. That is $sigma_(P_0) approx 0.5%$, the measured order. The near cancellation between BC and LA makes the prediction sensitive to $b_1$, $b_2$ and the nonlinear response at $k gt.tilde 0.1$. This is the main uncertainty of the tree-level model.

= Limitations

- *Tree-level responses.* At $k gt.tilde 0.1 h"Mpc"^(-1)$ the nonlinear response differs from tree level, and fingers of God reduce the redshift-space response. Response coefficients from simulations, or a separate-universe calibration, could replace $a_(ell n)$, $c_(ell n)$ without changing anything else.
- *Bias parameters.* $b_2$ and $b_(s^2)$ come from relations ($b_2(b_1)$, local Lagrangian), unless they are given. Their effect is largest on $R_0^((0))$ and $R_2^((2))$.
- *Auto-spectra of one tracer.* Cross-spectra need the responses of two tracers.
- *Not included.* The response of the shot noise; the connected trispectrum of short modes (the part of the excess that grows with the bin width); evolution of the long mode across the redshift range (the linear $P$ at $z_"eff"$ is used).
