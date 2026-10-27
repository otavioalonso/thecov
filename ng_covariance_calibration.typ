// Non-Gaussian covariance of windowed power-spectrum multipoles from the catalogue itself
// Companion to pk_covariance.typ (the Gaussian term). Derivations and estimators for the
// discreteness, super-sample and trispectrum terms, built from one realisation of the data.

#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#show link: set text(fill: rgb("#36a"))

#let defbox(title, body) = block(
  width: 100%, stroke: 0.6pt + luma(110), inset: 9pt, radius: 3pt, fill: luma(247),
  [*#title* #v(3pt) #body],
)
#let resultbox(body) = block(
  width: 100%, stroke: 1pt + rgb("#a33"), inset: 10pt, radius: 4pt, fill: rgb("#fff3f3"), body,
)
#let notebox(body) = block(
  width: 100%, stroke: 0.6pt + rgb("#36a"), inset: 8pt, radius: 3pt, fill: rgb("#f0f5ff"), body,
)
#let todobox(body) = block(
  width: 100%, stroke: 0.6pt + rgb("#b80"), inset: 8pt, radius: 3pt, fill: rgb("#fff9e8"), body,
)
#let Cov = $op("Cov")$
#let Var = $op("Var")$
#let E = $op("E")$
#let Re = $op("Re")$
#let avg(x) = $chevron.l #x chevron.r$
#let dd = $dif$

#align(center)[
  #text(size: 16pt, weight: "bold")[The non-Gaussian covariance from the catalogue itself] \
  #v(2pt)
  #text(size: 12pt)[Discreteness, super-sample and trispectrum terms calibrated on one realisation of the data] \
  #v(4pt)
  #text(size: 9.5pt, fill: luma(90))[Companion to _Gaussian covariance of windowed power-spectrum multipoles_ (`pk_covariance.typ`). thecov `thecov2` at `bdc247b`; mock results from HANDOFF.md sections 7, 9, 12 and the 2026-10-15/19 updates.]
]
#v(8pt)

= Purpose and the rule

The analytic covariance of the DESI power-spectrum multipoles will be built once, for one realisation of the survey. This note sets one rule for everything beyond the Gaussian term and then derives, term by term, how the rule can be met.

#defbox[The rule][
  Every input of the covariance is one of
  + a quantity *measured on the catalogue itself*: the multipoles $hat(P)_ell (k)$ and the parameters fitted to them ($b_1$, $f$, $sigma_v$), pair and quadruplet sums over the galaxies, and the response of the small-scale power to the long modes present in the survey;
  + a quantity *fixed by the window*: pair counts of the randoms and the long-mode variances of the footprint;
  + a quantity *fixed by cosmology*: the linear power spectrum $P_"L" (k)$ and the separate-universe growth responses.

  No input is an amplitude fitted to a mock ensemble. Mocks are judges: the whole covariance is built from one mock catalogue as if it were the data and compared with the covariance of the ensemble (848 clean LRG1 mocks, 859 QSO mocks). The spread of the result over a few single catalogues is the uncertainty to carry on the data.
]

The covariance is organised as

$
C_(i j) = G_(i j) [hat(P)_ell, W] + C^"disc"_(i j) [B\/overline(n), P\/overline(n)^2] + C^"SSC"_(i j) [sigma^2_L, R_ell] + C^(T_0)_(i j) [R_1, R_2 ; "tree"],
$ <eq:decomp>

with $i, j$ labelling (multipole, $k$-bin) pairs. The Gaussian term $G$ is the subject of the companion note; it is a functional of the window (randoms) and of the spectrum, with nothing to calibrate once the spectrum is the measured one. The three non-Gaussian terms are the subject here. @sec:poisson sets up the estimator and shows where each term comes from; @sec:thinning derives the discreteness terms from a thinning of the galaxies, in closed form; @sec:response derives the response estimator that fixes the super-sample term and the squeezed trispectrum; @sec:ssc and @sec:t0 assemble those two terms; @sec:validation gives the validation protocol and @sec:status the state of the implementation.

#notebox[
  *Reading guide.* Boxed results are what gets implemented. Each derivation starts from the discrete estimator, so that the discreteness terms appear with their exact weights; a reader who knows the continuum derivation of the Gaussian term can skip to the boxes and to @sec:validation.
]

= The estimator and where the terms come from <sec:poisson>

== The FKP field and the estimator

Galaxies $g$ sit at $bold(x)_g$ with total weights $w_g$ (completeness, systematics and FKP); randoms $r$ at $bold(x)_r$ with weights $w_r$, and $alpha = sum_g w_g \/ sum_r w_r$. The density field and its Fourier transform are

$
F(bold(k)) = sum_g w_g e^(-i bold(k) dot bold(x)_g) - alpha sum_r w_r e^(-i bold(k) dot bold(x)_r) equiv sum_g w_g e_g (bold(k)) - F_r (bold(k)), quad e_g (bold(k)) equiv e^(-i bold(k) dot bold(x)_g).
$ <eq:F>

For the monopole (the multipoles add a Legendre weight at one of the two points, see the companion note; nothing below depends on it until @sec:thinning-rsd) the estimator in a shell $i$ with $N_i$ modes is

$
hat(P)(k_i) = frac(1, cal(N)) [ frac(1, N_i) sum_(bold(k) in i) |F(bold(k))|^2 - S ], quad S = sum_g w_g^2 + alpha^2 sum_r w_r^2,
$ <eq:Phat>

where $cal(N)$ is the estimator's normalisation (pypower / jaxpower `norm`) and $S$ the *realised* shot noise (`num_shotnoise`), i.e. the self-pair term of $|F|^2$ computed from the catalogue, not an expectation. Both are inputs of the Gaussian term (`PowerSpectrum(norm=, shotnoise=)`).

== Poisson sampling and the three non-Gaussian terms

The galaxies are a Poisson sampling of a clustered continuous field: in a cell $dd^3 x$ the number of galaxies is Poisson with mean $overline(n)(bold(x)) [1 + delta_g (bold(x))] dd^3 x$. Averages over the sampling at fixed $delta_g$ turn sums over galaxies into integrals, with one rule: a product of the same galaxy with itself does not factorise. For a sum over galaxies of a product of $m$ factors, the terms where $m$ indices coincide carry the weight $w^m$ once and the density $overline(n)$ once, instead of $(overline(n) w)^m$:

$
E_"P" [ sum_(g_1, dots, g_m) w_(g_1) dots.c w_(g_m) thin f(bold(x)_(g_1), dots, bold(x)_(g_m)) ]
&= integral product_(a = 1)^m dd^3 x_a thin (overline(n) w)(bold(x)_a) [1 + delta_g (bold(x)_a)] thin f(bold(x)_1, dots, bold(x)_m) \
&quad + sum_("partitions") integral product_("blocks" b) dd^3 x_b thin (overline(n) w^(m_b))(bold(x)_b) [1 + delta_g (bold(x)_b)] thin f|_(bold(x)_(g) = bold(x)_b "for" g in b) ,
$ <eq:poisson-moments>

The second line runs over the ways of grouping the $m$ indices into coincident blocks (a block of $m_b$ indices is one galaxy counted $m_b$ times); each block contributes one density and the weight to the power $m_b$. Applied to the covariance of @eq:Phat, which is a four-point object in $F$:

- *no coincidence*: the connected four-point function of the clustered field → super-sample covariance (long modes coupling through the window) and the connected trispectrum $T_0$;
- *two of the four points coincide*: a three-point function times a shot-noise factor → the *collapsed bispectrum* term, $prop B\/overline(n)$;
- *two pairs coincide*: a two-point function times two shot-noise factors → the *pair* term, $prop P\/overline(n)^2$;
- *three or four coincide*: a $1\/overline(n)^3$ term, which the realised shot noise removes exactly (every self-pair is subtracted), so it does not appear;
- the *disconnected* pairings give the Gaussian term, whose shot-noise parts are the $S$ window of the companion note.

The weights are what matters for @sec:thinning. With weights, the shot-noise factor of a coincidence is $sum_g w_g^m$ for the appropriate $m$: $m = 2$ for the pair that becomes a single galaxy in the collapsed bispectrum and in the Gaussian $S$ window, and $m = 2$ twice for the pair term. Written with the measured quantities,

$
C^"disc" = underbrace(frac(1, cal(N)^2) sum_g w_g^2 thin cal(B)_g, "collapsed bispectrum, " B\/overline(n)) + underbrace(frac(1, cal(N)^2) sum_(g != h) w_g^2 w_h^2 thin cal(P)_(g h), "pair term, " P\/overline(n)^2),
$ <eq:disc-structure>

where $cal(B)_g$ and $cal(P)_(g h)$ are the bispectrum and power-spectrum kernels of the configuration (two shells, the shared galaxy's position); their explicit forms are not needed, because @sec:thinning measures the two sums directly.

#notebox[
  *Why one cannot stop here.* Tree-level perturbation theory gives $cal(B)_g$ and $cal(P)_(g h)$ as integrals of $P_"L"$ with kernels in $b_1, b_2, f$. For the holi LRG1 mocks that prediction is $1.2$--$1.6 times$ too small for $B\/overline(n)$ and for QSO $1.9 times$ too small for $P\/overline(n)^2$ (HANDOFF 9.4 corrected in the 2026-10-19 update), because the galaxies' own small-scale statistics (one-halo pairs, exclusion, the real galaxy bispectrum at $k approx 0.2$--$0.3$) are not tree level. The rule says to measure the sums instead.
]

= Discreteness terms from a thinning of the catalogue <sec:thinning>

== Thinning as a probe of the sampling terms

Keep each galaxy independently with probability $p$ and reweight the survivors by $1\/p$. The thinned catalogue is again a Poisson sampling of the *same* clustered field (a $p$-thinning of a Poisson process is a Poisson process with intensity $p overline(n)$), with the same window, the same $cal(N)$ (the expected sum of weights is unchanged) and the same long modes. Only the coincidence weights change: a coincidence of $m$ factors now carries $sum_g (w_g\/p)^m$ over the kept galaxies, whose expectation is $p^(1 - m) sum_g w_g^m$. Hence, writing $a equiv 1\/p - 1$,

$
S arrow.r S_p &= (1 + a) sum_g w_g^2 + alpha^2 sum_r w_r^2, \
B\/overline(n) arrow.r (1 + a) thin B\/overline(n), quad
P\/overline(n)^2 &arrow.r (1 + a)^2 thin P\/overline(n)^2, quad
"SSC", T_0 arrow.r "unchanged".
$ <eq:scaling>

The covariance of the thinned estimator over both the clustering and the thinning is therefore

$
C(p) = G(S_p) + C^"SSC" + C^(T_0) + (1 + a) B\/overline(n) + (1 + a)^2 P\/overline(n)^2 .
$ <eq:Cp>

$G(S_p)$ is thecov's Gaussian term with the shot-noise window scaled by $1 + a$ (`PowerSpectrum(shotnoise=S_p)`), i.e. known exactly. Everything else in @eq:Cp except the two discreteness terms is independent of $p$.

== The law of total covariance

Let $epsilon_g in {0, 1}$ be the Bernoulli variables and $eta_g equiv epsilon_g \/ p - 1$, so that $E eta = 0$, $E eta^2 = a$. The thinned field is $F_p = F_1 + D$ with

$
D(bold(k)) = sum_g w_g eta_g e_g (bold(k)),
$ <eq:D>

and the thinned estimator subtracts its own realised shot noise $hat(S)_p = sum_g (w_g epsilon_g \/ p)^2 + alpha^2 sum_r w_r^2$. Two facts about the expectation over the thinning at fixed catalogue:

$
E_eta |D|^2 = a sum_g w_g^2, quad E_eta hat(S)_p = sum_g frac(w_g^2, p) + alpha^2 sum_r w_r^2 = S_1 + a sum_g w_g^2,
$

so that $E_eta hat(P)_p = hat(P)_1$ exactly: thinning does not bias the estimator. The law of total covariance, $Cov[X] = E[Cov(X | "cat")] + Cov[E(X | "cat")]$ with $X = hat(P)_p$ and the conditioning on the unthinned catalogue, then reads $C(p) = E_"cat" [Cov_eta hat(P)_p] + C(1)$, and with @eq:Cp:

#resultbox[
  $
  E_"cat" [ Cov_eta [hat(P)_p (k), hat(P)_p (k')] ] = [G(S_p) - G(S_1)] + a thin B\/overline(n) + (2a + a^2) thin P\/overline(n)^2 .
  $ <eq:thin-expect>
  The thinning covariance of one catalogue, over the Bernoulli variables only, is the difference of two known Gaussian terms plus the two discreteness terms with known coefficients in $a = 1\/p - 1$. SSC and $T_0$ cancel. Two values of $a$ (or the exact $a$-polynomial below) separate $B\/overline(n)$ from $P\/overline(n)^2$.
]

The left-hand side is the quantity to compute; it needs no thinning at all, because the expectation over $eta$ can be done in closed form.

== The closed form over the Bernoulli variables

Expand $|F_p|^2 - hat(S)_p$ around the unthinned values. With $hat(S)_p - S_1 = sum_g w_g^2 [(eta_g + 1)\/p - 1] = sum_g w_g^2 (eta_g \/ p + a)$,

$
|F_p|^2 - hat(S)_p - (|F_1|^2 - S_1) = 2 Re[F_1^* D] + sum_g w_g^2 eta_g^2 + sum_(g != h) w_g w_h eta_g eta_h e_g e_h^* - sum_g w_g^2 (eta_g \/ p + a).
$ <eq:expand>

The diagonal of $|D|^2$ is linear in $eta$, because $epsilon_g^2 = epsilon_g$:

$
eta_g^2 = (epsilon_g \/ p - 1)^2 = epsilon_g frac(1 - 2p, p^2) + 1 = (eta_g + 1) frac(1 - 2p, p) + 1 quad arrow.r.double quad eta_g^2 - eta_g \/ p - a = -2 eta_g .
$ <eq:diag>

So the realised shot noise removes the $eta^2$ diagonal exactly and leaves $-2 sum_g w_g^2 eta_g$. Per shell, with $overline((thin dots.c thin))^((i))$ denoting the average over the $N_i$ modes of bin $i$,

$
hat(P)_p (k_i) - hat(P)_1 (k_i) = frac(1, cal(N)) [ sum_g eta_g overline(G)_g^((i)) + sum_(g != h) eta_g eta_h w_g w_h overline(Re[e_g e_h^*])^((i)) ], quad
overline(G)_g^((i)) equiv 2 w_g Re[ overline(F_1^* e_g)^((i)) ] - 2 w_g^2 .
$ <eq:linquad>

The covariance over $eta$ needs $E[eta_g eta_h] = a delta_(g h)$ and, for the quadratic piece, $E[eta_g eta_h eta_(g') eta_(h')]$ with $g != h$, $g' != h'$, which is $a^2$ when ${g, h} = {g', h'}$ and zero otherwise; the cross term between the linear and the quadratic piece needs $E[eta_g eta_(g') eta_(h')]$ with $g' != h'$, which vanishes (no index pattern survives). Hence

#resultbox[
  $
  Cov_eta [hat(P)_p (k_i), hat(P)_p (k_j)] = frac(1, cal(N)^2) { a sum_g overline(G)_g^((i)) overline(G)_g^((j)) + a^2 [ overline(|M_1|^2)^((i j)) + overline(|M_2|^2)^((i j)) - 2 sum_g w_g^4 ] },
  $ <eq:closed>
  $
  M_1 (bold(k), bold(k)') = sum_g w_g^2 e_g (bold(k)) e_g^* (bold(k)'), quad M_2 (bold(k), bold(k)') = sum_g w_g^2 e_g (bold(k)) e_g (bold(k)'),
  $
  with $overline(|M|^2)^((i j))$ the average over $bold(k) in i$, $bold(k)' in j$. Exact in $eta$ for a given catalogue: no Monte-Carlo noise, every $p$ at once.
]

Every piece is a sum over the galaxies of products of the data field $F_1$ and of the transform of the $w^2$-weighted catalogue: the linear term needs $F_1$ at all modes and one pass over the galaxies (chunked, $N_g times N_"modes"$); $|M_1|^2$ and $|M_2|^2$ are shell averages of the $w^2$-catalogue's power at the mode difference and sum, i.e. convolutions done with one FFT each. The cost is that of one window computation (about a node-hour per cap); `thinning_closed_form.py` does this with NUFFTs on the grid of the Monte-Carlo thinning it was checked against.

== Extracting the two terms

Comparing @eq:closed with @eq:thin-expect term by term in $a$:

$
underbrace(frac(a, cal(N)^2) sum_g overline(G)_g overline(G)_g, "lin" times a) = a [ g_1 + B\/overline(n) + 2 P\/overline(n)^2 ], quad
underbrace(frac(a^2, cal(N)^2) [ |M_1|^2 + |M_2|^2 - 2 sum w^4 ], "quad" times a^2) = a^2 [ g_2 + P\/overline(n)^2 ],
$ <eq:match>

where $G(S_p) - G(S_1) = a g_1 + a^2 g_2$ is the Gaussian difference, itself exactly quadratic in $a$ because $G$ is bilinear in the windows and only the $S$ window scales (its $P$--$S$ cross terms give $g_1$, its $S$--$S$ term gives $g_1$ and $g_2$). Hence

#resultbox[
  $
  P\/overline(n)^2 = "quad" - g_2, quad B\/overline(n) = "lin" - g_1 - 2 thin P\/overline(n)^2 ,
  $ <eq:extract>
  with $g_1, g_2$ from two thecov Gaussian runs (shot-noise windows at $S_1$ and $S_p$) and lin, quad from @eq:closed. The $-2 P\/overline(n)^2$ in $B\/overline(n)$ is the linear part of $(1 + a)^2$; omitting it overestimates $B\/overline(n)$ by $2 P\/overline(n)^2$ (this was the correction of 2026-10-19).
]

The extraction is linear algebra on matrices; both terms come out for all $(i, j)$ of the block at once.

== What the terms contain, and the one place they must not be used

The sums in @eq:closed are over the actual galaxies. The bispectrum of @eq:disc-structure is the galaxies' own three-point function at the configurations of the two shells, including the one-halo contribution and whatever the fibre assignment did to close pairs; the pair term is the galaxies' own two-point function at the mode difference, including exclusion and duplicates. None of this is modelled. For the QSO mocks the pair term equals the clustering expectation to $1$--$4 %$ (the pair diagnostics of 2026-10-19), which is itself a result about the mocks.

One configuration must be excluded from the measured terms: at $|i - j| <= 1$ the kernels $cal(P)_(g h)$ of @eq:disc-structure contain the modes with $|bold(k) - bold(k)'| approx 0$, a ridge whose weight scales as $1 \/ N_"modes"$ like a Gaussian term and which belongs to the bin-width-dependent part of $G$ (HANDOFF 9.4). The measured terms at $|i - j| <= 1$ are replaced by continuity from $|i - j| = 2, 3$ (`cont_diag` in `t0_tree_fit.py`); this is part of the estimator's definition.

== Multipoles <sec:thinning-rsd>

For $ell > 0$ the estimator weights one of the two points of each pair by $cal(L)_ell (hat(bold(k)) dot hat(bold(x)))$ with the local line of sight (companion note, estimator section). In @eq:F this means a second field per multipole, $F_ell (bold(k)) = sum_g w_g cal(L)_ell (hat(bold(k)) dot hat(bold(x))_g) e_g (bold(k)) - (dots)_r$, and $|F|^2$ becomes $Re[F_ell^* F_0]$; the thinning algebra of @eq:expand to @eq:closed goes through unchanged with $overline(G)_g^((i))$ built from both fields and $M_(1, 2)$ carrying the Legendre weights of one point. The closed form has so far been run for the $P_0$--$P_0$ block only; the other blocks are the first item of @sec:status, because until they exist the $ell = 2, 4$ discreteness blocks are the tree-level ones scaled by the $P_0$ off-diagonal ratio, a stopgap that breaks positive-definiteness when $P_4$ is included.

== Results on mock 173 (LRG1 and QSO, NGC)

In units of the Gaussian term at $k > 0.15$: LRG1 $B\/overline(n)$ $0.09$ on the diagonal ($1.2 times$ tree level) and $0.12$ off the diagonal ($1.6 times$), $P\/overline(n)^2$ $0.018$ ($1.2 times$); QSO $P\/overline(n)^2$ $0.039$ ($1.9 times$) and $B\/overline(n)$ small, consistent with ordinary clustering. The closed form reproduces the Monte-Carlo thinning on the same catalogue within the latter's noise. These numbers replace the "mocks want $2 times$ the discreteness terms" of the fitted analysis: they are the model's values, measured.

#todobox[
  *Cosmic variance of a single catalogue.* The closed form has no estimator noise, but it measures the realisation's own three- and four-point sums. Their scatter between catalogues is the irreducible uncertainty of this route. Measure it by running the closed form on 5--10 mocks (@sec:validation); the expectation is a few per cent of the term, i.e. below $1 %$ of $G$.
]

= Responses of the small-scale power to long modes, measured on the catalogue <sec:response>

Both the super-sample term and the squeezed limit of the trispectrum are *responses*: how $P(k)$ at a small scale changes when a long-wavelength mode $delta_L$ is present. This section defines the responses, shows what fixes them, and derives the estimator that measures the first-order one on the catalogue.

== Definition and the separate-universe decomposition <sec:su>

Inside a region where the long mode is $delta_L$ (per unit *matter* overdensity, averaged over the region), the small-scale galaxy power is

$
P_g (k | delta_L) = P_g (k) [ 1 + R_1 (k) delta_L + frac(1, 2) R_2 (k) delta_L^2 + dots ],
$ <eq:response-def>

with $R_n$ the $n$-th order response. A long mode of wavelength much larger than $1\/k$ acts on the small scales like a change of the background cosmology (the *separate universe*), through three effects, and $R_n$ is the chain rule of the three:

+ *Growth.* The region evolves as a universe of slightly different density; its small-scale fluctuations grow with $D_W (delta_L) = D [1 + frac(13, 21) delta_L + O(delta_L^2)]$ (the spherical-collapse expansion), so $P prop D_W^2$ gives $frac(26, 21) delta_L$ at first order.
+ *Reference density.* Small-scale contrasts are defined with respect to the global mean, but the region has mean $overline(n)_g (1 + b_1 delta_L + dots)$: the small-scale galaxy fluctuation measured globally is $(b_1 + b_2 delta_L) (1 + delta_L) delta_s^"loc"$ for a local bias expansion $delta_g = b_1 delta + frac(b_2, 2) delta^2$, so $P_g$ acquires $[1 + (1 + b_2 \/ b_1) delta_L]^2$: $2 + 2 b_2 \/ b_1$ at first order.
+ *Dilation.* Comoving coordinates inside the region are rescaled by $(1 + delta_L)^(-1\/3)$, so a wavenumber $k$ in global units is $k (1 + delta_L)^(1\/3)$ locally and the mode volume changes: $P(k) arrow.r (1 + delta_L)^(-1) P(k (1 + delta_L)^(1\/3))$, i.e. $-1 - frac(1, 3) dd ln P \/ dd ln k$ at first order.

#resultbox[
  First-order response of the galaxy power to a matter long mode, real space, local bias:
  $
  R_1 (k) = frac(26, 21) + 2 + frac(2 b_2, b_1) - 1 - frac(1, 3) frac(dd ln P_g, dd ln k) = frac(47, 21) + frac(2 b_2, b_1) - frac(1, 3) frac(dd ln P_g, dd ln k).
  $ <eq:R1>
  Per unit *galaxy* long mode $delta_(g, L) = b_1 delta_L$, which is what a galaxy survey measures: $R_1^((g)) = R_1 \/ b_1$. For LRG1 ($b_1 = 2.2$, $b_2 approx 0.3$, $dd ln P \/ dd ln k approx -1.5$ at $k approx 0.2$): $R_1^((g)) approx 1.4$; the mocks give $1.9$ (@sec:ibisp).
]

The same expansion to second order gives $R_2$ as the second derivative of the product of the three factors: the second-order separate-universe growth, the second-order reference-density factor (which brings in $b_3$ through the local bias expansion), and the second-order dilation ($dd^2 ln P \/ dd ln k^2$ and the square of the first derivative). The structure is fixed by the derivation above; the growth coefficients beyond tree level are the *growth-only responses* measured in separate-universe simulations (Wagner et al. 2015), which are cosmology, not sample properties. Redshift space adds $f$ and the orientation of the long mode to the line of sight (tidal and velocity terms); thecov's tree-level `responses()` carries these through the $Z_1, Z_2$ kernels. Two statements matter for the rule:

- the *dilation* parts of $R_1$ and $R_2$ are derivatives of the *measured* $P_g (k)$ (class 1), not of a model;
- the *growth* parts are cosmology (class 3) and the *bias* parts are the fitted $b_1, b_2$ (class 1), with $b_3$ the one poorly constrained input.

== The in-survey response: integrated bispectrum in Fourier space <sec:ibisp>

The first-order response can be measured on the catalogue, because the survey contains long modes: $delta_L$ varies across the volume, and the small-scale power measured locally follows it. The clean way to do this without sub-volumes is the cross-spectrum of the field with the field modulated by its own long modes.

Let $m(bold(x)) = alpha overline(n)_r (bold(x)) w(bold(x))$ be the window (the `NW` density of the randoms) and define the long-mode overdensity inside the survey,

$
delta_L (bold(x)) = frac([F ast W_L](bold(x)), m(bold(x))), quad W_L (bold(k)) = e^(-k^2 \/ 2 k_L^2) "or" Theta(k < k_L),
$ <eq:deltaL>

set to zero where $m = 0$. Consider the three-point statistic

$
C_ell (k) equiv frac(1, cal(N)) chevron.l F_ell^* (bold(k)) thin "FT"[delta_L F](bold(k)) chevron.r_"shell" .
$ <eq:Cl>

Inserting the Fourier representation of $delta_L F$, $"FT"[delta_L F](bold(k)) = integral frac(dd^3 q, (2pi)^3) delta_L (bold(q)) F(bold(k) - bold(q))$, the expectation is the bispectrum of the field at the configuration $(bold(k), -bold(k) + bold(q), -bold(q))$ integrated against $W_L (q)$: the *integrated bispectrum* of Chiang et al. (2014), with the filter in place of sub-volumes. For $q < k_L << k$ the bispectrum is in its squeezed limit, where by @eq:response-def

$
B(bold(k), -bold(k) + bold(q), -bold(q)) arrow.r R_1^((g)) (k) P_g (k) P_(g, L) (q) + O(q \/ k),
$ <eq:squeezed-B>

(the response is the squeezed bispectrum: $chevron.l delta_g (bold(q)) P_g (k | delta_L) chevron.r = R_1^((g)) P_g (k) chevron.l delta_g (bold(q)) delta_(g, L) chevron.r$). Carrying out the $q$ integral,

#resultbox[
  $
  C_ell (k) = R_ell^((g)) (k) thin P_ell (k) thin sigma_L^2, quad sigma_L^2 = frac(integral dd^3 x thin m^2 (bold(x)) delta_L^2 (bold(x)), integral dd^3 x thin m^2 (bold(x))) ,
  $ <eq:ibisp>
  so the measured response per unit galaxy long mode is $R_ell^((g)) (k) = C_ell (k) \/ [P_ell (k) sigma_L^2]$, with $sigma_L^2$ the variance of the long modes actually present in the survey ($18.8 %$ at $k_L = 0.03$, $11.6 %$ at $0.02$ for LRG1 NGC). Everything on the right is computed from the catalogue: two FFTs per $k_L$ once $F$ is on the mesh.
]

Three details of the estimator:

- *Shot noise.* The discrete three-point function has a term where the long-mode leg and one short leg share a galaxy: $frac(1, overline(n)) P$-like, the same collapsed term as @eq:disc-structure. It is removed exactly by replacing $F$ in the modulated leg by the $w^2$-weighted catalogue minus its random counterpart, $C_ell^"SN" = chevron.l F_ell^* thin "FT"[delta_L S] chevron.r$ with $S = sum_g w_g^2 delta_D - alpha^2 sum_r w_r^2 delta_D$, and subtracting it.
- *Multipoles.* With the first-point line of sight on both legs, $C_ell$ for $ell = 0, 2$ gives the response of $P_0$ and $P_2$. For an isotropic field the $ell = 2$ cross-spectrum is dipole-free and measures the quadrupole response directly.
- *Validation.* On lognormal fields with a known response ($R = 2$ at every $q$, Poisson-sampled on the LRG1 footprint) the estimator recovers $2$ to $5 %$.

#figure(
  image("report_figures/response_kL_LRG1_NGC.png", width: 100%),
  caption: [Responses of $P_0$ and $P_2$ to the in-survey long modes, LRG1 NGC, six holi mocks, as a function of the filter scale $k_L$. (a) The monopole response is flat at $R_0 = 1.9$ for $k_L >= 0.015$, $1.2$--$1.5 times$ tree level. (b) The quadrupole response rises from $2.8$ to $3.9$ as $k_L$ drops below $2pi \/ ("shell thickness")$: at small $k_L$ the in-survey long modes are transverse, and the quadrupole responds to the orientation of the long mode. (c) The net response of $hat(P)_0$ to the survey-mean mode, $R_0 - 2$, from this estimator and from the regression of 848 mocks on their shell density: a near cancellation.],
) <fig:response>

Per catalogue the statistical error is $2$--$3 %$ on $R_0$ and $5$--$10 %$ on $R_2$ (the scatter among the six mocks); the limit is the cosmic variance of the long modes in one survey, not the estimator.

== The survey-mean mode and the local average <sec:la>

The one long mode the estimator cannot see is the survey-mean overdensity $delta_u$: $delta_L$ of @eq:deltaL is defined relative to the realised mean of the catalogue (through $alpha$), so a uniform mode is invisible in-survey. This is also the mode to which the estimator's normalisation reacts. Take the pypower / jaxpower convention, $cal(N) = alpha sum_c n_D (c) n_R (c)$ over mesh cells: the data mesh and $alpha$ each carry one power of $(1 + delta_(g, u))$, so $cal(N) prop (1 + delta_(g, u))^2$, while $|F|^2$ responds as $1 + R^((g)) delta_(g, u)$. Hence

#resultbox[
  $
  frac(dd ln hat(P)_ell, dd delta_(g, u)) = R_ell^((g)) (q arrow.r 0) - 2 ,
  $ <eq:la>
  the *local-average* response (for a normalisation fixed by the randoms alone, $cal(N) prop 1 + delta_(g, u)$ and the $2$ becomes $1$). It is not a separate measurement: it is the in-survey response at $k_L arrow.r 0$ minus the known reaction of the normalisation. The identity was checked two ways on the LRG1 mocks (@fig:response c, HANDOFF update 2026-10-15 (d)): the regression of 848 mocks' $hat(P)_0$ on their realised shell density gives $-0.09$ to $-0.19$ per unit galaxy mode across the $k$ bands, equal within $1 sigma$ to $R_0 - 2$ from @eq:ibisp; and a synthetic lognormal catalogue with an imposed uniform mode gives $dd ln hat(P)_0 \/ dd delta_u = 0.00 plus.minus 0.02$ with data-matched randoms (expected $2 - 2$) and $1.00 plus.minus 0.02$ with fixed randoms (expected $2 - 1$).
]

For $P_0$ the two terms nearly cancel ($1.9 - 2$): the survey-mean mode barely changes the measured monopole, and the sign of the small remainder is set by whether the measured $R_0$ is above or below 2 --- tree level ($R_0 approx 1.4$) gets the sign wrong. For $P_2$ the response is $3.4$--$4$ and nothing cancels.

== The second-order response <sec:r2>

The same construction at second order, $J_ell (k) prop chevron.l F_ell^* "FT"[delta_L^2 F] chevron.r$, measures $R_2$ in principle. In practice the signal is $frac(1, 2) R_2 chevron.l delta_L^4 chevron.r_c$-like, with $sigma_L approx 0.1$--$0.2$: a $10^(-3)$ modulation of a spectrum measured to $10^(-2)$ per bin. It cannot be measured on one survey; averaged over many mocks it is a check of the formalism's $R_2$ (@sec:validation), not an input. $R_2$ therefore comes from the separate-universe decomposition of @sec:su, where its only non-measured ingredients are the growth-only coefficients (cosmology) and $b_3$.

= The super-sample covariance <sec:ssc>

== The term

A long mode changes the small-scale power everywhere in the survey by @eq:response-def; the long modes vary from realisation to realisation; their variance makes all small-scale bins fluctuate together:

$
C^"SSC"_(ell ell') (k, k') = R_ell (k) R_(ell') (k') thin P_ell (k) P_(ell') (k') thin sigma^2_W, quad
sigma^2_W = integral frac(dd^3 q, (2pi)^3) P_"L" (q) frac(|W(bold(q))|^2, I^2),
$ <eq:ssc>

for an isotropic response and $W(bold(q))$ the Fourier transform of the clustering window ($I = W(0)$). In redshift space the response depends on the orientation of the long mode to the line of sight and the variance splits into components $sigma^2_(n n')$ weighted by the long-mode projections; thecov computes these from the window pair counts with the same tripolar machinery as the Gaussian term (`SuperSampleCovariance`, the $cal(Q)_(2 2 Lambda)$ functions of the companion note). Two features of the implementation follow from the rule:

- *Beat coupling and local average.* The in-survey long modes enter weighted by the window (beat coupling, response $R$); the survey-mean mode enters with the net response of @eq:la. For the data--randoms normalisation the latter is $R - 2$ per unit galaxy mode.
- *Poisson part.* The realised $alpha$ and $cal(N)$ also fluctuate with the finite number of galaxies; this gives the $"Var"_P (epsilon_N)$ and $T_ell$ terms of `ssc.py`, integrals over the randoms and data weights, with the collapsed-bispectrum piece in $b_2 + frac(2, 3) b_(s^2)$.

== What the rule requires

Each ingredient of @eq:ssc has a class:

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + luma(150), inset: 6pt,
  [*ingredient*], [*source*], [*class*],
  [$sigma^2_W$, radial and transverse components], [$P_"L"$ and the randoms' pair counts; the radial part uses the shell's $n(z)$, which is the data's], [2 + 3],
  [$R_ell (k)$ at $q arrow.r 0$], [@eq:ibisp on the catalogue, several $k_L$, extrapolated to small $k_L$; for $P_2$ the transverse limit ($3.4$--$4$ for LRG1), since the super-survey mode of a thin shell is transverse], [1],
  [local-average response], [$R_ell - 2$ from the same measurement, @eq:la], [1],
  [Poisson part], [weights of the randoms and the data, $b_2$ from the fit], [1 + 2],
)

#todobox[
  *Implementation.* `SuperSampleCovariance(responses=...)`: accept $R_ell (k)$ as measured arrays per multipole and $k$ band in place of the tree-level $a P + c thin dd P \/ dd ln k$ (`responses()`), which stays as the default for checks; and weight $sigma^2$ anisotropically (radial from $P_"L" |W|^2$ with the shell's $n(z)$, transverse from the same window at $q_parallel = 0$). SSC is subdominant above $k approx 0.1$ for these samples, so the $3 %$ error on $R$ is harmless; the term matters for the first bins and the $P_0$--$P_2$ coupling.
]

= The connected trispectrum $T_0$ <sec:t0>

== Tree level and what the mocks say about it

At tree level the connected four-point function of the galaxy field is the sum of the $T_(2211)$ ("snake", two $Z_2$ kernels) and $T_(3111)$ ("star", one $Z_3$ kernel) diagrams; for the covariance configuration $(bold(k)_1, -bold(k)_1, bold(k)_2, -bold(k)_2)$ it enters as

$
C^(T_0)_(ell_1 ell_2) (k_1, k_2) &= J_4 thin chevron.l (2 ell_1 + 1)(2 ell_2 + 1) cal(L)_(ell_1) (mu_1) cal(L)_(ell_2) (mu_2) thin T(bold(k)_1, -bold(k)_1, bold(k)_2, -bold(k)_2) chevron.r_(hat(bold(k))_1, hat(bold(k))_2, "shells"), \
J_4 &= frac(integral W^4, cal(N)^2),
$ <eq:t0>

with $J_4$ the window factor ($1\/V$ for a uniform window) and the term $bold(k)_1 + bold(k)_3 = 0$ excluded (it is the SSC). thecov's `TrispectrumCovariance` computes this with the exact redshift-space $Z_1 dots Z_3$ and the Galileon bias basis, validated against PowerSpecCovFFT and an independent quadrature.

Against the holi LRG1 mocks, with the measured discreteness terms of @sec:thinning in the base model (HANDOFF 12; @fig:t0): the likelihood prefers $0.38 plus.minus 0.05$ of the tree-level $T_0$ for $P_0$, $0.61 plus.minus 0.05$ for $P_2$, $0.43 plus.minus 0.03$ for both; the matrix is positive definite at amplitude 1 (the earlier failure used a $b_3$ in the wrong bias basis); and the excess is in the *squeezed* coupling between the first bins and $k > 0.2$, where tree level predicts twice the mocks' correlation, while the hard--hard part is roughly right for $P_0$ and low for $P_2$. A fitted amplitude is excluded by the rule, so the two regimes are treated separately.

#figure(
  image("report_figures/t0_tree_reanalysis.png", width: 92%),
  caption: [Tree-level $T_0$ (snake + star, $b_3 = 0$) against the 848 LRG1 NGC mocks, base model $G + P\/overline(n)^2 + B\/overline(n)$ with the measured discreteness terms. (a) Wishart likelihood against the amplitude of the template; the dashed curve is an earlier template that summed the tree with its response-approach rebuild and gave a spurious $0.1$. (b) Mock/model variance along the leading non-Gaussian direction. (c) Mean correlation between bins at $k = 0.02$--$0.05$ and $0.2$--$0.3$ (squeezed) and between bins both at $k = 0.15$--$0.3$ with $|i - j| >= 4$ (hard). (d) Separate amplitudes for the two regimes.],
) <fig:t0>

== The squeezed block from the responses

For $k_1 << k_2$ the two soft legs are a long mode of wavenumber $q = k_1$ seen by the hard power at $k = k_2$. Expanding the hard power to second order in the long mode as in @eq:response-def, now keeping the direction of $bold(q)$,

$
chevron.l delta(bold(q)) delta(-bold(q)) delta(bold(k)) delta(-bold(k)) chevron.r_c
= chevron.l delta(bold(q)) delta(-bold(q)) [P(k | delta_L) - P(k)] chevron.r_c
arrow.r cal(R)_2 (k; hat(bold(q)) dot hat(bold(k))) thin P_"L" (q)^2 thin P(k) ,
$ <eq:t0-squeezed>

where the term linear in $delta_L$ averages to the bispectrum $B(q, -q, 0)$, i.e. the super-sample term already counted, and the quadratic term pairs the two long modes with the two factors of $delta_L$ (Gaussian long modes), leaving the coefficient $cal(R)_2$ of $delta_L (bold(q)) delta_L (-bold(q))$ in the hard power. In the tree-level diagrams this is the combination of the star ($Z_3$ with two soft legs) and of the part of the snake where the soft momentum is transferred through the hard propagator; their $(k\/q)^2$ infrared pieces cancel against each other (HANDOFF 7.1 checked this numerically) and what remains is the second-order response, angle-dependent through the tidal and velocity terms. In the separate-universe decomposition of @sec:su, $cal(R)_2$ is the second derivative of growth $times$ reference density $times$ dilation, with the measured $P_g$ in the dilation terms and the fitted biases in the density terms.

#resultbox[
  Squeezed block of $T_0$ from the measured spectrum:
  $
  C^(T_0, "sq")_(ell_1 ell_2) (k_1, k_2) = J_4 thin chevron.l cal(L)_(ell_1) cal(L)_(ell_2) thin cal(R)_2^((g)) (k_2; mu_(1 2), mu_2) chevron.r_"angles" thin P_(g, "L") (k_1)^2 thin P_g (k_2) quad (k_1 < k_"split" <= k_2) ,
  $ <eq:t0-sq>
  with $cal(R)_2^((g))$ assembled from (i) the growth-only second-order response (cosmology), (ii) $dd ln P_g \/ dd ln k$ and $dd^2 ln P_g \/ dd ln k^2$ of the *measured* multipoles, (iii) $b_1, b_2$ from the fit, $b_3$ from the fit or a $b_3 (b_1)$ relation, and $f$. Nothing is fitted to a covariance. The first-order part of the same decomposition must reproduce the measured $R_1^((g))$ of @sec:ibisp on the same catalogue; a discrepancy there is carried as the systematic of $cal(R)_2$.
]

#todobox[
  *Implementation.* A module `thecov/tree/squeezed.py` with $cal(R)_2^((g)) (k; mu)$ in the separate-universe form (isotropic growth and dilation parts; tidal and redshift-space parts from the squeezed limit of the $Z_2, Z_3$ kernels, which thecov already has), the angle averages of @eq:t0-sq on the shells, and the assembly for $k_1 < k_"split"$. The explicit second-order coefficients are to be derived from the expansion of @sec:su and cross-checked against the squeezed limit of thecov's tree $T_0$ with $P_g = b_1^2 P_"L"$, where the two must agree identically.
]

== The hard block

For $k_1, k_2 >= k_"split"$ there is no response shortcut: the trispectrum is needed in full. The inputs allowed by the rule are the tree-level kernels with the *measured* nonlinear multipoles in place of $b_1^2 P_"L"$ inside the diagrams (the `Dressed` power already used for the SSC responses: IR-damped, with the data's $sigma_v$ in a Gaussian fingers-of-God factor on the external legs), $b_1, b_2, f$ from the fit, $b_3 = 0$ or a $b_3 (b_1)$ relation. Whether this moves the hard $P_0$ amplitude of @fig:t0 ($0.35$--$0.5$ against the mocks) towards 1 is to be tested; the hard block and $B\/overline(n)$ are both smooth at high $k$ and partly degenerate, so the test must hold $B\/overline(n)$ at its measured value. If the hard block stays low, that is a limitation of the tree level to be stated and bounded with the mocks, not tuned away.

= Validation protocol <sec:validation>

The mocks enter only here.

+ *One catalogue as the data.* Build $G$ (kernel window, the catalogue's own smoothed $hat(P)_ell$), $B\/overline(n)$ and $P\/overline(n)^2$ (closed form, all blocks), SSC (measured $R_ell$, anisotropic $sigma^2$), $T_0$ (squeezed from $cal(R)_2^((g))$, hard from the dressed tree) from mock 173 alone, LRG1 and QSO, NGC and SGC, with the pipeline that will run on the data.
+ *Compare with the ensemble covariance* of 848 / 859 mocks: Wishart deviance per multipole set and $k_max$, diagonal ratios per block and $k$ band, the mock/model variance along the leading non-Gaussian direction, the $G$-whitened eigenvalue spectrum against the Marchenko--Pastur edge, the squeezed and hard mean correlations of @fig:t0 c. The targets are the residuals of HANDOFF 7.9 and 9.4 ($chi^2 \/ n = 1.02$--$1.03$ for LRG1 today).
+ *Realisation scatter.* Repeat step 1 on mocks 174--178. The spread of the single-catalogue covariances, per term and propagated to the parameter variances, is the uncertainty to quote for the data covariance. Only the closed-form sums and the responses are noisy; the window pieces are not.
+ *Second-order response.* On the six mocks, compare $cal(R)_2^((g))$ of @sec:su with the mock-averaged $J_ell$ estimator of @sec:r2.
+ *A different galaxy physics.* The 25 AbacusSummit complete mocks (HANDOFF 7.7) as an independent ensemble for the same single-catalogue construction; their SSC differs (periodic boxes) and is left out of that comparison.
+ *Data.* After the blinding sign-off, the same pipeline on the DR2 catalogues. The data's own $B\/overline(n)$, $P\/overline(n)^2$ and $R_ell$ compared with the mocks' values is a test of the mocks' small-scale galaxy physics that the two-point function does not provide.

= Noise and feedback of data-derived inputs

- *$hat(P)_ell$ in $G$ and in the dilation responses.* The measured multipoles carry $approx 0.5 %$ noise per bin ($P_0$; more for $P_2, P_4$), entering $G$ quadratically and the responses through derivatives. Smoothing over a few bins (the cubic spline of `PowerSpectrum`, or a smoothing spline) removes most of it. Quantify on the mocks: $G$ from each mock's own $hat(P)_ell$ against $G$ from the mocks' mean; the windows are shared, so hundreds of cases are cheap.
- *Correlation between the covariance and the data.* The closed-form sums and the responses are statistics of the same realisation whose covariance they describe. The induced error on parameter uncertainties is of order the fractional cosmic variance of those statistics times the term's weight in $C$ (a few per cent of a $10$--$20 %$ term), i.e. below a per cent on parameter errors; step 3 of @sec:validation bounds it directly.
- *$b_2$ and $b_3$.* They enter the reference-density parts of $R_1$ and $cal(R)_2$ and the collapsed bispectrum of the local-average Poisson term. Use the fit's values with their errors propagated, or the Lazeyras et al. relations as priors; the sensitivity is small for SSC and is to be quantified for the squeezed $T_0$.

= Status and work list <sec:status>

#table(
  columns: (1fr, auto, auto),
  stroke: 0.5pt + luma(150), inset: 6pt,
  [*step*], [*status*], [*cost per tracer and cap*],
  [Thinning closed form, $P_0$--$P_0$ block (@sec:thinning)], [done, validated against Monte Carlo], [$approx 1$ node-hour],
  [Closed form for the $ell = 2, 4$ and cross blocks (@sec:thinning-rsd)], [to do], [same],
  [Response estimator $R_ell (k; k_L)$ (@sec:ibisp)], [done, validated; six mocks measured], [$approx 0.5$ node-hour],
  [SSC with measured responses and anisotropic $sigma^2$ (@sec:ssc)], [to do], [minutes],
  [Squeezed $T_0$ from $cal(R)_2^((g))$ of the measured $P_ell$ (@sec:t0)], [to do; coefficients to derive and cross-check], [minutes],
  [Hard $T_0$ with the dressed measured power], [to do (option of `TrispectrumCovariance`)], [$approx 10$ min, 64 processes],
  [Single-catalogue build on mock 173, ensemble comparison], [to do (`t0_tree_fit.py` has the comparison code)], [hours],
  [Realisation scatter, mocks 174--178], [to do], [$approx 10$ node-hours],
  [Abacus complete cross-check], [prepared (`run_abacus_t0_test.sh`)], [$approx 5$ node-hours],
)

Order: the $ell = 2, 4$ closed form first (it removes the stopgap that breaks $P_0 + P_2 + P_4$), then SSC with measured responses, then the squeezed $T_0$, then the hard-block test. Each step is validated against the ensemble before the next.

#v(6pt)
#line(length: 100%, stroke: 0.4pt + luma(150))
#text(size: 9pt)[
  *References.* Chiang, Wagner, Schmidt, Komatsu 2014 (position-dependent power spectrum / integrated bispectrum); Wagner, Schmidt, Chiang, Komatsu 2015 (separate-universe responses to second order); Barreira & Schmidt 2017a,b (response formalism, squeezed covariance); Wadekar & Scoccimarro 2020 (discreteness terms of the windowed covariance); Kobayashi et al. 2020 (PowerSpecCovFFT, tree-level $T_0$ decomposition); thecov `desi_validation/HANDOFF.md` and `report/` for the mock results quoted.
]
