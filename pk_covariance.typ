// Gaussian covariance of windowed power-spectrum multipoles
// Separation-space (tripolar) formulation, written for numerical implementation.

#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

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

#align(center)[
  #text(size: 16pt, weight: "bold")[Gaussian covariance of windowed power-spectrum multipoles] \
  #v(2pt)
  #text(size: 12pt)[A separation-space formulation for numerical implementation]
]
#v(8pt)

= Purpose and result at a glance

This note derives the Gaussian (disconnected) part of the covariance of the FKP--Yamamoto power-spectrum multipole estimator, with the survey window included, for one or several tracers. The derivation is organised so that every object that appears in the final formula is either (a) a one-dimensional integral of the model power spectrum, (b) a one-dimensional function of the survey geometry obtainable from pair counts of the random catalogue, or (c) a pure number built from Wigner symbols. The final expression has the form

$
C^(A B C D)_(ell_1 ell_2)(i, j) = frac(1, I_(A B) I_(C D)) sum_"terms" c times integral s^2 dif s space overline(p)^((i))(s) space overline(p)^((j))(s) space cal(Q)(s),
$ <eq:glance>

where $i, j$ label $k$-bins and

- $overline(p)^((i))_(L lambda)(s)$ is the model power-spectrum multipole $P_L (k)$ averaged over bin $i$ against a spherical Bessel function $j_lambda (k s)$ (cosmology enters only here);
- $cal(Q)_(Lambda_1 Lambda_2 Lambda)(s)$ are _tripolar window functions_: multipoles of the window pair-correlation with line-of-sight weights at both points (geometry enters only here, computed once per survey);
- $c$ are numerical coupling coefficients (Gaunt coefficients and a $3j$ symbol).

All harmonic sums are finite and truncate exactly; the $k$-bin average is exact; no fine $k$-grid is needed. The only physical approximations are:

+ Gaussianity (the connected four-point and super-sample terms are not included);
+ the local plane-parallel approximation: each pair of correlated points shares the line of sight of one of them;
+ the window varies slowly over a correlation length, so that $W(bold(x)') approx W(bold(x))$ inside a correlator. This is the leading order in $1 \/ (k R)$ with $R$ the scale over which the selection function varies.

#notebox[
  *Reading guide.* Sections 2--3 fix conventions and definitions. Section 4 is the derivation. Section 5 states the final result and the selection rules. Section 6 explains how to compute each ingredient and assemble the matrix. Section 7 lists checks. Appendix A justifies the reduction to scalar window functions.
]

= Conventions

#defbox[Fourier transform][
  $ tilde(f)(bold(k)) = integral dif^3 bold(x) space e^(-i bold(k) dot bold(x)) f(bold(x)), quad
    f(bold(x)) = integral frac(dif^3 bold(k), (2 pi)^3) e^(i bold(k) dot bold(x)) tilde(f)(bold(k)). $
]

#defbox[Spherical harmonics][
  Orthonormal complex harmonics $Y_(ell m)(hat(bold(n)))$ with $integral dif Omega space Y_(ell m) overline(Y)_(ell' m') = delta_(ell ell') delta_(m m')$, conjugation $overline(Y)_(ell m) = (-1)^m Y_(ell, -m)$, parity $Y_(ell m)(-hat(bold(n))) = (-1)^ell Y_(ell m)(hat(bold(n)))$, and $Y_(00) = (4 pi)^(-1\/2)$.
]

#defbox[Addition theorem][
  $ cal(L)_ell (hat(bold(a)) dot hat(bold(b))) = frac(4 pi, 2 ell + 1) sum_(m = -ell)^(ell) Y_(ell m)(hat(bold(a))) overline(Y)_(ell m)(hat(bold(b))). $ <eq:addition>
  Because $cal(L)_ell$ is real, the conjugate may be placed on either factor.
]

#defbox[Plane wave][
  $ e^(i bold(k) dot bold(s)) = 4 pi sum_(lambda = 0)^(infinity) sum_(nu = -lambda)^(lambda) i^lambda space j_lambda (k s) space overline(Y)_(lambda nu)(hat(bold(k))) Y_(lambda nu)(hat(bold(s))), $ <eq:planewave>
  and $e^(-i bold(k) dot bold(s))$ is obtained by $i^lambda -> (-i)^lambda$ (equivalently $hat(bold(s)) -> -hat(bold(s))$).
]

#defbox[Gaunt coefficients and the merge rule][
  For $n >= 2$ define the $n$-harmonic integral
  $ cal(G)^(m_1 dots m_n)_(ell_1 dots ell_n) equiv integral dif Omega space product_(a = 1)^n Y_(ell_a m_a)(hat(bold(n))). $ <eq:gaunt>
  For $n = 3$ this is the ordinary (real) Gaunt coefficient,
  $ cal(G)^(m_1 m_2 m_3)_(ell_1 ell_2 ell_3) = sqrt(frac((2 ell_1 + 1)(2 ell_2 + 1)(2 ell_3 + 1), 4 pi))
    mat(delim: "(", ell_1, ell_2, ell_3; 0, 0, 0) mat(delim: "(", ell_1, ell_2, ell_3; m_1, m_2, m_3), $
  non-zero only if $m_1 + m_2 + m_3 = 0$, $ell_1 + ell_2 + ell_3$ is even and the triangle inequality holds. For $n = 2$, $cal(G)^(m_1 m_2)_(ell_1 ell_2) = (-1)^(m_1) delta_(ell_1 ell_2) delta_(m_1, -m_2)$. Higher $n$ follow by recursion,
  $ cal(G)^(m_1 dots m_n mu)_(ell_1 dots ell_n Lambda) = sum_(Lambda' mu') (-1)^(mu') space cal(G)^(m_1 dots m_(n-1) mu')_(ell_1 dots ell_(n-1) Lambda') space cal(G)^(-mu', m_n, mu)_(Lambda', ell_n, Lambda). $ <eq:gaunt-rec>
  The _merge rule_ expresses a product of harmonics at one direction as a single harmonic:
  $ product_(a = 1)^n Y_(ell_a m_a)(hat(bold(n))) = sum_(Lambda mu) cal(G)^(m_1 dots m_n mu)_(ell_1 dots ell_n Lambda) space overline(Y)_(Lambda mu)(hat(bold(n))). $ <eq:merge>
]

#defbox[Triangle sets][
  $"tri"(a, b) = { |a - b|, |a - b| + 2, dots, a + b }$ (same-parity triangle set); $"tri"(a, b, c)$ denotes the set of $Lambda$ for which $cal(G)^((4))_(a b c Lambda)$ can be non-zero, i.e. $Lambda <= a + b + c$ with $a + b + c + Lambda$ even and some intermediate coupling allowed.
]

Throughout, all multipole indices ($ell, L, lambda, Lambda$) are even. This holds for the estimator multipoles considered here and for the power-spectrum multipoles of a single line of sight; odd (imaginary) cross-spectrum multipoles are not included.

= Fields, windows and the estimator

== Weighted density fields and windows

For tracer $A$ with mean density $overline(n)_A (bold(x))$, FKP-type weights $w_A (bold(x))$, and a random catalogue with density $overline(n)_A \/ alpha_A$, the weighted density contrast is $delta_A (bold(x)) = w_A (bold(x))[n_A (bold(x)) - alpha_A n_A^"ran"(bold(x))]$. We need two kinds of window:

$
W^(A B)(bold(x)) &equiv overline(n)_A (bold(x)) overline(n)_B (bold(x)) w_A (bold(x)) w_B (bold(x)) & "(clustering window)," \
S^A (bold(x)) &equiv (1 + alpha_A) space overline(n)_A (bold(x)) w_A (bold(x))^2 & "(shot-noise window)," 
$ <eq:windows>

and the normalisation $I_(A B) equiv integral dif^3 bold(x) space W^(A B)(bold(x))$.

== Two-point function in the local plane-parallel approximation

For two points $bold(x)$ and $bold(x)' = bold(x) - bold(r)$ within a correlation length, with the line of sight taken along $hat(bold(x))$,

$
chevron.l delta_A (bold(x)) delta_B (bold(x) - bold(r)) chevron.r approx W^(A B)(bold(x)) space xi^(A B)(bold(r); hat(bold(x))) + delta^K_(A B) space S^A (bold(x)) space delta_D (bold(r)),
$ <eq:twopoint>

where $W^(A B)(bold(x)')$ has been replaced by $W^(A B)(bold(x))$ (approximation (iii)). Its Fourier transform in $bold(r)$ defines the multipoles $P_L^(A B)(k)$:

$
integral dif^3 bold(r) space e^(-i bold(k) dot bold(r)) chevron.l delta_A (bold(x)) delta_B (bold(x) - bold(r)) chevron.r &approx sum_L cal(L)_L (hat(bold(k)) dot hat(bold(x))) space Pi^(A B)_L (k; bold(x)), \
Pi^(A B)_L (k; bold(x)) &equiv W^(A B)(bold(x)) P^(A B)_L (k) + delta^K_(A B) delta^K_(L 0) S^A (bold(x)).
$ <eq:Pi>

It is convenient to regard $Pi^(A B)_L$ as a sum over _spectrum--window pairs_ $(omega, p)$:

$
Pi^(A B)_L (k; bold(x)) = sum_((omega, p) in cal(P)^(A B)) omega(bold(x)) space p_L (k),
quad
cal(P)^(A B) = { (W^(A B), P^(A B)_L) } union { (S^A, delta_(L 0)) "if" A = B }.
$ <eq:pairs>

Everything below is linear in these pairs, so shot noise is handled by simply extending the sum. For even multipoles $P^(A B)_L = P^(B A)_L$, so the order of the tracer labels in a pair is immaterial.

== Estimator

The multipole field and the binned estimator are

$
F^A_ell (bold(k)) &= integral dif^3 bold(x) space e^(-i bold(k) dot bold(x)) cal(L)_ell (hat(bold(k)) dot hat(bold(x))) space delta_A (bold(x)), \
hat(P)^(A B)_ell (k_i) &= frac(2 ell + 1, I_(A B)) frac(1, V_i) integral_(bold(k) in i) frac(dif^3 bold(k), (2 pi)^3) space F^A_ell (bold(k)) F^B_0(-bold(k)),
quad
V_i equiv integral_(bold(k) in i) frac(dif^3 bold(k), (2 pi)^3) = frac(1, 2 pi^2) integral_i k^2 dif k.
$ <eq:estimator>

Here "$bold(k) in i$" means $k$ in the $i$-th radial bin, with the full sphere of directions. Note the line of sight is that of the field carrying the Legendre weight. With @eq:Pi one checks $chevron.l hat(P)^(A B)_ell (k_i) chevron.r = overline(P^(A B)_ell)^((i))$ at this order, which fixes the normalisation.

= Derivation

== Wick expansion

Write $a = F^A_(ell_1)(bold(k)_1)$, $b = F^B_0(-bold(k)_1)$, $c = F^C_(ell_2)(bold(k)_2)$, $d = F^D_0(-bold(k)_2)$. The Gaussian part of $chevron.l a b c d chevron.r - chevron.l a b chevron.r chevron.l c d chevron.r$ is $chevron.l a d chevron.r chevron.l b c chevron.r + chevron.l a c chevron.r chevron.l b d chevron.r$. Hence

$
C^(A B C D)_(ell_1 ell_2)(i, j) = frac((2 ell_1 + 1)(2 ell_2 + 1), I_(A B) I_(C D))
frac(1, V_i V_j) integral_(bold(k)_1 in i) integral_(bold(k)_2 in j) frac(dif^3 bold(k)_1 dif^3 bold(k)_2, (2 pi)^6)
[ T_1(bold(k)_1, bold(k)_2) + T_2(bold(k)_1, bold(k)_2) ],
$ <eq:cov-wick>

$
T_1 &= chevron.l F^A_(ell_1)(bold(k)_1) F^D_0(-bold(k)_2) chevron.r space chevron.l F^C_(ell_2)(bold(k)_2) F^B_0(-bold(k)_1) chevron.r, \
T_2 &= chevron.l F^A_(ell_1)(bold(k)_1) F^C_(ell_2)(-bold(k)_2) chevron.r space chevron.l F^D_0(bold(k)_2) F^B_0(-bold(k)_1) chevron.r,
$ <eq:T12>

where in $T_2$ the substitution $bold(k)_2 -> -bold(k)_2$ has been made (the shell is symmetric and $ell_2$ is even). There is no extra factor of 2: the two Wick pairings are written out explicitly.

== Each two-point function becomes a single plane wave in a separation

Take the first factor of $T_1$. Insert the definition @eq:estimator, set $bold(x)' = bold(x) - bold(r)$ for the position of $delta_D$, and use @eq:Pi:

$
chevron.l F^A_(ell_1)(bold(k)_1) F^D_0(-bold(k)_2) chevron.r
&= integral dif^3 bold(x) space dif^3 bold(r) space e^(-i bold(k)_1 dot bold(x) + i bold(k)_2 dot (bold(x) - bold(r))) cal(L)_(ell_1)(hat(bold(k))_1 dot hat(bold(x))) chevron.l delta_A (bold(x)) delta_D (bold(x) - bold(r)) chevron.r \
&approx sum_(L_2) integral dif^3 bold(x) space e^(-i (bold(k)_1 - bold(k)_2) dot bold(x)) space cal(L)_(ell_1)(hat(bold(k))_1 dot hat(bold(x))) space cal(L)_(L_2)(hat(bold(k))_2 dot hat(bold(x))) space Pi^(A D)_(L_2)(k_2; bold(x)).
$ <eq:factor1>

Identically, with $bold(x)'$ the position of $delta_C$ and line of sight $hat(bold(x))'$,

$
chevron.l F^C_(ell_2)(bold(k)_2) F^B_0(-bold(k)_1) chevron.r approx sum_(L_1) integral dif^3 bold(x)' space e^(-i (bold(k)_2 - bold(k)_1) dot bold(x)') space cal(L)_(ell_2)(hat(bold(k))_2 dot hat(bold(x))') space cal(L)_(L_1)(hat(bold(k))_1 dot hat(bold(x))') space Pi^(C B)_(L_1)(k_1; bold(x)').
$ <eq:factor2>

For $T_2$ the same steps give, with the additional local approximation $cal(L)_(ell_2)(hat(bold(k))_2 dot hat(bold(x))') -> cal(L)_(ell_2)(hat(bold(k))_2 dot hat(bold(x)))$ inside the first correlator (both points share the line of sight $hat(bold(x))$),

$
chevron.l F^A_(ell_1)(bold(k)_1) F^C_(ell_2)(-bold(k)_2) chevron.r &approx sum_(L_2) integral dif^3 bold(x) space e^(-i (bold(k)_1 - bold(k)_2) dot bold(x)) space cal(L)_(ell_1)(hat(bold(k))_1 dot hat(bold(x))) cal(L)_(ell_2)(hat(bold(k))_2 dot hat(bold(x))) cal(L)_(L_2)(hat(bold(k))_2 dot hat(bold(x))) space Pi^(A C)_(L_2)(k_2; bold(x)), \
chevron.l F^D_0(bold(k)_2) F^B_0(-bold(k)_1) chevron.r &approx sum_(L_1) integral dif^3 bold(x)' space e^(-i (bold(k)_2 - bold(k)_1) dot bold(x)') space cal(L)_(L_1)(hat(bold(k))_1 dot hat(bold(x))') space Pi^(D B)_(L_1)(k_1; bold(x)').
$ <eq:factor34>

== The separation variable and the shell kernels

Multiply the two factors and change variables to $bold(s) = bold(x)' - bold(x)$. The phases combine to

$
e^(-i (bold(k)_1 - bold(k)_2) dot bold(x)) space e^(-i (bold(k)_2 - bold(k)_1) dot (bold(x) + bold(s))) = e^(+i bold(k)_1 dot bold(s)) space e^(-i bold(k)_2 dot bold(s)).
$ <eq:phases>

This is the key step: each binned momentum now appears in exactly one plane wave, and both plane waves involve the same separation $bold(s)$. Consequently, at fixed $bold(s)$, the double shell average in @eq:cov-wick factorises exactly into two single shell averages. Define the *shell kernel* for a spectrum $p_L$ and bin $i$:

#resultbox[
$
cal(K)^((i))_(ell L)[p](hat(bold(a)), hat(bold(b)); bold(s)) equiv frac(1, V_i) integral_(bold(k) in i) frac(dif^3 bold(k), (2 pi)^3) space p_L (k) space cal(L)_ell (hat(bold(k)) dot hat(bold(a))) space cal(L)_L (hat(bold(k)) dot hat(bold(b))) space e^(i bold(k) dot bold(s)).
$ <eq:kernel-def>
]

The first direction $hat(bold(a))$ always carries the estimator multipole $ell$, the second $hat(bold(b))$ the power-spectrum multipole $L$. In terms of the kernels, using the pair decomposition @eq:pairs,

$
overline(T)_n (i, j) &equiv frac(1, V_i V_j) integral_i integral_j frac(dif^3 bold(k)_1 dif^3 bold(k)_2, (2 pi)^6) T_n (bold(k)_1, bold(k)_2), quad n = 1, 2, \
overline(T)_1 (i, j) &= sum_(L_1 L_2) sum_((omega, p) in cal(P)^(A D)) sum_((omega', p') in cal(P)^(C B))
integral dif^3 bold(x) space dif^3 bold(s) space omega(bold(x)) space omega'(bold(x) + bold(s)) \
& quad quad times cal(K)^((i))_(ell_1 L_1)[p'](hat(bold(x)), hat(bold(x))'; bold(s)) space cal(K)^((j))_(ell_2 L_2)[p](hat(bold(x))', hat(bold(x)); -bold(s)), \
overline(T)_2 (i, j) &= sum_(L_1 L_2) sum_((omega, p) in cal(P)^(A C)) sum_((omega', p') in cal(P)^(D B))
integral dif^3 bold(x) space dif^3 bold(s) space omega(bold(x)) space omega'(bold(x) + bold(s)) \
& quad quad times cal(K)^((i))_(ell_1 L_1)[p'](hat(bold(x)), hat(bold(x))'; bold(s)) space cal(K)^((j))_(ell_2 L_2)[p](hat(bold(x)), hat(bold(x)); -bold(s)),
$ <eq:master>

with $hat(bold(x))' equiv (bold(x) + bold(s)) \/ |bold(x) + bold(s)|$. @eq:master is the master formula. Note the bookkeeping: the kernel for bin $i$ (momentum $bold(k)_1$) always carries the spectrum of the correlator that contains $F^B_0(-bold(k)_1)$, i.e. $p^(C B)$ in $T_1$ and $p^(D B)$ in $T_2$; the kernel for bin $j$ carries $p^(A D)$ in $T_1$ and $p^(A C)$ in $T_2$. In $T_2$ the second kernel has both Legendre weights at $hat(bold(x))$.

== The shell kernel in harmonics: exact truncation

Expand the two Legendre polynomials with @eq:addition, the plane wave with @eq:planewave, and integrate over $hat(bold(k))$. The angular integral is an ordinary Gaunt coefficient,

$
integral dif Omega_k space overline(Y)_(ell m)(hat(bold(k))) overline(Y)_(L M)(hat(bold(k))) overline(Y)_(lambda nu)(hat(bold(k))) = cal(G)^(m M nu)_(ell L lambda),
$

which forces $lambda in "tri"(ell, L)$: *the spherical-Bessel expansion terminates exactly at $lambda = ell + L$.* The radial integral is a bin average; since $V_i = (4 pi \/ (2 pi)^3) integral_i k^2 dif k$,

$
frac(1, V_i) integral_i frac(k^2 dif k, (2 pi)^3) p_L (k) j_lambda (k s) = frac(1, 4 pi) overline(p)^((i))_(L lambda)(s),
quad
overline(p)^((i))_(L lambda)(s) equiv frac(integral_i k^2 dif k space p_L (k) space j_lambda (k s), integral_i k^2 dif k).
$ <eq:pbar>

Collecting factors $(4 pi)^2 \/ [(2 ell + 1)(2 L + 1)]$ from the addition theorems, $4 pi$ from the plane wave and $1 \/ (4 pi)$ from the radial average:

#resultbox[
$
cal(K)^((i))_(ell L)[p](hat(bold(a)), hat(bold(b)); bold(s)) = frac((4 pi)^2, (2 ell + 1)(2 L + 1))
sum_(lambda in "tri"(ell, L)) i^lambda space overline(p)^((i))_(L lambda)(s)
sum_(m M nu) cal(G)^(m M nu)_(ell L lambda) space Y_(ell m)(hat(bold(a))) space Y_(L M)(hat(bold(b))) space Y_(lambda nu)(hat(bold(s))).
$ <eq:kernel-harm>
]

For the kernel evaluated at $-bold(s)$ use $Y_(lambda nu)(-hat(bold(s))) = (-1)^lambda Y_(lambda nu)(hat(bold(s)))$, which is $+1$ for even $lambda$; equivalently $i^lambda -> (-i)^lambda$, which for even $lambda$ is the same number $(-1)^(lambda \/ 2)$. Two useful checks of @eq:kernel-harm: for $ell = L = 0$ it reduces to $cal(K)^((i))_(00)[p](bold(s)) = overline(p)^((i))_(00)(s)$, the bin-averaged $p(k) j_0(k s)$; for $ell = 2$, $L = 0$ it gives $-overline(p)^((i))_(02)(s) space cal(L)_2(hat(bold(a)) dot hat(bold(s)))$, matching $integral frac(dif Omega_k, 4 pi) cal(L)_2(hat(bold(k)) dot hat(bold(a))) e^(i bold(k) dot bold(s)) = -j_2(k s) cal(L)_2(hat(bold(a)) dot hat(bold(s)))$.

== Contracting two kernels: window functions and coupling coefficients

Insert @eq:kernel-harm twice into @eq:master. The product of the two kernels contains harmonics at three directions, $hat(bold(x))$, $hat(bold(x))'$ and $hat(bold(s))$. The lists of harmonics at each direction are:

#align(center)[
#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  stroke: 0.5pt + luma(150),
  [], [at $hat(bold(x))$], [at $hat(bold(x))'$], [at $hat(bold(s))$],
  [$T_1$], [$Y_(ell_1 m_1) Y_(L_2 M_2)$], [$Y_(L_1 M_1) Y_(ell_2 m_2)$], [$Y_(lambda nu) Y_(lambda' nu')$],
  [$T_2$], [$Y_(ell_1 m_1) Y_(ell_2 m_2) Y_(L_2 M_2)$], [$Y_(L_1 M_1)$], [$Y_(lambda nu) Y_(lambda' nu')$],
)
]

Here $(lambda, nu)$ belong to the bin-$i$ kernel and $(lambda', nu')$ to the bin-$j$ kernel. Apply the merge rule @eq:merge at each direction, producing $overline(Y)_(Lambda_1 mu_1)(hat(bold(x))) overline(Y)_(Lambda_2 mu_2)(hat(bold(x))') overline(Y)_(Lambda mu)(hat(bold(s)))$. The integral over $hat(bold(s))$ and $bold(x)$ then only involves the windows and these three harmonics. Define the *window pair tensor*

$
Xi^(omega omega' ; mu_1 mu_2 mu)_(Lambda_1 Lambda_2 Lambda)(s) equiv integral dif Omega_s integral dif^3 bold(x) space omega(bold(x)) space omega'(bold(x) + bold(s)) space
overline(Y)_(Lambda_1 mu_1)(hat(bold(x))) space overline(Y)_(Lambda_2 mu_2)(hat(bold(x))') space overline(Y)_(Lambda mu)(hat(bold(s))).
$ <eq:Xi>

The coefficient multiplying $Xi$ is a contraction of Gaunt coefficients over all magnetic numbers; call it $T^(mu_1 mu_2 mu)_(Lambda_1 Lambda_2 Lambda)$. Because the whole expression is a rotational scalar and $T$ is built solely from invariant tensors, $T$ must be proportional to the unique invariant of three angular momenta, the $3j$ symbol (Appendix A):

$
T^(mu_1 mu_2 mu)_(Lambda_1 Lambda_2 Lambda) = t_(Lambda_1 Lambda_2 Lambda) space mat(delim: "(", Lambda_1, Lambda_2, Lambda; mu_1, mu_2, mu),
quad
t_(Lambda_1 Lambda_2 Lambda) = sum_(mu_1 mu_2 mu) mat(delim: "(", Lambda_1, Lambda_2, Lambda; mu_1, mu_2, mu) T^(mu_1 mu_2 mu)_(Lambda_1 Lambda_2 Lambda),
$ <eq:t-def>

using $sum_(mu_1 mu_2 mu) (3j)^2 = 1$. Therefore only the scalar projection of the window tensor is needed:

#resultbox[
$
cal(Q)^(omega omega')_(Lambda_1 Lambda_2 Lambda)(s) equiv sum_(mu_1 mu_2 mu) mat(delim: "(", Lambda_1, Lambda_2, Lambda; mu_1, mu_2, mu) space Xi^(omega omega' ; mu_1 mu_2 mu)_(Lambda_1 Lambda_2 Lambda)(s)
= integral dif Omega_s integral dif^3 bold(x) space omega(bold(x)) space omega'(bold(x) + bold(s)) space cal(S)_(Lambda_1 Lambda_2 Lambda)(hat(bold(x)), hat(bold(x))', hat(bold(s))),
$ <eq:Q>
$
cal(S)_(Lambda_1 Lambda_2 Lambda)(hat(bold(a)), hat(bold(b)), hat(bold(c))) equiv sum_(mu_1 mu_2 mu) mat(delim: "(", Lambda_1, Lambda_2, Lambda; mu_1, mu_2, mu) overline(Y)_(Lambda_1 mu_1)(hat(bold(a))) overline(Y)_(Lambda_2 mu_2)(hat(bold(b))) overline(Y)_(Lambda mu)(hat(bold(c))).
$ <eq:S>
]

$cal(S)$ is a tripolar spherical harmonic; for even $Lambda_1, Lambda_2, Lambda$ it is real, hence so are $cal(Q)$ and $t$. These $cal(Q)$ are the covariance analogues of the familiar window multipoles $Q_ell (s)$ used to convolve the mean power spectrum ($Q_ell$ corresponds to $Lambda_2 = 0$, $Lambda_1 = Lambda = ell$ with the $W_A$ rather than $W^(A B)$ weights).

== Explicit coupling coefficients

Writing out the contraction, with the magnetic numbers fixed by the Gaunt selection rules:

$
t^((1)) &= sum_(m_1 M_1 m_2 M_2) cal(G)^(m_1 M_1 nu)_(ell_1 L_1 lambda) space cal(G)^(m_2 M_2 nu')_(ell_2 L_2 lambda') space
cal(G)^(m_1 M_2 mu_1)_(ell_1 L_2 Lambda_1) space cal(G)^(M_1 m_2 mu_2)_(L_1 ell_2 Lambda_2) space cal(G)^(nu nu' mu)_(lambda lambda' Lambda) space
mat(delim: "(", Lambda_1, Lambda_2, Lambda; mu_1, mu_2, mu), \
& quad nu = -m_1 - M_1, quad nu' = -m_2 - M_2, quad mu_1 = -m_1 - M_2, quad mu_2 = -M_1 - m_2, quad mu = -nu - nu';
$ <eq:t1>
$
t^((2)) &= sum_(m_1 M_1 m_2 M_2) cal(G)^(m_1 M_1 nu)_(ell_1 L_1 lambda) space cal(G)^(m_2 M_2 nu')_(ell_2 L_2 lambda') space
cal(G)^(m_1 m_2 M_2 mu_1)_(ell_1 ell_2 L_2 Lambda_1) space cal(G)^(M_1 mu_2)_(L_1 Lambda_2) space cal(G)^(nu nu' mu)_(lambda lambda' Lambda) space
mat(delim: "(", Lambda_1, Lambda_2, Lambda; mu_1, mu_2, mu), \
& quad nu, nu', mu "as above", quad mu_1 = -m_1 - m_2 - M_2, quad Lambda_2 = L_1, quad mu_2 = -M_1.
$ <eq:t2>

In $t^((2))$ the four-harmonic coefficient is obtained from @eq:gaunt-rec and the two-harmonic one is $cal(G)^(M_1 mu_2)_(L_1 Lambda_2) = (-1)^(M_1) delta_(L_1 Lambda_2) delta_(mu_2, -M_1)$. Each sum has at most $(2 ell_1 + 1)(2 L_1 + 1)(2 ell_2 + 1)(2 L_2 + 1) <= 9^4$ terms and is computed once.

= Final result

Combining @eq:cov-wick, @eq:master, @eq:kernel-harm, @eq:Q and the phase $i^lambda (-i)^(lambda') = (-1)^((lambda + lambda') \/ 2)$:

#resultbox[
$
C^(A B C D)_(ell_1 ell_2)(i, j) &= frac((4 pi)^4, I_(A B) I_(C D)) sum_(L_1 L_2) frac(1, (2 L_1 + 1)(2 L_2 + 1))
sum_(lambda lambda') (-1)^((lambda + lambda') \/ 2) sum_(Lambda_1 Lambda_2 Lambda)
[ t^((1)) space cal(I)^((1))_(i j) + t^((2)) space cal(I)^((2))_(i j) ], \
cal(I)^((1))_(i j) &= sum_((omega, p) in cal(P)^(A D)) space sum_((omega', p') in cal(P)^(C B))
integral s^2 dif s space overline(p')^((i))_(L_1 lambda)(s) space overline(p)^((j))_(L_2 lambda')(s) space cal(Q)^(omega omega')_(Lambda_1 Lambda_2 Lambda)(s), \
cal(I)^((2))_(i j) &= sum_((omega, p) in cal(P)^(A C)) space sum_((omega', p') in cal(P)^(D B))
integral s^2 dif s space overline(p')^((i))_(L_1 lambda)(s) space overline(p)^((j))_(L_2 lambda')(s) space cal(Q)^(omega omega')_(Lambda_1 Lambda_2 Lambda)(s).
$ <eq:final>
]

Here $overline(p')^((i))_(L_1 lambda)$ is the bin-$i$ average of the spectrum $p'$ of the primed pair (the one attached to $bold(k)_1$) and $overline(p)^((j))_(L_2 lambda')$ that of the unprimed pair (attached to $bold(k)_2$). For a shot-noise pair $(S, delta_(L 0))$, the multipole is forced to $L = 0$ and the radial function is the bin-averaged Bessel function $overline(j_lambda (k s))^((i))$.

== Selection rules

All indices even. In addition:

- $lambda in "tri"(ell_1, L_1)$, $lambda' in "tri"(ell_2, L_2)$;
- $T_1$: $Lambda_1 in "tri"(ell_1, L_2)$, $Lambda_2 in "tri"(L_1, ell_2)$;
- $T_2$: $Lambda_1 in "tri"(ell_1, ell_2, L_2)$, $Lambda_2 = L_1$;
- $Lambda in "tri"(lambda, lambda') inter "tri"(Lambda_1, Lambda_2)$;
- shot-noise pairs: $L_1 = 0$ (if the primed pair is shot noise) and/or $L_2 = 0$ (if the unprimed pair is).

For $ell_1, ell_2, L_1, L_2 <= 4$: $lambda, lambda' <= 8$, $Lambda_1 <= 8$ ($T_1$) or $<= 12$ ($T_2$), $Lambda_2 <= 8$, $Lambda <= 16$. The number of distinct window functions $cal(Q)_(Lambda_1 Lambda_2 Lambda)$ per pair of windows $(omega, omega')$ is of order one hundred.

== Structure

@eq:final separates cleanly into geometry, cosmology and numbers:

#align(center)[
#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  stroke: 0.5pt + luma(150),
  [Object], [Depends on], [Dimension], [Computed],
  [$cal(Q)^(omega omega')_(Lambda_1 Lambda_2 Lambda)(s)$], [survey geometry, weights], [1D in $s$], [once per survey],
  [$overline(p)^((i))_(L lambda)(s)$], [model $P_L (k)$, binning], [1D in $s$, per bin], [per cosmology (fast)],
  [$t^((1)), t^((2))$], [multipole labels only], [scalars], [once, cached],
  [$I_(A B)$], [survey geometry, weights], [scalar], [once per survey],
)
]

= Computing the ingredients

== Radial kernels $overline(p)^((i))_(L lambda)(s)$

For a bin $[k_i^-, k_i^+]$ and a tabulated model $P_L (k)$,
$
overline(p)^((i))_(L lambda)(s) = frac(3, (k_i^+)^3 - (k_i^-)^3) integral_(k_i^-)^(k_i^+) k^2 dif k space P_L (k) space j_lambda (k s),
$
evaluated by direct quadrature on an $s$-grid. Guidelines:

- $s$-range: $0 <= s <= s_max$ with $s_max$ the largest pair separation in the survey ($approx 2 R_"survey"$; $cal(Q)(s)$ vanishes beyond it).
- $s$-step: the integrand of @eq:final oscillates with period $2 pi \/ k$; use $Delta s <= 1$--$2 space h^(-1)"Mpc"$ for $k_max approx 0.3 space h space "Mpc"^(-1)$, or a finer grid near $s = 0$ and test convergence.
- The bin average is what makes the large-$s$ tail decay: $overline(j_lambda (k s))^((i)) tilde (k s)^(-1) "sinc"(Delta k space s \/ 2)$ for narrow bins. The tail must nonetheless be integrated to $s_max$ against $cal(Q)(s)$; do not truncate early.
- Any bin shape (top-hat, or the exact discrete-mode weights of an FFT grid) can be used by replacing $integral_i k^2 dif k$ with the corresponding weighted sum; the formula @eq:pbar then remains an average over the modes actually used by the estimator.

== Window functions $cal(Q)^(omega omega')_(Lambda_1 Lambda_2 Lambda)(s)$ from pair counts

Each window in @eq:windows is a product of a mean density and a smooth weight, so it is sampled by the random catalogue. If $R_X$ has density $overline(n)_X \/ alpha_X$,
$
integral dif^3 bold(x) space omega(bold(x)) f(bold(x)) approx alpha_X sum_(r in R_X) tilde(omega)(bold(x)_r) f(bold(x)_r),
quad
tilde(omega) equiv omega \/ overline(n)_X,
$
with, for the windows needed here, $tilde(W)^(A B) = overline(n)_B w_A w_B$ (sampled by $R_A$) and $tilde(S)^A = (1 + alpha_A) w_A^2$ (sampled by $R_A$). Then, for a radial bin $b$ of width $Delta s$ centred on $s_b$,

#resultbox[
$
cal(Q)^(omega omega')_(Lambda_1 Lambda_2 Lambda)(s_b) approx frac(alpha alpha', s_b^2 Delta s) sum_(r in R) sum_(r' in R') Theta_b (|bold(s)_(r r')|) space
tilde(omega)(bold(x)_r) space tilde(omega)'(bold(x)_(r')) space cal(S)_(Lambda_1 Lambda_2 Lambda)(hat(bold(x))_r, hat(bold(x))_(r'), hat(bold(s))_(r r')), \
bold(s)_(r r') equiv bold(x)_(r') - bold(x)_r .
$ <eq:Q-pairs>
]

Notes:

- The pairs are *ordered*: $r$ carries $omega$ and the first line-of-sight index $Lambda_1$, $r'$ carries $omega'$ and $Lambda_2$; $bold(s)$ points from $r$ to $r'$. Both orderings of a given pair of points contribute (as different terms) unless $omega = omega'$ and $Lambda_1 = Lambda_2$.
- $cal(S)$ in @eq:S is evaluated per pair from the three unit vectors; precompute $overline(Y)_(Lambda mu)$ up to $Lambda = 16$ for $hat(bold(s))$ and up to $12$ for $hat(bold(x))$, and the $3j$ symbols once.
- The window functions are smooth; a heavily subsampled random catalogue ($10^5$--$10^6$ points) with a tree code is adequate. Convergence can be checked by varying the subsample.
- Radial binning of $cal(Q)$ must be at least as fine as the $s$-grid used for @eq:final, or $cal(Q)$ must be interpolated; $cal(Q)$ is smooth, so interpolation is safe.
- Alternative (grid): paint the fields $omega(bold(x)) overline(Y)_(Lambda_1 mu_1)(hat(bold(x)))$ on a mesh, cross-correlate by FFT with $omega'(bold(x)) overline(Y)_(Lambda_2 mu_2)(hat(bold(x)))$, project the result on $overline(Y)_(Lambda mu)(hat(bold(s)))$ over spherical shells in $bold(s)$, and contract with the $3j$ symbol. Pair counts avoid aliasing and any embedding box and are preferred.

The normalisations $I_(A B) = alpha_A sum_(r in R_A) tilde(W)^(A B)(bold(x)_r)$ are computed from the same randoms.

== Coupling coefficients

Compute the Gaunt coefficients from Wigner $3j$ symbols (e.g. `sympy.physics.wigner`, `py3nj`, or `wigxjpf`), build $cal(G)^((4))$ by @eq:gaunt-rec, and evaluate @eq:t1 and @eq:t2 by explicit loops over $(m_1, M_1, m_2, M_2)$. As a correctness test, compute the unprojected tensor $T^(mu_1 mu_2 mu)$ (same sums without the $3j$ factor) for a few index sets and verify that it is proportional to the $3j$ symbol to machine precision; this checks the whole harmonic bookkeeping.

== Assembling the matrix

For each block $(ell_1, ell_2)$ and each tracer combination $(A B, C D)$:

+ Enumerate all index tuples $(L_1, L_2, lambda, lambda', Lambda_1, Lambda_2, Lambda, "term", "pairs")$ allowed by the selection rules; attach the prefactor
  $ c = frac((4 pi)^4 (-1)^((lambda + lambda') \/ 2), (2 L_1 + 1)(2 L_2 + 1)) space t^(("term")). $
+ For each tuple form the vectors $u_i (s) = overline(p')^((i))_(L_1 lambda)(s)$ over bins $i$ and $v_j (s) = overline(p)^((j))_(L_2 lambda')(s)$ over bins $j$, on the common $s$-grid.
+ Accumulate $C_(i j) += frac(c, I_(A B) I_(C D)) sum_s Delta s space s^2 space cal(Q)(s) space u_i (s) v_j (s)$, i.e. one matrix product $U space "diag"(Delta s space s^2 c cal(Q)) space V^T$ per tuple.

With $N_"bin"$ bins and $N_s$ grid points, each tuple costs $N_"bin"^2 N_s$ operations; a few thousand tuples with $N_"bin" tilde 30$, $N_s tilde 3000$ take seconds. Changing the cosmology only requires recomputing the $overline(p)$ vectors.

= Checks

+ *Distant-observer, periodic-box limit.* Let the window be uniform, $W = overline(n)^2$, in a volume $V$ far from the observer so that $hat(bold(x)) approx hat(bold(n))$ is constant. Then $cal(Q)^(W W)_(Lambda_1 Lambda_2 Lambda)(s) -> overline(n)^4 V_"ov"(s) space cal(S)_(Lambda_1 Lambda_2 Lambda)(hat(bold(n)), hat(bold(n)), dot)$ integrated over $hat(bold(s))$, i.e. only $Lambda = 0$ survives, and the $s$-integral of two bin-averaged Bessel functions of the same order yields $delta_(i j) \/ V_i$ (orthogonality). The result must reduce to
  $ C_(ell_1 ell_2)(k_i) = frac(2 (2 ell_1 + 1)(2 ell_2 + 1), N_i) sum_(L_1 L_2) P_(L_1)(k_i) P_(L_2)(k_i) integral_(-1)^(1) frac(dif mu, 2) cal(L)_(ell_1)(mu) cal(L)_(ell_2)(mu) cal(L)_(L_1)(mu) cal(L)_(L_2)(mu), \
    N_i = V V_i, $
  the standard Gaussian multipole covariance (with $P_L -> P_L + delta_(L 0) \/ overline(n)$ when shot noise is included). The simplest sub-case, $ell_1 = ell_2 = L_1 = L_2 = 0$, gives $2 P^2 \/ N_i$ and fixes all $4 pi$ factors.
+ *Symmetry.* The exact covariance is symmetric under $(ell_1, i, A B) <-> (ell_2, j, C D)$. In @eq:final this symmetry is exact for $T_1$ and broken only by the $hat(bold(x))' -> hat(bold(x))$ step in $T_2$; the size of the residual asymmetry measures that approximation.
+ *Brute force.* For one bin pair, evaluate @eq:cov-wick directly on a $bold(q) = bold(k)_2 - bold(k)_1$ grid using FFTs of the window multipole fields and Monte-Carlo shell averages. Agreement checks the coefficients, phases and $cal(Q)$ normalisation end to end.
+ *Mocks.* Compare the diagonal and the first off-diagonals with the sample covariance of mocks that share the window; residual differences at small scales are expected from the non-Gaussian terms not included here.

#pagebreak()
#set heading(numbering: "A.1", supplement: "Appendix")
#counter(heading).update(0)

= Why only the scalar projection of the window tensor survives

Fix $s$ and consider the product of the two kernels @eq:kernel-harm appearing in @eq:master, as a function of the three directions $hat(bold(x))$, $hat(bold(x))'$, $hat(bold(s))$ with the radial functions held fixed:
$
Phi(hat(bold(x)), hat(bold(x))', hat(bold(s))) = sum_(Lambda_1 mu_1 Lambda_2 mu_2 Lambda mu) T^(mu_1 mu_2 mu)_(Lambda_1 Lambda_2 Lambda) space overline(Y)_(Lambda_1 mu_1)(hat(bold(x))) overline(Y)_(Lambda_2 mu_2)(hat(bold(x))') overline(Y)_(Lambda mu)(hat(bold(s))).
$
Each kernel is, by its definition @eq:kernel-def, a scalar function of its arguments: it depends on $hat(bold(a)), hat(bold(b)), bold(s)$ only through dot products, because the integration over $hat(bold(k))$ covers the full sphere. Hence $Phi$ is invariant under a common rotation of its three arguments. The coefficients of an invariant function in the basis $overline(Y) overline(Y) overline(Y)$ form an invariant tensor of $V_(Lambda_1) times.o V_(Lambda_2) times.o V_Lambda$, and by the Wigner--Eckart theorem that space is one-dimensional, spanned by the $3j$ symbol. This gives @eq:t-def. Contracting $T$ with the window tensor $Xi$ therefore only picks out the projection of $Xi$ on the $3j$ symbol, which is @eq:Q.

If one prefers not to rely on this argument numerically, the unreduced form is equally valid: keep the full tensor $Xi^(mu_1 mu_2 mu)$ from pair counts (with $overline(Y)_(Lambda_1 mu_1) overline(Y)_(Lambda_2 mu_2) overline(Y)_(Lambda mu)$ accumulated per pair instead of $cal(S)$) and contract it with $T^(mu_1 mu_2 mu)$. The storage is larger by the number of magnetic components, but no $3j$ projection is needed. The two routes must agree.

= Phase bookkeeping

The only phases in the derivation are $i^lambda$ from $e^(i bold(k)_1 dot bold(s))$ and $(-i)^(lambda')$ from $e^(-i bold(k)_2 dot bold(s))$. Parity of the Gaunt coefficient $cal(G)^(m M nu)_(ell L lambda)$ forces $lambda equiv ell + L$ (mod 2), hence even, so both phases are real and their product is $(-1)^((lambda + lambda') \/ 2)$. No other sign appears: $Y_(lambda' nu')(-hat(bold(s))) = Y_(lambda' nu')(hat(bold(s)))$ for even $lambda'$, and the merge rule @eq:merge has been written so that every summed magnetic index pairs a harmonic with a conjugate harmonic.

= Symbol table

#table(
  columns: (auto, auto),
  align: (left, left),
  stroke: 0.5pt + luma(150),
  [Symbol], [Meaning],
  [$A, B, C, D$], [tracer labels; $hat(P)^(A B)$ is the cross spectrum of $A$ (Legendre-weighted) and $B$],
  [$W^(A B)$, $S^A$], [clustering and shot-noise windows, @eq:windows],
  [$I_(A B)$], [$integral dif^3 bold(x) W^(A B)$],
  [$cal(P)^(A B)$], [set of spectrum--window pairs $(omega, p)$, @eq:pairs],
  [$V_i$], [$k$-space volume of bin $i$, $(2 pi^2)^(-1) integral_i k^2 dif k$],
  [$ell_1, ell_2$], [estimator multipoles of bins $i, j$],
  [$L_1, L_2$], [power-spectrum multipoles attached to $bold(k)_1$ (bin $i$) and $bold(k)_2$ (bin $j$)],
  [$lambda, lambda'$], [Bessel orders of the bin-$i$ and bin-$j$ kernels],
  [$Lambda_1, Lambda_2, Lambda$], [multipoles at $hat(bold(x))$, $hat(bold(x))'$, $hat(bold(s))$ of the window function],
  [$overline(p)^((i))_(L lambda)(s)$], [bin-averaged Bessel transform of $p_L$, @eq:pbar],
  [$cal(K)^((i))_(ell L)$], [shell kernel, @eq:kernel-def and @eq:kernel-harm],
  [$cal(S)_(Lambda_1 Lambda_2 Lambda)$], [tripolar spherical harmonic, @eq:S],
  [$cal(Q)^(omega omega')_(Lambda_1 Lambda_2 Lambda)(s)$], [tripolar window function, @eq:Q and @eq:Q-pairs],
  [$t^((1)), t^((2))$], [coupling coefficients, @eq:t1 and @eq:t2],
)
