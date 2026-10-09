#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, it)

#let x = $bold(x)$
#let y = $bold(y)$
#let k = $bold(k)$
#let q = $bold(q)$
#let p = $bold(p)$
#let r = $bold(r)$
#let s = $bold(s)$
#let nh = $hat(bold(n))$
#let xh = $hat(bold(x))$
#let avg(body) = $lr(chevron.l chevron.l body chevron.r chevron.r)$
#let G = $cal(G)$
#let tj(a, b, c, d, e, f) = math.mat(delim: "(", (a, b, c), (d, e, f))
#let note(body) = block(fill: luma(242), inset: 8pt, radius: 3pt, width: 100%, body)

#align(center)[
  #text(size: 16pt, weight: "bold")[Gaussian covariance of windowed bispectrum multipoles\ in the tripolar (Sugiyama) basis]
  #v(0.3em)
  #text(size: 11pt)[The same formalism as for the power spectrum: local plane-parallel correlators, plane-wave and Legendre expansions, Wigner--Eckart, and a window function from pair counts]
  #v(0.3em)
  #text(size: 9.5pt, fill: luma(80))[Derivation note, branch `thecov2`, October 2026. Conventions follow Sugiyama, Saito, Beutler & Seo (2019, 2020).]
]

#v(0.8em)
#note[
*Summary.* The Gaussian ($P P P$) covariance of the bispectrum multipoles $B_(ell_1 ell_2 L)(k_1, k_2)$ of a windowed survey can be derived exactly as the power-spectrum covariance was: each of the three two-point correlators that a Wick pairing produces is a windowed, locally anisotropic power spectrum; the shell integrals turn into Bessel averages; the Legendre weights of the estimators and of the anisotropic correlators turn into spherical harmonics; and the Wigner--Eckart theorem collapses the product of Gaunt coefficients onto rotational invariants. Two things are new. First, the three correlators sit at three points, so the window enters through its *three-point* function $integral dif^3 x thin W(x) W(x + r_1) W(x + r_2)$, weighted by an invariant combination of harmonics of the two separations $hat(r)_1, hat(r)_2$ and of the three lines of sight. Second, the third side of the triangle, $k_3 = |k_1 + k_2|$, is not binned, so its power spectrum depends on the opening angle of the two binned momenta; this couples the two separations through a bipolar expansion. The six pairings split into two classes: in the two pairings where the two line-of-sight galaxies pair with each other the window function is anchored at that galaxy and is a product of two pair sums per anchor --- no triplet counting; in the other four pairings one shell momentum runs along the third side of the triangle. For the elements that matter most, the diagonal $(k_1, k_2) = (k'_1, k'_2)$, the first class is the leading term and the second is suppressed by $tilde.op Delta k \/ k$. In a periodic box the result reduces to eq. (49) of Sugiyama et al. (2020). Nothing here is specific to $B_(000)$ or $B_(202)$; their index sets are listed at the end.
]


#note[
*Status (October 2026).* The Gaussian part was reviewed by independent numerical checks; the derivation stands, with these corrections not yet folded into the text: (i) $H_(202) = +1 \/ sqrt(5)$ (in general $H_(ell 0 ell) = (-1)^ell \/ sqrt(2 ell + 1)$); (ii) the invariant basis $E_(K_1 K_2)$ after @eq:T must couple the line-of-sight harmonics with the same variance as $T$: write $overline(Y)_(L M) = (-1)^M Y_(L, -M)$, so that all five harmonics are unconjugated, and use $E = sum chevron.l lambda_1 rho_1 thin L_1 M_1 | K_1 kappa_1 chevron.r chevron.l lambda_2 rho_2 thin L_2 M_2 | K_2 kappa_2 chevron.r tj(K_1, K_2, Lambda_3, kappa_1, kappa_2, mu_3)$, which is orthonormal ($cal(T) = sum T E$) and makes the scalar of @eq:Q3 and the anchored moments $A^((a))$ rotation invariant; the same correction applies to the pair-sum formulae; (iii) in a periodic box only $lambda_1 = lambda_2 = 0$ survive, and $cal(Q) -> V sqrt(N_(L_1 L_2 Lambda_3) \/ 4 pi) thin H_(L_1 L_2 Lambda_3)$ carries the redshift-space ranks $L_a, Lambda_3$ of eq. (49); (iv) in class 2 the window does not bound the $(lambda, lambda')$ sums in the $(p_1, p_2)$ variables; use $(p_1, p_3 = p_1 + p_2)$ so that each anchored side carries one plane wave, and the ranks are bounded by $lambda_w + ell_a + ell'_a + L_a$ as in class 1 (the class-1 $(a, b)$ bound also reads so); (v) the pairing $(1 2)(1' 2')(3 3')$ is not negligible: on the elements with $i = j$, $i' = j'$ it has the same coincidence measure as class 1 and is suppressed only by the integral constraint, $C^((D))(i i; i' i') approx (4 pi)^3 H H' sqrt(N N') \/ (I_3^2 V_i V_(i')) thin integral dif^3 x dif^3 x' thin W(x) W(x') chevron.l overline(tilde(cal(S))) P chevron.r_i chevron.l tilde(cal(S))' P chevron.r_(i') chevron.l F(x) F(x') chevron.r_"IC"$, the long-mode variance of the $W$-weighted field (zero for uniform $omega$, tens of percent of the $(i, i)$ variance in a Gaussian test with a radially varying $omega$); (vi) the self-pair $y_1 = y_2$ must be subtracted from the $r_1 = r_2$ bins of the pair sums, and the product cost is $tilde.op 10^(12)$ with $mu_3 = 0$, not $10^(11)$.
]

= Conventions

We use the identities of the power-spectrum note: the addition theorem, the plane-wave expansion, the generalised Gaunt coefficient $G^(m_1 dots m_n)_(ell_1 dots ell_n) equiv integral dif Omega thin product_a Y_(ell_a m_a)$, and the product rule
$ product_(a = 1)^n Y_(ell_a m_a)(nh) = sum_(Lambda mu) G^(m_1 dots m_n mu)_(ell_1 dots ell_n Lambda) thin overline(Y)_(Lambda mu)(nh), quad
  product_(a = 1)^n overline(Y)_(ell_a m_a)(nh) = sum_(Lambda mu) G^(m_1 dots m_n mu)_(ell_1 dots ell_n Lambda) thin Y_(Lambda mu)(nh). $ <eq:product>
Spherical harmonics are the standard $Y_(ell m)$; Sugiyama et al. use $y^m_ell = sqrt(4 pi \/ (2 ell + 1)) thin Y_(ell m)$.

*Fields and windows.* One tracer, weighted field $F(x) = w(x) [n_g (x) - alpha n_r (x)]$, Fourier transform $F(k) = integral dif^3 x thin e^(-i k dot x) F(x)$, so $F(-k) = overline(F(k))$. With $omega(x) equiv overline(n)(x) w(x)$:
$ W(x) equiv omega^2(x) quad "(pair window)", wide S(x) equiv (1 + alpha) thin overline(n)(x) w^2(x) quad "(shot-noise window)", \
  I_3 equiv integral dif^3 x thin omega^3(x) . $
$I_3$ is the normalisation $I = integral dif^3 x thin overline(n)^3 w^3$ of Sugiyama et al. (2019, eq. 37).

*Shell averages.* For a shell $i$ of the wavenumber magnitude, $V_i = integral_i dif^3 q \/ (2 pi)^3$ and
$ avg(f)_i equiv 1 / V_i integral_i (dif^3 q) / (2 pi)^3 f(q) = (integral_i q^2 dif q integral (dif Omega_q) / (4 pi) f(q)) / (integral_i q^2 dif q) , $
so that the angular part is a normalised average; nested averages are abbreviated, $avg(f)_(i j) equiv avg(avg(f)_i)_j$ and $avg(f)_(i j i' j')$ for the four shells of a covariance element. As in the power-spectrum note, with $P_L$ the multipoles of the model power spectrum,
$ overline(j)^((i))_lambda (r) equiv (integral_i q^2 dif q thin j_lambda (q r)) / (integral_i q^2 dif q), quad
  P^((i))_(L lambda)(r) equiv (integral_i q^2 dif q thin j_lambda (q r) thin P_L (q)) / (integral_i q^2 dif q) . $

*Plane-wave shell averages.* From the plane-wave expansion and the orthogonality of the harmonics,
$ avg(e^(-i q dot r) overline(Y)_(ell m)(hat(q)))_i = (-i)^ell thin overline(j)^((i))_ell (r) thin overline(Y)_(ell m)(hat(r)), wide
  avg(e^(+i p dot r) overline(Y)_(Lambda mu)(hat(p)) thin g(p))_i = i^Lambda thin overline(g)^((i))_Lambda (r) thin overline(Y)_(Lambda mu)(hat(r)), $ <eq:pw>
with $overline(g)_Lambda$ the Bessel-weighted radial average of $g$. The first is used for the momenta of the estimator whose covariance we take, the second for the primed one.

*Tripolar basis and estimator (Sugiyama et al. 2019, eqs. 12--13 and 34--35).* With $H_(ell_1 ell_2 L) = tj(ell_1, ell_2, L, 0, 0, 0)$ and $N_(ell_1 ell_2 L) = (2 ell_1 + 1)(2 ell_2 + 1)(2 L + 1)$,
$ B(k_1, k_2, nh) = sum_(ell_1 ell_2 L) B_(ell_1 ell_2 L)(k_1, k_2) thin cal(S)_(ell_1 ell_2 L)(hat(k)_1, hat(k)_2, nh), \
  cal(S)_(ell_1 ell_2 L)(hat(a), hat(b), nh) = 1 / H_(ell_1 ell_2 L) sum_(m_1 m_2 M) tj(ell_1, ell_2, L, m_1, m_2, M) y^(m_1)_(ell_1)(hat(a)) y^(m_2)_(ell_2)(hat(b)) y^M_L (nh), $
$ B_(ell_1 ell_2 L)(k_1, k_2) = N_(ell_1 ell_2 L) H^2_(ell_1 ell_2 L) integral (dif Omega_1 dif Omega_2 dif Omega_n) / (4 pi)^3 thin overline(cal(S))_(ell_1 ell_2 L) thin B . $
It is convenient to write $cal(S)$ through the unnormalised combination of ordinary harmonics,
$ tilde(cal(S))_(ell_1 ell_2 L)(hat(a), hat(b), nh) equiv sum_(m_1 m_2 M) tj(ell_1, ell_2, L, m_1, m_2, M) Y_(ell_1 m_1)(hat(a)) Y_(ell_2 m_2)(hat(b)) Y_(L M)(nh), quad
  cal(S) = (4 pi)^(3\/2) / (H sqrt(N)) thin tilde(cal(S)) . $
The estimator attaches the line of sight to the third galaxy, which is the one not integrated over a shell:
$ hat(B)_(ell_1 ell_2 L)(i, j) &= (N H^2) / I_3 avg( integral dif^3 x thin overline(cal(S))_(ell_1 ell_2 L)(hat(q)_1, hat(q)_2, xh) thin e^(i (q_1 + q_2) dot x) thin F(q_1) F(q_2) F(x) )_(i j) \
  &= ((4 pi)^(3\/2) H sqrt(N)) / I_3 avg( integral dif^3 x thin overline(tilde(cal(S)))_(ell_1 ell_2 L)(hat(q)_1, hat(q)_2, xh) thin e^(i (q_1 + q_2) dot x) F(q_1) F(q_2) F(x) )_(i j) . $ <eq:est>
Written out, $e^(i(q_1 + q_2) dot x) F(q_1) F(q_2) F(x) = integral dif^3 y_1 dif^3 y_2 thin F(y_1) F(y_2) F(x) thin e^(-i q_1 dot (y_1 - x)) e^(-i q_2 dot (y_2 - x))$: the two shell momenta are the sides of the triangle from the line-of-sight galaxy at $x$ to the galaxies at $y_1$, $y_2$.

*Unbiasedness.* In the local approximation $chevron.l F(y_1) F(y_2) F(x) chevron.r approx omega^3(x) zeta(y_1 - x, y_2 - x; xh)$, so $chevron.l hat(B) chevron.r = (N H^2 \/ I_3) integral dif^3 x thin omega^3 chevron.l chevron.l integral dif Omega_1 dif Omega_2 \/ (4 pi)^2 thin overline(cal(S))(hat(q)_1, hat(q)_2, xh) B(q_1, q_2, xh) chevron.r chevron.r$, which is $B_(ell_1 ell_2 L)$ averaged over the shells and over the survey, because for fixed $nh$ the $(hat(q)_1, hat(q)_2)$ average of $overline(cal(S))_(ell_1 ell_2 L) cal(S)_(ell'_1 ell'_2 L')$ is $delta_(ell_1 ell'_1) delta_(ell_2 ell'_2) delta_(L L') \/ (N H^2)$ (orthogonality of the 3-$j$ symbols at fixed $M$). The window dilutes it by $integral omega^3 \/ I_3 = 1$; with a pair-averaged window this ratio is the bispectrum analogue of $I_k \/ "norm"$.

= Wick expansion

Write the covariance as $chevron.l hat(B) thin overline(hat(B)') chevron.r_c$ with the primed estimator at shells $(i', j')$ and multipoles $(ell'_1 ell'_2 L')$. Since $overline(F(p)) = F(-p)$,
$ overline(hat(B)')_(ell'_1 ell'_2 L')(i', j') = ((4 pi)^(3\/2) H' sqrt(N')) / I_3 avg( integral dif^3 x' thin tilde(cal(S))_(ell'_1 ell'_2 L')(hat(p)_1, hat(p)_2, xh') thin e^(-i (p_1 + p_2) dot x') thin F(-p_1) F(-p_2) F(x') )_(i' j') . $
Label the six fields $1 equiv F(q_1)$, $2 equiv F(q_2)$, $3 equiv F(x)$, $1' equiv F(-p_1)$, $2' equiv F(-p_2)$, $3' equiv F(x')$. The Gaussian part of the six-point function is a sum over pairings; those with a pair inside one estimator, e.g. $chevron.l F(q_1) F(q_2) chevron.r$, are proportional to the window at $q_1 + q_2$ and only survive when $|q_1 + q_2|$ is below the window scale, a fraction $tilde.op (Delta k \/ k)^2$ of the shells: we drop them, as the power-spectrum note drops the $k approx 0$ terms. What remains are the $3! = 6$ pairings between $\{1, 2, 3\}$ and $\{1', 2', 3'\}$:
$ "class 1:" quad (1 1')(2 2')(3 3'), quad (1 2')(2 1')(3 3'); \
  "class 2:" quad (1 1')(2 3')(3 2'), quad (1 2')(2 3')(3 1'), quad (1 3')(2 1')(3 2'), quad (1 3')(2 2')(3 1') . $
In class 1 the two line-of-sight galaxies pair with each other; in class 2 each pairs with a shell galaxy of the other estimator.

= The correlators in the local approximation

Three kinds of two-point correlator occur. With $P(p; nh) = sum_L cal(L)_L (hat(p) dot nh) P_L (p)$, a pair of Fourier fields,
$ chevron.l F(q) F(-p) chevron.r = integral dif^3 y thin dif^3 y' thin e^(-i q dot y + i p dot y') chevron.l F(y) F(y') chevron.r
  approx sum_L integral dif^3 y thin e^(-i (q - p) dot y) thin W(y) thin cal(L)_L (hat(p) dot hat(y)) P_L (p) , $ <eq:FF>
exactly as in the power-spectrum note (window at $y$, anisotropy about $hat(y)$, pair separation only in the Fourier kernel; the power spectrum is evaluated at the primed momentum). A Fourier field and a configuration-space field, whose position carries the phase $e^(-i p_3 dot x')$ of the primed estimator and is integrated,
$ integral dif^3 x' thin e^(-i p_3 dot x') chevron.l F(q) F(x') chevron.r approx integral dif^3 y thin e^(-i (q + p_3) dot y) thin W(y) thin P(p_3; hat(y)) , $ <eq:Fx>
obtained by writing $x' = y + u$ and $integral dif^3 u thin e^(-i p_3 dot u) xi(u; hat(y)) = P(p_3; hat(y))$; and the two configuration-space fields, with both estimator phases,
$ integral dif^3 x' thin e^(i (q_1 + q_2) dot x - i (p_1 + p_2) dot x') chevron.l F(x) F(x') chevron.r approx W(x) thin e^(i (q_1 + q_2 - p_1 - p_2) dot x) thin P(p_1 + p_2; xh) . $ <eq:xx>
In every case the line-of-sight harmonic of the primed estimator, $Y_(L' M')(xh')$, is moved to the position of the galaxy its partner sits at ($xh$ in @eq:xx, $hat(y)$ in @eq:Fx), as the Legendre weight of the estimator was moved in the power-spectrum note. Shot noise adds to each correlator the same expression with $W -> S$ and $P_L -> delta^K_(L 0)$; we restore it at the end.

= Class 1: the two lines of sight pair together

== Phases and the three-point window function

For the pairing $(1 1')(2 2')(3 3')$ the three correlators are @eq:FF for $(q_1, p_1)$ at $y_1$, @eq:FF for $(q_2, p_2)$ at $y_2$, and @eq:xx at $x$. The phases combine into
$ e^(-i (q_1 - p_1) dot y_1) thin e^(-i (q_2 - p_2) dot y_2) thin e^(i (q_1 + q_2 - p_1 - p_2) dot x) = e^(-i (q_1 - p_1) dot r_1) thin e^(-i (q_2 - p_2) dot r_2), quad r_1 equiv y_1 - x, quad r_2 equiv y_2 - x : $
each momentum pair rides a plane wave on one side of the triangle from the line-of-sight galaxy. Collecting everything,
$ C^((1))_(ell ell') (i j; i' j') = ((4 pi)^3 H H' sqrt(N N')) / I_3^2 avg( cal(I)^((1)) )_(i j i' j'), wide
  cal(I)^((1)) equiv integral dif^3 x thin dif^3 r_1 thin dif^3 r_2 thin W(x) W(x + r_1) W(x + r_2) \
  times e^(-i (q_1 - p_1) dot r_1) e^(-i (q_2 - p_2) dot r_2) thin overline(tilde(cal(S))_(ell_1 ell_2 L))(hat(q)_1, hat(q)_2, xh) thin tilde(cal(S))_(ell'_1 ell'_2 L')(hat(p)_1, hat(p)_2, xh) thin P(p_1; hat(y)_1) P(p_2; hat(y)_2) P(p_1 + p_2; xh) , $ <eq:master1>
with $hat(y)_a = (x + r_a) \/ |x + r_a|$. This is the master formula of class 1. The second class-1 pairing, $(1 2')(2 1')(3 3')$, is the same expression with the primed labels exchanged, $(p_1, ell'_1, i') arrow.l.r (p_2, ell'_2, j')$; for the diagonal elements with $i = j$ it adds to the first with equal weight.

#note[*Box limit.* For a periodic box of volume $V$ with a global line of sight, $W = 1$, $xh = nh$, $I_3 = V$, and $integral dif^3 r_1 e^(-i (q_1 - p_1) dot r_1) = (2 pi)^3 delta_D (q_1 - p_1)$, whose shell average over $p_1$ is $delta_(i i') \/ V_i$. Hence
$ C^((1))_"box" &= ((4 pi)^3 H H' sqrt(N N')) / (V V_i V_j) thin delta_(i i') delta_(j j') avg( overline(tilde(cal(S))_ell) tilde(cal(S))_(ell') thin P(q_1) P(q_2) P(|q_1 + q_2|) )_(i j) \
  &= (N N' H^2 H'^2 thin V) / (tilde(N)_"mode"(k_i) tilde(N)_"mode"(k_j)) thin delta_(i i') delta_(j j') avg( cal(S)_ell cal(S)_(ell') thin P P P )_(i j) , $
using $(4 pi)^3 overline(tilde(cal(S))) tilde(cal(S))' = H H' sqrt(N N') cal(S) cal(S)'$ and $V_i = tilde(N)_"mode"(k_i) \/ V$ with $tilde(N)_"mode" = 4 pi k^2 Delta k V \/ (2 pi)^3$. This is the first term of eq. (49) of Sugiyama et al. (2020), with their top-hat $W(k, k')$ as $delta_(i i')$; the "5 perms." there are the other five pairings. With shot noise restored, $P -> P + 1 \/ overline(n)$ in each factor, as in their $P^((N))$.]

== Harmonic expansion

*Momenta of the unprimed estimator.* $overline(tilde(cal(S))_ell)(hat(q)_1, hat(q)_2, xh)$ carries $overline(Y)_(ell_1 m_1)(hat(q)_1) overline(Y)_(ell_2 m_2)(hat(q)_2) overline(Y)_(L M)(xh)$ and nothing else depends on $hat(q)_1, hat(q)_2$, so by @eq:pw
$ avg(e^(-i q_1 dot r_1) overline(Y)_(ell_1 m_1)(hat(q)_1))_i = (-i)^(ell_1) thin overline(j)^((i))_(ell_1)(r_1) thin overline(Y)_(ell_1 m_1)(hat(r)_1), $ <eq:qside>
and likewise for $q_2$. The unprimed estimator contributes only bin-averaged Bessel functions, with no power spectrum, as in the power-spectrum note.

*The vertex term.* The third correlator depends on $hat(p)_1$ and $hat(p)_2$ jointly, through $p_3 = p_1 + p_2$:
$ P(p_1 + p_2; xh) = sum_(L_3) (4 pi) / (2 L_3 + 1) sum_gamma overline(Y)_(L_3 gamma)(xh) thin [Y_(L_3 gamma)(hat(p)_3) P_(L_3)(|p_1 + p_2|)] . $
The bracket is a function of $(hat(p)_1, hat(p)_2)$ at fixed magnitudes that transforms as a rank-$L_3$ tensor; its bipolar expansion
$ Y_(L_3 gamma)(hat(p)_3) P_(L_3)(|p_1 + p_2|) = sum_(a b) c^(L_3)_(a b)(p_1, p_2) thin [Y_a (hat(p)_1) ⊗ Y_b (hat(p)_2)]_(L_3 gamma), \
  [Y_a ⊗ Y_b]_(L gamma) equiv sum_(alpha beta) chevron.l a alpha thin b beta | L gamma chevron.r Y_(a alpha)(hat(p)_1) Y_(b beta)(hat(p)_2), $ <eq:bipolar>
has coefficients $c^(L_3)_(a b)(p_1, p_2) = integral dif Omega_1 dif Omega_2 thin overline([Y_a ⊗ Y_b]_(L_3 gamma)) thin Y_(L_3 gamma)(hat(p)_3) P_(L_3)(|p_1 + p_2|)$ (independent of $gamma$), a one-dimensional integral over the opening angle after the rotational symmetry is used. Parity gives $a + b$ even, and $|a - b| <= L_3 <= a + b$. For $L_3 = 0$ the coefficients are the Legendre coefficients of $P_0 (|p_1 + p_2|)$ in $hat(p)_1 dot hat(p)_2$: $a = b$ and $c^0_(a a) prop p_a (p_1, p_2)$ where $P_0 (|p_1 + p_2|) = sum_a p_a (p_1, p_2) cal(L)_a (hat(p)_1 dot hat(p)_2)$. This expansion is the one new ingredient of the three-point case: it is finite in practice (the function is smooth in the opening angle; a few tens of terms resolve the BAO), and it is the only place where the two legs are coupled before the window is involved.

*Momenta of the primed estimator.* On leg 1 the factors of $hat(p)_1$ are $Y_(ell'_1 m'_1)(hat(p)_1)$ from $tilde(cal(S))_(ell')$, $Y_(L_1 M_1)(hat(p)_1)$ from $cal(L)_(L_1)(hat(p)_1 dot hat(y)_1) = (4 pi \/ (2 L_1 + 1)) sum_(M_1) Y_(L_1 M_1)(hat(p)_1) overline(Y)_(L_1 M_1)(hat(y)_1)$, $Y_(a alpha)(hat(p)_1)$ from @eq:bipolar, and the plane wave $e^(i p_1 dot r_1)$. Collapsing the three harmonics with @eq:product and applying @eq:pw,
$ avg(e^(i p_1 dot r_1) Y_(ell'_1 m'_1) Y_(L_1 M_1) Y_(a alpha)(hat(p)_1) thin g(p_1))_(i') = sum_(Lambda mu) G^(m'_1 M_1 alpha mu)_(ell'_1 L_1 a Lambda) thin i^Lambda thin overline(g)^((i'))_Lambda (r_1) thin overline(Y)_(Lambda mu)(hat(r)_1), $
and the same on leg 2 with $(ell'_2, m'_2, L_2, M_2, b, beta, Lambda', mu', r_2)$. Because the vertex coefficient $c^(L_3)_(a b)(p_1, p_2)$ depends on both magnitudes, the two radial shell averages do not factorise; they form one two-dimensional kernel,
$ cal(P)^((i' j'))_(L_1 L_2 L_3; a b; Lambda Lambda')(r_1, r_2) equiv (integral_(i') p_1^2 dif p_1 integral_(j') p_2^2 dif p_2 thin j_Lambda (p_1 r_1) thin j_(Lambda')(p_2 r_2) thin P_(L_1)(p_1) P_(L_2)(p_2) thin c^(L_3)_(a b)(p_1, p_2)) / (integral_(i') p_1^2 dif p_1 integral_(j') p_2^2 dif p_2) , $ <eq:P2d>
the analogue of $P^((i))_(L lambda)(s)$, now with the third power spectrum folded in through $c^(L_3)_(a b)$.

*Harmonics at the five directions.* After these steps the integrand of @eq:master1 is a product of harmonics of $hat(r)_1, hat(r)_2, hat(y)_1, hat(y)_2, xh$:
$ hat(r)_1 : overline(Y)_(ell_1 m_1) overline(Y)_(Lambda mu) = sum_(lambda_1 rho_1) G^(m_1 mu rho_1)_(ell_1 Lambda lambda_1) Y_(lambda_1 rho_1)(hat(r)_1), wide
  hat(r)_2 : overline(Y)_(ell_2 m_2) overline(Y)_(Lambda' mu') = sum_(lambda_2 rho_2) G^(m_2 mu' rho_2)_(ell_2 Lambda' lambda_2) Y_(lambda_2 rho_2)(hat(r)_2), $
$ hat(y)_1 : overline(Y)_(L_1 M_1)(hat(y)_1), wide hat(y)_2 : overline(Y)_(L_2 M_2)(hat(y)_2), wide
  xh : overline(Y)_(L M) Y_(L' M') overline(Y)_(L_3 gamma) = (-1)^(M') sum_(Lambda_3 mu_3) G^(M, -M', gamma, mu_3)_(L L' L_3 Lambda_3) Y_(Lambda_3 mu_3)(xh) . $
The phase $(-i)^(ell_1 + ell_2) i^(Lambda + Lambda')$ is real: $Lambda + Lambda' equiv a + b equiv 0$ (mod 2).

== Wigner--Eckart and the invariant window function

The coefficient of $Y_(lambda_1 rho_1)(hat(r)_1) Y_(lambda_2 rho_2)(hat(r)_2) overline(Y)_(L_1 M_1)(hat(y)_1) overline(Y)_(L_2 M_2)(hat(y)_2) Y_(Lambda_3 mu_3)(xh)$ is the tensor
$ T^(rho_1 rho_2 M_1 M_2 mu_3) = sum_(m_1 m_2 M thin m'_1 m'_2 M' thin alpha beta gamma thin mu mu')
  tj(ell_1, ell_2, L, m_1, m_2, M) tj(ell'_1, ell'_2, L', m'_1, m'_2, M') chevron.l a alpha thin b beta | L_3 gamma chevron.r \

  times G^(m'_1 M_1 alpha mu)_(ell'_1 L_1 a Lambda) G^(m'_2 M_2 beta mu')_(ell'_2 L_2 b Lambda') G^(m_1 mu rho_1)_(ell_1 Lambda lambda_1) G^(m_2 mu' rho_2)_(ell_2 Lambda' lambda_2) (-1)^(M') G^(M, -M', gamma, mu_3)_(L L' L_3 Lambda_3) , $ <eq:T>
built from Gaunt coefficients only, hence rotationally invariant. With five free magnetic indices the space of invariants is no longer one-dimensional: a coupling scheme is needed. The natural one follows the geometry --- leg 1 $(hat(r)_1, hat(y)_1)$, leg 2 $(hat(r)_2, hat(y)_2)$, and the anchor $xh$:
$ E^(rho_1 M_1 rho_2 M_2 mu_3)_(K_1 K_2) equiv sum_(kappa_1 kappa_2) chevron.l lambda_1 rho_1 thin L_1 M_1 | K_1 kappa_1 chevron.r chevron.l lambda_2 rho_2 thin L_2 M_2 | K_2 kappa_2 chevron.r chevron.l K_1 kappa_1 thin K_2 kappa_2 | Lambda_3 mu_3 chevron.r, $
in terms of which $T = sum_(K_1 K_2) cal(T)_(K_1 K_2) E_(K_1 K_2)$ with $cal(T)_(K_1 K_2) = (sum T E_(K_1 K_2)) \/ (sum E_(K_1 K_2)^2)$ (sums over all five indices; the $E$ for different $(K_1, K_2)$ are orthogonal). The invariant window function is the three-point correlation of the window weighted by the corresponding scalar,
$ cal(Q)^(lambda_1 L_1 K_1 ; lambda_2 L_2 K_2 ; Lambda_3)(r_1, r_2) equiv integral dif^3 x integral dif^2 hat(r)_1 dif^2 hat(r)_2 thin W(x) W(x + r_1) W(x + r_2) \

  times sum_(rho_1 M_1 rho_2 M_2 mu_3) E^(rho_1 M_1 rho_2 M_2 mu_3)_(K_1 K_2) thin Y_(lambda_1 rho_1)(hat(r)_1) overline(Y)_(L_1 M_1)(hat(y)_1) Y_(lambda_2 rho_2)(hat(r)_2) overline(Y)_(L_2 M_2)(hat(y)_2) Y_(Lambda_3 mu_3)(xh) , $ <eq:Q3>
the analogue of $cal(Q)_(Lambda_1 Lambda_2 Lambda)(s)$ with one more separation and one more line of sight. It depends on the geometry only.

== Result for class 1

$ C^((1))_(ell ell')(i j; i' j') = ((4 pi)^3 H H' sqrt(N N')) / I_3^2 sum_(L_1 L_2 L_3) ((4 pi)^3) / ((2 L_1 + 1)(2 L_2 + 1)(2 L_3 + 1))
  sum_(a b) sum_(Lambda Lambda') sum_(lambda_1 lambda_2 Lambda_3) sum_(K_1 K_2) (-1)^((Lambda + Lambda' - ell_1 - ell_2) \/ 2) thin cal(T)_(K_1 K_2) \
  times integral r_1^2 dif r_1 integral r_2^2 dif r_2 thin overline(j)^((i))_(ell_1)(r_1) thin overline(j)^((j))_(ell_2)(r_2) thin cal(P)^((i' j'))_(L_1 L_2 L_3; a b; Lambda Lambda')(r_1, r_2) thin cal(Q)^(lambda_1 L_1 K_1 ; lambda_2 L_2 K_2 ; Lambda_3)(r_1, r_2) , $ <eq:result1>
where $cal(T)$ carries all the multipole labels $(ell_1 ell_2 L; ell'_1 ell'_2 L'; L_1 L_2 L_3; a b; Lambda Lambda'; lambda_1 lambda_2 Lambda_3)$ and is a pure number computed once from 3-$j$ symbols. The selection rules are those of the Gaunt coefficients: $Lambda in "tri"(ell'_1, L_1, a)$, $lambda_1 in "tri"(ell_1, Lambda)$, $K_1 in "tri"(lambda_1, L_1)$, $Lambda_3 in "tri"(L, L', L_3) inter "tri"(K_1, K_2)$, and the same for leg 2. Every sum is finite except those over $(a, b)$, which are the Legendre orders of the third power spectrum in the opening angle.

*Shot noise.* Each of the three correlators is linear in its window-times-spectrum, so restoring $W P_L -> W P_L + S delta^K_(L 0)$ at each of the three points gives eight terms: at a leg, $S$ at that point with $L_a = 0$ and $P_(L_a) -> 1$ inside @eq:P2d; at the vertex, $S(x)$ with $L_3 = 0$ and $c^0_(a b) -> delta_(a 0) delta_(b 0) sqrt(4 pi)$ (the constant $Y_(00) P_0 -> Y_(00)$). The window functions become $cal(Q)^(X Y Z)$ with $X, Y, Z in {W, S}$ for the anchor and the two legs. The estimator's own shot-noise subtraction removes the self-pairs inside each estimator and leaves this structure intact, as it does for the power spectrum (Sugiyama et al. 2020, sec. 2.3).

= Class 2: a line of sight pairs with a shell galaxy

For the pairing $(1 1')(2 3')(3 2')$ the correlators are @eq:FF for $(q_1, p_1)$ at $y_1$, @eq:Fx for $(q_2, p_3)$ at $y_2$ with $p_3 = p_1 + p_2$, and @eq:FF read backwards for $(x, -p_2)$: $integral dif^3 y'_2 thin e^(i p_2 dot y'_2) chevron.l F(x) F(y'_2) chevron.r = e^(i p_2 dot x) W(x) P(p_2; xh)$. The line-of-sight harmonic of the primed estimator now sits at $hat(y)_2$. The phases combine into
$ e^(-i (q_1 - p_1) dot r_1) thin e^(-i (q_2 + p_1 + p_2) dot r_2) = e^(-i q_1 dot r_1) e^(-i q_2 dot r_2) e^(-i p_2 dot r_2) thin e^(-i p_1 dot (r_2 - r_1)) , $
and the master formula is
$ C^((2))_(ell ell') = ((4 pi)^3 H H' sqrt(N N')) / I_3^2 avg( cal(I)^((2)) )_(i j i' j'), wide
  cal(I)^((2)) equiv integral dif^3 x thin dif^3 r_1 thin dif^3 r_2 thin W(x) W(x + r_1) W(x + r_2) \
  times e^(-i (q_1 - p_1) dot r_1) e^(-i (q_2 + p_1 + p_2) dot r_2) thin overline(tilde(cal(S))_ell)(hat(q)_1, hat(q)_2, xh) thin tilde(cal(S))_(ell')(hat(p)_1, hat(p)_2, hat(y)_2) thin P(p_1; hat(y)_1) P(p_1 + p_2; hat(y)_2) P(p_2; xh) . $ <eq:master2>
The other three class-2 pairings follow by the two relabellings $(p_1, ell'_1, i') arrow.l.r (p_2, ell'_2, j')$ and $(q_1, ell_1, i, r_1) arrow.l.r (q_2, ell_2, j, r_2)$.

*Box limit.* $integral dif^3 r_1 dif^3 r_2$ gives $(2 pi)^6 delta_D (q_1 - p_1) delta_D (q_2 + p_1 + p_2)$: the shell of $q_2$ must contain the third side of the primed triangle. With the shell averages,
$ C^((2))_"box" = ((4 pi)^3 H H' sqrt(N N')) / (V V_i V_j) thin delta_(i i') avg( bb(1)[ |p_1 + p_2| in j ] thin overline(tilde(cal(S))_ell)(hat(p)_1, -hat(p)_3, nh) thin tilde(cal(S))_(ell')(hat(p)_1, hat(p)_2, nh) \
  times P(p_1) P(p_2) P(|p_1 + p_2|) )_(i j') , $
the terms of eq. (49) of Sugiyama et al. (2020) in which $k_3$ falls in a bin. On the diagonal $j' = j$ the indicator selects the configurations with $|p_1 + p_2| in j$, a fraction $tilde.op Delta k \/ (2 k)$ of the shell of $p_2$ (the third side ranges over $|p_1 - p_2| <= p_3 <= p_1 + p_2$, a width $tilde.op 2 k$): class 2 is suppressed by $tilde.op Delta k \/ k$ on the diagonal, $5%$ at $k = 0.1$ for $Delta k = 0.005$, and it is the only Gaussian contribution off the diagonal, where it is of the same relative size.

*Structure.* In the anchored basis the two plane waves of $p_1$ factorise, $e^(-i p_1 dot (r_2 - r_1)) = e^(i p_1 dot r_1) e^(-i p_1 dot r_2)$, so the $hat(p)_1$ average couples harmonics of $hat(r)_1$ and $hat(r)_2$:
$ integral (dif Omega_(p_1)) / (4 pi) e^(i p_1 dot r_1) e^(-i p_1 dot r_2) [Y dots (hat(p)_1)] = sum_(lambda lambda') i^(lambda - lambda') j_lambda (p_1 r_1) j_(lambda')(p_1 r_2) \
  times integral (dif Omega_(p_1)) / (4 pi) Y_(lambda)(hat(p)_1) overline(Y)_(lambda')(hat(p)_1) [Y dots (hat(p)_1)] thin overline(Y)_lambda (hat(r)_1) Y_(lambda')(hat(r)_2) . $
The Gaunt coefficient bounds $|lambda - lambda'|$ by the rank of the other harmonics of $hat(p)_1$, but not $lambda$ itself: the expansion in harmonics of the two anchored separations does not terminate (it is the Bessel addition theorem for $j_lambda (p_1 |r_2 - r_1|)$, which converges once $lambda > p_1 max(r_1, r_2)$). The rest of the derivation is as in class 1, with $cal(Q)$ functions whose line-of-sight labels sit at $xh$ for $L$ and at $hat(y)_2$ for $L'$. Which orders are needed in practice is decided by the window, not by the kernel: see the next section.

= Computing the window function from pair sums

The scalar in @eq:Q3 couples harmonics of $(hat(r)_1, hat(y)_1)$ into $K_1$, of $(hat(r)_2, hat(y)_2)$ into $K_2$, and the two into $Lambda_3$ at the anchor. Therefore
$ cal(Q)^(dots)(r_1, r_2) = sum_(kappa_1 kappa_2 mu_3) chevron.l K_1 kappa_1 thin K_2 kappa_2 | Lambda_3 mu_3 chevron.r integral dif^3 x thin W(x) thin Y_(Lambda_3 mu_3)(xh) thin A^((1))_(K_1 kappa_1)(x; r_1) thin A^((2))_(K_2 kappa_2)(x; r_2), $
$ A^((a))_(K kappa)(x; r) equiv integral dif^2 hat(r) thin W(x + r) sum_(rho M) chevron.l lambda_a rho thin L_a M | K kappa chevron.r Y_(lambda_a rho)(hat(r)) overline(Y)_(L_a M)(hat(y)) , $
and $A^((a))_(K kappa)(x; r)$ is a *pair* quantity: the harmonic moments of the window on the shell of radius $r$ around the anchor $x$, accumulated from the pairs $(x, x + r)$ exactly as the pair counts of `WindowLibrary`, in bins of $r$. The three-point function is then the sum over anchors of the product of two such moments. This is the Slepian--Eisenstein construction of the three-point function, and it needs no triplets: the cost is (anchors) $times$ (neighbours per anchor) for the moments, plus (anchors) $times$ (bins of $r_1$) $times$ (bins of $r_2$) $times$ (label pairs) for the products. The same construction holds in class 2, where the two plane waves of $p_1$ are each attached to one anchored side.

*What bounds the ranks.* The exact expansions contain unbounded sums: over $(a, b)$ in class 1 through $lambda_(1,2) in "tri"(ell, Lambda)$, and over $(lambda, lambda')$ in class 2. The product of a kernel of rank $lambda$ in $hat(r)_1$ with $cal(Q)$ integrates to zero unless $cal(Q)$ has that rank, and $cal(Q)$ is smooth in the opening angle of the triangle: the window varies on hundreds of Mpc$\/h$, so $cal(Q)^(dots)(r_1, r_2)$ at ranks beyond $lambda tilde.op 6$--$8$ is negligible for $r_1, r_2$ below the window scale, and exactly zero in a periodic box, where only the fully isotropic component $lambda_1 = lambda_2 = Lambda_3 = L_a = 0$ survives and reproduces the box formulae above (the isotropic component of the kernel is the full angular average, which is what the delta functions select). Truncating the window's ranks at $lambda_w$ therefore truncates every sum, with an error set by the window's angular smoothness and not by the kernels. The $(a, b)$ sums of class 1 are then bounded by $lambda_w + ell + L_a$, and the class-2 $(lambda, lambda')$ sums by $lambda_w$; alternatively the kernel's projection on the low-rank basis is computed by direct angular quadrature of @eq:master1 and @eq:master2 at each $(r_1, r_2)$, which avoids the Gaunt algebra for the primed momenta altogether. The convergence in $lambda_w$ is checked by computing $cal(Q)$ to a higher rank, at pair-count cost.

*Cost.* With $n_"sub" = 5000$ far randoms as anchors and neighbours: $2.5 times 10^7$ pair evaluations for the moments (as now), and for the products $5000 times n_r^2 times$ (label pairs) $times$ (magnetic contractions). For $lambda_w = 4$, $L_a <= 2$ and the diagonal elements the label pairs are $tilde.op 2500$ and the contraction $tilde.op 10^2$, i.e. $tilde.op 10^(11)$ operations per $100 times 100$ radial grid: minutes on a GPU, an hour on a node. The near-pair regime ($s < s_"split"$) uses the tree neighbours as now. Rotating each anchor's frame to $xh = hat(z)$ makes $mu_3 = 0$ and removes one magnetic sum.

= $B_(000)$ and $B_(202)$ on the diagonal

*$B_(000)$.* $ell_1 = ell_2 = L = 0$ on both sides, $H = 1$, $N = 1$. The six $Y_(00)$ factors supply $(4 pi)^(-3)$ and cancel the $(4 pi)^3$ of the prefactor. The $hat(r)_1$ collapse forces $lambda_1 = Lambda$, the $hat(r)_2$ one $lambda_2 = Lambda'$, and the anchor $Lambda_3 = L_3$, so @eq:result1 becomes
$ C^((1))_(000)(i j; i' j') = 1 / I_3^2 sum_(L_1 L_2 L_3) ((4 pi)^3) / ((2 L_1 + 1)(2 L_2 + 1)(2 L_3 + 1)) sum_(a b Lambda Lambda' K_1 K_2) (-1)^((Lambda + Lambda') \/ 2) cal(T)_(K_1 K_2) \

  times integral integral r_1^2 r_2^2 dif r_1 dif r_2 thin overline(j)^((i))_0 (r_1) overline(j)^((j))_0 (r_2) thin cal(P)^((i' j'))_(L_1 L_2 L_3; a b; Lambda Lambda') thin cal(Q)^(Lambda L_1 K_1; Lambda' L_2 K_2; L_3) , $
with $Lambda in "tri"(L_1, a)$, $Lambda' in "tri"(L_2, b)$. In real space ($L_1 = L_2 = L_3 = 0$, $a = b$, $Lambda = Lambda' = K_1 = K_2 = a$) the scalar of @eq:Q3 is proportional to $cal(L)_a (hat(r)_1 dot hat(r)_2)$: the window enters through its three-point function projected on Legendre polynomials of the opening angle, and the sum over $a$ is the Legendre expansion of $P(|p_1 + p_2|)$; in the box only $a = 0$ survives and the kernel reduces to the angle-averaged $P(|q_1 + q_2|)$. In redshift space the ranks are $L_(1,2,3) in {0, 2, 4}$, with $Lambda_3 = L_3$: the window's line-of-sight dependence at the anchor is the quadrupole and hexadecapole of the vertex correlator, and at the legs those of the two leg correlators.

*$B_(202)$.* $ell_1 = 2$, $ell_2 = 0$, $L = 2$ (and the same primed), $cal(S)_(202)(hat(k)_1, hat(k)_2, nh) = cal(L)_2 (hat(k)_1 dot nh)$, $H_(202) = -sqrt(1 \/ 5)$, $N = 25$. The 3-$j$ symbols are $tj(2, 0, 2, m_1, 0, -m_1) = (-1)^(m_1) \/ sqrt(5)$ and reduce the double sums to one magnetic index per estimator. The index sets: at $hat(r)_2$ still $lambda_2 = Lambda'$ (because $ell_2 = 0$); at $hat(r)_1$, $lambda_1 in "tri"(2, Lambda)$, with $Lambda in "tri"(2, L_1, a)$; at the anchor $Lambda_3 in "tri"(2, 2, L_3) inter "tri"(K_1, K_2)$, up to $4 + L_3$. The quadrupole weight on leg 1 is a rank-2 harmonic of $hat(r)_1$ in the unprimed kernel and a rank-2 harmonic of $hat(p)_1$ in the primed one; the anchor carries the two line-of-sight quadrupoles of the estimators and the vertex anisotropy. For $B_(202)$ the second class-1 pairing is not the mirror of the first ($ell_1 != ell_2$), so both must be evaluated; the diagonal of $B_(202)$ at $k_1 = k_2$ receives the swapped term with the roles of the two legs exchanged.

*Recipe for the diagonal.* (i) Pair counts with the anchored moments $A^((a))_(K kappa)(x; r)$ for $lambda <= lambda_w$, $L_a in {0, 2, 4}$, and the eight $(W, S)$ combinations; (ii) the invariants $cal(Q)$ for the labels above, contracted per anchor; (iii) the kernels: $overline(j)^((i))_(ell)$ from the shells, the two-dimensional $cal(P)$ of @eq:P2d from the model multipoles and the bipolar coefficients $c^(L_3)_(a b)(p_1, p_2)$ (a one-dimensional integral per $(a, b, L_3)$ and per node of the two shells), and for class 2 the projected kernel by quadrature; (iv) the coefficients $cal(T)$ from 3-$j$ symbols; (v) the double radial integral. The class-1 terms give the diagonal to $O(Delta k \/ k)$; adding the four class-2 terms completes the Gaussian part, on and off the diagonal.

= Non-Gaussian terms: $P B$, $B B$ and $P T$

The connected correlators localise. In the local approximation a connected three-point function of three fields puts all three within a correlation length of one point, so with the phases of the fields written out,
$ chevron.l F(k_a) F(k_b) F(k_c) chevron.r_c approx integral dif^3 y thin e^(-i (k_a + k_b + k_c) dot y) thin omega^3 (y) thin B(k_a, k_b; hat(y)), $ <eq:B3>
where $B(k_a, k_b; nh)$ is the bispectrum with third momentum $-(k_a + k_b)$ and line of sight $nh$. A configuration-space field $F(x)$ that carries an estimator phase $e^(i K dot x)$ is a Fourier field of momentum $-K$ at the point $x$; if it is the anchor the $y$ integral is absent and $omega^3 (x)$ appears instead, exactly as $W(x)$ in @eq:xx. The same rule gives the connected four-point function with $omega^4 (y)$ and the trispectrum $T(k_a, k_b, k_c; hat(y))$, the fourth momentum fixed by conservation. In the Sugiyama basis $B(k_a, k_b; nh) = sum_(ell'') B_(ell'')(k_a, k_b) thin cal(S)_(ell'')(hat(k)_a, hat(k)_b, nh)$ with $ell'' = (ell''_1 ell''_2 L'')$.

Two consequences hold for every term below. First, each term has exactly one plane wave and one separation, between the point of one connected factor and the point of the other, so *the window enters through a two-point function*, a pair count with the weights $(omega^2, omega^3)$, $(omega^3, omega^3)$ or $(omega^2, omega^4)$ in place of $(omega^2, omega^2)$: the functions $cal(Q)_(Lambda_1 Lambda_2 Lambda)(s)$ of the power-spectrum note with other weights, and no three-point window function. Second, with one separation and two line-of-sight points the invariant space is one-dimensional and the Wigner--Eckart step is the $3$-$j$ symbol of the power-spectrum note. The difficulty moves to the kernels: the bispectrum and trispectrum models, whose angular dependence in the shells is reduced by the same bipolar expansion as @eq:bipolar.

We write the power-spectrum estimator with the line of sight at the configuration-space point, $I_2 equiv integral dif^3 x thin omega^2$, and shells $i$ of $k$:
$ hat(P)_ell (i) = (2 ell + 1) / I_2 avg( integral dif^3 x thin e^(-i k dot x) cal(L)_ell (hat(k) dot xh) thin F(x) F(-k) )_i , $
so that $F(x)$ carries momentum $k$ at $x$ and $F(-k)$ is the Fourier field; its fields are labelled $a equiv F(-k)$ and $b equiv F(x)$.

== The $P B$ cross-covariance

$chevron.l hat(P)_ell (i) thin overline(hat(B)')_(ell')(i', j') chevron.r_c$ has five fields, $a, b$ and $1', 2', 3'$. There is no Gaussian term (five fields), and the leading term pairs one field of $hat(P)$ with one of $hat(B)'$ and leaves the other three connected; the pairings with the pair inside $hat(B)'$ are the beat-mode terms (window at $p_1 + p_2$, as in the Gaussian case) and the pairing $(a b)$ is disconnected. The six cross pairings come in four structures (the mirror $1' arrow.l.r 2'$ doubles the first two):
$ (a 1')(b 2' 3'), quad (b 1')(a 2' 3') quad & "leg: " k approx p_1 "(and " p_2 ")", \
  (a 3')(b 1' 2'), quad (b 3')(a 1' 2') quad & "third side: " k approx |p_1 + p_2| . $
For $(a 1')(b 2' 3')$ the two-point factor is @eq:FF with $q = -k$: $integral dif^3 y thin e^(i (k + p_1) dot y) W(y) P(p_1; hat(y))$, and the connected factor, anchored at the line-of-sight point $x$ of $hat(P)$ with the phases $e^(-i k dot x)$ of $hat(P)$ and $e^(-i (p_1 + p_2) dot x')$ of $hat(B)'$, is by @eq:B3
$ integral dif^3 x' thin e^(-i (p_1 + p_2) dot x') chevron.l F(x) F(-p_2) F(x') chevron.r_c approx omega^3 (x) thin e^(-i p_1 dot x) thin B(-p_2, p_1 + p_2; xh) , $
with the line-of-sight harmonic of $hat(B)'$ moved from $xh'$ to $xh$. The phases combine into $e^(i (k + p_1) dot s)$, $s equiv y - x$, and
$ C^(P B, (a 1'))_(ell, ell')(i; i' j') = ((2 ell + 1) (4 pi)^(3\/2) H' sqrt(N')) / (I_2 I_3) avg( cal(I)^(P B) )_(i i' j'), wide
  cal(I)^(P B) equiv integral dif^3 x thin dif^3 s thin omega^3 (x) W(x + s) thin e^(i (k + p_1) dot s) \
  times cal(L)_ell (hat(k) dot xh) thin tilde(cal(S))_(ell')(hat(p)_1, hat(p)_2, xh) thin P(p_1; hat(y)) thin B(-p_2, p_1 + p_2; xh) , $ <eq:PB>
$hat(y) = (x + s) \/ |x + s|$. The box limit, $integral dif^3 s -> (2 pi)^3 delta_D (k + p_1)$, gives $delta_(i i') \/ V_i$ and $P(k) B(k, -p_2, p_2 - k)$ at $hat(k) = -hat(p)_1$: the $P B$ terms of Sugiyama et al. (2020) with the leg coincidence; $(b 1')(a 2' 3')$ gives the same box limit with $W$ at the anchor and $omega^3$ at the far point, $P(p_1; xh) B(-k, -p_2; hat(y))$, the analogue of the two orderings $T_(A C, D B)$ and $T_(A D, C B)$ of the power-spectrum note. The third-side pairings carry $P(p_1 + p_2; dot)$ in the two-point factor and $B(-p_1, -p_2; dot)$ in the connected one, with the plane wave $e^(i (k - p_1 - p_2) dot s)$ (or $k + p_1 + p_2$): in the box they need $|p_1 + p_2| in i$, a fraction $tilde.op Delta k \/ (2 k)$ of the $p_2$ shell on the "diagonal" $i = j'$.

*Harmonic reduction of @eq:PB.* $cal(L)_ell (hat(k) dot xh) = (4 pi \/ (2 ell + 1)) sum_m Y_(ell m)(hat(k)) overline(Y)_(ell m)(xh)$ and the conjugate of @eq:pw give $avg(e^(i k dot s) Y_(ell m)(hat(k)))_i = i^ell thin overline(j)^((i))_ell (s) thin Y_(ell m)(hat(s))$. The bispectrum model is expanded in its own basis, $B(-p_2, p_1 + p_2; xh) = sum_(ell'') B_(ell'')(p_2, |p_1 + p_2|) thin (4 pi)^(3\/2) / (H'' sqrt(N'')) tilde(cal(S))_(ell'')(-hat(p)_2, hat(p)_3, xh)$, and the factor that depends on $hat(p)_3$ is bipolar in $(hat(p)_1, hat(p)_2)$ exactly as in @eq:bipolar,
$ Y_(ell''_2 m''_2)(hat(p)_3) thin B_(ell'')(p_2, |p_1 + p_2|) = sum_(a b) c^(ell'')_(a b)(p_1, p_2) thin [Y_a (hat(p)_1) ⊗ Y_b (hat(p)_2)]_(ell''_2 m''_2) . $
Then $hat(p)_2$ carries $Y_(ell'_2)$, $Y_(ell''_1)$ and $Y_b$ and no plane wave: its shell average is a Gaunt coefficient, $G^(m'_2 m''_1 beta)_(ell'_2 ell''_1 b) \/ (4 pi)$, with $b in "tri"(ell'_2, ell''_1)$. $hat(p)_1$ carries $Y_(ell'_1)$, $Y_(L_1)$ from $P(p_1; hat(y))$, $Y_a$ and the plane wave $e^(i p_1 dot s)$: as on leg 1 of class 1, $sum_Lambda G^(m'_1 M_1 alpha mu)_(ell'_1 L_1 a Lambda) thin i^Lambda thin overline(g)^((i'))_Lambda (s) overline(Y)_(Lambda mu)(hat(s))$. Three directions remain, $hat(s)$ with $Y_ell overline(Y)_Lambda -> lambda$, $xh$ with the three line-of-sight harmonics $overline(Y)_ell (xh) Y_(L' M')(xh) Y_(L'' M'')(xh) -> Lambda_x$, and $hat(y)$ with $overline(Y)_(L_1 M_1)$. The coefficient tensor has three magnetic indices and is $cal(T)_(lambda Lambda_x L_1) thin tj(lambda, Lambda_x, L_1, rho, mu_x, M_1)$ after the conjugated harmonics are written as $overline(Y)_(L M) = (-1)^M Y_(L, -M)$, so
$ C^(P B, (a 1'))_(ell, ell')(i; i' j') = ((2 ell + 1) (4 pi)^(3\/2) H' sqrt(N')) / (I_2 I_3) sum_(ell'' L_1 a b Lambda) ((4 pi)^(5\/2)) / ((2 ell + 1)(2 L_1 + 1) H'' sqrt(N'')) sum_(lambda Lambda_x) i^(ell + Lambda) thin cal(T)_(lambda Lambda_x L_1) \
  times integral s^2 dif s thin overline(j)^((i))_ell (s) thin cal(P)^((i' j'))_(L_1 ell''; a b; Lambda)(s) thin cal(Q)^(omega^3, W)_(Lambda_x L_1 lambda)(s), $ <eq:PBresult>
$ cal(P)^((i' j'))_(L_1 ell''; a b; Lambda)(s) equiv (integral_(i') p_1^2 dif p_1 integral_(j') p_2^2 dif p_2 thin j_Lambda (p_1 s) thin P_(L_1)(p_1) thin c^(ell'')_(a b)(p_1, p_2)) / (integral_(i') p_1^2 dif p_1 integral_(j') p_2^2 dif p_2) , $
with $cal(Q)^(omega^3, W)_(Lambda_x L_1 lambda)(s) = integral dif^3 x integral dif^2 hat(s) thin omega^3 (x) W(x + s) sum (3"-"j) Y_(Lambda_x)(xh) Y_(L_1)(hat(y)) Y_lambda (hat(s))$ the window function of the power-spectrum note with the weights $(omega^3, omega^2)$, and $cal(T)$ the contraction of the Gaunt coefficients of the chain above (the same product as @eq:T with one leg, three directions and the model's $3$-$j$ symbol in place of the second estimator's). The phase is real: $Lambda equiv ell'_1 + L_1 + a$, $a + b equiv ell''_2$, $b equiv ell'_2 + ell''_1$ (mod 2), so $ell + Lambda equiv L' + L'' equiv 0$. The radial kernel is one-dimensional: there is one separation. The third-side pairings have the same form with $P_(L_1)(p_1) -> P_(L_1)(|p_1 + p_2|)$ inside the bipolar coefficient and $B_(ell'')(p_1, p_2)$ with both arguments on the shells, i.e. with $c$ built from $Y(hat(p)_3) P_(L_1)(p_3)$ and the plane wave of $p_3$ carried by the pair $(hat(p)_1, hat(p)_2)$ jointly, which is where the box factor $Delta k \/ (2 k)$ comes from.

== The $B B$ term

The six fields split into two connected triples. Of the ten splittings, $(1 2 3)(1' 2' 3')$ is disconnected and the other nine each have one unprimed field with two primed ones. Writing the singleton of the unprimed estimator and the pair of the primed one, the coincidence that the plane wave enforces is
$ (3 | 1' 2'): k_3 approx k'_3, quad (3 | 1' 3'), (3 | 2' 3'): k_3 approx k'_2, k'_1, quad (1 | 1' 2'), (2 | 1' 2'): k_1, k_2 approx k'_3, \
  (1 | 2' 3'), (1 | 1' 3'), (2 | 1' 3'), (2 | 2' 3'): k_1 approx k'_1, k_1 approx k'_2, k_2 approx k'_2, k_2 approx k'_1 , $
where $k_3 = |q_1 + q_2|$ and $k'_3 = |p_1 + p_2|$ are the unbinned third sides. The four leg--leg pairings give the $delta_(i i')$-type terms of eq. (32) of Sugiyama et al. (2020) with full weight; the five that involve a third side are suppressed by $tilde.op Delta k \/ (2 k)$ on the diagonal, as class 2 is. For $(1 | 2' 3')$, with the first triple anchored at $x'$ and the second at $x$,
$ integral dif^3 x' e^(-i (p_1 + p_2) dot x') chevron.l F(q_1) F(-p_1) F(x') chevron.r_c & approx omega^3 (x') e^(-i (q_1 + p_2) dot x') B(q_1, -p_1; xh'), \
  e^(i (q_1 + q_2) dot x) chevron.l F(q_2) F(x) F(-p_2) chevron.r_c & approx omega^3 (x) e^(i (q_1 + p_2) dot x) B(q_2, -p_2; xh), $
so with $s = x' - x$
$ C^(B B, (1 | 2' 3'))_(ell ell')(i j; i' j') = ((4 pi)^3 H H' sqrt(N N')) / I_3^2 avg( cal(I)^(B B) )_(i j i' j'), wide
  cal(I)^(B B) equiv integral dif^3 x thin dif^3 s thin omega^3 (x) omega^3 (x + s) thin e^(-i (q_1 + p_2) dot s) \
  times overline(tilde(cal(S))_ell)(hat(q)_1, hat(q)_2, xh) thin tilde(cal(S))_(ell')(hat(p)_1, hat(p)_2, hat(y)) thin B(q_2, -p_2; xh) thin B(q_1, -p_1; hat(y)) , $ <eq:BB>
$hat(y) = (x + s) \/ |x + s|$: the coincidence is $q_1 approx -p_2$ ($i = j'$), each estimator's line of sight sits at its own anchor, the unprimed estimator's line of sight with $B(q_2, -p_2; xh)$ and the primed one's with $B(q_1, -p_1; hat(y))$. The box limit is $(delta_(i j') \/ V_i) thin avg(dots B(q_2, -p_2, p_2 - q_2) B(q_1, -p_1, p_1 - q_1) )$ with $hat(p)_2 = -hat(q)_1$, the first term of eq. (32).

*Reduction.* Each bispectrum is expanded in its basis and the third-side arguments, here $|p_2 - q_2|$ and $|p_1 - q_1|$ through $B(q_2, -p_2; xh) = sum B_(ell'')(q_2, p_2) cal(S)_(ell'')(hat(q)_2, -hat(p)_2, xh)$ with the magnitudes of the two binned momenta as arguments, need no bipolar expansion when both arguments are shell momenta: $B_(ell'')(q_2, p_2)$ depends on the magnitudes only and the opening angle is in $cal(S)_(ell'')$. In the pairings that involve a third side, one bispectrum argument is $|q_1 + q_2|$ or $|p_1 + p_2|$ and the bipolar expansion of @eq:bipolar applies to $Y(hat(q)_3) B_(ell'')(dot, q_3)$. In @eq:BB the momenta without a plane wave, $hat(q)_2$ and $hat(p)_1$, carry two harmonics each ($tilde(cal(S))$ and the model) and average to Gaunt coefficients; $hat(q)_1$ and $hat(p)_2$ carry the plane wave $e^(-i (q_1 + p_2) dot s)$ and give $overline(j)^((i))_(ell_1 + dots)$ and $overline(j)^((j'))_(dots)$ through @eq:pw, two Bessel averages at the same $s$ and no model power spectrum on the plane wave. The three remaining directions are $hat(s)$, $xh$ (with $overline(Y)_L$, $Y_(L'')$) and $hat(y)$ (with $Y_(L')$, $Y_(L''')$ of the two models' lines of sight), so
$ C^(B B) = sum (dots) thin cal(T)_(lambda Lambda_x Lambda_y) integral s^2 dif s thin cal(K)^((i j i' j'))(s) thin cal(Q)^(omega^3, omega^3)_(Lambda_x Lambda_y lambda)(s), $
with a kernel $cal(K)(s)$ built from products of $B_(ell'')(q, p)$ averaged over the two shells of each bispectrum and Bessel functions of the two plane-wave momenta, and $cal(Q)^(omega^3, omega^3)$ the pair-count window function with weights $omega^3$ at both points. On the diagonal $(i, j) = (i', j')$ the four leg--leg pairings contribute; each is a product of two bispectra at the binned sides, $B(k_1, k_2, k_3)^2$ in the box, so this is the term that grows fastest with the signal and the reason the Gaussian-only covariance fails at $k gt.tilde 0.1$.

== The $P T$ term

The six fields split into a cross pair and a connected four-point function of the other four (the pairs inside one estimator are again beat-mode terms). The nine cross pairs have the same coincidence table as the Gaussian pairings: four leg--leg, $(1 1'), (1 2'), (2 1'), (2 2')$; two leg--third, $(1 3'), (2 3')$; two third--leg, $(3 1'), (3 2')$; one third--third, $(3 3')$. For $(1 1')$ the pair is @eq:FF, $integral dif^3 y thin e^(-i (q_1 - p_1) dot y) W(y) P(p_1; hat(y))$, and the four-point function, anchored at $x$ with both configuration-space fields,
$ integral dif^3 x' thin e^(i (q_1 + q_2) dot x - i (p_1 + p_2) dot x') chevron.l F(q_2) F(x) F(-p_2) F(x') chevron.r_c \
  approx omega^4 (x) thin e^(i (q_1 - p_1) dot x) thin T(q_2, -p_2, p_1 + p_2; xh) , $
the trispectrum with fourth momentum $-(q_2 + p_1)$, which closes the quadrilateral exactly when $q_1 = p_1$. With $s = y - x$,
$ C^(P T, (1 1'))_(ell ell')(i j; i' j') = ((4 pi)^3 H H' sqrt(N N')) / I_3^2 avg( cal(I)^(P T) )_(i j i' j'), wide
  cal(I)^(P T) equiv integral dif^3 x thin dif^3 s thin omega^4 (x) W(x + s) thin e^(-i (q_1 - p_1) dot s) \
  times overline(tilde(cal(S))_ell)(hat(q)_1, hat(q)_2, xh) thin tilde(cal(S))_(ell')(hat(p)_1, hat(p)_2, xh) thin P(p_1; hat(y)) thin T(q_2, -p_2, p_1 + p_2; xh) , $ <eq:PT>
whose box limit $(delta_(i i') \/ V_i) avg(dots P(q_1) T(q_2, -p_2, q_1 + p_2, -q_1 - q_2))$ is the $P T$ term of eq. (33) of Sugiyama et al. (2020). The window enters through $cal(Q)^(omega^4, W)(s)$, the pair count with weights $(omega^4, omega^2)$, and the plane wave carries only the beat momentum $q_1 - p_1$: the whole angular structure of $T$ --- it depends on $hat(q)_2$, $hat(p)_2$, $hat(p)_1$ and $xh$ jointly --- is averaged over the shells of $q_2$ and $p_2$ with the estimator harmonics, and only its dependence on $hat(p)_1$ (through the plane wave, ranks bounded by the window) and on $xh$ (the line-of-sight ranks) survives into the Wigner--Eckart contraction. In practice this is a Monte Carlo over the shell triples, as in `thecov.trispectrum`, with the integrand projected on $Y_Lambda (hat(p)_1)$ and on the line-of-sight harmonics at $xh$ before the $s$ integral; the shell of $q_1$ enters only through $overline(j)^((i))_(ell_1)(s)$.

*The squeezed limit.* The beat momentum $epsilon = q_1 - p_1$ is not zero under the window: $T$ is evaluated on quadrilaterals whose momenta sum to $-epsilon$, and for $|epsilon| lt.tilde k_w$ it contains the response terms $prop P(epsilon)$ of the connected four-point function. Integrated against $|tilde(W)(epsilon)|^2$ these are the super-sample covariance of the bispectrum, the analogue of the $T_0$ piece of the power-spectrum covariance. Either $T$ is evaluated at $epsilon = 0$ (the standard $P T$ term) and the super-sample term is added separately, with its own responses, or the full $epsilon$ dependence is kept and no separate super-sample term is added. Doing both double counts; `thecov` does the former for $P$.

== Shot noise, models and calibration

*Shot noise.* Each connected correlator carries its Poisson terms, which in the local approximation change the weights: for the three-point function $omega^3 B -> omega^3 B + S thin omega [P(k_a) + P(k_b) + P(k_c)] + (1 - alpha^2) overline(n) w^3$, with $S = (1 + alpha) overline(n) w^2$ for the one-shared-galaxy terms (the pair window of the two coincident fields) and the three-shared-galaxy constant; for the four-point function the terms with one, two and three shared galaxies follow the same rule ($omega^4 T + S omega^2 B + dots$). `thecov.discreteness` has the one- and two-shared-galaxy pieces. The pair counts needed are therefore those with weights $omega^2, omega^3, omega^4$ and $S omega$, $S omega^2$, $S^2$, which `WindowLibrary` can accumulate in one pass.

*Models.* Tree level throughout, with the kernels already in `thecov`: $B = 2 Z_1 Z_1 Z_2 P P + 2$ perms. (SCF99, $b_1, b_2, b_(s^2), f$, in `discreteness.py` and `ssc.py`) and $T = T_(2211) + T_(3111)$ (`trispectrum.py`, Galileon bias), on the same damped $P_"lin"$ as the Gaussian term, with the finger-of-god damping applied to every $Z_1$ leg so that $P$, $B$ and $T$ are damped consistently. The $B B$ and $P B$ kernels need $B_(ell'')(k, k')$ in the Sugiyama basis on a two-magnitude grid, obtained once by angular quadrature of the model; the $P T$ kernel needs $T$ on general quadrilaterals, which the kernel code gives but the current integration driver (parallelograms $(k, -k, k', -k')$) does not. Loop corrections and the connected five- and six-point terms ($P_5$ in $P B$, $P_6$ in $B B$) are not worth their cost here: they are corrections to terms that are themselves corrections.

*Calibration from the data.* $B_(ell'')(k, k')$ can be taken from the measured multipoles, binned on the covariance grid and lightly smoothed, with the shot-noise pieces left in, as the measured $P$ replaces the model in the Gaussian term; the window convolution of the measurement matters only within a few $k_w$ of the window scale, and in noisy bins the tree-level model with $b_1, b_2, f$ fitted to $hat(B)$ is safer. $T$ cannot be measured, but the configurations the $P T$ term integrates over can: the squeezed part is the correlation of the local power spectrum with the local mean density (the position-dependent power spectrum), and the collapsed part is the covariance of the power spectrum across subvolumes of the survey, both available from the data at the ten-percent level. The residual is one amplitude per term, fitted to the mock covariance and checked against those data-side estimates.

= Checks and caveats

- *Box limit.* Both classes reduce to eq. (49) of Sugiyama et al. (2020) when $W = 1$; in the truncated scheme only the $lambda_1 = lambda_2 = 0$ components of $cal(Q)$ survive, with the closed form of the status box, so the box limit tests the kernels, the counting factors and the $cal(T)$ coefficients independently of the pair counts (a 4000-realisation Gaussian box reproduced the class-1 and class-2 box formulae for $B_(000)$, $B_(202)$, $B_(220)$ during the review).
- *Gaussian-field test.* As for the power spectrum, Gaussian realisations on the real footprint (`run_gaussian_footprint.py`) measured with the Sugiyama estimator give the exact Gaussian covariance of the windowed multipoles, including the window three-point effects at all ranks, and test $lambda_w$.
- *Which momentum carries the power spectrum.* In @eq:FF to @eq:xx the spectrum is evaluated at the primed momentum; the symmetric choice (the mean over the two shells, or half the sum of the two assignments) differs at $O(Delta k \/ k)$ and is the one to use for the final product, as for the power spectrum.
- *Beat-mode pairings.* Of the pairings with a pair inside each estimator, eight need a shell momentum below the window scale and vanish; $(1 2)(1' 2')(3 3')$ does not (status box, item v) and is kept as a seventh Gaussian term on the $(i i; i' i')$ elements. The same beat-mode pairings exist in the $P B$, $B B$ and $P T$ terms.
- *Scope.* The Gaussian term is derived in full; the $P B$, $B B$ and $P T$ terms are derived to the point where their window functions (pair counts) and kernels are identified, with the harmonic chain written out for $P B$ only; $P_5$ and $P_6$ are dropped; the super-sample term is separate (see the squeezed limit of $P T$). The window is the local one; a pair-averaged window enters, as for the power spectrum, through the dilution of the mean and through $W$ in @eq:Q3.
