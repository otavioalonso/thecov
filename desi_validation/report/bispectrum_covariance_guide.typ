#set document(title: "Covariance of windowed bispectrum multipoles: a guided derivation")
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
#let u = $bold(u)$
#let nh = $hat(bold(n))$
#let xh = $hat(bold(x))$
#let avg(body) = $lr(chevron.l body chevron.r)$
#let sh(body) = $lr(chevron.l chevron.l body chevron.r chevron.r)$
#let tj(a, b, c, d, e, f) = math.mat(delim: "(", (a, b, c), (d, e, f))
#let note(body) = block(fill: luma(242), inset: 8pt, radius: 3pt, width: 100%, body)
#let idea(body) = block(fill: rgb("#eaf2fb"), inset: 8pt, radius: 3pt, width: 100%, body)
#show table: set par(justify: false)
#show table: set text(size: 9.5pt)
#let result(body) = block(stroke: 0.6pt + rgb("#2b5797"), inset: 8pt, radius: 3pt, width: 100%, body)

#align(center)[
  #text(size: 16pt, weight: "bold")[Covariance of windowed bispectrum multipoles:\ a guided derivation]
  #v(0.3em)
  #text(size: 11pt)[A step-by-step companion to `bispectrum_covariance.typ`, with its corrections folded in]
  #v(0.3em)
  #text(size: 9.5pt, fill: luma(80))[Branch `thecov2`, October 2026. Conventions of `pk_covariance.typ` (power spectrum) and of Sugiyama, Saito, Beutler & Seo (2019, 2020) for the bispectrum basis.]
]

#v(0.6em)
#idea[
*Road map.* The derivation has the same skeleton as the power-spectrum one; each step is one idea.
+ *Geometry.* The estimator is a triangle anchored at the galaxy that carries the line of sight, with two legs $#r _1, #r _2$ ending at the galaxies whose momenta are binned (@sec:est).
+ *Locality.* Every two-point correlator is a window at one point times a locally anisotropic power spectrum; connected correlators collapse to a point (@sec:local).
+ *Wick.* Of the 15 Gaussian pairings of six fields, six connect the two estimators; one more survives on the $k_1 = k_2$ elements (@sec:wick).
+ *Plane waves on the legs.* In each pairing the phases recombine into one plane wave per leg of the triangle (@sec:class1).
+ *Shell averages.* Averaging the momenta of one estimator over their shells moves its angular weight from the momenta onto the legs, at the cost of bin-averaged Bessel functions. The other estimator brings the power spectra, including that of the unbinned third side (@sec:shell).
+ *Rotational invariance.* What multiplies the window is a rotational scalar of five directions. So only a few invariant projections of the window's three-point function are needed (@sec:inv), and these are computed from pair sums without triplets (@sec:pairs).
What is new relative to $P(k)$: a three-point window function, the third side $k_3 = |#k _1 + #k _2|$ that is not binned, and a "$k_3 approx 0$" term on the $k_1 = k_2$ elements. The non-Gaussian terms ($P B$, $B B$, $P T$) turn out to be *simpler*: they reduce to the power-spectrum formula with other window weights (@sec:ng).
]

#outline(depth: 1, indent: 1em)

= Notation and three tools <sec:not>

*Fourier and harmonics.* $tilde(f)(#k) = integral dif^3 x thin e^(-i #k dot #x) f(#x)$; orthonormal $Y_(ell m)$ with $overline(Y)_(ell m) = (-1)^m Y_(ell, -m)$; Legendre polynomials $cal(L)_ell$. All estimator and power-spectrum multipoles are even.

*Shell averages.* For a radial bin $i$ (with all directions),
$ sh(f)_i equiv (integral_i q^2 dif q integral dif Omega_q \/ (4 pi) thin f(#q)) / (integral_i q^2 dif q), quad V_i equiv integral_i (dif^3 q) / (2 pi)^3 = 1 / (2 pi^2) integral_i q^2 dif q . $
Nested averages over several bins are written $sh(f)_(i j)$ and so on. Radial bin averages of Bessel functions carry a bar:
$ overline(j)^((i))_lambda (r) equiv (integral_i q^2 dif q thin j_lambda (q r)) / (integral_i q^2 dif q), quad overline(g)^((i))_lambda (r) equiv (integral_i q^2 dif q thin j_lambda (q r) thin g(q)) / (integral_i q^2 dif q) . $

#note[
*Tool 1 (addition theorem).* $cal(L)_ell (hat(bold(a)) dot hat(bold(b))) = 4 pi \/ (2 ell + 1) sum_m Y_(ell m)(hat(bold(a))) overline(Y)_(ell m)(hat(bold(b)))$. The conjugate may sit on either factor.

*Tool 2 (plane wave over a shell, i.e. Funk--Hecke).* For any function $h_ell$ in the span of the $Y_(ell m)$ (for example $Y_(ell m)$ or $overline(Y)_(ell m)$),
$ sh(e^(plus.minus i #q dot #r) thin h_ell (hat(#q)) thin g(q))_i = (plus.minus i)^ell thin overline(g)^((i))_ell (r) thin h_ell (hat(#r)) . $ <eq:fh>
_A harmonic weight on the momentum reappears unchanged on the separation._ This is the workhorse of the whole derivation; with $g = 1$ it gives $overline(j)^((i))_ell$.

*Tool 3 (opening-angle expansion).* For a function of the opening angle, $f(hat(#p)_1 dot hat(#p)_2) = sum_a f_a cal(L)_a (hat(#p)_1 dot hat(#p)_2)$, two applications of @eq:fh and Tool 1 give
$ integral (dif Omega_1 dif Omega_2) / (4 pi)^2 thin e^(i #p _1 dot #r _1) e^(plus.minus i #p _2 dot #r _2) cal(L)_a (hat(#p)_1 dot hat(#p)_2) = (minus.plus 1)^a thin j_a (p_1 r_1) thin j_a (p_2 r_2) thin cal(L)_a (hat(#r)_1 dot hat(#r)_2) , $ <eq:open>
that is, $(-1)^a$ for two plane waves of the same sign and $+1$ for opposite signs. (Both signs were checked by Monte Carlo.)
]

*The Sugiyama basis.* With $y^m_ell = sqrt(4 pi \/ (2 ell + 1)) Y_(ell m)$, $H_(ell_1 ell_2 L) = tj(ell_1, ell_2, L, 0, 0, 0)$ and $N_(ell_1 ell_2 L) = (2 ell_1 + 1)(2 ell_2 + 1)(2 L + 1)$, the multipoles are defined by
$ B(#k _1, #k _2\; #nh) = sum_(ell) B_ell (k_1, k_2) thin cal(S)_ell (hat(#k)_1, hat(#k)_2, #nh), \
  cal(S)_ell (hat(bold(a)), hat(bold(b)), #nh) = 1 / H_ell sum_(m_1 m_2 M) tj(ell_1, ell_2, L, m_1, m_2, M) y^(m_1)_(ell_1)(hat(bold(a))) y^(m_2)_(ell_2)(hat(bold(b))) y^M_L (#nh), $
with $ell equiv (ell_1 ell_2 L)$. The $3$-$j$ symbol makes $cal(S)_ell$ a rotational scalar of its three arguments. It is real when $ell_1 + ell_2 + L$ is even; we keep the bar $overline(cal(S))$ anyway, for bookkeeping. Two things are used repeatedly:
- *Orthogonality.* At fixed $#nh$, averaging $overline(cal(S))_ell cal(S)_(ell'')$ over $hat(bold(a))$ and $hat(bold(b))$ gives $delta_(ell ell'') \/ (N_ell H_ell^2)$.
- *Unnormalised version.* $tilde(cal(S))_ell equiv sum tj(ell_1, ell_2, L, m_1, m_2, M) Y_(ell_1 m_1) Y_(ell_2 m_2) Y_(L M)$, so that $cal(S)_ell = (4 pi)^(3\/2) \/ (H_ell sqrt(N_ell)) thin tilde(cal(S))_ell$.

Examples: $cal(S)_(000) = 1$. $H_(ell 0 ell) = (-1)^ell \/ sqrt(2 ell + 1)$, so $H_(202) = +1 \/ sqrt(5)$, $N_(202) = 25$ and $cal(S)_(202)(hat(bold(a)), hat(bold(b)), #nh) = cal(L)_2 (hat(bold(a)) dot #nh)$. (All of these were checked numerically.)

*Fields and windows.* One tracer, $F(#x) = w(#x)[n_g (#x) - alpha n_r (#x)]$, real, so $F(-#k) = overline(F(#k))$. With $omega = overline(n) avg(w)$ the mean weighted density (`weight_average.typ`),
$ W = omega^2 thin "(pair window)", quad S = (1 + alpha) thin overline(n) avg(w^2) thin "(pair shot noise)", quad I_3 = integral dif^3 x thin omega^3 thin "(normalisation)". $

= The estimator is a triangle anchored at the line of sight <sec:est>

Sugiyama's estimator puts the line of sight on one galaxy, at $#x$, and bins the momenta $#q _1, #q _2$ of the other two:
$ hat(B)_ell (i, j) = ((4 pi)^(3\/2) H_ell sqrt(N_ell)) / I_3 thin sh( integral dif^3 x thin overline(tilde(cal(S)))_ell (hat(#q)_1, hat(#q)_2, #xh) thin e^(i (#q _1 + #q _2) dot #x) F(#q _1) F(#q _2) F(#x) )_(i j) . $ <eq:est>
Writing the two Fourier fields out,
$ e^(i (#q _1 + #q _2) dot #x) F(#q _1) F(#q _2) = integral dif^3 y_1 dif^3 y_2 thin e^(-i #q _1 dot (#y _1 - #x)) e^(-i #q _2 dot (#y _2 - #x)) F(#y _1) F(#y _2) . $
Each binned momentum is thus conjugate to one *leg* of the triangle, $#r _a = #y _a - #x$, measured from the anchor.

#note[
*Warm-up: the mean.* Assume the three-point function is local, $chevron.l F(#y _1) F(#y _2) F(#x) chevron.r approx omega^3 (#x) zeta(#r _1, #r _2\; #xh)$, i.e. the window is constant over the triangle. The legs then integrate to the bispectrum $B(#q _1, #q _2\; #xh)$ (third momentum $-#q _1 - #q _2$), and
$ chevron.l hat(B)_ell chevron.r = ((4 pi)^(3\/2) H_ell sqrt(N_ell)) / I_3 integral dif^3 x thin omega^3 (#x) thin sh(overline(tilde(cal(S)))_ell thin B)_(i j) = (integral omega^3) / I_3 thin sh(B_ell)_(i j) , $
by orthogonality. This fixes the prefactor of @eq:est. The window dilutes the multipole by $integral omega^3 \/ I_3 = 1$; a pair-averaged window would replace this by the bispectrum analogue of $I_k \/ "norm"$.
]

= The local approximation: three rules <sec:local>

We use the same approximations as for the power spectrum:
- *(A1)* The window is constant over a correlation length.
- *(A2)* Nearby points share the line of sight.
- *(A3)* Connected correlators are confined within a correlation length.

Under A1--A2, $chevron.l F(#y) F(#y + #u) chevron.r approx W(#y) thin xi(#u\; hat(#y))$ (plus $S(#y) delta_D (#u)$), with $P(#p\; #nh) = sum_L cal(L)_L (hat(#p) dot #nh) P_L (p)$ its transform. In the covariance the fields appear in three guises: Fourier fields, configuration fields integrated against an estimator phase, and the anchor. This gives three rules, each a one-line change of variables $#u = $ (separation) followed by $integral dif^3 u thin e^(-i #p dot #u) xi(#u) = P(#p)$:
$ "(R1)" quad & chevron.l F(#q) F(-#p) chevron.r approx integral dif^3 y thin e^(-i (#q - #p) dot #y) thin W(#y) thin P(#p\; hat(#y)), \
  "(R2)" quad & integral dif^3 x' thin e^(-i #p dot #x') chevron.l F(#q) F(#x') chevron.r approx integral dif^3 y thin e^(-i (#q + #p) dot #y) thin W(#y) thin P(#p\; hat(#y)), \
  "(R3)" quad & integral dif^3 x' thin e^(-i #p dot #x') chevron.l F(#x) F(#x') chevron.r approx e^(-i #p dot #x) thin W(#x) thin P(#p\; #xh) . $ <eq:rules>
#idea[
*Reading the rules.*
- A correlator is the window *at one point* times the anisotropic spectrum with the line of sight *at that point*.
- The spectrum is evaluated at the momentum of the partner field, conventionally the primed one. Choosing the other one, or the mean, changes the result at $O(Delta k \/ k)$.
- When a field that carries an estimator's line-of-sight harmonic is absorbed into a correlator anchored elsewhere, the harmonic moves to that anchor (A2).
]
Shot noise is restored at the end by $W P_L -> W P_L + S thin delta^K_(L 0)$ in each correlator: everything is linear in these window--spectrum pairs.

= Wick: which pairings survive <sec:wick>

The covariance is $C = chevron.l hat(B) thin overline(hat(B)') chevron.r - chevron.l hat(B) chevron.r chevron.l overline(hat(B)') chevron.r$. The primed estimator has shells $(i', j')$, multipoles $ell' = (ell'_1 ell'_2 L')$ and momenta $#p _1, #p _2$; conjugating @eq:est turns its phase into $e^(-i (#p _1 + #p _2) dot #x')$ and its fields into $F(-#p _1) F(-#p _2) F(#x')$. Label the six fields
$ 1 = F(#q _1), quad 2 = F(#q _2), quad 3 = F(#x), quad 1' = F(-#p _1), quad 2' = F(-#p _2), quad 3' = F(#x') . $
A Gaussian six-point function is a sum over the 15 ways of pairing six fields. Sorting them:
#align(center, table(columns: 3, align: (left, left, left), stroke: 0.5pt + luma(160), inset: 5pt,
  [*pairings*], [*what the momenta must satisfy*], [*status*],
  [class 1: $(1 1')(2 2')(3 3')$, $(1 2')(2 1')(3 3')$], [$#q _a approx #p _b$: leg to leg], [leading],
  [class 2: $(1 1')(2 3')(3 2')$ and 3 relabellings], [a leg of one triangle equals the _third side_ of the other], [$O(Delta k \/ k)$ on the diagonal],
  [$(1 2)(1' 2')(3 3')$], [$#q _1 + #q _2 approx 0$, $#p _1 + #p _2 approx 0$, i.e. $k_3, k'_3 approx 0$], [on $i = j$, $i' = j'$ only (@sec:k3)],
  [the other 8 with a pair inside one estimator], [a shell momentum below the window scale $k_w$], [vanish],
))
For example, $(1 3)$ forces $#q _2 approx 0$ after the $#x$ integral. Only $(1 2)(1' 2')(3 3')$ can be satisfied with all momenta in their shells, by the folded triangles with $k_3 -> 0$ that exist when $i = j$.

= Class 1: the two anchors pair together <sec:class1>

== Phases become plane waves on the legs

For $(1 1')(2 2')(3 3')$ use R1 for $(1 1')$ at $#y _1$ and for $(2 2')$ at $#y _2$, and R3 with $#p = #p _1 + #p _2$ for $(3 3')$, multiplied by the unprimed phase $e^(i (#q _1 + #q _2) dot #x)$. The phases combine as
$ e^(-i (#q _1 - #p _1) dot #y _1) thin e^(-i (#q _2 - #p _2) dot #y _2) thin e^(i (#q _1 + #q _2 - #p _1 - #p _2) dot #x) = e^(-i (#q _1 - #p _1) dot #r _1) thin e^(-i (#q _2 - #p _2) dot #r _2) . $
Each momentum pair rides a plane wave on one leg. The three windows sit at the three vertices of the triangle:
#result[
$ C^((1))_(ell ell') = & ((4 pi)^3 H H' sqrt(N N')) / I_3^2 \ & times integral dif^3 x thin dif^3 r_1 thin dif^3 r_2 thin W(#x) W(#x + #r _1) W(#x + #r _2) thin cal(K)^((1))(#r _1, #r _2\; hat(#y)_1, hat(#y)_2, #xh), $ <eq:master1>
$ cal(K)^((1)) = & sh( e^(-i #q _1 dot #r _1 - i #q _2 dot #r _2) thin overline(tilde(cal(S)))_ell (hat(#q)_1, hat(#q)_2, #xh) )_(i j) \ & times sh( e^(i #p _1 dot #r _1 + i #p _2 dot #r _2) thin tilde(cal(S))_(ell')(hat(#p)_1, hat(#p)_2, #xh) thin P(#p _1\; hat(#y)_1) thin P(#p _2\; hat(#y)_2) thin P(|#p _1 + #p _2|\; #xh) )_(i' j'), $
]
with $hat(#y)_a = (#x + #r _a) \/ |#x + #r _a|$. The double shell average has *factorised* into one average per estimator. This is the analogue of the "one plane wave per binned momentum" step of the power-spectrum note.

The second class-1 pairing $(1 2')(2 1')(3 3')$ is the same with $(#p _1, ell'_1, i') <-> (#p _2, ell'_2, j')$. It needs $i = j'$ and $j = i'$, so on a diagonal element $(i, j\; i, j)$ it contributes only if $i = j$, and then with equal weight.

#note[
*Check: the periodic box.* Set $W = 1$, $#xh = hat(#y)_a = #nh$, $I_3 = V$. The $#r _a$ integrals give $(2 pi)^3 delta_D (#q _a - #p _a)$, whose shell average is $delta_(i i') \/ V_i$:
$ C^((1))_"box" = & ((4 pi)^3 H H' sqrt(N N')) / (V V_i V_j) thin delta_(i i') delta_(j j') thin sh(overline(tilde(cal(S)))_ell tilde(cal(S))_(ell') P(#q _1) P(#q _2) P(|#q _1 + #q _2|))_(i j) \ = & (N N' H^2 H'^2 thin V) / (N_"mode"(k_i) N_"mode"(k_j)) delta_(i i') delta_(j j') sh(overline(cal(S))_ell cal(S)_(ell') P P P)_(i j) , $
with $N_"mode"(k_i) = V V_i$. This is the first term of eq. (49) of Sugiyama et al. (2020); the "5 perms." there are the other five pairings.
]

== Warm-up: $B_(000)$ in real space <sec:warm>

Everything essential is visible in the simplest case. Here $tilde(cal(S))_(000) = (4 pi)^(-3\/2)$, $H = N = 1$, and $P(#p\; #nh) = P(p)$. In $cal(K)^((1))$:
+ *Unprimed average.* By @eq:fh with $ell = 0$, $sh(e^(-i #q _1 dot #r _1))_i = overline(j)^((i))_0 (r_1)$, and the same for leg 2.
+ *Primed average.* The only coupling between the legs is the third side. Expand it in the opening angle, $P(|#p _1 + #p _2|) = sum_a p_a (p_1, p_2) cal(L)_a (hat(#p)_1 dot hat(#p)_2)$ with $p_a = (2 a + 1) \/ 2 integral_(-1)^1 dif mu thin cal(L)_a (mu) P(sqrt(p_1^2 + p_2^2 + 2 p_1 p_2 mu))$. Tool 3 (same signs) gives
  $ sh(dots)_(i' j') = sum_a (-1)^a thin cal(P)^((i' j'))_a (r_1, r_2) thin cal(L)_a (hat(#r)_1 dot hat(#r)_2) , $
  $ cal(P)^((i' j'))_a (r_1, r_2) equiv (integral_(i') p_1^2 dif p_1 integral_(j') p_2^2 dif p_2 thin j_a (p_1 r_1) j_a (p_2 r_2) P(p_1) P(p_2) p_a (p_1, p_2)) / (integral_(i') p_1^2 dif p_1 integral_(j') p_2^2 dif p_2) . $
+ *Window.* What remains of the window is its three-point function projected on the opening angle of the triangle:
#result[
$ C^((1))_(000) = 1 / I_3^2 sum_a (-1)^a integral r_1^2 dif r_1 integral r_2^2 dif r_2 thin overline(j)^((i))_0 (r_1) thin overline(j)^((j))_0 (r_2) thin cal(P)^((i' j'))_a (r_1, r_2) thin cal(Q)_a (r_1, r_2), $ <eq:b000>
$ cal(Q)_a (r_1, r_2) equiv integral dif^3 x integral dif Omega_(r_1) dif Omega_(r_2) thin W(#x) W(#x + #r _1) W(#x + #r _2) thin cal(L)_a (hat(#r)_1 dot hat(#r)_2) . $
]
In the box, $cal(Q)_a = (4 pi)^2 V delta_(a 0)$. With $4 pi integral r^2 dif r thin overline(j)^((i))_0 (r) thin overline(g)^((i'))_0 (r) = delta_(i i') sh(g)_i \/ V_i$, @eq:b000 becomes $delta_(i i') delta_(j j') sh(P P p_0)_(i j) \/ (V V_i V_j)$: the box result, in which only the *angle-averaged* third side $p_0$ survives. A window feeds the higher Legendre orders $a$ of the third side into the covariance, in proportion to the anisotropy of its three-point function, $cal(Q)_(a > 0)$. The rest of this section only adds indices to @eq:b000.

== General multipoles: the two shell averages <sec:shell>

*Unprimed estimator.* $overline(tilde(cal(S)))_ell$ is a product of $overline(Y)_(ell_1 m_1)(hat(#q)_1) overline(Y)_(ell_2 m_2)(hat(#q)_2)$ and a harmonic of $#xh$, and nothing else in its average depends on $hat(#q)_1, hat(#q)_2$. Tool 2 applied twice gives the exact statement
$ sh( e^(-i #q _1 dot #r _1 - i #q _2 dot #r _2) thin overline(tilde(cal(S)))_ell (hat(#q)_1, hat(#q)_2, #xh) )_(i j) = (-i)^(ell_1 + ell_2) thin overline(j)^((i))_(ell_1)(r_1) thin overline(j)^((j))_(ell_2)(r_2) thin overline(tilde(cal(S)))_ell (hat(#r)_1, hat(#r)_2, #xh) . $ <eq:unprimed>
_The estimator's angular weight is simply transferred from the momenta to the legs._

*Primed estimator.* Here three power spectra multiply the weight. Expand the two leg spectra with Tool 1, putting the conjugates on the momenta,
$ P(#p _a\; hat(#y)_a) = sum_(L_a) (4 pi) / (2 L_a + 1) sum_(M_a) overline(Y)_(L_a M_a)(hat(#p)_a) Y_(L_a M_a)(hat(#y)_a) P_(L_a)(p_a), $
and do the same for the vertex, $P(#p _3\; #xh) = sum_(L_3) 4 pi \/ (2 L_3 + 1) sum_gamma overline(Y)_(L_3 gamma)(hat(#p)_3) Y_(L_3 gamma)(#xh) P_(L_3)(p_3)$.

*The vertex is the one new ingredient.* $overline(Y)_(L_3 gamma)(hat(#p)_3) P_(L_3)(|#p _1 + #p _2|)$ is a function of both momenta, and it transforms as a rank-$L_3$ object. Its expansion in bipolar harmonics, the tensor generalisation of the opening-angle expansion in @sec:warm, is
$ overline(Y)_(L_3 gamma)(hat(#p)_3) thin P_(L_3)(p_3) = sum_(a b) c^(L_3)_(a b)(p_1, p_2) sum_(alpha beta) chevron.l a alpha thin b beta | L_3 gamma chevron.r thin overline(Y)_(a alpha)(hat(#p)_1) thin overline(Y)_(b beta)(hat(#p)_2) . $ <eq:bipolar>
The coefficients $c^(L_3)_(a b)$ are real and independent of $gamma$. Each is a one-dimensional integral over the opening angle. Parity requires $a + b + L_3$ even and $|a - b| <= L_3 <= a + b$. For $L_3 = 0$ they reduce to the Legendre coefficients $p_a$ of @sec:warm, up to normalisation. A few tens of orders resolve the BAO in the third side.

Each primed momentum now carries a plane wave and three harmonics: those of $tilde(cal(S))_(ell')$, of the leg spectrum, and of the vertex. Tool 2 moves their product to the leg. Let
$ g^(m M alpha nu)_(ell L a Lambda) equiv integral dif Omega thin Y_(ell m) thin overline(Y)_(L M) thin overline(Y)_(a alpha) thin overline(Y)_(Lambda nu) $
be the four-harmonic Gaunt integral that re-expands the product as harmonics of rank $Lambda$, with $Lambda <= ell + L + a$. The two radial averages *do not* factorise, because $c^(L_3)_(a b)$ depends on both magnitudes. They form one two-dimensional kernel:
$ cal(P)^((i' j'))_(L_1 L_2 L_3\; a b\; Lambda Lambda')(r_1, r_2) equiv (integral_(i') p_1^2 dif p_1 integral_(j') p_2^2 dif p_2 thin j_Lambda (p_1 r_1) j_(Lambda')(p_2 r_2) thin P_(L_1)(p_1) P_(L_2)(p_2) thin c^(L_3)_(a b)(p_1, p_2)) / (integral_(i') p_1^2 dif p_1 integral_(j') p_2^2 dif p_2) . $ <eq:P2d>
The primed average is then
$ sum_(L_1 L_2 L_3) ((4 pi)^3) / ((2 L_1 + 1)(2 L_2 + 1)(2 L_3 + 1)) sum_(a b) sum_(Lambda Lambda') i^(Lambda + Lambda') thin cal(P)^((i' j'))_(L_1 L_2 L_3\; a b\; Lambda Lambda')(r_1, r_2) thin cal(A)(hat(#r)_1, hat(#r)_2, hat(#y)_1, hat(#y)_2, #xh), $
where $cal(A)$ is a fixed polynomial in harmonics:
$ cal(A) = & sum tj(ell'_1, ell'_2, L', m'_1, m'_2, M') chevron.l a alpha thin b beta | L_3 gamma chevron.r thin g^(m'_1 M_1 alpha nu_1)_(ell'_1 L_1 a Lambda) thin g^(m'_2 M_2 beta nu_2)_(ell'_2 L_2 b Lambda') \ & times Y_(Lambda nu_1)(hat(#r)_1) Y_(Lambda' nu_2)(hat(#r)_2) Y_(L_1 M_1)(hat(#y)_1) Y_(L_2 M_2)(hat(#y)_2) Y_(L' M')(#xh) Y_(L_3 gamma)(#xh) . $

== Rotational invariance: why the window enters through a few numbers <sec:inv>

Collect the result: $cal(K)^((1))$ is a sum of products of radial functions (of $r_1, r_2$) and angular functions of the five directions $hat(#r)_1, hat(#y)_1, hat(#r)_2, hat(#y)_2, #xh$. Every ingredient is built from dot products, so $cal(K)^((1))$ is unchanged when all five directions are rotated together. A function of five directions with this property lies in the span of the *invariant* functions
$ Phi_n (hat(#r)_1, hat(#y)_1, hat(#r)_2, hat(#y)_2, #xh) = & sum tj(K_1, K_2, Lambda_3, kappa_1, kappa_2, mu_3) \ & times [Y_(lambda_1)(hat(#r)_1) times.o Y_(L_1)(hat(#y)_1)]_(K_1 kappa_1) [Y_(lambda_2)(hat(#r)_2) times.o Y_(L_2)(hat(#y)_2)]_(K_2 kappa_2) Y_(Lambda_3 mu_3)(#xh), $
where $[Y_lambda times.o Y_L]_(K kappa) = sum chevron.l lambda rho thin L M | K kappa chevron.r Y_(lambda rho) Y_(L M)$, and $n = (lambda_1 L_1 K_1\; lambda_2 L_2 K_2\; Lambda_3)$. The coupling tree follows the geometry: leg 1, leg 2, and the anchor. The $Phi_n$ are orthonormal on the five spheres. So
$ cal(K)^((1)) = sum_n cal(K)_n (r_1, r_2) thin Phi_n, quad cal(K)_n = integral dif^5 Omega thin overline(Phi)_n thin cal(K)^((1)) . $
The physical directions are not independent ($hat(#y)_a$ is fixed by $#x$ and $#r _a$), but this identity holds for independent directions, so it holds in particular for the physical ones. Inserting it in @eq:master1, the window is needed only through its projections on the same functions. This is the Wigner--Eckart step:
#result[
$ C^((1))_(ell ell')(i j\; i' j') = ((4 pi)^3 H H' sqrt(N N')) / I_3^2 sum_n integral r_1^2 dif r_1 integral r_2^2 dif r_2 thin cal(K)_n (r_1, r_2) thin cal(Q)_n (r_1, r_2), $ <eq:result1>
$ cal(Q)_n (r_1, r_2) equiv integral dif^3 x integral dif Omega_(r_1) dif Omega_(r_2) thin W(#x) W(#x + #r _1) W(#x + #r _2) thin Phi_n (hat(#r)_1, hat(#y)_1, hat(#r)_2, hat(#y)_2, #xh), $
$ cal(K)_n = & (-i)^(ell_1 + ell_2) thin overline(j)^((i))_(ell_1)(r_1) thin overline(j)^((j))_(ell_2)(r_2) \ & times sum_(L_3 a b Lambda Lambda') ((4 pi)^3 thin i^(Lambda + Lambda')) / ((2 L_1 + 1)(2 L_2 + 1)(2 L_3 + 1)) thin cal(P)^((i' j'))_(L_1 L_2 L_3\; a b\; Lambda Lambda')(r_1, r_2) thin T_n, $
$ T_n equiv integral dif^5 Omega thin overline(Phi)_n thin overline(tilde(cal(S)))_ell (hat(#r)_1, hat(#r)_2, #xh) thin cal(A) quad "(a pure number, from 3-j algebra or quadrature)". $
]
The $L_1, L_2$ of $cal(K)_n$ are the ranks of $n$ at $hat(#y)_1, hat(#y)_2$. Three kinds of object appear, cleanly separated:
- *Window* ($cal(Q)_n$): geometry only.
- *Radial kernels* ($overline(j)$, $cal(P)$): cosmology and binning.
- *Coupling numbers* ($T_n$): angular-momentum algebra, computed once.

*Selection rules.*
- $lambda_1 in "tri"(ell_1, Lambda)$ and $lambda_2 in "tri"(ell_2, Lambda')$ (merging $overline(Y)_(ell_a)$ with $Y_Lambda$ on the leg).
- $Lambda <= ell'_1 + L_1 + a$ and $Lambda' <= ell'_2 + L_2 + b$.
- $Lambda_3$ even, $Lambda_3 <= L + L' + L_3$.
- $K_a$ runs over *all* integers $|lambda_a - L_a| <= K_a <= lambda_a + L_a$. Two different directions are coupled here, so there is no parity restriction; for example, $(hat(#r)_1 times hat(#y)_1) dot (hat(#r)_2 times hat(#y)_2)$ is a legitimate invariant.

The phase $(-i)^(ell_1 + ell_2) i^(Lambda + Lambda')$ is real, because $Lambda + Lambda' equiv a + b equiv L_3 equiv 0$ (mod 2).

#note[
*Check: the periodic box.* With $W = 1$ and a global line of sight, the leg integrals $integral dif Omega_(r_a) Y_(lambda_a rho_a)(hat(#r)_a)$ force $lambda_1 = lambda_2 = 0$, hence $K_a = L_a$. Using $sum_(M_1 M_2 mu_3) tj(L_1, L_2, Lambda_3, M_1, M_2, mu_3) Y_(L_1 M_1) Y_(L_2 M_2) Y_(Lambda_3 mu_3)(#nh) = H_(L_1 L_2 Lambda_3) sqrt(N_(L_1 L_2 Lambda_3)) \/ (4 pi)^(3\/2)$ (checked numerically),
$ cal(Q)_n^"box" = V sqrt(N_(L_1 L_2 Lambda_3) \/ 4 pi) thin H_(L_1 L_2 Lambda_3) thin delta_(lambda_1 0) delta_(lambda_2 0) . $
Then $Lambda = ell_1$ and $Lambda' = ell_2$, and @eq:result1 reduces to the box formula above. The box therefore tests the kernels, the counting factors and $T_n$ independently of any pair count. This was done during the review with 4000 Gaussian boxes, for $B_(000)$, $B_(202)$ and $B_(220)$.
]

== Truncation is set by the window, not by the kernel <sec:trunc>

The vertex sums over $(a, b)$ do not terminate. But a kernel term of rank $lambda_1$ at $hat(#r)_1$ only survives if $cal(Q)_n$ has that rank. The window varies on scales of hundreds of $h^(-1)"Mpc"$, so the high-rank projections $cal(Q)_n$ are small. In a box only $lambda_a = 0$ survives. Keeping $lambda_1, lambda_2 <= lambda_w$ bounds every sum:
$ Lambda <= lambda_w + ell_1, quad a <= lambda_w + ell_1 + ell'_1 + L_1, quad b <= lambda_w + ell_2 + ell'_2 + L_2 . $
Convergence in $lambda_w$ is checked by computing $cal(Q)_n$ to higher rank, at pair-count cost, and against Gaussian realisations on the footprint. The truncation error is set by the window's angular smoothness on the scales $r tilde.op 1 \/ Delta k$ that dominate the near-diagonal elements.

== Shot noise

Each of the three correlators is linear in its window--spectrum pair. Replacing $W P_L -> W P_L + S delta^K_(L 0)$ at the two legs and the anchor gives eight terms:
- A shot-noise *leg* has $L_a = 0$ and $P_(L_a) -> 1$ in @eq:P2d.
- A shot-noise *anchor* has $L_3 = 0$ and $c^0_(a b) -> sqrt(4 pi) thin delta_(a 0) delta_(b 0)$.
The window functions become $cal(Q)_n^(X Y Z)$ with $X, Y, Z in {W, S}$.

= Class 2: an anchor pairs with a leg <sec:class2>

Take $(1 1')(2 3')(3 2')$. R1 gives $(1 1')$ at $#y _1$. R2 with $#p = #p _3 equiv #p _1 + #p _2$ gives $(2 3')$ at $#y _2$, where the primed line of sight now sits. R1 read backwards gives $integral dif^3 y' e^(i #p _2 dot #y') chevron.l F(#x) F(#y') chevron.r = e^(i #p _2 dot #x) W(#x) P(#p _2\; #xh)$. The phases are
$ e^(-i (#q _1 - #p _1) dot #r _1) thin e^(-i (#q _2 + #p _3) dot #r _2) . $
*Change variables from $#p _2$ to $#p _3$.* The Jacobian is 1, and the shell condition on $#p _2$ becomes the indicator $bb(1)[|#p _3 - #p _1| in j']$, while $#p _3$ runs over all magnitudes. Now each leg again carries exactly one primed momentum ($#p _1$ on leg 1, $#p _3$ on leg 2), and the structure is that of class 1:
#result[
$ C^((2))_(ell ell') = & ((4 pi)^3 H H' sqrt(N N')) / I_3^2 \ & times integral dif^3 x thin dif^3 r_1 thin dif^3 r_2 thin W(#x) W(#x + #r _1) W(#x + #r _2) thin cal(K)^((2))(#r _1, #r _2\; hat(#y)_1, hat(#y)_2, #xh), $ <eq:master2>
$ cal(K)^((2)) = & (-i)^(ell_1 + ell_2) thin overline(j)^((i))_(ell_1)(r_1) thin overline(j)^((j))_(ell_2)(r_2) thin overline(tilde(cal(S)))_ell (hat(#r)_1, hat(#r)_2, #xh) \
  & times sh( e^(i #p _1 dot #r _1 - i #p _3 dot #r _2) thin tilde(cal(S))_(ell')(hat(#p)_1, hat(#p)_2, hat(#y)_2) thin P(#p _1\; hat(#y)_1) thin P(#p _3\; hat(#y)_2) thin P(#p _2\; #xh) )_(i', #p _3), $
]
where the last average runs over $#p _1 in i'$ and $#p _3$ with $|#p _3 - #p _1| in j'$, normalised by $integral_(i') p_1^2 dif p_1 integral_(j') p_2^2 dif p_2$. Everything that depends on $#p _2 = #p _3 - #p _1$ forms the *class-2 vertex*: the indicator, $Y_(ell'_2)(hat(#p)_2)$ from $tilde(cal(S))_(ell')$, and $P(#p _2\; #xh)$. It is expanded in bipolar harmonics of $(hat(#p)_1, hat(#p)_3)$ exactly as in @eq:bipolar.
- The five directions are the same as in class 1, so the *same invariant window functions* $cal(Q)_n$ appear. Only the rank assignment differs: $hat(#y)_2$ carries the primed line of sight coupled with the leg-2 spectrum, and $#xh$ carries the unprimed line of sight with the vertex spectrum.
- In real space, $B_(000)$ gives @eq:b000 with $(-1)^a -> +1$ (Tool 3, opposite signs) and $cal(P)_a -> cal(P)^((2))_a$. Here $j_a (p_3 r_2)$ is integrated over all $p_3$, and $p_a$ is replaced by the Legendre coefficients of $bb(1)[|#p _3 - #p _1| in j'] P(|#p _3 - #p _1|)$ in $hat(#p)_1 dot hat(#p)_3$.
- The other three class-2 pairings follow by $(#p _1, ell'_1, i') <-> (#p _2, ell'_2, j')$ and $(#q _1, ell_1, i, #r _1) <-> (#q _2, ell_2, j, #r _2)$.

#note[
*Check: the box, and why class 2 is small on the diagonal.* The leg integrals force $#q _1 = #p _1$ and $#q _2 = -#p _3$: the shell of $#q _2$ must contain the *third side* of the primed triangle,
$ C^((2))_"box" = & ((4 pi)^3 H H' sqrt(N N')) / (V V_i V_j) thin delta_(i i') \ & times sh(bb(1)[|#p _1 + #p _2| in j] thin overline(tilde(cal(S)))_ell (hat(#p)_1, -hat(#p)_3, #nh) thin tilde(cal(S))_(ell')(hat(#p)_1, hat(#p)_2, #nh) thin P(p_1) P(p_2) P(p_3))_(i j') . $
Since $p_3^2 = p_1^2 + p_2^2 + 2 p_1 p_2 mu$, the indicator keeps a fraction $Delta k thin p_3 \/ (2 p_1 p_2) approx Delta k \/ (2 k)$ of the opening angles. Counting the pairings that survive (two of class 2 against one of class 1 for $i != j$; four against two for $i = j$), class 2 is $approx Delta k \/ k$ of class 1 on the diagonal, i.e. 5% at $k = 0.1$ with $Delta k = 0.005$. Off the diagonal it is the only Gaussian contribution.
]

= The $k_3 approx 0$ term <sec:k3>

For $(1 2)(1' 2')(3 3')$, R1 gives $chevron.l F(#q _1) F(#q _2) chevron.r = integral dif^3 y thin e^(-i bold(epsilon) dot #y) W(#y) P(#q _2\; hat(#y))$ with $bold(epsilon) = #q _1 + #q _2$. This needs $|bold(epsilon)| lt.tilde k_w$, i.e. folded triangles with $#q _2 approx -#q _1$, which exist only if $i = j$. The average over the $#q _2$ shell turns $integral dif^3 epsilon \/ (2 pi)^3 e^(i bold(epsilon) dot (#x - #y))$ into $delta_D (#x - #y) \/ V_i$. The same happens on the primed side, and the anchors remain connected by $(3 3')$:
#result[
$ C^((7))(i i\; i' i') = & ((4 pi)^3 H H' sqrt(N N')) / (I_3^2 V_i V_(i')) \ & times integral dif^3 x thin dif^3 x' thin W(#x) W(#x') thin sh(overline(tilde(cal(S)))_ell (hat(#q), -hat(#q), #xh) P(#q\; #xh))_i \ & times sh(tilde(cal(S))_(ell')(hat(#p), -hat(#p), #xh') P(#p\; #xh'))_(i') thin chevron.l F(#x) F(#x') chevron.r_"IC" . $
]
#idea[
*What this is.* On the $k_1 = k_2$ elements the Sugiyama multipole includes the triangles with $k_3 -> 0$, where $hat(B)$ is the local power $P(k)$ times the $W$-weighted mean overdensity of the survey.
- In a box with a fixed mean the latter vanishes (the integral constraint, "IC").
- In the survey, $alpha$ forces $integral F = 0$ but not $integral omega^2 F = 0$, so the term is zero for uniform $omega$.
- It reached tens of percent of the $(i, i)$ variance in a Gaussian test with a radially varying $omega$.

It has the same mode-counting factor $1 \/ (V_i V_(i'))$ as class 1, so it is not suppressed by $Delta k \/ k$.
]

= The window invariants from pair sums <sec:pairs>

The coupling tree of $Phi_n$ was chosen so that $cal(Q)_n$ factorises at each anchor:
$ cal(Q)_n (r_1, r_2) = integral dif^3 x thin W(#x) sum_(kappa_1 kappa_2 mu_3) tj(K_1, K_2, Lambda_3, kappa_1, kappa_2, mu_3) thin Y_(Lambda_3 mu_3)(#xh) thin A^(lambda_1 L_1)_(K_1 kappa_1)(#x\; r_1) thin A^(lambda_2 L_2)_(K_2 kappa_2)(#x\; r_2), $
$ A^(lambda L)_(K kappa)(#x\; r) equiv integral dif Omega_r thin W(#x + #r) thin [Y_lambda (hat(#r)) times.o Y_L (hat(#y))]_(K kappa) . $
$A$ is a *pair* quantity: the harmonic moments of the window on the sphere of radius $r$ about the anchor. It is accumulated from the pairs (anchor, neighbour) in bins of $r$, exactly like the pair counts of `WindowLibrary`. The three-point function is the sum over anchors of products of two such moments, so no triplets are counted (the Slepian--Eisenstein construction). Practical points:
- *Frame.* Rotate each anchor's frame so that $#xh = hat(bold(z))$. Then $Y_(Lambda_3 mu_3)(#xh) prop delta_(mu_3 0)$, which removes one magnetic sum.
- *Self-pairs.* When $r_1$ and $r_2$ fall in the same bin, the product $A A$ contains the terms with the same neighbour in both factors. These have zero measure in the continuum and must be subtracted.
- *Cost.* With 5000 anchors and the neighbours of the power-spectrum pair counts, the moments cost as now. The products cost about $10^(12)$ operations for $lambda_w = 4$, $L_a <= 2$ on a $100 times 100$ radial grid. The near-pair regime uses the tree neighbours as now.
- *Weights.* The eight shot-noise combinations need the moments with $W$ or $S$ at the neighbour and the anchor.

= Worked index sets: $B_(000)$ and $B_(202)$ <sec:examples>

*$B_(000)$.* $tilde(cal(S))_(000) = (4 pi)^(-3\/2)$ on both sides cancels the $(4 pi)^3$ of the prefactor. Leg 1 forces $lambda_1 = Lambda in "tri"(L_1, a)$ and leg 2 forces $lambda_2 = Lambda' in "tri"(L_2, b)$; at the anchor $Lambda_3 = L_3$. In real space this is @eq:b000. In redshift space $L_(1, 2, 3) in {0, 2, 4}$: the line-of-sight ranks of the window function at the two far vertices and at the anchor are those of the two leg spectra and of the vertex spectrum.

*$B_(202)$.* Here $H = 1 \/ sqrt(5)$, $N = 25$, $cal(S)_(202) = cal(L)_2 (hat(#k)_1 dot #nh)$, and by @eq:unprimed the unprimed weight becomes $-overline(j)^((i))_2 (r_1) overline(j)^((j))_0 (r_2) thin cal(L)_2 (hat(#r)_1 dot #xh)$ up to normalisation: a quadrupole of leg 1 about the anchor's line of sight. Then:
- Leg 2 still has $lambda_2 = Lambda'$.
- Leg 1 has $lambda_1 in "tri"(2, Lambda)$ with $Lambda <= 2 + L_1 + a$.
- At the anchor, $Lambda_3$ is even and $Lambda_3 <= 4 + L_3$.

Since $ell_1 != ell_2$, the second class-1 pairing is not the mirror of the first and must be evaluated separately (it contributes on $i = j$).

= Non-Gaussian terms: back to the power-spectrum formula <sec:ng>

*The localisation rule (A3).* A connected correlator of three fields is confined to one point:
$ chevron.l F(#k _a) F(#k _b) F(#k _c) chevron.r_c approx integral dif^3 y thin e^(-i (#k _a + #k _b + #k _c) dot #y) thin omega^3 (#y) thin B(#k _a, #k _b\; hat(#y)) . $
For four fields the same holds with $omega^4$ and the trispectrum. A configuration field with estimator phase $e^(i bold(K) dot #x)$ acts as a Fourier field of momentum $-bold(K)$ at $#x$. If the correlator contains the anchor, there is no $#y$ integral and $omega^3 (#x)$ appears.

#idea[
*Why these terms are simpler than the Gaussian one.* $P B$ is a pair plus a connected triple, $B B$ is two connected triples, and $P T$ is a pair plus a connected quadruple. In each there are exactly *two points*, so there is *one separation* $#s$ and one plane wave. The window enters through a pair count with weights $(omega^3, W)$, $(omega^3, omega^3)$ or $(omega^4, W)$, i.e. the functions $cal(Q)^(omega omega')_(Lambda_1 Lambda_2 Lambda)(s)$ of the power-spectrum note with other weights. There is no three-point window function. The momenta that do not carry the plane wave are averaged over their shells *before* anything else, and they turn the bispectrum or trispectrum into an *effective spectrum*. What is left is literally the power-spectrum shell kernel $cal(K)^((i))_(ell L)[p](hat(bold(a)), hat(bold(b))\; #s)$ of `pk_covariance.typ` (its eq. "kernel-def"), and its final formula applies.
]

One more approximation is needed, at the order already accepted:
- *(A4)* Inside slowly varying factors, set the beat momentum ($bold(epsilon) = $ the sum of the two momenta tied by the plane wave, $|bold(epsilon)| lt.tilde k_w$) to zero.

This is the same $O(k_w \/ k)$ freedom as "which momentum carries the spectrum". For $P T$ it is exactly the separation from super-sample covariance (below).

== $P B$

With $hat(P)_ell (i) = (2 ell + 1) \/ I_2 thin sh(integral dif^3 x thin e^(-i #k dot #x) cal(L)_ell (hat(#k) dot #xh) F(#x) F(-#k))_i$ and $I_2 = integral omega^2$, label $a = F(-#k)$ and $b = F(#x)$. Five fields have no Gaussian part. The leading term pairs one field of $hat(P)$ with one of $hat(B)'$ and connects the other three. The pairings inside $hat(B)'$ are beat modes and $(a b)$ is disconnected. That leaves six:
#align(center, table(columns: 4, align: left, stroke: 0.5pt + luma(160), inset: 5pt,
  [*pairing*], [*points: pair / triple*], [*plane wave*], [*box coincidence*],
  [$(a 1')(b 2' 3')$, mirror $1' <-> 2'$], [$W(#x + #s)$ / $omega^3 (#x)$], [$e^(i (#k + #p _1) dot #s)$], [$k approx p_1$: full weight],
  [$(b 1')(a 2' 3')$, mirror], [$W(#x)$ / $omega^3 (#x + #s)$], [$e^(i (#k - #p _1) dot #s)$], [$k approx p_1$: full weight],
  [$(a 3')(b 1' 2')$, $(b 3')(a 1' 2')$], [as above], [carries $#k$ and $#p _3$], [$k approx p'_3$: $O(Delta k \/ k)$],
))
For $(a 1')(b 2' 3')$, R1 gives the pair and the localisation rule gives the triple, anchored at $#x$, $omega^3 (#x) e^(-i #p _1 dot #x) B(-#p _2, #p _1 + #p _2\; #xh)$, with the $hat(B)'$ line of sight moved to $#xh$. Only $#k$ and $#p _1$ carry the plane wave, so average over $#p _2$ first:
$ beta(#p _1\; #xh) equiv sh(tilde(cal(S))_(ell')(hat(#p)_1, hat(#p)_2, #xh) thin B(-#p _2, #p _1 + #p _2\; #xh))_(j') = sum_J beta_J (p_1) thin cal(L)_J (hat(#p)_1 dot #xh) , $
a Legendre series because it is a rotational scalar of two directions. Then
#result[
$ C^(P B, (a 1'))_(ell, ell')(i\; i' j') = & ((2 ell + 1)(4 pi)^(3\/2) H' sqrt(N')) / (I_2 I_3) sum_(L_1 J) \ & times integral dif^3 x thin dif^3 s thin omega^3 (#x) W(#x + #s) \ & times cal(K)^((i))_(ell 0)[1](#xh, dot\; #s) thin cal(K)^((i'))_(J L_1)[P_(L_1) beta_J](#xh, hat(#y)\; #s), $
]
with $hat(#y) = (#x + #s) \/ |#x + #s|$. This is the power-spectrum master formula with weights $(omega^3, W)$ and effective spectrum $P_(L_1) beta_J$. In the box it gives $delta_(i i') \/ V_i$ times $P(k) B(k, p_2, |#k - #p _2|)$, averaged with the weights: the $P B$ term of Sugiyama et al. (2020).

The pairings $(b 1')(a 2' 3')$ are the same with the weights swapped, $W(#x)$ and $omega^3 (#x + #s)$, and the $hat(B)'$ line of sight at $hat(#y)$. In the third-side pairings use $(#p _1, #p _3)$: the plane-wave momentum is $#p _3$, integrated over all magnitudes with the indicator, which is where the $Delta k \/ k$ comes from.

== $B B$

The six fields split into two connected triples in 10 ways. $(1 2 3)(1' 2' 3')$ is $chevron.l hat(B) chevron.r chevron.l hat(B)' chevron.r$; the other nine each join one unprimed field to two primed ones. The triple containing the unprimed singleton $u$ and the primed pair $(a', b')$ forces $k_u approx k'_(c')$, where $c'$ is the remaining primed field:
#align(center, table(columns: 3, align: left, stroke: 0.5pt + luma(160), inset: 5pt,
  [*singleton $|$ pair*], [*coincidence*], [*weight on the diagonal $(i, j) = (i', j')$*],
  [$(1 | 2' 3')$, $(2 | 1' 3')$], [$k_1 approx k'_1$, $k_2 approx k'_2$], [full],
  [$(1 | 1' 3')$, $(2 | 2' 3')$], [$k_1 approx k'_2$, $k_2 approx k'_1$], [full if $i = j$, else zero],
  [$(3 | 1' 3')$, $(3 | 2' 3')$, $(1 | 1' 2')$, $(2 | 1' 2')$, $(3 | 1' 2')$], [involve a third side], [$O(Delta k \/ k)$],
))
Take $(1 | 2' 3')$: the triple $(1, 2', 3')$ is anchored at $#x'$ and the triple $(2, 3, 1')$ at $#x$. Their phases give $e^(-i (#q _1 + #p _1) dot #s)$ with $#s = #x' - #x$, so $#q _1 approx -#p _1$. Each estimator's line of sight stays at its own anchor. With A4 the two bispectra become the unprimed and primed triangles, and the averages over the non-plane-wave momenta define two effective spectra:
$ beta^u (#q _1\; #xh) = sh(overline(tilde(cal(S)))_ell (hat(#q)_1, hat(#q)_2, #xh) B(#q _1, #q _2\; #xh))_j, quad beta^p (#p _1\; hat(#y)) = sh(tilde(cal(S))_(ell')(hat(#p)_1, hat(#p)_2, hat(#y)) B(-#p _1, -#p _2\; hat(#y)))_(j') , $
each a Legendre series in its momentum and line of sight. Then
#result[
$ C^(B B, (1 | 2' 3'))_(ell ell') = & ((4 pi)^3 H H' sqrt(N N')) / I_3^2 sum_(J J') \ & times integral dif^3 x thin dif^3 s thin omega^3 (#x) omega^3 (#x + #s) \ & times cal(K)^((i))_(J 0)[beta^u_J](#xh, dot\; -#s) thin cal(K)^((i'))_(J' 0)[beta^p_(J')](hat(#y), dot\; -#s), $
]
which uses the power-spectrum formula with weights $(omega^3, omega^3)$. In the box this is $delta_(i i') \/ V_i$ times the product of the two bispectra averaged with the estimator weights, the first term of eq. (32) of Sugiyama et al. (2020). It grows as $B^2$, faster than any other term, which is why the Gaussian-only covariance fails at $k gt.tilde 0.1$.

== $P T$ and the super-sample overlap

Here a cross pair plus a connected quadruple. The nine cross pairs have the same coincidence table as the Gaussian pairings. For $(1 1')$ the pair sits at $#y = #x + #s$ (R1, $P(#p _1\; hat(#y))$), and the quadruple $(2, 3, 2', 3')$ at the anchor, $omega^4 (#x) e^(i (#q _1 - #p _1) dot #x) T(#q _2, -#p _2, #p _1 + #p _2\; #xh)$. The plane wave carries the beat $#q _1 - #p _1$. With A4 ($#p _1 -> #q _1$ inside $T$),
$ tau(#q _1\; #xh) = & sh(overline(tilde(cal(S)))_ell (hat(#q)_1, hat(#q)_2, #xh) tilde(cal(S))_(ell')(hat(#q)_1, hat(#p)_2, #xh) thin T(#q _2, -#p _2, #q _1 + #p _2, -#q _1 - #q _2\; #xh))_(j j') \ = & sum_J tau_J (q_1) cal(L)_J (hat(#q)_1 dot #xh), $
#result[
$ C^(P T, (1 1'))_(ell ell') = & ((4 pi)^3 H H' sqrt(N N')) / I_3^2 sum_(J L_1) \ & times integral dif^3 x thin dif^3 s thin omega^4 (#x) W(#x + #s) \ & times cal(K)^((i))_(J 0)[tau_J](#xh, dot\; -#s) thin cal(K)^((i'))_(0 L_1)[P_(L_1)](dot, hat(#y)\; #s), $
]
i.e. weights $(omega^4, W)$. The box limit is the $P T$ term of eq. (33) of Sugiyama et al. (2020). $tau$ is a Monte Carlo over the two shells, as in `thecov.trispectrum`.

#idea[
*Do not double count super-sample covariance.* Setting $bold(epsilon) = #q _1 - #p _1 = 0$ inside $T$ (A4) drops its response terms $prop P(epsilon)$. Integrated against the window, those terms *are* the super-sample covariance of the bispectrum, the analogue of $T_0$'s squeezed part for $P(k)$. So either use $T$ at $epsilon = 0$ and add the super-sample term separately (what `thecov` does for $P$), or keep the full $bold(epsilon)$ dependence and add nothing. Not both.
]

== Shot noise, models, calibration

*Poisson terms.* For $F = w(n_g - alpha n_r)$ the connected three-point function has:
- a one-shared-galaxy term with weight $overline(n) w^2 dot omega$, from galaxies only, because randoms do not cluster with the third field;
- a three-shared term with weight $(1 - alpha^2) overline(n) w^3$.

Hence $omega^3 B -> omega^3 B + overline(n) w^2 omega [P(k_a) + P(k_b) + P(k_c)] + (1 - alpha^2) overline(n) w^3$. The four-point function follows the same rule. The pair counts needed are those with weights built from $omega^2, omega^3, omega^4, overline(n) w^2 omega, overline(n) w^2 omega^2, S^2$, all of which `WindowLibrary` can accumulate in one pass.

*Models.*
- $B$ and $T$ at tree level with the SCF99 kernels already in `thecov` ($B$ in `discreteness.py` and `ssc.py`, $T$ in `trispectrum.py`, Galileon bias), on the same damped $P_"lin"$ as the Gaussian term, with the fingers-of-God damping on every $Z_1$ leg.
- $beta$ and $tau$ need $B$ on two-magnitude grids and $T$ on general quadrilaterals. The kernels support this, but the current $T$ driver integrates only parallelograms.
- $P_5$ (in $P B$) and $P_6$ (in $B B$) are corrections to corrections and are dropped.

*Calibration.*
- The measured multipoles can replace the model $B_(ell'')$ in $beta$ and $beta^(u, p)$, lightly smoothed; in noisy bins the tree model fitted to $hat(B)$ is safer.
- $T$ cannot be measured, but its squeezed part (position-dependent power) and its collapsed part (covariance of $hat(P)$ across sub-volumes) can.
- One residual amplitude per term is then fitted to the mocks and checked against these data-side estimates.

= Summary

#align(center, table(columns: 4, align: left, stroke: 0.5pt + luma(160), inset: 5pt,
  [*term*], [*window enters through*], [*radial kernel*], [*size on the diagonal*],
  [Gaussian class 1 (2 pairings)], [$cal(Q)_n (r_1, r_2)$: three-point, anchored pair sums], [$overline(j) overline(j) cal(P)$, 2D], [leading],
  [Gaussian class 2 (4)], [same $cal(Q)_n$], [$overline(j) overline(j) cal(P)^((2))$, 2D], [$approx Delta k \/ k$],
  [$k_3 approx 0$ (1)], [long-mode variance of $omega^2 F$], [none], [on $i = j$, $i' = j'$; zero for uniform $omega$],
  [$P B$], [$cal(Q)^(omega^3 W)(s)$], [power-spectrum kernel with $P beta$], [first order in $B$],
  [$B B$], [$cal(Q)^(omega^3 omega^3)(s)$], [same, with $beta^u, beta^p$], [grows as $B^2$],
  [$P T$], [$cal(Q)^(omega^4 W)(s)$], [same, with $tau$], [plus separate SSC],
))

= What changed relative to `bispectrum_covariance.typ`

The physics and the results are the same. These are the differences:
+ $H_(202) = +1 \/ sqrt(5)$ (in general $H_(ell 0 ell) = (-1)^ell \/ sqrt(2 ell + 1)$). This is what makes $cal(S)_(202) = cal(L)_2$; the old sign gave $-cal(L)_2$. [status (i)]
+ All harmonics at the five window directions are unconjugated, the invariant basis $Phi_n$ is orthonormal for that variance, and $T_n$ is defined as a projection. This removes the mismatch between $T$ and $E$. [status (ii)]
+ $K_1, K_2$ run over all integers in their triangles, not same-parity sets.
+ The box value of the window invariants is stated: $cal(Q)_n^"box" = V sqrt(N \/ 4 pi) H delta_(lambda_1 0) delta_(lambda_2 0)$. [status (iii)]
+ Class 2 is written in the variables $(#p _1, #p _3)$, one primed momentum per leg, so it shares the window functions $cal(Q)_n$ with class 1. [status (iv)]
+ The rank bound reads $a <= lambda_w + ell_1 + ell'_1 + L_1$.
+ The $k_3 approx 0$ term $(1 2)(1' 2')(3 3')$ is included as a seventh Gaussian term. [status (v)]
+ Self-pair subtraction is added and the cost corrected to $tilde.op 10^(12)$. [status (vi)]
+ $B B$:
  - The displayed pairing was $(1 | 1' 3')$ (coincidence $k_1 approx k'_2$), not $(1 | 2' 3')$; the derivation above uses $(1 | 2' 3')$.
  - On the diagonal with $i != j$ only two of the four leg--leg pairings survive.
  - The same caveat applies to the second class-1 pairing.
+ The one-shared-galaxy Poisson weight of the three-point function is $overline(n) w^2 omega$, without the factor $(1 + alpha)$.
+ The non-Gaussian terms are reorganised: averaging the non-plane-wave momenta first turns each into the power-spectrum formula with an effective spectrum. Approximation A4 is stated explicitly.
+ New: the real-space $B_(000)$ warm-up @eq:b000, Tools 2--3 (with both signs of Tool 3 verified numerically), and the explicit class-2 suppression count ($approx Delta k \/ k$ overall).
