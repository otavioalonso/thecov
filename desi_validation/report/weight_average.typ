#set document(title: "Why the clustering window uses the locally averaged weight")
#set page(paper: "a4", margin: (x: 1.7cm, y: 1.3cm))
#set text(font: "New Computer Modern", size: 9.5pt)
#set par(justify: true, leading: 0.55em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#show heading: it => block(above: 0.7em, below: 0.4em, text(size: 11pt, it))

#let x = $bold(x)$
#let y = $bold(y)$
#let avg(body) = $lr(chevron.l body chevron.r)$
#let note(body) = block(fill: luma(242), inset: 7pt, radius: 3pt, width: 100%, body)

#align(center)[
  #text(size: 13.5pt, weight: "bold")[Why the clustering window uses the locally averaged weight]
  #v(0.1em)
  #text(size: 9pt, fill: luma(80))[Validation note, branch `thecov2` --- October 2026]
]

= What is computed
Galaxies $g$ at $#x _g$ with weights $w_g$, randoms $r$ with weights $w_r$, $alpha = sum_g w_g \/ sum_r w_r$:
$ F(#x) = rho_w (#x) - alpha rho_(w,r)(#x), quad rho_w (#x) = sum_g w_g delta_D (#x - #x _g), quad hat(P) = (|tilde(F)|^2 - N_S) \/ "norm". $
The covariance needs window integrals $integral W f$ and $integral S f$ for smooth test functions $f$ (e.g. $e^(-i bold(q) dot #x)$), evaluated as sums over randoms.

= Model of the catalogue (the one assumption)
Galaxies are a Poisson sample of $macron(n)(#x)[1 + delta(#x)]$. Each object carries a weight drawn from a local distribution $p(w|#x)$, *independently of the other objects and of $delta$* (assumption A). Write $avg(w^p)(#x) = integral w^p p(w|#x) dif w$.
Randoms are a Poisson sample of $n_r (#x) = macron(n)(#x) \/ alpha$ whose weights are drawn from the same $p(w|#x)$ (DESI randoms inherit the weights of random data objects).

= Exact two-point function
Split the double sum $sum_(g, g')$ into distinct pairs and self-pairs. For distinct pairs A gives $avg(w_g w_(g')) = avg(w)(#x) avg(w)(#y)$; the self-pairs carry $w_g^2$:
$ avg(rho_w (#x) rho_w (#y)) = m(#x) m(#y) [1 + xi(#x - #y)] + delta_D (#x - #y) thin macron(n) avg(w^2)(#x), quad m(#x) equiv macron(n)(#x) avg(w)(#x). $ <eq:2pt>
Adding the randoms, $avg(F(#x) F(#y)) = m(#x) m(#y) xi(#x - #y) + delta_D (#x - #y) S(#x)$ with $S = (1 + alpha) macron(n) avg(w^2)$.
So *clustering* terms see the mean weighted density $m$, and the window is $W = m^2 = macron(n)^2 avg(w)^2$; only *shot-noise* terms see $avg(w^2)$.

= Evaluating the window with randoms
A sum over randoms with each random's own weight as sampling weight is unbiased for $m$:
$ E[alpha sum_r w_r f(#x _r)] = alpha integral n_r avg(w) f = integral m f. $ <eq:mc>
For $integral W f = integral m^2 f$ the second factor $m(#x _r)$ must be supplied at the random.
*Naive choice*, the random's own weight, $m(#x _r) -> macron(n)(#x _r) w_r$:
$ E[alpha sum_r w_r^2 macron(n) f] = integral macron(n)^2 avg(w^2) f = integral m^2 f thin (1 + "var"(w) \/ avg(w)^2). $
The product $w_r dot w_r$ is a *self-pair*: it puts the shot-noise moment into the clustering window (+7% for DESI weights).

= The approximation: replace $avg(w)(#x _r)$ by a neighbour average
Use the $k$ nearest *other* randoms, $hat(w)_r = k^(-1) sum_(j in cal(N)_k (r), j != r) w_j$. By A, $w_r$ and the $w_j$ are independent, and the estimator is linear in $hat(w)_r$:
$ E[w_r hat(w)_r] = avg(w)(#x _r) dot 1/k sum_j avg(w)(#x _j) = avg(w)^2 (#x _r) [1 + O(ell_k^2 nabla^2 avg(w) \/ avg(w))], $ <eq:approx>
where $ell_k$ is the neighbourhood size (the term linear in $#x _j - #x _r$ cancels for a symmetric neighbourhood). Two remarks:
- *No noise bias:* because @eq:approx is linear in $hat(w)_r$, the scatter of $hat(w)_r$ does not bias $integral W f$ (a $"var"(w)\/k$ term appears only for higher powers of $m$, e.g. $integral m^4$, suppressed by $1\/k approx 3%$ relative to the naive $"var"(w)\/avg(w)^2$).
- *Smoothing error:* $O(ell_k^2 nabla^2 avg(w))$: negligible where $avg(w)$ varies on scales $>> ell_k$ ($ell_k approx 15 h^(-1)$Mpc for $k = 32$ in 3D; an angular neighbourhood, $approx 0.1 degree$, resolves sharp tile-completeness boundaries).

= Result
With $rho_r$ the unweighted random density, $alpha rho_r = macron(n)$, so the smooth mean weighted density and the window sums are
$ m(#x _r) = alpha thin rho_r (#x _r) thin hat(w)_r, quad integral W f approx alpha sum_r w_r thin m(#x _r) thin f(#x _r), quad integral S f approx (1 + alpha) alpha sum_r w_r^2 f(#x _r). $
This is `NW` in `thecov.Tracer`: the averaged weight enters once, the random's own weight once (sampling), and the shot noise keeps the own weight squared.

#note[
*When it fails.* The averaging is only an estimator of $avg(w)(#x)$; the physics is assumption A. If weights correlate with each other or with $delta$ (fiber-assignment/completeness weights depend on local target density and close pairs), then $avg(w_g w_(g')) != avg(w)(#x _g) avg(w)(#x _(g'))$ at small separation. That is a two-point effect: no one-point $m(#x)$, averaged or not, can absorb it. Check: the GRF test with the high-resolution window and the smooth $avg(w)$ agree to $0.2%$.
]
