// Mock validation of the thecov Gaussian covariance (mock_version 4 run).
// Build: python reports/validation_v5/make_figures.py <run_dir> <control_run_dir>; typst compile report.typ

#set document(title: "Mock validation of the separation-space Gaussian covariance")
#set page(paper: "a4", margin: (x: 2.1cm, y: 2.2cm), numbering: "1")
#set text(size: 10pt, font: "New Computer Modern")
#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#show heading: set block(above: 1.2em, below: 0.7em)
#show figure.caption: set text(size: 8.8pt)
#show figure: set block(above: 1.1em, below: 1.1em)

#align(center)[
  #text(size: 15pt, weight: "bold")[Mock validation of the separation-space Gaussian covariance \ of windowed multi-tracer power-spectrum multipoles]
  #v(4pt)
  #text(size: 10pt)[`thecov` validation note — run `results_v5` (mock version 4) · 30 September 2026]
]
#v(6pt)

#block(inset: (x: 1.2cm))[
  #set text(size: 9.2pt)
  *Abstract.* We compare the Gaussian covariance of power-spectrum multipoles computed by `thecov` — a
  separation-space (tripolar) formulation that includes the survey window exactly and approximates only
  Gaussianity, a local plane-parallel line of sight and a window slowly varying over a correlation length —
  with the sample covariance of 1000 Gaussian mocks of a two-tracer survey with a realistic, wide-angle
  footprint. For the 174-element data vector ($ell = 0, 2$ of the two auto- and one cross-spectrum,
  $0.01 < k < 0.30 thin h thin "Mpc"^(-1)$) the mean $chi^2$ computed with the analytic matrix is
  $174.26$ against an expectation of $173.83 plus.minus 0.59$ ($+0.7 sigma$); its variance, its
  distribution, the eigenvalues of $C^(-1\/2) hat(C) C^(-1\/2)$ and the correlation-coefficient residuals
  are all consistent with an exact covariance, and every block agrees within $1.5 sigma$. A control run that
  subtracts the _expected_ instead of the _realised_ shot noise fails at $+7.1 sigma$ in the high-$k$
  monopole-auto blocks, a failure reproduced without free parameters by the Poisson fluctuation of the
  self-pair sum. The only departure from exactness we can see is a variance deficit of $1.9 plus.minus 1.5 %$
  in the three lowest $k$ bins, of the sign expected from the unmodelled integral constraint.
]

= Purpose

`thecov` computes the disconnected covariance of FKP–Yamamoto power-spectrum multipoles as
$
C^(A B C D)_(ell_1 ell_2)(i,j) = frac((4 pi)^4, I_(A B) I_(C D)) sum_(L_1 L_2)
frac(1, (2L_1+1)(2L_2+1)) sum_(lambda lambda') (-1)^((lambda+lambda')\/2)
sum_(Lambda_1 Lambda_2 Lambda) [t^((1)) cal(I)^((1))_(i j) + t^((2)) cal(I)^((2))_(i j)],
$ <eq:cov>
where $cal(I)^((n))_(i j) = integral s^2 dif s thin overline(p)'^((i))_(L_1 lambda)(s) thin overline(p)^((j))_(L_2 lambda')(s) thin cal(Q)^(omega omega')_(Lambda_1 Lambda_2 Lambda)(s)$
couples bin-averaged Bessel transforms of the model multipoles to tripolar window functions
$cal(Q)(s)$ measured from pair counts of the random catalogues. Every other test in the suite checks the
implementation against exact identities; the mocks are the only test of the three _approximations_ behind
@eq:cov. Because each mock galaxy is displaced along its own line of sight inside a $30 degree$ cap,
nothing in their construction assumes a plane-parallel geometry.

= Set-up

*Survey.* A spherical cap of half-opening $30 degree$ between comoving distances $400$ and
$900 thin h^(-1)"Mpc"$, with three circular holes and a linear completeness gradient of $plus.minus 25 %$ across the
cap (@fig:geometry). Two tracers share the angular mask but have different radial selections:
$overline(n)_A$ peaks at $6 times 10^(-4) thin h^3"Mpc"^(-3)$ at $550 thin h^(-1)"Mpc"$, $overline(n)_B$
at $3 times 10^(-4)$ at $750 thin h^(-1)"Mpc"$. The expected numbers of galaxies are
$63 thin 800$ (A) and $47 thin 500$ (B); each tracer has its own random catalogue, $15 times$ denser
($alpha = 0.0667$). FKP weights $w = (1 + overline(n) P_0)^(-1)$ with $P_0 = 10^4 thin h^(-3)"Mpc"^3$.

#figure(image("fig_geometry.pdf", width: 100%),
  caption: [Survey geometry. _Left:_ radial selection of the two tracers. _Right:_ the angular footprint,
  shown by a subsample of the randoms of tracer A coloured by $overline(n)_A$; the three holes and the
  completeness gradient are visible.]) <fig:geometry>

*Mocks.* A Gaussian linear density field with a smooth, CDM-like spectrum with a small oscillatory
feature, generated on a $256^3$ grid in a periodic box of side $1800 thin h^(-1)"Mpc"$ (twice the survey
extent; cell $7.03 thin h^(-1)"Mpc"$, $k_"Nyq" = 0.447 thin h thin "Mpc"^(-1)$). Galaxies are Poisson-sampled
cell by cell from $overline(n)(1 + b delta)$ with $b_A = 1.9$, $b_B = 1.2$, and displaced by the linear
Zel'dovich velocity $f (bold(Psi) dot hat(bold(x))) hat(bold(x))$, $f = 0.78$, along _each galaxy's own_
line of sight. The amplitude is scaled so that $sigma(b_A delta) = 0.31$ per cell, for which clipping of
$1 + b delta$ at zero affects $6 times 10^(-4)$ of the cells and removes $approx 0.1 %$ of the power. The field is
generated with the cell window pre-divided, so the sampled point process has exactly the input spectrum,
and has no power above $0.8 thin k_"Nyq"$. The survey mask and weights are applied at the observed
(redshift-space) position. The expected multipoles are the linear Kaiser ones,
$P_(X Y)(k, mu) = (b_X + f mu^2)(b_Y + f mu^2) P_"lin"(k)$.

*Estimator.* Yamamoto–FKP multipoles with the line of sight at the first field, TSC assignment with
interlacing, and the pypower/jaxpower conventions: $alpha$ and the shot noise
$sum_g w_g^2 + alpha^2 sum_r w_r^2$ are those of each realisation. We measure $P^(A A)_ell$, $P^(A B)_ell$
and $P^(B B)_ell$ for $ell = 0, 2$ in 29 bins of width $0.01$ between $k = 0.01$ and $0.30$
($k_"max" = 0.67 thin k_"Nyq"$; a dedicated aliasing test on a common catalogue bounds the aliasing bias of
this estimator by $0.2 %$ at $0.56 thin k_"Nyq"$ and by $0.4 %$ ($0.9 %$ for $A B$) at $0.73 thin k_"Nyq"$). The normalisation is
fixed to $I_(X Y) = integral overline(n)_X overline(n)_Y w_X w_Y$. @fig:multipoles shows the mean of the
1000 mocks.

*Analytic covariance.* `thecov` with the Kaiser multipoles above as the model (up to $L = 4$), window
functions from pair counts of $5000$ (all pairs, $s > 80 thin h^(-1)"Mpc"$) and $2 times 10^5$ (neighbours,
$s < 80$) randoms per window, $Delta s = 10 thin h^(-1)"Mpc"$ for the pair counts and $2 thin h^(-1)"Mpc"$ for
the radial integrals. @fig:windows shows representative window functions. The model is exact for the
mocks at linear order; the Zel'dovich displacement of a discrete sample adds small non-linear corrections.

#figure(image("fig_multipoles.pdf", width: 100%),
  caption: [Mean of the 1000 mocks, $k P_ell(k)$, for the monopole (_left_) and quadrupole (_right_) of the
  three spectra. Bands: $plus.minus 1 sigma$ of a single realisation from the analytic covariance.])
  <fig:multipoles>

= Statistics

With $N = 1000$ mocks $bold(d)_i$ and $n = 174$ elements, the principal whole-matrix test is
$
chi^2_i = (bold(d)_i - overline(bold(d)))^T C^(-1) (bold(d)_i - overline(bold(d))), quad
chevron.l chi^2 chevron.r = n (1 - 1\/N) plus.minus n (1 - 1\/N) sqrt(2\/(n N)),
$ <eq:chi2>
which, with $C$ exact, is accurate to $0.34 %$ here and sensitive to the whole correlation structure.
Its variance should be $approx 2 n$ and its distribution $chi^2_n$ rescaled by $1 - 1\/N$. The
eigenvalues of $C^(-1\/2) hat(C) C^(-1\/2)$, with $hat(C)$ the sample covariance, are _not_ expected to lie
within $plus.minus sqrt(2\/N)$ of unity: for an exact $C$ they fill the Marchenko–Pastur interval
$[(1 - sqrt(q))^2, (1 + sqrt(q))^2]$ with $q = n\/(N-1)$, i.e. $[0.339, 2.009]$. To localise a failure we
also evaluate @eq:chi2 restricted to each block and to its low- and high-$k$ halves (split at
$k = 0.155$), the ratio $sigma_"mock"\/sigma_"thecov"$ per element (statistical scatter
$1\/sqrt(2(N-1)) = 0.022$) and the correlation-coefficient residuals
$z_(i j) = (hat(rho)_(i j) - rho_(i j)) sqrt(N-1) \/ (1 - rho_(i j)^2)$, which are unit Gaussians for an
exact $C$.

= Results

#figure(image("fig_chi2_eigen.pdf", width: 100%),
  caption: [_Left:_ distribution of $chi^2_i$ over the mocks against the rescaled $chi^2_(174)$ density.
  _Right:_ eigenvalues of $C^(-1\/2) hat(C) C^(-1\/2)$ against the Marchenko–Pastur density of an exact
  covariance with $n = 174$, $N = 1000$.]) <fig:chi2>

*Whole matrix.* $chevron.l chi^2 chevron.r = 174.26$ against $173.83 plus.minus 0.59$ ($+0.7 sigma$); the
variance is $345.6$ against $347.7$, and a Kolmogorov–Smirnov test of the $chi^2_i$ against the
rescaled $chi^2_(174)$ distribution gives $p = 0.30$ (@fig:chi2, left). The eigenvalues span
$[0.342, 1.991]$, inside the Marchenko–Pastur interval $[0.339, 2.009]$, and their density follows it
(@fig:chi2, right): there is no direction in data space along which the mocks fluctuate more, or less, than
`thecov` predicts, beyond what $1000$ samples allow.

*Blocks.* @tab:blocks lists @eq:chi2 restricted to each block. All 18 entries are within $1.53 sigma$;
their spread is what 18 unit Gaussians give.

#figure(
  table(columns: 4, align: (left, center, center, center), stroke: none,
    table.hline(),
    [block], [all $k$], [$k < 0.155$], [$k > 0.155$],
    table.hline(stroke: 0.5pt),
    [$A A$, $ell=0$], [$0.994 space (-0.7)$], [$0.988 space (-1.0)$], [$1.002 space (+0.1)$],
    [$A A$, $ell=2$], [$1.008 space (+0.9)$], [$0.997 space (-0.2)$], [$1.018 space (+1.5)$],
    [$A B$, $ell=0$], [$0.995 space (-0.6)$], [$1.002 space (+0.1)$], [$0.990 space (-0.9)$],
    [$A B$, $ell=2$], [$0.995 space (-0.6)$], [$0.995 space (-0.4)$], [$0.994 space (-0.5)$],
    [$B B$, $ell=0$], [$1.001 space (+0.1)$], [$0.987 space (-1.1)$], [$1.014 space (+1.2)$],
    [$B B$, $ell=2$], [$1.000 space (0.0)$], [$1.015 space (+1.2)$], [$0.987 space (-1.1)$],
    table.hline(),
  ),
  caption: [$chevron.l chi^2 chevron.r$ restricted to each block, divided by its expectation; the deviation in
  units of its standard error in brackets.]) <tab:blocks>

*Diagonal.* @fig:sigma shows $sigma_"mock"\/sigma_"thecov"$ for all 174 elements. The ratios scatter
about unity with an rms of $0.025$, against $0.022$ expected from sampling alone (the small excess is
the positive correlation between neighbouring bins, $rho approx 0.3$, which makes runs of points move
together); the mean variance ratio over all elements is $0.997$. No trend with $k$ is visible in any block,
in particular none at high $k$ in the monopole-auto blocks, where the control run below fails.

#figure(image("fig_sigma_ratio.pdf", width: 100%),
  caption: [Ratio of the mock to the analytic standard deviation for every element of the data vector.
  Shaded: $plus.minus 1 sigma$ and $plus.minus 2 sigma$ sampling scatter for $N = 1000$.]) <fig:sigma>

*Correlations.* The full correlation matrices are compared in @fig:correlation. The analytic matrix
reproduces the structure of the mocks: nearest-neighbour correlations $rho approx 0.3$ from the window,
the coupling between $ell = 0$ and $ell = 2$ of the same spectrum, and the correlations between the auto-
and cross-spectra, which carry the multi-tracer information. The normalised residuals $z_(i j)$ have mean
$0.035$ and standard deviation $1.008$; a fraction $0.27 %$ of them exceed $3$ in modulus, against
$0.27 %$ for a Gaussian.

#figure(image("fig_correlation.pdf", width: 100%),
  caption: [_Left:_ correlation matrix of the mocks (lower triangle) and of `thecov` (upper triangle),
  ordered by spectrum, multipole and $k$ bin. _Right:_ the difference in units of its sampling error,
  $z_(i j)$.]) <fig:correlation>

= Discussion

*What is tested.* With Gaussian mocks, the disconnected covariance is the complete answer, so a pass
tests the window treatment and the two geometric approximations of @eq:cov: the local plane-parallel
line of sight — each pair sharing the direction of one of its points in a $30 degree$ cap, with
redshift-space displacements along individual lines of sight — and the slowly varying window, tested on a
footprint with holes, a completeness gradient and steep radial selections. Neither leaves a detectable
imprint at the $0.3 %$ precision of @eq:chi2, nor at the $2 %$ per-element level of @fig:sigma. The same
holds for the multi-tracer structure: the auto–cross correlations, which depend on the cross window
$W^(A B)$ and on the separate shot-noise windows of the two tracers, are reproduced as well as the
auto-spectrum blocks.

*The shot-noise convention matters.* A control run on the same survey that subtracted the _expected_
shot noise $(1 + alpha) alpha sum_r w_r^2$ instead of the realised $sum_g w_g^2 + alpha^2 sum_r w_r^2$
failed at $chevron.l chi^2 chevron.r = 178.03$ ($+7.1 sigma$). With the expected value, the self-pairs stay in
$hat(P)$ and the Poisson fluctuation of their sum adds to the monopole-auto covariance the fully
correlated term
$
Delta C_(i j) = [ integral overline(n) w^4 + 2 (P_0(k_i) + P_0(k_j)) integral overline(n)^2 w^4 ] \/ I^2 ,
$ <eq:selfpair>
which is $2 %$, $10 %$ and $19 %$ of the $A A$ diagonal at $k = 0.12$, $0.22$ and $0.30$ here, and absent
from any Gaussian formula. Adding @eq:selfpair to `thecov` without any free parameter brings the control
run to $175.67$ and removes its high-$k$ excess (@fig:control). With the realised shot noise — the
convention of pypower and jaxpower, the estimators `thecov` is meant to be used with — the self-pairs are
removed exactly and @eq:selfpair does not arise, which is what the present run confirms. We checked the
premise directly: the variance over the mocks of $sum_g w_g^2$ is $429$ ($A$) and $344$ ($B$), against a
prediction of $330 + 116$ and $304 + 59$ from the Poisson term $integral overline(n) w^4$ plus the
clustering of the weighted counts, $integral integral overline(n) w^2 overline(n) w^2 xi$ (ratios
$0.96$ and $0.95$, $plus.minus 0.045$).

#figure(image("fig_control.pdf", width: 100%),
  caption: [Monopole-auto blocks. _Orange:_ control run with the expected shot noise, against the Gaussian
  `thecov` matrix. _Blue:_ the same mocks against `thecov` plus the self-pair term @eq:selfpair. _Aqua:_
  the present run (realised shot noise) against the Gaussian matrix. Shaded: $plus.minus 1 sigma$ sampling
  scatter.]) <fig:control>

*Low $k$.* The only feature suggestive of a departure is in the three lowest bins
($k = 0.015$–$0.035$), where the variance ratio averaged over the six blocks is $1.037$, $0.966$ and
$0.941$; over the three bins together it is $0.981 plus.minus 0.015$ (bootstrap over mocks), against
$0.998 plus.minus 0.004$ for the remaining 26 bins. The deficit is $1.2 sigma$ from unity and not significant, but its
sign and location are those expected from the integral constraint: the realised
$alpha = sum w_d \/ sum w_r$ forces the zero mode of the FKP field to vanish, which removes variance at
$k tilde 1\/R_"survey"$ and which @eq:cov does not model. An order-of-magnitude larger number of mocks, or
the corresponding correction to @eq:cov (a Gaussian-level term involving the window integrated against
$overline(n) w$), would settle it.

*Limits of the test.* (i) The mocks are Gaussian: the trispectrum, super-sample covariance and the
bispectrum and $P\/overline(n)^2$ discreteness terms are absent or negligible by construction, so the
test says nothing about them. (ii) Redshift-space distortions are linear (Kaiser); there is no
fingers-of-God damping. (iii) One box size (twice the survey extent) was used: the periodic images of the
survey lie beyond $900 thin h^(-1)"Mpc"$, where the correlation function is negligible, and modes below
$2 pi \/ L = 0.0035 thin h thin "Mpc"^(-1)$ are missing, both small effects that we did not test by varying
the box. (iv) The normalisation was fixed; with the mesh-based normalisation of pypower and jaxpower the
covariance must be rescaled by the ratio of normalisations (a few per cent, handled by
`GaussianCovariance.set_normalization`), and the fluctuation of that normalisation adds a term below
$10^(-3)$ of the diagonal for this survey. (v) The window functions carry Monte-Carlo noise from the
random subsamples: two seeds differ by $approx 10^(-2)$ per element of the covariance and by
$approx 2 times 10^(-3)$ in $chevron.l chi^2 chevron.r\/n$ for a subsample four times smaller than the one used here.

#figure(image("fig_windows.pdf", width: 55%),
  caption: [Monopole tripolar window functions $cal(Q)_(000)(s)$ of the three clustering windows, and one
  anisotropic function, $cal(Q)_(202)$, normalised to $cal(Q)_(000)$ at the first separation bin.])
  <fig:windows>

= Conclusion

For a two-tracer survey with a wide-angle, masked footprint, the separation-space Gaussian covariance
reproduces the covariance of 1000 Gaussian mocks to the precision the mocks allow: $chevron.l chi^2
chevron.r$ within $0.7 sigma$ ($0.25 %$), eigenvalues within the Marchenko–Pastur range of an exact matrix,
every block within $1.5 sigma$, and correlation residuals indistinguishable from sampling noise. Two
conditions are essential and are enforced by the validation driver: the estimator must remove the
self-pairs (realised shot noise, as in pypower and jaxpower), and the normalisation used by `thecov` must
be the estimator's. The remaining open question is the integral constraint at $k lt.tilde 0.04 thin h thin
"Mpc"^(-1)$, where we see a $1.2 sigma$ variance deficit of the expected sign.

#v(6pt)
#text(size: 8.5pt)[_Reproducibility._ Mocks and covariance:
`python -m mocks.run_validation --n-mocks 1000 --grid 256 --box-factor 2 --kmax 0.30 --interlace --scheme tsc --amplitude 0.6 --out results_v5`
(branch `thecov2`); figures and numbers: `reports/validation_v5/make_figures.py`.]
