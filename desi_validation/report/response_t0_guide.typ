#set document(title: "The response approach to the non-Gaussian power-spectrum covariance")
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, it)

#let k = $bold(k)$
#let q = $bold(q)$
#let p = $bold(p)$
#let k1 = $bold(k)_1$
#let k2 = $bold(k)_2$
#let x = $bold(x)$
#let nh = $hat(bold(n))$
#let avg(body) = $lr(chevron.l body chevron.r)$
#let note(body) = block(fill: luma(242), inset: 8pt, radius: 3pt, width: 100%, body)
#let idea(body) = block(fill: rgb("#eaf2fb"), inset: 8pt, radius: 3pt, width: 100%, body)

#align(center)[
  #text(size: 16pt, weight: "bold")[The response approach to the non-Gaussian\ power-spectrum covariance]
  #v(0.3em)
  #text(size: 11pt)[A beginner's guide, and why we want it for the DESI covariance in `thecov`]
  #v(0.3em)
  #text(size: 9.5pt, fill: luma(80))[Branch `thecov2`, October 2026. Companion to `ssc_formalism.typ`; status of the code in `desi_validation/HANDOFF.md` §7.]
]

#v(0.6em)
#note[
*In one paragraph.* The covariance of a measured power spectrum has a Gaussian part, which we compute accurately, and a non-Gaussian part coming from the four-point function (the trispectrum) of the galaxy field. The textbook way to compute the trispectrum, tree-level perturbation theory, works only on large scales; on the DESI LRG mocks it overpredicts the non-Gaussian couplings by factors of 2--10 at $k approx 0.2-0.3 thin h"/Mpc"$ and even makes the covariance non-positive. The _response approach_ keeps only the configurations of the four-point function in which one wavevector is much smaller than the others. In those, the small-scale power simply _responds_ to a long-wavelength mode as if it lived in a slightly different universe, and that response can be written in terms of the _measured, nonlinear_ power spectrum (or measured directly in sub-volumes). This note explains the idea from scratch.
]

= What we are computing

== The power spectrum and its scatter

Write the galaxy density contrast in Fourier space as $delta(#k)$. Statistical homogeneity means that different wavevectors are uncorrelated on average, and the power spectrum is the variance of each mode,
$ avg(delta(#k) delta(#k')) = (2 pi)^3 delta_D (#k + #k') P(k). $
In a box of volume $V$ we estimate $P$ in a shell (a "$k$-bin") by averaging $|delta(#k)|^2$ over the $N_k$ modes it contains,
$ hat(P)(k_i) = 1/(N_(k_i)) sum_(#k in "bin" i) (|delta(#k)|^2)/V . $
This estimate scatters from one realisation of the universe (or one mock) to the next. Its covariance matrix
$ C_(i j) = avg(hat(P)(k_i) hat(P)(k_j)) - avg(hat(P)(k_i)) avg(hat(P)(k_j)) $
is what enters the likelihood of every cosmological fit, so errors in it become errors on the inferred cosmology.

== The Gaussian part

If $delta$ were a Gaussian random field, each $|delta(#k)|^2$ would be an independent (exponentially distributed) number with variance $P^2$, and averaging $N_k$ of them (each mode counted together with its partner $-#k$) gives
$ C_(i j)^"G" = (2 P(k_i)^2)/(N_(k_i)) delta_(i j). $
The Gaussian covariance is diagonal (in a box) and shrinks as $1\/N_k$. In a real survey the window function couples neighbouring bins and the multipoles; `thecov` computes that part accurately --- we have checked it against 859 DESI mocks.

== The non-Gaussian part: the trispectrum

Gravity couples different Fourier modes, so the field is not Gaussian and the modes are not independent. The extra covariance is controlled by the connected four-point function, the _trispectrum_ $T$, evaluated on the "parallelogram" configuration $(#k1, -#k1, #k2, -#k2)$ and averaged over the two shells:
$ C_(i j) = C_(i j)^"G" + 1/V overline(T)(k_i, k_j), quad overline(T)(k_i, k_j) = avg(T(#k1, -#k1, #k2, -#k2))_(#k1 in i, #k2 in j). $ <eq-cov>
$T = 0$ for a Gaussian field. Unlike the Gaussian part, the $T$ term does not shrink with the number of modes in a bin: it is a _coherent_ coupling between all bins, which is why it matters for fits that use many bins.

(In a survey, $1\/V$ is replaced by a window integral, and there are also discreteness (shot-noise) versions of these terms; we set those aside here --- `thecov` already handles them.)

= Why the obvious calculation fails

== Tree-level perturbation theory

On large scales one expands the density in powers of the linear field, $delta = delta^((1)) + delta^((2)) + delta^((3)) + dots$, with the higher orders built from the linear one through known kernels ($F_2, F_3$; in redshift space and for galaxies $Z_1, Z_2, Z_3$, which contain the bias parameters $b_1, b_2, dots$ and the growth rate $f$). The lowest-order ("tree-level") trispectrum has two pieces,
$ T = underbrace(4 sum Z_1 Z_1 Z_2 Z_2 P_L P_L P_L, "snake") + underbrace(6 sum Z_1 Z_1 Z_1 Z_3 P_L P_L P_L, "star"), $
products of three linear power spectra $P_L$ with kernels. This is what `thecov.TrispectrumCovariance` computes, in the same way as Kobayashi's `PowerSpecCovFFT` (we reproduce its results exactly where its code is correct).

== What the DESI mocks say

Perturbation theory is an expansion in $delta^((1))$, so it is accurate only where the field is weakly nonlinear: $k lt.tilde 0.1 thin h"/Mpc"$ at $z approx 0.5$, and less in redshift space, where the random velocities inside haloes smear galaxies along the line of sight (the "fingers of God"). The covariance we need extends to $k = 0.3$. Figure @fig-mocks compares the off-diagonal covariance of 859 LRG mocks with our prediction. Without the tree-level trispectrum (blue) the prediction is close; adding it (orange, red, green) overshoots almost everywhere, worst in the quadrupole blocks and in the couplings between the lowest-$k$ bins and $k approx 0.2-0.3$. With it the total covariance matrix even acquires negative eigenvalues, i.e. it is not a valid covariance. Damping the BAO wiggles or adding a fingers-of-God factor does not cure this (`HANDOFF.md` §7.1).

#figure(
  image("t0_offdiag_LRG1_NGC.png", width: 100%),
  caption: [Off-diagonal covariance of the LRG1 NGC multipoles beyond the Gaussian part, in units of $sqrt(C^"G"_(i i) C^"G"_(j j))$, averaged over the $k_j$ in the shaded band. Points: 859 mocks. Blue: super-sample + discreteness terms. Orange/red/green: adding the tree-level trispectrum (green: with fingers-of-God damping).],
) <fig-mocks>

The lesson is not that the trispectrum is negligible --- the mocks clearly need some extra coupling in the monopole at high $k$ --- but that tree-level perturbation theory gets its _size and shape_ wrong at the scales we use. We need a way to compute the trispectrum that stays correct when the small-scale modes are nonlinear.

= The response idea

== A long mode looks like a different universe

Take a Fourier mode with a long wavelength, wavenumber $q$, and look at a region much smaller than $1\/q$. Inside it the long mode is nearly constant: it is just a small _overdensity_ $delta_L$ (plus, more generally, a constant tidal field and a bulk flow with a constant velocity gradient). The small-scale structure inside the region evolves exactly as it would in a universe with slightly higher mean density. This is the _separate-universe_ picture, and it is exact in the limit $q -> 0$.

So the small-scale power spectrum measured inside the region is a function of $delta_L$:
$ P(k | delta_L) = P(k) [1 + R_1(k) delta_L + 1/2 R_2(k) delta_L^2 + dots]. $ <eq-resp>
The coefficients $R_1, R_2$ are the first- and second-order _response functions_. They describe how small-scale power reacts to its large-scale environment, and they know nothing about how the long mode was generated.

#idea[
*Key point.* The response of the small scales to a long mode can be computed, or measured, using the _nonlinear_ small-scale power spectrum. Nothing in the separate-universe argument requires the small scales to be perturbative --- only the long mode has to be small. This is what tree-level theory lacks: it describes the small scales with the linear $P_L$ and perturbative kernels.
]

== Example: the matter response $R_1$

For dark matter in real space $R_1$ has three physical contributions, each easy to understand:

+ *Growth.* An overdense region behaves like a closed universe: structure grows faster. To first order the small-scale amplitude is enhanced by $13\/21 thin delta_L$, so the power by $26\/21 thin delta_L$.
+ *Dilation.* The overdense region expands less than the background, so physical scales are compressed by a factor $(1 - delta_L\/3)$. A fixed comoving $k$ in global coordinates corresponds to a slightly different $k$ locally, which shifts the spectrum: $-1\/3 thin dif ln (k^3 P)\/dif ln k = -1 - 1\/3 thin dif ln P\/dif ln k$.
+ *Reference density.* We measure $delta$ relative to the _global_ mean density, not the local one, so the local power is multiplied by $(1 + delta_L)^2$: a contribution $+2$.

Adding them up,
$ R_1(k) = 26/21 - 1 - 1/3 (dif ln P)/(dif ln k) + 2 = 47/21 - 1/3 (dif ln P)/(dif ln k). $ <eq-r1>
Tree-level perturbation theory gives exactly this formula with $P -> P_L$ (the linear spectrum). Separate-universe $N$-body simulations show that using the _nonlinear_ $P$ in @eq-r1 stays accurate to a few per cent well into the nonlinear regime, while the tree-level version fails --- for example because $dif ln P_L\/dif ln k$ contains the full BAO wiggles, which nonlinear evolution erases. The same happens with a tidal (anisotropic) long mode: $R_K = 8\/7 - dif ln P\/dif ln k$.

= From responses to the covariance

== Three kinds of configurations

The covariance @eq-cov needs $T(#k1, -#k1, #k2, -#k2)$. Besides $k_1$ and $k_2$, two other wavenumbers characterise this configuration: the "internal" momenta $|#k1 + #k2|$ and $|#k1 - #k2|$. The response approach looks at which of these four is small:

*Collapsed* ($p = |#k1 plus.minus #k2| << k_1, k_2$). Possible only when $k_1 approx k_2$ (near-diagonal elements). Both short modes are modulated by the same long mode $p$: if $p$ happens to be overdense in our realisation, both $hat(P)(k_1)$ and $hat(P)(k_2)$ go up together. To first order in the long mode,
$ T_"coll" approx sum_(plus.minus) R_1(k_1) R_1(k_2) P(k_1) P(k_2) P(|#k1 plus.minus #k2|), $
written here for the isotropic (density) part of the long mode; its tidal part adds angular factors.
_This is the same physics as the super-sample covariance_, which `thecov.SuperSampleCovariance` already computes for long modes larger than the survey ($p lt.tilde 1\/L$). The collapsed term is its continuation to long modes _inside_ the survey.

*Squeezed* ($k_1 = q << k_2$). One of the measured modes is itself the long mode. Its power $|delta_q|^2$ fluctuates from realisation to realisation, and the short-scale power responds to it at _second_ order:
$ T_"sq" approx R_2(k_2) thin P(q)^2 thin P(k_2) quad "(angle-averaged, schematically)". $
This is the term that dominated the failure in Figure @fig-mocks (the low-$k_i$ end of each panel). In tree-level theory $R_2$ comes from a delicate cancellation between large snake and star pieces, evaluated with the linear power spectrum at $k_2 approx 0.2-0.3$. The response approach evaluates it with the nonlinear one.

*Hard* (all four wavenumbers comparable and large). Here no separation of scales exists and responses do not apply. For dark matter, response-based calculations compared with large suites of simulations (Barreira & Schmidt 2017) find that the collapsed and squeezed terms capture most of the non-Gaussian covariance for $k gt.tilde 0.1 thin h"/Mpc"$; the hard part is a minor correction. Where all wavenumbers are small ($lt.tilde 0.1$) tree-level theory is accurate and can be used as is.

== Putting it together

The recipe is a patchwork controlled by a splitting scale $k_s$ (a few $times 0.01 thin h"/Mpc"$):
$ T approx cases(
  T_"tree" & "if all wavenumbers" < k_"PT",
  T_"coll" & "if" |#k1 plus.minus #k2| < k_s,
  T_"sq" & "if" min(k_1, k_2) < k_s,
  "small (neglected or a free template)" & "otherwise"
) $
with checks that the result does not depend on the exact $k_s$.

= Galaxies in redshift space

Two complications, both already met in the super-sample covariance note:

*Redshift-space distortions.* The long mode now also has a velocity field. Its line-of-sight gradient stretches or compresses the redshift-space map along the line of sight, so the response depends on the angle $nu$ between the long mode and the line of sight, and different multipoles respond differently. We write the responses as $R_ell^((n))(k)$: the response of the multipole $ell$ to the $n$-th Legendre component of the long mode's orientation ($n = 0, 2$ suffice at first order).

*Bias.* Galaxies are biased tracers, so their density responds through $b_1$ (linear bias), $b_2$, and the tidal bias. These enter the response at first order through $b_1$ and $b_2$, and at second order through third-order bias parameters, which are poorly known.

Both are handled the same way: start from the tree-level response kernels, but evaluate them on a _dressed_ power spectrum $P(k, mu)$ that already contains what tree-level theory lacks --- BAO damping and fingers of God, or directly the measured multipoles. This is what we just implemented for the super-sample term (`thecov.ssc.response_multipoles` with `thecov.power.Dressed`): on the LRG mocks it brought the coherent quadrupole amplitude from 1.10% to 0.95% (mocks: 0.94%) in NGC with a single parameter, $sigma_v = 2.1 thin "Mpc"\/h$, fitted to the mean $P_2\/P_0$ (`HANDOFF.md` §7.2).

#idea[
*Responses can also be measured.* Split the survey (or a mock) into sub-volumes, measure in each the local power spectrum $hat(P)_"sub"(k)$ and the mean density $overline(delta)_"sub"$, and correlate them: the slope $dif ln hat(P)_"sub" \/ dif overline(delta)_"sub"$ is $R_1(k)$ ("position-dependent power spectrum", Chiang et al. 2014). The same works for the anisotropic responses by splitting along and across the line of sight. This connects the response approach to our plan of fitting the non-Gaussian terms to sub-volumes of the data: instead of fitting the amplitudes of arbitrary templates, we would measure the few response functions that build the covariance.
]

= Plan in `thecov`

+ *First-order responses from the dressed power* --- done for the super-sample term (§7.2 of the hand-off).
+ *Collapsed term inside the survey.* Extend the super-sample long-mode integral from window scales up to $p < k_s$, with the same responses. Near-diagonal blocks only; cheap.
+ *Squeezed term.* Second-order dressed responses $R_(2,ell)^((n))$ for pairs with $min(k_1, k_2) < k_s$; the third-order bias enters here, so give it a free amplitude (and an independent check from sub-volumes).
+ *Large-scale corner.* Keep tree-level $T$ where all wavenumbers are below $k_"PT" approx 0.1$.
+ *Validation* on the 859 LRG and QSO mocks with the existing diagnostics (`ssc_check.py`, `ssc_plots.py`): off-diagonal blocks, whitened eigenvalues, parameter-variance ratios; positivity of the total covariance.
+ *Sub-volume amplitudes.* Collapsed and squeezed terms become templates with amplitudes $A_"coll"$, $A_"sq"$ in `thecov.CovarianceTemplates`, fitted to sub-volumes (and validated on mocks).

*What can go wrong.* (i) The squeezed term needs the long mode to be well inside the survey; for the thin LRG shells the lowest $k$ bins are close to the window scale, where the local approximation fails and the term must be matched to the super-sample treatment. (ii) Second-order galaxy responses are less certain than first-order ones. (iii) The hard part may not be negligible for galaxies at $k = 0.3$; the mocks will tell, and it can be absorbed in a template amplitude.

= Glossary

/ Mode coupling: correlation between different Fourier modes caused by nonlinear evolution; zero for a Gaussian field.
/ Trispectrum $T$: connected four-point function in Fourier space; sets the non-Gaussian covariance of $hat(P)$.
/ Tree level: lowest non-vanishing order of perturbation theory; uses the linear power spectrum.
/ Separate universe: the equivalence between a constant long-wavelength overdensity and a change of the background cosmology.
/ Response $R_n(k)$: $n$-th derivative of $ln P(k)$ with respect to the long-mode amplitude, @eq-resp.
/ Collapsed / squeezed: four-point configurations in which an internal momentum, or one of the external momenta, is much smaller than the rest.
/ Super-sample covariance: the collapsed term for long modes larger than the survey.
/ Dressed power: a power spectrum that includes nonlinear effects (BAO damping, fingers of God), used in place of the linear one.

= Further reading

- M. Takada & W. Hu (2013), _Power spectrum super-sample covariance_, Phys. Rev. D 87, 123504 --- the super-sample term and its response formulation.
- C.-T. Chiang, C. Wagner, F. Schmidt & E. Komatsu (2014), JCAP 05, 048 --- the position-dependent power spectrum (measuring responses in sub-volumes).
- C. Wagner, F. Schmidt, C.-T. Chiang & E. Komatsu (2015), MNRAS 448, L11 --- separate-universe simulations and nonlinear responses.
- A. Barreira & F. Schmidt (2017), JCAP 06, 053 --- the response approach to the matter power spectrum covariance (the method this note describes).
- D. Wadekar & R. Scoccimarro (2020), Phys. Rev. D 102, 123517 --- tree-level covariance of redshift-space galaxy multipoles, the starting point of `thecov`'s non-Gaussian terms.
- Y. Kobayashi (2023), `PowerSpecCovFFT` --- tree-level trispectrum covariance of redshift-space multipoles (the code we validated against).
