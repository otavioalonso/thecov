#set document(title: "Bispectrum covariance from DESI DR2 mocks")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#show math.equation: set text(font: "New Computer Modern Math")
#set par(justify: true)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 11.5pt)
#show figure.caption: set text(size: 9pt)
#set table(stroke: 0.4pt + luma(170), inset: 4pt)

#align(center)[
  #text(size: 15pt, weight: "bold")[Bispectrum multipole covariance from DESI DR2 mocks] \
  #v(2pt)
  #text(size: 10pt)[holi, GLAM-Uchuu and AbacusSummit suites: a mock-to-mock comparison]
]

#v(6pt)
#block(inset: 8pt, stroke: 0.4pt + luma(170), radius: 2pt)[
*Summary.* We compare the sample covariance of the diagonal Sugiyama bispectrum multipoles $B_(000)(k, k)$ and $B_(202)(k, k)$, $0.02 <= k < 0.205 thin h"/Mpc"$, across four DR2 mock suites (holi v3 altmtl, GLAM-Uchuu v2 altmtl, Abacus-HF v2 complete and Abacus-HF v2 altmtl) plus the two Abacus 2nd-gen suites. For LRG and ELG, holi and GLAM agree to $lt.tilde 15%$ in variance and give mutually consistent $chi^2$ ($chi^2"/"p = 0.93$--$1.11$). The correlation structure is the same in every suite. Holi QSO is the exception: its $B_(000)$ variance is $1.6$--$4$ times that of every other suite at $k > 0.08 thin h"/Mpc"$, it has a coherent amplitude mode that is absent elsewhere, and 11 realisations are $chi^2$ outliers. We do not recommend the holi QSO bispectrum covariance until this is understood. The N-body Abacus realisations show 10--40% less LRG variance than the approximate mocks at $k > 0.14 thin h"/Mpc"$ after normalising by $P_"tot"^3$. Fibre assignment raises the variance by factors of 1.8 (ELG) and 2.4 (QSO), most of which is the expected effect of the lower effective density.
]

= Data

== Measurements

All measurements are the DR2 full-shape products in
`/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base/<suite>/mock<N>/`,
`mesh3_spectrum_sugiyama-diagonal_poles_<TRACER>_<ZRANGE>_<REGION>_weight-default-FKP.h5`, computed with `jaxpower`. We use:

- Diagonal configurations $k_1 = k_2 = k$ of the multipoles $(ell_1 ell_2 L) = (000)$ and $(202)$, local line of sight. There are 41 bins of width $Delta k = 0.005 thin h"/Mpc"$; we keep the 37 bins with $0.02 <= k < 0.205$, so the data vector has $p = 74$ elements.
- The `value` field is shot-noise subtracted. The normalisation is $alpha sum n_r^3$ on $10 thin "Mpc"/h$ cells, which varies slightly between realisations. Mesh sizes are 750, 960 and 1152 in boxes of 5625, 7200 and 8640 $"Mpc"/h$ (LRG, ELG, QSO).
- Six tracer bins: LRG1 ($0.4<z<0.6$), LRG2 ($0.6<z<0.8$), LRG3 ($0.8<z<1.1$), ELG1 (`ELG_LOPnotqso`, $0.8<z<1.1$), ELG2 ($1.1<z<1.6$) and QSO ($0.8<z<2.1$), each in NGC, SGC and GCcomb.
- The power spectrum multipoles from the matching `mesh2_spectrum_poles_*` files, used only for the normalisation below.

GCcomb is exactly the normalisation-weighted average of NGC and SGC in every suite: the maximum relative residual is $10^(-14)$. Its covariance therefore carries no extra information, and we quote NGC results unless stated otherwise.

== Suites

The inventory found 30 suites with `mesh3` products. We use the six in @tab:suites. The others are variants (Lorentzian redshift errors, petal or $z$-failure masks, BGS, older holi/GLAM versions, lightcone and single-box Abacus) and are listed in `inventory_mesh3.csv`.

#figure(
  text(size: 8.5pt, table(columns: (auto, auto, auto, 1fr), align: (left, center, left, left),
  table.header([*suite* (directory)], [*dirs*], [*used*], [*notes*]),
  [holi v3 altmtl \ #text(size: 7pt)[`holi-v3-altmtl`]], [1000], [859 (LRG1: 848)], [141 `dubious_mock_*` dirs not used; 11 LRG1 mocks with defective n(z) excluded],
  [GLAM-Uchuu v2 altmtl \ #text(size: 7pt)[`glam-uchuu-v2-altmtl`]], [1091], [1007], [84 `dubious_mock*` dirs not used],
  [Abacus-HF v2 complete (rotated) \ #text(size: 7pt)[`abacus-hf-dr2-v2-rotated-complete-claude`]], [100], [100 (25 boxes × 4 rotations)], [mocks 0--24 are the boxes of the altmtl suite],
  [Abacus-HF v2 altmtl \ #text(size: 7pt)[`abacus-hf-dr2-v2-altmtl`]], [25], [25], [paired with complete mocks 0--24],
  [Abacus 2nd-gen complete \ #text(size: 7pt)[`abacus-2ndgen-dr2-complete`]], [26], [25], [ELG sample is `ELG_LOP` (not `notqso`); paired with 2nd-gen altmtl],
  [Abacus 2nd-gen altmtl \ #text(size: 7pt)[`abacus-2ndgen-dr2-altmtl`]], [25], [25], [paired with 2nd-gen complete],
  )),
  caption: [Mock suites used in this report. "dirs" counts all `mock*` directories, including the ones flagged as dubious.],
) <tab:suites>

The rotated Abacus-HF complete suite reuses each of the 25 AbacusSummit boxes four times: mock $i$, $i + 25$, $i + 50$ and $i + 75$ come from the same box. We measured the mean correlation between their data vectors (across all elements and tracer bins, NGC). At lag 25, 50 and 75 it is 0.043 ± 0.005, 0.049 ± 0.006 and 0.048 ± 0.008, compared with -0.012 ± 0.0006 for unrelated pairs. The negative value for unrelated pairs is the expected $-1"/"(N-1)$ from mean subtraction. The rotations are therefore close to independent ($N_"eff" approx 85$), and we treat the 100 realisations as independent. Abacus-HF altmtl mocks 0--24 are the same boxes as complete mocks 0--24, with per-element correlations of 0.5--0.9. This pairing is what isolates the effect of fibre assignment in @sec:fa.

= Normalisation and estimators

The suites differ in number density (complete vs altmtl), bias and HOD. We compare variances both raw and after dividing out the Gaussian expectation $"Var"[B] prop P_"tot" (k)^3 \/ N_"tri"$. Here $P_"tot" = P_0 + P_"shot"$ is the mean monopole of each suite plus its shot noise, interpolated to the bispectrum bins. The ratio
$ R(k) = ("Var"_s [B(k)] \/ P_("tot", s)^3) / ("Var"_"holi" [B(k)] \/ P_("tot", "holi")^3) $
is 1 if two suites differ only through $P_"tot"$ in the Gaussian term. The scaling ignores the non-Gaussian $B B$, $P T$ and $P_6$ terms, which scale differently with bias and density, so $R != 1$ by itself is not a discrepancy. A band ratio is the ratio of the summed (scaled) variances over the bins in the band. Its error comes from a delete-block jackknife with 40 blocks, applied to both suites and added in quadrature. Per-element errors would understate the uncertainty because neighbouring elements are correlated.

To test consistency at the level of the full covariance, we compute, for each realisation of suite $s$, $chi^2 = (bold(d) - bold(mu)_s)^T hat(C)^(-1) (bold(d) - bold(mu)_s)$. Here $hat(C)^(-1)$ is the Hartlap-corrected inverse sample covariance of the reference suite. If the two covariances agree, $chevron.l chi^2 chevron.r \/ p = 1$. For the reference suite itself we split the realisations into two halves and use one half's covariance on the other half. The Hartlap factors are 0.91 (holi), 0.925 (GLAM) and 0.24 (Abacus-HF complete). The 25-realisation suites have $N < p$ and give a singular covariance, so we use them only as test samples, never as references.

= Results

== Mean signal and relative errors

#figure(image("../figures/mean_bispectra.pdf", width: 100%), caption: [Mean $k^2 B_(000)$ (top) and $k^2 B_(202)$ (bottom), NGC. GLAM ELG has a higher amplitude than the other suites. Holi QSO $B_(000)$ is lower at $k < 0.1$ and flatter at high $k$ than every other suite.]) <fig:mean>

#figure(image("../figures/relerr.pdf", width: 100%), caption: [Relative error $sigma(B) \/ |chevron.l B_(000) chevron.r|$ per bin, NGC. For LRG all suites lie on top of each other. Abacus complete has the smallest ELG and QSO errors because its density is higher. Holi QSO has the largest $B_(000)$ error of all suites and grows with $k$.]) <fig:relerr>

== Variances relative to holi

#figure(image("../figures/varratio_scaled.pdf", width: 100%), caption: [$P_"tot"^3$-scaled variance ratio $R(k)$ relative to holi, NGC.]) <fig:var>

#figure(
  text(size: 8pt, table(columns: 8, align: (left, left, center, center, center, center, center, center),
  table.header(table.cell(rowspan: 2)[*suite*], table.cell(rowspan: 2)[*bin*], table.cell(colspan: 3)[$B_(000)$, $k$ band], table.cell(colspan: 3)[$B_(202)$, $k$ band],
    [0.02--0.08], [0.08--0.14], [0.14--0.205], [0.02--0.08], [0.08--0.14], [0.14--0.205]),
  [GLAM v2 altmtl], [LRG1], [$0.95 plus.minus 0.04$], [$1.08 plus.minus 0.04$], [$1.14 plus.minus 0.04$], [$0.97 plus.minus 0.04$], [$1.06 plus.minus 0.03$], [$1.05 plus.minus 0.03$],
  [], [LRG2], [$1.00 plus.minus 0.05$], [$1.09 plus.minus 0.04$], [$1.13 plus.minus 0.04$], [$1.01 plus.minus 0.04$], [$1.11 plus.minus 0.03$], [$1.09 plus.minus 0.03$],
  [], [LRG3], [$1.03 plus.minus 0.05$], [$0.92 plus.minus 0.03$], [$0.84 plus.minus 0.03$], [$1.04 plus.minus 0.05$], [$1.00 plus.minus 0.02$], [$0.97 plus.minus 0.02$],
  [], [ELG1], [$1.02 plus.minus 0.04$], [$1.07 plus.minus 0.03$], [$1.08 plus.minus 0.02$], [$1.10 plus.minus 0.05$], [$1.05 plus.minus 0.03$], [$1.07 plus.minus 0.03$],
  [], [ELG2], [$1.02 plus.minus 0.05$], [$1.09 plus.minus 0.03$], [$1.02 plus.minus 0.02$], [$0.99 plus.minus 0.04$], [$1.09 plus.minus 0.03$], [$1.02 plus.minus 0.02$],
  [], [QSO], [$1.05 plus.minus 0.04$], [$0.61 plus.minus 0.03$], [$0.26 plus.minus 0.02$], [$1.09 plus.minus 0.05$], [$0.97 plus.minus 0.02$], [$0.84 plus.minus 0.03$],
  [Abacus-HF complete], [LRG1], [$0.88 plus.minus 0.09$], [$0.88 plus.minus 0.08$], [$0.74 plus.minus 0.05$], [$1.01 plus.minus 0.11$], [$0.82 plus.minus 0.06$], [$0.79 plus.minus 0.05$],
  [], [LRG2], [$0.98 plus.minus 0.11$], [$0.92 plus.minus 0.07$], [$0.89 plus.minus 0.05$], [$1.04 plus.minus 0.10$], [$0.89 plus.minus 0.05$], [$0.80 plus.minus 0.04$],
  [], [LRG3], [$1.05 plus.minus 0.13$], [$0.85 plus.minus 0.06$], [$0.59 plus.minus 0.04$], [$0.81 plus.minus 0.08$], [$0.80 plus.minus 0.05$], [$0.67 plus.minus 0.04$],
  [], [ELG1], [$1.05 plus.minus 0.11$], [$0.89 plus.minus 0.05$], [$0.99 plus.minus 0.05$], [$1.06 plus.minus 0.11$], [$0.99 plus.minus 0.07$], [$0.90 plus.minus 0.05$],
  [], [ELG2], [$1.34 plus.minus 0.16$], [$1.11 plus.minus 0.07$], [$1.01 plus.minus 0.04$], [$0.98 plus.minus 0.11$], [$1.00 plus.minus 0.07$], [$0.87 plus.minus 0.04$],
  [], [QSO], [$1.41 plus.minus 0.13$], [$0.72 plus.minus 0.05$], [$0.29 plus.minus 0.02$], [$0.98 plus.minus 0.10$], [$1.02 plus.minus 0.04$], [$0.95 plus.minus 0.05$],
  [Abacus-HF altmtl], [LRG1], [$0.91 plus.minus 0.19$], [$0.94 plus.minus 0.12$], [$0.75 plus.minus 0.09$], [$1.57 plus.minus 0.28$], [$0.84 plus.minus 0.15$], [$0.70 plus.minus 0.09$],
  [], [LRG2], [$0.81 plus.minus 0.14$], [$0.95 plus.minus 0.12$], [$0.88 plus.minus 0.10$], [$1.40 plus.minus 0.27$], [$1.03 plus.minus 0.10$], [$0.87 plus.minus 0.11$],
  [], [LRG3], [$1.36 plus.minus 0.35$], [$0.87 plus.minus 0.09$], [$0.74 plus.minus 0.06$], [$0.99 plus.minus 0.19$], [$1.07 plus.minus 0.15$], [$1.09 plus.minus 0.11$],
  [], [ELG1], [$0.91 plus.minus 0.23$], [$0.94 plus.minus 0.11$], [$0.93 plus.minus 0.08$], [$1.05 plus.minus 0.13$], [$0.87 plus.minus 0.08$], [$1.02 plus.minus 0.11$],
  [], [ELG2], [$1.37 plus.minus 0.35$], [$1.08 plus.minus 0.13$], [$0.89 plus.minus 0.07$], [$0.81 plus.minus 0.10$], [$0.91 plus.minus 0.11$], [$1.00 plus.minus 0.09$],
  [], [QSO], [$1.18 plus.minus 0.23$], [$0.53 plus.minus 0.05$], [$0.26 plus.minus 0.03$], [$0.74 plus.minus 0.12$], [$0.77 plus.minus 0.07$], [$0.76 plus.minus 0.05$],
  [Abacus 2nd-gen altmtl], [LRG1], [$0.91 plus.minus 0.15$], [$0.73 plus.minus 0.08$], [$0.74 plus.minus 0.09$], [$1.17 plus.minus 0.27$], [$0.79 plus.minus 0.10$], [$0.70 plus.minus 0.09$],
  [], [LRG2], [$1.02 plus.minus 0.22$], [$0.99 plus.minus 0.13$], [$0.70 plus.minus 0.08$], [$0.73 plus.minus 0.23$], [$0.85 plus.minus 0.13$], [$0.93 plus.minus 0.11$],
  [], [LRG3], [$0.92 plus.minus 0.20$], [$0.88 plus.minus 0.12$], [$0.67 plus.minus 0.10$], [$0.62 plus.minus 0.11$], [$1.04 plus.minus 0.12$], [$1.10 plus.minus 0.16$],
  [], [ELG1], [$0.95 plus.minus 0.23$], [$1.13 plus.minus 0.16$], [$1.11 plus.minus 0.10$], [$0.66 plus.minus 0.11$], [$0.98 plus.minus 0.12$], [$0.93 plus.minus 0.09$],
  [], [ELG2], [$0.95 plus.minus 0.21$], [$0.93 plus.minus 0.11$], [$1.11 plus.minus 0.09$], [$0.96 plus.minus 0.15$], [$1.20 plus.minus 0.14$], [$1.02 plus.minus 0.09$],
  [], [QSO], [$1.23 plus.minus 0.20$], [$0.56 plus.minus 0.07$], [$0.26 plus.minus 0.03$], [$0.97 plus.minus 0.14$], [$0.95 plus.minus 0.10$], [$0.79 plus.minus 0.06$],
  )),
  caption: [$P_"tot"^3$-scaled band variance ratio relative to holi in three $k$ bands, NGC, with jackknife errors.],
) <tab:bands>

Holi and GLAM agree within 15% for all LRG and ELG bins (@fig:var, @tab:bands). The differences are significant, though:

- GLAM is 8--14% higher for LRG1 and LRG2 at $k > 0.08$.
- GLAM is 16% lower for LRG3 at $k > 0.14$.
- GLAM is 2--10% higher for ELG.

The raw ratios are nearly the same (for example $1.19 plus.minus 0.04$, $1.16$, $0.84$ for LRG1--3 at $k > 0.14$), because the two suites have similar $P_"tot"$. SGC shows the same pattern with larger excursions: for example 1.26 and 1.31 for LRG1 and LRG2 at $k > 0.14$.

All three Abacus suites agree with each other and are 10--40% below holi for LRG at $k > 0.14$: for example 0.74 (LRG1), 0.59 (LRG3) in Abacus complete. For ELG they are within $approx 10%$ of holi. The mean $B_(000)$ of Abacus LRG is about 10--15% lower than holi at high $k$ (@fig:mean). Non-Gaussian terms of the form $B^2$ would scale as the square of this, so part of the deficit may follow from the signal rather than from the covariance model of the approximate mocks. Resolving this requires the analytic non-Gaussian terms.

== Correlation structure

#figure(image("../figures/corr_LRG1_NGC.pdf", width: 100%), caption: [Correlation matrices of $[B_(000), B_(202)]$ for LRG1 NGC. The number of realisations is in parentheses.]) <fig:corr>

#figure(image("../figures/corr_summary.pdf", width: 100%), caption: [Median nearest-neighbour correlation and the mean correlation at separations $|i - j| >= 10$ ("floor"), NGC. Error bars are the sampling errors $(1 - rho^2)\/sqrt(N)$ and $1\/sqrt(N N_"pairs")$.]) <fig:corrsum>

#figure(
  text(size: 8pt, table(columns: 8, align: (left, left, center, center, center, center, center, center),
  table.header([*suite*], [*quantity*], [LRG1], [LRG2], [LRG3], [ELG1], [ELG2], [QSO]),
  [holi], [$rho_(i, i+1)$, $B_(000)$], [0.45], [0.42], [0.34], [0.15], [0.10], [0.52],
  [], [floor, $B_(000)$], [0.15], [0.11], [0.10], [0.01], [0.01], [0.28],
  [], [$rho_(i, i+1)$, $B_(202)$], [0.38], [0.38], [0.27], [0.20], [0.11], [0.12],
  [GLAM], [$rho_(i, i+1)$, $B_(000)$], [0.46], [0.45], [0.33], [0.17], [0.12], [0.07],
  [], [floor, $B_(000)$], [0.15], [0.14], [0.08], [0.02], [0.01], [0.02],
  [], [$rho_(i, i+1)$, $B_(202)$], [0.38], [0.41], [0.29], [0.21], [0.13], [0.07],
  [Abacus-HF compl.], [$rho_(i, i+1)$, $B_(000)$], [0.38], [0.40], [0.30], [0.16], [0.18], [0.07],
  [], [floor, $B_(000)$], [0.13], [0.06], [0.08], [0.03], [0.02], [0.03],
  [], [$rho_(i, i+1)$, $B_(202)$], [0.35], [0.34], [0.26], [0.24], [0.12], [0.07],
  [Abacus-HF altmtl], [$rho_(i, i+1)$, $B_(000)$], [0.40], [0.38], [0.25], [0.14], [0.06], [0.04],
  [], [floor, $B_(000)$], [0.12], [0.01], [0.02], [0.02], [-0.00], [-0.00],
  [], [$rho_(i, i+1)$, $B_(202)$], [0.35], [0.37], [0.32], [0.26], [0.15], [0.11],
  [Abacus 2nd altmtl], [$rho_(i, i+1)$, $B_(000)$], [0.32], [0.40], [0.26], [0.09], [0.10], [0.02],
  [], [floor, $B_(000)$], [0.15], [0.10], [0.02], [0.03], [0.01], [0.02],
  [], [$rho_(i, i+1)$, $B_(202)$], [0.36], [0.39], [0.32], [0.21], [0.17], [0.04],
  )),
  caption: [Correlation summary, NGC.],
) <tab:corr>

The correlation structure is the same in every suite except holi QSO (@fig:corr, @fig:corrsum, @tab:corr):

- Nearest-neighbour correlations fall from $approx 0.45$ (LRG1, LRG2) to $0.33$ (LRG3), $approx 0.15$ (ELG1), $0.10$ (ELG2) and $0.07$ (QSO). This ordering follows the relative importance of the non-Gaussian terms, which grows with density and bias.
- For LRG there is a broad positive floor at large separations ($0.15$ in LRG1, $0.10$ in LRG3). It is consistent with a coherent amplitude modulation, of the kind a long-mode or $k_3 approx 0$ term produces. For ELG and QSO the floor is $lt.tilde 0.02$.
- The $B_(000)$--$B_(202)$ cross-correlation at equal $k$ is small and negative, between $-0.05$ and $-0.09$.

The marginal distributions are close to Gaussian: the median excess kurtosis is between $-0.15$ and $0.12$ in all groups except holi QSO ($0.32$).

== $chi^2$ consistency

#figure(image("../figures/chi2_consistency.pdf", width: 100%), caption: [Mean $chi^2 \/ p$ of each suite's realisations under the holi (left) and GLAM (right) covariance, NGC. The diagonal entries use the half-split. Statistical errors are $approx 0.01$ for holi and GLAM and $approx 0.03$ for the 25-realisation suites.]) <fig:chi2>

@fig:chi2 shows the following:

- The half-split self-consistency is $0.97$--$1.04$ for holi and $0.98$--$1.03$ for GLAM.
- GLAM under the holi covariance gives $1.09$, $1.11$, $0.93$ (LRG1--3), $1.06$, $1.05$ (ELG) and $0.96$ (QSO).
- Holi under the GLAM covariance gives $0.90$--$1.07$, except QSO at $1.41$.

The QSO asymmetry arises because the holi covariance is larger than GLAM's in a few directions, so GLAM realisations look "too good" in holi's metric while holi realisations look "too bad" in GLAM's.

Abacus-HF altmtl under the holi covariance gives $chi^2 \/ p = 0.88$--$1.12$. It is therefore compatible with holi to $approx 10%$ overall, even though the high-$k$ LRG variances are lower. The complete suites have much smaller $chi^2$ ($0.3$--$0.8$) because their density is higher. This is the expected sign and size, and it does not indicate a problem.

== Fibre assignment <sec:fa>

#figure(image("../figures/fibre_assignment.pdf", width: 100%), caption: [Paired variance ratio $"Var"("altmtl") \/ "Var"("complete")$ for the same 25 Abacus-HF boxes, NGC, raw (solid) and $P_"tot"^3$-scaled (dashed).]) <fig:fa>

#figure(
  table(columns: 6, align: center,
  table.header([*bin*], [$P_"tot"$ ratio ($k = 0.1$)], [$B_(000)$ raw], [$B_(000)$ scaled], [$B_(202)$ raw], [$B_(202)$ scaled]),
  [LRG1], [1.01], [1.00], [0.94], [1.11], [1.09],
  [LRG2], [1.01], [1.13], [1.09], [1.04], [1.00],
  [LRG3], [1.02], [1.25], [1.16], [1.41], [1.35],
  [ELG1], [1.21], [1.79], [1.04], [1.74], [1.02],
  [ELG2], [1.25], [1.82], [0.91], [2.24], [1.11],
  [QSO], [1.50], [2.37], [0.77], [2.72], [0.83],
  ),
  caption: [Median paired variance ratio, altmtl over complete, for Abacus-HF v2, NGC.],
) <tab:fa>

Since the altmtl and complete Abacus-HF mocks share the same boxes, their ratio isolates the effect of fibre assignment with little cosmic variance. The results are in @fig:fa and @tab:fa:

- The raw $B_(000)$ variance grows by 1.0--1.25 for LRG, 1.8 for ELG and 2.4 for QSO.
- After $P_"tot"^3$ scaling the ratios are $0.9$--$1.1$ for LRG1, LRG2 and ELG, and $0.77$--$0.83$ for QSO. Most of the increase is therefore the loss of effective density.
- LRG3 retains a residual excess of 16% ($B_(000)$) and 35% ($B_(202)$) that the scaling does not explain.
- The altmtl mean $B_(000)$ is lower by 0.3--0.8 standard deviations.

The 2nd-gen LRG pair gives 1.2--1.4. Its ELG pair is not like-for-like (`ELG_LOP` vs `ELG_LOPnotqso`) and is not interpreted.

== The holi QSO anomaly

#figure(image("../figures/holi_qso.pdf", width: 100%), caption: [Left: distribution of the QSO $B_(000)$ common mode, i.e. the mean standardised residual over $k$. Middle: standard deviation of the common mode for every tracer bin. Right: $chi^2$ of each QSO realisation under its own suite's covariance, compared with $chi^2_(74)$.]) <fig:qso>

Holi QSO differs from every other suite in several independent ways:

+ Variance. The $B_(000)$ variance is $1.6$ times GLAM's at $0.08 < k < 0.14$ and $3.8$ times at $k > 0.14$. All four other suites agree with each other here (scaled ratios $0.26$--$0.30$ in the top band). $B_(202)$ is affected much less (ratios $0.8$--$1.0$).
+ Coherence. The nearest-neighbour correlation of $B_(000)$ is $0.52$ and the floor is $0.28$, against $0.07$ and $approx 0.02$ elsewhere. A single amplitude mode carries the excess: its standard deviation is $0.60$ against $0.18$--$0.25$ in the other suites (@fig:qso, middle). In the same realisations the $P_0$ common mode is normal ($0.16$). The anomaly is therefore specific to the three-point function.
+ Tail. The common-mode distribution has a long positive tail (@fig:qso, left), and the excess kurtosis is $0.32$. Under holi's own covariance, 11 realisations have $chi^2$ $p < 10^(-4)$: NGC 401, 416; SGC 109, 139, 198, 266, 275, 367, 605, 894, 979. Among the other tracer bins only one realisation is flagged, LRG1 NGC 333.
+ Mean. The mean holi QSO $B_(000)$ has a different shape from the other suites (@fig:mean): it is lower at large scales and flatter at high $k$.

This combination points to a property of the holi QSO catalogues that varies from realisation to realisation and adds small-scale three-point power: for example, the satellite fraction or the redshift-error model. The excess sits partly in the tail and partly in the bulk. In SGC, removing the 9 flagged realisations moves the GLAM/holi $B_(000)$ variance ratio at $k > 0.14$ from 0.31 to 0.38. Removing the 5% of realisations with the largest common mode moves it to 0.66 (NGC: from 0.25 to 0.39). Neither removal brings it to 1. The QSO power spectrum is unaffected, so the problem would not have been visible in the two-point validation. For QSO bispectrum covariances we currently recommend GLAM.

== Signal-to-noise

The cumulative $S\/N = (bold(mu)^T hat(C)^(-1) bold(mu))^(1\/2)$ for the $p = 74$ data vector in GCcomb is:

#figure(
  table(columns: 13, align: center,
  table.header([], table.cell(colspan: 2)[LRG1], table.cell(colspan: 2)[LRG2], table.cell(colspan: 2)[LRG3], table.cell(colspan: 2)[ELG1], table.cell(colspan: 2)[ELG2], table.cell(colspan: 2)[QSO]),
  [], [holi], [GLAM], [holi], [GLAM], [holi], [GLAM], [holi], [GLAM], [holi], [GLAM], [holi], [GLAM],
  [$S\/N$], [51], [52], [71], [65], [60], [69], [37], [44], [38], [50], [14], [32],
  ),
  caption: [Cumulative $S\/N$, $0.02 <= k < 0.205 thin h"/Mpc"$, GCcomb, with the Hartlap-corrected inverse covariance.],
) <tab:snr>

The inflated holi QSO covariance halves the QSO $S\/N$ (14 against 32). The GLAM ELG $S\/N$ is higher mainly because its signal is higher (@fig:mean).

= Findings and implications for the analytic covariance

+ For LRG and ELG, holi and GLAM give the same bispectrum covariance to within 15% and are mutually $chi^2$-consistent at the 10% level. Either can serve as the mock reference for validating the analytic model, provided we allow a $plus.minus 10%$ systematic band between the two.
+ Holi QSO $B_(000)$ is anomalous: its variance is 2--4 times too large, it has a coherent amplitude mode, and 11 realisations are outliers. This needs to be reported to the holi mock producers. GLAM and Abacus are mutually consistent for QSO.
+ The N-body Abacus mocks show 10--40% less LRG variance at high $k$ than the approximate mocks. With 25--100 realisations this is a 3--10$sigma$ effect in the 0.14--0.205 band (@tab:bands), and it is the most informative target for the non-Gaussian terms of the analytic model.
+ Fibre assignment raises the variance mostly through the loss of effective density, which our model captures through $n_w$ and the window. A residual of $approx 15$--$35%$ remains for LRG3.
+ With only diagonal configurations available, the $k_3 approx 0$ (squeezed) term contributes to every element of the covariance. The LRG correlation floor of 0.10--0.15 is a direct test of that term together with the long-mode terms.
+ Convergence is not a limitation for holi or GLAM: the Hartlap factor is above 0.9 at $p = 74$. The Abacus suites cannot provide a full-rank reference covariance.

#v(6pt)
#text(size: 8.5pt)[Scripts: `inventory.py`, `collect.py`, `analysis.py`, `make_report_figures.py`. Numerical results: `results.json`, `figures/figure_results.json`.]
