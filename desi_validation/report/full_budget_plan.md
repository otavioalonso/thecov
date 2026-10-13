# Plan: the full covariance budget, term by term (2026-10-13)

Goal set by the user: include every term that is supposed to exist (Gaussian, SSC with the local average, the
discreteness terms B/n and P/n^2, T0), then understand why the mocks agree or disagree with each. This note fixes
what "understand" has to mean after HANDOFF 7-9 and `report/ssc_t0_review.md`, and lists the tests in order.

## 1. Where each term stands

| term | implementation | validated against | status against the mocks |
|---|---|---|---|
| Gaussian, xi-kernel window | `WindowSmoothing` | exact Gaussian fields on the DESI randoms (holes resolved): P0 to ~1 %, LRG1 and LRG2 (`grf_randoms.py`); PS/SS terms, P2/P4 and RSD in progress | k < 0.1 agrees once SSC + disc are added; l = 4 given l = 0, 2 is 7-11 % high (7.9) |
| SSC + local average | `ssc.py` | algebra rederived (review 1.1); sphere test (lam = 0 channel only) | P2 response right, P0 net response ~3x weaker in the mocks (9.2); norm scatter half-modelled (9.3) |
| discreteness B/n, P/n^2 | `discreteness.py` (= WS20 SN-NG) | counting and window integrals rederived (review 1.1); **no test on a field with known statistics yet** | mocks want B/n x 1.8-2.3 (LRG1), P/n^2 x 2.2-4.1 (QSO) (9.4) |
| T0 tree level | `trispectrum.py` | PowerSpecCovFFT (l1 = l2), independent quadrature, IR cancellation (review 2.1) | amplitude 0.2-0.4 (holi and Abacus); not PD at amplitude 1 (7.1, 7.7) |

The three high-k terms (B/n, P/n^2, T0) are all smooth, broadband P(k) P(k')-like surfaces. The Wishart fits that
produced the amplitudes above split one measured excess between them by shape, with a 2-3 % error in the Gaussian
baseline underneath (review 2.2). So "T0 is 0.3 and disc is 2" is one solution of a degenerate system, not a
measurement of either term. Before any term is declared right or wrong, the degeneracy has to be broken by
something other than shape.

## 2. Two handles that are not shape

**Dilution (review 3.C).** Recompute each mock's multipoles on a random 50 % and 25 % subsample of its galaxies.
Under dilution by p: the Gaussian term changes as thecov predicts (shot noise 1/p), B/n scales as 1/p, P/n^2 as
1/p^2, SSC and T0 do not change (clustering statistics of a random subsample are those of the parent, including any
one-halo contribution). Three dilution levels give three equations for the three smooth terms plus the two
invariant ones, with no shape prior. This is the model-independent decomposition and it is what makes "include all
terms" testable. Cost: jaxpower on 859 mocks x 2 dilutions x 2 caps, a few node-hours in the debug queue, one job per
(cap, dilution); the catalogues and the pipeline exist (`dump_report_data` reads the spectra, so the new spectra go
into a sibling directory with the same layout).

**Fields with known statistics (what `grf_randoms.py` does).** Each term can be isolated on the real randoms:
- Gaussian field: only the Gaussian term (done; variants in the queue add PS/SS, multipoles, RSD).
- Gaussian field + Poisson sampling (`--poisson`, queued): B = 0, T = 0, no SSC beyond the box; the only non-Gaussian
  term is P/n^2, at a known amplitude. This tests `DiscretenessCovariance`'s P/n^2 term, including its window
  integrals and the local approximation of P(|k1 - k2|), i.e. candidates (i) and (iii) of 9.4 at once. If it passes,
  QSO's x2-4 is in the mocks (non-Poisson sampling), not in thecov.
- Lognormal field + Poisson sampling (next): B and T are nonzero and known in configuration space
  (1 + xi = exp(xi_g); the connected 3- and 4-point functions follow from exp(sum xi_g)). This tests B/n at a known
  B that is not tree level and that includes the squeezed configurations, and the sum SSC + T0 for a field whose
  super-sample content is known (the box). It does not test the tree-level *shapes*, which is the point: the question
  is whether thecov's discreteness machinery is right for any B, before asking which B the galaxies have.

## 3. Hypotheses for the disagreements, each with the test that decides it

1. **B/n x 2 (LRG1).** (a) Implementation or window integrals: decided by the Poisson and lognormal field tests.
   (b) The galaxies' bispectrum at k = 0.1-0.3 is larger than tree level (one-halo term, satellites): decided by
   the dilution test, which measures the 1/p term with whatever B the galaxies have; the ratio to tree level is then
   the one-halo excess, a statement about the HOD, not about thecov. (c) Per-mock FKP weights in J_Smm: a
   10-20 % effect at most (9.3), cannot give x2.
2. **P/n^2 x 2-4 (QSO).** Same split as 1(a)/(b); QSO at ~1 galaxy per halo should be close to Poisson, so a
   factor 2-4 that survives the Poisson field test would point at the estimator's shot-noise subtraction in the files
   (realised vs expected, the (1 + alpha) factor with the mocks' alpha) rather than at clustering.
3. **T0 amplitude 0.2-0.4.** Once the discreteness terms are pinned by dilution, the invariant remainder minus SSC
   is T0 at the mocks' nonlinear level. Its size relative to tree level is the physics result; the review's reading
   (tree-level Z2, Z3 with (f k mu)^n terms overpredict at k > 0.1 for LRG, part of the squeezed LL block is the SSC
   again) stands or falls on that number. No new thecov code is needed for this step.
4. **SSC P0 response 3x weaker (9.2).** Not a covariance-code question: the response is a property of the galaxies.
   Review 3.B (position-dependent power spectrum from sub-volumes of the mocks) measures it and replaces
   `responses()` by a table. The dilution spectra can be reused for this: sub-volume splits are a post-processing
   of the same catalogues.
5. **l = 4 given l = 0, 2 excess (7.9).** Gaussian-level or not: decided by the RSD variant of `grf_randoms.py`
   (queued; P0, P2, P4 on the DESI randoms with shot noise at the files' level) through
   `grf_randoms_blocks.py`. If the Gaussian fields show 1.00 and the mocks 1.07-1.11, it is a non-Gaussian l = 4
   coupling or the estimator's line-of-sight convention; if the fields show it too, it is thecov's (2,4)/(4,4) window
   treatment.

## 4. Order

1. Read the queued variants when they land: PS/SS (`_snL`), P2/P4 and RSD (`_rsd`, then `grf_randoms_blocks.py`
   for item 5), P/n^2 (`_poi`, item 1a/2a).
2. Dilution spectra on the mocks (section 2, first handle): the one new NERSC job set, 4-8 debug jobs.
3. Lognormal + Poisson field on the randoms (B/n at known B): a `--lognormal` option in `grf_randoms.py`, two
   field jobs and one thecov job for LRG1 NGC.
4. With 1-3 in hand, the amplitude table of 9.4 is replaced by measured per-term amplitudes with no shape prior,
   and the remaining T0 is compared with tree level. Then SSC responses from sub-volumes (item 4).

What not to do: fit amplitudes of smooth templates to the full mocks again, or window-convolve local terms
(`window_convolve` inflates them by (sum_j K_ij)^2, 9.4), before the dilution decomposition exists.
