# Converged pair-averaged clustering window (`WindowSmoothing`): validation results

Runs at NERSC on `thecov2` @ `1a6794b` (two-stage basis), holi-v3-altmtl mock 173, 859 realisations,
caches `~/thecov_desi/holi_v3_mock173`. Steps 1-5 of `TASK_kernel_validation.md`.
Logs: `~/thecov/logs/kcore2_{NGC,SGC,all,QSO_NGC,QSO_SGC,QSO_all}.log`.
Labels: `holi-kcore2-{NGC,SGC,}-LRG1`, `holi-kcore2-{NGC,SGC,}-QSO`, all with `--cache-tag _r8p10`,
`--target-near-pairs 1e10 --n-randoms-max 8e6`.

## Bottom line

The converged two-stage basis **fixes the first-bin normalisation** (basis I_k error 6.6e-2 -> 1.1e-5;
predicted calibration 1.324 -> 1.298 NGC against the GRF 1.276) but **does not reduce chi2/n**:
LRG1 GCcomb is 1.055 at k = 0.02-0.2, the same as the unconverged single-stage run (1.049) and still
above hand-tuned fill-production (1.019). The residual is therefore **not basis truncation**. The
variance ratios locate it at k > 0.06: the kernel window is slightly *better* than fill below
k = 0.06 and ~1-2% low in predicted variance above it, and the excess sits entirely in the
bin-width-independent coefficient a, so it is still a **Gaussian-term** error rather than
non-Gaussianity. Independently of the mocks, the kernel's I_k misses the GRF calibration curve by
+1.5% in the first bin and -1.7% in the last — a ~3% shape error that convergence did not remove.
The QSO control is unaffected (chi2/n 0.996 GCcomb, indistinguishable from the default window).

## Basis sizes, errors, timings (step 1)

Both convergence errors are below the 2e-3 target everywhere; the window basis stays at B <= 9, so
the step-4 escape (`--kernel-tol 5e-3`) was **not triggered** and was not run.

| run | kernel basis N (I_k err, kernel err) | window basis B (I_k err, window err) | smoothing | pair counts (n_near) | job total |
|---|---|---|---|---|---|
| LRG1 NGC | 19 (1.3e-3, 1.3e-3) | 8 (1.1e-5, 1.4e-3) | 261 s | 798 s (2.97e6) | 1086 s |
| LRG1 SGC | 19 (1.7e-3, 1.6e-3) | 8 (8.4e-6, 1.6e-3) | 126 s | 597 s (2.03e6) | 749 s |
| QSO NGC  | 20 (1.3e-3, 3.6e-4) | 8 (6.1e-5, 1.9e-3) | 987 s | 1241 s (4.00e6) | 1613 s |
| QSO SGC  | 21 (4.3e-5, 6.1e-5) | 9 (1.8e-5, 1.2e-3) | 524 s | 856 s (4.00e6) | 1202 s |

All four fitted in the 30 min debug queue. `I_k / int m_A m_B` (first, last bin):
LRG1 0.7707, 0.8457 (NGC) and 0.7804, 0.8627 (SGC); QSO 0.8775, 0.9430 (NGC) and 0.8793, 0.9487 (SGC).

`int m (K_k * m) / norm` (first bin ... middle ... last):

| run | first | middle | last | int m^2 / norm |
|---|---|---|---|---|
| LRG1 NGC | 0.9443 | 1.0171 | 1.0362 | 1.2252 |
| LRG1 SGC | 0.9446 | 1.0230 | 1.0443 | 1.2104 |
| QSO NGC  | 0.9740 | 1.0291 | 1.0468 | 1.1101 |
| QSO SGC  | 0.9734 | 1.0312 | 1.0502 | 1.1070 |

## LRG1: chi2/n (step 3)

`python -m desi_validation.compare_kernel holi-kcore2-LRG1:kernel holi-fillprod-LRG1:fill
holi-kcore-LRG1:kernel holi-kcore-LRG1:random-density`

chi2/n, k = 0.02-0.2, dk = 0.005 (x1), ells 0/2/4 — the updated reference table:

| | NGC | SGC | GCcomb |
|---|---|---|---|
| default m^2 (random-density) | 1.090 | 1.095 | 1.095 |
| kernel, single-stage, 12 kernels (I_k err 6.6e-2), default sampling | 1.048 | 1.044 | 1.049 |
| fill (m x healpix fill fraction, nside 128, hand-tuned), default sampling | 1.034 | 1.029 | - |
| fill, production sampling | 1.014 | 1.023 | 1.019 |
| **kernel, two-stage converged, production sampling** | **1.049** | **1.061** | **1.055** |

Over k = 0.02-0.3: default 1.096 / 1.099 / 1.098; kernel single-stage 1.059 / 1.050 / 1.056;
fill-production 1.024 / 1.031 / 1.026; **kernel two-stage 1.062 / 1.070 / 1.065**.

chi2/n of the converged kernel per k range (NGC / SGC / GCcomb):

| k range | 0.02-0.06 | 0.06-0.12 | 0.12-0.2 | 0.2-0.3 |
|---|---|---|---|---|
| kernel two-stage | 1.020 / 1.005 / 1.024 | 1.039 / 1.045 / 1.043 | 1.059 / 1.065 / 1.058 | 1.084 / 1.073 / 1.078 |
| fill-production | 1.000 / 0.982 / 1.003 | 1.009 / 1.014 / 1.012 | 1.023 / 1.030 / 1.022 | 1.045 / 1.036 / 1.040 |
| kernel single-stage | 1.024 / 0.989 / 1.021 | 1.040 / 1.031 / 1.039 | 1.056 / 1.047 / 1.050 | 1.077 / 1.047 / 1.066 |

## LRG1: variance ratios — where the residual sits

var(mocks)/var(thecov), **kernel two-stage / fill-production / default**:

| | 0.02-0.06 | 0.06-0.12 | 0.12-0.2 | 0.2-0.3 |
|---|---|---|---|---|
| NGC ell=0 | 1.040 / 1.058 / 1.136 | 1.053 / 1.052 / 1.123 | 1.118 / 1.105 / 1.169 | 1.278 / 1.258 / 1.312 |
| NGC ell=2 | 1.015 / 1.028 / 1.111 | 1.048 / 1.045 / 1.122 | 1.074 / 1.061 / 1.126 | 1.085 / 1.065 / 1.117 |
| NGC ell=4 | 1.004 / 1.013 / 1.095 | 0.999 / 0.996 / 1.070 | 1.035 / 1.022 / 1.087 | 1.013 / 0.996 / 1.044 |
| SGC ell=0 | 1.020 / 1.028 / 1.108 | 1.054 / 1.043 / 1.120 | 1.160 / 1.136 / 1.212 | 1.202 / 1.171 / 1.240 |
| SGC ell=2 | 1.019 / 1.021 / 1.107 | 1.062 / 1.050 / 1.131 | 1.056 / 1.035 / 1.106 | 1.110 / 1.081 / 1.147 |
| SGC ell=4 | 1.017 / 1.014 / 1.103 | 1.048 / 1.035 / 1.116 | 1.033 / 1.011 / 1.082 | 1.026 / 0.999 / 1.060 |

At k = 0.02-0.06 — the range the converged basis was expected to change — the kernel is now **better**
than fill-production for ell = 0 and 2 in NGC (1.040 vs 1.058, 1.015 vs 1.028) and comparable in SGC.
Above k = 0.06 the kernel is consistently 0.010-0.025 higher (i.e. its predicted variance is that much
low), and that is what the chi2/n gap is made of.

Bin-width split, var ratio - 1 = a + b (dk/0.005), NGC+SGC mean, ell = 0 / 2 / 4:

| | a (k < 0.2) | b (k < 0.2) | a (0.2-0.3) |
|---|---|---|---|
| default | 0.106 / 0.096 / 0.079 | 0.046 / 0.022 / 0.011 | 0.162 / 0.084 / 0.032 |
| kernel single-stage | 0.036 / 0.025 / 0.008 | 0.042 / 0.019 / 0.009 | 0.115 / 0.038 / -0.011 |
| fill-production | 0.035 / 0.021 / 0.004 | 0.043 / 0.021 / 0.011 | 0.103 / 0.025 / -0.023 |
| **kernel two-stage** | **0.045 / 0.032 / 0.016** | **0.040 / 0.018 / 0.008** | **0.130 / 0.052 / 0.001** |

b is the same for every window (0.04 / 0.02 / 0.01), confirming it is non-Gaussian and not a window
effect. a is 2-4x smaller than the default for both kernel runs, but ~0.01 larger than
fill-production, consistently with the ratios above.

## LRG1: window calibration vs the GRF (step 3)

Predicted `int m^2 / int m (K_k * m)` against the exact-Gaussian factor `c(k) = norm / I_k` measured on
the real footprint with a Gaussian random field (`run_gaussian_footprint.py`; the bands are
0.02-0.05, 0.05-0.1, 0.1-0.2, 0.2-0.3, per `report/window_kernel.typ`). The k bins compared are the
first (k = 0.0225, band 1), middle (k ~ 0.16, band 3) and last (k = 0.2975, band 4):

| | first bin vs 0.02-0.05 | middle vs 0.1-0.2 | last vs 0.2-0.3 |
|---|---|---|---|
| NGC GRF c(k) | 1.276 | 1.214 | 1.202 |
| NGC two-stage | **1.2975** (+1.7%) | 1.2046 (-0.8%) | 1.1824 (-1.6%) |
| NGC single-stage | 1.3236 (+3.7%) | 1.2018 (-1.0%) | 1.1960 (-0.5%) |
| SGC GRF c(k) | 1.262 | 1.193 | 1.179 |
| SGC two-stage | **1.2814** (+1.5%) | 1.1832 (-0.8%) | 1.1591 (-1.7%) |
| SGC single-stage | 1.3087 (+3.7%) | 1.1810 (-1.0%) | 1.1714 (-0.7%) |

(The 0.05-0.1 band, 1.242 NGC / 1.220 SGC, has no logged per-bin counterpart.)

The first bin moved from 1.324 to 1.298 (NGC) and 1.309 to 1.281 (SGC), i.e. towards the GRF values
1.276 / 1.262 as expected from the converged basis (task prediction: "~1.28"), halving the deviation
from +3.7% to +1.5-1.7%. But validation step 1 of `window_kernel.typ` — "I_k from the kernel must
reproduce the GRF calibration curve, with no mocks involved" — is **not** met: the kernel's I_k is
~1.5% too low in the first bin and ~1.6-1.7% too high in the last, a ~3% shape error across
0.02-0.3 that convergence did not remove (and the last bin got slightly worse, -0.5% -> -1.6%).
This is the cleanest statement of the residual, because it involves no mocks and no non-Gaussian
contamination.

Cross-check against the GRF *variance* (Var GRF / thecov-default = 1.094, 1.038, 1.064, 1.050 NGC and
1.071, 1.099, 1.074, 1.059 SGC): if the kernel window reproduced the exact Gaussian variance, the
measured var(mocks)/var(thecov-kernel) should equal the default ratio divided by those factors.
For ell = 0, measured minus expected is +0.002, -0.029, +0.019, +0.028 (NGC) and -0.015, +0.035,
+0.032, +0.031 (SGC) — agreement in the lowest band, and the kernel delivering 2-3% less variance
than the exact Gaussian answer above k ~ 0.1. The band edges of the two calculations differ
(0.02-0.06 vs 0.02-0.05, etc.), so this is indicative only; the clean test is validation step 2,
below.

## Step 4 — not triggered

Gate: B > ~10 or pair counts that do not fit 30 min. Measured B = 8, 8, 8, 9 and pair counts
597-1241 s inside 1086-1613 s jobs, so no `--kernel-tol 5e-3` run was made. Tolerance insensitivity is
already demonstrated more strongly by the pair of runs in hand: 12 kernels with an I_k error of 6.6e-2
and 19 kernels with 1.1e-5 give chi2/n 1.048 and 1.049 (NGC, k = 0.02-0.2), a 0.001 difference.

## QSO control (step 5)

`python -m desi_validation.compare_kernel --bin QSO holi-kcore2-QSO:kernel holi-altmtl:random-density
holi-fill-QSO:fill`

chi2/n, k = 0.02-0.2 (x1):

| | NGC | SGC | GCcomb |
|---|---|---|---|
| kernel two-stage | 1.003 | 0.993 | 0.996 |
| default m^2 (random-density) | 1.007 | 0.995 | 1.000 |
| fill | 1.000 | 0.989 | 0.993 |

Over 0.02-0.3: kernel 1.007 / 1.000 / 1.003; default 1.009 / 1.000 / 1.004; fill 1.003 / 0.995 / 0.998.
Per k range the three windows agree to within 0.016 everywhere (largest at k = 0.02-0.06 NGC:
kernel 0.992, default 1.008, fill 0.998; elsewhere <= 0.008); kernel - default ranges over
-0.016 to +0.006, i.e. the kernel is never worse than the default by more than 0.006. a (k < 0.2) is
-0.007 / -0.009 / -0.011 (kernel) vs -0.005 / -0.004 / -0.008 (default). The kernel does not spoil QSO, as required; QSO has
int m^2 / norm = 1.11 (vs 1.23 for LRG1), so there is little for the pair-averaging to correct.

Predicted calibration for QSO: NGC 1.1398 / 1.0787 / 1.0605 and SGC 1.1372 / 1.0735 / 1.0541
(first / middle / last bin). No GRF run exists for QSO to compare against.

## Interpretation and what is left

1. Converging the kernel basis is necessary for a correct I_k at low k, and it works: the first-bin
   calibration deviation halves (+3.7% -> +1.7% NGC), the k < 0.06 variance ratios improve past
   fill-production, and the basis itself now reproduces I_k to 1e-5.
2. It is not sufficient. What remains is still a **Gaussian-term** error, not non-Gaussianity: the
   excess is in a (the dk-independent intercept, 0.045 / 0.032 / 0.016 for ell = 0 / 2 / 4) and not in
   b (0.040 / 0.018 / 0.008, identical across all four windows including the default). Validation plan
   step 3 of `window_kernel.typ` expected a to vanish and leave only the non-Gaussian remainder; it did
   not. Comparing against the GRF, the default window's Gaussian deficit is 5-9% in variance, and the
   kernel recovers roughly half to two-thirds of it.
3. Candidates for the residual, in order:
   (a) **The isotropic kernel.** K_k(r) = xi_0(r) j_0(kr)/P_0(k) uses the real-space monopole, but the
       estimator sees the anisotropic redshift-space xi, and the multipole covariances weight
       directions unequally. The rms kernel width collapses from 55.7 Mpc/h in the first bin to
       8.2 Mpc/h in the middle bin (LRG1 NGC), so at high k the kernel samples exactly the small-r,
       most anisotropic part of xi — which is where the I_k error and the missing variance both are.
       `window_kernel.typ` caveat (i) argued the top-hat test suggested the isotropic kernel suffices;
       these numbers do not support that, and it is now the leading untested assumption.
   (b) **Mesh resolution of the narrow kernels.** The painted density uses 3-5 Mpc/h cells, so an
       8.2 Mpc/h (LRG1) or 1.0-3.0 Mpc/h (QSO) rms kernel is barely or not resolved. This is a listed
       convergence test (halve the cell) that has not been run at production sampling. QSO validating
       is a weak counter-argument: its int m^2 / norm is 1.11, so there is little correction to get
       wrong there.
   (c) **The k ~ k' approximation** inside a window width, caveat (ii) of the note — a low-k
       limitation, so it does not explain a residual concentrated above k = 0.06.
   (d) Residual non-Gaussianity: excluded by the b coefficients above.
4. The hand-tuned fill window still fits better (GCcomb 1.019 vs 1.055), but it has a tuned scale and
   the kernel has none, so `kernel` is not yet justified as the DESI default on chi2/n.
5. **Next test.** Not more mock chi2 — validation step 2 of the note, which has never been run:
   `run_gaussian_footprint.py` with NW = K_k * m, compared against the GRF variance at every k and bin
   width. That isolates whether W_k is right, with no mocks and no non-Gaussian contamination, and it
   decides between (a) and (b) when repeated with a halved mesh cell. The anisotropic
   redshift-space kernel, tested against a GRF in redshift space, follows if (a) is confirmed.
