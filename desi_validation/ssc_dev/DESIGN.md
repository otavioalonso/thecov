# SSC module for thecov: design (started 2026-10-05, not yet implemented)

Goal: Cov_SSC[P_l1(k1), P_l2(k2)] = sum_{n,n'} R_l1^(n)(k1) R_l2^(n')(k2) sigma^2_{nn'}, redshift space,
local line of sight, from thecov's existing window pair counts. Targets: HANDOFF.md item 4.
(arXiv is blocked from the cloud session, so Wadekar & Scoccimarro 2020 could not be read; the module is
derived from the same ingredients. Cross-check the responses against their Appendix when possible.)

## Responses (derive_responses.py, sympy; run it: ~minutes)
R(k, k^; q^) = lim_{eps->0} 2[Z2(ka,-q)Z1(ka)P(ka) + Z2(kb,-q)Z1(kb)P(kb)], ka = k, kb = -k+q, q = eps q^,
with the SCF99 Z1/Z2 kernels (b1, b2, bs2, f), P(kb)/P(k) = (|kb|/k)^n_eff. The 1/eps terms (bulk flow)
cancel; the O(1) remainder is the response (growth, dilation, bias, RSD/LOS-velocity-gradient terms).
Checked analytically for real space: isotropic 47/21 - n_eff/3, tidal (8/7 - n_eff)((k^.q^)^2 - 1/3).
Azimuthal average about n^, Legendre in mu_k (ell = 0,2,4) and in nu = q^.n^ (n = 0,2; check n = 4 is 0).
Output: R_l^(n)(k) = [a_ln(b1,b2,bs2,f) + c_ln(...) n_eff(k)] P_lin(k). Hard-code these in thecov/ssc.py,
with a test that re-derives them when sympy is importable.
Local average (randoms normalised by the data, as jaxpower/pypower): add R_LA = -2 (b1 + f nu^2) P_l(k),
i.e. n = 0: -2(b1 + f/3) P_l, n = 2: -(4f/3) P_l. Option to switch off.
Measured spectra are diluted by I_k/norm: multiply R by cov.I_k()/cov.I() per bin (masked convention).

## Long-mode variances from the window pair counts
With thecov's Q_{L1 L2 L}(s) = int dOmega_s int d^3x W(x) W(x+s) S_{L1 L2 L}(x^, x'^, s^) and
xi_lam(s) = int q^2 dq/(2 pi^2) P_lin(q) j_lam(qs):
  sigma^2_{nn'} = (1/I^2) (4 pi)^{3/2} sum_lam i^lam sqrt(2 lam+1) (n n' lam; 0 0 0) / sqrt((2n+1)(2n'+1))
                  int s^2 ds xi_lam(s) Q^{WW}_{n n' lam}(s),     I = int W.
Check: n = n' = 0 gives (1/I^2) int d^3s xi(s) Q_WW(s), the usual sigma_b^2.
Triples needed: (0,0,0), (0,2,2), (2,0,2), (2,2,0), (2,2,2), (2,2,4); the Gaussian run with ells and
L up to 4 already requests them (cached window files). Use the local window Window('W', A, B).
Test: uniform sphere with a distant observer: sigma^2_{02} = 0, sigma^2_{22} = sigma^2_{00}/5, and
sigma^2_00 = int q^2 dq/(2 pi^2) P(q) [3 j1(qR)/(qR)]^2.

## P_lin
User-supplied (k, P) at z_eff in (Mpc/h)^3 (cosmoprimo at NERSC). Keep an Eisenstein-Hu no-wiggle helper
for tests only.

## API sketch
    ssc = SuperSampleCovariance(cov, p_lin=(k, P), f=0.8, b1=2.0, b2=0.5, bs2=None, local_average=True)
    C_ssc, labels = ssc.covariance([('LRG','LRG')], ells=(0,2,4))   # same layout as cov.covariance
Validation (desi_validation): predicted sigma_0 = sqrt(C_ssc) along the P0 template vs 0.51/0.66%,
sigma_2/P0 vs 0.94/1.22% (NGC/SGC), ratio 1.30, correlation ~0; then rank1_excess.py with C_G + C_ssc.
