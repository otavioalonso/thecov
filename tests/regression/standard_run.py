"""Regression reference for the refactoring: the standard production call sequence on a small synthetic survey.

It mirrors what our standard runs do, with the same arguments:
  * Gaussian term : desi_validation.desi_compare.thecov_covariance (kernel smoothing, masked model, estimator norm);
  * SSC           : desi_validation.ssc_check (local-window GaussianCovariance as container, LA + Poisson,
                    norm_kind variants, dilution from the kernel I_k);
  * discreteness  : DiscretenessCovariance.components, window_convolve with the Gaussian C;
  * T0            : TrispectrumCovariance.components + collapsed_covariance, response_split / response_covariance.

    python tests/regression/standard_run.py write    # store the reference (run on the pre-refactoring version)
    python tests/regression/standard_run.py check    # recompute and compare with the stored reference

Everything is seeded, so with the same backend the outputs must agree to rounding.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))

from thecov import (Tracer, PowerSpectrum, GaussianCovariance, WindowSmoothing,  # noqa: E402
                    SuperSampleCovariance, DiscretenessCovariance, TrispectrumCovariance,
                    galileon_bias, response_split, response_covariance)
from thecov.tree.windowing import window_convolve  # noqa: E402
from thecov.tree.response_approach import collapsed_covariance  # noqa: E402

REF = os.path.join(HERE, 'standard_run_reference.npz')
CENTER = np.array([0.0, 0.0, 1200.0])        # close enough for the line of sight to vary across the survey


def p_lin():
    k = np.logspace(-4, 1, 3000)
    return k, 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)


def survey(R, n, holes, seed):
    """Uniform sphere of radius R around CENTER with spherical holes (veto-mask-like)."""
    rng = np.random.default_rng(seed)
    out, m = [], 0
    while m < n:
        x = rng.uniform(-R, R, size=(4 * n, 3))
        keep = np.sum(x * x, 1) < R * R
        for c, rh in holes:
            keep &= np.sum((x - c) ** 2, 1) > rh * rh
        out.append(x[keep])
        m += int(keep.sum())
    return np.concatenate(out)[:n] + CENTER


def run():
    t0 = time.time()
    R, nbar, b1, f = 250.0, 5e-4, 2.0, 0.7
    b2, bs2 = 0.3, -4.0 / 7.0 * (b1 - 1.0)
    rng = np.random.default_rng(1)
    holes = [(rng.uniform(-200, 200, 3), rng.uniform(4, 12)) for _ in range(200)]
    pos = survey(R, 20_000, holes, seed=2)
    ref = survey(R, 200_000, holes, seed=3)
    wfun = lambda x: 1.0 / (1.0 + nbar * 1e4) * (1 + 0.1 * np.sin(x[:, 0] / 50.0))   # FKP-like, smoothly varying
    w = wfun(pos)
    V_eff = 4 / 3 * np.pi * R ** 3 * len(survey(R, 50_000, holes, seed=4)) / len(survey(R, 50_000, (), seed=4))
    alpha = nbar * V_eff / len(pos)
    tr = Tracer('T', {'POSITION': pos, 'WEIGHT': w, 'NZ': np.full(len(pos), nbar), 'NW': nbar * w}, alpha)
    kl, Pl = p_lin()
    k_edges = np.arange(0.02, 0.1601, 0.02)
    kc = 0.5 * (k_edges[1:] + k_edges[:-1])
    g0 = b1 + f / 3
    multipoles = {0: (kl, (b1 ** 2 + 2 * b1 * f / 3 + f ** 2 / 5) * Pl),
                  2: (kl, (4 * b1 * f / 3 + 4 * f ** 2 / 7) * Pl), 4: (kl, 8 * f ** 2 / 35 * Pl)}
    out = {}

    # --- Gaussian, kernel smoothing (desi_compare.thecov_covariance) ---
    sm = WindowSmoothing(cell=6.0, r_max=100.0, tol=5e-3)
    sm.add_density('T', ref, wfun(ref), alpha * len(pos) / len(ref))
    sm.set_power('T', 'T', PowerSpectrum({0: (kc, np.interp(kc, kl, g0 ** 2 * Pl))}))
    cov = GaussianCovariance([tr], k_edges, ds=2.0, ds_pair=10.0, n_sub=1500, n_near=8000, smoothing=sm)
    cov.compute_windows([('T', 'T')])
    norm = 0.98 * cov.I_local('T', 'T')
    power = PowerSpectrum(multipoles, kedges=k_edges, window_convolved=True, norm=norm)
    cov.set_power(power)
    C, _ = cov.covariance([('T', 'T')], ells=(0, 2, 4))
    out['C_gauss_kernel'] = C
    out['I_k_kernel'] = np.asarray(cov.I_k('T', 'T'))
    print(f'gaussian kernel done {time.time() - t0:.0f} s', flush=True)

    # --- local-window container used by the NG terms (ssc_check) ---
    covl = GaussianCovariance([tr], power, n_sub=1500, n_near=8000, ds=2.0, ds_pair=10.0, s_split=80.0,
                              smoothing=None)
    out['I_local'] = np.array(covl.I_local('T', 'T'))
    out['C_gauss_local'] = covl.covariance([('T', 'T')], ells=(0, 2, 4))[0]
    dil = out['I_k_kernel'] / norm

    # --- SSC (ssc_check variants) ---
    windows = None
    for key, la, pois in (('LA', True, True), ('LA_noPoisson', True, False), ('noLA', False, False)):
        ssc = SuperSampleCovariance(covl, (kl, Pl), b1=b1, f=f, b2=b2, bs2=bs2, local_average=la, n_near=8000,
                                    dilution=dil, discreteness=pois, norm_kind='data-randoms')
        if windows is None:
            ssc.compute_windows('T')
            windows = ssc.windows
        else:
            ssc.windows = windows
        out[f'C_ssc_{key}'] = ssc.covariance([('T', 'T')], ells=(0, 2, 4))[0]
        if key == 'LA':
            sig = ssc.sigma2('T')
            out['sigma2'] = np.array([sig[k_] for k_ in sorted(sig)])
            out['LA_integrals'] = np.array(ssc.discreteness_integrals('T'), dtype=float)
            out['ssc_a'], out['ssc_c'] = np.asarray(ssc.a), np.asarray(ssc.c)
    for kind in ('randoms', 'alpha'):
        sk = SuperSampleCovariance(covl, (kl, Pl), b1=b1, f=f, b2=b2, bs2=bs2, local_average=True, n_near=8000,
                                   dilution=dil, discreteness=True, norm_kind=kind)
        sk.windows = windows
        out[f'C_ssc_norm_{kind}'] = sk.covariance([('T', 'T')], ells=(0, 2, 4))[0]
    print(f'ssc done {time.time() - t0:.0f} s', flush=True)

    # --- discreteness (tree) ---
    disc = DiscretenessCovariance(covl, (kl, Pl), b1=b1, f=f, b2=b2, bs2=bs2)
    dcomp = disc.components([('T', 'T')], ells=(0, 2, 4))
    out['C_disc_B'], out['C_disc_P'] = dcomp['B'], dcomp['P']
    out['C_disc_windowed'] = window_convolve(dcomp['B'] + dcomp['P'], C)
    out['J_disc'] = np.array(disc.window_integrals('T'), dtype=float)
    print(f'discreteness done {time.time() - t0:.0f} s', flush=True)

    # --- T0 (tree) + response approach ---
    gb = galileon_bias(b1, b2, bs2, b3=0.0)
    t0c = TrispectrumCovariance(covl, (kl, Pl), gb, f=f, n_workers=1, n_mid=32, n_end=16)
    tc = t0c.components([('T', 'T')], ells=(0, 2, 4))
    for key, M in tc.items():
        out[f'C_T0_{key}'] = M
    k_split = 0.07
    Ccoll = collapsed_covariance(t0c, [('T', 'T')], k_split, ells=(0, 2, 4))
    out['C_T0_collapsed'] = Ccoll
    out['J4'] = np.array(t0c.window_integral('T'))
    Ct = tc['snake'] + tc['star']
    sp = response_split(window_convolve(Ct, C), kc, k_split, C + out['C_ssc_LA'], C_collapsed=window_convolve(Ccoll, C))
    for key, M in sp.items():
        out[f'C_T0_split_{key}'] = M
    out['C_T0_response'] = response_covariance(sp)
    print(f'T0 done {time.time() - t0:.0f} s', flush=True)
    return out


def main(mode):
    out = run()
    if mode == 'write':
        np.savez(REF, **out)
        print('reference written:', REF, len(out), 'arrays')
        return 0
    ref = np.load(REF)
    worst, bad = 0.0, []
    for key in ref.files:
        a, b = np.asarray(ref[key], float), np.asarray(out.get(key, np.nan), float)
        if a.shape != b.shape:
            bad.append(f'{key}: shape {a.shape} -> {b.shape}')
            continue
        d = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))
        worst = max(worst, d)
        if d > 1e-10:
            bad.append(f'{key}: max |diff| / max |ref| = {d:.2e}')
    missing = sorted(set(ref.files) - set(out))
    print(f'{len(ref.files)} arrays compared; worst relative difference {worst:.2e}')
    for line in bad + [f'missing: {m}' for m in missing]:
        print('  ', line)
    return 1 if bad or missing else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'check'))
