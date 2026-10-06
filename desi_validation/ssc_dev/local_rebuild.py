"""Rebuild the non-Gaussian covariance terms locally from a NERSC ssc_check bundle and evaluate them on the mocks.

    python desi_validation/ssc_dev/local_rebuild.py --bundle DIR --bin LRG1 --region NGC --plin plin_z051.npz

Needs in DIR: report_data_<label>.npz (mocks + Gaussian covariance) and ssc_<label>.npz (long-mode variances,
dilution, biases, SSC pieces; with the parts saved by the current ssc_check also the window integrals and P_lin).
--plin: (k, P, f) table at z_eff when the npz has no P_lin (older runs; e.g. from CAMB). For older runs the
discreteness window integrals are passed with --jsmm --jss (from the log).
Recomputes SSC (normalised and BAO/FoG damped responses, long-mode variances x A) and the discreteness 4-point
terms (normalised, damped), optionally window-convolved, and prints chi2 / -2 dlnL / coherent amplitudes, unfitted.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from thecov.ssc import response_multipoles  # noqa: E402
from thecov.power import Dressed, no_wiggle, ir_damped  # noqa: E402
from thecov.discreteness import DiscretenessCovariance  # noqa: E402
from thecov.covariance_tools import window_convolve  # noqa: E402
from thecov import CovarianceTemplates  # noqa: E402
from desi_validation.compare_kernel import get, chi2  # noqa: E402
from desi_validation.ssc_check import summarize, fit_sigma_v, fit_amplitude  # noqa: E402


def rebuild(bundle, b, r, label=None, plin=None, jsmm=None, jss=None, ir_sigma=6.0, cosmo=(0.6766, 0.14239, 0.15745, 0.9665)):
    label = label or f'holi-kcore2-{b}'
    z = np.load(os.path.join(bundle, f'report_data_{label}.npz'))
    s = np.load(os.path.join(bundle, f'ssc_{label}.npz'))
    V, C, k = get(z, b, r, 1, 'kernel')
    nb = len(k)
    edges = z[f'{b}/{r}/x1/k_edges']
    b1, b2, bs2, f, zeff = s[f'{r}/params']
    if f'{r}/plin_k' in s.files:
        kl, Pl = s[f'{r}/plin_k'], s[f'{r}/plin_P']
    else:
        pl = np.load(plin)
        kl, Pl = pl['k'], pl['P']
    if jsmm is None:
        jsmm, jss = s[f'{r}/J_disc']
    dil = s[f'{r}/dilution']
    m = V.mean(0)
    sig = fit_sigma_v(k, m[:nb], m[nb:2 * nb], b1, f)
    A = fit_amplitude(k, m[:nb], dil, b1, f, sig, lambda x: np.interp(x, kl, Pl))
    h, om, fb, ns = cosmo
    Pir = A * ir_damped(kl, Pl, no_wiggle(kl, Pl, h, om, fb, n_s=ns), ir_sigma)
    # SSC: long-mode variances x A, responses from the dressed power, local average from the mock mean
    sig2 = {tuple(ast.literal_eval(p) for p in key.split('|')): v for key, v in zip(s[f'{r}/sigma2_keys'], s[f'{r}/sigma2'])}
    X = [('W', 0), ('W', 2), ('M', 0), ('M', 2)]
    Sm = A * np.array([[sig2.get((x, y), sig2.get((y, x))) for y in X] for x in X])
    xk, wk = np.polynomial.legendre.leggauss(4)
    kn = np.array([0.5 * (hi - lo) * xk + 0.5 * (hi + lo) for lo, hi in zip(edges[:-1], edges[1:])])
    wn = wk * kn ** 2
    wn /= wn.sum(1, keepdims=True)
    R = response_multipoles(kn.ravel(), Dressed(kl, Pir, sig), b1, f, b2, bs2)
    R = {key: np.sum(wn * v.reshape(kn.shape), 1) for key, v in R.items()}
    Pm = m.reshape(3, nb)
    g = {0: b1 + f / 3, 2: 2 * f / 3}
    Vc = np.array([np.concatenate([-Pm[i] * g[n] + (dil * R[(l, n)] if x == 'W' else 0)
                                   for i, l in enumerate((0, 2, 4))]) for (x, n) in X])
    ssc = Vc.T @ Sm @ Vc + (s[f'{r}/C_ssc'] - s[f'{r}/C_ssc_LA_noPoisson'])

    class Cov:
        pass
    cv = Cov()
    cv.k_edges, cv.ells = edges, (0, 2, 4)
    d = DiscretenessCovariance(cv, (kl, Pir), b1=b1, f=f, b2=b2, bs2=bs2, sigma_fog=sig, n_mu=16, n_phi=24)
    d.window_integrals = lambda _: (jsmm, jss)
    disc = d.components([('T', 'T')])
    return dict(V=V, C=C, k=k, nb=nb, ssc=ssc, disc_B=disc['B'], disc_P=disc['P'], sigma_v=sig, A=A, s=s, r=r)


def evaluate(p, extra=()):
    V, C, k, nb, s, r = p['V'], p['C'], p['k'], p['nb'], p['s'], p['r']
    N, Smp = len(V), np.cov(V.T)
    disc = p['disc_B'] + p['disc_P']
    variants = [('NERSC run (as saved)', s[f'{r}/C_ssc'] + s[f'{r}/C_disc']),
                ('SSC + disc (normalised, damped)', p['ssc'] + disc),
                ('SSC + disc windowed', p['ssc'] + window_convolve(disc, C))] + list(extra)
    ref = CovarianceTemplates(C).loglike(Smp, N)
    print(f"  sigma_v {p['sigma_v']:.2f} Mpc/h, A {p['A']:.3f}")
    for name, Cx in variants:
        print(f'  {name}: -2 dlnL vs Gaussian {CovarianceTemplates(C + Cx).loglike(Smp, N) - ref:+.1f}')
        summarize(f'{name[:30]:30s}', V, C, Cx, k, nb)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--label', default=None)
    ap.add_argument('--plin', default=None)
    ap.add_argument('--jsmm', type=float, default=None)
    ap.add_argument('--jss', type=float, default=None)
    a = ap.parse_args()
    p = rebuild(a.bundle, a.bin, a.region, a.label, a.plin, a.jsmm, a.jss)
    print(f'{a.bin} {a.region}')
    evaluate(p)
