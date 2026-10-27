"""Tree-level T0 (snake + star) for the LRG1/QSO bins from the saved inputs, with a choice of the power inside the kernels.

    python t0_run.py <tracer> <region> <variant> <out.npz> [n_workers]

variant: 'prod'      P_ir A (the production inputs: IR-damped, amplitude-normalised linear power), sigma_v from the fit
         'fog<x>'    same with sigma_v = x Mpc/h
         'meas'      the measured nonlinear P_g divided by the Kaiser-FoG monopole factor as the 'matter' power inside the
                     kernels (b1 fixed), sigma_v from the fit; beyond the measured range it continues as P_ir A
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thecov import GaussianCovariance, PowerSpectrum, Tracer, TrispectrumCovariance, galileon_bias

tr, reg, variant, out = sys.argv[1:5]
n_workers = int(sys.argv[5]) if len(sys.argv) > 5 else 1
Z = np.load(os.environ.get('T0_INPUTS', 'handoff/t0_inputs.npz'))   # bundle built by ssc_check/t0_bundle on Perlmutter
F = lambda key: Z[f'F/{tr}/{reg}/{key}']
b1, b2, bs2, f, zeff = F('params'); ir_sigma, sigma_v, A_norm = F('damping'); J4 = float(F('J4'))
kl, PirA = F('plin_k'), F('plin_P_damped_normalised')
kedges, k = F('k_edges'), F('k')
nb = len(k)

if variant.startswith('fog'):
    sigma_v = float(variant[3:])
P_in = PirA
if variant == 'meas':
    # P_g,0 measured (window-convolved, mocks' mean) / dilution -> P_g; divide by the Kaiser-FoG monopole with the fitted b1, f
    P0 = F('mock_mean')[:nb] / F('dilution')
    x, w = np.polynomial.legendre.leggauss(32)
    kaiser0 = np.sum(w[None, :] * (b1 + f * x[None, :] ** 2) ** 2 * np.exp(-(k[:, None] * x[None, :] * sigma_v) ** 2), 1) / 2
    Pm_meas = P0 / kaiser0
    # splice: measured where available (log-interpolated), P_ir A outside, matched at the ends by the ratio at the edges
    lo, hi = k[0], k[-1]
    ratio = np.exp(np.interp(np.log(np.clip(kl, lo, hi)), np.log(k), np.log(Pm_meas / np.interp(k, kl, PirA))))
    P_in = PirA * ratio
    print('meas/PirA ratio at k = 0.03, 0.1, 0.2, 0.3:', np.round(np.interp([0.03, 0.1, 0.2, 0.3], kl, ratio), 3), flush=True)

rng = np.random.default_rng(0); pos = rng.uniform(0, 500, (2000, 3)) + [0, 0, 1500]
dummy = Tracer('T', {'POSITION': pos, 'WEIGHT': np.ones(2000), 'NZ': np.full(2000, 1e-4)}, 1.0)
cov = GaussianCovariance([dummy], PowerSpectrum({0: (kl, P_in)}, kedges=kedges, ells=(0, 2, 4)), smoothing=None)
gb = galileon_bias(b1, b2, bs2, b3=0.0)
t0 = time.time()
tc = TrispectrumCovariance(cov, (kl, P_in), gb, f=f, n_workers=n_workers, sigma_fog=sigma_v, n_mu=20, n_psi=40, J4=J4)
comp = tc.components([('T', 'T')], ells=(0, 2, 4), verbose=True)
np.savez(out, snake=comp['snake'], star=comp['star'], sigma_v=sigma_v, variant=variant, P_in_k=kl, P_in=P_in)
print(f'{variant}: {time.time() - t0:.0f} s, sigma_v {sigma_v:.2f}', flush=True)
