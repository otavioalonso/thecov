import numpy as np, sys, io, contextlib, time
sys.path.insert(0,'/home/user/thecov'); sys.path.insert(0,'/home/user/thecov/tests')
from test_discreteness import p_lin
from thecov.trispectrum import multipoles, Bias
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
with contextlib.redirect_stdout(io.StringIO()):
    from powercovfft import PowerSpecCovFFT
    pc = PowerSpecCovFFT()
k, P = p_lin()
pc.set_power_law_decomp({'nu': -0.3, 'kmin': 1e-5, 'kmax': 1e1, 'nmax': 512})
spl = InterpolatedUnivariateSpline(np.log(k), np.log(P)); pc.pk_lin_spl = spl; pc.decomp.compute(pc.get_pk_lin)
Pf = lambda q: np.where(q > 1e-6, np.exp(spl(np.log(np.clip(q, 1e-6, None)))), 0.0)
ks = np.array([0.05, 0.1, 0.2])
K1, K2 = np.meshgrid(ks, ks, indexing='ij')
pc.calc_master_integral(K1, K2)
V = 1.0
cases = [dict(b1=2.0), dict(b1=2.0, b2=0.8), dict(b1=2.0, bG2=-0.3), dict(b1=2.0, b3=0.5), dict(b1=2.0, bG3=0.5),
         dict(b1=2.0, bdG2=0.5), dict(b1=2.0, bGamma3=0.5)]
f = 0.7
for c in cases:
    full = dict(b1=2.0, b2=0.0, bG2=0.0, b3=0.0, bG3=0.0, bdG2=0.0, bGamma3=0.0); full.update(c)
    pc.set_params(V, f, full, 1e-3); pc.calc_base_integral()
    B = Bias(**full)
    t0 = time.time()
    mine = {(i, j): multipoles(ks[i], ks[j], Pf, B, f) for i in range(3) for j in range(3)}
    dt = time.time() - t0
    print(c, f'({dt / 9:.2f} s per k-pair)')
    for (l1, l2) in [(0, 0), (2, 2), (4, 4), (0, 2), (0, 4), (2, 4)]:
        sn = pc.get_cov_T2211(l1, l2, K1, K2); st = pc.get_cov_T3111(l1, l2, K1, K2)
        ms = np.array([[mine[(i, j)]['snake'][(l1, l2)] for j in range(3)] for i in range(3)])
        mt = np.array([[mine[(i, j)]['star'][(l1, l2)] for j in range(3)] for i in range(3)])
        print(f'  ({l1},{l2}) snake mine/theirs', np.round(ms / sn, 4).ravel(), ' star', np.round(mt / st, 4).ravel())
