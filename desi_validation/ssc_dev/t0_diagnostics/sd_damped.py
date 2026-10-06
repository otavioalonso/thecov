"""Local rebuild of SSC (+LA) and discreteness 4-pt for LRG1 from the NERSC run's saved pieces, with BAO (IR) and
fingers-of-God damping; compared with the mocks."""
import numpy as np, sys, ast, time
sys.path.insert(0, '/home/user/thecov')
from thecov.ssc import response_multipoles
from thecov.power import Dressed, no_wiggle, ir_damped
from thecov.discreteness import DiscretenessCovariance
from desi_validation.compare_kernel import get
from desi_validation.ssc_check import summarize
S = '/tmp/claude-0/-home-user-thecov/90e4caa1-064c-5b2e-b2e3-4ff61a970c03/scratchpad'
JD = {'NGC': (2.208e-06, 5.058e-03), 'SGC': (4.699e-06, 1.211e-02)}
SIGV = {'NGC': 2.09, 'SGC': 2.75}

def main():
    r = sys.argv[1]
    sig_list = [float(x) for x in sys.argv[2:]] or [SIGV[r]]
    pl = np.load(S + '/plin_z051.npz'); kl, Pl, f = pl['k'], pl['P'], float(pl['f'])
    h = 0.6766; om = 0.02242 + 0.11933 + 0.06 / 93.14
    Pir = ir_damped(kl, Pl, no_wiggle(kl, Pl, h, om, 0.02242 / om), 6.0)
    z = np.load(S + '/t0run/report_data_holi-kcore2-LRG1.npz'); s = np.load(S + '/t0run/ssc_holi-kcore2-LRG1.npz')
    V, C, k = get(z, 'LRG1', r, 1, 'kernel'); nb = len(k)
    edges = z[f'LRG1/{r}/x1/k_edges']
    b1, b2, bs2, _, _ = s[f'{r}/params']
    dil = s[f'{r}/dilution']
    sig2 = {tuple(ast.literal_eval(p) for p in key.split('|')): v for key, v in zip(s[f'{r}/sigma2_keys'], s[f'{r}/sigma2'])}
    X = [('W', 0), ('W', 2), ('M', 0), ('M', 2)]
    Sm = np.array([[sig2.get((x, y), sig2.get((y, x))) for y in X] for x in X])
    xk, wk = np.polynomial.legendre.leggauss(4)
    kn = np.array([0.5 * (hi - lo) * xk + 0.5 * (hi + lo) for lo, hi in zip(edges[:-1], edges[1:])])
    wn = wk * kn ** 2; wn /= wn.sum(1, keepdims=True)
    Pm = V.mean(0).reshape(3, nb)
    g = {0: b1 + f / 3, 2: 2 * f / 3}
    def ssc(pd):
        R = response_multipoles(kn.ravel(), pd, b1, f, b2, bs2)
        R = {key: np.sum(wn * v.reshape(kn.shape), 1) for key, v in R.items()}
        Vc = []
        for (x, n) in X:
            vec = []
            for i, l in enumerate((0, 2, 4)):
                v = -Pm[i] * g[n]
                if x == 'W':
                    v = v + dil * R[(l, n)]
                vec.append(v)
            Vc.append(np.concatenate(vec))
        Vc = np.array(Vc)
        return Vc.T @ Sm @ Vc
    la_pois = s[f'{r}/C_ssc'] - s[f'{r}/C_ssc_LA_noPoisson']
    class Cov: pass
    cv = Cov(); cv.k_edges = edges; cv.ells = (0, 2, 4)
    def disc(P, sv):
        d = DiscretenessCovariance(cv, (kl, P), b1=b1, f=f, b2=b2, bs2=bs2, sigma_fog=sv,
                                   n_mu=16 if sv else 12, n_phi=24 if sv else 16)
        d.window_integrals = lambda A: JD[r]
        return d.covariance([('T', 'T')])[0]
    t = time.time()
    C_ssc0 = ssc(Dressed(kl, Pl))
    print(f'{r}: rebuild check, tree SSC (no Poisson LA): diag ratio rebuilt/NERSC l=0 bins 6,26,46:',
          np.round(np.diag(C_ssc0)[[6, 26, 46]] / np.diag(s[f'{r}/C_ssc_LA_noPoisson'])[[6, 26, 46]], 3),
          ' l=2:', np.round(np.diag(C_ssc0)[nb + np.array([6, 26, 46])] / np.diag(s[f'{r}/C_ssc_LA_noPoisson'])[nb + np.array([6, 26, 46])], 3), flush=True)
    Cd0 = disc(Pl, 0.0)
    print(f'   disc rebuild/NERSC diag l=0:', np.round(np.diag(Cd0)[[6, 26, 46]] / np.diag(s[f'{r}/C_disc'])[[6, 26, 46]], 3), f'({time.time() - t:.0f} s)', flush=True)
    variants = [('NERSC: SSC + disc', s[f'{r}/C_ssc'] + s[f'{r}/C_disc']),
                ('rebuilt: SSC + disc', C_ssc0 + la_pois + Cd0),
                ('BAO-damped', ssc(Dressed(kl, Pir)) + la_pois + disc(Pir, 0.0))]
    out = {}
    for sv in sig_list:
        Cs, Cd = ssc(Dressed(kl, Pir, sv)), disc(Pir, sv)
        variants.append((f'BAO + FoG {sv}', Cs + la_pois + Cd))
        out[f'ssc_s{sv}'], out[f'disc_s{sv}'] = Cs + la_pois, Cd
    np.savez(S + f'/sd_damped_{r}.npz', **out)
    for name, Cx in variants:
        summarize(f'{name:24s}', V, C, Cx, k, nb)

if __name__ == '__main__':
    main()
