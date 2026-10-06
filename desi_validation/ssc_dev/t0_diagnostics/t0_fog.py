import numpy as np, sys, time
sys.path.insert(0, '/home/user/thecov')
from scipy.optimize import least_squares
from thecov.trispectrum import TrispectrumCovariance, Bias
from desi_validation.compare_kernel import get, chi2
S = '/tmp/claude-0/-home-user-thecov/90e4caa1-064c-5b2e-b2e3-4ff61a970c03/scratchpad'
D = S + '/t0run'
pl = np.load(S + '/plin_z051.npz'); kl, Pl, f = pl['k'], pl['P'], float(pl['f'])
z = np.load(D + '/report_data_holi-kcore2-LRG1.npz'); s = np.load(D + '/ssc_holi-kcore2-LRG1.npz')
r = sys.argv[1] if len(sys.argv) > 1 else 'NGC'
J4 = {'NGC': 9.700e-10, 'SGC': 1.829e-09}[r]
V, C, k = get(z, 'LRG1', r, 1, 'kernel'); nb = len(k)
edges = z[f'LRG1/{r}/x1/k_edges']
dil = s[f'{r}/dilution']
# --- sigma_v from the mock mean multipoles: A * dil * int (b1 + f mu^2)^2 P exp(-(k mu sig)^2) L_l
def main():
    x, w = np.polynomial.legendre.leggauss(32)
    Pk = np.interp(k, kl, Pl)
    b1g = float(s[f'{r}/params'][0])
    def model(p):
        A, sig = p; b1 = b1g
        kern = (b1 + f * x[None] ** 2) ** 2 * np.exp(-(k[:, None] * x[None] * sig) ** 2)
        return np.concatenate([A * dil * Pk * (2 * l + 1) / 2 * np.sum(w * kern * np.polynomial.legendre.Legendre.basis(l)(x), 1) for l in (0, 2)])
    m = V.mean(0)[:2 * nb]; e = np.sqrt(np.diag(C))[:2 * nb]
    sel = (np.r_[k, k] <= 0.3) & (np.r_[k, k] >= 0.05)
    fit = least_squares(lambda p: ((model(p) - m) / e)[sel], [0.7, 3.0])
    A, sig = fit.x; sig = abs(sig)
    print(f'{r}: FoG fit (b1 {b1g:.3f} fixed) A {A:.3f} sigma_v {sig:.2f} Mpc/h; chi2/dof {np.sum(fit.fun ** 2) / (sel.sum() - 2):.2f}', flush=True)
    res = (model(fit.x) - m) / m
    print('  rel. residual P0 at k 0.05,0.15,0.25,0.3:', np.round(res[[6, 26, 46, nb - 1]], 3), ' P2:', np.round(res[nb + np.array([6, 26, 46, nb - 1])], 3))
    class Cov: pass
    cov = Cov(); cov.k_edges = edges; cov.ells = (0, 2, 4)
    gb = Bias(*s[f'{r}/bias_galileon'])
    out = {}
    for name, sv, nm, npsi in [('base', 0.0, 12, 24), ('fog', sig, 20, 40)]:
        t = time.time()
        tc = TrispectrumCovariance(cov, (kl, Pl), gb, f=f, J4=J4, n_workers=4, sigma_fog=sv, n_mu=nm, n_psi=npsi).components([('T', 'T')])
        out[name] = tc
        print(name, f'{time.time() - t:.0f} s', flush=True)
    np.savez(S + f'/t0_fog_{r}.npz', sigma_v=sig, **{f'{n}_{p}': M for n, d in out.items() for p, M in d.items()})
    ref = s[f'{r}/C_T0_snake'] + s[f'{r}/C_T0_star']
    loc = out['base']['snake'] + out['base']['star']
    print('calibration local/run diag l=0 at bins 6,26,46:', np.round(np.diag(loc)[[6, 26, 46]] / np.diag(ref)[[6, 26, 46]], 3), ' l=2:', np.round(np.diag(loc)[[nb + 6, nb + 26]] / np.diag(ref)[[nb + 6, nb + 26]], 3))

if __name__ == '__main__':
    main()
