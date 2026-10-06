import numpy as np, sys, time
sys.path.insert(0, '/home/user/thecov')
from thecov.trispectrum import TrispectrumCovariance, Bias
S = '/tmp/claude-0/-home-user-thecov/90e4caa1-064c-5b2e-b2e3-4ff61a970c03/scratchpad'
def main():
    r, sig = sys.argv[1], float(sys.argv[2])
    pl = np.load(S + '/plin_z051.npz'); kl, Pl, f = pl['k'], pl['P'], float(pl['f'])
    z = np.load(S + '/t0run/report_data_holi-kcore2-LRG1.npz'); s = np.load(S + '/t0run/ssc_holi-kcore2-LRG1.npz')
    class Cov: pass
    cov = Cov(); cov.k_edges = z[f'LRG1/{r}/x1/k_edges']; cov.ells = (0, 2, 4)
    J4 = {'NGC': 9.700e-10, 'SGC': 1.829e-09}[r]
    t = time.time()
    tc = TrispectrumCovariance(cov, (kl, Pl), Bias(*s[f'{r}/bias_galileon']), f=f, J4=J4, n_workers=4, sigma_fog=sig,
                               n_mu=20, n_psi=40).components([('T', 'T')])
    np.savez(S + f'/t0_fog_{r}_s{sig:.2f}.npz', **tc)
    print(r, sig, f'{time.time() - t:.0f} s')
if __name__ == '__main__':
    main()
