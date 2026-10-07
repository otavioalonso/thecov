import sys, numpy as np
sys.path.insert(0, '/home/user/thecov')
from thecov.trispectrum import TrispectrumCovariance, Bias
from thecov import response_split, CovarianceTemplates
from thecov.covariance_tools import window_convolve
from desi_validation.compare_kernel import get, chi2
def main():
    D, label, b = sys.argv[1], sys.argv[2], sys.argv[3]
    s = np.load(f'{D}/ssc_{label}.npz'); z = np.load(f'{D}/report_data_{label}.npz')
    for r in ('NGC', 'SGC'):
        V, C, k = get(z, b, r, 1, 'kernel'); nb = len(k); N = len(V); S = np.cov(V.T)
        ssc, disc = s[f'{r}/C_ssc'], s[f'{r}/C_disc']
        base = C + ssc + disc
        Ct = window_convolve(s[f'{r}/C_T0_snake'] + s[f'{r}/C_T0_star'], C)
        class Cv: pass
        cv = Cv(); cv.k_edges = s[f'{r}/k_edges']; cv.ells = (0, 2, 4)
        f = float(s[f'{r}/params'][3]); sv = float(s[f'{r}/damping'][1])
        t0 = TrispectrumCovariance(cv, (s[f'{r}/plin_k'], s[f'{r}/plin_P_damped_normalised']), Bias(*s[f'{r}/bias_galileon']),
                                   f=f, J4=float(s[f'{r}/J4']), sigma_fog=sv, n_mu=20, n_psi=40)
        tpl = CovarianceTemplates(C); ref = tpl.loglike(S, N)
        L = np.linalg.cholesky(C); Li = np.linalg.inv(L)
        def row(name, X):
            M = C + X; v = CovarianceTemplates(M).loglike(S, N)
            lam = np.linalg.eigvalsh(Li @ M @ Li.T).min()
            try: c2 = f'{chi2(V, M, np.arange(3 * nb)):.4f}'
            except np.linalg.LinAlgError: c2 = ' n/PD '
            print(f'    {name:42s} ' + (f'{v - ref:+9.1f}' if np.isfinite(v) else '   not PD') + f'  {c2}  min eig {lam:.3f}', flush=True)
        print(f'== {b} {r}: -2 dlnL vs Gaussian (unfitted), chi2/n')
        row('SSC + disc (recommended)', ssc + disc)
        for ks in (0.04, 0.06, 0.08, 0.10, 0.12):
            coll = window_convolve(t0.collapsed([('T', 'T')], ks), C)
            p = response_split(Ct, k, ks, base - C, C_collapsed=coll)
            print(f'  k_split {ks}:')
            row('+ tree LL (both k < k_split)', ssc + disc + p['LL'])
            row('+ tree LL + squeezed LH + completion', ssc + disc + p['LL'] + p['LH'] + p['completion'])
            row('+ collapsed (response) only', ssc + disc + p['HH_collapsed'])
            row('+ all response pieces', ssc + disc + p['LL'] + p['LH'] + p['completion'] + p['HH_collapsed'])
if __name__ == '__main__':
    main()
