import numpy as np, sys
sys.path.insert(0, '/home/user/thecov')
from desi_validation.compare_kernel import get
from desi_validation.ssc_check import summarize
from desi_validation.ssc_plots import fig_offdiag, fig_variance
S = '/tmp/claude-0/-home-user-thecov/90e4caa1-064c-5b2e-b2e3-4ff61a970c03/scratchpad'
r, sig = sys.argv[1], sys.argv[2]
z = np.load(S + '/t0run/report_data_holi-kcore2-LRG1.npz'); s = np.load(S + '/t0run/ssc_holi-kcore2-LRG1.npz')
fog = np.load(S + f'/t0_fog_{r}_s{sig}.npz')
V, C, k = get(z, 'LRG1', r, 1, 'kernel'); nb = len(k)
b3 = s[f'{r}/bias_galileon'][3]
base = s[f'{r}/C_ssc'] + s[f'{r}/C_disc']
T0 = s[f'{r}/C_T0_snake'] + s[f'{r}/C_T0_star']
T0f = fog['snake'] + fog['star']
M = {'SSC + disc': base, 'SSC + disc + T0': base + T0, f'SSC + disc + T0 FoG {sig}': base + T0f,
     f'SSC + disc + T0 FoG {sig} b3=0': base + T0f - b3 * fog['star_b3']}
for n, X in M.items():
    summarize(f'{n:34s}', V, C, X, k, nb)
d = np.diag(C)
for n, X in [('T0', T0), ('T0 FoG', T0f)]:
    print(n, 'diag/C l=0 k~0.05,0.15,0.25,0.3', np.round(np.diag(X)[[6, 26, 46, nb - 1]] / d[[6, 26, 46, nb - 1]], 3), ' l=2', np.round(np.diag(X)[nb + np.array([6, 26, 46, nb - 1])] / d[nb + np.array([6, 26, 46, nb - 1])], 3))
fig_offdiag(r, V, C, k, M, S + f'/t0_offdiag_{r}.png')
fig_variance({r: (V, C, k, M)}, S + f'/t0_variance_{r}.png')
