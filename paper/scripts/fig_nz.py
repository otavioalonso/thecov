"""Figure: the radial long modes of the holi mocks, from the weighted n(z) of all 1000 mocks (20 bins of dz = 0.01 for
LRG1). (a) n(z) of every LRG1 NGC mock relative to the median: twelve mocks (red) have 55-65% more galaxies in
0.46 < z < 0.47, in both caps; they are excluded from every statistic in the paper. (b) rms of the shell-count
fluctuation (Poisson removed) over linear theory for a slab of the same thickness, (b1 + f) sigma_slab, as a function of
the slab thickness, with all mocks (open) and without the defective ones (filled)."""
import json
import os
from scipy.integrate import quad
from common import *

OM = 0.3153


def chi(z):
    return 2997.92458 * quad(lambda x: 1 / np.sqrt(OM * (1 + x) ** 3 + 1 - OM), 0, z)[0]


def slab_sigma(dr, A, kl, P):
    q = np.linspace(1e-4, 2.0, 200000)
    return np.sqrt(np.trapezoid(np.interp(q, kl, P) * np.sinc(q * dr / 2 / np.pi) ** 2, q) / np.pi / A)


bad = {b: excluded(b) for b in ('LRG1', 'QSO')}
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
z = load('nz_scatter_LRG1')
NZ, ids, e = z['NGC/NZ'], z['NGC/mocks'], z['edges']
zc = 0.5 * (e[1:] + e[:-1])
d = NZ / np.median(NZ, 0) - 1
ax = axes[0]
isbad = np.isin(ids, bad['LRG1'])
for row in d[~isbad][:300]:
    ax.plot(zc, row * 100, color=MUTED, lw=0.3, alpha=0.4)
for row in d[isbad]:
    ax.plot(zc, row * 100, color=RED, lw=0.9)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$n(z) / {\rm median} - 1$ [%]')
ax.set_title(f'(a) LRG1 NGC: {isbad.sum()} defective mocks (red)', fontsize=8)

ax = axes[1]
out = {}
for b, c in (('LRG1', BLUE), ('QSO', ORANGE)):
    z = load(f'nz_scatter_{b}')
    h = holi(b, 'NGC')
    b1, f = h['params'][0], h['params'][3]
    kl, P = h['plin_k'], h['plin_P_damped_normalised']
    e = z['edges']
    c0, c1 = chi(e[0]), chi(e[-1])
    for cap, ls in (('NGC', '-'), ('SGC', '--')):
        NZ, ids = z[f'{cap}/NZ'], z[f'{cap}/mocks']
        A = float(z[f'{cap}/area']) * (c1 ** 3 - c0 ** 3) / 3 / (c1 - c0)
        for clean, mk in ((False, 'o'), (True, 'o')):
            M = NZ[~np.isin(ids, bad[b])] if clean else NZ
            n = len(M)
            xs, ys, es = [], [], []
            for m in (1, 2, 5, 10, 20):
                mg = M.reshape(n, m, -1).sum(2)
                meas = mg.std(0, ddof=1) / mg.mean(0)
                clus = np.sqrt(np.maximum(meas ** 2 - 1 / mg.mean(0), 0)).mean()
                th = (b1 + f) * slab_sigma((c1 - c0) / m, A, kl, P)
                xs.append((c1 - c0) / m)
                ys.append(clus / th)
                es.append(clus / th / np.sqrt(2 * (n - 1)) * np.sqrt(m))
            if clean or bad[b]:
                ax.errorbar(xs, ys, es, fmt=mk + ls, color=c, ms=3, lw=1.0, mfc=c if clean else 'white',
                            label=(f'{b} {cap}' if clean else None))
            out[f'{b}/{cap}/{"clean" if clean else "all"}'] = dict(scale=xs, ratio=ys)
ax.axhline(1, color=INK2, lw=0.7)
ax.set_xscale('log')
ax.set_xlabel(r'slab thickness [$h^{-1}$Mpc]')
ax.set_ylabel(r'$\sigma_{\rm mocks} / \sigma_{\rm linear}$')
ax.set_title('(b) radial modes: mocks / linear theory', fontsize=8)
ax.legend(fontsize=6.5, ncol=2)
fig.tight_layout(w_pad=0.8)
save(fig, 'nz')
write_numbers('nz', out)
