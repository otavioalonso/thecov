"""Figure: the mocks in whitened coordinates, along directions chosen without looking at the mocks being plotted.

The 859 LRG1 NGC mocks are split at random into halves A and B. Directions are the leading eigenvectors of the whitened
sample covariance of A, C^{-1/2} S_A C^{-1/2}; the clouds show B projected on them, y = u^T C^{-1/2} (v - <v>_B), which
has unit variance in every direction if C is right (circles: 1 and 2 sigma; ellipse: B's own 1-sigma contour).
(a) Gaussian model, (b) recommended model; (c) physical directions, chosen a priori: the amplitudes of the monopole and
quadrupole (projections on the mean multipoles), all mocks, model ellipse dashed. (d) out-of-sample variance along A's
20 leading directions, measured on B, with the Wishart error, and the in-sample eigenvalues; (e) the two leading
directions of (a) as patterns in the data vector (in units of the Gaussian error per bin)."""
from matplotlib.patches import Ellipse, Circle
from common import *

d = holi('LRG1', 'NGC')
V, k, nb, N = d['V'], d['k'], len(d['k']), len(d['V'])
rng = np.random.default_rng(7)
perm = rng.permutation(N)
A, B = V[perm[:N // 2]], V[perm[N // 2:]]
NB = len(B)


def cv_directions(C):
    W, Wi = whitened_cov(A, C)
    lam, U = np.linalg.eigh(W)
    o = np.argsort(-lam)
    yB = (Wi @ (B - B.mean(0)).T).T @ U[:, o]
    return lam[o], U[:, o], yB, Wi


def ellipse(ax, cov2, **kw):
    w, v = np.linalg.eigh(cov2)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang, fill=False, **kw))


def cloud(ax, y, title, model_cov=None):
    ax.scatter(y[:, 0], y[:, 1], s=2.5, color=INK2, alpha=0.55, lw=0)
    if model_cov is None:
        for r in (1, 2):
            ax.add_patch(Circle((0, 0), r, fill=False, color=BLUE, lw=1.0))
    else:
        for r in (1, 2):
            ellipse(ax, r ** 2 * model_cov, color=BLUE, lw=1.0, ls='--')
    ellipse(ax, np.cov(y.T), color=RED, lw=1.4)
    s = np.var(y, axis=0, ddof=1)
    ax.text(0.03, 0.97, rf'$\hat\sigma^2 = {s[0]:.2f},\ {s[1]:.2f}$', transform=ax.transAxes, va='top', fontsize=6.8)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8)


fig = plt.figure(figsize=(TEXTWIDTH, 4.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.45, wspace=0.35)
res = {}
for j, m in enumerate(('G', 'rec')):
    res[m] = cv_directions(d['C'][m])
    ax = fig.add_subplot(gs[0, j])
    cloud(ax, res[m][2][:, :2], f'({"ab"[j]}) ' + ('Gaussian' if m == 'G' else 'recommended'))
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$', labelpad=0)

# physical directions: amplitude of P0 and of P2, generalised-least-squares projections (unit variance under C)
C = d['C']['rec']
Ci = np.linalg.inv(C)
mu = V.mean(0)
T = np.zeros((V.shape[1], 2))
T[:nb, 0], T[nb:2 * nb, 1] = mu[:nb], mu[nb:2 * nb]
F = T.T @ Ci @ T
P = np.linalg.solve(F, T.T @ Ci)                   # amplitude estimators, covariance F^{-1} under C
amp = (P @ (V - mu).T).T / np.sqrt(np.diag(np.linalg.inv(F)))
mc = np.linalg.inv(F) / np.sqrt(np.outer(np.diag(np.linalg.inv(F)), np.diag(np.linalg.inv(F))))
ax = fig.add_subplot(gs[0, 2])
cloud(ax, amp, '(c) $P_0$, $P_2$ amplitudes', model_cov=mc)
ax.set_xlabel(r'$\delta A_0 / \sigma$')
ax.set_ylabel(r'$\delta A_2 / \sigma$', labelpad=0)

ax = fig.add_subplot(gs[1, 0:2])
nd = 20
for m, off in (('G', -0.15), ('rec', 0.15)):
    s = np.var(res[m][2][:, :nd], axis=0, ddof=1)
    ax.errorbar(np.arange(1, nd + 1) + off, s, s * np.sqrt(2 / (NB - 1)), fmt='o', ms=3, lw=0.9,
                color=MODEL_COLOR[m], label=MODEL_LABEL[m] + ' (out of sample)')
    ax.plot(np.arange(1, nd + 1), res[m][0][:nd], color=MODEL_COLOR[m], lw=0.8, ls=':',
            label=None if m == 'G' else 'in-sample eigenvalue (biased)')
ax.axhline(1, color=INK2, lw=0.7)
ax.set_xlabel('direction (rank in half A)')
ax.set_ylabel(r'variance in half B')
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_title('(d) cross-validated variance along the leading directions', fontsize=8)
ax.legend(fontsize=6.5, loc='upper right')

ax = fig.add_subplot(gs[1, 2])
Wh = np.linalg.inv(res['G'][3])
C = d['C']['G']
sg = np.sqrt(np.diag(C))
for i, ls in zip(range(2), ('-', '--')):
    pat = Wh @ res['G'][1][:, i] / sg
    pat *= np.sign(pat[np.argmax(np.abs(pat))])
    for e, c in ELL_COLOR.items():
        sl = slice((e // 2) * nb, (e // 2 + 1) * nb)
        ax.plot(k, pat[sl], color=c, ls=ls, lw=1.0, label=(rf'$\ell={e}$' if i == 0 else None))
ax.axhline(0, color=BASE, lw=0.6)
ax.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$')
ax.set_title('(e) directions of (a)', fontsize=8)
ax.legend(fontsize=6.3, loc='lower right', ncol=1, handlelength=1.2)
save(fig, 'clouds')
