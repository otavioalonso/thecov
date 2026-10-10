"""Shared style, data access and statistics for the paper figures and numbers.

Every figure script imports this module, reads only paper/products/ (see make_products.py) and saves through `save`,
which stamps the PDF with the script name, the git commit and the sha256 of the products it read.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROD = os.environ.get('PAPER_PRODUCTS', os.path.join(ROOT, 'products'))
FIGS = os.path.join(ROOT, 'figures')

# ------------------------------------------------------------------ style (reference palette, fixed slot order)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
                                                          '#008300', '#4a3aa7', '#e34948')
INK, INK2, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
ELL_COLOR = {0: BLUE, 2: ORANGE, 4: VIOLET}
# one colour per covariance model, used identically in every figure
MODEL_COLOR = {'G_local': MUTED, 'G': BLUE, 'G+SSC': ORANGE, 'rec': AQUA, 'rec+T0tree': RED, 'rec+T0resp': VIOLET}
MODEL_LABEL = {'G_local': r'Gaussian, local window $m^2$', 'G': r'Gaussian, $\xi$-kernel window',
               'G+SSC': 'Gaussian + SSC', 'rec': 'Gaussian + SSC + discreteness',
               'rec+T0tree': r'$\ldots$ + tree-level $T_0$', 'rec+T0resp': r'$\ldots$ + response $T_0$'}
DIVERGING = LinearSegmentedColormap.from_list('div', ['#184f95', '#6da7ec', '#f0efec', '#ec835a', '#b8322f'])
SEQUENTIAL = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#9ec5f4', '#3987e5', '#184f95', '#0d366b'])

TEXTWIDTH = 6.3   # inches, JCAP text block

plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.edgecolor': BASE, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.axisbelow': True, 'lines.linewidth': 1.4, 'lines.markersize': 3.5,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'figure.dpi': 150,
    'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
})

_read = {}


def load(name):
    """products/<name>.npz (cached); records its sha256 for the provenance stamp"""
    if name not in _read:
        fn = os.path.join(PROD, name + '.npz')
        if not os.path.exists(fn):
            sys.exit(f'{fn} missing: run make_products.py (see paper/README.md)')
        with open(fn, 'rb') as fh:
            _read[name] = (np.load(fn, allow_pickle=False), hashlib.sha256(fh.read()).hexdigest()[:16])
    return _read[name][0]


def git_commit():
    try:
        return subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                              text=True).stdout.strip() or 'unknown'
    except OSError:
        return 'unknown'


def in_notebook():
    try:
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'   # noqa: F821
    except NameError:
        return False


def save(fig, name):
    """figures/<name>.pdf with provenance metadata; in a notebook the figure is also shown inline"""
    os.makedirs(FIGS, exist_ok=True)
    script = 'paper.ipynb' if in_notebook() else os.path.basename(sys.argv[0])
    meta = {'Title': name, 'Creator': f'paper/scripts/{script} @ {git_commit()}',
            'Subject': 'products: ' + ', '.join(f'{k}.npz:{h}' for k, (_, h) in sorted(_read.items()))}
    fn = os.path.join(FIGS, name + '.pdf')
    fig.savefig(fn, metadata=meta)
    if in_notebook():
        import io
        from IPython.display import Image, display
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        display(Image(data=buf.getvalue()))
    plt.close(fig)
    print(f'wrote {fn}')


# ------------------------------------------------------------------ data access
def excluded(b):
    """mock ids excluded from the holi statistics (defective n(z); products/excluded_mocks.json from make_products.py)"""
    fn = os.path.join(PROD, 'excluded_mocks.json')
    if os.environ.get('PAPER_KEEP_ALL_MOCKS') or not os.path.exists(fn):
        return []
    return json.load(open(fn)).get(b, {}).get('mocks', [])


def holi(b, cap, f=1):
    """mock vectors, k grid and every covariance model for holi tracer b, cap, binning factor f (defective mocks removed;
    PAPER_KEEP_ALL_MOCKS=1 keeps them)"""
    z = load(f'holi_{b}')
    p = f'{cap}/x{f}/'
    keep = ~np.isin(z[p + 'mock_ids'], excluded(b))
    d = dict(V=z[p + 'V'][keep], k=z[p + 'k'], edges=z[p + 'k_edges'], nmodes=z[p + 'nmodes'], norm=z[p + 'norm'][keep],
             mock_ids=z[p + 'mock_ids'][keep])
    d['C'] = models(z, p)
    for key in ('mock_mean', 'params', 'template_fits', 'plin_k', 'plin_P_damped_normalised', 'damping'):
        if p + key in z.files:
            d[key] = z[p + key]
    d['terms'] = {key[2:]: z[p + key] for key in ('C_ssc', 'C_ssc_noLA', 'C_disc', 'C_disc_local', 'C_disc_B',
                                                    'C_disc_P', 'C_T0_tree', 'C_T0_response') if p + key in z.files}
    return d


def models(z, p):
    C = {'G': z[p + 'C_G']}
    if p + 'C_G_local' in z.files:
        C['G_local'] = z[p + 'C_G_local']
    if p + 'C_ssc' in z.files:
        C['G+SSC'] = C['G'] + z[p + 'C_ssc']
        C['rec'] = C['G+SSC'] + z[p + 'C_disc']
        if p + 'C_T0_tree' in z.files:
            C['rec+T0tree'] = C['rec'] + z[p + 'C_T0_tree']
        if p + 'C_T0_response' in z.files:
            C['rec+T0resp'] = C['rec'] + z[p + 'C_T0_response']
    return C


def rebin_matrix(nmodes, f, nl=3):
    """R such that R @ P_x1 is the mode-weighted average over groups of f bins (per multipole)"""
    nb = len(nmodes)
    R1 = np.zeros((nb // f, nb))
    for i in range(nb // f):
        w = nmodes[i * f:(i + 1) * f]
        R1[i, i * f:(i + 1) * f] = w / w.sum()
    return np.kron(np.eye(nl), R1)


def kidx(k, kmax=0.3, kmin=0.0, ells=(0, 2, 4)):
    nb = len(k)
    s = np.flatnonzero((k <= kmax + 1e-9) & (k >= kmin))
    return np.concatenate([s + (e // 2) * nb for e in ells])


# ------------------------------------------------------------------ statistics
def chi2(V, C, idx=None):
    """chi^2 of every mock about the mock mean; its expectation is n (N - 1) / N"""
    idx = np.arange(V.shape[1]) if idx is None else idx
    L = np.linalg.cholesky(C[np.ix_(idx, idx)])
    x = np.linalg.solve(L, (V[:, idx] - V[:, idx].mean(0)).T)
    return (x ** 2).sum(0)


def whitened_cov(V, C, idx=None):
    """C^{-1/2} S C^{-1/2} with the symmetric square root"""
    idx = np.arange(V.shape[1]) if idx is None else idx
    S = np.cov(V[:, idx].T)
    w, U = np.linalg.eigh(C[np.ix_(idx, idx)])
    Wi = (U / np.sqrt(w)) @ U.T
    return Wi @ S @ Wi, Wi


def mp_edges(n, N):
    q = n / (N - 1)
    return (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2


def wishart_sigma(C, N):
    """standard deviation of the sample covariance elements for N Gaussian vectors with covariance C"""
    d = np.diag(C)
    return np.sqrt((np.outer(d, d) + C ** 2) / (N - 1))


def corr(C):
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def ell_axes_label(ax, k, nl=3):
    """x axis of a concatenated (l=0, 2, 4) vector: tick at the start of each block"""
    nb = len(k)
    for j in range(1, nl):
        ax.axvline(j * nb - 0.5, color=BASE, lw=0.6)


def write_numbers(prefix, values):
    """append \\newcommand macros to products/numbers_<prefix>.json (collected into tex/numbers.tex by numbers.py)"""
    fn = os.path.join(PROD, f'numbers_{prefix}.json')
    json.dump(values, open(fn, 'w'), indent=1)


def fit_templates(V, CG, templates, start=None):
    """Wishart maximum-likelihood amplitudes theta of C = CG + sum_a theta_a T_a for the mocks V, with Fisher errors
    F_ab = (N - 1) / 2 tr(C^-1 T_a C^-1 T_b) at the best fit. Returns (theta, sigma, -2 Delta lnL vs theta = 0)."""
    from scipy.optimize import minimize
    S, N = np.cov(V.T), len(V)
    T = list(templates.values())

    def m2l(th):
        C = CG + sum(t * Ta for t, Ta in zip(th, T))
        try:
            L = np.linalg.cholesky(C)          # also rejects matrices with an even number of negative eigenvalues
        except np.linalg.LinAlgError:
            return 1e300
        X = np.linalg.solve(L, S)
        return (N - 1) * (np.trace(np.linalg.solve(L, X.T)) + 2 * np.log(np.diag(L)).sum())
    x0 = np.ones(len(T)) if start is None else np.asarray(start, float)
    if m2l(x0) >= 1e299:
        x0 = 0.1 * np.ones(len(T))
    res = minimize(m2l, x0, method='Powell', options=dict(xtol=1e-4, ftol=1e-6, maxiter=20000))
    th = res.x
    C = CG + sum(t * Ta for t, Ta in zip(th, T))
    Ci = np.linalg.inv(C)
    A = [Ci @ Ta for Ta in T]
    F = np.array([[0.5 * (N - 1) * np.trace(a @ b) for b in A] for a in A])
    sig = np.sqrt(np.diag(np.linalg.inv(F)))
    return dict(zip(templates, th)), dict(zip(templates, sig)), m2l(np.zeros(len(T))) - res.fun
