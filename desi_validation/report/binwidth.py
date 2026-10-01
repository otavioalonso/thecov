from data import *
F = (1, 2, 4)
def ratios(name, b, r, mode=None, boot=None):
    """mean var ratio per (ell, coarse x4 bin) for each factor; returns array (3 factors, 3 ells, nJ)"""
    out = []
    for f in F:
        d = get(name, b, r, f, mode)
        V = d['V'] if boot is None else d['V'][boot]
        rr = V.var(0, ddof=1) / np.diag(d['C'])
        nb = len(d['k']); per = 4 // f
        out.append(rr.reshape(3, nb // per, per).mean(-1))
    return np.array(out)
def decompose(R):
    """least squares R-1 = a + b f per (ell, J)"""
    A = np.vstack([np.ones(3), np.array(F, float)]).T
    coef, *_ = np.linalg.lstsq(A, (R - 1).reshape(3, -1), rcond=None)
    return coef.reshape(2, *R.shape[1:])
def run_decomp(name, b, regions=('NGC', 'SGC'), nboot=200, seed=1):
    rng = np.random.default_rng(seed)
    N = len(get(name, b, regions[0])['V'])
    R = np.mean([ratios(name, b, r) for r in regions], 0)
    ab = decompose(R)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, N, N)
        bs.append(decompose(np.mean([ratios(name, b, r, boot=i) for r in regions], 0)))
    bs = np.array(bs)
    Rb = []
    return R, ab, bs.std(0), get(name, b, regions[0], 4)['k']
