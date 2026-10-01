import numpy as np, json, os
RD = os.environ.get('REPORT_DATA', os.path.expanduser('~/thecov_desi'))   # directory with the unpacked report_data.tgz
FILES = {'holi': 'holi_v3_mock173/report_data_holi-altmtl',
         'complete': 'abacus-2ndgen-dr2-complete_mock0/report_data_abacus-complete',
         'altmtl': 'abacus-2ndgen-dr2-altmtl_mock0/report_data_abacus-altmtl'}
_cache = {}
def run(name):
    if name not in _cache:
        z = np.load(os.path.join(RD, FILES[name] + '.npz'))
        m = json.load(open(os.path.join(RD, FILES[name] + '.json')))
        _cache[name] = (z, m[0] if isinstance(m, list) else m)
    return _cache[name]
def primary(region):
    return 'combined-regions' if region == 'GCcomb' else 'random-density'
def get(name, b, r, f=1, mode=None):
    z, _ = run(name)
    p = f'{b}/{r}/x{f}'
    mode = mode or primary(r)
    if mode == 'nx' and r == 'GCcomb': mode = 'combined-regions [nx]'
    return dict(V=z[p + '/V'].astype(float), C=z[p + f'/C/{mode}'], k=z[p + '/k'], edges=z[p + '/k_edges'],
                norm=z[p + '/norm'], sn=z[p + '/num_shotnoise'], ids=z[p + '/mock_ids'], ells=(0, 2, 4))
def chi2_i(V, C, idx=None):
    idx = np.arange(V.shape[1]) if idx is None else idx
    L = np.linalg.cholesky(C[np.ix_(idx, idx)])
    zz = np.linalg.solve(L, (V[:, idx] - V[:, idx].mean(0)).T)
    return (zz ** 2).sum(0) / (len(idx) * (1 - 1 / len(V)))
def kidx(k, kmax, nl=3, kmin=0.0):
    s = np.flatnonzero((k <= kmax + 1e-9) & (k >= kmin))
    nb = len(k)
    return np.concatenate([s + j * nb for j in range(nl)])
def mp_edges(n, N):
    q = n / (N - 1)
    return (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
