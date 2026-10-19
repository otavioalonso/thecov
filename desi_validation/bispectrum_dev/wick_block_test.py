"""Brute-force Wick test: variance of the windowed mini-bispectrum sum_{a in A, b in B} F(a) F(b) F(-a-b) over two mode blocks
against the full Wick sum with the exact lattice two-point function G(a, b).  usage: wick_block_test.py N n_real half_width"""
import sys, time, itertools
sys.path.insert(0, 'desi_validation/bispectrum_dev')
import numpy as np, scipy.fft as sfft
from multiprocessing import Pool
from grf_sphere import Mesh, P_model
L = 1300.0; N = int(sys.argv[1]); n_real = int(sys.argv[2]); hw = int(sys.argv[3])
m = Mesh(L=L, N=N)
mask = np.where(m.r < 600, np.exp(-m.r ** 2 / (2 * 200.0 ** 2)), 0.0)
shape = m.k.shape; kf = 2 * np.pi / L
c0 = np.array([12, 3, 0]); c2 = np.array([20, -20, 8]) if len(sys.argv) < 5 else np.array([-6, 18, 6])
off = np.array([(a, b, c) for a in range(-hw, hw + 1) for b in range(-hw, hw + 1) for c in range(-hw, hw + 1)])
A = c0[None, :] + off; B = c2[None, :] + off
if len(sys.argv) >= 5:           # class-2 geometry: B also contains the block around -(c0 + c2), so third sides fall in B
    B = np.concatenate([B, -(c0 + c2)[None, :] + off])
nA = len(A); nB = len(B)
tri = np.array([(ia, ib) for ia in range(nA) for ib in range(nB)])
negQ = -(A[tri[:, 0]] + B[tri[:, 1]])
negQ_u, inv = np.unique(negQ, axis=0, return_inverse=True)
modes = np.concatenate([A, B, negQ_u]); nm = len(modes)
flat = np.ravel_multi_index(tuple(modes.T % N), shape)


def one(seed):
    d = m.grf(seed); d = d - np.sum(mask * d) / np.sum(mask)
    return (sfft.fftn(mask * d, workers=1) * m.dV).ravel()[flat]


if __name__ == '__main__':
    t0 = time.time()
    with Pool(3) as pool:
        F = np.array(pool.map(one, range(n_real), chunksize=20))
    print(f'{n_real} realisations {time.time() - t0:.0f}s', flush=True)
    FA, FB, FQ = F[:, :nA], F[:, nA:nA + nB], F[:, nA + nB:]
    Bmini = np.sum(FA[:, tri[:, 0]] * FB[:, tri[:, 1]] * FQ[:, inv], axis=1)
    var_meas = Bmini.var(ddof=1); err = np.sqrt(2 / n_real)
    mt = sfft.fftn(mask, workers=3) * m.dV; P = P_model(np.where(m.k > 0, m.k, 1e-10)); P[m.k == 0] = 0
    mt_neg = np.conj(mt); FTm = sfft.fftn(mt_neg, workers=3); ijk = np.indices(shape)
    negidx = tuple((-ijk) % N)
    t0 = time.time()
    Gm = np.zeros((nm, nm), dtype=complex)
    fl_neg = np.ravel_multi_index(tuple((-modes).T % N), shape)
    for i, a in enumerate(modes):
        h = P * np.roll(mt_neg, shift=tuple(a % N), axis=(0, 1, 2))
        c = sfft.ifftn(sfft.fftn(h, workers=3) * FTm, workers=3)
        Gm[i] = (c[negidx] / L ** 3).ravel()[fl_neg]                        # G(a, -y)
    print(f'G matrix {nm} in {time.time() - t0:.0f}s', flush=True)
    I = [tri[:, 0], nA + tri[:, 1], nA + nB + inv]
    tot = 0.0; parts = {}
    nt = len(tri)
    for p in itertools.permutations([0, 1, 2]):
        tp = 0.0
        for c0_ in range(0, nt, 2000):
            sl = slice(c0_, c0_ + 2000)
            tp += np.sum(Gm[np.ix_(I[0][sl], I[p[0]])] * Gm[np.ix_(I[1][sl], I[p[1]])] * Gm[np.ix_(I[2][sl], I[p[2]])])
        parts[p] = tp.real; tot += tp
    wick = tot.real
    print('pairing shares:', {p: round(v / wick, 4) for p, v in parts.items()}, flush=True)
    print(f'blocks ({2*hw+1})^3: var measured / Wick(6 cross pairings) = {var_meas / wick:.4f} +- {err:.4f}', flush=True)
    diag = np.mean(np.sum(np.abs(FA[:, tri[:, 0]] * FB[:, tri[:, 1]] * FQ[:, inv]) ** 2, axis=1))
    tot_d = sum(np.sum(Gm[I[0], I[p[0]]] * Gm[I[1], I[p[1]]] * Gm[I[2], I[p[2]]]) for p in itertools.permutations([0, 1, 2]))
    print(f'diagonal share of Wick {tot_d.real / wick:.3f}; diagonal measured/Wick {diag / tot_d.real:.4f}')
