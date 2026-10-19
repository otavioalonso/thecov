"""Exact class-1 and class-2 Gaussian covariance of the windowed B_000 estimator on the mesh, with NO truncation of the
mode coupling: for each sampled (q1, q2) the sums over all (p1 in i', p2 in j') are done with FFTs,
    G_a(b) = <F(a) F(b)> = (1/L^3) sum_k P(k) m~(a - k) m~(b + k)        (one FFT pair per a),
    class 1: sum_{p1 p2} G_{q1}(-p1) G_{q2}(-p2) G_{-Q}(p1 + p2) = sum_P G_{-Q}(P) (f * g)(P),
    class 2: sum_{p1 p2} G_{q1}(-p1) G_{-Q}(-p2) G_{q2}(p1 + p2).
usage: exact_mesh_gaussian3.py kind R L N n_samples "i,j;i,j;..." out.npz
"""
import sys, time
from pathlib import Path
import numpy as np
import scipy.fft as sfft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grf_sphere import Mesh, P_model, K_EDGES  # noqa: E402
W = 3
S_BALL = 5


def main(kind, R, L, N, n_samp, pairs, out, seed=0):
    t0 = time.time()
    m = Mesh(L=L, N=N)
    nb = len(K_EDGES) - 1
    if kind == 'sphere':
        mask = (m.r < R).astype(float)
    elif kind == 'box':
        mask = np.ones_like(m.r)
    else:
        mask = np.where(m.r < R, np.exp(-m.r ** 2 / (2 * (R / 3.0) ** 2)), 0.0)
    I3 = np.sum(mask ** 3) * m.dV
    mt = sfft.fftn(mask, workers=W) * m.dV
    P = P_model(np.where(m.k > 0, m.k, 1e-10)) / 1e4; P[m.k == 0] = 0.0
    shape = m.k.shape
    idx = {i: np.flatnonzero(s.ravel()) for i, s in enumerate(m.shells)}
    nmod = np.array([len(idx[i]) for i in range(nb)])
    shells = [s.astype(float) for s in m.shells]
    mt_neg = np.conj(mt)                                   # m~(-u)
    FT_mt = sfft.fftn(mt, workers=W)

    def G_of(a_ijk):
        """G_a(b) on the whole grid for the lattice vector a: sum_k P(k) m~(a - k) m~(b + k) / L^3."""
        h = P * np.roll(mt_neg, shift=tuple(a_ijk), axis=(0, 1, 2))      # m~(a - k) = conj(m~)(k - a) -> roll by a
        # sum_k h(k) m~(b + k) = correlation: FFT^-1[ conj(FFT(h)) * FFT(m~) ] ... use the convolution with m~(-.)
        # sum_k h(k) mt(b + k) = sum_k h(k) mt_neg_rev... write as convolution of h with mt_neg: sum_k h(k) mt(-( -b - k))
        c = sfft.ifftn(sfft.fftn(h, workers=W) * sfft.fftn(mt_neg, workers=W), workers=W)   # sum_k h(k) mt_neg(b' - k) at b' = -b
        # c(b') = sum_k h(k) mt(k - b'), so G_a(b) = c(-b)
        ijk = np.indices(shape)
        return c[tuple((-ijk) % N)] / L ** 3

    def flip(arr):
        ijk = np.indices(shape)
        return arr[tuple((-ijk) % N)]

    def conv(f, g):
        return sfft.ifftn(sfft.fftn(f, workers=W) * sfft.fftn(g, workers=W), workers=W)

    rng = np.random.default_rng(seed)
    res = {}
    for (i, j) in pairs:
        acc1 = np.zeros((nb, nb)); acc2 = np.zeros((nb, nb)); acc1b = np.zeros((nb, nb)); acc2b = np.zeros((nb, nb))
        for n in range(n_samp):
            q1 = np.array(np.unravel_index(rng.choice(idx[i]), shape)); q2 = np.array(np.unravel_index(rng.choice(idx[j]), shape))
            Q = (q1 + q2) % N
            G1 = G_of(q1); G2 = G_of(q2); G3 = G_of((-Q) % N)
            # class 1: f(p1) = G_{q1}(-p1) 1[p1 in i'], g(p2) = G_{q2}(-p2) 1[p2 in j'], sum_P G_{-Q}(P) (f*g)(P)
            f1 = flip(G1); g2 = flip(G2); g3f = flip(G3)
            # s-ball truncated versions (|p1 - q1| <= s_ball k_f etc.) for a paired comparison with exact_mesh_gaussian2
            kf = 2 * np.pi / L
            def ball(center):
                ijk = np.indices(shape)
                d = ((ijk - np.asarray(center)[:, None, None, None] + N // 2) % N - N // 2)
                return (np.sum(d ** 2, axis=0) <= S_BALL ** 2).astype(float)
            b1 = ball(q1); b2 = ball(q2); b3 = ball((-Q) % N)
            for ip in range(nb):
                for jp in range(nb):
                    Fj = conv(f1 * shells[ip], g2 * shells[jp])
                    acc1[ip, jp] += np.sum(G3 * Fj).real
                    Fjb = conv(f1 * shells[ip] * b1, g2 * shells[jp] * b2)
                    acc1b[ip, jp] += np.sum(G3 * Fjb).real
                    # class 2: f(p1) = G_{q1}(-p1) 1[i'], g(p2) = G_{-Q}(-p2) 1[j'], h(P) = G_{q2}(P)
                    Fc = conv(f1 * shells[ip], g3f * shells[jp])
                    acc2[ip, jp] += np.sum(G2 * Fc).real
                    Fcb = conv(f1 * shells[ip] * b1, g3f * shells[jp] * b3)
                    acc2b[ip, jp] += np.sum(G2 * Fcb).real
            if n % 20 == 19:
                print(f'  ({i},{j}) {n + 1}/{n_samp} {time.time() - t0:.0f}s', flush=True)
        norm = nmod[i] * nmod[j] / n_samp * 1e12
        C1 = np.zeros((nb, nb)); C2 = np.zeros((nb, nb))
        for ip in range(nb):
            for jp in range(nb):
                d = I3 ** 2 * nmod[i] * nmod[j] * nmod[ip] * nmod[jp]
                C1[ip, jp] = acc1[ip, jp] * norm / d
                C2[ip, jp] = acc2[ip, jp] * norm / d
        res[(i, j)] = (C1, C2)
        print(f'  ({i},{j}) full / s-ball: class1 {acc1[i, j] / acc1b[i, j]:.4f}, class2 (i\'=i) {acc2[i].sum() / acc2b[i].sum():.4f}', flush=True)
        print(f'({i},{j}) done {time.time() - t0:.0f}s', flush=True)
    np.savez(out, **{f'C1_{i}{j}': v[0] for (i, j), v in res.items()}, **{f'C2_{i}{j}': v[1] for (i, j), v in res.items()},
             pairs=np.array(pairs), n_samp=n_samp)


if __name__ == '__main__':
    a = sys.argv[1:]
    pairs = [tuple(int(v) for v in p.split(',')) for p in a[5].split(';')]
    main(a[0], float(a[1]), float(a[2]), int(a[3]), int(a[4]), pairs, a[6])
