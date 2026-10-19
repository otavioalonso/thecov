"""Sharp sphere: per sampled triangle (q1 in i, q2 in j), the exact class-1 (+ class-2) variance sum over (p1 in i, p2 in j)
with the exact lattice G (FFT grids) against the local-approximation formula sum W~(q1-p1) P(p1) W~(q2-p2) P(p2) W~(P-Q) P(P)
that thecov evaluates in configuration space.  usage: sphere_exact_vs_local.py R L N n_samples "i,j;..." [P_at: primed|mid]"""
import sys, time
sys.path.insert(0, 'desi_validation/bispectrum_dev')
import numpy as np, scipy.fft as sfft
from grf_sphere import Mesh, P_model, K_EDGES
W = 3
R, L, N, n_samp = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
pairs = [tuple(int(v) for v in p.split(',')) for p in sys.argv[5].split(';')]
kind = sys.argv[6] if len(sys.argv) > 6 else 'sphere'
m = Mesh(L=L, N=N)
mask = (m.r < R).astype(float) if kind == 'sphere' else np.where(m.r < R, np.exp(-m.r ** 2 / (2 * (R / 3.0) ** 2)), 0.0)
shape = m.k.shape; nb = len(K_EDGES) - 1
mt = sfft.fftn(mask, workers=W) * m.dV
P = P_model(np.where(m.k > 0, m.k, 1e-10)) / 1e4; P[m.k == 0] = 0.0
mt_neg = np.conj(mt); FTm = sfft.fftn(mt_neg, workers=W); ijk = np.indices(shape); negidx = tuple((-ijk) % N)
Wt = sfft.ifftn(sfft.fftn(mt, workers=W) * sfft.fftn(mt, workers=W), workers=W)   # W~(s) = sum_u m~(u) m~(s - u)
shells = [s.astype(float) for s in m.shells]
idx = {i: np.flatnonzero(s.ravel()) for i, s in enumerate(m.shells)}


def G_of(a):
    h = P * np.roll(mt_neg, shift=tuple(a % N), axis=(0, 1, 2)); c = sfft.ifftn(sfft.fftn(h, workers=W) * FTm, workers=W)
    return c[negidx] / L ** 3                           # G_a(b)


def conv(f, g):
    return sfft.ifftn(sfft.fftn(f, workers=W) * sfft.fftn(g, workers=W), workers=W)


def flip(a):
    return a[negidx]


def roll_to(arr, a):
    """arr(b - a) as an array over b: W~(q - p) over p is roll of W~(-.) ... W~(a - b) = flip(W~)(b - a) -> np.roll(flip(W~), a)"""
    return np.roll(arr, shift=tuple(a % N), axis=(0, 1, 2))


rng = np.random.default_rng(11)
t0 = time.time()
for (i, j) in pairs:
    ex1 = ex2 = lo1 = lo2 = 0.0; ratios = []; v1 = []; v2 = []
    for n in range(n_samp):
        q1 = np.array(np.unravel_index(rng.choice(idx[i]), shape)); q2 = np.array(np.unravel_index(rng.choice(idx[j]), shape)); Q = (q1 + q2) % N
        G1, G2, G3 = G_of(q1), G_of(q2), G_of((-Q) % N)
        f1 = flip(G1) * shells[i]; g2 = flip(G2) * shells[j]; g3f = flip(G3) * shells[j]
        e1 = np.sum(G3 * conv(f1, g2)).real                      # class 1: G(q1,-p1) G(q2,-p2) G(-Q, P)
        e2 = np.sum(G2 * conv(f1, g3f)).real                     # class 2: G(q1,-p1) G(-Q,-p2) G(q2, P)
        # local approximation: W~(q1 - p1) P(p1) etc.  (P at the primed momenta; the anchor pairing W~(P - Q) P(P))
        Wq1 = roll_to(flip(Wt), q1)                              # W~(q1 - p1) over p1
        Wq2 = roll_to(flip(Wt), q2)
        WQ = roll_to(flip(Wt), (-Q) % N)                         # W~(-Q - b) over b; anchor pairing needs W~(P - Q) = W~(-Q - (-P)) -> at b = -P
        h1 = Wq1 * P * shells[i]; h2 = Wq2 * P * shells[j]
        WQP = flip(WQ) * P                                        # as a function of P: W~(P - Q) P(P)
        l1 = np.sum(WQP * conv(h1, h2)).real / L ** 9
        # class 2 local: W~(q1 - p1) P(p1) W~(-Q - p2) P(p2) W~(q2 + P) P(P): sum over p2 in j with P = p1 + p2
        h3 = flip(WQ) * P * shells[j]                             # over p2: W~(-Q - p2)?  WQ(b) = W~(-Q - b) -> at b = p2 : flip? use WQ directly
        h3 = WQ * P * shells[j]
        Wq2P = roll_to(Wt, (-q2) % N)                             # W~(q2 + P) over P: Wt(P + q2) = roll(Wt, -q2)
        l2 = np.sum(Wq2P * P * conv(h1, h3)).real / L ** 9
        ex1 += e1; ex2 += e2; lo1 += l1; lo2 += l2; ratios.append(l1 / e1); v1.append(e1); v2.append(e2)
        if n % 50 == 49:
            print(f'  ({i},{j}) {n + 1}: local/exact class1 {lo1 / ex1:.4f} class2 {lo2 / ex2:.4f}; per-sample ratio rms {np.std(ratios):.3f}  [{time.time() - t0:.0f}s]', flush=True)
    I3 = np.sum(mask ** 3) * m.dV; nm = np.array([len(idx[k]) for k in range(nb)])
    norm = 1e12 / (I3 ** 2 * nm[i] * nm[j] * nm[i] * nm[j]) * nm[i] * nm[j]      # MC mean -> sum over (q1, q2)
    v1 = np.array(v1) * norm; v2 = np.array(v2) * norm
    # class 2 variance element needs the two class-2 pairings; the second is the (j, i) relabelling: not sampled here -> report the first
    print(f'({i},{j}) FINAL local/exact class1 {lo1 / ex1:.4f}, class2 {lo2 / ex2:.4f}, class2/class1 (exact) {ex2 / ex1:.3f}; '
          f'exact class1 {v1.mean():.6e} +- {v1.std() / np.sqrt(len(v1)) / v1.mean():.3f}, class2 (first pairing) {v2.mean():.6e} +- {v2.std() / np.sqrt(len(v2)) / v2.mean():.3f}', flush=True)
