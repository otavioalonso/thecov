"""B' = F(q1*) sum_{q2 in shell j} F(q2) F(-q1*-q2) for a fixed leg mode q1* (|q1*| ~ 0.05, shell j = 2): its variance has
exactly one Wick pairing, var' = G(q1*, -q1*) sum_{q2, p2 in j} G(q2, -p2) G(-q1*-q2, q1*+p2), evaluated with no Monte Carlo
(explicit sum over q2 and the window-coupling offsets s = q2 - p2).  usage: single_mode_test.py N n_real"""
import sys, time
sys.path.insert(0, 'desi_validation/bispectrum_dev')
import numpy as np, scipy.fft as sfft
from multiprocessing import Pool
from grf_sphere import Mesh, P_model, K_EDGES
L = 1300.0; N = int(sys.argv[1]); n_real = int(sys.argv[2])
m = Mesh(L=L, N=N)
mask = np.where(m.r < 600, np.exp(-m.r ** 2 / (2 * 200.0 ** 2)), 0.0)
shape = m.k.shape; kf = 2 * np.pi / L
q1 = np.array([10, 3, 1])                      # |q1| = 10.5 kf = 0.051
J = 2
s2 = m.shells[J]
flat_q1 = np.ravel_multi_index(tuple(q1 % N), shape)


def one(seed):
    d = m.grf(seed); d = d - np.sum(mask * d) / np.sum(mask)
    F = mask * d
    Fk = sfft.fftn(F, workers=1) * m.dV
    Fj = sfft.ifftn(Fk * s2, workers=1) / m.dV                 # F_j(x) = (1/L^3) sum_{q in j} F(q) e^{iqx} * (N^3 dV / L^3 ...) -> see below
    # sum_{q2 in j} F(q2) F(-q1-q2) = dV sum_x F(x) e^{i q1 x} [sum_{q2 in j} F(q2) e^{i q2 x}] = dV sum_x F(x) e^{iq1x} Fj(x) * (L^3/dV... )
    # with Fj as defined: ifftn(Fk s2)/dV = (1/(N^3 dV)) sum_{q2 in j} F(q2) e^{iq2x} -> sum_{q2} F(q2) e^{iq2x} = N^3 dV Fj
    G = F * Fj * N ** 3 * m.dV
    Gk = sfft.fftn(G, workers=1) * m.dV                        # = dV sum_x G e^{-ikx}; we need e^{+iq1x}: take k = -q1
    val = Fk.ravel()[flat_q1] * Gk[tuple((-q1) % N)]
    return val


if __name__ == '__main__':
    t0 = time.time()
    import os
    if os.path.exists('/tmp/single_mode_B.npy') and len(np.load('/tmp/single_mode_B.npy')) == n_real:
        B = np.load('/tmp/single_mode_B.npy')
    else:
        with Pool(3) as pool:
            B = np.array(pool.map(one, range(n_real), chunksize=20))
    print(f'{n_real} realisations {time.time() - t0:.0f}s; <B>/sigma = {abs(B.mean()) / B.std() * np.sqrt(n_real):.2f}', flush=True)
    var_meas = np.mean(np.abs(B) ** 2)
    # exact: G(q1,-q1) and sum_{q2 in j} sum_s 1_j(q2 - s) K_s(q2) K_{-s}(-q1-q2)
    mt = sfft.fftn(mask, workers=3) * m.dV
    P_SCALE = 1e4
    P = P_model(np.where(m.k > 0, m.k, 1e-10)) / P_SCALE; P[m.k == 0] = 0
    grid = lambda r: np.array([(a, b, c) for a in range(-r, r + 1) for b in range(-r, r + 1) for c in range(-r, r + 1) if a * a + b * b + c * c <= r * r])
    U = grid(4); S = grid(6)
    mt_u = mt.ravel()[np.ravel_multi_index(tuple(U.T % N), shape)]
    SU = (S[:, None, :] - U[None, :, :]) % N
    mt_su = mt.ravel()[np.ravel_multi_index(tuple(SU.transpose(2, 0, 1)), shape)]
    neg_index = np.array([np.flatnonzero((S == -s).all(1))[0] for s in S])

    def flat_of(ijk):
        return np.ravel_multi_index(tuple(ijk % N), shape)

    def K_all_s(a_ijk):
        au = a_ijk[:, :, None] - U.T[:, None, :]
        Pa = P.ravel()[flat_of(au)]
        return (Pa * mt_u[None, :]) @ mt_su.T / L ** 3

    def kmag(ijk):
        kk = ((ijk + N // 2) % N - N // 2) * kf
        return np.sqrt((kk ** 2).sum(0))

    idx2 = np.flatnonzero(s2.ravel())
    tot = 0.0; tot2 = 0.0
    for b0 in range(0, len(idx2), 500):
        q2 = np.array(np.unravel_index(idx2[b0:b0 + 500], shape))             # (3, nb)
        K2 = K_all_s(q2)                                                        # K_s(q2)
        K3 = K_all_s(-(q1[:, None] + q2))                                       # K_s(-q1-q2); need K_{-s}
        in_j = (np.digitize(kmag(q2[:, :, None] - S.T[:, None, :]), K_EDGES) - 1) == J     # p2 = q2 - s in shell j
        tot += np.sum(K2 * K3[:, neg_index] * in_j)
        # class-2-type pairing (q2, P)(-Q, -p2): p2 = -Q + s in shell j, value K_s(q2) K_{-s}(-Q)
        in_j2 = (np.digitize(kmag(-(q1[:, None] + q2)[:, :, None] + S.T[:, None, :]), K_EDGES) - 1) == J
        tot2 += np.sum(K2 * K3[:, neg_index] * in_j2)
    G11 = K_all_s(q1[:, None])[0, np.flatnonzero((S == 0).all(1))[0]]           # G(q1, -q1) = K_0(q1)
    wick1 = (G11 * tot).real * P_SCALE ** 3; wick2 = (G11 * tot2).real * P_SCALE ** 3
    np.save('/tmp/single_mode_B.npy', B)
    print(f'single-mode estimator (|q1| = {np.linalg.norm(q1) * kf:.3f}, shell {J}): var measured / Wick(class1) = {var_meas / wick1:.4f}; class2/class1 = {wick2 / wick1:.4f}; var / (class1 + class2) = {var_meas / (wick1 + wick2):.4f} +- {np.sqrt(2 / n_real):.4f}')
