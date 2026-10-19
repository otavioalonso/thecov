"""Exact Gaussian covariance of the windowed B_000 estimator on the GRF mesh by mode coupling, for windows whose Fourier
transform m~(u) is compact (Gaussian window: |u| <= u_max k_f).  K_s(a) = G(a, s - a) = (nbar^2/L^3) sum_u m~(u) P(a - u)
m~(s - u) is evaluated per sampled mode over the compact u support, for all s in a ball; the class-1, class-2 and k3
pairings are then exact sums over (s1, s2) and Monte-Carlo sums over (q1, q2).  See exact_mesh_gaussian.py for the
formulae.  usage: exact_mesh_gaussian2.py kind R L N n_samples u_max s_max out.npz
"""
import sys, time
from pathlib import Path
import numpy as np
import scipy.fft as sfft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grf_sphere import Mesh, P_model, K_EDGES  # noqa: E402


def main(kind, R, L, N, n_samp, u_max, s_max, out, seed=0, batch=400):
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
    mt = sfft.fftn(mask, workers=1) * m.dV
    P_SCALE = 1e4
    P = P_model(np.where(m.k > 0, m.k, 1e-10)) / P_SCALE; P[m.k == 0] = 0.0
    shape = m.k.shape
    kf = 2 * np.pi / L
    grid = lambda r: np.array([(a, b, c) for a in range(-r, r + 1) for b in range(-r, r + 1) for c in range(-r, r + 1)
                               if a * a + b * b + c * c <= r * r])
    U = grid(u_max); S = grid(s_max)                                   # (nu, 3), (ns, 3) integer lattice vectors
    mt_u = mt.ravel()[np.ravel_multi_index(tuple((U.T) % N), shape)]  # m~(u)
    # m~(s - u) table (ns, nu)
    SU = (S[:, None, :] - U[None, :, :]) % N
    mt_su = mt.ravel()[np.ravel_multi_index(tuple(SU.transpose(2, 0, 1)), shape)]
    s_index = {tuple(s): n for n, s in enumerate(S)}
    print(f'nu={len(U)} ns={len(S)}; tail of m~ at |u| = u_max: {np.abs(mt_u[np.sum(U**2,1)==u_max**2]).max()/np.abs(mt_u).max():.2e}', flush=True)

    def ijk(flat):
        return np.array(np.unravel_index(flat, shape))

    def flat_of(ijk_arr):
        return np.ravel_multi_index(tuple(ijk_arr % N), shape)

    def kmag(ijk_arr):
        kk = ((ijk_arr + N // 2) % N - N // 2) * kf
        return np.sqrt((kk ** 2).sum(0))

    def shell_of(kk):
        b = np.digitize(kk, K_EDGES) - 1
        return np.where((b >= 0) & (b < nb), b, -1)

    def K_all_s(a_ijk):
        """K_s(a) for all s in S, for a batch of modes a (3, nbatch): (nbatch, ns)."""
        au = a_ijk[:, :, None] - U.T[:, None, :]                            # (3, nbatch, nu)
        Pa = P.ravel()[flat_of(au)]                                          # P(a - u)  (nbatch, nu)
        return (Pa * mt_u[None, :]) @ mt_su.T / L ** 3                        # (nbatch, ns)

    idx = {i: np.flatnonzero(s.ravel()) for i, s in enumerate(m.shells)}
    nmod = np.array([len(idx[i]) for i in range(nb)])
    # pair table for the sums over (s1, s2): index of -(s1 + s2) in S (or -1 if outside the ball)
    neg_sum = np.full((len(S), len(S)), -1, dtype=int)
    for a, s1 in enumerate(S):
        for b, s2 in enumerate(S):
            neg_sum[a, b] = s_index.get(tuple(-(s1 + s2)), -1)
    valid = neg_sum >= 0
    ia, ib = np.nonzero(valid); ic = neg_sum[ia, ib]
    print(f'{len(ia)} (s1, s2) pairs inside the ball', flush=True)
    rng = np.random.default_rng(seed)
    C1 = np.zeros((nb, nb, nb, nb)); C2 = np.zeros((nb, nb, nb, nb))
    Cq1 = np.zeros((nb, nb, 10)); Cq2 = np.zeros((nb, nb, 10))
    for i in range(nb):
        for j in range(nb):
            acc1 = np.zeros((nb, nb)); acc2 = np.zeros((nb, nb))
            QB = np.arange(0, 0.4001, 0.04); nqb = len(QB) - 1
            accq1 = np.zeros(nqb); accq2 = np.zeros(nqb)
            for b0 in range(0, n_samp, batch):
                nbt = min(batch, n_samp - b0)
                q1 = ijk(rng.choice(idx[i], nbt)); q2 = ijk(rng.choice(idx[j], nbt))
                Q = q1 + q2
                K1 = K_all_s(q1); K2 = K_all_s(q2); K3 = K_all_s(-Q)        # (nbt, ns)
                # shells of p1 = q1 - s1 and p2 = q2 - s2 (class 1), p2 = -Q + s1 + s2 (class 2)
                b1 = shell_of(kmag(q1[:, :, None] - S.T[:, None, :]))        # (nbt, ns)
                b2 = shell_of(kmag(q2[:, :, None] - S.T[:, None, :]))
                b2c = shell_of(kmag(-Q[:, :, None] - S.T[:, None, :]))       # |-Q - s3| with s3 = -(s1+s2): = |-Q + s1 + s2|
                qbin = np.digitize(kmag(Q), QB) - 1                                            # (nbt,)
                kP1 = kmag(Q[:, :, None] + S.T[:, None, :])                                   # |P| = |Q - s3| for class 1 (index by C_)
                pbin1 = np.digitize(kP1, QB) - 1
                kP2 = kmag(-q2[:, :, None] + S.T[:, None, :])                                 # |P| = |-q2 + s2| for class 2 (index by B_)
                pbin2 = np.digitize(kP2, QB) - 1
                for n0 in range(0, len(ia), 20000):
                    sl = slice(n0, n0 + 20000)
                    A_, B_, C_ = ia[sl], ib[sl], ic[sl]
                    val = (K1[:, A_] * K2[:, B_] * K3[:, C_]).real          # (nbt, npairs)
                    bb1 = b1[:, A_]; bb2 = b2[:, B_]; bb2c = b2c[:, C_]
                    ok = (bb1 >= 0) & (bb2 >= 0)
                    np.add.at(acc1, (bb1[ok], bb2[ok]), val[ok])
                    okv = ok & (bb1 == i) & (bb2 == j) & (pbin1[:, C_] == qbin[:, None]) & (qbin[:, None] >= 0) & (qbin[:, None] < nqb)
                    np.add.at(accq1, np.broadcast_to(qbin[:, None], val.shape)[okv], val[okv])
                    ok = (bb1 >= 0) & (bb2c >= 0)
                    np.add.at(acc2, (bb1[ok], bb2c[ok]), val[ok])
                    okv = ok & (bb1 == i) & (bb2c == j) & (pbin2[:, B_] == qbin[:, None]) & (qbin[:, None] >= 0) & (qbin[:, None] < nqb)
                    np.add.at(accq2, np.broadcast_to(qbin[:, None], val.shape)[okv], val[okv])
            norm = nmod[i] * nmod[j] / n_samp * P_SCALE ** 3
            Cq1[i, j] = accq1 * norm / (I3 ** 2 * nmod[i] ** 2 * nmod[j] ** 2)
            Cq2[i, j] = accq2 * norm / (I3 ** 2 * nmod[i] ** 2 * nmod[j] ** 2)
            for ip in range(nb):
                for jp in range(nb):
                    d = I3 ** 2 * nmod[i] * nmod[j] * nmod[ip] * nmod[jp]
                    C1[i, j, ip, jp] = acc1[ip, jp] * norm / d
                    C2[i, j, ip, jp] = acc2[ip, jp] * norm / d
            print(f'  ({i},{j}) done {time.time() - t0:.0f}s', flush=True)
    C1_tot = C1 + C1.transpose(0, 1, 3, 2)
    C2_tot = C2 + C2.transpose(1, 0, 2, 3) + C2.transpose(0, 1, 3, 2) + C2.transpose(1, 0, 3, 2)
    # k3 pairing: A_ij(s) = sum_{q1 in i, |s - q1| in j} K_s(q1); anchor G(-s, s') with the mean subtraction
    A = np.zeros((nb, nb, len(S)), dtype=complex)
    for i in range(nb):
        for b0 in range(0, nmod[i], batch):
            q1 = ijk(idx[i][b0:b0 + batch])
            Kq = K_all_s(q1)                                                   # (nbt, ns)
            bj = shell_of(kmag(S.T[:, None, :] - q1[:, :, None]))             # |s - q1|  (nbt, ns)
            for j in range(nb):
                A[i, j] += np.sum(np.where(bj == j, Kq, 0.0), axis=0)
    zero = np.zeros((3, 1), dtype=int)
    Kzero = K_all_s(zero)[0]                                                   # K_s(0) = G0(0, s) = C0(s)
    m0 = mt.ravel()[0]
    mt_s = mt.ravel()[flat_of(S.T)]                                            # m~(s)
    mt_negs = mt.ravel()[flat_of(-S.T)]
    K_negs = K_all_s(-S.T)                                                     # (ns, ns): K_{s'}(-s) over s' index
    Gk = np.zeros((len(S), len(S)), dtype=complex)
    for n, s in enumerate(S):
        for n2, s2 in enumerate(S):
            d = s_index.get(tuple(s2 - s))
            if d is None:
                continue
            G0 = K_negs[n, d]                                                  # G0(-s, s') = K_{s'-s}(-s)
            Gk[n, n2] = G0 - mt_negs[n] / m0 * Kzero[n2] - mt_s[n2] / m0 * Kzero[s_index[tuple(-s)]] \
                + mt_negs[n] * mt_s[n2] / m0 ** 2 * Kzero[s_index[(0, 0, 0)]]
    Ck3 = np.zeros((nb, nb, nb, nb))
    for i in range(nb):
        for j in range(nb):
            for ip in range(nb):
                for jp in range(nb):
                    Ck3[i, j, ip, jp] = (A[i, j] @ Gk @ np.conj(A[ip, jp])).real * P_SCALE ** 3 / (I3 ** 2 * nmod[i] * nmod[j] * nmod[ip] * nmod[jp])
    np.savez(out, C_class1=C1_tot, C_class2=C2_tot, C_k3=Ck3, k_edges=K_EDGES, R=R, L=L, N=N, nmodes=nmod, n_samp=n_samp,
             u_max=u_max, s_max=s_max, Cq1=Cq1, Cq2=Cq2, Q_bins=np.arange(0, 0.4001, 0.04))
    print(f'saved {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], float(a[1]), float(a[2]), int(a[3]), int(a[4]), int(a[5]), int(a[6]), a[7])
