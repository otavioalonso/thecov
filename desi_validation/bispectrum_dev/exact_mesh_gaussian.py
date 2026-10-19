"""Exact Gaussian covariance of the windowed B_000 estimator on the GRF mesh, by mode coupling (no local approximation).

With F(a) = dV sum_x F(x) e^{-iax}, F = nbar m (delta - delta_bar), the two-point function on the lattice is
    G(a, b) = <F(a) F(b)> = (nbar^2 / L^3) sum_k P(k) m~(a - k) m~(b + k)   (+ mean-subtraction terms),
peaked at a + b ~ 0 within the window width.  B_000(i, j) = (1 / (I3 n_i n_j)) sum_{q1 in i, q2 in j} F(q1) F(q2) F(-q1-q2),
and the Gaussian covariance is the sum over Wick pairings of products of three G's.  K_s(a) = G(a, s - a) is one
circular convolution per small lattice vector s; the pairings are then Monte-Carlo sums over (q1, q2) with the small
offsets summed exactly.  Class 1 (2 pairings), class 2 (4) and the k3 ~ 0 pairing are evaluated; the other 8 vanish on
the shells.  usage: exact_mesh_gaussian.py kind(sphere|smooth) R L N n_samples out.npz
"""
import sys, time
from pathlib import Path
import numpy as np
import scipy.fft as sfft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grf_sphere import Mesh, P_model, K_EDGES  # noqa: E402


def main(kind, R, L, N, n_samp, out, s_leg=1, s_anchor=2, seed=0):
    t0 = time.time()
    m = Mesh(L=L, N=N)
    nb = len(K_EDGES) - 1
    if kind == 'sphere':
        mask = (m.r < R).astype(float)
    elif kind == 'box':
        mask = np.ones_like(m.r)
    else:
        mask = np.where(m.r < R, np.exp(-m.r ** 2 / (2 * (R / 3.0) ** 2)), 0.0)
    nbar = 1.0                                      # cancels between I3 and the G's
    I3 = nbar ** 3 * np.sum(mask ** 3) * m.dV
    mt = sfft.fftn(mask, workers=2) * m.dV           # m~(k)
    P = P_model(np.where(m.k > 0, m.k, 1e-10)); P[m.k == 0] = 0.0
    P_SCALE = 1e4
    Pf = sfft.fftn(P / P_SCALE, workers=2)
    cube = [(a, b, c) for a in range(-s_anchor, s_anchor + 1) for b in range(-s_anchor, s_anchor + 1)
            for c in range(-s_anchor, s_anchor + 1)]
    sidx = {s: n for n, s in enumerate(cube)}
    K = np.empty((len(cube),) + mt.shape, dtype=np.complex64)
    mtc = np.conj(mt)
    for n, s in enumerate(cube):
        f = mt * np.roll(mtc, shift=s, axis=(0, 1, 2))                    # m~(u) m~(s - u)
        K[n] = (sfft.ifftn(sfft.fftn(f, workers=2) * Pf, workers=2) * (nbar ** 2 / L ** 3)).astype(np.complex64)
    print(f'K grids ({len(cube)}) in {time.time() - t0:.0f}s', flush=True)
    kf = 2 * np.pi / L
    # lattice indices of the shells
    idx = {i: np.flatnonzero(s.ravel()) for i, s in enumerate(m.shells)}
    nmod = np.array([len(idx[i]) for i in range(nb)])
    shape = m.k.shape

    def kmag_of(flat_idx, shift):
        """|k| of the lattice vectors flat_idx shifted by the integer vector shift (periodic)."""
        ijk = np.array(np.unravel_index(flat_idx, shape))             # (3, n)
        ijk = (ijk + np.asarray(shift)[:, None]) % N
        kk = ((ijk + N // 2) % N - N // 2) * kf
        return np.sqrt((kk ** 2).sum(0)), np.ravel_multi_index(tuple(ijk), shape)

    def shell_of(kmag):
        b = np.digitize(kmag, K_EDGES) - 1
        return np.where((b >= 0) & (b < nb), b, -1)

    def neg(flat_idx):
        ijk = np.array(np.unravel_index(flat_idx, shape))
        return np.ravel_multi_index(tuple((-ijk) % N), shape)

    legs = [s for s in cube if max(abs(c) for c in s) <= s_leg]
    rng = np.random.default_rng(seed)
    C1 = np.zeros((nb, nb, nb, nb)); C2 = np.zeros((nb, nb, nb, nb))
    Kflat = K.reshape(len(cube), -1)
    for i in range(nb):
        for j in range(nb):
            q1 = rng.choice(idx[i], n_samp); q2 = rng.choice(idx[j], n_samp)
            Q = kmag_of(q1, (0, 0, 0))[1]                                  # placeholder for index arithmetic
            ijk1 = np.array(np.unravel_index(q1, shape)); ijk2 = np.array(np.unravel_index(q2, shape))
            Qidx = np.ravel_multi_index(tuple((ijk1 + ijk2) % N), shape)
            negQ = neg(Qidx)
            acc1 = np.zeros((nb, nb)); acc2 = np.zeros((nb, nb))
            for s1 in legs:
                # class 1: p1 = q1 - s1 (s1 = q1 - p1), value K_{s1}(q1); p2 = q2 - s2; anchor K_{-(s1+s2)}(-Q)
                kp1, _ = kmag_of(q1, tuple(-c for c in s1)); b1 = shell_of(kp1)
                g1 = Kflat[sidx[s1], q1].astype(np.complex128)
                for s2 in legs:
                    kp2, _ = kmag_of(q2, tuple(-c for c in s2)); b2 = shell_of(kp2)
                    g2 = Kflat[sidx[s2], q2].astype(np.complex128)
                    s3 = tuple(-(a + b) for a, b in zip(s1, s2))
                    g3 = Kflat[sidx[s3], negQ].astype(np.complex128)
                    val = (g1 * g2 * g3).real * P_SCALE ** 3
                    ok = (b1 >= 0) & (b2 >= 0)
                    np.add.at(acc1, (b1[ok], b2[ok]), val[ok])
                    # class 2 (1 1')(2 3')(3 2'): p1 = q1 - s1, p2 = -Q + s1 + s2, value K_{s1}(q1) K_{s2}(q2) K_{-(s1+s2)}(-Q)
                    kp2c, _ = kmag_of(negQ, tuple(a + b for a, b in zip(s1, s2))); b2c = shell_of(kp2c)
                    ok = (b1 >= 0) & (b2c >= 0)
                    np.add.at(acc2, (b1[ok], b2c[ok]), val[ok])
            norm = nmod[i] * nmod[j] / n_samp                               # MC sum -> full sum over (q1, q2)
            for ip in range(nb):
                for jp in range(nb):
                    C1[i, j, ip, jp] = acc1[ip, jp] * norm / (I3 ** 2 * nmod[i] * nmod[j] * nmod[ip] * nmod[jp])
                    C2[i, j, ip, jp] = acc2[ip, jp] * norm / (I3 ** 2 * nmod[i] * nmod[j] * nmod[ip] * nmod[jp])
            print(f'  ({i},{j}) done {time.time() - t0:.0f}s', flush=True)
    # second class-1 pairing (1 2')(2 1')(3 3'): primed shells swapped; class 2: the four transposes
    C1_tot = C1 + C1.transpose(0, 1, 3, 2)
    C2_tot = C2 + C2.transpose(1, 0, 2, 3) + C2.transpose(0, 1, 3, 2) + C2.transpose(1, 0, 3, 2)
    # k3 pairing (1 2)(1' 2')(3 3') with the mean-subtracted anchor correlator
    A = np.zeros((nb, nb, len(cube)), dtype=complex)
    for i in range(nb):
        for n, s in enumerate(cube):
            kq2, _ = kmag_of(idx[i], s)                                      # |s - q1| = |q1 - s| (even shells)
            b = shell_of(kq2)
            for j in range(nb):
                A[i, j, n] = np.sum(Kflat[n, idx[i]][b == j].astype(np.complex128))
    zero = np.ravel_multi_index((0, 0, 0), shape)
    m0 = mt.ravel()[zero]
    def mt_at(s):
        return mt.ravel()[np.ravel_multi_index(tuple(np.array(s) % N), shape)]
    def C0(s):   # G0(0, s) = K_s(0)
        return complex(Kflat[sidx[s], zero])
    Gk = np.zeros((len(cube), len(cube)), dtype=complex)
    for n, s in enumerate(cube):
        for n2, s2 in enumerate(cube):
            d = tuple(b - a for a, b in zip(s, s2))                          # s' - s
            if d in sidx:
                negs = np.ravel_multi_index(tuple((-np.array(s)) % N), shape)
                G0 = complex(Kflat[sidx[d], negs])
                ms, ms2 = mt_at(tuple(-c for c in s)), mt_at(s2)
                Gk[n, n2] = G0 - ms / m0 * C0(s2) - ms2 / m0 * C0(tuple(-c for c in s)) + ms * ms2 / m0 ** 2 * C0((0, 0, 0))
    Ck3 = np.zeros((nb, nb, nb, nb))
    for i in range(nb):
        for j in range(nb):
            for ip in range(nb):
                for jp in range(nb):
                    Ck3[i, j, ip, jp] = (A[i, j] @ Gk @ np.conj(A[ip, jp])).real * P_SCALE ** 3 / (I3 ** 2 * nmod[i] * nmod[j] * nmod[ip] * nmod[jp])
    np.savez(out, C_class1=C1_tot, C_class2=C2_tot, C_k3=Ck3, k_edges=K_EDGES, R=R, L=L, N=N, nmodes=nmod, n_samp=n_samp)
    print(f'saved {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    kind, R, L, N, n_samp, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    main(kind, R, L, N, n_samp, out)
