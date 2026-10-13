r"""Three-point window invariants Q_n(r1, r2) from anchored pair sums (guide, sec. 'The window invariants from pair sums').

    Q_n(r1, r2) = int d^3x W_a(x) sum_{k1 k2 m3} (K1 K2 Lam3; k1 k2 m3) Y_{Lam3 m3}(x^) A^{lam1 L1}_{K1 k1}(x; r1) A^{lam2 L2}_{K2 k2}(x; r2),
    A^{lam L}_{K k}(x; r) = int dOmega_r W_l(x + r) [Y_lam(r^) x Y_L(y^)]_{K k},    y = x + r,

with n = (lam1, L1, K1, lam2, L2, K2, Lam3).  A is a PAIR quantity: the harmonic moments of the leg window on the shell
of radius r about the anchor x, accumulated from the (anchor, neighbour) pairs in bins of r exactly like the pair counts
of windows.py.  The three-point function is the sum over anchors of products of two moments (Slepian-Eisenstein): no
triplets.  Each anchor's frame is rotated so that x^ = z^, which forces m3 = 0 and k2 = -k1.

Estimator.  Anchors are a subsample of the host randoms (weights w~_a: the W or S window at the anchor, alpha_a the
subsample's effective alpha); neighbours of a shell bin b are the far subsample (r >= s_split, all pairs) or the KD-tree
neighbours of the near subsample (r < s_split), with tilde weights w~_l of the leg window and their own alpha_l, so that
    A(x; b) = alpha_l sum_{y in b} w~_l(y) [Y x Y]_{K k}(y) / int_b r^2 dr,
    Q_n(b1, b2) = alpha_a sum_x w~_a(x) sum_k (K1 K2 Lam3; k -k 0) Y_{Lam3 0}(z^) A1(x; b1)[k] A2(x; b2)[-k]
                  - (self pairs: the same neighbour y in both factors when b1 = b2).
The eight (anchor, leg1, leg2) in {W, S}^3 combinations use the same pairs with different weight columns.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .bispectrum_algebra import cg_tensor
from .harmonics import sph_harm_table, angles
from .tracers import Window
from .wigner import w3j_tensor, FOUR_PI


def rotation_to_z(xhat: np.ndarray) -> np.ndarray:
    """Rotation matrices R (n, 3, 3) with R xhat = z^ (R = R_y(-theta) R_z(-phi))."""
    ct = np.clip(xhat[:, 2], -1.0, 1.0)
    st = np.sqrt(1.0 - ct ** 2)
    phi = np.arctan2(xhat[:, 1], xhat[:, 0])
    cp, sp = np.cos(phi), np.sin(phi)
    R = np.empty((len(xhat), 3, 3))
    R[:, 0, 0] = ct * cp
    R[:, 0, 1] = ct * sp
    R[:, 0, 2] = -st
    R[:, 1, 0] = -sp
    R[:, 1, 1] = cp
    R[:, 1, 2] = 0.0
    R[:, 2, 0] = st * cp
    R[:, 2, 1] = st * sp
    R[:, 2, 2] = ct
    return R


class MomentLabels:
    """The (lam, L, K) labels of the leg moments and the n's that use them."""

    def __init__(self, ns):
        self.ns = sorted(set(tuple(int(v) for v in n) for n in ns))
        labs = set()
        for (lam1, L1, K1, lam2, L2, K2, Lam3) in self.ns:
            labs.add((lam1, L1, K1))
            labs.add((lam2, L2, K2))
        self.labels = sorted(labs)
        self.index = {lab: i for i, lab in enumerate(self.labels)}
        self.lam_max = max(l[0] for l in self.labels) if self.labels else 0
        self.L_max = max(l[1] for l in self.labels) if self.labels else 0
        self.K_max = max(l[2] for l in self.labels) if self.labels else 0
        self.nk = 2 * self.K_max + 1

    def bipolar(self, rhat: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """[Y_lam(r^) x Y_L(y^)]_{K k} for all labels: (n_pairs, n_labels, nk), k index = kappa + K_max."""
        lmax = max(self.lam_max, self.L_max)
        ct, ph = angles(rhat)
        Yr = sph_harm_table(ct, ph, lmax)                  # [l, m + lmax, pair]
        ct, ph = angles(yhat)
        Yy = sph_harm_table(ct, ph, lmax)
        out = np.zeros((len(rhat), len(self.labels), self.nk), dtype=complex)
        for i, (lam, L, K) in enumerate(self.labels):
            CG = cg_tensor(lam, L, K)                      # [rho, M, kappa]
            Yl = Yr[lam, lmax - lam:lmax + lam + 1]        # [rho, pair]
            YL = Yy[L, lmax - L:lmax + L + 1]              # [M, pair]
            YY = (Yl[:, None, :] * YL[None, :, :]).reshape((2 * lam + 1) * (2 * L + 1), -1)   # [(rho M), pair]
            out[:, i, self.K_max - K:self.K_max + K + 1] = YY.T @ CG.reshape(-1, 2 * K + 1)
        return out


class AnchoredMoments:
    """Pair sums for Q_n of one host tracer: several anchor windows (weights at x) and leg windows (weights at y).

    Parameters: see BispectrumGaussianCovariance.  `r_edges` are the radial bins of the moments.
    """

    def __init__(self, anchor_windows, leg_windows, r_edges, n_anchor=2000, n_sub=5000, n_near=200000,
                 s_split=80.0, seed=0, chunk_pairs=50000, n_threads=None):
        self.anchor_windows = list(anchor_windows)
        self.leg_windows = list(leg_windows)
        hosts = {w.host.name for w in self.anchor_windows + self.leg_windows}
        if len(hosts) != 1:
            raise NotImplementedError("anchored moments need all windows sampled by the same tracer's randoms")
        self.host = self.anchor_windows[0].host
        self.r_edges = np.asarray(r_edges, dtype=float)
        self.nb = len(self.r_edges) - 1
        self.vol = (self.r_edges[1:] ** 3 - self.r_edges[:-1] ** 3) / 3.0
        self.n_anchor, self.n_sub, self.n_near = int(n_anchor), int(n_sub), int(n_near)
        self.s_split = float(self.r_edges[int(np.argmin(np.abs(self.r_edges - s_split)))])   # snap to an edge
        self.seed = int(seed)
        self.chunk_pairs = int(chunk_pairs)
        self.Q = {}          # (ia, il1, il2, n) -> (nb, nb) real array
        self.imag_fraction = {}
        self.npairs = None   # unweighted pairs per r bin (anchor-neighbour), diagnostics

    # ------------------------------------------------------------------------------------------ sampling
    def _anchors(self):
        A = self.host
        idx = A.subsample_indices(self.n_anchor, self.seed + 7919)
        pos = A.pos[idx]
        w = np.stack([aw.tilde_weights()[idx] for aw in self.anchor_windows], axis=1)   # (n_anchor, n_akinds)
        alpha = A.alpha * A.size / len(idx)
        return pos, w, alpha

    def _neighbours(self, n):
        A = self.host
        idx = A.subsample_indices(n, self.seed)
        pos = A.pos[idx]
        w = np.stack([lw.tilde_weights()[idx] for lw in self.leg_windows], axis=1)       # (n, n_lkinds)
        alpha = A.alpha * A.size / len(idx)
        return pos, w, alpha

    # ------------------------------------------------------------------------------------------ moments
    def _accumulate(self, labels: MomentLabels, xpos, R, i_pair, d, wl, alpha_l, bins, A, SP, groups):
        """Add the pairs (anchor index i_pair within the block, separation d, leg weights wl (npair, n_lkinds)) to the
        moments A[kind, x, label, k, b] and the self-pair sums SP[(il1, il2, n)][x, b]."""
        r = np.linalg.norm(d, axis=1)
        ok = (r > 0) & (bins >= 0) & (bins < self.nb)
        if not np.any(ok):
            return
        i_pair, d, wl, r, bins = i_pair[ok], d[ok], wl[ok], r[ok], bins[ok]
        y = xpos[i_pair] + d
        Rp = R[i_pair]
        rhat = np.einsum('pij,pj->pi', Rp, d / r[:, None])
        yhat = np.einsum('pij,pj->pi', Rp, y / np.linalg.norm(y, axis=1)[:, None])
        B = labels.bipolar(rhat, yhat)                                  # (npair, nlab, nk)
        cell = i_pair * self.nb + bins
        ncell = len(xpos) * self.nb
        norm = alpha_l / self.vol[bins]
        for kind in range(wl.shape[1]):
            S = csr_matrix((wl[:, kind] * norm, (cell, np.arange(len(cell)))), shape=(ncell, len(cell)))
            Bk = (S @ B.reshape(len(cell), -1)).reshape(len(xpos), self.nb, len(labels.labels), labels.nk)
            A[kind] += np.transpose(Bk, (0, 2, 3, 1))
        # self pairs (same neighbour in both factors): grouped by the two moment labels, one anti-diagonal product
        # P[p, k] = B1[p, k] B2[p, -k] per group, then one 3j contraction and one bincount per n
        if not groups:
            return
        kmax = labels.K_max
        S1 = csr_matrix((np.ones(len(cell)), (cell, np.arange(len(cell)))), shape=(ncell, len(cell)))
        for (il1, il2, lab1, lab2), members in groups.items():
            K1, K2 = lab1[2], lab2[2]
            Kc = min(K1, K2)
            b1 = B[:, labels.index[lab1], kmax - Kc:kmax + Kc + 1]
            b2 = B[:, labels.index[lab2], kmax - Kc:kmax + Kc + 1][:, ::-1]        # k -> -k
            P = b1 * b2 * (wl[:, il1] * wl[:, il2] * norm ** 2)[:, None]
            W3 = np.stack([w3 for (_, w3) in members], axis=1)                       # [k, member], (K1 K2 Lam3; k -k 0)
            acc = (S1 @ (P @ W3)).reshape(len(xpos), self.nb, len(members))
            for m, (n, _) in enumerate(members):
                SP[(il1, il2, n)] += acc[:, :, m]

    @staticmethod
    def _self_groups(self_keys, labels: MomentLabels):
        groups = {}
        for (il1, il2, n) in self_keys:
            lam1, L1, K1, lam2, L2, K2, Lam3 = n
            Kc = min(K1, K2)
            W3 = w3j_tensor(K1, K2, Lam3)[:, :, Lam3]
            w3 = np.array([W3[k + K1, -k + K2] for k in range(-Kc, Kc + 1)])
            groups.setdefault((il1, il2, (lam1, L1, K1), (lam2, L2, K2)), []).append((n, w3))
        return groups

    def _products(self, requests, labels: MomentLabels, w_anchor, A, SP, Q):
        """Q[(ia, il1, il2, n)] += alpha_a sum_x w~_a(x) sum_k (K1 K2 Lam3; k -k 0) Y_{Lam3 0}(z^) A1 A2 - self pairs."""
        kmax = labels.K_max
        for (ia, il1, il2), ns in requests.items():
            w = w_anchor[:, ia]
            for n in ns:
                lam1, L1, K1, lam2, L2, K2, Lam3 = n
                Kc = min(K1, K2)
                W3 = w3j_tensor(K1, K2, Lam3)[:, :, Lam3]
                w3 = np.array([W3[k + K1, -k + K2] for k in range(-Kc, Kc + 1)])
                Y0 = math.sqrt((2 * Lam3 + 1) / FOUR_PI)
                A1 = A[il1][:, labels.index[(lam1, L1, K1)], kmax - Kc:kmax + Kc + 1, :]          # (x, k, b1)
                A2 = A[il2][:, labels.index[(lam2, L2, K2)], kmax - Kc:kmax + Kc + 1, :][:, ::-1, :]   # k -> -k
                nx = len(w)
                left = (A1 * (w[:, None, None] * w3[None, :, None])).reshape(nx * (2 * Kc + 1), -1)
                right = A2.reshape(nx * (2 * Kc + 1), -1)
                dQ = Y0 * (left.T @ right)
                dQ -= Y0 * np.diag(w @ SP[(il1, il2, n)])
                Q[(ia, il1, il2, n)] = Q.get((ia, il1, il2, n), 0.0) + dQ

    def compute(self, requests, verbose=False, block_anchors=400):
        """requests: {(ia, il1, il2): set of n}.  Fills self.Q[(ia, il1, il2, n)].  Anchors are processed in blocks
        (the moments of a block are (kinds, anchors, labels, 2 K_max + 1, bins) complex)."""
        log = print if verbose else (lambda *a: None)
        all_n = set()
        for ns in requests.values():
            all_n |= set(ns)
        labels = MomentLabels(all_n)
        self_keys = sorted({(il1, il2, n) for (ia, il1, il2), ns in requests.items() for n in ns})
        groups = self._self_groups(self_keys, labels)
        xpos_all, wa_all, alpha_a = self._anchors()
        R_all = rotation_to_z(xpos_all / np.linalg.norm(xpos_all, axis=1)[:, None])
        nl = len(self.leg_windows)
        self.npairs = np.zeros(self.nb)
        Q = {}
        yfar, wfar, alpha_far = self._neighbours(self.n_sub)
        near = np.any(self.r_edges[:-1] < self.s_split - 1e-9)
        if near:
            ynear, wnear, alpha_near = self._neighbours(self.n_near)
            tree = cKDTree(ynear)
        t0 = time.time()
        tf = tn = tp = 0.0
        for b0 in range(0, len(xpos_all), block_anchors):
            b1 = min(len(xpos_all), b0 + block_anchors)
            xpos, wa, R = xpos_all[b0:b1], wa_all[b0:b1], R_all[b0:b1]
            nx = len(xpos)
            A = np.zeros((nl, nx, len(labels.labels), labels.nk, self.nb), dtype=complex)
            SP = {key: np.zeros((nx, self.nb), dtype=complex) for key in self_keys}
            # far neighbours: all pairs with the n_sub subsample, r >= s_split
            t = time.time()
            step = max(1, self.chunk_pairs // max(len(yfar), 1))
            for i0 in range(0, nx, step):
                i1 = min(nx, i0 + step)
                d = (yfar[None, :, :] - xpos[i0:i1, None, :]).reshape(-1, 3)
                ip = np.repeat(np.arange(i0, i1), len(yfar))
                r = np.linalg.norm(d, axis=1)
                keep = r >= self.s_split
                bins = np.digitize(r[keep], self.r_edges) - 1
                self.npairs += np.bincount(bins[(bins >= 0) & (bins < self.nb)], minlength=self.nb)
                wl = np.broadcast_to(wfar[None, :, :], (i1 - i0,) + wfar.shape).reshape(-1, nl)[keep]
                self._accumulate(labels, xpos, R, ip[keep], d[keep], wl, alpha_far, bins, A, SP, groups)
            tf += time.time() - t
            # near neighbours: KD-tree of the n_near subsample, r < s_split
            if near:
                t = time.time()
                m = cKDTree(xpos).sparse_distance_matrix(tree, self.s_split, output_type='ndarray')
                if len(m):
                    ip = m['i'].astype(np.int64)
                    j = m['j'].astype(np.int64)
                    for c0 in range(0, len(ip), self.chunk_pairs):
                        ipc, jc = ip[c0:c0 + self.chunk_pairs], j[c0:c0 + self.chunk_pairs]
                        d = ynear[jc] - xpos[ipc]
                        r = np.linalg.norm(d, axis=1)
                        keep = (r > 0) & (r < self.s_split)
                        bins = np.digitize(r[keep], self.r_edges) - 1
                        self.npairs += np.bincount(bins[(bins >= 0) & (bins < self.nb)], minlength=self.nb)
                        self._accumulate(labels, xpos, R, ipc[keep], d[keep], wnear[jc][keep], alpha_near, bins, A, SP, groups)
                tn += time.time() - t
            t = time.time()
            self._products(requests, labels, alpha_a * wa, A, SP, Q)
            tp += time.time() - t
            log(f"  anchors {b1}/{len(xpos_all)}: far {tf:.1f}s, near {tn:.1f}s, products {tp:.1f}s")
        for key, val in Q.items():
            # Phi_n is real or imaginary according to the parity of K1 + K2 + Lam3 (the coupling numbers are real,
            # so the imaginary invariants never enter); keep the real part and the imaginary fraction as a check
            scale = np.max(np.abs(val))
            self.imag_fraction[key] = float(np.max(np.abs(val.imag)) / scale) if scale > 0 else 0.0
            self.Q[key] = np.ascontiguousarray(val.real)
        log(f"  {len(Q)} invariants from {len(xpos_all)} anchors in {time.time() - t0:.1f}s")
        return self


class ThreePointWindowLibrary:
    """Requests, computes, interpolates and stores Q_n(r1, r2) for the window-kind triples of one tracer.

    Window kinds are 'W' (clustering, W = m^2) and 'S' (shot noise).  `get(kinds, n, r)` returns Q_n on the (r, r) grid
    (cubic spline through the bin centres, constant beyond them).
    """

    def __init__(self, r_edges, n_anchor=2000, n_sub=5000, n_near=200000, s_split=80.0, seed=0, chunk_pairs=50000):
        self.r_edges = np.asarray(r_edges, dtype=float)
        self.opts = dict(n_anchor=n_anchor, n_sub=n_sub, n_near=n_near, s_split=s_split, seed=seed, chunk_pairs=chunk_pairs)
        self._requests = {}      # tracer name -> {(ia, il1, il2): set(n)}
        self._windows = {}       # tracer name -> ({kind: window}, AnchoredMoments)
        self._tracers = {}
        self._interp = {}

    @staticmethod
    def kinds_index(kind):
        return {'W': 0, 'S': 1}[kind]

    def request(self, tracer, kinds, ns):
        """kinds = (anchor, leg1, leg2) in {'W', 'S'}^3."""
        self._tracers[tracer.name] = tracer
        key = tuple(self.kinds_index(k) for k in kinds)
        self._requests.setdefault(tracer.name, {}).setdefault(key, set()).update(tuple(int(v) for v in n) for n in ns)

    def missing(self):
        out = []
        for name, req in self._requests.items():
            have = self._windows.get(name)
            for key, ns in req.items():
                if have is None or any((key + (n,)) not in have.Q for n in ns):
                    out.append((name, key))
                    break
        return out

    def compute_all(self, verbose=False):
        todo = {m[0] for m in self.missing()}
        for name, req in self._requests.items():
            if name in todo:
                T = self._tracers[name]
                wins = [Window('W', T, T), Window('S', T)]
                am = AnchoredMoments(wins, wins, self.r_edges, **self.opts)
                if verbose:
                    print(f"three-point window of {name}: {sum(len(v) for v in req.values())} invariants")
                am.compute(req, verbose=verbose)
                self._windows[name] = am
        self._interp = {}

    def get(self, tracer_name, kinds, n, r):
        key = (tracer_name, tuple(self.kinds_index(k) for k in kinds), tuple(n), id(r))
        if key not in self._interp:
            am = self._windows[tracer_name]
            Q = am.Q.get(key[1] + (tuple(n),))
            if Q is None:
                self._interp[key] = None
            else:
                c = 0.5 * (self.r_edges[1:] + self.r_edges[:-1])
                good = am.npairs > 0
                spl = RectBivariateSpline(c[good], c[good], Q[np.ix_(good, good)], kx=3, ky=3)
                rc = np.clip(r, c[good][0], c[good][-1])
                self._interp[key] = spl(rc, rc)
        return self._interp[key]

    # ------------------------------------------------------------------------------------------ persistence
    def save(self, path):
        data = {'r_edges': self.r_edges}
        meta = []
        for name, am in self._windows.items():
            data[f'npairs_{name}'] = am.npairs
            for (ia, il1, il2, n), Q in am.Q.items():
                qn = f'Q_{len(meta)}'
                data[qn] = Q
                meta.append({'tracer': name, 'kinds': [ia, il1, il2], 'n': list(n), 'q': qn})
        data['meta'] = np.array(json.dumps(meta))
        np.savez(path, **data)

    def load(self, path):
        with np.load(path) as f:
            if not np.allclose(f['r_edges'], self.r_edges):
                raise ValueError("r_edges of the stored windows differ from the current ones")
            meta = json.loads(str(f['meta'].item()))
            for rec in meta:
                name = rec['tracer']
                if name not in self._windows:
                    am = AnchoredMoments.__new__(AnchoredMoments)
                    am.r_edges, am.nb = self.r_edges, len(self.r_edges) - 1
                    am.Q, am.npairs, am.imag_fraction = {}, None, {}
                    self._windows[name] = am
                am = self._windows[name]
                am.Q[tuple(rec['kinds']) + (tuple(rec['n']),)] = f[rec['q']]
                if am.npairs is None:
                    am.npairs = f[f'npairs_{name}']
        self._interp = {}
        return self
