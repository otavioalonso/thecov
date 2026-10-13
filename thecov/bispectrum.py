r"""Gaussian covariance of windowed bispectrum multipoles B_{l1 l2 L}(k_i, k_j) in the Sugiyama basis: class-1 term.

Derivation: desi_validation/report/bispectrum_covariance_guide.typ (and bispectrum_covariance.typ).  The class-1 pairings
(1 1')(2 2')(3 3') and (1 2')(2 1')(3 3') give

    C^{(1)}_{ell ell'}(i j; i' j') = (4 pi)^3 H H' sqrt(N N') / I_3^2
        sum_{L1 L2 L3 a b Lam Lam'} (4 pi)^3 / ((2L1+1)(2L2+1)(2L3+1)) (-i)^{l1+l2} i^{Lam+Lam'}
        sum_n T_n  int r1^2 dr1 r2^2 dr2  jbar^{(i)}_{l1}(r1) jbar^{(j)}_{l2}(r2)  P^{(i'j')}_{L1 L2 L3; ab; Lam Lam'}(r1, r2)  Q_n(r1, r2),

    P^{(i'j')}(r1, r2) = << j_Lam(p1 r1) j_Lam'(p2 r2) P_L1(p1) P_L2(p2) c^{L3}_{ab}(p1, p2) >>_{i' j'}  (radial shell averages),

with T_n and c^{L3}_{ab} from bispectrum_algebra and the window invariants Q_n(r1, r2) from bispectrum_windows.  The
unprimed estimator (shells i, j) contributes bin-averaged Bessel functions only; the primed one (i', j') carries the
three power spectra.  Shot noise: a leg with S instead of W has L_a = 0 and P -> 1; an anchor with S has L3 = 0 and
c -> sqrt(4 pi) delta_{a0} delta_{b0}.

Radial assembly.  Writing P^{(i'j')} through the shell quadrature nodes, P = sum_{q1 in i', q2 in j'} A_{q1}(r1) c_{q1 q2} B_{q2}(r2),
the double integral for ALL (i, j; i', j') at once is

    C[i, j, i', j'] = sum_{q1 q2} c[i' q1, j' q2]  M[i q1, j q2],    M = U Qt V^T,
    U[(i, q1), r1] = r1^2 dr1 jbar^{(i)}_{l1}(r1) w_{q1} P_L1(k_{q1}) j_Lam(k_{q1} r1) / norm_{i'},   Qt = sum_n T_n Q_n,

so the (nr x nr) window matrix is multiplied once per radial label set instead of once per shell pair.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.special import spherical_jn

from .bispectrum_algebra import (BispectrumCoupling, bipolar_coefficients, H, N, phase, box_window_value)
from .harmonics import normalized_legendre
from .kernels import PowerSpectrumModel
from .wigner import FOUR_PI, tri, tri_multi


# ---------------------------------------------------------------------------------------------------- model helpers
def extrapolated_power(model: PowerSpectrumModel, A, B, L):
    """P_L^{AB}(k) as a callable that extends the tabulated range: a power law fitted to the two lowest tabulated points
    below k_min (the third side |p1 + p2| reaches 0 on the i' = j' shells), and P(k_max) (k/k_max)^-3 above k_max."""
    spl = model._spl.get(model._key(A, B) + (int(L),))
    if spl is None:
        return lambda k: np.zeros_like(np.asarray(k, dtype=float))
    kmin, kmax = spl.x[0], spl.x[-1]
    k1 = min(spl.x[1], 1.5 * kmin)
    p0, p1 = float(spl(kmin)), float(spl(k1))
    slope = math.log(abs(p1 / p0)) / math.log(k1 / kmin) if p0 != 0 and p1 != 0 and p1 / p0 > 0 else 1.0
    pmax = float(spl(kmax))

    def P(k):
        k = np.asarray(k, dtype=float)
        kc = np.clip(k, kmin, kmax)
        out = spl(kc)
        out = np.where(k < kmin, p0 * (np.maximum(k, 1e-30) / kmin) ** slope, out)
        out = np.where(k > kmax, pmax * (k / kmax) ** -3, out)
        return out
    return P


# ---------------------------------------------------------------------------------------------------- shell kernels
class BispectrumShellKernels:
    """Gauss-Legendre nodes on the k shells, bin-averaged Bessel functions jbar^{(i)}_l(r), the node-wise Bessel tables
    j_Lam(k_q r) and the vertex matrices c^{L3}_{ab}(k_q1, k_q2) over all node pairs.

    Parameters
    ----------
    k_edges : shell edges (h/Mpc)
    r       : separation grid for the radial integrals (Mpc/h); the integration weights are r^2 dr (midpoint)
    n_quad  : nodes per shell (default: enough to resolve j(kr) across a bin at r.max())
    n_mu    : nodes of the opening-angle quadrature of the vertex coefficients
    """

    def __init__(self, k_edges, r, n_quad: int | None = None, n_mu: int = 64):
        self.k_edges = np.asarray(k_edges, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.nbins = len(self.k_edges) - 1
        self.nr = len(self.r)
        dr = np.gradient(self.r) if self.nr > 1 else np.array([1.0])
        self.r_weights = self.r ** 2 * dr
        if n_quad is None:
            dk = np.max(np.diff(self.k_edges))
            osc = dk * self.r.max() / (2 * np.pi)
            n_quad = int(10 * osc) + 16
        self.nq = int(n_quad)
        x, w = np.polynomial.legendre.leggauss(self.nq)
        lo, hi = self.k_edges[:-1], self.k_edges[1:]
        self.kq = 0.5 * (hi - lo)[:, None] * x[None, :] + 0.5 * (hi + lo)[:, None]      # (nbins, nq)
        self.wq = 0.5 * (hi - lo)[:, None] * w[None, :] * self.kq ** 2                   # includes k^2
        self.norm = (hi ** 3 - lo ** 3) / 3.0                                            # int_i k^2 dk
        self.n_mu = int(n_mu)
        self._bessel = {}
        self._bessel_fine = {}
        self._p3 = {}
        self._vertex = {}

    def bessel(self, lam: int) -> np.ndarray:
        """j_lam(k_q r) for all nodes: (nbins, nq, nr)."""
        if lam not in self._bessel:
            self._bessel[lam] = spherical_jn(lam, self.kq[:, :, None] * self.r[None, None, :])
        return self._bessel[lam]

    def jbar(self, ell: int) -> np.ndarray:
        """Bin-averaged jbar^{(i)}_ell(r): (nbins, nr)."""
        return np.einsum('iq,iqr->ir', self.wq, self.bessel(ell)) / self.norm[:, None]

    # ------------------------------------------------------------------------------------------ class 2
    def p3_nodes(self, ip: int, jp: int, n3: int = 10):
        """Composite Gauss-Legendre nodes in p3 = |p1 + p2| for every p1 node of shell ip, with p2 in shell jp:
        p3 in [max(klo - p1, p1 - khi, 0), p1 + khi], split at the kinks |klo -+ p1|, khi -+ p1 where the opening-angle
        range of the indicator changes AND at every shell edge (the radial integral over r2 selects the shell j of p3
        with a kernel that is a step on the scale of the window, which a piece must not straddle).
        Returns (p3, w3) of shape (nq, n_nodes) with w3 including p3^2 (zero padding)."""
        key = (ip, jp, n3)
        if key not in self._p3:
            klo, khi = self.k_edges[jp], self.k_edges[jp + 1]
            x, w = np.polynomial.legendre.leggauss(n3)
            P3, W3 = [], []
            for p1 in self.kq[ip]:
                lo, hi = max(klo - p1, p1 - khi, 0.0), p1 + khi
                kinks = sorted({lo, hi} | {k for k in (abs(klo - p1), abs(khi - p1), klo + p1, khi + p1, *self.k_edges)
                                          if lo < k < hi})
                nodes, wts = [], []
                for a, b in zip(kinks[:-1], kinks[1:]):
                    nodes.append(0.5 * (b - a) * x + 0.5 * (b + a))
                    wts.append(0.5 * (b - a) * w)
                nodes, wts = np.concatenate(nodes), np.concatenate(wts)
                P3.append(nodes)
                W3.append(wts * nodes ** 2)
            nmax = max(len(v) for v in P3)
            p3 = np.zeros((self.nq, nmax))
            w3 = np.zeros((self.nq, nmax))
            for q, (v, u) in enumerate(zip(P3, W3)):
                p3[q, :len(v)] = v
                w3[q, :len(u)] = u
            self._p3[key] = (p3, w3)
        return self._p3[key]

    def bessel_at(self, lam: int, p: np.ndarray) -> np.ndarray:
        """j_lam(p r) for arbitrary p (any shape) by linear interpolation of a fine table in p (step 1e-4 h/Mpc,
        error <~ (dp r)^2 / 8 ~ 1e-3 at r = 1000): shape p.shape + (nr,)."""
        if lam not in self._bessel_fine:
            g = np.arange(0.0, 2.0 * self.k_edges[-1] + 2e-3, 1e-4)
            self._bessel_fine[lam] = (g, spherical_jn(lam, g[:, None] * self.r[None, :]))
        g, J = self._bessel_fine[lam]
        p = np.asarray(p, dtype=float)
        u = np.clip(p / 1e-4, 0, len(g) - 1.000001)
        i0 = u.astype(int)
        t = (u - i0)[..., None]
        return (1 - t) * J[i0] + t * J[i0 + 1]

    def vertex2(self, lpp: int, a: int, b: int, pfunc, ip: int, jp: int, tag=None, n3: int = 10) -> np.ndarray:
        """Class-2 vertex d^{l''}_{ab}(p1, p3) = (-1)^a c^{l''}_{ab}(p1, p3)[h] with h(p2) = 1(p2 in shell jp) P_L2(p2),
        on the p3 nodes of (ip, jp): shape (nq, n_nodes).  pfunc None -> shot-noise anchor (h = indicator)."""
        key = (tag, lpp, a, b, ip, jp, n3)
        if tag is not None and key in self._vertex:
            return self._vertex[key]
        p3, w3 = self.p3_nodes(ip, jp, n3)
        P1 = np.broadcast_to(self.kq[ip][:, None], p3.shape)
        h = (lambda k: np.ones_like(np.asarray(k, dtype=float))) if pfunc is None else pfunc
        out = (-1) ** a * vertex_coefficients(lpp, a, b, P1.ravel(), p3.ravel(), h, self.n_mu,
                                              k_range=(self.k_edges[jp], self.k_edges[jp + 1])).reshape(p3.shape)
        out = np.where(w3 > 0, out, 0.0)
        if tag is not None:
            self._vertex[key] = out
        return out

    def leg(self, pfunc, L: int, lam: int) -> np.ndarray:
        """w_q P_L(k_q) j_lam(k_q r) / norm_i for a leg carrying the spectrum: (nbins, nq, nr).  pfunc None -> 1."""
        pk = np.ones_like(self.kq) if pfunc is None else np.asarray(pfunc(self.kq), dtype=float)
        return (self.wq * pk / self.norm[:, None])[:, :, None] * self.bessel(lam)

    def vertex(self, L3: int, a: int, b: int, pfunc, tag=None) -> np.ndarray:
        """c^{L3}_{ab}(k_q1, k_q2) over all node pairs: (nbins, nq, nbins, nq).  pfunc None -> shot-noise anchor,
        c = sqrt(4 pi) delta_{a0} delta_{b0} (and L3 must be 0)."""
        key = (tag, L3, a, b)
        if tag is not None and key in self._vertex:
            return self._vertex[key]
        shape = (self.nbins, self.nq, self.nbins, self.nq)
        if pfunc is None:
            out = np.full(shape, math.sqrt(FOUR_PI) if (L3 == 0 and a == 0 and b == 0) else 0.0)
        else:
            k = self.kq.ravel()
            P1, P2 = np.meshgrid(k, k, indexing='ij')
            out = vertex_coefficients(L3, a, b, P1.ravel(), P2.ravel(), pfunc, self.n_mu).reshape(shape)
        if tag is not None:
            self._vertex[key] = out
        return out


def vertex_coefficients(L3: int, a: int, b: int, p1, p2, P_L3, n_quad: int = 64, chunk: int = 4000,
                        k_range=None) -> np.ndarray:
    """Vectorised c^{L3}_{ab}(p1, p2) for flat arrays p1, p2 (same formula as bispectrum_algebra.bipolar_coefficients).

    k_range = (lo, hi): the third side |p1 + p2| is restricted to that shell (the indicator of the class-2 vertex);
    the opening-angle integral then runs over the intersection of [|p1 - p2|, p1 + p2] with the shell (zero if empty).
    """
    from .bispectrum_algebra import cg_tensor
    p1 = np.asarray(p1, dtype=float).ravel()
    p2 = np.asarray(p2, dtype=float).ravel()
    out = np.zeros(len(p1))
    if (a + b + L3) % 2 or L3 < abs(a - b) or L3 > a + b:
        return out
    C = cg_tensor(a, b, L3)
    lmax = max(b, L3)
    x, w = np.polynomial.legendre.leggauss(n_quad)
    gs = range(-min(b, L3), min(b, L3) + 1)
    for c0 in range(0, len(p1), chunk):
        q1, q2 = p1[c0:c0 + chunk, None], p2[c0:c0 + chunk, None]
        lo, hi = np.abs(q1 - q2), q1 + q2
        if k_range is not None:
            lo, hi = np.maximum(lo, k_range[0]), np.minimum(hi, k_range[1])
            hi = np.maximum(hi, lo)                                   # empty intersection -> zero-length interval
        p3 = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        wmu = 0.5 * (hi - lo) * w * p3 / (q1 * q2)
        mu = np.clip((p3 ** 2 - q1 ** 2 - q2 ** 2) / (2 * q1 * q2), -1.0, 1.0)
        ct3 = np.clip((q1 + q2 * mu) / p3, -1.0, 1.0)
        Pb2 = normalized_legendre(mu, lmax)                       # [l, m, pair, node]; Y_{l m}(theta, 0) = Pbar, m >= 0
        Pb3 = normalized_legendre(ct3, lmax)
        f = np.asarray(P_L3(p3), dtype=float) * wmu
        acc = 0.0
        for g in gs:
            s = (-1.0) ** g if g < 0 else 1.0                      # Y_{l,-m}(theta, 0) = (-1)^m Pbar[l, m]
            acc = acc + C[a, g + b, g + L3] * np.sum(f * Pb2[b, abs(g)] * Pb3[L3, abs(g)], axis=1) * s * s
        out[c0:c0 + chunk] = acc
    return out * 8 * math.pi ** 2 * math.sqrt((2 * a + 1) / FOUR_PI) / (2 * L3 + 1)


# ---------------------------------------------------------------------------------------------------- radial assembly
class Class1Assembler:
    """Radial double integrals of the class-1 term for one (ell, ell') and one window pattern, all (i j; i' j') at once.

    `windows(n)` returns Q_n(r1, r2) on the (nr, nr) grid (None if absent); the three correlators are given as
    (tag, {L: pfunc}) with pfunc(k) the multipole P_L(k) ({0: None} for shot noise, P -> 1); the tag identifies the
    spectrum in the vertex cache.  `lam_w` truncates the window ranks.  Returns C[i, j, i', j'] without the overall
    prefactor (4 pi)^3 H H' sqrt(N N') / I_3^2.
    """

    def __init__(self, kernels: BispectrumShellKernels, coupling: BispectrumCoupling, lam_w: int = 4):
        self.K = kernels
        self.coup = coupling
        self.lam_w = int(lam_w)

    def radial_labels(self, ell, ellp, Ls1, Ls2, Ls3):
        """Distinct (L1, L2, L3, a, b, Lam, Lam') with their list of n = (lam1 L1 K1 lam2 L2 K2 Lam3)."""
        l1, l2, L = ell
        lp1, lp2, Lp = ellp
        out = {}
        for L1 in Ls1:
            for L2 in Ls2:
                for L3 in Ls3:
                    amax = self.lam_w + l1 + lp1 + L1
                    bmax = self.lam_w + l2 + lp2 + L2
                    for a in range(0, amax + 1):
                        for b in range(abs(a - L3), min(a + L3, bmax) + 1):
                            if (a + b + L3) % 2:
                                continue
                            for Lam in tri_multi((lp1, L1, a)):
                                for Lamp in tri_multi((lp2, L2, b)):
                                    ns = []
                                    for lam1 in tri(l1, Lam):
                                        if lam1 > self.lam_w:
                                            continue
                                        for lam2 in tri(l2, Lamp):
                                            if lam2 > self.lam_w:
                                                continue
                                            for (K1, K2, Lam3), T in self.coup.t_all(ell, ellp, L1, L2, L3, a, b, Lam,
                                                                                     Lamp, lam1, lam2).items():
                                                if abs(T) > 1e-15:
                                                    ns.append(((lam1, L1, K1, lam2, L2, K2, Lam3), T))
                                    if ns:
                                        out[(L1, L2, L3, a, b, Lam, Lamp)] = ns
        return out

    def needed_windows(self, ell, ellp, Ls1, Ls2, Ls3):
        return sorted({n for ns in self.radial_labels(ell, ellp, Ls1, Ls2, Ls3).values() for (n, _) in ns})

    def assemble(self, ell, ellp, leg1, leg2, anchor, windows, log=None) -> np.ndarray:
        """C[i, j, i', j'] (no prefactor).  leg1 = (tag, {L: pfunc}) at y1 (shell i'), leg2 at y2 (shell j'), anchor at x."""
        K = self.K
        tag1, p1f = leg1
        tag2, p2f = leg2
        tag3, p3f = anchor
        Ls1, Ls2, Ls3 = tuple(sorted(p1f)), tuple(sorted(p2f)), tuple(sorted(p3f))
        labels = self.radial_labels(ell, ellp, Ls1, Ls2, Ls3)
        nb, nq, nr = K.nbins, K.nq, K.nr
        C = np.zeros((nb, nb, nb, nb))
        jb1 = K.jbar(ell[0]) * K.r_weights[None, :]            # (nb, nr) with r^2 dr
        jb2 = K.jbar(ell[1]) * K.r_weights[None, :]
        t0 = time.time()
        for nlab, ((L1, L2, L3, a, b, Lam, Lamp), ns) in enumerate(labels.items()):
            Qt = None
            for (n, T) in ns:
                Q = windows(n)
                if Q is None:
                    continue
                Qt = T * Q if Qt is None else Qt + T * Q
            if Qt is None:
                continue
            pref = FOUR_PI ** 3 / ((2 * L1 + 1) * (2 * L2 + 1) * (2 * L3 + 1)) * phase(ell, Lam, Lamp)
            A = K.leg(p1f[L1], L1, Lam).reshape(nb * nq, nr)     # [(i', q1), r1]
            B = K.leg(p2f[L2], L2, Lamp).reshape(nb * nq, nr)    # [(j', q2), r2]
            c = K.vertex(L3, a, b, p3f[L3], tag=tag3).reshape(nb * nq, nb * nq)   # [(i', q1), (j', q2)]
            # X[i, i', q1, r2] = sum_r1 jb1[i, r1] A[(i' q1), r1] Qt[r1, r2]      (one matrix product per label)
            X = (np.einsum('ir,Qr->iQr', jb1, A).reshape(nb * nb * nq, nr) @ Qt).reshape(nb, nb, nq, nr)
            c4 = c.reshape(nb, nq, nb, nq)
            B3 = B.reshape(nb, nq, nr)
            for I in range(nb):                                  # shell i' of the primed estimator
                Y = np.einsum('qJp,iqr->iJpr', c4[I], X[:, I])   # [i, j', q2, r2]: vertex applied
                Z = np.einsum('iJpr,Jpr->iJr', Y, B3)            # [i, j', r2]: leg 2 spectrum and Bessel
                C[:, :, I, :] += pref * np.einsum('iJr,jr->ijJ', Z, jb2)
            if log is not None and (nlab % 50 == 0):
                log(f"    label {nlab + 1}/{len(labels)} ({time.time() - t0:.1f}s)")
        return C

    # ------------------------------------------------------------------------------------------ class 2
    def radial_labels2(self, ell, ellp, Ls1, Ls2, Ls3):
        """Class-2 pairing (1 1')(2 3')(3 2'): distinct (L1, L2, L3, l'', a, b, Lam, Lam', Ly) with their n list.
        L1: P(p1; y1^) (leg 1), L2: P(p2; x^) (anchor, inside the vertex), L3: P(p3; y2^) (leg 2)."""
        from .bispectrum_algebra import all_tri
        l1, l2, L = ell
        lp1, lp2, Lp = ellp
        out = {}
        for L1 in Ls1:
            for L2 in Ls2:
                for L3 in Ls3:
                    for lpp in tri(lp2, L2):
                        amax = self.lam_w + l1 + lp1 + L1
                        bmax = self.lam_w + l2 + L3
                        for a in range(0, amax + 1):
                            for b in range(abs(a - lpp), min(a + lpp, bmax) + 1):
                                if (a + b + lpp) % 2:
                                    continue
                                for Lam in tri_multi((lp1, L1, a)):
                                    for Lamp in tri(L3, b):
                                        for Ly in tri(L3, Lp):
                                            ns = []
                                            for lam1 in tri(l1, Lam):
                                                if lam1 > self.lam_w:
                                                    continue
                                                for lam2 in tri(l2, Lamp):
                                                    if lam2 > self.lam_w:
                                                        continue
                                                    for (K1, K2, Lam3), T in self.coup.t2_all(ell, ellp, L1, L2, L3, lpp, a, b,
                                                                                             Lam, Lamp, Ly, lam1, lam2).items():
                                                        if abs(T) > 1e-15:
                                                            ns.append(((lam1, L1, K1, lam2, Ly, K2, Lam3), T))
                                            if ns:
                                                out[(L1, L2, L3, lpp, a, b, Lam, Lamp, Ly)] = ns
        return out

    def needed_windows2(self, ell, ellp, Ls1, Ls2, Ls3):
        return sorted({n for ns in self.radial_labels2(ell, ellp, Ls1, Ls2, Ls3).values() for (n, _) in ns})

    def assemble2(self, ell, ellp, leg1, anchor, leg2, windows, log=None, n3: int = 10) -> np.ndarray:
        """Class-2 pairing (1 1')(2 3')(3 2'): C[i, j, i', j'] (no prefactor).  leg1 = (tag, {L: pfunc}) is the
        spectrum P(p1; y1^) (shell i'), anchor the spectrum P(p2; x^) inside the vertex (p2 in shell j'), leg2 the
        spectrum P(p3; y2^) of the unbinned third side carried by leg 2 (shell j of the unprimed estimator)."""
        from .bispectrum_algebra import phase2
        K = self.K
        tag1, p1f = leg1
        tag2, p2f = anchor
        tag3, p3f = leg2
        Ls1, Ls2, Ls3 = tuple(sorted(p1f)), tuple(sorted(p2f)), tuple(sorted(p3f))
        labels = self.radial_labels2(ell, ellp, Ls1, Ls2, Ls3)
        nb, nq, nr = K.nbins, K.nq, K.nr
        C = np.zeros((nb, nb, nb, nb))
        jb1 = K.jbar(ell[0]) * K.r_weights[None, :]
        jb2 = K.jbar(ell[1]) * K.r_weights[None, :]
        t0 = time.time()
        for nlab, ((L1, L2, L3, lpp, a, b, Lam, Lamp, Ly), ns) in enumerate(labels.items()):
            Qt = None
            for (n, T) in ns:
                Q = windows(n)
                if Q is None:
                    continue
                Qt = T * Q if Qt is None else Qt + T * Q
            if Qt is None:
                continue
            pref = FOUR_PI ** 3 / ((2 * L1 + 1) * (2 * L2 + 1) * (2 * L3 + 1)) * phase2(ell, Lam, Lamp)
            A = K.leg(p1f[L1], L1, Lam).reshape(nb * nq, nr)                      # [(i', q1), r1]
            X = (np.einsum('ir,Qr->iQr', jb1, A).reshape(nb * nb * nq, nr) @ Qt).reshape(nb, nb, nq, nr)
            for I in range(nb):                                                     # shell i' of p1
                for J in range(nb):                                                 # shell j' of p2 (the indicator)
                    D = K.vertex2(lpp, a, b, p2f[L2], I, J, tag=tag2, n3=n3)        # [q1, q3]
                    if not np.any(D):
                        continue
                    p3, w3 = K.p3_nodes(I, J, n3)
                    P3 = np.ones_like(p3) if p3f[L3] is None else np.asarray(p3f[L3](p3), dtype=float)
                    B3 = (w3 * P3 / K.norm[J])[:, :, None] * K.bessel_at(Lamp, p3)   # [q1, q3, r2]
                    Z = np.einsum('qt,iqr,qtr->ir', D, X[:, I], B3)                  # [i, r2]
                    C[:, :, I, J] += pref * (Z @ jb2.T)
            if log is not None and (nlab % 50 == 0):
                log(f"    class-2 label {nlab + 1}/{len(labels)} ({time.time() - t0:.1f}s)")
        return C

    def box_windows(self, V: float):
        """windows(n) for a periodic box of volume V (bispectrum_algebra.box_window_value, constant in r1, r2)."""
        nr = self.K.nr

        def windows(n):
            h = box_window_value(n, V)
            return None if h == 0.0 else np.full((nr, nr), h)
        return windows


# ---------------------------------------------------------------------------------------------------- user-facing class
class BispectrumGaussianCovariance:
    """Gaussian (class-1) covariance of the bispectrum multipoles B_{l1 l2 L}(k_i, k_j) of one tracer with survey window.

    Parameters
    ----------
    tracer     : Tracer (randoms, weights, nbar); single tracer, as in the derivation
    k_edges    : edges of the k shells (h/Mpc); B(k_i, k_j) is averaged over shells i and j
    multipoles : Sugiyama multipoles (l1, l2, L), e.g. ((0, 0, 0), (2, 0, 2))
    L_max      : highest power-spectrum multipole of the model (even)
    lam_w      : highest rank of the window invariants kept at the two legs (guide, sec. 'Truncation'); 0 is the
                 box-like (isotropic-window) approximation
    r_max, dr  : separation grid of the radial integrals (default r_max: bounding box of the randoms)
    dr_pair    : radial bin width of the anchored pair moments (Q_n is smooth; coarse is fine)
    shot_noise : include the shot-noise window S at the anchor and the legs (eight (W, S) combinations)
    n_anchor, n_sub, n_near, s_split, seed : sampling of the pair sums (bispectrum_windows.AnchoredMoments)
    n_mu, n_quad : quadrature of the vertex coefficients and of the shells

    Usage
    -----
        cov = BispectrumGaussianCovariance(tracer, k_edges, multipoles=((0,0,0), (2,0,2)))
        cov.compute_windows(verbose=True)              # pair sums (geometry), once per survey
        cov.set_model(model)                           # PowerSpectrumModel with P_0, P_2, P_4 of the tracer
        C, labels = cov.covariance()                   # labels: (l1, l2, L, i, j)

    Scope: the two class-1 pairings (1 1')(2 2')(3 3') and (1 2')(2 1')(3 3') (the diagonal to O(dk/k)) and, with
    class2=True, the four class-2 pairings in which a leg of one triangle is the third side of the other (~dk/k on the
    diagonal, the whole Gaussian off-diagonal structure).  The k3 ~ 0 term (`k3_zero_block`) is not implemented; the
    non-Gaussian P B, B B, P T terms are outside this class.  `block4(..., terms=)` returns the classes separately.
    """

    def __init__(self, tracer, k_edges, multipoles=((0, 0, 0), (2, 0, 2)), L_max=4, lam_w=4, r_max=None, dr=4.0,
                 dr_pair=20.0, shot_noise=True, n_anchor=2000, n_sub=5000, n_near=200000, s_split=80.0, seed=0,
                 n_mu=64, n_quad=None, class2=True, n3=10):
        from .bispectrum_windows import ThreePointWindowLibrary
        self.tracer = tracer
        self.class2 = bool(class2)
        self.n3 = int(n3)
        self.k_edges = np.asarray(k_edges, dtype=float)
        self.nbins = len(self.k_edges) - 1
        self.multipoles = tuple(tuple(int(x) for x in m) for m in multipoles)
        self.Ls = tuple(range(0, int(L_max) + 1, 2))
        self.lam_w = int(lam_w)
        self.shot_noise = bool(shot_noise)
        if r_max is None:
            r_max = tracer.s_max()
        self.r_max = float(r_max)
        self.r = np.arange(0.5 * dr, self.r_max, dr)
        r_edges = np.arange(0.0, self.r_max + dr_pair, dr_pair)
        self.kernels = BispectrumShellKernels(self.k_edges, self.r, n_quad=n_quad, n_mu=n_mu)
        self.coupling = BispectrumCoupling()
        self.assembler = Class1Assembler(self.kernels, self.coupling, lam_w=self.lam_w)
        self.windows = ThreePointWindowLibrary(r_edges, n_anchor=n_anchor, n_sub=n_sub, n_near=n_near,
                                               s_split=s_split, seed=seed)
        self.model = None
        self._I3 = None
        self._blocks = {}

    # ------------------------------------------------------------------------------------------ normalisation
    def I3(self) -> float:
        """I_3 = int omega^3 = alpha sum_r w_r m_r^2 from the randoms, unless set_normalization was used."""
        if self._I3 is None:
            T = self.tracer
            self._I3 = T.alpha * float(np.sum(T.w * T.mw ** 2))
        return self._I3

    def set_normalization(self, I3: float):
        """Use the normalisation the bispectrum estimator actually divided by (the covariance scales as 1 / I_3^2)."""
        self._I3 = float(I3)
        self._blocks = {}
        return self

    # ------------------------------------------------------------------------------------------ correlator specs
    def _kinds(self):
        return ('W', 'S') if self.shot_noise else ('W',)

    def _spec(self, kind):
        """(tag, {L: pfunc}) of a correlator with window kind 'W' (model multipoles) or 'S' (shot noise, P -> 1)."""
        if kind == 'S':
            return ('S', {0: None})
        if self.model is None:
            return ('P', {L: None for L in self.Ls})
        A = self.tracer.name
        avail = self.model.multipoles(A, A)
        return ('P', {L: extrapolated_power(self.model, A, A, L) for L in self.Ls if L in avail})

    def _ellp_pairings(self, ellp):
        """The primed multipole as seen by the two class-1 pairings: (ell', transpose primed shells?)."""
        sw = (ellp[1], ellp[0], ellp[2])
        return [(ellp, False), (sw, True)]

    @staticmethod
    def _class2_pairings(ell, ellp):
        """The four class-2 pairings as relabellings of (1 1')(2 3')(3 2'): (ell_eff, ell'_eff, swap unprimed shells,
        swap primed shells).  Swapping the primed labels (p1, l'1, i') <-> (p2, l'2, j') gives (1 2')(2 3')(3 1'); swapping
        the unprimed ones (q1, l1, i, r1) <-> (q2, l2, j, r2) gives (1 3')(2 1')(3 2'); both give (1 3')(2 2')(3 1')."""
        ell_sw = (ell[1], ell[0], ell[2])
        ellp_sw = (ellp[1], ellp[0], ellp[2])
        return [(ell, ellp, False, False), (ell, ellp_sw, False, True), (ell_sw, ellp, True, False), (ell_sw, ellp_sw, True, True)]

    # ------------------------------------------------------------------------------------------ geometry
    def request_windows(self):
        Ls = lambda k: (0,) if k == 'S' else tuple(self._spec('W')[1])  # noqa: E731
        for ell in self.multipoles:
            for ellp in self.multipoles:
                for ka in self._kinds():
                    for k1 in self._kinds():
                        for k2 in self._kinds():
                            ns = set()
                            for ellp_eff, _ in self._ellp_pairings(ellp):
                                ns |= set(self.assembler.needed_windows(ell, ellp_eff, Ls(k1), Ls(k2), Ls(ka)))
                            if self.class2:
                                for ell_e, ellp_e, _, _ in self._class2_pairings(ell, ellp):
                                    ns |= set(self.assembler.needed_windows2(ell_e, ellp_e, Ls(k1), Ls(ka), Ls(k2)))
                            self.windows.request(self.tracer, (ka, k1, k2), ns)
        return self

    def compute_windows(self, verbose=False):
        """Anchored pair sums for every invariant the requested multipoles need."""
        self.request_windows()
        t0 = time.time()
        self.windows.compute_all(verbose=verbose)
        if verbose:
            print(f"three-point window functions done in {time.time() - t0:.1f}s")
        self._blocks = {}
        return self

    def save_windows(self, path):
        self.windows.save(path)

    def load_windows(self, path):
        self.windows.load(path)
        self._blocks = {}
        return self

    # ------------------------------------------------------------------------------------------ model
    def set_model(self, model: PowerSpectrumModel):
        """Model multipoles P_L(k) of the tracer (unmasked; tabulated over the whole range the shells reach, the third
        side |p1 + p2| being extrapolated below k_min as a power law)."""
        self.model = model
        self.kernels._vertex.clear()
        self._blocks = {}
        return self

    # ------------------------------------------------------------------------------------------ results
    def _windows_for(self, kinds):
        name, r = self.tracer.name, self.r
        return lambda n: self.windows.get(name, kinds, n, r)

    def block4(self, ell, ellp, verbose=False, terms=None) -> np.ndarray:
        """Cov[B_ell(k_i, k_j), B_ell'(k_i', k_j')] as the (nbins,)*4 array C[i, j, i', j'] summed over the requested
        Gaussian terms (default: class 1 and, if enabled, class 2; terms=('class1',) or ('class2',) for one of them),
        all window combinations.  The power spectra are evaluated at the primed momenta; see `block` for the
        symmetrised combination."""
        ell, ellp = tuple(ell), tuple(ellp)
        if terms is None:
            terms = ('class1', 'class2') if self.class2 else ('class1',)
        terms = tuple(terms)
        key = (ell, ellp, terms)
        if key in self._blocks:
            return self._blocks[key]
        if self.model is None:
            raise RuntimeError("set_model() first")
        self.request_windows()
        if self.windows.missing():
            self.windows.compute_all(verbose=verbose)
        log = print if verbose else None
        nb = self.nbins
        C = np.zeros((nb, nb, nb, nb))
        kinds = [(ka, k1, k2) for ka in self._kinds() for k1 in self._kinds() for k2 in self._kinds()]
        if 'class1' in terms:
            for ellp_eff, swap in self._ellp_pairings(ellp):
                pref = FOUR_PI ** 3 * H(*ell) * H(*ellp_eff) * math.sqrt(N(*ell) * N(*ellp_eff))
                for (ka, k1, k2) in kinds:
                    if verbose:
                        print(f"  class 1 {ell} x {ellp_eff}{' (swapped)' if swap else ''}: anchor {ka}, legs {k1} {k2}")
                    part = self.assembler.assemble(ell, ellp_eff, self._spec(k1), self._spec(k2), self._spec(ka),
                                                   self._windows_for((ka, k1, k2)), log=log)
                    C += pref * (part.transpose(0, 1, 3, 2) if swap else part)
        if 'class2' in terms:
            for ell_e, ellp_e, swap_u, swap_p in self._class2_pairings(ell, ellp):
                pref = FOUR_PI ** 3 * H(*ell_e) * H(*ellp_e) * math.sqrt(N(*ell_e) * N(*ellp_e))
                for (ka, k1, k2) in kinds:
                    if verbose:
                        print(f"  class 2 {ell_e} x {ellp_e} (swaps u={swap_u} p={swap_p}): anchor {ka}, legs {k1} {k2}")
                    part = self.assembler.assemble2(ell_e, ellp_e, self._spec(k1), self._spec(ka), self._spec(k2),
                                                    self._windows_for((ka, k1, k2)), log=log, n3=self.n3)
                    if swap_p:
                        part = part.transpose(0, 1, 3, 2)
                    if swap_u:
                        part = part.transpose(1, 0, 2, 3)
                    C += pref * part
        C /= self.I3() ** 2
        self._blocks[key] = C
        return C

    @staticmethod
    def default_elements(ell, nbins):
        """(i, j) pairs of B_ell(k_i, k_j): i <= j when l1 = l2 (the estimator is then symmetric), all pairs otherwise."""
        if ell[0] == ell[1]:
            return [(i, j) for i in range(nbins) for j in range(i, nbins)]
        return [(i, j) for i in range(nbins) for j in range(nbins)]

    def block(self, ell, ellp, elements=None, elements_p=None, symmetrize=True, terms=None) -> np.ndarray:
        """Covariance block between the elements (i, j) of B_ell and (i', j') of B_ell'.

        The derivation evaluates the power spectra at the momenta of the primed estimator; the other choice differs
        at O(dk/k) and symmetrize=True returns the average of the two (the direct block and the transposed mirror),
        as GaussianCovariance.block does.  For l1 = l2 the (i, j) and (j, i) elements are the same measurement and the
        class-1 pairings make C symmetric under i <-> j by construction.
        """
        ell, ellp = tuple(ell), tuple(ellp)
        if elements is None:
            elements = self.default_elements(ell, self.nbins)
        if elements_p is None:
            elements_p = self.default_elements(ellp, self.nbins)
        if symmetrize:
            direct = self.block(ell, ellp, elements, elements_p, symmetrize=False, terms=terms)
            mirror = self.block(ellp, ell, elements_p, elements, symmetrize=False, terms=terms)
            return 0.5 * (direct + mirror.T)
        C4 = self.block4(ell, ellp, terms=terms)
        ii, jj = np.array([e[0] for e in elements]), np.array([e[1] for e in elements])
        Ii, Jj = np.array([e[0] for e in elements_p]), np.array([e[1] for e in elements_p])
        return C4[ii[:, None], jj[:, None], Ii[None, :], Jj[None, :]]

    def covariance(self, multipoles=None, elements=None, symmetrize=True, verbose=False, terms=None):
        """Full matrix for the data vector [B_ell(k_i, k_j)] ordered by multipole then element; returns (C, labels)
        with labels (l1, l2, L, i, j).  `elements` maps ell -> list of (i, j) (default: default_elements)."""
        if multipoles is None:
            multipoles = self.multipoles
        multipoles = [tuple(m) for m in multipoles]
        elements = elements or {}
        els = {m: elements.get(m, self.default_elements(m, self.nbins)) for m in multipoles}
        sizes = [len(els[m]) for m in multipoles]
        offs = np.concatenate([[0], np.cumsum(sizes)])
        C = np.zeros((offs[-1], offs[-1]))
        t0 = time.time()
        for a, ma in enumerate(multipoles):
            for b, mb in enumerate(multipoles):
                C[offs[a]:offs[a + 1], offs[b]:offs[b + 1]] = self.block(ma, mb, els[ma], els[mb], symmetrize=False, terms=terms)
                if verbose:
                    print(f"block {ma} x {mb} done ({time.time() - t0:.1f}s)")
        if symmetrize:
            C = 0.5 * (C + C.T)
        labels = [m + e for m in multipoles for e in els[m]]
        return C, labels

    # ------------------------------------------------------------------------------------------ not implemented
    def k3_zero_block(self, ell, ellp):
        """The pairing (1 2)(1' 2')(3 3') on the (i i; i' i') elements: the long-mode variance of the W-weighted field
        (guide sec. 'The k3 ~ 0 term'); zero for uniform omega, tens of per cent of the (i, i) variance for a radially
        varying omega.  Needs <F(x) F(x')>_IC, i.e. the integral-constraint-corrected long-mode correlation, which the
        super-sample machinery (thecov.ssc) can provide."""
        raise NotImplementedError("the k3 ~ 0 term is not implemented yet; see the guide sec. 'The k3 ~ 0 term'")
