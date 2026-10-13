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
        self._vertex = {}

    def bessel(self, lam: int) -> np.ndarray:
        """j_lam(k_q r) for all nodes: (nbins, nq, nr)."""
        if lam not in self._bessel:
            self._bessel[lam] = spherical_jn(lam, self.kq[:, :, None] * self.r[None, None, :])
        return self._bessel[lam]

    def jbar(self, ell: int) -> np.ndarray:
        """Bin-averaged jbar^{(i)}_ell(r): (nbins, nr)."""
        return np.einsum('iq,iqr->ir', self.wq, self.bessel(ell)) / self.norm[:, None]

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


def vertex_coefficients(L3: int, a: int, b: int, p1, p2, P_L3, n_quad: int = 64, chunk: int = 4000) -> np.ndarray:
    """Vectorised c^{L3}_{ab}(p1, p2) for flat arrays p1, p2 (same formula as bispectrum_algebra.bipolar_coefficients)."""
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

    def box_windows(self, V: float):
        """windows(n) for a periodic box of volume V (bispectrum_algebra.box_window_value, constant in r1, r2)."""
        nr = self.K.nr

        def windows(n):
            h = box_window_value(n, V)
            return None if h == 0.0 else np.full((nr, nr), h)
        return windows
