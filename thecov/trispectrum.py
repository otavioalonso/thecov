r"""Tree-level connected trispectrum (T0) covariance of power-spectrum multipoles, in redshift space.

Same decomposition as Kobayashi (2023, PowerSpecCovFFT):

  C^{T0}_{l1 l2}(k1, k2) = J_4 (2 l1 + 1)(2 l2 + 1) < L_l1(k1^.n^) L_l2(k2^.n^) T(k1, -k1, k2, -k2) >,

  T = T_2211 ("snake", P(k_a) P(k_b) P(|k_a + k_c|) Z1 Z1 Z2 Z2) + T_3111 ("star", P P P Z1 Z1 Z1 Z3),

with the average over the line of sight n^ (local, both spectra) and over the orientations of k1^, k2^;
J_4 = int m^4 / norm^2 (m = nbar w; = 1/V for a uniform window, Kobayashi's 1/V_eff = I44 / I22^2). The
kernels Z1, Z2, Z3 are the SCF99 redshift-space kernels of a biased tracer, built from the matter kernels
F_n, G_n (standard recursion) and the galaxy density kernels in the Galileon basis

  delta_g = b1 delta + b2/2 delta^2 + bG2 G2(Phi) + b3/6 delta^3 + bG3 G3(Phi) + bdG2 delta G2(Phi) + bGamma3 Gamma3,

through the exact mapping delta_s(k) = sum_m (f k_z)^m / m! prod (q_z / q^2) theta(q) * (1 + delta_g).
Convert from the (b2, bs2) basis of thecov.ssc (delta_g contains b2/2 delta^2 + bs2/2 s^2) with
`galileon_bias`.

Differences from PowerSpecCovFFT (the fixes):
  * every block (l1, l2), including l1 > l2, is computed from its own angular average, so the matrix is
    symmetric by construction, C_{l1 l2}(k1, k2) = C_{l2 l1}(k2, k1). PowerSpecCovFFT stores coefficient
    functions for l1 <= l2 only and obtains the k1 <-> k2 swapped terms by swapping k without swapping l,
    which is wrong for l1 != l2;
  * the angular averages are numerical but exact: at fixed mu12 = k1^.k2^ the integrand is a polynomial
    in the components of n^, integrated exactly by Gauss-Legendre in mu1 = k1^.n^ and a uniform rule in
    the azimuth of k2^ around k1^ (`n_mu`, `n_psi`); this replaces the Mathematica coefficient files and
    covers any bias / any l;
  * the mu12 integral (Kobayashi: FFTLog + analytic master integrals for the P(k12) terms, closed forms with
    log((k1+k2)/|k1-k2|) for the star terms) is a Gauss-Legendre rule clustered logarithmically at
    mu12 = +-1, where the integrands carry the integrable singularities |k1 -+ k2|^-2 (snake P(k12) / k12^2,
    star log terms); it resolves the BAO in P(k12) with `n_mid` interior nodes;
  * the configurations with k_a + k_c = 0 exactly (P(0), the beat-coupling / super-sample piece) are
    excluded; they are the SuperSampleCovariance's;
  * k-bin averages (Gauss-Legendre in |k|, `n_k` nodes per bin, k^2-weighted).

The snake and star parts are returned separately (`components`, with d star / d b3) so that their amplitudes
can be fitted (e.g. to subvolumes of the data) with thecov.CovarianceTemplates.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.interpolate import CubicSpline

_TINY = 1e-24


# ----------------------------------------------------------------------------- vector helpers
class _Gram:
    """Momenta as linear combinations of a few basis vectors: every dot product and z-component is a
    combination of the basis' Gram matrix and z-components, computed once and cached (the kernels below
    take either plain (3, ...) arrays or _Vec's of one _Gram)."""

    def __init__(self, *vectors):
        self.n = len(vectors)
        self.g = [[np.sum(a * b, 0) for b in vectors] for a in vectors]
        self.zc = [v[2] for v in vectors]
        self._dot, self._z, self._inv2 = {}, {}, {}

    def basis(self):
        return [_Vec(self, tuple(float(i == j) for j in range(self.n))) for i in range(self.n)]

    def dot(self, c1, c2):
        key = (c1, c2) if c1 <= c2 else (c2, c1)
        if key not in self._dot:
            out = 0.0
            for i, a in enumerate(c1):
                if a:
                    for j, b in enumerate(c2):
                        if b:
                            out = out + (a * b) * self.g[i][j]
            self._dot[key] = out
        return self._dot[key]

    def z(self, c):
        if c not in self._z:
            self._z[c] = sum((a * z for a, z in zip(c, self.zc) if a), 0.0)
        return self._z[c]


class _Vec:
    __slots__ = ('ctx', 'c')

    def __init__(self, ctx, c):
        self.ctx, self.c = ctx, c

    def __add__(self, o):
        return _Vec(self.ctx, tuple(a + b for a, b in zip(self.c, o.c)))

    def __sub__(self, o):
        return _Vec(self.ctx, tuple(a - b for a, b in zip(self.c, o.c)))

    def __neg__(self):
        return _Vec(self.ctx, tuple(-a for a in self.c))

    def __mul__(self, x):
        return _Vec(self.ctx, tuple(x * a for a in self.c))

    __rmul__ = __mul__

    def __getitem__(self, i):
        if i != 2:
            raise IndexError('only the z (line-of-sight) component is available')
        return self.ctx.z(self.c)


def _dot(a, b):
    if isinstance(a, _Vec):
        return a.ctx.dot(a.c, b.c)
    return np.sum(a * b, 0)


def _inv(x):
    with np.errstate(divide='ignore'):
        return np.where(x > _TINY, 1.0 / np.where(x > _TINY, x, 1.0), 0.0)


def _inv2(a):
    """1 / |a|^2 (0 for a = 0), cached for _Vec's"""
    if isinstance(a, _Vec):
        ctx, key = a.ctx, a.c
        if key not in ctx._inv2:
            ctx._inv2[key] = _inv(ctx.dot(key, key))
        return ctx._inv2[key]
    return _inv(_dot(a, a))


def _sig2(a, b):
    """Fourier kernel of G2(Phi): (a^.b^)^2 - 1 (0 if a vector vanishes, where it is multiplied by 0)"""
    ab = _dot(a, b)
    return ab * ab * _inv2(a) * _inv2(b) - 1.0


def _alpha(a, b):
    return _dot(a + b, a) * _inv2(a)


def _beta(a, b):
    s = a + b
    return _dot(s, s) * _dot(a, b) * _inv2(a) * _inv2(b) / 2


# ----------------------------------------------------------------------------- matter kernels
def F2(a, b):
    return 5 / 7 + 0.5 * _dot(a, b) * (_inv2(a) + _inv2(b)) + 2 / 7 * (_sig2(a, b) + 1)


def G2(a, b):
    return 3 / 7 + 0.5 * _dot(a, b) * (_inv2(a) + _inv2(b)) + 4 / 7 * (_sig2(a, b) + 1)


def _FG3_unsym(q1, q2, q3):
    """unsymmetrised F3, G3 from the recursion (n = 3)"""
    k12, k23 = q1 + q2, q2 + q3
    # m = 1: G1(q1) [ .. F2(q2, q3), G2(q2, q3) ];  m = 2: G2(q1, q2) [ .. F1(q3), G1(q3) ]
    f23, g23, g12 = F2(q2, q3), G2(q2, q3), G2(q1, q2)
    a1, b1_ = _alpha(q1, k23), _beta(q1, k23)
    a2, b2_ = _alpha(k12, q3), _beta(k12, q3)
    F = (7 * a1 * f23 + 2 * b1_ * g23 + g12 * (7 * a2 + 2 * b2_)) / 18
    G = (3 * a1 * f23 + 6 * b1_ * g23 + g12 * (3 * a2 + 6 * b2_)) / 18
    return F, G


_PERMS3 = list(itertools.permutations(range(3)))


def FG3(q1, q2, q3):
    """symmetrised F3, G3"""
    qs = (q1, q2, q3)
    F = G = 0.0
    for p in _PERMS3:
        f, g = _FG3_unsym(*(qs[i] for i in p))
        F, G = F + f, G + g
    return F / 6, G / 6


# ----------------------------------------------------------------------------- galaxy kernels (Galileon basis)
class Bias:
    """Galileon-basis bias parameters (b1, b2, bG2, b3, bG3, bdG2, bGamma3)."""

    def __init__(self, b1, b2=0.0, bG2=None, b3=0.0, bG3=0.0, bdG2=0.0, bGamma3=None):
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.bG2 = -2 / 7 * (self.b1 - 1) if bG2 is None else float(bG2)          # local Lagrangian
        self.b3, self.bG3, self.bdG2 = float(b3), float(bG3), float(bdG2)
        self.bGamma3 = 23 / 42 * (self.b1 - 1) if bGamma3 is None else float(bGamma3)

    def as_dict(self):
        return dict(b1=self.b1, b2=self.b2, bG2=self.bG2, b3=self.b3, bG3=self.bG3, bdG2=self.bdG2, bGamma3=self.bGamma3)

    def __repr__(self):
        return 'Bias(' + ', '.join(f'{k}={v:.4g}' for k, v in self.as_dict().items()) + ')'


def galileon_bias(b1, b2=0.0, bs2=None, **third):
    """Bias from the (b2, bs2) basis of thecov.ssc (b2/2 delta^2 + bs2/2 s^2): bG2 = bs2/2, b2_G = b2 + 2/3 bs2."""
    bs2 = -4 / 7 * (b1 - 1) if bs2 is None else bs2
    return Bias(b1, b2 + 2 / 3 * bs2, bs2 / 2, **third)


def Fg2(a, b, bias):
    return bias.b1 * F2(a, b) + bias.b2 / 2 + bias.bG2 * _sig2(a, b)


def _G3gal(q1, q2, q3):
    m12 = _dot(q1, q2) * np.sqrt(_inv2(q1) * _inv2(q2))
    m23 = _dot(q2, q3) * np.sqrt(_inv2(q2) * _inv2(q3))
    m31 = _dot(q3, q1) * np.sqrt(_inv2(q3) * _inv2(q1))
    return -0.5 * (2 * m12 * m23 * m31 + 1 - m12 ** 2 - m23 ** 2 - m31 ** 2)


def Fg3(q1, q2, q3, bias, F3=None):
    """symmetrised third-order galaxy density kernel (F3: the symmetric matter F3, if already computed)"""
    if F3 is None:
        F3 = FG3(q1, q2, q3)[0]
    qs = (q1, q2, q3)
    out = 0.0
    for p in _PERMS3:
        a, b, c = (qs[i] for i in p)
        bc = b + c
        f2, g2 = F2(b, c), G2(b, c)
        s = _sig2(a, bc)
        out = out + (bias.b2 * f2 + 2 * bias.bG2 * s * f2 + bias.bdG2 * _sig2(b, c)
                     + 2 * bias.bGamma3 * s * (f2 - g2))
    return bias.b1 * F3 + out / 6 + bias.b3 / 6 + bias.bG3 * _G3gal(q1, q2, q3)


# ----------------------------------------------------------------------------- redshift-space kernels (n^ = z^)
def Z1(q, bias, f):
    return bias.b1 + f * q[2] ** 2 * _inv2(q)


def _uz(q):
    """q_z / q^2 (0 for q = 0)"""
    return q[2] * _inv2(q)


def Z2(q1, q2, bias, f):
    k = q1 + q2
    fk = f * k[2]
    return (Fg2(q1, q2, bias) + fk * _uz(k) * G2(q1, q2) + 0.5 * fk * bias.b1 * (_uz(q1) + _uz(q2))
            + 0.5 * fk ** 2 * _uz(q1) * _uz(q2))


def Z3(q1, q2, q3, bias, f):
    """symmetrised third-order redshift-space kernel"""
    F3, G3 = FG3(q1, q2, q3)
    k = q1 + q2 + q3
    fk = f * k[2]
    out = Fg3(q1, q2, q3, bias, F3=F3) + fk * _uz(k) * G3
    qs = (q1, q2, q3)
    acc = 0.0
    for p in _PERMS3:
        a, b, c = (qs[i] for i in p)
        ua, ub, uc = _uz(a), _uz(b), _uz(c)
        u_ab, u_bc = _uz(a + b), _uz(b + c)
        g_ab, g_bc = G2(a, b), G2(b, c)
        acc = acc + (fk * (u_ab * g_ab * bias.b1 + ua * Fg2(b, c, bias))
                     + fk ** 2 / 2 * (ua * ub * bias.b1 + u_ab * g_ab * uc + ua * u_bc * g_bc)
                     + fk ** 3 / 6 * ua * ub * uc)
    return out + acc / 6


# ----------------------------------------------------------------------------- trispectrum
def trispectrum(ks, P, bias, f, parts=('snake', 'star')):
    """Tree-level T(ks[0], ks[1], ks[2], ks[3]) (vectors (3, ...), sum 0; n^ = z^), split into parts.

    P: callable P_lin(|k|) (0 at k = 0, so the k_a + k_c = 0 terms vanish)."""
    mags = [np.sqrt(_dot(k, k)) for k in ks]
    Pk = [P(m) for m in mags]
    z1 = [Z1(k, bias, f) for k in ks]
    out = {}
    if 'snake' in parts:
        T = 0.0
        for a, b in itertools.combinations(range(4), 2):
            c, d = (i for i in range(4) if i not in (a, b))
            pre = 4 * z1[a] * z1[b] * Pk[a] * Pk[b]
            for cc, dd in ((c, d), (d, c)):
                s = ks[a] + ks[cc]
                Ps = P(np.sqrt(_dot(s, s)))
                T = T + pre * Ps * Z2(-ks[a], s, bias, f) * Z2(-ks[b], ks[b] + ks[dd], bias, f)
        out['snake'] = T
    if 'star' in parts:
        T = 0.0
        for d in range(4):
            a, b, c = (i for i in range(4) if i != d)
            T = T + 6 * Z3(-ks[a], -ks[b], -ks[c], bias, f) * z1[a] * z1[b] * z1[c] * Pk[a] * Pk[b] * Pk[c]
        out['star'] = T
    return out


def parallelogram(k1, k2, P, bias, f, parts=('snake', 'star', 'star_b3')):
    """T(k1, -k1, k2, -k2) = trispectrum([k1, -k1, k2, -k2]) using its symmetries (kernels are even under
    q -> -q of all arguments; the P(0) terms dropped): 6 snake terms instead of 24, 2 Z3 instead of 4."""
    if not isinstance(k1, _Vec):
        k1, k2 = _Gram(k1, k2).basis()
    m1, m2 = np.sqrt(_dot(k1, k1)), np.sqrt(_dot(k2, k2))
    P1, P2 = P(m1), P(m2)
    za, zb = Z1(k1, bias, f), Z1(k2, bias, f)
    out = {}
    if 'snake' in parts:
        sp, sm = k1 + k2, k1 - k2
        Pp, Pm = P(np.sqrt(np.maximum(_dot(sp, sp), 0))), P(np.sqrt(np.maximum(_dot(sm, sm), 0)))
        z_1p, z_1m = Z2(-k1, sp, bias, f), Z2(-k1, sm, bias, f)            # Z2(-k1, k1 +- k2)
        z_2p, z_2m = Z2(-k2, sp, bias, f), Z2(k2, sm, bias, f)              # Z2(-k2, k1 + k2), Z2(k2, k1 - k2)
        z_1p_, z_1m_ = Z2(k1, -sp, bias, f), Z2(k1, -sm, bias, f)          # = Z2(-k1, sp), Z2(-k1, sm) (even)
        # pair (k1, -k1): c = k2: s = k1 + k2, Z2(-k1, k1 + k2) Z2(k1, -k1 - k2); c = -k2: s = k1 - k2
        T = 4 * za ** 2 * P1 ** 2 * (Pp * z_1p * z_1p_ + Pm * z_1m * z_1m_)
        # pair (k2, -k2): s = k2 + k1 and k2 - k1
        z_2p_, z_2m_ = Z2(k2, -sp, bias, f), Z2(-k2, -sm, bias, f)
        T = T + 4 * zb ** 2 * P2 ** 2 * (Pp * z_2p * z_2p_ + Pm * z_2m * z_2m_)
        # pairs (k1, k2) and (-k1, -k2) [s = k1 - k2], (k1, -k2) and (-k1, k2) [s = k1 + k2]
        T = T + 2 * 4 * za * zb * P1 * P2 * (Pm * Z2(-k1, sm, bias, f) * Z2(-k2, -sm, bias, f)
                                             + Pp * Z2(-k1, sp, bias, f) * Z2(k2, -sp, bias, f))
        out['snake'] = T
    if 'star' in parts:
        out['star'] = 2 * 6 * (Z3(k1, -k1, k2, bias, f) * za ** 2 * zb * P1 ** 2 * P2
                               + Z3(k2, -k2, k1, bias, f) * zb ** 2 * za * P2 ** 2 * P1)
    if 'star_b3' in parts:                                                  # d(star) / d(b3): Z3 contains b3 / 6
        out['star_b3'] = 2 * (za ** 2 * zb * P1 ** 2 * P2 + zb ** 2 * za * P2 ** 2 * P1)
    return out


def mu12_nodes(n_mid=64, n_end=32, delta=0.05, eps=1e-12):
    """(mu, weight) for int_{-1}^{1} dmu / 2: Gauss-Legendre on [-1 + delta, 1 - delta] plus Gauss-Legendre
    in log(1 -+ mu) on the end segments (integrable singularities at mu = +-1)."""
    x, w = np.polynomial.legendre.leggauss(n_mid)
    mu = [(1 - delta) * x]
    wt = [(1 - delta) * w]
    t, wt_ = np.polynomial.legendre.leggauss(n_end)
    lo, hi = np.log(eps), np.log(delta)
    u = np.exp(0.5 * (hi - lo) * t + 0.5 * (hi + lo))                    # 1 -+ mu in [eps, delta]
    wu = 0.5 * (hi - lo) * wt_ * u
    mu += [1 - u, -1 + u]
    wt += [wu, wu]
    return np.concatenate(mu), np.concatenate(wt) / 2


def multipoles(k1, k2, P, bias, f, ells=(0, 2, 4), n_mu=12, n_psi=24, mu12=None, parts=('snake', 'star', 'star_b3'),
               sigma_fog=0.0):
    """{part: {(l1, l2): (2 l1 + 1)(2 l2 + 1) < L_l1(mu1) L_l2(mu2) T(k1, -k1, k2, -k2) >}} at |k1|, |k2|.

    sigma_fog: Gaussian fingers-of-God damping of the four external fields, T -> T exp(-(k1 mu1 s)^2 - (k2 mu2 s)^2)
    (phenomenological; the angular rule is then no longer exact: raise n_mu, n_psi)."""
    mu, wmu = mu12_nodes() if mu12 is None else mu12
    xg, wg = np.polynomial.legendre.leggauss(n_mu)
    psi = 2 * np.pi * (np.arange(n_psi) + 0.5) / n_psi
    M, C1, PS = np.meshgrid(mu, xg, psi, indexing='ij')
    W = np.meshgrid(wmu, wg / 2, np.full(n_psi, 1 / n_psi), indexing='ij')
    W = W[0] * W[1] * W[2]
    S1 = np.sqrt(1 - C1 ** 2)
    SM = np.sqrt(np.clip(1 - M ** 2, 0, None))
    e1 = np.stack([S1, np.zeros_like(S1), C1])                       # k1^ (n^ = z^)
    ea = np.stack([C1, np.zeros_like(C1), -S1])                      # orthonormal to k1^ in the x-z plane
    eb = np.stack([np.zeros_like(C1), np.ones_like(C1), np.zeros_like(C1)])
    e2 = M * e1 + SM * (np.cos(PS) * ea + np.sin(PS) * eb)          # k2^ at angle mu12 from k1^
    mu1, mu2 = C1, e2[2]
    T = parallelogram(k1 * e1, k2 * e2, P, bias, f, parts=parts)
    if sigma_fog:
        D = np.exp(-(k1 * mu1 * sigma_fog) ** 2 - (k2 * mu2 * sigma_fog) ** 2)
        T = {p: t * D for p, t in T.items()}
    L = {l: (np.polynomial.legendre.Legendre.basis(l)(mu1), np.polynomial.legendre.Legendre.basis(l)(mu2)) for l in ells}
    return {part: {(l1, l2): (2 * l1 + 1) * (2 * l2 + 1) * float(np.sum(W * L[l1][0] * L[l2][1] * Tp))
                   for l1 in ells for l2 in ells} for part, Tp in T.items()}


# ----------------------------------------------------------------------------- covariance
class LinearPower:
    """P_lin(k) from a table (cubic spline in ln k; 0 outside the table, in particular at k = 0); picklable"""

    def __init__(self, k, P):
        k, P = np.asarray(k, float), np.asarray(P, float)
        self._spl = CubicSpline(np.log(k), P)
        self.kmin, self.kmax = k[0], k[-1]

    def __call__(self, k):
        k = np.asarray(k, float)
        out = np.zeros_like(k)
        ok = (k > self.kmin) & (k < self.kmax)
        out[ok] = self._spl(np.log(k[ok]))
        return out


def _row(args):
    """multipoles of all pairs (bin i, bin j >= i), bin-averaged: [{part: {(l1, l2): value}}] (one per j)"""
    i, kn, kw, P, bias, f, ells, n_mu, n_psi, mu12, sigma_fog = args
    rows = []
    for j in range(i, len(kn)):
        acc = {}
        for wa, ka in zip(kw[i], kn[i]):
            for wb, kb in zip(kw[j], kn[j]):
                m = multipoles(ka, kb, P, bias, f, ells=ells, n_mu=n_mu, n_psi=n_psi, mu12=mu12, sigma_fog=sigma_fog)
                for p, d in m.items():
                    a = acc.setdefault(p, {})
                    for key, v in d.items():
                        a[key] = a.get(key, 0.0) + wa * wb * v
        rows.append(acc)
    return rows


class TrispectrumCovariance:
    """Tree-level T0 covariance for the set-up of a GaussianCovariance (auto-spectra of one tracer).

    Parameters: cov (k bins, normalisation, tracers), p_lin = (k, P) linear matter power at z_eff, bias
    (a Bias, or b1 with the other Galileon parameters as keywords), f; quadrature n_mu, n_psi (line of
    sight; exact for n_mu >= 11, n_psi >= 21 up to l = 4), n_mid, n_end (mu12; relative error ~3e-7 at the
    defaults), n_k (|k| nodes per bin); sigma_fog: Gaussian FoG damping of the external fields [Mpc/h]; J4: override of int m^4 / norm^2 (e.g. 1/V); n_workers: processes
    (rows of the k x k matrix in parallel; ~0.12 s per pair of k nodes and process).
    """

    def __init__(self, cov, p_lin, bias, f, n_mu=12, n_psi=24, n_mid=64, n_end=32, n_k=1, J4=None, n_workers=1,
                 sigma_fog=0.0, **bias_kw):
        self.cov = cov
        self.P = LinearPower(*p_lin)
        self.bias = bias if isinstance(bias, Bias) else Bias(bias, **bias_kw)
        self.f = float(f)
        self.n_mu, self.n_psi, self.n_k = int(n_mu), int(n_psi), int(n_k)
        self.mu12 = mu12_nodes(n_mid=n_mid, n_end=n_end)
        self._J4 = J4
        self.n_workers = int(n_workers)
        self.sigma_fog = float(sigma_fog)

    def window_integral(self, A):
        """J4 = int m^4 / norm^2 = alpha sum_r w_r m_r^3 / norm^2 (m = nbar w at the randoms)"""
        if self._J4 is not None:
            return float(self._J4)
        T = self.cov._tracer(A)
        w, m = np.asarray(T.w, float), np.asarray(T.mw, float)
        return T.alpha * float(np.sum(w * m ** 3)) / self.cov.I(A, A) ** 2

    def _nodes(self):
        edges = self.cov.k_edges
        xk, wk = np.polynomial.legendre.leggauss(self.n_k)
        kn, kw = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            kk = 0.5 * (hi - lo) * xk + 0.5 * (hi + lo)
            ww = wk * kk ** 2
            kn.append(kk)
            kw.append(ww / ww.sum())
        return kn, kw

    def components(self, spectra, ells=None, verbose=False):
        """{'snake': C, 'star': C, 'star_b3': d C_star / d b3} for [P^{AA}_ell(k_i)] ordered by ell, bin (as
        GaussianCovariance). The covariance is snake + star; star_b3 (linear in b3, the least known bias) is
        for varying or fitting b3."""
        cov = self.cov
        ells = cov.ells if ells is None else tuple(ells)
        spectra = [tuple(str(x) for x in sp) for sp in spectra]
        if len(spectra) != 1 or spectra[0][0] != spectra[0][1]:
            raise NotImplementedError('T0: auto-spectrum of one tracer')
        A = spectra[0][0]
        J4 = self.window_integral(A)
        kn, kw = self._nodes()
        nb, nl = len(kn), len(ells)
        tasks = [(i, kn, kw, self.P, self.bias, self.f, ells, self.n_mu, self.n_psi, self.mu12, self.sigma_fog)
                 for i in range(nb)]
        if self.n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp
            with ProcessPoolExecutor(self.n_workers, mp_context=mp.get_context('spawn')) as ex:
                rows = list(ex.map(_row, tasks[::-1]))[::-1]           # long rows first
        else:
            rows = []
            for t in tasks:
                if verbose:
                    print(f'T0: bin {t[0] + 1}/{nb}', flush=True)
                rows.append(_row(t))
        out = {p: np.zeros((nl * nb, nl * nb)) for p in ('snake', 'star', 'star_b3')}
        for i, row in enumerate(rows):
            for dj, acc in enumerate(row):
                j = i + dj
                for p in out:
                    for a, l1 in enumerate(ells):
                        for b, l2 in enumerate(ells):
                            v = J4 * acc[p][(l1, l2)]
                            out[p][a * nb + i, b * nb + j] = v
                            out[p][b * nb + j, a * nb + i] = v
        return out

    def covariance(self, spectra, ells=None, verbose=False):
        comps = self.components(spectra, ells=ells, verbose=verbose)
        ells = self.cov.ells if ells is None else tuple(ells)
        A = str(spectra[0][0])
        return comps['snake'] + comps['star'], [(A, A, l, i) for l in ells for i in range(len(self.cov.k_edges) - 1)]


# ----------------------------------------------------------------------------- templates
class CovarianceTemplates:
    """C(amplitudes) = C_fixed + sum_i A_i C_i: the container for fitting the amplitudes of non-Gaussian
    terms (e.g. to subvolumes of the data). Templates are added by name; unspecified amplitudes are 1."""

    def __init__(self, fixed=None):
        self.fixed = fixed
        self.templates = {}

    def add(self, name, C):
        self.templates[str(name)] = np.asarray(C, float)
        return self

    def update(self, comps, prefix=''):
        for name, C in comps.items():
            self.add(prefix + name, C)
        return self

    @property
    def names(self):
        return list(self.templates)

    def __call__(self, **amplitudes):
        unknown = set(amplitudes) - set(self.templates)
        if unknown:
            raise KeyError(f'unknown templates {sorted(unknown)}; have {self.names}')
        C = 0.0 if self.fixed is None else np.array(self.fixed, float)
        for name, T in self.templates.items():
            C = C + amplitudes.get(name, 1.0) * T
        return C

    def save(self, fn):
        np.savez(fn, fixed=np.asarray(self.fixed if self.fixed is not None else np.nan), names=np.array(self.names),
                 **{f't_{n}': T for n, T in self.templates.items()})

    @classmethod
    def load(cls, fn):
        with np.load(fn) as z:
            fixed = z['fixed']
            out = cls(None if fixed.ndim == 0 else fixed)
            for n in z['names']:
                out.add(str(n), z[f't_{n}'])
        return out
