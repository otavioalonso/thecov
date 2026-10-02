"""Pair-averaged clustering windows: the Gaussian covariance beyond the local approximation W = m^2.

The clustering part of the covariance contains m(x) m(x + r) weighted by xi(r) e^{i k.r} over a
correlation length. The local approximation replaces it by m(x)^2, which fails wherever the window
changes on that scale (veto holes, footprint edges, completeness patterns). Keeping the partner inside
the r integral gives, to the same order in the window width,

    G_P(k, k') = P(k) W~_k(k - k'),   W_k(x) = m_A(x) (K_k * m_B)(x),   K_k(r) = xi(r) e^{i k.r} / P(k),

with int K_k = 1, and the mean of the estimator is <P_hat(k)> = P(k) I_k / I with I_k = int W_k and I the
estimator's normalisation (see reports / the note window_kernel). The kernel is set by xi: there is no
free smoothing scale.

Implementation (isotropic kernel, angle average xi_0(r) j_0(k r) / P_0(k), averaged over each k-bin):
* two-stage compression, so that W_{k_i} = sum_b c_ib W_b and the pair counts are done once per basis
  pair (all in the same pass, as extra weight columns):
  1. the per-bin kernels K_i(r) are expanded on a kernel basis, K_i ~ sum_a d_ia K_a (weighted SVD with
     the window's own autocorrelation as the metric). This basis is converged to `tol` (typically ~20
     kernels for 0.005 bins up to k = 0.3: at large r the kernels oscillate as j_0(k r) and are nearly
     orthogonal), and (K_a * m_B) is evaluated at the randoms for each of them. I_k comes from this stage;
  2. the smoothed windows (K_i * m_B)(x) at the randoms are much lower rank than the kernels (the
     oscillating large-r tails average against a smooth window): they are compressed again by an SVD
     in the metric int m_A^2 v v', to the B window basis functions that enter the pair counts
     (the cost of the covariance pair counts grows as B^2);
* (K_b * m_B) is evaluated at the randoms of every tracer by a direct neighbour sum over a reference
  random catalogue of B for r < r_split (where the kernel is steep: xi ~ r^-1.8) and an FFT of the
  reference randoms painted on a mesh for the smooth remainder. The reference randoms should be denser
  than (and may include) the tracer's randoms; a random never averages with itself (zero-distance pairs
  are dropped), which would otherwise bring back the <w^2> bias of per-object weights.

Usage:
    sm = WindowSmoothing()
    sm.add_density('LRG', ref_positions, ref_weights, ref_alpha)   # many random files
    sm.set_power('LRG', 'LRG', k, P0)                                # fiducial shape (amplitude cancels)
    cov = GaussianCovariance([lrg], k_edges, ..., smoothing=sm)
    cov.compute_windows([('LRG', 'LRG')])
    cov.set_model(model, masked=True)     # model = window-convolved (measured) multipoles; or masked=False
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import scipy.fft as sfft
from scipy.spatial import cKDTree
from scipy.special import spherical_jn


# ----------------------------------------------------------------------------- kernels
def extend_power(k, p, k_max=5.0, damping=1.0, n_low=1.0, n_k=20001):
    """P on [1e-4, k_max] from its values on [k0, k1]: (k/k0)^n_low below k0, a power law fitted to
    the last 8 points above k1, times exp(-(k damping)^2) [damping in Mpc/h] to regularise xi(r -> 0)."""
    k, p = np.asarray(k, float), np.asarray(p, float)
    kk = np.linspace(1e-4, k_max, n_k)
    tail = slice(max(0, len(k) - 8), len(k))
    slope = np.polyfit(np.log(k[tail]), np.log(np.abs(p[tail]) + 1e-300), 1)[0]
    out = np.interp(kk, k, p)
    out = np.where(kk < k[0], p[0] * (kk / k[0]) ** n_low, out)
    out = np.where(kk > k[-1], p[-1] * (kk / k[-1]) ** slope, out)
    return kk, out * np.exp(-(kk * damping) ** 2)


def xi0_from_power(kk, pk, r):
    """xi_0(r) = int k^2 dk / (2 pi^2) P(k) j_0(k r)."""
    xi = np.empty(len(r))
    for i in range(0, len(r), 64):
        rr = np.asarray(r[i:i + 64])[:, None]
        xi[i:i + 64] = np.trapezoid(kk ** 2 * pk * np.sinc(kk * rr / np.pi), kk, axis=1) / (2 * np.pi ** 2)
    return xi


def bin_kernels(k_edges, r, xi, nk=16):
    """K_i(r) = xi_0(r) <j_0(k r)>_i / norm for each k-bin i (mode-weighted average over the bin),
    normalised to int 4 pi r^2 K_i dr = 1 on the tabulated r. Shape (nbins, nr)."""
    out = np.empty((len(k_edges) - 1, len(r)))
    for i, (lo, hi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
        kb = np.linspace(lo, hi, nk)
        wk = kb ** 2 / np.sum(kb ** 2)
        K = xi * np.sum(wk[:, None] * spherical_jn(0, kb[:, None] * r[None, :]), axis=0)
        out[i] = K / np.trapezoid(4 * np.pi * r ** 2 * K, r)
    return out


def hankel0(r, K, q):
    """K~(q) = int 4 pi r^2 K(r) j_0(q r) dr, for K of shape (..., nr)."""
    K = np.atleast_2d(K)
    out = np.empty((K.shape[0], len(q)))
    for i in range(0, len(q), 256):
        jq = np.sinc(np.asarray(q[i:i + 256])[:, None] * r[None, :] / np.pi)
        out[:, i:i + 256] = np.trapezoid(4 * np.pi * r ** 2 * K[:, None, :] * jq[None], r, axis=-1)
    return out


def taper(r, r_split, width):
    """1 below r_split - width, 0 above r_split + width, cos^2 in between (near part of the kernel)."""
    x = np.clip((r - (r_split - width)) / (2 * width), 0.0, 1.0)
    return np.cos(0.5 * np.pi * x) ** 2


# ----------------------------------------------------------------------------- density
class DensityField:
    """m = alpha sum_r w_r delta_D(x - x_r) from a reference random catalogue: painted (CIC) on a
    zero-padded mesh for the FFT part of the smoothing, and kept as points for the near part."""

    def __init__(self, positions, weights, alpha, cell=3.0, pad=200.0, workers=None):
        self.pos = np.ascontiguousarray(positions, dtype=float)
        self.val = float(alpha) * np.asarray(weights, dtype=float)
        self.cell, self.pad = float(cell), float(pad)
        self.workers = workers or os.cpu_count()
        lo = self.pos.min(0) - pad - 2 * cell
        hi = self.pos.max(0) + pad + 2 * cell
        self.lo = lo
        self.shape = tuple(sfft.next_fast_len(int(np.ceil((h - l) / cell)) + 1) for l, h in zip(lo, hi))
        self._mk = None
        self._tree = None

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.pos)
        return self._tree

    # -- mesh helpers
    def _cic(self, pos):
        u = (pos - self.lo) / self.cell
        i0 = np.floor(u).astype(np.int64)
        return i0, u - i0

    def _corners(self, pos):
        i0, f = self._cic(pos)
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    yield np.ravel_multi_index((i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz), self.shape), wx * wy * wz

    def mesh_fft(self, chunk=4_000_000):
        if self._mk is None:
            size = int(np.prod(self.shape))
            mesh = np.zeros(size, np.float64)
            for s in range(0, len(self.pos), chunk):
                v = self.val[s:s + chunk] / self.cell ** 3
                idx, wts = zip(*self._corners(self.pos[s:s + chunk]))
                mesh += np.bincount(np.concatenate(idx), weights=np.concatenate([v * w for w in wts]), minlength=size)
            self._mk = sfft.rfftn(mesh.reshape(self.shape).astype(np.float32), workers=self.workers)
            del mesh
        return self._mk

    def _kgrid(self):
        kx = 2 * np.pi * np.fft.fftfreq(self.shape[0], d=self.cell)
        ky = 2 * np.pi * np.fft.fftfreq(self.shape[1], d=self.cell)
        kz = 2 * np.pi * np.fft.rfftfreq(self.shape[2], d=self.cell)
        return kx, ky, kz

    def _apply(self, filt):
        """irfft(mesh_fft * filt(|k|) / W_cic^2) slab by slab; filt maps |k| arrays to values."""
        mk = self.mesh_fft()
        kx, ky, kz = self._kgrid()
        half = self.cell / 2
        wy = np.maximum(np.sinc(ky * half / np.pi) ** 2, 0.1)
        wz = np.maximum(np.sinc(kz * half / np.pi) ** 2, 0.1)
        out = np.empty(mk.shape, np.complex64)
        for i, kxi in enumerate(kx):
            kk = np.sqrt(kxi ** 2 + ky[:, None] ** 2 + kz[None, :] ** 2)
            wc = max(np.sinc(kxi * half / np.pi) ** 2, 0.1) * wy[:, None] * wz[None, :]
            out[i] = mk[i] * (filt(kk) / wc ** 2)        # CIC painting and reading compensated
        return sfft.irfftn(out, s=self.shape, workers=self.workers).astype(np.float32)

    def _read(self, field, pos):
        flat = field.reshape(-1)
        out = np.zeros(len(pos))
        for s in range(0, len(pos), 2_000_000):
            acc = np.zeros(min(2_000_000, len(pos) - s))
            for idx, w in self._corners(pos[s:s + 2_000_000]):
                acc += flat[idx] * w
            out[s:s + 2_000_000] = acc
        return out

    # -- quantities
    def autocorrelation(self, r_edges):
        """Q_mm(r) = int m(x) m(x + r) d^3x, angle-averaged in r bins (mesh resolution)."""
        mk = self.mesh_fft()
        kx, ky, kz = self._kgrid()
        half = self.cell / 2
        wy = np.maximum(np.sinc(ky * half / np.pi) ** 2, 0.1)
        wz = np.maximum(np.sinc(kz * half / np.pi) ** 2, 0.1)
        pw = np.empty(mk.shape, np.complex64)
        for i, kxi in enumerate(kx):
            wc = max(np.sinc(kxi * half / np.pi) ** 2, 0.1) * wy[:, None] * wz[None, :]
            pw[i] = (np.abs(mk[i]) ** 2 / wc ** 2) * self.cell ** 3
        corr = sfft.irfftn(pw, s=self.shape, workers=self.workers)
        del pw
        dx = [np.fft.fftfreq(n, d=1.0 / n) * self.cell for n in self.shape]
        nb = len(r_edges) - 1
        num, cnt = np.zeros(nb + 1), np.zeros(nb + 1)
        for i in range(self.shape[0]):
            if abs(dx[0][i]) >= r_edges[-1]:
                continue
            rr = np.sqrt(dx[0][i] ** 2 + dx[1][:, None] ** 2 + dx[2][None, :] ** 2).ravel()
            ib = np.clip(np.digitize(rr, r_edges) - 1, 0, nb)
            ib[rr >= r_edges[-1]] = nb
            num += np.bincount(ib, weights=corr[i].ravel(), minlength=nb + 1)
            cnt += np.bincount(ib, minlength=nb + 1)
        return num[:nb] / np.maximum(cnt[:nb], 1)

    def smooth_far(self, r, kernels, positions, q_max=None):
        """(K_b * m) at positions for radial kernels (B, nr) on r, by FFT. Returns (B, N)."""
        kx, ky, kz = self._kgrid()
        q = np.linspace(0, np.sqrt(kx.max() ** 2 + ky.max() ** 2 + kz.max() ** 2) * 1.01, 4096)
        Kq = hankel0(r, kernels, q)
        out = np.empty((len(kernels), len(positions)))
        for b in range(len(kernels)):
            field = self._apply(lambda kk, b=b: np.interp(kk, q, Kq[b]).astype(np.float32))
            out[b] = self._read(field, positions)
            del field
        return out

    def smooth_near(self, r, kernels, positions, r_max, chunk=50_000):
        """(K_b * m) at positions for kernels (B, nr) supported on r < r_max, by direct sums over the
        reference points within r_max (zero-distance pairs, i.e. a random with itself, excluded)."""
        dr = r[1] - r[0]
        out = np.zeros((len(kernels), len(positions)))
        tree = self.tree
        for s in range(0, len(positions), chunk):
            sub = cKDTree(positions[s:s + chunk])
            m = sub.sparse_distance_matrix(tree, r_max, output_type='ndarray')
            d = m['v']
            keep = d > 1e-9
            i, j, d = m['i'][keep].astype(np.int64), m['j'][keep].astype(np.int64), d[keep]
            x = np.clip((d - r[0]) / dr, 0.0, len(r) - 1.000001)     # linear interpolation in r (the
            ir = x.astype(np.int64)                                  # kernel is steep: xi ~ r^-1.8)
            f = x - ir
            v = self.val[j]
            n = min(chunk, len(positions) - s)
            for b in range(len(kernels)):
                kb = kernels[b]
                out[b, s:s + n] = np.bincount(i, weights=v * (kb[ir] * (1 - f) + kb[ir + 1] * f), minlength=n)
        return out


# ----------------------------------------------------------------------------- smoothing
class WindowSmoothing:
    """Pair-averaged clustering windows W_b = m_A (K_b * m_B) on a kernel basis, with per-k-bin
    coefficients c_ib (K_{k_i} = sum_b c_ib K_b).

    Parameters
    ----------
    cell     : FFT mesh cell [Mpc/h] for the smooth part of the kernels
    r_split  : the kernels are split at r_split (smooth cos^2 taper of half-width `taper_width`):
               direct neighbour sums below, FFT above
    r_max    : kernel truncation [Mpc/h]; also the zero-padding of the mesh
    tol      : basis sizes: the smallest kernel basis with relative errors on I_k and on the kernels
               (window-weighted L2) below tol for every bin; then the smallest window basis B with relative
               errors on I_k and on the windows m_A (K_k * m_B) (L2 over the survey) below tol
    max_kernels: upper limit on the kernel basis (stage 1)
    max_basis: upper limit on B (stage 2; None: set by tol only)
    damping  : Gaussian damping [Mpc/h] of the power extrapolated beyond the tabulated k range
    """

    def __init__(self, cell=3.0, r_split=8.0, taper_width=2.0, r_max=200.0, dr=0.25, tol=2e-3,
                 max_basis=None, max_kernels=48, damping=1.0, workers=None):
        self.cell, self.r_split, self.taper_width = float(cell), float(r_split), float(taper_width)
        self.r_max, self.dr = float(r_max), float(dr)
        self.tol, self.damping = float(tol), float(damping)
        self.max_basis, self.max_kernels = (None if max_basis is None else int(max_basis)), int(max_kernels)
        self.workers = workers
        self.r = np.arange(self.dr / 2, self.r_max, self.dr)
        self._densities = {}
        self._power = {}
        self._pairs = {}           # (A, B) -> dict(coeffs, I_b, n_basis, kernels, ...)
        self._values = {}          # (A, B) -> {tracer name: (B, N) array}

    # -- inputs
    @staticmethod
    def pair_key(A, B):
        return tuple(sorted((str(A), str(B))))

    def add_density(self, name, positions, weights, alpha):
        """Reference randoms of tracer `name` (positions, weights, alpha such that alpha sum w = the
        weighted number of galaxies): they sample m_name. Use many random files."""
        self._densities[str(name)] = DensityField(positions, weights, alpha, cell=self.cell, pad=self.r_max,
                                                  workers=self.workers)
        return self

    def set_power(self, A, B, k, p0):
        """Fiducial monopole P_0^{AB}(k) whose shape sets the kernel (the amplitude cancels)."""
        self._power[self.pair_key(A, B)] = (np.asarray(k, float), np.asarray(p0, float))
        return self

    @property
    def tag(self):
        """identifies the kernels (part of the basis windows' keys): settings and fiducial power; a
        loaded smoothing keeps the tag it was saved with, so that its cached windows are recognised"""
        if getattr(self, '_loaded_tag', None):
            return self._loaded_tag
        key = repr((self.cell, self.r_split, self.taper_width, self.r_max, self.dr, self.tol, self.max_basis,
                    self.max_kernels, self.damping, sorted((k, float(np.sum(v[0])), float(np.sum(v[1]))) for k, v in self._power.items())))
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def has(self, A, B):
        return self.pair_key(A, B) in self._pairs

    def n_basis(self, A, B):
        return self._pairs[self.pair_key(A, B)]['n_basis']

    def kernel_coeffs(self, A, B, k_edges=None):
        """(nbins, n_kernels) coefficients d_ia of the k-bin kernels on the kernel basis (stage 1). Other
        binnings are projected on the basis (the smoothing and the pair counts do not depend on it)."""
        p = self._pairs[self.pair_key(A, B)]
        if k_edges is None or (len(k_edges) == len(p['k_edges']) and np.allclose(k_edges, p['k_edges'])):
            return p['kcoeffs']
        ck = ('proj', tuple(np.round(np.asarray(k_edges, float), 10)))
        if ck not in p:
            Ki = bin_kernels(np.asarray(k_edges, float), self.r, p['xi'])
            p[ck] = (Ki * p['sw'][None, :]) @ (p['basis'] * p['sw'][None, :]).T    # basis orthonormal in the metric
        return p[ck]

    def coeffs(self, A, B, k_edges=None):
        """(nbins, n_basis) coefficients c_ib of the k-bins `k_edges` (default: those of build) on the
        window basis W_b, whose pair counts enter the covariance."""
        p = self._pairs[self.pair_key(A, B)]
        if k_edges is None or (len(k_edges) == len(p['k_edges']) and np.allclose(k_edges, p['k_edges'])):
            return p['coeffs']
        return self.kernel_coeffs(A, B, k_edges) @ p['proj']

    def integrals(self, A, B):
        """I_a = int m_A (K_a * m_B) for the kernel basis, from the randoms of the host A
        (alpha_A sum_r w_r (K_a * m_B))."""
        return self._pairs[self.pair_key(A, B)]['I_b']

    def I_k(self, A, B, k_edges=None):
        """I_{k_i} = int m_A (K_{k_i} * m_B) per k-bin (from the converged kernel basis)."""
        return self.kernel_coeffs(A, B, k_edges) @ self.integrals(A, B)

    def values(self, A, B, b, at):
        """(K_b * m_B)(x) of window basis function b at the randoms of tracer `at` (B the partner of the
        canonical pair)."""
        return self._values[self.pair_key(A, B)][str(at)][b]

    # -- build
    def _basis(self, k_edges, key, log):
        k, p0 = self._power[key]
        kk, pk = extend_power(k, p0, damping=self.damping)
        xi = xi0_from_power(kk, pk, self.r)
        Ki = bin_kernels(np.asarray(k_edges, float), self.r, xi)                 # (nbins, nr)
        # metric: the window's autocorrelation (the error on I_k = int K_k Q_mm is then controlled)
        dens = self._densities[key[1]]
        r_edges = np.concatenate([[0.0], np.arange(self.cell, self.r_max + self.cell, self.cell)])
        q_mm = dens.autocorrelation(r_edges)
        rc = 0.5 * (r_edges[1:] + r_edges[:-1])
        rho = np.interp(self.r, rc, q_mm / q_mm[0])
        w = 4 * np.pi * self.r ** 2 * self.dr * np.maximum(rho, 0.0) + 1e-30
        sw = np.sqrt(w)
        U, S, Vt = np.linalg.svd(Ki * sw[None, :], full_matrices=False)
        Iex = Ki @ w
        for nb in range(1, min(self.max_kernels, len(S)) + 1):
            rec = (U[:, :nb] * S[:nb]) @ Vt[:nb] / sw[None, :]
            err_I = np.max(np.abs((rec - Ki) @ w) / np.abs(Iex))
            err_K = np.max(np.sqrt(((rec - Ki) ** 2 * w).sum(1) / (Ki ** 2 * w).sum(1)))
            if max(err_I, err_K) < self.tol:
                break
        basis = Vt[:nb] / sw[None, :]                    # (nb, nr): K_b(r), orthonormal with the metric w
        coeffs = U[:, :nb] * S[:nb]                       # (nbins, nb)
        widths = [float(np.sqrt(abs(np.trapezoid(4 * np.pi * self.r ** 4 * K, self.r)
                                    / np.trapezoid(4 * np.pi * self.r ** 2 * K, self.r)))) for K in Ki[[0, len(Ki) // 2, -1]]]
        log(f'  smoothing {key}: {nb} kernels (errors: I_k {err_I:.1e}, kernels {err_K:.1e}); '
            f'kernel rms widths {widths[0]:.1f} / {widths[1]:.1f} / {widths[2]:.1f} Mpc/h (first / middle / last bin)')
        return basis, coeffs, dict(err_I=float(err_I), err_K=float(err_K), widths=widths, sw=sw, xi=xi,
                                   k_edges=np.asarray(k_edges, float))

    def _window_basis(self, d, v, host, I_a):
        """Stage 2: compress the smoothed windows v_i = sum_a d_ia v_a (v_a = (K_a * m_B) at the host's
        randoms) in the metric G_aa' = int m_A^2 v_a v_a' (alpha sum_r w_r mw_r v_a v_a'). Returns the
        projection P (n_kernels, B) with c = d P, the coefficients c, the window basis T (B, n_kernels),
        W_j = sum_a T_ja W_a, and the errors on I_k and on the windows."""
        v = v.astype(float)
        G = host.alpha * (v * (host.w * host.mw)[None, :]) @ v.T
        g, E = np.linalg.eigh(0.5 * (G + G.T))
        g = np.maximum(g, g.max() * 1e-14)
        L, Linv = E * np.sqrt(g)[None, :], (E / np.sqrt(g)[None, :]).T          # G = L L^T
        M = d @ L
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        Iex = d @ I_a
        nmax = len(S) if self.max_basis is None else min(self.max_basis, len(S))
        for nb in range(1, nmax + 1):
            R = M - (U[:, :nb] * S[:nb]) @ Vt[:nb]
            err_W = np.max(np.linalg.norm(R, axis=1) / np.linalg.norm(M, axis=1))
            err_I = np.max(np.abs((R @ Linv) @ I_a) / np.abs(Iex))
            if max(err_W, err_I) < self.tol:
                break
        P = L @ Vt[:nb].T                     # (n_kernels, B): c = d P
        return P, d @ P, Vt[:nb] @ Linv, dict(err_W=float(err_W), err_I=float(err_I))

    def build(self, k_edges, tracers, pairs, log=print):
        """Basis, coefficients, and (K_b * m_B) at the randoms of every tracer, for each pair (A, B).
        `tracers`: {name: Tracer}; `pairs`: iterable of (A, B) names."""
        for A, B in pairs:
            key = self.pair_key(A, B)
            if key in self._pairs:
                continue
            if key[1] not in self._densities:
                raise ValueError(f"no reference density for {key[1]}: add_density first")
            if key not in self._power:
                raise ValueError(f"no fiducial power for {key}: set_power first")
            t0 = time.time()
            basis, coeffs, diag = self._basis(k_edges, key, log)
            tp = taper(self.r, self.r_split, self.taper_width)
            near, far = basis * tp[None, :], basis * (1 - tp)[None, :]
            dens = self._densities[key[1]]
            vals = {}
            for name, tr in tracers.items():
                v = dens.smooth_far(self.r, far, tr.pos)
                v += dens.smooth_near(self.r, near, tr.pos, self.r_split + self.taper_width)
                vals[name] = v.astype(np.float32)
            host = tracers[key[0]]
            I_b = host.alpha * (vals[key[0]].astype(float) @ host.w)
            proj, wcoeffs, T, d2 = self._window_basis(coeffs, vals[key[0]], host, I_b)
            self._values[key] = {n: (T @ v.astype(float)).astype(np.float32) for n, v in vals.items()}
            self._pairs[key] = dict(kcoeffs=coeffs, coeffs=wcoeffs, proj=proj, I_b=I_b,
                                    n_basis=len(T), n_kernels=len(basis), basis=basis,
                                    err_IW=d2['err_I'], err_W=d2['err_W'], **diag)
            Ik = self.I_k(*key)
            loc = host.alpha * float(np.sum(host.w * host.mw))
            log(f'  smoothing {key}: {len(basis)} kernels -> {len(T)} windows (errors: I_k {d2["err_I"]:.1e}, '
                f'windows {d2["err_W"]:.1e}); built in {time.time() - t0:.0f} s; I_k / int m_A m_B (first, last bin) = '
                f'{Ik[0] / loc:.4f}, {Ik[-1] / loc:.4f}')
        return self

    # -- persistence (the pair counts are saved by the window library; this keeps what the covariance
    #    assembly and the masked model need: coefficients and integrals)
    def save(self, path):
        data, meta = {}, []
        for n, (key, p) in enumerate(self._pairs.items()):
            data[f'coeffs_{n}'], data[f'I_b_{n}'], data[f'basis_{n}'] = p['coeffs'], p['I_b'], p['basis']
            data[f'kcoeffs_{n}'], data[f'proj_{n}'] = p['kcoeffs'], p['proj']
            data[f'sw_{n}'], data[f'xi_{n}'], data[f'k_edges_{n}'] = p['sw'], p['xi'], p['k_edges']
            meta.append(dict(key=list(key), n=n, err_I=p['err_I'], err_K=p['err_K'], err_IW=p['err_IW'], err_W=p['err_W'],
                             widths=p['widths']))
        data['meta'] = np.array(json.dumps(dict(pairs=meta, tag=self.tag)))
        np.savez(path, **data)

    def load(self, path):
        with np.load(path) as f:
            meta = json.loads(str(f['meta'].item()))
            self._loaded_tag = meta['tag']
            for rec in meta['pairs']:
                n = rec['n']
                self._pairs[tuple(rec['key'])] = dict(coeffs=f[f'coeffs_{n}'], I_b=f[f'I_b_{n}'], basis=f[f'basis_{n}'],
                                                      sw=f[f'sw_{n}'], xi=f[f'xi_{n}'], k_edges=f[f'k_edges_{n}'],
                                                      kcoeffs=f[f'kcoeffs_{n}'], proj=f[f'proj_{n}'],
                                                      n_basis=int(f[f'coeffs_{n}'].shape[1]),
                                                      n_kernels=int(f[f'basis_{n}'].shape[0]), err_I=rec['err_I'],
                                                      err_K=rec['err_K'], err_IW=rec['err_IW'], err_W=rec['err_W'],
                                                      widths=rec['widths'])
        return self
