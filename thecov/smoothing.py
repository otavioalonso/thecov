"""Pair-averaged clustering windows: the Gaussian covariance beyond the local approximation W^{AB} = W_A W_B.

The clustering part of the covariance contains W_A(x) W_B(x + r) weighted by xi(r) e^{i k.r} over a
correlation length. The local approximation replaces it by W_A(x) W_B(x), which fails wherever the window
changes on that scale (veto holes, footprint edges, completeness patterns). Keeping the partner inside
the r integral gives, to the same order in the window width,

    G_P(k, k') = P(k) W~_k(k - k'),   W^{AB}_k(x) = W_A(x) (K_k * W_B)(x),   K_k(r) = xi(r) e^{i k.r} / P(k),

with int K_k = 1, and the mean of the estimator is <P_hat(k)> = P(k) I_k / I with I_k = int W^{AB}_k and I the
estimator's normalisation (see reports / the note window_kernel). The kernel is set by xi: there is no
free smoothing scale.

Implementation (isotropic kernel, angle average xi_0(r) j_0(k r) / P_0(k), averaged over each k-bin):
* two-stage compression, so that W^{AB}_{k_i} = sum_b c_ib W^{AB}_b and the pair counts are done once per basis
  pair (all in the same pass, as extra weight columns):
  1. the per-bin kernels K_i(r) are expanded on a kernel basis, K_i ~ sum_a d_ia K_a (weighted SVD with
     the window's own autocorrelation as the metric). This basis is converged to `tol` (typically ~20
     kernels for 0.005 bins up to k = 0.3: at large r the kernels oscillate as j_0(k r) and are nearly
     orthogonal), and (K_a * W_B) is evaluated at the randoms for each of them. I_k comes from this stage;
  2. the smoothed windows (K_i * W_B)(x) at the randoms are much lower rank than the kernels (the
     oscillating large-r tails average against a smooth window): they are compressed again by an SVD
     in the metric int W_A^2 v v', to the B window basis functions that enter the pair counts
     (the cost of the covariance pair counts grows as B^2);
* (K_b * W_B) is evaluated at the randoms of every tracer by a direct neighbour sum over a reference
  random catalogue of B for r < r_split (where the kernel is steep: xi ~ r^-1.8) and an FFT of the
  reference randoms painted on a mesh for the smooth remainder. The reference randoms should be denser
  than (and may include) the tracer's randoms; a random never averages with itself (zero-distance pairs
  are dropped), which would otherwise bring back the <w^2> bias of per-object weights.

Usage:
    sm = WindowSmoothing()
    sm.add_density('LRG', ref_positions, ref_weights, ref_alpha)   # many random files
    sm.set_power('LRG', 'LRG', fiducial)       # optional: a PowerSpectrum for the kernel shape (default: the covariance's)
    cov = GaussianCovariance([lrg], power, ..., smoothing=sm)        # power: a PowerSpectrum (power.py)
    cov.compute_windows([('LRG', 'LRG')])
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import scipy.fft as sfft
from scipy.spatial import cKDTree

from . import cache, defaults as D
from .power import PowerSpectrum, BinnedHankel, hankel_integral

VERSION = 2          # part of the kernel tag: 2 = xi_0 and the bin-averaged j_0 by BinnedHankel quadrature


# ----------------------------------------------------------------------------- kernels
def kernel_power(power, low_slope=D.KERNEL_LOW_SLOPE):
    """The P_0 that shapes the kernel, as a callable defined at every k: a PowerSpectrum's P_0, continued beyond its
    arrays as a power law (k^low_slope below, the slope of the last points above) unless it extrapolates already."""
    if callable(power) and not isinstance(power, PowerSpectrum):
        return power
    if power.extrapolation is False:
        power = power.extrapolated('power-law', low_slope=low_slope)
    return power.function(0)


def xi0_from_power(power, r, damping=D.KERNEL_DAMPING, k_max=D.KERNEL_K_MAX):
    """xi_0(r) = int_0^k_max k^2 dk / (2 pi^2) P(k) exp(-(k damping)^2) j_0(k r) for a PowerSpectrum (its P_0,
    see kernel_power) or a callable P(k); the damping [Mpc/h] regularises xi at r -> 0."""
    damp = lambda k: np.exp(-(k * damping) ** 2)
    return hankel_integral(kernel_power(power), np.linspace(1e-4, k_max, 101), r, factor=damp) / (2 * np.pi ** 2)


def bin_kernels(kedges, r, xi):
    """K_i(r) = xi_0(r) <j_0(k r)>_i / norm for each k-bin i (<.>_i: the k^2-weighted bin average),
    normalised to int 4 pi r^2 K_i dr = 1 on the tabulated r. Shape (nbins, nr)."""
    K = xi[None, :] * BinnedHankel(kedges, r).transform(None, 0)
    return K / np.trapezoid(4 * np.pi * r ** 2 * K, r, axis=1)[:, None]


def hankel0(r, K, q):
    """K~(q) = int 4 pi r^2 K(r) j_0(q r) dr, for K of shape (..., nr) tabulated on r (trapezoid rule: K is known
    on the grid only, and Gauss-Legendre nodes would need it interpolated, which is less accurate)."""
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
    """W = alpha sum_r w_r delta_D(x - x_r) from a reference random catalogue: painted (CIC) on a zero-padded mesh for
    the FFT part of the smoothing, and kept as points for the near part."""

    def __init__(self, positions, weights, alpha, cell=3.0, pad=200.0, workers=None):
        self.positions = np.ascontiguousarray(positions, dtype=float)
        self.values = float(alpha) * np.asarray(weights, dtype=float)       # alpha w_r: W sampled by the points
        self.cell, self.pad = float(cell), float(pad)
        self.workers = workers or os.cpu_count()
        self.lo = self.positions.min(0) - pad - 2 * cell
        hi = self.positions.max(0) + pad + 2 * cell
        self.shape = tuple(sfft.next_fast_len(int(np.ceil((h - l) / cell)) + 1) for l, h in zip(self.lo, hi))
        self._mesh_fft = None
        self._tree = None

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.positions)
        return self._tree

    # -- mesh helpers
    def _cic_corners(self, positions):
        """(flat mesh index, CIC weight) of the 8 cells around each position."""
        u = (positions - self.lo) / self.cell
        i0 = np.floor(u).astype(np.int64)
        f = u - i0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                         * (f[:, 2] if dz else 1 - f[:, 2]))
                    yield np.ravel_multi_index((i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz), self.shape), w

    def mesh_fft(self, chunk=4_000_000):
        """rfft of the CIC-painted density (float32), computed once."""
        if self._mesh_fft is None:
            size = int(np.prod(self.shape))
            mesh = np.zeros(size, np.float64)
            for s in range(0, len(self.positions), chunk):
                v = self.values[s:s + chunk] / self.cell ** 3
                idx, wts = zip(*self._cic_corners(self.positions[s:s + chunk]))
                mesh += np.bincount(np.concatenate(idx), weights=np.concatenate([v * w for w in wts]), minlength=size)
            self._mesh_fft = sfft.rfftn(mesh.reshape(self.shape).astype(np.float32), workers=self.workers)
            del mesh
        return self._mesh_fft

    def _wavevectors(self):
        return (2 * np.pi * np.fft.fftfreq(self.shape[0], d=self.cell), 2 * np.pi * np.fft.fftfreq(self.shape[1], d=self.cell),
                2 * np.pi * np.fft.rfftfreq(self.shape[2], d=self.cell))

    def _slabs(self):
        """(i, |k| on slab i, CIC window^2 on slab i) for every x slab of the rfft mesh (one CIC window per painting
        or reading, floored at 0.1 to avoid amplifying the corners)."""
        kx, ky, kz = self._wavevectors()
        half = self.cell / 2
        wy = np.maximum(np.sinc(ky * half / np.pi) ** 2, 0.1)
        wz = np.maximum(np.sinc(kz * half / np.pi) ** 2, 0.1)
        for i, kxi in enumerate(kx):
            k = np.sqrt(kxi ** 2 + ky[:, None] ** 2 + kz[None, :] ** 2)
            yield i, k, max(np.sinc(kxi * half / np.pi) ** 2, 0.1) * wy[:, None] * wz[None, :]

    def _filtered(self, filt):
        """irfft(mesh_fft * filt(|k|) / W_cic^2): the density convolved with a radial kernel, CIC painting and
        reading compensated."""
        mk = self.mesh_fft()
        out = np.empty(mk.shape, np.complex64)
        for i, k, wc in self._slabs():
            out[i] = mk[i] * (filt(k) / wc ** 2)
        return sfft.irfftn(out, s=self.shape, workers=self.workers).astype(np.float32)

    def _read(self, field, positions, chunk=2_000_000):
        """CIC interpolation of a mesh field at positions."""
        flat = field.reshape(-1)
        out = np.zeros(len(positions))
        for s in range(0, len(positions), chunk):
            for idx, w in self._cic_corners(positions[s:s + chunk]):
                out[s:s + chunk] += flat[idx] * w
        return out

    # -- quantities
    def autocorrelation(self, r_edges):
        """Q_WW(r) = int W(x) W(x + r) d^3x, angle-averaged in r bins (mesh resolution)."""
        mk = self.mesh_fft()
        power = np.empty(mk.shape, np.complex64)
        for i, _, wc in self._slabs():
            power[i] = (np.abs(mk[i]) ** 2 / wc ** 2) * self.cell ** 3
        corr = sfft.irfftn(power, s=self.shape, workers=self.workers)
        del power
        dx = [np.fft.fftfreq(n, d=1.0 / n) * self.cell for n in self.shape]
        nb = len(r_edges) - 1
        num, cnt = np.zeros(nb + 1), np.zeros(nb + 1)
        for i in range(self.shape[0]):
            if abs(dx[0][i]) >= r_edges[-1]:
                continue
            r = np.sqrt(dx[0][i] ** 2 + dx[1][:, None] ** 2 + dx[2][None, :] ** 2).ravel()
            ib = np.clip(np.digitize(r, r_edges) - 1, 0, nb)
            ib[r >= r_edges[-1]] = nb
            num += np.bincount(ib, weights=corr[i].ravel(), minlength=nb + 1)
            cnt += np.bincount(ib, minlength=nb + 1)
        return num[:nb] / np.maximum(cnt[:nb], 1)

    def convolve_by_fft(self, r, kernels, positions):
        """(K_b * W) at positions for radial kernels (B, nr) on r, by FFT of the mesh. Returns (B, N)."""
        kx, ky, kz = self._wavevectors()
        q = np.linspace(0, np.sqrt(kx.max() ** 2 + ky.max() ** 2 + kz.max() ** 2) * 1.01, 4096)
        Kq = hankel0(r, kernels, q)
        out = np.empty((len(kernels), len(positions)))
        for b in range(len(kernels)):
            out[b] = self._read(self._filtered(lambda k, b=b: np.interp(k, q, Kq[b]).astype(np.float32)), positions)
        return out

    def convolve_by_neighbours(self, r, kernels, positions, r_max, chunk=50_000):
        """(K_b * W) at positions for kernels (B, nr) supported on r < r_max, by direct sums over the reference points
        within r_max (zero-distance pairs, i.e. a random with itself, excluded)."""
        dr = r[1] - r[0]
        out = np.zeros((len(kernels), len(positions)))
        for s in range(0, len(positions), chunk):
            m = cKDTree(positions[s:s + chunk]).sparse_distance_matrix(self.tree, r_max, output_type='ndarray')
            keep = m['v'] > 1e-9
            i, j, d = m['i'][keep].astype(np.int64), m['j'][keep].astype(np.int64), m['v'][keep]
            x = np.clip((d - r[0]) / dr, 0.0, len(r) - 1.000001)     # linear interpolation in r (the kernel is steep:
            ir = x.astype(np.int64)                                  # xi ~ r^-1.8)
            f = x - ir
            v = self.values[j]
            n = min(chunk, len(positions) - s)
            for b, kb in enumerate(kernels):
                out[b, s:s + n] = np.bincount(i, weights=v * (kb[ir] * (1 - f) + kb[ir + 1] * f), minlength=n)
        return out


# ----------------------------------------------------------------------------- smoothing
def _smallest_rank(errors, n_max, tol):
    """(rank, errors) for the smallest rank in 1..n_max whose errors(rank) are all below tol (n_max if none is)."""
    for rank in range(1, n_max + 1):
        err = errors(rank)
        if max(err) < tol:
            break
    return rank, err


class WindowSmoothing:
    """Pair-averaged clustering windows W^{AB}_b = W_A (K_b * W_B) on a kernel basis, with per-k-bin coefficients c_ib
    (K_{k_i} = sum_b c_ib K_b).

    Parameters
    ----------
    cell     : FFT mesh cell [Mpc/h] for the smooth part of the kernels
    r_split  : the kernels are split at r_split (smooth cos^2 taper of half-width `taper_width`): direct neighbour
               sums below, FFT above
    r_max    : kernel truncation [Mpc/h]; also the zero-padding of the mesh
    tol      : basis sizes: the smallest kernel basis with relative errors on I_k and on the kernels (window-weighted
               L2) below tol for every bin; then the smallest window basis B with relative errors on I_k and on the
               windows W_A (K_k * W_B) (L2 over the survey) below tol
    max_kernels: upper limit on the kernel basis (stage 1)
    max_basis: upper limit on B (stage 2; None: set by tol only)
    damping  : Gaussian damping exp(-(k damping)^2) [Mpc/h] of the kernel's power (regularises xi at r -> 0)
    k_max, low_slope : the kernel's xi_0 integrates P to k_max, continued beyond its arrays as a power law
               (k^low_slope at low k, the slope of the last points at high k)

    Inputs, per tracer and pair (defaults when the covariance builds the smoothing): the reference density of a tracer
    (add_density; default: the tracer's own randoms) and the fiducial power of a pair (set_power; default: the
    covariance's power spectrum of the pair).
    """

    # what is stored per pair (A, B) (and saved, under these names):
    #   kcoeffs (nbins, n_kernels) d_ia, k-bin kernels on the kernel basis      basis (n_kernels, nr) K_a(r)
    #   proj (n_kernels, B) P with c = d P                                     coeffs (nbins, B) c_ib
    #   I_b (n_kernels,) int W_A (K_a * W_B)                                    sw, xi, k_edges: metric, xi_0, binning
    _SAVED = ('coeffs', 'I_b', 'basis', 'kcoeffs', 'proj', 'sw', 'xi', 'k_edges')      # k_edges: name kept in the files
    _ERRORS = ('err_I', 'err_K', 'err_IW', 'err_W', 'widths')

    def __init__(self, cell=D.KERNEL_CELL, r_split=D.KERNEL_R_SPLIT, taper_width=D.KERNEL_TAPER, r_max=D.KERNEL_R_MAX,
                 dr=D.KERNEL_DR, tol=D.KERNEL_TOL, max_basis=None, max_kernels=D.KERNEL_MAX_KERNELS,
                 damping=D.KERNEL_DAMPING, k_max=D.KERNEL_K_MAX, low_slope=D.KERNEL_LOW_SLOPE, workers=None):
        self.cell, self.r_split, self.taper_width = float(cell), float(r_split), float(taper_width)
        self.r_max, self.dr = float(r_max), float(dr)
        self.tol, self.damping, self.k_max, self.low_slope = float(tol), float(damping), float(k_max), float(low_slope)
        self.max_basis, self.max_kernels = (None if max_basis is None else int(max_basis)), int(max_kernels)
        self.workers = workers
        self.r = np.arange(self.dr / 2, self.r_max, self.dr)
        self._densities = {}
        self._power = {}
        self._pairs = {}           # (A, B) -> the arrays above, n_basis, n_kernels and the errors
        self._values = {}          # (A, B) -> {tracer name: (B, N) window-basis values at its randoms}

    # -- inputs
    @staticmethod
    def pair_key(A, B):
        return tuple(sorted((str(A), str(B))))

    def add_density(self, name, positions, weights, alpha):
        """Reference randoms of tracer `name` (positions, weights, alpha such that alpha sum w = the weighted number
        of galaxies): they sample W_name. Use many random files."""
        self._densities[str(name)] = DensityField(positions, weights, alpha, cell=self.cell, pad=self.r_max,
                                                  workers=self.workers)
        return self

    def set_power(self, A, B, power):
        """The fiducial power of the pair: a PowerSpectrum whose P_0 shapes the kernel (the amplitude cancels)."""
        if not isinstance(power, PowerSpectrum) or power.entry(0) is None:
            raise ValueError("set_power takes a PowerSpectrum with a monopole")
        self._power[self.pair_key(A, B)] = power
        return self

    @staticmethod
    def _fingerprint(power):
        """Numbers that identify the fiducial P_0 (for the tag): the sums of its arrays, or of its values on a grid."""
        entry = power.entry(0)
        if callable(entry):
            return float(np.sum(entry(np.geomspace(1e-3, 1.0, 64)))), power.extrapolation
        return float(np.sum(entry[0])), float(np.sum(entry[1])), power.extrapolation

    @property
    def tag(self):
        """identifies the kernels (part of the basis windows' keys): settings and fiducial power; a loaded smoothing
        keeps the tag it was saved with, so that its cached windows are recognised"""
        if getattr(self, '_loaded_tag', None):
            return self._loaded_tag
        return cache.tag(VERSION, self.cell, self.r_split, self.taper_width, self.r_max, self.dr, self.tol,
                         self.max_basis, self.max_kernels, self.damping, self.k_max, self.low_slope,
                         sorted((k, self._fingerprint(v)) for k, v in self._power.items()))

    def has_density(self, name):
        return str(name) in self._densities

    def has_power(self, A, B):
        return self.pair_key(A, B) in self._power

    # -- results
    def has(self, A, B):
        return self.pair_key(A, B) in self._pairs

    def n_basis(self, A, B):
        return self._pairs[self.pair_key(A, B)]['n_basis']

    def _same_binning(self, p, kedges):
        return kedges is None or (len(kedges) == len(p['k_edges']) and np.allclose(kedges, p['k_edges']))

    def kernel_coeffs(self, A, B, kedges=None):
        """(nbins, n_kernels) coefficients d_ia of the k-bin kernels on the kernel basis (stage 1). The kernels of other
        binnings are computed from the stored xi_0 and projected on the basis (no new pair counts); a warning reports a
        projection error above tol (exact for unions of the original bins, not guaranteed otherwise)."""
        p = self._pairs[self.pair_key(A, B)]
        if self._same_binning(p, kedges):
            return p['kcoeffs']
        key = ('proj', tuple(np.round(np.asarray(kedges, float), 10)))
        if key not in p:
            Ki = bin_kernels(np.asarray(kedges, float), self.r, p['xi'])
            w = p['sw'] ** 2
            d = (Ki * w[None, :]) @ p['basis'].T                  # basis orthonormal in the metric w
            R = Ki - d @ p['basis']
            err_K = np.max(np.sqrt((R ** 2 * w).sum(1) / (Ki ** 2 * w).sum(1)))
            err_I = np.max(np.abs(R @ w) / np.abs(Ki @ w))
            if max(err_K, err_I) > self.tol:
                warnings.warn(f"smoothing {self.pair_key(A, B)}: the kernels of the requested k bins are represented "
                              f"by the kernel basis with errors {err_I:.1e} (I_k) and {err_K:.1e} (kernels) > tol "
                              f"{self.tol:.0e}; rebuild the smoothing for these bins")
            p[key] = d
        return p[key]

    def coeffs(self, A, B, kedges=None):
        """(nbins, n_basis) coefficients c_ib of the k-bins `kedges` (default: those of build) on the window basis
        W^{AB}_b, whose pair counts enter the covariance."""
        p = self._pairs[self.pair_key(A, B)]
        return p['coeffs'] if self._same_binning(p, kedges) else self.kernel_coeffs(A, B, kedges) @ p['proj']

    def integrals(self, A, B):
        """I_a = int W_A (K_a * W_B) for the kernel basis, from the randoms of the host A (alpha_A sum_r w_r (K_a * W_B))."""
        return self._pairs[self.pair_key(A, B)]['I_b']

    def I_k(self, A, B, kedges=None):
        """I_{k_i} = int W_A (K_{k_i} * W_B) per k-bin (from the converged kernel basis)."""
        return self.kernel_coeffs(A, B, kedges) @ self.integrals(A, B)

    def values(self, A, B, b, at):
        """(K_b * W_B)(x) of window basis function b at the randoms of tracer `at` (B the partner of the canonical pair)."""
        return self._values[self.pair_key(A, B)][str(at)][b]

    # -- build
    def _kernel_basis(self, kedges, key, log):
        """Stage 1: the k-bin kernels K_i(r) on the smallest basis K_a (SVD in the metric of the window's own
        autocorrelation, so that the error on I_k = int K_k Q_WW is controlled)."""
        power = kernel_power(self._power[key], low_slope=self.low_slope)
        xi = xi0_from_power(power, self.r, damping=self.damping, k_max=self.k_max)
        Ki = bin_kernels(np.asarray(kedges, float), self.r, xi)                  # (nbins, nr)
        r_edges = np.concatenate([[0.0], np.arange(self.cell, self.r_max + self.cell, self.cell)])
        q_WW = self._densities[key[1]].autocorrelation(r_edges)
        rho = np.interp(self.r, 0.5 * (r_edges[1:] + r_edges[:-1]), q_WW / q_WW[0])
        w = 4 * np.pi * self.r ** 2 * self.dr * np.maximum(rho, 0.0) + 1e-30
        sw = np.sqrt(w)
        U, S, Vt = np.linalg.svd(Ki * sw[None, :], full_matrices=False)
        I_exact = Ki @ w

        def errors(rank):                     # relative errors on I_k and on the kernels, worst bin
            rec = (U[:, :rank] * S[:rank]) @ Vt[:rank] / sw[None, :]
            return (np.max(np.abs((rec - Ki) @ w) / np.abs(I_exact)),
                    np.max(np.sqrt(((rec - Ki) ** 2 * w).sum(1) / (Ki ** 2 * w).sum(1))))
        rank, (err_I, err_K) = _smallest_rank(errors, min(self.max_kernels, len(S)), self.tol)
        basis = Vt[:rank] / sw[None, :]                    # (rank, nr): K_a(r), orthonormal with the metric w
        coeffs = U[:, :rank] * S[:rank]                     # (nbins, rank)
        widths = [float(np.sqrt(abs(np.trapezoid(4 * np.pi * self.r ** 4 * K, self.r)
                                    / np.trapezoid(4 * np.pi * self.r ** 2 * K, self.r)))) for K in Ki[[0, len(Ki) // 2, -1]]]
        log(f'  smoothing {key}: {rank} kernels (errors: I_k {err_I:.1e}, kernels {err_K:.1e}); '
            f'kernel rms widths {widths[0]:.1f} / {widths[1]:.1f} / {widths[2]:.1f} Mpc/h (first / middle / last bin)')
        return basis, coeffs, dict(err_I=float(err_I), err_K=float(err_K), widths=widths, sw=sw, xi=xi,
                                   k_edges=np.asarray(kedges, float))

    def _window_basis(self, d, v, host, I_a):
        """Stage 2: compress the smoothed windows v_i = sum_a d_ia v_a (v_a = (K_a * W_B) at the host's randoms) in
        the metric G_aa' = int W_A^2 v_a v_a' (alpha sum_r w_r W_r v_a v_a'). Returns the projection P (n_kernels, B)
        with c = d P, the coefficients c, the window basis T (B, n_kernels), W_j = sum_a T_ja W_a, and the errors
        on I_k and on the windows."""
        v = v.astype(float)
        G = host.alpha * (v * (host.weights * host.weighted_density)[None, :]) @ v.T
        g, E = np.linalg.eigh(0.5 * (G + G.T))
        g = np.maximum(g, g.max() * 1e-14)
        L, Linv = E * np.sqrt(g)[None, :], (E / np.sqrt(g)[None, :]).T          # G = L L^T
        M = d @ L
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        I_exact = d @ I_a

        def errors(rank):                     # relative errors on the windows and on I_k, worst bin
            R = M - (U[:, :rank] * S[:rank]) @ Vt[:rank]
            return (np.max(np.linalg.norm(R, axis=1) / np.linalg.norm(M, axis=1)),
                    np.max(np.abs((R @ Linv) @ I_a) / np.abs(I_exact)))
        n_max = len(S) if self.max_basis is None else min(self.max_basis, len(S))
        rank, (err_W, err_I) = _smallest_rank(errors, n_max, self.tol)
        P = L @ Vt[:rank].T                     # (n_kernels, B): c = d P
        return P, d @ P, Vt[:rank] @ Linv, dict(err_W=float(err_W), err_I=float(err_I))

    def build(self, kedges, tracers, pairs, power_of=None, log=print):
        """Basis, coefficients, and (K_b * W_B) at the randoms of every tracer, for each pair (A, B).
        `tracers`: {name: Tracer}; `pairs`: iterable of (A, B) names; power_of(A, B): the fiducial power of a pair
        without one (e.g. GaussianCovariance.power_of). A tracer without a reference density gets its own randoms."""
        for A, B in pairs:
            key = self.pair_key(A, B)
            if key in self._pairs:
                continue
            if not self.has_density(key[1]):
                T = tracers[key[1]]
                self.add_density(key[1], T.positions, T.weights, T.alpha)
            if not self.has_power(*key):
                power = None if power_of is None else power_of(*key)
                if power is None or power.entry(0) is None:
                    raise RuntimeError(f"the kernel of {key} takes its shape from P_0: set the power spectrum (of the "
                                     "covariance, before compute_windows, or of the smoothing with set_power)")
                self.set_power(*key, power)
            t0 = time.time()
            basis, coeffs, diag = self._kernel_basis(kedges, key, log)
            near_part = taper(self.r, self.r_split, self.taper_width)
            density = self._densities[key[1]]
            kernel_values = {name: (density.convolve_by_fft(self.r, basis * (1 - near_part)[None, :], tr.positions)
                                    + density.convolve_by_neighbours(self.r, basis * near_part[None, :], tr.positions,
                                                                     self.r_split + self.taper_width)).astype(np.float32)
                             for name, tr in tracers.items()}
            host = tracers[key[0]]
            I_b = host.alpha * (kernel_values[key[0]].astype(float) @ host.weights)
            proj, wcoeffs, T, d2 = self._window_basis(coeffs, kernel_values[key[0]], host, I_b)
            self._values[key] = {n: (T @ v.astype(float)).astype(np.float32) for n, v in kernel_values.items()}
            self._pairs[key] = dict(kcoeffs=coeffs, coeffs=wcoeffs, proj=proj, I_b=I_b, n_basis=len(T),
                                    n_kernels=len(basis), basis=basis, err_IW=d2['err_I'], err_W=d2['err_W'], **diag)
            Ik = self.I_k(*key)
            local = host.alpha * float(np.sum(host.weights * host.weighted_density))
            log(f'  smoothing {key}: {len(basis)} kernels -> {len(T)} windows (errors: I_k {d2["err_I"]:.1e}, '
                f'windows {d2["err_W"]:.1e}); built in {time.time() - t0:.0f} s; I_k / int W_A W_B (first, last bin) = '
                f'{Ik[0] / local:.4f}, {Ik[-1] / local:.4f}')
        return self

    # -- persistence (the pair counts are saved by the window library; this keeps what the covariance assembly and
    #    window-convolved spectra need: coefficients and integrals)
    def save(self, path):
        arrays = {f'{name}_{n}': p[name] for n, p in enumerate(self._pairs.values()) for name in self._SAVED}
        meta = [dict(key=list(key), n=n, **{e: p[e] for e in self._ERRORS}) for n, (key, p) in enumerate(self._pairs.items())]
        cache.save(path, arrays, dict(pairs=meta, tag=self.tag))

    def load(self, path):
        arrays, meta = cache.load(path)
        self._loaded_tag = meta['tag']
        for rec in meta['pairs']:
            p = {name: arrays[f"{name}_{rec['n']}"] for name in self._SAVED}
            p.update(n_basis=int(p['coeffs'].shape[1]), n_kernels=int(p['basis'].shape[0]), **{e: rec[e] for e in self._ERRORS})
            self._pairs[tuple(rec['key'])] = p
        return self
