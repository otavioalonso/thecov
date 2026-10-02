"""Pair-averaged clustering window for thecov (isotropic kernel).

The clustering part of the Gaussian covariance contains m(x) m(x + r) weighted by xi(r) e^{ik.r}, which
the local approximation replaces by m(x)^2. Keeping the partner inside the r integral gives (see
report/window_kernel.pdf)

    W_k(x) = m(x) (K_k * m)(x),   K_k(r) = xi(r) e^{ik.r} / P(k),   int K_k = 1,

and the mean of the estimator <P_hat(k)> = P(k) int W_k / norm. In thecov only the value of m at each
random (NW) changes: NW_r = (K_k * m)(x_r), so that alpha sum_r w_r NW_r = int m (K_k * m) = I_k, which
is also what thecov's model normalisation (norm / I) then uses.

Isotropic kernel: angle-averaged K_k(r) = xi_0(r) j_0(k r) / P_0(k), averaged over a k-band, from the
shape of the model monopole (the amplitude cancels). (K * m) is computed by FFT: m is painted (CIC) from
many random files on a zero-padded mesh, multiplied by the kernel's Fourier transform (with the CIC
painting and reading windows compensated) and read back (CIC) at thecov's randoms.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.fft as sfft
from scipy.special import spherical_jn


# ----------------------------------------------------------------------------- xi and kernels
def extend_power(k, p, k_max=5.0, damping=1.0, n_low=1.0):
    """P0 on a wide k range from its values on [k0, k1]: (k/k0)^n_low below k0 (n_low = 1, the large-
    scale slope; only the kernel's far tail cares), a power law fitted to the last 8 points above k1,
    and a Gaussian damping exp(-(k damping)^2) [damping in Mpc/h] that regularises xi at r -> 0."""
    k, p = np.asarray(k, float), np.asarray(p, float)
    kk = np.linspace(1e-4, k_max, 20001)
    tail = slice(max(0, len(k) - 8), len(k))
    slope = np.polyfit(np.log(k[tail]), np.log(np.abs(p[tail])), 1)[0]
    out = np.interp(kk, k, p)
    out = np.where(kk < k[0], p[0] * (kk / k[0]) ** n_low, out)
    out = np.where(kk > k[-1], p[-1] * (kk / k[-1]) ** slope, out)
    return kk, out * np.exp(-(kk * damping) ** 2)


def xi0_from_power(kk, pk, r):
    """xi_0(r) = int k^2 dk / (2 pi^2) P(k) j_0(k r), trapezoid on the dense grid kk."""
    xi = np.empty(len(r))
    for i in range(0, len(r), 64):
        rr = r[i:i + 64, None]
        xi[i:i + 64] = np.trapezoid(kk ** 2 * pk * np.sinc(kk * rr / np.pi), kk, axis=1) / (2 * np.pi ** 2)
    return xi


def band_kernel(r, xi, k_lo, k_hi, nk=64):
    """Radial kernel of a k-band: xi_0(r) <j_0(k r)>_band (mode-weighted, k^2), normalised so that
    int 4 pi r^2 K dr = 1 over the tabulated r (i.e. truncated at r.max()). Returns (K, raw norm)."""
    kb = np.linspace(k_lo, k_hi, nk)
    wk = kb ** 2 / np.sum(kb ** 2)
    j0 = np.sum(wk[:, None] * spherical_jn(0, kb[:, None] * r[None, :]), axis=0)
    K = xi * j0
    norm = np.trapezoid(4 * np.pi * r ** 2 * K, r)
    return K / norm, norm


def kernel_fourier(r, K, q):
    """K~(q) = int 4 pi r^2 K(r) j_0(q r) dr."""
    out = np.empty(len(q))
    for i in range(0, len(q), 256):
        qq = q[i:i + 256, None]
        out[i:i + 256] = np.trapezoid(4 * np.pi * r ** 2 * K * np.sinc(qq * r / np.pi), r, axis=1)
    return out


# ----------------------------------------------------------------------------- smoothing
class KernelSmoother:
    """m painted from (positions, weights) with alpha: m = alpha sum_r w_r delta_D, on a mesh padded by
    `rmax` (so that the kernel does not wrap around), and its convolution with radial kernels."""

    def __init__(self, pos, w, alpha, cell=5.0, rmax=200.0, workers=None, log=print):
        lo, hi = pos.min(0) - rmax - 2 * cell, pos.max(0) + rmax + 2 * cell
        self.cell, self.lo = float(cell), lo
        self.shape = tuple(sfft.next_fast_len(int(np.ceil((h - l) / cell)) + 1) for l, h in zip(lo, hi))
        self.workers = workers or os.cpu_count()
        mesh = np.zeros(self.shape, np.float32)
        self._cic_paint(mesh, pos, alpha * w / cell ** 3)
        self.mk = sfft.rfftn(mesh, workers=self.workers)
        del mesh
        kx = 2 * np.pi * np.fft.fftfreq(self.shape[0], d=cell)
        ky = 2 * np.pi * np.fft.fftfreq(self.shape[1], d=cell)
        kz = 2 * np.pi * np.fft.rfftfreq(self.shape[2], d=cell)
        self.k1 = (kx, ky, kz)
        log(f'  kernel mesh {self.shape} ({np.prod(self.shape) / 1e6:.0f}M cells, cell {cell} Mpc/h, padding {rmax} Mpc/h)')

    def _cic_weights(self, pos):
        u = (pos - self.lo) / self.cell
        i0 = np.floor(u).astype(np.int64)
        f = u - i0
        return i0, f

    def _cic_paint(self, mesh, pos, val, chunk=2_000_000):
        flat = mesh.reshape(-1)
        for s in range(0, len(pos), chunk):
            i0, f = self._cic_weights(pos[s:s + chunk])
            v = val[s:s + chunk]
            for dx in (0, 1):
                wx = f[:, 0] if dx else 1 - f[:, 0]
                for dy in (0, 1):
                    wy = f[:, 1] if dy else 1 - f[:, 1]
                    for dz in (0, 1):
                        wz = f[:, 2] if dz else 1 - f[:, 2]
                        idx = np.ravel_multi_index((i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz), self.shape)
                        flat += np.bincount(idx, weights=v * wx * wy * wz, minlength=flat.size).astype(np.float32)

    def _cic_read(self, mesh, pos, chunk=2_000_000):
        flat = mesh.reshape(-1)
        out = np.zeros(len(pos))
        for s in range(0, len(pos), chunk):
            i0, f = self._cic_weights(pos[s:s + chunk])
            acc = np.zeros(len(i0))
            for dx in (0, 1):
                wx = f[:, 0] if dx else 1 - f[:, 0]
                for dy in (0, 1):
                    wy = f[:, 1] if dy else 1 - f[:, 1]
                    for dz in (0, 1):
                        wz = f[:, 2] if dz else 1 - f[:, 2]
                        idx = np.ravel_multi_index((i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz), self.shape)
                        acc += flat[idx] * wx * wy * wz
            out[s:s + chunk] = acc
        return out

    def smooth_at(self, r, K, positions):
        """(K * m) at `positions`, K a radial kernel tabulated on r."""
        kx, ky, kz = self.k1
        kmax = np.sqrt(kx.max() ** 2 + ky.max() ** 2 + kz.max() ** 2) * 1.01
        q = np.linspace(0, kmax, 4096)
        Kq = kernel_fourier(r, K, q)
        out = np.empty(self.mk.shape, np.complex64)
        half = self.cell / 2
        wz = np.sinc(kz * half / np.pi) ** 2                    # CIC window per axis
        wz = np.maximum(wz, 0.1)
        for i, kxi in enumerate(kx):                           # slab by slab (memory)
            kk = np.sqrt(kxi ** 2 + ky[:, None] ** 2 + kz[None, :] ** 2)
            wcic = (max(np.sinc(kxi * half / np.pi) ** 2, 0.1) * np.maximum(np.sinc(ky * half / np.pi) ** 2, 0.1)[:, None]
                    * wz[None, :])
            out[i] = self.mk[i] * (np.interp(kk, q, Kq) / wcic ** 2)   # paint + read compensation
        field = sfft.irfftn(out, s=self.shape, workers=self.workers).astype(np.float32)
        del out
        return self._cic_read(field, positions)


# ----------------------------------------------------------------------------- per region
def region_kernels(k, p0, band_edges, rmax=200.0, dr=0.25, damping=1.0):
    """Normalised radial kernels for each k-band from the model monopole shape (k, p0)."""
    kk, pk = extend_power(k, p0, damping=damping)
    r = np.arange(dr / 2, rmax, dr)
    xi = xi0_from_power(kk, pk, r)
    kernels, norms = [], []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        Kb, nb = band_kernel(r, xi, lo, hi)
        kernels.append(Kb)
        # raw normalisation vs the band power: ~1 if the truncated xi transform recovers P (diagnostic)
        sel = (kk >= lo) & (kk <= hi)
        norms.append(float(nb / np.mean(pk[sel])))
    return r, kernels, norms


def effective_width(r, K):
    """rms radius of the kernel, sqrt(int 4 pi r^4 K / int 4 pi r^2 K) [Mpc/h] (diagnostic)."""
    return float(np.sqrt(abs(np.trapezoid(4 * np.pi * r ** 4 * K, r) / np.trapezoid(4 * np.pi * r ** 2 * K, r))))
