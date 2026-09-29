r"""Tripolar window functions Q^{omega omega'}_{Lambda1 Lambda2 Lambda}(s) from pair counts.

    Q(s_b) = alpha alpha' / int_b s^2 ds  *  sum_{r, r'} Theta_b(|s_rr'|)
             omega~(x_r) omega~'(x_r') S_{Lam1 Lam2 Lam}(x_r^, x_r'^, s_rr'^),   s = x_r' - x_r.

Count and contract
------------------
Because x' = x + s, the tripolar weight depends on the pair only through (r1, s, mu) with
r1 = |x| and mu = x^ . s^ (see harmonics.coplanar_geometry: the azimuth about s^ vanishes
identically). The estimator therefore factorises into

  * a COUNT: histogram the weighted pairs into cells of (r1, s, mu) -- the inner loop is one
    distance and one dot product, with no spherical harmonics at all;
  * a CONTRACT: evaluate S once per cell and sum.

All triples come from the same counts, and S is evaluated on the cell's weighted MEAN (r1, s, mu)
rather than its geometric centre, which makes the result correct to first order in the cell size
and the shell/mu resolution largely uncritical. The counting step is the natural place for an
external pair counter (Corrfunc/pycorr bin in exactly these variables).

Backends
--------
The counting step is dispatched on `backend`: 'numpy' is the reference implementation, 'jax' runs
the same arithmetic under XLA (fused, no temporaries, and GPU-ready), and 'auto' prefers jax when
it imports. Both produce the same histogram, so the two can be compared directly.

Sampling
--------
* far pairs (s >= s_split): all pairs of an n_sub subsample per window;
* near pairs (s <  s_split): KD-tree neighbours of a much larger n_near subsample;
* s = 0: the exact one-point anchor, which needs no pairs (Window.overlap_integral);
* s_split is snapped to a bin edge -- a bin straddling it would receive near pairs only below and
  far pairs only above while being normalised by its whole volume.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from .harmonics import tripolar_coplanar
from .tracers import Window
from .wigner import FOUR_PI

S000 = FOUR_PI ** -1.5


def _resolve_backend(name: str) -> str:
    """'auto' picks jax when it imports, otherwise numpy."""
    if name == 'numpy':
        return 'numpy'
    try:
        from . import _jax_backend  # noqa: F401
    except Exception:
        if name == 'jax':
            raise
        return 'numpy'
    return 'jax'


def _jax():
    from . import _jax_backend
    return _jax_backend


class PairHistogram:
    """Weighted pair counts on a grid of (r1, s, mu), with the weighted mean of each cell.

    `ncol` weight columns share one set of pairs (window pairs sampled by the same randoms differ
    only in their weights); the unweighted count is common to all columns.
    """

    def __init__(self, r_edges, s_edges, mu_edges, ncol: int = 1):
        self.r_edges = np.asarray(r_edges, dtype=float)
        self.s_edges = np.asarray(s_edges, dtype=float)
        self.mu_edges = np.asarray(mu_edges, dtype=float)
        self.shape = (len(self.r_edges) - 1, len(self.s_edges) - 1, len(self.mu_edges) - 1)
        self.ncol = int(ncol)
        n = int(np.prod(self.shape))
        self.w = np.zeros((self.ncol, n))      # sum of pair weights
        self.wr = np.zeros((self.ncol, n))     # sum of w * r1
        self.ws = np.zeros((self.ncol, n))     # sum of w * s
        self.wmu = np.zeros((self.ncol, n))    # sum of w * mu
        self.n = np.zeros(n)                   # unweighted count

    def add(self, x1, r1, d, w):
        """Add pairs given the primary positions, their radii, the separation vectors and weights
        (shape (n,) or (n, ncol))."""
        w = np.asarray(w, dtype=float).reshape(len(r1), -1)
        s = np.sqrt(np.einsum('ij,ij->i', d, d))
        ok = (s > 0) & (r1 > 0)
        if not np.any(ok):
            return
        x1, r1, d, w, s = x1[ok], r1[ok], d[ok], w[ok], s[ok]
        mu = np.clip(np.einsum('ij,ij->i', x1, d) / (r1 * s), -1.0, 1.0)
        ia = np.digitize(r1, self.r_edges) - 1
        ib = np.digitize(s, self.s_edges) - 1
        im = np.digitize(mu, self.mu_edges) - 1
        na, nb, nm = self.shape
        np.clip(ia, 0, na - 1, out=ia)
        np.clip(im, 0, nm - 1, out=im)
        good = (ib >= 0) & (ib < nb)
        idx = ((ia * nb + ib) * nm + im)[good]
        ln = self.n.size
        rg, sg, mg = r1[good], s[good], mu[good]
        for c in range(self.ncol):
            ww = w[good, c]
            self.w[c] += np.bincount(idx, weights=ww, minlength=ln)
            self.wr[c] += np.bincount(idx, weights=ww * rg, minlength=ln)
            self.ws[c] += np.bincount(idx, weights=ww * sg, minlength=ln)
            self.wmu[c] += np.bincount(idx, weights=ww * mg, minlength=ln)
        self.n += np.bincount(idx, minlength=ln)

    def add_arrays(self, w, wr, ws, wmu, n):
        """Absorb the accumulations produced by an accelerated backend ((ncol, ncell) and (ncell,))."""
        self.w += np.asarray(w).reshape(self.w.shape)
        self.wr += np.asarray(wr).reshape(self.w.shape)
        self.ws += np.asarray(ws).reshape(self.w.shape)
        self.wmu += np.asarray(wmu).reshape(self.w.shape)
        self.n += np.asarray(n)

    def contract(self, triples, col: int = 0):
        """sum over cells of w * S_T, per s bin, plus the weighted and unweighted counts per s bin."""
        nb = self.shape[1]
        wc = self.w[col]
        nz = np.flatnonzero(wc != 0)
        out = {t: np.zeros(nb) for t in triples}
        wsum = wc.reshape(self.shape).sum(axis=(0, 2))
        nsum = self.n.reshape(self.shape).sum(axis=(0, 2))
        if len(nz) == 0:
            return out, wsum, nsum
        w = wc[nz]
        r1 = self.wr[col][nz] / w
        s = self.ws[col][nz] / w
        mu = self.wmu[col][nz] / w
        S = tripolar_coplanar(triples, r1, s, mu)               # (n_triples, n_cells)
        ib = (nz // self.shape[2]) % nb
        for n, t in enumerate(triples):
            out[t] = np.bincount(ib, weights=w * S[n], minlength=nb)
        return out, wsum, nsum


def _n_threads(n_threads):
    if n_threads is None:
        import os
        return max(1, min(16, os.cpu_count() or 1))
    return max(1, int(n_threads))


class _Counter:
    """Pair counts of one pair of HOST catalogues, for every window pair they sample.

    Window pairs whose windows are sampled by the same randoms see exactly the same pairs (the
    subsample depends only on the host and the seed) and differ only in their weights, so one
    neighbour search and one binning serve all of them: each is one weight column.
    """

    def __init__(self, windows, opts, backend, n_threads=None):
        self.tws = windows
        self.backend = backend
        self.n_threads = _n_threads(n_threads)
        self.chunk_pairs = int(opts['chunk_pairs'])

    def _samples(self, n):
        tw0 = self.tws[0]
        pos1, _, a1 = tw0.omega.sample(n, tw0.seed)
        pos2, _, a2 = tw0.omega_p.sample(n, tw0.seed)
        w1 = np.stack([tw.omega.sample(n, tw.seed)[1] for tw in self.tws], axis=1)
        w2 = np.stack([tw.omega_p.sample(n, tw.seed)[1] for tw in self.tws], axis=1)
        return pos1, w1, a1, pos2, w2, a2

    def _fold(self, fn, blocks):
        """Run fn over the blocks, split into one contiguous group per thread; each group sums its
        results where they live (on the device for jax) and is copied back once."""
        groups = [g for g in np.array_split(np.arange(len(blocks)), self.n_threads) if len(g)]

        def run(g):
            acc = None
            for b in g:
                out = fn(blocks[b])
                if out is not None:
                    acc = out if acc is None else tuple(a + o for a, o in zip(acc, out))
            return None if acc is None else [np.asarray(a) for a in acc]
        if len(groups) == 1:
            return [run(groups[0])]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(groups)) as ex:
            return list(ex.map(run, groups))

    # ------------------------------------------------------------------ far: all pairs of n_sub
    def far(self, hist, n_sub, s_lo, s_hi, verbose):
        pos1, w1, a1, pos2, w2, a2 = self._samples(n_sub)
        r1all = np.linalg.norm(pos1, axis=1)
        n1, n2 = len(pos1), len(pos2)
        step = max(1, self.chunk_pairs // max(n2, 1))
        blocks = [(i0, min(n1, i0 + step)) for i0 in range(0, n1, step)]
        if self.backend == 'jax':
            jb = _jax()
            import jax.numpy as jnp
            p2, ww2 = jnp.asarray(pos2), jnp.asarray(w2)
            edges = tuple(jnp.asarray(e) for e in (hist.r_edges, hist.s_edges, hist.mu_edges))
            ncell = int(np.prod(hist.shape))
            uni = tuple(jb.is_uniform(e) for e in (hist.r_edges, hist.s_edges, hist.mu_edges))

            def one(b):
                i0, i1 = b
                if i1 - i0 < step:            # pad the last block: one compiled shape only
                    pad = step - (i1 - i0)
                    p1 = np.vstack([pos1[i0:i1], np.zeros((pad, 3))])
                    ww1 = np.vstack([w1[i0:i1], np.zeros((pad, w1.shape[1]))])
                    rr = np.concatenate([r1all[i0:i1], np.zeros(pad)])
                else:
                    p1, ww1, rr = pos1[i0:i1], w1[i0:i1], r1all[i0:i1]
                return jb.far_block(jnp.asarray(p1), jnp.asarray(ww1), jnp.asarray(rr), p2, ww2, *edges,
                                    hist.shape, ncell, float(s_lo), float(s_hi), uni)
        else:
            def one(b):
                i0, i1 = b
                h = PairHistogram(hist.r_edges, hist.s_edges, hist.mu_edges, hist.ncol)
                d = pos2[None, :, :] - pos1[i0:i1, None, :]
                s = np.linalg.norm(d, axis=-1)
                keep = (s >= s_lo) & (s < s_hi)
                if np.any(keep):
                    x1 = np.broadcast_to(pos1[i0:i1, None, :], d.shape)[keep]
                    r1 = np.broadcast_to(r1all[i0:i1, None], s.shape)[keep]
                    w = (w1[i0:i1, None, :] * w2[None, :, :])[keep]
                    h.add(x1, r1, d[keep], w)
                return (h.w, h.wr, h.ws, h.wmu, h.n)
        t0 = time.time()
        for out in self._fold(one, blocks):
            if out is not None:
                hist.add_arrays(*out)
        if verbose:
            print(f"  far pairs {self.tws[0].omega.key}x{self.tws[0].omega_p.key} (+{len(self.tws) - 1} "
                  f"sharing them): {n1}x{n2} in {time.time() - t0:.1f}s")
        return a1 * a2

    # ------------------------------------------------------------------ near: KD-tree neighbours
    def near(self, hist, n_near, s_split, verbose):
        pos1, w1, a1, pos2, w2, a2 = self._samples(n_near)
        r1all = np.linalg.norm(pos1, axis=1)
        tree2 = cKDTree(pos2)
        # blocks of primaries sized so that each yields ~ a few chunks of pairs
        dens = len(pos2) / max(float(np.prod(np.ptp(pos2, axis=0))), 1e-30)
        per_point = max(1.0, dens * 4.0 / 3.0 * np.pi * s_split ** 3)
        bsize = int(np.clip(4 * self.chunk_pairs / per_point, 64, 20000))
        blocks = [(i0, min(len(pos1), i0 + bsize)) for i0 in range(0, len(pos1), bsize)]
        jb = _jax() if self.backend == 'jax' else None
        if jb is not None:
            import jax.numpy as jnp
            edges = tuple(jnp.asarray(e) for e in (hist.r_edges, hist.s_edges, hist.mu_edges))
            ncell = int(np.prod(hist.shape))
            uni = tuple(jb.is_uniform(e) for e in (hist.r_edges, hist.s_edges, hist.mu_edges))
        C = self.chunk_pairs

        def one(b):
            i0, i1 = b
            # all pairs within s_split, as index arrays built in C (no per-point Python lists)
            m = cKDTree(pos1[i0:i1]).sparse_distance_matrix(tree2, s_split, output_type='ndarray')
            i = m['i'].astype(np.int64) + i0
            j = m['j'].astype(np.int64)
            if len(i) == 0:
                return None
            if jb is None:
                h = PairHistogram(hist.r_edges, hist.s_edges, hist.mu_edges, hist.ncol)
                h.add(pos1[i], r1all[i], pos2[j] - pos1[i], w1[i] * w2[j])
                return (h.w, h.wr, h.ws, h.wmu, h.n)
            acc = None                           # summed on the device
            for c0 in range(0, len(i), C):       # fixed-size, padded chunks: one compiled kernel
                ii, jj = i[c0:c0 + C], j[c0:c0 + C]
                nv = len(ii)
                x1, r1, d, w = pos1[ii], r1all[ii], pos2[jj] - pos1[ii], w1[ii] * w2[jj]
                if nv < C:
                    pad = C - nv
                    x1 = np.vstack([x1, np.zeros((pad, 3))])
                    d = np.vstack([d, np.ones((pad, 3))])
                    r1 = np.concatenate([r1, np.ones(pad)])
                    w = np.vstack([w, np.zeros((pad, w.shape[1]))])
                valid = jnp.arange(C) < nv
                out = jb.flat_pairs(jnp.asarray(x1), jnp.asarray(r1), jnp.asarray(d), jnp.asarray(w),
                                    *edges, hist.shape, ncell, valid, uni)
                acc = out if acc is None else tuple(a + o for a, o in zip(acc, out))
            return acc
        t0 = time.time()
        for out in self._fold(one, blocks):
            if out is not None:
                hist.add_arrays(*out)
        if verbose:
            print(f"  near pairs {self.tws[0].omega.key}x{self.tws[0].omega_p.key} (+{len(self.tws) - 1} "
                  f"sharing them): {len(pos1)} points in {time.time() - t0:.1f}s")
        return a1 * a2

    # ------------------------------------------------------------------ driver
    def compute(self, verbose=False):
        tw0 = self.tws[0]
        split = tw0._split_edge()
        if abs(split - tw0.s_split) > 1e-9 and verbose:
            print(f"  s_split snapped from {tw0.s_split:g} to the bin edge {split:g}")
        n_big = max(tw0.n_sub, tw0.n_near)
        p1, _, _ = tw0.omega.sample(n_big, tw0.seed)
        p2, _, _ = tw0.omega_p.sample(n_big, tw0.seed)
        r_edges = tw0._radial_edges(p1, p2)
        # the finest mu grid any member needs (a finer grid is only more accurate)
        mu_edges = np.linspace(-1.0, 1.0, max(tw.n_mu for tw in self.tws) + 1)
        K = len(self.tws)
        near = tw0.s_edges[1:] <= split + 1e-9
        far = PairHistogram(r_edges, tw0.s_edges, mu_edges, K)
        norm_f = self.far(far, tw0.n_sub, split, tw0.s_edges[-1], verbose)
        nearh, norm_n = None, None
        if np.any(near):
            # near pairs only fall in the s bins below the split: a histogram over those alone is
            # ~15x smaller (exact; a small gain in the scatter-bound binning)
            nearh = PairHistogram(r_edges, tw0.s_edges[:int(np.count_nonzero(near)) + 1], mu_edges, K)
            norm_n = self.near(nearh, tw0.n_near, split, verbose)
        for c, tw in enumerate(self.tws):
            tw._finish(split, far, norm_f, nearh, norm_n, col=c)


class TripolarWindow:
    """Pair-count estimate of Q^{omega omega'}_{Lam1 Lam2 Lam}(s) for a set of triples."""

    def __init__(self, omega: Window, omega_p: Window, triples, s_edges: np.ndarray,
                 n_sub: int = 5000, n_near: int = 200000, s_split: float = 80.0,
                 seed: int = 0, chunk_pairs: int = 200000, min_pairs: int = 20,
                 n_shells: int = 16, n_mu=None, backend: str = 'auto', n_threads=None):
        self.omega = omega
        self.omega_p = omega_p
        self.triples = sorted(set(tuple(int(x) for x in t) for t in triples))
        self.s_edges = np.asarray(s_edges, dtype=float)
        self.n_sub = int(n_sub)
        self.n_near = int(n_near)
        self.s_split = float(s_split)
        self.seed = int(seed)
        self.chunk_pairs = int(chunk_pairs)
        self.min_pairs = int(min_pairs)
        self.n_shells = int(n_shells)
        # S depends on mu through Pbar_{Lam1 m}(mu) and Pbar_{Lam2 m}(c2) with c2 ~ mu, so the mu
        # grid has to resolve oscillations of order max(Lam1, Lam2) -- roughly Lam/2 nodes over
        # [-1, 1]. Six cells per node is comfortable; too few silently biases the high multipoles
        # (with ells and L up to 4, Lam1 reaches 12, and n_mu = 20 is badly under-resolved).
        lam_max = max(max(t[0], t[1]) for t in self.triples) if self.triples else 0
        self.n_mu = int(n_mu) if n_mu is not None else max(24, 6 * lam_max)
        self.backend = _resolve_backend(backend)
        self.n_threads = n_threads
        self.Q = None          # dict triple -> array over s bins
        self.npairs = None     # unweighted pairs per s bin actually used
        self.Q0 = None         # exact value at s = 0
        self._splines = {}

    # ------------------------------------------------------------------
    @property
    def s_centers(self) -> np.ndarray:
        return 0.5 * (self.s_edges[1:] + self.s_edges[:-1])

    def _split_edge(self) -> float:
        return float(self.s_edges[int(np.argmin(np.abs(self.s_edges - self.s_split)))])

    def _radial_edges(self, *point_sets) -> np.ndarray:
        r = np.concatenate([np.linalg.norm(p, axis=1) for p in point_sets])
        lo, hi = float(r.min()), float(r.max())
        pad = 1e-6 * max(hi, 1.0)
        return np.linspace(lo - pad, hi + pad, self.n_shells + 1)

    def _opts(self):
        return dict(chunk_pairs=self.chunk_pairs)

    # ------------------------------------------------------------------ driver
    def compute(self, verbose: bool = False):
        _Counter([self], self._opts(), self.backend, self.n_threads).compute(verbose)
        return self

    def _finish(self, split, far, norm_f, nearh, norm_n, col=0):
        """Q from the (shared) histograms, column `col` being this window pair's weights."""
        self._split = split
        vol = (self.s_edges[1:] ** 3 - self.s_edges[:-1] ** 3) / 3.0
        near = self.s_edges[1:] <= split + 1e-9
        acc_f, w_f, n_f = far.contract(self.triples, col)
        acc, self.npairs, norm = acc_f, n_f.copy(), np.full(len(vol), norm_f)
        if nearh is not None and np.any(near):
            acc_n, w_n, n_n = nearh.contract(self.triples, col)
            pad = len(vol) - len(n_n)            # the near histogram covers the near bins only
            acc_n = {t: np.concatenate([v, np.zeros(pad)]) for t, v in acc_n.items()}
            w_n, n_n = np.concatenate([w_n, np.zeros(pad)]), np.concatenate([n_n, np.zeros(pad)])
            for t in self.triples:
                acc[t] = np.where(near, acc_n[t], acc_f[t])
            self.npairs[near] = n_n[near]
            norm = np.where(near, norm_n, norm_f)
        self.Q = {t: norm * acc[t] / vol for t in self.triples}

        # exact s = 0 anchor:  Q(0) = sqrt(4 pi) (-1)^Lam1 sqrt(2 Lam1 + 1) / (4 pi) int omega omega'
        ov = self.omega.overlap_integral(self.omega_p)
        self.Q0 = {(L1, L2, L): ((-1) ** L1 * math.sqrt(FOUR_PI * (2 * L1 + 1)) / FOUR_PI * ov
                                 if (L == 0 and L1 == L2) else 0.0)
                   for (L1, L2, L) in self.triples}
        self._splines = {}
        return self

    # ------------------------------------------------------------------
    def _spline(self, key):
        if key not in self._splines:
            good = (self.npairs >= self.min_pairs) if self.npairs is not None else np.ones(len(self.s_centers), bool)
            x = np.concatenate([[0.0], self.s_centers[good], [self.s_edges[-1]]])
            y0 = self.Q0[key] if self.Q0 is not None else self.Q[key][0]
            y = np.concatenate([[y0], self.Q[key][good], [0.0]])
            self._splines[key] = CubicSpline(x, y, extrapolate=False)
        return self._splines[key]

    def __call__(self, Lam1: int, Lam2: int, Lam: int, s: np.ndarray) -> np.ndarray:
        if self.Q is None:
            raise RuntimeError("call compute() first")
        return np.nan_to_num(self._spline((Lam1, Lam2, Lam))(s), nan=0.0)


class WindowLibrary:
    """Collects the (omega, omega') pairs and triples needed, computes them once, serves Q(s).

    Uses the symmetry  Q^{omega' omega}_{Lam1 Lam2 Lam} = Q^{omega omega'}_{Lam2 Lam1 Lam}
    (even multipoles) so that each unordered pair of windows is counted once.
    """

    def __init__(self, s_edges, n_sub=5000, n_near=200000, s_split=80.0, seed=0, chunk_pairs=200000,
                 min_pairs=20, n_shells=16, n_mu=None, backend='auto', n_threads=None):
        self.s_edges = np.asarray(s_edges, dtype=float)
        self.opts = dict(n_sub=n_sub, n_near=n_near, s_split=s_split, seed=seed,
                         chunk_pairs=chunk_pairs, min_pairs=min_pairs, n_shells=n_shells,
                         n_mu=n_mu, backend=backend, n_threads=n_threads)
        self._requests = {}
        self._windows = {}
        self._interp_cache = {}

    @staticmethod
    def _canonical(omega: Window, omega_p: Window):
        if omega_p.key < omega.key:
            return (omega_p.key, omega.key), True
        return (omega.key, omega_p.key), False

    def request(self, omega: Window, omega_p: Window, triples):
        key, swapped = self._canonical(omega, omega_p)
        if key not in self._requests:
            self._requests[key] = ((omega_p, omega) if swapped else (omega, omega_p), set())
        trip = self._requests[key][1]
        for (L1, L2, L) in triples:
            trip.add((L2, L1, L) if swapped else (L1, L2, L))

    def missing(self):
        return [key for key, (_, trip) in self._requests.items()
                if key not in self._windows or not set(self._windows[key].triples) >= trip]

    def compute_all(self, verbose=False):
        """Count all missing window pairs, grouped by the pair of host catalogues that samples them:
        each group shares one neighbour search and one binning (one weight column per window pair)."""
        groups = {}
        for key in self.missing():
            wins, trip = self._requests[key]
            tw = TripolarWindow(wins[0], wins[1], sorted(trip), self.s_edges, **self.opts)
            hosts = (wins[0].host.name, wins[1].host.name)
            groups.setdefault(hosts, []).append((key, tw))
        for hosts, members in groups.items():
            if verbose:
                print(f"pair counts for hosts {hosts}: {len(members)} window pairs "
                      f"({', '.join(str(k) for k, _ in members)})")
            tws = [tw for _, tw in members]
            _Counter(tws, tws[0]._opts(), tws[0].backend, self.opts.get('n_threads')).compute(verbose)
            for key, tw in members:
                self._windows[key] = tw
        self._interp_cache = {}

    def get(self, omega: Window, omega_p: Window, Lam1: int, Lam2: int, Lam: int, s: np.ndarray) -> np.ndarray:
        key, swapped = self._canonical(omega, omega_p)
        ckey = (key, Lam1, Lam2, Lam, id(s))
        if ckey not in self._interp_cache:
            tw = self._windows[key]
            if swapped:
                Lam1, Lam2 = Lam2, Lam1
            self._interp_cache[ckey] = tw(Lam1, Lam2, Lam, s)
        return self._interp_cache[ckey]

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Store the pair-count results (the expensive, cosmology-independent part)."""
        data = {'s_edges': self.s_edges}
        meta = []
        for w, (key, tw) in enumerate(self._windows.items()):
            np_name = f"npairs_{w}"
            data[np_name] = np.asarray(tw.npairs if tw.npairs is not None else [])
            for trip, Q in tw.Q.items():
                q_name = f"Q_{len(meta)}"
                data[q_name] = np.asarray(Q)
                meta.append({'key': [list(k) for k in key], 'trip': [int(x) for x in trip],
                             'q': q_name, 'npairs': np_name,
                             'q0': float(tw.Q0[trip]) if tw.Q0 is not None else 0.0})
        data['meta'] = np.array(json.dumps(meta))
        np.savez(path, **data)

    def load(self, path):
        """Load pair-count results saved with save(); windows are matched by their names."""
        with np.load(path) as f:
            if len(f['s_edges']) != len(self.s_edges) or not np.allclose(f['s_edges'], self.s_edges):
                raise ValueError("s_edges of the stored windows differ from the current ones")
            meta = json.loads(str(f['meta'].item()))
            groups, q0s, npairs = {}, {}, {}
            for rec in meta:
                key = tuple(tuple(k) for k in rec['key'])
                trip = tuple(rec['trip'])
                groups.setdefault(key, {})[trip] = f[rec['q']]
                q0s.setdefault(key, {})[trip] = rec['q0']
                if key not in npairs:
                    arr = f[rec['npairs']]
                    npairs[key] = arr if arr.size else None
        for key, Qs in groups.items():
            tw = TripolarWindow(None, None, list(Qs), self.s_edges, **self.opts)
            tw.Q, tw.Q0, tw.npairs = Qs, q0s[key], npairs.get(key)
            self._windows[key] = tw
        self._interp_cache = {}
        return self
