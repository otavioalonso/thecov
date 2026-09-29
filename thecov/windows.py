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


def _cell_tasks(pos1, pos2, r, same):
    """Pairs of cubes (side >= r) that can hold pairs closer than r, for a cell-list search.

    Returns (cubes1, cubes2, tasks): point indices per occupied cube and the (a, b) cube pairs to
    search. With `same` (one catalogue) only half the neighbour stencil is listed, so every
    unordered pair of points is found exactly once; otherwise the full stencil.
    """
    lo = np.minimum(pos1.min(axis=0), pos2.min(axis=0))

    def cubes(pos):
        ijk = np.floor((pos - lo) / r).astype(np.int64)
        keys = ijk[:, 0] * 1_000_003 ** 2 + ijk[:, 1] * 1_000_003 + ijk[:, 2]
        order = np.argsort(keys, kind='stable')
        uk, start = np.unique(keys[order], return_index=True)
        stop = np.append(start[1:], len(order))
        out = {}
        for k, a, b in zip(uk, start, stop):
            c = order[a]
            out[tuple(ijk[c])] = order[a:b]
        return out
    c1 = cubes(pos1)
    c2 = c1 if same else cubes(pos2)
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    if same:
        offsets = [o for o in offsets if o > (0, 0, 0)]     # half stencil; o = 0 handled apart
    tasks = []
    for a in c1:
        if same:
            tasks.append((a, a))
        for o in offsets:
            b = (a[0] + o[0], a[1] + o[1], a[2] + o[2])
            if b in c2:
                tasks.append((a, b))
    return c1, c2, tasks


class _Counter:
    """Pair counts of one UNORDERED pair of host catalogues, for every window pair they sample.

    Window pairs whose windows are sampled by the same randoms see exactly the same pairs (the
    subsample depends only on the host and the seed) and differ only in their weights, so one
    neighbour search and one binning serve all of them: each is one weight column. Each pair of
    points is also found only once and binned from both ends: for hosts (H1, H2) the window pairs
    with omega on H1 ("forward") take (x1 = x_i, s = x_j - x_i), those with omega on H2 ("reverse")
    take (x1 = x_j, s = x_i - x_j); for a single host, every unordered pair feeds both orientations
    of the same histogram, which is exactly the ordered-pair count.
    """

    def __init__(self, fwd, rev, opts, backend, n_threads=None):
        self.fwd, self.rev = list(fwd), list(rev)
        self.same = (self.fwd[0].omega.host is self.fwd[0].omega_p.host) if self.fwd else False
        if self.same and self.rev:
            raise ValueError("a single-host group has no reverse members")
        self.tws = self.fwd + self.rev
        self.backend = backend
        self.n_threads = _n_threads(n_threads)
        self.chunk_pairs = int(opts['chunk_pairs'])

    def _samples(self, n):
        """Positions, tilde weights (forward: omega on H1 / omega' on H2; reverse: omega on H2 /
        omega' on H1) and alpha_eff of the two host subsamples."""
        tw0 = self.fwd[0] if self.fwd else self.rev[0]
        h1 = tw0.omega if self.fwd else tw0.omega_p        # a window hosted by H1
        h2 = tw0.omega_p if self.fwd else tw0.omega
        pos1, _, a1 = h1.sample(n, tw0.seed)
        pos2, _, a2 = h2.sample(n, tw0.seed)

        def stack(ws, n1):
            return np.stack([w.sample(n, tw0.seed)[1] for w in ws], axis=1) if ws else np.zeros((n1, 0))
        wf1 = stack([tw.omega for tw in self.fwd], len(pos1))
        wf2 = stack([tw.omega_p for tw in self.fwd], len(pos2))
        wr1 = stack([tw.omega for tw in self.rev], len(pos2))       # hosted by H2
        wr2 = stack([tw.omega_p for tw in self.rev], len(pos1))     # hosted by H1
        return pos1, pos2, wf1, wf2, wr1, wr2, a1 * a2

    def _fold(self, fn, blocks):
        """Run fn over the blocks, split into one contiguous group per thread; each group sums its
        results where they live (on the device for jax) and is copied back once. fn may keep
        state per group through the `finish` attribute of the returned closure factory."""
        groups = [g for g in np.array_split(np.arange(len(blocks)), self.n_threads) if len(g)]

        def run(g):
            worker = fn()
            for b in g:
                worker.feed(blocks[b])
            return worker.result()
        if len(groups) == 1:
            return [run(groups[0])]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(groups)) as ex:
            return list(ex.map(run, groups))

    def _kernel_setup(self, hist):
        jb = _jax()
        import jax.numpy as jnp
        edges = tuple(jnp.asarray(e) for e in (hist.r_edges, hist.s_edges, hist.mu_edges))
        uni = tuple(jb.is_uniform(e) for e in (hist.r_edges, hist.s_edges, hist.mu_edges))
        return jb, jnp, edges, uni, int(np.prod(hist.shape))

    # ------------------------------------------------------------------ far: all pairs of n_sub
    def _far_one_orientation(self, hist, pos1, w1, pos2, w2, s_lo, s_hi):
        """All ordered pairs (i in pos1, j in pos2), x1 = pos1[i]; the brute force is cheap."""
        r1all = np.linalg.norm(pos1, axis=1)
        n1, n2 = len(pos1), len(pos2)
        step = max(1, self.chunk_pairs // max(n2, 1))
        blocks = [(i0, min(n1, i0 + step)) for i0 in range(0, n1, step)]
        jax_ = self.backend == 'jax'
        if jax_:
            jb, jnp, edges, uni, ncell = self._kernel_setup(hist)
            p2, ww2 = jnp.asarray(pos2), jnp.asarray(w2)
        counter = self

        class Worker:
            def __init__(self):
                self.acc = None

            def feed(self, b):
                i0, i1 = b
                if jax_:
                    if i1 - i0 < step:        # pad the last block: one compiled shape only
                        pad = step - (i1 - i0)
                        p1 = np.vstack([pos1[i0:i1], np.zeros((pad, 3))])
                        ww1 = np.vstack([w1[i0:i1], np.zeros((pad, w1.shape[1]))])
                        rr = np.concatenate([r1all[i0:i1], np.zeros(pad)])
                    else:
                        p1, ww1, rr = pos1[i0:i1], w1[i0:i1], r1all[i0:i1]
                    out = jb.far_block(jnp.asarray(p1), jnp.asarray(ww1), jnp.asarray(rr), p2, ww2, *edges,
                                       hist.shape, ncell, float(s_lo), float(s_hi), uni)
                else:
                    h = PairHistogram(hist.r_edges, hist.s_edges, hist.mu_edges, hist.ncol)
                    d = pos2[None, :, :] - pos1[i0:i1, None, :]
                    s = np.linalg.norm(d, axis=-1)
                    keep = (s >= s_lo) & (s < s_hi)
                    if np.any(keep):
                        x1 = np.broadcast_to(pos1[i0:i1, None, :], d.shape)[keep]
                        r1 = np.broadcast_to(r1all[i0:i1, None], s.shape)[keep]
                        w = (w1[i0:i1, None, :] * w2[None, :, :])[keep]
                        h.add(x1, r1, d[keep], w)
                    out = (h.w, h.wr, h.ws, h.wmu, h.n)
                self.acc = out if self.acc is None else tuple(a + o for a, o in zip(self.acc, out))

            def result(self):
                return None if self.acc is None else [np.asarray(a) for a in self.acc]
        for out in counter._fold(Worker, blocks):
            if out is not None:
                hist.add_arrays(*out)

    def far(self, hf, hr, n_sub, s_lo, s_hi, verbose):
        pos1, pos2, wf1, wf2, wr1, wr2, norm = self._samples(n_sub)
        t0 = time.time()
        if self.fwd:
            self._far_one_orientation(hf, pos1, wf1, pos2, wf2, s_lo, s_hi)
        if self.rev:
            self._far_one_orientation(hr, pos2, wr1, pos1, wr2, s_lo, s_hi)
        if verbose:
            print(f"  far pairs {self._label()}: {len(pos1)}x{len(pos2)} in {time.time() - t0:.1f}s")
        return norm

    # ------------------------------------------------------------------ near: cell list + KD-trees
    def near(self, hf, hr, n_near, s_split, verbose):
        pos1, pos2, wf1, wf2, wr1, wr2, norm = self._samples(n_near)
        r1all = np.linalg.norm(pos1, axis=1)
        r2all = np.linalg.norm(pos2, axis=1)
        c1, c2, tasks = _cell_tasks(pos1, pos2, s_split, self.same)
        trees1 = {a: cKDTree(pos1[ix]) for a, ix in c1.items()}
        trees2 = trees1 if self.same else {b: cKDTree(pos2[ix]) for b, ix in c2.items()}
        # biggest cube pairs first, so that the threads finish together
        tasks.sort(key=lambda t: -len(c1[t[0]]) * len(c2[t[1]]))
        tasks = [tasks[i::self.n_threads] for i in range(self.n_threads)]
        tasks = [t for grp in tasks for t in grp]
        C = self.chunk_pairs
        jax_ = self.backend == 'jax'
        if jax_:
            jb, jnp, edges, uni, ncell = self._kernel_setup(hf if self.fwd else hr)
        same, fwd, rev = self.same, bool(self.fwd), bool(self.rev)

        def pairs(t):
            a, b = t
            if same and a == b:
                m = trees1[a].query_pairs(s_split, output_type='ndarray')
                if len(m) == 0:
                    return None
                return c1[a][m[:, 0]], c1[a][m[:, 1]]
            m = trees1[a].sparse_distance_matrix(trees2[b], s_split, output_type='ndarray')
            if len(m) == 0:
                return None
            return c1[a][m['i']], c2[b][m['j']]

        if jax_:
            # resident on the device for the whole group; only index pairs are sent per chunk
            dev = dict(p1=jnp.asarray(pos1), r1=jnp.asarray(r1all), p2=jnp.asarray(pos2), r2=jnp.asarray(r2all),
                       wf1=jnp.asarray(wf1), wf2=jnp.asarray(wf2), wr1=jnp.asarray(wr1), wr2=jnp.asarray(wr2))

            def zeros(hist):
                K = hist.ncol
                return (jnp.zeros((K, ncell)), jnp.zeros((K, ncell)), jnp.zeros((K, ncell)),
                        jnp.zeros((K, ncell)), jnp.zeros(ncell))

        class Worker:
            """Buffers pairs across cube pairs and bins them in fixed-size chunks."""

            def __init__(self):
                self.bi, self.bj, self.n = [], [], 0
                if jax_:
                    self.acc_f = zeros(hf) if fwd else None
                    self.acc_r = zeros(hr) if rev else None
                else:
                    self.acc_f = PairHistogram(hf.r_edges, hf.s_edges, hf.mu_edges, hf.ncol) if fwd else None
                    self.acc_r = PairHistogram(hr.r_edges, hr.s_edges, hr.mu_edges, hr.ncol) if rev else None

            def _dispatch(self, i, j):
                if not jax_:
                    d = pos2[j] - pos1[i]
                    if fwd:
                        self.acc_f.add(pos1[i], r1all[i], d, wf1[i] * wf2[j])
                        if same:      # the same unordered pair seen from its other end
                            self.acc_f.add(pos1[j], r1all[j], -d, wf1[j] * wf2[i])
                    if rev:
                        self.acc_r.add(pos2[j], r2all[j], -d, wr1[j] * wr2[i])
                    return
                nv = len(i)
                if nv < C:
                    i = np.concatenate([i, np.zeros(C - nv, dtype=i.dtype)])
                    j = np.concatenate([j, np.zeros(C - nv, dtype=j.dtype)])
                ii, jj = jnp.asarray(i.astype(np.int32)), jnp.asarray(j.astype(np.int32))
                valid = jnp.arange(C) < nv
                args = (hf.shape if fwd else hr.shape, ncell, valid, uni)
                if fwd:
                    self.acc_f = jb.indexed_pairs(dev['p1'], dev['r1'], dev['wf1'], dev['p2'], dev['wf2'], ii, jj,
                                                  *edges, *args, self.acc_f)
                    if same:      # the same unordered pair seen from its other end
                        self.acc_f = jb.indexed_pairs(dev['p1'], dev['r1'], dev['wf1'], dev['p2'], dev['wf2'], jj, ii,
                                                      *edges, *args, self.acc_f)
                if rev:
                    self.acc_r = jb.indexed_pairs(dev['p2'], dev['r2'], dev['wr1'], dev['p1'], dev['wr2'], jj, ii,
                                                  *edges, *args, self.acc_r)

            def _drain(self, final=False):
                if self.n == 0:
                    return
                i, j = np.concatenate(self.bi), np.concatenate(self.bj)
                full = (len(i) // C) * C
                for c0 in range(0, full, C):
                    self._dispatch(i[c0:c0 + C], j[c0:c0 + C])
                i, j = i[full:], j[full:]
                if final and len(i):
                    self._dispatch(i, j)
                    i, j = i[:0], j[:0]
                self.bi, self.bj, self.n = [i], [j], len(i)

            def feed(self, t):
                p = pairs(t)
                if p is None:
                    return
                self.bi.append(p[0]); self.bj.append(p[1]); self.n += len(p[0])
                if self.n >= C:
                    self._drain()

            def result(self):
                self._drain(final=True)

                def conv(acc):
                    if acc is None:
                        return None
                    if isinstance(acc, PairHistogram):
                        return [acc.w, acc.wr, acc.ws, acc.wmu, acc.n]
                    return [np.asarray(a) for a in acc]
                return conv(self.acc_f), conv(self.acc_r)
        t0 = time.time()
        for out_f, out_r in self._fold(Worker, tasks):
            if out_f is not None:
                hf.add_arrays(*out_f)
            if out_r is not None:
                hr.add_arrays(*out_r)
        if verbose:
            print(f"  near pairs {self._label()}: {len(pos1)}x{len(pos2)} points, {len(tasks)} cube pairs "
                  f"in {time.time() - t0:.1f}s")
        return norm

    def _label(self):
        names = [f"{tw.omega.key}x{tw.omega_p.key}" for tw in self.fwd] + \
                [f"{tw.omega.key}x{tw.omega_p.key} (reversed)" for tw in self.rev]
        return f"{names[0]} (+{len(names) - 1} sharing them)"

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
        near = tw0.s_edges[1:] <= split + 1e-9

        def hists(s_edges):
            return (PairHistogram(r_edges, s_edges, mu_edges, max(len(self.fwd), 1)),
                    PairHistogram(r_edges, s_edges, mu_edges, max(len(self.rev), 1)))
        far_f, far_r = hists(tw0.s_edges)
        norm_f = self.far(far_f, far_r, tw0.n_sub, split, tw0.s_edges[-1], verbose)
        near_f = near_r = norm_n = None
        if np.any(near):
            # near pairs only ever fall in the s bins below the split: a histogram over those alone
            # is ~15x smaller and stays in cache, which is what the scatter-adds are bound by
            n_nb = int(np.count_nonzero(near))
            near_f, near_r = hists(tw0.s_edges[:n_nb + 1])
            norm_n = self.near(near_f, near_r, tw0.n_near, split, verbose)
        for c, tw in enumerate(self.fwd):
            tw._finish(split, far_f, norm_f, near_f, norm_n, col=c)
        for c, tw in enumerate(self.rev):
            tw._finish(split, far_r, norm_f, near_r, norm_n, col=c)


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
        _Counter([self], [], self._opts(), self.backend, self.n_threads).compute(verbose)
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
        """Count all missing window pairs, grouped by the UNORDERED pair of host catalogues that
        samples them: each group shares one neighbour search and one binning (a weight column per
        window pair), and each pair of points is found once and binned from both ends."""
        groups = {}
        for key in self.missing():
            wins, trip = self._requests[key]
            tw = TripolarWindow(wins[0], wins[1], sorted(trip), self.s_edges, **self.opts)
            h1, h2 = wins[0].host.name, wins[1].host.name
            hosts = tuple(sorted((h1, h2)))
            g = groups.setdefault(hosts, ([], []))
            (g[0] if h1 == hosts[0] else g[1]).append((key, tw))
        for hosts, (fwd, rev) in groups.items():
            if verbose:
                print(f"pair counts for hosts {hosts}: {len(fwd) + len(rev)} window pairs")
            tws = [tw for _, tw in fwd + rev]
            _Counter([tw for _, tw in fwd], [tw for _, tw in rev], tws[0]._opts(), tws[0].backend,
                     self.opts.get('n_threads')).compute(verbose)
            for key, tw in fwd + rev:
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
