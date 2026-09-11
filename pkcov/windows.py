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
    """Weighted pair counts on a grid of (r1, s, mu), with the weighted mean of each cell."""

    def __init__(self, r_edges, s_edges, mu_edges):
        self.r_edges = np.asarray(r_edges, dtype=float)
        self.s_edges = np.asarray(s_edges, dtype=float)
        self.mu_edges = np.asarray(mu_edges, dtype=float)
        self.shape = (len(self.r_edges) - 1, len(self.s_edges) - 1, len(self.mu_edges) - 1)
        n = int(np.prod(self.shape))
        self.w = np.zeros(n)        # sum of pair weights
        self.wr = np.zeros(n)       # sum of w * r1
        self.ws = np.zeros(n)       # sum of w * s
        self.wmu = np.zeros(n)      # sum of w * mu
        self.n = np.zeros(n)        # unweighted count

    def add(self, x1, r1, d, w):
        """Add pairs given the primary positions, their radii, the separation vectors and weights."""
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
        ww = w[good]
        ln = self.w.size
        self.w += np.bincount(idx, weights=ww, minlength=ln)
        self.wr += np.bincount(idx, weights=ww * r1[good], minlength=ln)
        self.ws += np.bincount(idx, weights=ww * s[good], minlength=ln)
        self.wmu += np.bincount(idx, weights=ww * mu[good], minlength=ln)
        self.n += np.bincount(idx, minlength=ln)

    def add_arrays(self, w, wr, ws, wmu, n):
        """Absorb the five accumulations produced by an accelerated backend."""
        self.w += np.asarray(w)
        self.wr += np.asarray(wr)
        self.ws += np.asarray(ws)
        self.wmu += np.asarray(wmu)
        self.n += np.asarray(n)

    def contract(self, triples):
        """sum over cells of w * S_T, per s bin, plus the weighted and unweighted counts per s bin."""
        nb = self.shape[1]
        nz = np.flatnonzero(self.w != 0)
        out = {t: np.zeros(nb) for t in triples}
        wsum = self.w.reshape(self.shape).sum(axis=(0, 2))
        nsum = self.n.reshape(self.shape).sum(axis=(0, 2))
        if len(nz) == 0:
            return out, wsum, nsum
        w = self.w[nz]
        r1 = self.wr[nz] / w
        s = self.ws[nz] / w
        mu = self.wmu[nz] / w
        S = tripolar_coplanar(triples, r1, s, mu)               # (n_triples, n_cells)
        ib = (nz // self.shape[2]) % nb
        for n, t in enumerate(triples):
            out[t] = np.bincount(ib, weights=w * S[n], minlength=nb)
        return out, wsum, nsum


class _PadBuffer:
    """Feeds variable-length pair lists to a jitted kernel in fixed-size chunks.

    jit specialises on shape, so a kernel called with every neighbour-list length would be
    recompiled constantly. Pairs are buffered and dispatched in blocks of exactly `size`, the tail
    padded with zero-weight entries that the kernel masks out.
    """

    def __init__(self, size, hist, jb):
        self.size, self.hist, self.jb = int(size), hist, jb
        import jax.numpy as jnp
        self.jnp = jnp
        self.edges = (jnp.asarray(hist.r_edges), jnp.asarray(hist.s_edges), jnp.asarray(hist.mu_edges))
        self.ncell = int(np.prod(hist.shape))
        self.uni = tuple(jb.is_uniform(e) for e in (hist.r_edges, hist.s_edges, hist.mu_edges))
        self._x1, self._r1, self._d, self._w = [], [], [], []
        self._n = 0

    def push(self, x1, r1, d, w):
        self._x1.append(x1); self._r1.append(r1); self._d.append(d); self._w.append(w)
        self._n += len(w)
        while self._n >= self.size:
            self._emit(self.size)

    def _emit(self, take):
        x1 = np.concatenate(self._x1); r1 = np.concatenate(self._r1)
        d = np.concatenate(self._d); w = np.concatenate(self._w)
        head, tail = slice(0, take), slice(take, None)
        self._dispatch(x1[head], r1[head], d[head], w[head], take)
        self._x1, self._r1, self._d, self._w = [x1[tail]], [r1[tail]], [d[tail]], [w[tail]]
        self._n = len(w) - take

    def _dispatch(self, x1, r1, d, w, nvalid):
        jnp = self.jnp
        pad = self.size - len(w)
        if pad > 0:
            x1 = np.vstack([x1, np.zeros((pad, 3))])
            d = np.vstack([d, np.ones((pad, 3))])
            r1 = np.concatenate([r1, np.ones(pad)])
            w = np.concatenate([w, np.zeros(pad)])
        valid = jnp.arange(self.size) < nvalid
        out = self.jb.flat_pairs(jnp.asarray(x1), jnp.asarray(r1), jnp.asarray(d), jnp.asarray(w),
                                 *self.edges, self.hist.shape, self.ncell, valid, self.uni)
        self.hist.add_arrays(*[np.asarray(a) for a in out])

    def flush(self):
        if self._n > 0:
            n = self._n
            self._emit(n)


class TripolarWindow:
    """Pair-count estimate of Q^{omega omega'}_{Lam1 Lam2 Lam}(s) for a set of triples."""

    def __init__(self, omega: Window, omega_p: Window, triples, s_edges: np.ndarray,
                 n_sub: int = 5000, n_near: int = 200000, s_split: float = 80.0,
                 seed: int = 0, chunk_pairs: int = 200000, min_pairs: int = 20,
                 n_shells: int = 16, n_mu=None, backend: str = 'auto'):
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

    # ------------------------------------------------------------------ counting
    def _far_pairs(self, hist, verbose):
        pos1, tw1, a1 = self.omega.sample(self.n_sub, self.seed)
        pos2, tw2, a2 = self.omega_p.sample(self.n_sub, self.seed)
        r1all = np.linalg.norm(pos1, axis=1)
        n1, n2 = len(pos1), len(pos2)
        step = max(1, self.chunk_pairs // max(n2, 1))
        t0 = time.time()
        if self.backend == 'jax':
            jb = _jax()
            import jax.numpy as jnp
            p2 = jnp.asarray(pos2)
            w2 = jnp.asarray(tw2)
            re, se, me = (jnp.asarray(hist.r_edges), jnp.asarray(hist.s_edges), jnp.asarray(hist.mu_edges))
            ncell = int(np.prod(hist.shape))
            uni = tuple(jb.is_uniform(e) for e in (hist.r_edges, hist.s_edges, hist.mu_edges))
            acc = None
            for i0 in range(0, n1, step):
                i1 = min(n1, i0 + step)
                out = jb.far_block(jnp.asarray(pos1[i0:i1]), jnp.asarray(tw1[i0:i1]),
                                   jnp.asarray(r1all[i0:i1]), p2, w2, re, se, me,
                                   hist.shape, ncell, float(self._split), float(self.s_edges[-1]), uni)
                acc = out if acc is None else tuple(a + b for a, b in zip(acc, out))
                if verbose and (i0 // step) % 50 == 0:
                    print(f"  far pairs [jax] {self.omega.key} x {self.omega_p.key}: {i1}/{n1}, {time.time() - t0:.1f}s")
            if acc is not None:
                hist.add_arrays(*[np.asarray(a) for a in acc])
            return a1 * a2
        for i0 in range(0, n1, step):
            i1 = min(n1, i0 + step)
            d = pos2[None, :, :] - pos1[i0:i1, None, :]
            s = np.linalg.norm(d, axis=-1)
            keep = (s >= self._split) & (s < self.s_edges[-1])
            if not np.any(keep):
                continue
            x1 = np.broadcast_to(pos1[i0:i1, None, :], d.shape)[keep]
            r1 = np.broadcast_to(r1all[i0:i1, None], s.shape)[keep]
            w = (tw1[i0:i1, None] * tw2[None, :])[keep]
            hist.add(x1, r1, d[keep], w)
            if verbose and (i0 // step) % 50 == 0:
                print(f"  far pairs {self.omega.key} x {self.omega_p.key}: {i1}/{n1}, {time.time() - t0:.1f}s")
        return a1 * a2

    def _near_pairs(self, hist, verbose):
        pos1, tw1, a1 = self.omega.sample(self.n_near, self.seed)
        pos2, tw2, a2 = self.omega_p.sample(self.n_near, self.seed)
        r1all = np.linalg.norm(pos1, axis=1)
        tree2 = cKDTree(pos2)
        step = 2000
        t0 = time.time()
        jb = _jax() if self.backend == 'jax' else None
        buf = _PadBuffer(self.chunk_pairs, hist, jb) if jb is not None else None
        for i0 in range(0, len(pos1), step):
            i1 = min(len(pos1), i0 + step)
            lists = tree2.query_ball_point(pos1[i0:i1], r=self._split)
            lens = np.array([len(l) for l in lists])
            if lens.sum() == 0:
                continue
            i = np.repeat(np.arange(i0, i1), lens)
            j = np.concatenate([np.asarray(l, dtype=int) for l in lists])
            x1, r1 = pos1[i], r1all[i]
            d = pos2[j] - pos1[i]
            w = tw1[i] * tw2[j]
            if buf is not None:
                buf.push(x1, r1, d, w)
            else:
                hist.add(x1, r1, d, w)
            if verbose and (i0 // step) % 20 == 0:
                tag = '[jax] ' if jb is not None else ''
                print(f"  near pairs {tag}{self.omega.key} x {self.omega_p.key}: {i1}/{len(pos1)}, {time.time() - t0:.1f}s")
        if buf is not None:
            buf.flush()
        return a1 * a2

    # ------------------------------------------------------------------ driver
    def compute(self, verbose: bool = False):
        self._split = self._split_edge()
        if abs(self._split - self.s_split) > 1e-9 and verbose:
            print(f"  s_split snapped from {self.s_split:g} to the bin edge {self._split:g}")
        vol = (self.s_edges[1:] ** 3 - self.s_edges[:-1] ** 3) / 3.0
        near = self.s_edges[1:] <= self._split + 1e-9
        p1, _, _ = self.omega.sample(max(self.n_sub, self.n_near), self.seed)
        p2, _, _ = self.omega_p.sample(max(self.n_sub, self.n_near), self.seed)
        r_edges = self._radial_edges(p1, p2)
        mu_edges = np.linspace(-1.0, 1.0, self.n_mu + 1)

        far = PairHistogram(r_edges, self.s_edges, mu_edges)
        norm_f = self._far_pairs(far, verbose)
        acc_f, w_f, n_f = far.contract(self.triples)
        acc, wsum, self.npairs, norm = acc_f, w_f, n_f.copy(), np.full(len(vol), norm_f)
        if np.any(near):
            nearh = PairHistogram(r_edges, self.s_edges, mu_edges)
            norm_n = self._near_pairs(nearh, verbose)
            acc_n, w_n, n_n = nearh.contract(self.triples)
            for t in self.triples:
                acc[t] = np.where(near, acc_n[t], acc_f[t])
            wsum = np.where(near, w_n, w_f)
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
                 min_pairs=20, n_shells=16, n_mu=None, backend='auto'):
        self.s_edges = np.asarray(s_edges, dtype=float)
        self.opts = dict(n_sub=n_sub, n_near=n_near, s_split=s_split, seed=seed,
                         chunk_pairs=chunk_pairs, min_pairs=min_pairs, n_shells=n_shells,
                         n_mu=n_mu, backend=backend)
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
        for key in self.missing():
            wins, trip = self._requests[key]
            if verbose:
                print(f"pair counts for {key}: {len(trip)} triples")
            tw = TripolarWindow(wins[0], wins[1], sorted(trip), self.s_edges, **self.opts)
            tw.compute(verbose=verbose)
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
