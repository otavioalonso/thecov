r"""Tripolar window functions Q^{omega omega'}_{Lambda1 Lambda2 Lambda}(s) from pair counts.

    Q(s_b) = alpha alpha' / int_b s^2 ds  *  sum_{r, r'} Theta_b(|s_rr'|)
             omega~(x_r) omega~'(x_r') S_{Lam1 Lam2 Lam}(x_r^, x_r'^, s_rr'^),   s = x_r' - x_r,

for a pair of windows (omega at x, omega' at x' = x + s; in covariance.py window_x and window_xp).

Count and contract
------------------
Because x' = x + s, the tripolar weight depends on the pair only through (r1, s, mu) with r1 = |x| and
mu = x^ . s^ (harmonics.coplanar_geometry: the azimuth about s^ vanishes identically). So:

  * COUNT: histogram the weighted pairs into cells of (r1, s, mu) -- one distance and one dot product per pair,
    no spherical harmonics;
  * CONTRACT: evaluate S once per cell, at the cell's mean (r1, s, mu), and sum.

All triples (Lam1, Lam2, Lam) come from the same counts. Window pairs sampled by the same two random catalogues see
the same pairs and differ only in their weights: they are counted together, one weight column each.

Sampling
--------
* far pairs (s >= s_split): all pairs of an n_sub subsample per window;
* near pairs (s <  s_split): KD-tree neighbours of a much larger n_near subsample;
* s = 0: the exact one-point anchor, which needs no pairs (Window.overlap_integral);
* s_split is snapped to a bin edge -- a bin straddling it would receive near pairs only below and far pairs only
  above while being normalised by its whole volume.

Backends: 'numpy' (reference) or 'jax' (the same arithmetic fused under XLA, GPU-ready; pair_counts_jax.py); 'auto'
prefers jax when it imports. Both provide the same two binners (far: a block of x against every x'; near: a list of
pairs) and produce the same histogram.
"""
from __future__ import annotations

import math
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from . import cache, defaults as D
from .harmonics import tripolar_coplanar
from .tracers import Window, SHOT_NOISE

# 2: shot-noise windows at unit scale (the estimator's shot noise is applied by the covariance)
WINDOW_FILE_FORMAT = 2
from .wigner import FOUR_PI

S000 = FOUR_PI ** -1.5


def resolve_backend(name: str) -> str:
    """'auto' picks jax when it imports, otherwise numpy."""
    if name == 'numpy':
        return 'numpy'
    try:
        from . import pair_counts_jax  # noqa: F401
    except Exception:
        if name == 'jax':
            raise
        return 'numpy'
    return 'jax'


# ----------------------------------------------------------------------------- histogram
class PairHistogram:
    """Weighted pair counts on a grid of (r1, s, mu), with the mean (r1, s, mu) of each cell.

    `n_columns` weight columns share one set of pairs (window pairs sampled by the same randoms differ only in their
    weights). The cell means are unweighted and common to all columns: K + 4 sums per pair for K columns instead of
    4K + 1 with per-column weighted means; the two agree to first order in the cell size (2e-4 in the covariance on
    the mock survey, against a cell-size error ~1e-3).
    """

    def __init__(self, r_edges, s_edges, mu_edges, n_columns: int = 1):
        self.r_edges, self.s_edges, self.mu_edges = (np.asarray(e, dtype=float) for e in (r_edges, s_edges, mu_edges))
        self.shape = (len(self.r_edges) - 1, len(self.s_edges) - 1, len(self.mu_edges) - 1)
        self.n_columns = int(n_columns)
        n_cells = int(np.prod(self.shape))
        self.weight = np.zeros((self.n_columns, n_cells))     # sum of pair weights, per column
        self.sum_r1 = np.zeros(n_cells)                        # sums of r1, s, mu (unweighted, for the cell means)
        self.sum_s = np.zeros(n_cells)
        self.sum_mu = np.zeros(n_cells)
        self.count = np.zeros(n_cells)                         # unweighted number of pairs

    def empty_like(self) -> 'PairHistogram':
        return PairHistogram(self.r_edges, self.s_edges, self.mu_edges, self.n_columns)

    def sums(self):
        return self.weight, self.sum_r1, self.sum_s, self.sum_mu, self.count

    def add_sums(self, weight, sum_r1, sum_s, sum_mu, count):
        """Absorb sums produced elsewhere (another histogram, a thread, the jax backend)."""
        self.weight += np.asarray(weight).reshape(self.weight.shape)
        self.sum_r1 += np.asarray(sum_r1)
        self.sum_s += np.asarray(sum_s)
        self.sum_mu += np.asarray(sum_mu)
        self.count += np.asarray(count)

    def add(self, x1, r1, d, w):
        """Add pairs: positions of the first point x1, their radii r1, separation vectors d = x2 - x1 and weights
        w, shape (n,) or (n, n_columns)."""
        w = np.asarray(w, dtype=float).reshape(len(r1), -1)
        s = np.sqrt(np.einsum('ij,ij->i', d, d))
        ok = (s > 0) & (r1 > 0)
        if not np.any(ok):
            return
        x1, r1, d, w, s = x1[ok], r1[ok], d[ok], w[ok], s[ok]
        mu = np.clip(np.einsum('ij,ij->i', x1, d) / (r1 * s), -1.0, 1.0)
        n_r, n_s, n_mu = self.shape
        i_r = np.clip(np.digitize(r1, self.r_edges) - 1, 0, n_r - 1)
        i_s = np.digitize(s, self.s_edges) - 1
        i_mu = np.clip(np.digitize(mu, self.mu_edges) - 1, 0, n_mu - 1)
        good = (i_s >= 0) & (i_s < n_s)
        cell = ((i_r * n_s + i_s) * n_mu + i_mu)[good]
        n_cells = self.count.size
        for c in range(self.n_columns):
            self.weight[c] += np.bincount(cell, weights=w[good, c], minlength=n_cells)
        self.sum_r1 += np.bincount(cell, weights=r1[good], minlength=n_cells)
        self.sum_s += np.bincount(cell, weights=s[good], minlength=n_cells)
        self.sum_mu += np.bincount(cell, weights=mu[good], minlength=n_cells)
        self.count += np.bincount(cell, minlength=n_cells)

    def contract(self, triples, column: int = 0):
        """({triple: sum over cells of w S_triple, per s bin}, weighted pairs per s bin, pairs per s bin)."""
        n_s = self.shape[1]
        weight = self.weight[column]
        per_s = lambda x: x.reshape(self.shape).sum(axis=(0, 2))
        out = {t: np.zeros(n_s) for t in triples}
        filled = np.flatnonzero(weight != 0)
        if len(filled):
            n = self.count[filled]
            S = tripolar_coplanar(triples, self.sum_r1[filled] / n, self.sum_s[filled] / n, self.sum_mu[filled] / n)
            s_bin = (filled // self.shape[2]) % n_s
            out = {t: np.bincount(s_bin, weights=weight[filled] * S[k], minlength=n_s) for k, t in enumerate(triples)}
        return out, per_s(weight), per_s(self.count)


# ----------------------------------------------------------------------------- numpy binners
def _numpy_far_binner(histogram, positions_xp, weights_xp, block_size, s_lo, s_hi):
    """bin_block(positions_x, weights_x, radii_x) -> histogram sums of all pairs (block of x) x (every x')."""
    def bin_block(positions_x, weights_x, radii_x):
        h = histogram.empty_like()
        d = positions_xp[None, :, :] - positions_x[:, None, :]
        s = np.linalg.norm(d, axis=-1)
        keep = (s >= s_lo) & (s < s_hi)
        if np.any(keep):
            h.add(np.broadcast_to(positions_x[:, None, :], d.shape)[keep], np.broadcast_to(radii_x[:, None], s.shape)[keep],
                  d[keep], (weights_x[:, None, :] * weights_xp[None, :, :])[keep])
        return h.sums()
    return bin_block


def _numpy_near_binner(histogram, chunk):
    """bin_pairs(x1, r1, d, w) -> histogram sums of an explicit list of pairs."""
    def bin_pairs(x1, r1, d, w):
        h = histogram.empty_like()
        h.add(x1, r1, d, w)
        return h.sums()
    return bin_pairs


def _binners(backend):
    if backend == 'jax':
        from . import pair_counts_jax
        return pair_counts_jax.far_binner, pair_counts_jax.near_binner
    return _numpy_far_binner, _numpy_near_binner


def _default_threads(n_threads):
    return max(1, min(16, os.cpu_count() or 1)) if n_threads is None else max(1, int(n_threads))


def _sum_over_threads(fn, blocks, n_threads):
    """sum of fn(block) over the blocks, one contiguous group of blocks per thread; each group sums where its
    results live (on the device for jax) and is copied back once. Returns one sum per group (None if empty)."""
    def run(group):
        total = None
        for b in group:
            out = fn(blocks[b])
            if out is not None:
                total = out if total is None else tuple(a + o for a, o in zip(total, out))
        return None if total is None else [np.asarray(a) for a in total]
    groups = [g for g in np.array_split(np.arange(len(blocks)), n_threads) if len(g)]
    if len(groups) == 1:
        return [run(groups[0])]
    with ThreadPoolExecutor(len(groups)) as ex:
        return list(ex.map(run, groups))


# ----------------------------------------------------------------------------- pair counter
class _PairCounter:
    """Pair counts of one pair of host catalogues, for every window pair they sample (one weight column each)."""

    def __init__(self, pair_windows, backend, n_threads=None):
        self.pair_windows = pair_windows
        self.first = pair_windows[0]                  # settings (subsamples, seed, s bins) are common to the group
        self.far_binner, self.near_binner = _binners(backend)
        self.n_threads = _default_threads(n_threads)
        self.chunk_pairs = int(self.first.chunk_pairs)

    def _subsamples(self, n):
        """positions, weight columns and effective alpha of the x and x' subsamples of size n."""
        first = self.first
        positions_x, _, alpha_x = first.window_x.sample(n, first.seed)
        positions_xp, _, alpha_xp = first.window_xp.sample(n, first.seed)
        weights_x = np.stack([pw.window_x.sample(n, pw.seed)[1] for pw in self.pair_windows], axis=1)
        weights_xp = np.stack([pw.window_xp.sample(n, pw.seed)[1] for pw in self.pair_windows], axis=1)
        return positions_x, weights_x, alpha_x, positions_xp, weights_xp, alpha_xp

    def _log(self, verbose, what, t0):
        if verbose:
            first = self.first
            print(f"  {what} {first.window_x.key}x{first.window_xp.key} (+{len(self.pair_windows) - 1} sharing them) "
                  f"in {time.time() - t0:.1f}s")

    def count_far(self, histogram, s_lo, s_hi, verbose):
        """all pairs of the n_sub subsamples with s in [s_lo, s_hi); returns the pair normalisation alpha alpha'."""
        positions_x, weights_x, alpha_x, positions_xp, weights_xp, alpha_xp = self._subsamples(self.first.n_sub)
        radii_x = np.linalg.norm(positions_x, axis=1)
        step = max(1, self.chunk_pairs // max(len(positions_xp), 1))
        blocks = [(i0, min(len(positions_x), i0 + step)) for i0 in range(0, len(positions_x), step)]
        bin_block = self.far_binner(histogram, positions_xp, weights_xp, step, s_lo, s_hi)
        t0 = time.time()
        for sums in _sum_over_threads(lambda b: bin_block(positions_x[b[0]:b[1]], weights_x[b[0]:b[1]], radii_x[b[0]:b[1]]),
                                      blocks, self.n_threads):
            if sums is not None:
                histogram.add_sums(*sums)
        self._log(verbose, f'far pairs ({len(positions_x)}x{len(positions_xp)})', t0)
        return alpha_x * alpha_xp

    def count_near(self, histogram, s_split, verbose):
        """KD-tree neighbours within s_split of the n_near subsamples; returns alpha alpha'."""
        positions_x, weights_x, alpha_x, positions_xp, weights_xp, alpha_xp = self._subsamples(self.first.n_near)
        radii_x = np.linalg.norm(positions_x, axis=1)
        tree_xp = cKDTree(positions_xp)
        # blocks of first points sized so that each yields a few chunks of pairs
        density = len(positions_xp) / max(float(np.prod(np.ptp(positions_xp, axis=0))), 1e-30)
        pairs_per_point = max(1.0, density * 4.0 / 3.0 * np.pi * s_split ** 3)
        block_size = int(np.clip(4 * self.chunk_pairs / pairs_per_point, 64, 20000))
        blocks = [(i0, min(len(positions_x), i0 + block_size)) for i0 in range(0, len(positions_x), block_size)]
        bin_pairs = self.near_binner(histogram, self.chunk_pairs)

        def bin_block(block):
            i0, i1 = block
            # all pairs within s_split, as index arrays built in C (no per-point Python lists)
            m = cKDTree(positions_x[i0:i1]).sparse_distance_matrix(tree_xp, s_split, output_type='ndarray')
            i, j = m['i'].astype(np.int64) + i0, m['j'].astype(np.int64)
            if len(i) == 0:
                return None
            return bin_pairs(positions_x[i], radii_x[i], positions_xp[j] - positions_x[i], weights_x[i] * weights_xp[j])
        t0 = time.time()
        for sums in _sum_over_threads(bin_block, blocks, self.n_threads):
            if sums is not None:
                histogram.add_sums(*sums)
        self._log(verbose, f'near pairs ({len(positions_x)} points)', t0)
        return alpha_x * alpha_xp

    def compute(self, verbose=False):
        first = self.first
        split = first.split_edge()
        if abs(split - first.s_split) > 1e-9 and verbose:
            print(f"  s_split snapped from {first.s_split:g} to the bin edge {split:g}")
        n_big = max(first.n_sub, first.n_near)
        r_edges = first.radial_edges(first.window_x.sample(n_big, first.seed)[0], first.window_xp.sample(n_big, first.seed)[0])
        # the finest mu grid any member needs (a finer grid is only more accurate)
        mu_edges = np.linspace(-1.0, 1.0, max(pw.n_mu for pw in self.pair_windows) + 1)
        n_columns = len(self.pair_windows)
        far = PairHistogram(r_edges, first.s_edges, mu_edges, n_columns)
        norm_far = self.count_far(far, split, first.s_edges[-1], verbose)
        near, norm_near = None, None
        n_near_bins = int(np.count_nonzero(first.s_edges[1:] <= split + 1e-9))
        if n_near_bins:
            # near pairs only fall in the s bins below the split: a histogram over those alone (~15x smaller)
            near = PairHistogram(r_edges, first.s_edges[:n_near_bins + 1], mu_edges, n_columns)
            norm_near = self.count_near(near, split, verbose)
        for column, pw in enumerate(self.pair_windows):
            pw.set_from_histograms(split, far, norm_far, near, norm_near, column=column)


# ----------------------------------------------------------------------------- one pair of windows
class TripolarWindow:
    """Q^{omega omega'}_{Lam1 Lam2 Lam}(s) of one pair of windows (window_x = omega, window_xp = omega') for a set
    of triples, from pair counts; Q(s) is then a cubic spline through the bins and the exact s = 0 value."""

    def __init__(self, window_x: Window, window_xp: Window, triples, s_edges: np.ndarray,
                 n_sub: int = D.N_SUB, n_near: int = D.N_NEAR, s_split: float = D.S_SPLIT,
                 seed: int = D.SEED, chunk_pairs: int = D.CHUNK_PAIRS, min_pairs: int = D.MIN_PAIRS,
                 n_shells: int = D.N_SHELLS, n_mu=D.N_MU, backend: str = 'auto', n_threads=None):
        self.window_x, self.window_xp = window_x, window_xp
        self.triples = sorted(set(tuple(int(x) for x in t) for t in triples))
        self.s_edges = np.asarray(s_edges, dtype=float)
        self.n_sub, self.n_near, self.s_split, self.seed = int(n_sub), int(n_near), float(s_split), int(seed)
        self.chunk_pairs, self.min_pairs, self.n_shells = int(chunk_pairs), int(min_pairs), int(n_shells)
        # S depends on mu through Pbar_{Lam1 m}(mu) and Pbar_{Lam2 m}(c2) with c2 ~ mu, so the mu grid has to resolve
        # oscillations of order max(Lam1, Lam2) -- roughly Lam/2 nodes over [-1, 1]. Six cells per node is
        # comfortable; too few silently biases the high multipoles (with ells and L up to 4, Lam1 reaches 12).
        lam_max = max((max(t[0], t[1]) for t in self.triples), default=0)
        self.n_mu = int(n_mu) if n_mu is not None else max(24, 6 * lam_max)
        self.backend = resolve_backend(backend)
        self.n_threads = n_threads
        self.Q = None          # {triple: values per s bin}
        self.npairs = None     # unweighted pairs per s bin actually used
        self.Q0 = None         # exact values at s = 0
        self._splines = {}

    # backwards-compatible names of the two windows (the note's omega, omega')
    omega = property(lambda self: self.window_x)
    omega_p = property(lambda self: self.window_xp)

    @property
    def s_centers(self) -> np.ndarray:
        return 0.5 * (self.s_edges[1:] + self.s_edges[:-1])

    def split_edge(self) -> float:
        return float(self.s_edges[int(np.argmin(np.abs(self.s_edges - self.s_split)))])

    def radial_edges(self, *point_sets) -> np.ndarray:
        r = np.concatenate([np.linalg.norm(p, axis=1) for p in point_sets])
        pad = 1e-6 * max(float(r.max()), 1.0)
        return np.linspace(float(r.min()) - pad, float(r.max()) + pad, self.n_shells + 1)

    def compute(self, verbose: bool = False):
        _PairCounter([self], self.backend, self.n_threads).compute(verbose)
        return self

    def set_from_histograms(self, split, far, norm_far, near, norm_near, column=0):
        """Q from the far and near histograms (column `column` holds this window pair's weights)."""
        volume = (self.s_edges[1:] ** 3 - self.s_edges[:-1] ** 3) / 3.0
        sums, _, self.npairs = far.contract(self.triples, column)
        norm = np.full(len(volume), norm_far)
        if near is not None:
            is_near = self.s_edges[1:] <= split + 1e-9
            sums_near, _, npairs_near = near.contract(self.triples, column)
            pad = lambda x: np.concatenate([x, np.zeros(len(volume) - len(x))])      # near bins only
            sums = {t: np.where(is_near, pad(sums_near[t]), sums[t]) for t in self.triples}
            self.npairs = np.where(is_near, pad(npairs_near), self.npairs)
            norm = np.where(is_near, norm_near, norm_far)
        self.Q = {t: norm * sums[t] / volume for t in self.triples}
        # exact s = 0 anchor: Q(0) = sqrt(4 pi) (-1)^Lam1 sqrt(2 Lam1 + 1) / (4 pi) int omega omega'
        overlap = self.window_x.overlap_integral(self.window_xp)
        self.Q0 = {(L1, L2, L): ((-1) ** L1 * math.sqrt(FOUR_PI * (2 * L1 + 1)) / FOUR_PI * overlap
                                 if (L == 0 and L1 == L2) else 0.0) for (L1, L2, L) in self.triples}
        self._splines = {}
        return self

    def _spline(self, triple):
        if triple not in self._splines:
            good = (self.npairs >= self.min_pairs) if self.npairs is not None else np.ones(len(self.s_centers), bool)
            x = np.concatenate([[0.0], self.s_centers[good], [self.s_edges[-1]]])
            y0 = self.Q0[triple] if self.Q0 is not None else self.Q[triple][0]
            y = np.concatenate([[y0], self.Q[triple][good], [0.0]])
            self._splines[triple] = CubicSpline(x, y, extrapolate=False)
        return self._splines[triple]

    def __call__(self, Lam1: int, Lam2: int, Lam: int, s: np.ndarray) -> np.ndarray:
        if self.Q is None:
            raise RuntimeError("call compute() first")
        return np.nan_to_num(self._spline((Lam1, Lam2, Lam))(s), nan=0.0)


# ----------------------------------------------------------------------------- all pairs of windows
class WindowLibrary:
    """Collects the window pairs and triples needed, computes them once, serves Q(s).

    Uses the symmetry  Q^{omega' omega}_{Lam1 Lam2 Lam} = Q^{omega omega'}_{Lam2 Lam1 Lam}  (even multipoles) so that
    each unordered pair of windows is counted once: pairs are stored under their canonical (sorted) key.
    """

    def __init__(self, s_edges, n_sub=D.N_SUB, n_near=D.N_NEAR, s_split=D.S_SPLIT, seed=D.SEED,
                 chunk_pairs=D.CHUNK_PAIRS, min_pairs=D.MIN_PAIRS, n_shells=D.N_SHELLS, n_mu=D.N_MU,
                 backend='auto', n_threads=None):
        self.s_edges = np.asarray(s_edges, dtype=float)
        self.opts = dict(n_sub=n_sub, n_near=n_near, s_split=s_split, seed=seed, chunk_pairs=chunk_pairs,
                         min_pairs=min_pairs, n_shells=n_shells, n_mu=n_mu, backend=backend, n_threads=n_threads)
        self.requests = {}       # canonical key -> ((window_x, window_xp), set of triples)
        self.computed = {}       # canonical key -> TripolarWindow
        self._values = {}        # cache of Q(s) on the grids asked for

    @staticmethod
    def _canonical(window_x: Window, window_xp: Window):
        """(sorted key, whether the pair was swapped to get it)."""
        if window_xp.key < window_x.key:
            return (window_xp.key, window_x.key), True
        return (window_x.key, window_xp.key), False

    def request(self, window_x: Window, window_xp: Window, triples):
        key, swapped = self._canonical(window_x, window_xp)
        pair, wanted = self.requests.setdefault(key, ((window_xp, window_x) if swapped else (window_x, window_xp), set()))
        wanted.update((L2, L1, L) if swapped else (L1, L2, L) for (L1, L2, L) in triples)

    def missing(self):
        return [key for key, (_, wanted) in self.requests.items()
                if key not in self.computed or not set(self.computed[key].triples) >= wanted]

    def compute_all(self, verbose=False):
        """Count all missing window pairs, grouped by the pair of host catalogues that samples them: each group shares
        one neighbour search and one binning (one weight column per window pair)."""
        groups = {}
        for key in self.missing():
            (window_x, window_xp), wanted = self.requests[key]
            pw = TripolarWindow(window_x, window_xp, sorted(wanted), self.s_edges, **self.opts)
            groups.setdefault((window_x.host.name, window_xp.host.name), []).append((key, pw))
        for hosts, members in groups.items():
            if verbose:
                print(f"pair counts for hosts {hosts}: {len(members)} window pairs ({', '.join(str(k) for k, _ in members)})")
            pair_windows = [pw for _, pw in members]
            _PairCounter(pair_windows, pair_windows[0].backend, self.opts.get('n_threads')).compute(verbose)
            self.computed.update(members)
        self._values = {}

    def get(self, window_x: Window, window_xp: Window, Lam1: int, Lam2: int, Lam: int, s: np.ndarray) -> np.ndarray:
        key, swapped = self._canonical(window_x, window_xp)
        cache_key = (key, Lam1, Lam2, Lam, id(s))
        if cache_key not in self._values:
            self._values[cache_key] = self.computed[key](*((Lam2, Lam1) if swapped else (Lam1, Lam2)), Lam, s)
        return self._values[cache_key]

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Store the pair-count results (the expensive, cosmology-independent part)."""
        data, meta = {'s_edges': self.s_edges}, []
        for n, (key, pw) in enumerate(self.computed.items()):
            npairs_name = f"npairs_{n}"
            data[npairs_name] = np.asarray(pw.npairs if pw.npairs is not None else [])
            for triple, Q in pw.Q.items():
                q_name = f"Q_{len(meta)}"
                data[q_name] = np.asarray(Q)
                meta.append({'key': [list(k) for k in key], 'trip': [int(x) for x in triple], 'q': q_name,
                             'npairs': npairs_name, 'q0': float(pw.Q0[triple]) if pw.Q0 is not None else 0.0})
        cache.save(path, data, {'format': WINDOW_FILE_FORMAT, 'windows': meta})

    def load(self, path):
        """Load pair-count results saved with save(); windows are matched by their names."""
        arrays, meta = cache.load(path)
        if isinstance(meta, list):          # format 1: shot-noise windows were stored with the tracer's shot-noise scale
            n_meta = len(meta)
            meta = [rec for rec in meta if not any(k[0] == SHOT_NOISE for k in rec['key'])]
            if len(meta) < n_meta:
                warnings.warn(f"{path}: written before the shot-noise windows were stored at unit scale; their pair "
                              "counts are dropped and will be recomputed (the clustering windows are kept)")
        else:
            meta = meta['windows']
        if len(arrays['s_edges']) != len(self.s_edges) or not np.allclose(arrays['s_edges'], self.s_edges):
            raise ValueError("s_edges of the stored windows differ from the current ones")
        Qs, Q0s, npairs = {}, {}, {}
        for rec in meta:
            key, triple = tuple(tuple(k) for k in rec['key']), tuple(rec['trip'])
            Qs.setdefault(key, {})[triple] = arrays[rec['q']]
            Q0s.setdefault(key, {})[triple] = rec['q0']
            if key not in npairs:
                npairs[key] = arrays[rec['npairs']] if arrays[rec['npairs']].size else None
        for key, Q in Qs.items():
            pw = TripolarWindow(None, None, list(Q), self.s_edges, **self.opts)
            pw.Q, pw.Q0, pw.npairs = Q, Q0s[key], npairs[key]
            self.computed[key] = pw
        self._values = {}
        return self
