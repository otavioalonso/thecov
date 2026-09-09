"""Tripolar window functions Q^{omega omega'}_{Lambda1 Lambda2 Lambda}(s) from pair counts.

Implements eq. Q-pairs of the note:

    Q(s_b) = alpha alpha' / int_b s^2 ds  *  sum_{r in R, r' in R'} Theta_b(|s_rr'|)
             omega~(x_r) omega~'(x_r') S_{Lam1 Lam2 Lam}(x_r^, x_r'^, s_rr'^),   s = x_r' - x_r.

S is evaluated in the frame where s^ is the polar axis (harmonics.tripolar_frame_weights), so
that only two associated-Legendre tables and cos(mu * dphi) are needed per pair.

Sampling strategy
-----------------
* far pairs (s >= s_split): all pairs of a random subsample of n_sub points per window;
* near pairs (s <  s_split): KD-tree neighbour search on a much larger subsample (n_near), where
  the far subsample would contain too few pairs (their number grows as s^2 ds);
* s = 0: the exact one-point anchor Q(0) = sqrt(4 pi) delta_{Lam 0} delta_{Lam1 Lam2}
  sqrt(2 Lam1 + 1) / (4 pi) * int d^3x omega omega', which needs no pairs at all
  (Window.overlap_integral).
The radial spline through the bin values uses only bins with at least `min_pairs` pairs plus the
anchors at s = 0 and s = s_max.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from .harmonics import normalized_legendre, cos_dphi, cos_multiples, tripolar_frame_weights, unit_vectors
from .tracers import Window
from .wigner import FOUR_PI


class TripolarWindow:
    """Pair-count estimate of Q^{omega omega'}_{Lam1 Lam2 Lam}(s) for a set of triples."""

    def __init__(self, omega: Window, omega_p: Window, triples, s_edges: np.ndarray,
                 n_sub: int = 5000, n_near: int = 200000, s_split: float = 80.0,
                 seed: int = 0, chunk_pairs: int = 40000, min_pairs: int = 20):
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
        self.Q = None          # dict triple -> array over s bins
        self.npairs = None     # pairs per s bin actually used
        self.Q0 = None         # dict triple -> exact value at s = 0
        self._splines = {}

    # ------------------------------------------------------------------
    @property
    def s_centers(self) -> np.ndarray:
        return 0.5 * (self.s_edges[1:] + self.s_edges[:-1])

    def _groups(self):
        groups = {}
        for (L1, L2, L) in self.triples:
            groups.setdefault((L1, L2), []).append(L)
        Wmats = {k: tripolar_frame_weights(k[0], k[1], v) for k, v in groups.items()}
        return groups, Wmats

    def _accumulate(self, x1, x2, wpair, d, groups, Wmats, acc, npairs):
        """Add the contribution of the pairs (x1[i] -> x2[i]) with separation vectors d[i]."""
        nb = len(self.s_edges) - 1
        s = np.linalg.norm(d, axis=-1)
        shat = d / s[:, None]
        c1 = np.einsum('ij,ij->i', x1, shat)
        c2 = np.einsum('ij,ij->i', x2, shat)
        cx = np.einsum('ij,ij->i', x1, x2)
        lmax1 = max(k[0] for k in groups)
        lmax2 = max(k[1] for k in groups)
        mumax = max(min(k) for k in groups)
        cm = cos_multiples(cos_dphi(c1, c2, cx), mumax)
        P1 = normalized_legendre(c1, lmax1)
        P2 = normalized_legendre(c2, lmax2)
        ib = np.digitize(s, self.s_edges) - 1
        ok = (ib >= 0) & (ib < nb)
        npairs += np.bincount(ib[ok], minlength=nb)
        for (L1, L2), Ls in groups.items():
            mm = min(L1, L2)
            A = (P1[L1, :mm + 1] * P2[L2, :mm + 1] * cm[:mm + 1]).T
            S = A @ Wmats[(L1, L2)]
            for j, L in enumerate(Ls):
                acc[(L1, L2, L)] += np.bincount(ib[ok], weights=(wpair * S[:, j])[ok], minlength=nb)

    def _split_edge(self) -> float:
        """s_split snapped to the nearest s-bin edge.

        Near and far pairs are counted on different subsamples, so a bin that STRADDLES the split
        would receive near pairs only below it and far pairs only above it -- with each estimator
        normalised by the whole bin volume, the bin comes out low by the volume fraction it is
        missing. Snapping the split to a bin edge makes every bin belong entirely to one regime.
        """
        return float(self.s_edges[int(np.argmin(np.abs(self.s_edges - self.s_split)))])

    def _far_pairs(self, groups, Wmats, verbose):
        pos1, tw1, a1 = self.omega.sample(self.n_sub, self.seed)
        pos2, tw2, a2 = self.omega_p.sample(self.n_sub, self.seed)
        xhat1, xhat2 = unit_vectors(pos1), unit_vectors(pos2)
        n1, n2 = len(pos1), len(pos2)
        nb = len(self.s_edges) - 1
        acc = {t: np.zeros(nb) for t in self.triples}
        npairs = np.zeros(nb)
        step = max(1, self.chunk_pairs // n2)
        t0 = time.time()
        for i0 in range(0, n1, step):
            i1 = min(n1, i0 + step)
            d = pos2[None, :, :] - pos1[i0:i1, None, :]
            s = np.linalg.norm(d, axis=-1)
            keep = (s >= self._split) & (s < self.s_edges[-1])
            if not np.any(keep):
                continue
            x1 = np.broadcast_to(xhat1[i0:i1, None, :], d.shape)[keep]
            x2 = np.broadcast_to(xhat2[None, :, :], d.shape)[keep]
            w = (tw1[i0:i1, None] * tw2[None, :])[keep]
            self._accumulate(x1, x2, w, d[keep], groups, Wmats, acc, npairs)
            if verbose and (i0 // step) % 50 == 0:
                print(f"  far pairs {self.omega.key} x {self.omega_p.key}: {i1}/{n1}, {time.time() - t0:.1f}s")
        return acc, npairs, a1 * a2

    def _near_pairs(self, groups, Wmats, verbose):
        pos1, tw1, a1 = self.omega.sample(self.n_near, self.seed)
        pos2, tw2, a2 = self.omega_p.sample(self.n_near, self.seed)
        xhat1, xhat2 = unit_vectors(pos1), unit_vectors(pos2)
        tree2 = cKDTree(pos2)
        nb = len(self.s_edges) - 1
        acc = {t: np.zeros(nb) for t in self.triples}
        npairs = np.zeros(nb)
        step = 2000
        t0 = time.time()
        for i0 in range(0, len(pos1), step):
            i1 = min(len(pos1), i0 + step)
            lists = tree2.query_ball_point(pos1[i0:i1], r=self._split)
            lens = np.array([len(l) for l in lists])
            if lens.sum() == 0:
                continue
            i = np.repeat(np.arange(i0, i1), lens)
            j = np.concatenate([np.asarray(l, dtype=int) for l in lists])
            d = pos2[j] - pos1[i]
            keep = np.linalg.norm(d, axis=-1) > 0
            i, j, d = i[keep], j[keep], d[keep]
            self._accumulate(xhat1[i], xhat2[j], tw1[i] * tw2[j], d, groups, Wmats, acc, npairs)
            if verbose and (i0 // step) % 20 == 0:
                print(f"  near pairs {self.omega.key} x {self.omega_p.key}: {i1}/{len(pos1)}, {time.time() - t0:.1f}s")
        return acc, npairs, a1 * a2

    def compute(self, verbose: bool = False):
        groups, Wmats = self._groups()
        self._split = self._split_edge()
        if abs(self._split - self.s_split) > 1e-9 and verbose:
            print(f"  s_split snapped from {self.s_split:g} to the bin edge {self._split:g}")
        vol = (self.s_edges[1:] ** 3 - self.s_edges[:-1] ** 3) / 3.0
        near = self.s_edges[1:] <= self._split + 1e-9
        acc_f, np_f, norm_f = self._far_pairs(groups, Wmats, verbose)
        self.Q = {t: norm_f * acc_f[t] / vol for t in self.triples}
        self.npairs = np_f.copy()
        if np.any(near):
            acc_n, np_n, norm_n = self._near_pairs(groups, Wmats, verbose)
            for t in self.triples:
                self.Q[t][near] = (norm_n * acc_n[t] / vol)[near]
            self.npairs[near] = np_n[near]
        # exact s = 0 anchor
        ov = self.omega.overlap_integral(self.omega_p)
        # Q(0) = sqrt(4 pi) (-1)^Lam1 sqrt(2 Lam1 + 1) / (4 pi) * int omega omega'  for Lam = 0,
        # Lam1 = Lam2, and zero otherwise (see the note; the sign is +1 for even Lam1).
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
        """Q interpolated on the grid s (zero beyond the last pair bin)."""
        if self.Q is None:
            raise RuntimeError("call compute() first")
        return np.nan_to_num(self._spline((Lam1, Lam2, Lam))(s), nan=0.0)


class WindowLibrary:
    """Collects the (omega, omega') pairs and triples needed, computes them once, serves Q(s).

    Uses the symmetry  Q^{omega' omega}_{Lam1 Lam2 Lam} = Q^{omega omega'}_{Lam2 Lam1 Lam}
    (even multipoles) so that each unordered pair of windows is counted once.
    """

    def __init__(self, s_edges, n_sub=5000, n_near=200000, s_split=80.0, seed=0, chunk_pairs=40000, min_pairs=20):
        self.s_edges = np.asarray(s_edges, dtype=float)
        self.opts = dict(n_sub=n_sub, n_near=n_near, s_split=s_split, seed=seed,
                         chunk_pairs=chunk_pairs, min_pairs=min_pairs)
        self._requests = {}   # canonical key -> ((omega, omega_p), set of triples)
        self._windows = {}    # canonical key -> TripolarWindow
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
        """Canonical keys that were requested but are not (fully) computed."""
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
        """Store the pair-count results (the expensive, cosmology-independent part).

        The metadata is written as a JSON string, so loading needs no pickling.
        """
        data = {'s_edges': self.s_edges}
        meta = []
        for w, (key, tw) in enumerate(self._windows.items()):
            np_name = f"npairs_{w}"
            data[np_name] = np.asarray(tw.npairs if tw.npairs is not None else [])
            for trip, Q in tw.Q.items():
                q_name = f"Q_{len(meta)}"
                data[q_name] = np.asarray(Q)
                meta.append({'key': [list(k) for k in key], 'trip': list(int(x) for x in trip),
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
