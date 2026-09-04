"""Tripolar window functions Q^{omega omega'}_{Lambda1 Lambda2 Lambda}(s) from pair counts.

Implements eq. Q-pairs of the note:

    Q(s_b) = alpha alpha' / int_b s^2 ds  *  sum_{r in R, r' in R'} Theta_b(|s_rr'|)
             omega~(x_r) omega~'(x_r') S_{Lam1 Lam2 Lam}(x_r^, x_r'^, s_rr'^),   s = x_r' - x_r.

S is evaluated in the frame where s^ is the polar axis (harmonics.tripolar_frame_weights), so
that only two associated-Legendre tables and cos(mu * dphi) are needed per pair.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import CubicSpline

from .harmonics import normalized_legendre, cos_dphi, cos_multiples, tripolar_frame_weights, unit_vectors
from .tracers import Window


class TripolarWindow:
    """Pair-count estimate of Q^{omega omega'}_{Lam1 Lam2 Lam}(s) for a set of triples."""

    def __init__(self, omega: Window, omega_p: Window, triples, s_edges: np.ndarray,
                 n_sub: int = 5000, seed: int = 0, chunk_pairs: int = 40000):
        self.omega = omega
        self.omega_p = omega_p
        self.triples = sorted(set(tuple(int(x) for x in t) for t in triples))
        self.s_edges = np.asarray(s_edges, dtype=float)
        self.n_sub = int(n_sub)
        self.seed = int(seed)
        self.chunk_pairs = int(chunk_pairs)
        self.Q = None  # dict triple -> array over s bins
        self._splines = {}

    # ------------------------------------------------------------------
    @property
    def s_centers(self) -> np.ndarray:
        return 0.5 * (self.s_edges[1:] + self.s_edges[:-1])

    def compute(self, verbose: bool = False):
        pos1, tw1, a1 = self.omega.sample(self.n_sub, self.seed)
        pos2, tw2, a2 = self.omega_p.sample(self.n_sub, self.seed)
        xhat1, xhat2 = unit_vectors(pos1), unit_vectors(pos2)
        n1, n2 = len(pos1), len(pos2)
        nb = len(self.s_edges) - 1
        smax = self.s_edges[-1]

        # group triples by (Lam1, Lam2)
        groups = {}
        for (L1, L2, L) in self.triples:
            groups.setdefault((L1, L2), []).append(L)
        Wmats = {k: tripolar_frame_weights(k[0], k[1], v) for k, v in groups.items()}
        lmax1 = max(k[0] for k in groups)
        lmax2 = max(k[1] for k in groups)
        mumax = max(min(k) for k in groups)
        acc = {t: np.zeros(nb) for t in self.triples}

        step = max(1, self.chunk_pairs // n2)
        t0 = time.time()
        for i0 in range(0, n1, step):
            i1 = min(n1, i0 + step)
            d = pos2[None, :, :] - pos1[i0:i1, None, :]              # (c, n2, 3)
            s = np.linalg.norm(d, axis=-1)
            keep = (s > 0) & (s < smax)
            if not np.any(keep):
                continue
            shat = (d / np.where(s > 0, s, 1.0)[..., None])[keep]
            s = s[keep]
            x1 = np.broadcast_to(xhat1[i0:i1, None, :], d.shape)[keep]
            x2 = np.broadcast_to(xhat2[None, :, :], d.shape)[keep]
            wpair = (tw1[i0:i1, None] * tw2[None, :])[keep]
            c1 = np.einsum('ij,ij->i', x1, shat)
            c2 = np.einsum('ij,ij->i', x2, shat)
            cx = np.einsum('ij,ij->i', x1, x2)
            cm = cos_multiples(cos_dphi(c1, c2, cx), mumax)          # (mumax+1, npairs)
            P1 = normalized_legendre(c1, lmax1)                       # (lmax1+1, lmax1+1, npairs)
            P2 = normalized_legendre(c2, lmax2)
            ib = np.digitize(s, self.s_edges) - 1
            ok = (ib >= 0) & (ib < nb)
            for (L1, L2), Ls in groups.items():
                mm = min(L1, L2)
                A = (P1[L1, :mm + 1] * P2[L2, :mm + 1] * cm[:mm + 1]).T   # (npairs, mm+1)
                S = A @ Wmats[(L1, L2)]                                   # (npairs, nL)
                for j, L in enumerate(Ls):
                    acc[(L1, L2, L)] += np.bincount(ib[ok], weights=(wpair * S[:, j])[ok], minlength=nb)
            if verbose and (i0 // step) % 20 == 0:
                print(f"  pairs {self.omega.key} x {self.omega_p.key}: {i1}/{n1} primaries, {time.time()-t0:.1f}s")

        vol = (self.s_edges[1:] ** 3 - self.s_edges[:-1] ** 3) / 3.0
        self.Q = {t: a1 * a2 * a / vol for t, a in acc.items()}
        self._splines = {}
        return self

    # ------------------------------------------------------------------
    def __call__(self, Lam1: int, Lam2: int, Lam: int, s: np.ndarray) -> np.ndarray:
        """Q interpolated on the grid s (zero beyond the last pair bin)."""
        key = (Lam1, Lam2, Lam)
        if self.Q is None:
            raise RuntimeError("call compute() first")
        if key not in self._splines:
            self._splines[key] = CubicSpline(self.s_centers, self.Q[key], extrapolate=True)
        out = self._splines[key](s)
        out[s > self.s_edges[-1]] = 0.0
        return out


class WindowLibrary:
    """Collects the (omega, omega') pairs and triples needed, computes them once, serves Q(s).

    Uses the symmetry  Q^{omega' omega}_{Lam1 Lam2 Lam} = Q^{omega omega'}_{Lam2 Lam1 Lam}
    (even multipoles) so that each unordered pair of windows is counted once.
    """

    def __init__(self, s_edges, n_sub=5000, seed=0, chunk_pairs=40000):
        self.s_edges = np.asarray(s_edges, dtype=float)
        self.n_sub, self.seed, self.chunk_pairs = n_sub, seed, chunk_pairs
        self._requests = {}   # canonical key -> (omega, omega_p, set of triples)
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

    def compute_all(self, verbose=False):
        for key, (wins, trip) in self._requests.items():
            if key in self._windows and set(self._windows[key].triples) >= trip:
                continue
            if verbose:
                print(f"pair counts for {key}: {len(trip)} triples")
            tw = TripolarWindow(wins[0], wins[1], sorted(trip), self.s_edges,
                                n_sub=self.n_sub, seed=self.seed, chunk_pairs=self.chunk_pairs)
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

    def missing(self):
        """Canonical keys that were requested but are not (fully) computed."""
        out = []
        for key, (_, trip) in self._requests.items():
            if key not in self._windows or not set(self._windows[key].triples) >= trip:
                out.append(key)
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Store the pair-count results (the expensive, cosmology-independent part)."""
        data = {'s_edges': self.s_edges}
        meta = []
        for key, tw in self._windows.items():
            for trip, Q in tw.Q.items():
                name = f"Q_{len(meta)}"
                data[name] = Q
                meta.append((key, trip, name))
        data['meta'] = np.array(meta, dtype=object)
        np.savez(path, **data, allow_pickle=True)

    def load(self, path):
        """Load pair-count results saved with save(); keys are matched by window names."""
        with np.load(path, allow_pickle=True) as f:
            s_edges = f['s_edges']
            if len(s_edges) != len(self.s_edges) or not np.allclose(s_edges, self.s_edges):
                raise ValueError("s_edges of the stored windows differ from the current ones")
            groups = {}
            for key, trip, name in f['meta']:
                key = tuple(tuple(k) for k in key)
                groups.setdefault(key, {})[tuple(int(x) for x in trip)] = f[name]
        for key, Qs in groups.items():
            tw = TripolarWindow(None, None, list(Qs), self.s_edges)
            tw.Q = Qs
            self._windows[key] = tw
        self._interp_cache = {}
        return self
