"""Tracers (random catalogues) and the windows built from them.

A tracer is described by its random catalogue: a dict with keys
    'POSITION' : (N, 3) comoving Cartesian coordinates with the observer at the origin,
    'WEIGHT'   : (N,)   total weight w(x) applied to the density field (FKP x completeness ...),
    'NZ'       : (N,)   optional, mean density nbar(x) at each random.
plus alpha = (weighted number of galaxies) / (weighted number of randoms), i.e. the
random catalogue samples nbar / alpha.

Windows (eq. windows of the note):
    W^{AB}(x) = nbar_A nbar_B w_A w_B      (clustering)
    S^{A}(x)  = (1 + alpha_A) nbar_A w_A^2 (shot noise)
Both are of the form nbar_X(x) * omega~(x) and are sampled by the randoms of X.
"""
from __future__ import annotations

import warnings
import zlib

import numpy as np
from scipy.spatial import cKDTree


class Tracer:
    def __init__(self, name: str, randoms: dict, alpha: float, nbar=None, knn: int = 64):
        self.name = str(name)
        self.pos = np.ascontiguousarray(randoms['POSITION'], dtype=float)
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("randoms['POSITION'] must have shape (N, 3)")
        self.w = np.asarray(randoms['WEIGHT'], dtype=float)
        self.alpha = float(alpha)
        self._tree = None
        if nbar is None:
            nbar = randoms.get('NZ', None)
        if nbar is None:
            warnings.warn(f"Tracer {self.name}: no 'NZ' given; estimating nbar from the randoms "
                          f"with a {knn}-nearest-neighbour density estimate.")
            nbar = self._knn_nbar(knn)
        elif callable(nbar):
            nbar = nbar(self.pos)
        self.nbar = np.asarray(nbar, dtype=float)
        if self.nbar.shape != (len(self.pos),):
            raise ValueError("nbar must have one value per random")
        self._sub_cache = {}

    # -- geometry helpers ------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.pos)

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.pos)
        return self._tree

    def _knn_nbar(self, k: int) -> np.ndarray:
        d, _ = self.tree.query(self.pos, k=k + 1)
        r = d[:, -1]
        return self.alpha * k / (4.0 / 3.0 * np.pi * r ** 3)

    def nbar_w_at(self, positions: np.ndarray):
        """(nbar(x), w(x)) of this tracer at arbitrary positions (nearest random)."""
        if positions is self.pos:
            return self.nbar, self.w
        _, idx = self.tree.query(positions, k=1)
        return self.nbar[idx], self.w[idx]

    def nw_at(self, positions: np.ndarray) -> np.ndarray:
        """nbar(x) * w(x) of this tracer at arbitrary positions (nearest random)."""
        nb, w = self.nbar_w_at(positions)
        return nb * w

    def subsample_indices(self, n: int, seed: int = 0) -> np.ndarray:
        key = (n, seed)
        if key not in self._sub_cache:
            rng = np.random.default_rng(seed + zlib.crc32(self.name.encode()) % 10000)
            n = min(n, self.size)
            self._sub_cache[key] = np.sort(rng.choice(self.size, size=n, replace=False))
        return self._sub_cache[key]

    def s_max(self) -> float:
        """Upper bound on pair separations (bounding-box diagonal)."""
        return float(np.linalg.norm(self.pos.max(0) - self.pos.min(0)))


class Window:
    """omega(x) = nbar_host(x) * tilde_omega(x), sampled by the randoms of `host`."""

    def __init__(self, kind: str, A: Tracer, B: Tracer | None = None):
        if kind == 'W':
            if B is None:
                raise ValueError("clustering window needs two tracers")
            # canonical ordering: the window is symmetric, host = alphabetically first
            if B.name < A.name:
                A, B = B, A
            self.tracers = (A, B)
        elif kind == 'S':
            self.tracers = (A,)
        else:
            raise ValueError("kind must be 'W' or 'S'")
        self.kind = kind
        self.key = (kind,) + tuple(t.name for t in self.tracers)

    @property
    def host(self) -> Tracer:
        return self.tracers[0]

    def tilde_weights(self) -> np.ndarray:
        """omega / nbar_host at the host randoms."""
        A = self.host
        if self.kind == 'W':
            B = self.tracers[1]
            return A.w * B.nw_at(A.pos)
        return (1.0 + A.alpha) * A.w ** 2

    def value_at(self, positions: np.ndarray) -> np.ndarray:
        """omega(x) at arbitrary positions (nearest-random interpolation of nbar and w)."""
        if self.kind == 'W':
            A, B = self.tracers
            return A.nw_at(positions) * B.nw_at(positions)
        A = self.host
        nb, w = A.nbar_w_at(positions)
        return (1.0 + A.alpha) * nb * w ** 2

    def integral(self) -> float:
        """int d^3x omega(x)  (e.g. I_AB for kind 'W')."""
        return self.host.alpha * float(np.sum(self.tilde_weights()))

    def overlap_integral(self, other: "Window") -> float:
        """int d^3x omega(x) omega'(x): the s -> 0 anchor of the window pair function."""
        A = self.host
        return A.alpha * float(np.sum(self.tilde_weights() * other.value_at(A.pos)))

    def sample(self, n_sub: int, seed: int = 0):
        """(positions, tilde weights, effective alpha) of a subsample of the host randoms."""
        A = self.host
        idx = A.subsample_indices(n_sub, seed)
        tw = self.tilde_weights()[idx]
        alpha_eff = A.alpha * A.size / len(idx)
        return A.pos[idx], tw, alpha_eff

    def __repr__(self):
        return "Window" + str(self.key)


def spectrum_window_pairs(A: Tracer, B: Tracer, shot_noise: bool = True):
    """The set P^{AB} of (window, spectrum-label) pairs, eq. pairs of the note.

    Spectrum labels: ('P', nameA, nameB) for the clustering multipoles and ('S', nameA) for
    shot noise (p_L = delta_{L0}).
    """
    pairs = [(Window('W', A, B), ('P',) + tuple(sorted((A.name, B.name))))]
    if shot_noise and A.name == B.name:
        pairs.append((Window('S', A), ('S', A.name)))
    return pairs
