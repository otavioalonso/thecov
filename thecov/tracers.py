"""Tracers (random catalogues) and the windows built from them.

A tracer is described by its random catalogue: a dict with keys
    'POSITION' : (N, 3) comoving Cartesian coordinates with the observer at the origin,
    'WEIGHT'   : (N,)   total weight w(x) applied to the density field (FKP x completeness ...),
    'NZ'       : (N,)   optional, mean density nbar(x) at each random,
    'NW'       : (N,)   optional, the mean WEIGHTED galaxy density m(x) = E[sum_g w_g delta_D(x - x_g)]
                        at each random, a smooth function of position;
plus alpha = (weighted number of galaxies) / (weighted number of randoms), i.e. the weighted random
catalogue samples the weighted galaxy density.

Per-object weights. The clustering window is W^{AB}(x) = m_A(x) m_B(x). When the weight is a smooth
function of position, m = nbar w and 'NZ' suffices. When it varies from object to object (survey
completeness, imaging-systematics and redshift-failure weights, as in DESI, where each random also
inherits the weight of a random data object), nbar(x_r) w_r evaluated with a random's OWN weight
gives the auto window ~ <w^2> instead of <w>^2, too large by 1 + var(w)/<w>^2. Pass the smooth 'NW'
then; the windows use it in place of NZ * WEIGHT, while every random still carries its own weight as
a sampling weight (sum_r w_r f(x_r) ~ int m f / alpha) and in the shot-noise window.
`shotnoise_scale` multiplies the shot-noise window, e.g. to match (sum_d w^2 + alpha^2 sum_r w^2)
when the random weights do not reproduce the distribution of the data weights.

Cross windows W^{AB} are sampled by the randoms of A, with nbar_B w_B taken from the nearest random
of B and set to zero when that random is farther than `mask_factor` x the LOCAL spacing of B's
randoms, so that B's footprint is respected. The footprint edge is therefore resolved to a few
inter-random spacings: use dense random catalogues for cross windows, and check I_AB against an
independent estimate (it is exposed as GaussianCovariance.I).

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
    def __init__(self, name: str, randoms: dict, alpha: float, nbar=None, knn: int = 64,
                 mask_factor: float = 2.5, mask_knn: int = 8, shotnoise_scale: float = 1.0):
        self.name = str(name)
        self.pos = np.ascontiguousarray(randoms['POSITION'], dtype=float)
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("randoms['POSITION'] must have shape (N, 3)")
        self.w = np.asarray(randoms['WEIGHT'], dtype=float)
        self.alpha = float(alpha)
        self.shotnoise_scale = float(shotnoise_scale)
        self._tree = None
        nw = randoms.get('NW', None)
        if nbar is None:
            nbar = randoms.get('NZ', None)
        if nbar is None and nw is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                nbar = np.where(self.w != 0, np.asarray(nw, dtype=float) / self.w, 0.0)
        if nbar is None:
            warnings.warn(f"Tracer {self.name}: no 'NZ' given; estimating nbar from the randoms "
                          f"with a {knn}-nearest-neighbour density estimate.")
            nbar = self._knn_nbar(knn)
        elif callable(nbar):
            nbar = nbar(self.pos)
        self.nbar = np.asarray(nbar, dtype=float)
        if self.nbar.shape != (len(self.pos),):
            raise ValueError("nbar must have one value per random")
        # m(x): the mean weighted galaxy density entering the clustering windows
        self.has_nw = nw is not None
        self.mw = np.asarray(nw, dtype=float) if self.has_nw else self.nbar * self.w
        if self.mw.shape != (len(self.pos),):
            raise ValueError("NW must have one value per random")
        self._sub_cache = {}
        # Footprint mask used when this tracer's density is needed at foreign positions: a position
        # is outside the footprint if it is farther than mask_factor x the LOCAL random spacing from
        # the nearest random. The threshold has to be local: with a global value, a tracer whose
        # density varies strongly (e.g. a steep n(z)) would have its sparse outskirts masked away.
        # The spacing is estimated from the k-th neighbour (k = mask_knn) rather than the first: the
        # first-neighbour distance of a Poisson set has ~50 % scatter, so a threshold built on it
        # masks out interior positions whose nearest random happens to sit in a close pair.
        self.mask_factor = float(mask_factor)
        kk = min(int(mask_knn), self.size - 1)
        d, _ = self.tree.query(self.pos, k=kk + 1)
        self.spacing = d[:, kk] / kk ** (1.0 / 3.0)
        # consistency of alpha with NZ: the randoms must sample nbar/alpha. The pair counts use alpha
        # and the randoms, the s = 0 anchor and cross windows use NZ; an inconsistent pair of
        # (alpha, NZ) would mis-normalise them relative to each other.
        ratio = self._implied_alpha(knn) / self.alpha
        if abs(ratio - 1) > 0.15:
            what = 'NW' if self.has_nw else 'NZ'
            warnings.warn(f"Tracer {self.name}: alpha={self.alpha:.4g} but the random density and {what} imply "
                          f"alpha~{self.alpha * ratio:.4g} (ratio {ratio:.2f}); "
                          + ("NW must equal alpha x (random density) x (local mean random weight)."
                             if self.has_nw else "nbar/alpha must equal the density of the randoms."))

    def _implied_alpha(self, k: int) -> float:
        """alpha implied by NZ (or NW) and a k-nearest-neighbour density of the randoms: median over
        a subsample of nbar / n_ran, or of m / (n_ran <w>_k) with <w>_k the mean weight of the
        neighbours (excluding the random itself) when NW is given."""
        sub = self.pos[: min(self.size, 20000)]
        d, idx = self.tree.query(sub, k=k + 1)
        n_ran = k / (4.0 / 3.0 * np.pi * d[:, -1] ** 3)
        if self.has_nw:
            wmean = self.w[idx[:, 1:]].mean(axis=1)
            return float(np.median(self.mw[: len(sub)] / (n_ran * wmean)))
        return float(np.median(self.nbar[: len(sub)] / n_ran))

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
        d, idx = self.tree.query(positions, k=1)
        inside = d <= self.mask_factor * self.spacing[idx]
        return np.where(inside, self.nbar[idx], 0.0), np.where(inside, self.w[idx], 0.0)

    def nbar_w_at_randoms_of(self, T: "Tracer"):
        """nbar_w_at(T.pos), cached: the nearest-random lookup at every random of T is the costly
        part of building the cross windows, and several of them need the same one."""
        if T is self:
            return self.nbar, self.w
        cache = self.__dict__.setdefault('_at_cache', {})
        if T.name not in cache:
            cache[T.name] = self.nbar_w_at(T.pos)
        return cache[T.name]

    def mw_w_at(self, positions: np.ndarray):
        """(m(x), w(x)) at arbitrary positions: the smooth mean weighted density and the weight of the
        nearest random (both zero outside the footprint)."""
        if positions is self.pos:
            return self.mw, self.w
        d, idx = self.tree.query(positions, k=1)
        inside = d <= self.mask_factor * self.spacing[idx]
        return np.where(inside, self.mw[idx], 0.0), np.where(inside, self.w[idx], 0.0)

    def mw_w_at_randoms_of(self, T: "Tracer"):
        """mw_w_at(T.pos), cached per tracer T."""
        if T is self:
            return self.mw, self.w
        cache = self.__dict__.setdefault('_mw_cache', {})
        if T.name not in cache:
            cache[T.name] = self.mw_w_at(T.pos)
        return cache[T.name]

    def nw_at(self, positions: np.ndarray) -> np.ndarray:
        """m(x) = nbar(x) w(x) (or NW) of this tracer at arbitrary positions (nearest random)."""
        return self.mw_w_at(positions)[0]

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
        """omega / nbar_host at the host randoms, i.e. the per-random weight with which the host
        randoms sample omega (cached on the host: a cross window needs a nearest-random lookup of
        the other tracer at every host random).

        W^{AB}: w_r m_B(x_r) (m_B = nbar_B w_B, or NW), so that alpha sum_r -> int m_A m_B.
        S^A:    scale (1 + alpha) w_r^2, each random with its OWN weight (the shot noise is <w^2>).
        """
        A = self.host
        cache = A.__dict__.setdefault('_tilde_cache', {})
        if self.key not in cache:
            if self.kind == 'W':
                m, _ = self.tracers[1].mw_w_at_randoms_of(A)
                cache[self.key] = A.w * m
            else:
                cache[self.key] = A.shotnoise_scale * (1.0 + A.alpha) * A.w ** 2
        return cache[self.key]

    def value_at(self, positions: np.ndarray) -> np.ndarray:
        """omega(x) at arbitrary positions (nearest-random interpolation).

        For S the nearest random's weight stands in for w(x): (1 + alpha) m w, whose average is
        exact for a smooth weight and ~<w>^2/<w^2> low with per-object weights; it only enters the
        s = 0 anchor of window pairs that involve S.
        """
        if self.kind == 'W':
            A, B = self.tracers
            return A.nw_at(positions) * B.nw_at(positions)
        A = self.host
        m, w = A.mw_w_at(positions)
        return A.shotnoise_scale * (1.0 + A.alpha) * m * w

    def integral(self) -> float:
        """int d^3x omega(x)  (e.g. I_AB for kind 'W')."""
        return self.host.alpha * float(np.sum(self.tilde_weights()))

    def value_at_host_randoms(self, T: Tracer) -> np.ndarray:
        """value_at(T.pos), cached on T (each needs a nearest-random lookup at every random of T)."""
        cache = T.__dict__.setdefault('_value_cache', {})
        if self.key not in cache:
            if self.kind == 'W':
                A, B = self.tracers
                cache[self.key] = A.mw_w_at_randoms_of(T)[0] * B.mw_w_at_randoms_of(T)[0]
            else:
                A = self.host
                m, w = A.mw_w_at_randoms_of(T)
                cache[self.key] = A.shotnoise_scale * (1.0 + A.alpha) * m * w
        return cache[self.key]

    def overlap_integral(self, other: "Window") -> float:
        """int d^3x omega(x) omega'(x): the s -> 0 anchor of the window pair function."""
        A = self.host
        return A.alpha * float(np.sum(self.tilde_weights() * other.value_at_host_randoms(A)))

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
