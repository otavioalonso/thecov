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
`shotnoise` sets the integral of the shot-noise window, int S = sum_g w_g^2 + alpha^2 sum_r w_r^2 as the estimator
subtracts it (pypower / jaxpower `num_shotnoise`), when the random weights do not reproduce the distribution of the
data weights; by default it is the randoms' own (1 + alpha) alpha sum_r w_r^2.

Cross windows W^{AB} are sampled by the randoms of A, with m_B taken from the nearest random of B and set to
zero when that random is farther than `mask_factor` x the LOCAL spacing of B's randoms, so that B's footprint is
respected. The footprint edge is therefore resolved to a few inter-random spacings: use dense random catalogues
for cross windows, and check I_AB against an independent estimate (it is exposed as GaussianCovariance.I).

Windows (eq. windows of the note), each sampled by the randoms of its host (the first tracer):
    W^{AB}(x) = m_A m_B                  (clustering; or m_A (K_b * m_B) with kernel smoothing)
    S^{A}(x)  = (1 + alpha_A) m_A w_A    (shot noise)
    M^{A}(x)  = m_A                      (mean weighted density; used by the super-sample covariance)
"""
from __future__ import annotations

import warnings
import zlib

import numpy as np
from scipy.spatial import cKDTree

CLUSTERING, SHOT_NOISE, MEAN_DENSITY = 'W', 'S', 'M'      # window kinds (the letters are part of the cache keys)


class Tracer:
    """A tracer's random catalogue with its weights, mean densities and footprint (see the module docstring).

    Attributes: positions (N, 3), weights w (N,), alpha, nbar (N,), weighted_density m (N,) (= NW, or nbar w),
    shotnoise_scale (S window scale giving the requested int S), random_spacing (local, for the footprint mask).
    `nbar` may be given (array or callable of positions) instead of 'NZ'; without NZ, NW or nbar it is estimated
    from a k-nearest-neighbour density of the randoms (`knn`).
    """

    def __init__(self, name: str, randoms: dict, alpha: float, nbar=None, knn: int = 64,
                 mask_factor: float = 2.5, mask_knn: int = 8, shotnoise=None):
        self.name = str(name)
        self.positions = np.ascontiguousarray(randoms['POSITION'], dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("randoms['POSITION'] must have shape (N, 3)")
        self.weights = np.asarray(randoms['WEIGHT'], dtype=float)
        self.alpha = float(alpha)
        # the S window is (1 + alpha) w^2 at the randoms, scaled to the requested int S
        self.shotnoise_scale = 1.0 if shotnoise is None else float(shotnoise) / (
            (1.0 + self.alpha) * self.alpha * float(np.sum(self.weights ** 2)))
        self._tree = None
        self._subsamples = {}
        nw = randoms.get('NW', None)
        self.nbar = self._mean_density(randoms.get('NZ', None) if nbar is None else nbar, nw, knn)
        # m(x): the mean weighted galaxy density entering the clustering windows
        self.has_weighted_density = nw is not None
        self.weighted_density = np.asarray(nw, dtype=float) if self.has_weighted_density else self.nbar * self.weights
        if self.weighted_density.shape != (self.size,):
            raise ValueError("NW must have one value per random")
        # Footprint mask used when this tracer's density is needed at foreign positions: a position is outside the
        # footprint if it is farther than mask_factor x the LOCAL random spacing from the nearest random. The
        # threshold has to be local: with a global value, a tracer whose density varies strongly (e.g. a steep n(z))
        # would have its sparse outskirts masked away. The spacing is estimated from the k-th neighbour
        # (k = mask_knn) rather than the first: the first-neighbour distance of a Poisson set has ~50 % scatter, so
        # a threshold built on it masks out interior positions whose nearest random happens to sit in a close pair.
        self.mask_factor = float(mask_factor)
        k = min(int(mask_knn), self.size - 1)
        self.random_spacing = self.tree.query(self.positions, k=k + 1)[0][:, k] / k ** (1.0 / 3.0)
        self._check_alpha(knn)

    # -- construction helpers --------------------------------------------------
    def _mean_density(self, nbar, nw, knn):
        """nbar at the randoms: given (array or callable of positions), NW / w, or a k-NN estimate."""
        if nbar is None and nw is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                nbar = np.where(self.weights != 0, np.asarray(nw, dtype=float) / self.weights, 0.0)
        if nbar is None:
            warnings.warn(f"Tracer {self.name}: no 'NZ' given; estimating nbar from the randoms "
                          f"with a {knn}-nearest-neighbour density estimate.")
            r = self.tree.query(self.positions, k=knn + 1)[0][:, -1]
            nbar = self.alpha * knn / (4.0 / 3.0 * np.pi * r ** 3)
        elif callable(nbar):
            nbar = nbar(self.positions)
        nbar = np.asarray(nbar, dtype=float)
        if nbar.shape != (self.size,):
            raise ValueError("nbar must have one value per random")
        return nbar

    def _check_alpha(self, knn):
        """Warn if alpha is inconsistent with NZ (or NW) and the density of the randoms.

        The randoms must sample nbar / alpha: the pair counts use alpha and the randoms, the s = 0 anchor and cross
        windows use NZ, and an inconsistent (alpha, NZ) would mis-normalise them relative to each other. The k-NN
        density averages over a sphere of ~(k / n_ran)^(1/3) (~20 Mpc/h for DESI randoms), so it includes the veto
        holes inside it while m is the density between them: for a footprint with fine veto masks the ratio is
        ~1 / (unvetoed fraction at that scale), e.g. 1.2 for DESI LRG, without anything being wrong.
        """
        # a random subsample: catalogues are often sorted (by z or sky position), and the first randoms then sit at
        # an edge, where the k-NN density is underestimated
        rng = np.random.default_rng(0)
        sub = np.sort(rng.choice(self.size, size=min(self.size, 20000), replace=False))
        d, idx = self.tree.query(self.positions[sub], k=knn + 1)
        n_randoms = knn / (4.0 / 3.0 * np.pi * d[:, -1] ** 3)
        if self.has_weighted_density:      # m / (n_ran <w>_k), <w>_k the mean weight of the neighbours
            implied = np.median(self.weighted_density[sub] / (n_randoms * self.weights[idx[:, 1:]].mean(axis=1)))
        else:
            implied = np.median(self.nbar[sub] / n_randoms)
        ratio = float(implied) / self.alpha
        if abs(ratio - 1) > 0.15:
            sub = np.random.default_rng(0).choice(self.size, size=min(self.size, 2000), replace=False)
            scale = float(np.median(self.tree.query(self.positions[sub], k=knn + 1)[0][:, -1]))
            what = 'NW' if self.has_weighted_density else 'NZ'
            warnings.warn(f"Tracer {self.name}: alpha={self.alpha:.4g} but the random density and {what} imply "
                          f"alpha~{self.alpha * ratio:.4g} (ratio {ratio:.2f}). Expected if the footprint has veto holes "
                          f"(ratio ~ 1 / unvetoed fraction on ~{scale:.0f} Mpc/h scales); otherwise "
                          + ("NW must equal alpha x (random density) x (local mean random weight)."
                             if self.has_weighted_density else "nbar/alpha must equal the density of the randoms."))

    # -- geometry ---------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.positions)

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.positions)
        return self._tree

    def s_max(self) -> float:
        """Upper bound on pair separations (bounding-box diagonal)."""
        return float(np.linalg.norm(self.positions.max(0) - self.positions.min(0)))

    def subsample_indices(self, n: int, seed: int = 0) -> np.ndarray:
        if (n, seed) not in self._subsamples:
            rng = np.random.default_rng(seed + zlib.crc32(self.name.encode()) % 10000)
            self._subsamples[(n, seed)] = np.sort(rng.choice(self.size, size=min(n, self.size), replace=False))
        return self._subsamples[(n, seed)]

    def density_and_weight_at(self, positions: np.ndarray):
        """(m(x), w(x)) at arbitrary positions: the smooth mean weighted density and the weight of the nearest random
        (both zero outside the footprint)."""
        if positions is self.positions:
            return self.weighted_density, self.weights
        d, idx = self.tree.query(positions, k=1)
        inside = d <= self.mask_factor * self.random_spacing[idx]
        return np.where(inside, self.weighted_density[idx], 0.0), np.where(inside, self.weights[idx], 0.0)

    def density_and_weight_at_randoms_of(self, other: "Tracer"):
        """density_and_weight_at(other.positions), cached per tracer: the nearest-random lookup at every random of
        the other tracer is the costly part of building the cross windows, and several of them need the same one."""
        if other is self:
            return self.weighted_density, self.weights
        cache = self.__dict__.setdefault('_at_randoms_of', {})
        if other.name not in cache:
            cache[other.name] = self.density_and_weight_at(other.positions)
        return cache[other.name]


class Window:
    """A window omega(x) over the survey, sampled by the randoms of its host (the first tracer).

    kind CLUSTERING ('W', two tracers), SHOT_NOISE ('S') or MEAN_DENSITY ('M', one tracer); see the module docstring.
    With `smoothing` (a WindowSmoothing) and `basis` b, the clustering window is the pair-averaged
    W_b^{AB}(x) = m_A(x) (K_b * m_B)(x) instead of m_A m_B (see smoothing.py); `coeffs_for` then gives the per-k-bin
    coefficients c_ib of K_{k_i} = sum_b c_ib K_b.
    """

    def __init__(self, kind: str, A: Tracer, B: Tracer | None = None, smoothing=None, basis: int | None = None):
        if kind not in (CLUSTERING, SHOT_NOISE, MEAN_DENSITY):
            raise ValueError("kind must be 'W', 'S' or 'M'")
        if basis is not None and (kind != CLUSTERING or smoothing is None):
            raise ValueError("a basis index needs kind 'W' and a smoothing")
        if kind == CLUSTERING and B is None:
            raise ValueError("clustering window needs two tracers")
        # the clustering window is symmetric: host = alphabetically first
        self.tracers = (tuple(sorted((A, B), key=lambda t: t.name)) if kind == CLUSTERING else (A,))
        self.kind, self.smoothing, self.basis = kind, smoothing, basis
        self.key = (kind,) + tuple(t.name for t in self.tracers) + ((f'k{basis}:{smoothing.tag}',) if basis is not None else ())

    @property
    def host(self) -> Tracer:
        return self.tracers[0]

    def __repr__(self):
        return "Window" + str(self.key)

    def coeffs_for(self, k_edges):
        """(nbins,) coefficients of this basis window in each k-bin (None for an unsmoothed window)."""
        if self.basis is None:
            return None
        A, B = self.tracers
        return self.smoothing.coeffs(A.name, B.name, k_edges)[:, self.basis]

    def sampling_weights(self) -> np.ndarray:
        """The weight with which each host random samples omega, so that alpha sum_r (.) f(x_r) -> int omega f
        (cached on the host: a cross window needs a nearest-random lookup of the other tracer at every host random).

            W^{AB}: w_r m_B(x_r)              (m_A at the random is carried by the sampling itself)
            W_b^{AB}: w_r (K_b * m_B)(x_r)
            S^A:    scale (1 + alpha) w_r^2   (each random with its OWN weight: the shot noise is <w^2>)
            M^A:    w_r
        """
        A = self.host
        cache = A.__dict__.setdefault('_sampling_weights', {})
        if self.key not in cache:
            if self.kind == CLUSTERING and self.basis is not None:
                cache[self.key] = A.weights * self.smoothing.values(A.name, self.tracers[1].name, self.basis, A.name)
            elif self.kind == CLUSTERING:
                cache[self.key] = A.weights * self.tracers[1].density_and_weight_at_randoms_of(A)[0]
            elif self.kind == MEAN_DENSITY:
                cache[self.key] = np.asarray(A.weights, float).copy()
            else:
                cache[self.key] = A.shotnoise_scale * (1.0 + A.alpha) * A.weights ** 2
        return cache[self.key]

    def values_at_randoms_of(self, T: Tracer) -> np.ndarray:
        """omega(x) at the randoms of T (nearest-random interpolation of the other tracers; cached on T).

        For S the nearest random's weight stands in for w(x): (1 + alpha) m w, whose average is exact for a smooth
        weight and ~<w>^2/<w^2> low with per-object weights; it only enters the s = 0 anchor of window pairs with S.
        """
        cache = T.__dict__.setdefault('_window_values', {})
        if self.key not in cache:
            m_host, w_host = self.host.density_and_weight_at_randoms_of(T)
            if self.kind == CLUSTERING and self.basis is not None:
                A, B = self.tracers
                cache[self.key] = m_host * self.smoothing.values(A.name, B.name, self.basis, T.name)
            elif self.kind == CLUSTERING:
                cache[self.key] = m_host * self.tracers[1].density_and_weight_at_randoms_of(T)[0]
            elif self.kind == MEAN_DENSITY:
                cache[self.key] = m_host
            else:
                cache[self.key] = self.host.shotnoise_scale * (1.0 + self.host.alpha) * m_host * w_host
        return cache[self.key]

    def integral(self) -> float:
        """int d^3x omega(x)  (e.g. I_AB for kind 'W')."""
        return self.host.alpha * float(np.sum(self.sampling_weights()))

    def overlap_integral(self, other: "Window") -> float:
        """int d^3x omega(x) omega'(x): the s -> 0 anchor of the window pair function."""
        A = self.host
        return A.alpha * float(np.sum(self.sampling_weights() * other.values_at_randoms_of(A)))

    def sample(self, n_sub: int, seed: int = 0):
        """(positions, sampling weights, effective alpha) of a subsample of the host randoms."""
        A = self.host
        idx = A.subsample_indices(n_sub, seed)
        return A.positions[idx], self.sampling_weights()[idx], A.alpha * A.size / len(idx)


def spectrum_window_pairs(A: Tracer, B: Tracer, shot_noise: bool = True, smoothing=None):
    """The set P^{AB} of (window, power) pairs of one correlator, eq. pairs of the note.

    power labels: ('P', nameA, nameB) for the clustering multipoles and ('S', nameA) for shot noise (p_L = delta_L0).
    With a built `smoothing` for (A, B), the clustering window is replaced by its pair-averaged basis windows (each
    with its per-k-bin coefficients).
    """
    power = ('P',) + tuple(sorted((A.name, B.name)))
    if smoothing is not None and smoothing.has(A.name, B.name):
        pairs = [(Window(CLUSTERING, A, B, smoothing=smoothing, basis=b), power)
                 for b in range(smoothing.n_basis(A.name, B.name))]
    else:
        pairs = [(Window(CLUSTERING, A, B), power)]
    if shot_noise and A.name == B.name:
        pairs.append((Window(SHOT_NOISE, A), ('S', A.name)))
    return pairs
