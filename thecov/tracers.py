"""Tracers (random catalogues) and the windows built from them.

A tracer is described by its randoms:
    positions : (N, 3) comoving Cartesian coordinates with the observer at the origin,
    weights   : (N,)   total weight w of each random (FKP x completeness ...), as applied to the density field,
    alpha     :        (weighted number of galaxies) / (weighted number of randoms),
so that the weighted randoms sample the mean weighted galaxy density W(x) = E[sum_g w_g delta_D(x - x_g)]:
sum_r w_r f(x_r) ~ int W f / alpha for any smooth f. The windows need W at each random, W = nbar <w>, with two
independent ingredients:

nbar (the mean density). 'auto' (default): alpha x the randoms' own number density, from the angular-radial split
    of the randoms: rho_r(chi) = (randoms in a radial shell) / (Omega x shell volume), with Omega the footprint
    solid angle from the occupancy of an equal-area grid on the sky (footprint_solid_angle). This assumes the
    randoms' sky distribution does not depend on distance (true for DESI randoms, up to Poisson noise). Omega is
    resolved to the grid cells (~30 randoms each), so veto holes smaller than a cell are counted as footprint:
    nbar is then low by the unvetoed fraction at that scale. That cancels in a Gaussian covariance whose model is
    a window-convolved measurement (GaussianCovariance with masked=True), not in terms that need the absolute
    density (super-sample, discreteness, trispectrum): pass nbar as an array when it is known (e.g. DESI's NX).
    An array (N,) or a callable of positions is used as is.

window_weighting (how the clustering window sees the weights). The clustering window is W_A W_B. When the weight
    varies from object to object (survey completeness, imaging-systematics and redshift-failure weights, as in
    DESI, where each random inherits the weight of a random data object), nbar w_r with a random's OWN weight gives
    the auto window ~ <w^2> instead of <w>^2, too large by 1 + var(w) / <w>^2. 'nearest' (default): W = nbar <w>_k,
    <w>_k the mean weight of the k_nearest nearest other randoms (3D); 'own': W = nbar w_r. Both agree for
    constant weights. Every random still carries its own weight as a sampling weight and in the shot-noise window.
The shot-noise window has the randoms' own integral, int S = (1 + alpha) alpha sum_r w_r^2; when the estimator
subtracted a different one (PowerSpectrum.shotnoise, pypower / jaxpower `num_shotnoise`), the covariance scales it
(GaussianCovariance.shotnoise_scale).

Cross windows W^{AB} are sampled by the randoms of A, with W_B taken from the nearest random of B and set to
zero when that random is farther than `mask_factor` x the LOCAL spacing of B's randoms, so that B's footprint is
respected. The footprint edge is therefore resolved to a few inter-random spacings: use dense random catalogues
for cross windows, and check I_AB against an independent estimate (it is exposed as GaussianCovariance.I).

Windows (eq. windows of the note), each sampled by the randoms of its host (the first tracer):
    W^{AB}(x) = W_A W_B                  (clustering; or W_A (K_b * W_B) with kernel smoothing)
    S^{A}(x)  = (1 + alpha_A) W_A w_A    (shot noise)
    M^{A}(x)  = W_A                      (mean weighted density; used by the super-sample covariance)
"""
from __future__ import annotations

import warnings
import zlib

import numpy as np
from scipy.spatial import cKDTree

CLUSTERING, SHOT_NOISE, MEAN_DENSITY = 'W', 'S', 'M'      # window kinds (the letters are part of the cache keys)


def footprint_solid_angle(directions, n_per_cell=30):
    """Solid angle [sr] covered by unit vectors `directions` (N, 3), uniformly distributed inside it: N x (cell area)
    / (mean count of the interior cells) on an equal-area grid (uniform in cos(theta) and phi) with ~n_per_cell
    points per occupied cell. Interior cells (all 8 neighbours occupied) are fully inside, so their mean count is an
    unbiased Poisson mean; the footprint edge does not enter."""
    u = np.clip(directions[:, 2], -1.0, 1.0)
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    index = lambda n: (np.minimum(((u + 1) / 2 * n).astype(np.int64), n - 1),
                       np.minimum(((phi + np.pi) / (2 * np.pi) * 2 * n).astype(np.int64), 2 * n - 1))
    n = 1
    while len(u) / len(np.unique(np.ravel_multi_index(index(2 * n), (2 * n, 4 * n)))) >= n_per_cell:
        n *= 2
    counts = np.zeros((n, 2 * n))
    np.add.at(counts, index(n), 1.0)
    occupied = counts > 0
    interior = occupied.copy()
    interior[0], interior[-1] = False, False                                 # no neighbours beyond the poles
    for du in (-1, 0, 1):
        for dp in (-1, 0, 1):
            interior &= np.roll(np.roll(occupied, du, axis=0), dp, axis=1)   # phi wraps; u rows checked above
    if not interior.any():
        raise ValueError('footprint_solid_angle: no interior cells (too few points or a too thin footprint)')
    return float(len(u) * 4 * np.pi / (2 * n * n) / counts[interior].mean())


def randoms_number_density(positions, dchi=10.0, n_per_cell=30):
    """Number density of the randoms at each random from the angular-radial split: counts in radial shells of
    width ~dchi [Mpc/h] (edges aligned with the randoms' range) over Omega (chi_2^3 - chi_1^3) / 3."""
    chi = np.linalg.norm(positions, axis=1)
    omega = footprint_solid_angle(positions / chi[:, None], n_per_cell)
    lo, hi = float(chi.min()), float(chi.max())
    edges = np.linspace(lo, hi * (1 + 1e-12), max(1, int(round((hi - lo) / dchi))) + 1)
    shell = np.minimum(np.searchsorted(edges, chi, side='right') - 1, len(edges) - 2)
    counts = np.bincount(shell, minlength=len(edges) - 1)
    return (counts / (omega * np.diff(edges ** 3) / 3))[shell]


def mean_weight_of_nearest(tree, weights, k=32, chunk=500_000):
    """Mean weight of the k nearest OTHER points of `tree` at each of its points."""
    out = np.empty(tree.n)
    for i0 in range(0, tree.n, chunk):
        _, idx = tree.query(tree.data[i0:i0 + chunk], k=k + 1, workers=-1)
        out[i0:i0 + chunk] = weights[idx[:, 1:]].mean(axis=1)
    return out


class Tracer:
    """A tracer's randoms with their weights, mean densities and footprint (see the module docstring).

        Tracer(name, positions, weights, alpha, nbar='auto', window_weighting='nearest', k_nearest=32)

    nbar             : 'auto' (the randoms' density from the angular-radial split, times alpha), or an array (N,) /
                       callable of positions
    window_weighting : 'nearest' (W = nbar x mean weight of the k_nearest nearest randoms) or 'own' (W = nbar w)
    mask_factor, mask_knn : footprint mask for cross windows (positions farther than mask_factor x the local random
                       spacing, from the mask_knn-th neighbour, are outside)
    Attributes: positions (N, 3), weights w (N,), alpha, nbar (N,), weighted_density W (N,), random_spacing.

    Deprecated: Tracer(name, randoms, alpha) with randoms = {'POSITION', 'WEIGHT'[, 'NZ'][, 'NW']} keeps its former
    meaning exactly: W = NW if given, else nbar w with nbar = NZ, or a knn-neighbour density estimate without NZ.
    """

    def __init__(self, name: str, positions, weights=None, alpha=None, nbar='auto', window_weighting: str = 'nearest',
                 k_nearest: int = 32, mask_factor: float = 2.5, mask_knn: int = 8, knn: int = 64):
        self.name = str(name)
        legacy = isinstance(positions, dict)
        if legacy:
            warnings.warn("Tracer(name, randoms_dict, alpha) is deprecated: use Tracer(name, positions, weights, alpha, "
                          "nbar=..., window_weighting=...)", DeprecationWarning, stacklevel=2)
            randoms, alpha = positions, (weights if alpha is None else alpha)
            positions, weights = randoms['POSITION'], randoms['WEIGHT']
        if weights is None or alpha is None:
            raise TypeError('Tracer needs positions, weights and alpha')
        self.positions = np.ascontiguousarray(positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError('positions must have shape (N, 3)')
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (self.size,):
            raise ValueError('weights must have one value per random')
        self.alpha = float(alpha)
        self._tree = None
        self._subsamples = {}
        if legacy:
            self.nbar, self.weighted_density = self._legacy_densities(randoms, None if isinstance(nbar, str) else nbar, knn)
            self.window_weighting = 'given' if 'NW' in randoms else 'own'
        else:
            self.nbar = self._as_array(self.alpha * randoms_number_density(self.positions) if isinstance(nbar, str) and nbar == 'auto'
                                       else (nbar(self.positions) if callable(nbar) else nbar), 'nbar')
            if window_weighting == 'nearest':
                self.weighted_density = self.nbar * mean_weight_of_nearest(self.tree, self.weights, k=int(k_nearest))
            elif window_weighting == 'own':
                self.weighted_density = self.nbar * self.weights
            else:
                raise ValueError(f"window_weighting must be 'nearest' or 'own', not {window_weighting!r}")
            self.window_weighting = window_weighting
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
    def _as_array(self, x, what):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.size,):
            raise ValueError(f'{what} must have one value per random')
        return x

    def _legacy_densities(self, randoms, nbar, knn):
        """(nbar, W) of the former dict interface: NW if given (nbar = NW / w), else nbar = the `nbar` argument, NZ, or
        a knn-neighbour density estimate, and W = nbar w."""
        nw = randoms.get('NW', None)
        if nbar is None:
            nbar = randoms.get('NZ', None)
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
        nbar = self._as_array(nbar, 'nbar')
        return nbar, (self._as_array(nw, 'NW') if nw is not None else nbar * self.weights)

    def _check_alpha(self, knn):
        """Warn if alpha is inconsistent with W and the density of the randoms.

        The randoms must sample W / alpha: the pair counts use alpha and the randoms, the s = 0 anchor and cross
        windows use W, and an inconsistent (alpha, W) would mis-normalise them relative to each other. The k-NN
        density averages over a sphere of ~(k / n_ran)^(1/3) (~20 Mpc/h for DESI randoms), so it includes the veto
        holes inside it while W is the density between them: for a footprint with fine veto masks the ratio is
        ~1 / (unvetoed fraction at that scale), e.g. 1.2 for DESI LRG, without anything being wrong.
        """
        # a random subsample: catalogues are often sorted (by z or sky position), and the first randoms then sit at
        # an edge, where the k-NN density is underestimated
        rng = np.random.default_rng(0)
        sub = np.sort(rng.choice(self.size, size=min(self.size, 20000), replace=False))
        d, idx = self.tree.query(self.positions[sub], k=knn + 1)
        n_randoms = knn / (4.0 / 3.0 * np.pi * d[:, -1] ** 3)
        ratio = float(np.median(self.weighted_density[sub] / (n_randoms * self.weights[idx[:, 1:]].mean(axis=1)))) / self.alpha
        if abs(ratio - 1) > 0.15:
            sub = np.random.default_rng(0).choice(self.size, size=min(self.size, 2000), replace=False)
            scale = float(np.median(self.tree.query(self.positions[sub], k=knn + 1)[0][:, -1]))
            warnings.warn(f"Tracer {self.name}: alpha={self.alpha:.4g} but the random density and W imply "
                          f"alpha~{self.alpha * ratio:.4g} (ratio {ratio:.2f}). Expected if the footprint has veto holes "
                          f"(ratio ~ 1 / unvetoed fraction on ~{scale:.0f} Mpc/h scales); otherwise W / nbar must be the "
                          f"local mean weight and nbar / alpha the density of the randoms.")

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
        """(W(x), w(x)) at arbitrary positions: the smooth mean weighted density and the weight of the nearest random
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
    W_b^{AB}(x) = W_A(x) (K_b * W_B)(x) instead of W_A W_B (see smoothing.py); `coeffs_for` then gives the per-k-bin
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

    def coeffs_for(self, kedges):
        """(nbins,) coefficients of this basis window in each k-bin (None for an unsmoothed window)."""
        if self.basis is None:
            return None
        A, B = self.tracers
        return self.smoothing.coeffs(A.name, B.name, kedges)[:, self.basis]

    def sampling_weights(self) -> np.ndarray:
        """The weight with which each host random samples omega, so that alpha sum_r (.) f(x_r) -> int omega f
        (cached on the host: a cross window needs a nearest-random lookup of the other tracer at every host random).

            W^{AB}: w_r W_B(x_r)              (W_A at the random is carried by the sampling itself)
            W_b^{AB}: w_r (K_b * W_B)(x_r)
            S^A:    (1 + alpha) w_r^2         (each random with its OWN weight: the shot noise is <w^2>)
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
                cache[self.key] = (1.0 + A.alpha) * A.weights ** 2
        return cache[self.key]

    def values_at_randoms_of(self, T: Tracer) -> np.ndarray:
        """omega(x) at the randoms of T (nearest-random interpolation of the other tracers; cached on T).

        For S the nearest random's weight stands in for w(x): (1 + alpha) W w, whose average is exact for a smooth
        weight and ~<w>^2/<w^2> low with per-object weights; it only enters the s = 0 anchor of window pairs with S.
        """
        cache = T.__dict__.setdefault('_window_values', {})
        if self.key not in cache:
            W_host, w_host = self.host.density_and_weight_at_randoms_of(T)
            if self.kind == CLUSTERING and self.basis is not None:
                A, B = self.tracers
                cache[self.key] = W_host * self.smoothing.values(A.name, B.name, self.basis, T.name)
            elif self.kind == CLUSTERING:
                cache[self.key] = W_host * self.tracers[1].density_and_weight_at_randoms_of(T)[0]
            elif self.kind == MEAN_DENSITY:
                cache[self.key] = W_host
            else:
                cache[self.key] = (1.0 + self.host.alpha) * W_host * w_host
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
    """The set P^{AB} of (window, spectrum) pairs of one correlator, eq. pairs of the note.

    spectrum labels: ('P', nameA, nameB) for the clustering multipoles and ('S', nameA) for shot noise (p_L = delta_L0).
    With a built `smoothing` for (A, B), the clustering window is replaced by its pair-averaged basis windows (each
    with its per-k-bin coefficients).
    """
    spectrum = ('P',) + tuple(sorted((A.name, B.name)))
    if smoothing is not None and smoothing.has(A.name, B.name):
        pairs = [(Window(CLUSTERING, A, B, smoothing=smoothing, basis=b), spectrum)
                 for b in range(smoothing.n_basis(A.name, B.name))]
    else:
        pairs = [(Window(CLUSTERING, A, B), spectrum)]
    if shot_noise and A.name == B.name:
        pairs.append((Window(SHOT_NOISE, A), ('S', A.name)))
    return pairs
