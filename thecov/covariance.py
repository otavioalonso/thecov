"""Assembly of the Gaussian covariance of windowed power-spectrum multipoles (eq. final of the note).

    C^{ABCD}_{l1 l2}(i, j) = (4 pi)^4 / (I_AB I_CD) sum_{L1 L2} 1/((2L1+1)(2L2+1))
        sum_{lam lam'} (-1)^{(lam+lam')/2} sum_{Lam1 Lam2 Lam} [ t^(1) I^(1)_ij + t^(2) I^(2)_ij ],

    I^(1)_ij = sum_{(w,p) in P^{AD}} sum_{(w',p') in P^{CB}} int s^2 ds  pbar'^{(i)}_{L1 lam}  pbar^{(j)}_{L2 lam'}  Q^{w w'}_{Lam1 Lam2 Lam},
    I^(2)_ij = sum_{(w,p) in P^{AC}} sum_{(w',p') in P^{DB}} int s^2 ds  pbar'^{(i)}_{L1 lam}  pbar^{(j)}_{L2 lam'}  Q^{w w'}_{Lam1 Lam2 Lam}.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .kernels import PowerSpectrumModel, ShellKernels
from .tracers import Tracer, Window, spectrum_window_pairs
from .wigner import CouplingCoefficients, tri, tri_multi, FOUR_PI
from .windows import WindowLibrary


class GaussianCovariance:
    """Gaussian covariance of P_ell^{AB}(k_i) for one or several tracers with survey window.

    Parameters
    ----------
    tracers   : list of Tracer
    k_edges   : edges of the k bins of the covariance (h/Mpc)
    ells      : estimator multipoles (even)
    L_max     : highest power-spectrum multipole used in the model (even)
    s_max     : maximum separation (default: bounding box of the randoms)
    ds        : step of the s grid used for the radial integrals (Mpc/h)
    ds_pair   : radial bin width of the pair counts (Mpc/h); Q is smooth, coarse is fine
    shot_noise: include the shot-noise windows S^A
    n_sub     : randoms per tracer used for the far pairs (s >= s_split; all pairs, cost ~ n_sub^2)
    n_near    : randoms per tracer used for the near pairs (s < s_split; KD-tree, cost ~ n_near * density)
    s_split   : separation splitting the two regimes (Mpc/h)
    min_pairs : s bins with fewer pairs are dropped from the radial spline of Q
    n_shells, n_mu : resolution of the (r1, s, mu) histogram the pairs are binned into. S is
                evaluated at each cell's weighted mean, so the result is first-order accurate in the
                cell size. n_mu must resolve oscillations of order max(Lam1, Lam2) in mu and
                defaults to 6 x that (None = auto); n_shells controls only the weak s/r1 dependence.
    backend   : 'auto' (jax if importable, else numpy), 'jax', or 'numpy'. Only the pair-counting
                step differs; both backends produce the same histogram.
    cell_means: 'shared' (default) evaluates S in each (r1, s, mu) cell at the cell's unweighted mean,
                common to all window pairs counted together; 'weighted' uses each window pair's own
                weighted mean. The binning is bound by scatter-adds, and 'shared' needs K + 4 per pair
                instead of 4K + 1 for K window pairs. Both are exact to first order in the cell size;
                on the mock survey they differ by 2e-4 in the covariance (the cell-size error itself
                is ~1e-3 and the Monte-Carlo noise ~1e-2).
    n_threads : threads for the pair counts (None: min(16, cpu count)). Window pairs sampled by the
                same pair of randoms catalogues are counted together, in one pass.
    smoothing : optional WindowSmoothing (smoothing.py): pair-averaged clustering windows
                m_A (K_k * m_B) with a xi-based kernel per k-bin instead of the local m_A m_B. Its
                reference densities and fiducial power must be set; it is built (for the k bins and
                tracers here) by compute_windows.
    """

    def __init__(self, tracers, k_edges, ells=(0, 2, 4), L_max=4, s_max=None, ds=2.0, ds_pair=10.0,
                 shot_noise=True, n_sub=5000, n_near=200000, s_split=80.0, min_pairs=20, seed=0,
                 chunk_pairs=200000, n_shells=16, n_mu=None, backend='auto', n_threads=None,
                 cell_means='shared', smoothing=None):
        self.tracers = {t.name: t for t in tracers}
        self.k_edges = np.asarray(k_edges, dtype=float)
        self.nbins = len(self.k_edges) - 1
        self.ells = tuple(int(l) for l in ells)
        self.Ls = tuple(range(0, int(L_max) + 1, 2))
        self.shot_noise = bool(shot_noise)
        if s_max is None:
            s_max = max(t.s_max() for t in tracers)
        self.s_max = float(s_max)
        self.s = np.arange(0.5 * ds, self.s_max, ds)
        self.s_weights = self.s ** 2 * ds
        s_edges = np.arange(0.0, self.s_max + ds_pair, ds_pair)
        self.windows = WindowLibrary(s_edges, n_sub=n_sub, n_near=n_near, s_split=s_split,
                                     min_pairs=min_pairs, seed=seed, chunk_pairs=chunk_pairs,
                                     n_shells=n_shells, n_mu=n_mu,
                                     backend=backend, n_threads=n_threads, cell_means=cell_means)
        self.coeffs = CouplingCoefficients()
        self.kernels = ShellKernels(self.k_edges, self.s)
        self.model: PowerSpectrumModel | None = None
        self.masked = False
        self.smoothing = smoothing
        self._I = {}

    # ------------------------------------------------------------------ helpers
    def _tracer(self, name) -> Tracer:
        return self.tracers[str(name)]

    def _pairs(self, X, Y):
        return spectrum_window_pairs(self._tracer(X), self._tracer(Y), self.shot_noise, self.smoothing)

    def _term_pairs(self, AB, CD, term):
        """(P at x-hat [kernel j, L2], P at x'-hat [kernel i, L1]) for the two Wick terms."""
        A, B = AB
        C, D = CD
        if term == 1:
            return self._pairs(A, D), self._pairs(C, B)
        return self._pairs(A, C), self._pairs(D, B)

    def _Ls_for(self, spec):
        if spec[0] == 'S':
            return (0,)
        if self.model is None:
            return self.Ls
        avail = self.model.multipoles(spec[1], spec[2])
        return tuple(L for L in self.Ls if L in avail)

    @staticmethod
    def index_tuples(term, ell1, ell2, L1, L2):
        """All (lam, lam', Lam1, Lam2, Lam) allowed by the selection rules (Section 5.1)."""
        for lam in tri(ell1, L1):
            for lamp in tri(ell2, L2):
                Lam1s = tri(ell1, L2) if term == 1 else tri_multi((ell1, ell2, L2))
                Lam2s = tri(L1, ell2) if term == 1 else (L1,)
                for Lam1 in Lam1s:
                    for Lam2 in Lam2s:
                        for Lam in sorted(set(tri(lam, lamp)) & set(tri(Lam1, Lam2))):
                            yield lam, lamp, Lam1, Lam2, Lam

    def I(self, A, B) -> float:
        """The normalisation of P^AB: int W^AB from the randoms, unless set_normalization was used."""
        key = tuple(sorted((str(A), str(B))))
        if key not in self._I:
            self._I[key] = Window('W', self._tracer(A), self._tracer(B)).integral()
        return self._I[key]

    def set_normalization(self, A, B, norm: float):
        """Use the normalisation the power-spectrum estimator actually divided by.

        The covariance scales as 1 / (I_AB I_CD), so it must match the estimator. pypower and
        jaxpower default to painting data x randoms on a 10 Mpc/h mesh (their `wnorm` / `norm`),
        which differs from int nbar_A nbar_B w_A w_B by a few per cent (2 % low in the mock
        validation, i.e. 4 % in the variance). Pass their value here, e.g.
        `cov.set_normalization('A', 'A', poles.wnorm)`.
        """
        self._I[tuple(sorted((str(A), str(B))))] = float(norm)
        return self

    def I_local(self, A, B) -> float:
        """int m_A m_B (the local-approximation window integral)."""
        return Window('W', self._tracer(A), self._tracer(B)).integral()

    def I_k(self, A, B) -> np.ndarray:
        """Per k-bin integral of the clustering window, int m_A (K_k * m_B) with smoothing, else
        int m_A m_B for every bin: the mean of the estimator is P(k) I_k / I(A, B)."""
        if self.smoothing is not None and self.smoothing.has(str(A), str(B)):
            return self.smoothing.I_k(str(A), str(B), self.k_edges)
        return np.full(self.nbins, self.I_local(A, B))

    def mask_factor(self, A, B, k) -> np.ndarray:
        """I(A, B) / I_k(k): converts a masked (window-convolved) model into the power spectrum the
        covariance needs (interpolated between bin centres, constant beyond the first and last)."""
        centres = 0.5 * (self.k_edges[1:] + self.k_edges[:-1])
        return self.I(A, B) / np.interp(k, centres, self.I_k(A, B))

    def _kernel(self, spec, L, lam):
        if spec[0] == 'S':
            return self.kernels.average(None, lam, tag=('S',))
        A, B = spec[1], spec[2]
        if self.masked:
            return self.kernels.average(lambda k: self.model(A, B, L, k) * self.mask_factor(A, B, k), lam,
                                        tag=('P', A, B, L, 'masked'))
        return self.kernels.average(lambda k: self.model(A, B, L, k), lam, tag=('P', A, B, L))

    # ------------------------------------------------------------------ geometry
    def request_windows(self, spectra):
        """Register all window pairs / triples needed for the covariance of the listed spectra."""
        spectra = [tuple(str(x) for x in sp) for sp in spectra]
        self.build_smoothing(spectra)            # before the windows are listed (they depend on it)
        for AB in spectra:
            for CD in spectra:
                for term in (1, 2):
                    pairsX, pairsXp = self._term_pairs(AB, CD, term)
                    for (omega, spec) in pairsX:
                        for (omega_p, spec_p) in pairsXp:
                            trip = set()
                            for ell1 in self.ells:
                                for ell2 in self.ells:
                                    for L1 in self._Ls_for(spec_p):
                                        for L2 in self._Ls_for(spec):
                                            for (_, _, Lam1, Lam2, Lam) in self.index_tuples(term, ell1, ell2, L1, L2):
                                                trip.add((Lam1, Lam2, Lam))
                            self.windows.request(omega, omega_p, trip)

    def build_smoothing(self, spectra, log=print):
        """Build the smoothing (basis, coefficients, smoothed m at the randoms) for the spectra's pairs."""
        if self.smoothing is not None:
            pairs = [tuple(str(x) for x in sp) for sp in spectra]
            todo = [p for p in pairs if not self.smoothing.has(*p)]
            if todo:
                self.smoothing.build(self.k_edges, self.tracers, todo, log=log)
        return self

    def compute_windows(self, spectra, verbose=False):
        """Pair counts for everything needed by `spectra` (a list of (A, B) tracer-name pairs)."""
        self.build_smoothing(spectra, log=print if verbose else (lambda *a: None))
        self.request_windows(spectra)
        t0 = time.time()
        self.windows.compute_all(verbose=verbose)
        if verbose:
            print(f"window functions done in {time.time() - t0:.1f}s")
        return self

    def save_windows(self, path):
        self.windows.save(path)
        if self.smoothing is not None:
            self.smoothing.save(str(path) + '.smoothing.npz')

    def load_windows(self, path):
        self.windows.load(path)
        if self.smoothing is not None and os.path.exists(str(path) + '.smoothing.npz'):
            self.smoothing.load(str(path) + '.smoothing.npz')
        return self

    # ------------------------------------------------------------------ model
    def set_model(self, model: PowerSpectrumModel, masked: bool = False):
        """The model multipoles P_L^{AB}(k).

        masked=False: the power spectrum itself (e.g. theory), used as is.
        masked=True : window-convolved multipoles as measured, normalised by I(A, B) (e.g. the mean of
                      mocks): their mean is P(k) I_k / I(A, B), so they are multiplied by
                      I(A, B) / I_k (mask_factor). Without smoothing, I_k = int m_A m_B. Set the
                      estimator's normalisation (set_normalization) first.
        """
        self.model = model
        self.masked = bool(masked)
        self.kernels._cache.clear()
        return self

    # ------------------------------------------------------------------ results
    def block(self, AB, CD, ell1: int, ell2: int, symmetrize: bool = True) -> np.ndarray:
        """Covariance block Cov[P^{AB}_{ell1}(k_i), P^{CD}_{ell2}(k_j)] as an (nbins, nbins) array.

        In each correlator the derivation evaluates the power spectrum at one of the two momenta
        (eq. Pi of the note uses the momentum of the field without the Legendre weight). The other
        choice differs by O(q) = O(1/(k L_survey)) and is equally valid at this order. The two
        choices agree on the diagonal but not off it, so an individual block is not exactly the
        transpose of its exchanged partner: C^{ABCD}_{l1 l2}(i,j) vs C^{CDAB}_{l2 l1}(j,i). With
        symmetrize=True (default) the average of the two is returned, which is symmetric by
        construction and is the combination that enters covariance(). Pass symmetrize=False for the
        raw expression; the difference between them measures the ambiguity and is confined to the
        off-diagonal elements.
        """
        if symmetrize:
            direct = self.block(AB, CD, ell1, ell2, symmetrize=False)
            mirror = self.block(CD, AB, ell2, ell1, symmetrize=False)
            return 0.5 * (direct + mirror.T)
        return self._block_raw(AB, CD, ell1, ell2)

    def _block_raw(self, AB, CD, ell1: int, ell2: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("set_model() first")
        AB = tuple(str(x) for x in AB)
        CD = tuple(str(x) for x in CD)
        self.request_windows([AB, CD])
        if self.windows.missing():          # lazily compute whatever pair counts are missing
            self.windows.compute_all()
        C = np.zeros((self.nbins, self.nbins))
        for term in (1, 2):
            pairsX, pairsXp = self._term_pairs(AB, CD, term)
            for (omega, spec) in pairsX:            # at x-hat, kernel j, multipole L2
                for (omega_p, spec_p) in pairsXp:   # at x'-hat, kernel i, multipole L1
                    for L1 in self._Ls_for(spec_p):
                        for L2 in self._Ls_for(spec):
                            # group the Lambda sums per (lam, lam'): one matrix product each
                            qsum = {}
                            for (lam, lamp, Lam1, Lam2, Lam) in self.index_tuples(term, ell1, ell2, L1, L2):
                                t = self.coeffs.t(term, ell1, ell2, L1, L2, lam, lamp, Lam1, Lam2, Lam)
                                if abs(t) < 1e-15:
                                    continue
                                c = FOUR_PI ** 4 * (-1) ** ((lam + lamp) // 2) / ((2 * L1 + 1) * (2 * L2 + 1)) * t
                                Q = self.windows.get(omega, omega_p, Lam1, Lam2, Lam, self.s)
                                qsum[(lam, lamp)] = qsum.get((lam, lamp), 0.0) + c * Q
                            ci, cj = omega_p.coeffs_for(self.k_edges), omega.coeffs_for(self.k_edges)   # basis windows
                            for (lam, lamp), q in qsum.items():
                                u = self._kernel(spec_p, L1, lam)     # (nbins, ns), bin i
                                v = self._kernel(spec, L2, lamp)      # (nbins, ns), bin j
                                if ci is not None:
                                    u = u * ci[:, None]
                                if cj is not None:
                                    v = v * cj[:, None]
                                C += (u * (self.s_weights * q)) @ v.T
        return C / (self.I(*AB) * self.I(*CD))

    def covariance(self, spectra, ells=None, symmetrize=True, verbose=False):
        """Full covariance matrix for the data vector [P^{sp}_{ell}(k_i)] ordered by spectrum, ell, bin.

        Returns (matrix, labels) with labels a list of (A, B, ell, i).
        """
        if ells is None:
            ells = self.ells
        spectra = [tuple(str(x) for x in sp) for sp in spectra]
        blocks = [(sp, l) for sp in spectra for l in ells]
        n = len(blocks) * self.nbins
        C = np.zeros((n, n))
        t0 = time.time()
        for a, (spA, lA) in enumerate(blocks):
            for b, (spB, lB) in enumerate(blocks):
                blk = self.block(spA, spB, lA, lB, symmetrize=False)
                C[a * self.nbins:(a + 1) * self.nbins, b * self.nbins:(b + 1) * self.nbins] = blk
                if verbose:
                    print(f"block {spA} l={lA} x {spB} l={lB} done ({time.time() - t0:.1f}s)")
        if symmetrize:
            C = 0.5 * (C + C.T)
        labels = [(sp[0], sp[1], l, i) for (sp, l) in blocks for i in range(self.nbins)]
        return C, labels
