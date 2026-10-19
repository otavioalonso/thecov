"""Assembly of the Gaussian covariance of windowed power-spectrum multipoles (eq. final of the note).

    C^{ABCD}_{l1 l2}(i, j) = (4 pi)^4 / (I_AB I_CD) sum_{L1 L2} 1/((2L1+1)(2L2+1))
        sum_{lam lam'} (-1)^{(lam+lam')/2} sum_{Lam1 Lam2 Lam} [ t^(1) I^(1)_ij + t^(2) I^(2)_ij ],

    I^(1)_ij = sum_{(w,p) in P^{AD}} sum_{(w',p') in P^{CB}} int s^2 ds  pbar'^{(i)}_{L1 lam}  pbar^{(j)}_{L2 lam'}  Q^{w w'}_{Lam1 Lam2 Lam},
    I^(2)_ij = sum_{(w,p) in P^{AC}} sum_{(w',p') in P^{DB}} int s^2 ds  pbar'^{(i)}_{L1 lam}  pbar^{(j)}_{L2 lam'}  Q^{w w'}_{Lam1 Lam2 Lam}.

Structure: `_wick_terms` lists, for a pair of spectra, every (Wick term, window pair, model multipoles) combination;
`_couplings` gives, for one combination of multipoles, the (lam, lam') groups of window triples with their coefficients
(selection rules and t-coefficients, computed once). Both the window requests and the assembly iterate over these.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

import numpy as np

from . import defaults as D
from .kernels import PowerSpectrumModel, ShellKernels
from .tracers import Tracer, Window, spectrum_window_pairs
from .wigner import CouplingCoefficients, tri, tri_multi, FOUR_PI
from .windows import WindowLibrary

_COEFFS = CouplingCoefficients()


@lru_cache(maxsize=None)
def index_tuples(term, ell1, ell2, L1, L2):
    """All (lam, lam', Lam1, Lam2, Lam) allowed by the selection rules (Section 5.1)."""
    Lam1s = tri(ell1, L2) if term == 1 else tri_multi((ell1, ell2, L2))
    Lam2s = tri(L1, ell2) if term == 1 else (L1,)
    return tuple((lam, lamp, Lam1, Lam2, Lam)
                 for lam in tri(ell1, L1) for lamp in tri(ell2, L2) for Lam1 in Lam1s for Lam2 in Lam2s
                 for Lam in sorted(set(tri(lam, lamp)) & set(tri(Lam1, Lam2))))


@lru_cache(maxsize=None)
def _couplings(term, ell1, ell2, L1, L2):
    """{(lam, lam'): ((c, (Lam1, Lam2, Lam)), ...)}: the window triples of each (lam, lam') with their coefficient
    c = (4 pi)^4 (-1)^((lam+lam')/2) t / ((2L1+1)(2L2+1)); vanishing t are dropped."""
    out = {}
    for lam, lamp, Lam1, Lam2, Lam in index_tuples(term, ell1, ell2, L1, L2):
        t = _COEFFS.t(term, ell1, ell2, L1, L2, lam, lamp, Lam1, Lam2, Lam)
        if abs(t) >= 1e-15:
            c = FOUR_PI ** 4 * (-1) ** ((lam + lamp) // 2) / ((2 * L1 + 1) * (2 * L2 + 1)) * t
            out.setdefault((lam, lamp), []).append((c, (Lam1, Lam2, Lam)))
    return {k: tuple(v) for k, v in out.items()}


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
    shot_noise: include the shot-noise windows S^A
    smoothing : optional WindowSmoothing (smoothing.py): pair-averaged clustering windows m_A (K_k * m_B) with a
                xi-based kernel per k-bin instead of the local m_A m_B. Its reference densities and fiducial power
                must be set; it is built (for the k bins and tracers here) by compute_windows.
    Pair counts (see thecov/defaults.py): ds_pair, n_sub, n_near, s_split, min_pairs, seed, chunk_pairs, n_shells,
                n_mu; backend 'auto' (jax if importable) / 'jax' / 'numpy' (same histogram); n_threads.
    """

    def __init__(self, tracers, k_edges, ells=(0, 2, 4), L_max=4, s_max=None, ds=D.DS, ds_pair=D.DS_PAIR,
                 shot_noise=True, n_sub=D.N_SUB, n_near=D.N_NEAR, s_split=D.S_SPLIT, min_pairs=D.MIN_PAIRS,
                 seed=D.SEED, chunk_pairs=D.CHUNK_PAIRS, n_shells=D.N_SHELLS, n_mu=D.N_MU, backend='auto',
                 n_threads=None, smoothing=None):
        self.tracers = {t.name: t for t in tracers}
        self.k_edges = np.asarray(k_edges, dtype=float)
        self.nbins = len(self.k_edges) - 1
        self.ells = tuple(int(l) for l in ells)
        self.Ls = tuple(range(0, int(L_max) + 1, 2))
        self.shot_noise = bool(shot_noise)
        self.s_max = float(s_max if s_max is not None else max(t.s_max() for t in tracers))
        self.s = np.arange(0.5 * ds, self.s_max, ds)
        self.s_weights = self.s ** 2 * ds
        self.windows = WindowLibrary(np.arange(0.0, self.s_max + ds_pair, ds_pair), n_sub=n_sub, n_near=n_near,
                                     s_split=s_split, min_pairs=min_pairs, seed=seed, chunk_pairs=chunk_pairs,
                                     n_shells=n_shells, n_mu=n_mu, backend=backend, n_threads=n_threads)
        self.coeffs = _COEFFS
        self.kernels = ShellKernels(self.k_edges, self.s)
        self.model: PowerSpectrumModel | None = None
        self.masked = False
        self.smoothing = smoothing
        self._I = {}
        self._requested = set()

    index_tuples = staticmethod(index_tuples)

    # ------------------------------------------------------------------ the sums of the formula
    def _tracer(self, name) -> Tracer:
        return self.tracers[str(name)]

    def _Ls_for(self, power):
        """Model multipoles of a window's spectrum: 0 for shot noise, those of the model otherwise."""
        if power[0] == 'S':
            return (0,)
        if self.model is None:
            return self.Ls
        avail = self.model.multipoles(power[1], power[2])
        return tuple(L for L in self.Ls if L in avail)

    def _wick_terms(self, AB, CD):
        """(term, window_x, power_x, window_xp, power_xp) for the two Wick terms.

        Each correlator of a Wick term is a sum of (window, power) pieces, the set P^{XY} of the note
        (spectrum_window_pairs): the clustering window W^{XY} = m_X m_Y (or its smoothed basis windows) with the model
        P_L^{XY}, and for X = Y the shot-noise window S^X with p_L = delta_L0. window_x (the note's omega) is the piece
        of the first correlator, at x^, with the kernel of bin j and multipole L2; window_xp (omega') the piece of the
        second, at x'^, with bin i and L1. Each (window_x, window_xp) needs its own tripolar window Q^{omega omega'}.
        Term 1 pairs A with D and C with B, term 2 A with C and D with B."""
        (A, B), (C, D) = AB, CD
        pairs = lambda X, Y: spectrum_window_pairs(self._tracer(X), self._tracer(Y), self.shot_noise, self.smoothing)
        for term, (X, Xp) in ((1, ((A, D), (C, B))), (2, ((A, C), (D, B)))):
            for window_x, power_x in pairs(*X):
                for window_xp, power_xp in pairs(*Xp):
                    yield term, window_x, power_x, window_xp, power_xp

    # ------------------------------------------------------------------ normalisation
    def I(self, A, B) -> float:
        """The normalisation of P^AB: int W^AB from the randoms, unless set_normalization was used."""
        key = tuple(sorted((str(A), str(B))))
        if key not in self._I:
            self._I[key] = self.I_local(A, B)
        return self._I[key]

    def set_normalization(self, A, B, norm: float):
        """Use the normalisation the power-spectrum estimator actually divided by.

        Default (no call): I_AB = int m_A m_B = alpha sum_r m_B(x_r) w_r over the randoms, the standard FKP
        normalisation int nbar^2 w^2 of an FKP estimator normalised with the randoms; nothing to do for it.
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

    def _kernel(self, power, L, lam):
        """pbar^{(i)}_{L lam}(s) for every bin i, (nbins, ns): shot noise, the model, or the masked model."""
        if power[0] == 'S':
            return self.kernels.average(None, lam, tag=('S',))
        A, B = power[1], power[2]
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
                if (AB, CD, id(self.model)) in self._requested:      # the model sets the multipoles needed
                    continue
                for term, window_x, power_x, window_xp, power_xp in self._wick_terms(AB, CD):
                    trip = {t[2:] for ell1 in self.ells for ell2 in self.ells for L1 in self._Ls_for(power_xp)
                            for L2 in self._Ls_for(power_x) for t in index_tuples(term, ell1, ell2, L1, L2)}
                    self.windows.request(window_x, window_xp, trip)
                self._requested.add((AB, CD, id(self.model)))

    def build_smoothing(self, spectra, log=print):
        """Build the smoothing (basis, coefficients, smoothed m at the randoms) for the spectra's pairs."""
        if self.smoothing is not None:
            todo = [p for p in (tuple(str(x) for x in sp) for sp in spectra) if not self.smoothing.has(*p)]
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
            return 0.5 * (self._block_raw(AB, CD, ell1, ell2) + self._block_raw(CD, AB, ell2, ell1).T)
        return self._block_raw(AB, CD, ell1, ell2)

    def _block_raw(self, AB, CD, ell1: int, ell2: int) -> np.ndarray:
        """sum over Wick terms, window pairs and multipoles of u_i (s^2 ds sum_c c Q) v_j, / (I_AB I_CD)."""
        if self.model is None:
            raise RuntimeError("set_model() first")
        AB, CD = tuple(str(x) for x in AB), tuple(str(x) for x in CD)
        self.request_windows([AB, CD])
        if self.windows.missing():          # lazily compute whatever pair counts are missing
            self.windows.compute_all()
        C = np.zeros((self.nbins, self.nbins))
        for term, window_x, power_x, window_xp, power_xp in self._wick_terms(AB, CD):
            ci, cj = window_xp.coeffs_for(self.k_edges), window_x.coeffs_for(self.k_edges)   # basis-window coefficients
            for L1 in self._Ls_for(power_xp):
                for L2 in self._Ls_for(power_x):
                    for (lam, lamp), group in _couplings(term, ell1, ell2, L1, L2).items():
                        q = sum(c * self.windows.get(window_x, window_xp, *trip, self.s) for c, trip in group)
                        u = self._kernel(power_xp, L1, lam)     # (nbins, ns), bin i
                        v = self._kernel(power_x, L2, lamp)      # (nbins, ns), bin j
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
        ells = self.ells if ells is None else ells
        blocks = [(tuple(str(x) for x in sp), l) for sp in spectra for l in ells]
        t0 = time.time()

        def one(a, b):
            blk = self._block_raw(a[0], b[0], a[1], b[1])
            if verbose:
                print(f"block {a[0]} l={a[1]} x {b[0]} l={b[1]} done ({time.time() - t0:.1f}s)")
            return blk
        C = np.block([[one(a, b) for b in blocks] for a in blocks])
        labels = [(sp[0], sp[1], l, i) for (sp, l) in blocks for i in range(self.nbins)]
        return (0.5 * (C + C.T) if symmetrize else C), labels
