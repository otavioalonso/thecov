"""Assembly of the Gaussian covariance of windowed power-spectrum multipoles (eq. final of the note).

    C^{ABCD}_{l1 l2}(i, j) = (4 pi)^4 / (I_AB I_CD) sum_{L1 L2} 1/((2L1+1)(2L2+1))
        sum_{lam lam'} (-1)^{(lam+lam')/2} sum_{Lam1 Lam2 Lam} [ t^(1) I^(1)_ij + t^(2) I^(2)_ij ],

    I^(1)_ij = sum_{(w,p) in P^{AD}} sum_{(w',p') in P^{CB}} int s^2 ds  pbar'^{(i)}_{L1 lam}  pbar^{(j)}_{L2 lam'}  Q^{w w'}_{Lam1 Lam2 Lam},
    I^(2)_ij = sum_{(w,p) in P^{AC}} sum_{(w',p') in P^{DB}} int s^2 ds  pbar'^{(i)}_{L1 lam}  pbar^{(j)}_{L2 lam'}  Q^{w w'}_{Lam1 Lam2 Lam}.

Structure: `_wick_terms` lists, for a pair of spectra, every (Wick term, window pair, power-spectrum multipoles) combination;
`_couplings` gives, for one combination of multipoles, the (lam, lam') groups of window triples with their coefficients
(selection rules and t-coefficients, computed once). Both the window requests and the assembly iterate over these.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

import numpy as np

from . import defaults as D
from .power import PowerSpectrum, BinnedHankel
from .smoothing import WindowSmoothing
from .tracers import Tracer, Window, spectrum_window_pairs
from .wigner import CouplingCoefficients, tri, tri_multi, FOUR_PI
from .pair_counts import WindowLibrary

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
    power     : the power spectrum (power.py): a PowerSpectrum (one tracer), or {(A, B): PowerSpectrum} for several;
                the k bins of the covariance are those of the spectra (kedges, common to all), the multipoles L of
                the formula those given, the normalisation I(A, B) the spectrum's norm (else the FKP integral of the
                randoms), and the shot-noise window is scaled to the spectrum's shotnoise. Or only the bin edges
                kedges (h/Mpc), to compute the windows first and set_power later.
    ells      : the multipoles of the data vector (default: those of the spectra, else 0, 2, 4)
    s_max     : maximum separation (default: bounding box of the randoms)
    ds        : step of the s grid used for the radial integrals (Mpc/h)
    shot_noise: include the shot-noise windows S^A
    smoothing : the clustering window (smoothing.py):
                'auto' (default): pair-averaged windows W_A (K_k * W_B) with the xi-based kernel of each k-bin, i.e.
                    WindowSmoothing() with its defaults: reference density = each tracer's own randoms, kernel shape
                    = the spectrum's P_0 (so set the power spectrum before compute_windows);
                a WindowSmoothing: the same, with the reference densities (add_density; for DESI-like footprints
                    denser randoms than the tracer's: the near part of the kernel is a sum over them) or fiducial
                    powers (set_power) set by the caller, the defaults for the rest;
                None: the local window W_A W_B (exact for a box, 7-9 % low on footprints with fine veto masks).
                The smoothing is built (for the k bins and tracers here) by compute_windows.
    Pair counts (see thecov/defaults.py): ds_pair, n_sub, n_near, s_split, min_pairs, seed, chunk_pairs, n_shells,
                n_mu; backend 'auto' (jax if importable) / 'jax' / 'numpy' (same histogram); n_threads.
    """

    DEFAULT_MULTIPOLES = (0, 2, 4)     # ells and L when no power spectrum is set

    def __init__(self, tracers, power, ells=None, s_max=None, ds=D.DS, ds_pair=D.DS_PAIR,
                 shot_noise=True, n_sub=D.N_SUB, n_near=D.N_NEAR, s_split=D.S_SPLIT, min_pairs=D.MIN_PAIRS,
                 seed=D.SEED, chunk_pairs=D.CHUNK_PAIRS, n_shells=D.N_SHELLS, n_mu=D.N_MU, backend='auto',
                 n_threads=None, smoothing='auto'):
        self.tracers = {t.name: t for t in tracers}
        spectra = None if not isinstance(power, (PowerSpectrum, dict)) else self._as_spectra(power)
        if spectra is None:
            self.kedges = np.asarray(power, dtype=float)
        else:
            binnings = [p.kedges for p in spectra.values() if p.kedges is not None]
            if not binnings:
                raise ValueError("the power spectrum has no k binning: give it kedges")
            self.kedges = binnings[0]
        self.nbins = len(self.kedges) - 1
        self._ells = None if ells is None else tuple(int(l) for l in ells)
        self.shot_noise = bool(shot_noise)
        self.s_max = float(s_max if s_max is not None else max(t.s_max() for t in tracers))
        self.s = np.arange(0.5 * ds, self.s_max, ds)
        self.s_weights = self.s ** 2 * ds
        self.windows = WindowLibrary(np.arange(0.0, self.s_max + ds_pair, ds_pair), n_sub=n_sub, n_near=n_near,
                                     s_split=s_split, min_pairs=min_pairs, seed=seed, chunk_pairs=chunk_pairs,
                                     n_shells=n_shells, n_mu=n_mu, backend=backend, n_threads=n_threads)
        self.coeffs = _COEFFS
        self.hankel = BinnedHankel(self.kedges, self.s)
        self.power = {}
        self._power_version = 0
        self.smoothing = WindowSmoothing() if isinstance(smoothing, str) and smoothing == 'auto' else smoothing
        self._I_local = {}
        self._requested = set()
        if spectra is not None:
            self.set_power(spectra)

    index_tuples = staticmethod(index_tuples)

    @property
    def kmid(self) -> np.ndarray:
        return 0.5 * (self.kedges[1:] + self.kedges[:-1])

    @property
    def ells(self) -> tuple:
        """The multipoles of the data vector: as given, else those of the spectra (if they agree), else 0, 2, 4."""
        if self._ells is not None:
            return self._ells
        found = {p.ells for p in self.power.values()}
        if len(found) > 1:
            raise ValueError(f"the spectra have different ells {sorted(found)}: pass ells= to the covariance")
        return found.pop() if found else self.DEFAULT_MULTIPOLES

    # ------------------------------------------------------------------ power spectrum
    def _as_spectra(self, power) -> dict:
        """{(A, B) sorted: PowerSpectrum}; a single PowerSpectrum belongs to the only tracer."""
        if isinstance(power, PowerSpectrum):
            if len(self.tracers) != 1:
                raise ValueError("several tracers: pass {(A, B): PowerSpectrum} for the spectra")
            name = next(iter(self.tracers))
            power = {(name, name): power}
        return {tuple(sorted((str(A), str(B)))): p for (A, B), p in power.items()}

    def set_power(self, power):
        """The power spectrum P_L^{AB}(k): a PowerSpectrum (one tracer) or {(A, B): PowerSpectrum}.

        window_convolved=False: the power spectrum itself (e.g. theory), used as is.
        window_convolved=True : multipoles as measured (e.g. the data or the mean of mocks), normalised by
                                I(A, B) = power.norm (else int W_A W_B): their mean is P(k) I_k / I(A, B), so
                                they are multiplied by I(A, B) / I_k (mask_factor). Without smoothing,
                                I_k = int W_A W_B.
        Spectra without kedges take those of the covariance; others must have the same binning.
        """
        spectra = self._as_spectra(power)
        for key, p in spectra.items():
            if p.kedges is None:
                spectra[key] = p.binned_on(self.kedges)
            elif len(p.kedges) != len(self.kedges) or not np.allclose(p.kedges, self.kedges):
                raise ValueError(f"the k binning of the spectrum {key} differs from the covariance's")
        self.power = spectra
        self._power_version += 1
        self.hankel._cache.clear()
        return self

    def power_of(self, A, B) -> PowerSpectrum | None:
        """The PowerSpectrum of the pair (A, B) (either order), or None."""
        return self.power.get(tuple(sorted((str(A), str(B)))))

    # ------------------------------------------------------------------ the sums of the formula
    def _tracer(self, name) -> Tracer:
        return self.tracers[str(name)]

    def _Ls_for(self, spectrum):
        """Multipoles L of a window's spectrum: 0 for shot noise, those of the power spectrum otherwise (0, 2, 4
        before any is set, to request the windows)."""
        if spectrum[0] == 'S':
            return (0,)
        if not self.power:
            return self.DEFAULT_MULTIPOLES
        p = self.power_of(spectrum[1], spectrum[2])
        return () if p is None else tuple(p.multipoles())

    def _wick_terms(self, AB, CD):
        """(term, window_x, spectrum_x, window_xp, spectrum_xp) for the two Wick terms.

        Each correlator of a Wick term is a sum of (window, spectrum) pieces, the set P^{XY} of the note
        (spectrum_window_pairs): the clustering window W^{XY} = W_X W_Y (or its smoothed basis windows) with the
        multipoles P_L^{XY}, and for X = Y the shot-noise window S^X with p_L = delta_L0. window_x (the note's omega)
        is the piece of the first correlator, at x^, with the kernel of bin j and multipole L2; window_xp (omega') the
        piece of the second, at x'^, with bin i and L1. Each (window_x, window_xp) needs its own tripolar window
        Q^{omega omega'}. Term 1 pairs A with D and C with B, term 2 A with C and D with B."""
        (A, B), (C, D) = AB, CD
        pairs = lambda X, Y: spectrum_window_pairs(self._tracer(X), self._tracer(Y), self.shot_noise, self.smoothing)
        for term, (X, Xp) in ((1, ((A, D), (C, B))), (2, ((A, C), (D, B)))):
            for window_x, spectrum_x in pairs(*X):
                for window_xp, spectrum_xp in pairs(*Xp):
                    yield term, window_x, spectrum_x, window_xp, spectrum_xp

    # ------------------------------------------------------------------ normalisation and shot noise
    def I(self, A, B) -> float:
        """The normalisation of P^AB: the spectrum's norm (the estimator's), else int W_A W_B from the randoms.

        int W_A W_B = alpha sum_r W_B(x_r) w_r is the FKP normalisation int nbar^2 w^2 of an estimator normalised
        with the randoms. pypower and jaxpower paint data x randoms on a 10 Mpc/h mesh instead (their `norm`), a few
        per cent different (2 % low in the mock validation, i.e. 4 % in the variance): give it to the PowerSpectrum.
        """
        p = self.power_of(A, B)
        if p is not None and p.norm is not None:
            return p.norm
        return self.I_local(A, B)

    def I_local(self, A, B) -> float:
        """int W_A W_B (the local-approximation window integral)."""
        key = tuple(sorted((str(A), str(B))))
        if key not in self._I_local:
            self._I_local[key] = Window('W', self._tracer(A), self._tracer(B)).integral()
        return self._I_local[key]

    def shotnoise_scale(self, A) -> float:
        """Scale of the shot-noise window S^A: the auto spectrum's shotnoise (the estimator's int S) over the randoms'
        own (1 + alpha) alpha sum_r w_r^2, or 1."""
        p = self.power_of(A, A)
        if p is None or p.shotnoise is None:
            return 1.0
        T = self._tracer(A)
        return p.shotnoise / ((1.0 + T.alpha) * T.alpha * float(np.sum(T.weights ** 2)))

    def _window_scale(self, spectrum) -> float:
        return self.shotnoise_scale(spectrum[1]) if spectrum[0] == 'S' else 1.0

    def I_k(self, A, B) -> np.ndarray:
        """Per k-bin integral of the clustering window, int W_A (K_k * W_B) with smoothing, else
        int W_A W_B for every bin: the mean of the estimator is P(k) I_k / I(A, B)."""
        if self.smoothing is not None and self.smoothing.has(str(A), str(B)):
            return self.smoothing.I_k(str(A), str(B), self.kedges)
        return np.full(self.nbins, self.I_local(A, B))

    def mask_factor(self, A, B, k) -> np.ndarray:
        """I(A, B) / I_k(k): converts window-convolved multipoles into the power spectrum the
        covariance needs (interpolated between bin centres, constant beyond the first and last)."""
        return self.I(A, B) / np.interp(k, self.kmid, self.I_k(A, B))

    def _kernel(self, spectrum, L, lam):
        """pbar^{(i)}_{L lam}(s) for every bin i, (nbins, ns): shot noise, the power spectrum, or the window-convolved
        one times mask_factor."""
        if spectrum[0] == 'S':
            return self.hankel.transform(None, lam, tag=('S',))
        A, B = spectrum[1], spectrum[2]
        p = self.power_of(A, B)
        if p.window_convolved:
            return self.hankel.transform(p.function(L), lam, tag=('P', A, B, L, 'window_convolved'),
                                         factor=lambda k: self.mask_factor(A, B, k))
        return self.hankel.transform(p.function(L), lam, tag=('P', A, B, L))

    # ------------------------------------------------------------------ geometry
    def request_windows(self, spectra):
        """Register all window pairs / triples needed for the covariance of the listed spectra."""
        spectra = [tuple(str(x) for x in sp) for sp in spectra]
        self.build_smoothing(spectra)            # before the windows are listed (they depend on it)
        for AB in spectra:
            for CD in spectra:
                if (AB, CD, self._power_version) in self._requested:      # the spectra set the multipoles needed
                    continue
                for term, window_x, spectrum_x, window_xp, spectrum_xp in self._wick_terms(AB, CD):
                    trip = {t[2:] for ell1 in self.ells for ell2 in self.ells for L1 in self._Ls_for(spectrum_xp)
                            for L2 in self._Ls_for(spectrum_x) for t in index_tuples(term, ell1, ell2, L1, L2)}
                    self.windows.request(window_x, window_xp, trip)
                self._requested.add((AB, CD, self._power_version))

    def build_smoothing(self, spectra, log=print):
        """Build the smoothing (basis, coefficients, smoothed W at the randoms) for the spectra's pairs."""
        if self.smoothing is not None:
            todo = [p for p in (tuple(str(x) for x in sp) for sp in spectra) if not self.smoothing.has(*p)]
            if todo:
                self.smoothing.build(self.kedges, self.tracers, todo, power_of=self.power_of, log=log)
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

    def _block_raw(self, AB, CD, ell1: int, ell2: int, split: bool = False):
        """sum over Wick terms, window pairs and multipoles of u_i (s^2 ds sum_c c Q) v_j, / (I_AB I_CD).

        split=True returns {'PP', 'PS', 'SS'}: the contributions with zero, one and two shot-noise pieces, from the same
        windows (their sum is the block). The shot-noise windows enter through _window_scale only, so a covariance at
        shot noise S' is PP + (S'/S) PS + (S'/S)^2 SS exactly (single tracer; for cross pairs per tracer)."""
        if not self.power:
            raise RuntimeError("set the power spectrum first (set_power)")
        AB, CD = tuple(str(x) for x in AB), tuple(str(x) for x in CD)
        self.request_windows([AB, CD])
        if self.windows.missing():          # lazily compute whatever pair counts are missing
            self.windows.compute_all()
        C = np.zeros((self.nbins, self.nbins))
        parts = {0: np.zeros_like(C), 1: np.zeros_like(C), 2: np.zeros_like(C)} if split else None
        for term, window_x, spectrum_x, window_xp, spectrum_xp in self._wick_terms(AB, CD):
            n_s = (spectrum_x[0] == 'S') + (spectrum_xp[0] == 'S')
            ci, cj = window_xp.coeffs_for(self.kedges), window_x.coeffs_for(self.kedges)   # basis-window coefficients
            scale = self._window_scale(spectrum_x) * self._window_scale(spectrum_xp)        # shot-noise windows
            for L1 in self._Ls_for(spectrum_xp):
                for L2 in self._Ls_for(spectrum_x):
                    for (lam, lamp), group in _couplings(term, ell1, ell2, L1, L2).items():
                        q = sum(c * self.windows.get(window_x, window_xp, *trip, self.s) for c, trip in group)
                        if scale != 1.0:
                            q = scale * q
                        u = self._kernel(spectrum_xp, L1, lam)     # (nbins, ns), bin i
                        v = self._kernel(spectrum_x, L2, lamp)      # (nbins, ns), bin j
                        if ci is not None:
                            u = u * ci[:, None]
                        if cj is not None:
                            v = v * cj[:, None]
                        contrib = (u * (self.s_weights * q)) @ v.T
                        C += contrib
                        if split:
                            parts[n_s] += contrib
        norm = self.I(*AB) * self.I(*CD)
        if split:
            return {name: parts[n] / norm for n, name in ((0, 'PP'), (1, 'PS'), (2, 'SS'))}
        return C / norm

    def covariance_parts(self, spectra, ells=None, symmetrize=True):
        """The covariance split by the number of shot-noise pieces: {'PP', 'PS', 'SS'}, labels; PP + PS + SS equals
        covariance(). Gives the covariance at any other shot-noise level from one set of windows (e.g. for thinning or
        the discreteness decomposition) without recomputing the windows."""
        ells = self.ells if ells is None else ells
        blocks = [(tuple(str(x) for x in sp), l) for sp in spectra for l in ells]
        raw = [[self._block_raw(a[0], b[0], a[1], b[1], split=True) for b in blocks] for a in blocks]
        out = {}
        for name in ('PP', 'PS', 'SS'):
            C = np.block([[raw[i][j][name] for j in range(len(blocks))] for i in range(len(blocks))])
            out[name] = 0.5 * (C + C.T) if symmetrize else C
        labels = [(sp[0], sp[1], l, i) for (sp, l) in blocks for i in range(self.nbins)]
        return out, labels

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
