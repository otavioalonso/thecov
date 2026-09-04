"""Assembly of the Gaussian covariance of windowed power-spectrum multipoles (eq. final of the note).

    C^{ABCD}_{l1 l2}(i, j) = (4 pi)^4 / (I_AB I_CD) sum_{L1 L2} 1/((2L1+1)(2L2+1))
        sum_{lam lam'} (-1)^{(lam+lam')/2} sum_{Lam1 Lam2 Lam} [ t^(1) I^(1)_ij + t^(2) I^(2)_ij ],

    I^(1)_ij = sum_{(w,p) in P^{AD}} sum_{(w',p') in P^{CB}} int s^2 ds  pbar'^{(i)}_{L1 lam}  pbar^{(j)}_{L2 lam'}  Q^{w w'}_{Lam1 Lam2 Lam},
    I^(2)_ij = sum_{(w,p) in P^{AC}} sum_{(w',p') in P^{DB}} int s^2 ds  pbar'^{(i)}_{L1 lam}  pbar^{(j)}_{L2 lam'}  Q^{w w'}_{Lam1 Lam2 Lam}.
"""
from __future__ import annotations

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
    """

    def __init__(self, tracers, k_edges, ells=(0, 2, 4), L_max=4, s_max=None, ds=2.0, ds_pair=10.0,
                 shot_noise=True, n_sub=5000, n_near=200000, s_split=80.0, min_pairs=20, seed=0,
                 chunk_pairs=40000):
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
                                     min_pairs=min_pairs, seed=seed, chunk_pairs=chunk_pairs)
        self.coeffs = CouplingCoefficients()
        self.kernels = ShellKernels(self.k_edges, self.s)
        self.model: PowerSpectrumModel | None = None
        self._I = {}

    # ------------------------------------------------------------------ helpers
    def _tracer(self, name) -> Tracer:
        return self.tracers[str(name)]

    def _pairs(self, X, Y):
        return spectrum_window_pairs(self._tracer(X), self._tracer(Y), self.shot_noise)

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
        key = tuple(sorted((str(A), str(B))))
        if key not in self._I:
            self._I[key] = Window('W', self._tracer(A), self._tracer(B)).integral()
        return self._I[key]

    def _kernel(self, spec, L, lam):
        if spec[0] == 'S':
            return self.kernels.average(None, lam, tag=('S',))
        A, B = spec[1], spec[2]
        return self.kernels.average(lambda k: self.model(A, B, L, k), lam, tag=('P', A, B, L))

    # ------------------------------------------------------------------ geometry
    def request_windows(self, spectra):
        """Register all window pairs / triples needed for the covariance of the listed spectra."""
        spectra = [tuple(str(x) for x in sp) for sp in spectra]
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

    def compute_windows(self, spectra, verbose=False):
        """Pair counts for everything needed by `spectra` (a list of (A, B) tracer-name pairs)."""
        self.request_windows(spectra)
        t0 = time.time()
        self.windows.compute_all(verbose=verbose)
        if verbose:
            print(f"window functions done in {time.time() - t0:.1f}s")
        return self

    def save_windows(self, path):
        self.windows.save(path)

    def load_windows(self, path):
        self.windows.load(path)
        return self

    # ------------------------------------------------------------------ model
    def set_model(self, model: PowerSpectrumModel):
        self.model = model
        self.kernels._cache.clear()
        return self

    # ------------------------------------------------------------------ results
    def block(self, AB, CD, ell1: int, ell2: int) -> np.ndarray:
        """Covariance block Cov[P^{AB}_{ell1}(k_i), P^{CD}_{ell2}(k_j)] as an (nbins, nbins) array."""
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
                            for (lam, lamp), q in qsum.items():
                                u = self._kernel(spec_p, L1, lam)     # (nbins, ns), bin i
                                v = self._kernel(spec, L2, lamp)      # (nbins, ns), bin j
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
                blk = self.block(spA, spB, lA, lB)
                C[a * self.nbins:(a + 1) * self.nbins, b * self.nbins:(b + 1) * self.nbins] = blk
                if verbose:
                    print(f"block {spA} l={lA} x {spB} l={lB} done ({time.time() - t0:.1f}s)")
        if symmetrize:
            # exact for T1; the residual asymmetry comes from the x'^ -> x^ step in T2 (Section 7)
            C = 0.5 * (C + C.T)
        labels = [(sp[0], sp[1], l, i) for (sp, l) in blocks for i in range(self.nbins)]
        return C, labels
