import logging, os
import itertools as itt

import numpy as np

from . import base, geometry, math, utils

cache_dir = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(cache_dir, exist_ok=True)


class MultiTracerGaussianCovariance:
    """Computes the Gaussian Covariance matrix C(AB, CD) for arbitrary tracers.
    
    A, B, C, D are identifiers (e.g., strings) for the tracers. AB is the first 
    cross spectrum and CD is the second.
    """

    def __init__(self, pk_ellmax=4, mask_ellmax=4):
        self.logger = logging.getLogger('MultiTracerGaussianCovariance')
        self.pk_ellmax = pk_ellmax
        self.mask_ellmax = mask_ellmax
        
        self.auto_windows = {}  # Map tracer_name -> SingleTracerSurveyGeometry
        self.cross_windows = {}  # Map frozenset(tracerA, tracerB) -> MultiTracerSurveyGeometry
        self._pk = {}      # Map frozenset(tracerA, tracerB) -> dict(ell -> pk)

    def add_auto_window(self, name, geometry):
        """Add a SingleTracerSurveyGeometry object for a specific tracer."""
        self.auto_windows[name] = geometry

    def add_cross_window(self, nameA, nameB, geometry):
        """Add a MultiTracerSurveyGeometry object describing the combined window of tracer A and B."""
        pair = frozenset([nameA, nameB])
        self.cross_windows[pair] = geometry

    def set_pk_multipole(self, nameA, nameB, ell, pk):
        """Set P_AB_ell(k)."""
        pair = frozenset([nameA, nameB])
        if pair not in self._pk:
            self._pk[pair] = {}
        self._pk[pair][ell] = pk

    def get_pk(self, nameA, nameB, ell):
        """Get P_AB_ell(k)."""
        pair = frozenset([nameA, nameB])
        if pair in self._pk and ell in self._pk[pair]:
            return self._pk[pair][ell]
        return None

    def _get_window(self, A, B):
        """Helper to get window_matrix for pair (A, B).
        If A==B, gets the density window matrix from SingleTracerSurveyGeometry.
        If A!=B, gets from MultiTracerSurveyGeometry.
        Always returns shape [power_configs, ellm, nu1, nu2, k1, k2]
        For MultiTracer it only has power_configs=1, while SingleTracer has 2.
        """
        if A == B:
            return self.auto_windows[A].window_matrix
        else:
            return self.cross_windows[frozenset([A, B])].window_matrix

    def _get_shotnoise_window(self, A):
        """Helper to get the shotnoise geometry for a tracer A. Uses window_matrix[1]."""
        return self.auto_windows[A].shotnoise_window_matrix

    def compute_covariance_block(self, A, B, C, D):
        """Computes the full Gaussian Covariance Block C(AB, CD)."""
        from thecov.math import get_gaunt_coefficients
        
        # We need the 4 combinations of cross-windows
        W_AC = self._get_window(A, C)[0]
        W_BD = self._get_window(B, D)[0]
        W_AD = self._get_window(A, D)[0]
        W_BC = self._get_window(B, C)[0]
        
        window_kernel = {}

        ells = 0, 2, 4
        accumulator = \
            get_gaunt_coefficients('cosmic_variance_ACBD', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ \
                np.einsum('abcij,xyzij->bcayzxij', W_AC, W_BD) @ \
                np.array([[np.outer(self.get_pk(A, C, l1), self.get_pk(B, D, l2)) for l1 in ells] for l2 in ells])

        accumulator += \
            get_gaunt_coefficients('cosmic_variance_ADBC', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ \
            np.einsum('abcij,xyzij->bcayzxij', W_AD, W_BC) @ \
            np.array([[np.outer(self.get_pk(A, D, l1), self.get_pk(B, C, l2)) for l1 in ells] for l2 in ells])

        ### 2. Mixed Term (Requires A==C, A==D, B==C, or B==D)
        
        # For each valid delta_Kronecker, we compute W_Shot * W_Cross
        if A == C:
            accumulator += \
                get_gaunt_coefficients('mixed_ACBD', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ \
                np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(A), W_BD) @ \
                np.array([self.get_pk(A, C, l) for l in ells])
        if A == D:
            accumulator += \
                get_gaunt_coefficients('mixed_ADBC', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ \
                np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(A), W_BC) @ \
                np.array([self.get_pk(A, D, l) for l in ells])
        if B == C:
            accumulator += \
                get_gaunt_coefficients('mixed_BCAD', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ \
                np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(B), W_AD) @ \
                np.array([self.get_pk(B, C, l) for l in ells])
        if B == D:
            accumulator += \
                get_gaunt_coefficients('mixed_BDAC', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ \
                np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(B), W_AC) @ \
                np.array([self.get_pk(B, D, l) for l in ells])

        ### 3. Shotnoise Term (Requires (A==C and B==D) or (A==D and B==C))
        
        if A == C and B == D:
            accumulator += get_gaunt_coefficients('shotnoise_ACBD', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ \
                np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(A), self._get_shotnoise_window(B))

        if A == D and B == C:
            accumulator += \
                get_gaunt_coefficients('shotnoise_ADBC', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ \
                np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(A), self._get_shotnoise_window(B))
            
        self.cov = accumulator

        return accumulator

# Retain old wrapper for single-tracer
class GaussianCovariance(MultiTracerGaussianCovariance):
    def __init__(self):
        super().__init__()

    def set_tracer(self, geometry):
        """Simplifies setup for 1 tracer context."""
        self.add_tracer("0", geometry)

    def set_galaxy_pk_multipole(self, pk, ell):
        self.set_pk_multipole("0", "0", ell, pk)

    def get_pk(self, ell):
        return super().get_pk("0", "0", ell)
        
    def compute_window_kernels(self):
        """Returns C(AA, AA) window kernels."""
        return self.compute_covariance_block("0", "0", "0", "0")