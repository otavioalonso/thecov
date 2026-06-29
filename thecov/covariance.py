import logging, os
import numpy as np

cache_dir = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(cache_dir, exist_ok=True)


class MultiTracerGaussianCovariance:
    """Computes the Gaussian Covariance matrix C(AB, CD) for arbitrary tracers.
    
    A, B, C, D are identifiers (e.g., strings) for the tracers. AB is the first 
    cross spectrum and CD is the second.
    """

    def __init__(self, pk_ellmax=4, mask_ellmax=4, alpha=0.0):
        self.logger = logging.getLogger('MultiTracerGaussianCovariance')
        self.pk_ellmax = pk_ellmax
        self.mask_ellmax = mask_ellmax
        self.auto_windows = {}  # Map tracer_name -> SingleTracerSurveyGeometry
        self.cross_windows = {}  # Map frozenset(tracerA, tracerB) -> MultiTracerSurveyGeometry
        self._pk = {}      # Map frozenset(tracerA, tracerB) -> dict(ell -> pk)
        self.alpha = alpha  # Shotnoise scaling factor

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
        Always returns shape [ellm, nu1, nu2, k1, k2]
        """
        if A == B:
            return self.auto_windows[A].window_matrix
        else:
            return self.cross_windows[frozenset([A, B])].window_matrix

    def _get_shotnoise_window(self, A):
        """Helper to get the shotnoise geometry for a tracer A. Uses window_matrix[1]."""
        return (1 + self.alpha) * self.auto_windows[A].shotnoise_window_matrix

    def _binning_geometry(self):
        """Return any registered geometry, used for the k-binning and volume
        (all windows in a block share the same k-bins and survey volume)."""
        if self.auto_windows:
            return next(iter(self.auto_windows.values()))
        return next(iter(self.cross_windows.values()))

    def _pair_normalization(self, A, B):
        """FKP normalization I_AB of the (A, B) power-spectrum estimator.

        For an auto pair (A == B) this is I_22 = sum(nbar * w^2 * w_sys),
        matching SingleTracerSurveyGeometry.normalization(2, 2) and the
        reference cosmodesi/thecov I('22'). For a cross pair it falls back to
        MultiTracerSurveyGeometry.normalization().
        """
        if A == B:
            return self.auto_windows[A].normalization(2, 2)
        return self.cross_windows[frozenset([A, B])].normalization()

    def compute_covariance_block(self, A, B, C, D):
        """Computes the full Gaussian Covariance Block C(AB, CD)."""
        from thecov.math import get_gaunt_coefficients
        
        # We need the 4 combinations of cross-windows
        W_AC = self._get_window(A, C)
        W_BD = self._get_window(B, D)
        W_AD = self._get_window(A, D)
        W_BC = self._get_window(B, C)
        
        # All even multipoles 0, 2, ..., pk_ellmax. This must match the
        # ell-dimension of the Gaunt tensors (pk_ellmax//2 + 1 entries);
        # a single-element range would silently broadcast in the einsum and
        # use P_{pk_ellmax} for every input multipole.
        ells = np.arange(0, self.pk_ellmax + 1, 2)

        product = get_gaunt_coefficients('cosmic_variance_ACBD', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ np.einsum('abcij,xyzij->bcayzxij', W_AC, W_BD)
        covariance = np.einsum('abcdij,ci,dj->abij',
                product.to_dense(),
                np.array([(4*np.pi)*self.get_pk(A, C, l)/(2*l + 1) for l in ells]),
                np.array([(4*np.pi)*self.get_pk(B, D, l)/(2*l + 1) for l in ells]))

        product = get_gaunt_coefficients('cosmic_variance_ADBC', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ np.einsum('abcij,xyzij->bcayzxij', W_AD, W_BC)
        covariance += np.einsum('abcdij,ci,dj->abij',
                product.to_dense(),
                np.array([(4*np.pi)*self.get_pk(A, D, l)/(2*l + 1) for l in ells]),
                np.array([(4*np.pi)*self.get_pk(B, C, l)/(2*l + 1) for l in ells]))

        ### 2. Mixed Term (Requires A==C, A==D, B==C, or B==D)
        
        # For each valid delta_Kronecker, we compute W_Shot * W_Cross
        # Each mixed term is SN_{delta-pair} * P_{complementary-pair}:
        # delta_AC -> SN_A * P_BD,  delta_BD -> SN_B * P_AC, etc.
        if A == C:
            product = get_gaunt_coefficients('mixed_ACBD', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(A), W_BD)
            covariance += np.einsum('abcij,cj->abij',
                product.to_dense(),
                np.array([(4*np.pi)*self.get_pk(B, D, l)/(2*l + 1) for l in ells]))
        if A == D:
            product = get_gaunt_coefficients('mixed_ADBC', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(A), W_BC)
            covariance += np.einsum('abcij,cj->abij',
                product.to_dense(),
                np.array([(4*np.pi)*self.get_pk(B, C, l)/(2*l + 1) for l in ells]))
        if B == C:
            product = get_gaunt_coefficients('mixed_BCAD', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(B), W_AD)
            covariance += np.einsum('abcij,cj->abij',
                product.to_dense(),
                np.array([(4*np.pi)*self.get_pk(A, D, l)/(2*l + 1) for l in ells]))
        if B == D:
            product = get_gaunt_coefficients('mixed_BDAC', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(B), W_AC)
            covariance += np.einsum('abcij,cj->abij',
                product.to_dense(),
                np.array([(4*np.pi)*self.get_pk(A, C, l)/(2*l + 1) for l in ells]))

        ### 3. Shotnoise Term (Requires (A==C and B==D) or (A==D and B==C))
        
        if A == C and B == D:
            product = get_gaunt_coefficients('shotnoise_ACBD', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(A), self._get_shotnoise_window(B))
            covariance += product.to_dense()

        if A == D and B == C:
            product = get_gaunt_coefficients('shotnoise_ADBC', pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax) @ np.einsum('abcij,xyzij->bcayzxij', self._get_shotnoise_window(A), self._get_shotnoise_window(B))
            covariance += product.to_dense()

        covariance /= self._pair_normalization(A, B) * self._pair_normalization(C, D)
        covariance *= 2*(4*np.pi)**2

        b = self._binning_geometry()
        nmodes = b.nmodes
        covariance = covariance * np.sqrt(np.multiply.outer(nmodes, nmodes))[None, None, :, :]

        self.cov = covariance

        return covariance

# Retain old wrapper for single-tracer
class SingleTracerGaussianCovariance(MultiTracerGaussianCovariance):
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