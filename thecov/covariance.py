"""Module for computing the covariance matrix of power spectrum multipoles.

Example
-------
>>> from thecov import SurveyGeometry, GaussianCovariance
>>> geometry = SurveyGeometry(random_catalog=randoms, alpha=1. / 10)
>>> covariance = GaussianCovariance(geometry)
>>> covariance.set_kbins(kmin=0, kmax=0.25, dk=0.005)
>>> covariance.set_galaxy_pk_multipole(P0, 0, has_shotnoise=False)
>>> covariance.set_galaxy_pk_multipole(P2, 2)
>>> covariance.set_galaxy_pk_multipole(P4, 4)
>>> covariance.compute_covariance_box()
"""

import logging, os
import itertools as itt
import numpy as np

from . import base, geometry, math

__all__ = ['GaussianCovariance',
           'TrispectrumCovariance',
           'SuperSampleCovariance']

TRACER_LABELS = ['A', 'B', 'C', 'D']

cache_dir = os.path.join(os.path.dirname(__file__), "cache")
# Create the cache directory only on rank 0 when running under MPI to
# avoid race conditions
try:
    from mpi4py import MPI
    if MPI.Is_initialized():
        _cov_rank = MPI.COMM_WORLD.Get_rank()
        if _cov_rank == 0:
            os.makedirs(cache_dir, exist_ok=True)
        MPI.COMM_WORLD.Barrier()
    else:
        os.makedirs(cache_dir, exist_ok=True)
except Exception:
    os.makedirs(cache_dir, exist_ok=True)
    
class PowerSpectrumCovariance(base.MultipoleFourierCovariance):
    '''Parent covariance matrix class handling power spectrum multipoles assignment, shotnoise, etc.

    Attributes
    ----------
    geometry : survey_geometry.Geometry
        Geometry of the survey. Can be a BoxGeometry or a SurveyGeometry object.
    alpha : list of float
        List of alpha parameters for each tracer, where alpha = N_galaxies / N_randoms. This is the alpha used in the P(k) measurements, and can be different from the alpha used in the geometry object.
    pk_renorm : float
        Relative normalization of the power spectrum. This is determined by comparing the estimated FKP shotnoise with the given shotnoise value, and is used to rescale the input power spectrum for the covariance calculation.
    '''

    def __init__(self, geometry:geometry.SurveyGeometry=None):
        super().__init__()
        self.logger = logging.getLogger('PowerSpectrumCovariance')
        self.geometry = geometry
        
        self._pk = {}
        self.num_tracers = geometry.num_tracers
        self.num_spectra = int(self.num_tracers * (self.num_tracers+1)/2)
        self.pk_renorm = 1

    @property
    def alpha(self):
        '''The value of alpha. This is the alpha used in the Pk measurements.
           It can be different from the alpha used in the geometry object.

        Returns
        -------
        float
            The value of alpha.
        '''
        if not hasattr(self, '_alpha'):
            return self.geometry.alphas
        else:
            return self._alpha
    
    @alpha.setter
    def alpha(self, alpha):
        '''Sets the value of alpha. This is the alpha used in the P(k) measurements.
           It can be different from the alpha used in the geometry object.

        Parameters
        ----------
        alpha : float
            The value of alpha.
        '''
        self._alpha = alpha

    def compute_covariance(self):
        '''Compute the covariance matrix for the given geometry and power spectra.

        Parameters
        ----------
        ells : tuple, optional
            Multipoles of the power spectra to have the covariance calculated for.
        '''

        if isinstance(self.geometry, geometry.BoxGeometry):
            return self._compute_covariance_box()

        if isinstance(self.geometry, geometry.SurveyGeometry):
            return self._compute_covariance_survey()

    def _compute_covariance_box(self):
        raise NotImplementedError

    def _compute_covariance_survey(self):
        raise NotImplementedError

    def _build_covariance_survey(self, func):

        # If kbins are set for the covariance matrix but not for the geometry,
        # set them for the geometry as well
        if self.k_binning.is_kbins_set and not self.geometry.is_kbins_set:
            self.geometry.set_kbins(self.k_binning)

        # has shape [tracer, tracer, ell, ell, k, k]
        cov = np.zeros((self.num_spectra, self.num_spectra, 3, 3, self.k_binning.kbins, self.k_binning.kbins))
        n_AB = 0
        for idx_A, idx_B in itt.product(range(self.num_tracers), repeat=2):
            if idx_B < idx_A: continue
            n_CD = 0
            for idx_C, idx_D in itt.product(range(self.num_tracers), repeat=2):
                if idx_D < idx_C: continue
                cov[n_AB, n_CD, :, :, :, :] += func(idx_A, idx_B, idx_C, idx_D)
                n_CD += 1
            n_AB += 1

        # enforce symmetry
        # axes 2,3 = ell1,ell2 ; axes 4,5 = k1,k2
        cov = (cov + cov.transpose(0, 1, 3, 2, 5, 4)) / 2

        cov *= (self.pk_renorm)
        return cov

    def _set_survey_covariance(self, cov_array):

        ell1 = self.ells[0]
        num_spectra = self.num_tracers * (self.num_tracers + 1) // 2
        for l1, l2 in itt.product(ell1, repeat=2):
            for t1, t2 in itt.product(range(num_spectra), repeat=2):
                if t2 < t1: continue
                #ell_idx = (l1 * len(self.ells[1]) + l2) // 2
                self.set_ell_tracer_cov(l1, l2, t1, t2, cov_array[t1, t2, l1//2, l2//2, :, :])

    @property
    def shotnoise(self):
        '''Shotnoise of the sample in the same normalization as the power spectrum.

        Returns
        -------
        float
            Shotnoise value.'''
        
        shotnoise = []
        if isinstance(self.geometry, geometry.SurveyGeometry):
            alphas = np.array(self.alpha)
            for t1 in range(self.num_tracers):
                shotnoise.append(self.pk_renorm * (1 + alphas[t1]) * self.geometry.I(t1, t1, 1, 2, 0, 0)/self.geometry.I(t1, t1, 1, 1, 1, 1))
            return np.array(shotnoise)
        elif isinstance(self.geometry, geometry.BoxGeometry):
            return self.pk_renorm * self.geometry.shotnoise

    def set_shotnoise(self, shotnoise):
        '''Determines the relative normalization of the power spectrum by comparing
           the estimated FKP shotnoise with the given shotnoise value.

        Parameters
        ----------
        shotnoise : float
            shotnoise with same normalization as the power spectrum.
        '''

        if self.rank == 0:
            self.logger.info(f'Estimated shotnoise was {self.shotnoise}')
            self.logger.info(f'Forcing it to be {shotnoise}.')

            self.logger.info(f'Setting pk_renorm to {self.pk_renorm} based on given shotnoise value.')

        self.pk_renorm *= shotnoise / self.shotnoise

class GaussianCovariance(PowerSpectrumCovariance):
    '''Gaussian covariance matrix of power spectrum multipoles in a given geometry.

    Attributes
    ----------
    cov : np.ndarray
        Full 2D Gaussian covariance matrix, ordered first by tracer, then by multipole combination.
    geometry : geometry.Geometry
        Geometry of the survey. Can be a BoxGeometry or a SurveyGeometry object.
    '''

    def __init__(self, geometry=None):
        super().__init__(geometry=geometry)

        self.logger = logging.getLogger('GaussianCovariance')

    def set_galaxy_pk_multipole(self, pk:np.ndarray, ell:int, tracer1:int=0, tracer2:int=0, has_shotnoise:bool=False):
        '''Set the input power spectrum to be used for the covariance calculation.

        Parameters
        ----------
        pk : np.ndarray
            Power spectrum input as a 1D numpy array.
        ell : int
            Multipole of the power spectrum.
        tracer1 : int
            Index of the first tracer the power spectrum corresponds to.
            If equal to tracer2, then p(k) is an auto-spectrum. If different, then p(k) is a cross-spectrum.
        tracer2 : int
            Index of the second tracer the power spectrum corresponds to.
            If equal to tracer1, then p(k) is an auto-spectrum. If different, then p(k) is a cross-spectrum.
        has_shotnoise : bool, optional
            Whether the power spectrum has shotnoise included or not.

        Raises
        ------
        ValueError
            If the length of the power spectrum does not match the number of k-bins, or if the tracer indices are invalid.
        '''

        if len(pk) != self.k_binning.kbins:
            raise ValueError(f"Error in PowerSpectrumMultipolesCovariance.set_galaxy_pk_multipole: Power spectrum must have the same number of k-bins ({len(pk)}) as the covariance matrix ({self.k_binning.kbins}).")

        if tracer1 >= self.num_tracers or tracer2 >= self.num_tracers:
            raise ValueError(f"Error in PowerSpectrumMultipolesCovariance.set_galaxy_pk_multipole: Requested tracer combo ({tracer1}, {tracer2}) must both be < total number of tracers ({self.num_tracers})")

        # NOTE: Find a better way to set ells
        if not self.has_ells(ell, ell):
            self.ells = [list(range(0, ell + 1, 2)), list(range(0, ell + 1, 2))]

        if ell == 0 and has_shotnoise and tracer1 == tracer2:
            if self.rank == 0: self.logger.info(f'Removing shotnoise = {self.shotnoise} from ell = 0.')
            pk = pk - self.shotnoise[tracer1]

        self._pk[ell, tracer1, tracer2] = pk
        if tracer1 != tracer2: # <- assuming cross spectra are symmetric (P(t1, t2) = P(t2, t1))
            self._pk[ell, tracer2, tracer1] = pk

    def get_pk(self, ell, tracer1=0, tracer2=0, force_return=False, remove_shotnoise=True, renorm=True):
        '''Get the input power spectrum to be used for the covariance calculation.

        Parameters
        ----------
        ell : int
            Multipole of the power spectrum to retrieve.
        tracer1 : int
            Index of the first tracer the power spectrum corresponds to.
            If equal to tracer2, then p(k) is an auto-spectrum. If different, then p(k) is a cross-spectrum.
        tracer2 : int
            Index of the second tracer the power spectrum corresponds to.
            If equal to tracer1, then p(k) is an auto-spectrum. If different, then p(k) is a cross-spectrum.
        force_return : bool or float, optional
            If the power spectrum for the given ell is not set, return a zero array if True or the specified value if a float.
        remove_shotnoise : bool, optional
            Whether to remove the shotnoise from the power spectrum monopole. Default True.
        renorm : bool, optional
            Whether to renormalize the power spectrum by pk_renorm. Default True.
        '''

        pk_renorm = self.pk_renorm if renorm else 1.0

        if (ell, tracer1, tracer2) in self._pk.keys():
            pk = self._pk[ell, tracer1, tracer2]
            if (not remove_shotnoise) and ell == 0:
                if self.rank == 0: self.logger.info(f'Adding shotnoise = {self.shotnoise} to ell = 0.')
                return pk / pk_renorm + self.shotnoise
            return pk / pk_renorm
        elif type(force_return) != bool:
            return force_return*np.ones(self.kbins)
        elif force_return:
            return np.zeros(self.kbins)
        else:
            raise ValueError(f"Power spectrum for ell = {ell} and tracer combo ({tracer1}, {tracer2}) not set.")

    def _compute_covariance_box(self):
        '''Compute the covariance matrix for a box geometry.
        NOTE: This functionality uses the older single-tracer methodology. Use at your own risk!
        
        Returns
        -------
        GaussianCovariance
            Covariance matrix object.
        '''

        # TODO: Upgrade to multi-tracer (if we really want...)
        # If the power spectrum for a given ell is not set, use a zero array instead
        P0 = self.get_pk(0, force_return=True, remove_shotnoise=False)
        P2 = self.get_pk(2, force_return=True)
        P4 = self.get_pk(4, force_return=True)

        cov = {}

        cov[0, 0] = P0**2 + 1/5*P2**2 + 1/9*P4**2
        cov[0, 2] = 2*P0*P2 + 2/7*P2 ** 2 + 4/7*P2*P4 + 100/693*P4**2
        cov[0, 4] = 2*P0*P4 + 18/35*P2**2 + 40/77*P2*P4 + 162/1001*P4**2
        cov[2, 2] = 5*P0**2 + 20/7*P0*P2 + 20/7*P0*P4 + \
            15/7*P2**2 + 120/77*P2*P4 + 8945/9009*P4**2
        cov[2, 4] = 36/7*P0*P2 + 200/77*P0*P4 + 108 / \
            77*P2**2 + 3578/1001*P2*P4 + 900/1001*P4**2
        cov[4, 4] = 9*P0**2 + 360/77*P0*P2 + 2916/1001*P0*P4 + \
            16101/5005*P2**2 + 3240/1001*P2*P4 + 42849/17017*P4**2

        for l1, l2 in itt.combinations_with_replacement(self.ells, r=2):
            self.set_ell_tracer_cov(l1, l2, 0, 0, 2/self.nmodes * np.diag(cov[l1, l2]))

        if (self.eigvals < 0).any():
            self.logger.warning('Covariance matrix is not positive definite.')

        return self

    def _compute_covariance_survey(self):
        '''Compute the covariance matrix for a survey geometry.

        Returns
        -------
        GaussianCovariance
            Covariance matrix object.
        '''

        # terms without the power spectrum have to be multiplied by its relative normalization pk_renorm 
        def func(A, B, C, D): 
            
            return self._get_cosmic_variance_term(A, B, C, D) + \
                   self._get_mixed_term(A, B, C, D) + \
                   self._get_shotnoise_term(A, B, C, D)

        self._set_survey_covariance(self._build_covariance_survey(func))

        if self.rank == 0:
            eigvals = self.eigvals
            if (eigvals < 0).any():
                self.logger.warning(
                    f'Covariance matrix is not positive definite. Worst of {sum(eigvals < 0)} negative eigenvalues is {eigvals.min():.2e}.')
                # extra_modes = int(0.2*self.geometry.kmodes_sampled)
                # self.geometry.kmodes_sampled += extra_modes
                # self.logger.warning(f'Sampling {extra_modes} more kmodes. Total = {self.geometry.kmodes_sampled}.')
                # self.geometry.compute_window_kernels()
                # self._compute_covariance_survey()
            self.logger.info(
                f'Condition number is {eigvals.max()/eigvals[eigvals > 0].min():.2e}.')
            self.logger.info(
                f'Lowest positive eigval is {eigvals[eigvals > 0].min():.2e}.')

            if not np.allclose(self.cov, self.cov.T):
                self.logger.warning('Covariance matrix is not symmetric.')

        return self

    def _diagnose_covariance(self):

        eigvals = self.eigvals
        if (eigvals < 0).any():
            self.logger.warning(
                f'Covariance matrix is not positive definite. Worst of {sum(eigvals < 0)} negative eigenvalues is {eigvals.min():.2e}.')

        self.logger.info(
            f'Condition number is {eigvals.max()/eigvals[eigvals > 0].min():.2e}.')
        self.logger.info(
            f'Lowest positive eigval is {eigvals[eigvals > 0].min():.2e}.')

        if not np.allclose(self.cov, self.cov.T):
            self.logger.warning('Covariance matrix is not symmetric.')

    # TODO for Otavio: Upgrade to multi-tracer support
    def load_pypower_file(self, filename, **kwargs):
        '''Load power spectrum from pypower file and set it to be used for the covariance calculation.

        Parameters
        ----------
        filename : str
            Name of the pypower file containing the power spectrum.
        tracer1 : int
            Index of the first tracer the pypower file corresponds to.
            If equal to tracer2, then p(k) is an auto-spectrum. If different, then p(k) is a cross-spectrum.
        tracer2 : int
            Index of the second tracer the pypower file corresponds to.
            If equal to tracer1, then p(k) is an auto-spectrum. If different, then p(k) is a cross-spectrum.
        remove_shotnoise : bool, optional
            Whether pypower should be used to remove the shotnoise from the power spectrum monopole.
            If None, will be determined based on the geometry used.
        set_shotnoise : bool, optional
            Whether to rescale shotnoise matching the value in the power spectrum file.
        '''
        from pypower import PowerSpectrumMultipoles
        self.logger.info(f'Loading power spectrum from {filename}.')
        pypower = PowerSpectrumMultipoles.load(filename)
        return self.load_pypower(pypower, **kwargs)

    # TODO for Otavio: Upgrade to multi-tracer support
    def load_pypower(self, pypower, tracer1=0, tracer2=0, remove_shotnoise=None, set_shotnoise=False, naverage=1):
        '''Load power spectrum from pypower object and set it to be used for the covariance calculation.

        Parameters
        ----------
        pypower : PowerSpectrumMultipoles
            pypower object containing the power spectrum.
        tracer1 : int
            Index of the first tracer the pypower file corresponds to.
            If equal to tracer2, then p(k) is an auto-spectrum. If different, then p(k) is a cross-spectrum.
        tracer2 : int
            Index of the second tracer the pypower file corresponds to.
            If equal to tracer1, then p(k) is an auto-spectrum. If different, then p(k) is a cross-spectrum.
        remove_shotnoise : bool, optional
            Whether pypower should be used to remove the shotnoise from the power spectrum monopole.
            If None, will be determined based on the geometry used.
        set_shotnoise : bool, optional
            Whether to rescale shotnoise matching the value in the power spectrum file.
        '''

        kmin_file, kmax_file = pypower.kedges[[0, -1]]
        dk_file = np.diff(pypower.kedges).mean()

        if not self.is_kbins_set:
            self.set_kbins(kmin_file, kmax_file, dk_file)

        if self.kmin < kmin_file or self.kmax > kmax_file:
            raise ValueError(
                'kmin and kmax of the covariance matrix must be within the range of the power spectrum file.')

        imin = np.round((self.kmin - kmin_file)/dk_file).astype(int)
        imax = np.round((self.kmax - kmin_file)/dk_file).astype(int)
        di = self.dk/dk_file

        self.logger.info(
            f'Cutting power spectrum from {kmin_file} < k < {kmax_file} to {self.kmin} < k < {self.kmax}.')

        if np.allclose(np.round(di), di):
            di = np.round(di).astype(int)
        else:
            raise ValueError(
                f'dk = {self.dk} must be a multiple of dk_file = {dk_file}.')
        if di != 1:
            self.logger.info(
                f'Rebinning power spectrum by a factor of {di}. From dk = {dk_file} to dk = {self.dk}.')

        self.logger.info(f'Grouping {di} bins.')

        if remove_shotnoise is None:
            if self.geometry is None:
                remove_shotnoise = True
            elif isinstance(self.geometry, geometry.BoxGeometry):
                remove_shotnoise = False
            elif isinstance(self.geometry, geometry.SurveyGeometry):
                remove_shotnoise = True
            else:
                remove_shotnoise = True

        if remove_shotnoise and self.rank == 0:
            self.logger.info(
                'pypower is removing shotnoise from the power spectrum.')
        elif self.rank == 0:
            self.logger.info(
                'pypower is NOT removing shotnoise from the power spectrum.')

        P0, P2, P4 = pypower[imin:imax:di].get_power(
            remove_shotnoise=remove_shotnoise, complex=False)

        self.set_galaxy_pk_multipole(P0, 0, tracer1, tracer2, has_shotnoise=not remove_shotnoise)
        self.set_galaxy_pk_multipole(P2, 2, tracer1, tracer2)
        self.set_galaxy_pk_multipole(P4, 4, tracer1, tracer2)

        self.alpha = pypower.attrs['sum_data_weights1'] / \
            pypower.attrs['sum_randoms_weights1']
        self.logger.info(
            f'alpha = sum_data_weights/sum_randoms_weights estimated from pypower is {self.alpha:.2f}')
        self.logger.info(
            f'Renormalizing by a factor of {self.pk_renorm:.2f} to match pypower power spectrum normalization.')

        if self.geometry is not None:
            if set_shotnoise:
                self.set_shotnoise(shotnoise=pypower.shotnoise)
            else:
                self.pk_renorm = self.geometry.I(tracer1, tracer2, 1,1,1,1) / pypower.wnorm * naverage
                self.logger.info(
                    f'Renormalizing by a factor of {self.pk_renorm:.2f} to match pypower power spectrum normalization.')

    def load_npy_file(self, ps_file):
        """Loads power spectra that are stored as npy files, which is mainly for spherex use

        Parameters
        ----------
        ps_file : str
            Path to the npy file containing the power spectra.

        Raises
        ------
        IOError
            If the file does not exist.
        """

        self.logger.info(f"Loading power spectrum from {ps_file}")

        if os.path.exists(ps_file):
            pk_data = np.load(ps_file)
        else:
            raise IOError(f"Could not find power spectrum file at {ps_file}")
        
        #[nz, nt, nk, nl]
        pk_galaxy_raw = pk_data

        if pk_galaxy_raw.ndim != 4:
            raise ValueError(f"Error in load_npy_file: input power spectrum must have 4 dimensions, found {pk_galaxy_raw.ndim} instead.")
        if pk_galaxy_raw.shape[1] == self.num_spectra:
            pk_galaxy_raw = pk_galaxy_raw.transpose(1, 0, 2, 3)
        if pk_galaxy_raw.shape[2] == self.k_binning.kbins:
            pk_galaxy_raw = pk_galaxy_raw.transpose(0, 1, 3, 2)

        self.logger.info(f"input power spectrum has shape {pk_galaxy_raw.shape}")

        idx = 0
        for (i, j) in itt.product(range(self.num_tracers), range(self.num_tracers)):
            if i > j: continue
            self.set_galaxy_pk_multipole(pk_galaxy_raw[idx, 0, 0, :], 0, i, j, has_shotnoise=True)
            self.set_galaxy_pk_multipole(pk_galaxy_raw[idx, 0, 1, :], 2, i, j, has_shotnoise=True)
            self.set_galaxy_pk_multipole(pk_galaxy_raw[idx, 0, 2, :], 4, i, j, has_shotnoise=True)
            idx += 1

    def _get_cosmic_variance_term(self, A:int, B:int, C:int, D:int):
        """Calculates elements of the cosmic variance Gaussian term

        Parameters
        ----------
        A : int
            First tracer index.
        B : int
            Second tracer index.
        C : int
            Third tracer index.
        D : int
            Fourth tracer index.

        Returns
        -------
        np.ndarray
            Cosmic variance contribution with shape [n_ells, n_ells, nk, nk].
        """
        WinKernel_1 = self.geometry.cosmic_variance_kernel(A,B,C,D)[0]
        WinKernel_2 = self.geometry.cosmic_variance_kernel(A,B,C,D)[1]

        P_AD = np.array([self.get_pk(ell, A, D, force_return=True, remove_shotnoise=True) for ell in [0,2,4]])
        P_BC = np.array([self.get_pk(ell, B, C, force_return=True, remove_shotnoise=True) for ell in [0,2,4]])
        P_BD = np.array([self.get_pk(ell, B, D, force_return=True, remove_shotnoise=True) for ell in [0,2,4]])
        P_AC = np.array([self.get_pk(ell, A, C, force_return=True, remove_shotnoise=True) for ell in [0,2,4]])

        cov = np.einsum('ijklxy,kx,ly->ijxy', WinKernel_1, P_AD, P_BC) + \
              np.einsum('ijklxy,kx,ly->ijxy', WinKernel_2, P_BD, P_AC)
        
        return cov

    def _get_mixed_term(self, A:int, B:int, C:int, D:int):
        """Calculates elements of the mixed Gaussian term

        Parameters
        ----------
        A : int
            First tracer index.
        B : int
            Second tracer index.
        C : int
            Third tracer index.
        D : int
            Fourth tracer index.

        Returns
        -------
        np.ndarray
            Mixed term contribution with shape [n_ells, n_ells, nk, nk].
        """
        W_mixed = self.geometry.mixed_kernel(A,B,C,D)

        P_AC = np.array([self.get_pk(ell, A, C, force_return=True, remove_shotnoise=True) for ell in [0, 2, 4]])
        P_AD = np.array([self.get_pk(ell, A, D, force_return=True, remove_shotnoise=True) for ell in [0, 2, 4]])
        P_BC = np.array([self.get_pk(ell, B, C, force_return=True, remove_shotnoise=True) for ell in [0, 2, 4]])
        P_BD = np.array([self.get_pk(ell, B, D, force_return=True, remove_shotnoise=True) for ell in [0, 2, 4]])

        cov = np.zeros((3, 3, self.k_binning.kbins, self.k_binning.kbins))
        if A == D:
            cov += (1 + self.alpha[A]) / 2 * \
            (np.einsum('ijkxy,kx->ijxy', W_mixed[0], P_BC) + \
             np.einsum('ijkxy,ky->ijxy', W_mixed[0], P_BC))
        if B == C:
            cov += (1 + self.alpha[B]) / 2 * \
                (np.einsum('ijkxy,kx->ijxy', W_mixed[1], P_AD) + \
                 np.einsum('ijkxy,ky->ijxy', W_mixed[1], P_AD))
        if A == C:
            cov += (1 + self.alpha[A]) / 2 * \
                (np.einsum('ijkxy,kx->ijxy', W_mixed[2], P_BD) + \
                 np.einsum('ijkxy,ky->ijxy', W_mixed[2], P_BD))
        if B == D:
            cov += (1 + self.alpha[B]) / 2 * \
                (np.einsum('ijkxy,kx->ijxy', W_mixed[3], P_AC) + \
                 np.einsum('ijkxy,ky->ijxy', W_mixed[3], P_AC))

        return cov

    def _get_shotnoise_term(self, A:int, B:int, C:int, D:int):
        """Calculates elements of the shotnoise Gaussian term

        Parameters
        ----------
        A : int
            First tracer index.
        B : int
            Second tracer index.
        C : int
            Third tracer index.
        D : int
            Fourth tracer index.

        Returns
        -------
        np.ndarray
            Shotnoise term contribution with shape [n_ells, n_ells, nk, nk].
        """
        WinKernel = self.geometry.shotnoise_kernel(A,B)
        if A == D and B == C:
            #ells = np.array([0, 2, 4])
            #ell_factor = np.outer(2*ells + 1, 2*ells + 1).reshape(3, 3, 1, 1) * np.ones((3, 3, self.k_binning.kbins, self.k_binning.kbins))
            return (1 + self.alpha[A]) * (1 + self.alpha[B]) * WinKernel
        else:
            return np.zeros((3, 3, self.k_binning.kbins, self.k_binning.kbins))


# TODO: Update this to multi-tracer
class RegularTrispectrumCovariance(PowerSpectrumCovariance):
    '''Regular trispectrum covariance matrix of power spectrum multipoles in a given geometry.

    Attributes
    ----------
    geometry : geometry.Geometry
        Geometry of the survey. Can be a BoxGeometry or a SurveyGeometry object.
    '''

    def __init__(self, geometry=None):

        super().__init__(self, geometry=geometry)
        self.logger = logging.getLogger('RegularTrispectrumCovariance')

        from powercovfft import PowerSpecCovFFT
        self.calculator = PowerSpecCovFFT()

    def set_kbins(self, kmin, kmax, dk, ells=(0, 2, 4)):
        '''Set the k-binning for the covariance matrix.

        Parameters
        ----------
        kmin : float
            Minimum k value.
        kmax : float
            Maximum k value.
        dk : float
            Width of the k bins.
        '''

        self.ells = ells
        base.PowerSpectrumMultipolesCovariance.set_kbins(self, kmin, kmax, dk)

        # Set the FFTLog
        config_fft = {'nu': -0.3, 'kmin': 1e-5, 'kmax': 1e+1, 'nmax': 512}
        self.calculator.set_power_law_decomp(config_fft)

        k1, k2 = self.kmid_matrices

        # Precompute the master integrals
        self.calculator.calc_master_integral(k1, k2)

        return self

    def set_linear_matter_pk(self, pk_linear, k=None):
        '''Set the input linear power spectrum to be used for the covariance calculation.

        Parameters
        ----------
        k : array_like
            Wavenumbers.
        pk : array_like
            Power spectrum.
        '''

        if callable(pk_linear):
            self.pk_linear = pk_linear
            self.calculator.pk_lin_spl = lambda logk: np.log(pk_linear(np.exp(logk)))
            self.calculator.decomp.compute(self.calculator.get_pk_lin)
        else:
            if k is None:
                self.logger.error('k must be set if pk is not callable.')
            if len(k) != len(pk_linear):
                self.logger.error('k and pk must have the same length.')
            if 0 in k:
                self.logger.error('k must not contain zero.')
            
            from scipy.interpolate import InterpolatedUnivariateSpline
            self.calculator.pk_lin_spl = InterpolatedUnivariateSpline(
                np.log(k), np.log(pk_linear), ext='extrapolate')

            self.calculator.decomp.compute(self.calculator.get_pk_lin)

            self.pk_linear = self.calculator.get_pk_lin

    def set_params(self, fgrowth, b1, b2=None, g2=None, b3=None, g3=None, g2x=None, g21=None):
        '''Set the bias parameters to be used for the covariance calculation. If the optional
           parameters are not set, will use expressions for non-local bias (g_i) from local
           lagrangian approximation and non-linear bias (b_i) from peak-background split fit
           of arXiv:1511.01096 rescaled using Appendix C.2 of arXiv:1812.03208, (useful if
           those parameters aren't constrained).

        Parameters
        ----------
        fgrowth : float
            Growth rate of the linear power spectrum.
        b1 : float
            Linear bias.
        b2 : float, optional
            Quadratic bias.
        g2 : float, optional
            gamma_2 non-local bias.
        b3 : float, optional
            Cubic bias.
        g3 : float, optional
            gamma_3 non-local bias.
        g2x : float, optional
            gamma_2^x non-local bias.
        g21 : float, optional
            gamma_{21} non-local  bias.
        '''

        if g2 is None:
            g2 = -2/7*(b1 - 1)
        if g3 is None:
            g3 = 11/63*(b1 - 1)
        if b2 is None:
            b2 = 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 + 4/3*g2
        if g2x is None:
            g2x = -2/7*b2
        if g21 is None:
            g21 = -22/147*(b1 - 1)
        if b3 is None:
            b3 = -1.028 + 7.646*b1 - 6.227*b1**2 + 0.912 * \
                b1**3 + 4*g2x - 4/3*g3 - 8/3*g21 - 32/21*g2

        #    equation 78 of arXiv:1812.03208 (Wadekar   basis)
        # vs equation A1 of arXiv:2308.08593 (Kobayashi basis)
        self.calculator.bias = {
            'b1': b1,
            'b2': b2,
            'bG2': g2,
            'b3': b3,
            'bG3': g3,
            'bdG2': g2x,
            'bGamma3': -4/7*g21,  # equation 44 of arXiv:1812.03208
        }

        self.calculator.fgrowth = fgrowth

        return self

    def _compute_covariance_box(self):
        '''Compute the covariance matrix for a box geometry.

        Returns
        -------
        self : TrispectrumCovariance
            Covariance matrix.
        '''

        self.calculator.vol = self.geometry.volume
        self.calculator.ndens = self.geometry.nbar

        self._build_covariance()

        return self

    def _compute_covariance_survey(self):
        '''Compute the covariance matrix for a survey geometry.

        Returns
        -------
        self : TrispectrumCovariance
            Covariance matrix.
        '''

        self.calculator.vol = self.geometry.I(2,2)**2 / self.geometry.I(4,4)
        self.calculator.ndens = self.geometry.I(4,4) / self.geometry.I(3,4)
        self.calculator.ndens2 = self.geometry.I(4,4) / self.geometry.I(2,4)

        self._build_covariance()

        return self

    def _build_covariance(self):

        # Compute the elementary integrals Eq. (13)
        self.calculator.calc_base_integral()

        k1, k2 = self.kmid_matrices

        # Compute the non-Gaussian covariance for each combination of multipoles
        self.set_ell_cov(0, 0, self.pk_renorm**2 *
                         self.calculator.get_cov_T0(0, 0, k1, k2))
        self.set_ell_cov(2, 2, self.pk_renorm**2 *
                         self.calculator.get_cov_T0(2, 2, k1, k2))
        self.set_ell_cov(4, 4, self.pk_renorm**2 *
                         self.calculator.get_cov_T0(4, 4, k1, k2))
        self.set_ell_cov(0, 2, self.pk_renorm**2 *
                         self.calculator.get_cov_T0(0, 2, k1, k2))
        self.set_ell_cov(0, 4, self.pk_renorm**2 *
                         self.calculator.get_cov_T0(0, 4, k1, k2))
        self.set_ell_cov(2, 4, self.pk_renorm**2 *
                         self.calculator.get_cov_T0(2, 4, k1, k2))

        return self


class SuperSampleCovariance(PowerSpectrumCovariance):
    '''Regular super sample covariance matrix of power spectrum multipoles in a given geometry.

    Attributes
    ----------
    geometry : geometry.Geometry
        Geometry of the survey. Can be a BoxGeometry or a SurveyGeometry object.
    '''

    def __init__(self, geometry=None):

        base.PowerSpectrumMultipolesCovariance.__init__(
            self, geometry=geometry)
        self.logger = logging.getLogger('SuperSampleCovariance')

    def set_kbins(self, kmin, kmax, dk, ells=(0, 2, 4)):
        '''Set the k-binning for the covariance matrix.

        Parameters
        ----------
        kmin : float
            Minimum k value.
        kmax : float
            Maximum k value.
        dk : float
            Width of the k bins.
        '''

        self.ells = ells
        base.PowerSpectrumMultipolesCovariance.set_kbins(self, kmin, kmax, dk)

        return self

    def set_linear_matter_pk(self, pk_linear, k=None, dPk=None):
        '''Set the input linear power spectrum to be used for the covariance calculation.

        Parameters
        ----------
        k : array_like
            Wavenumbers.
        pk : array_like
            Power spectrum.
        '''

        kmid = self.kmid

        if callable(dPk):
            dPk = dPk(kmid)

        if callable(pk_linear):
            self.pk_linear = pk_linear
            if dPk is None:
                from scipy.misc import derivative
                dPk = derivative(self.pk_linear, kmid, dx=1e-4)
        else:
            if len(k) != len(pk_linear):
                self.logger.error('k and pk must have the same length.')

            from scipy.interpolate import InterpolatedUnivariateSpline
            self.pk_linear = InterpolatedUnivariateSpline(k, pk_linear)
            if dPk is None:
                dPk = self.pk_linear.derivative()(kmid)

        self._dlnPk = dPk * kmid/self.pk_linear(kmid)

    def set_params(self, fgrowth, b1, b2=None, g2=None):
        '''Set the bias parameters to be used for the covariance calculation. If the optional
           parameters are not set, will use expressions for non-local bias (g_i) from local
           lagrangian approximation and non-linear bias (b_i) from peak-background split fit
           of arXiv:1511.01096 rescaled using Appendix C.2 of arXiv:1812.03208, (useful if
           those parameters aren't constrained).

        Parameters
        ----------
        fgrowth : float
            Growth rate of the linear power spectrum.

        b1 : float
            Linear bias.

        b2 : float, optional
            Quadratic bias.

        g2 : float, optional
            gamma_2 non-local bias.
        '''

        if g2 is None:
            g2 = -2/7*(b1 - 1)
        if b2 is None:
            b2 = 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 + 4/3*g2

        self.params = {
            'b1': b1,
            'b2': b2,
            'g2': g2,
            'f': fgrowth,
        }

        return self

    def Z12Multipoles(self, ell_kernel, ell_legendre):
        b1, be, b2, g2 = self.params['b1'], \
            self.params['f']/self.params['b1'], \
            self.params['b2'], \
            self.params['g2']

        dlnpk = self._dlnPk

        # Expressions from CovaPT (arXiv:1910.02914)

        if ell_kernel == 0:
            def Z12(mu): return (7*b1**2*be*(70 + 42*be + (-35*(-3 + dlnpk) + 3*be*(28 + 13*be - 14*dlnpk - 5*be*dlnpk))*mu**2) +
                                 b1*(35*(47 - 7*dlnpk) + be*(798 + 153*be - 98*dlnpk - 21*be*dlnpk +
                                                             4*(84 + be*(48 - 21*dlnpk) - 49*dlnpk)*mu**2)) +
                                 98*(5*b2*(3 + be) + 4*g2*(-5 + be*(-2 + mu**2))))/(735.*b1**2)
        elif ell_kernel == 2:
            def Z12(mu): return (14*b1**2*be*(14 + 12*be + (2*be*(12 + 7*be) - (1 + be)*(7 + 5*be)*dlnpk)*mu**2) +
                                 b1*(4*be*(69 + 19*be) - 7*be*(2 + be)*dlnpk +
                                     (24*be*(11 + 6*be) - 7*(21 + be*(22 + 9*be))*dlnpk)*mu**2 + 7*(-8 + 7*dlnpk + 24*mu**2)) +
                                 28*(7*b2*be + g2*(-7 - 13*be + (21 + 11*be)*mu**2)))/(147.*b1**2)
        elif ell_kernel == 4:
            def Z12(mu): return (8*be*(b1*(-132 + 77*dlnpk + be*(23 + 154*b1 + 14*dlnpk)) - 154*g2 +
                                       (b1*(396 - 231*dlnpk + be*(272 + 308*b1 + 343*b1*be - 7*(17 + b1*(22 + 15*be))*dlnpk)) +
                                        462*g2)*mu**2))/(2695.*b1**2)

        Z12 = np.vectorize(Z12)
        legendre = math.legendre(ell_legendre)

        from scipy.integrate import quad
        return np.array([quad(lambda mu: legendre(mu)*Z12(mu)[i], -1, 1)[0] for i in range(len(dlnpk))])

    def _compute_covariance_survey(self):
        '''Compute the covariance matrix for a survey geometry.

        Returns
        -------
        self : SuperSampleCovariance
            Covariance matrix.
        '''

        self.logger.debug(f'Calculating the variance of super-survey modes.')

        b1 = self.params['b1']
        b2 = self.params['b2']
        be = self.params['f']/b1

        # Convolving the window function power with linear power spectrum to
        # obtain the variance of super-survey modes. Done for all multipole combinations.
        from scipy.integrate import quad
        sigmas = b1**2 * np.array([4 * np.pi / (2 * np.pi)**3 * quad(lambda k: k**2*self.pk_linear(k)*Pwin(k), 0, self.geometry.kmax)[0]
                                   for Pwin in self.geometry.get_window_power_interpolators()])

        # indices of respective values from sigmas
        sigma22x22 = [[0,  1,  2],
                      [1,  6,  7],
                      [2,  7, 11]]

        sigma10x10 = 15

        sigma22x10 = [[3,  4,  5],
                      [8,  9, 10],
                      [12, 13, 14]]

        # evaluating indices to actual values
        sigma22x22 = sigmas[sigma22x22]
        sigma10x10 = sigmas[sigma10x10]
        sigma22x10 = sigmas[sigma22x10]

        self.sigma22x22 = sigma22x22
        self.sigma10x10 = sigma10x10
        self.sigma22x10 = sigma22x10

        Plin = self.pk_linear(self.kmid)

        # shape here is (ell)
        Z1 = np.array([quad(lambda mu: math.legendre(l)(mu) * (1 + be*mu**2), -1, 1)[0]
                       for l in self.ells])

        # shape here is (ell_kernel, ell_legendre, k)
        Z12 = np.array([self.Z12Multipoles(ell_kernel=l1, ell_legendre=l2)
                        for l1 in self.ells for l2 in self.ells]).reshape(3, 3, -1)

        # same shape as Z12
        # in einsum, lmij are used for ells, kq are used for k
        response_function = np.einsum('k,lmk->lmk', Plin, Z12)

        # shapes are (ell_kernel, ell_legendre, k), (ell, ell), (ell_kernel, ell_legendre, k)
        # final shape is (l1, l2, k1, k2)
        covBC = 1/4. * np.einsum('lik,ij,mjq->lmkq',
                                 response_function, sigma22x22, response_function)

        self.covBC = self.pk_renorm**2 * covBC

        # Kaiser approximation used for large-scale modes in redshift space
        # shape is (ell, k)
        P_kaiser = np.array([
            (1 + 2/3*be + 1/5 * be**2) * Plin,
            (4/3*be + 4/7 * be**2) * Plin,
            (8/35*be**2) * Plin,
        ])

        # shape is (ell_kernel, ell, k),  (ell, ell), (ell) -> (ell_kernel, k)
        LA_term = 1/4. * np.einsum('lik,ij,j->lk', Z12, sigma22x10, Z1)
        # shape is (ell, k)
        LA_term += b2 * P_kaiser/b1**2 * \
            self.geometry.I(3,2)/self.geometry.I(2,2)/self.geometry.I(1,0)

        # output shape is (lmkq) = (l1,l2,k1,k2) final shape of covariance
        covLA = np.einsum('lk,mq  ->lmkq', P_kaiser, P_kaiser) * sigma10x10
        covLA -= np.einsum('lk,q,mq->lmkq', P_kaiser, Plin, LA_term)
        covLA -= np.einsum('lk,q,mq->mlqk', P_kaiser, Plin,
                           LA_term)  # symmetric of the above

        self.covLA = self.pk_renorm**2 * covLA

        cov = self.covBC + self.covLA

        for l1, l2 in itt.combinations_with_replacement(self.ells, r=2):
            self.set_ell_cov(l1, l2, cov[l1//2, l2//2])

        return self
