"""Module containing classes for creating the window function to be used in calculating the Gaussian covariance term.

Classes
-------
SurveyWindow
    Class re
SurveyGeometry
"""

import logging

import numpy as np
import os, time
import itertools as itt

from tqdm import tqdm as shell_tqdm
import multiprocessing as mp
# from scipy.integrate import lebedev_rule

import mockfactory
from pypower import CatalogMesh

import functools

from . import base, math

__all__ = ['SurveyWindow']

class SurveyWindow(base.BaseClass, base.LinearBinning):

    def __init__(self, randoms1, alpha1, randoms2=None, alpha2=None, nmesh=None, cellsize=None, boxsize=None, boxpad=2., kmax=0.02, dk=None, ellmax=4, shotnoise=False, **kwargs):

        base.LinearBinning.__init__(self)

        self.logger = logging.getLogger('SurveyWindow')
        self.tqdm = shell_tqdm

        self._ellmax = ellmax
        self._is_shotnoise = shotnoise
        self._kmax = kmax
        self._dk = dk

        self.mesh1 = self._parse_randoms(
            randoms=randoms1,
            alpha=alpha1,
            nmesh=nmesh,
            cellsize=cellsize,
            boxsize=boxsize,
            boxpad=boxpad,
            kmax=kmax
        )
        self.boxsize = self.mesh1.boxsize[0]
        self.nmesh = self.mesh1.nmesh[0]

        if randoms2 is not None:

            assert alpha2 is not None, "If randoms2 is provided, alpha2 must also be provided."

            self.mesh2 = self._parse_randoms(
                randoms=randoms1.append(randoms2),
                alpha=alpha2,
                nmesh=nmesh,
                cellsize=cellsize,
                boxsize=boxsize,
                boxpad=boxpad,
                kmax=kmax
            )

            self.boxsize = self.mesh2.boxsize[0]
            self.nmesh = self.mesh2.nmesh[0]

            self.mesh2 = self._parse_randoms(
                randoms=randoms2,
                alpha=alpha2,
                nmesh=nmesh,
                cellsize=cellsize,
                boxsize=boxsize,
                boxpad=boxpad,
                kmax=kmax
            )
            
            self.mesh1._set_box(nmesh=self.nmesh, boxsize=self.boxsize, wrap=False)
            self.mesh2._set_box(nmesh=self.nmesh, boxsize=self.boxsize, wrap=False)
        
        self.logger.warn(f'Using box size {self.boxsize}, box center {self.mesh1.boxcenter} and nmesh {self.nmesh}.')
        self.logger.warn(f'Fundamental wavenumber of window meshes = {self.kfun}.')
        self.logger.warn(f'Nyquist wavenumber of window meshes = {self.knyquist}.')

        if kmax is not None and self.knyquist < kmax:
            self.logger.warning(f'Nyquist wavelength {self.knyquist} smaller than required window kmax = {kmax}.')

        self.logger.warn(f'Average of {self.mesh1.data_size / self.nmesh**3} objects per voxel.')

        # Initialize rebin parameters
        self._rebin_parameters(dk, kmax)

    def _parse_randoms(self, randoms, alpha, nmesh, cellsize, boxsize, boxpad, kmax):
        """Parse the randoms into a mesh, filling in missing information as needed."""
        start_time = time.time()
        if not isinstance(randoms, mockfactory.Catalog):
            randoms = mockfactory.Catalog(randoms)

        # Check if the randoms have weights, otherwise set them to 1
        for name in ['WEIGHT', 'WEIGHT_FKP']:
            if name not in randoms:
                self.logger.warning(f'{name} column not found in randoms. Setting it to 1.')
                randoms[name] = np.ones(self.randoms.size, dtype='f8')
        
        randoms['WEIGHT'] *= alpha
        
        # Check if the randoms have a number density column, otherwise estimate it using RedshiftDensityInterpolator
        if 'NZ' not in randoms:
            self.logger.warning('NZ column not found in randoms. Estimating it with RedshiftDensityInterpolator.')
            import healpy as hp
            nside = 512
            distance = np.sqrt(np.sum(randoms['POSITION']**2, axis=-1))
            xyz = randoms['POSITION'] / distance[:, None]
            hpixel = hp.vec2pix(nside, *xyz.T)
            unique_hpixels = np.unique(hpixel)
            fsky = len(unique_hpixels) / hp.nside2npix(nside)
            self.logger.warn(f'fsky estimated from randoms: {fsky:.3f}')
            nbar = mockfactory.RedshiftDensityInterpolator(z=distance, weights=randoms['WEIGHT'], fsky=fsky)
            randoms['NZ'] = nbar(distance)

        # Check if the randoms have nmesh and cellsize, otherwise set them using the kmax parameter
        if nmesh is None and cellsize is None:
            # Pick value that will give at least k_mask = kmax_window in the FFTs
            self.cellsize = np.pi / kmax / (1. + 1e-9)

        self.logger.warn(f'Parsed randoms in {time.time() - start_time:.2f} seconds.')

        return CatalogMesh(
                data_positions=randoms['POSITION'],
                data_weights=randoms['WEIGHT'],
                position_type='pos',
                nmesh=nmesh,
                cellsize=cellsize,
                boxsize=boxsize,
                boxpad=boxpad,
                dtype='c16',
                **{'interlacing': 3, 'resampler': 'tsc'}
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['logger', 'tqdm', '_randoms', '_mesh', '_resume_file']:
            del state[key]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    @property
    def knyquist(self):
        if hasattr(self, 'knmesh'):
            return np.pi * self.knmesh / self.kboxsize
        
        return np.pi * self.nmesh / self.boxsize
    
    @property
    def kfun(self):
        if hasattr(self, 'knmesh'):
            return 2 * np.pi / self.kboxsize
        return 2 * np.pi / self.boxsize
    
    @property
    def alpha1(self):
        return self._alpha1
    
    @property
    def alpha2(self):
        return self._alpha2
    
    @property
    def alpha(self):
        return self.alpha1

    @property
    def ikgrid(self):
        """Grid of wavenumber indices."""
        ikgrid = []
        nmesh = self.knmesh if hasattr(self, 'knmesh') else self.nmesh
        for _ in range(3):
            iik = np.arange(nmesh)
            iik[iik >= nmesh // 2] -= nmesh
            ikgrid.append(iik)
        return ikgrid
    
    def I(self, nbar_power, fkp_power):
        return (self._randoms[0]['NZ']**(nbar_power-1) * \
                self._randoms[0]['WEIGHT_FKP']**fkp_power * \
                self._randoms[0]['WEIGHT'] * \
                self.alpha1).sum().tolist()
    
    def _rebin_parameters(self, dk, kmax):

        # If dk and kmax are provided, they determine the target boxsize and nmesh
        target_boxsize = 2*np.pi/dk
        target_nmesh = int(np.ceil(target_boxsize * kmax / np.pi))

        # Trim the mesh to achieve the target boxsize
        trim_to_nmesh = int(np.ceil(target_boxsize/self.boxsize * self.nmesh))
        
        # Rebin mesh to obtain the target nmesh
        rebin_factor = trim_to_nmesh//target_nmesh

        # Ensure that trim_to_nmesh is a multiple of rebin_factor
        if (trim_to_nmesh % rebin_factor) != 0:
            trim_to_nmesh +=  rebin_factor - (trim_to_nmesh % rebin_factor)

        self.kboxsize = trim_to_nmesh/self.nmesh * self.boxsize
        self.knmesh = trim_to_nmesh//rebin_factor

        return trim_to_nmesh, rebin_factor

    @functools.cache
    def mesh(self, ell, m, fourier=True, threshold=None):
        """Compute the product of meshes and multiply by real Ylm evaluated at the same coordinates.

        Parameters
        ----------
        ell : int
            Degree of the spherical harmonic.

        m : int
            Order of the spherical harmonic.

        shotnoise : bool, optional
            If True, the shotnoise mesh is used instead of the original mesh. Default is False.

        fourier : bool, optional
            If True, the Fourier transform of the mesh is returned. Default is False.

        Returns
        -------
        mesh
            Resulting mesh after computation with size [nmesh, nmesh, nmesh]
        """

        assert ell >= 0, "ell must be non-negative"
        assert abs(m) <= ell, "m must be less than or equal to ell"

        Ylm = math.get_real_Ylm(ell, m)

        time_start = time.time()
        # Initialize the result mesh
        result = self.mesh1.clone(
                data_positions=self.mesh1.data_positions,
                data_weights=self.mesh1.data_weights*Ylm(self.mesh1.data_positions.T[0],
                                                         self.mesh1.data_positions.T[1],
                                                         self.mesh1.data_positions.T[2]),
                position_type='pos',
            ).to_mesh(compensate=True)

        self.logger.warn(f"Mesh computation with Ylm done in {time.time() - time_start:.2f} seconds")
        time_start = time.time()

        if hasattr(self, 'mesh2'):
            result *= self.mesh2.to_mesh(compensate=True)
        else:
            result *= self.mesh1.to_mesh(compensate=True)

        self.logger.warn(f"Second mesh computation done in {time.time() - time_start:.2f} seconds")

        if self._dk is not None and self._kmax is not None:
            trim_to_nmesh, rebin_factor = self._rebin_parameters(self._dk, self._kmax)
            
            if trim_to_nmesh <= self.nmesh and self.kboxsize <= self.boxsize:
                result = result[:trim_to_nmesh, :trim_to_nmesh, :trim_to_nmesh]
                self.logger.warn(f"Trimmed mesh from {self.nmesh} to {trim_to_nmesh} and boxsize from {self.boxsize:.0f} to {self.kboxsize:.0f}.")

                # Sum mesh values in the new mesh
                if rebin_factor > 1:
                    result = result.reshape((self.knmesh, rebin_factor, self.knmesh, rebin_factor, self.knmesh, rebin_factor)).sum(axis=(1, 3, 5))

                self.logger.warn(f"Rebinned mesh from {trim_to_nmesh} to {result.shape[0]} with factor {rebin_factor}.")


        if hasattr(result, 'value'):
            result = result.value

        # pmesh fft convention is F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r); let us correct it here

        if fourier:
            result *= self.knmesh**3
            time_start = time.time()
            result = np.fft.fftn(result, axes=(0, 1, 2), norm='backward')
            self.logger.warn(f"Mesh Fourier transform done in {time.time() - time_start:.2f} seconds")

        # result = result.value if not fourier else result.r2c().value

        if threshold is not None:
            # Convert the result to a sparse array to save memory
            result[np.abs(result) < threshold] = 0
            result = base.SparseNDArray.from_dense(result, shape_in=(self.nmesh,self.nmesh), shape_out=self.nmesh)
        
        return result
    

class SurveyGeometry(base.BaseClass, base.LinearBinning):

    # survey window needs randoms1, alpha1, randoms2=None, alpha2=None, nmesh=None, cellsize=None, boxsize=None, boxpad=2., kmax=0.02, ellmax=4, **kwargs):
    def __init__(self, random_files, alphas, nmesh, boxsize, box_padding, kmax=0.2, dk=None, delta_k_max=3, ellmax=4, 
                 sample_mode="lebedev", lebedev_degree=25, nthreads=None):

        base.LinearBinning.__init__(self)

        self.logger = logging.getLogger('SurveyGeometry')
        self.tqdm = shell_tqdm

        if   isinstance(random_files, list): self.num_tracers = len(random_files)
        elif isinstance(random_files, str):  self.num_tracers = 1
        self.alphas = alphas
            
        if self.num_tracers > 4 or self.num_tracers < 1:
            raise ValueError(f"Error in SurveyGeometry.__init__: num_tracers must be between [1, 4] but is {self.num_tracers}")
        if self.num_tracers >= 2 and self.num_tracers != len(alphas):
            raise ValueError(f"Error in SurveyGeometry.__init__: number of alpha values provided ({len(alphas)}) not equal to number of tracers ({self.num_tracers})")
        
        self.nmesh = nmesh
        self.boxsize = boxsize
        self.box_padding = box_padding
        self._kmax=kmax
        self._dk = dk
        self.delta_k_max = delta_k_max
        self.ellmax = ellmax
        self.sample_mode = sample_mode
        self.lebedev_degree = lebedev_degree
        self.nthreads = nthreads

        # NOTE: This method + logic could instead go in SurveyWindow
        self.load_randoms(random_files)

        self._init_survey_windows()
        self._init_ell_m_combos()
    

    def load_randoms(self, random_files):
        """Loads random catalogs into mockfactory CatalogMesh objects"""
        
        self.randoms = []
        for file in random_files:
            self.logger.info(f"loading in {file}...")
            self.randoms.append(mockfactory.Catalog.read(file))


    def _init_survey_windows(self):

        self.W_AB, self.W_CD = None, None
        if self.num_tracers == 1:
            self.window_AB = SurveyWindow(self.randoms[0], self.alphas[0], None, None, 
                                          self.nmesh, self._dk, self.boxsize, self.box_padding, self._kmax, self.ellmax, shotnoise=False)
            self.window_CD = self.window_AB
        elif self.num_tracers == 2:
            self.window_AB = SurveyWindow(self.randoms[0], self.alphas[0], self.randoms[1], self.alphas[1], 
                                          self.nmesh, self._dk, self.boxsize, self.box_padding, self._kmax, self.ellmax, shotnoise=False)
            self.window_CD = self.window_AB
        elif self.num_tracers == 3:
            self.window_AB = SurveyWindow(self.randoms[0], self.alphas[0], self.randoms[1], self.alphas[1], 
                                          self.nmesh, self._dk, self.boxsize, self.box_padding, self._kmax, self.ellmax, shotnoise=False)
            self.window_CD = SurveyWindow(self.randoms[2], self.alphas[2], None, None, 
                                          self.nmesh, self._dk, self.boxsize, self.box_padding, self._kmax, self.ellmax, shotnoise=False)
        elif self.num_tracers == 4:
            self.window_AB = SurveyWindow(self.randoms[0], self.alphas[0], self.randoms[1], self.alphas[1], 
                                          self.nmesh, self._dk, self.boxsize, self.box_padding, self._kmax, self.ellmax, shotnoise=False)
            self.window_CD = SurveyWindow(self.randoms[2], self.alphas[2], self.randoms[3], self.alphas[3], 
                                          self.nmesh, self._dk, self.boxsize, self.box_padding, self._kmax, self.ellmax, shotnoise=False)

    def get_survey_window(self):

        self.W_ABCD = base.SparseNDArray(shape_out=(7,7,7,7), shape_in=(self.nmesh,self.nmesh,self.nmesh))

        for la, lb in itt.product(range(0, self.ellmax+1, 2), repeat=2):
            for ma in range(-la, la+1):
                for mb in range(-lb, lb+1):
                    self.W_ABCD[la,lb,ma,mb] = self.window_AB.mesh(la, ma) * self.window_CD.mesh(lb, mb)

        return self.W_ABCD

        # calls survey_window

        if self.W_AB is None:
            self.W_AB = base.SparseNDArray(shape_out=(7,7), shape_in=(self.nmesh,self.nmesh,self.nmesh))
            self.W_CD = base.SparseNDArray(shape_out=(7,7), shape_in=(self.nmesh,self.nmesh,self.nmesh))
            self.ikgrid = self.window_AB.ikgrid

            #self.W_AB = np.zeros(len(self.ells_and_ms, self.nmesh, self.nmesh, self.nmesh))
            #self.W_CD = np.zeros_like(self.W_AB) if self.num_tracers > 1 else None
            for idx in range(len(self.ells_and_ms)):
                ell = self.ells_and_ms[idx][0]
                m = self.ells_and_ms[idx][1]
                self.W_AB[ell, m] = self.window_AB.mesh(ell, m)
                if self.num_tracers > 1: self.W_CD[ell, m] = self.window_CD.mesh(ell, m)
    

    def _init_ell_m_combos(self):
        """creates the unique combinations of l and m and stores them in a nested list"""
        self.ells_and_ms = []
        for ell in range(self.ellmax, step=2):
            for m in range(-ell, ell+1):
                self.ells_and_ms.append([ell, m])

    # TODO: Possibly move this into math.py
    @staticmethod
    def get_gaunt_coefficients(cache_dir=None):
        """Calculates all relavent Gaunt coefficients, or loads them from file"""

        # Load mask coupling Gaunt coefficients if cache exists, otherwise compute them
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, "cosmic_variance_coefficients.npz")

        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            import sympy.physics.wigner

            # shape_out = l1, l2, l3, l4, m1, m2, m3, m4
            # shape_in =  la, lb, ma, mb
            # Only including positive m values, as -m is equivalent to m
            # when Ylm is real and m is even
            gaunt_coefficients = base.SparseNDArray(shape_out=(3,3,3,3,3,3,3,3), shape_in=(7,7,7,7))

            for l1, l2, l3, l4 in itt.product((0,2,4), repeat=4):
                for m1, m2, m3, m4 in itt.product(*[np.arange(-l, l+1, 2) for l in (l1, l2, l3, l4)]):
                    for la in np.arange(np.abs(l1-l4), l1+l4+1, 2):
                        for lb in np.arange(np.abs(l2-l3), l2+l3+1, 2):
                            for ma, mb in itt.product(*[np.arange(-l, l+1, 2) for l in (la, lb)]):

                                value = np.float64(sympy.physics.wigner.gaunt(l1,l4,la,m1,m4,ma)*\
                                                   sympy.physics.wigner.gaunt(l2,l3,lb,m2,m3,mb))
                                if value != 0.:
                                    # Taking absolute values of all m as -m is equivalent to m
                                    # when Ylm is real and m is even
                                    m1, m2, m3, m4 = np.abs(m1), np.abs(m2), np.abs(m3), np.abs(m4)
                                    ma, mb = np.abs(ma), np.abs(mb)
                                    gaunt_coefficients[l1//2,l2//2,
                                                            l3//2,l4//2,
                                                            m1//2,m2//2,
                                                            m3//2,m4//2,
                                                            la//2,lb//2,
                                                            ma//2,mb//2] += value
                                    
                    for lc in np.arange(np.abs(l1-l2), l1+l2+1, 2):
                        for la in np.arange(np.abs(lc-l4), lc+l4+1, 2):
                            for ma, mc in itt.product(*[np.arange(-l, l+1, 2) for l in (la, lc)]):
                                value = np.float64(sympy.physics.wigner.gaunt(l1,l2,lc,m1,m2,mc)*\
                                                   sympy.physics.wigner.gaunt(lc,l4,la,mc,m4,ma))
                                lb, mb = l3, m3
                                if value != 0.:
                                    # Taking absolute values of all m as -m is equivalent to m
                                    # when Ylm is real and m is even
                                    m1, m2, m3, m4 = np.abs(m1), np.abs(m2), np.abs(m3), np.abs(m4)
                                    ma, mb = np.abs(ma), np.abs(mb)
                                    gaunt_coefficients[l1//2,l2//2,
                                                            l3//2,l4//2,
                                                            m1//2,m2//2,
                                                            m3//2,m4//2,
                                                            la//2,lb//2,
                                                            ma//2,mb//2] += value
            gaunt_coefficients.save(filename)

        return gaunt_coefficients


    def compute_window_kernels_old(self):
        '''Computes the window kernels to be used in the calculation of the covariance.

        Notes
        -----
        The window kernels are computed using the method described in [1]_.

        References
        ----------
        .. [1] https://arxiv.org/abs/1910.02914
        '''

        # sample kmodes from each k1 bin

        # SAMPLE FROM SHELL
        # kfun = 2 * np.pi / self.boxsize
        # kmodes = np.array([[math.sample_from_shell(kmin/kfun, kmax/kfun) for _ in range(
        #                    self.kmodes_sampled)] for kmin, kmax in zip(self.kedges[:-1], self.kedges[1:])])
        # Nmodes = math.nmodes(self.boxsize**3, self.kedges[:-1], self.kedges[1:])

        # SAMPLE FROM CUBE
        # kmodes, Nmodes = math.sample_from_cube(self.kmax/kfun, self.dk/kfun, self.kmodes_sampled)

        # HYBRID SAMPLING
        kmodes, Nmodes =  math.sample_kmodes(kmin=self.kmin,
                                            kmax=self.kmax,
                                            dk=self.dk,
                                            boxsize=self.boxsize,
                                            max_modes=self.kmodes_sampled,
                                            k_shell_approx=0.1,
                                            sample_mode="monte-carlo")

        if len(kmodes) != self.kbins or len(Nmodes) != self.kbins:
            raise ValueError(f'Error in thecov.utils.sample_kmodes: results should have length {self.kbins}, but had {len(kmodes)}. Parameters were kmin={self.kmin},kmax={self.kmax},dk={self.dk},boxsize={self.boxsize},max_modes={self.kmodes_sampled},k_shell_approx={0.1}).')

        # Calculate window FFTs if they haven't been initialized yet
        self.get_survey_window()

        init_params = {
            'boxsize': self.boxsize,
            'dk': self.dk,
            'nmesh': self.nmesh,
            'ikgrid': self.ikgrid,
            'delta_k_max': self.delta_k_max,
        }

        def init_worker(*args):
            global shared_w
            global shared_params
            shared_w = {}
            for w, l in zip(args, W_LABELS):
                shared_w[l] = np.frombuffer(w).view(np.complex128).reshape(self.nmesh, self.nmesh, self.nmesh)
            shared_params = args[-1]

        if self.WinKernel is None and self.WinKernel_error is None:
            # Format is [k1_bins, k2_bins, P_i x P_j term, Cov_ij]
            self.WinKernel = np.empty([self.kbins, 2*self.delta_k_max+1, 15, 6])
            self.WinKernel.fill(np.nan)

            self.WinKernel_error = np.empty([self.kbins, 2*self.delta_k_max+1, 15, 6])
            self.WinKernel_error.fill(np.nan)

        ell_factor = lambda l1,l2: (2*l1 + 1) * (2*l2 + 1) * (2 if 0 in (l1, l2) else 1)

        last_save = time.time()

        for i, km in self.tqdm(enumerate(kmodes), desc='Computing window kernels', total=self.kbins):

            if self._resume_file is not None:
                # Skip rows that were already computed
                if not np.isnan(self.WinKernel[i,0,0,0]):
                    # self.logger.debug(f'Skipping bin {i} of {self.kbins}.')
                    continue

            init_params['k1_bin_index'] = i + self.kmin//self.dk
            kmodes_sampled = len(km)

            # Splitting kmodes in chunks to be sent to each worker
            chunks = np.array_split(km, self.nthreads)

            with mp.Pool(processes=min(self.nthreads, len(chunks)),
                         initializer=init_worker,
                         initargs=[*[self._W[w] for w in W_LABELS], init_params]) as pool:
                
                results = pool.map(self._compute_window_kernel_row_old, chunks)

                self.WinKernel[i] = np.sum(results, axis=0) / kmodes_sampled

                std_results = np.std(results, axis=0) / np.sqrt(len(results))
                mean_results = np.mean(results, axis=0)
                mean_results[std_results == 0] = 1
                self.WinKernel_error[i] =  std_results / mean_results
        
                for k2_bin_index in range(0, 2*self.delta_k_max + 1):
                    if (k2_bin_index + i - self.delta_k_max >= self.kbins or k2_bin_index + i - self.delta_k_max < 0):
                        self.WinKernel[i, k2_bin_index, :, :] = 0
                    else:
                        self.WinKernel[i, k2_bin_index, :, :] /= Nmodes[i + k2_bin_index - self.delta_k_max]

            self.WinKernel[i, ..., 0] *= ell_factor(0,0)
            self.WinKernel[i, ..., 1] *= ell_factor(2,2)
            self.WinKernel[i, ..., 2] *= ell_factor(4,4)
            self.WinKernel[i, ..., 3] *= ell_factor(2,0)
            self.WinKernel[i, ..., 4] *= ell_factor(4,0)
            self.WinKernel[i, ..., 5] *= ell_factor(4,2)
            
            if self._resume_file is not None and (time.time() - last_save) > 600:
                self.save(self._resume_file)
                last_save = time.time()
                
        self.logger.info('Window kernels computed.')

        if self._resume_file is not None:
            self.save(self._resume_file)

    def clean(self):
        '''Clean window kernels and power spectra.'''
        self.WinKernel = None
        self.WinKernel_error = None
        self._window_power = None
        self._W = {}
        self._I = {}

    def compute_window_kernels_lebedev(self):

        # points on the unit sphere with corresponding integration weights
        x, y, z, w = math.get_lebedev_points(self.lebedev_degree)

        # Gaunt coefficients
        # cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/")
        G = self.get_gaunt_coefficients()

        # W_AB and W_CD
        self.get_survey_window()
        
        init_params = {
            'boxsize': self.boxsize,
            'dk': self.dk,
            'nmesh': self.nmesh,
            'ikgrid': self.ikgrid,
            'delta_k_max': self.delta_k_max,
        }

        # HYBRID SAMPLING
        kmodes, Nmodes, weights = math.sample_kmodes(kmin=self.kmin,
                                                     kmax=self.kmax,
                                                     dk=self.dk,
                                                     boxsize=self.boxsize,
                                                     max_modes=kmodes_sampled,
                                                     k_shell_approx=0.1,
                                                     sample_mode="monte-carlo")

        def init_worker(*args): #<- args is [W_AB, W_CD, ]
            global shared_W_ABCD
            global shared_params
            global shared_G

            for w_abcd, G, idx in zip(args):
               shared_W_ABCD = np.frombuffer(args[0]).view(np.complex128).reshape(self.nmesh, self.nmesh, self.nmesh)
            shared_G = G
            shared_params = args[-1]

        if self.WinKernel is None:
            # Format is [k1_bins, k2_bins, l1, l2, l3, l4]
            self.WinKernel = np.empty([self.kbins, 2*self.delta_k_max+1, 3, 3, 3, 3])
            self.WinKernel.fill(np.nan)

        ell_factor = lambda l1,l2: (2*l1 + 1) * (2*l2 + 1) * (2 if 0 in (l1, l2) else 1)
        last_save = time.time()
        for i, km in self.tqdm(enumerate(kmodes), desc='Computing window kernels', total=self.kbins):

            if self._resume_file is not None:
                # Skip rows that were already computed
                if not np.isnan(self.WinKernel[i,0,0,0,0,0]):
                    # self.logger.debug(f'Skipping bin {i} of {self.kbins}.')
                    continue

            init_params['k1_bin_index'] = i + self.kmin//self.dk
            kmodes_sampled = len(km)

            # Splitting k_bins in chunks to be sent to each worker
            chunks = np.array_split(km, self.nthreads)

            with mp.Pool(processes=min(self.nthreads, len(chunks)),
                         initializer=init_worker,
                         initargs=[*[self.W_AB[idx] for idx in self.ells_and_ms], 
                                   *[self.W_CD[idx] for idx in self.ells_and_ms],
                                   *G, init_params]) as pool:
                
                results = pool.map(self._compute_window_kernel_row, chunks)

                self.WinKernel[i] = np.sum(results * weights, axis=0) / kmodes_sampled

                std_results = np.std(results * weights, axis=0) / np.sqrt(len(results))
                avg_results = np.average(results * weights, weights=weights, axis=0)
                avg_results[std_results == 0] = 1
                self.WinKernel_error[i] =  std_results / avg_results
        
                for k2_bin_index in range(0, 2*self.delta_k_max + 1):
                    if (k2_bin_index + i - self.delta_k_max >= self.kbins or k2_bin_index + i - self.delta_k_max < 0):
                        self.WinKernel[i, k2_bin_index, :, :] = 0
                    else:
                        self.WinKernel[i, k2_bin_index, :, :] = avg_results[i, k2_bin_index, :, :]
        
            if self._resume_file is not None and (time.time() - last_save) > 600:
                self.save(self._resume_file)
                last_save = time.time()

        self.logger.info('Window kernels computed.')

        if self._resume_file is not None:
            self.save(self._resume_file)

    @staticmethod
    def _compute_window_kernel_row(bin_kmodes):
        '''Computes a row of the window kernels. This function is called in parallel for each k1 bin.
        Gives window kernels for L=0,2,4 auto and cross covariance (instead of only L=0 above)

        Returns:
            WinKernel: an array with [2*delta_k_max+1,num_ell,num_ell,num_ell,num_ell] dimensions.
                The first dim corresponds to the k-bin of k2
                (only 3 bins on each side of diagonal are included by default as the Gaussian covariance drops quickly away from diagonal)
                The remaining dims correspond to specific ells
        '''

        k1_bin_index = shared_params['k1_bin_index']
        boxsize = shared_params['boxsize']
        kfun = 2 * np.pi / boxsize
        dk = shared_params['dk']

        G = shared_G
        W = shared_w

        # The Gaussian covariance drops quickly away from diagonal.
        # Only delta_k_max points to each side of the diagonal are calculated.
        delta_k_max = shared_params['delta_k_max']

        WinKernel = np.zeros((2*delta_k_max+1, 3,3,3,3))
        iix, iiy, iiz = np.meshgrid(*shared_params['ikgrid'], indexing='ij')

        k2xh = np.zeros_like(iix)
        k2yh = np.zeros_like(iiy)
        k2zh = np.zeros_like(iiz)

        for ik1x, ik1y, ik1z, ik1r in bin_kmodes:

            if ik1r <= 1e-10:
                k1xh = 0
                k1yh = 0
                k1zh = 0
            else:
                k1xh = ik1x/ik1r
                k1yh = ik1y/ik1r
                k1zh = ik1z/ik1r

            # Build a 3D array of modes around the selected mode
            k2xh = ik1x-iix
            k2yh = ik1y-iiy
            k2zh = ik1z-iiz

            k2r = np.sqrt(k2xh**2 + k2yh**2 + k2zh**2)

            # to decide later which shell the k2 mode belongs to
            k2_bin_index = (k2r * kfun / dk).astype(int)
            k2r[k2r <= 1e-10] = np.inf
            k2xh /= k2r
            k2yh /= k2r
            k2zh /= k2r

            # k2_bin_index has shape (nmesh, nmesh, nmesh)
            # k1_bin_index is a scalar
            
            # give 3x3x3x3 x nmesh x nmesh x nmesh
            temp = np.sum(G @ np.einsum('ak,bk->abk', shared_W_AB, shared_W_CD), axis=[4,5,6,7])
            
            for l1, l2, l3, l4 in itt.product((0,2,4), repeat=4):
                for m1, m2, m3, m4 in itt.product(*[np.arange(-l, l+1, 2) for l in (l1, l2, l3, l4)]):
                    Ylm_1 = math.get_real_Ylm(l1, m1)
                    Ylm_2 = math.get_real_Ylm(l2, m2)
                    Ylm_3 = math.get_real_Ylm(l3, m3)
                    Ylm_4 = math.get_real_Ylm(l3, m4)

                    temp[l1,l2,l3,l4] += Ylm_1(k1xh, k1yh, k1zh) * \
                                         Ylm_2(k2xh, k2yh, k2zh) * \
                                         Ylm_3(k1xh, k1yh, k1zh) * \
                                         Ylm_4(k2xh, k2yh, k2zh)

            for delta_k in range(-delta_k_max, delta_k_max + 1):
                modes = (k2_bin_index - k1_bin_index == delta_k)
                WinKernel[delta_k] = temp[modes]

        return WinKernel