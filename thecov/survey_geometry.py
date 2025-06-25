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
        return np.pi * self.nmesh / self.boxsize
    @property
    def kfun(self):
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
        for _ in range(3):
            iik = np.arange(self.nmesh)
            iik[iik >= self.nmesh // 2] -= self.nmesh
            ikgrid.append(iik)
        return ikgrid
    
    def I(self, nbar_power, fkp_power):
        return (self._randoms[0]['NZ']**(nbar_power-1) * \
                self._randoms[0]['WEIGHT_FKP']**fkp_power * \
                self._randoms[0]['WEIGHT'] * \
                self.alpha1).sum().tolist()
    
    @staticmethod
    def _shotnoise_mesh(mesh, randoms, alpha):
        """Compute the shotnoise mesh S_AB = nbar * fkp^2."""

        return mesh.clone(
            data_positions=randoms['POSITION'],
            data_weights=randoms['WEIGHT_FKP']**2 * randoms[f'WEIGHT'] * alpha,
            position_type='pos',
        ).to_mesh(compensate=True)

    def mesh(self, ell, m, fourier=True, output_nmesh=None, threshold=None):
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

                
        target_boxsize = 2*np.pi/self._dk
        target_nmesh = int(np.ceil(target_boxsize * self._kmax / np.pi))

        trim_to_nmesh = int(np.ceil(target_boxsize/self.boxsize * self.nmesh))
        rebin_factor = trim_to_nmesh//target_nmesh

        new_boxsize = trim_to_nmesh/self.nmesh * self.boxsize
        new_nmesh = trim_to_nmesh//(trim_to_nmesh//self.nmesh)

        if new_nmesh <= self.nmesh and new_boxsize <= self.boxsize:
            result = result[:trim_to_nmesh:rebin_factor, :trim_to_nmesh:rebin_factor, :trim_to_nmesh:rebin_factor]
            self.logger.warn(f"Trimmed mesh from {self.nmesh} to {trim_to_nmesh} and boxsize from {self.boxsize} to {new_boxsize}.")
            self.logger.warn(f"Rebinned mesh from {self.nmesh} to {new_nmesh} with factor {rebin_factor}.")
        else:
            result = result.value

        # pmesh fft convention is F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r); let us correct it here
        if fourier:
            result *= new_nmesh**3

            result = numpy.fft.rfftn

        result = result.value if not fourier else result.r2c().value

        self.logger.warn(f"Mesh Fourier transform done in {time.time() - time_start:.2f} seconds")

        if threshold is not None:
            # Convert the result to a sparse array to save memory
            result[np.abs(result) < threshold] = 0
            result = base.SparseNDArray.from_dense(result, shape_in=(self.nmesh,self.nmesh), shape_out=self.nmesh)
        
        return result
    

class SurveyGeometry(base.BaseClass, base.LinearBinning):

    # survey window needs randoms1, alpha1, randoms2=None, alpha2=None, nmesh=None, cellsize=None, boxsize=None, boxpad=2., kmax=0.02, ellmax=4, **kwargs):
    def __init__(self, random_files, alphas, nmesh, boxsize, box_padding, k1, delta_k_max=3, ellmax=4, 
                 sample_mode="lebedev", lebedev_degree=25):

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
        self.k1 = k1
        self.ddelta_k_max = delta_k_max
        self.ellmax = ellmax
        self.sample_mode = sample_mode
        self.lebedev_degree = lebedev_degree

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
                                              self.nmesh, None, self.boxsize, self.box_padding, self.k1[-1], self.ellmax, shotnoise=False)
            self.window_CD = None
        elif self.num_tracers == 2:
            raise NotImplementedError
        elif self.num_tracers == 3:
            raise NotImplementedError
        elif self.num_tracers == 4:
            self.window_AB = SurveyWindow(self.randoms[0], self.alphas[0], self.randoms[1], self.alphas[1], 
                                     self.nmesh, None, self.boxsize, self.box_padding, self.k1[-1], self.ellmax, shotnoise=False)
            self.window_CD = SurveyWindow(self.randoms[2], self.alphas[2], self.randoms[3], self.alphas[3], 
                                     self.nmesh, None, self.boxsize, self.box_padding, self.k1[-1], self.ellmax, shotnoise=False)
    
    def get_survey_window(self):
        # calls survey_window

        if self.W_AB is None:

            self.W_AB = np.zeros(len(self.ells_and_ms, self.nmesh, self.nmesh, self.nmesh))
            self.W_CD = np.zeros_like(self.W_AB) if self.num_tracers > 1 else None
            for idx in range(len(self.ells_and_ms)):
                ell = self.ells_and_ms[idx][0]
                m = self.ells_and_ms[idx][1]
                self.W_AB[idx] = self.window_AB.mesh(ell, m)
                if self.num_tracers > 1: self.W_CD[idx] = self.window_CD.mesh(ell, m)
    

    def _init_ell_m_combos(self):
        """creates the unique combinations of l and m and stores them in a nested list"""
        self.ells_and_ms = []
        for ell in range(self.ellmax, step=2):
            for m in range(-ell, ell+1):
                self.ells_and_ms.append([ell, m])


    def get_gaunt_coefficients(self, cache_dir):
        """Calculates all relavent Gaunt coefficients, or loads them from file"""

        # Load mask coupling Gaunt coefficients if cache exists, otherwise compute them
        filename = os.path.join(cache_dir, "cosmic_variance_coefficients.npz")

        if os.path.exists(filename):
            self.gaunt_coefficients = base.SparseNDArray.load(filename)
        else:
            import sympy.physics.wigner

            # shape_out = l1, l2, l3, l4, m1, m2, m3, m4
            # shape_in =  la, lb, ma, mb
            # Only including positive m values, as -m is equivalent to m
            # when Ylm is real and m is even
            self.gaunt_coefficients = base.SparseNDArray(shape_out=(3,3,3,3,3,3,3,3), shape_in=(7,7,7,7))

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
                                    self.gaunt_coefficients[l1//2,l2//2,
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
                                    self.gaunt_coefficients[l1//2,l2//2,
                                                            l3//2,l4//2,
                                                            m1//2,m2//2,
                                                            m3//2,m4//2,
                                                            la//2,lb//2,
                                                            ma//2,mb//2] += value
            self.gaunt_coefficients.save(filename)


    def compute_window_kernels(self):
        if self.sample_mode == "monte-carlo":
            self.logger.warn("WARNING! This mode is old!")
            self.compute_window_kernels_monte_carlo()
        elif self.sample_mode == "lebedev":
            self.compute_window_kernels_lebedev()

    def compute_window_kernels_monte_carlo(self):
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
                                            k_shell_approx=0.1)

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

    @staticmethod
    def _compute_window_kernel_row_old(bin_kmodes):
        '''Computes a row of the window kernels. This function is called in parallel for each k1 bin.'''
        # Gives window kernels for L=0,2,4 auto and cross covariance (instead of only L=0 above)

        # Returns an array with [2*delta_k_max+1,15,6] dimensions.
        #    The first dim corresponds to the k-bin of k2
        #    (only 3 bins on each side of diagonal are included as the Gaussian covariance drops quickly away from diagonal)

        #    The second dim corresponds to elements to be multiplied by various power spectrum multipoles
        #    to obtain the final covariance (see function 'Wij' below)

        #    The last dim corresponds to multipoles: [L0xL0,L2xL2,L4xL4,L2xL0,L4xL0,L4xL2]

        k1_bin_index = shared_params['k1_bin_index']
        boxsize = shared_params['boxsize']
        kfun = 2 * np.pi / boxsize
        dk = shared_params['dk']

        W = shared_w

        # The Gaussian covariance drops quickly away from diagonal.
        # Only delta_k_max points to each side of the diagonal are calculated.
        delta_k_max = shared_params['delta_k_max']

        WinKernel = np.zeros((2*delta_k_max+1, 15, 6))

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
            # k2 hat arrays built

            # Expressions below come straight from CovaPT (arXiv:1910.02914)

            # Now calculating window multipole kernels by taking dot products of cartesian FFTs with k1-hat, k2-hat arrays
            # W corresponds to W22(k) and Wc corresponds to conjugate of W22(k)
            # L(i) refers to multipoles
            # TODO: replace all of the below with calculations done in a spherical basis

            W_L0 = W['22']
            Wc_L0 = np.conj(W['22'])

            xx = W['22xx']*k1xh**2 + W['22yy']*k1yh**2 + W['22zz']*k1zh**2 + 2. * \
                W['22xy']*k1xh*k1yh + 2.*W['22yz'] * \
                k1yh*k1zh + 2.*W['22xz']*k1zh*k1xh
            W_k1L2 = 1.5*xx - 0.5*W['22']
            W_k2L2 = 1.5*(W['22xx']*k2xh**2 + W['22yy']*k2yh**2 + W['22zz']*k2zh**2
                          + 2.*W['22xy']*k2xh*k2yh + 2.*W['22yz']*k2yh*k2zh + 2.*W['22xz']*k2zh*k2xh) - 0.5*W['22']
            Wc_k1L2 = np.conj(W_k1L2)
            Wc_k2L2 = np.conj(W_k2L2)

            W_k1L4 = 35./8.*(W['22xxxx']*k1xh**4 + W['22yyyy']*k1yh**4 + W['22zzzz']*k1zh**4
                             + 4.*W['22xxxy']*k1xh**3*k1yh + 4.*W['22xxxz'] *
                             k1xh**3*k1zh + 4.*W['22xyyy']*k1yh**3*k1xh
                             + 4.*W['22yyyz']*k1yh**3*k1zh + 4.*W['22xzzz'] *
                             k1zh**3*k1xh + 4.*W['22yzzz']*k1zh**3*k1yh
                             + 6.*W['22xxyy']*k1xh**2*k1yh**2 + 6.*W['22xxzz'] *
                             k1xh**2*k1zh**2 + 6.*W['22yyzz']*k1yh**2*k1zh**2
                             + 12.*W['22xxyz']*k1xh**2*k1yh*k1zh + 12.*W['22xyyz']*k1yh**2*k1xh*k1zh + 12.*W['22xyzz']*k1zh**2*k1xh*k1yh) \
                - 5./2.*W_k1L2 - 7./8.*W_L0

            Wc_k1L4 = np.conj(W_k1L4)

            k1k2 = W['22xxxx']*(k1xh*k2xh)**2 + W['22yyyy']*(k1yh*k2yh)**2+W['22zzzz']*(k1zh*k2zh)**2 \
                + W['22xxxy']*(k1xh*k1yh*k2xh**2 + k1xh**2*k2xh*k2yh)*2 \
                + W['22xxxz']*(k1xh*k1zh*k2xh**2 + k1xh**2*k2xh*k2zh)*2 \
                + W['22yyyz']*(k1yh*k1zh*k2yh**2 + k1yh**2*k2yh*k2zh)*2 \
                + W['22yzzz']*(k1zh*k1yh*k2zh**2 + k1zh**2*k2zh*k2yh)*2 \
                + W['22xyyy']*(k1yh*k1xh*k2yh**2 + k1yh**2*k2yh*k2xh)*2 \
                + W['22xzzz']*(k1zh*k1xh*k2zh**2 + k1zh**2*k2zh*k2xh)*2 \
                + W['22xxyy']*(k1xh**2*k2yh**2 + k1yh**2*k2xh**2 + 4.*k1xh*k1yh*k2xh*k2yh) \
                + W['22xxzz']*(k1xh**2*k2zh**2 + k1zh**2*k2xh**2 + 4.*k1xh*k1zh*k2xh*k2zh) \
                + W['22yyzz']*(k1yh**2*k2zh**2 + k1zh**2*k2yh**2 + 4.*k1yh*k1zh*k2yh*k2zh) \
                + W['22xyyz']*(k1xh*k1zh*k2yh**2 + k1yh**2*k2xh*k2zh + 2.*k1yh*k2yh*(k1zh*k2xh + k1xh*k2zh))*2 \
                + W['22xxyz']*(k1yh*k1zh*k2xh**2 + k1xh**2*k2yh*k2zh + 2.*k1xh*k2xh*(k1zh*k2yh + k1yh*k2zh))*2 \
                + W['22xyzz']*(k1yh*k1xh*k2zh**2 + k1zh**2*k2yh *
                               k2xh + 2.*k1zh*k2zh*(k1xh*k2yh + k1yh*k2xh))*2

            W_k2L4 = 35./8.*(W['22xxxx']*k2xh**4 + W['22yyyy']*k2yh**4 + W['22zzzz']*k2zh**4
                             + 4.*W['22xxxy']*k2xh**3*k2yh + 4.*W['22xxxz'] *
                             k2xh**3*k2zh + 4.*W['22xyyy']*k2yh**3*k2xh
                             + 4.*W['22yyyz']*k2yh**3*k2zh + 4.*W['22xzzz'] *
                             k2zh**3*k2xh + 4.*W['22yzzz']*k2zh**3*k2yh
                             + 6.*W['22xxyy']*k2xh**2*k2yh**2 + 6.*W['22xxzz'] *
                             k2xh**2*k2zh**2 + 6.*W['22yyzz']*k2yh**2*k2zh**2
                             + 12.*W['22xxyz']*k2xh**2*k2yh*k2zh + 12.*W['22xyyz']*k2yh**2*k2xh*k2zh + 12.*W['22xyzz']*k2zh**2*k2xh*k2yh) \
                - 5./2.*W_k2L2 - 7./8.*W_L0

            Wc_k2L4 = np.conj(W_k2L4)

            W_k1L2_k2L2 = 9./4.*k1k2 - 3./4.*xx - 1./2.*W_k2L2
            # approximate as 6th order FFTs not simulated
            W_k1L2_k2L4 = 2/7.*W_k1L2 + 20/77.*W_k1L4
            W_k1L4_k2L2 = W_k1L2_k2L4  # approximate
            W_k1L4_k2L4 = 1/9.*W_L0 + 100/693.*W_k1L2 + 162/1001.*W_k1L4

            Wc_k1L2_k2L2 = np.conj(W_k1L2_k2L2)
            Wc_k1L2_k2L4 = np.conj(W_k1L2_k2L4)
            Wc_k1L4_k2L2 = Wc_k1L2_k2L4
            Wc_k1L4_k2L4 = np.conj(W_k1L4_k2L4)

            k1k2W12 = np.conj(W['12xxxx'])*(k1xh*k2xh)**2 + np.conj(W['12yyyy'])*(k1yh*k2yh)**2 + np.conj(W['12zzzz'])*(k1zh*k2zh)**2 \
                + np.conj(W['12xxxy'])*(k1xh*k1yh*k2xh**2 + k1xh**2*k2xh*k2yh)*2 \
                + np.conj(W['12xxxz'])*(k1xh*k1zh*k2xh**2 + k1xh**2*k2xh*k2zh)*2 \
                + np.conj(W['12yyyz'])*(k1yh*k1zh*k2yh**2 + k1yh**2*k2yh*k2zh)*2 \
                + np.conj(W['12yzzz'])*(k1zh*k1yh*k2zh**2 + k1zh**2*k2zh*k2yh)*2 \
                + np.conj(W['12xyyy'])*(k1yh*k1xh*k2yh**2 + k1yh**2*k2yh*k2xh)*2 \
                + np.conj(W['12xzzz'])*(k1zh*k1xh*k2zh**2 + k1zh**2*k2zh*k2xh)*2 \
                + np.conj(W['12xxyy'])*(k1xh**2*k2yh**2 + k1yh**2*k2xh**2 + 4.*k1xh*k1yh*k2xh*k2yh) \
                + np.conj(W['12xxzz'])*(k1xh**2*k2zh**2 + k1zh**2*k2xh**2 + 4.*k1xh*k1zh*k2xh*k2zh) \
                + np.conj(W['12yyzz'])*(k1yh**2*k2zh**2 + k1zh**2*k2yh**2 + 4.*k1yh*k1zh*k2yh*k2zh) \
                + np.conj(W['12xyyz'])*(k1xh*k1zh*k2yh**2 + k1yh**2*k2xh*k2zh + 2.*k1yh*k2yh*(k1zh*k2xh + k1xh*k2zh))*2 \
                + np.conj(W['12xxyz'])*(k1yh*k1zh*k2xh**2 + k1xh**2*k2yh*k2zh + 2.*k1xh*k2xh*(k1zh*k2yh + k1yh*k2zh))*2 \
                + np.conj(W['12xyzz'])*(k1yh*k1xh*k2zh**2 + k1zh**2*k2yh *
                               k2xh + 2.*k1zh*k2zh*(k1xh*k2yh + k1yh*k2xh))*2

            xxW12 = np.conj(W['12xx'])*k1xh**2 + np.conj(W['12yy'])*k1yh**2 + np.conj(W['12zz'])*k1zh**2 \
                + 2.*np.conj(W['12xy'])*k1xh*k1yh + 2.*np.conj(W['12yz']) * \
                k1yh*k1zh + 2.*np.conj(W['12xz'])*k1zh*k1xh

            W12c_L0 = np.conj(W['12'])
            W12_k1L2 = 1.5*xxW12 - 0.5*np.conj(W['12'])
            W12_k1L4 = 35./8.*(np.conj(W['12xxxx'])*k1xh**4 + np.conj(W['12yyyy'])*k1yh**4 + np.conj(W['12zzzz'])*k1zh**4
                               + 4.*np.conj(W['12xxxy'])*k1xh**3*k1yh + 4.*np.conj(W['12xxxz']) *
                               k1xh**3*k1zh + 4.*np.conj(W['12xyyy'])*k1yh**3*k1xh
                               + 6.*np.conj(W['12xxyy'])*k1xh**2*k1yh**2 + 6.*np.conj(W['12xxzz']) *
                               k1xh**2*k1zh**2 + 6.*np.conj(W['12yyzz'])*k1yh**2*k1zh**2
                               + 12.*np.conj(W['12xxyz'])*k1xh**2*k1yh*k1zh + 12.*np.conj(W['12xyyz'])*k1yh**2*k1xh*k1zh + 12.*np.conj(W['12xyzz'])*k1zh**2*k1xh*k1yh) \
                - 5./2.*W12_k1L2 - 7./8.*W12c_L0

            W12_k1L4_k2L2 = 2/7.*W12_k1L2 + 20/77.*W12_k1L4
            W12_k1L4_k2L4 = 1/9.*W12c_L0 + 100/693.*W12_k1L2 + 162/1001.*W12_k1L4

            W12_k2L2 = 1.5*(np.conj(W['12xx'])*k2xh**2 + np.conj(W['12yy'])*k2yh**2 + np.conj(W['12zz'])*k2zh**2
                            + 2.*np.conj(W['12xy'])*k2xh*k2yh + 2.*np.conj(W['12yz'])*k2yh*k2zh + 2.*np.conj(W['12xz'])*k2zh*k2xh) - 0.5*np.conj(W['12'])

            W12_k2L4 = 35./8.*(np.conj(W['12xxxx'])*k2xh**4 + np.conj(W['12yyyy'])*k2yh**4 + np.conj(W['12zzzz'])*k2zh**4
                               + 4.*np.conj(W['12xxxy'])*k2xh**3*k2yh + 4.*np.conj(W['12xxxz']) *
                               k2xh**3*k2zh + 4.*np.conj(W['12xyyy'])*k2yh**3*k2xh
                               + 4.*np.conj(W['12yyyz'])*k2yh**3*k2zh + 4.*np.conj(W['12xzzz']) *
                               k2zh**3*k2xh + 4.*np.conj(W['12yzzz'])*k2zh**3*k2yh
                               + 6.*np.conj(W['12xxyy'])*k2xh**2*k2yh**2 + 6.*np.conj(W['12xxzz']) *
                               k2xh**2*k2zh**2 + 6.*np.conj(W['12yyzz'])*k2yh**2*k2zh**2
                               + 12.*np.conj(W['12xxyz'])*k2xh**2*k2yh*k2zh + 12.*np.conj(W['12xyyz'])*k2yh**2*k2xh*k2zh + 12.*np.conj(W['12xyzz'])*k2zh**2*k2xh*k2yh) \
                - 5./2.*W12_k2L2 - 7./8.*W12c_L0

            W12_k1L2_k2L2 = 9./4.*k1k2W12 - 3./4.*xxW12 - 1./2.*W12_k2L2

            W_k1L2_Sumk2L22 = 1/5.*W_k1L2 + 2/7.*W_k1L2_k2L2 + 18/35.*W_k1L2_k2L4
            W_k1L2_Sumk2L24 = 2/7.*W_k1L2_k2L2 + 20/77.*W_k1L2_k2L4
            W_k1L4_Sumk2L22 = 1/5.*W_k1L4 + 2/7.*W_k1L4_k2L2 + 18/35.*W_k1L4_k2L4
            W_k1L4_Sumk2L24 = 2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4
            W_k1L4_Sumk2L44 = 1/9.*W_k1L4 + 100/693.*W_k1L4_k2L2 + 162/1001.*W_k1L4_k2L4

            C00exp = [Wc_L0 * W_L0, Wc_L0 * W_k2L2, Wc_L0 * W_k2L4,
                      Wc_k1L2*W_L0, Wc_k1L2*W_k2L2, Wc_k1L2*W_k2L4,
                      Wc_k1L4*W_L0, Wc_k1L4*W_k2L2, Wc_k1L4*W_k2L4]

            C00exp += [2.*W_L0 * W12c_L0, W_k1L2*W12c_L0,         W_k1L4 * W12c_L0,
                       W_k2L2*W12c_L0, W_k2L4*W12c_L0, np.conj(W12c_L0)*W12c_L0]

            C22exp = [Wc_k2L2*W_k1L2 + Wc_L0*W_k1L2_k2L2,
                      Wc_k2L2*W_k1L2_k2L2 + Wc_L0*W_k1L2_Sumk2L22,
                      Wc_k2L2*W_k1L2_k2L4 + Wc_L0*W_k1L2_Sumk2L24,
                      Wc_k1L2_k2L2*W_k1L2 + Wc_k1L2*W_k1L2_k2L2,
                      Wc_k1L2_k2L2*W_k1L2_k2L2 + Wc_k1L2*W_k1L2_Sumk2L22,
                      Wc_k1L2_k2L2*W_k1L2_k2L4 + Wc_k1L2*W_k1L2_Sumk2L24,
                      Wc_k1L4_k2L2*W_k1L2 + Wc_k1L4*W_k1L2_k2L2,
                      Wc_k1L4_k2L2*W_k1L2_k2L2 + Wc_k1L4*W_k1L2_Sumk2L22,
                      Wc_k1L4_k2L2*W_k1L2_k2L4 + Wc_k1L4*W_k1L2_Sumk2L24]

            C22exp += [W_k1L2*W12_k2L2 + W_k2L2*W12_k1L2 + W_k1L2_k2L2*W12c_L0+W_L0*W12_k1L2_k2L2,

                       0.5*((1/5.*W_L0+2/7.*W_k1L2 + 18/35.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L2
                            + (1/5.*W_k2L2+2/7.*W_k1L2_k2L2 + 18/35.*W_k1L4_k2L2)*W12c_L0 + W_k1L2*W12_k1L2_k2L2),

                       0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L2
                            + (2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12c_L0 + W_k1L4*W12_k1L2_k2L2),

                       0.5*(W_k1L2_k2L2*W12_k2L2 + (1/5.*W_L0 + 2/7.*W_k2L2 + 18/35.*W_k2L4)*W12_k1L2
                            + (1/5.*W_k1L2 + 2/7.*W_k1L2_k2L2 + 18/35.*W_k1L2_k2L4)*W12c_L0 + W_k2L2*W12_k1L2_k2L2),

                       0.5*(W_k1L2_k2L4*W12_k2L2 + (2/7.*W_k2L2 + 20/77.*W_k2L4)*W12_k1L2
                            + W_k2L4*W12_k1L2_k2L2 + (2/7.*W_k1L2_k2L2 + 20/77.*W_k1L2_k2L4)*W12c_L0),

                       np.conj(W12_k1L2_k2L2)*W12c_L0 + np.conj(W12_k1L2)*W12_k2L2]

            C44exp = [Wc_k2L4 * W_k1L4 + Wc_L0 * W_k1L4_k2L4,
                      Wc_k2L4 * W_k1L4_k2L2 + Wc_L0 * W_k1L4_Sumk2L24,
                      Wc_k2L4 * W_k1L4_k2L4 + Wc_L0 * W_k1L4_Sumk2L44,
                      Wc_k1L2_k2L4*W_k1L4 + Wc_k1L2*W_k1L4_k2L4,
                      Wc_k1L2_k2L4*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L24,
                      Wc_k1L2_k2L4*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L44,
                      Wc_k1L4_k2L4*W_k1L4 + Wc_k1L4*W_k1L4_k2L4,
                      Wc_k1L4_k2L4*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L24,
                      Wc_k1L4_k2L4*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L44]

            C44exp += [W_k1L4 * W12_k2L4 + W_k2L4*W12_k1L4
                       + W_k1L4_k2L4*W12c_L0 + W_L0 * W12_k1L4_k2L4,

                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)*W12_k2L4 + W_k1L2_k2L4*W12_k1L4
                            + (2/7.*W_k1L2_k2L4 + 20/77.*W_k1L4_k2L4)*W12c_L0 + W_k1L2 * W12_k1L4_k2L4),

                       0.5*((1/9.*W_L0 + 100/693.*W_k1L2 + 162/1001.*W_k1L4)*W12_k2L4 + W_k1L4_k2L4*W12_k1L4
                            + (1/9.*W_k2L4 + 100/693.*W_k1L2_k2L4 + 162/1001.*W_k1L4_k2L4)*W12c_L0 + W_k1L4 * W12_k1L4_k2L4),

                       0.5*(W_k1L4_k2L2*W12_k2L4 + (2/7.*W_k2L2 + 20/77.*W_k2L4)*W12_k1L4
                            + W_k2L2*W12_k1L4_k2L4 + (2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4)*W12c_L0),

                       0.5*(W_k1L4_k2L4*W12_k2L4 + (1/9.*W_L0 + 100/693.*W_k2L2 + 162/1001.*W_k2L4)*W12_k1L4
                            + W_k2L4*W12_k1L4_k2L4 + (1/9.*W_k1L4 + 100/693.*W_k1L4_k2L2 + 162/1001.*W_k1L4_k2L4)*W12c_L0),

                       np.conj(W12_k1L4_k2L4)*W12c_L0 + np.conj(W12_k1L4)*W12_k2L4]  # 1/(nbar)^2

            C20exp = [Wc_L0 * W_k1L2,   Wc_L0*W_k1L2_k2L2, Wc_L0 * W_k1L2_k2L4,
                      Wc_k1L2*W_k1L2, Wc_k1L2*W_k1L2_k2L2, Wc_k1L2*W_k1L2_k2L4,
                      Wc_k1L4*W_k1L2, Wc_k1L4*W_k1L2_k2L2, Wc_k1L4*W_k1L2_k2L4]

            C20exp += [W_k1L2*W12c_L0 + W['22']*W12_k1L2,
                       0.5*((1/5.*W['22'] + 2/7.*W_k1L2 + 18 /
                            35.*W_k1L4)*W12c_L0 + W_k1L2*W12_k1L2),
                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)
                            * W12c_L0 + W_k1L4*W12_k1L2),
                       0.5*(W_k1L2_k2L2*W12c_L0 + W_k2L2*W12_k1L2),
                       0.5*(W_k1L2_k2L4*W12c_L0 + W_k2L4*W12_k1L2),
                       np.conj(W12_k1L2)*W12c_L0]

            C40exp = [Wc_L0*W_k1L4,   Wc_L0 * W_k1L4_k2L2, Wc_L0 * W_k1L4_k2L4,
                      Wc_k1L2*W_k1L4, Wc_k1L2*W_k1L4_k2L2, Wc_k1L2*W_k1L4_k2L4,
                      Wc_k1L4*W_k1L4, Wc_k1L4*W_k1L4_k2L2, Wc_k1L4*W_k1L4_k2L4]

            C40exp += [W_k1L4*W12c_L0 + W['22']*W12_k1L4,
                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)
                            * W12c_L0 + W_k1L2*W12_k1L4),
                       0.5*((1/9.*W['22'] + 100/693.*W_k1L2+162 /
                            1001.*W_k1L4)*W12c_L0 + W_k1L4*W12_k1L4),
                       0.5*(W_k1L4_k2L2*W12c_L0 + W_k2L2*W12_k1L4),
                       0.5*(W_k1L4_k2L4*W12c_L0 + W_k2L4*W12_k1L4),
                       np.conj(W12_k1L4)*W12c_L0]

            C42exp = [Wc_k2L2*W_k1L4 + Wc_L0 * W_k1L4_k2L2,
                      Wc_k2L2*W_k1L4_k2L2 + Wc_L0 * W_k1L4_Sumk2L22,
                      Wc_k2L2*W_k1L4_k2L4 + Wc_L0 * W_k1L4_Sumk2L24,
                      Wc_k1L2_k2L2*W_k1L4 + Wc_k1L2*W_k1L4_k2L2,
                      Wc_k1L2_k2L2*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L22,
                      Wc_k1L2_k2L2*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L24,
                      Wc_k1L4_k2L2*W_k1L4 + Wc_k1L4*W_k1L4_k2L2,
                      Wc_k1L4_k2L2*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L22,
                      Wc_k1L4_k2L2*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L24]

            C42exp += [W_k1L4*W12_k2L2 + W_k2L2*W12_k1L4
                       + W_k1L4_k2L2*W12c_L0 + W['22']*W12_k1L4_k2L2,

                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L4
                            + (2/7.*W_k1L2_k2L2 + 20/77.*W_k1L4_k2L2)*W12c_L0 + W_k1L2 * W12_k1L4_k2L2),

                       0.5*((1/9.*W['22'] + 100/693.*W_k1L2 + 162/1001.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L4
                            + (1/9.*W_k2L2 + 100/693.*W_k1L2_k2L2 + 162/1001.*W_k1L4_k2L2)*W12c_L0 + W_k1L4*W12_k1L4_k2L2),

                       0.5*(W_k1L4_k2L2*W12_k2L2 + (1/5.*W['22'] + 2/7.*W_k2L2 + 18/35.*W_k2L4)*W12_k1L4
                            + W_k2L2*W12_k1L4_k2L2 + (1/5.*W_k1L4 + 2/7.*W_k1L4_k2L2 + 18/35.*W_k1L4_k2L4)*W12c_L0),

                       0.5*(W_k1L4_k2L4*W12_k2L2 + (2/7.*W_k2L2 + 20/77.*W_k2L4)*W12_k1L4
                            + W_k2L4*W12_k1L4_k2L2 + (2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4)*W12c_L0),

                       np.conj(W12_k1L4_k2L2)*W12c_L0+np.conj(W12_k1L4)*W12_k2L2]  # 1/(nbar)^2

            for delta_k in range(-delta_k_max, delta_k_max + 1):
                # k2_bin_index has shape (nmesh, nmesh, nmesh)
                # k1_bin_index is a scalar
                modes = (k2_bin_index - k1_bin_index == delta_k)

                # Iterating over terms (m,m') that will multiply P_m(k1)*P_m'(k2) in the sum
                for term in range(15):
                    WinKernel[delta_k + delta_k_max, term, 0] += np.sum(np.real(C00exp[term][modes]))
                    WinKernel[delta_k + delta_k_max, term, 1] += np.sum(np.real(C22exp[term][modes]))
                    WinKernel[delta_k + delta_k_max, term, 2] += np.sum(np.real(C44exp[term][modes]))
                    WinKernel[delta_k + delta_k_max, term, 3] += np.sum(np.real(C20exp[term][modes]))
                    WinKernel[delta_k + delta_k_max, term, 4] += np.sum(np.real(C40exp[term][modes]))
                    WinKernel[delta_k + delta_k_max, term, 5] += np.sum(np.real(C42exp[term][modes]))
        
        return WinKernel


    def compute_window_kernels_lebedev(self):

        # points on the unit sphere with corresponding integration weights
        x, y, z, w = math.get_lebedev_points(self.lebedev_degree)

        # Gaunt coefficients
        cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/")
        G = self.get_gaunt_coefficients(cache_dir)

        # W_AB and W_CD
        self.get_survey_window()

        def init_worker(*args):
            global shared_w
            global shared_params
            shared_w = {}
            #for w, l in zip(args, W_LABELS):
            #    shared_w[l] = np.frombuffer(w).view(np.complex128).reshape(self.nmesh, self.nmesh, self.nmesh)
            shared_params = args[-1]

        if self.WinKernel is None and self.WinKernel_error is None:
            # Format is [k1_bins, k2_bins, l1, l2, l3, l4]
            self.WinKernel = np.empty([self.kbins, 2*self.delta_k_max+1, 3, 3, 3, 3])
            self.WinKernel.fill(np.nan)

        ell_factor = lambda l1,l2: (2*l1 + 1) * (2*l2 + 1) * (2 if 0 in (l1, l2) else 1)

        last_save = time.time()

        for i, km in self.tqdm(enumerate(self.k1), desc='Computing window kernels', total=len(self.k1)):

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

        print("hi!")