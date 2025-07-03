"""Module containing classes for creating the window function to be used in calculating the Gaussian covariance term.

Classes
-------
SurveyWindow
    Class re
SurveyGeometry
"""

import logging
logging.basicConfig(level = logging.WARN)

import numpy as np
import os, time
import itertools as itt

from tqdm import tqdm as shell_tqdm
import multiprocessing as mp
import multiprocessing.shared_memory
# from scipy.integrate import lebedev_rule

import mockfactory
from pypower import CatalogMesh

import functools

from . import base, math

MASK_ELL_MAX = 12
PK_ELL_MAX = 4

__all__ = ['SurveyWindow', 'SurveyGeometry']

class SurveyWindow(base.BaseClass, base.LinearBinning):

    def __init__(self, randoms1, alpha1, randoms2=None, alpha2=None, nmesh=None, cellsize=None, boxsize=None, boxpad=2., kmin=0.0, kmax=0.02, dk=None, shotnoise=False, **kwargs):

        super().__init__(kmin, kmax, dk)

        self.logger = logging.getLogger('SurveyWindow')
        self.logger.setLevel(logging.INFO)
        self.tqdm = shell_tqdm

        self._is_shotnoise = shotnoise
        self.kmax = kmax
        self.dk = dk

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
        self.I_1 = self.I(randoms1, alpha1, 1, 2)

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
            self.I_2 = self.I(randoms2, alpha2, 1, 2)

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
        
        self.logger.info(f'Using box size {self.boxsize}, box center {self.mesh1.boxcenter} and nmesh {self.nmesh}.')
        self.logger.info(f'Fundamental wavenumber of window meshes = {self.kfun}.')
        self.logger.info(f'Nyquist wavenumber of window meshes = {self.knyquist}.')

        if kmax is not None and self.knyquist < kmax:
            self.logger.warning(f'Nyquist wavelength {self.knyquist} smaller than required window kmax = {kmax}.')

        self.logger.info(f'Average of {self.mesh1.data_size / self.nmesh**3} objects per voxel.')

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
            self.logger.info(f'fsky estimated from randoms: {fsky:.3f}')
            nbar = mockfactory.RedshiftDensityInterpolator(z=distance, weights=randoms['WEIGHT'], fsky=fsky)
            randoms['NZ'] = nbar(distance)

        # Check if the randoms have nmesh and cellsize, otherwise set them using the kmax parameter
        if nmesh is None and cellsize is None:
            # Pick value that will give at least k_mask = kmax_window in the FFTs
            self.cellsize = np.pi / kmax / (1. + 1e-9)

        self.logger.info(f'Parsed randoms in {time.time() - start_time:.2f} seconds.')

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

    def I(self, randoms, alpha, nbar_power, fkp_power):
        return (randoms['NZ']**(nbar_power-1) * \
                randoms['WEIGHT_FKP']**fkp_power * \
                randoms['WEIGHT'] * \
                alpha).sum().tolist()
    
    def _rebin_parameters(self, dk, kmax):

        # If dk and kmax are provided, they determine the target boxsize and nmesh
        target_boxsize = 2*np.pi/dk
        target_nmesh = int(np.ceil(target_boxsize * kmax / np.pi))

        # Trim the mesh to achieve the target boxsize
        trim_to_nmesh = int(np.ceil(target_boxsize/self.boxsize * self.nmesh))
        
        # Rebin mesh to obtain the target nmesh
        rebin_factor = trim_to_nmesh//target_nmesh

        if rebin_factor == 0:
            self.logger.error(f"Rebin factor = 0 with the given values of dk ({dk}) and kmax ({kmax})! This might mean your nmesh is too small")
            raise ZeroDivisionError

        # Ensure that trim_to_nmesh is a multiple of rebin_factor
        if (trim_to_nmesh % rebin_factor) != 0:
            trim_to_nmesh +=  rebin_factor - (trim_to_nmesh % rebin_factor)

        self.kboxsize = trim_to_nmesh/self.nmesh * self.boxsize
        self.knmesh = trim_to_nmesh//rebin_factor

        # NOTE: idk if this return is necesary
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

        self.logger.info(f"Mesh computation with Ylm ({ell}, {m}) done in {time.time() - time_start:.2f} seconds")
        time_start = time.time()

        if hasattr(self, 'mesh2'):
            result *= self.mesh2.to_mesh(compensate=True)
        else:
            result *= self.mesh1.to_mesh(compensate=True)

        self.logger.info(f"Second mesh computation done in {time.time() - time_start:.2f} seconds")

        if self.dk is not None and self.kmax is not None:
            trim_to_nmesh, rebin_factor = self._rebin_parameters(self.dk, self.kmax)
            
            if trim_to_nmesh <= self.nmesh and self.kboxsize <= self.boxsize:
                result = result[:trim_to_nmesh, :trim_to_nmesh, :trim_to_nmesh]
                self.logger.info(f"Trimmed mesh from {self.nmesh} to {trim_to_nmesh} and boxsize from {self.boxsize:.0f} to {self.kboxsize:.0f}.")

                # Sum mesh values in the new mesh
                if rebin_factor > 1:
                    result = result.reshape((self.knmesh, rebin_factor, self.knmesh, rebin_factor, self.knmesh, rebin_factor)).sum(axis=(1, 3, 5))

                self.logger.info(f"Rebinned mesh from {trim_to_nmesh} to {result.shape[0]} with factor {rebin_factor}.")


        if hasattr(result, 'value'):
            result = result.value

        # pmesh fft convention is F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r); let us correct it here

        if fourier:
            result *= self.knmesh**3
            time_start = time.time()
            result = np.fft.fftn(result, axes=(0, 1, 2), norm='backward')
            self.logger.info(f"Mesh Fourier transform done in {time.time() - time_start:.2f} seconds")

        # result = result.value if not fourier else result.r2c().value

        if threshold is not None:
            # Convert the result to a sparse array to save memory
            result[np.abs(result) < threshold] = 0
            result = base.SparseNDArray.from_dense(result, shape_in=(self.nmesh,self.nmesh), shape_out=self.nmesh)
        
        return result
    
# barebones class so covariance.py compiles without error for now
class BoxGeometry(base.BaseClass, base.LinearBinning):

    def __init__(self):
        pass


class SurveyGeometry(base.BaseClass, base.LinearBinning):

    # survey window needs randoms1, alpha1, randoms2=None, alpha2=None, nmesh=None, cellsize=None, boxsize=None, boxpad=2., kmax=0.02, ellmax=4, **kwargs):
    def __init__(self,
                 randoms_a,      alpha_a,
                 randoms_b=None, alpha_b=None,
                 randoms_c=None, alpha_c=None,
                 randoms_d=None, alpha_d=None,
                 nmesh=None, boxsize=None, boxpad=2.,
                 kmin=0, kmax=0.2, dk=None, mask_ellmax=12, pk_ellmax=4,
                 sample_mode="lebedev", lebedev_degree=25, nthreads=None):

        # set's k-binning
        super().__init__(kmin, kmax, dk)

        self.logger = logging.getLogger('SurveyGeometry')
        self.logger.setLevel(logging.INFO)
        self.tqdm = shell_tqdm
                
        #self.delta_k_max = 3
        self.mask_ellmax = mask_ellmax
        self.pk_ellmax = pk_ellmax
        self.sample_mode = sample_mode
        self.lebedev_degree = lebedev_degree
        self.nthreads = nthreads if nthreads is not None else int(os.environ.get('OMP_NUM_THREADS', os.cpu_count()))

        self._init_randoms(randoms_a, alpha_a, randoms_b, alpha_b, randoms_c, alpha_c, randoms_d, alpha_d)
        self._init_survey_windows(nmesh=nmesh, boxsize=boxsize, boxpad=boxpad, kmin=kmin, kmax=kmax, dk=dk)
    
    def _init_randoms(self, randoms_a, alpha_a,
                            randoms_b, alpha_b,
                            randoms_c, alpha_c,
                            randoms_d, alpha_d):
        
        self.randoms = {}
        self.alphas = {}
        
        self.randoms['A'] = randoms_a
        self.alphas['A'] = alpha_a
        
        if randoms_b is not None and alpha_b is not None:
            self.randoms['B'] = randoms_b
            self.alphas['B'] = alpha_b
        
        if randoms_c is not None and alpha_c is not None:
            self.randoms['C'] = randoms_c
            self.alphas['C'] = alpha_c

        if randoms_d is not None and alpha_d is not None:
            self.randoms['D'] = randoms_d
            self.alphas['D'] = alpha_d

    def _init_survey_windows(self, **kwargs):
        
        if 'B' in self.randoms:
            self.window_AB = SurveyWindow(self.randoms['A'], self.alphas['A'], self.randoms['B'], self.alphas['B'], **kwargs)
        else:
            self.window_AB = SurveyWindow(self.randoms['A'], self.alphas['A'], None, None, **kwargs)
        if 'C' in self.randoms or 'D' in self.randoms:
            if 'D' in self.randoms:
                self.window_CD = SurveyWindow(self.randoms['C'], self.alphas['C'], self.randoms['D'], self.alphas['D'], **kwargs)
            else:
                self.window_CD = SurveyWindow(self.randoms['C'], self.alphas['C'], None, None, **kwargs)
        else:
            self.window_CD = self.window_AB
    
    @property
    def delta_k_max(self):
        return self.nmesh // 2 - 1

    @functools.cache
    def get_combined_survey_window(self, cache_dir=None):

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"W_ABCD.npz")

        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            window_ABCD = base.SparseNDArray(shape_out=(MASK_ELL_MAX//2+1,MASK_ELL_MAX//2+1,2*MASK_ELL_MAX+1,2*MASK_ELL_MAX+1),
                                            shape_in=(self.nmesh,self.nmesh,self.nmesh))

            for la, lb in itt.product(range(0, self.mask_ellmax+1, 2), repeat=2):
                for ma in range(-la, la+1):
                    for mb in range(-lb, lb+1):
                        window_ABCD[la//2,lb//2,ma+la,mb+lb] = self.window_AB.mesh(la, ma) * self.window_CD.mesh(lb, mb)

            window_ABCD.save(filename)
            return window_ABCD

    @property
    def get_window_kernels(self):
        return self.WinKernel
    
    @property
    def nmesh(self):
        return self.window_AB.knmesh
    
    @property
    def boxsize(self):
        return self.window_AB.kboxsize

    @property
    def get_num_tracers(self):
        return self.num_tracers

    @staticmethod
    def get_gaunt_coefficients(cache_dir=None, mask_ellmax=12, pk_ellmax=4):
        """Calculates all relavent Gaunt coefficients, or loads them from file"""

        # Load mask coupling Gaunt coefficients if cache exists, otherwise compute them
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"cosmic_variance_coefficients_{pk_ellmax}_{mask_ellmax}.npz")

        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            import sympy.physics.wigner

            # shape_out = l1, l2, l3, l4, m1, m2, m3, m4
            # shape_in =  la, lb, ma, mb
            # Only including positive m values, as -m is equivalent to m
            # when Ylm is real and m is even
            shape_out = 4*[PK_ELL_MAX//2 + 1] + 4*[2*PK_ELL_MAX + 1]
            shape_in = 2*[MASK_ELL_MAX//2 + 1] + 2*[2*MASK_ELL_MAX + 1]
            gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)

            for l1, l2, l3, l4 in itt.product(np.arange(0, pk_ellmax + 1, 2), repeat=4):
                for m1, m2, m3, m4 in itt.product(*[np.arange(-l, l+1, 2) for l in (l1, l2, l3, l4)]):
                    for la in np.arange(np.abs(l1-l4), l1+l4+1, 2):
                        for lb in np.arange(np.abs(l2-l3), l2+l3+1, 2):
                            for ma, mb in itt.product(*[np.arange(-l, l+1, 2) for l in (la, lb)]):

                                value = np.float64(sympy.physics.wigner.gaunt(l1,l4,la,m1,m4,ma)*\
                                                   sympy.physics.wigner.gaunt(l2,l3,lb,m2,m3,mb))
                                if value != 0.:
                                    # Taking absolute values of all m as -m is equivalent to m
                                    # when Ylm is real and m is even
                                    # m1, m2, m3, m4 = np.abs(m1), np.abs(m2), np.abs(m3), np.abs(m4)
                                    # ma, mb = np.abs(ma), np.abs(mb)
                                    gaunt_coefficients[l1//2,l2//2,
                                                       l3//2,l4//2,
                                                       m1+l1,m2+l2,
                                                       m3+l3,m4+l4,
                                                       la//2,lb//2,
                                                       ma+la,mb+lb] += value
                                    
                    for lc in np.arange(np.abs(l1-l2), l1+l2+1, 2):
                        for la in np.arange(np.abs(lc-l4), lc+l4+1, 2):
                            for ma, mc in itt.product(*[np.arange(-l, l+1, 2) for l in (la, lc)]):
                                value = np.float64(sympy.physics.wigner.gaunt(l1,l2,lc,m1,m2,mc)*\
                                                   sympy.physics.wigner.gaunt(lc,l4,la,mc,m4,ma))
                                lb, mb = l3, m3
                                if value != 0.:
                                    # Taking absolute values of all m as -m is equivalent to m
                                    # when Ylm is real and m is even
                                    # m1, m2, m3, m4 = np.abs(m1), np.abs(m2), np.abs(m3), np.abs(m4)
                                    # ma, mb = np.abs(ma), np.abs(mb)
                                    gaunt_coefficients[l1//2,l2//2,
                                                       l3//2,l4//2,
                                                       m1+l1,m2+l2,
                                                       m3+l3,m4+l4,
                                                       la//2,lb//2,
                                                       ma+la,mb+lb] += value
            gaunt_coefficients.save(filename)

        return gaunt_coefficients

    def clean(self):
        '''Clean window kernels and power spectra.'''
        self.WinKernel = None
        self.WinKernel_error = None
        self._window_power = None
        self._W = {}
        self._I = {}

    def compute_window_kernels(self):

        # points on the unit sphere with corresponding integration weights
        # x, y, z, w = math.get_lebedev_points(self.lebedev_degree)

        # Gaunt coefficients
        # cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/")
        # self.get_gaunt_coefficients(mask_ellmax=self.mask_ellmax, pk_ellmax=self.pk_ellmax)
        self.logger.info("Calculating or loading Gaunt coefficients...")
        self.get_gaunt_coefficients()

        # W_AB * W_CD (outer product)
        self.logger.info("Retrieving survey window outer product W_AB x W_CD...")
        W_ABCD = self.get_combined_survey_window()

        # create shared memory objects
        shm_data = multiprocessing.shared_memory.SharedMemory(create=True, size=W_ABCD._matrix.data.nbytes*2)
        shm_indices = multiprocessing.shared_memory.SharedMemory(create=True, size=W_ABCD._matrix.indices.nbytes)
        shm_indptr = multiprocessing.shared_memory.SharedMemory(create=True, size=W_ABCD._matrix.indptr.nbytes)

        # create views to shared memory
        data_shared = np.ndarray(W_ABCD._matrix.data.shape, dtype=W_ABCD._matrix.data.dtype, buffer=shm_data.buf)
        indices_shared = np.ndarray(W_ABCD._matrix.indices.shape, dtype=W_ABCD._matrix.indices.dtype, buffer=shm_indices.buf)
        indptr_shared = np.ndarray(W_ABCD._matrix.indptr.shape, dtype=W_ABCD._matrix.indptr.dtype, buffer=shm_indptr.buf)

        # copy data to shared memory objects
        # NOTE: This operation duplicates W_ABCD temporarily, which might become a problem for large nmesh
        np.copyto(data_shared, W_ABCD._matrix.data)
        np.copyto(indices_shared, W_ABCD._matrix.indices)
        np.copyto(indptr_shared, W_ABCD._matrix.indptr)
        # del W_ABCD # <- W_ABCD now lives in shared memory, so delete old one

        init_params = {
            'kfun':  2 * np.pi / self.boxsize,
            'dk': self.dk,
            'ikgrid': self.window_AB.ikgrid,
            'delta_k_max': self.delta_k_max,
            'mask_ellmax': self.mask_ellmax,
            'pk_ellmax': self.pk_ellmax,
            'sparse_shape': [W_ABCD._matrix.data.shape,
                             W_ABCD._matrix.indices.shape,
                             W_ABCD._matrix.indptr.shape,
                             W_ABCD.shape_in,
                             W_ABCD.shape_out],
        }

        # HYBRID SAMPLING
        kmodes_sampled = 1000
        kmodes, Nmodes, weights = math.sample_kmodes(kmin=self.kmin,
                                                     kmax=self.kmax,
                                                     dk=self.dk,
                                                     boxsize=self.boxsize,
                                                     max_modes=kmodes_sampled,
                                                     k_shell_approx=0.1,
                                                     sample_mode="monte-carlo")

        def init_worker(data_name, indices_name, indptr_name, init_params):
            global shared_params

            # buffers for W_ABCD
            global shared_data
            global shared_indices
            global shared_indptr

            shared_data = multiprocessing.shared_memory.SharedMemory(name=data_name)
            shared_indices = multiprocessing.shared_memory.SharedMemory(name=indices_name)
            shared_indptr = multiprocessing.shared_memory.SharedMemory(name=indptr_name)

            shared_params = init_params

        #delta_k_max = 3
        delta_k_max = self.nmesh // 2 - 1

        if not hasattr(self, 'WinKernel') or self.WinKernel is None:
            # Format is [k1_bins, k2_bins, l1, l2, l3, l4]
            self.WinKernel = np.empty([self.kbins, 2*delta_k_max+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1])
            self.WinKernel.fill(np.nan)

        #ell_factor = lambda l1,l2: (2*l1 + 1) * (2*l2 + 1) * (2 if 0 in (l1, l2) else 1)
        last_save = time.time()
        self.logger.info(f"Beginning window kernel calculations with {self.nthreads} threads...")
        for i, km in self.tqdm(enumerate(kmodes), desc='Computing window kernels', total=self.kbins):

            if hasattr(self, '_resume_file') and self._resume_file is not None:
                # Skip rows that were already computed
                if not np.isnan(self.WinKernel[i,0,0,0,0,0]):
                    # self.logger.debug(f'Skipping bin {i} of {self.kbins}.')
                    continue

            init_params['k1_bin_index'] = i + self.kmin//self.dk
            kmodes_sampled = len(km)

            # Splitting kmodes in chunks to be sent to each worker
            chunks = np.array_split(km, self.nthreads)

            with mp.Pool(processes=min(self.nthreads, len(chunks)),
                         initializer=init_worker,
                         initargs=[shm_data.name,
                                   shm_indices.name,
                                   shm_indptr.name,
                                   init_params]) as pool:
                
                results = pool.map(self._compute_window_kernel_row, chunks)
                self.WinKernel[i] = np.sum(results, axis=0) * weights[i] / kmodes_sampled

                # std_results = np.std(results * weights, axis=0) / np.sqrt(len(results))
                # avg_results = np.average(results, weights=weights, axis=0)
                # avg_results[std_results == 0] = 1
                # self.WinKernel_error[i] =  std_results / avg_results
        
                for k2_bin_index in range(0, 2*delta_k_max + 1):
                    if (k2_bin_index + i - delta_k_max >= self.kbins or k2_bin_index + i - delta_k_max < 0):
                        self.WinKernel[i, k2_bin_index, :, :] = 0
                    else:
                        self.WinKernel[i, k2_bin_index, :, :] /= Nmodes[i + k2_bin_index - self.delta_k_max]

            if hasattr(self, '_resume_file') and self._resume_file is not None and (time.time() - last_save) > 600:
                self.save(self._resume_file)
                last_save = time.time()

        self.logger.info('Window kernels computed.')

        shm_data.close()
        shm_data.unlink()
        shm_indices.close()
        shm_indices.unlink()
        shm_indptr.close()
        shm_indptr.unlink()

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

        data = np.ndarray(shared_params['sparse_shape'][0], dtype=np.complex128, buffer=shared_data.buf)
        indices = np.ndarray(shared_params['sparse_shape'][1], dtype=np.int32, buffer=shared_indices.buf)
        indptr = np.ndarray(shared_params['sparse_shape'][2], dtype=np.int32, buffer=shared_indptr.buf)

        W_ABCD = base.SparseNDArray.from_arrays(data, indices, indptr,
                                                shape_in=shared_params['sparse_shape'][3],
                                                shape_out=shared_params['sparse_shape'][4])
        
        # k1_bin_index is a scalar
        k1_bin_index = shared_params['k1_bin_index']
        kfun = shared_params['kfun']
        dk = shared_params['dk']
        pk_ellmax = shared_params['pk_ellmax']
        mask_ellmax = shared_params['mask_ellmax']

        G = SurveyGeometry.get_gaunt_coefficients(mask_ellmax=mask_ellmax,
                                                  pk_ellmax=pk_ellmax)

        # The Gaussian covariance drops quickly away from diagonal.
        # Only delta_k_max points to each side of the diagonal are calculated.
        delta_k_max = shared_params['delta_k_max']

        WinKernel = np.zeros((2*delta_k_max+1, pk_ellmax//2+1, pk_ellmax//2+1, pk_ellmax//2+1, pk_ellmax//2+1), dtype=np.complex128)
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
            # k2_bin_index has shape (nmesh, nmesh, nmesh)
            k2_bin_index = (k2r * kfun / dk).astype(int)
            k2r[k2r <= 1e-10] = np.inf
            k2xh /= k2r
            k2yh /= k2r
            k2zh /= k2r
            
            # multiply by Gaunt factors
            # give 3x3x3x3x9x9x9x9 x nmesh x nmesh x nmesh
            product = G @ W_ABCD
            result = np.zeros((list(product.shape_in) + [3,3,3,3]), dtype=np.complex128)

            # multiply by Ylms
            for l1, l2, l3, l4 in itt.product(np.arange(0, pk_ellmax+1, 2), repeat=4):
                l1_idx = int(l1 / 2)
                l2_idx = int(l2 / 2)
                l3_idx = int(l3 / 2)
                l4_idx = int(l4 / 2)
                
                for m1, m2, m3, m4 in itt.product(*[np.arange(-l, l+1, 2) for l in (l1, l2, l3, l4)]):
                    m1_idx = int((m1 + l1) / 2)
                    m2_idx = int((m2 + l2) / 2)
                    m3_idx = int((m3 + l3) / 2)
                    m4_idx = int((m4 + l4) / 2)

                    W_times_G = product[l1_idx,l2_idx,l3_idx,l4_idx,m1_idx,m2_idx,m3_idx,m4_idx]

                    Ylms = math.get_real_Ylm(l1, m1)(k1xh, k1yh, k1zh) * \
                           math.get_real_Ylm(l2, m2)(k2xh, k2yh, k2zh) * \
                           math.get_real_Ylm(l3, m3)(k1xh, k1yh, k1zh) * \
                           math.get_real_Ylm(l4, m4)(k2xh, k2yh, k2zh)
                    if not isinstance(Ylms, float):
                        Ylms = np.array(Ylms)
                    
                    result[:,:,:,l1_idx,l2_idx,l3_idx,l4_idx] += Ylms * W_times_G.toarray().reshape(product.shape_in)

            for delta_k in range(-delta_k_max, delta_k_max + 1):
                modes = (k2_bin_index - k1_bin_index == delta_k)
                if np.any(modes == True):
                    WinKernel[delta_k] = np.sum(result[modes], axis=0)

        return WinKernel