"""Module containing classes that represent the geometry to be used in the covariance calculation.

Classes
-------
Geometry
    Abstract class that defines the interface for the geometry classes.
BoxGeometry
    Class that represents the geometry of a periodic cubic box.
SurveyGeometry
    Class that represents the geometry of a survey in cut-sky.
"""

import os, time
import itertools as itt
import multiprocessing as mp
import logging
import functools

import numpy as np
import scipy as sp

from tqdm import tqdm as shell_tqdm

import mockfactory
from pypower import CatalogMesh

from . import base, utils, math

__all__ = ['BoxGeometry',
           'SurveyGeometry']

MASK_ELL_MAX = 12
PK_ELL_MAX = 4

class Geometry(base.BaseClass):
    pass

class Catalog:

    def __init__(self, randoms, alpha, nmesh=None, boxsize=None, boxpad=2., **kwargs):

        self.logger = logging.getLogger('SurveyGeometry')
        self.tqdm = shell_tqdm

        self._alpha = alpha

        self._randoms = randoms

        if not isinstance(randoms, mockfactory.Catalog):
            randoms = mockfactory.Catalog(randoms)

        # Check if the randoms have weights, otherwise set them to 1
        if 'WEIGHT' not in randoms:
            self.logger.warning(f'WEIGHT column not found in randoms. Setting it to 1.')
            self._randoms['WEIGHT'] = np.ones(self._randoms.size, dtype='f8')

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
            self.logger.warning(f'fsky = {fsky:.3f}')
            self.logger.info(f'fsky estimated from randoms: {fsky:.3f}')
            nbar = mockfactory.RedshiftDensityInterpolator(z=distance, fsky=fsky)
            self._randoms['NZ'] = alpha * nbar(distance)

        self._mesh = CatalogMesh(data_positions=self._randoms['POSITION'], data_weights=self._randoms['WEIGHT']*alpha,
                                 position_type='pos', nmesh=nmesh, boxsize=boxsize, boxpad=boxpad, dtype='c16',
                                 **{'interlacing': 3, 'resampler': 'tsc', **kwargs})
        
        self.logger.info(f'Loaded catalog with {self._mesh.boxsize}, box center {self._mesh.boxcenter}.')
        self.boxsize = self._mesh.boxsize[0]
        self.nmesh = self._mesh.nmesh[0]
        assert np.allclose(self._mesh.boxsize, self.boxsize) and np.all(self._mesh.nmesh == self.nmesh)

        self.logger.info(f'Average of {self._mesh.data_size / self.nmesh**3} objects per voxel.')

class SurveyGeometry(Geometry, base.LinearBinning):

    def __init__(self, catalog_a, catalog_b=None, kmax=0.02, ellmax=4, kmodes_sampled=2000, **kwargs):

        base.LinearBinning.__init__(self)

        self.logger = logging.getLogger('SurveyGeometry')
        self.tqdm = shell_tqdm

        self._kmax = kmax
        self._ellmax = ellmax
        self.kmodes_sampled = kmodes_sampled


        self.window_matrix = None

        self._resume_file = None
        # Determine number of threads to use
        if 'nthreads' in kwargs:
            self.nthreads = kwargs.pop('nthreads')
        else:
            self.nthreads = int(os.environ.get('OMP_NUM_THREADS', mp.cpu_count()))

        self.catalog_a = catalog_a
        if catalog_b is not None:
            self.catalog_b = catalog_b
        else:
            self.catalog_b = catalog_a

        self.boxsize = max((self.catalog_a._mesh.boxsize[0], self.catalog_b._mesh.boxsize[0]))
        self.nmesh = max((self.catalog_a._mesh.nmesh[0], self.catalog_b._mesh.nmesh[0]))

        catalog_a._mesh = catalog_a._mesh.clone(boxsize=self.boxsize, nmesh=self.nmesh)
        catalog_b._mesh = catalog_b._mesh.clone(boxsize=self.boxsize, nmesh=self.nmesh)

        self.logger.info(f'Using box size {self.boxsize}, box center {self._mesh.boxcenter} and nmesh {self._mesh.nmesh}.')
        self.boxsize = self._mesh.boxsize[0]
        self.nmesh = self._mesh.nmesh[0]
        assert np.allclose(self._mesh.boxsize, self.boxsize) and np.all(self._mesh.nmesh == self.nmesh)

        self.logger.info(f'Fundamental wavenumber of window FFTs = {self.kfun}.')
        self.logger.info(f'Nyquist wavenumber of window FFTs = {self.knyquist}.')

        if kmax is not None and self.knyquist < kmax:
            self.logger.warning(f'Nyquist wavelength {self.knyquist} smaller than required window kmax = {kmax}.')

    @functools.lru_cache    
    def compute_mesh(self, nbar_power, weight_power, ell, m, threshold=None):
        """Compute the Fourier transform of nbar**nbar_power * weight**weight_power * Ylm

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
            Resulting mesh after computation.
        """

        assert ell >= 0, "ell must be non-negative"
        assert abs(m) <= ell, "m must be less than or equal to ell"

        # can probably be made faster by properly vectorizing it
        Ylm = np.vectorize(math.get_real_Ylm(ell, m))

        self.logger.info(f'Computing mesh nbar^{nbar_power} * weight^{weight_power} (ell={ell}, m={m})')

        result = self._mesh.copy(
            data_positions=self._randoms['POSITION'],
            data_weights=self._randoms['NZ']**(nbar_power-1)*self._randoms['WEIGHT']**(weight_power) * self.alpha * Ylm(*self._randoms['POSITION'].T),
            position_type='pos',
            ).to_mesh(compensate=True).r2c().value * self.nmesh**3

        if threshold is not None:
            # Convert the result to a sparse array to save memory
            result[np.abs(result) < threshold] = 0
            result = base.SparseNDArray.from_dense(result, shape_in=(self.nmesh,self.nmesh), shape_out=self.nmesh)
        
        return result

    def compute_window_matrix(self, pk_ellmax = PK_ELL_MAX, mask_ellmax = MASK_ELL_MAX):
        '''Computes the window matrix to be used in the calculation of the covariance.

        Notes
        -----
        The window matrices are computed using the method described in [1]_.

        References
        ----------
        .. [1] https://arxiv.org/abs/1910.02914
        '''

        self.logger.info('='*60)
        self.logger.info('Computing window matrices')
        self.logger.info(f'pk_ellmax={pk_ellmax}, mask_ellmax={mask_ellmax}')
        self.logger.info('='*60)

        # sample kmodes from each k1 bin

        # SAMPLE FROM SHELL
        # kfun = 2 * np.pi / self.boxsize
        # kmodes = np.array([[math.sample_from_shell(kmin/kfun, kmax/kfun) for _ in range(
        #                    self.kmodes_sampled)] for kmin, kmax in zip(self.kedges[:-1], self.kedges[1:])])
        # Nmodes = math.nmodes(self.boxsize**3, self.kedges[:-1], self.kedges[1:])

        # SAMPLE FROM CUBE
        # kmodes, Nmodes = math.sample_from_cube(self.kmax/kfun, self.dk/kfun, self.kmodes_sampled)

        # HYBRID SAMPLING
        self.logger.info('Sampling k-modes for binning...')
        kmodes, Nmodes =  math.sample_kmodes(kmin=self.kmin,
                                             kmax=self.kmax,
                                             dk=self.dk,
                                             boxsize=self.boxsize,
                                             max_modes=self.kmodes_sampled,
                                             k_shell_approx=0.1)
        self.logger.info(f'Sampled k-modes for {self.kbins} bins')

        assert len(kmodes) == self.kbins and len(Nmodes) == self.kbins, \
            f'Error in thecov.utils.sample_kmodes: results should have length {self.kbins}, but had {len(kmodes)}. Parameters were kmin={self.kmin},kmax={self.kmax},dk={self.dk},boxsize={self.boxsize},max_modes={self.kmodes_sampled},k_shell_approx={0.1}).'

        init_params = {
            'boxsize':     self.boxsize,
            'dk':          self.dk,
            'nbins':       self.kbins,
            'nmesh':       self.nmesh,
            'ikgrid':      self.ikgrid,
            'delta_k_max': self.delta_k_max,
            'pk_ellmax':   pk_ellmax
        }

        windows_cosmicvar = base.SparseNDArray(shape_out=2*[mask_ellmax//2+1] + 2*[2*mask_ellmax+1], shape_in=3*[self.nmesh], dtype=np.complex128)
        windows_mixed     = base.SparseNDArray(shape_out=2*[mask_ellmax//2+1] + 2*[2*mask_ellmax+1], shape_in=3*[self.nmesh], dtype=np.complex128)
        windows_shotnoise = base.SparseNDArray(shape_out=2*[mask_ellmax//2+1] + 2*[2*mask_ellmax+1], shape_in=3*[self.nmesh], dtype=np.complex128)

        self.logger.info('Computing window function multipoles...')
        for la, lb, ma, mb in utils.ellmiter(mask_ellmax, 2):
            windows_cosmicvar[la//2,lb//2,ma+la,mb+lb] = self.compute_mesh(2,2,la,ma) * self.compute_mesh(2,2,lb,mb)
            windows_mixed[la//2,lb//2,ma+la,mb+lb]     = self.compute_mesh(2,2,la,ma) * self.compute_mesh(1,2,lb,mb)
            windows_shotnoise[la//2,lb//2,ma+la,mb+lb] = self.compute_mesh(1,2,la,ma) * self.compute_mesh(1,2,lb,mb)

        self.logger.info('Contracting Gaunt coefficients with window meshes...')
        # Create shared memory for the sparse arrays
        sparse_arrays = {
            'first_cosmic_variance':  self.get_first_cosmic_variance_gaunt_coefficients(mask_ellmax, pk_ellmax)  @ windows_cosmicvar,
            'second_cosmic_variance': self.get_second_cosmic_variance_gaunt_coefficients(mask_ellmax, pk_ellmax) @ windows_cosmicvar,
            'mixed_term':             self.get_mixed_gaunt_coefficients(mask_ellmax, pk_ellmax)     @ windows_mixed,
            'shotnoise':              self.get_shotnoise_gaunt_coefficients(mask_ellmax, pk_ellmax) @ windows_shotnoise,
        }
        
        self.logger.info('Creating shared memory blocks for multiprocessing...')
        shared_memory_blocks = []
        sparse_metadata = {}
        
        for name, sparse_arr in sparse_arrays.items():
            matrix = sparse_arr._matrix
            
            # Create shared memory for data, indices, indptr
            from multiprocessing import shared_memory as shm
            
            shm_data = shm.SharedMemory(create=True, size=matrix.data.nbytes)
            shm_indices = shm.SharedMemory(create=True, size=matrix.indices.nbytes)
            shm_indptr = shm.SharedMemory(create=True, size=matrix.indptr.nbytes)
            
            # Copy data to shared memory
            np.ndarray(matrix.data.shape, dtype=matrix.data.dtype, buffer=shm_data.buf)[:] = matrix.data
            np.ndarray(matrix.indices.shape, dtype=matrix.indices.dtype, buffer=shm_indices.buf)[:] = matrix.indices
            np.ndarray(matrix.indptr.shape, dtype=matrix.indptr.dtype, buffer=shm_indptr.buf)[:] = matrix.indptr
            
            shared_memory_blocks.extend([shm_data, shm_indices, shm_indptr])
            
            sparse_metadata[name] = {
                'shape_out': sparse_arr.shape_out.tolist(),
                'shape_in': sparse_arr.shape_in.tolist(),
                'matrix_shape': matrix.shape,
                'data_shape': matrix.data.shape,
                'data_dtype': str(matrix.data.dtype),
                'indices_shape': matrix.indices.shape,
                'indices_dtype': str(matrix.indices.dtype),
                'indptr_shape': matrix.indptr.shape,
                'indptr_dtype': str(matrix.indptr.dtype),
                'shm_data_name': shm_data.name,
                'shm_indices_name': shm_indices.name,
                'shm_indptr_name': shm_indptr.name,
            }

        init_params['sparse_metadata'] = sparse_metadata

        if self.window_matrix is None:
            self.window_matrix = {}
            
            self.window_matrix['cosmic_variance'] = np.empty(4*[pk_ellmax//2] + 2*[self.kbins])
            self.window_matrix['cosmic_variance'].fill(np.nan)

            self.window_matrix['mixed_term'] = np.empty(3*[pk_ellmax//2] + 2*[self.kbins])
            self.window_matrix['mixed_term'].fill(np.nan)

            self.window_matrix['shotnoise'] = np.empty(2*[pk_ellmax//2] + 2*[self.kbins])
            self.window_matrix['shotnoise'].fill(np.nan)

        last_save = time.time()

        self.logger.info(f'Starting parallel computation with {self.nthreads} threads...')
        try:
            for i, km in self.tqdm(enumerate(kmodes), desc='Computing window matrix', total=self.kbins):

                if self._resume_file is not None and 'shotnoise' in self.window_matrix:
                    # Skip rows that were already computed
                    if not np.isnan(self.window_matrix['shotnoise'][0,0,i,i]).all():
                        # self.logger.debug(f'Skipping bin {i} of {self.kbins}.')
                        continue

                init_params['k1_bin_index'] = i + self.kmin//self.dk

                # Splitting kmodes in chunks to be sent to each worker
                chunks = np.array_split(km, self.nthreads)

                with mp.Pool(processes=min(self.nthreads, len(chunks)),
                             initializer=_init_sparse_worker,
                             initargs=(init_params,)) as pool:
                    
                    results = pool.map(self._compute_window_matrix_row, chunks)

                    self.window_matrix['cosmic_variance'][:,:,:,:,i,:] = \
                        (4*np.pi)**2 / self.normalization(2,2)**2 * \
                            np.array([r[0] for r in results]).transpose(axes=(1,2,3,4,0,5))/Nmodes[None,None,None,None,:]

                    self.window_matrix['mixed_term'][:,:,:,i,:] = \
                        (4*np.pi)**2 / self.normalization(2,2)**2 * \
                            np.array([r[1] for r in results]).transpose(axes=(1,2,3,0,4))/Nmodes[None,None,None,:]
                    
                    self.window_matrix['shotnoise'][:,:,i,:] = \
                        (4*np.pi)**2 / self.normalization(2,2)**2 * \
                            np.array([r[2] for r in results]).transpose(axes=(1,2,0,3))/Nmodes[None,None,:]

                if self._resume_file is not None and not self._resume_file_readonly and (time.time() - last_save) > 600:
                    self.logger.info(f'Auto-saving intermediate results (bin {i+1}/{self.kbins})...')
                    self.save(self._resume_file)
                    last_save = time.time()
                    
            self.logger.info('Window matrix computation completed successfully!')

            if self._resume_file is not None and not self._resume_file_readonly:
                self.logger.info(f'Saving final results to {self._resume_file}')
                self.save(self._resume_file)
                
        finally:
            # Clean up shared memory
            self.logger.info('Cleaning up shared memory...')
            for shm_block in shared_memory_blocks:
                shm_block.close()
                shm_block.unlink()
            self.logger.info('Shared memory cleanup complete.')

    @staticmethod
    def _compute_window_matrix_row(bin_kmodes):
        '''Computes a row of the window kernels. This function is called in parallel for each k1 bin.'''
        # Gives window kernels for L=0,2,4 auto and cross covariance (instead of only L=0 above)

        # Returns an array with [2*delta_k_max+1,15,6] dimensions.
        #    The first dim corresponds to the k-bin of k2
        #    (only 3 bins on each side of the diagonal are calculated as the Gaussian covariance drops quickly away from diagonal)

        #    The second dim corresponds to elements to be multiplied by various power spectrum multipoles
        #    to obtain the final covariance (see function 'Wij' below)

        #    The last dim corresponds to multipoles: [L0xL0,L2xL2,L4xL4,L2xL0,L4xL0,L4xL2]

        boxsize = shared_params['boxsize']
        kfun = 2 * np.pi / boxsize
        dk = shared_params['dk']
        pk_ellmax = shared_params['pk_ellmax']
        nmesh = shared_params['nmesh']
        nbins = shared_params['nbins']

        # Access the shared sparse arrays
        first_cosmic_variance_sparse  = shared_sparse['first_cosmic_variance']
        second_cosmic_variance_sparse = shared_sparse['second_cosmic_variance']
        mixed_term_sparse             = shared_sparse['mixed_term']
        shotnoise_term_sparse         = shared_sparse['shotnoise']

        delta_ik = np.array(np.meshgrid(*shared_params['ikgrid'], indexing='ij'))

        window_matrix = [0,0,0]

        for ik1x, ik1y, ik1z, ik1r in bin_kmodes:

            ik1 = np.array([ik1x, ik1y, ik1z])
            ik2 = ik1[:,None,None,None] + delta_ik

            k2_bin_index = (np.sqrt(np.sum(ik2**2, axis=0)) * kfun / dk).astype(int)

            Ylm = {}
            for l, m in utils.ellmiter(pk_ellmax, 1):
                Ylm[l,m,1] = np.vectorize(math.get_real_Ylm(l,m))(*ik1)
                Ylm[l,m,2] = np.vectorize(math.get_real_Ylm(l,m))(*ik2)

            cosmic_variance = np.zeros(4*[pk_ellmax//2+1] + 3*[nmesh])
            mixed_term      = np.zeros(3*[pk_ellmax//2+1] + 3*[nmesh])
            shotnoise_term  = np.zeros(2*[pk_ellmax//2+1] + 3*[nmesh])

            for l1, l2, l3, l4, m1, m2, m3, m4 in utils.ellmiter(pk_ellmax, 4):
                cosmic_variance[l1//2,l2//2,l3//2,l4//2] += \
                    first_cosmic_variance_sparse[l1//2,l2//2,l3//2,l4//2,m1+l1,m2+l2,m3+l3,m4+l4].real * \
                    Ylm[l1,m1,1]*Ylm[l2,m2,1]*Ylm[l3,m3,2]*Ylm[l4,m4,2]
                
                cosmic_variance[l1//2,l2//2,l3//2,l4//2] += \
                    second_cosmic_variance_sparse[l1//2,l2//2,l3//2,l4//2,m1+l1,m2+l2,m3+l3,m4+l4].real * \
                    Ylm[l1,m1,1]*Ylm[l2,m2,2]*Ylm[l3,m3,1]*Ylm[l4,m4,2]
                    
            for l1, l2, l3, m1, m2, m3 in utils.ellmiter(pk_ellmax, 3):
                mixed_term[l1//2,l2//2,l3//2] += \
                    mixed_term_sparse[l1//2,l2//2,l3//2,m1+l1,m2+l2,m3+l3].real * \
                    Ylm[l1,m1,1]*Ylm[l2,m2,2]*Ylm[l3,m3,2]

            for l1, l2, m1, m2 in utils.ellmiter(pk_ellmax, 2):
                shotnoise_term[l1//2,l2//2] += \
                    shotnoise_term_sparse[l1//2,l2//2,m1+l1,m2+l2].real * \
                    Ylm[l1,m1,1]*Ylm[l2,m2,2]

            window_matrix[0] += np.array([
                np.bincount(
                    k2_bin_index.ravel(),
                    weights=cosmic_variance[*np.array(ls)//2].ravel(),
                    minlength=nbins
                ) for ls in utils.elliter(pk_ellmax, 4)]
            ).reshape(4*[pk_ellmax//2+1] + [nbins])/len(bin_kmodes)

            window_matrix[1] += np.array([
                np.bincount(
                    k2_bin_index.ravel(),
                    weights=mixed_term[*np.array(ls)//2].ravel(),
                    minlength=nbins
                ) for ls in utils.elliter(pk_ellmax, 3)]
            ).reshape(3*[pk_ellmax//2+1] + [nbins])/len(bin_kmodes)

            window_matrix[2] += np.array([
                np.bincount(
                    k2_bin_index.ravel(),
                    weights=shotnoise_term[*np.array(ls)//2].ravel(),
                    minlength=nbins
                ) for ls in utils.elliter(pk_ellmax, 2)]
            ).reshape(2*[pk_ellmax//2+1] + [nbins])/len(bin_kmodes)

        return window_matrix

    @staticmethod
    def get_first_cosmic_variance_gaunt_coefficients(mask_ellmax=MASK_ELL_MAX, pk_ellmax=PK_ELL_MAX, cache_dir=None):
        """Calculates all relavent Gaunt coefficients for the cosmic variance term, or loads them from file"""

        # Load mask coupling Gaunt coefficients if cache exists, otherwise compute them
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"first_cosmic_variance_coefficients_{pk_ellmax:d}_{mask_ellmax:d}.npz")

        logger = logging.getLogger('SurveyGeometry')

        if os.path.exists(filename):
            logger.info(f'Loading first cosmic variance Gaunt coefficients from cache: {filename}')
            return base.SparseNDArray.load(filename)
        else:
            logger.info(f'Computing first cosmic variance Gaunt coefficients (pk_ellmax={pk_ellmax}, mask_ellmax={mask_ellmax})...')
            import sympy.physics.wigner

            # shape_out = l1, l2, l3, l4, m1, m2, m3, m4
            # shape_in =  la, lb, ma, mb
            shape_out = 4*[pk_ellmax//2 + 1] + 4*[2*pk_ellmax + 1]
            shape_in = 2*[mask_ellmax//2 + 1] + 2*[2*mask_ellmax + 1]
            gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)

            for l1, l2, l3, l4, m1, m2, m3, m4 in utils.ellmiter(pk_ellmax, 4):
                for la in np.arange(np.abs(l1-l4), min(l1+l4, mask_ellmax)+1, 2):
                    for lb in np.arange(np.abs(l2-l3), min(l2+l3, mask_ellmax)+1, 2):
                        for ma, mb in itt.product(*[np.arange(-l, l+1, 2) for l in (la, lb)]):

                            value = np.float64(sympy.physics.wigner.gaunt(l1,l4,la,m1,m4,ma)*\
                                                sympy.physics.wigner.gaunt(l2,l3,lb,m2,m3,mb))
                            if value != 0.:
                                gaunt_coefficients[l1//2,l2//2,
                                                    l3//2,l4//2,
                                                    m1+l1,m2+l2,
                                                    m3+l3,m4+l4,
                                                    la//2,lb//2,
                                                    ma+la,mb+lb] += value

            
            logger.info(f'Computed {gaunt_coefficients._matrix.nnz} non-zero Gaunt coefficients')
            logger.info(f'Saving first cosmic variance Gaunt coefficients to: {filename}')
            gaunt_coefficients.save(filename)
            return gaunt_coefficients

    @staticmethod
    def get_second_cosmic_variance_gaunt_coefficients(mask_ellmax=MASK_ELL_MAX, pk_ellmax=PK_ELL_MAX, cache_dir=None):
        """Calculates all relavent Gaunt coefficients for the cosmic variance term, or loads them from file"""
        
        # Load mask coupling Gaunt coefficients if cache exists, otherwise compute them
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"second_cosmic_variance_coefficients_{pk_ellmax:d}_{mask_ellmax:d}.npz")

        logger = logging.getLogger('SurveyGeometry')

        if os.path.exists(filename):
            logger.info(f'Loading second cosmic variance Gaunt coefficients from cache: {filename}')
            return base.SparseNDArray.load(filename)
        else:
            logger.info(f'Computing second cosmic variance Gaunt coefficients (pk_ellmax={pk_ellmax}, mask_ellmax={mask_ellmax})...')
            import sympy.physics.wigner

            # shape_out = l1, l2, l3, l4, m1, m2, m3, m4
            # shape_in =  la, lb, ma, mb  (a for W22 and b for W12)
            shape_out = 4*[pk_ellmax//2 + 1] + 4*[2*pk_ellmax + 1]
            shape_in = 2*[mask_ellmax//2 + 1] + 2*[2*mask_ellmax + 1]
            gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)

            for l1, l2, l3, l4, m1, m2, m3, m4 in utils.ellmiter(pk_ellmax, 4):
                for lc in np.arange(np.abs(l1-l2), min(l1+l2, mask_ellmax)+1, 2):
                    for la in np.arange(np.abs(lc-l4), min(lc+l4, mask_ellmax)+1, 2):
                        for ma, mc in itt.product(*[np.arange(-l, l+1, 2) for l in (la, lc)]):
                            value = np.float64(sympy.physics.wigner.gaunt(l1,l2,lc,m1,m2,mc)*\
                                                sympy.physics.wigner.gaunt(lc,l4,la,mc,m4,ma))
                            lb, mb = l3, m3
                            if value != 0.:
                                gaunt_coefficients[l1//2,l2//2,
                                                    l3//2,l4//2,
                                                    m1+l1,m2+l2,
                                                    m3+l3,m4+l4,
                                                    la//2,lb//2,
                                                    ma+la,mb+lb] += value
            logger.info(f'Computed {gaunt_coefficients._matrix.nnz} non-zero Gaunt coefficients')
            logger.info(f'Saving second cosmic variance Gaunt coefficients to: {filename}')
            gaunt_coefficients.save(filename)
            return gaunt_coefficients

    @staticmethod
    def get_mixed_gaunt_coefficients(mask_ellmax=MASK_ELL_MAX, pk_ellmax=PK_ELL_MAX, cache_dir=None):
        """Calculates all relavent Gaunt coefficients for the shotnoise term, or loads them from file"""
        
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"mixed_coefficients_{pk_ellmax:d}_{mask_ellmax:d}.npz")

        logger = logging.getLogger('SurveyGeometry')

        if os.path.exists(filename):
            logger.info(f'Loading mixed Gaunt coefficients from cache: {filename}')
            return base.SparseNDArray.load(filename)
        else:
            logger.info(f'Computing mixed Gaunt coefficients (pk_ellmax={pk_ellmax}, mask_ellmax={mask_ellmax})...')
            import sympy.physics.wigner

            # shape_out = l1, l2, l3, m1, m2, l3
            # shape_in =  la, ma, lb, mb
            # Only including positive m values, as -m is equivalent to m
            # when Ylm is real and m is even
            shape_out = 3*[pk_ellmax//2 + 1] + 3*[2*pk_ellmax + 1]
            shape_in = 2*[mask_ellmax//2 + 1] + 2*[2*mask_ellmax + 1]
            gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)

            for l1, l2, l3, m1, m2, m3 in utils.ellmiter(pk_ellmax, 3):

                lb, mb = l1, m1
                if lb <= mask_ellmax:
                    for la in np.arange(np.abs(l2-l3), min(l2+l3, mask_ellmax)+1, 2):
                        for ma in np.arange(-la, la+1, 2):
                            value = np.float64(sympy.physics.wigner.gaunt(l2,l3,la,m2,m3,ma))
                            if value != 0:
                                gaunt_coefficients[l1//2, l2//2, l3//2, m1+l1, m2+l2, m3+l3, la//2, lb//2, ma+la, mb+lb] += value

                lb, mb = l2, m2
                if lb <= mask_ellmax:
                    for la in np.arange(np.abs(l1-l3), min(l1+l3, mask_ellmax)+1, 2):
                        for ma in np.arange(-la, la+1, 2):
                            value = np.float64(sympy.physics.wigner.gaunt(l1,l3,la,m1,m3,ma))
                            if value != 0:
                                gaunt_coefficients[l1//2, l2//2, l3//2, m1+l1, m2+l2, m3+l3, la//2, lb//2, ma+la, mb+lb] += value

                la, ma = l3, m3
                if la <= mask_ellmax:
                    for lb in np.arange(np.abs(l1-l2), min(l1+l2, mask_ellmax)+1, 2):
                        for mb in np.arange(-lb, lb+1, 2):
                            
                            value = np.float64(sympy.physics.wigner.gaunt(l1,l2,lb,m1,m2,mb))
                            if value != 0:
                                gaunt_coefficients[l1//2, l2//2, l3//2, m1+l1, m2+l2, m3+l3, la//2, lb//2, ma+la, mb+lb] += value
                                
                lb, mb = 0,0
                for lc in np.arange(np.abs(l1-l2), min(l1+l2, mask_ellmax)+1, 2):
                    for la in range(np.abs(lc-l3), min(lc+l3, mask_ellmax)+1, 2):
                        for ma in np.arange(-la, la+1, 2):
                            for mc in range(-lc, lc+1, 2):
                                value = np.float64(sympy.physics.wigner.gaunt(l1,l2,lc,m1,m2,mc)*\
                                                   sympy.physics.wigner.gaunt(lc,l3,la,mc,m3,ma))
                                if value != 0:
                                    gaunt_coefficients[l1//2, l2//2, l3//2, m1+l1, m2+l2, m3+l3, la//2, lb//2, ma+la, mb+lb] += value
                                    
            logger.info(f'Computed {gaunt_coefficients._matrix.nnz} non-zero Gaunt coefficients')
            logger.info(f'Saving mixed Gaunt coefficients to: {filename}')
            gaunt_coefficients.save(filename)
            return gaunt_coefficients
        
    @staticmethod
    def get_shotnoise_gaunt_coefficients(mask_ellmax=MASK_ELL_MAX, pk_ellmax=PK_ELL_MAX, cache_dir=None):
        """Calculates all relavent Gaunt coefficients for the shotnoise term, or loads them from file"""
       
        logger = logging.getLogger('SurveyGeometry')

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"shotnoise_coefficients_{pk_ellmax:d}_{mask_ellmax:d}.npz")

        if os.path.exists(filename):
            logger.info(f'Loading shotnoise Gaunt coefficients from cache: {filename}')
            return base.SparseNDArray.load(filename)
        else:
            logger.info(f'Computing shotnoise Gaunt coefficients (pk_ellmax={pk_ellmax}, mask_ellmax={mask_ellmax})...')
            import sympy.physics.wigner

            # shape_out = l1, l2, m1, m2
            # shape_in =  la, ma
            shape_out = 2*[pk_ellmax//2 + 1] + 2*[2*pk_ellmax + 1]
            shape_in = 2*[mask_ellmax//2 + 1] + 2*[2*mask_ellmax + 1]
            gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)

            for l1, l2, m1, m2 in utils.ellmiter(pk_ellmax, 2):

                la, ma = l1,m1
                lb, mb = l2,m2
                gaunt_coefficients[l1//2, l2//2, m1+l1, m2+l2, la//2, lb//2, ma+la, mb+lb] += 1

                lb,mb = 0,0
                for la in range(np.abs(l1-l2), min(l1+l2+1, mask_ellmax), 2):
                    for ma in range(-la, la+1, 2):
                        value = np.float64(sympy.physics.wigner.gaunt(l1,l2,la,m1,m2,ma))
                        if value != 0:
                            gaunt_coefficients[l1//2, l2//2, m1+l1, m2+l2, la//2, lb//2, ma+la, mb+lb] += value

            logger.info(f'Computed {gaunt_coefficients._matrix.nnz} non-zero Gaunt coefficients')
            logger.info(f'Saving shotnoise Gaunt coefficients to: {filename}')
            gaunt_coefficients.save(filename)
            return gaunt_coefficients

    def normalization(self, nbar_power, weight_power):
        return (self._randoms['NZ']**(nbar_power-1) * \
                self._randoms['WEIGHT']**(weight_power) * \
                self.alpha).sum().tolist()

    @property
    def knyquist(self):
        return np.pi * self.nmesh / self.boxsize
    
    @property
    def kfun(self):
        return 2 * np.pi / self.boxsize
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def ikgrid(self):
        """Grid of wavenumber indices."""
        ikgrid = []
        for _ in range(3):
            iik = np.arange(self.nmesh)
            iik[iik >= self.nmesh // 2] -= self.nmesh
            ikgrid.append(iik)
        return ikgrid
    
    @property
    def delta_k_max(self):
        return self.nmesh // 2 - 1

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['logger', 'tqdm', '_randoms', '_mesh', '_resume_file']:
            state.pop(key, None)
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

    def clean(self):
        '''Clean window matrix.'''
        self.window_matrix = None
        self.window_matrix_error = None

# Module-level globals for multiprocessing workers
shared_params = None
shared_sparse = None

def _init_sparse_worker(params):
    """Initialize worker process with shared sparse arrays from shared memory."""
    global shared_params
    global shared_sparse
    
    from multiprocessing import shared_memory as shm
    import scipy.sparse
    
    shared_params = params
    shared_sparse = {}
    
    for name, meta in params['sparse_metadata'].items():
        # Attach to shared memory blocks
        shm_data = shm.SharedMemory(name=meta['shm_data_name'])
        shm_indices = shm.SharedMemory(name=meta['shm_indices_name'])
        shm_indptr = shm.SharedMemory(name=meta['shm_indptr_name'])
        
        # Create numpy arrays backed by shared memory
        data = np.ndarray(meta['data_shape'], dtype=np.dtype(meta['data_dtype']), buffer=shm_data.buf)
        indices = np.ndarray(meta['indices_shape'], dtype=np.dtype(meta['indices_dtype']), buffer=shm_indices.buf)
        indptr = np.ndarray(meta['indptr_shape'], dtype=np.dtype(meta['indptr_dtype']), buffer=shm_indptr.buf)
        
        # Reconstruct CSR matrix (uses the shared memory buffers directly, no copy)
        matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=meta['matrix_shape'])
        
        # Reconstruct SparseNDArray
        sparse_arr = base.SparseNDArray.__new__(base.SparseNDArray)
        sparse_arr.shape_out = np.array(meta['shape_out'])
        sparse_arr.shape_in = np.array(meta['shape_in'])
        sparse_arr._matrix = matrix
        
        shared_sparse[name] = sparse_arr