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
import logging
import warnings
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress JAX fork warning - we're using multiprocessing intentionally
warnings.filterwarnings("ignore", message="os.fork\\(\\) was called")

import numpy as np

import mockfactory
from pypower import CatalogMesh

from . import base, utils, math
from .monitor import ResourceMonitor, HAS_PSUTIL

__all__ = ['BoxGeometry',
           'SurveyGeometry']

MASK_ELL_MAX = 12
PK_ELL_MAX = 4

class Geometry(base.BaseClass):
    pass

class BoxGeometry(Geometry):
    '''Class that represents the geometry of a periodic cubic box.

    Attributes
    ----------
    boxsize : float
        Size of the box.
    nmesh : int
        Number of mesh points in each dimension.
    alpha : float
        <number of galaxies>/<number of randoms> in the box.

    Methods
    -------
    set_boxsize
        Set the size of the box.
    set_nmesh
        Set the number of mesh points in each dimension.
    set_alpha
        Set the alpha parameter.
    '''
    logger = logging.getLogger('BoxGeometry')

    def __init__(self, volume=None, nbar=None):
        self._volume = volume
        self._nbar = nbar
        self._zmin = None
        self._zmax = None
        self.fsky = 1.0

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, volume):
        self._volume = volume

    @property
    def nbar(self):
        return self._nbar

    @nbar.setter
    def nbar(self, nbar):
        self._nbar = nbar

    @property
    def shotnoise(self):
        '''Estimates the Poissonian shotnoise of the sample as 1/nbar.

        Returns
        -------
        float
            Poissonian shotnoise of the sample = 1/nbar.'''
        return 1. / self.nbar

    @shotnoise.setter
    def shotnoise(self, shotnoise):
        '''Sets the Poissonian shotnoise of the sample and nbar = 1/shotnoise.

        Parameters
        ----------
        shotnoise : float
            Shotnoise of the sample.'''
        self.nbar = 1. / shotnoise

    @property
    def zedges(self):
        return self._zedges

    @property
    def zmid(self):
        return (self.zedges[1:] + self.zedges[:-1])/2

    @property
    def zmin(self):
        return self._zmin if self._zmin is not None else self.zedges[0]

    @property
    def zmax(self):
        return self._zmax if self._zmax is not None else self.zedges[-1]

    @property
    def zavg(self):
        bin_volume = np.diff(self.cosmo.comoving_radial_distance(self.zedges)**3)
        return np.average(self.zmid, weights=self.nz * bin_volume)

    def set_effective_volume(self, zmin, zmax, fsky=None):
        '''Set the effective volume of the box based on the redshift limits of the sample and the fraction of the sky covered.

        Parameters
        ----------
        zmin : float
            Minimum redshift of the sample.
        zmax : float
            Maximum redshift of the sample.
        fsky : float, optional
            Fraction of the sky covered by the sample. If not given, the current value of fsky is used.

        Returns
        -------
        float
            Effective volume of the box.'''

        if fsky is not None:
            self.fsky = fsky

        self._zmin = zmin
        self._zmax = zmax

        self.volume = self.fsky * 4. / 3. * np.pi * \
            (self.cosmo.comoving_radial_distance(zmax)**3 -
             self.cosmo.comoving_radial_distance(zmin)**3)

        return self.volume

    def set_nz(self, zedges, nz, *args, **kwargs):
        '''Set the effective volume and number density of the box based on the
        redshift distribution of the sample.

        Parameters
        ----------
        zedges : array_like
            Array of redshift bin edges.
        nz : array_like
            Array of redshift distribution of the sample.
        *args, **kwargs
            Arguments and keyword arguments to be passed to set_effective_volume.
        '''
        assert len(zedges) == len(nz) + \
            1, "Length of zedges should equal length of nz + 1."

        self._zedges = np.array(zedges)
        self._nz = np.array(nz)[np.argsort(self.zmid)]
        self._zedges.sort()

        self.set_effective_volume(
            zmin=self.zmin, zmax=self.zmax, *args, **kwargs)
        self.logger.info(f'Effective volume: {self.volume:.3e} (Mpc/h)^3')

        bin_volume = self.fsky * \
            np.diff(self.cosmo.comoving_radial_distance(self.zedges)**3)
        self.nbar = np.average(self.nz, weights=bin_volume)
        self.logger.info(f'Estimated nbar: {self.nbar:.3e} (Mpc/h)^-3')

    def set_randoms(self, randoms, alpha=1.0, bins=None, fsky=None):
        '''Estimates the effective volume and number density of the box based on a
        provided catalog of randoms.

        Parameters
        ----------
        randoms : array_like
            Catalog of randoms.
        alpha : float, optional
            Factor to multiply the number density of the randoms. Default is 1.0.
        '''
        from mockfactory import RedshiftDensityInterpolator

        if fsky is None:
            import healpy as hp

            nside = 512
            hpixel = hp.ang2pix(nside, randoms['RA'], randoms['DEC'], lonlat=True)
            unique_hpixels = np.unique(hpixel)
            self.fsky = len(unique_hpixels) / hp.nside2npix(nside)

            self.logger.info(f'fsky estimated from randoms: {self.fsky:.3f}')
        else:
            self.fsky = fsky

        nz_hist = RedshiftDensityInterpolator(z=randoms['Z'], bins=bins, fsky=self.fsky, distance=self.cosmo.comoving_radial_distance)
        self.set_nz(zedges=nz_hist.edges, nz=nz_hist.nbar * alpha)

    @property
    def area(self):
        return self.fsky * 360**2 / np.pi

    @area.setter
    def area(self, area):
        self.fsky = area / 360**2 * np.pi

    @property
    def nz(self):
        return self._nz

    @property
    def ngals(self):
        return self.nbar * self.volume

    @property
    def cosmo(self):
        if not hasattr(self, '_cosmo'):
            self.logger.info('Cosmology object not set. Using fiducial cosmology DESI.')
            from cosmoprimo.fiducial import DESI
            self._cosmo = DESI()
        return self._cosmo

    @cosmo.setter
    def cosmo(self, cosmo):
        self._cosmo = cosmo


class SurveyGeometry(Geometry, base.LinearBinning):

    def __init__(self, randoms, alpha, nmesh=None, cellsize=None, boxsize=None, boxpad=2., kmax=0.02, **kwargs):

        base.LinearBinning.__init__(self)

        self.logger = logging.getLogger('SurveyGeometry')

        self._alpha = alpha

        self._kmax = kmax

        self.window_matrix = None

        self._resume_file = None

        self._randoms = mockfactory.Catalog(randoms) if not isinstance(randoms, mockfactory.Catalog) else randoms

        # Check if the randoms have weights, otherwise set them to 1
        if 'WEIGHT' not in self._randoms:
            self.logger.warning(f'WEIGHT column not found in randoms. Setting it to 1.')
            self._randoms['WEIGHT'] = np.ones(self._randoms.size, dtype='f8')

        # Check if the randoms have a number density column, otherwise estimate it using RedshiftDensityInterpolator
        if 'NZ' not in self._randoms:
            self.logger.warning('NZ column not found in randoms. Estimating it with RedshiftDensityInterpolator.')
            import healpy as hp
            nside = 512
            distance = np.sqrt(np.sum(self._randoms['POSITION']**2, axis=-1))
            xyz = self._randoms['POSITION'] / distance[:, None]
            hpixel = hp.vec2pix(nside, *xyz.T)
            unique_hpixels = np.unique(hpixel)
            fsky = len(unique_hpixels) / hp.nside2npix(nside)
            self.logger.warning(f'fsky = {fsky:.3f}')
            self.logger.info(f'fsky estimated from randoms: {fsky:.3f}')
            nbar = mockfactory.RedshiftDensityInterpolator(z=distance, fsky=fsky)
            self._randoms['NZ'] = alpha * nbar(distance)

        # Check if the randoms have nmesh and cellsize, otherwise set them using the kmax parameter
        if nmesh is None and cellsize is None:
            # Pick value that will give at least k_mask = kmax_window in the FFTs
            cellsize = np.pi / kmax / (1. + 1e-9)

        self._mesh = CatalogMesh(data_positions=self._randoms['POSITION'], data_weights=self._randoms['WEIGHT']*alpha,
                                position_type='pos', nmesh=nmesh, cellsize=cellsize, boxsize=boxsize, boxpad=boxpad,
                                dtype='c16', **{'interlacing': 3, 'resampler': 'tsc', **kwargs})
        
        self.boxsize = self._mesh.boxsize[0]
        self.nmesh = self._mesh.nmesh[0]

        self.logger.info(f'Using box size {self._mesh.boxsize}, box center {self._mesh.boxcenter} and nmesh {self._mesh.nmesh}.')
    

        self.logger.info(f'Fundamental wavenumber of window FFTs = {self.kfun}.')
        self.logger.info(f'Nyquist wavenumber of window FFTs = {self.knyquist}.')

        if kmax is not None and self.knyquist < kmax:
            self.logger.warning(f'Nyquist wavelength {self.knyquist} smaller than required window kmax = {kmax}.')

        self.logger.info(f'Average of {self._mesh.data_size / self.nmesh**3} objects per voxel.')

    def compute_mesh(self, nbar_power, weight_power, ell, m):
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

        # get_real_Ylm returns a vectorized function (scipy.lpmv or numexpr)
        # that releases the GIL - no need for np.vectorize!
        Ylm = math.get_real_Ylm(ell, m)

        self.logger.info(f'Computing mesh nbar^{nbar_power} * weight^{weight_power} (ell={ell}, m={m})')
        start = time.time()

        result = self._mesh.copy(
            data_positions=self._randoms['POSITION'],
            data_weights=self._randoms['NZ']**(nbar_power-1)*self._randoms['WEIGHT']**(weight_power) * self.alpha * Ylm(*self._randoms['POSITION'].T),
            position_type='pos',
        ).to_mesh(compensate=True).r2c().value * self.nmesh**3

        # if threshold is not None:
        #     # Convert the result to a sparse array to save memory
        #     result[np.abs(result) < threshold] = 0
        #     result = base.SparseNDArray.from_dense(result, shape_in=(self.nmesh,self.nmesh), shape_out=self.nmesh)

        self.logger.info(f'Mesh computed in {time.time() - start:.0f} seconds.')

        return result

    @base.cache
    def compute_window_matrix(self, pk_ellmax=PK_ELL_MAX, mask_ellmax=MASK_ELL_MAX, kmodes_sampled=2000, n_workers=None, monitor=False, monitor_interval=5.0):
        '''Computes the window matrix using multiprocessing with shared memory.

        Parameters
        ----------
        pk_ellmax : int, optional
            Maximum ell for the power spectrum multipoles. Default is PK_ELL_MAX.
        mask_ellmax : int, optional
            Maximum ell for the mask multipoles. Default is MASK_ELL_MAX.
        kmodes_sampled : int, optional
            Number of k-modes to sample per bin. Default is 2000.
        n_workers : int, optional
            Number of worker processes. Default is cpu_count().
        monitor : bool, optional
            Whether to monitor resource usage during computation. Default is False.
            Requires psutil to be installed.
        monitor_interval : float, optional
            Resource monitoring interval in seconds. Default is 5.0.

        Notes
        -----
        The window matrices are computed using the method described in [1]_.

        References
        ----------
        .. [1] https://arxiv.org/abs/1910.02914
        '''

        if n_workers is None:
            n_workers = min(cpu_count(), 32)

        self.logger.info('=' * 60)
        self.logger.info(f'Computing window matrices with {n_workers} workers')
        self.logger.info(f'pk_ellmax={pk_ellmax}, mask_ellmax={mask_ellmax}')
            self.logger.info(f'pk_ellmax={pk_ellmax}, mask_ellmax={mask_ellmax}')
        self.logger.info('=' * 60)

        # Start resource monitor if requested
        resource_monitor = None
        if monitor:
            if HAS_PSUTIL:
                resource_monitor = ResourceMonitor(interval=monitor_interval)
                resource_monitor.start()
            else:
                self.logger.warning("psutil not installed, resource monitoring disabled. Install with: pip install psutil")

        # HYBRID SAMPLING
        self.logger.info('Sampling k-modes for binning...')
        kmodes, Nmodes =  math.sample_kmodes(kmin=self.kmin,
                                             kmax=self.kmax,
                                             dk=self.dk,
                                             boxsize=self.boxsize,
                                             max_modes=kmodes_sampled,
                                             k_shell_approx=0.1)

        delta_ik = np.array(np.meshgrid(*self.ikgrid, indexing='ij'))

        self.logger.info(f'Sampled k-modes for {self.kbins} bins')
        assert len(kmodes) == self.kbins and len(Nmodes) == self.kbins, \
            f'Error in sample_kmodes: results should have length {self.kbins}, but had {len(kmodes)}.'

        # Compute window products
        self.logger.info('Beginning of window multipole computation')
        
        # Collect all unique (nbar_power, weight_power, ell, m) combinations
        unique_mesh_params = set()
        for la, lb, ma, mb in utils.ellmiter(mask_ellmax, 2):
            unique_mesh_params.add((2, 2, la, ma))  # cosmic variance
            unique_mesh_params.add((2, 2, lb, mb))  # cosmic variance
            unique_mesh_params.add((1, 2, la, ma))  # mixed/shotnoise
            unique_mesh_params.add((1, 2, lb, mb))  # mixed/shotnoise
        
        self.logger.info(f'Computing {len(unique_mesh_params)} window meshes with {n_workers} threads')
        mesh_start = time.time()
        
        # Compute all meshes in parallel using threads (FFT releases GIL)
        def compute_mesh_wrapper(params):
            return params, self.compute_mesh(*params)
        
        mesh_cache = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(compute_mesh_wrapper, p): p for p in unique_mesh_params}
            for i, future in enumerate(as_completed(futures)):
                params, result = future.result()
                mesh_cache[params] = result
        
        self.logger.info(f'All meshes computed in {time.time() - mesh_start:.0f} seconds')
        
        # Get shape from any cached mesh
        shape_slab = next(iter(mesh_cache.values())).shape

        # Load Gaunt coefficients
        self.logger.info('Loading Gaunt coefficients...')
        coefficients = {
            'first_cosmic_variance': self.get_first_cosmic_variance_gaunt_coefficients(mask_ellmax, pk_ellmax),
            'second_cosmic_variance': self.get_second_cosmic_variance_gaunt_coefficients(mask_ellmax, pk_ellmax),
            'mixed_term': self.get_mixed_gaunt_coefficients(mask_ellmax, pk_ellmax),
            'shotnoise': self.get_shotnoise_gaunt_coefficients(mask_ellmax, pk_ellmax),
        }

        # window_product will be populated in the loop below
        window_product = {}

        self.logger.info('Contracting Gaunt coefficients with window mesh products...')
        start = time.time()

        nbar_weight_indices = {
            'first_cosmic_variance': ((2, 2), (2, 2)),
            'second_cosmic_variance': ((2, 2), (2, 2)),
            'mixed_term': ((2, 2), (1, 2)),
            'shotnoise': ((1, 2), (1, 2)),
        }

        # Process each term separately to limit memory
        for term_name, coeff in coefficients.items():
            self.logger.info(f'Computing {term_name} term')

            nw1, nw2 = nbar_weight_indices[term_name]
            
            product = base.SparseNDArray(
                shape_out=coeff.shape_in,  # (la, lb, ma, mb) indices
                shape_in=shape_slab,
                dtype=np.complex128
            )
            
            # Only compute products for non-zero Gaunt indices
            for index in coeff.T.nonzero_indices_out():
                la, lb = 2*index[0], 2*index[1]
                ma, mb = index[2] - la, index[3] - lb
                
                product[index] = mesh_cache[(*nw1, la, ma)] * mesh_cache[(*nw2, lb, mb)]
                
            window_product[term_name] = coeff @ product

            del product
            
        del mesh_cache

        self.logger.info(f'Gaunt contraction completed in {time.time() - start:.0f} seconds')

        # Create shared memory for sparse arrays
        self.logger.info('Setting up shared memory for parallel processing')
        shm_metadata = {}
        all_shm_handles = []
        for key, sparse_arr in window_product.items():
            metadata, handles = sparse_arr.to_shared_memory()
            shm_metadata[key] = metadata
            all_shm_handles.extend(handles)
        
        # Now we can delete window_product to free memory
        del window_product

        # Also share delta_ik
        delta_ik_shm = _create_shared_ndarray(delta_ik)
        all_shm_handles.append(delta_ik_shm['handle'])

        # Prepare worker arguments (only small data, no large arrays)
        worker_args = {
            'shm_metadata': shm_metadata,
            'delta_ik_info': {
                'name': delta_ik_shm['name'],
                'shape': delta_ik.shape,
                'dtype': delta_ik.dtype,
            },
            'pk_ellmax': pk_ellmax,
            'kfun': self.kfun,
            'dk': self.dk,
            'kbins': self.kbins,
            'kmin': self.kmin,
        }

        # Initialize output arrays
        window_matrix = {
            'cosmic_variance': np.zeros(4*[pk_ellmax//2+1] + 2*[self.kbins]),
            'mixed_term': np.zeros(3*[pk_ellmax//2+1] + 2*[self.kbins]),
            'shotnoise': np.zeros(2*[pk_ellmax//2+1] + 2*[self.kbins]),
        }
        
        # Target ~32-64 modes per worker for good efficiency
        avg_modes = np.mean([len(km) for km in kmodes[2:]])
        target_modes_per_worker = 32
        workers_per_bin = max(1, min(n_workers, int(avg_modes / target_modes_per_worker)))
        bins_parallel = max(1, n_workers // workers_per_bin)
        
        self.logger.info(f'Starting HYBRID integration with {n_workers} workers')
        self.logger.info(f'  - {bins_parallel} bins in parallel')
        self.logger.info(f'  - {workers_per_bin} workers per bin')
        self.logger.info(f'  - ~{avg_modes / workers_per_bin:.0f} modes per worker')
        start_time = time.time()

        try:
            with Pool(n_workers, initializer=_init_worker, initargs=(worker_args,)) as pool:
                
                # Process bins in batches
                n_bins = len(kmodes)
                for batch_start in range(0, n_bins, bins_parallel):
                    batch_end = min(batch_start + bins_parallel, n_bins)
                    batch_bins = list(range(batch_start, batch_end))
                    
                    # Create tasks for all bins in this batch
                    all_tasks = []
                    for i in batch_bins:
                        km = kmodes[i]
                        k1_bin_index = int(i + self.kmin // self.dk)
                        
                        # Split modes into chunks (workers_per_bin chunks per bin)
                        km_array = np.array(km)
                        chunks = np.array_split(km_array, min(workers_per_bin, len(km_array)))
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            if len(chunk) > 0:
                                all_tasks.append((i, chunk.tolist(), k1_bin_index, chunk_idx))
                    
                    # Process all tasks for this batch in parallel
                    # Accumulate results per bin
                    bin_results = {i: {
                        'cosmic_variance': np.zeros((pk_ellmax//2+1,) * 4 + (self.kbins,)),
                        'mixed_term': np.zeros((pk_ellmax//2+1,) * 3 + (self.kbins,)),
                        'shotnoise': np.zeros((pk_ellmax//2+1,) * 2 + (self.kbins,)),
                        'k1_bin_index': int(i + self.kmin // self.dk)
                    } for i in batch_bins}
                    
                    for result in pool.imap_unordered(_process_modes_chunk, all_tasks):
                        i, k1_bin_index, cosmic_variance, mixed_term, shotnoise, chunk_idx, profiling = result
                        bin_results[i]['cosmic_variance'] += cosmic_variance
                        bin_results[i]['mixed_term'] += mixed_term
                        bin_results[i]['shotnoise'] += shotnoise

                        # Print profiling for first chunk of first 4 bins
                        if profiling:
                            total = profiling['total']
                            print(f"\n=== PROFILING k-bin {i} chunk {chunk_idx} ({profiling['n_modes']} modes) ===")
                            print(f"  k2 computation:    {profiling['t_k2']:6.2f}s ({100*profiling['t_k2']/total:5.1f}%)")
                            print(f"  Sparse matrix:     {profiling['t_sparse']:6.2f}s ({100*profiling['t_sparse']/total:5.1f}%)")
                            print(f"  Ylm2 evaluation:   {profiling['t_ylm2']:6.2f}s ({100*profiling['t_ylm2']/total:5.1f}%)")
                            print(f"  CV weights:        {profiling['t_cv']:6.2f}s ({100*profiling['t_cv']/total:5.1f}%)")
                            print(f"  MT weights:        {profiling['t_mt']:6.2f}s ({100*profiling['t_mt']/total:5.1f}%)")
                            print(f"  SN weights:        {profiling['t_sn']:6.2f}s ({100*profiling['t_sn']/total:5.1f}%)")
                            print(f"  TOTAL:             {total:6.2f}s")
                            print("=" * 45)
                    
                    # Add batch results to window matrix
                    for i in batch_bins:
                        k1_idx = bin_results[i]['k1_bin_index']
                        window_matrix['cosmic_variance'][..., k1_idx, :] += bin_results[i]['cosmic_variance']
                        window_matrix['mixed_term'][..., k1_idx, :] += bin_results[i]['mixed_term']
                        window_matrix['shotnoise'][..., k1_idx, :] += bin_results[i]['shotnoise']

                    self.logger.info(f'Completed bins {batch_start+1}-{batch_end}/{n_bins}')

        finally:
            # Stop resource monitor and save results
            if resource_monitor is not None:
                resource_monitor.stop()
                resource_monitor.summary()
                resource_monitor.plot('window_matrix_resources.png')
            
            # Cleanup shared memory
            self.logger.info('Cleaning up shared memory...')
            base.SparseNDArray.cleanup_shared_memory(all_shm_handles)

        # Apply normalization
        self.logger.info('Applying normalization...')
        norm = (4*np.pi)**2 / self.normalization(2, 2)**2
        for i, Nm in enumerate(Nmodes):
            k_bin_index = int(i + self.kmin // self.dk)
            window_matrix['cosmic_variance'][..., k_bin_index, :] *= norm / Nm
            window_matrix['mixed_term'][..., k_bin_index, :] *= norm / Nm
            window_matrix['shotnoise'][..., k_bin_index, :] *= norm / Nm

        self.window_matrix = window_matrix
        self.logger.info(f'Window matrix computation completed in {time.time() - start_time:.0f} seconds!')

        return self.window_matrix

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
        """Calculates all relavent Gaunt coefficients for the mixed term, or loads them from file"""
        
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

    @property
    def has_mpi(self):
        return self.mpicomm is not None