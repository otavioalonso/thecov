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
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import numba

import mockfactory
from pypower import CatalogMesh

from . import base, utils, math
from .monitor import ResourceMonitor, HAS_PSUTIL


__all__ = ['BoxGeometry',
           'SurveyGeometry']

MASK_ELL_MAX = 4 # max = 3*PK_ELL_MAX
PK_ELL_MAX = 4

complex_dtype = np.complex128
float_dtype = np.float32

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
        self.logger.info('=' * 60)

        # Profiling timers
        profiling = {
            'total_start': time.time(),
            'kmode_sampling': 0.0,
            'mesh_computation': 0.0,
            'gaunt_loading': 0.0,
            'gaunt_contraction': 0.0,
            'shared_memory_setup': 0.0,
            'mode_integration': 0.0,
            'normalization': 0.0,
        }

        # Start resource monitor if requested
        resource_monitor = None
        if monitor:
            if HAS_PSUTIL:
                resource_monitor = ResourceMonitor(interval=monitor_interval)
                resource_monitor.start()
            else:
                self.logger.warning("psutil not installed, resource monitoring disabled. Install with: pip install psutil")

        # ==================== PHASE 1: K-MODE SAMPLING ====================
        phase_start = time.time()
        self.logger.info('Sampling k-modes for binning...')
        kmodes, Nmodes = math.sample_kmodes(
            kmin=self.kmin, kmax=self.kmax, dk=self.dk,
            boxsize=self.boxsize, max_modes=kmodes_sampled, k_shell_approx=0.1
        )

        delta_ik = np.array(np.meshgrid(*self.ikgrid, indexing='ij'))

        self.logger.info(f'Sampled k-modes for {self.kbins} bins')
        assert len(kmodes) == self.kbins and len(Nmodes) == self.kbins, \
            f'Error in sample_kmodes: results should have length {self.kbins}, but had {len(kmodes)}.'
        profiling['kmode_sampling'] = time.time() - phase_start
        self.logger.info(f'[PROFILING] K-mode sampling: {profiling["kmode_sampling"]:.2f}s')

        # ==================== PHASE 2: MESH COMPUTATION ====================
        phase_start = time.time()
        self.logger.info('Beginning of window multipole computation')
        
        # Collect all unique (nbar_power, weight_power, ell, m) combinations
        unique_mesh_params = set()
        for la, lb, ma, mb in utils.ellmiter(mask_ellmax, 2):
            unique_mesh_params.add((2, 2, la, ma))  # cosmic variance
            unique_mesh_params.add((2, 2, lb, mb))  # cosmic variance
            unique_mesh_params.add((1, 2, la, ma))  # mixed/shotnoise
            unique_mesh_params.add((1, 2, lb, mb))  # mixed/shotnoise
        
        self.logger.info(f'Computing {len(unique_mesh_params)} window meshes with {n_workers} threads')
        
        # Compute all meshes in parallel using threads (FFT releases GIL)
        def compute_mesh_wrapper(params):
            return params, self.compute_mesh(*params)
        
        mesh_cache = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(compute_mesh_wrapper, p): p for p in unique_mesh_params}
            for i, future in enumerate(as_completed(futures)):
                params, result = future.result()
                mesh_cache[params] = result
        
        profiling['mesh_computation'] = time.time() - phase_start
        self.logger.info(f'[PROFILING] Mesh computation: {profiling["mesh_computation"]:.2f}s')

        # ==================== PHASE 3: GAUNT COEFFICIENT LOADING ====================
        phase_start = time.time()
        self.logger.info('Loading Gaunt coefficients...')
        coefficients = {
            'first_cosmic_variance': self.get_first_cosmic_variance_gaunt_coefficients(mask_ellmax, pk_ellmax),
            'second_cosmic_variance': self.get_second_cosmic_variance_gaunt_coefficients(mask_ellmax, pk_ellmax),
            'mixed_term': self.get_mixed_gaunt_coefficients(mask_ellmax, pk_ellmax),
            'shotnoise': self.get_shotnoise_gaunt_coefficients(mask_ellmax, pk_ellmax),
        }
        profiling['gaunt_loading'] = time.time() - phase_start
        self.logger.info(f'[PROFILING] Gaunt loading: {profiling["gaunt_loading"]:.2f}s')

        # ==================== PHASE 4: GAUNT CONTRACTION ====================
        phase_start = time.time()
        # window_product will be populated in the loop below
        window_product = {}

        self.logger.info('Contracting Gaunt coefficients with window mesh products...')

        nbar_weight_indices = {
            'first_cosmic_variance': ((2, 2), (2, 2)),
            'second_cosmic_variance': ((2, 2), (2, 2)),
            'mixed_term': ((2, 2), (1, 2)),
            'shotnoise': ((1, 2), (1, 2)),
        }

        # Track per-term timing
        term_timings = {}

        # Process each term separately to limit memory
        for term_name, coeff in coefficients.items():
            term_start = time.time()
            self.logger.info(f'Computing {term_name} term')

            nw1, nw2 = nbar_weight_indices[term_name]
            
            product = base.SparseNDArray(
                shape_out=coeff.shape_in,  # (la, lb, ma, mb) indices
                shape_in=[self.nmesh**3],
                dtype=float_dtype,
            )
            
            # Only compute products for non-zero Gaunt indices
            n_nonzero = 0
            for index in coeff.T.nonzero_indices_out():
                la, lb = 2*index[0], 2*index[1]
                ma, mb = index[2] - la, index[3] - lb

                product[index] = (mesh_cache[(*nw1, la, ma)] * np.conj(mesh_cache[(*nw2, lb, mb)])).real.ravel().astype(float_dtype)
                n_nonzero += 1

            window_product[term_name] = coeff @ product

            del product
            term_timings[term_name] = time.time() - term_start
            self.logger.info(f'  {term_name}: {n_nonzero} products, {term_timings[term_name]:.2f}s')
            
        del mesh_cache

        profiling['gaunt_contraction'] = time.time() - phase_start
        self.logger.info(f'[PROFILING] Gaunt contraction: {profiling["gaunt_contraction"]:.2f}s')

        # ==================== PHASE 5: SHARED MEMORY SETUP ====================
        phase_start = time.time()
        self.logger.info('Setting up shared memory for parallel processing')
        
        # Convert sparse window_product to dense arrays for efficient worker access
        # This eliminates sparse indexing overhead in the hot loop
        all_shm_handles = []
        window_shm_info = {}
        
        for key, sparse_arr in window_product.items():
            indices, values = sparse_arr.get_nonzero_rows_dense()
            self.logger.info(f'  {key}: {len(indices)} nonzero rows, values shape {values.shape}')
            
            # Create shared memory for indices and values
            indices_shm = _create_shared_ndarray(indices)
            values_shm = _create_shared_ndarray(values.astype(float_dtype))
            all_shm_handles.append(indices_shm['handle'])
            all_shm_handles.append(values_shm['handle'])
            
            window_shm_info[key] = {
                'indices_name': indices_shm['name'],
                'indices_shape': indices.shape,
                'indices_dtype': indices.dtype,
                'values_name': values_shm['name'],
                'values_shape': values.shape,
                'values_dtype': float_dtype,
                'shape_out': sparse_arr.shape_out,
            }
        
        del window_product  # Free memory

        # Also share delta_ik
        delta_ik_shm = _create_shared_ndarray(delta_ik)
        all_shm_handles.append(delta_ik_shm['handle'])

        # Prepare worker arguments (only small data, no large arrays)
        worker_args = {
            'window_shm_info': window_shm_info,
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

        profiling['shared_memory_setup'] = time.time() - phase_start
        self.logger.info(f'[PROFILING] Shared memory setup: {profiling["shared_memory_setup"]:.2f}s')

        # ==================== PHASE 6: MODE INTEGRATION ====================
        phase_start = time.time()
        
        # Target ~32-64 modes per worker for good efficiency
        avg_modes = np.mean([len(km) for km in kmodes[2:]])
        target_modes_per_worker = 32
        workers_per_bin = max(1, min(n_workers, int(avg_modes / target_modes_per_worker)))
        bins_parallel = max(1, n_workers // workers_per_bin)
        
        total_modes = sum(len(km) for km in kmodes)
        self.logger.info(f'Starting mode integration with {n_workers} workers')
        self.logger.info(f'  Total modes to process: {total_modes}')
        self.logger.info(f'  {bins_parallel} bins in parallel')
        self.logger.info(f'  {workers_per_bin} workers per bin')
        self.logger.info(f'  {avg_modes / workers_per_bin:.0f} modes per worker (avg)')

        # Worker profiling aggregation
        worker_timers = {
            'ylm1_setup': 0.0,
            'k2_computation': 0.0,
            'bin_matrix': 0.0,
            'ylm2_computation': 0.0,
            'cosmic_variance': 0.0,
            'mixed_term': 0.0,
            'shotnoise': 0.0,
        }
        
        # Memory profiling aggregation (track max across all workers)
        memory_stats = {
            'baseline': [],
            'after_ylm1': [],
            'after_ik2': [],
            'after_bin_matrix': [],
            'after_ylm2': [],
            'peak_cosmic_variance': [],
            'peak_mixed_term': [],
            'peak_shotnoise': [],
            'final': [],
        }
        n_tasks_profiled = 0

        try:
            with Pool(n_workers, initializer=_init_worker, initargs=(worker_args,)) as pool:
                
                # Process bins in batches
                n_bins = len(kmodes)
                modes_processed = 0
                for batch_start in range(0, n_bins, bins_parallel):
                    batch_time = time.time()
                    batch_end = min(batch_start + bins_parallel, n_bins)
                    batch_bins = list(range(batch_start, batch_end))
                    
                    # Create tasks for all bins in this batch
                    all_tasks = []
                    batch_modes = 0
                    for i in batch_bins:
                        km = np.array(kmodes[i])
                        k1_bin_index = int(i + self.kmin // self.dk)
                        batch_modes += len(km)
                        
                        # Split modes into chunks (workers_per_bin chunks per bin)
                        chunks = np.array_split(km, min(workers_per_bin, len(km)))
                        
                        for chunk in chunks:
                            if len(chunk) > 0:
                                all_tasks.append((i, chunk.tolist(), k1_bin_index))
                    
                    # Process all tasks for this batch in parallel
                    # Accumulate results per bin
                    bin_results = {i: {
                        'cosmic_variance': np.zeros((pk_ellmax//2+1,) * 4 + (self.kbins,)),
                        'mixed_term': np.zeros((pk_ellmax//2+1,) * 3 + (self.kbins,)),
                        'shotnoise': np.zeros((pk_ellmax//2+1,) * 2 + (self.kbins,)),
                        'k1_bin_index': int(i + self.kmin // self.dk)
                    } for i in batch_bins}
                    
                    for result in pool.imap_unordered(_process_modes, all_tasks):
                        i, k1_bin_index, cosmic_variance, mixed_term, shotnoise, timers, memory_profile = result
                        bin_results[i]['cosmic_variance'] += cosmic_variance
                        bin_results[i]['mixed_term'] += mixed_term
                        bin_results[i]['shotnoise'] += shotnoise
                        # Aggregate worker timers
                        for key in worker_timers:
                            worker_timers[key] += timers[key]
                        # Aggregate memory stats
                        for key in memory_stats:
                            if memory_profile.get(key, 0) > 0:
                                memory_stats[key].append(memory_profile[key])
                        n_tasks_profiled += 1
                    
                    # Add batch results to window matrix
                    for i in batch_bins:
                        k1_idx = bin_results[i]['k1_bin_index']
                        window_matrix['cosmic_variance'][..., k1_idx, :] += bin_results[i]['cosmic_variance']
                        window_matrix['mixed_term'][..., k1_idx, :] += bin_results[i]['mixed_term']
                        window_matrix['shotnoise'][..., k1_idx, :] += bin_results[i]['shotnoise']

                    modes_processed += batch_modes
                    batch_elapsed = time.time() - batch_time
                    modes_per_sec = batch_modes / batch_elapsed if batch_elapsed > 0 else 0
                    self.logger.info(f'Completed bins {batch_start+1}-{batch_end}/{n_bins} | '
                                   f'{batch_modes} modes in {batch_elapsed:.1f}s ({modes_per_sec:.1f} modes/s) | '
                                   f'Progress: {modes_processed}/{total_modes} ({100*modes_processed/total_modes:.1f}%)')
        finally:
            # Stop resource monitor and save results
            if resource_monitor is not None:
                resource_monitor.stop()
                resource_monitor.summary()
                resource_monitor.plot('window_matrix_resources.png')
            
            # Cleanup shared memory
            self.logger.info('Cleaning up shared memory...')
            for shm in all_shm_handles:
                shm.close()
                shm.unlink()

        profiling['mode_integration'] = time.time() - phase_start
        profiling['worker_timers'] = worker_timers
        profiling['n_tasks_profiled'] = n_tasks_profiled
        self.logger.info(f'[PROFILING] Mode integration: {profiling["mode_integration"]:.2f}s')

        # ==================== PHASE 7: NORMALIZATION ====================
        phase_start = time.time()
        self.logger.info('Applying normalization...')
        norm = (4*np.pi)**2 / self.normalization(2, 2)**2
        for i, Nm in enumerate(Nmodes):
            k_bin_index = int(i + self.kmin // self.dk)
            window_matrix['cosmic_variance'][..., k_bin_index, :] *= norm / Nm
            window_matrix['mixed_term'][..., k_bin_index, :] *= norm / Nm
            window_matrix['shotnoise'][..., k_bin_index, :] *= norm / Nm

        profiling['normalization'] = time.time() - phase_start
        profiling['memory_stats'] = memory_stats
        self.logger.info(f'[PROFILING] Normalization: {profiling["normalization"]:.2f}s')

        # ==================== PROFILING SUMMARY ====================
        profiling['total'] = time.time() - profiling['total_start']
        self.logger.info('=' * 60)
        self.logger.info('PROFILING SUMMARY')
        self.logger.info('=' * 60)
        self.logger.info(f'  K-mode sampling:      {profiling["kmode_sampling"]:8.2f}s ({100*profiling["kmode_sampling"]/profiling["total"]:5.1f}%)')
        self.logger.info(f'  Mesh computation:     {profiling["mesh_computation"]:8.2f}s ({100*profiling["mesh_computation"]/profiling["total"]:5.1f}%)')
        self.logger.info(f'  Gaunt loading:        {profiling["gaunt_loading"]:8.2f}s ({100*profiling["gaunt_loading"]/profiling["total"]:5.1f}%)')
        self.logger.info(f'  Gaunt contraction:    {profiling["gaunt_contraction"]:8.2f}s ({100*profiling["gaunt_contraction"]/profiling["total"]:5.1f}%)')
        self.logger.info(f'  Shared memory setup:  {profiling["shared_memory_setup"]:8.2f}s ({100*profiling["shared_memory_setup"]/profiling["total"]:5.1f}%)')
        self.logger.info(f'  Mode integration:     {profiling["mode_integration"]:8.2f}s ({100*profiling["mode_integration"]/profiling["total"]:5.1f}%)')
        self.logger.info(f'  Normalization:        {profiling["normalization"]:8.2f}s ({100*profiling["normalization"]/profiling["total"]:5.1f}%)')
        self.logger.info('-' * 60)
        self.logger.info(f'  TOTAL:                {profiling["total"]:8.2f}s')
        self.logger.info('=' * 60)

        # Worker timing breakdown (aggregated across all tasks)
        if profiling.get('n_tasks_profiled', 0) > 0:
            wt = profiling['worker_timers']
            wt_total = sum(wt.values())
            self.logger.info('')
            self.logger.info('WORKER TIMING BREAKDOWN (aggregated CPU time across all tasks)')
            self.logger.info('=' * 60)
            self.logger.info(f'  Ylm1 setup:           {wt["ylm1_setup"]:8.2f}s ({100*wt["ylm1_setup"]/wt_total:5.1f}%)')
            self.logger.info(f'  k2 computation:       {wt["k2_computation"]:8.2f}s ({100*wt["k2_computation"]/wt_total:5.1f}%)')
            self.logger.info(f'  Bin matrix build:     {wt["bin_matrix"]:8.2f}s ({100*wt["bin_matrix"]/wt_total:5.1f}%)')
            self.logger.info(f'  Ylm2 computation:     {wt["ylm2_computation"]:8.2f}s ({100*wt["ylm2_computation"]/wt_total:5.1f}%)')
            self.logger.info(f'  Cosmic variance:      {wt["cosmic_variance"]:8.2f}s ({100*wt["cosmic_variance"]/wt_total:5.1f}%)')
            self.logger.info(f'  Mixed term:           {wt["mixed_term"]:8.2f}s ({100*wt["mixed_term"]/wt_total:5.1f}%)')
            self.logger.info(f'  Shot noise:           {wt["shotnoise"]:8.2f}s ({100*wt["shotnoise"]/wt_total:5.1f}%)')
            self.logger.info('-' * 60)
            self.logger.info(f'  Total worker CPU:     {wt_total:8.2f}s')
            self.logger.info(f'  Tasks profiled:       {profiling["n_tasks_profiled"]}')
            self.logger.info(f'  Parallelization eff:  {wt_total / profiling["mode_integration"] / n_workers * 100:.1f}%')
            self.logger.info('=' * 60)

        # Memory profiling breakdown
        if profiling.get('memory_stats') and any(profiling['memory_stats'].values()):
            ms = profiling['memory_stats']
            self.logger.info('')
            self.logger.info('WORKER MEMORY PROFILING (MB per worker)')
            self.logger.info('=' * 60)
            for key in ['baseline', 'after_ylm1', 'after_ik2', 'after_bin_matrix', 'after_ylm2', 
                        'peak_cosmic_variance', 'peak_mixed_term', 'peak_shotnoise', 'final']:
                values = ms.get(key, [])
                if values:
                    self.logger.info(f'  {key:24s}: min={min(values):7.1f}  max={max(values):7.1f}  avg={sum(values)/len(values):7.1f}')
            # Calculate memory deltas to identify hotspots
            self.logger.info('-' * 60)
            self.logger.info('MEMORY HOTSPOTS (max increase from baseline)')
            baseline_avg = sum(ms.get('baseline', [0])) / max(len(ms.get('baseline', [0])), 1)
            for key in ['peak_cosmic_variance', 'peak_mixed_term', 'peak_shotnoise']:
                values = ms.get(key, [])
                if values:
                    delta = max(values) - baseline_avg
                    self.logger.info(f'  {key:24s}: +{delta:7.1f} MB from baseline')
            self.logger.info('=' * 60)

        self.window_matrix = window_matrix
        self._profiling = profiling  # Store for later analysis

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


# ============================================================================
# Module-level functions for multiprocessing (must be at module level for pickling)
# ============================================================================

_worker_data = {}

def _create_shared_ndarray(arr):
    """Create a shared memory array from a numpy array."""
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shared_arr[:] = arr[:]
    return {'handle': shm, 'name': shm.name}

def _init_worker(args):
    """Initialize worker with shared memory references."""
    import os
    from multiprocessing import shared_memory
    
    # Limit NumPy/BLAS threading to 1 per worker to prevent thread contention
    # This is optimal because _process_kbin uses element-wise ops, not BLAS
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    global _worker_data
    _worker_data = args.copy()
    _worker_data['_shm_refs'] = []

    # Attach to shared memory dense arrays for window_product
    # Build lookup dict: window_data[key] = {'indices': array, 'values': array, 'index_map': dict}
    _worker_data['window_data'] = {}
    
    for key, info in args['window_shm_info'].items():
        # Attach to indices shared memory
        indices_shm = shared_memory.SharedMemory(name=info['indices_name'])
        indices = np.ndarray(info['indices_shape'], dtype=info['indices_dtype'], buffer=indices_shm.buf)
        _worker_data['_shm_refs'].append(indices_shm)
        
        # Attach to values shared memory
        values_shm = shared_memory.SharedMemory(name=info['values_name'])
        values = np.ndarray(info['values_shape'], dtype=info['values_dtype'], buffer=values_shm.buf)
        _worker_data['_shm_refs'].append(values_shm)
        
        # Build index lookup map: tuple(index) -> row number in values array
        index_map = {tuple(idx): i for i, idx in enumerate(indices)}
        
        _worker_data['window_data'][key] = {
            'indices': indices,
            'values': values,
            'index_map': index_map,
            'shape_out': info['shape_out'],
        }

    # Attach to delta_ik shared array
    delta_info = args['delta_ik_info']
    delta_shm = shared_memory.SharedMemory(name=delta_info['name'])
    _worker_data['delta_ik'] = np.ndarray(
        delta_info['shape'], dtype=delta_info['dtype'], buffer=delta_shm.buf
    )
    _worker_data['_shm_refs'].append(delta_shm)

def _get_memory_mb():
    """Get current process memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def _process_modes(task):
    import time
    
    i, kmodes, k1_bin_index = task

    global _worker_data
    window_data = _worker_data['window_data']
    delta_ik = _worker_data['delta_ik']
    pk_ellmax = _worker_data['pk_ellmax']
    kfun = _worker_data['kfun']
    dk = _worker_data['dk']
    kbins = _worker_data['kbins']

    # Profiling accumulators
    timers = {
        'ylm1_setup': 0.0,
        'k2_computation': 0.0,
        'bin_matrix': 0.0,
        'ylm2_computation': 0.0,
        'cosmic_variance': 0.0,
        'mixed_term': 0.0,
        'shotnoise': 0.0,
    }
    
    # Memory profiling - track peak memory at each stage
    memory_profile = {
        'baseline': 0.0,
        'after_ylm1': 0.0,
        'after_ik2': 0.0,
        'after_bin_matrix': 0.0,
        'after_ylm2': 0.0,
        'peak_cosmic_variance': 0.0,
        'peak_mixed_term': 0.0,
        'peak_shotnoise': 0.0,
        'final': 0.0,
    }
    
    memory_profile['baseline'] = _get_memory_mb()

    # Initialize local accumulators
    cosmic_variance = np.zeros((pk_ellmax//2+1,) * 4 + (kbins,))
    mixed_term = np.zeros((pk_ellmax//2+1,) * 3 + (kbins,))
    shotnoise = np.zeros((pk_ellmax//2+1,) * 2 + (kbins,))

    kmodes = np.array(kmodes)  # shape (n_modes, 4)
    
    if kmodes.shape[0] == 0:
        return i, k1_bin_index, cosmic_variance, mixed_term, shotnoise, timers, memory_profile
    
    # Grid size
    grid_shape = delta_ik.shape[1:]  # (nmesh, nmesh, nmesh)
    n_grid = int(np.prod(grid_shape))
    lmax = pk_ellmax
    
    # Extract indices and values for each term
    first_cosmic_variance_indices = window_data['first_cosmic_variance']['indices']
    first_cosmic_variance_values = window_data['first_cosmic_variance']['values']
    second_cosmic_variance_indices = window_data['second_cosmic_variance']['indices']
    second_cosmic_variance_values = window_data['second_cosmic_variance']['values']
    mixed_term_indices = window_data['mixed_term']['indices']
    mixed_term_values = window_data['mixed_term']['values']
    shotnoise_indices = window_data['shotnoise']['indices']
    shotnoise_values = window_data['shotnoise']['values']
    
    # Cache Ylm functions
    t0 = time.perf_counter()
    Ylm_funcs = {(l, m): math.get_real_Ylm(l, m) for l, m in utils.ellmiter(pk_ellmax, 1)}
    
    # Precompute Yk1 for all modes: dict -> flat array for Numba
    # Layout: Yk1_all[mode_idx, l*(2*lmax+1) + m+lmax]
    n_modes = kmodes.shape[0]
    n_lm = (lmax + 1) * (2 * lmax + 1)
    Yk1_all = np.zeros((n_modes, n_lm), dtype=np.float64)
    for l, m in utils.ellmiter(pk_ellmax, 1):
        Yk1_all[:, l*(2*lmax + 1) + m + lmax] = Ylm_funcs[l, m](kmodes[:, 0], kmodes[:, 1], kmodes[:, 2])
    
    timers['ylm1_setup'] = time.perf_counter() - t0
    memory_profile['after_ylm1'] = max(memory_profile['after_ylm1'], _get_memory_mb())

    # Process each k-mode
    for mode_idx in range(n_modes):
        ik1 = kmodes[mode_idx, :3]
        
        # k2 = k1 + delta_k for all grid points
        t0 = time.perf_counter()
        ik2 = ik1[:, None, None, None] + delta_ik  # (3, nmesh, nmesh, nmesh)
        timers['k2_computation'] += time.perf_counter() - t0
        memory_profile['after_ik2'] = max(memory_profile['after_ik2'], _get_memory_mb())

        # Compute k2 bin indices
        t0 = time.perf_counter()
        k2_bin_index = (np.sqrt(np.sum(ik2**2, axis=0)) * kfun / dk).astype(np.int32).ravel()
        valid_mask = (k2_bin_index >= 0) & (k2_bin_index < kbins)
        timers['bin_matrix'] += time.perf_counter() - t0
        memory_profile['after_bin_matrix'] = max(memory_profile['after_bin_matrix'], _get_memory_mb())
        
        # Precompute Yk2 as 3D array for Numba: (lmax+1, 2*lmax+1, n_grid)
        t0 = time.perf_counter()
        Yk2_3d = np.zeros((lmax + 1, 2 * lmax + 1, n_grid), dtype=np.float64)
        for l, m in utils.ellmiter(pk_ellmax, 1):
            Yk2_3d[l, m + lmax, :] = np.broadcast_to(Ylm_funcs[l, m](*ik2), grid_shape).ravel()
        timers['ylm2_computation'] += time.perf_counter() - t0
        memory_profile['after_ylm2'] = max(memory_profile['after_ylm2'], _get_memory_mb())

        # Get Yk1 values for this mode as flat array
        Yk1_flat = Yk1_all[mode_idx, :]

        # ============ COSMIC VARIANCE (using Numba kernels) ============
        t0 = time.perf_counter()
        
        # First cosmic variance term: Yk1[l1,m1] * Yk1[l2,m2] * Yk2[l3,m3] * Yk2[l4,m4]
        if len(first_cosmic_variance_indices) > 0:
            first_cosmic_variance_contribution = _compute_cosmic_variance_first(
                k2_bin_index, valid_mask, first_cosmic_variance_values, first_cosmic_variance_indices,
                Yk1_flat, Yk2_3d, pk_ellmax, kbins
            )
            cosmic_variance += first_cosmic_variance_contribution
            memory_profile['peak_cosmic_variance'] = max(memory_profile['peak_cosmic_variance'], _get_memory_mb())
        
        # Second cosmic variance term: Yk1[l1,m1] * Yk1[l3,m3] * Yk2[l2,m2] * Yk2[l4,m4]
        if len(second_cosmic_variance_indices) > 0:
            second_cosmic_variance_contribution = _compute_cosmic_variance_second(
                k2_bin_index, valid_mask, second_cosmic_variance_values, second_cosmic_variance_indices,
                Yk1_flat, Yk2_3d, pk_ellmax, kbins
            )
            cosmic_variance += second_cosmic_variance_contribution
            memory_profile['peak_cosmic_variance'] = max(memory_profile['peak_cosmic_variance'], _get_memory_mb())
        
        timers['cosmic_variance'] += time.perf_counter() - t0

        # ============ MIXED TERM ============
        t0 = time.perf_counter()
        
        if len(mixed_term_indices) > 0:
            mixed_term_contribution = _compute_mixed_term_contribution(
                k2_bin_index, valid_mask, mixed_term_values, mixed_term_indices,
                Yk1_flat, Yk2_3d, pk_ellmax, kbins
            )
            mixed_term += mixed_term_contribution
            memory_profile['peak_mixed_term'] = max(memory_profile['peak_mixed_term'], _get_memory_mb())
        
        timers['mixed_term'] += time.perf_counter() - t0

        # ============ SHOT NOISE ============
        t0 = time.perf_counter()
        
        if len(shotnoise_indices) > 0:
            shotnoise_contribution = _compute_shotnoise_contribution(
                k2_bin_index, valid_mask, shotnoise_values, shotnoise_indices,
                Yk1_flat, Yk2_3d, pk_ellmax, kbins
            )
            shotnoise += shotnoise_contribution
            memory_profile['peak_shotnoise'] = max(memory_profile['peak_shotnoise'], _get_memory_mb())
        
        timers['shotnoise'] += time.perf_counter() - t0

    # Final memory measurement
    memory_profile['final'] = _get_memory_mb()
    
    # Return profiling info with results
    return i, k1_bin_index, cosmic_variance, mixed_term, shotnoise, timers, memory_profile



# ============================================================================
# Numba-optimized kernels for window matrix computation
# ============================================================================

@numba.njit(fastmath=True, cache=True)
def _compute_cosmic_variance_first(
    bin_indices,        # (n_grid,) - which k-bin each grid point belongs to
    valid_mask,         # (n_grid,) - boolean mask for valid grid points
    window_product,     # (n_nonzero, n_grid) - window product values
    coeff_indices,      # (n_nonzero, 8) - l1/2, l2/2, l3/2, l4/2, m1+l1, m2+l2, m3+l3, m4+l4
    Yk1_flat,           # (n_lm,) - Ylm values for k1 at this mode, indexed as l*(2*lmax+1) + m+lmax
    Yk2_3d,             # (lmax+1, 2*lmax+1, n_grid) - Ylm values for all k2 grid points
    pk_ellmax,          # Maximum ell
    kbins               # Number of k-bins
):
    """Compute FIRST cosmic variance contribution: Yk1[l1,m1] * Yk1[l2,m2] * Yk2[l3,m3] * Yk2[l4,m4]
    
    Returns array of shape (pk_ellmax//2+1, pk_ellmax//2+1, pk_ellmax//2+1, pk_ellmax//2+1, kbins)
    
    Note: Uses sequential loop to avoid race conditions when multiple coefficients
    share the same (l1,l2,l3,l4) output indices.
    """
    n_nonzero = coeff_indices.shape[0]
    n_grid = len(bin_indices)
    n_ell = pk_ellmax // 2 + 1
    lmax = pk_ellmax
    
    # Output accumulator
    result = np.zeros((n_ell, n_ell, n_ell, n_ell, kbins), dtype=np.float64)
    
    # Process each nonzero coefficient sequentially to avoid race conditions
    for i in range(n_nonzero):
        l1_half = coeff_indices[i, 0]
        l2_half = coeff_indices[i, 1]
        l3_half = coeff_indices[i, 2]
        l4_half = coeff_indices[i, 3]
        l1, l2, l3, l4 = 2*l1_half, 2*l2_half, 2*l3_half, 2*l4_half
        m1 = coeff_indices[i, 4] - l1
        m2 = coeff_indices[i, 5] - l2
        m3 = coeff_indices[i, 6] - l3
        m4 = coeff_indices[i, 7] - l4
        
        # First term: Yk1[l1,m1] * Yk1[l2,m2] for k1
        Yk1_l1m1 = Yk1_flat[l1 * (2*lmax + 1) + m1 + lmax]
        Yk1_l2m2 = Yk1_flat[l2 * (2*lmax + 1) + m2 + lmax]
        Yk1_term = Yk1_l1m1 * Yk1_l2m2
        
        # Accumulate over grid points into k-bins
        for j in range(n_grid):
            if valid_mask[j]:
                k2_bin = bin_indices[j]
                # Yk2[l3,m3] * Yk2[l4,m4] for k2
                Yk2_l3m3 = Yk2_3d[l3, m3 + lmax, j]
                Yk2_l4m4 = Yk2_3d[l4, m4 + lmax, j]
                contrib = window_product[i, j] * Yk1_term * Yk2_l3m3 * Yk2_l4m4
                result[l1_half, l2_half, l3_half, l4_half, k2_bin] += contrib
    
    return result


@numba.njit(fastmath=True, cache=True)
def _compute_cosmic_variance_second(
    bin_indices,        # (n_grid,)
    valid_mask,         # (n_grid,)
    window_values,      # (n_nonzero, n_grid)
    coeff_indices,      # (n_nonzero, 8) - l1/2, l2/2, l3/2, l4/2, m1+l1, m2+l2, m3+l3, m4+l4
    Yk1_flat,           # (n_lm,)
    Yk2_3d,             # (lmax+1, 2*lmax+1, n_grid)
    pk_ellmax,
    kbins
):
    """Compute SECOND cosmic variance contribution: Yk1[l1,m1] * Yk1[l3,m3] * Yk2[l2,m2] * Yk2[l4,m4]
    
    Note: This uses l1,l3 from Yk1 and l2,l4 from Yk2 (different from first term).
    """
    n_nonzero = coeff_indices.shape[0]
    n_grid = len(bin_indices)
    n_ell = pk_ellmax // 2 + 1
    lmax = pk_ellmax
    
    result = np.zeros((n_ell, n_ell, n_ell, n_ell, kbins), dtype=np.float64)
    
    for idx in range(n_nonzero):
        l1_half = coeff_indices[idx, 0]
        l2_half = coeff_indices[idx, 1]
        l3_half = coeff_indices[idx, 2]
        l4_half = coeff_indices[idx, 3]
        l1, l2, l3, l4 = 2*l1_half, 2*l2_half, 2*l3_half, 2*l4_half
        m1 = coeff_indices[idx, 4] - l1
        m2 = coeff_indices[idx, 5] - l2
        m3 = coeff_indices[idx, 6] - l3
        m4 = coeff_indices[idx, 7] - l4
        
        # Second term: Yk1[l1,m1] * Yk1[l3,m3] for k1
        Yk1_l1m1 = Yk1_flat[l1 * (2*lmax + 1) + m1 + lmax]
        Yk1_l3m3 = Yk1_flat[l3 * (2*lmax + 1) + m3 + lmax]
        Yk1_term = Yk1_l1m1 * Yk1_l3m3
        
        for i in range(n_grid):
            if valid_mask[i]:
                k2_bin = bin_indices[i]
                # Yk2[l2,m2] * Yk2[l4,m4] for k2
                Yk2_l2m2 = Yk2_3d[l2, m2 + lmax, i]
                Yk2_l4m4 = Yk2_3d[l4, m4 + lmax, i]
                contrib = window_values[idx, i] * Yk1_term * Yk2_l2m2 * Yk2_l4m4
                result[l1_half, l2_half, l3_half, l4_half, k2_bin] += contrib
    
    return result


@numba.njit(fastmath=True, cache=True)
def _compute_mixed_term_contribution(
    bin_indices,        # (n_grid,)
    valid_mask,         # (n_grid,)
    window_values,      # (n_nonzero, n_grid)
    coeff_indices,      # (n_nonzero, 6) - l1/2, l2/2, l3/2, m1+l1, m2+l2, m3+l3
    Yk1_flat,           # (n_lm,)
    Yk2_3d,             # (lmax+1, 2*lmax+1, n_grid)
    pk_ellmax,
    kbins
):
    """Compute mixed term contribution with fused operations."""
    n_nonzero = coeff_indices.shape[0]
    n_grid = len(bin_indices)
    n_ell = pk_ellmax // 2 + 1
    lmax = pk_ellmax
    
    result = np.zeros((n_ell, n_ell, n_ell, kbins), dtype=np.float64)
    
    for idx in range(n_nonzero):
        l1_half = coeff_indices[idx, 0]
        l2_half = coeff_indices[idx, 1]
        l3_half = coeff_indices[idx, 2]
        l1, l2, l3 = 2*l1_half, 2*l2_half, 2*l3_half
        m1 = coeff_indices[idx, 3] - l1
        m2 = coeff_indices[idx, 4] - l2
        m3 = coeff_indices[idx, 5] - l3
        
        Yk1_l1m1 = Yk1_flat[l1 * (2*lmax + 1) + m1 + lmax]
        
        for i in range(n_grid):
            if valid_mask[i]:
                k2_bin = bin_indices[i]
                Yk2_l2m2 = Yk2_3d[l2, m2 + lmax, i]
                Yk2_l3m3 = Yk2_3d[l3, m3 + lmax, i]
                contrib = window_values[idx, i] * Yk1_l1m1 * Yk2_l2m2 * Yk2_l3m3
                result[l1_half, l2_half, l3_half, k2_bin] += contrib
    
    return result


@numba.njit(fastmath=True, cache=True)
def _compute_shotnoise_contribution(
    bin_indices,        # (n_grid,)
    valid_mask,         # (n_grid,)
    window_values,      # (n_nonzero, n_grid)
    coeff_indices,      # (n_nonzero, 4) - l1/2, l2/2, m1+l1, m2+l2
    Yk1_flat,           # (n_lm,)
    Yk2_3d,             # (lmax+1, 2*lmax+1, n_grid)
    pk_ellmax,
    kbins
):
    """Compute shotnoise contribution with fused operations."""
    n_nonzero = coeff_indices.shape[0]
    n_grid = len(bin_indices)
    n_ell = pk_ellmax // 2 + 1
    lmax = pk_ellmax
    
    result = np.zeros((n_ell, n_ell, kbins), dtype=np.float64)
    
    for idx in range(n_nonzero):
        l1_half = coeff_indices[idx, 0]
        l2_half = coeff_indices[idx, 1]
        l1, l2 = 2*l1_half, 2*l2_half
        m1 = coeff_indices[idx, 2] - l1
        m2 = coeff_indices[idx, 3] - l2
        
        Yk1_l1m1 = Yk1_flat[l1 * (2*lmax + 1) + m1 + lmax]
        
        for i in range(n_grid):
            if valid_mask[i]:
                k2_bin = bin_indices[i]
                Yk2_l2m2 = Yk2_3d[l2, m2 + lmax, i]
                contrib = window_values[idx, i] * Yk1_l1m1 * Yk2_l2m2
                result[l1_half, l2_half, k2_bin] += contrib
    
    return result