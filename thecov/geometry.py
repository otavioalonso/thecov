import logging
import numpy as np
from . import base, math, utils
from functools import lru_cache

MASK_ELL_MAX = 4 # max = 3*PK_ELL_MAX
PK_ELL_MAX = 4

class Randoms():
    def __init__(self, pos, nz=None, nz_weight=1, weight=1):
        self.logger = logging.getLogger('Randoms')

        self.pos = pos
        self.nz = nz
        self.nz_weight = nz_weight
        self.weight = weight

        # Set nz (number density), estimating it if not provided
        if nz is None:
            self.logger.warning('nz not provided. Estimating it with RedshiftDensityInterpolator.')
            import healpy as hp
            import mockfactory
            nside = 512
            distance = np.sqrt(np.sum(self.pos**2, axis=-1))
            xyz = self.pos / distance[:, None]
            hpixel = hp.vec2pix(nside, *xyz.T)
            unique_hpixels = np.unique(hpixel)
            fsky = len(unique_hpixels) / hp.nside2npix(nside)
            self.logger.warning(f'fsky = {fsky:.3f}')
            self.logger.info(f'fsky estimated from randoms: {fsky:.3f}')
            nbar = mockfactory.RedshiftDensityInterpolator(z=distance, weights=self.nz_weight, fsky=fsky)
            nz = nbar(distance)

        self.nz = nz

    def __getitem__(self, index):
        def _idx(x):
            return x[index] if isinstance(x, np.ndarray) else x
        return Randoms(pos=self.pos[index],
                       nz=self.nz[index] if self.nz is not None else None,
                       nz_weight=_idx(self.nz_weight),
                       weight=_idx(self.weight))

    # This method allows len() to work on your object
    def __len__(self):
        return self.pos.shape[0]

    def __add__(self, other):
        if not isinstance(other, Randoms):
            raise TypeError(f"Cannot add Randoms with {type(other)}")
        return Randoms(pos=np.concatenate([self.pos, other.pos]),
                       nz=np.concatenate([self.nz, other.nz]) if self.nz is not None and other.nz is not None else None,
                       nz_weight=np.concatenate([self.nz_weight, other.nz_weight]),
                       weight=np.concatenate([self.weight, other.weight]))

class SurveyGeometry(base.LinearBinning):

    def __init__(self):
        base.LinearBinning.__init__(self)

        self.logger = logging.getLogger('SurveyGeometry')

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

class SingleTracerSurveyGeometry(SurveyGeometry):

    def __init__(self, randoms_pos, randoms_nz=None, randoms_nz_weight=1, randoms_weight=1, ellmax=4):

        SurveyGeometry.__init__(self)

        self.randoms = Randoms(pos=randoms_pos,
                               nz=randoms_nz,
                               nz_weight=randoms_nz_weight,
                               weight=randoms_weight)
        
        self.ellmax = ellmax

        self._window_matrix = None

    @property
    def shotnoise_window_matrix(self):
        return self._window_matrix[1]

    @property
    def window_matrix(self):
        return self._window_matrix[0]

    def compute_window_matrix(self, nchunks=4, nthreads=None):
        order  = np.random.permutation(self.randoms.pos.shape[1])
        chunks = [self.randoms[idx] for idx in np.array_split(order, nchunks)]

        self.logger.info(f"Computing window matrices using {len(chunks)} chunks of {len(chunks[0])} randoms processed by {nthreads} threads.")

        import multiprocessing
        from functools import partial
        from tqdm import tqdm

        with multiprocessing.get_context('spawn').Pool(processes=nthreads) as pool:
            all_chunks = np.stack(list(tqdm(
                pool.imap_unordered(
                    partial(self._compute_window_matrix, self.kedges, ellmax=self.ellmax), chunks),
                total=nchunks, desc='Window matrix chunks')))

        result         = all_chunks.sum(axis=0)
        relative_error = (np.sqrt(nchunks) * all_chunks.std(axis=0) /
                          np.abs(result).clip(1e-30))

        self.logger.info(f'Max relative error: {relative_error.max():.3e}')

        self._window_matrix = result
        self._window_matrix_error = relative_error

    # Computes all window matrix elements for a given array of randoms.
    # Can be run with a subset of randoms and aggregated
    @staticmethod
    def _compute_window_matrix(kedges, randoms, ellmax=4):
        power_configs = [(2, 2), (1, 2)]
        nus   = list(range(0, ellmax + 1, 2))
        ellms = [(ell, m) for ell in range(0, ellmax + 1, 2) for m in range(-ell, ell + 1)]

        r2      = (randoms.pos**2).sum(axis=1)                                            # [N]
        bessels = 4*np.pi *  np.stack([math.spherical_bessel(nu, r2, kedges) * np.real((- 1j)**(nu)) for nu in nus])      # [nu, k, N]
        Ylm     = np.stack([math.Ylm(ell, m)(*randoms.pos.T) for ell, m in ellms])    # [ellm, N]


        # Compute all outer-product sums at once.
        # out[c, i_ellm, a, b, i, j] = Σ_p  w_c[p] · Y[i_ellm,p] · B[a,i,p] · B[b,j,p]
        results = np.stack([
            np.stack([
                np.einsum('aip,bjp->abij',
                          bessels * (randoms.nz_weight * randoms.nz**(nbar_power-1) * randoms.weight**weight_power * Ylm[i_ellm])[None, None, :],
                          bessels)
                for i_ellm in range(len(ellms))
            ])
            for nbar_power, weight_power in power_configs
        ])
        # shape: [power_configs, ellm, nu1, nu2, k1, k2]
        return results

    def normalization(self, nbar_power, weight_power):
        return (self._randoms_nz**(nbar_power-1) * \
                self._randoms_weight**(weight_power) * \
                self._randoms_nz_weight).sum().tolist()


class MultiTracerSurveyGeometry(SurveyGeometry):

    def __init__(self, randoms_pos1, randoms_pos2, randoms_nz1=None, randoms_nz2=None, randoms_nz_weight1=1, randoms_nz_weight2=1, randoms_weight1=1, randoms_weight2=1, ellmax=4):

        SurveyGeometry.__init__(self)

        self.randoms1 = Randoms(pos=randoms_pos1,
                                nz=randoms_nz1,
                                nz_weight=randoms_nz_weight1,
                                weight=randoms_weight1)

        self.randoms2 = Randoms(pos=randoms_pos2,
                                nz=randoms_nz2,
                                nz_weight=randoms_nz_weight2,
                                weight=randoms_weight2)

        self.ellmax = ellmax

        self._window_matrix = None

    @lru_cache
    def compute_window_profile(self, ell, m, nmesh=512, boxsize=None, boxpad=None):
        """Compute the window radial profile for nbar**nbar_power * weight**weight_power * Y_lm.

        Build a mesh from the random catalog weighted by nbar and weight powers and
        multiplied by the real spherical harmonic Y_lm(r̂). The mesh is compensated
        after gridding and collapsed into a radial profile by binning |mesh| on
        spherical shells.

        Parameters
        ----------
        nbar_power : int
            Exponent applied to the local number density (randoms_nz).
        weight_power : int
            Exponent applied to the randoms weight (randoms_weight).
        ell : int
            Degree of the spherical harmonic.
        m : int
            Order of the spherical harmonic.
        rbins : array_like, optional
            Radial bin edges used to accumulate the profile. If None, a default
            set of bins is created adaptively using a greedy minimum-weight scheme
            with an automatically determined minimum bin width.

        Returns
        -------
        W : ndarray
            Radial profile (binned sum of |mesh|) evaluated on rbins.
        rbins : ndarray
            The radial bin edges used to compute W.
        """
        from pypower import CatalogMesh

        assert ell >= 0, "ell must be non-negative"
        assert abs(m) <= ell, "m must be less than or equal to ell"

        Ylm = math.Ylm(ell, m)

        self.logger.info(f'Computing mesh (ell={ell}, m={m})')

        if boxpad is None and boxsize is None:
            boxpad = 1.4

        # Create CatalogMesh just to determine box properties
        mesh = CatalogMesh(data_positions=np.concatenate([self.randoms1.pos, self.randoms2.pos]), 
                           position_type='pos', nmesh=nmesh, boxsize=boxsize, boxpad=boxpad)

        # Mesh for tracer A
        mesh = CatalogMesh(data_positions=self.randoms1.pos, data_weights=self.randoms1.nz_weight*self.randoms1.weight * Ylm(*self.randoms1.pos.T),
                           position_type='pos', nmesh=nmesh, boxsize=mesh.boxsize, boxcenter=mesh.boxcenter,
                           dtype='c16', **{'interlacing': 3, 'resampler': 'tsc'}).to_mesh(compensate=True).value
        
        # Multiply by mesh for tracer B
        mesh *= CatalogMesh(data_positions=self.randoms2.pos, data_weights=self.randoms2.nz_weight*self.randoms2.weight,
                            position_type='pos', nmesh=nmesh, boxsize=mesh.boxsize, boxcenter=mesh.boxcenter,
                            dtype='c16', **{'interlacing': 3, 'resampler': 'tsc'}).to_mesh(compensate=True).value

        self.logger.info(f'Binning power...')

        W, rbins = utils.bin(
            r=np.sqrt(sum((x.real**2 for x in mesh.x))).ravel(),
            mesh=np.abs(mesh).ravel(),
            rbins=rbins)

        return W, rbins

    def compute_window_matrix(self, nthreads=None):
        import multiprocessing
        from functools import partial
        from tqdm import tqdm

        # output: [ellm, nu1, nu2, k1, k2]
        nus = list(range(0, self.ellmax + 1, 2))
        ellms = [(ell, m) for ell in range(0, self.ellmax + 1, 2) for m in range(-ell, ell + 1)]
        
        # Initialize output array
        # Shape: [ellm, nu1, nu2, k1, k2]
        # Where k1 and k2 have length = len(self.kedges) - 1
        nk = len(self.kedges) - 1
        result = np.zeros((len(ellms), len(nus), len(nus), nk, nk))
        
        self.logger.info(f"Computing window matrix for MultiTracerSurveyGeometry using {multiprocessing.cpu_count() if nthreads is None else nthreads} threads...")
        
        # We parallelize over (ell, m). Each task computes the mesh and then its associated (nu1, nu2) bessels.
        # This prevents redundant mesh creations while parallelizing the heavy work.
        tasks = []
        for i_ellm, (ell, m) in enumerate(ellms):
            tasks.append((i_ellm, ell, m, nus, self.kedges))

        worker = partial(self._multitracer_window_profile_worker, obj=self)

        with multiprocessing.get_context('spawn').Pool(processes=nthreads) as pool:
            for i_ellm, layer_result in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="Window matrix components"):
                result[i_ellm] = layer_result

        self._window_matrix = result
    
    @staticmethod
    def _multitracer_window_profile_worker(task_args, obj):
        """Worker function for MultiTracerSurveyGeometry.compute_window_matrix."""
        i_ellm, ell, m, nus, kedges = task_args
        import numpy as np
        from thecov import math

        nk = len(kedges) - 1
        result = np.zeros((len(nus), len(nus), nk, nk))

        # 1. Compute mesh and profile for this (ell, m) 
        # (Since this method may use heavy catalog mesh code, we evaluate it inside the worker)
        W, rbins = obj.compute_window_profile(ell=ell, m=m)
        
        # 2. Compute the double spherical bessel transformations for all (nu1, nu2)
        for i_nu1, nu1 in enumerate(nus):
            for i_nu2, nu2 in enumerate(nus):
                result[i_nu1, i_nu2] = \
                    math.double_spherical_bessel(xbins=rbins, W=W, kedges=kedges, l1=nu1, l2=nu2)
                
        return i_ellm, result

    @property
    def window_matrix(self):
        return self._window_matrix