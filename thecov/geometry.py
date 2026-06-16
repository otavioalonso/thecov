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
import logging

import numpy as np

import mockfactory

from . import base


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


class SurveyGeometry(Geometry, base.LinearBinning):

    def __init__(self, randoms_pos, randoms_nz=None, randoms_nz_weight=1, randoms_weight=1, pk_ellmax=4, mask_ellmax=4):

        base.LinearBinning.__init__(self)

        self.logger = logging.getLogger('SurveyGeometry')

        self.randoms = Randoms(pos=randoms_pos,
                               nz=randoms_nz,
                               nz_weight=randoms_nz_weight,
                               weight=randoms_weight)
        
        self.pk_ellmax = pk_ellmax
        self.mask_ellmax = mask_ellmax

        self.window_matrix = None

    def compute_window_kernels(self):
        
        # window matrix: [power_configs, ellm, nu1, nu2, k1, k2]
        # window product: a1, a2, lma, b1, b2, lmb
        product = np.einsum('abcij,xyzij->bcayzxij', self.window_matrix[0], self.window_matrix[0])
        coefficients = self.get_cosmic_variance_gaunt_coefficients(pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax)
        
        # shape_out = l1, l2, L1, L2
        self.window_kernel = coefficients @ product
        

    def compute_window_matrix(self, nchunks=4, nthreads=None):
        order  = np.random.permutation(len(self.randoms.nz))
        chunks = [self.randoms[idx] for idx in np.array_split(order, nchunks)]

        self.logger.info(f"Computing window matrices using {len(chunks)} chunks of {len(chunks[0])} randoms processed by {nthreads} threads.")

        import multiprocessing
        from functools import partial
        from tqdm import tqdm

        with multiprocessing.get_context('spawn').Pool(processes=nthreads) as pool:
            all_chunks = np.stack(list(tqdm(
                pool.imap_unordered(
                    partial(self._compute_window_matrix, self.kedges, ellmax=self.pk_ellmax), chunks),
                total=nchunks, desc='Window matrix chunks')))

        result         = all_chunks.sum(axis=0)
        relative_error = (np.sqrt(nchunks) * all_chunks.std(axis=0) /
                          np.abs(result).clip(1e-30))

        self.logger.info(f'Max relative error: {relative_error.max():.3e}')

        self.window_matrix = result

        return result, relative_error

    # Computes all window matrix elements for a given array of randoms.
    # Can be run with a subset of randoms and aggregated
    @staticmethod
    def _compute_window_matrix(kedges, randoms, ellmax=4):
        import os
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
        from thecov.math import get_real_Ylm, spherical_bessel

        power_configs = [(2, 2)]  #, (1, 2)]
        nus   = list(range(0, ellmax + 1, 2))
        ellms = [(ell, m) for ell in range(0, ellmax + 1, 2) for m in range(-ell, ell + 1)]

        r2      = (randoms.pos**2).sum(axis=1)                                            # [N]
        bessels = 4*np.pi *  np.stack([spherical_bessel(nu, r2, kedges) * np.real((- 1j)**(nu)) for nu in nus])      # [nu, k, N]
        Ylm     = np.stack([get_real_Ylm(ell, m)(*randoms.pos.T) for ell, m in ellms])    # [ellm, N]


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

    @staticmethod
    def get_cosmic_variance_gaunt_coefficients(mask_ellmax=MASK_ELL_MAX, pk_ellmax=PK_ELL_MAX, cache_dir=None):
        """Calculates all relevant Gaunt coefficients for the cosmic variance term, or loads them from file"""

        logger = logging.getLogger('SurveyGeometry')

        # Load mask coupling Gaunt coefficients if cache exists, otherwise compute them
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
            logger.info(f'Using cache directory: {cache_dir}')

        filename = os.path.join(cache_dir, f"cosmic_variance_coefficients_{pk_ellmax:d}_{mask_ellmax:d}.npz")

        if os.path.exists(filename):
            logger.info(f'Loading first cosmic variance Gaunt coefficients from cache: {filename}')
            return base.SparseNDArray.load(filename)
        else:
            logger.info(f'Computing first cosmic variance Gaunt coefficients (pk_ellmax={pk_ellmax}, mask_ellmax={mask_ellmax})...')

            from thecov.utils import ellmiter, elliter, n_ellm
            from multiprocessing import Pool, cpu_count
            from tqdm import tqdm

            # shape_out = l1, l2, L1, L2
            # shape_in =  a1, a2, lma, b1, b2, lmb
            shape_out = 4*[pk_ellmax//2 + 1]
            shape_in = (2*[mask_ellmax//2+1] + [n_ellm(mask_ellmax)] +
                        2*[mask_ellmax//2+1] + [n_ellm(mask_ellmax)])
            
            gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)

            outer_args = [
                (l1, l2, L1, L2, a1, a2, la, ma, b1, b2, lb, mb, mask_ellmax)
                for l1, l2, L1, L2 in elliter(pk_ellmax, 4)
                for a1, a2, b1, b2 in elliter(mask_ellmax, 4)
                for la, lb, ma, mb in ellmiter(mask_ellmax, 2)
            ]
            n = len(outer_args)
            chunksize = max(1, n // (cpu_count() * 20))

            logger.info(f'Computing Gaunt coefficients: {n:,} tasks across {cpu_count()} CPUs '
                        f'(chunksize={chunksize})...')

            with Pool() as pool:
                for result in tqdm(pool.imap_unordered(_gaunt_row_worker, outer_args, chunksize=chunksize),
                                   total=n, desc='Gaunt coefficients'):
                    if result is not None:
                        index, val = result
                        gaunt_coefficients[index] = val

            gaunt_coefficients.save(filename)

            return gaunt_coefficients

    def normalization(self, nbar_power, weight_power):
        return (self._randoms_nz**(nbar_power-1) * \
                self._randoms_weight**(weight_power) * \
                self._randoms_nz_weight).sum().tolist()


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


def _gaunt_row_worker(args):
    """Compute one Gaunt coefficient entry.

    Performs the m-sum for a single (l1,l2,L1,L2, a1,a2,la,ma, b1,b2,lb,mb)
    combination. Returns (index, val) if non-zero, else None.
    Must be at module level to be picklable by multiprocessing.
    """
    l1, l2, L1, L2, a1, a2, la, ma, b1, b2, lb, mb, mask_ellmax = args
    from thecov.utils import ellm_to_index, miter
    from thecov.math import gaunt

    val = 0.0
    for m1, m2, M1, M2, nu1, nu2, rho1, rho2 in miter(l1, l2, L1, L2, a1, a2, b1, b2):
        val += (gaunt((l2, L1, a1, a2, la), (m2, M1, nu1, nu2, ma)) *
                gaunt((l1, L2, b1, b2, lb), (m1, M2, rho1, rho2, mb)) *
                gaunt((l1, L1, a1, b1), (m1, M1, nu1, rho1)) *
                gaunt((l2, L2, a2, b2), (m2, M2, nu2, rho2)) +
                \
                gaunt((L1, a1, a2, la), (M1, nu1, nu2, ma)) *
                gaunt((l1, l2, L2, b1, b2, lb), (m1, m2, M2, rho1, rho2, mb)) *
                gaunt((l1, L1, a1, b1), (m1, M1, nu1, rho1)) *
                gaunt((l2, L2, a2, b2), (m2, M2, nu2, rho2)))

    if val == 0.0:
        return None

    index = (l1//2, l2//2, L1//2, L2//2,
             a1//2, a2//2, ellm_to_index(la, ma, mask_ellmax),
             b1//2, b2//2, ellm_to_index(lb, mb, mask_ellmax))
    
    return index, val
