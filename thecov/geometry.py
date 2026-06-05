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

import numpy as np

import mockfactory
from pypower import CatalogMesh

from . import base, utils, math


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

    def __init__(self, randoms_pos, randoms_nz=None, randoms_nz_weight=None, randoms_weight=None, nmesh=None, cellsize=None, boxsize=None, boxpad=2., kmax=0.02, **kwargs):

        base.LinearBinning.__init__(self)

        self.logger = logging.getLogger('SurveyGeometry')

        self._kmax = kmax

        self.window_matrix = None
        self.window_matrix_error = None

        self._resume_file = None
        self._randoms_pos = randoms_pos

        # Set randoms_nz_weight (replaces WEIGHT), defaulting to 1 if not provided
        if randoms_nz_weight is None:
            self.logger.warning('randoms_nz_weight not provided. Setting it to 1.')
            self._randoms_nz_weight = np.ones(len(self._randoms_pos), dtype='f8')
        else:
            self._randoms_nz_weight = randoms_nz_weight

        # Set randoms_weight (replaces WEIGHT_FKP), defaulting to 1 if not provided
        if randoms_weight is None:
            self._randoms_weight = np.ones(len(self._randoms_pos), dtype='f8')
        else:
            self._randoms_weight = randoms_weight

        # Set randoms_nz (number density), estimating it if not provided
        if randoms_nz is None:
            self.logger.warning('randoms_nz not provided. Estimating it with RedshiftDensityInterpolator.')
            import healpy as hp
            nside = 512
            distance = np.sqrt(np.sum(self._randoms_pos**2, axis=-1))
            xyz = self._randoms_pos / distance[:, None]
            hpixel = hp.vec2pix(nside, *xyz.T)
            unique_hpixels = np.unique(hpixel)
            fsky = len(unique_hpixels) / hp.nside2npix(nside)
            self.logger.warning(f'fsky = {fsky:.3f}')
            self.logger.info(f'fsky estimated from randoms: {fsky:.3f}')
            nbar = mockfactory.RedshiftDensityInterpolator(z=distance, weights=self._randoms_nz_weight, fsky=fsky)
            self._randoms_nz = nbar(distance)
        else:
            self._randoms_nz = randoms_nz

        # Check if the randoms have nmesh and cellsize, otherwise set them using the kmax parameter
        if nmesh is None and cellsize is None:
            # Pick value that will give at least k_mask = kmax_window in the FFTs
            cellsize = np.pi / kmax / (1. + 1e-9)

        self._mesh = CatalogMesh(data_positions=self._randoms_pos, data_weights=self._randoms_nz_weight,
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

    @base.cache
    def compute_window_profile(self, nbar_power, weight_power, ell, m, rbins=None):
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

        assert ell >= 0, "ell must be non-negative"
        assert abs(m) <= ell, "m must be less than or equal to ell"

        Ylm = math.get_real_Ylm(ell, m)

        self.logger.info(f'Computing mesh nbar^{nbar_power} * weight^{weight_power} (ell={ell}, m={m})')
        start = time.time()

        mesh = self._mesh.copy(
            data_positions=self._randoms_pos,
            data_weights=self._randoms_nz_weight * self._randoms_nz**(nbar_power-1)*self._randoms_weight**(weight_power) * Ylm(*self._randoms_pos.T),
            position_type='pos',
        ).to_mesh(compensate=True)

        self.logger.info(f'Mesh computed in {time.time() - start:.0f} seconds.')
        
        start = time.time()
        self.logger.info(f'Binning power...')

        W, rbins = math.bin(
            r=np.sqrt(sum((x.real**2 for x in mesh.x))).ravel(),
            mesh=np.abs(mesh.value).ravel(),
            rbins=rbins)

        self.logger.info(f'Power binned in {time.time() - start:.0f} seconds.')

        return W, rbins

    @base.cache
    def compute_window_matrix(self, pk_ellmax=PK_ELL_MAX, mask_ellmax=MASK_ELL_MAX):
        '''Computes the window matrix using multiprocessing with shared memory.

        Parameters
        ----------
        pk_ellmax : int, optional
            Maximum ell for the power spectrum multipoles. Default is PK_ELL_MAX.
        mask_ellmax : int, optional
            Maximum ell for the mask multipoles. Default is MASK_ELL_MAX.'''

        self.get_first_cosmic_variance_gaunt_coefficients()

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

            # shape_out = l1, l2, l3, l4, m1, m2, m3, m4
            # shape_in =  la, lb, ma, mb
            shape_out = 4*[pk_ellmax//2 + 1] + 4*[2*pk_ellmax + 1]
            shape_in = 2*[mask_ellmax//2 + 1] + 2*[2*mask_ellmax + 1]
            gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)


    def normalization(self, nbar_power, weight_power):
        return (self._randoms_nz**(nbar_power-1) * \
                self._randoms_weight**(weight_power) * \
                self._randoms_nz_weight).sum().tolist()

    @property
    def knyquist(self):
        return np.pi * self.nmesh / self.boxsize
    
    @property
    def kfun(self):
        return 2 * np.pi / self.boxsize
    
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

