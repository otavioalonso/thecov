"""Module containing classes for creating the window function to be used in calculating the Gaussian covariance term.

Classes
-------
SurveyWindow
SurveyGeometry
"""

import logging
#logging.basicConfig(level = logging.INFO)
logging.basicConfig(level = logging.INFO)

import numpy as np
import os, time, sys
import itertools as itt

from tqdm import tqdm as shell_tqdm
from mpi4py import MPI
import mockfactory
from pypower import CatalogMesh

# from scipy.integrate import lebedev_rule
import functools

from . import base, utils, math, binning

MASK_ELL_MAX = 12
PK_ELL_MAX = 4

__all__ = ['SurveyWindow', 'SurveyGeometry']

class SurveyWindow(base.BaseClass):

    def __init__(self, randoms1, alpha1, randoms2=None, alpha2=None, mpi_comm=MPI.COMM_WORLD,
                 nmesh=None, cellsize=None, boxsize=None, boxpad=2., kmin=0.0, kmax=0.02, 
                 dk=None, binning_type="linear", shotnoise=False, **kwargs):

        super().__init__()

        self.comm = mpi_comm
        self.rank = mpi_comm.Get_rank()
        self.size = mpi_comm.Get_size()

        if binning_type == "linear":
            self.k_binning = binning.LinearBinning(kmin, kmax, dk)
        elif binning_type == "log":
            self.k_binning = binning.LogBinning(kmin, kmax, dk)
        else:
            raise ValueError("binning_type must be either 'linear' or 'log'")

        self.logger = logging.getLogger('SurveyWindow')
        self.logger.setLevel(logging.INFO)
        self.tqdm = shell_tqdm

        self._is_shotnoise = shotnoise
        self.kmax = kmax
        self.dk = dk
        self.alpha1, self.alpha2 = alpha1, alpha2

        self.mesh1, self.shotnoise_mesh1 = self._create_mesh(
            randoms=randoms1,
            alpha=alpha1,
            nmesh=nmesh,
            cellsize=cellsize,
            boxsize=boxsize,
            boxpad=boxpad,
            kmax=kmax,
            shotnoise=shotnoise
        )
        self.boxsize = self.mesh1.boxsize[0]
        self.nmesh = self.mesh1.nmesh[0]

        if randoms2 is not None:
            assert alpha2 is not None, "If randoms2 is provided, alpha2 must also be provided."

            self.mesh2, self.shotnoise_mesh2 = self._create_mesh(
                randoms=randoms2,
                alpha=alpha2,
                nmesh=nmesh,
                cellsize=cellsize,
                boxsize=boxsize,
                boxpad=boxpad,
                kmax=kmax,
                shotnoise=shotnoise
            )

            self.boxsize = max(self.mesh1.boxsize[0], self.mesh2.boxsize[0])
            self.nmesh = max(self.mesh1.nmesh[0], self.mesh2.nmesh[0])
            
            self.mesh1._set_box(nmesh=self.nmesh, boxsize=self.boxsize, wrap=False)
            self.mesh2._set_box(nmesh=self.nmesh, boxsize=self.boxsize, wrap=False)
            if shotnoise:
                self.shotnoise_mesh1._set_box(nmesh=self.nmesh, boxsize=self.boxsize, wrap=False)
                self.shotnoise_mesh2._set_box(nmesh=self.nmesh, boxsize=self.boxsize, wrap=False)

        if self.rank == 0:
            self.logger.info(f'Using box size {self.boxsize}, box center {self.mesh1.boxcenter} and nmesh {self.nmesh}.')
            self.logger.info(f'Fundamental wavenumber of window meshes = {self.kfun}.')
            self.logger.info(f'Nyquist wavenumber of window meshes = {self.knyquist}.')

            if kmax is not None and self.knyquist < kmax:
                self.logger.warning(f'Nyquist wavelength {self.knyquist} smaller than required window kmax = {kmax}.')

            self.logger.info(f'Average of {self.mesh1.data_size / self.nmesh**3} objects per voxel.')

        # Initialize rebin parameters
        self._rebin_parameters(dk, kmax)

    def _create_mesh(self, randoms, alpha, nmesh, cellsize, boxsize, boxpad, kmax, shotnoise):
        """Parse the randoms into a mesh, filling in missing information as needed."""        

        start_time = time.time()
        # Check if the randoms have nmesh and cellsize, otherwise set them using the kmax parameter
        if nmesh is None and cellsize is None:
            # Pick value that will give at least k_mask = kmax_window in the FFTs
            self.cellsize = np.pi / kmax / (1. + 1e-9)
        if boxsize is None:
            self.logger.debug("boxsize not provided, estimating from randoms' positions.")
            boxsize_rank = max(np.amax(randoms['POSITION'], axis=0) - np.amin(randoms['POSITION'], axis=0))
            boxsize = self.comm.allreduce(boxsize_rank, op=MPI.MAX) * 1.05

        if self.rank == 0: self.logger.info("Creating survey mesh W...")
        mesh = CatalogMesh(
            data_positions=randoms['POSITION'],
            data_weights=randoms["NZ"] * randoms['WEIGHT']**2 * alpha,
            position_type='pos',
            nmesh=nmesh,
            cellsize=cellsize,
            boxsize=boxsize,
            boxpad=boxpad,
            dtype='c16',
            mpicomm=self.comm,
            **{'interlacing': 3, 'resampler': 'tsc'}
        )
        self.comm.Barrier()
        if shotnoise==True:
            if self.rank == 0: self.logger.info("Creating shotnoise mesh S...")
            shotnoise_mesh = CatalogMesh(
                data_positions=randoms['POSITION'],
                data_weights=randoms['WEIGHT']**2 * alpha,
                position_type='pos',
                nmesh=nmesh,
                cellsize=cellsize,
                boxsize=boxsize,
                boxpad=boxpad,
                dtype='c16',
                mpicomm=self.comm,
                **{'interlacing': 3, 'resampler': 'tsc'}
            )
        else:
            shotnoise_mesh = None

        self.comm.Barrier()
        if self.rank == 0: self.logger.info(f'Created meshes in {time.time() - start_time:.2f} seconds.')
        return mesh, shotnoise_mesh
        
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
    def ikgrid(self):
        """Grid of wavenumber indices."""
        ikgrid = []
        nmesh = self.knmesh if hasattr(self, 'knmesh') else self.nmesh
        for _ in range(3):
            iik = np.arange(nmesh)
            iik[iik >= nmesh // 2] -= nmesh
            ikgrid.append(iik)
        return ikgrid
    
    def _rebin_parameters(self, dk, kmax):

        # If dk and kmax are provided, they determine the target boxsize and nmesh
        target_boxsize = 2*np.pi/dk
        target_nmesh = int(np.ceil(target_boxsize * kmax / np.pi))

        # Trim the mesh to achieve the target boxsize
        trim_to_nmesh = int(np.ceil(target_boxsize/self.boxsize * self.nmesh))
        
        # Rebin mesh to obtain the target nmesh
        rebin_factor = trim_to_nmesh//target_nmesh

        if rebin_factor == 0:
            self.logger.error(f"trim_to_nmesh ({trim_to_nmesh}) smaller than target mesh {target_nmesh} with the given values of dk ({dk}) and kmax ({kmax})! Try increasing nmesh.")
            self.logger.error(f"HINT: minimum mesh size for this configuration is {utils.get_minimum_mesh_size(dk, kmax, self.boxsize)}")
            raise ZeroDivisionError

        # Ensure that trim_to_nmesh is a multiple of rebin_factor
        if (trim_to_nmesh % rebin_factor) != 0:
            trim_to_nmesh += rebin_factor - (trim_to_nmesh % rebin_factor)

        self.kboxsize = trim_to_nmesh/self.nmesh * self.boxsize
        self.knmesh = trim_to_nmesh//rebin_factor

        return trim_to_nmesh, rebin_factor

    # @staticmethod
    # def _shotnoise_mesh(mesh, randoms, alpha):
    #     """Compute the shotnoise mesh S_AB = nbar * fkp^2."""

    #     return mesh.clone(
    #         data_positions=randoms['POSITION'],
    #         data_weights=randoms['WEIGHT_FKP']**2 * randoms[f'WEIGHT'] * alpha,
    #         position_type='pos',
    #     ).to_mesh(compensate=True)

    @functools.cache
    def compute_mesh(self, ell:int, m:int, mesh_1:str="W", mesh_2:str=None, fourier=True, threshold=None):
        """Compute the product of meshes and multiply by real Ylm evaluated at the same coordinates.

        Args:
            ell (int): Degree of the spherical harmonic.
            m (int): Order of the spherical harmonic.
            mesh_1 (str): Which mesh to use for the first window. Options are "W" for the original mesh and "S" for the shotnoise mesh. Default is "W".
            mesh_2 (str): Which mesh to use for the second window. Options are "W" for the original mesh and "S" for the shotnoise mesh. Default is None.
            fourier (bool, optional): If True, the Fourier transform of the mesh is returned. Default is False.
            threshold (float, optional): If provided, values in the resulting mesh below this threshold are set to zero to save memory. Default is None.

        Returns:
            np.ndarray: mesh * Ylm(ell, m) with shape: [nmesh, nmesh, nmesh]
        """

        assert ell >= 0, "ell must be non-negative"
        assert abs(m) <= ell, "m must be less than or equal to ell"

        time_start = time.time()
        # Initialize the result mesh
        if mesh_1 == "S":
            mesh_to_clone = self.shotnoise_mesh1
        else:
            mesh_to_clone = self.mesh1

        Ylm = math.get_real_Ylm(ell, m)
        unit_positions = mesh_to_clone.data_positions / np.sqrt(np.sum(mesh_to_clone.data_positions**2, axis=-1))[:, None]

        self.comm.Barrier()
        result = mesh_to_clone.clone(
                data_positions=mesh_to_clone.data_positions,
                data_weights=mesh_to_clone.data_weights*Ylm(*unit_positions.T),
                position_type='pos',
                mpicomm=self.comm, mpiroot = 0
            ).to_mesh(compensate=True)

        if mesh_2 is not None and hasattr(self, 'mesh2'):
            if mesh_2 == "S":
                result *= self.shotnoise_mesh2.to_mesh(compensate=True)
            else:
                result *= self.mesh2.to_mesh(compensate=True)
        elif mesh_2 is not None and not hasattr(self, 'mesh2'):
            if mesh_2 == "S":
                result *= self.shotnoise_mesh1.to_mesh(compensate=True)
            else:
                result *= self.mesh1.to_mesh(compensate=True)

        # pmesh fft convention is F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r); let us correct it here
        # result = result.r2c() * self.nmesh**3 if fourier else result

        if self.rank == 0: self.logger.info(f"Mesh computation with Ylm ({ell}, {m}) done in {time.time() - time_start:.2f} seconds")

        # element-wise addition and send to root rank
        if hasattr(result, 'value'):
            result_combined = utils.gather_field_to_root(result, root=0)
        else:
            full_shape = tuple(int(n) for n in result.Nmesh)
            result_combined = np.zeros(full_shape, dtype=result.dtype)
        self.comm.Barrier()

        # trim mesh to desired size and rebin if needed
        if self.rank == 0:
            if self.dk is not None and self.kmax is not None:
                trim_to_nmesh, rebin_factor = self._rebin_parameters(self.dk, self.kmax)

                if trim_to_nmesh <= self.nmesh and self.kboxsize <= self.boxsize:
                    result_combined = result_combined[:trim_to_nmesh, :trim_to_nmesh, :trim_to_nmesh]
                    self.logger.info(f"Trimmed mesh from {self.nmesh} to {trim_to_nmesh} and boxsize from {self.boxsize:.0f} to {self.kboxsize:.0f}.")

                    # Sum mesh values in the new mesh
                    if rebin_factor > 1:
                        result_combined = result_combined.reshape((self.knmesh, rebin_factor, self.knmesh, rebin_factor, self.knmesh, rebin_factor)).sum(axis=(1, 3, 5))

                    self.logger.info(f"Rebinned mesh from {trim_to_nmesh} to {result_combined.shape[0]} with factor {rebin_factor}.")

            # pmesh fft convention is F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r); let us correct it here
            if fourier:
                result_combined *= self.knmesh**3
                time_start = time.time()
                result_combined = np.fft.fftn(result_combined, axes=(0, 1, 2), norm='backward')
                self.logger.info(f"Mesh Fourier transform done in {time.time() - time_start:.2f} seconds")

            # result = result.value if not fourier else result.r2c().value
            if threshold is not None:
                result_combined[np.abs(result_combined) < threshold] = 0

        self.comm.Barrier()    
        return result_combined

    
# barebones class so covariance.py compiles without error for now
# TODO for Otavio: restore this class?
class BoxGeometry(base.BaseClass):

    def __init__(self):
        pass


class SurveyGeometry(base.BaseClass):

    def __init__(self,
                 randoms:list=None, alphas:list=None,
                 nmesh=None, boxsize=None, boxpad=2.,
                 kmin=0, kmax=0.2, dk=None, binning_type="linear", mask_ellmax=12, pk_ellmax=4,
                 sample_mode="monte-carlo", lebedev_degree=25, resume_file=None, comm=MPI.COMM_WORLD):

        # set's k-binning
        super().__init__()

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.logger = logging.getLogger('SurveyGeometry')
        self.logger.setLevel(logging.INFO)
        self.tqdm = shell_tqdm
                
        self.mask_ellmax = mask_ellmax
        self.pk_ellmax = pk_ellmax
        self.sample_mode = sample_mode
        self.lebedev_degree = lebedev_degree
        self.window_matrix = {}

        if resume_file is not None:
            self.set_resume_file(resume_file)
        else:
            self.set_resume_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache/WinKernel.npy"))

        self._init_randoms(randoms, alphas)
        self._init_I_factors()
        self.comm.Barrier()
        self._init_survey_windows(nmesh=nmesh, boxsize=boxsize, boxpad=boxpad, kmin=kmin, kmax=kmax, dk=dk)
        self.comm.Barrier()
        del self.randoms

    def load_resume_file(self, filename):
        '''Load the window kernels from a file on each rank (sequentually).

        Parameters
        ----------
        filename : str
            Name of the file to load the window kernels from.
        '''

        if self.rank == 0: self.logger.info(f'Loading window kernels from {filename}.')
        new = self.load(filename)
        self.__setstate__(new.__getstate__())
        self.comm.Barrier()

    def set_resume_file(self, filename):
        '''Set the resume file for the window kernels.

        Parameters
        ----------
        filename : str
            Name of the file to save the window kernels.
        '''
        self._resume_file = filename

        if self._resume_file is not None:
            if os.path.exists(self._resume_file):
                self.load_resume_file(self._resume_file)
                if self.rank == 0: self.logger.warning(f'Loaded resume file {self._resume_file}. This might override your settings. See debug messages for more details on the loaded attributes.')
            else:
                if self.rank == 0: self.logger.info(f'File {self._resume_file} not found. Creating resume file.')
                self.save(self._resume_file)

        self.comm.Barrier()

    @property
    def is_kbins_set(self):
        return getattr(self, 'k_binning.is_kbins_set', False)

    def set_kbins(self, binning_obj:binning.FourierBinning):
        '''Set the k-bins for the window kernels.

        Args:
            binning_obj: binning.FourierBinning
                Either a linear or log binning object.
        '''
        if not isinstance(binning_obj, binning.LinearBinning) and not isinstance(binning_obj, binning.LogBinning):
            raise ValueError("binning must be either a linear or log binning object")
        
        self.k_binning = binning_obj

    def _init_randoms(self, randoms_list, alphas_list):
        
        self.randoms = []
        self.alphas  = []
        self.num_tracers = len(randoms_list)

        for i, (randoms, alpha) in enumerate(zip(randoms_list, alphas_list)):
            self.randoms.append(self._parse_randoms(randoms, alpha))
            self.alphas.append(alpha)
            if self.rank == 0: self.logger.info(f"Randoms for tracer {i} initialized with {self.randoms[-1].size} objects and alpha = {alpha}.")

    def _parse_randoms(self, randoms, alpha):
        """Parse the randoms into a catalog, filling in missing information as needed."""
        
        if not isinstance(randoms, mockfactory.Catalog):
            randoms = mockfactory.Catalog(randoms)

        # Check if the randoms have weights, otherwise set them to 1
        #for name in ['WEIGHT', 'WEIGHT_FKP']:
        if 'WEIGHT' not in randoms:
            if 'WEIGHT_FKP' in randoms:
                if self.rank == 0: self.logger.info('Setting WEIGHT column in randoms to WEIGHT_FKP values.')
                randoms['WEIGHT'] = randoms['WEIGHT_FKP'].copy()
            else:
                if self.rank == 0: self.logger.warning(f'WEIGHT column not found in randoms. Setting it to 1.')
                randoms['WEIGHT'] = np.ones(randoms.size, dtype='f8')
            
        if 'NZ' not in randoms:
            if self.rank == 0: self.logger.warning('NZ column not found in randoms. Estimating it with RedshiftDensityInterpolator.')
            import healpy as hp
            nside = 512
            distance = np.sqrt(np.sum(randoms['POSITION']**2, axis=-1))
            xyz = randoms['POSITION'] / distance[:, None]
            hpixel = hp.vec2pix(nside, *xyz.T)
            unique_hpixels_rank = np.unique(hpixel)
            unique_hpixels_total = self.comm.allgather(unique_hpixels_rank)
            unique_hpixels_total = np.unique(np.concatenate(unique_hpixels_total).ravel())

            fsky = len(unique_hpixels_total) / hp.nside2npix(nside)
            if self.rank == 0: self.logger.info(f'fsky estimated from randoms: {fsky:.3f}')
            nbar = mockfactory.RedshiftDensityInterpolator(z=distance, weights=randoms['WEIGHT'], fsky=fsky)
            randoms['NZ'] = nbar(distance)

        return randoms

    def _init_survey_windows(self, **kwargs):
        
        self.windows = {}
        for (t1, t2) in itt.product(range(self.num_tracers), repeat=2):
            if t2 > t1: continue

            self.windows[(t1, t2)] = SurveyWindow(self.randoms[t1], self.alphas[t1], 
                                                  self.randoms[t2], self.alphas[t2], shotnoise=True, **kwargs)

        # self.window_AB = SurveyWindow(self.randoms['A'], self.alphas['A'], self.randoms['B'], self.alphas['B'], shotnoise=True, **kwargs)
        # self.window_AC = SurveyWindow(self.randoms['A'], self.alphas['A'], self.randoms['C'], self.alphas['C'], **kwargs)
        # self.window_AD = SurveyWindow(self.randoms['A'], self.alphas['A'], self.randoms['D'], self.alphas['D'], **kwargs)

        # if self.randoms['B'] != None and self.randoms['C'] != None:
        #     self.window_CD = SurveyWindow(self.randoms['C'], self.alphas['C'], self.randoms['D'], self.alphas['D'], **kwargs)
        #     self.window_BC = SurveyWindow(self.randoms['B'], self.alphas['B'], self.randoms['C'], self.alphas['C'], shotnoise=True, **kwargs)
        #     self.window_BD = SurveyWindow(self.randoms['B'], self.alphas['B'], self.randoms['D'], self.alphas['D'], **kwargs)
        # elif self.randoms['B'] != None and self.randoms['C'] == None:
        #     self.window_CD = self.window_AB.copy()
        #     self.window_BC = SurveyWindow(self.randoms['B'], self.alphas['B'], self.randoms['C'], self.alphas['C'], shotnoise=True, **kwargs)
        #     self.window_BD = SurveyWindow(self.randoms['B'], self.alphas['B'], self.randoms['D'], self.alphas['D'], **kwargs)
        # elif self.randoms['B'] == None and self.randoms['C'] != None:
        #     self.window_CD = SurveyWindow(self.randoms['C'], self.alphas['C'], self.randoms['D'], self.alphas['D'], **kwargs)
        #     self.window_BC = self.window_AB.copy()
        #     self.window_BD = self.window_AB.copy()
        # else:
        #     self.window_CD = self.window_AB.copy()
        #     self.window_BC = self.window_AB.copy()
        #     self.window_BD = self.window_AB.copy()

    def _init_I_factors(self):
        """initializes all relavent I factors from the input randoms"""

        self.I_LABELS = ['12', '22', '10', '24', '14', '34', '44', '32']
        self._I = np.full((len(self.I_LABELS), self.num_tracers), 1.0)
        for tracer in range(self.num_tracers):

            if self.randoms[tracer] is not None:
                if self.rank == 0: 
                    self.logger.info(f"Initializing I factors from random {tracer}...")
                    pbar = self.tqdm(total=len(self.I_LABELS), desc=f"I factors for tracer {tracer}")

                for i, label in enumerate(self.I_LABELS):
                    nbar_power = int(label[0])
                    fkp_power = int(label[1])
                    I_sub = (self.randoms[tracer]['NZ']**(nbar_power-1) * \
                            self.randoms[tracer]['WEIGHT']**fkp_power).sum().item()
                    I = self.comm.allreduce(I_sub, op=MPI.SUM)

                    self._I[i, tracer] = I
                    if self.rank == 0: pbar.update(1)

                if self.rank == 0: pbar.close()

    def I(self, tracer:int, nbar_power:int, fkp_power:int, apply_alpha=False):
        """Retrieve the I normalization factor for the given tracer.

        Args:
        tracer (int): Tracer label.
        nbar_power (int): Power of nbar in the I factor.
        fkp_power (int): Power of FKP weight in the I factor.
        apply_alpha (bool, optional): Whether to apply alpha(tracer) Default is False.

        Returns
        -------
        I factor
        """

        if tracer > self.num_tracers - 1:
            raise ValueError(f"tracer must be between 0 and {self.num_tracers - 1}")
        if f"{nbar_power}{fkp_power}" not in self.I_LABELS:
            raise ValueError(f"Invalid combination of nbar_power ({nbar_power}) and fkp_power ({fkp_power}). Must be one of {self.I_LABELS}")

        label_idx = self.I_LABELS.index(f"{nbar_power}{fkp_power}")
        if apply_alpha:
            return self._I[label_idx, tracer] * self.alphas[tracer]
        else:
            return self._I[label_idx, tracer]

    @functools.cache
    def get_cosmic_variance_window(self, cache_dir=None, A=0, B=0, C=0, D=0, term="first"):

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"W_{A}{B}{C}{D}_{term}.npz")

        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            window_cv = base.SparseNDArray(shape_out=2*[self.mask_ellmax//2+1] + 2*[2*self.mask_ellmax+1],
                                             shape_in=(self.nmesh,self.nmesh,self.nmesh))

            total_iterations = 0
            for la, lb, ma, mb in utils.ellmiter(self.mask_ellmax, 2):
                total_iterations+=1

            self.comm.Barrier()
            if self.rank == 0: pbar = self.tqdm(total=total_iterations, desc=f"{term} Cosmic variance mesh calculation")

            for la, lb, ma, mb in utils.ellmiter(self.mask_ellmax, 2):
                if term == "first":
                    window_cv[la//2,lb//2,ma+la,mb+lb] = np.conj(self.windows[A, D].compute_mesh(la, ma, "W", "W")) * \
                                                           self.windows[B, C].compute_mesh(lb, mb, "W", "W")
                elif term == "second":
                    window_cv[la//2,lb//2,ma+la,mb+lb] = np.conj(self.windows[A, C].compute_mesh(la, ma, "W", "W")) * \
                                                           self.windows[B, D].compute_mesh(lb, mb, "W", "W")
                if self.rank == 0: pbar.update(1)

            if self.rank == 0: pbar.close()
            window_cv.save(filename)
            return window_cv

    @functools.cache
    def get_mixed_window(self, cache_dir:str=None, term="first"):

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"W_mixed_{term}.npz")

        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            window = base.SparseNDArray(shape_out=2*[self.mask_ellmax//2+1] + 2*[2*self.mask_ellmax+1],
                                        shape_in=(self.nmesh,self.nmesh,self.nmesh))
            
            total_iterations = 0
            for la, lb, ma, mb in utils.ellmiter(self.mask_ellmax, 2):
                total_iterations+=1

            self.comm.Barrier()
            if self.rank == 0: pbar = self.tqdm(total=total_iterations, desc=f"Mixed mesh calculation ({term})")
            for la, lb, ma, mb in utils.ellmiter(self.mask_ellmax, 2):
                if term == "first":
                    window[la//2,lb//2,ma+la,mb+lb] = self.window_AB.compute_mesh(la, ma, "S", None) * \
                                                      self.window_BC.compute_mesh(lb, mb, "W", "W")
                elif term == "second":
                    window[la//2,lb//2,ma+la,mb+lb] = self.window_AD.compute_mesh(la, ma, "W", "W") * \
                                                      self.window_BC.compute_mesh(lb, mb, "S", None)
                elif term == "third":
                    window[la//2,lb//2,ma+la,mb+lb] = self.window_AB.compute_mesh(la, ma, "S", None) * \
                                                      self.window_BD.compute_mesh(lb, mb, "W", "W")
                elif term == "fourth":
                    window[la//2,lb//2,ma+la,mb+lb] = self.window_AC.compute_mesh(la, ma, "W", "W") * \
                                                      self.window_BC.compute_mesh(lb, mb, "S", None)
                if self.rank == 0: pbar.update(1)

            if self.rank == 0: pbar.close()                 
            window.save(filename)
            return window

    @functools.cache
    def get_shotnoise_window(self, cache_dir:str=None, A=0, B=0):

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"S_{A}{B}.npz")
        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            window = base.SparseNDArray(shape_out=2*[self.mask_ellmax//2+1] + 2*[2*self.mask_ellmax+1],
                                        shape_in=(self.nmesh,self.nmesh,self.nmesh))

            total_iterations = 0
            for la, lb, ma, mb in utils.ellmiter(self.mask_ellmax, 2):
                total_iterations+=1

            self.comm.Barrier()
            if self.rank == 0: pbar = self.tqdm(total=total_iterations, desc="Shotnoise mesh calculation")
            for la, lb, ma, mb in utils.ellmiter(self.mask_ellmax, 2):
                window[la//2,lb//2,ma+la,mb+lb] = self.windows[A, B].compute_mesh(la, ma, "S", None) * \
                                                  self.windows[B, A].compute_mesh(lb, mb, "S", None)
                
                if self.rank == 0: pbar.update(1)

            if self.rank == 0: pbar.close()
            window.save(filename)
            return window

    @property
    def delta_k_max(self):
        # TODO: This will usually give much larger values than we probably need. Would be good to test that
        return self.nmesh // 2 - 1

    def cosmic_variance_kernel(self, A, B, C, D):
        """
        The survey window kernel corresponding to the cosmic variance Gaussian term with the specific tracer combination.
        Upon first calling this property, the window kernels are computed and cached.
        Subsequent calls return the cached value.
        """
        key = f"cosmic_variance_{A}{B}{C}{D}"
        if key not in self.window_matrix or np.any(np.isnan(self.window_matrix[key])):
            self.compute_window_matrix(None, A, B, C, D)
        return self.window_matrix[key]
    
    def mixed_kernel(self, A, B, C, D):
        """
        The survey window kernel corresponding to the mixed Gaussian term.
        Upon first calling this property, the window kernels are computed and cached.
        Subsequent calls return the cached value.
        """
        key = f"mixed_term_{A}{B}{C}{D}"
        if key not in self.window_matrix or np.any(np.isnan(self.window_matrix[key])):
            self.compute_window_matrix(None, A, B, C, D)
        return self.window_matrix[key]

    def shotnoise_kernel(self, A, B, C=0, D=0):
        """
        The survey window kernel corresponding to the shotnoise Gaussian term.
        Upon first calling this property, the window kernels are computed and cached.
        Subsequent calls return the cached value.
        """
        key = f"shotnoise_{A}{B}"
        if key not in self.window_matrix or np.any(np.isnan(self.window_matrix[key])):
            self.compute_window_matrix(None, A, B, C, D)
        return self.window_matrix[key]

    @property
    def nmesh(self):
        return self.windows[0,0].knmesh
    
    @property
    def boxsize(self):
        return self.windows[0,0].kboxsize
    
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
    def get_mixed_gaunt_coefficients(mask_ellmax=MASK_ELL_MAX, pk_ellmax=PK_ELL_MAX, cache_dir=None, term="first"):
        """Calculates all relavent Gaunt coefficients for the mixed term, or loads them from file"""
        
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"mixed_coefficients_{pk_ellmax:d}_{mask_ellmax:d}_{term}.npz")

        logger = logging.getLogger('SurveyGeometry')

        if os.path.exists(filename):
            logger.info(f'Loading mixed Gaunt coefficients from cache: {filename}')
            return base.SparseNDArray.load(filename)
        else:
            logger.info(f'Computing mixed Gaunt coefficients (term= {term}, pk_ellmax={pk_ellmax}, mask_ellmax={mask_ellmax})...')
            import sympy.physics.wigner

            # shape_out = l1, l2, l3, m1, m2, m3
            # shape_in =  la, ma, lb, mb
            # Only including positive m values, as -m is equivalent to m
            # when Ylm is real and m is even
            shape_out = 3*[pk_ellmax//2 + 1] + 3*[2*pk_ellmax + 1]
            shape_in = 2*[mask_ellmax//2 + 1] + 2*[2*mask_ellmax + 1]
            gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)

            for l1, l2, l3, m1, m2, m3 in utils.ellmiter(pk_ellmax, 3):

                lb, mb = l1, m1
                if lb <= mask_ellmax and term == "first":
                    for la in np.arange(np.abs(l2-l3), min(l2+l3, mask_ellmax)+1, 2):
                        for ma in np.arange(-la, la+1, 2):
                            value = np.float64(sympy.physics.wigner.gaunt(l2,l3,la,m2,m3,ma))
                            if value != 0:
                                gaunt_coefficients[l1//2, l2//2, l3//2, m1+l1, m2+l2, m3+l3, la//2, lb//2, ma+la, mb+lb] += value

                lb, mb = l2, m2
                if lb <= mask_ellmax and term == "second":
                    for la in np.arange(np.abs(l1-l3), min(l1+l3, mask_ellmax)+1, 2):
                        for ma in np.arange(-la, la+1, 2):
                            value = np.float64(sympy.physics.wigner.gaunt(l1,l3,la,m1,m3,ma))
                            if value != 0:
                                gaunt_coefficients[l1//2, l2//2, l3//2, m1+l1, m2+l2, m3+l3, la//2, lb//2, ma+la, mb+lb] += value

                la, ma = l3, m3
                if la <= mask_ellmax and term == "third":
                    for lb in np.arange(np.abs(l1-l2), min(l1+l2, mask_ellmax)+1, 2):
                        for mb in np.arange(-lb, lb+1, 2):
                            
                            value = np.float64(sympy.physics.wigner.gaunt(l1,l2,lb,m1,m2,mb))
                            if value != 0:
                                gaunt_coefficients[l1//2, l2//2, l3//2, m1+l1, m2+l2, m3+l3, la//2, lb//2, ma+la, mb+lb] += value
                                
                lb, mb = 0,0
                if term == "fourth":
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
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"shotnoise_coefficients_{pk_ellmax:d}_{mask_ellmax:d}.npz")
        logger = logging.getLogger('SurveyGeometry')

        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            import sympy.physics.wigner

            # shape_out = l1, l2, m1, m2
            # shape_in =  la, ma
            # Only including positive m values, as -m is equivalent to m
            # when Ylm is real and m is even
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

    def clean(self):
        '''Clean window kernels and power spectra.'''
        self.WinKernel = None
        self.WinKernel_error = None
        self._window_power = None
        self._I = {}

    @base.cache
    def compute_window_matrix(self, cache_dir:str=None, A:int=0, B:int=0, C:int=0, D:int=0, kmodes_sampled=50):
        '''Computes the window matrix to be used in the calculation of the covariance.

        Notes
        -----
        The window matrices are computed using the method described in [1]_.

        References
        ----------
        .. [1] https://arxiv.org/abs/1910.02914
        '''

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")

        window_matrix_file = os.path.join(cache_dir, f'window_matrix_{A}{B}{C}{D}.npz')
        if os.path.exists(window_matrix_file):
            self.window_matrix = {}
            if self.rank == 0: self.logger.info(f'Loading window matrices from cache: {window_matrix_file}')
            for r in range(self.size):
                self.comm.Barrier()
                if self.rank == r:
                    data = np.load(window_matrix_file, allow_pickle=True)
                    self.window_matrix[f'cosmic_variance_{A}{B}{C}{D}'] = data['cosmic_variance']
                    self.window_matrix[f'mixed_term_{A}{B}{C}{D}']      = data['mixed_term']
                    self.window_matrix[f'shotnoise_{A}{B}']             = data['shotnoise']

            return self.window_matrix

        if self.rank == 0:
            self.logger.info('='*60)
            self.logger.info('Computing window matrices')
            self.logger.info(f'pk_ellmax={self.pk_ellmax}, mask_ellmax={self.mask_ellmax}')
            self.logger.info('='*60)

        # sample kmodes from each k1 bin

        # SAMPLE FROM SHELL
        # kfun = 2 * np.pi / self.boxsize
        # kmodes = np.array([[math.sample_from_shell(kmin/kfun, kmax/kfun) for _ in range(
        #                    kmodes_sampled)] for kmin, kmax in zip(self.kedges[:-1], self.kedges[1:])])
        # Nmodes = math.nmodes(self.boxsize**3, self.kedges[:-1], self.kedges[1:])

        # SAMPLE FROM CUBE
        # kmodes, Nmodes = math.sample_from_cube(self.kmax/kfun, self.dk/kfun, kmodes_sampled)

        # HYBRID SAMPLING
        # Sample k1 modes (in units of k / k_func)
        if self.rank == 0:
            self.logger.info('Sampling k-modes for binning...')
            kmodes, Nmodes, weights = math.sample_kmodes(self.k_binning, boxsize=self.boxsize,
                                        max_modes=kmodes_sampled, k_shell_approx=0.05, sample_mode="monte-carlo")
        else:
            kmodes, Nmodes, weights = None, None, None
        
        kmodes = self.comm.bcast(kmodes, root=0)
        Nmodes = self.comm.bcast(Nmodes, root=0)
        weights = self.comm.bcast(weights, root=0)

        delta_ik = np.array(np.meshgrid(*self.ikgrid, indexing='ij'))

        if self.rank == 0:
            self.logger.info(f'Sampled k-modes for {self.k_binning.kbins} bins')

            assert len(kmodes) == self.k_binning.kbins and len(Nmodes) == self.k_binning.kbins, \
                f'Error in thecov.utils.sample_kmodes: results should have length {self.k_binning.kbins}, but had {len(kmodes)}. Parameters were kmin={self.k_binning.kmin},kmax={self.k_binning.kmax},dk={self.k_binning.dk},boxsize={self.boxsize},max_modes={kmodes_sampled},k_shell_approx={0.1}).'

            self.logger.info('Computing window function multipoles...')        

        # Compute the shape of the slab
        #shape_slab = self.compute_mesh(2,2,0,0).shape

        survey_window = {}
        survey_window['first_cosmic_variance']  = self.get_cosmic_variance_window(cache_dir, A, B, C, D, term="first")
        survey_window['second_cosmic_variance'] = self.get_cosmic_variance_window(cache_dir, A, B, C, D, term="second")
        # survey_window['first_mixed_term']       = self.get_mixed_window(cache_dir=cache_dir, term="first")
        # survey_window['second_mixed_term']      = self.get_mixed_window(cache_dir=cache_dir, term="second")
        # survey_window['third_mixed_term']       = self.get_mixed_window(cache_dir=cache_dir, term="third")
        # survey_window['fourth_mixed_term']      = self.get_mixed_window(cache_dir=cache_dir, term="fourth")
        survey_window['shotnoise']              = self.get_shotnoise_window(cache_dir, A, B)

        # Move survey_window to shared memory
        for key in survey_window:
            survey_window[key] = survey_window[key].to_shared_memory()

        # Read Gaunt coefficients only on rank 0 to avoid IO race conditions
        if self.rank == 0:
            self.logger.info('Contracting Gaunt coefficients with window meshes...')
            coefficients = {
                'first_cosmic_variance':  self.get_first_cosmic_variance_gaunt_coefficients(self.mask_ellmax, self.pk_ellmax),
                'second_cosmic_variance': self.get_second_cosmic_variance_gaunt_coefficients(self.mask_ellmax, self.pk_ellmax),
                'first_mixed_term':       self.get_mixed_gaunt_coefficients(self.mask_ellmax, self.pk_ellmax, term="first"),
                'second_mixed_term':      self.get_mixed_gaunt_coefficients(self.mask_ellmax, self.pk_ellmax, term="second"),
                'third_mixed_term':       self.get_mixed_gaunt_coefficients(self.mask_ellmax, self.pk_ellmax, term="third"),
                'fourth_mixed_term':      self.get_mixed_gaunt_coefficients(self.mask_ellmax, self.pk_ellmax, term="fourth"),
                'shotnoise':              self.get_shotnoise_gaunt_coefficients(self.mask_ellmax, self.pk_ellmax),
            }
        else:
            coefficients = None

        coefficients = self.comm.bcast(coefficients)
        
        window_product = {
            'first_cosmic_variance':  coefficients['first_cosmic_variance'] @ survey_window['first_cosmic_variance'],
            'second_cosmic_variance': coefficients['second_cosmic_variance'] @ survey_window['second_cosmic_variance'],
            # 'first_mixed_term':       coefficients['first_mixed_term'] @ survey_window['first_mixed_term'],
            # 'second_mixed_term':      coefficients['second_mixed_term'] @ survey_window['second_mixed_term'],
            # 'third_mixed_term':       coefficients['third_mixed_term'] @ survey_window['third_mixed_term'],
            # 'fourth_mixed_term':      coefficients['fourth_mixed_term'] @ survey_window['fourth_mixed_term'],
            'shotnoise':              coefficients['shotnoise'] @ survey_window['shotnoise'],
        }

        # load in ylm callables
        Ylm_table = math.build_Ylm_table(self.pk_ellmax)

        window_matrix = {}
        window_matrix['cosmic_variance'] = np.zeros([2] + 4*[self.pk_ellmax//2+1] + 2*[self.k_binning.kbins])
        window_matrix['mixed_term']      = np.zeros([4] + 3*[self.pk_ellmax//2+1] + 2*[self.k_binning.kbins])
        window_matrix['shotnoise']       = np.zeros(2*[self.pk_ellmax//2+1] + 2*[self.k_binning.kbins])

        if self.rank == 0:
            self.logger.info(f'Starting mode integration with {self.size} MPI ranks...')
            
        for i, km in enumerate(kmodes):
            if self.rank == 0:
                self.logger.info(f'Computing window matrix for bin {i+1}/{self.k_binning.kbins} with {len(km)} modes.')

            #k1_bin_index = int(i + self.k_binning.kmin // self.k_binning.dk)
            k1_bin_index = i

            # Split kmodes in chunks
            chunks = np.array_split(km, self.size)

            for ik1x, ik1y, ik1z, ik1r in chunks[self.rank]:
                ik1 = np.array([ik1x, ik1y, ik1z])
                ik1_norm = np.sqrt(np.sum(ik1**2))
                if ik1_norm == 0:
                    ik1_norm = 1.0
                ik1_hat = ik1 / ik1_norm

                # Compute and normalize ik2 = ik1 + delta_ik
                ik2 = ik1[:, None, None, None] + delta_ik
                ik2_norm = np.sqrt(np.sum(ik2**2, axis=0))
                ik2_norm_safe = ik2_norm.copy()
                ik2_norm_safe[ik2_norm_safe == 0] = 1.0
                ik2_hat = ik2 / ik2_norm_safe[None, ...]
                ik2_hat[:, ik2_norm == 0] = np.array([1, 0, 0])[:, None]  # Arbitrary direction for zero vector

                #k2_bin_index = (np.sqrt(np.sum(ik2**2, axis=0)) * self.kfun / self.k_binning.dk).astype(int)
                #k2_bin_index = (ik2_norm * self.kfun / self.k_binning.dk).astype(int)
                k2_bin_index = (((ik2_norm * self.kfun) - self.k_binning.kmin) / self.k_binning.dk).astype(int)

                Ylm_k1 = math.evaluate_Ylms(Ylm_table, self.pk_ellmax, *ik1_hat)
                Ylm_k2 = math.evaluate_Ylms(Ylm_table, self.pk_ellmax, *ik2_hat)

                # Cosmic Variance Term
                for l1, l2, l3, l4, m1, m2, m3, m4 in utils.ellmiter(self.pk_ellmax, 4):

                    # mesh is shape [nmesh, nmesh, nmesh]
                    mesh1 = window_product['first_cosmic_variance']\
                        [l1//2,l2//2,l3//2,l4//2,m1+l1,m2+l2,m3+l3,m4+l4].real
                    mesh1 = mesh1.toarray().reshape(window_product['first_cosmic_variance'].shape_in) # <- [nmesh, nmesh, nmesh]
                    mesh1 *= Ylm_k1[l1//2][(m1+l1)//2]*Ylm_k1[l2//2][(m2+l2)//2]*Ylm_k2[l3//2][(m3+l3)//2]*Ylm_k2[l4//2][(m4+l4)//2]
                    
                    mesh2 = window_product['second_cosmic_variance']\
                        [l1//2,l2//2,l3//2,l4//2,m1+l1,m2+l2,m3+l3,m4+l4].real
                    mesh2 = mesh2.toarray().reshape(window_product['second_cosmic_variance'].shape_in)
                    mesh2 *= Ylm_k1[l1//2][(m1+l1)//2]*Ylm_k2[l2//2][(m2+l2)//2]*Ylm_k2[l3//2][(m3+l3)//2]*Ylm_k1[l4//2][(m4+l4)//2]

                    idx = k2_bin_index.ravel()
                    valid = (idx >= 0) & (idx < self.k_binning.kbins)
                    if valid.any():
                        window_matrix['cosmic_variance'][0, l1//2,l2//2,l3//2,l4//2,k1_bin_index,:] += \
                            np.bincount(k2_bin_index.ravel()[valid], weights=mesh1.ravel()[valid], minlength=self.k_binning.kbins)[:self.k_binning.kbins]
                        window_matrix['cosmic_variance'][1, l1//2,l2//2,l3//2,l4//2,k1_bin_index,:] += \
                            np.bincount(k2_bin_index.ravel()[valid], weights=mesh2.ravel()[valid], minlength=self.k_binning.kbins)[:self.k_binning.kbins]

                    # window_matrix['cosmic_variance'][0, l1//2,l2//2,l3//2,l4//2,k1_bin_index,:] += \
                    #    (np.bincount(k2_bin_index.ravel(), weights=mesh.ravel(), minlength=self.k_binning.kbins)[:self.k_binning.kbins])
                    
                    # window_matrix['cosmic_variance'][1, l1//2,l2//2,l3//2,l4//2,k1_bin_index,:] += \
                    #     (np.bincount(k2_bin_index.ravel(), weights=mesh.ravel(), minlength=self.k_binning.kbins)[:self.k_binning.kbins])

                # Mixed Term
                # for l1, l2, l3, m1, m2, m3 in utils.ellmiter(self.pk_ellmax, 3):
                #     for (i, term) in enumerate(["first_mixed_term", "second_mixed_term", "third_mixed_term", "fourth_mixed_term"]):
                #         mesh = window_product[term]\
                #             [l1//2,l2//2,l3//2,m1+l1,m2+l2,m3+l3].real
                #         mesh = mesh.toarray().reshape(window_product[term].shape_in)
                #         mesh *= Ylm_k1[l1//2][(m1+l1)//2]*Ylm_k2[l2//2][(m2+l2)//2]*Ylm_k2[l3//2][(m3+l3)//2]

                #         window_matrix["mixed_term"][i, l1//2,l2//2,l3//2,k1_bin_index,:] += \
                #             (np.bincount(k2_bin_index.ravel(), weights=mesh.ravel(), minlength=self.k_binning.kbins)[:self.k_binning.kbins])      
                
                # Shotnoise Term
                for l1, l2, m1, m2 in utils.ellmiter(self.pk_ellmax, 2):
                    mesh = window_product['shotnoise'][l1//2,l2//2,m1+l1,m2+l2].real + \
                           survey_window['shotnoise'][l1//2,l2//2,m1+l1,m2+l2].real
                    mesh = mesh.toarray().reshape(window_product['shotnoise'].shape_in)
                    mesh *=Ylm_k1[l1//2][(m1+l1)//2]*Ylm_k2[l2//2][(m2+l2)//2]

                    window_matrix['shotnoise'][l1//2,l2//2,k1_bin_index,:] += \
                        (np.bincount(k2_bin_index.ravel(), weights=mesh.ravel(), minlength=self.k_binning.kbins)[:self.k_binning.kbins])

            # for key in window_matrix.keys():
            #     window_matrix[key] /= len(km)

        self.comm.Barrier()
        # Sum contributions from all ranks
        #if self.with_mpi:
        window_matrix_combined = {}
        for key in window_matrix.keys():
            window_matrix_combined[key] = np.zeros_like(window_matrix[key])
            self.comm.Allreduce(window_matrix[key], window_matrix_combined[key], op=MPI.SUM)
        #else:
        #    self.window_matrix = window_matrix

        # alpha and I22 factors are handled in covariance.py
        for k1 in range(self.k_binning.kbins):
            window_matrix_combined['cosmic_variance'][:,:,:,:,:,k1,:] *= \
                (4*np.pi)**4 / (len(kmodes[k1]) * Nmodes[None,None,None,None,None,k1])

            window_matrix_combined['mixed_term'][:,:,:,:,k1,:] *= \
                (4*np.pi)**2 / (len(kmodes[k1]) * Nmodes[None,None,None,k1])
            
            window_matrix_combined['shotnoise'][:,:,k1,:] *= \
                (4*np.pi)**2 / (len(kmodes[k1]) * Nmodes[None,None,k1])
        
        self.window_matrix[f"cosmic_variance_{A}{B}{C}{D}"] = window_matrix_combined['cosmic_variance']
        self.window_matrix[f"mixed_term_{A}{B}{C}{D}"]      = window_matrix_combined['mixed_term']
        self.window_matrix[f"shotnoise_{A}{B}"]             = window_matrix_combined['shotnoise']

        if self.rank == 0:
            self.logger.info('Window matrix computation completed successfully!')
            np.savez(window_matrix_file, **window_matrix_combined)