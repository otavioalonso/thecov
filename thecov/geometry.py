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
import os, time
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
            boxsize = max(np.amax(randoms['POSITION'], axis=0) - np.amin(randoms['POSITION'], axis=0))

        if self.rank == 0: self.logger.info("Creating survey mesh W...")
        mesh = CatalogMesh(
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

        if shotnoise==True:
            if self.rank == 0: self.logger.info("Creating shotnoise mesh S...")
            shotnoise_mesh = CatalogMesh(
                data_positions=randoms['POSITION'],
                data_weights=randoms['WEIGHT_FKP']**2 * randoms[f'WEIGHT'] * alpha,
                position_type='pos',
                nmesh=nmesh,
                cellsize=cellsize,
                boxsize=boxsize,
                boxpad=boxpad,
                dtype='c16',
                **{'interlacing': 3, 'resampler': 'tsc'}
            )
        else:
            shotnoise_mesh = None

        self.comm.Barrier()
        if self.rank == 0: self.logger.info(f'Created meshes in {time.time() - start_time:.2f} seconds.')
        return mesh, shotnoise_mesh

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     for key in ['logger', 'tqdm', 'mesh1',  'mesh2', '_resume_file']:
    #         if hasattr(self, key):
    #             del state[key]
    #     return state
    
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
        
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
    def mesh(self, ell, m, shotnoise=False, combine_windows=True, fourier=True, threshold=None):
        """Compute the product of meshes and multiply by real Ylm evaluated at the same coordinates.

        Parameters
        ----------
        ell : int
            Degree of the spherical harmonic.

        m : int
            Order of the spherical harmonic.

        shotnoise : bool, optional
            If True, the shotnoise mesh is used instead of the original mesh. Default is False.

        combine_windows : bool, optional
            Determines whether or not to multiply mesh1 by mesh2. Default True
            
        fourier : bool, optional
            If True, the Fourier transform of the mesh is returned. Default is True.

        Returns
        -------
        mesh
            Resulting mesh (numpy array) after computation with size [nmesh, nmesh, nmesh] on rank 0
        """

        assert ell >= 0, "ell must be non-negative"
        assert abs(m) <= ell, "m must be less than or equal to ell"

        Ylm = math.get_real_Ylm(ell, m)
        time_start = time.time()
        # Initialize the result mesh
        if shotnoise:
            mesh_to_clone = self.shotnoise_mesh1
        else:
            mesh_to_clone = self.mesh1

        self.comm.Barrier()
        print("got past barrier")

        result = mesh_to_clone.clone(
                data_positions=mesh_to_clone.data_positions,
                data_weights=mesh_to_clone.data_weights*Ylm(mesh_to_clone.data_positions.T[0],
                                                            mesh_to_clone.data_positions.T[1],
                                                            mesh_to_clone.data_positions.T[2]),
                position_type='pos',
                mpicomm=self.comm
            ).to_mesh(compensate=True)

        print("got past to_mesh")
        if hasattr(self, 'mesh2') and not shotnoise:
            result *= self.mesh2.to_mesh(compensate=True)
        elif shotnoise and combine_windows:
            result *= self.shotnoise_mesh2.to_mesh(compensate=True)

        if not shotnoise or combine_windows and self.rank == 0:
            if self.rank == 0: self.logger.info(f"Mesh computation with Ylm ({ell}, {m}) done in {time.time() - time_start:.2f} seconds")

        # element-wise addition and send to root rank
        if hasattr(result, 'value'):
            result_per_rank = result.value # <- is now a numpy array
        else:
            result_per_rank = np.zeros((result.shape))

        print("got past result_per_rank")
        result_combined = np.zeros((result_per_rank.shape[0],
                                    result_per_rank.shape[1] * self.size,
                                    result_per_rank.shape[2]))
        result_combined[:,result_per_rank.shape[1]*self.rank:result_per_rank.shape[1]*(self.rank+1), :] = result_per_rank
        self.comm.Barrier()
        result_combined = self.comm.reduce(result_combined, op=MPI.SUM, root=0)
        print("got past reduce")

        if self.dk is not None and self.kmax is not None:
            trim_to_nmesh, rebin_factor = self._rebin_parameters(self.dk, self.kmax)

            if trim_to_nmesh <= self.nmesh and self.kboxsize <= self.boxsize:
                result_combined = result_combined[:trim_to_nmesh, :trim_to_nmesh, :trim_to_nmesh]
                if self.rank == 0: self.logger.info(f"Trimmed mesh from {self.nmesh} to {trim_to_nmesh} and boxsize from {self.boxsize:.0f} to {self.kboxsize:.0f}.")

                # Sum mesh values in the new mesh
                if rebin_factor > 1:
                    result_combined = result_combined.reshape((self.knmesh, rebin_factor, self.knmesh, rebin_factor, self.knmesh, rebin_factor)).sum(axis=(1, 3, 5))

                if self.rank == 0:self.logger.info(f"Rebinned mesh from {trim_to_nmesh} to {result.shape[0]} with factor {rebin_factor}.")

            # pmesh fft convention is F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r); let us correct it here
            if fourier:
                result_combined *= self.knmesh**3
                time_start = time.time()
                result_combined = np.fft.fftn(result_combined, axes=(0, 1, 2), norm='backward')
                self.logger.info(f"Mesh Fourier transform done in {time.time() - time_start:.2f} seconds")

        # result = result.value if not fourier else result.r2c().value
        if threshold is not None:
            # Convert the result to a sparse array to save memory
            result_combined[np.abs(result_combined) < threshold] = 0
            result_combined = base.SparseNDArray.from_dense(result_combined, shape_in=(self.nmesh,self.nmesh), shape_out=self.nmesh)

        self.comm.Barrier()
        return result_combined

    
# barebones class so covariance.py compiles without error for now
# TODO for Otavio: restore this class?
class BoxGeometry(base.BaseClass):

    def __init__(self):
        pass


class SurveyGeometry(base.BaseClass):

    def __init__(self,
                 randoms_a,      alpha_a,
                 randoms_b=None, alpha_b=None,
                 randoms_c=None, alpha_c=None,
                 randoms_d=None, alpha_d=None,
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

        if resume_file is not None:
            self.set_resume_file(resume_file)
        else:
            self.set_resume_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache/WinKernel.npy"))

        self._init_randoms(randoms_a, alpha_a, randoms_b, alpha_b, randoms_c, alpha_c, randoms_d, alpha_d)
        self._init_I_factors()
        self._init_survey_windows(nmesh=nmesh, boxsize=boxsize, boxpad=boxpad, kmin=kmin, kmax=kmax, dk=dk)
        del self.randoms

    def load_resume_file(self, filename):
        '''Load the window kernels from a file on each rank (sequentually).

        Parameters
        ----------
        filename : str
            Name of the file to load the window kernels from.
        '''
        for r in range(self.size):
            if self.rank == r: 
                if self.rank == 0: self.logger.info(f'Loading window kernels from {filename}.')
                self.logger.debug(f'rank {r} loading window kernels...')
                self.__setstate__(self.load(filename))
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
            try:
                self.load_resume_file(self._resume_file)
                if self.rank == 0: self.logger.warning(f'Loaded resume file {self._resume_file}. This might override your settings. See debug messages for more details on the loaded attributes.')
            except FileNotFoundError:
                if self.rank == 0:
                    self.logger.info(f'File {self._resume_file} not found. Creating resume file.')
                    utils.mkdir(os.path.dirname(self._resume_file))
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

    def _init_randoms(self, randoms_a, alpha_a,
                            randoms_b, alpha_b,
                            randoms_c, alpha_c,
                            randoms_d, alpha_d):
        
        self.randoms = {'A' : None, 'B' : None, 'C' : None, 'D': None}
        self.alphas  = {'A' : None, 'B' : None, 'C' : None, 'D': None}

        self.randoms['A'] = self._parse_randoms(randoms_a, alpha_a)
        self.alphas['A'] = alpha_a
        self._num_tracers = 1

        if randoms_b is not None and alpha_b is not None:
            self.randoms['B'] = self._parse_randoms(randoms_b, alpha_b)
            self.alphas['B'] = alpha_b
            self._num_tracers+=1
        
        if randoms_c is not None and alpha_c is not None:
            self.randoms['C'] = self._parse_randoms(randoms_c, alpha_c)
            self.alphas['C'] = alpha_c
            self._num_tracers+=1

        if randoms_d is not None and alpha_d is not None:
            self.randoms['D'] = self._parse_randoms(randoms_d, alpha_d)
            self.alphas['D'] = alpha_d
            self._num_tracers+=1

    def _parse_randoms(self, randoms, alpha):
        """Parse the randoms into a catalog, filling in missing information as needed."""
        
        if not isinstance(randoms, mockfactory.Catalog):
            randoms = mockfactory.Catalog(randoms)

        # Check if the randoms have weights, otherwise set them to 1
        for name in ['WEIGHT', 'WEIGHT_FKP']:
            if name not in randoms:
                if self.rank == 0: self.logger.warning(f'{name} column not found in randoms. Setting it to 1.')
                randoms[name] = np.ones(randoms.size, dtype='f8')
        
        randoms['WEIGHT'] *= alpha
        
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
        
        # 'A' will always have an associated random / alpha
        # TODO: When num_tracer = 1, this code runs thru the same calculation 3 times. Find a way to optimize it.
        self.window_AB = SurveyWindow(self.randoms['A'], self.alphas['A'], self.randoms['B'], self.alphas['B'], shotnoise=True, **kwargs)
        self.window_AC = SurveyWindow(self.randoms['A'], self.alphas['A'], self.randoms['C'], self.alphas['C'], **kwargs)
        self.window_AD = SurveyWindow(self.randoms['A'], self.alphas['A'], self.randoms['D'], self.alphas['D'], **kwargs)

        if self.randoms['B'] != None and self.randoms['C'] != None:
            self.window_CD = SurveyWindow(self.randoms['C'], self.alphas['C'], self.randoms['D'], self.alphas['D'], **kwargs)
            self.window_BC = SurveyWindow(self.randoms['B'], self.alphas['B'], self.randoms['C'], self.alphas['C'], shotnoise=True, **kwargs)
            self.window_BD = SurveyWindow(self.randoms['B'], self.alphas['B'], self.randoms['D'], self.alphas['D'], **kwargs)
        elif self.randoms['B'] != None and self.randoms['C'] == None:
            self.window_CD = self.window_AB
            self.window_BC = SurveyWindow(self.randoms['B'], self.alphas['B'], self.randoms['C'], self.alphas['C'], shotnoise=True, **kwargs)
            self.window_BD = SurveyWindow(self.randoms['B'], self.alphas['B'], self.randoms['D'], self.alphas['D'], **kwargs)
        elif self.randoms['B'] == None and self.randoms['C'] != None:
            self.window_CD = SurveyWindow(self.randoms['C'], self.alphas['C'], self.randoms['D'], self.alphas['D'], **kwargs)
            self.window_BC = self.window_AB
            self.window_BD = self.window_AB
        else:
            self.window_CD = self.window_AB
            self.window_BC = self.window_AB
            self.window_BD = self.window_AB

    def _init_I_factors(self):
        """initializes all relavent I factors from the input randoms"""

        self.I_LABELS = ['12', '22', '10', '24', '14', '34', '44', '32']
        self.TRACER_LABELS = ['A', 'B', 'C', 'D']
        self._I = np.full((len(self.I_LABELS), len(self.TRACER_LABELS)), 1.0)
        for tracer in self.TRACER_LABELS:

            if self.randoms[tracer] is not None:
                if self.rank == 0: 
                    self.logger.info(f"Initializing I factors from random {tracer}...")
                    pbar = self.tqdm(total=len(self.I_LABELS), desc=f"I factors for tracer {tracer}")

                for i, label in enumerate(self.I_LABELS):
                    nbar_power = int(label[0])
                    fkp_power = int(label[1])
                    I_sub = (self.randoms[tracer]['NZ']**(nbar_power-1) * \
                        self.randoms[tracer]['WEIGHT_FKP']**fkp_power * \
                        self.randoms[tracer]['WEIGHT'] * \
                        self.alphas[tracer]).sum().item()
                    I = self.comm.allreduce(I_sub, op=MPI.SUM)
                    print(self.rank, I_sub, I)
                    self._I[i, self.TRACER_LABELS.index(tracer)] = I
                    if self.rank == 0: pbar.update(1)

                if self.rank == 0: pbar.close()

    def I(self, tracer:str, nbar_power:int, fkp_power:int):
        """Retrieve the I normalization factor for the given tracer.

        Args:
        tracer (str, optional): Tracer label. Must be one of 'A', 'B', 'C', 'D'.
        nbar_power (int, optional): Power of nbar in the I factor.
        fkp_power (int, optional): Power of FKP weight in the I factor.

        Returns
        -------
        I factor
        """

        if tracer not in ['A', 'B', 'C', 'D']:
            raise ValueError("tracer must be one of 'A', 'B', 'C', 'D'")
        if f"{nbar_power}{fkp_power}" not in self.I_LABELS:
            raise ValueError(f"Invalid combination of nbar_power ({nbar_power}) and fkp_power ({fkp_power}). Must be one of {self.I_LABELS}")

        label_idx = self.I_LABELS.index(f"{nbar_power}{fkp_power}")
        tracer_idx = self.TRACER_LABELS.index(tracer)
        return self._I[label_idx, tracer_idx]


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

            total_iterations = 0
            for la, lb in itt.product(range(0, self.mask_ellmax+1, 2), repeat=2):
                for ma in range(-la, la+1):
                    for mb in range(-lb, lb+1):
                        total_iterations+=1

            if self.rank == 0: pbar = self.tqdm(total=total_iterations, desc="W * Ylm mesh calculation")
            for la, lb in itt.product(range(0, self.mask_ellmax+1, 2), repeat=2):
                for ma in range(-la, la+1):
                    for mb in range(-lb, lb+1):
                        window_ABCD[la//2,lb//2,ma+la,mb+lb] = self.window_AB.mesh(la, ma) * self.window_CD.mesh(lb, mb)
                        if self.rank == 0:pbar.update(1)
            if self.rank == 0:pbar.close()
            window_ABCD.save(filename)
            return window_ABCD

    @functools.cache
    def get_survey_window(self, idx_1="A", idx_2="C", cache_dir:str=None):
        """Retrieves the survey window for the given tracer indices as a base.SparseNDArray object.

        This function should be called by all ranks. The resulting survey window is only stored on rank 0.

        Args:
            idx_1 (str, optional): First tracer index. Defaults to "A".
            idx_2 (str, optional): Second tracer index. Defaults to "C".
            cache_dir (str, optional): Directory to cache the survey window. Defaults to None.

        Raises:
            ValueError: If idx_1 or idx_2 are invalid.

        Returns:
            base.SparseNDArray: Survey window for the given tracer indices.
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, "W_"+idx_1+idx_2+".npz")

        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            window = base.SparseNDArray(shape_out=(MASK_ELL_MAX//2+1,2*MASK_ELL_MAX+1),
                                        shape_in=(self.nmesh,self.nmesh,self.nmesh))

            if self.rank == 0:
                for l in range(0, self.mask_ellmax+1, 2):
                    for m in range(-l, l+1):
                        if idx_1 == "A" and idx_2 == "C":
                            window[l//2,m] = self.window_AC.mesh(l, m)
                        elif idx_1 == "A" and idx_2 == "D":
                            window[l//2,m] = self.window_AD.mesh(l, m)
                        elif idx_1 == "B" and idx_2 == "C":
                            window[l//2,m] = self.window_BC.mesh(l, m)
                        elif idx_1 == "B" and idx_2 == "D":
                            window[l//2,m] = self.window_BD.mesh(l, m)
                        else:
                            raise ValueError(f"ERROR! invalid values for A ({idx_1}) and B ({idx_2})")

            window.save(filename)
            return window

    def get_shotnoise_window(self, idx="A", cache_dir:str=None):
        """Retrieves the shotnoise window for the given tracer indices as a base.SparseNDArray object.

        This function should be called by all ranks. The resulting shotnoise window is only stored on rank 0.

        Args:
            idx (str, optional): Tracer index. Must be one of ["A", "B", "AB"], Defaults to "A".
            cache_dir (str, optional): Directory to cache the survey window. Defaults to None.

        Raises:
            ValueError: If idx is invalid.
        Returns:
            base.SparseNDArray: Survey window for the given tracer indices.
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, "S_"+idx+".npz")
        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            if self.rank == 0:
                window = base.SparseNDArray(shape_out=(MASK_ELL_MAX//2+1,2*MASK_ELL_MAX+1),
                                            shape_in=(self.nmesh,self.nmesh,self.nmesh))
            else: window = None

            for l in range(0, self.mask_ellmax+1, 2):
                for m in range(-l, l+1):
                    if idx == "A":
                        result = self.window_AB.mesh(l, m, shotnoise=True, combine_windows=False)
                    elif idx == "B":
                        result = self.window_BC.mesh(l, m, shotnoise=True, combine_windows=False)
                    elif idx == "AB":
                        result = self.window_AB.mesh(l, m, shotnoise=True, combine_windows=True)
                    else:
                        raise ValueError(f"ERROR! invalid value for input index. Should be one of [A, B, AB] but was {idx}")
                    if self.rank == 0: window[l//2,m+MASK_ELL_MAX] = result
            
            if self.rank == 0: window.save(filename)
            return window

    @property
    def delta_k_max(self):
        # TODO: This will usually give much larger values than we probably need. Would be good to test that
        return self.nmesh // 2 - 1

    @property
    def cosmic_variance_kernel(self):
        """
        The survey window kernel corresponding to the cosmic variance Gaussian term.
        Upon first calling this property, the window kernels are computed and cached.
        Subsequent calls return the cached value.
        """
        if not hasattr(self, 'WinKernel_cosmic') or np.any(np.isnan(self.WinKernel_cosmic)):
            self.compute_window_kernels()
        return self.WinKernel_cosmic
    
    @property
    def mixed_kernel(self):
        """
        The survey window kernel corresponding to the mixed Gaussian term.
        Upon first calling this property, the window kernels are computed and cached.
        Subsequent calls return the cached value.
        """
        if not hasattr(self, 'WinKernel_mixed') or np.any(np.isnan(self.WinKernel_mixed)):
            self.compute_window_kernels()
        return self.WinKernel_mixed

    @property
    def shotnoise_kernel(self):
        """
        The survey window kernel corresponding to the shotnoise Gaussian term.
        Upon first calling this property, the window kernels are computed and cached.
        Subsequent calls return the cached value.
        """
        if not hasattr(self, 'WinKernel_shotnoise') or np.any(np.isnan(self.WinKernel_shotnoise)):
            self.compute_window_kernels()
        return self.WinKernel_shotnoise

    @property
    def nmesh(self):
        return self.window_AB.knmesh
    
    @property
    def boxsize(self):
        return self.window_AB.kboxsize

    @property
    def num_tracers(self):
        return self._num_tracers

    @staticmethod
    def get_cosmic_variance_gaunt_coefficients(cache_dir=None, mask_ellmax=12, pk_ellmax=4):
        """Calculates all relavent Gaunt coefficients for the cosmic variance term, or loads them from file"""

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

    @staticmethod
    def get_mixed_gaunt_coefficients(cache_dir=None, mask_ellmax=12, pk_ellmax=4):
        """Calculates all relavent Gaunt coefficients for the shotnoise term, or loads them from file"""
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"mixed_coefficients_{pk_ellmax}_{mask_ellmax}.npz")
        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            import sympy.physics.wigner

            # shape_out = l1, l2, l3, m1, m2, l3
            # shape_in =  la, ma
            # Only including positive m values, as -m is equivalent to m
            # when Ylm is real and m is even
            shape_out = 3*[PK_ELL_MAX//2 + 1] + 3*[2*PK_ELL_MAX + 1]
            shape_in = [MASK_ELL_MAX//2 + 1] + [2*MASK_ELL_MAX + 1]
            gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)

            for l1, l2, l3 in itt.product(np.arange(0, pk_ellmax + 1, 2), repeat=3):
                for m1, m2, m3 in itt.product(*[np.arange(-l, l+1, 2) for l in (l1, l2, l3)]):
                    for la in range(np.abs(l1-l2), l1+l2+1, 2):
                        for ma in range(-la, la+1, 2):

                            # NOTE: We only have to compute one of the Gaunt coefficient objects here, as
                            # TODO for Otavio: How do we calculate this one? Eq 23 in the overleaf
                            value = 0
                            if value != 0: 
                                gaunt_coefficients[l1//2, l2//2, l3//2, m1+l1, m2+l2, m3+l3, la//2, ma+la] += value

            gaunt_coefficients.save(filename)
            return gaunt_coefficients
        
    @staticmethod
    def get_shotnoise_gaunt_coefficients(cache_dir=None, mask_ellmax=12, pk_ellmax=4):
        """Calculates all relavent Gaunt coefficients for the shotnoise term, or loads them from file"""
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        filename = os.path.join(cache_dir, f"shotnoise_coefficients_{pk_ellmax}_{mask_ellmax}.npz")

        if os.path.exists(filename):
            return base.SparseNDArray.load(filename)
        else:
            import sympy.physics.wigner

            # shape_out = l1, l2, m1, m2
            # shape_in =  la, ma
            # Only including positive m values, as -m is equivalent to m
            # when Ylm is real and m is even
            shape_out = 2*[PK_ELL_MAX//2 + 1] + 2*[2*PK_ELL_MAX + 1]
            shape_in = [MASK_ELL_MAX//2 + 1] + [2*MASK_ELL_MAX + 1]
            gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)

            for l1, l2 in itt.product(np.arange(0, pk_ellmax + 1, 2), repeat=2):
                for m1, m2 in itt.product(*[np.arange(-l, l+1, 2) for l in (l1, l2)]):
                    # TODO: Verify this loop is correct
                    for la in range(np.abs(l1-l2), l1+l2+1, 2):
                        for ma in range(-la, la+1, 2):
                            
                            value = np.float64(sympy.physics.wigner.gaunt(l1,l2,la,m1,m2,ma))
                            if value == 0:
                                # Taking absolute values of all m as -m is equivalent to m
                                # when Ylm is real and m is even
                                # m1, m2 = np.abs(m1), np.abs(m2)
                                # ma, mb = np.abs(ma), np.abs(mb)
                                gaunt_coefficients[l1//2, l2//2, m1+l1, m2+l2, la//2, ma+la] += value
            gaunt_coefficients.save(filename)
            return gaunt_coefficients

    def clean(self):
        '''Clean window kernels and power spectra.'''
        self.WinKernel = None
        self.WinKernel_error = None
        self._window_power = None
        self._I = {}

    def compute_window_kernels(self, cache_dir:str=None, kmodes_sampled:int=250):
        """Wrapper function that sequentially runs all kernel computations.
        Each term is calculated seperately in order to save memory
        
        Args:
            cache_dir (str): Directory to save/load window kernels. If None, uses default cache directory. Default None
            kmodes_sampled (int): Number of k-modes to randomly sample from each k1 bin. Default 250
        """

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")

        # Sample k1 modes
        if self.rank == 0:
            kmodes, Nmodes, weights = math.sample_kmodes(self.k_binning, boxsize=self.boxsize,
                                        max_modes=kmodes_sampled, k_shell_approx=0.05, sample_mode="monte-carlo")
        else:
            kmodes, Nmodes, weights = None, None, None
        
        kmodes = self.comm.bcast(kmodes, root=0)
        Nmodes = self.comm.bcast(Nmodes, root=0)
        weights = self.comm.bcast(weights, root=0)

        self._compute_cosmic_variance_kernel(cache_dir, kmodes, Nmodes, weights)
        self._compute_mixed_kernel(cache_dir, kmodes, Nmodes, weights)
        self._compute_shotnoise_kernel(cache_dir, kmodes, Nmodes, weights)

    def _compute_cosmic_variance_kernel(self, cache_dir:str, kmodes:np.ndarray, nmodes:np.ndarray, weights:np.ndarray):
        
        # points on the unit sphere with corresponding integration weights
        # x, y, z, w = math.get_lebedev_points(self.lebedev_degree)
        if hasattr(self, "WinKernel_cosmic") and self.WinKernel_cosmic is not None:
            return
        else:
            if self.rank == 0: self.logger.info("Computing cosmic variance term kernels...")
            # Format is [k1_bins, k2_bins, l1, l2, l3, l4]
            self.WinKernel_cosmic = np.empty([self.k_binning.kbins, 2*self.delta_k_max+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1])
            self.WinKernel_cosmic.fill(np.nan)

        # Gaunt coefficients
        if self.rank == 0:
            # calculate Gaunt coefficients first to avoid race conditions
            self.logger.info("Calculating or loading Gaunt coefficients...")
            G = self.get_cosmic_variance_gaunt_coefficients(cache_dir=cache_dir, mask_ellmax=self.mask_ellmax, pk_ellmax=self.pk_ellmax)
        else:
            G = None

        # W_AB * W_CD (outer product)
        self.logger.info("Retrieving survey window outer product W_AB x W_CD...")
        W_ABCD_not_shared = self.get_combined_survey_window(cache_dir=cache_dir)

        self.comm.Barrier()
        W_ABCD = W_ABCD_not_shared.to_shared_memory()
        del W_ABCD_not_shared
        
        self.comm.Barrier()
        G = self.comm.bcast(G, root=0)

        # multiply by Gaunt factors
        # give 3x3x3x3x9x9x9x9 x nmesh x nmesh x nmesh
        G_times_W = G @ W_ABCD
        self.comm.Barrier()
        del W_ABCD

        # load in ylm callables
        Ylm_table = math.build_Ylm_table(self.pk_ellmax)

        last_save = time.time()
        if self.rank == 0:
            self.logger.info(f"Beginning window kernel calculations on {self.size} ranks...")
            pbar = self.tqdm(desc='Computing window kernels', total=math.num_sampled_modes(kmodes))
        
        # NOTE: We're still parallelizing each k1 bin as before, but this could
        # be changed now that we're using mpi4py if we wanted
        for i, km in enumerate(kmodes):

            self.comm.Barrier()
            if hasattr(self, '_resume_file') and self._resume_file is not None and self.rank == 0:
                # Skip rows that were already computed
                if not np.isnan(self.WinKernel_cosmic[i,0,0,0,0,0]):
                    self.logger.debug(f'Skipping bin {i} of {self.k_binning.kbins}.')
                    continue

            kmodes_sampled = len(km)
            # Splitting kmodes in chunks to be sent to each rank
            kmodes_per_rank = np.array_split(km, self.size)[self.rank]

            results_per_rank = self._compute_cosmic_variance_kernel_row(i, kmodes_per_rank, G_times_W, Ylm_table)
            self.comm.Barrier()

            results_per_rank = np.sum(results_per_rank, axis=0)
            if self.rank == 0:
                results_combined = np.zeros_like(results_per_rank)
            else:
                results_combined = None # None on non-root processes
            self.comm.Reduce(results_per_rank, results_combined, op=MPI.SUM, root=0)

            # std_results = np.std(results * weights, axis=0) / np.sqrt(len(results))
            # avg_results = np.average(results, weights=weights, axis=0)
            # avg_results[std_results == 0] = 1
            # self.WinKernel_error[i] =  std_results / avg_results
    
            if self.rank == 0:
                self.WinKernel_cosmic[i] = results_combined * weights[i] / kmodes_sampled
                for k2_bin_index in range(0, 2*self.delta_k_max + 1):
                    if (k2_bin_index + i - self.delta_k_max >= self.k_binning.kbins or k2_bin_index + i - self.delta_k_max < 0):
                        self.WinKernel_cosmic[i, k2_bin_index, :, :] = 0
                    else:
                        self.WinKernel_cosmic[i, k2_bin_index, :, :] /= nmodes[i + k2_bin_index - self.delta_k_max]

                pbar.update(len(kmodes[i]))
                if hasattr(self, '_resume_file') and self._resume_file is not None and (time.time() - last_save) > 600:
                    self.logger.debug("Saving progress...")
                    self.save(self._resume_file)
                    last_save = time.time()
            
        self.logger.info('Cosmic variance window kernel computed.')
        if self._resume_file is not None and self.rank == 0:
            self.save(self._resume_file)

    def _compute_cosmic_variance_kernel_row(self, idx:int, bin_kmodes:np.ndarray, product:base.SparseNDArray, Ylm_table:np.ndarray):
        '''Computes a row of the window kernels. This function is called in parallel for each k1 bin.
        Gives window kernels for L=0,2,4 auto and cross covariance

        Args:
            idx (int):, the index of the current k1 bin
            bin_kmodes (np.ndarray): 4D array of x, y, z, and r coordinates of sampled modes in the current k1 bin
            product (SparseNDArray): Precomputed product of the Gaunt coefficients and the survey window
            Ylm_table (np.ndarray): Precomputed Ylm callables for each ell
        Returns:
            WinKernel (np.ndarray): an array with [2*delta_k_max+1,num_ell,num_ell,num_ell,num_ell] dimensions.
                The first dim corresponds to the k-bin of k2
                (only 3 bins on each side of diagonal are included by default as the Gaussian covariance drops quickly away from diagonal)
                The remaining dims correspond to specific ells
        '''

        # k1_bin_index is a scalar
        k1_bin_index = idx + self.k_binning.kmin//self.dk

        WinKernel = np.zeros((2*self.delta_k_max+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1), dtype=np.complex128)
        iix, iiy, iiz = np.meshgrid(*self.window_AB.ikgrid, indexing='ij')

        k2xh = np.zeros_like(iix)
        k2yh = np.zeros_like(iiy)
        k2zh = np.zeros_like(iiz)
        kfun = 2 * np.pi / self.boxsize

        mode_idx = 1
        t_avg = 0
        for ik1x, ik1y, ik1z, ik1r in bin_kmodes:
            t_start = time.time()
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
            k2_bin_index = (k2r * kfun / self.dk).astype(int)
            k2r[k2r <= 1e-10] = np.inf
            k2xh /= k2r
            k2yh /= k2r
            k2zh /= k2r
            
            # Evaluate ylm factors at the given k1 and k2 modes
            Ylm_k1 = math.evaluate_Ylms(Ylm_table, self.pk_ellmax, k1xh, k1yh, k1zh)
            Ylm_k2 = math.evaluate_Ylms(Ylm_table, self.pk_ellmax, k2xh, k2yh, k2zh)

            result = np.zeros((list(product.shape_in) + [3,3,3,3]), dtype=np.complex128)
            # multiply by Ylms
            for l1, l2, l3, l4 in itt.product(np.arange(0, self.pk_ellmax+1, 2), repeat=4):
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

                    Ylms = Ylm_k1[l1_idx][m1_idx] * \
                           Ylm_k2[l2_idx][m2_idx] * \
                           Ylm_k1[l3_idx][m3_idx] * \
                           Ylm_k2[l4_idx][m4_idx]
                    
                    result[:,:,:,l1_idx,l2_idx,l3_idx,l4_idx] += Ylms * W_times_G.toarray().reshape(product.shape_in)

            for delta_k in range(-self.delta_k_max, self.delta_k_max + 1):
                modes = (k2_bin_index - k1_bin_index == delta_k)
                if np.any(modes == True):
                    WinKernel[delta_k] = np.sum(result[modes], axis=0)

            t_avg += time.time() - t_start
            self.logger.debug(f"process {os.getpid()}, mode {mode_idx} / {len(bin_kmodes)} done. Avg time per iteration = {t_avg / mode_idx:.1f}s")
            mode_idx += 1

        return WinKernel
    
    def _compute_mixed_kernel(self, cache_dir:str, kmodes, Nmodes, weights):

        if hasattr(self, "WinKernel_mixed") and self.WinKernel_mixed is not None:
            return
        else:
            self.logger.info("Computing mixed term kernels...")
            # Format is [k1_bins, term, k2_bins, l1, l2, l3]
            self.WinKernel_mixed = np.empty([self.k_binning.kbins, 4, 2*self.delta_k_max+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1])
            self.WinKernel_mixed.fill(np.nan)

        if self.rank == 0:
            self.logger.info("Retrieving survey and shotnoise windows...")
            G_shot = self.get_shotnoise_gaunt_coefficients(pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax)
            G_mixed = self.get_mixed_gaunt_coefficients(pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax)
        else:
            G_shot, G_mixed = None, None
        G_shot = self.comm.bcast(G_shot, root=0)
        G_mixed = self.comm.bcast(G_mixed, root=0)
        self.comm.Barrier()

        SA_times_WBC = self.get_shotnoise_window("A", cache_dir) * self.get_survey_window("B", "C", cache_dir)
        WAD_times_SA = self.get_survey_window("A", "D", cache_dir) * self.get_shotnoise_window("B", cache_dir)
        SA_times_WBD = self.get_shotnoise_window("A", cache_dir) * self.get_survey_window("B", "D", cache_dir)
        WAC_times_SA = self.get_survey_window("A", "C", cache_dir) * self.get_shotnoise_window("B", cache_dir)
        self.comm.Barrier()
        
        SA_times_WBC = SA_times_WBC.to_shared_memory(self.comm)
        WAD_times_SA = WAD_times_SA.to_shared_memory(self.comm)
        SA_times_WBD = SA_times_WBD.to_shared_memory(self.comm)
        WAC_times_SA = WAC_times_SA.to_shared_memory(self.comm)

        G_times_SA_times_WBC = G_shot @ SA_times_WBC
        G_times_WAD_times_SB = G_shot @ WAD_times_SA
        G_times_SA_times_WBD = G_shot @ SA_times_WBD
        G_times_WAC_times_SB = G_mixed @ WAC_times_SA
        self.comm.Barrier()
        del SA_times_WBC, WAD_times_SA, SA_times_WBD, WAC_times_SA

         # load in ylm callables
        Ylm_table = math.build_Ylm_table(self.pk_ellmax)

        #ell_factor = lambda l1,l2: (2*l1 + 1) * (2*l2 + 1) * (2 if 0 in (l1, l2) else 1)
        last_save = time.time()
        self.logger.info(f"Beginning window kernel calculations with {self.nthreads} threads...")
        for i, km in self.tqdm(enumerate(kmodes), desc='Computing mixed window kernels', total=self.k_binning.kbins):

            if hasattr(self, '_resume_file') and self._resume_file is not None:
                # Skip rows that were already computed
                if not np.isnan(self.WinKernel_mixed[i,0,0,0,0,0,0]):
                    self.logger.debug(f'Skipping bin {i} of {self.k_binning.kbins}.')
                    continue
    
            kmodes_sampled = len(km)
            # Splitting kmodes in chunks to be sent to each rank
            kmodes_per_rank = np.array_split(km, self.size)[self.rank]

            results_per_rank = self._compute_mixed_kernel_row(i, kmodes_per_rank, G_times_SA_times_WBC, G_times_WAD_times_SB, 
                                                              G_times_SA_times_WBD, G_times_WAC_times_SB, Ylm_table)
            self.comm.Barrier()

            results_per_rank = np.sum(results_per_rank, axis=0)
            if self.rank == 0:
                results_combined = np.zeros_like(results_per_rank)
            else:
                results_combined = None # None on non-root processes
            self.comm.Reduce(results_per_rank, results_combined, op=MPI.SUM, root=0)

            if self.rank == 0:
                self.WinKernel_mixed[i] = results_combined.real * weights[i] / kmodes_sampled
                for k2_bin_index in range(0, 2*self.delta_k_max + 1):
                    if (k2_bin_index + i - self.delta_k_max >= self.k_binning.kbins or k2_bin_index + i - self.delta_k_max < 0):
                        self.WinKernel_mixed[i, :, k2_bin_index, :, :, :, :] = 0
                    else:
                        self.WinKernel_mixed[i, :, k2_bin_index, :, :, :, :] /= Nmodes[i + k2_bin_index - self.delta_k_max]

            if hasattr(self, '_resume_file') and self._resume_file is not None and (time.time() - last_save) > 600:
                self.save(self._resume_file)
                last_save = time.time()

            self.logger.info('Mixed term Window kernel computed.')

            if self._resume_file is not None:
                self.save(self._resume_file)


    def _compute_mixed_kernel_row(self, idx:int, bin_kmodes:np.ndarray, 
                                  G_times_SA_times_WBC:base.SparseNDArray, 
                                  G_times_WAD_times_SB:base.SparseNDArray,
                                  G_times_SA_times_WBD:base.SparseNDArray,
                                  G_times_WAC_times_SB:base.SparseNDArray,
                                  Ylm_table:np.ndarray):
        '''Computes a row of the window kernels. This function is called in parallel for each k1 bin.
        Gives window kernels for L=0,2,4 auto and cross covariance

        Args:
            idx (int):, the index of the current k1 bin
            bin_kmodes (np.ndarray): 4D array of x, y, z, and r coordinates of sampled modes in the current k1 bin
            G_times_SA_times_WBC (SparseNDArray): Precomputed product of the Gaunt coefficients and S_A * W_BC
            G_times_WAD_times_SB (SparseNDArray): Precomputed product of the Gaunt coefficients and W_AD * S_B
            G_times_SA_times_WBD (SparseNDArray): Precomputed product of the Gaunt coefficients and S_A * W_BD
            G_times_WAC_times_SB (SparseNDArray): Precomputed product of the Gaunt coefficients and W_AC * S_B
            Ylm_table (np.ndarray): Precomputed Ylm callables for each ell
        Returns:
            WinKernel (np.ndarray): an array with [2*delta_k_max+1,num_ell,num_ell,num_ell,num_ell] dimensions.
                The first dim corresponds to the k-bin of k2
                (only 3 bins on each side of diagonal are included by default as the Gaussian covariance drops quickly away from diagonal)
                The remaining dims correspond to specific ells
        '''

        # k1_bin_index is a scalar
        k1_bin_index = idx + self.k_binning.kmin//self.dk
        
        WinKernel_mixed = np.zeros((4, 2*self.delta_k_max+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1), dtype=np.complex128)
        iix, iiy, iiz = np.meshgrid(*self.window_AB.ikgrid, indexing='ij')

        k2xh = np.zeros_like(iix)
        k2yh = np.zeros_like(iiy)
        k2zh = np.zeros_like(iiz)
        kfun = 2 * np.pi / self.boxsize

        mode_idx = 1
        t_avg = 0
        for ik1x, ik1y, ik1z, ik1r in bin_kmodes:
            t_start = time.time()
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
            k2_bin_index = (k2r * kfun / self.dk).astype(int)
            k2r[k2r <= 1e-10] = np.inf
            k2xh /= k2r
            k2yh /= k2r
            k2zh /= k2r
            
            # Evaluate ylm factors at the given k1 and k2 modes
            Ylm_k1 = math.evaluate_Ylms(Ylm_table, self.pk_ellmax, k1xh, k1yh, k1zh)
            Ylm_k2 = math.evaluate_Ylms(Ylm_table, self.pk_ellmax, k2xh, k2yh, k2zh)

            result_1 = np.zeros((list(G_times_SA_times_WBC.shape_in) + [3,3,3]), dtype=np.complex128)
            result_2 = np.zeros((list(G_times_WAD_times_SB.shape_in) + [3,3,3]), dtype=np.complex128)
            result_3 = np.zeros((list(G_times_SA_times_WBD.shape_in) + [3,3,3]), dtype=np.complex128)
            result_4 = np.zeros((list(G_times_WAC_times_SB.shape_in) + [3,3,3]), dtype=np.complex128)
            # multiply by Ylms
            for l1, l2, l3 in itt.product(np.arange(0, self.pk_ellmax+1, 2), repeat=3):
                l1_idx = int(l1 / 2)
                l2_idx = int(l2 / 2)
                l3_idx = int(l3 / 2)
                
                for m1, m2, m3 in itt.product(*[np.arange(-l, l+1, 2) for l in (l1, l2, l3)]):
                    m1_idx = int((m1 + l1) / 2)
                    m2_idx = int((m2 + l2) / 2)
                    m3_idx = int((m3 + l3) / 2)

                    G_SA_WBC = G_times_SA_times_WBC[l1_idx,l2_idx,l3_idx,m1_idx,m2_idx,m3_idx]
                    G_WAD_SB = G_times_WAD_times_SB[l1_idx,l2_idx,l3_idx,m1_idx,m2_idx,m3_idx]
                    G_SA_WBD = G_times_SA_times_WBD[l1_idx,l2_idx,l3_idx,m1_idx,m2_idx,m3_idx]
                    G_WAC_SB = G_times_WAC_times_SB[l1_idx,l2_idx,l3_idx,m1_idx,m2_idx,m3_idx]

                    Ylms = Ylm_k1[l1_idx][m1_idx] * \
                           Ylm_k2[l2_idx][m2_idx] * \
                           Ylm_k2[l3_idx][m3_idx]
                    
                    result_1[:,:,:,l1_idx,l2_idx,l3_idx] += Ylms * G_SA_WBC.toarray().reshape(G_times_SA_times_WBC.shape_in)
                    result_2[:,:,:,l1_idx,l2_idx,l3_idx] += Ylms * G_WAD_SB.toarray().reshape(G_times_WAD_times_SB.shape_in)
                    result_3[:,:,:,l1_idx,l2_idx,l3_idx] += Ylms * G_SA_WBD.toarray().reshape(G_times_SA_times_WBD.shape_in)
                    result_4[:,:,:,l1_idx,l2_idx,l3_idx] += Ylms * G_WAC_SB.toarray().reshape(G_times_WAC_times_SB.shape_in)

            for delta_k in range(-self.delta_k_max, self.delta_k_max + 1):
                modes = (k2_bin_index - k1_bin_index == delta_k)
                if np.any(modes == True):
                    WinKernel_mixed[0, delta_k] = np.sum(result_1[modes], axis=0)
                    WinKernel_mixed[1, delta_k] = np.sum(result_2[modes], axis=0)
                    WinKernel_mixed[2, delta_k] = np.sum(result_3[modes], axis=0)
                    WinKernel_mixed[3, delta_k] = np.sum(result_4[modes], axis=0)

            t_avg += time.time() - t_start
            self.logger.debug(f"process {os.getpid()}, mode {mode_idx} / {len(bin_kmodes)} done. Avg time per iteration = {t_avg / mode_idx:.1f}s")
            mode_idx += 1

        return WinKernel_mixed

    def _compute_shotnoise_kernel(self, cache_dir, kmodes, Nmodes, weights):

        if hasattr(self, "WinKernel_shotnoise") and self.WinKernel_shotnoise is not None:
            return
        else:
            if self.rank == 0: self.logger.info("Computing shotnoise term kernels...")
            # Format is [k1_bins, k2_bins, l1, l2, l3, l4]
            self.WinKernel_shotnoise = np.empty([self.k_binning.kbins, 2*delta_k_max+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1])
            self.WinKernel_shotnoise.fill(np.nan)

        if self.rank == 0:
            self.logger.info("Retrieving shotnoise windows S_AB...")
            G = self.get_shotnoise_gaunt_coefficients(pk_ellmax=self.pk_ellmax, mask_ellmax=self.mask_ellmax)
        else:
            G = None
        S_A_temp = self.get_shotnoise_window("A", cache_dir)
        S_B_temp = self.get_shotnoise_window("B", cache_dir)
        S_AB_temp = self.get_shotnoise_window("AB", cache_dir)

        self.comm.Barrier()
        
        S_A = S_A_temp.to_shared_memory(self.comm)
        S_B = S_B_temp.to_shared_memory(self.comm)
        S_AB = S_AB_temp.to_shared_memory(self.comm)
        del S_A_temp, S_B_temp, S_AB_temp
        G = self.comm.bcast(G, root=0)

        G_times_S = G @ S_AB
        self.comm.Barrier()
        del S_AB

        # load in ylm callables
        Ylm_table = math.build_Ylm_table(self.pk_ellmax)

        #ell_factor = lambda l1,l2: (2*l1 + 1) * (2*l2 + 1) * (2 if 0 in (l1, l2) else 1)
        last_save = time.time()

        if self.rank == 0:
            self.logger.info(f"Beginning window kernel calculations on {self.size} ranks...")
            pbar = self.tqdm(desc='Computing window kernels', total=math.num_sampled_modes(kmodes))
        
        # TODO: Now that we're using mpi4py, come up with a more efficient way to loop thru modes
        for i, km in enumerate(kmodes):

            self.comm.Barrier()
            if hasattr(self, '_resume_file') and self._resume_file is not None and self.rank == 0:
                # Skip rows that were already computed
                if not np.isnan(self.WinKernel_shotnoise[i,0,0,0,0,0]):
                    self.logger.debug(f'Skipping bin {i} of {self.k_binning.kbins}.')
                    continue

            kmodes_sampled = len(km)
            # Splitting kmodes in chunks to be sent to each rank
            kmodes_per_rank = np.array_split(km, self.size)[self.rank]

            results_per_rank = self._compute_shotnoise_kernel_row(i, kmodes_per_rank, G_times_S, S_A, S_B, Ylm_table)
            self.comm.Barrier()

            results_per_rank = np.sum(results_per_rank, axis=0)
            if self.rank == 0:
                results_combined = np.zeros_like(results_per_rank)
            else:
                results_combined = None # None on non-root processes
            self.comm.Reduce(results_per_rank, results_combined, op=MPI.SUM, root=0)
    
            if self.rank == 0:
                self.WinKernel_shotnoise[i] = results_combined.real * weights[i] / kmodes_sampled
                for k2_bin_index in range(0, 2*self.delta_k_max + 1):
                    if (k2_bin_index + i - self.delta_k_max >= self.k_binning.kbins or k2_bin_index + i - self.delta_k_max < 0):
                        self.WinKernel_shotnoise[i, k2_bin_index, :, :] = 0
                    else:
                        self.WinKernel_shotnoise[i, k2_bin_index, :, :] /= Nmodes[i + k2_bin_index - self.delta_k_max]

                pbar.update(len(kmodes[i]))
                if hasattr(self, '_resume_file') and self._resume_file is not None and (time.time() - last_save) > 600:
                    self.logger.debug("Saving progress...")

                    self.save(self._resume_file)
                    last_save = time.time()
            
        if self.rank == 0: self.logger.info('Shotnoise window kernel computed.')

        if self._resume_file is not None and self.rank == 0:
            self.save(self._resume_file)


    def _compute_shotnoise_kernel_row(self, idx, bin_kmodes, product, S_A, S_B, Ylm_table):

        k1_bin_index = idx + self.k_binning.kmin//self.dk

        WinKernel = np.zeros((2*self.delta_k_max+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1, self.pk_ellmax//2+1), dtype=np.complex128)
        iix, iiy, iiz = np.meshgrid(*self.window_AB.ikgrid, indexing='ij')

        k2xh = np.zeros_like(iix)
        k2yh = np.zeros_like(iiy)
        k2zh = np.zeros_like(iiz)
        kfun = 2 * np.pi / self.boxsize

        mode_idx = 1
        t_avg = 0
        for ik1x, ik1y, ik1z, ik1r in bin_kmodes:
            t_start = time.time()
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
            k2_bin_index = (k2r * kfun / self.dk).astype(int)
            k2r[k2r <= 1e-10] = np.inf
            k2xh /= k2r
            k2yh /= k2r
            k2zh /= k2r
            
            # Evaluate ylm factors at the given k1 and k2 modes
            Ylm_k1 = math.evaluate_Ylms(Ylm_table, self.pk_ellmax, k1xh, k1yh, k1zh)
            Ylm_k2 = math.evaluate_Ylms(Ylm_table, self.pk_ellmax, k2xh, k2yh, k2zh)

            result = np.zeros((list(product.shape_in) + [3,3]), dtype=np.complex128)
            # multiply by Ylms
            for l1, l2 in itt.product(np.arange(0, self.pk_ellmax+1, 2), repeat=2):
                l1_idx = int(l1 / 2)
                l2_idx = int(l2 / 2)
                
                for m1, m2 in itt.product(*[np.arange(-l, l+1, 2) for l in (l1, l2)]):
                    m1_idx = int((m1 + l1) / 2)
                    m2_idx = int((m2 + l2) / 2)

                    G_times_S = product[l1_idx,l2_idx,m1_idx,m2_idx]
                    s_a = S_A[l1_idx,m1_idx].toarray().reshape([self.nmesh, self.nmesh, self.nmesh])
                    s_b = S_B[l2_idx,m2_idx].toarray().reshape([self.nmesh, self.nmesh, self.nmesh])

                    Ylms = Ylm_k1[l1_idx][m1_idx] * \
                           Ylm_k2[l2_idx][m2_idx]
                    
                    result[:,:,:,l1_idx,l2_idx] += Ylms * (s_a @ s_b + (G_times_S).toarray().reshape(product.shape_in))
            
            for delta_k in range(-self.delta_k_max, self.delta_k_max + 1):
                modes = (k2_bin_index - k1_bin_index == delta_k)
                if np.any(modes == True):
                    WinKernel[delta_k] = np.sum(result[modes], axis=0)

            t_avg += time.time() - t_start
            self.logger.debug(f"process {os.getpid()}, mode {mode_idx} / {len(bin_kmodes)} done. Avg time per iteration = {t_avg / mode_idx:.1f}s")
            mode_idx += 1

        return WinKernel