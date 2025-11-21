'''Module containing basic classes to deal with covariance matrices.'''

import os, time, copy
from mpi4py import MPI
import pickle
import itertools as itt

import numpy as np
import scipy

from . import utils, binning
import logging

__all__ = ['BaseClass',
           'Covariance',
           'MultipoleCovariance',
           'MultipoleFourierCovariance']

class BaseClass:
    """
    Base class that implements copy, save/load, etc.
    """
    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self, **kwargs):
        new = self.__copy__()
        new.__dict__.update(kwargs)
        return new

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.__dict__.get("comm") is None:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def get_pickleable_state(self):
        """Same as __getstate__, except only returns picklable variables"""
        state = self.__dict__.copy()
        # Drop MPI communicators or other non-pickleable attributes
        for key, _ in state.items():
            if "comm" in key or "rank" in key or "size" in key or "window_" in key:
                state[key] = None

        return state

    # @classmethod
    # def from_state(cls, state):
    #     new = cls.__new__(cls)
    #     new.__setstate__(state)
    #     return new

    @property
    def with_mpi(self):
        """Whether to use MPI."""
        return getattr(self, 'mpicomm', None) is not None and self.mpicomm.size > 1

    def save(self, filename):
        """Save to ``filename``."""
        start = time.time()
        if not self.with_mpi or self.mpicomm.rank == 0:
            
            utils.mkdir(os.path.dirname(filename))
            with open(filename, "wb") as f:
                pickle.dump(self.get_pickleable_state(), f, protocol=pickle.HIGHEST_PROTOCOL)
        # if self.with_mpi:
        #     self.mpicomm.Barrier()

        if hasattr(self, 'logger'):
            self.logger.info(f'Saved to {filename} in {time.time() - start:.3f}s.')

    @classmethod
    def load(cls, filename):
        # state = np.load(filename, allow_pickle=True)[()]
        # new = cls.from_state(state)
        with open(filename, "rb") as f:
            state = pickle.load(f)
        return state
        #     new = cls.from_state(state)
        # return new

class Covariance(BaseClass):
    '''A class that represents a covariance matrix.
    Implements basic operations such as correlation matrix computation, etc.
    '''

    def __init__(self, covariance=None):
        '''Initializes a Covariance object.

        Parameters
        ----------
        covariance : numpy.ndarray
            (n,n) numpy array with elements corresponding to the covariance.
        '''
        self._cov = covariance
        self._ells = []
        self._mshape = (0, 0)

    @property
    def cov(self):
        '''The covariance matrix.

        Returns
        -------
        numpy.ndarray
            (n,n) numpy array corresponding to the elements of the covariance matrix.
        '''

        return self._cov

    @cov.setter
    def cov(self, covariance):
        '''Sets the covariance matrix.

        Parameters
        ----------
        covariance : numpy.ndarray
            (n,n) numpy array with elements corresponding to the covariance.
        '''

        self._cov = covariance

    @property
    def cor(self):
        '''Returns the correlation matrix.

        The correlation matrix is obtained by dividing each element of the covariance matrix by
        the product of the standard deviations of the corresponding variables.

        Returns
        -------
        numpy.ndarray
            (n,n) numpy array corresponding to the elements of the correlation matrix.
        '''

        cov = self.cov
        v = np.sqrt(np.diag(cov))
        outer_v = np.outer(v, v)
        outer_v[outer_v == 0] = np.inf
        cor = cov / outer_v
        cor[cov == 0] = 0
        return cor

    def symmetrize(self):
        """Symmetrizes the covariance matrix in place."""
        self.cov = (self.cov + self.cov.T)/2

    def symmetrized(self):
        '''Returns a symmetrized copy of the covariance matrix.

        Returns
        -------
        Covariance
            Covariance object corresponding to the symmetrized covariance matrix.
        '''
        new_cov = self.copy()
        new_cov.symmetrize()
        return new_cov
    
    def regularize(self, mode='zero'):
        eigvals, eigvecs = self.eig
        if mode == 'zero':
            eigvals[eigvals < 0] = 0
        elif mode == 'flip':
            eigvals = np.abs(eigvals)
        elif mode == 'minpos':
            eigvals[eigvals < 0] = min(eigvals[eigvals > 0])
        self.cov = np.einsum('ij,jk,kl->il', eigvecs, np.diag(eigvals), eigvecs.T)
    
    def regularized(self):
        new_cov = self.copy()
        new_cov.regularize()
        return new_cov

    def __add__(self, y):
        return Covariance(self.cov + (y.cov if isinstance(y, Covariance) else y))

    def __sub__(self, y):
        return self.__add__(-y)

    def __mul__(self, y):
        return Covariance(self.cov * y)

    def __truediv__(self, y):
        return Covariance(self.cov / y)

    @property
    def T(self):
        '''Returns the transpose of the covariance matrix.

        Returns
        -------
        Covariance
            Covariance object corresponding to the transpose of the covariance matrix.
        '''

        new_cov = self.copy()
        new_cov.cov = new_cov.cov.T
        return new_cov

    @property
    def shape(self):
        '''Returns the shape of the covariance.

        Returns
        -------
        tuple
            A tuple with the shape of the covariance matrix.
        '''

        return self.cov.shape

    @property
    def eig(self):
        '''Compute the eigenvalues and right eigenvectors of the covariance.

        Returns
        -------
        A namedtuple with the following attributes:
            eigenvalues
            (..., M) array
                The eigenvalues, each repeated according to its multiplicity.
                The eigenvalues are not necessarily ordered. The resulting
                array will be of complex type, unless the imaginary part is
                zero in which case it will be cast to a real type. When a is
                real the resulting eigenvalues will be real (0 imaginary
                part) or occur in conjugate pairs

            eigenvectors
            (...), M, M) array
                The normalized (unit “length”) eigenvectors, such that the
                column eigenvectors[:,i] is the eigenvector corresponding to
                the eigenvalue eigenvalues[i].
        '''

        return np.linalg.eig(self.cov)

    @property
    def eigvals(self):
        '''Compute the eigenvalues of the covariance.

        Returns
        -------
        (..., M,) ndarray
            The eigenvalues, each repeated according to its multiplicity.
            They are not necessarily ordered, nor are they necessarily
            real for real matrices.
        '''

        return np.linalg.eigvals(self.cov)

    def savetxt(self, filename):
        '''Saves the covariance as a text file with a specified filename.

        Parameters
        -------
        filename : string
            The name of the file where the covariance matrix will be saved.
        '''
        utils.mkdir(os.path.dirname(filename))
        np.savetxt(filename, self.cov)

    @classmethod
    def loadtxt(cls, *args, **kwargs):
        '''Loads the covariance from a text file with a specified filename.

        Parameters
        -------
        *args
            Arguments to be passed to numpy.loadtxt.
        **kwargs
            Keyword arguments to be passed to numpy.loadtxt.

        Returns
        -------
        Covariance
            Covariance object.
        '''

        return cls.from_array(np.loadtxt(*args, **kwargs))

    @classmethod
    def from_array(cls, a):
        '''Creates a Covariance object from a numpy array.

        Parameters
        -------
        numpy.ndarray
            (n,n) numpy array with elements corresponding to the covariance.

        Returns
        -------
        Covariance
            Covariance object.
        '''

        return cls(covariance=a)

    @property
    def ells(self):
        '''The multipoles for which the covariance matrix is defined. Sorted in ascending order.

        Returns
        -------
        tuple
            A tuple of multipole values.
        '''

        return sorted(self._ells)

class MultipoleCovariance(Covariance):
    '''A class to represent a covariance matrix for a set of multipoles.

    Attributes
    ----------
    cov : numpy.ndarray
        The covariance matrix.
    cor : numpy.ndarray
        The correlation matrix.
    '''

    def __init__(self, symmetric=False):
        super().__init__() # <- calls Covariance.__init__()
        self._multipole_covariance = {}
        self._symmetric = symmetric

    def set_ell_cov(self, l1, l2, cov, cls=Covariance):
        '''Sets the covariance matrix for a given pair of multipoles.

        Parameters
        ----------
        l1 : int
            The first multipole.
        l2 : int
            The second multipole.
        cov : Covariance or numpy.ndarray
            The covariance matrix. Can be an instance of Covariance or a numpy array.
        cls : class, optional
            The class to be used to create the covariance matrix if cov is a numpy array.
        '''

        if l1 > l2:
            return self.set_ell_cov(l2, l1, cov.T if cov is not None else None)

        if self._ells == []:
            self._mshape = cov.shape
        else:
            if l1 not in self.ells:
                self._ells.append(l1)
            if l2 not in self.ells:
                self._ells.append(l2)

        cov = cov if isinstance(cov, cls) else cls(cov)

        self._multipole_covariance[l1, l2] = cov

        return cov

    def get_ell_cov(self, l1, l2,  force_return=False, cls=Covariance):
        '''Returns the covariance matrix for a given pair of multipoles.

        Parameters
        ----------
        l1
            the first multipole.
        l2
            the second multipole.

        Returns
        -------
        Covariance
            A Covariance object corresponding to the covariance matrix for the given multipoles.
        '''

        if l1 > l2:
            return self.get_ell_cov(l2, l1, cls=cls).T

        if (l1, l2) in self._multipole_covariance:
            return self._multipole_covariance[l1, l2]
        elif type(force_return) != bool:
            return cls(force_return*np.ones(self._mshape))
        elif force_return:
            return cls(np.zeros(self._mshape))

    def is_ell_set(self, l1, l2):
        return (l1,l2) in self._multipole_covariance.keys()

    @property
    def ells(self):
        '''Returns sorted lists of unique first and second multipoles used in the covariance matrices.

        Returns
        -------
        tuple of two lists
        '''
        ells1, ells2 = set(), set()
        
        for (l1, l2) in self._multipole_covariance.keys():
            ells1.add(l1)
            ells2.add(l2)
            
        return sorted(ells1), sorted(ells2)

    def has_ells(self, l1, l2):
        if self._symmetric and l1 > l2:
            return self.has_ells(l2, l1)
        return (l1, l2) in self._multipole_covariance.keys()

    def foreach(self, func):
        '''Applies a function to each covariance matrix.

        Parameters
        ----------
        func : function
            The function to be applied to each covariance matrix.
        '''

        for (l1, l2), cov in self._multipole_covariance.items():
            self.set_ell_cov(l1, l2, func(cov))
        
        return self

    @property
    def symmetric(self):
        return self._symmetric

    @ells.setter
    def ells(self, ells):
        '''Initializes all entries in the self._multipole_covariance dict based on the input ells tuple.

        Parameters
        ----------
        ells : tuple
            A tuple of two lists: (l1s, l2s), where l1s and l2s are lists of multipoles.
        '''
        
        # Initialize the covariance matrices for each pair
        for l1 in ells[0]:
            for l2 in ells[1]:
                if not self.has_ells(l1,l2):
                    self.set_ell_cov(l1, l2, None)

    @property
    def cov(self):
        '''This function calculates the full covariance matrix by stacking covariances for different multipoles
        in ascending order.

        Returns
        -------
        numpy.ndarray
            An (n,n) numpy array corresponding to the elements of the covariance matrix.
        '''

        ells1, ells2 = self.ells

        cov_return = np.zeros(np.array(self._mshape)*len(ells1))
        for (i, l1), (j, l2) in itt.product(enumerate(ells1), enumerate(ells2)):
            cov_return[i*self._mshape[0]:(i+1)*self._mshape[0],
                j*self._mshape[1]:(j+1)*self._mshape[1]] = self.get_ell_cov(l1, l2).cov
        return cov_return

    @cov.setter
    def cov(self, cov):
        '''Sets the full covariance matrix from covariances for different multipoles stacked
        in ascending order.

        Parameters
        ----------
        cov : numpy.ndarray
            An (n,n) numpy array corresponding to the elements of the covariance matrix.
        '''

        ells1, ells2 = self.ells

        assert cov.ndim == 2, "Covariance should be a matrix (ndim == 1)."
        assert cov.shape[0] % len(ells1) == 0, \
            "Can't resolve covariance structure as shape is not a multiple of the number of ells."
        assert cov.shape[1] % len(ells2) == 0, \
            "Can't resolve covariance structure as shape is not a multiple of the number of ells."

        size1 = cov.shape[0]//len(ells1)
        size2 = cov.shape[1]//len(ells2)

        for i1,l1 in enumerate(ells1):
            for i2,l2 in enumerate(ells2):
                self.set_ell_cov(l1,l2,Covariance(cov[i1*size1:(i1+1)*size1,i2*size2:(i2+1)*size2]))

    def __add__(self, y):
        assert isinstance(y, MultipoleCovariance)

        cov = MultipoleCovariance(symmetric=self.symmetric and y.symmetric)
        ells1, ells2 = self.ells
        for l1 in ells1:
            for l2 in ells2:
                cov.set_ell_cov(l1,l2, self.get_ell_cov(l1,l2) + y.get_ell_cov(l1,l2))
        return cov

    def __sub__(self, y):
        return self.__add__(-y)

    def __mul__(self, y):
        cov = self.deepcopy()
        cov.foreach(lambda x: x*y)
        return cov

    def __truediv__(self, y):
        return self * (1/y)

    @classmethod
    def from_array(cls, cov):
        '''Creates a MultipoleCovariance object from a numpy array corresponding to the full covariance matrix.

        Parameters
        ----------
        cov
            (n,n) numpy array with elements corresponding to the covariance.
        ells
            the multipoles for which the covariance matrix is defined.

        Returns
        -------
        MultipoleCovariance
            A MultipoleCovariance object.
        '''

        cov = cls()
        cov.cov = cov

        return cov


class MultipoleFourierCovariance(MultipoleCovariance):

    def __init__(self, binning_type="linear"):
        
        super().__init__()
        if binning_type == "linear":
            self.k_binning = binning.LinearBinning()
        elif binning_type == "log":
            self.k_binning = binning.LogBinning()
        else:
            raise ValueError("Binning must be either 'linear' or 'log'.")
        self.logger = logging.getLogger('MultipoleFourierCovariance')

    @property
    def kmid_matrices(self):
        k1 = np.einsum('i,j->ij', self.k_binning.kmid, np.ones(self.k_binning.kbins))
        k2 = np.einsum('i,j->ji', self.k_binning.kmid, np.ones(self.k_binning.kbins))

        return k1, k2

    @property
    def kmin_matrices(self):
        k1 = np.einsum('i,j->ij', self.k_binning.kedges[:-1], np.ones(self.k_binning.kbins))
        k2 = np.einsum('i,j->ji', self.k_binning.kedges[:-1], np.ones(self.k_binning.kbins))

        return k1, k2

    def kcut(self, kmin=None, kmax=None):
        if kmin is None:
            kmin = self.k_binning.kmin

        if kmax is None:
            kmax = self.k_binning.kmax

        imin = (self.k_binning.kmid >= kmin).argmax()
        imax = len(self.k_binning.kmid) if (self.k_binning.kmid <= kmax).all() else (self.k_binning.kmid <= kmax).argmin()

        self._covariance = self._covariance[imin:imax, imin:imax]
        self.k_binning.kmin, self.k_binning.kmax = kmin, kmax

        return self

    @property
    def kmid_ell_matrices(self):
        ells1, ells2 = self.ells

        kfull1 = np.concatenate([self.k_binning.kmid for _ in ells1])
        kfull2 = np.concatenate([self.k_binning.kmid for _ in ells2])

        k1 = np.einsum('i,j->ij', kfull1, np.ones_like(kfull2))
        k2 = np.einsum('i,j->ji', kfull2, np.ones_like(kfull1))

        return k1, k2

    @property
    def ell_matrices(self):
        ells1, ells2 = self.ells

        kells1 = np.einsum('i,j->ij', ells1, np.ones(self.k_binning.kbins)).flatten()
        kells2 = np.einsum('i,j->ji', ells2, np.ones(self.k_binning.kbins)).flatten()

        ell1 = np.einsum('i,j->ij', kells1, np.ones_like(kells2))
        ell2 = np.einsum('i,j->ji', kells2, np.ones_like(kells1))

        return ell1, ell2

    def savecsv(self, filename, ells_both_ways=False, fmt=['%.d', '%.d', '%.4f', '%.4f', '%.8e']):
        k1, k2 = self.kmid_ell_matrices
        ell1, ell2 = self.ell_matrices
        cov = self.cov

        mask = ell1 <= ell2 if ells_both_ways else np.ones_like(ell1, dtype=bool)

        utils.mkdir(os.path.dirname(filename))
        np.savetxt(filename, np.concatenate([ell1[mask].reshape(-1, 1),
                                             ell2[mask].reshape(-1, 1),
                                               k1[mask].reshape(-1, 1),
                                               k2[mask].reshape(-1, 1),
                                              cov[mask].reshape(-1, 1)], axis=1), fmt=fmt, header='ell1 ell2 kmid1 kmid2 cov')
    def loadcsv(self, filename):
        ell1, ell2, k1, k2, value = np.loadtxt(filename).T

        k1 = np.unique(k1)
        kbins = len(k1)

        assert np.allclose(k1, np.unique(k2)), "k1 and k2 are not consistent"

        dk = np.mean(np.diff(k1))
        kmin = k1.min() - dk/2
        kmax = k1.max() + dk/2
        self.set_kbins(kmin, kmax, dk=dk)

        ells = np.unique(ell1)
        assert np.allclose(ells, np.unique(ell2)), "ell1 and ell2 are not consistent"

        ells_both_ways = len(value) == (len(ells)*kbins)**2
        ells_one_way   = len(value) == (len(ells)**2 + len(ells))/2 * kbins**2

        assert ells_one_way or ells_both_ways, 'length of covariance file doesn\'nt match'

        assert np.allclose(np.unique(k1), self.k_binning.kmid), "k bins are not linearly spaced"

        kmid_matrix = np.einsum('i,j->ij', k1, np.ones_like(k1))

        for l1, l2 in itt.combinations_with_replacement(ells, r=2):
            block_mask = (ell1 == l1) & (ell2 == l2)
            print(block_mask , block_mask.shape)
            #assert np.allclose(k1[block_mask].reshape(kmid_matrix.shape),   kmid_matrix)
            #assert np.allclose(k2[block_mask].reshape(kmid_matrix.T.shape), kmid_matrix.T)
            c = value[block_mask].reshape(kbins, kbins)
            self.set_ell_cov(l1, l2, c)

        return self

    @classmethod
    def fromcsv(cls, filename):
        cov = cls()
        cov.loadcsv(filename)
        return cov
    
    def set_ell_cov(self, l1, l2, cov, cls=Covariance):
        cov = super().set_ell_cov(l1, l2, cov, cls=cls)
        if not self.k_binning.is_kbins_set:
            self.k_binning.set_kbins(self.k_binning.kmin, 
                                     self.k_binning.kmax, 
                                     self.k_binning.dk)
        return cov

    def kcut(self, kmin=None, kmax=None):
        self.foreach(lambda cov: cov.kcut(kmin, kmax))
        self.set_kbins(kmin, kmax, self.k_binning.dk)
        
        self.logger.info(f'kcut to {self.kmin} < k < {self.kmax}')

        return self
    
    def set_kbins(self, kmin, kmax, dk, nmodes=None):
        return self.k_binning.set_kbins(kmin, kmax, dk, nmodes)

    @property
    def kbins(self):
        return self.k_binning.kbins

class SparseNDArray:
    """
    A class to represent a sparse ND array using scipy.sparse.csr_matrix.
    Indices are split between shape_out and shape_in, as if the array is
    a 2D matrix (shape_out x shape_in). Matrix multiplication is done using
    the @ operator and requires the shapes to be compatible, i.e., shape_in
    of the leftmost array must match shape_out of the rightmost array.
    """
    def __init__(self, shape_out, shape_in):
        self.shape_in = np.asarray(shape_in).astype(int)
        self.shape_out = np.asarray(shape_out).astype(int)
        self._matrix = scipy.sparse.csr_matrix((np.prod(shape_out), np.prod(shape_in)))

    def _nd_to_2d_indices(self, *indices):
        indices = np.asarray(indices).astype(int)
        if len(indices) == len(self.shape_out) + len(self.shape_in):
            i = np.ravel_multi_index(indices[:len(self.shape_out)], self.shape_out)
            j = np.ravel_multi_index(indices[len(self.shape_out):], self.shape_in)
            return i,j
        elif len(indices) == len(self.shape_out):
            i = np.ravel_multi_index(indices, self.shape_out)
            # j = np.arange(np.prod(self.shape_in))
            return i
        
    
    def __setitem__(self, indices, value):
        indices = np.asarray(indices).astype(int)
        if len(indices) == len(self.shape_out) + len(self.shape_in):
            try:
                self._matrix[self._nd_to_2d_indices(*indices)] = value
            except IndexError:
                raise IndexError(f"Indices {indices} are out of bounds for array with shape shape_out={self.shape_out}, shape_in={self.shape_in}.")
        elif len(indices) == len(self.shape_out):
            if isinstance(value, SparseNDArray):
                self._matrix[self._nd_to_2d_indices(*indices)] = value._matrix
            elif isinstance(value, scipy.sparse.csr_matrix):
                self._matrix[self._nd_to_2d_indices(*indices)] = value
            elif isinstance(value, scipy.sparse.csc_matrix):
                self._matrix[self._nd_to_2d_indices(*indices)] = value.T
            else:
                self._matrix[self._nd_to_2d_indices(*indices)] = value.flatten()
        else:
            raise ValueError(f"Invalid number of indices: {len(indices)}. Expected {len(self.shape_out) + len(self.shape_in)} or {len(self.shape_out)}.")

    def __getitem__(self, indices):
        indices = np.asarray(indices).astype(int)
        return self._matrix[self._nd_to_2d_indices(*indices)]

    def __repr__(self):
        return f"SparseNDArray(shape_out={self.shape_out} -> {np.prod(self.shape_out)}, shape_in={self.shape_in} -> {np.prod(self.shape_in)}, nnz={self._matrix.nnz})"
    
    def to_dense(self):
        """
        Convert the sparse matrix back to a dense ND array.
        """
        return self._matrix.toarray().reshape(self.shape_out.tolist() + self.shape_in.tolist())

    @staticmethod
    def from_dense(dense_array, shape_out=None, shape_in=None):
        """
        Create a SparseNDArray from a dense array.
        """
        if shape_out is None:
            shape_out = dense_array.shape[:-len(dense_array.shape)//2]
        if shape_in is None:
            shape_in = dense_array.shape[len(dense_array.shape)//2:]
        sparse_array = SparseNDArray(shape_in, shape_out)
        sparse_array._matrix = scipy.sparse.csr_matrix(dense_array.reshape(np.prod(shape_out), np.prod(shape_in)))
        return sparse_array
    
    def __add__(self, other):
        if isinstance(other, SparseNDArray):
            assert (self.shape_in == other.shape_in) and (self.shape_out == other.shape_out), \
                "Shapes do not match for multiplication."
            
            import copy
            other = copy.deepcopy(other)
            other._matrix += self._matrix
            return other
        else:
            raise ValueError(f"Operation not supported between {self.__class__} and {other.__class__}.")
        
    def __mul__(self, other):
        if isinstance(other, SparseNDArray):
            assert (self.shape_in == other.shape_in) and (self.shape_out == other.shape_out), \
                "Shapes do not match for multiplication."
            
            import copy
            other = copy.deepcopy(other)
            other._matrix *= self._matrix
            return other
        elif isinstance(other, scipy.sparse.csr_matrix):

            import copy
            result = copy.deepcopy(self)
            result._matrix = self._matrix * other
            return result
        else:
            raise ValueError(f"Operation not supported between {self.__class__} and {other.__class__}.")
        
    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, SparseNDArray):
            assert (np.all(self.shape_in == other.shape_out)), \
                "Shapes do not match for matrix multiplication."
            other = copy.deepcopy(other)
            other._matrix = self._matrix.dot(other._matrix)
            other.shape_out = self.shape_out
            return other
        
        elif isinstance(other, np.ndarray):
            result = copy.deepcopy(self)
            result._matrix = scipy.sparse.csr_matrix(self._matrix.dot(other.reshape(np.prod(self.shape_in), -1)))
            result.shape_in = other.shape[len(self.shape_in):]
            return result
        
        else:
            raise ValueError(f"Operation not supported between {self.__class__} and {other.__class__}.")
        
    def __sizeof__(self):
        return self._matrix.data.nbytes + self._matrix.indptr.nbytes + self._matrix.indices.nbytes
    
    def save(self, filename):
        """
        Save the sparse matrix to a file.
        """
        np.savez(filename,
                 data=self._matrix.data,
                 indices=self._matrix.indices,
                 indptr=self._matrix.indptr,
                 shape=self._matrix.shape,
                 shape_out=self.shape_out,
                 shape_in=self.shape_in)

    @classmethod
    def load(cls, filename):
        """
        Load the sparse matrix from a file.
        """
        loader = np.load(filename)
        obj = cls(loader['shape_out'], loader['shape_in'])
        obj._matrix = scipy.sparse.csr_matrix((loader['data'],
                                               loader['indices'],
                                               loader['indptr']),
                                               shape=loader['shape'])
        return obj

    def reshape(self, shape_out=None, shape_in=None):
        """
        Reshape the sparse matrix.
        """
        result = copy.deepcopy(self)
        result._matrix = self._matrix.reshape(np.prod(shape_out), np.prod(shape_in))
        if shape_out is not None:
            result.shape_out = shape_out
        if shape_in is not None:
            result.shape_in = shape_in
        return result

    def transpose(self):
        """
        Transpose the sparse matrix.
        """
        result = copy.deepcopy(self)
        result._matrix = self._matrix.transpose()
        result.shape_out, result.shape_in = self.shape_in, self.shape_out
        return result
    
    @property
    def T(self):
        """
        Transpose the sparse matrix.
        """
        return self.transpose()
    
    # def outer(self, other):
    #     """
    #     Compute the outer product of two SparseNDArrays. 
    #     """
    #     if isinstance(other, SparseNDArray):
    #         assert (self.shape_in == other.shape_in), \
    #             "Shapes do not match for outer product."

    #         result_shape_out =  np.atleast_1d(self.shape_out).tolist() + np.atleast_1d(other.shape_out).tolist()

    #         result = SparseNDArray(shape_out=result_shape_out, shape_in=self.shape_in)

    #         # Perform the equivalent of np.einsum('ab,cb->acb', sparse_matrix_a, sparse_matrix_b)
    #         result._matrix = scipy.sparse.csr_matrix(
    #             (self._matrix.data[:, None] * other._matrix.data[None, :],
    #              (np.outer(self._matrix.indices, np.ones(other._matrix.shape[1], dtype=int)).flatten(),
    #              np.outer(np.ones(self._matrix.shape[0], dtype=int), other._matrix.indices).flatten())),
    #             shape=(np.prod(result_shape_out), np.prod(other.shape_out))
    #         )

    #         return result
    #     else:
    #         raise ValueError(f"Operation not supported between {self.__class__} and {other.__class__}.")

    @classmethod
    def from_arrays(cls, data, indices, indptr, shape_out, shape_in):
        """
        Create a SparseNDArray from arrays of data, indices, and indptr, shape_out, and shape_in.
        """
        result = cls(shape_out, shape_in)
        result._matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=(np.prod(shape_out), np.prod(shape_in)))
        result.shape_out = shape_out
        result.shape_in = shape_in
        return result