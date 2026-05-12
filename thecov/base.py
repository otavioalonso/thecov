'''Module containing basic classes to deal with covariance matrices.'''

import copy
import logging
import os
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
import scipy
from scipy.sparse import SparseEfficiencyWarning

# Suppress sparse efficiency warnings (intentional element-wise access in SparseNDArray)
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)

from . import utils, math


__all__ = ['Covariance',
           'MultipoleCovariance',
           'Binning',
           'LinearBinning',
           'LogarithmicBinning',
           'FourierCovariance',
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

    def save_cache(self, cache_dir='./cache'):
        """Save cached function results to disk.
        
        Parameters
        ----------
        cache_dir : str
            Directory where cache files will be saved.
        """
        import pickle
        class_name = self.__class__.__name__
        for attr_name in dir(self):
            try:
                # Get the function from the class, not a bound method from the instance
                attr = getattr(self.__class__, attr_name, None)
                if attr is None:
                    continue
                # Get the underlying function if it's a method
                func = getattr(attr, '__func__', attr)
                if callable(func) and hasattr(func, 'cached') and func.cached:
                    cache_filename = os.path.join(cache_dir, f"{class_name}_{attr_name}.pkl")
                    utils.mkdir(os.path.dirname(cache_filename))
                    with open(cache_filename, 'wb') as f:
                        pickle.dump(func.cached, f)
            except Exception:
                # Skip attributes that can't be accessed or aren't suitable for caching
                continue

    def load_cache(self, cache_dir='./cache'):
        """Load cached function results from disk.
        
        Parameters
        ----------
        cache_dir : str
            Directory where cache files are stored.
        """
        import pickle
        class_name = self.__class__.__name__
        for attr_name in dir(self):
            try:
                # Get the function from the class, not a bound method from the instance
                # This ensures we modify the actual function's cached dict
                attr = getattr(self.__class__, attr_name, None)
                if attr is None:
                    continue
                # Get the underlying function if it's a method
                func = getattr(attr, '__func__', attr)
                if callable(func) and hasattr(func, 'cached'):
                    cache_filename = os.path.join(cache_dir, f"{class_name}_{attr_name}.pkl")
                    if os.path.isfile(cache_filename):
                        with open(cache_filename, 'rb') as f:
                            func.cached.update(pickle.load(f))
            except Exception:
                # Skip attributes that can't be accessed or raise errors
                continue

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    @property
    def with_mpi(self):
        """Whether to use MPI."""
        return getattr(self, 'mpicomm', None) is not None and self.mpicomm.size > 1

    def save(self, filename):
        """Save to ``filename``."""
        start = time.time()
        if not self.with_mpi or self.mpicomm.rank == 0:
            if hasattr(self, 'logger'):
                self.logger.info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            np.save(filename, self.__getstate__(), allow_pickle=True)

        if hasattr(self, 'logger'):
            self.logger.info(f'Saved to {filename} in {time.time() - start:.3f}s.')

    @classmethod
    def load(cls, filename):
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new

class Covariance(BaseClass):
    '''A class that represents a covariance matrix.
    
    Implements basic operations such as correlation matrix computation,
    symmetrization, regularization, and arithmetic operations.

    Attributes
    ----------
    cov : numpy.ndarray
        The covariance matrix.
    cor : numpy.ndarray
        The correlation matrix (read-only property).
    '''

    def __init__(self, covariance=None):
        '''Initializes a Covariance object.

        Parameters
        ----------
        covariance : numpy.ndarray, optional
            (n, n) numpy array with elements corresponding to the covariance.
        '''
        super().__init__()
        self._cov = covariance

    def __getstate__(self):
        '''Get state for pickling.'''
        return {'_cov': self._cov}

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
        self.cov = (self.cov + self.cov.T) / 2

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
        '''Regularize the covariance matrix by handling negative eigenvalues.

        Parameters
        ----------
        mode : str, optional
            Method for handling negative eigenvalues:
            - 'zero': Set negative eigenvalues to zero (default)
            - 'flip': Take absolute value of eigenvalues
            - 'minpos': Set negative eigenvalues to the minimum positive eigenvalue
        
        Raises
        ------
        ValueError
            If mode is not one of 'zero', 'flip', or 'minpos'.
        '''
        valid_modes = ('zero', 'flip', 'minpos')
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
        
        eigvals, eigvecs = self.eig
        if mode == 'zero':
            eigvals[eigvals < 0] = 0
        elif mode == 'flip':
            eigvals = np.abs(eigvals)
        elif mode == 'minpos':
            eigvals[eigvals < 0] = min(eigvals[eigvals > 0])
        self.cov = np.einsum('ij,jk,kl->il', eigvecs, np.diag(eigvals), eigvecs.T)
    
    def regularized(self, mode='zero'):
        '''Returns a regularized copy of the covariance matrix.

        Parameters
        ----------
        mode : str, optional
            Method for handling negative eigenvalues (see regularize()).

        Returns
        -------
        Covariance
            Covariance object corresponding to the regularized covariance matrix.
        '''
        new_cov = self.copy()
        new_cov.regularize(mode=mode)
        return new_cov

    def __add__(self, y):
        '''Add a covariance matrix or array.'''
        return Covariance(self.cov + (y.cov if isinstance(y, Covariance) else y))

    def __radd__(self, y):
        '''Right-add for commutative addition.'''
        return self.__add__(y)

    def __sub__(self, y):
        '''Subtract a covariance matrix or array.'''
        return self.__add__(-y)

    def __mul__(self, y):
        '''Multiply the covariance by a scalar.'''
        return Covariance(self.cov * y)

    def __rmul__(self, y):
        '''Right-multiply for commutative scalar multiplication.'''
        return self.__mul__(y)

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


class MultipoleCovariance(Covariance):
    '''A class to represent a covariance matrix for a set of multipoles.

    The underlying data structure is a single numpy array representing the full covariance matrix.
    This class provides view-based access to specific (l1, l2) multipole blocks.

    Attributes
    ----------
    cov : numpy.ndarray
        The full covariance matrix.
    cor : numpy.ndarray
        The correlation matrix.
    ells : tuple
        A tuple of two lists containing the multipoles (ells1, ells2).
    block_shape : tuple
        The shape (n, m) of each multipole sub-block.
    '''

    def __init__(self, ells=(0, 2, 4), block_shape=None):
        '''Initializes a MultipoleCovariance object.

        Parameters
        ----------
        ells : tuple or list, optional
            A tuple of two lists (ells1, ells2) specifying the multipoles.
            If a single list is provided, it will be used for both dimensions.
            Defaults to (0, 2, 4).
        block_shape : tuple, optional
            The shape (n, m) of each (l1, l2) sub-covariance block.
            If None, must be set later before accessing blocks.
        '''
        # Note: We don't call super().__init__() because we manage _cov directly
        self._ells1 = []
        self._ells2 = []
        self._block_shape = block_shape
        self._cov = None

        if ells is not None:
            if isinstance(ells, (tuple, list, np.ndarray)) and not isinstance(ells[0], (list, np.ndarray, tuple)):
                # Single list provided, use for both dimensions
                ells = (list(ells), list(ells))
            self._ells1 = sorted(list(ells[0]))
            self._ells2 = sorted(list(ells[1]))
            if block_shape is not None:
                self._initialize_cov()

    def _initialize_cov(self):
        '''Initialize the full covariance array based on ells and block_shape.
        
        Raises
        ------
        ValueError
            If block_shape is not set.
        '''
        if self._block_shape is None:
            raise ValueError("block_shape must be set before initializing covariance.")
        n1, n2 = self._block_shape
        total_rows = len(self._ells1) * n1
        total_cols = len(self._ells2) * n2
        self._cov = np.zeros((total_rows, total_cols))

    def _get_block_indices(self, l1, l2):
        '''Get the slice indices for the (l1, l2) block.

        Parameters
        ----------
        l1 : int
            First multipole.
        l2 : int
            Second multipole.

        Returns
        -------
        tuple
            (row_slice, col_slice) for accessing the block in the full covariance.
        
        Raises
        ------
        KeyError
            If l1 or l2 is not in the ells lists.
        '''
        if l1 not in self._ells1:
            raise KeyError(f"Multipole l1={l1} not in ells1={self._ells1}")
        if l2 not in self._ells2:
            raise KeyError(f"Multipole l2={l2} not in ells2={self._ells2}")

        i1 = self._ells1.index(l1)
        i2 = self._ells2.index(l2)
        n1, n2 = self._block_shape

        row_start, row_end = i1 * n1, (i1 + 1) * n1
        col_start, col_end = i2 * n2, (i2 + 1) * n2

        return slice(row_start, row_end), slice(col_start, col_end)

    def set_ell_cov(self, l1, l2, cov, cls=None):
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
            Unused, kept for backward compatibility.
        '''
        # Extract the numpy array from Covariance objects
        if isinstance(cov, Covariance):
            cov_array = cov.cov
        else:
            cov_array = cov

        if cov_array is None:
            return

        # Add ells if not present
        if l1 not in self._ells1:
            self._ells1.append(l1)
            self._ells1.sort()
        if l2 not in self._ells2:
            self._ells2.append(l2)
            self._ells2.sort()

        # Set block shape if not yet set
        if self._block_shape is None:
            self._block_shape = cov_array.shape

        # Initialize covariance array if needed
        if self._cov is None:
            self._initialize_cov()

        # Set the block
        row_slice, col_slice = self._get_block_indices(l1, l2)
        self._cov[row_slice, col_slice] = cov_array

    def get_ell_cov(self, l1, l2, cls=Covariance):
        '''Returns the covariance matrix for a given pair of multipoles.

        Parameters
        ----------
        l1
            the first multipole.
        l2
            the second multipole.
        cls : class, optional
            The class to wrap the result in. Defaults to Covariance.

        Returns
        ------- 
        Covariance
            A Covariance object corresponding to the covariance matrix for the given multipoles.
        '''

        if not self.has_ells(l1, l2):
            return None

        if self._cov is None:
            return None

        row_slice, col_slice = self._get_block_indices(l1, l2)
        block = self._cov[row_slice, col_slice]

        if cls is None:
            return block
        return cls(block)

    def is_ell_set(self, l1, l2):
        '''Check if a given (l1, l2) block is set (non-zero).'''
        return self.has_ells(l1, l2)

    @property
    def ells(self):
        '''Returns sorted lists of unique first and second multipoles used in the covariance matrices.

        Returns
        -------
        tuple of two lists
        '''
        return self._ells1.copy(), self._ells2.copy()

    def has_ells(self, l1, l2):
        '''Check if the given multipoles are in the covariance structure.
        
        Parameters
        ----------
        l1 : int
            First multipole.
        l2 : int
            Second multipole.
            
        Returns
        -------
        bool
            True if both l1 and l2 are in the ells lists.
        '''
        return l1 in self._ells1 and l2 in self._ells2

    @property
    def block_shape(self):
        '''The shape of each (l1, l2) sub-block.'''
        return self._block_shape

    @block_shape.setter
    def block_shape(self, shape):
        '''Set the block shape and initialize/resize the covariance array.'''
        self._block_shape = shape
        if self._ells1 and self._ells2:
            self._initialize_cov()

    @ells.setter
    def ells(self, ells):
        '''Initializes the ells structure.

        Parameters
        ----------
        ells : tuple
            A tuple of two lists: (l1s, l2s), where l1s and l2s are lists of multipoles.
        '''
        if isinstance(ells, (list, np.ndarray)) and not isinstance(ells[0], (list, np.ndarray, tuple)):
            ells = (list(ells), list(ells))
        self._ells1 = sorted(list(ells[0]))
        self._ells2 = sorted(list(ells[1]))
        if self._block_shape is not None:
            self._initialize_cov()

    @property
    def cov(self):
        '''Returns the full covariance matrix.

        Returns
        -------
        numpy.ndarray
            An (n,n) numpy array corresponding to the elements of the covariance matrix.
        '''
        return self._cov

    @cov.setter
    def cov(self, cov):
        '''Sets the full covariance matrix.

        Parameters
        ----------
        cov : numpy.ndarray
            An (n, n) numpy array corresponding to the elements of the covariance matrix.
        
        Raises
        ------
        ValueError
            If cov is not a 2D array, or if ells are not set, or if dimensions
            don't match the multipole structure.
        '''
        if cov is None:
            self._cov = None
            return

        if cov.ndim != 2:
            raise ValueError(f"Covariance should be a 2D matrix, got ndim={cov.ndim}.")

        if not self._ells1 or not self._ells2:
            raise ValueError("ells must be set before setting the full covariance matrix.")

        ells1, ells2 = self.ells

        if cov.shape[0] % len(ells1) != 0:
            raise ValueError(
                f"Can't resolve covariance structure: shape[0]={cov.shape[0]} "
                f"is not a multiple of the number of ells1={len(ells1)}."
            )
        if cov.shape[1] % len(ells2) != 0:
            raise ValueError(
                f"Can't resolve covariance structure: shape[1]={cov.shape[1]} "
                f"is not a multiple of the number of ells2={len(ells2)}."
            )

        size1 = cov.shape[0] // len(ells1)
        size2 = cov.shape[1] // len(ells2)

        self._block_shape = (size1, size2)
        self._cov = cov.copy()

    def __add__(self, y):
        '''Add two MultipoleCovariance objects.
        
        Parameters
        ----------
        y : MultipoleCovariance
            The covariance to add.
            
        Returns
        -------
        MultipoleCovariance
            The sum of the two covariances.
        
        Raises
        ------
        TypeError
            If y is not a MultipoleCovariance.
        ValueError
            If the multipoles don't match.
        '''
        if not isinstance(y, MultipoleCovariance):
            raise TypeError(f"Can only add MultipoleCovariance objects, got {type(y).__name__}")
        if self.ells != y.ells:
            raise ValueError(f"Multipoles must match for addition: {self.ells} != {y.ells}")

        result = MultipoleCovariance(
            ells=self.ells,
            block_shape=self._block_shape
        )
        result._cov = self._cov + y._cov
        return result

    def __sub__(self, y):
        '''Subtract two MultipoleCovariance objects.'''
        return self.__add__(-y)

    def __mul__(self, y):
        '''Multiply the covariance by a scalar.'''
        result = self.deepcopy()
        result._cov = result._cov * y
        return result

    def __rmul__(self, y):
        '''Right multiply the covariance by a scalar.'''
        return self.__mul__(y)

    def __truediv__(self, y):
        '''Divide the covariance by a scalar.'''
        return self * (1/y)

    def deepcopy(self):
        '''Create a deep copy of the MultipoleCovariance object.'''
        new = MultipoleCovariance(
            ells=(self._ells1.copy(), self._ells2.copy()),
            block_shape=self._block_shape
        )
        if self._cov is not None:
            new._cov = self._cov.copy()
        return new

    def foreach(self, func):
        '''Applies a function to each covariance block.

        Parameters
        ----------
        func : function
            The function to be applied to each covariance block.
            Should accept a Covariance object and return a Covariance or array.
        '''
        ells1, ells2 = self.ells
        for l1 in ells1:
            for l2 in ells2:
                cov_block = self.get_ell_cov(l1, l2)
                result = func(cov_block)
                if isinstance(result, Covariance):
                    result = result.cov
                row_slice, col_slice = self._get_block_indices(l1, l2)
                self._cov[row_slice, col_slice] = result

        return self

    def symmetrize(self):
        '''Symmetrize the diagonal blocks in place.'''
        ells1, ells2 = self.ells
        for l in set(ells1) & set(ells2):
            row_slice, col_slice = self._get_block_indices(l, l)
            block = self._cov[row_slice, col_slice]
            self._cov[row_slice, col_slice] = (block + block.T) / 2

    def __getstate__(self):
        '''Get state for pickling.'''
        return {
            '_ells1': self._ells1,
            '_ells2': self._ells2,
            '_block_shape': self._block_shape,
            '_cov': self._cov,
        }

    def __setstate__(self, state):
        '''Set state for unpickling.'''
        self.__dict__.update(state)


class Binning(ABC):
    '''Abstract base class for binning schemes.
    
    Subclasses must implement the `bins`, `edges`, and `midpoints` properties.
    '''
    
    @property
    @abstractmethod
    def bins(self):
        '''Returns the total number of bins.'''
        pass
    
    @property
    @abstractmethod
    def edges(self):
        '''Returns the bin edges.'''
        pass
    
    @property
    @abstractmethod
    def midpoints(self):
        '''Returns the bin midpoints.'''
        pass
    
    @property
    def is_set(self):
        '''Check if binning has been configured.'''
        return False


class LinearBinning(Binning, BaseClass):
    '''A class to represent an observable linearly binned in wavenumber k.

    Attributes
    ----------
    kmin : float
        The minimum value of the wavenumber k.
    kmax : float
        The maximum value of the wavenumber k.
    dk : float
        The spacing between k-bins.
    volume : float, optional
        The volume of the survey/box (used for nmodes calculation).
    '''

    def __init__(self, kmin=None, kmax=None, dk=None, volume=None):
        '''Initialize a LinearBinning object.
        
        Parameters
        ----------
        kmin : float, optional
            The minimum value of the wavenumber k.
        kmax : float, optional
            The maximum value of the wavenumber k.
        dk : float, optional
            The spacing between k-bins.
        volume : float, optional
            The volume of the survey/box.
        '''
        super().__init__()
        self.kmin = kmin
        self.kmax = kmax
        self.dk = dk
        self.volume = volume

    def __getstate__(self):
        '''Get state for pickling.'''
        return {
            'kmin': self.kmin,
            'kmax': self.kmax,
            'dk': self.dk,
            'volume': self.volume,
            '_nmodes': getattr(self, '_nmodes', None),
        }

    def set_kbins(self, kmin, kmax, dk):
        '''Define the k-bins.

        Parameters
        ----------
        kmin : float
            The minimum value of the wavenumber k.
        kmax : float
            The maximum value of the wavenumber k.
        dk : float
            The spacing between k-bins.
        '''
        self.kmin = kmin
        self.kmax = kmax
        self.dk = dk

    @property
    def is_set(self):
        '''Check if k-bins were defined.

        Returns
        -------
        bool
            True if k-bins were defined, False otherwise.
        '''
        return None not in (self.dk, self.kmin, self.kmax)

    # Alias for backward compatibility
    @property
    def is_kbins_set(self):
        '''Alias for is_set for backward compatibility.'''
        return self.is_set

    @property
    def bins(self):
        '''Returns the total number of k-bins.

        Returns
        -------
        int
            The total number of k-bins.
        '''
        return len(self.midpoints)

    # Alias for backward compatibility
    @property
    def kbins(self):
        '''Alias for bins for backward compatibility.'''
        return self.bins

    @property
    def midpoints(self):
        '''Returns the midpoints of the k-bins.

        Returns
        -------
        numpy.ndarray
            The midpoints of the k-bins.
        '''
        return np.arange(self.kmin + self.dk / 2, self.kmax + self.dk / 2, self.dk)

    @property
    def kmid(self):
        '''Alias for midpoints.'''
        return self.midpoints

    @property
    def kavg(self):
        '''Returns the average k of the k-bins.
        
        Assumes spherical approximation to integrate k-modes, 
        which fails for small k.

        Returns
        -------
        numpy.ndarray
            The average k of the k-bins.
        '''
        edges = self.edges
        return 3/4 * (edges[1:]**4 - edges[:-1]**4) / (edges[1:]**3 - edges[:-1]**3)

    @property
    def edges(self):
        '''Returns the edges of the k-bins.

        Returns
        -------
        numpy.ndarray
            The edges of the k-bins.
        '''
        return np.arange(self.kmin, self.kmax + self.dk / 2, self.dk)

    # Alias for backward compatibility
    @property
    def kedges(self):
        '''Alias for edges for backward compatibility.'''
        return self.edges

    @property
    def kfun(self):
        '''Fundamental wavenumber of the box 2*pi/Lbox.

        Returns
        -------
        float
            The fundamental wavenumber of the box.
        
        Raises
        ------
        ValueError
            If volume is not set.
        '''
        if self.volume is None:
            raise ValueError("volume must be set to compute kfun")
        return 2 * np.pi / self.volume**(1/3)

    @property
    def nmodes(self):
        '''Calculate the number of modes per k-bin shell.
        
        If nmodes was not manually set, it is estimated from the volume of each shell.

        Returns
        -------
        numpy.ndarray
            The number of modes per k-bin shell.
        
        Raises
        ------
        ValueError
            If volume is not set and nmodes was not manually set.
        '''
        if hasattr(self, '_nmodes') and self._nmodes is not None:
            return self._nmodes
        
        if self.volume is None:
            raise ValueError("volume must be set to compute nmodes")

        return math.nmodes(self.volume, self.edges[:-1], self.edges[1:])

    @nmodes.setter
    def nmodes(self, nmodes):
        '''Manually set the number of modes per k-bin shell.

        Parameters
        ----------
        nmodes : numpy.ndarray
            The number of modes per k-bin shell.
        '''
        self._nmodes = nmodes


class LogarithmicBinning(Binning, BaseClass):
    '''A class to represent an observable logarithmically binned in wavenumber k.

    Attributes
    ----------
    kmin : float
        The minimum value of the wavenumber k.
    kmax : float
        The maximum value of the wavenumber k.
    nbins : int
        The number of bins.
    volume : float, optional
        The volume of the survey/box (used for nmodes calculation).
    '''

    def __init__(self, kmin=None, kmax=None, nbins=None, volume=None):
        '''Initialize a LogarithmicBinning object.
        
        Parameters
        ----------
        kmin : float, optional
            The minimum value of the wavenumber k (must be > 0).
        kmax : float, optional
            The maximum value of the wavenumber k.
        nbins : int, optional
            The number of logarithmically spaced bins.
        volume : float, optional
            The volume of the survey/box.
        '''
        super().__init__()
        self.kmin = kmin
        self.kmax = kmax
        self.nbins = nbins
        self.volume = volume

    def __getstate__(self):
        '''Get state for pickling.'''
        return {
            'kmin': self.kmin,
            'kmax': self.kmax,
            'nbins': self.nbins,
            'volume': self.volume,
            '_nmodes': getattr(self, '_nmodes', None),
        }

    @property
    def is_set(self):
        '''Check if k-bins were defined.

        Returns
        -------
        bool
            True if k-bins were defined, False otherwise.
        '''
        return None not in (self.kmin, self.kmax, self.nbins)

    @property
    def dlogk(self):
        '''The logarithmic bin width (spacing in log10(k)).

        Returns
        -------
        float
            The spacing between bins in log10(k).
        '''
        if not self.is_set:
            return None
        return (np.log10(self.kmax) - np.log10(self.kmin)) / self.nbins

    @property
    def bins(self):
        '''Returns the total number of k-bins.

        Returns
        -------
        int
            The total number of k-bins.
        '''
        return self.nbins

    # Alias for backward compatibility
    @property
    def kbins(self):
        '''Alias for bins for backward compatibility.'''
        return self.bins

    @property
    def edges(self):
        '''Returns the edges of the k-bins (logarithmically spaced).

        Returns
        -------
        numpy.ndarray
            The edges of the k-bins.
        '''
        if not self.is_set:
            return None
        return np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.nbins + 1)

    # Alias for backward compatibility
    @property
    def kedges(self):
        '''Alias for edges for backward compatibility.'''
        return self.edges

    @property
    def midpoints(self):
        '''Returns the midpoints of the k-bins (geometric mean of edges).

        The geometric mean is used because bins are logarithmically spaced,
        so the geometric mean gives the center in log-space.

        Returns
        -------
        numpy.ndarray
            The midpoints of the k-bins.
        '''
        edges = self.edges
        if edges is None:
            return None
        return np.sqrt(edges[:-1] * edges[1:])

    @property
    def kmid(self):
        '''Alias for midpoints.'''
        return self.midpoints

    @property
    def kavg(self):
        '''Returns the average k of the k-bins.
        
        Assumes spherical approximation to integrate k-modes.

        Returns
        -------
        numpy.ndarray
            The average k of the k-bins.
        '''
        edges = self.edges
        if edges is None:
            return None
        return 3/4 * (edges[1:]**4 - edges[:-1]**4) / (edges[1:]**3 - edges[:-1]**3)

    @property
    def dk(self):
        '''Returns the width of each k-bin (varies across bins).

        Returns
        -------
        numpy.ndarray
            The width of each k-bin.
        '''
        edges = self.edges
        if edges is None:
            return None
        return edges[1:] - edges[:-1]

    @property
    def kfun(self):
        '''Fundamental wavenumber of the box 2*pi/Lbox.

        Returns
        -------
        float
            The fundamental wavenumber of the box.
        
        Raises
        ------
        ValueError
            If volume is not set.
        '''
        if self.volume is None:
            raise ValueError("volume must be set to compute kfun")
        return 2 * np.pi / self.volume**(1/3)

    @property
    def nmodes(self):
        '''Calculate the number of modes per k-bin shell.
        
        If nmodes was not manually set, it is estimated from the volume of each shell.

        Returns
        -------
        numpy.ndarray
            The number of modes per k-bin shell.
        
        Raises
        ------
        ValueError
            If volume is not set and nmodes was not manually set.
        '''
        if hasattr(self, '_nmodes') and self._nmodes is not None:
            return self._nmodes
        
        if self.volume is None:
            raise ValueError("volume must be set to compute nmodes")

        return math.nmodes(self.volume, self.edges[:-1], self.edges[1:])

    @nmodes.setter
    def nmodes(self, nmodes):
        '''Manually set the number of modes per k-bin shell.

        Parameters
        ----------
        nmodes : numpy.ndarray
            The number of modes per k-bin shell.
        '''
        self._nmodes = nmodes


class FourierCovariance(Covariance):
    '''A covariance matrix in Fourier space with k-binning information.
    
    Attributes
    ----------
    kbin1 : LinearBinning
        The k-binning for the first dimension.
    kbin2 : LinearBinning
        The k-binning for the second dimension.
    '''

    def __init__(self, covariance=None, kbin1=None, kbin2=None):
        '''Initialize a FourierCovariance object.
        
        Parameters
        ----------
        covariance : numpy.ndarray, optional
            The covariance matrix.
        kbin1 : LinearBinning, optional
            The k-binning for the first dimension.
        kbin2 : LinearBinning, optional
            The k-binning for the second dimension. If None, uses kbin1.
        '''
        super().__init__(covariance=covariance)
        if kbin2 is None:
            kbin2 = kbin1
        self.kbin1 = kbin1
        self.kbin2 = kbin2

    def __getstate__(self):
        '''Get state for pickling.'''
        state = super().__getstate__()
        state.update({
            'kbin1': self.kbin1,
            'kbin2': self.kbin2,
        })
        return state

    def kcut(self, kmin=None, kmax=None):
        '''Apply a k-cut to the covariance matrix.
        
        Parameters
        ----------
        kmin : float, optional
            Minimum k value to keep. Defaults to the larger of kbin1.kmin and kbin2.kmin.
        kmax : float, optional
            Maximum k value to keep. Defaults to the smaller of kbin1.kmax and kbin2.kmax.
            
        Returns
        -------
        FourierCovariance
            Self, for method chaining.
        '''
        if kmin is None:
            kmin = max(self.kbin1.kmin, self.kbin2.kmin)

        if kmax is None:
            kmax = min(self.kbin1.kmax, self.kbin2.kmax)

        imin1 = (self.kbin1.kmid >= kmin).argmax()
        imin2 = (self.kbin2.kmid >= kmin).argmax()

        imax1 = len(self.kbin1.kmid) if (self.kbin1.kmid <= kmax).all() else (self.kbin1.kmid <= kmax).argmin()
        imax2 = len(self.kbin2.kmid) if (self.kbin2.kmid <= kmax).all() else (self.kbin2.kmid <= kmax).argmin()

        self.cov = self.cov[imin1:imax1, imin2:imax2]

        self.kbin1.kmin, self.kbin1.kmax = kmin, kmax
        self.kbin2.kmin, self.kbin2.kmax = kmin, kmax

        return self

    @property
    def kmid_matrices(self):
        '''Returns 2D matrices of k midpoints for each axis.
        
        Returns
        -------
        tuple
            (k1, k2) where k1[i,j] = kbin1.kmid[i] and k2[i,j] = kbin2.kmid[j].
        '''
        k1 = np.einsum('i,j->ij', self.kbin1.kmid, np.ones(self.kbin2.kbins))
        k2 = np.einsum('i,j->ji', self.kbin2.kmid, np.ones(self.kbin1.kbins))

        return k1, k2

    @property
    def kmin_matrices(self):
        '''Returns 2D matrices of k bin lower edges for each axis.
        
        Returns
        -------
        tuple
            (k1, k2) where k1[i,j] = kbin1.kedges[i] and k2[i,j] = kbin2.kedges[j].
        '''
        k1 = np.einsum('i,j->ij', self.kbin1.kedges[:-1], np.ones(self.kbin2.kbins))
        k2 = np.einsum('i,j->ji', self.kbin2.kedges[:-1], np.ones(self.kbin1.kbins))

        return k1, k2

class MultipoleFourierCovariance(MultipoleCovariance, FourierCovariance):
    '''A covariance matrix for multipole power spectra in Fourier space.
    
    Combines multipole structure from MultipoleCovariance with k-binning
    from FourierCovariance.
    
    Attributes
    ----------
    ells : tuple
        A tuple of two lists containing the multipoles (ells1, ells2).
    kbin1 : LinearBinning
        The k-binning for the first dimension.
    kbin2 : LinearBinning
        The k-binning for the second dimension.
    '''

    def __init__(self, ells=(0, 2, 4)):
        '''Initialize a MultipoleFourierCovariance object.
        
        Parameters
        ----------
        ells : tuple or list, optional
            The multipoles to include. Defaults to (0, 2, 4).
        '''
        MultipoleCovariance.__init__(self, ells=ells)
        FourierCovariance.__init__(self)
        self.logger = logging.getLogger('MultipoleFourierCovariance')

    def __getstate__(self):
        '''Get state for pickling.'''
        return {
            '_ells1': self._ells1,
            '_ells2': self._ells2,
            '_block_shape': self._block_shape,
            '_cov': self._cov,
            'kbin1': self.kbin1,
            'kbin2': self.kbin2,
        }

    @property
    def kmid_ell_matrices(self):
        '''Returns 2D matrices of k midpoints repeated for each multipole.
        
        Returns
        -------
        tuple
            (k1, k2) matrices with k values for the full covariance structure.
        '''
        ells1, ells2 = self.ells

        kfull1 = np.concatenate([self.kbin1.kmid for _ in ells1])
        kfull2 = np.concatenate([self.kbin2.kmid for _ in ells2])

        k1 = np.einsum('i,j->ij', kfull1, np.ones_like(kfull2))
        k2 = np.einsum('i,j->ji', kfull2, np.ones_like(kfull1))

        return k1, k2

    @property
    def ell_matrices(self):
        '''Returns 2D matrices of multipole values for the full covariance.
        
        Returns
        -------
        tuple
            (ell1, ell2) matrices with multipole values.
        '''
        ells1, ells2 = self.ells

        kells1 = np.einsum('i,j->ij', ells1, np.ones(self.kbin1.kbins)).flatten()
        kells2 = np.einsum('i,j->ji', ells2, np.ones(self.kbin2.kbins)).flatten()

        ell1 = np.einsum('i,j->ij', kells1, np.ones_like(kells2))
        ell2 = np.einsum('i,j->ji', kells2, np.ones_like(kells1))

        return ell1, ell2

    def savetxt(self, filename, fmt='matrix'):
        '''Save the covariance to a text file.
        
        Parameters
        ----------
        filename : str
            The output filename.
        
        fmt : str
            The format for saving the file. Can be 'matrix' or 'list'.
        '''

        if fmt == 'matrix':
            utils.mkdir(os.path.dirname(filename))
            np.savetxt(filename, self.cov, fmt='%.8e')
            return

        if fmt == 'list':
                
            k1, k2 = self.kmid_ell_matrices
            ell1, ell2 = self.ell_matrices

            cov = self.cov

            mask = np.ones_like(ell1, dtype=bool)
            utils.mkdir(os.path.dirname(filename))
            np.savetxt(filename, np.concatenate([ell1[mask].reshape(-1, 1),
                                                ell2[mask].reshape(-1, 1),
                                                k1[mask].reshape(-1, 1),
                                                k2[mask].reshape(-1, 1),
                                                cov[mask].reshape(-1, 1)], axis=1), 
                    fmt=['%.d', '%.d', '%.4f', '%.4f', '%.8e'],
                    header='ell1 ell2 kmid1 kmid2 cov')
            return
        

    
    def set_ell_cov(self, l1, l2, cov, cls=None):
        '''Sets the covariance matrix for a given pair of multipoles.

        Parameters
        ----------
        l1 : int
            The first multipole.
        l2 : int
            The second multipole.
        cov : FourierCovariance, Covariance, or numpy.ndarray
            The covariance matrix.
        cls : class, optional
            Unused, kept for backward compatibility.
        '''
        super().set_ell_cov(l1, l2, cov, cls=cls)
    
    def get_ell_cov(self, l1, l2, cls=FourierCovariance):
        '''Returns the covariance matrix for a given pair of multipoles as a FourierCovariance.

        Parameters
        ----------
        l1 : int
            The first multipole.
        l2 : int
            The second multipole.
        cls : class, optional
            The class to wrap the result in. Defaults to FourierCovariance.

        Returns
        -------
        FourierCovariance or Covariance
            A covariance object for the given multipoles.
        '''
        block = super().get_ell_cov(l1, l2, cls=None)
        if block is None:
            return None
        if cls is FourierCovariance:
            fc = FourierCovariance(kbin1=self.kbin1, kbin2=self.kbin2)
            fc._cov = block
            return fc
        elif cls is None:
            return block
        return cls(block)

    def kcut(self, kmin=None, kmax=None):
        '''Apply a k-cut to the covariance matrix.

        Parameters
        ----------
        kmin : float, optional
            Minimum k value to keep.
        kmax : float, optional
            Maximum k value to keep.

        Returns
        -------
        MultipoleFourierCovariance
            Self, for method chaining.
        '''
        if kmin is None:
            kmin = self.kbin1.kmin
        if kmax is None:
            kmax = self.kbin1.kmax

        # Get indices for the k-cut
        imin = (self.kbin1.kmid >= kmin).argmax()
        imax = len(self.kbin1.kmid) if (self.kbin1.kmid <= kmax).all() else (self.kbin1.kmid <= kmax).argmin()

        # Update block shape
        new_size = imax - imin
        old_size = self._block_shape[0] if self._block_shape else self.kbin1.kbins

        # Create new covariance with cut data
        ells1, ells2 = self.ells
        new_cov = np.zeros((len(ells1) * new_size, len(ells2) * new_size))

        for i1, l1 in enumerate(ells1):
            for i2, l2 in enumerate(ells2):
                old_row = slice(i1 * old_size + imin, i1 * old_size + imax)
                old_col = slice(i2 * old_size + imin, i2 * old_size + imax)
                new_row = slice(i1 * new_size, (i1 + 1) * new_size)
                new_col = slice(i2 * new_size, (i2 + 1) * new_size)
                new_cov[new_row, new_col] = self._cov[old_row, old_col]

        self._cov = new_cov
        self._block_shape = (new_size, new_size)

        # Update k-bins
        self.kbin1.kmin, self.kbin1.kmax = kmin, kmax
        self.kbin2.kmin, self.kbin2.kmax = kmin, kmax

        self.logger.info(f'kcut to {kmin} < k < {kmax}')

        return self
    
    def set_linear_kbins(self, kmin, kmax, dk):
        '''Set the k-binning for this covariance.

        Parameters
        ----------
        kmin : float
            Minimum k value.
        kmax : float
            Maximum k value.
        dk : float
            k-bin width.
        '''
        size = (kmax - kmin)/dk
        size = (np.round(size) if np.allclose(np.round(size), size) else size).astype(int)
        self._block_shape = (size, size)
        self.kbin1 = LinearBinning(kmin, kmax, dk)
        self.kbin2 = LinearBinning(kmin, kmax, dk)
        
        # Initialize covariance array if ells are already set
        if self._ells1 and self._ells2:
            self._initialize_cov()

    # Alias
    set_kbins = set_linear_kbins

    # ---- Proxy properties delegating to kbin1 ----

    @property
    def kbins(self):
        '''Number of k-bins (proxy for kbin1.kbins).'''
        return self.kbin1.kbins

    @property
    def kmin(self):
        '''Minimum k value (proxy for kbin1.kmin).'''
        return self.kbin1.kmin

    @property
    def kmax(self):
        '''Maximum k value (proxy for kbin1.kmax).'''
        return self.kbin1.kmax

    @property
    def dk(self):
        '''k-bin width (proxy for kbin1.dk).'''
        return self.kbin1.dk

    @property
    def kmid(self):
        '''k-bin midpoints (proxy for kbin1.kmid).'''
        return self.kbin1.kmid

    @property
    def kedges(self):
        '''k-bin edges (proxy for kbin1.kedges).'''
        return self.kbin1.kedges

    @property
    def nmodes(self):
        '''Number of modes per k-bin (proxy for kbin1.nmodes).'''
        return self.kbin1.nmodes

    @property
    def is_kbins_set(self):
        '''Whether k-bins have been configured (proxy for kbin1.is_set).'''
        return self.kbin1 is not None and self.kbin1.is_set

class SparseNDArray:
    """
    A class to represent a sparse ND array using scipy.sparse.csr_matrix.
    Indices are split between shape_out and shape_in, as if the array is
    a 2D matrix (shape_out x shape_in). Matrix multiplication is done using
    the @ operator and requires the shapes to be compatible, i.e., shape_in
    of the leftmost array must match shape_out of the rightmost array.
    """
    def __init__(self, shape_out, shape_in, dtype=float):
        self.shape_in = np.asarray(shape_in).astype(int)
        self.shape_out = np.asarray(shape_out).astype(int)
        self._matrix = scipy.sparse.csr_matrix((np.prod(shape_out), np.prod(shape_in)), dtype=dtype)

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
            else:
                self._matrix[self._nd_to_2d_indices(*indices)] = value.flatten()
        else:
            raise ValueError(f"Invalid number of indices: {len(indices)}. Expected {len(self.shape_out) + len(self.shape_in)} or {len(self.shape_out)}.")

    def __getitem__(self, indices):
        indices = np.asarray(indices).astype(int)
        result = self._matrix[self._nd_to_2d_indices(*indices)]
        # If we indexed a single element (all indices provided), return a scalar
        if len(indices) == len(self.shape_out) + len(self.shape_in):
            # Extract scalar from sparse matrix (returns a matrix, need to get the value)
            if hasattr(result, 'toarray'):
                return result.toarray().item()
            return result
        elif len(indices) == len(self.shape_out):
            if hasattr(result, 'toarray'):
                return result.toarray().reshape(self.shape_in)
        return result

    def __repr__(self):
        return f"SparseNDArray(shape_out={self.shape_out} -> {np.prod(self.shape_out)}, shape_in={self.shape_in} -> {np.prod(self.shape_in)}, nnz={self._matrix.nnz}, sparsity={self._matrix.nnz / np.prod(self.shape_in) / np.prod(self.shape_out)})"

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
            if not ((self.shape_in == other.shape_in).all() and (self.shape_out == other.shape_out).all()):
                raise ValueError("Shapes do not match for addition.")
            
            result = copy.deepcopy(other)
            result._matrix += self._matrix
            return result
        else:
            raise TypeError(f"Operation not supported between {self.__class__.__name__} and {type(other).__name__}.")
        
    def __mul__(self, other):
        if isinstance(other, SparseNDArray):
            if not ((self.shape_in == other.shape_in).all() and (self.shape_out == other.shape_out).all()):
                raise ValueError("Shapes do not match for multiplication.")
            
            result = copy.deepcopy(other)
            result._matrix = result._matrix.multiply(self._matrix)
            return result
        else:
            raise TypeError(f"Operation not supported between {self.__class__.__name__} and {type(other).__name__}.")
        
    def __matmul__(self, other):
        if isinstance(other, SparseNDArray):
            if not (np.all(self.shape_in == other.shape_out)):
                raise ValueError("Shapes do not match for matrix multiplication.")
            result = copy.deepcopy(other)
            result._matrix = self._matrix.dot(result._matrix)
            result.shape_out = self.shape_out
            return result
        elif isinstance(other, np.ndarray):
            result = copy.deepcopy(self)
            result._matrix = scipy.sparse.csr_matrix(self._matrix.dot(other.reshape(np.prod(self.shape_in), -1)))
            result.shape_in = np.asarray(other.shape[len(self.shape_in):])
            return result
        else:
            raise TypeError(f"Operation not supported between {self.__class__.__name__} and {type(other).__name__}.")
        
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

    def nonzero_indices_out(self):
        """
        Yield shape_out indices (as tuples) that have at least one non-zero element.
        
        This is useful for iterating only over the populated rows of the sparse array.
        
        Yields
        ------
        tuple
            ND indices into shape_out that have non-zero entries.
        """
        nonzero_rows = np.unique(self._matrix.nonzero()[0])
        for row in nonzero_rows:
            yield tuple(np.unravel_index(row, self.shape_out))

    def get_nonzero_rows_dense(self):
        """
        Get all nonzero rows as dense arrays in a single operation.
        
        Much more efficient than iterating with __getitem__ when you need
        all nonzero rows, as it avoids repeated sparse-to-dense conversions.
        
        Returns
        -------
        indices : numpy.ndarray
            Shape (n_nonzero, len(shape_out)) array of ND indices into shape_out.
        values : numpy.ndarray
            Shape (n_nonzero, prod(shape_in)) array of row values.
            Each row corresponds to the flattened shape_in data.
        """
        # Get unique nonzero row indices
        nonzero_rows = np.unique(self._matrix.nonzero()[0])
        
        if len(nonzero_rows) == 0:
            return np.empty((0, len(self.shape_out)), dtype=int), \
                   np.empty((0, int(np.prod(self.shape_in))), dtype=self._matrix.dtype)
        
        # Convert flat row indices to ND indices
        indices = np.array(np.unravel_index(nonzero_rows, self.shape_out)).T
        
        # Extract all nonzero rows at once (much faster than repeated getrow)
        values = self._matrix[nonzero_rows].toarray()
        
        return indices, values

    def get_row_sparse(self, *indices):
        """
        Get a row as a sparse CSR matrix (1 x shape_in_flat), without densifying.
        
        Parameters
        ----------
        *indices : int
            Indices into shape_out dimensions.
            
        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse row vector of shape (1, prod(shape_in)).
        """
        if len(indices) != len(self.shape_out):
            raise ValueError(f"Expected {len(self.shape_out)} indices, got {len(indices)}")
        row_idx = self._nd_to_2d_indices(*indices)
        return self._matrix.getrow(row_idx)

    def __reduce__(self):
        """
        Custom pickling for MPI serialization.
        Returns the necessary data to reconstruct the SparseNDArray.
        """
        return (
            self.__class__._reconstruct,
            (self.shape_out, self.shape_in, self._matrix)
        )

    @classmethod
    def _reconstruct(cls, shape_out, shape_in, matrix):
        """Reconstruct a SparseNDArray from pickled components."""
        obj = cls.__new__(cls)
        obj.shape_out = shape_out
        obj.shape_in = shape_in
        obj._matrix = matrix
        return obj

    def allgather(self, mpicomm, axis=None):
        """
        Gather and concatenate SparseNDArray slabs from all ranks using allgather.
        Designed to be called as: result = SparseNDArray.allgather(mpicomm)

        Parameters
        ----------
        local_sparse : SparseNDArray
            Local slab on this rank
        comm : MPI.Comm
            MPI communicator
        axis : int
            Axis along which slabs are distributed (within shape_in)
            
        Returns
        -------
        SparseNDArray
            Complete SparseNDArray with all slabs concatenated
        """
        
        # Gather all objects from all ranks
        all_sparse = mpicomm.allgather(self)
        
        # Extract all CSR matrices and concatenate horizontally
        all_matrices = [s._matrix for s in all_sparse]
        combined_matrix = scipy.sparse.hstack(all_matrices, format='csr')

        # Calculate full shape_in by summing along concatenation axis
        shape_out = all_sparse[0].shape_out.copy()
        full_shape_in = all_sparse[0].shape_in.copy()

        if axis is None:
            # Pick the axis with length 1 if possible
            if (full_shape_in == 1).sum() == 1:
                axis = np.where(full_shape_in == 1)[0][0]
            else:
                # Default to axis 0
                axis = np.argmin(full_shape_in)

        full_shape_in[axis] = sum(s.shape_in[axis] for s in all_sparse)
        
        # Create new SparseNDArray
        result = self.__new__(self.__class__)
        result.shape_out = shape_out
        result.shape_in = full_shape_in
        result._matrix = combined_matrix
        
        return result

    def to_shared_memory(self):
        """
        Create shared memory arrays for the sparse matrix components.
        Returns metadata dict needed to reconstruct the sparse array.
        """
        from multiprocessing import shared_memory
        
        shm_handles = []
        
        # Share data array
        data_shm = shared_memory.SharedMemory(create=True, size=self._matrix.data.nbytes)
        data_arr = np.ndarray(self._matrix.data.shape, dtype=self._matrix.data.dtype, buffer=data_shm.buf)
        data_arr[:] = self._matrix.data[:]
        shm_handles.append(data_shm)
        
        # Share indices array
        indices_shm = shared_memory.SharedMemory(create=True, size=self._matrix.indices.nbytes)
        indices_arr = np.ndarray(self._matrix.indices.shape, dtype=self._matrix.indices.dtype, buffer=indices_shm.buf)
        indices_arr[:] = self._matrix.indices[:]
        shm_handles.append(indices_shm)
        
        # Share indptr array
        indptr_shm = shared_memory.SharedMemory(create=True, size=self._matrix.indptr.nbytes)
        indptr_arr = np.ndarray(self._matrix.indptr.shape, dtype=self._matrix.indptr.dtype, buffer=indptr_shm.buf)
        indptr_arr[:] = self._matrix.indptr[:]
        shm_handles.append(indptr_shm)
        
        metadata = {
            'data_shm_name': data_shm.name,
            'data_shape': self._matrix.data.shape,
            'data_dtype': self._matrix.data.dtype,
            'indices_shm_name': indices_shm.name,
            'indices_shape': self._matrix.indices.shape,
            'indices_dtype': self._matrix.indices.dtype,
            'indptr_shm_name': indptr_shm.name,
            'indptr_shape': self._matrix.indptr.shape,
            'indptr_dtype': self._matrix.indptr.dtype,
            'matrix_shape': self._matrix.shape,
            'shape_out': self.shape_out,
            'shape_in': self.shape_in,
        }
        
        return metadata, shm_handles

    @classmethod
    def from_shared_memory(cls, metadata):
        """
        Reconstruct a SparseNDArray from shared memory.
        Returns a read-only view into shared memory (no copying).
        """
        from multiprocessing import shared_memory
        
        # Attach to shared memory (no copy)
        data_shm = shared_memory.SharedMemory(name=metadata['data_shm_name'])
        data = np.ndarray(metadata['data_shape'], dtype=metadata['data_dtype'], buffer=data_shm.buf)
        
        indices_shm = shared_memory.SharedMemory(name=metadata['indices_shm_name'])
        indices = np.ndarray(metadata['indices_shape'], dtype=metadata['indices_dtype'], buffer=indices_shm.buf)
        
        indptr_shm = shared_memory.SharedMemory(name=metadata['indptr_shm_name'])
        indptr = np.ndarray(metadata['indptr_shape'], dtype=metadata['indptr_dtype'], buffer=indptr_shm.buf)
        
        # Create CSR matrix from shared memory arrays (no copy)
        matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=metadata['matrix_shape'], copy=False)
        
        # Create SparseNDArray
        obj = cls.__new__(cls)
        obj.shape_out = metadata['shape_out']
        obj.shape_in = metadata['shape_in']
        obj._matrix = matrix
        obj._shm_refs = [data_shm, indices_shm, indptr_shm]  # Keep references
        
        return obj

    def close_shared_memory(self):
        """Close shared memory references (call from worker processes)."""
        for shm in getattr(self, '_shm_refs', []):
            shm.close()

    @staticmethod
    def cleanup_shared_memory(shm_handles):
        """Cleanup shared memory (call from main process after workers complete)."""
        for shm in shm_handles:
            shm.close()
            shm.unlink()


def cache(func):
    """Cache decorator for instance methods.
    
    The cache key includes id(self) so each instance has its own cache,
    and also includes all arguments so repeated calls with the same args
    on the same instance are fast.
    """
    from functools import wraps
    import inspect
    func.cached = {}
    sig = inspect.signature(func)
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Bind arguments to get a consistent cache key
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        # Include id(self) so different instances don't share cached results.
        # Without this, a second instance with the same args gets a cached
        # return value but self.window_matrix (set as a side-effect) is never
        # assigned, leaving it as None.
        args_key = tuple(bound.arguments.items())[1:]  # Skip 'self'
        cache_key = (id(self), args_key)
        try:
            return wrapper.cached[cache_key]
        except KeyError:
            wrapper.cached[cache_key] = result = func(self, *args, **kwargs)
            return result
    wrapper.cached = func.cached
    return wrapper