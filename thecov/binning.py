import numpy as np

class FourierBinning:
    """A parent class to represent an observable binned in wavenumber k."""

    def __init__(self):
        self.kmin = None
        self.kmax = None
        self.dk = None
        self._nmodes = None

    def set_kbins(self, kmin:float, kmax:float, dk:float, nmodes=None):
        '''This function defines the k-bins.

        Parameters
        ----------
        kmin: float
            The minimum value of the wavenumber k.
        kmax: float
            The maximum value of the wavenumber k.
        dk: float
            The (log) spacing between k-bins.
        nmodes: numpy.ndarray, optional
            The number of modes to be used in the calculation. It is an optional parameter.
            If omitted, it is calculated from the volume of spherical shells.
        '''

        self.dk = dk
        self.kmax = kmax
        self.kmin = kmin
        self._nmodes = nmodes

    @property
    def kbins(self):
        '''Returns the total number of k-bins.

        Returns
        -------
        int
            The total number of k-bins.
        '''

        return len(self.kmid)
    
    @property
    def is_kbins_set(self):
        '''Check if k-bins were defined.

        Returns
        -------
            bool, True if k-bins were defined, False otherwise.
        '''
        return None not in (self.dk, self.kmin, self.kmax)
    
    @property
    def kavg(self):
        '''
        Returns the average k of the k-bins. Assumes spherical approximation to
        integrate k-modes, which fails for small k.

        Returns
        -------
        numpy.ndarray
            The average k of the k-bins.
        '''
        return 3/4*(self.kedges[1:]**4 - self.kedges[:-1]**4)/ \
                   (self.kedges[1:]**3 - self.kedges[:-1]**3)
    
    @property
    def kfun(self):
        '''Fundamental wavenumber of the box 2*pi/Lbox.

        Returns
        -------
        float
            The fundamental wavenumber of the box.
        '''

        return 2*np.pi/self.volume**(1/3)

    @property
    def kedges(self):
        raise NotImplementedError

    @property
    def nmodes(self):
        raise NotImplementedError

    @nmodes.setter
    def nmodes(self, nmodes):
        '''Manually sets the number of modes per k-bin shell.

        Parameters
        -------
        nmodes : numpy.ndarray
            The number of modes per k-bin shell.
        '''

        self._nmodes = nmodes

class LinearBinning(FourierBinning):
    '''A class to represent an observable linearly binned in wavenumber k.

    Attributes
    ----------
    kmin: float
        The minimum value of the wavenumber k.
    kmax: float
        The maximum value of the wavenumber k.
    dk: float
        The spacing between k-bins.
    '''

    def __init__(self, kmin:float=None, kmax:float=None, dk:float=None, num_kbins:int=None) -> None:
        super().__init__()
        if dk == None and num_kbins is not None:
            dk = (kmax - kmin) / num_kbins
        self.set_kbins(kmin, kmax, dk)

    @property
    def kmid(self):
        '''
        Returns the midpoints of the k-bins.

        Returns
        -------
        numpy.ndarray
            The midpoints of the k-bins.
        '''
        return np.arange(self.kmin + self.dk/2, self.kmax + self.dk/2, self.dk)

    @property
    def kedges(self):
        '''
        Returns the edges of the k-bins.

        Returns
        -------
        numpy.ndarray
            The edges of the k-bins.
        '''

        return np.arange(self.kmin, self.kmax + self.dk/2, self.dk)

    @property
    def nmodes(self):
        '''This function calculates the number of modes per k-bin shell. If nmodes was not provided, it is
        extimated from the volume of each shell.

        Returns
        -------
        numpy.ndarray
            The number of modes per k-bin shell.
        '''

        if hasattr(self, '_nmodes'):
            return self._nmodes

        return self.volume / 3. / (2*np.pi**2) * (self.kedges[:-1]**3 - self.kedges[1:]**3)


class LogBinning(FourierBinning):
    '''A class to represent an observable logarithmically binned in wavenumber k.

    Attributes
    ----------
    kmin: float
        The minimum value of the wavenumber k.
    kmax: float
        The maximum value of the wavenumber k.
    dk: float
        The logarithmic spacing between k-bins.
    '''

    def __init__(self, kmin:float=None, kmax:float=None, dk:float=None, num_kbins:int=None) -> None:
        super().__init__()
        if dk == None and num_kbins is not None:
            dk = (np.log(kmax) - np.log(kmin)) / num_kbins
        self.set_kbins(kmin, kmax, dk)

    @property
    def kmid(self):
        '''
        Returns the midpoints of the k-bins.

        Returns
        -------
        numpy.ndarray
            The midpoints of the k-bins.
        '''
        return np.sqrt(self.kedges[:-1]*self.kedges[1:])

    @property
    def kedges(self):
        '''
        Returns the edges of the k-bins.

        Returns
        -------
        numpy.ndarray
            The edges of the k-bins.
        '''
        nbins = int(np.floor(np.log(self.kmax/self.kmin)/self.dk))
        return np.geomspace(self.kmin, self.kmax, nbins + 1)

    @property
    def nmodes(self):
        '''This function calculates the number of modes per k-bin shell. If nmodes was not provided, it is
        extimated from the volume of each shell.

        Returns
        -------
        numpy.ndarray
            The number of modes per k-bin shell.
        '''

        if hasattr(self, '_nmodes'):
            return self._nmodes

        return self.volume / 3. / (2*np.pi**2) * (self.kedges[:-1]**3 - self.kedges[1:]**3)
