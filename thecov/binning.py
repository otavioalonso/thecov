import numpy as np

class FourierBinning:
    """A parent class to represent an observable binned in wavenumber k."""

    def __init__(self):
        self.kmin = None
        self.kmax = None
        self.dk = None
        self._nmodes = None

    def set_kbins(self, kmin:float, kmax:float, dk:float, kbins:int=None, nmodes=None):
        '''This function defines the k-bins.

        Parameters
        ----------
        kmin: float
            The minimum value of the wavenumber k.
        kmax: float
            The maximum value of the wavenumber k.
        dk: float
            The (log) spacing between k-bins.
        kbins: int, optional
            The number of k-bins.
        nmodes: numpy.ndarray, optional
            The number of modes to be used in the calculation. It is an optional parameter.
            If omitted, it is calculated from the volume of spherical shells.
        '''

        self.kmax = kmax
        self.kmin = kmin
        self.dk = dk
        self._kbins = kbins
        self._nmodes = nmodes
    
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

    def __init__(self, kmin:float=None, kmax:float=None, dk:float=None, kbins:int=None) -> None:
        super().__init__()
        if dk is None and kbins is not None:
            dk = (kmax - kmin) / kbins
        self.set_kbins(kmin, kmax, dk, kbins)

    @property
    def kbins(self):
        '''The number of k-bins. If not set, it is calculated from kmin, kmax and dk.

        Returns
        -------
        int
            The number of k-bins.
        '''
        return round((self.kmax - self.kmin) / self.dk)

    @property
    def kmid(self):
        '''
        Returns the midpoints of the k-bins.

        Returns
        -------
        numpy.ndarray
            The midpoints of the k-bins.
        '''
        return np.linspace(self.kmin + self.dk/2, self.kmax - self.dk/2, self.kbins)

    @property
    def kedges(self):
        '''
        Returns the edges of the k-bins.

        Returns
        -------
        numpy.ndarray
            The edges of the k-bins.
        '''
        if self.kbins is None:
            self.kbins = int((self.kmax - self.kmin) / self.dk)
        return np.linspace(self.kmin, self.kmax, self.kbins + 1)

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

    def __init__(self, kmin:float=None, kmax:float=None, dk:float=None, kbins:int=None) -> None:
        super().__init__()
        if kbins is not None and dk is None:
            dk = np.log(kmax/kmin) / kbins
        self.set_kbins(kmin, kmax, dk, kbins)

    @property
    def kbins(self):
        '''The number of k-bins. If not set, it is calculated from kmin, kmax and dk.

        Returns
        -------
        int
            The number of k-bins.
        '''
        return int(np.floor(np.log(self.kmax/self.kmin)/self.dk))

    @property
    def kmid(self):
        '''
        Returns the midpoints of the k-bins.

        Returns
        -------
        numpy.ndarray
            The midpoints of the k-bins.
        '''
        return np.exp(np.linspace(np.log(self.kmin) + self.dk/2, np.log(self.kmax) - self.dk/2, self.kbins))
        #return np.sqrt(self.kedges[:-1]*self.kedges[1:])

    @property
    def kedges(self):
        '''
        Returns the edges of the k-bins.

        Returns
        -------
        numpy.ndarray
            The edges of the k-bins.
        '''
        return np.geomspace(self.kmin, self.kmax, self.kbins + 1)

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
