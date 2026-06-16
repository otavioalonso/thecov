"""This module contains utility functions for thecov.
"""
import os, functools
import itertools as itt

def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return
    
# python's enumerate but with a custom step = 2
def enum2(xs, start=0, step=2):
    """Enumerate a sequence with a custom step.

    Parameters
    ----------
    xs : sequence
        Sequence to enumerate.
    start : int, optional
        Starting index. Default is 0.
    step : int, optional
        Step of the enumeration. Default is 2.

    Returns
    -------
    generator
        Generator of tuples (index, element).
    """
    for x in xs:
        yield (start, x)
        start += step

def limit(iterable, count):
    """
    Limit number of iterated elements from an iterable.
    count -- is the maximum number of elements to iterate through
    """
    while count > 0:
        yield next(iterable)
        count -= 1

def cache_method(func):
    '''Decorator to cache the result of a method.

    Parameters
    ----------
    func : callable
        Method to cache.

    Returns
    -------
    callable
        Cached method.
    '''

    @functools.wraps(func)
    def cached_func(self, *args, **kwargs):
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        if func.__name__ not in self._cache:
            self._cache[func.__name__] = {}
        
        if len(args) + len(kwargs) == 1:
            key = args[0] if args else next(iter(kwargs.values()))
        else:
            key = hash((args, frozenset(kwargs.items())))

        cache = self._cache[func.__name__]
        
        if key not in cache:
            cache[key] = func(self, *args, **kwargs)
    
        return cache[key]

    return cached_func

def ellmiter(lmax, n, enumerate=False):
    i = 0
    for ls in itt.product(range(0, lmax + 1, 2), repeat=n):
        for ms in itt.product(*[range(-l, l+1) for l in ls]):
            if enumerate:
                yield i, ls + ms
            else:
                yield ls + ms
            i += 1

def elliter(lmax, n):
    for ls in itt.product(range(0, lmax + 1, 2), repeat=n):
        yield ls


def miter(*ls):
    for ms in itt.product(*[range(-l, l+1) for l in ls]):
        yield ms

def ellm_to_index(ell, m, ellmax, even_only=True, only_positive_m=False):
    """Convert an (ell, m) pair to a unique flat index.

    The ordering iterates m from 0 to ellmax then -1 to -ellmax,
    and for each m iterates ell from the smallest valid value up to ellmax.

    With even_only=True (default), only even ells are included:
        (0,0), (2,0), (4,0), ..., (2,1), (4,1), ..., (4,2), ..., (2,-1), ...

    With even_only=False, all ells |m| <= ell <= ellmax are included.

    Parameters
    ----------
    ell : int
    m : int
    ellmax : int
    even_only : bool, optional
        If True (default), only even ells are used.
    only_positive_m : bool, optional
        If True, only m >= 0 are included. Default is False.

    Returns
    -------
    int
        Flat index of the (ell, m) pair.
    """
    step = 2 if even_only else 1
    idx = 0
    ms = list(range(0, ellmax + 1))
    if not only_positive_m:
        ms += list(range(-1, -ellmax - 1, -1))

    for mm in ms:
        ell_min = abs(mm)
        if even_only and ell_min % 2 != 0:
            ell_min += 1
        for ll in range(ell_min, ellmax + 1, step):
            if ll == ell and mm == m:
                return idx
            idx += 1
    raise ValueError(f'(ell={ell}, m={m}) not found for ellmax={ellmax}, even_only={even_only}')


def n_ellm(ellmax, even_only=True, only_positive_m=False):
    """Total number of (ell, m) pairs for the given ellmax and mode.

    Parameters
    ----------
    ellmax : int
    even_only : bool, optional
        If True (default), only even ells are used.
    only_positive_m : bool, optional
        If True, only m >= 0 are counted. Default is False.

    Returns
    -------
    int
    """
    step = 2 if even_only else 1
    ms = list(range(0, ellmax + 1))
    if not only_positive_m:
        ms += list(range(-1, -ellmax - 1, -1))
    count = 0
    for mm in ms:
        ell_min = abs(mm)
        if even_only and ell_min % 2 != 0:
            ell_min += 1
        count += len(range(ell_min, ellmax + 1, step))
    return count


def index_to_ellm(idx, ellmax, even_only=True, only_positive_m=False):
    """Inverse of ellm_to_index: convert a flat index back to (ell, m).

    Parameters
    ----------
    idx : int
        Flat index in [0, n_ellm(ellmax, even_only, only_positive_m)).
    ellmax : int
    even_only : bool, optional
        If True (default), only even ells are used.
    only_positive_m : bool, optional
        If True, only m >= 0 are included. Default is False.

    Returns
    -------
    (ell, m) : tuple of int
    """
    step = 2 if even_only else 1
    ms = list(range(0, ellmax + 1))
    if not only_positive_m:
        ms += list(range(-1, -ellmax - 1, -1))

    count = 0
    for mm in ms:
        ell_min = abs(mm)
        if even_only and ell_min % 2 != 0:
            ell_min += 1
        for ll in range(ell_min, ellmax + 1, step):
            if count == idx:
                return ll, mm
            count += 1
    raise IndexError(f'Index {idx} out of range for ellmax={ellmax}, '
                     f'even_only={even_only}, only_positive_m={only_positive_m} '
                     f'(max index is {count - 1})')