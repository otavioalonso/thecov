"""This module contains utility functions for thecov.
"""
import os, functools, psutil, sys
import numpy as np
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

def ellmiter(lmax, n):
    for ls in itt.product(range(0, lmax + 1, 2), repeat=n):
        for ms in itt.product(*[range(-l, l+1, 2) for l in ls]):
            yield ls + ms

def elliter(lmax, n):
    for ls in itt.product(range(0, lmax + 1, 2), repeat=n):
        yield ls


def miter(*ls):
    for ms in itt.product(*[range(-l, l+1, 2) for l in ls]):
        yield ms

def get_tqdm():
    """Get the tqdm module, compatible with Jupyter notebooks and terminals."""
    try: 
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            from tqdm.notebook import tqdm as tqdm
        else:
            # Terminal or other environment
            from tqdm import tqdm as tqdm
    except NameError:
        # Not in a Jupyter environment
        from tqdm import tqdm as tqdm
    return tqdm

def get_minimum_mesh_size(dk, kmax, boxsize):
    """Get the minimum mesh size for a given dk, kmax, and boxsize."""
    target_boxsize = 2*np.pi/dk
    min_nmesh = (target_boxsize * kmax / np.pi) / (target_boxsize/boxsize)
    return int(np.ceil(min_nmesh))

def get_available_memory():
    """Get the available system memory in Gigabytes."""
    return psutil.virtual_memory().available / (1024 ** 3)


def gather_field_to_root(field, root=0):
    """Gather a distributed 3D slab `field` onto `root`, preserving spatial layout.

    Args:
        field (pmesh Field): A pmesh Field (e.g. `RealField`) with attributes `pm`, `start`, `shape`, and
            `value` representing the local slab (numpy array) on each rank.
        root (int): MPI rank to gather to. Default is 0.

    Returns:
        (numpy.ndarray or None): On `root`, returns the reconstructed full array with global shape
            On `root`, returns the reconstructed full array with global shape
            `field.pm.Nmesh` (or `field.pm.Nmesh` for real fields). On non-root ranks,
            returns ``None``.
    """
    import numpy as _np

    pm = field.pm
    comm = pm.comm

    # local slab and its global start/shape
    local = _np.array(field.value, copy=False)
    start = tuple(int(s) for s in field.start)
    shape = tuple(int(s) for s in field.shape)

    # gather starts and shapes from all ranks to the root
    all_starts = comm.gather(start, root=root)
    all_shapes = comm.gather(shape, root=root)
    all_slabs = comm.gather(local, root=root)

    if comm.rank != root:
        return 0 # <- dummy number to avoid NoneType issues

    # allocate full array on root
    # use Nmesh for real-space fields, for complex fields use pm.Nmesh but their
    # represented storage may differ. We'll use pm.Nmesh for spatial layout.
    full_shape = tuple(int(n) for n in pm.Nmesh)
    full = _np.zeros(full_shape, dtype=local.dtype)

    # place each slab into the full array at the recorded start
    for st, sh, slab in zip(all_starts, all_shapes, all_slabs):
        slices = tuple(slice(s, s + n) for s, n in zip(st, sh))
        full[slices] = slab

    return full

def trim_fourier_mesh(mesh:np.ndarray, nmesh:int, new_nmesh:int):
    """Trim a fourier-space mesh to a smaller size, preserving positive and negative frequencies.
    
    Args:
        mesh (numpy.ndarray): The fourier-space mesh to trim.
        nmesh (int): The original size of the mesh.
        new_nmesh (int): The desired size of the mesh.

    Returns:
        numpy.ndarray: The trimmed fourier-space mesh.
    """
    mesh = np.fft.fftshift(mesh)
    center = nmesh // 2
    half   = new_nmesh // 2
    mesh = mesh[center-half:center+half,
                center-half:center+half,
                center-half:center+half]
    return np.fft.ifftshift(mesh)
