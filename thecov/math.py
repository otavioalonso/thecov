"""This module contains math functions for the covariance calculation
"""
import numpy as np

def cov2cor(covariance):
    '''Compute the correlation matrix from the covariance matrix.

    Parameters
    ----------
    covariance : array_like
        Covariance matrix.

    Returns
    -------
    array_like
        Correlation matrix.'''
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def fgrowth(Omega_m, z):
    '''Estimates the growth rate at redshift z.

    Parameters
    ----------
    Omega_m : float
        Matter density parameter.
    z : float
        Redshift.

    Returns
    -------
    float
        Growth rate.
    '''
    from scipy.special import hyp2f1

    return (1. + 6*(Omega_m-1)*hyp2f1(4/3., 2, 17/6., (1-1/Omega_m)/(1+z)**3) / \
          (11*Omega_m*(1+z)**3*hyp2f1(1/3., 1, 11/6., (1-1/Omega_m)/(1+z)**3) ))

def get_real_Ylm(ell, m, modules=None):
    """
    Return a function that computes the real spherical harmonic of order (ell, m).
    Copied from pypower.
    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py.

    Note
    ----
    Faster evaluation will be achieved if sympy and numexpr are available.
    Else, fallback to numpy and scipy's functions.

    Parameters
    ----------
    ell : int
        The degree of the harmonic.

    m : int
        The order of the harmonic; abs(m) <= ell.

    modules : str, default=None
        If 'sympy', use sympy + numexpr to speed up calculation.
        If 'scipy', use scipy.
        If ``None``, defaults to sympy if installed, else scipy.

    Returns
    -------
    Ylm : callable
        A function that takes 3 arguments: (x, y, z)
        Cartesian coordinates and returns the specified Ylm,
        after normalizing the input vector.

    References
    ----------
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    """
    # Make sure ell, m are integers
    ell = int(ell)
    m = int(m)

    # Normalization of Ylms
    amp = np.sqrt((2 * ell + 1) / (4 * np.pi))
    if m != 0:
        fac = 1
        for n in range(ell - abs(m) + 1, ell + abs(m) + 1): fac *= n  # (ell + |m|)!/(ell - |m|)!
        amp *= np.sqrt(2. / fac)

    sp = None

    if modules is None:
        try: import sympy as sp
        except ImportError: pass

    elif 'sympy' in modules:
        import sympy as sp

    elif 'scipy' not in modules:
        raise ValueError('modules must be either ["sympy", "scipy", None]')

    # sympy is not installed, fallback to scipy
    if sp is None:
        import scipy.special

        def Ylm(x, y, z):
            norm = np.sqrt(x**2 + y**2 + z**2)
            mask = norm == 0
            norm = np.where(mask, 1, norm)
            xhat, yhat, zhat = x / norm, y / norm, z / norm
            toret = amp * (-1)**m * scipy.special.lpmv(abs(m), ell, zhat)
            phi = np.arctan2(yhat, xhat)
            if m < 0:
                toret *= np.sin(abs(m) * phi)
            else:
                toret *= np.cos(abs(m) * phi)
            return np.where(mask, 0, toret)

        # Attach some meta-data
        Ylm.l = ell
        Ylm.m = m
        return Ylm

    # The relevant cartesian and spherical symbols
    # Using intermediate variable r helps sympy simplify expressions
    x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
    xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
    phi, theta = sp.symbols('phi theta')
    defs = [(sp.sin(phi), y / sp.sqrt(x**2 + y**2)),
            (sp.cos(phi), x / sp.sqrt(x**2 + y**2)),
            (sp.cos(theta), z / sp.sqrt(x**2 + y**2 + z**2))]

    # The cos(theta) dependence encoded by the associated Legendre polynomial
    expr = (-1)**m * sp.assoc_legendre(ell, abs(m), sp.cos(theta))

    # The phi dependence
    if m < 0:
        expr *= sp.expand_trig(sp.sin(abs(m) * phi))
    elif m > 0:
        expr *= sp.expand_trig(sp.cos(m * phi))

    # Simplify
    expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x / r, xhat), (y / r, yhat), (z / r, zhat)])

    try: import numexpr
    except ImportError: numexpr = None
    _Ylm = sp.lambdify((xhat, yhat, zhat), expr, modules='numexpr' if numexpr is not None else ['scipy', 'numpy'])

    def Ylm(x, y, z):
        norm = np.sqrt(x**2 + y**2 + z**2)
        mask = norm == 0
        norm = np.where(mask, 1, norm)
        xhat, yhat, zhat = x / norm, y / norm, z / norm
        result = _Ylm(xhat, yhat, zhat)
        return np.where(mask, 0, result)

    # Attach some meta-data
    Ylm.expr = expr
    Ylm.l = ell
    Ylm.m = m
    return Ylm

def double_spherical_bessel(xbins, W, kedges, l1=0, l2=0, nq=16):
    from scipy.special import spherical_jn
    
    xbins = np.asarray(xbins, float)
    x = (xbins[1:] + xbins[:-1])/2
    
    W = np.asarray(W, float)
    kedges = np.asarray(kedges, float)

    u, w = np.polynomial.legendre.leggauss(nq)
    nb = len(kedges) - 1
    out = np.zeros((nb, nb))

    wx = np.diff(xbins) * x**2 * W

    for i in range(nb):
        a1, b1 = kedges[i], kedges[i + 1]
        k1q = 0.5 * (b1 - a1) * u + 0.5 * (a1 + b1)
        w1 = 0.5 * (b1 - a1) * w
        j1 = spherical_jn(l1, np.outer(k1q, x))   # (nq, nx)

        for j in range(nb):
            a2, b2 = kedges[j], kedges[j + 1]
            k2q = 0.5 * (b2 - a2) * u + 0.5 * (a2 + b2)
            w2 = 0.5 * (b2 - a2) * w
            j2 = spherical_jn(l2, np.outer(k2q, x))   # (nq, nx)

            vals = np.einsum('ax,bx,x->ab', j1, j2, wx)   # integral over x
            out[i, j] = np.sum(np.outer(w1, w2) * vals) / ((b1 - a1) * (b2 - a2))

    return out

from functools import lru_cache
from sympy.physics.wigner import real_gaunt

@lru_cache(maxsize=None)
def gaunt(ells, ms, ellmax=12):
    """
    Generalized Gaunt coefficient: integral of N real spherical harmonics.
    Computed recursively by contracting the first two legs with a 3-point
    real_gaunt, summing over the intermediate (ell, m).

    Parameters
    ----------
    ells : tuple of int
    ms   : tuple of int, same length as ells
    ellmax : int, maximum ell in the intermediate sum
    """
    assert len(ells) == len(ms), "ells and ms must have the same length"

    if len(ells) == 3:
        return float(real_gaunt(*ells, *ms))

    if len(ells) < 3:
        raise ValueError("Need at least 3 harmonics")

    result = 0.0
    ell1, ell2 = ells[0], ells[1]
    m1,   m2   = ms[0],   ms[1]

    for ell in range(ellmax + 1):
        for m in range(-ell, ell + 1):
            g3 = float(real_gaunt(ell1, ell2, ell, m1, m2, m))
            if g3 == 0.0:
                continue  # skip: real_gaunt has hard selection rules
            rest = gaunt(
                (ell,) + ells[2:],
                (m,)   + ms[2:],
                ellmax,
            )
            result += g3 * rest

    return result

def bin(r, mesh, rbins=None):

    # build adaptive rbins if not supplied
    if rbins is None:
        order    = np.argsort(r)
        r_sorted = r[order]
        w_sorted = mesh[order]

        nrbins = max(10, 1.5*len(r)**(1/3))
        min_weight = 0.1*mesh.sum() / nrbins

        # Minimum bin width: r_max / nrbins, i.e. the width of a uniform bin
        # across the full radial range. This prevents the peak from being
        # over-resolved relative to a simple uniform grid with the same nrbins.
        min_dr = r_sorted[-1] / nrbins

        # Greedy forward pass: cut only when *both* the weight threshold is met
        # *and* the bin is at least min_dr wide. This smooths the peak (where
        # weight builds up fast) without widening the already-wide tail bins.
        edges = [0.0]
        cumw  = 0.0
        for i in range(len(r_sorted)):
            cumw += w_sorted[i]
            bin_width = r_sorted[i] - edges[-1]
            if cumw >= min_weight and bin_width >= min_dr and i < len(r_sorted) - 1:
                edges.append(0.5 * (r_sorted[i] + r_sorted[i + 1]))
                cumw = 0.0
        edges.append(r_sorted[-1] * (1.0 + 1e-9))
        rbins = np.array(edges)
        
    rbins = np.array(rbins, dtype=float)
    rbins[0]  = 0.0
    rbins[-1] = r.max() * (1.0 + 1e-9)

    # np.digitize returns 1-based indices; subtract 1 → 0-based, then clip
    # to guarantee indices lie in [0, len(rbins)-2] (i.e. N = len(rbins)-1 bins)
    d = np.clip(np.digitize(r, rbins) - 1, 0, len(rbins) - 2)
    W = np.bincount(d, weights=mesh, minlength=len(rbins) - 1)

    # Normalise by bin width so W is comparable across adaptive bins
    W = W / np.diff(rbins)

    return W, rbins
