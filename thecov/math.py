"""This module contains math functions for the covariance calculation
"""
import numpy as np
import sympy as sp

from functools import lru_cache
from sympy.physics.wigner import real_gaunt
from . import utils

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




@lru_cache(maxsize=None)
def gaunt(ells, ms=None, ellmax=12):
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
    if ms is None:
        ells, ms = utils.index_to_ellm(ells, ellmax=ellmax)

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

def Ylm(ell, m):
    """Return a JAX-traceable function that computes the real spherical harmonic Y_ell^m(x, y, z).

    Uses sympy to expand the associated Legendre polynomial into a pure polynomial
    in the unit-vector components (xhat, yhat, zhat), then lambdifies against
    jax.numpy. The result is fully JIT-compilable and runs on GPU.

    Parameters
    ----------
    ell : int
        Degree of the harmonic.
    m : int
        Order of the harmonic; abs(m) <= ell.

    Returns
    -------
    Ylm : callable
        Function of (x, y, z) — unnormalised Cartesian coordinates — returning Y_ell^m
        evaluated on the corresponding unit vectors. Returns 0 at the origin.
    """

    ell = int(ell)
    m   = int(m)

    # Normalisation amplitude
    amp = np.sqrt((2 * ell + 1) / (4 * np.pi))
    if m != 0:
        fac = 1
        for n in range(ell - abs(m) + 1, ell + abs(m) + 1): fac *= n
        amp *= np.sqrt(2. / fac)

    # Build symbolic expression in (xhat, yhat, zhat)
    x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
    xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
    phi, theta = sp.symbols('phi theta')
    defs = [(sp.sin(phi), y / sp.sqrt(x**2 + y**2)),
            (sp.cos(phi), x / sp.sqrt(x**2 + y**2)),
            (sp.cos(theta), z / sp.sqrt(x**2 + y**2 + z**2))]

    expr = (-1)**m * sp.assoc_legendre(ell, abs(m), sp.cos(theta))
    if m < 0:
        expr *= sp.expand_trig(sp.sin(abs(m) * phi))
    elif m > 0:
        expr *= sp.expand_trig(sp.cos(m * phi))

    expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x / r, xhat), (y / r, yhat), (z / r, zhat)])

    # Lambdify against jax.numpy: all arithmetic ops map to jnp equivalents,
    # making the result fully JAX-traceable and JIT-compilable.
    _Ylm = sp.lambdify((xhat, yhat, zhat), expr, modules=[np])

    def Ylm(x, y, z):
        norm = np.sqrt(x**2 + y**2 + z**2)
        mask = norm == 0
        norm = np.where(mask, 1.0, norm)
        result = _Ylm(x / norm, y / norm, z / norm)
        return np.where(mask, 0.0, result)

    Ylm.expr = expr
    Ylm.l = ell
    Ylm.m = m
    return Ylm

def spherical_bessel(nu, r, kedges, averaged=True):
    from jax.scipy.special import sici as _sici
    import jax.numpy as jnp

    SinIntegral = lambda z: _sici(z)[0]
    Sin, Cos = jnp.sin, jnp.cos

    kmin = kedges[:-1, None]
    kmax = kedges[1:, None]

    # Guard against r=0 (e.g. padded particles at the origin): replace with 1.0
    # for the computation, then restore the correct limit at the end.
    # Limits: j_nu(0) = 1 for nu=0, 0 for nu>0. Same holds for the averaged forms.
    limit  = 1.0 if nu == 0 else 0.0
    r_safe = jnp.where(r == 0, 1.0, r)
    x      = r_safe[None, :]        # shape [1, N], broadcasts with [nb, 1]
    mask   = (r == 0)[None, :]      # True where result should equal `limit`

    if averaged:
        norm = 3/(kmax**3 - kmin**3) # 1/Integrate[k^2, {k, kmin, kmax}]

        # Result of ToString[FortranForm[Simplify[
        #             Integrate[k^2 * SphericalBesselJ[nu, k*x], {k, kmin, kmax}],
        #             Assumptions->Element[{k, x}, PositiveReals]]]]
        if nu == 0:
            result = (-(kmax*x*Cos(kmax*x)) + kmin*x*Cos(kmin*x) + Sin(kmax*x) - Sin(kmin*x))/x**3
        elif nu == 1:
            result = (-2*Cos(kmax*x) + 2*Cos(kmin*x) - kmax*x*Sin(kmax*x) + kmin*x*Sin(kmin*x))/x**3
        elif nu == 2:
            result = (kmax*x*Cos(kmax*x) - kmin*x*Cos(kmin*x) - 4*Sin(kmax*x) + 4*Sin(kmin*x) + 3*SinIntegral(kmax*x) - 3*SinIntegral(kmin*x))/x**3
        elif nu == 3:
            result = ((-15*Sin(kmax*x))/kmax + (15*Sin(kmin*x))/kmin + x*(7*Cos(kmax*x) - 7*Cos(kmin*x) + kmax*x*Sin(kmax*x) - kmin*x*Sin(kmin*x)))/x**4
        elif nu == 4:
            result = ((105*x*Cos(kmax*x))/kmax - 2*kmax*x**3*Cos(kmax*x) - (105*x*Cos(kmin*x))/kmin + 2*kmin*x**3*Cos(kmin*x) - (105*Sin(kmax*x))/kmax**2 + 22*x**2*Sin(kmax*x) + (105*Sin(kmin*x))/kmin**2 - 22*x**2*Sin(kmin*x) + 15*x**2*SinIntegral(kmax*x) - 15*x**2*SinIntegral(kmin*x))/(2.*x**5)
        elif nu == 5:
            result = (kmax*kmin**3*x*(315 - 16*kmax**2*x**2)*Cos(kmax*x) - kmin**3*(315 - 105*kmax**2*x**2 + kmax**4*x**4)*Sin(kmax*x) + kmax**3*(kmin*x*(-315 + 16*kmin**2*x**2)*Cos(kmin*x) + (315 - 105*kmin**2*x**2 + kmin**4*x**4)*Sin(kmin*x)))/(kmax**3*kmin**3*x**6)
        elif nu == 6:
            result = ((20790*x*Cos(kmax*x))/kmax**3 - (1575*x**3*Cos(kmax*x))/kmax + 8*kmax*x**5*Cos(kmax*x) - (20790*x*Cos(kmin*x))/kmin**3 + (1575*x**3*Cos(kmin*x))/kmin - 8*kmin*x**5*Cos(kmin*x) - (20790*Sin(kmax*x))/kmax**4 + (8505*x**2*Sin(kmax*x))/kmax**2 - 176*x**4*Sin(kmax*x) + (20790*Sin(kmin*x))/kmin**4 - (8505*x**2*Sin(kmin*x))/kmin**2 + 176*x**4*Sin(kmin*x) + 105*x**4*SinIntegral(kmax*x) - 105*x**4*SinIntegral(kmin*x))/(8.*x**7)
        else:
            raise ValueError("Unsupported nu value for averaged spherical Bessel function")

        return jnp.where(mask, limit, norm * result)

    else:
        # Evaluate j_nu at the bin midpoint kmid = (kmin + kmax) / 2
        z = (kmin + kmax) / 2 * x
        if nu == 0:
            result = Sin(z) / z
        elif nu == 1:
            result = Sin(z)/z**2 - Cos(z)/z
        elif nu == 2:
            result = (3/z**3 - 1/z)*Sin(z) - 3*Cos(z)/z**2
        elif nu == 3:
            result = (15/z**4 - 6/z**2)*Sin(z) - (15/z**3 - 1/z)*Cos(z)
        elif nu == 4:
            result = (105/z**5 - 45/z**3 + 1/z)*Sin(z) - (105/z**4 - 10/z**2)*Cos(z)
        elif nu == 5:
            result = (945/z**6 - 420/z**4 + 15/z**2)*Sin(z) - (945/z**5 - 105/z**3 + 1/z)*Cos(z)
        elif nu == 6:
            result = (10395/z**7 - 4725/z**5 + 210/z**3 - 1/z)*Sin(z) - (10395/z**6 - 1260/z**4 + 21/z**2)*Cos(z)
        else:
            raise ValueError("Unsupported nu value for spherical Bessel function")

        return jnp.where(mask, limit, result)


def get_gaunt_coefficients(term, mask_ellmax=4, pk_ellmax=4, cache_dir=None):
    """Calculates all relevant Gaunt coefficients for the given term, or loads them from file"""

    import logging, os, multiprocessing
    from tqdm import tqdm
    from thecov.utils import ellmiter, elliter, n_ellm
    from thecov import base

    logger = logging.getLogger('SurveyGeometry')

    # Load mask coupling Gaunt coefficients if cache exists, otherwise compute them
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
        logger.info(f'Using cache directory: {cache_dir}')

    filename = os.path.join(cache_dir, f"gaunts_{term}_{pk_ellmax:d}_{mask_ellmax:d}.npz")

    if os.path.exists(filename):
        logger.info(f'Loading {term} Gaunt coefficients from cache: {filename}')
        return base.SparseNDArray.load(filename)
    else:
        logger.info(f'Computing {term} Gaunt coefficients (pk_ellmax={pk_ellmax}, mask_ellmax={mask_ellmax})...')

        shape_in = (2*[mask_ellmax//2+1] + [n_ellm(mask_ellmax)] +
                    2*[mask_ellmax//2+1] + [n_ellm(mask_ellmax)])
        
        if term.startswith('cosmic_variance_'):
            sub_term = term.split('_')[-1]
            shape_out = 4*[pk_ellmax//2 + 1]
            outer_args = [
                (l1, l2, L1, L2, a1, a2, la, ma, b1, b2, lb, mb, mask_ellmax, sub_term)
                for l1, l2, L1, L2 in elliter(pk_ellmax, 4)
                for a1, a2, b1, b2 in elliter(mask_ellmax, 4)
                for la, lb, ma, mb in ellmiter(mask_ellmax, 2)
            ]
            worker_func = _gaunt_cv_worker
        elif term.startswith('mixed_'):
            sub_term = term.split('_')[-1]
            shape_out = 3*[pk_ellmax//2 + 1]
            outer_args = [
                (l1, l2, L, a1, a2, la, ma, b1, b2, lb, mb, mask_ellmax, sub_term)
                for l1, l2, L in elliter(pk_ellmax, 3)
                for a1, a2, b1, b2 in elliter(mask_ellmax, 4)
                for la, lb, ma, mb in ellmiter(mask_ellmax, 2)
            ]
            worker_func = _gaunt_mixed_worker
        elif term.startswith('shotnoise_'):
            sub_term = term.split('_')[-1]
            shape_out = 2*[pk_ellmax//2 + 1]
            outer_args = [
                (l1, l2, a1, a2, la, ma, b1, b2, lb, mb, mask_ellmax, sub_term)
                for l1, l2 in elliter(pk_ellmax, 2)
                for a1, a2, b1, b2 in elliter(mask_ellmax, 4)
                for la, lb, ma, mb in ellmiter(mask_ellmax, 2)
            ]
            worker_func = _gaunt_shotnoise_worker
        else:
            raise ValueError(f"Unknown term: {term}")

        gaunt_coefficients = base.SparseNDArray(shape_out=shape_out, shape_in=shape_in)
        n = len(outer_args)
        chunksize = max(1, n // (multiprocessing.cpu_count() * 20))

        logger.info(f'Computing Gaunt coefficients: {n:,} tasks across {multiprocessing.cpu_count()} CPUs '
                    f'(chunksize={chunksize})...')

        with multiprocessing.Pool() as pool:
            for result in tqdm(pool.imap_unordered(worker_func, outer_args, chunksize=chunksize),
                               total=n, desc='Gaunt coefficients'):
                if result is not None:
                    index, val = result
                    gaunt_coefficients[index] = val

        # save to cache
        os.makedirs(cache_dir, exist_ok=True)
        gaunt_coefficients.save(filename)

        return gaunt_coefficients


def _gaunt_cv_worker(args):
    """Compute one Gaunt coefficient entry for cosmic variance terms."""
    l1, l2, L1, L2, a1, a2, la, ma, b1, b2, lb, mb, mask_ellmax, sub_term = args
    from thecov.utils import ellm_to_index, miter
    from thecov.math import gaunt

    val = 0.0
    for m1, m2, M1, M2, nu1, nu2, rho1, rho2 in miter(l1, l2, L1, L2, a1, a2, b1, b2):
        base_val = gaunt((l1, L1, a1, b1), (m1, M1, nu1, rho1)) * gaunt((l2, L2, a2, b2), (m2, M2, nu2, rho2))
        
        if sub_term == 'ACBD':
            val += base_val * gaunt((L1, a1, a2, la), (M1, nu1, nu2, ma)) * gaunt((l1, l2, L2, b1, b2, lb), (m1, m2, M2, rho1, rho2, mb))
        elif sub_term == 'ADBC':
            val += base_val * gaunt((l2, L1, a1, a2, la), (m2, M1, nu1, nu2, ma)) * gaunt((l1, L2, b1, b2, lb), (m1, M2, rho1, rho2, mb))

    if val == 0.0:
        return None

    index = (l1//2, l2//2, L1//2, L2//2,
             a1//2, a2//2, ellm_to_index(la, ma, mask_ellmax),
             b1//2, b2//2, ellm_to_index(lb, mb, mask_ellmax))
    
    return index, val


def _gaunt_mixed_worker(args):
    """Compute one Gaunt coefficient entry for mixed terms."""
    l1, l2, L, a1, a2, la, ma, b1, b2, lb, mb, mask_ellmax, sub_term = args
    from thecov.utils import ellm_to_index, miter
    from thecov.math import gaunt

    val = 0.0
    for m1, m2, M, nu1, nu2, rho1, rho2 in miter(l1, l2, L, a1, a2, b1, b2):
        base_val = gaunt(l1, 0, a1, b1, m1, 0, nu1, rho1) * gaunt(l2, L, a2, b2, m2, 0, nu2, rho2)
        
        if sub_term == 'ACBD':
            val += base_val * gaunt((l1, l2, a1, a2, la), (m1, m2, nu1, nu2, ma)) * gaunt((0,  L, b1, b2, lb), (0,  M, rho1, rho2, mb))
        elif sub_term == 'ADBC':
            val += base_val * gaunt((l1,  0, a1, a2, la), (m1,  0, nu1, nu2, ma)) * gaunt((l2, L, b1, b2, lb), (m2, M, rho1, rho2, mb))
        elif sub_term == 'BCAD':
            val += base_val * gaunt((l2,  0, a1, a2, la), (m2,  0, nu1, nu2, ma)) * gaunt((l1, L, b1, b2, lb), (m1, M, rho1, rho2, mb))
        elif sub_term == 'BDAC':
            val += base_val * gaunt((0, a1, a2, la), (0, nu1, nu2, ma)) * gaunt((l1, l2, L, b1, b2, lb), (m1, m2, M, rho1, rho2, mb))

    if val == 0.0:
        return None

    index = (l1//2, l2//2, L//2,
             a1//2, a2//2, ellm_to_index(la, ma, mask_ellmax),
             b1//2, b2//2, ellm_to_index(lb, mb, mask_ellmax))
    
    return index, val


def _gaunt_shotnoise_worker(args):
    """Compute one Gaunt coefficient entry for shotnoise terms."""
    l1, l2, a1, a2, la, ma, b1, b2, lb, mb, mask_ellmax, sub_term = args
    from thecov.utils import ellm_to_index, miter
    from thecov.math import gaunt

    val = 0.0
    for m1, m2, nu1, nu2, rho1, rho2 in miter(l1, l2, a1, a2, b1, b2):
        base_val = gaunt(l1, 0, a1, b1, m1, 0, nu1, rho1) * gaunt(l2, 0, a2, b2, m2, 0, nu2, rho2)
        
        if sub_term == 'ACBD':
            val += base_val * gaunt((l1, l2, a1, a2, la), (m1, m2, nu1, nu2, ma)) * gaunt((0, 0, b1, b2, lb), (0, 0, rho1, rho2, mb))
        elif sub_term == 'ADBC':
            val += base_val * gaunt((l1,  0, a1, a2, la), (m1,  0, nu1, nu2, ma)) * gaunt((l2, 0, b1, b2, lb), (m2, 0, rho1, rho2, mb))

    if val == 0.0:
        return None

    index = (l1//2, l2//2,
             a1//2, a2//2, ellm_to_index(la, ma, mask_ellmax),
             b1//2, b2//2, ellm_to_index(lb, mb, mask_ellmax))
    
    return index, val


def double_spherical_bessel_transform(xbins, W, kedges, l1=0, l2=0, averaged=True):

    xbins = np.asarray(xbins, float)
    x = (xbins[1:] + xbins[:-1])/2
    
    W = np.asarray(W, float)
    kedges = np.asarray(kedges, float)
    
    # 1. Compute integration weights over x
    wx = np.diff(xbins) * x**2 * W
    
    # 2. Get the pre-averaged bessel functions for all k-bins and all x at once
    # spherical_bessel returns shape (nbins, nx)
    j1 = spherical_bessel(l1, x, kedges, averaged=averaged)
    j2 = spherical_bessel(l2, x, kedges, averaged=averaged) if l2 != l1 else j1
    
    # 3. Contract the spatial dimension
    return np.einsum('kx,qx,x->kq', j1, j2, wx)