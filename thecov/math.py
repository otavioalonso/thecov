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

def get_real_Ylm(ell, m):
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
    import sympy as sp

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


def double_spherical_bessel_transform(xbins, W, kedges, l1=0, l2=0, nq=16):
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