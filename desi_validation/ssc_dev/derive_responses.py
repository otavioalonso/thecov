"""Squeezed limit of the tree-level redshift-space galaxy trispectrum: response of P_s(k, k^) to a
long matter mode along q^ (unit amplitude), with the Z1/Z2 kernels (Scoccimarro, Couchman & Frieman 1999).
R(k; q^) = lim_{eps->0} 2 [ Z2(ka, -q) Z1(ka) P(ka) + Z2(kb, -q) Z1(kb) P(kb) ],  ka = k, kb = -k + q, q = eps q^.
Then multipoles in mu_k = k^.n^ (after azimuthal average) and Legendre components in nu = q^.n^."""
import sympy as sp
eps, k, b1, b2, bs2, f, n = sp.symbols('epsilon k b1 b2 bs2 f n_eff', real=True)
ct, st, cp, sp_, ca, sa = sp.symbols('c_t s_t c_p s_p c_a s_a', real=True)   # k^ = (st cp, st sp, ct), q^ = (sa, 0, ca), n^ = z
khat = sp.Matrix([st*cp, st*sp_, ct]); qhat = sp.Matrix([sa, 0, ca]); nhat = sp.Matrix([0, 0, 1])
def mag2(v): return v.dot(v)
def Z1(kv):
    mu2 = (kv.dot(nhat))**2 / mag2(kv)
    return b1 + f*mu2
def F2(k1, k2):
    c = k1.dot(k2); m1, m2 = mag2(k1), mag2(k2)
    return sp.Rational(5,7) + c/2*(1/m1 + 1/m2) + sp.Rational(2,7)*c**2/(m1*m2)
def G2(k1, k2):
    c = k1.dot(k2); m1, m2 = mag2(k1), mag2(k2)
    return sp.Rational(3,7) + c/2*(1/m1 + 1/m2) + sp.Rational(4,7)*c**2/(m1*m2)
def S2(k1, k2):
    c = k1.dot(k2); m1, m2 = mag2(k1), mag2(k2)
    return c**2/(m1*m2) - sp.Rational(1,3)
def Z2(k1, k2):
    kk = k1 + k2
    mu = kk.dot(nhat); mk2 = mag2(kk)           # mu*|k| = kk.n
    mu1 = k1.dot(nhat); mu2 = k2.dot(nhat)      # mu_i * |k_i|
    m1, m2 = mag2(k1), mag2(k2)
    rsd = f*mu/2 * (mu1/m1*(b1 + f*mu2**2/m2) + mu2/m2*(b1 + f*mu1**2/m1))   # (f mu k/2)[mu1/k1 Z1(k2) + mu2/k2 Z1(k1)]
    return b1*F2(k1,k2) + f*mu**2/mk2*G2(k1,k2) + rsd + b2/2 + bs2/2*S2(k1,k2)
ka = k*khat; q = eps*qhat; kb = -ka + q
# P(|kb|)/P(k) = (|kb|/k)^n_eff to first order in eps
ratio = 1 + n*(sp.sqrt(mag2(kb))/k - 1)
expr = 2*(Z2(ka, -q)*Z1(ka) + Z2(kb, -q)*Z1(kb)*ratio)
# substitute unit-vector constraints and expand in eps
subs = {sp_**2: 1 - cp**2}
ser = sp.series(expr, eps, 0, 1).removeO()
ser = sp.expand(ser)
# check no 1/eps term
assert sp.simplify(ser.coeff(eps, -1)) == 0, ser.coeff(eps, -1)
R = sp.expand(ser.coeff(eps, 0))
R = R.subs({st**2: 1 - ct**2, sa**2: 1 - ca**2})
R = sp.expand(sp.simplify(R))
# azimuthal average over phi: cp -> cos(phi), sp -> sin(phi)
phi = sp.symbols('phi', real=True)
Rphi = R.subs({cp: sp.cos(phi), sp_: sp.sin(phi)})
Ravg = sp.integrate(sp.expand(Rphi), (phi, 0, 2*sp.pi))/(2*sp.pi)
Ravg = sp.expand(Ravg.subs({st: sp.sqrt(1-ct**2), sa: sp.sqrt(1-ca**2)}))
Ravg = sp.expand(sp.simplify(Ravg))
mu, nu = sp.symbols('mu nu', real=True)
Ravg = Ravg.subs({ct: mu, ca: nu})
print('R(k, mu; nu) azimuth-averaged, in units of P_lin(k):'); print(sp.collect(Ravg, [n], evaluate=True))
# Legendre projections: ell in 0,2,4 (mu), n in 0,2,4 (nu)
def leg(l, x): return sp.legendre(l, x)
out = {}
for l in (0, 2, 4):
    Rl = sp.Rational(2*l+1, 2)*sp.integrate(Ravg*leg(l, mu), (mu, -1, 1))
    for m in (0, 2, 4):
        c = sp.Rational(2*m+1, 2)*sp.integrate(sp.expand(Rl)*leg(m, nu), (nu, -1, 1))
        c = sp.factor(sp.simplify(c))
        out[(l, m)] = c
        print(f'R_{l}^({m}) =', sp.collect(sp.expand(c), n))
# checks: real space b1=1 b2=bs2=f=0
chk = out[(0,0)].subs({b1:1, b2:0, bs2:0, f:0}); print('real-space isotropic:', sp.simplify(chk), ' expected 47/21 - n/3')
chk = out[(2,2)].subs({b1:1, b2:0, bs2:0, f:0}); print('real-space ell=2,n=2:', sp.simplify(chk))
import pickle; pickle.dump({k_: sp.srepr(v) for k_, v in out.items()}, open('responses.pkl','wb'))
