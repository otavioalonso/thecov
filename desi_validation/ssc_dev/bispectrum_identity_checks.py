"""Numerical checks of the identities used in report/bispectrum_covariance_guide.typ: Sugiyama-basis orthogonality
and normalisation, H_202 and S_202 = L_2, the box value of the window invariant, and the opening-angle expansion
(Tool 3, both signs). Run: python desi_validation/ssc_dev/bispectrum_identity_checks.py"""
import numpy as np
from sympy.physics.wigner import wigner_3j
from scipy.special import sph_harm_y, spherical_jn, eval_legendre
rng = np.random.default_rng(1)
def Y(l, m, v):  # v (...,3) unit
    th = np.arccos(np.clip(v[..., 2], -1, 1)); ph = np.arctan2(v[..., 1], v[..., 0])
    return sph_harm_y(l, m, th, ph)
def unit(n):
    v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1)[:, None]
def tj(a, b, c, x, y, z): return float(wigner_3j(a, b, c, x, y, z))
def H(l1, l2, L): return tj(l1, l2, L, 0, 0, 0)
def S(ell, a, b, n):  # Sugiyama S with y = sqrt(4pi/(2l+1)) Y
    l1, l2, L = ell; out = 0
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            M = -m1 - m2
            if abs(M) > L: continue
            c = tj(l1, l2, L, m1, m2, M)
            if c == 0: continue
            out = out + c * np.sqrt(4*np.pi/(2*l1+1))*Y(l1, m1, a) * np.sqrt(4*np.pi/(2*l2+1))*Y(l2, m2, b) * np.sqrt(4*np.pi/(2*L+1))*Y(L, M, n)
    return out / H(*ell)
# 1. normalisation and H_202
print('H_202 =', H(2, 0, 2), ' 1/sqrt5 =', 1/np.sqrt(5))
n0 = np.array([0.3, -0.4, np.sqrt(1-0.25)])
N = 400000; a = unit(N); b = unit(N); n = np.broadcast_to(n0, a.shape)
for e1 in [(0,0,0), (2,0,2), (1,1,0), (2,2,0)]:
    for e2 in [(0,0,0), (2,0,2), (1,1,0), (2,2,0)]:
        v = np.mean(np.conj(S(e1, a, b, n)) * S(e2, a, b, n))
        NH2 = (2*e1[0]+1)*(2*e1[1]+1)*(2*e1[2]+1)*H(*e1)**2
        if abs(v) > 0.02 or e1 == e2: print(e1, e2, 'avg conj(S)S =', np.round(v, 3), ' 1/(N H^2) =', round(1/NH2, 3) if e1 == e2 else 0)
# S_202 = L2(a.n)
print('S_202 - L2 max:', np.max(np.abs(S((2,0,2), a[:1000], b[:1000], n[:1000]) - eval_legendre(2, np.sum(a[:1000]*n0, 1)))))
# 3. sum_m 3j Y Y Y (same direction) = H sqrt(N)/(4pi)^{3/2}
for (l1, l2, L) in [(0,0,0), (2,2,0), (2,0,2), (2,2,2), (2,2,4)]:
    s = sum(tj(l1, l2, L, m1, m2, -m1-m2) * Y(l1, m1, n0) * Y(l2, m2, n0) * Y(L, -m1-m2, n0)
            for m1 in range(-l1, l1+1) for m2 in range(-l2, l2+1) if abs(m1+m2) <= L)
    Nn = (2*l1+1)*(2*l2+1)*(2*L+1)
    print((l1, l2, L), np.round(s, 6), np.round(H(l1, l2, L)*np.sqrt(Nn)/(4*np.pi)**1.5, 6))
# 4. opening-angle identity: <e^{i p1.r1} e^{+-i p2.r2} L_a(p1^.p2^)>_{angles} = (+-1)^a... j_a j_a L_a(r1^.r2^)
p1, p2, r1, r2 = 0.13, 0.11, 37.0, 52.0
r1v = r1*np.array([0, 0, 1.]); r2v = r2*np.array([np.sin(1.1), 0, np.cos(1.1)])
u = unit(2_000_000); w = unit(2_000_000)
for a_ in [0, 1, 2, 3]:
    La = eval_legendre(a_, np.sum(u*w, 1))
    for sgn in (+1, -1):
        mc = np.mean(np.exp(1j*p1*u@r1v) * np.exp(sgn*1j*p2*w@r2v) * La)
        pred = (1 if sgn > 0 else -1)**0 * ((-1)**a_ if sgn > 0 else 1) * spherical_jn(a_, p1*r1)*spherical_jn(a_, p2*r2)*eval_legendre(a_, np.cos(1.1))/(2*a_+1)
        print(f'a={a_} sign={sgn:+d}: MC {mc.real:+.5f}{mc.imag:+.5f}i  pred {pred:+.5f}')
