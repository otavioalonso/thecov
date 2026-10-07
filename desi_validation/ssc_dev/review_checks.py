"""Independent numerical checks behind report/ssc_t0_review.md (2026-10-07). Run from the repository root:

    PYTHONPATH=. python desi_validation/ssc_dev/review_checks.py

1. the tree-level LRG1 response numbers and what they imply for the P_hat_0 - delta_norm regression slope
   under each normalisation convention (the sign test of the local-average term);
2. thecov.ssc.response_coefficients against thecov.trispectrum.squeezed_response (independent kernels, finite q);
3. the IR (k1 -> 0) limit of the T0 parallelogram: snake and star diverge as 1/k1^2 and cancel;
4. the real-space snake + star assembly against an independent one.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from thecov.ssc import response_coefficients
from thecov.trispectrum import squeezed_response, galileon_bias, parallelogram, LinearPower, Bias, FG3


def p_lin():
    k = np.logspace(-5, 1, 4000)
    return k, 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)


def responses_and_slopes(b1=2.2, f=0.76):
    b2 = 0.412 - 2.143 * b1 + 0.929 * b1 ** 2 + 0.008 * b1 ** 3
    bs2 = -4 / 7 * (b1 - 1)
    a, c = response_coefficients(b1, f, b2, bs2)
    K0 = b1 ** 2 + 2 / 3 * b1 * f + f ** 2 / 5
    K2 = 4 / 3 * b1 * f + 4 / 7 * f ** 2
    g0, g2 = b1 + f / 3, 2 * f / 3
    print(f'b1 {b1} f {f} b2 {b2:.3f} bs2 {bs2:.3f}; Kaiser K0 {K0:.3f} K2 {K2:.3f}; g0 {g0:.3f} g2 {g2:.3f}')
    np.set_printoptions(precision=3, suppress=True)
    print('a (rows ell = 0, 2, 4; cols n = 0, 2, 4):\n', a, '\nc:\n', c)
    print('uniform window, isotropic long mode D (D^W = D^M), per unit D, in units of the measured multipole:')
    for n in (0.5, -0.5, -1.5, -2.0):
        R00, R02 = (a[0, 0] + c[0, 0] * n) / K0, (a[0, 1] + c[0, 1] * n) / K0
        R20, R22 = (a[1, 0] + c[1, 0] * n) / K2, (a[1, 1] + c[1, 1] * n) / K2
        print(f'  n_eff {n:+.1f}: R0^(0) {R00:5.2f} R0^(2) {R02:5.2f} R2^(0) {R20:5.2f} R2^(2) {R22:5.2f} | net P0: '
              f'data-randoms {R00 - 2 * g0:+.2f} (slope {(R00 - 2 * g0) / (2 * g0):+.2f}), alpha {R00 - g0:+.2f} (slope {(R00 - g0) / g0:+.2f})')
    print('  (slope = d ln P_hat_0 / d delta_norm for the clustering part of delta_norm; the Poisson part adds -1 x its variance fraction)')
    for bb in (1.8, 2.0, 2.2, 2.4):
        b2b = 0.412 - 2.143 * bb + 0.929 * bb ** 2 + 0.008 * bb ** 3
        aa, cc = response_coefficients(bb, f, b2b, -4 / 7 * (bb - 1))
        K = bb ** 2 + 2 / 3 * bb * f + f ** 2 / 5
        R = (aa[0, 0] + cc[0, 0] * -1.5) / K
        print(f'  b1 {bb}: b2 {b2b:+.2f}  R0^(0)/P0 {R:.2f}  net(data-randoms) {R - 2 * (bb + f / 3):+.2f}  net(alpha) {R - (bb + f / 3):+.2f}')
    return a, c, b2, bs2


def cross_check_kernels(a, c, b1=2.2, f=0.76, b2=0.279, bs2=-0.686):
    k, P = p_lin()
    spl = CubicSpline(np.log(k), P)
    Pf = lambda kk: spl(np.log(np.maximum(kk, 1e-5)))
    gb = galileon_bias(b1, b2, bs2)
    xg, wg = np.polynomial.legendre.leggauss(24)
    phi = 2 * np.pi * np.arange(32) / 32
    mu, nu, ph = np.meshgrid(xg, xg, phi, indexing='ij')
    st, sn = np.sqrt(1 - mu ** 2), np.sqrt(1 - nu ** 2)
    kk = 0.1
    kv = kk * np.stack([st * np.cos(ph), st * np.sin(ph), mu])
    qh = np.stack([sn, 0 * nu, nu])
    worst = 0.0
    for eps in (1e-3, 1e-4):
        R = 0.5 * (squeezed_response(kv, eps * kk * qh, Pf, gb, f) + squeezed_response(kv, -eps * kk * qh, Pf, gb, f)).mean(2)
        for i, l in enumerate((0, 2, 4)):
            Ll = np.polynomial.legendre.Legendre.basis(l)(xg)
            for j, n in enumerate((0, 2, 4)):
                Ln = np.polynomial.legendre.Legendre.basis(n)(xg)
                val = (2 * l + 1) / 2 * (2 * n + 1) / 2 * np.sum(np.outer(wg * Ll, wg * Ln) * R)
                ref = a[i, j] * spl(np.log(kk)) + c[i, j] * spl(np.log(kk), 1)
                worst = max(worst, abs(val - ref) / abs(a[0, 0] * spl(np.log(kk))))
        print(f'squeezed_response (trispectrum.py kernels) vs response_coefficients, eps {eps}: max rel diff {worst:.1e}')


def t0_ir_limit():
    k, P = p_lin()
    PL = LinearPower(k, P)
    bias, f = galileon_bias(2.2, 0.28, -0.686), 0.76
    rng = np.random.default_rng(0)
    u1 = rng.normal(size=(3, 6)); u1 /= np.linalg.norm(u1, axis=0)
    u2 = rng.normal(size=(3, 6)); u2 /= np.linalg.norm(u2, axis=0)
    k2 = 0.2
    print('T0 parallelogram, k2 = 0.2, six random orientations: (snake + star) / (P1^2 P2) as k1 -> 0 (must stay finite)')
    for k1 in (0.02, 0.005, 0.001, 0.0002):
        T = parallelogram(k1 * u1, k2 * u2, PL, bias, f)
        tot = (T['snake'] + T['star']) / (PL(k1) ** 2 * PL(k2))
        print(f'  k1 {k1:7.4f}: total {np.array2string(tot, precision=1)}   |snake| ~ {np.max(np.abs(T["snake"])) / (PL(k1) ** 2 * PL(k2)):.1e}')

    def F2m(a, b):
        ab = np.sum(a * b, 0); ma, mb = np.sum(a * a, 0), np.sum(b * b, 0)
        return 5 / 7 + 0.5 * ab * (1 / ma + 1 / mb) + 2 / 7 * ab ** 2 / (ma * mb)

    def my_T(ks, Pfun):
        import itertools
        Pk = [Pfun(np.sqrt(np.sum(q * q, 0))) for q in ks]
        T = 0
        for a, b in itertools.permutations(range(4), 2):
            for cc in (i for i in range(4) if i not in (a, b)):
                d = [i for i in range(4) if i not in (a, b, cc)][0]
                s = ks[a] + ks[cc]
                if np.all(np.sum(s * s, 0) < 1e-20):
                    continue
                T = T + 2 * F2m(-ks[a], s) * F2m(-ks[b], ks[b] + ks[d]) * Pk[a] * Pk[b] * Pfun(np.sqrt(np.sum(s * s, 0)))
        for d in range(4):
            a, b, c = [i for i in range(4) if i != d]
            T = T + 6 * FG3(-ks[a], -ks[b], -ks[c])[0] * Pk[a] * Pk[b] * Pk[c]
        return T
    m = Bias(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    k1v, k2v = 0.1 * u1, 0.13 * u2
    ref = my_T([k1v, -k1v, k2v, -k2v], PL)
    got = parallelogram(k1v, k2v, PL, m, 0.0)
    print('real-space matter: parallelogram vs independent T2211 + T3111 assembly, max rel diff',
          f'{np.max(np.abs(got["snake"] + got["star"] - ref) / np.abs(ref)):.1e}')


if __name__ == '__main__':
    a, c, b2, bs2 = responses_and_slopes()
    cross_check_kernels(a, c, b2=b2, bs2=bs2)
    t0_ir_limit()
