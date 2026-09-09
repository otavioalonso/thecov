r"""The tripolar window functions Q_{Lam1 Lam2 Lam}(s) against direct numerical quadrature.

Until now Q has only ever been checked for (Lam1, Lam2, Lam) = (0,0,0) and (0,0,2), i.e. the
isotropic part; the anisotropic components -- the ones that carry the ell = 2 and ell = 4 signal and
the whole varying-line-of-sight effect -- were tested only through ratios in the distant-observer
limit. This module closes that gap for the hardest geometry available in closed form: a uniform
sphere CENTRED ON THE OBSERVER, where the line of sight sweeps the full sky.

Reference calculation
---------------------
For a uniform sphere of radius R centred at the origin the window is omega(x) = nbar^2 inside, so

    Q(s) = int dOmega_s int d^3x  omega(x) omega(x + s) S_{Lam1 Lam2 Lam}(x^, x'^, s^).

The inner integral is independent of the direction of s (rotating s maps the sphere onto itself and
S is invariant under a simultaneous rotation of its three arguments), so the dOmega_s integral gives
a factor 4 pi and we may fix s = s z^. The remaining configuration is azimuthally symmetric about
z^, giving another factor 2 pi, and the constraint |x + s z^| < R fixes the upper limit of mu:

    Q(s) = 8 pi^2 nbar^4 int_0^R r^2 dr int_{-1}^{mu_max(r)} dmu  S(x^, x'^, z^),
    mu_max = clip((R^2 - r^2 - s^2) / (2 r s), -1, 1),
    x = r (sqrt(1-mu^2), 0, mu),   x' = x + s z^.

S is evaluated with `tripolar_direct` (the explicit 3j sum over spherical harmonics), which is
independent of the s^-frame formula used by the pair counter, so the two routes share nothing but
the Wigner symbols.
"""
import numpy as np
import pytest

from pkcov import Tracer
from pkcov.harmonics import tripolar_direct, unit_vectors
from pkcov.tracers import Window
from pkcov.windows import TripolarWindow
from pkcov.wigner import FOUR_PI

R_SPHERE = 500.0
NBAR = 2e-4
TRIPLES = [(0, 0, 0), (2, 0, 2), (0, 2, 2), (2, 2, 0), (2, 2, 2), (2, 2, 4), (4, 2, 2), (4, 4, 0)]


def q_quadrature(s_values, triples, R=R_SPHERE, nbar=NBAR, n_r=400, n_mu=400):
    """Q_{Lam1 Lam2 Lam}(s) for an observer-centred uniform sphere, by 2D Gauss-Legendre."""
    xr, wr = np.polynomial.legendre.leggauss(n_r)
    xm, wm = np.polynomial.legendre.leggauss(n_mu)
    r = 0.5 * R * (xr + 1.0)
    wr = 0.5 * R * wr * r ** 2
    out = {t: np.zeros(len(s_values)) for t in triples}
    for i, s in enumerate(np.atleast_1d(s_values)):
        if s >= 2 * R:
            continue
        mu_max = np.clip((R ** 2 - r ** 2 - s ** 2) / np.where(r * s > 0, 2 * r * s, 1.0), -1.0, 1.0)
        if s == 0:
            mu_max = np.ones_like(r)
        half = 0.5 * (mu_max + 1.0)
        mu = (-1.0 + half[:, None] * (xm[None, :] + 1.0))            # (n_r, n_mu)
        wmu = half[:, None] * wm[None, :]
        sin = np.sqrt(np.clip(1 - mu ** 2, 0, None))
        rr = r[:, None]
        x = np.stack([rr * sin, np.zeros_like(mu), rr * mu], axis=-1)
        xp = x + np.array([0.0, 0.0, s])
        xhat = unit_vectors(x)
        xphat = unit_vectors(xp)
        shat = np.broadcast_to(np.array([0.0, 0.0, 1.0]), x.shape)
        for t in triples:
            S = tripolar_direct(t[0], t[1], t[2], xhat, xphat, shat).real
            out[t][i] = 8 * np.pi ** 2 * nbar ** 4 * np.sum(wr[:, None] * wmu * S)
    return out


def sphere_tracer(n=60000, R=R_SPHERE, nbar=NBAR, seed=41):
    rng = np.random.default_rng(seed)
    pts = []
    while sum(len(p) for p in pts) < n:
        x = rng.uniform(-R, R, size=(2 * n, 3))
        pts.append(x[np.sum(x * x, 1) < R * R])
    pos = np.concatenate(pts)[:n]
    V = 4 * np.pi / 3 * R ** 3
    return Tracer('S', {'POSITION': pos, 'WEIGHT': np.ones(n), 'NZ': np.full(n, nbar)}, nbar * V / n)


# --------------------------------------------------------------------------------- tests
def test_q0_anchor_matches_quadrature():
    """The analytic s = 0 anchor used to pin the radial spline.

    Q(0) = sqrt(4 pi) (-1)^Lam1 sqrt(2 Lam1 + 1) / (4 pi) * int omega omega' for Lam = 0 and
    Lam1 = Lam2, and zero otherwise. This formula was derived by hand and never verified.
    """
    quad = q_quadrature(np.array([0.0]), TRIPLES)
    ov = NBAR ** 4 * 4 * np.pi / 3 * R_SPHERE ** 3          # int omega omega' at s = 0
    for (L1, L2, L) in TRIPLES:
        expect = ((-1) ** L1 * np.sqrt(FOUR_PI * (2 * L1 + 1)) / FOUR_PI * ov
                  if (L == 0 and L1 == L2) else 0.0)
        got = quad[(L1, L2, L)][0]
        scale = max(abs(expect), 1e-6 * abs(ov))
        assert abs(got - expect) < 1e-3 * scale, ((L1, L2, L), got, expect)


def test_q000_quadrature_matches_analytic_overlap():
    """Sanity check on the reference itself: the (0,0,0) component is the sphere self-overlap."""
    s = np.array([0.0, 100.0, 400.0, 800.0])
    quad = q_quadrature(s, [(0, 0, 0)])
    Vov = np.pi / 12 * (4 * R_SPHERE + s) * (2 * R_SPHERE - s) ** 2
    expect = FOUR_PI ** (-1.5) * FOUR_PI * NBAR ** 4 * Vov
    assert np.allclose(quad[(0, 0, 0)], expect, rtol=2e-3), (quad[(0, 0, 0)], expect)


@pytest.mark.slow
def test_pair_counts_match_quadrature_anisotropic():
    """The pair counter reproduces the anisotropic tripolar components.

    This isolates the pair-count estimator from the covariance assembly, the way the analytic-Q
    injection did for ell = 0, and does it for an observer-centred sphere where the line of sight
    varies over the whole sky. Bins with few pairs and the region near s = 2R (where Q -> 0) are
    excluded; the tolerance is set by pair-count noise, not by the method.
    """
    tr = sphere_tracer()
    w = Window('W', tr, tr)
    s_edges = np.arange(0.0, 2 * R_SPHERE + 25.0, 25.0)
    tw = TripolarWindow(w, w, TRIPLES, s_edges, n_sub=6000, n_near=60000, s_split=80.0).compute()
    s = tw.s_centers
    quad = q_quadrature(s, TRIPLES)
    ref_scale = np.max(np.abs(quad[(0, 0, 0)]))
    good = (tw.npairs >= 200) & (s < 1.6 * R_SPHERE) & (s > 30.0)
    assert good.sum() > 10
    for t in TRIPLES:
        got, ref = tw.Q[t][good], quad[t][good]
        err = np.max(np.abs(got - ref)) / ref_scale
        assert err < 0.03, (t, err, got / np.where(np.abs(ref) > 0, ref, 1.0))


@pytest.mark.slow
def test_pair_counts_interpolation_matches_quadrature():
    """The spline served to the covariance (anchored at s = 0) also has to track the reference."""
    tr = sphere_tracer(n=40000, seed=42)
    w = Window('W', tr, tr)
    s_edges = np.arange(0.0, 2 * R_SPHERE + 25.0, 25.0)
    triples = [(0, 0, 0), (2, 2, 0), (2, 2, 2)]
    tw = TripolarWindow(w, w, triples, s_edges, n_sub=5000, n_near=50000, s_split=80.0).compute()
    s = np.arange(2.0, 1.5 * R_SPHERE, 4.0)
    quad = q_quadrature(s, triples)
    ref_scale = np.max(np.abs(quad[(0, 0, 0)]))
    for t in triples:
        got = tw(t[0], t[1], t[2], s)
        assert np.max(np.abs(got - quad[t])) / ref_scale < 0.05, t
