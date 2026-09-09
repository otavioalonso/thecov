"""Invariances the covariance must satisfy exactly (or, where noted, statistically).

These are the cheapest tests in the suite and the ones most likely to catch a normalisation or
bookkeeping error, because none of them needs a reference calculation: the same code is run twice
on inputs related by a transformation under which the answer is known not to change.

    1. w_A -> c w_A            : exact  (F^A and I_AB both scale, the estimator is invariant)
    2. randoms -> half, alpha -> 2 alpha : statistical (the randoms sample nbar/alpha)
    3. rigid rotation of every catalogue : exact (the covariance is a rotational scalar)
    4. L_max padded with zero multipoles : exact (the extra L terms vanish identically)

Note on determinism: the pair-count subsample is drawn from a seed that depends on the tracer NAME,
so a transformed catalogue must keep the same name for the two runs to use the same random subset.
"""
import numpy as np
import pytest

from pkcov import Tracer, PowerSpectrumModel, GaussianCovariance


# --------------------------------------------------------------------------- helpers
def sphere_randoms(rng, R, n, center=(0.0, 0.0, 1400.0)):
    pts = []
    while sum(len(p) for p in pts) < n:
        x = rng.uniform(-R, R, size=(2 * n, 3))
        pts.append(x[np.sum(x * x, 1) < R * R])
    return np.concatenate(pts)[:n] + np.asarray(center)


def toy_model(names_and_amps):
    k = np.linspace(0.0, 1.0, 200)
    shape = (k / 0.05 + 1e-3) / (1 + (k / 0.05) ** 2.2)
    model = PowerSpectrumModel()
    for pair, amps in names_and_amps.items():
        model.add(pair, {L: (k, a * shape) for L, a in amps.items()})
    return model


COV_OPTS = dict(ells=(0, 2), L_max=2, ds=2.0, ds_pair=10.0, n_sub=1200, n_near=15000,
                s_split=80.0, seed=0)
K_EDGES = np.arange(0.02, 0.13, 0.02)
BLOCKS = [(('A', 'A'), ('A', 'A'), 0, 0), (('A', 'A'), ('A', 'A'), 2, 2),
          (('A', 'A'), ('A', 'B'), 0, 0), (('A', 'B'), ('A', 'B'), 2, 0)]


def two_tracers(rng, R=400.0, n=12000, nA=3e-4, nB=6e-4, wA=None, wB=None, center=(0, 0, 1400.0)):
    V = 4 * np.pi / 3 * R ** 3
    posA, posB = sphere_randoms(rng, R, n, center), sphere_randoms(rng, R, n, center)
    wA = np.ones(len(posA)) if wA is None else wA(posA)
    wB = np.ones(len(posB)) if wB is None else wB(posB)
    A = Tracer('A', {'POSITION': posA, 'WEIGHT': wA, 'NZ': np.full(len(posA), nA)}, nA * V / n)
    B = Tracer('B', {'POSITION': posB, 'WEIGHT': wB, 'NZ': np.full(len(posB), nB)}, nB * V / n)
    return A, B


MODEL = toy_model({('A', 'A'): {0: 2.0e4, 2: 0.9e4}, ('B', 'B'): {0: 0.8e4, 2: 0.3e4},
                   ('A', 'B'): {0: 1.2e4, 2: 0.5e4}})


def blocks_of(cov):
    return {b: cov.block(b[0], b[1], b[2], b[3]) for b in BLOCKS}


# --------------------------------------------------------------------------- 1. weights
def test_weight_rescaling_invariance():
    """w_A -> c w_A leaves every covariance block unchanged.

    F^A scales by c and so does I_AB (through W^AB); the shot-noise window S^A and W^AA both scale
    by c^2, matching I_AA. Any mismatch between the normalisation of Q, of I, and of the s = 0
    anchor would break this.
    """
    c = 3.0
    A1, B1 = two_tracers(np.random.default_rng(21))
    A2, B2 = two_tracers(np.random.default_rng(21), wA=lambda p: np.full(len(p), c))
    cov1 = GaussianCovariance([A1, B1], K_EDGES, **COV_OPTS).set_model(MODEL)
    cov2 = GaussianCovariance([A2, B2], K_EDGES, **COV_OPTS).set_model(MODEL)
    b1, b2 = blocks_of(cov1), blocks_of(cov2)
    for key in BLOCKS:
        assert np.allclose(b1[key], b2[key], rtol=1e-10, atol=1e-12 * np.max(np.abs(b1[key]))), key


# --------------------------------------------------------------------------- 2. random density
def test_random_density_rescaling():
    """Halving the randoms while doubling alpha must leave the covariance unchanged.

    The randoms sample nbar/alpha, so this is the consistency that Tracer only *warns* about. It is
    a statistical identity (a different random subset is used), hence the loose tolerance and the
    comparison restricted to the diagonal. Shot noise is switched off because S^A = (1+alpha) nbar w^2
    depends on alpha explicitly and genuinely changes.
    """
    rng = np.random.default_rng(22)
    R, n, nA = 400.0, 24000, 3e-4
    V = 4 * np.pi / 3 * R ** 3
    pos = sphere_randoms(rng, R, n)
    full = Tracer('A', {'POSITION': pos, 'WEIGHT': np.ones(n), 'NZ': np.full(n, nA)}, nA * V / n)
    idx = rng.choice(n, size=n // 2, replace=False)
    half = Tracer('A', {'POSITION': pos[idx], 'WEIGHT': np.ones(n // 2), 'NZ': np.full(n // 2, nA)},
                  nA * V / (n // 2))
    opts = dict(COV_OPTS, shot_noise=False, n_sub=1200, n_near=10000)
    c1 = GaussianCovariance([full], K_EDGES, **opts).set_model(MODEL)
    c2 = GaussianCovariance([half], K_EDGES, **opts).set_model(MODEL)
    assert np.isclose(c1.I('A', 'A'), c2.I('A', 'A'), rtol=0.02)
    for ell in (0, 2):
        d1 = np.diag(c1.block(('A', 'A'), ('A', 'A'), ell, ell))
        d2 = np.diag(c2.block(('A', 'A'), ('A', 'A'), ell, ell))
        assert np.allclose(d2 / d1, 1.0, atol=0.05), (ell, d2 / d1)


# --------------------------------------------------------------------------- 3. rotation
def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else q[[1, 0, 2]]


def test_rotation_invariance():
    """A rigid rotation of every catalogue leaves the covariance unchanged.

    The covariance is a rotational scalar, so this exercises the tripolar basis, its evaluation in
    the s-hat frame, and the 3j projection end to end -- none of which is otherwise tested at this
    level. Pair separations are preserved exactly, so the only differences are float round-off (and,
    with vanishing probability, a pair sitting within ~1e-13 of an s-bin edge changing bin).
    """
    rng = np.random.default_rng(23)
    A1, B1 = two_tracers(np.random.default_rng(24))
    Rm = random_rotation(rng)
    rot = lambda t: Tracer(t.name, {'POSITION': t.pos @ Rm.T, 'WEIGHT': t.w, 'NZ': t.nbar}, t.alpha)
    cov1 = GaussianCovariance([A1, B1], K_EDGES, **COV_OPTS).set_model(MODEL)
    cov2 = GaussianCovariance([rot(A1), rot(B1)], K_EDGES, **COV_OPTS).set_model(MODEL)
    b1, b2 = blocks_of(cov1), blocks_of(cov2)
    for key in BLOCKS:
        d1, d2 = np.diag(b1[key]), np.diag(b2[key])
        assert np.allclose(d1, d2, rtol=1e-9), (key, d2 / d1)
        # Far off-diagonal elements are small differences of much larger contributions, so they
        # inherit the round-off of sums over ~1e7 pairs accumulated in a different order (the
        # KD-tree neighbour ordering depends on the coordinates). Judge them on the diagonal scale.
        scale = np.sqrt(np.outer(np.abs(d1), np.abs(d1)))
        assert np.max(np.abs(b1[key] - b2[key]) / scale) < 1e-5, (key, b2[key] / b1[key])


# --------------------------------------------------------------------------- 4. L_max padding
def test_Lmax_padding_is_exact():
    """Adding a vanishing P_4 and raising L_max must not change anything.

    The L = 4 terms enter with pbar = 0, so they contribute identically zero: this checks the
    selection-rule enumeration and the L sums for off-by-one or index-reuse errors.
    """
    A, B = two_tracers(np.random.default_rng(25))
    k = np.linspace(0.0, 1.0, 200)
    shape = (k / 0.05 + 1e-3) / (1 + (k / 0.05) ** 2.2)
    m2, m4 = PowerSpectrumModel(), PowerSpectrumModel()
    for pair, amps in {('A', 'A'): (2.0e4, 0.9e4), ('B', 'B'): (0.8e4, 0.3e4), ('A', 'B'): (1.2e4, 0.5e4)}.items():
        base = {0: (k, amps[0] * shape), 2: (k, amps[1] * shape)}
        m2.add(pair, base)
        m4.add(pair, {**base, 4: (k, np.zeros_like(k))})
    opts2 = dict(COV_OPTS, L_max=2)
    opts4 = dict(COV_OPTS, L_max=4)
    c2 = GaussianCovariance([A, B], K_EDGES, **opts2).set_model(m2)
    c4 = GaussianCovariance([A, B], K_EDGES, **opts4).set_model(m4)
    for key in BLOCKS:
        x2 = c2.block(key[0], key[1], key[2], key[3])
        x4 = c4.block(key[0], key[1], key[2], key[3])
        assert np.allclose(x2, x4, rtol=1e-10, atol=1e-12 * np.max(np.abs(x2))), key


# --------------------------------------------------------------------------- 5. weighted split sample
def test_split_sample_identity_weighted():
    """The split-sample identity of test_multitracer, now with spatially varying weights.

    The identity P^TT = (P^AA + P^AB + P^BA + P^BB)/4 holds for ARBITRARY weights, provided the two
    halves carry the same weight function as the full sample (they are the same galaxies) and half
    its density. This is the only test in the suite with a non-uniform w(x).
    """
    rng = np.random.default_rng(26)
    R, N, nbar, dist = 450.0, 30000, 3e-4, 1400.0
    pts = sphere_randoms(rng, R, N, center=(0, 0, dist))
    r = np.linalg.norm(pts, axis=1)
    w = 1.0 / (1.0 + nbar * 1e4 * (1.0 + 0.8 * (r - dist) / R))      # varying, order-unity weights
    V = 4 * np.pi / 3 * R ** 3
    alpha = nbar * V / N
    half = N // 2
    mk = lambda p, ww, nb: {'POSITION': p, 'WEIGHT': ww, 'NZ': np.full(len(p), nb)}
    T = Tracer('T', mk(pts, w, nbar), alpha)
    A = Tracer('A', mk(pts[:half], w[:half], nbar / 2), alpha)
    B = Tracer('B', mk(pts[half:], w[half:], nbar / 2), alpha)

    k = np.linspace(0.0, 1.0, 200)
    shape = (k / 0.05 + 1e-3) / (1 + (k / 0.05) ** 2.2)
    mult = {0: (k, 2e4 * shape), 2: (k, 0.8e4 * shape)}
    model = PowerSpectrumModel()
    for pair in [('T', 'T'), ('A', 'A'), ('A', 'B'), ('B', 'B')]:
        model.add(pair, mult)

    opts = dict(ells=(0, 2), L_max=2, s_max=2 * R, ds=2.0, ds_pair=10.0, shot_noise=True,
                n_sub=2500, n_near=30000, s_split=80.0, seed=0)
    single = GaussianCovariance([T], K_EDGES, **opts).set_model(model)
    multi = GaussianCovariance([A, B], K_EDGES, **opts).set_model(model)
    spectra = [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]
    for l1, l2 in [(0, 0), (2, 2), (0, 2)]:
        ref = single.block(('T', 'T'), ('T', 'T'), l1, l2)
        tot = sum(multi.block(s1, s2, l1, l2) for s1 in spectra for s2 in spectra) / 16.0
        # l1 != l2 blocks are small differences of larger terms, so they carry more pair-count
        # noise; the two sides also use independent random subsamples (different tracer names).
        atol = 0.03 if l1 == l2 else 0.06
        assert np.allclose(np.diag(tot) / np.diag(ref), 1.0, atol=atol), (l1, l2, np.diag(tot) / np.diag(ref))
        assert np.linalg.norm(tot - ref) / np.linalg.norm(ref) < 0.06, (l1, l2)


# --------------------------------------------------------------------------- 6. shot-noise-only limit
def test_shot_noise_only_limit():
    """With P = 0 the covariance is pure Poisson noise: C_ll = 2 (2l+1) S^2 / N_modes, C_02 = 0.

    Isolates the S-S window family, which is otherwise only ever tested inside a total where it
    happens to dominate. The cube's leakage factor is taken from the same Monte-Carlo used by the
    box test; it applies cleanly here because the "spectrum" is exactly white.
    """
    from tests.test_box import leakage_fractions

    L_box, nbar, alpha, dist = 1000.0, 1e-4, 0.5, 2e5
    N = int(round(nbar * L_box ** 3 / alpha))
    rng = np.random.default_rng(27)
    pos = rng.uniform(-L_box / 2, L_box / 2, size=(N, 3))
    pos[:, 0] += dist
    A = Tracer('A', {'POSITION': pos, 'WEIGHT': np.ones(N), 'NZ': np.full(N, nbar)}, alpha)
    k = np.linspace(0.0, 1.0, 20)
    model = PowerSpectrumModel()
    model.add(('A', 'A'), {0: (k, np.zeros_like(k))})
    k_edges = np.arange(0.0, 0.205, 0.02)
    cov = GaussianCovariance([A], k_edges, ells=(0, 2), L_max=0, ds=2.0, ds_pair=10.0,
                             shot_noise=True, n_sub=2500, n_near=80000, s_split=80.0,
                             seed=1).set_model(model)
    S = (1 + alpha) / nbar
    Vi = (k_edges[1:] ** 3 - k_edges[:-1] ** 3) / 3.0 / (2 * np.pi ** 2)
    N_modes = L_box ** 3 * Vi
    retained, _ = leakage_fractions(L_box, k_edges)
    sel = slice(2, -1)
    for ell in (0, 2):
        ref = 2 * (2 * ell + 1) * S ** 2 / N_modes
        got = np.diag(cov.block(('A', 'A'), ('A', 'A'), ell, ell))
        assert np.allclose((got / (ref * retained))[sel], 1.0, atol=0.05), (ell, got / (ref * retained))
    c02 = np.diag(cov.block(('A', 'A'), ('A', 'A'), 0, 2))
    scale = np.sqrt(np.diag(cov.block(('A', 'A'), ('A', 'A'), 0, 0))
                    * np.diag(cov.block(('A', 'A'), ('A', 'A'), 2, 2)))
    assert np.all(np.abs(c02 / scale)[sel] < 0.05), c02 / scale
