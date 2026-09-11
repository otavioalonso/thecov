"""Self-tests of the mock machinery, so that a failure of the end-to-end validation can be
attributed to thecov rather than to the mocks.

The estimator is checked where the answer is known exactly: a uniform periodic box far from the
observer, where the window is trivial and the measured multipoles must reproduce the input Kaiser
spectrum. The subtle point verified here is the cell window -- galaxies placed uniformly inside
their cell realise the CELL-AVERAGED density, which suppresses the clustering by T(k); the field
generator pre-divides by T to cancel it, and the RSD displacement is taken from the galaxy's own
cell so that it carries the same single power of T.
"""
import numpy as np
import pytest

from mocks.estimator import MultipoleFields, ShellBinner, cross_multipole, shot_noise
from mocks.field import Grid, GaussianField, cell_window
from mocks.survey import sample_points, radial_rsd


L_BOX, N_GRID = 1000.0, 64
K_EDGES = np.arange(0.01, 0.105, 0.01)


def box_grid(offset=0.0):
    return Grid(np.array([-L_BOX / 2, -L_BOX / 2, -L_BOX / 2 + offset]), L_BOX, N_GRID)


def pk(k, A=4.0e3, k0=0.05):
    """Deliberately low amplitude: the tests use biases up to b = 2, and Poisson sampling clips
    1 + b delta at zero, which removes power. sigma(b delta) is asserted below to stay small."""
    k = np.asarray(k, dtype=float)
    return A * (k / k0) / (1 + (k / k0) ** 2) ** 2


def shell_average_model(grid, binner, model_fn):
    """The model averaged over the grid modes of each bin, which is what the estimator measures."""
    k = grid.knorm().ravel()
    vals = np.where(k > 0, model_fn(np.where(k > 0, k, 1.0)), 0.0)
    return binner.average(vals.reshape(grid.knorm().shape))


def test_amplitude_is_safe_against_clipping():
    """1 + b delta must stay positive for the biases used here, or the mocks lose power."""
    g = box_grid()
    d = GaussianField(g, pk, np.random.default_rng(0)).delta()
    for b in (1.0, 1.8, 2.0):
        sigma = b * float(d.std())
        frac = float(np.mean(1 + b * d < 0))
        assert sigma < 0.35, (b, sigma)
        assert frac < 3e-3, (b, frac)


def test_cell_window_normalisation():
    g = box_grid()
    T = cell_window(g)
    assert np.isclose(T[0, 0, 0], 1.0)
    assert np.all(T > 0.25)                      # sinc(1/2)^3 = 0.258, never zero on the grid
    assert T.shape == (g.N, g.N, g.N // 2 + 1)


def test_field_variance():
    """Var[delta] must equal the sum of P over the grid modes divided by the box volume."""
    g = box_grid()
    rng = np.random.default_rng(0)
    f = GaussianField(g, pk, rng, deconvolve_cell=False)
    d = f.delta()
    k = g.knorm()
    mult = np.full(k.shape, 2.0)
    mult[:, :, 0] = 1.0
    mult[:, :, -1] = 1.0
    inside = (k > 0) & (k <= f.k_cut)
    expect = np.sum(mult[inside] * pk(k[inside])) / g.V_box
    assert np.isclose(d.var(), expect, rtol=0.05), (d.var(), expect)


def test_shot_noise_constant():
    """The subtracted constant must equal (1 + alpha) / nbar for uniform nbar and unit weights."""
    g = box_grid()
    rng = np.random.default_rng(1)
    nbar, nran = 3e-4, 20
    ran = sample_points(np.full((g.N,) * 3, nbar * g.V_cell * nran), g, rng)
    I = nbar ** 2 * g.V_box
    alpha = 1.0 / nran
    S = shot_noise(alpha, np.ones(len(ran)), I)
    assert np.isclose(S, (1 + alpha) / nbar, rtol=0.02), (S, (1 + alpha) / nbar)


@pytest.mark.slow
def test_monopole_recovered_in_periodic_box():
    """Mean of P_0 over realisations matches the input spectrum with no window."""
    g = box_grid()
    binner = ShellBinner(g, K_EDGES)
    nbar, nran, nrel = 3e-4, 20, 12
    I = nbar ** 2 * g.V_box
    alpha = 1.0 / nran
    acc = []
    for r in range(nrel):
        rng = np.random.default_rng(100 + r)
        f = GaussianField(g, pk, rng)
        d = f.delta()
        lam = np.full((g.N,) * 3, nbar * g.V_cell)
        gal = sample_points(lam * (1 + d), g, rng)
        ran = sample_points(lam * nran, g, rng)
        mf = MultipoleFields(g, gal, np.ones(len(gal)), ran, np.ones(len(ran)), alpha, ells=(0,))
        acc.append(cross_multipole(mf, mf, 0, binner, I) - shot_noise(alpha, np.ones(len(ran)), I))
    acc = np.array(acc)
    ref = shell_average_model(g, binner, pk)
    ratio = acc.mean(0) / ref
    err = acc.std(0) / np.sqrt(nrel) / ref
    pull = (ratio - 1) / np.maximum(err, 1e-6)
    assert np.abs(np.mean(ratio[2:] - 1)) < 0.04, ratio
    assert np.abs(np.mean(pull[2:])) < 3.0, pull


@pytest.mark.slow
def test_kaiser_multipoles_recovered_far_from_observer():
    """With the box pushed far along z the radial line of sight is nearly global, so the measured
    P_0 and P_2 must reproduce the Kaiser prediction for a bias b and growth rate f."""
    b, f_growth, offset = 1.8, 0.8, 60000.0
    g = box_grid(offset=offset)
    binner = ShellBinner(g, K_EDGES)
    nbar, nran, nrel = 5e-4, 20, 16
    I = nbar ** 2 * g.V_box
    alpha = 1.0 / nran
    acc = {0: [], 2: []}
    for r in range(nrel):
        rng = np.random.default_rng(200 + r)
        field = GaussianField(g, pk, rng)
        d = field.delta()
        psi = [field.displacement(i) for i in range(3)]
        lam = np.full((g.N,) * 3, nbar * g.V_cell)
        gal, cells = sample_points(lam * (1 + b * d), g, rng, return_cells=True)
        gal = radial_rsd(gal, psi, g, f_growth, cells=cells)
        ran = sample_points(lam * nran, g, rng)
        mf = MultipoleFields(g, gal, np.ones(len(gal)), ran, np.ones(len(ran)), alpha, ells=(0, 2))
        S = shot_noise(alpha, np.ones(len(ran)), I)
        acc[0].append(cross_multipole(mf, mf, 0, binner, I) - S)
        acc[2].append(cross_multipole(mf, mf, 2, binner, I))
    beta = f_growth / b
    fac = {0: b ** 2 * (1 + 2 * beta / 3 + beta ** 2 / 5), 2: b ** 2 * (4 * beta / 3 + 4 * beta ** 2 / 7)}
    # The monopole must be recovered tightly. The quadrupole of Zel'dovich-displaced Poisson points
    # matches linear Kaiser only to a few per cent -- the displacement is applied to a discrete
    # sample, so there are second-order RSD terms -- and it is far noisier per realisation, so it
    # gets a loose tolerance. This is a property of the mocks, not of the estimator: the covariance
    # comparison uses the mocks' own multipoles, not the linear prediction.
    tol = {0: 0.05, 2: 0.15}
    for ell in (0, 2):
        a = np.array(acc[ell])
        ref = fac[ell] * shell_average_model(g, binner, pk)
        ratio = a.mean(0)[2:] / ref[2:]
        assert np.abs(np.mean(ratio - 1)) < tol[ell], (ell, ratio)


@pytest.mark.slow
def test_cross_spectrum_has_no_shot_noise():
    """Two tracers Poisson-sampled independently from the same field: the cross monopole must sit on
    b_A b_B P with no additive constant.

    The two tracers must be given INDEPENDENT random catalogues. Sharing one makes the -alpha n_r
    piece of the FKP field common to both, so its Poisson noise survives in the cross spectrum as a
    spurious constant alpha/nbar -- 5 % of P at k = 0.02 here and 18 % by k = 0.1. Each tracer has
    its own randoms in the real pipeline (see survey.Catalogues), so this is a property of the test
    set-up, but it is an easy mistake to make with real catalogues too."""
    g = box_grid()
    binner = ShellBinner(g, K_EDGES)
    bA, bB, nbar, nran, nrel = 2.0, 1.2, 4e-4, 20, 12
    I = nbar ** 2 * g.V_box
    alpha = 1.0 / nran
    acc = []
    for r in range(nrel):
        rng = np.random.default_rng(300 + r)
        d = GaussianField(g, pk, rng).delta()
        lam = np.full((g.N,) * 3, nbar * g.V_cell)
        mfs = {}
        for name, bias in (('A', bA), ('B', bB)):
            gal = sample_points(lam * (1 + bias * d), g, rng)
            ran = sample_points(lam * nran, g, rng)          # independent randoms per tracer
            mfs[name] = MultipoleFields(g, gal, np.ones(len(gal)), ran, np.ones(len(ran)), alpha, ells=(0,))
        acc.append(cross_multipole(mfs['A'], mfs['B'], 0, binner, I))
    acc = np.array(acc)
    ref = bA * bB * shell_average_model(g, binner, pk)
    ratio = acc.mean(0)[2:] / ref[2:]
    assert np.abs(np.mean(ratio - 1)) < 0.05, ratio


@pytest.mark.slow
def test_footprint_and_catalogues_are_consistent():
    """The random catalogue must reproduce the intended nbar and the normalisation I_AB."""
    from mocks.survey import Catalogues, Footprint, make_grid
    fp = Footprint()
    grid = make_grid(fp, 96, box_factor=2.0)
    cat = Catalogues(fp, grid, n_random_factor=8.0, rng=np.random.default_rng(7))
    for t in ('A', 'B'):
        n_from_ran = cat.alpha[t] * len(cat.randoms[t])
        assert np.isclose(n_from_ran, cat.n_gal_expected[t], rtol=1e-9)
    # I_AB from the randoms (as thecov computes it) vs the grid integral
    from thecov import Tracer
    from thecov.tracers import Window
    trs = {}
    for t in ('A', 'B'):
        rnd, alpha = cat.thecov_randoms(t)
        trs[t] = Tracer(t, rnd, alpha)
    for (X, Y) in [('A', 'A'), ('B', 'B'), ('A', 'B')]:
        I_ran = Window('W', trs[X], trs[Y]).integral()
        assert np.isclose(I_ran, cat.I(X, Y), rtol=0.05), (X, Y, I_ran / cat.I(X, Y))
