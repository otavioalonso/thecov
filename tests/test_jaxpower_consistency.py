"""The native mock estimator must reproduce jaxpower (the estimator thecov is used with) on a common
catalogue: same alpha, shot noise, line of sight and normalisation conventions."""
import numpy as np
import pytest

pytest.importorskip("jaxpower")


def test_native_estimator_matches_jaxpower():
    from mocks.survey import Footprint, make_grid, Catalogues, make_mock
    from mocks.field import GaussianField
    from mocks.run_validation import pk_lin, BIAS, GROWTH
    from mocks.estimator import MultipoleFields, ShellBinner, cross_multipole, shot_noise, realised_alpha
    from mocks.jaxpower_estimator import JaxpowerEstimator

    fp = Footprint()
    g = make_grid(fp, 64, 2.0)
    cat = Catalogues(fp, g, n_random_factor=5.0, rng=np.random.default_rng(1))
    rng = np.random.default_rng(2)
    cats = make_mock(cat, GaussianField(g, lambda k: pk_lin(k, 0.6), rng), BIAS, {}, GROWTH, rng, rsd=True)
    gal, gw = {}, {}
    for t in cats:
        w = cat.weights_at(t, cats[t])
        gal[t], gw[t] = cats[t][w > 0], w[w > 0]
    k_edges = np.arange(0.01, 0.075, 0.01)
    ells, spectra = (0, 2), [('A', 'A'), ('A', 'B'), ('B', 'B')]
    I = {sp: cat.I(*sp) for sp in spectra}

    binner = ShellBinner(g, k_edges)
    alpha = {t: realised_alpha(gw[t], cat.w_ran[t]) for t in cats}
    f = {t: MultipoleFields(g, gal[t], gw[t], cat.randoms[t], cat.w_ran[t], alpha[t], ells=ells,
                            scheme='tsc', interlace=True) for t in cats}
    native = []
    for X, Y in spectra:
        for ell in ells:
            P = cross_multipole(f[X], f[Y], ell, binner, I[(X, Y)])
            if X == Y and ell == 0:
                P = P - shot_noise(alpha[X], cat.w_ran[X], I[(X, Y)], gal_w_A=gw[X])
            native.append(P)
    native = np.concatenate(native)
    jax, norms = JaxpowerEstimator(g, k_edges, ells)(gal, gw, cat.randoms, cat.w_ran, spectra, fixed_norm=I)

    nb = len(k_edges) - 1
    for i in range(len(spectra)):
        P0 = np.abs(native[2 * i * nb:(2 * i + 1) * nb])
        for j in range(len(ells)):
            s = slice((2 * i + j) * nb, (2 * i + j + 1) * nb)
            tol = 1e-3 if ells[j] == 0 else 3e-2       # l = 2: line-of-sight evaluation on the grid
            assert np.all(np.abs(jax[s] - native[s]) < tol * P0), (spectra[i], ells[j], (jax[s] - native[s]) / P0)
