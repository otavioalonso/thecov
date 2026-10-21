"""Tracer(name, positions, weights, alpha, nbar=..., window_weighting=...): the two ingredients of W = nbar <w>."""
import warnings

import numpy as np
import pytest

from thecov import Tracer
from thecov.tracers import footprint_solid_angle, randoms_number_density


def cap_randoms(n, cos_min=0.8, chi=(1000.0, 1400.0), seed=0):
    """Randoms uniform in a spherical-cap shell; returns positions and the exact number density."""
    rng = np.random.default_rng(seed)
    u, phi = rng.uniform(cos_min, 1, n), rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(chi[0] ** 3, chi[1] ** 3, n) ** (1 / 3)
    s = np.sqrt(1 - u ** 2)
    pos = r[:, None] * np.stack([s * np.cos(phi), s * np.sin(phi), u], axis=1)
    return pos, n / (2 * np.pi * (1 - cos_min) * (chi[1] ** 3 - chi[0] ** 3) / 3)


@pytest.mark.parametrize('cos_min', [0.99, 0.8, -0.2])
def test_footprint_solid_angle(cos_min):
    pos, _ = cap_randoms(400_000, cos_min)
    omega = footprint_solid_angle(pos / np.linalg.norm(pos, axis=1)[:, None])
    assert omega / (2 * np.pi * (1 - cos_min)) == pytest.approx(1, abs=3e-3)


def test_randoms_number_density():
    pos, rho = cap_randoms(400_000)
    r = randoms_number_density(pos)
    assert np.median(r) / rho == pytest.approx(1, abs=5e-3)
    assert np.std(r / rho) < 0.02                     # shot noise of ~10 Mpc/h shells


def test_nbar_auto_and_array():
    pos, rho = cap_randoms(200_000)
    w, alpha = np.ones(len(pos)), 0.05
    auto = Tracer('A', pos, w, alpha)
    assert np.median(auto.nbar) / (alpha * rho) == pytest.approx(1, abs=5e-3)
    given = Tracer('G', pos, w, alpha, nbar=np.full(len(pos), alpha * rho))
    assert np.all(given.nbar == alpha * rho)
    call = Tracer('C', pos, w, alpha, nbar=lambda x: np.full(len(x), alpha * rho))
    assert np.all(call.nbar == alpha * rho)


def test_window_weighting():
    pos, rho = cap_randoms(200_000)
    nbar = np.full(len(pos), 0.05 * rho)
    ones = np.ones(len(pos))
    # constant weights: the two agree
    a = Tracer('A', pos, ones, 0.05, nbar=nbar).weighted_density
    b = Tracer('B', pos, ones, 0.05, nbar=nbar, window_weighting='own').weighted_density
    assert np.allclose(a, b, rtol=1e-14)
    # per-object weights: 'nearest' is the local mean (<w>^2 window), 'own' keeps the scatter (<w^2>)
    w = np.random.default_rng(3).uniform(0.5, 1.5, len(pos))
    near = Tracer('N', pos, w, 0.05, nbar=nbar).weighted_density / nbar
    own = Tracer('O', pos, w, 0.05, nbar=nbar, window_weighting='own').weighted_density / nbar
    assert np.mean(near) == pytest.approx(1, abs=2e-3) and np.std(near) < 0.07     # ~ std(w) / sqrt(32) = 0.05
    assert np.mean(own ** 2) / np.mean(near ** 2) == pytest.approx(1 + np.var(w), rel=0.02)
    with pytest.raises(ValueError):
        Tracer('X', pos, w, 0.05, nbar=nbar, window_weighting='neighbours')


def test_legacy_dict_interface():
    pos, rho = cap_randoms(50_000)
    w = np.random.default_rng(4).uniform(0.5, 1.5, len(pos))
    nz = np.full(len(pos), 0.05 * rho)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        old = Tracer('L', {'POSITION': pos, 'WEIGHT': w, 'NZ': nz}, 0.05)
        old_nw = Tracer('M', {'POSITION': pos, 'WEIGHT': w, 'NW': 2 * nz}, alpha=0.05)
    assert any(issubclass(c.category, DeprecationWarning) for c in caught)
    new = Tracer('L', pos, w, 0.05, nbar=nz, window_weighting='own')
    assert np.array_equal(old.weighted_density, new.weighted_density) and np.array_equal(old.nbar, new.nbar)
    assert np.array_equal(old_nw.weighted_density, 2 * nz)
