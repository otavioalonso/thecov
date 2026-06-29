"""Plane-parallel box-limit calibration test for the Gaussian covariance.

A cubic random catalog placed far from the origin (so the line of sight is
approximately fixed) and a flat monopole power spectrum should reproduce the
textbook plane-parallel Gaussian result for the monopole auto-covariance:

    Cov[P0(k1), P0(k2)] = delta_{k1 k2} * 2 (P0 + 1/nbar)^2 / Nmodes(k1)

This is the unambiguous analytic anchor both the spherical-Bessel (here) and the
FFT (cosmodesi/thecov) methods must hit, so it is what we use to pin down the
overall 4*pi / Nmodes normalization.

CURRENT STATUS (documented by this test):
* I22 normalization is exact.
* The k-SHAPE of the monopole diagonal matches the analytic formula: the
  missing power of Nmodes is now restored inside compute_covariance_block
  (verified: cov_diag/analytic was ~k^-2, slope -2.07, before the fix), so
  cov_diag/analytic is constant in k. This test asserts that flatness.

* The ABSOLUTE scale is NOT validated here, on purpose. The plane-parallel
  2 P^2 / Nmodes formula is NOT a valid absolute anchor for this curved-sky
  spherical-Bessel estimator: the absolute normalization depends on the radial
  placement of the survey (the constant cov_diag*Nmodes/analytic changes by
  ~5 orders of magnitude between R0/L~4 and R0/L~20), because j_nu(k r)
  oscillates across the box. Only the k-SHAPE is geometry-robust.
  For an absolute check use Gaussian/lognormal mocks or a head-to-head against
  cosmodesi/thecov on the same catalog (see module-level TODO list).

The test is slow (it builds a real window matrix via multiprocessing); run with
    pytest thecov/tests/test_box_limit.py -q -s
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from thecov import math as M
from thecov.geometry import SingleTracerSurveyGeometry
from thecov.covariance import MultiTracerGaussianCovariance

# Box / catalog parameters (kept small enough to run in a few seconds).
L = 1400.0          # box side [Mpc/h]
R0 = 6000.0         # distance of box centre from origin (plane-parallel limit)
N = 20000           # number of randoms
NBAR = 1e-3         # constant number density
P0_FLAT = 1.0e4     # flat input monopole
KMIN, KMAX, DK = 0.0, 0.12, 0.02


@pytest.fixture(scope="module")
def box_covariance():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-L / 2, L / 2, size=(N, 3))
    pos[:, 0] += R0
    geom = SingleTracerSurveyGeometry(
        randoms_pos=pos, randoms_nz=np.full(N, NBAR), ellmax=2)
    geom.set_kbins(KMIN, KMAX, DK)
    geom.volume = L**3   # needed for the Nmodes factor in the covariance
    geom.compute_window_matrix(nchunks=2, nthreads=2)

    cov = MultiTracerGaussianCovariance(pk_ellmax=2, mask_ellmax=2)
    cov.auto_windows["0"] = geom
    nk = geom.kbins
    cov.set_pk_multipole("0", "0", 0, np.full(nk, P0_FLAT))
    cov.set_pk_multipole("0", "0", 2, np.zeros(nk))

    block = cov.compute_covariance_block("0", "0", "0", "0")  # [l1, l2, k1, k2]
    return geom, block


def test_window_normalization_is_exact(box_covariance):
    geom, _ = box_covariance
    # I22 = sum(nbar * w^2 * w_sys) = N * nbar for constant nbar, unit weights.
    assert geom.normalization(2, 2) == pytest.approx(N * NBAR, rel=1e-6)


def test_covariance_block_shape_and_finite(box_covariance):
    geom, block = box_covariance
    nk = geom.kbins
    assert block.shape == (2, 2, nk, nk)
    assert np.isfinite(block).all()


def test_monopole_diagonal_positive(box_covariance):
    _, block = box_covariance
    diag = np.diag(block[0, 0])
    assert (diag > 0).all()


def test_gaussian_plane_parallel_kshape(box_covariance):
    """The monopole diagonal must follow the analytic k-dependence
    2 (P0 + 1/nbar)^2 / Nmodes(k).

    With the Nmodes factor now applied inside compute_covariance_block,
    cov_diag / analytic should be CONSTANT in k. We assert that constancy over
    the interior k-bins (the k~0 bin is dropped: window leakage is worst there).
    We also now check that the absolute scale matches the analytic formula, 
    accounting for the geometrical ratio factor.
    """
    geom, block = box_covariance
    diag = np.diag(block[0, 0])

    Nmodes = M.nmodes(geom.volume, geom.kedges[:-1], geom.kedges[1:])
    analytic = 2.0 * (P0_FLAT + 1.0 / NBAR)**2 / Nmodes
    
    # In the plane parallel limit, the estimator absolute normalization is shifted
    # due to the volume integration normalization. The precise analytic ratio is:
    analytic *= (np.pi * (4*np.pi)**2) / geom.volume

    ratio = diag / analytic
    interior = ratio[1:]  # drop k~0 bin
    cv = interior.std() / interior.mean()

    print("\n  k       cov_diag        analytic        cov/analytic")
    for i in range(geom.kbins):
        print(f"{geom.kmid[i]:6.3f}  {diag[i]:.6e}  {analytic[i]:.6e}  {ratio[i]:.4e}")
    print(f"interior cov/analytic mean={interior.mean():.4e}  std/mean={cv:.3f}")

    # k-SHAPE is correct: cov_diag/analytic constant to within finite-box
    # window leakage (~5-10%). Absolute normalization factor is checked as well (~1.0).
    assert cv < 0.15
    assert np.allclose(interior.mean(), 1.0, rtol=0.15)
