r"""Shell kernels and the radial assembly of the class-1 term (thecov.bispectrum) against independent references.

* vertex_coefficients (vectorised over node pairs) equals bispectrum_algebra.bipolar_coefficients.
* Tapered periodic box: with Q_n(r1, r2) = Q_n^box exp(-(r1^2 + r2^2)/2 sigma^2) only lam = 0 contributes, the leg
  integrals reduce to smoothed shell indicators I_i(p) (1D integrals that converge fast in r_max), and the assembled
  C[i, j, i', j'] must equal V << I_i(p1) I_j(p2) << conj(S~_ell) S~_ell' P P P >>_Omega >>_{i' j'}
  (desi_validation/bispectrum_dev/radial_check.py).  The un-tapered box identity converges only as 1/r_max (the
  bin-averaged Bessel functions of adjacent top-hat shells overlap logarithmically); in a survey the window plays the
  role of the taper.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from thecov.bispectrum import BispectrumShellKernels, Class1Assembler, vertex_coefficients
from thecov.bispectrum_algebra import BispectrumCoupling, bipolar_coefficients, box_window_value

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'desi_validation' / 'bispectrum_dev'))
from angular_check import unit  # noqa: E402
from radial_check import tapered_reference  # noqa: E402

K0 = 0.1
PL = {0: lambda k: 1.0 + (k / K0) ** 2, 2: lambda k: 0.6 * (k / K0) ** 2, 4: lambda k: 0.2 * (k / K0) ** 4}
P_BAO = lambda k: 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6) * (1 + 0.1 * np.sin(k * 105))  # noqa: E731


def test_vertex_vectorised():
    rng = np.random.default_rng(0)
    p1, p2 = rng.uniform(0.05, 0.3, 20), rng.uniform(0.05, 0.3, 20)
    for (L3, a, b) in [(0, 3, 3), (2, 4, 2), (4, 5, 3), (2, 1, 1)]:
        v = vertex_coefficients(L3, a, b, p1, p2, P_BAO)
        ref = np.array([float(bipolar_coefficients(L3, a, b, x, y, P_BAO, n_quad=64)) for x, y in zip(p1, p2)])
        assert np.allclose(v, ref, rtol=1e-12, atol=1e-12 * np.max(np.abs(ref)))


def _setup(sigma=100.0, rmax=700.0, dr=2.0):
    r = np.arange(0.5 * dr, rmax, dr)
    K = BispectrumShellKernels(np.array([0.05, 0.1, 0.15, 0.2]), r)
    taper = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * sigma ** 2))

    def windows(n):
        h = box_window_value(n, 1.0)
        return None if h == 0.0 else h * taper
    return K, windows, sigma


def _check(ell, ellp, PLuse, nq_ref, nth, nph, tol):
    K, windows, sigma = _setup()
    asm = Class1Assembler(K, BispectrumCoupling(), lam_w=0)
    spec = ('P', PLuse)
    C = asm.assemble(ell, ellp, spec, spec, spec, windows)
    nhat = unit(1, np.random.default_rng(1))[0]
    ref = tapered_reference(K, ell, ellp, PLuse, sigma, nhat, nq_ref=nq_ref, nth=nth, nph=nph)
    assert np.max(np.abs(C - ref)) < tol * np.max(np.abs(ref))
    assert np.all(np.diag(C.reshape(K.nbins ** 2, K.nbins ** 2)) > 0)


def test_radial_assembly_isotropic():
    _check((0, 0, 0), (0, 0, 0), {0: PL[0]}, nq_ref=12, nth=12, nph=24, tol=1e-9)


@pytest.mark.slow
def test_radial_assembly_anisotropic():
    _check((2, 0, 2), (2, 0, 2), PL, nq_ref=12, nth=16, nph=32, tol=1e-9)
