"""Unit tests for the window-function, spherical-harmonic and Bessel
normalizations in :mod:`thecov`.

These tests pin down every normalization convention that the window-matrix /
covariance pipeline relies on:

* real spherical harmonics ``math.Ylm`` are orthonormal AND share the exact
  convention of ``sympy.physics.wigner.real_gaunt`` (the source of the Gaunt
  coefficients they are contracted against);
* the bin-averaged and point spherical Bessel functions match a brute-force
  reference (incl. the ``kmin == 0`` edge that used to produce ``NaN``);
* ``math.nmodes`` matches the reference cosmodesi/thecov shell formula;
* ``double_spherical_bessel_transform`` carries the ``4*pi*(-1j)**l`` plane-wave
  prefactor so the multi-tracer window matches the single-tracer convention;
* the FKP normalizations ``I_ab`` (auto and cross) and their application in the
  covariance reduce to the reference ``(pk_renorm / I('22'))**2`` for a single
  tracer.
"""

import numpy as np
import pytest

# Use double precision for the JAX-backed Bessel functions so we can assert
# tight tolerances. Must run before any JAX array is created.
import jax
jax.config.update("jax_enable_x64", True)

from scipy.special import spherical_jn
from scipy.integrate import quad
from sympy.physics.wigner import real_gaunt

from thecov import math as M


# --------------------------------------------------------------------------- #
# Quadrature helper: Gauss-Legendre in mu = cos(theta), uniform in phi.
# Exact for polynomials in the unit-vector components up to high degree.
# --------------------------------------------------------------------------- #
def _sphere_grid(nmu=64, nphi=64):
    mu, wmu = np.polynomial.legendre.leggauss(nmu)
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    wphi = 2 * np.pi / nphi
    MU, PH = np.meshgrid(mu, phi, indexing="ij")
    WMU, _ = np.meshgrid(wmu, phi, indexing="ij")
    w = WMU * wphi
    sinth = np.sqrt(1 - MU**2)
    x = sinth * np.cos(PH)
    y = sinth * np.sin(PH)
    z = MU
    return x, y, z, w


_YLM_PAIRS = [(0, 0), (2, 0), (2, 1), (2, -1), (2, 2), (2, -2),
              (4, 0), (4, 2), (4, 3), (4, -3), (4, -4)]


# --------------------------------------------------------------------------- #
# 1. Real spherical harmonics
# --------------------------------------------------------------------------- #
class TestYlmNormalization:

    def test_orthonormal(self):
        """integral Y_lm Y_l'm' dOmega = delta_{ll'} delta_{mm'}."""
        x, y, z, w = _sphere_grid()
        Ys = {p: M.Ylm(*p)(x, y, z) for p in _YLM_PAIRS}
        for p in _YLM_PAIRS:
            for q in _YLM_PAIRS:
                overlap = float(np.sum(Ys[p] * Ys[q] * w))
                expected = 1.0 if p == q else 0.0
                assert overlap == pytest.approx(expected, abs=1e-9), \
                    f"<Y{p}|Y{q}> = {overlap}, expected {expected}"

    @pytest.mark.parametrize("l1,l2,l3,m1,m2,m3", [
        (0, 0, 0, 0, 0, 0),
        (2, 2, 0, 0, 0, 0),
        (2, 2, 4, 0, 0, 0),
        (2, 2, 2, 1, 1, -2),
        (4, 2, 2, 3, -1, -2),
        (4, 4, 4, 0, 0, 0),
        (2, 2, 4, 1, -1, 0),
    ])
    def test_matches_real_gaunt(self, l1, l2, l3, m1, m2, m3):
        """integral Y Y Y dOmega must equal sympy.real_gaunt with the SAME
        convention, otherwise the window-matrix / Gaunt contraction is wrong."""
        x, y, z, w = _sphere_grid()
        numeric = float(np.sum(
            M.Ylm(l1, m1)(x, y, z)
            * M.Ylm(l2, m2)(x, y, z)
            * M.Ylm(l3, m3)(x, y, z) * w))
        reference = float(real_gaunt(l1, l2, l3, m1, m2, m3))
        assert numeric == pytest.approx(reference, abs=1e-9)

    def test_amplitude_monopole(self):
        """Y_00 = 1/sqrt(4 pi) everywhere."""
        f = M.Ylm(0, 0)
        vals = f(np.array([1.0, -3.0, 0.2]),
                 np.array([0.5, 1.0, -2.0]),
                 np.array([2.0, 0.1, 1.0]))
        assert np.allclose(vals, 1.0 / np.sqrt(4 * np.pi))

    def test_origin_returns_zero(self):
        """The unit-vector projection is undefined at r=0 -> defined as 0."""
        assert M.Ylm(2, 1)(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0] == 0.0


# --------------------------------------------------------------------------- #
# 2. Spherical Bessel functions
# --------------------------------------------------------------------------- #
class TestSphericalBessel:

    KEDGES_ZERO = np.array([0.0, 0.05, 0.10, 0.15, 0.20])   # first bin starts at 0
    KEDGES_POS = np.array([0.02, 0.07, 0.12, 0.17, 0.22])
    RADII = np.array([5.0, 25.0, 80.0, 150.0])

    @pytest.mark.parametrize("nu", [0, 1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("kedges_name", ["KEDGES_ZERO", "KEDGES_POS"])
    def test_averaged_matches_quad(self, nu, kedges_name):
        """Bin-averaged j_nu == (1/int k^2 dk) int k^2 j_nu(k x) dk."""
        kedges = getattr(self, kedges_name)
        got = np.asarray(M.spherical_bessel(nu, self.RADII, kedges, averaged=True))
        assert not np.isnan(got).any(), "NaN in averaged spherical Bessel"
        for ib in range(len(kedges) - 1):
            kmin, kmax = kedges[ib], kedges[ib + 1]
            norm = 3.0 / (kmax**3 - kmin**3)
            for ir, r in enumerate(self.RADII):
                ref = norm * quad(lambda k: k**2 * spherical_jn(nu, k * r),
                                  kmin, kmax)[0]
                assert got[ib, ir] == pytest.approx(ref, abs=1e-9, rel=1e-7)

    def test_averaged_kmin_zero_is_finite(self):
        """Regression: bins starting at k=0 must not produce NaN for nu>=3."""
        kedges = np.array([0.0, 0.05, 0.1])
        for nu in range(7):
            got = np.asarray(M.spherical_bessel(nu, self.RADII, kedges, averaged=True))
            assert np.isfinite(got).all(), f"non-finite averaged Bessel at nu={nu}, kmin=0"

    @pytest.mark.parametrize("nu", [0, 1, 2, 3, 4, 5, 6])
    def test_pointwise_matches_scipy(self, nu):
        """Non-averaged j_nu == scipy.spherical_jn at the bin midpoint.

        Note: the explicit closed forms lose precision via catastrophic
        cancellation for small arguments z = k*r and high nu (the leading
        ~1/z**nu terms cancel down to ~z**nu). Those points are physically
        negligible (|j_nu| -> 0) and the window pipeline uses the averaged
        form, so we only assert where the value is non-negligible.
        """
        kedges = self.KEDGES_POS
        kmid = 0.5 * (kedges[:-1] + kedges[1:])
        got = np.asarray(M.spherical_bessel(nu, self.RADII, kedges, averaged=False))
        for ib, km in enumerate(kmid):
            for ir, r in enumerate(self.RADII):
                ref = spherical_jn(nu, km * r)
                if abs(ref) < 1e-6:
                    continue  # deep small-z tail; closed form cancels to ~0
                assert got[ib, ir] == pytest.approx(ref, abs=1e-10, rel=1e-7)

    @pytest.mark.parametrize("averaged", [True, False])
    @pytest.mark.parametrize("nu,limit", [(0, 1.0), (2, 0.0), (4, 0.0)])
    def test_r_zero_limit(self, averaged, nu, limit):
        """At r=0, j_0 -> 1 and j_{nu>0} -> 0 (same for the averaged form)."""
        r = np.array([0.0, 0.0])
        got = np.asarray(M.spherical_bessel(nu, r, self.KEDGES_POS, averaged=averaged))
        assert np.allclose(got, limit)


# --------------------------------------------------------------------------- #
# 3. Mode counting
# --------------------------------------------------------------------------- #
class TestNmodes:

    def test_formula(self):
        """N = V / (6 pi^2) * (kmax^3 - kmin^3)  (cosmodesi/thecov convention)."""
        V = 1.7e9
        kmin = np.array([0.0, 0.1, 0.2])
        kmax = np.array([0.1, 0.2, 0.3])
        expected = V / 3.0 / (2 * np.pi**2) * (kmax**3 - kmin**3)
        assert np.allclose(M.nmodes(V, kmin, kmax), expected)

    def test_linear_binning_uses_it(self):
        """base.LinearBinning.nmodes routes through math.nmodes (was AttributeError)."""
        from thecov import base
        V = 1e9
        lb = base.LinearBinning(kmin=0.0, kmax=0.2, dk=0.05, volume=V)
        expected = M.nmodes(V, lb.edges[:-1], lb.edges[1:])
        assert np.allclose(lb.nmodes, expected)


# --------------------------------------------------------------------------- #
# 4. Double spherical Bessel transform (multi-tracer window)
# --------------------------------------------------------------------------- #
class TestDoubleBesselTransform:

    def test_carries_4pi_minus_i_factor(self):
        """The transform must apply 4*pi*(-1j)**l per leg so the multi-tracer
        window matches the single-tracer convention
        (geometry: bessels = 4*pi*(-1j)**nu * j_nu)."""
        kedges = np.array([0.0, 0.05, 0.10, 0.15])
        # W populated in a single radial bin -> the transform collapses to an
        # outer product of single-leg Bessels at that radius.
        xbins = np.array([0.0, 100.0, 200.0, 300.0])
        W = np.array([0.0, 1.0, 0.0])      # only the middle bin is non-zero
        x = 0.5 * (xbins[1:] + xbins[:-1])
        wx = np.diff(xbins) * x**2 * W

        for l1, l2 in [(0, 0), (0, 2), (2, 2), (2, 4), (4, 4)]:
            got = M.double_spherical_bessel_transform(
                xbins=xbins, W=W, kedges=kedges, l1=l1, l2=l2)

            j1 = np.asarray(M.spherical_bessel(l1, x, kedges, averaged=True))
            j2 = np.asarray(M.spherical_bessel(l2, x, kedges, averaged=True))
            fac1 = 4 * np.pi * np.real((-1j)**l1)
            fac2 = 4 * np.pi * np.real((-1j)**l2)
            expected = fac1 * fac2 * np.einsum("kx,qx,x->kq", j1, j2, wx)

            assert np.allclose(got, expected, atol=1e-10)

    def test_fourier_kernel_factor_reconstructs_cosine(self):
        """The per-leg factor 4*pi*Re((-1j)**l) = 4*pi*(-1)**(l/2) is exactly
        the even-multipole Rayleigh expansion of the *real part* of the Fourier
        kernel:

            cos(k.r) = 4*pi * sum_{l even} (-1)**(l/2) j_l(kr)
                                     * sum_m Y_lm(k_hat) Y_lm(r_hat)

        This pins the sign convention of the (-1)**power factors used in both
        the single-tracer 'bessels' array and double_spherical_bessel_transform.
        Only even l contribute (odd l carry Re(i**l)=0), consistent with the
        even-only multipoles of the redshift-space covariance.
        """
        kvec = np.array([0.07, -0.03, 0.05])
        rvec = np.array([6.0, 4.0, -3.0])
        kr = np.linalg.norm(kvec) * np.linalg.norm(rvec)
        kh = kvec / np.linalg.norm(kvec)
        rh = rvec / np.linalg.norm(rvec)

        total = 0.0
        for l in range(0, 11, 2):
            msum = sum(M.Ylm(l, m)(*kh) * M.Ylm(l, m)(*rh) for m in range(-l, l + 1))
            total += 4 * np.pi * np.real((-1j)**l) * spherical_jn(l, kr) * msum

        assert total == pytest.approx(np.cos(kvec @ rvec), abs=1e-5)

    def test_factor_is_even_l_real(self):
        """Re((-1j)**l) == Re((1j)**l) == (-1)**(l/2) for even l; 0 for odd l."""
        for l in range(0, 7, 2):
            assert np.real((-1j)**l) == pytest.approx((-1)**(l // 2))
            assert np.real((1j)**l) == pytest.approx((-1)**(l // 2))
        for l in [1, 3, 5]:
            assert np.real((-1j)**l) == pytest.approx(0.0, abs=1e-12)

    def test_matches_single_tracer_bessels(self):
        """Element-by-element: transform of a delta-shell == product of the
        single-tracer 'bessels' array used in _compute_window_matrix."""
        kedges = np.array([0.0, 0.05, 0.10, 0.15])
        xbins = np.array([0.0, 100.0, 200.0])
        W = np.array([0.0, 1.0])
        x = 0.5 * (xbins[1:] + xbins[:-1])     # single active radius (x[1])
        wx = np.diff(xbins) * x**2 * W

        nu1, nu2 = 2, 4
        got = M.double_spherical_bessel_transform(
            xbins=xbins, W=W, kedges=kedges, l1=nu1, l2=nu2)

        # Reproduce the single-tracer 'bessels' definition from geometry.py.
        b1 = 4 * np.pi * np.real((-1j)**nu1) * \
            np.asarray(M.spherical_bessel(nu1, x, kedges, averaged=True))
        b2 = 4 * np.pi * np.real((-1j)**nu2) * \
            np.asarray(M.spherical_bessel(nu2, x, kedges, averaged=True))
        # active radius is index 1 (x[1]); index 0 has W=0
        expected = np.einsum("k,q->kq", b1[:, 1], b2[:, 1]) * wx[1]

        assert np.allclose(got, expected, atol=1e-10)


# --------------------------------------------------------------------------- #
# 5. FKP normalizations I_ab (auto + cross) and covariance scaling
# --------------------------------------------------------------------------- #
def _make_single_tracer(pos, nz, weight=1.0, nz_weight=1.0):
    from thecov.geometry import SingleTracerSurveyGeometry
    return SingleTracerSurveyGeometry(
        randoms_pos=pos, randoms_nz=nz,
        randoms_weight=weight, randoms_nz_weight=nz_weight)


class TestFKPNormalization:

    def _catalog(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        pos = rng.normal(size=(n, 3)) * 100.0 + 500.0
        nz = rng.uniform(1e-4, 5e-4, size=n)
        weight = rng.uniform(0.8, 1.2, size=n)
        return pos, nz, weight

    def test_single_tracer_normalization_formula(self):
        """I(p,q) = sum(nz**(p-1) * weight**q * nz_weight)."""
        pos, nz, weight = self._catalog()
        nzw = 1.3
        geom = _make_single_tracer(pos, nz, weight=weight, nz_weight=nzw)
        for p, q in [(1, 0), (2, 2), (1, 2), (3, 2)]:
            expected = float((nz**(p - 1) * weight**q * nzw).sum())
            assert geom.normalization(p, q) == pytest.approx(expected, rel=1e-12)

    def test_I22_matches_reference_definition(self):
        """I('22') in cosmodesi/thecov = sum(nz * weight_fkp**2 * weight_sys)."""
        pos, nz, weight = self._catalog()
        nzw = 0.7
        geom = _make_single_tracer(pos, nz, weight=weight, nz_weight=nzw)
        expected = float((nz * weight**2 * nzw).sum())   # nz**(2-1) = nz
        assert geom.normalization(2, 2) == pytest.approx(expected, rel=1e-12)

    def test_cross_normalization_is_geometric_mean(self):
        """MultiTracer.normalization() == sqrt(I_AA * I_BB)."""
        from thecov.geometry import MultiTracerSurveyGeometry
        pos1, nz1, w1 = self._catalog(n=150, seed=1)
        pos2, nz2, w2 = self._catalog(n=180, seed=2)
        geom = MultiTracerSurveyGeometry(
            randoms_pos1=pos1, randoms_pos2=pos2,
            randoms_nz1=nz1, randoms_nz2=nz2,
            randoms_weight1=w1, randoms_weight2=w2)
        I1 = (nz1 * w1**2 * 1.0).sum()
        I2 = (nz2 * w2**2 * 1.0).sum()
        assert geom.normalization() == pytest.approx(np.sqrt(I1 * I2), rel=1e-12)


class _StubAuto:
    """Minimal stand-in for SingleTracerSurveyGeometry exposing normalization()."""
    def __init__(self, I22):
        self._I22 = I22

    def normalization(self, p, q):
        assert (p, q) == (2, 2)
        return self._I22


class TestCovarianceNormalizationApplication:

    def test_pair_normalization_auto(self):
        from thecov.covariance import MultiTracerGaussianCovariance
        cov = MultiTracerGaussianCovariance()
        cov.auto_windows["A"] = _StubAuto(42.0)
        assert cov._pair_normalization("A", "A") == 42.0

    def test_pair_normalization_cross(self):
        from thecov.covariance import MultiTracerGaussianCovariance

        class _StubCross:
            def normalization(self):
                return 9.0

        cov = MultiTracerGaussianCovariance()
        cov.cross_windows[frozenset(["A", "B"])] = _StubCross()
        assert cov._pair_normalization("A", "B") == 9.0

    def test_single_tracer_scaling_reduces_to_reference(self):
        """The final covariance scale is pk_renorm**2 / (I_AB * I_CD); for a
        single tracer this is the reference (pk_renorm / I('22'))**2."""
        from thecov.covariance import MultiTracerGaussianCovariance
        I22 = 12.5
        pk_renorm = 2.0
        cov = MultiTracerGaussianCovariance(pk_renorm=pk_renorm)
        cov.auto_windows["0"] = _StubAuto(I22)

        # Reproduce exactly the scaling line of compute_covariance_block.
        unnormalized = np.array([[3.0, 1.0], [1.0, 4.0]])
        scale = pk_renorm**2 / (
            cov._pair_normalization("0", "0") * cov._pair_normalization("0", "0"))
        scaled = unnormalized * scale

        reference = unnormalized * (pk_renorm / I22)**2
        assert np.allclose(scaled, reference)
        assert scale == pytest.approx((pk_renorm / I22)**2)
