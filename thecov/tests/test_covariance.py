import numpy as np
import pytest
from types import SimpleNamespace

from thecov import covariance, geometry, binning


class StubGeometry:
	"""Creates a dummy geometry object for testing multi-tracer covariance."""
	def __init__(self, num_tracers=2, alphas=None, I12=None, I22=None):
		self.num_tracers = num_tracers
		# keep insertion order consistent with TRACER_LABELS
		if alphas is None:
			alphas = {l: 0.0 for l in covariance.TRACER_LABELS[:num_tracers]}
		self.alphas = alphas
		self.I_12 = np.array(I12) if I12 is not None else np.ones(num_tracers)
		self.I_22 = np.array(I22) if I22 is not None else np.ones(num_tracers)


def test_set_galaxy_pk_multipole_stores_symmetric_keys():
	g = StubGeometry(num_tracers=3)
	cov = covariance.GaussianCovariance(geometry=g)

	# provide a k-binning stub matching pk length
	cov.set_kbins(0.001, 0.2, 0.005)
	pk = np.arange(cov.k_binning.kbins)

	ell = 0
	cov.set_galaxy_pk_multipole(pk.copy(), ell, tracer1=0, tracer2=1, has_shotnoise=False)

	# underlying storage uses tuple keys (ell, tracer1, tracer2)
	assert (ell, 0, 1) in cov._pk
	assert (ell, 1, 0) in cov._pk
	assert np.array_equal(cov._pk[(ell, 0, 1)], cov._pk[(ell, 1, 0)])

@pytest.mark.parametrize("num_tracers, tracer1, tracer2, expected", [
    (1, "A", "A", None),
    (1, "A", "B", ValueError),
    (2, "A", "A", None),
    (2, "A", "B", None),
    (2, "B", "A", None),
    (2, "B", "B", None),
    (3, "A", "D", ValueError),
    (3, "A", "C", None),
])
def test_get_tracer_cov_labels(num_tracers, tracer1, tracer2, expected):
	g = StubGeometry(num_tracers=num_tracers)
	cov = covariance.PowerSpectrumMultiTracerCovariance(geometry=g)
	cov.set_kbins(0.001, 0.2, 0.005)
	cov._cov = np.zeros((num_tracers * cov.k_binning.kbins * 2, num_tracers * cov.k_binning.kbins * 2))  # dummy covariance

	if isinstance(expected, type) and issubclass(expected, Exception):
		with pytest.raises(expected):
			C_dummy = cov.get_tracer_cov(tracer1, tracer2)
	else:
		C_dummy = cov.get_tracer_cov(tracer1, tracer2)

def test_shotnoise_computation_uses_geometry_I_and_alphas_and_pk_renorm():
	# Build a stub geometry and temporarily make isinstance checks pass by
	# treating geometry.SurveyGeometry as object during this test.
	g = StubGeometry(num_tracers=2, alphas={"A": 0.1, "B": 0.2}, I12=[10.0, 20.0], I22=[5.0, 4.0])

	cov = covariance.PowerSpectrumMultiTracerCovariance(geometry=g)

	# monkeypatch the type check in the geometry module so our stub is considered a SurveyGeometry
	orig_survey = geometry.SurveyGeometry
	try:
		geometry.SurveyGeometry = object
		# compute expected shotnoise per tracer
		alphas_array = np.array(list(g.alphas.values()))
		expected = []
		for t in range(g.num_tracers):
			expected.append(cov.pk_renorm * (1 + alphas_array[t]) * g.I_12[t] / g.I_22[t])
		expected = np.array(expected)

		sn = cov.shotnoise
		assert np.allclose(sn, expected)
	finally:
		geometry.SurveyGeometry = orig_survey


def test_load_npy_file_raises_on_wrong_dimensions(tmp_path):
	g = StubGeometry(num_tracers=1)
	cov = covariance.GaussianCovariance(geometry=g)

	arr = np.zeros((2, 2, 2))  # not 4D
	p = tmp_path / "pk_wrong.npy"
	np.save(p, arr)

	with pytest.raises(ValueError):
		cov.load_npy_file(str(p))
