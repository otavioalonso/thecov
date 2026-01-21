import numpy as np
import pytest

from mockfactory.make_survey import RandomBoxCatalog
from thecov import covariance, geometry, binning

def create_basic_randoms(num_tracers):

	nbar = np.random.rand(num_tracers) * 1e-5
	boxsize = 1000.0

	randoms = []
	for t in range(4):
		if t+1 <= num_tracers:
			randoms.append(RandomBoxCatalog(nbar=nbar[t], boxsize=boxsize))
			randoms[t]["POSITION"] = randoms[t]["Position"]
		else:
			randoms.append(None)
	return randoms

def test_set_galaxy_pk_multipole_stores_symmetric_keys():
	
	randoms = create_basic_randoms(num_tracers=3)
	alpha = [0.1, 0.1, 0.1]
	kmax = 0.05
	g = geometry.SurveyGeometry(randoms[0], alpha[0],
							    randoms[1], alpha[1],
							    randoms[2], alpha[2],
							    None, None,
							    nmesh=32, boxpad=1.2,
							    kmin=0.001, kmax=kmax, dk=0.005)
	cov = covariance.GaussianCovariance(geometry=g)

	# provide a k-binning stub matching pk length
	cov.set_kbins(0.001, kmax, 0.005)
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
	randoms = create_basic_randoms(num_tracers)
	alpha = [0.1, 0.1, 0.1, 0.1]
	kmax = 0.05
	g = geometry.SurveyGeometry(randoms[0], alpha[0],
							    randoms[1], alpha[1],
							    randoms[2], alpha[2],
							    randoms[3], alpha[3],
							    nmesh=32, boxpad=1.2,
							    kmin=0.001, kmax=kmax, dk=0.005)
	cov = covariance.PowerSpectrumMultiTracerCovariance(geometry=g)
	cov.set_kbins(0.001, kmax, 0.005)
	cov._cov = np.zeros((num_tracers * cov.k_binning.kbins * 2, num_tracers * cov.k_binning.kbins * 2))  # dummy covariance

	if isinstance(expected, type) and issubclass(expected, Exception):
		with pytest.raises(expected):
			C_dummy = cov.get_tracer_cov(tracer1, tracer2)
	else:
		C_dummy = cov.get_tracer_cov(tracer1, tracer2)

def test_shotnoise_computation_uses_geometry_I_and_alphas_and_pk_renorm():
	# Build a stub geometry and temporarily make isinstance checks pass by
	# treating geometry.SurveyGeometry as object during this test.
	TRACER_LABELS = ["A", "B", "C", "D"]
	randoms = create_basic_randoms(num_tracers=2)
	alpha = [0.1, 0.1]
	kmax = 0.05
	g = geometry.SurveyGeometry(randoms[0], alpha[0],
							    randoms[1], alpha[1],
								None, None,
							    None, None,
							    nmesh=32, boxpad=1.2,
							    kmin=0.001, kmax=kmax, dk=0.005)

	cov = covariance.PowerSpectrumMultiTracerCovariance(geometry=g)

	# monkeypatch the type check in the geometry module so our stub is considered a SurveyGeometry
	orig_survey = geometry.SurveyGeometry
	try:
		geometry.SurveyGeometry = object
		# compute expected shotnoise per tracer
		alphas_array = np.array(list(g.alphas.values()))
		expected = []
		for t in range(g.num_tracers):
			expected.append(cov.pk_renorm * (1 + alphas_array[t]) * g.I(TRACER_LABELS[t], 1, 2) / g.I(TRACER_LABELS[t], 2, 2))
		expected = np.array(expected)

		sn = cov.shotnoise
		assert np.allclose(sn, expected)
	finally:
		geometry.SurveyGeometry = orig_survey


def test_load_npy_file_raises_on_wrong_dimensions(tmp_path):
	randoms = create_basic_randoms(num_tracers=2)
	alpha = [0.1, 0.1, 0.1]
	kmax = 0.05
	g = geometry.SurveyGeometry(randoms[0], alpha[0],
							    randoms[1], alpha[1],
							    randoms[2], alpha[2],
							    None, None,
							    nmesh=32, boxpad=1.2,
							    kmin=0.001, kmax=kmax, dk=0.005)
	cov = covariance.GaussianCovariance(geometry=g)

	arr = np.zeros((2, 2, 2))  # not 4D
	p = tmp_path / "pk_wrong.npy"
	np.save(p, arr)

	with pytest.raises(ValueError):
		cov.load_npy_file(str(p))
