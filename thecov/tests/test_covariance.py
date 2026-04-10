import numpy as np
import pytest
import os

from mockfactory.make_survey import RandomBoxCatalog
from thecov import covariance, geometry

def create_basic_randoms(num_tracers):

	nbar = np.random.rand(num_tracers) * 1e-5
	boxsize = 1000.0

	randoms = []
	for t in range(num_tracers):
		randoms.append(RandomBoxCatalog(nbar=nbar[t], boxsize=boxsize))
		randoms[t]["POSITION"] = randoms[t]["Position"]
		randoms[t]["NZ"] = np.random.rand(len(randoms[t])) * 1e-5
	return randoms

def test_set_galaxy_pk_multipole_stores_symmetric_keys():
	
	randoms = create_basic_randoms(num_tracers=3)
	alpha = [0.1, 0.1, 0.1]
	kmax = 0.05
	g = geometry.SurveyGeometry(randoms, alpha,
							    nmesh=32, boxpad=1.2,
							    kmin=0.001, kmax=kmax, dk=0.005,
								resume_file="test.npy")
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
	os.remove("test.npy")
	
@pytest.mark.parametrize("num_tracers, tracer1, tracer2, expected", [
    (1, 0, 0, None),
    (1, 0, 1, KeyError),
    (2, 0, 0, None),
    (2, 0, 1, None),
    (2, 1, 0, None),
    (2, 1, 1, None),
    (3, 0, 3, KeyError),
    (3, 0, 2, None),
])
def test_get_tracer_cov_labels(num_tracers, tracer1, tracer2, expected):
	randoms = create_basic_randoms(num_tracers)
	alpha = [0.1, 0.1, 0.1, 0.1]
	kmax = 0.05
	g = geometry.SurveyGeometry(randoms, alpha,
							    nmesh=32, boxpad=1.2,
							    kmin=0.001, kmax=kmax, dk=0.005,
								resume_file="test.npy")
	cov = covariance.PowerSpectrumCovariance(geometry=g)
	cov.set_kbins(0.001, kmax, 0.005)
	for t1 in range(num_tracers):
		for t2 in range(num_tracers):
			cov.set_ell_tracer_cov(0, 0, t1, t2, np.zeros((cov.k_binning.kbins, cov.k_binning.kbins))+ (t1 + t2))

	if isinstance(expected, type) and issubclass(expected, Exception):
		with pytest.raises(expected):
			C_dummy = cov.get_ell_tracer_cov(0,0, tracer1, tracer2)
	else:
		C_dummy = cov.get_ell_tracer_cov(0,0, tracer1, tracer2)
		assert C_dummy is not None
		assert C_dummy.cov[0,0] == tracer1 + tracer2

	os.remove("test.npy")

def test_shotnoise_computation_uses_geometry_I_and_alphas_and_pk_renorm():
	# Build a stub geometry and temporarily make isinstance checks pass by
	# treating geometry.SurveyGeometry as object during this test.
	TRACER_LABELS = ["A", "B", "C", "D"]
	randoms = create_basic_randoms(num_tracers=2)
	alpha = [0.1, 0.1]
	kmax = 0.05
	g = geometry.SurveyGeometry(randoms, alpha,
							    nmesh=32, boxpad=1.2,
							    kmin=0.001, kmax=kmax, dk=0.005,
								resume_file="test.npy")

	cov = covariance.PowerSpectrumCovariance(geometry=g)

	# compute expected shotnoise per tracer
	alphas_array = np.array(g.alphas)
	expected = []
	for t in range(g.num_tracers):
		expected.append(cov.pk_renorm * (1 + alphas_array[t]) * g.I(t, t, 1, 2) / g.I(t, t, 2, 2))
	expected = np.array(expected)

	sn = cov.shotnoise
	assert np.allclose(sn, expected)

	os.remove("test.npy")

def test_load_npy_file_raises_on_wrong_dimensions(tmp_path):
	randoms = create_basic_randoms(num_tracers=2)
	alpha = [0.1, 0.1, 0.1]
	kmax = 0.05
	g = geometry.SurveyGeometry(randoms, alpha,
							    nmesh=32, boxpad=1.2,
							    kmin=0.001, kmax=kmax, dk=0.005,
								resume_file="test.npy")
	cov = covariance.GaussianCovariance(geometry=g)

	arr = np.zeros((2, 2, 2))  # not 4D
	p = tmp_path / "pk_wrong.npy"
	np.save(p, arr)

	with pytest.raises(ValueError):
		cov.load_npy_file(str(p))
	os.remove("test.npy")
