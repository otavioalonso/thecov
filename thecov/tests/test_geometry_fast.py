import numpy as np
import pytest
import logging
from thecov import geometry
from mockfactory.make_survey import RandomBoxCatalog
import os
import glob
from mpi4py import MPI

def get_cache_dir():
	return os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache/")

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

def make_surveywindow_stub():
    # Create an instance of SurveyWindow without running __init__ to avoid heavy setup
    w = geometry.SurveyWindow.__new__(geometry.SurveyWindow)
    w.logger = logging.getLogger('SurveyWindow')
    return w

def test_rebin_parameters_sets_attributes_and_returns_expected_values():

    w = make_surveywindow_stub()
    w.boxsize = 100.0
    w.nmesh = 64

    dk = 0.1
    kmax = 0.02
    trim_to_nmesh, rebin_factor = w._rebin_parameters(dk=dk, kmax=kmax)

    # Recompute expected values the same way as the implementation
    target_boxsize = 2 * np.pi / dk
    target_nmesh = int(np.ceil(target_boxsize * kmax / np.pi))
    expected_trim = int(np.ceil(target_boxsize / w.boxsize * w.nmesh))
    if expected_trim % 2 != 0:
        expected_trim += 1
    expected_rebin = expected_trim // target_nmesh

    assert trim_to_nmesh == expected_trim
    assert rebin_factor == expected_rebin
    assert hasattr(w, 'kboxsize') and hasattr(w, 'knmesh')
    assert np.isclose(w.kboxsize, trim_to_nmesh / w.nmesh * w.boxsize)
    assert w.knmesh == trim_to_nmesh // rebin_factor


def test_rebin_parameters_raises_when_rebin_factor_zero():
    w = make_surveywindow_stub()
    w.boxsize = 100.0
    w.nmesh = 8

    dk = 0.1
    # Choose kmax large so target_nmesh > trim_to_nmesh and rebin_factor == 0
    kmax = 10.0

    with pytest.raises(ZeroDivisionError):
        w._rebin_parameters(dk=dk, kmax=kmax)


def test_ikgrid_returns_wrapped_indices():

    w = make_surveywindow_stub()
    w.nmesh = 8

    ikgrid = w.ikgrid
    assert len(ikgrid) == 3
    expected = np.arange(8)
    expected[expected >= 8 // 2] -= 8

    for axis in ikgrid:
        assert np.array_equal(axis, expected)


def test_knyquist_and_kfun_use_knmesh_when_present():
    w = make_surveywindow_stub()
    # case using knmesh/kboxsize
    w.knmesh = 10
    w.kboxsize = 5.0
    assert np.isclose(w.knyquist, np.pi * w.knmesh / w.kboxsize)
    assert np.isclose(w.kfun, 2 * np.pi / w.kboxsize)

    # case falling back to nmesh/boxsize
    w2 = make_surveywindow_stub()
    w2.nmesh = 16
    w2.boxsize = 8.0
    assert np.isclose(w2.knyquist, np.pi * w2.nmesh / w2.boxsize)
    assert np.isclose(w2.kfun, 2 * np.pi / w2.boxsize)

@pytest.mark.parametrize("function, term", [
    ("first_cosmic_variance", None),
    ("second_cosmic_variance", None),
    ("mixed_term", "first"),
    ("mixed_term", "second"),
    ("mixed_term", "third"),
    ("mixed_term", "fourth"),
    ("shotnoise", None),
])
@pytest.mark.mpi(min_size=2)
def test_gaunt_coefficient_methods_are_mpi_safe(function, term):

    rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD
    
    if rank == 0:
        for f in glob.glob(os.path.join(get_cache_dir(), "*coefficients*.npz")):
            os.remove(f)
    comm.Barrier()  # Ensure all processes wait for the file to be removed before proceeding

    if function == "first_cosmic_variance":
          coefficients = geometry.SurveyGeometry.get_first_cosmic_variance_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    elif function == "second_cosmic_variance":
          coefficients = geometry.SurveyGeometry.get_second_cosmic_variance_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    elif function == "mixed_term":
          coefficients = geometry.SurveyGeometry.get_mixed_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm, term=term)
    elif function == "shotnoise":
          coefficients = geometry.SurveyGeometry.get_shotnoise_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    
    comm.Barrier()
    
    # try accessing some properties of coefficients to ensure they were loaded correctly
    assert coefficients is not None
    idx_nonzero, values_nonzero = coefficients.T.get_nonzero_rows_dense()
    assert type(idx_nonzero) == np.ndarray
    assert type(values_nonzero) == np.ndarray
    total_iterations = len(idx_nonzero)
    assert type(total_iterations) == int
    assert total_iterations > 0

    # Now do the exact same tests, but without clearing the cache, so that we test the loading path instead of the computation path. This ensures both paths are MPI-safe.
    if function == "first_cosmic_variance":
          coefficients = geometry.SurveyGeometry.get_first_cosmic_variance_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    elif function == "second_cosmic_variance":
          coefficients = geometry.SurveyGeometry.get_second_cosmic_variance_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    elif function == "mixed_term":
          coefficients = geometry.SurveyGeometry.get_mixed_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm, term=term)
    elif function == "shotnoise":
          coefficients = geometry.SurveyGeometry.get_shotnoise_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    
    comm.Barrier()
    # try accessing some properties of coefficients to ensure they were loaded correctly
    assert coefficients is not None
    idx_nonzero, values_nonzero = coefficients.T.get_nonzero_rows_dense()
    assert type(idx_nonzero) == np.ndarray
    assert type(values_nonzero) == np.ndarray
    total_iterations = len(idx_nonzero)
    assert type(total_iterations) == int
    assert total_iterations > 0

    if rank == 0:
        for f in glob.glob(os.path.join(get_cache_dir(), "*coefficients*.npz")):
            os.remove(f)
    # mpirun -n 2 python -m pytest -v --capture=tee-sys --tb=short --with-mpi thecov/tests -m mpi 

     