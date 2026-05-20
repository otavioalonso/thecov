import numpy as np
import thecov.base as base
import thecov.utils as utils
import os
import pytest
from mpi4py import MPI
from pypower import CatalogMesh

@pytest.mark.mpi(min_size=2)
def test_gather_field_to_root():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nmesh = 16
    # Create a mesh that is different on each rank
    local_array = np.random.rand(1000, 3) + rank
    local_weights = np.ones_like(local_array[:,0])

    local_mesh = CatalogMesh(
                data_positions=local_array,
                data_weights=local_weights,
                position_type='pos',
                nmesh=nmesh,
                dtype='c16',
                mpicomm=comm,
                **{'interlacing': 3, 'resampler': 'tsc'}
            )
    local_mesh = local_mesh.to_mesh(compensate=True)

    assert local_mesh.value.shape != (nmesh, nmesh, nmesh) # should be local slab shape, not full mesh shape
    comm.Barrier()
    # Gather the mesh to the root process as a numpy array
    gathered_mesh_as_array = utils.gather_field_to_root(local_mesh, root=0)
    local_mesh_as_array = local_mesh.value
    
    if rank == 0:
        assert gathered_mesh_as_array.shape == (nmesh, nmesh, nmesh)
    else:
        # Non-root ranks should receive None
        assert gathered_mesh_as_array == 0
    
    # Check that the gathered array contains the local arrays from all ranks
    scattered_mesh_as_array = np.empty((nmesh, nmesh, nmesh), dtype=local_mesh.value.dtype)
    comm.Scatter(gathered_mesh_as_array, scattered_mesh_as_array, root=0) # scatter back to all ranks for comparison
    assert np.allclose(np.intersect1d(scattered_mesh_as_array.flatten(), local_mesh_as_array.flatten()), 
                        np.sort(local_mesh_as_array.flatten()))

@pytest.mark.mpi(min_size=2)
def test_mpi_ellmiter_matches_ellmiter():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ellmax = 4
    mpi_iter = utils.mpi_ellmiter(ellmax, 2, comm)
    iter = utils.ellmiter(ellmax, 2)

    mpi_count = 0
    mpi_list_ellm_rank = np.zeros((ellmax, ellmax, 2*ellmax+1, 2*ellmax+1), dtype=int)# pre-allocate list of correct size
    mpi_list_ellm = np.zeros_like(mpi_list_ellm_rank) # pre-allocate list of correct size
    for ell1, ell2, m1, m2 in mpi_iter:
        m1_idx = m1 + ell1
        m2_idx = m2 + ell2
        assert ell1 <= ellmax and ell2 <= ellmax
        mpi_count += 1
        mpi_list_ellm_rank[ell1//2, ell2//2, m1_idx, m2_idx] = 1
    comm.Reduce(mpi_list_ellm_rank, mpi_list_ellm, op=MPI.SUM, root=0)
    mpi_count = comm.reduce(mpi_count, op=MPI.SUM, root=0)

    count = 0
    list_ellm = np.zeros((ellmax, ellmax, 2*ellmax+1, 2*ellmax+1), dtype=int)
    for ell1, ell2, m1, m2 in iter:
        m1_idx = m1 + ell1
        m2_idx = m2 + ell2
        assert ell1 <= ellmax and ell2 <= ellmax
        count += 1
        list_ellm[ell1//2, ell2//2, m1_idx, m2_idx] = 1
    
    if rank == 0:
        expected_count = ((ellmax//2 + 1) * (ellmax + 1))**2
        assert count == expected_count
        assert count == mpi_count

        assert np.allclose(mpi_list_ellm, list_ellm)

@pytest.mark.mpi(min_size=2)
def test_build_radial_profile():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Use values = r so the profile is a known function: f(r) = r
    positions = np.random.rand(1000, 3)
    r_local = np.sqrt(np.sum(positions**2, axis=-1))
    values = r_local

    # Build the radial profile
    profile = utils.build_radial_profile(positions, values, comm, n_bins=500)

    # Check that the profile is valid
    assert profile is not None

    # Check that the profile can be evaluated
    r = np.linspace(0, 1, 100)
    result = utils.interpolate_to_positions(profile, np.column_stack([r, r, r]))
    assert result is not None
    assert len(result) == 100

    # The profile averages f(r)=r in each bin, so interpolated value should be ~r
    test_pos = positions[0]
    expected_val = np.sqrt(np.sum(test_pos**2))
    interpolated_val = utils.interpolate_to_positions(profile, np.array([test_pos]))
    assert interpolated_val is not None
    assert np.isclose(interpolated_val[0], expected_val, rtol=0.05)