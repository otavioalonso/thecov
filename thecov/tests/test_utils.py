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