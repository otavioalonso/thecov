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