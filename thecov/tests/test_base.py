import numpy as np
import thecov.base as base
import os
import pytest
from mpi4py import MPI

def test_base_covariance_operations():
    rng = np.random.default_rng(1)
    a = rng.random((4, 4))
    # make symmetric positive-definite for stable eigvals
    mat = a @ a.T + np.eye(4) * 1e-3

    cov = base.Covariance(mat)

    assert np.allclose(cov.cov, mat)
    # transpose property
    assert np.allclose(cov.T.cov, mat.T)

    # correlation matrix diagonals are 1
    cor = cov.cor
    assert np.allclose(np.diag(cor), np.ones(4))

def test_multipole_multitracer_covariance_stack_and_get():
    rng = np.random.default_rng(2)
    # small 2x2 blocks
    b00 = rng.random((2, 2))
    b02 = rng.random((2, 2))
    b20 = rng.random((2, 2))
    b22 = rng.random((2, 2))

    m = base.MultipoleMultiTracerCovariance()

    m.set_ell_tracer_cov(0, 0, 0, 0, b00)
    m.set_ell_tracer_cov(0, 2, 0, 0, b02)
    m.set_ell_tracer_cov(2, 0, 0, 0, b20)
    m.set_ell_tracer_cov(2, 2, 0, 0, b22)

    # verify that the stacked matrix contains the blocks from get_ell_cov
    full = m.cov
    ells1, ells2 = m.ells
    # compute block sizes by inspecting diagonal blocks when present
    sizes1 = [m.get_ell_tracer_cov(l1, l1, 0, 0).cov.shape[0] for l1 in ells1]
    sizes2 = [m.get_ell_tracer_cov(l2, l2, 0, 0).cov.shape[1] for l2 in ells2]

    rstart = 0
    for i, l1 in enumerate(ells1):
        rend = rstart + sizes1[i]
        cstart = 0
        for j, l2 in enumerate(ells2):
            cend = cstart + sizes2[j]
            block = full[rstart:rend, cstart:cend]
            expected_block = m.get_ell_tracer_cov(l1, l2, 0, 0).cov
            assert np.allclose(block, expected_block)
            cstart = cend
        rstart = rend

def test_multipole_fourier_covariance_integration():
    rng = np.random.default_rng(3)
    cov = base.MultipoleFourierCovariance()
    # set k-bins: kmin=0.0, kmax=0.3, dk=0.1 -> 3 bins
    cov.set_kbins(0.0, 0.3, 0.1)
    kb = cov.kbins
    assert kb == 3

    # build two ells -> 0 and 2
    b00 = rng.random((kb, kb))
    b02 = rng.random((kb, kb))
    b20 = rng.random((kb, kb))
    b22 = rng.random((kb, kb))

    cov.set_ell_tracer_cov(0, 0, 0, 0, b00)
    cov.set_ell_tracer_cov(0, 2, 0, 0, b02)
    cov.set_ell_tracer_cov(2, 0, 0, 0, b20)
    cov.set_ell_tracer_cov(2, 2, 0, 0, b22)

    full = cov.cov
    assert full.shape == (2 * kb, 2 * kb)

    ells1, ells2 = cov.ells
    # all blocks are kb x kb
    row_offsets = [i * kb for i in range(len(ells1) + 1)]
    col_offsets = [i * kb for i in range(len(ells2) + 1)]

    for i, l1 in enumerate(ells1):
        for j, l2 in enumerate(ells2):
            r0, r1 = row_offsets[i], row_offsets[i + 1]
            c0, c1 = col_offsets[j], col_offsets[j + 1]
            block = full[r0:r1, c0:c1]
            expected_block = cov.get_ell_tracer_cov(l1, l2, 0, 0).cov
            assert np.allclose(block, expected_block)

def test_multipole_covariance_symmetrization():
    cov00, cov22, cov44, cov02, cov04, cov24 = np.random.rand(6, 100, 100)

    cov = base.MultipoleMultiTracerCovariance()

    cov.set_ell_tracer_cov(0, 0, 0, 0, cov00)
    cov.set_ell_tracer_cov(2, 2, 0, 0, cov22)
    cov.set_ell_tracer_cov(4, 4, 0, 0, cov44)
    cov.set_ell_tracer_cov(0, 2, 0, 0, cov02)
    cov.set_ell_tracer_cov(0, 4, 0, 0, cov04)
    cov.set_ell_tracer_cov(4, 2, 0, 0, cov24)

    assert (cov.get_ell_tracer_cov(0, 2, 0, 0).cov == cov02).all()
    assert (cov.get_ell_tracer_cov(2, 0, 0, 0).cov == cov02.T).all()

    assert (cov.get_ell_tracer_cov(0, 4, 0, 0).cov == cov04).all()
    assert (cov.get_ell_tracer_cov(4, 0, 0, 0).cov == cov04.T).all()
    assert (cov.get_ell_tracer_cov(4, 2, 0, 0).cov == cov24).all()
    assert (cov.get_ell_tracer_cov(2, 4, 0, 0).cov == cov24.T).all()

    assert not (cov.get_ell_tracer_cov(0, 0, 0, 0).cov == cov.get_ell_tracer_cov(0, 0, 0, 0).cov.T).all()
    assert not (cov.get_ell_tracer_cov(2, 2, 0, 0).cov == cov.get_ell_tracer_cov(2, 2, 0, 0).cov.T).all()
    assert not (cov.get_ell_tracer_cov(4, 4, 0, 0).cov == cov.get_ell_tracer_cov(4, 4, 0, 0).cov.T).all()

    cov.symmetrize()

    assert (cov.get_ell_tracer_cov(0, 2, 0, 0).cov == cov02).all()
    assert (cov.get_ell_tracer_cov(2, 0, 0, 0).cov == cov02.T).all()

    assert (cov.get_ell_tracer_cov(0, 4, 0, 0).cov == cov04).all()
    assert (cov.get_ell_tracer_cov(4, 0, 0, 0).cov == cov04.T).all()

    assert (cov.get_ell_tracer_cov(4, 2, 0, 0).cov == cov24).all()
    assert (cov.get_ell_tracer_cov(2, 4, 0, 0).cov == cov24.T).all()
    assert (cov.get_ell_tracer_cov(0, 0, 0, 0).cov == cov.get_ell_tracer_cov(0, 0, 0, 0).cov.T).all()
    assert (cov.get_ell_tracer_cov(2, 2, 0, 0).cov == cov.get_ell_tracer_cov(2, 2, 0, 0).cov.T).all()
    assert (cov.get_ell_tracer_cov(4, 4, 0, 0).cov == cov.get_ell_tracer_cov(4, 4, 0, 0).cov.T).all()

    assert not (cov.get_ell_tracer_cov(0, 2, 0, 0).cov == cov.get_ell_tracer_cov(0, 2, 0, 0).cov.T).all()
    assert not (cov.get_ell_tracer_cov(2, 4, 0, 0).cov == cov.get_ell_tracer_cov(2, 4, 0, 0).cov.T).all()
    assert not (cov.get_ell_tracer_cov(4, 0, 0, 0).cov == cov.get_ell_tracer_cov(4, 0, 0, 0).cov.T).all()
    assert (cov.get_ell_tracer_cov(0, 0, 0, 0).cov == (cov00 + cov00.T)/2).all()
    assert (cov.get_ell_tracer_cov(2, 2, 0, 0).cov == (cov22 + cov22.T)/2).all()
    assert (cov.get_ell_tracer_cov(4, 4, 0, 0).cov == (cov44 + cov44.T)/2).all()


def test_multipole_covariance_addition():
    cov1_00, cov1_22, cov1_44, cov1_02, cov1_04, cov1_24 = np.random.rand(6, 100, 100)
    cov2_00, cov2_22, cov2_44, cov2_02, cov2_04, cov2_24 = np.random.rand(6, 100, 100)

    cov1 = base.MultipoleMultiTracerCovariance()
    cov2 = base.MultipoleMultiTracerCovariance()

    cov1.set_ell_tracer_cov(0, 0, 0, 0, cov1_00)
    cov1.set_ell_tracer_cov(2, 2, 0, 0, cov1_22)
    cov1.set_ell_tracer_cov(4, 4, 0, 0, cov1_44)
    cov1.set_ell_tracer_cov(0, 2, 0, 0, cov1_02)
    cov1.set_ell_tracer_cov(0, 4, 0, 0, cov1_04)
    cov1.set_ell_tracer_cov(4, 2, 0, 0, cov1_24.T)

    cov2.set_ell_tracer_cov(0, 0, 0, 0, cov2_00)
    cov2.set_ell_tracer_cov(2, 2, 0, 0, cov2_22)
    cov2.set_ell_tracer_cov(4, 4, 0, 0, cov2_44)
    cov2.set_ell_tracer_cov(0, 2, 0, 0, cov2_02)
    cov2.set_ell_tracer_cov(0, 4, 0, 0, cov2_04)
    cov2.set_ell_tracer_cov(4, 2, 0, 0, cov2_24.T)
    addition = cov1 + cov2

    print(cov1._multipole_tracer_covariance.keys())
    print(cov2._multipole_tracer_covariance.keys())
    print(addition._multipole_tracer_covariance.keys())
    assert (addition.get_ell_tracer_cov(0, 0, 0, 0).cov == cov1_00 + cov2_00).all()
    assert (addition.get_ell_tracer_cov(2, 2, 0, 0).cov == cov1_22 + cov2_22).all()
    assert (addition.get_ell_tracer_cov(4, 4, 0, 0).cov == cov1_44 + cov2_44).all()

    assert (addition.get_ell_tracer_cov(0, 2, 0, 0).cov == cov1_02 + cov2_02).all()
    assert (addition.get_ell_tracer_cov(0, 4, 0, 0).cov == cov1_04 + cov2_04).all()
    assert (addition.get_ell_tracer_cov(4, 2, 0, 0).cov == cov1_24.T + cov2_24.T).all()

def test_multipole_covariance_composition_is_correct():
    rng = np.random.default_rng(5)
    cov = base.MultipoleMultiTracerCovariance()
    # set k-bins: kmin=0.0, kmax=0.3, dk=0.1 -> 3 bins
    kb = 3

    # build two ells -> 0 and 2
    b00 = rng.random((kb, kb))
    b02 = rng.random((kb, kb))
    b22 = rng.random((kb, kb))

    cov.set_ell_tracer_cov(0, 0, 0, 0, b00)
    cov.set_ell_tracer_cov(0, 2, 0, 0, b02)
    cov.set_ell_tracer_cov(2, 2, 0, 0, b22)

    full = cov.cov
    assert full.shape == (2 * kb, 2 * kb)

    # reconstruct full matrix manually
    manual = np.zeros((2 * kb, 2 * kb))
    manual[0:kb, 0:kb] = b00
    manual[0:kb, kb:2*kb] = b02
    manual[kb:2*kb, 0:kb] = b02.T
    manual[kb:2*kb, kb:2*kb] = b22

    print(full[0:kb, kb:2*kb], "\n")
    print(manual[0:kb, kb:2*kb])
    print(manual[kb:2*kb, 0:kb])

    assert np.allclose(full, manual)

def test_sparse_ndarray_basic():
    # shape_out: (2, ), shape_in: (3, ) => dense shape (2,3)
    s = base.SparseNDArray((2,), (3,))
    # set an element at (0, 1)
    s[0, 1] = 5.0
    dense = s.to_dense()
    assert dense.shape == (2, 3)
    assert dense[0, 1] == 5.0

@pytest.mark.mpi(min_size=2)
def test_sparse_ndarray_to_shared_memory():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # shape_out: (4, ), shape_in: (4, ) => dense shape (4,4)
    s = base.SparseNDArray((4,), (4,))
    s[1, 2] = 3.0
    s[3, 0] = 7.0

    # test that only rank 0 has the correct dense representation before to_shared_memory
    comm.Barrier()
    if rank != 0:
        dense = s.to_dense()
        assert dense.shape == (4, 4)
        assert dense[1, 2] == 0.0
        assert dense[3, 0] == 0.0

    comm.Barrier()
    s_shared = s.to_shared_memory()

    dense_shared = s_shared.to_dense()
    assert dense_shared.shape == (4, 4)
    assert dense_shared[1, 2] == 3.0
    assert dense_shared[3, 0] == 7.0

    # test that even after deleting the old s, we can still access the shared memory version
    del s
    dense_shared = s_shared.to_dense()
    assert dense_shared[1, 2] == 3.0
    assert dense_shared[3, 0] == 7.0

def test_save_load_one_rank():
    rng = np.random.default_rng(4)
    a = rng.random((3, 3))
    mat = a @ a.T + np.eye(3) * 1e-3

    cov = base.Covariance(mat)

    filename = "test_cov.npy"
    cov.save(filename)

    loaded_cov = base.Covariance.load(filename)

    assert np.allclose(cov.cov, loaded_cov.cov)
    os.remove(filename)

@pytest.mark.mpi(min_size=2)
def test_save_load_multi_rank():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # First, let's define data on all ranks
    rng = np.random.default_rng(4)
    a = rng.random((3, 3))
    mat = a @ a.T + np.eye(3) * 1e-3

    cov = base.Covariance(mat)
    filename = "test_cov.npy"
    cov.save(filename)

    loaded_cov = base.Covariance.load(filename)
    assert np.allclose(cov.cov, loaded_cov.cov)

    if rank == 0:
        os.remove(filename)

    # next, let's define data only on rank 0
    if rank == 0:
        a = rng.random((3, 3))
        mat = a @ a.T + np.eye(3) * 1e-3
    else:
        mat = np.zeros((3, 3))

    cov = base.Covariance(mat)
    filename = "test_cov.npy"
    cov.save(filename) # <- should save data on rank 0

    loaded_cov = base.Covariance.load(filename) # <- should load data onto all ranks
    if rank == 0:
        assert np.allclose(cov.cov, loaded_cov.cov)
    else:
        assert not np.allclose(cov.cov, loaded_cov.cov)

    if rank == 0:
        os.remove(filename)