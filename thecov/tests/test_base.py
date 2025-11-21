import numpy as np
import thecov.base
import thecov.base as base
import os

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

def test_multipole_covariance_stack_and_get():
    rng = np.random.default_rng(2)
    # small 2x2 blocks
    b00 = rng.random((2, 2))
    b02 = rng.random((2, 2))
    b20 = rng.random((2, 2))
    b22 = rng.random((2, 2))

    m = base.MultipoleCovariance()

    m.set_ell_cov(0, 0, b00)
    m.set_ell_cov(0, 2, b02)
    m.set_ell_cov(2, 0, b20)
    m.set_ell_cov(2, 2, b22)

    # verify that the stacked matrix contains the blocks from get_ell_cov
    full = m.cov
    ells1, ells2 = m.ells

    # compute block sizes by inspecting diagonal blocks when present
    sizes1 = [m.get_ell_cov(l1, l1).cov.shape[0] for l1 in ells1]
    sizes2 = [m.get_ell_cov(l2, l2).cov.shape[1] for l2 in ells2]

    rstart = 0
    for i, l1 in enumerate(ells1):
        rend = rstart + sizes1[i]
        cstart = 0
        for j, l2 in enumerate(ells2):
            cend = cstart + sizes2[j]
            block = full[rstart:rend, cstart:cend]
            expected_block = m.get_ell_cov(l1, l2).cov
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

    cov.set_ell_cov(0, 0, b00)
    cov.set_ell_cov(0, 2, b02)
    cov.set_ell_cov(2, 0, b20)
    cov.set_ell_cov(2, 2, b22)

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
            expected_block = cov.get_ell_cov(l1, l2).cov
            assert np.allclose(block, expected_block)

def test_multipole_covariance_symmetrization():
    cov00, cov22, cov44, cov02, cov04, cov24 = np.random.rand(6, 100, 100)

    cov = base.MultipoleCovariance()

    cov.set_ell_cov(0, 0, cov00)
    cov.set_ell_cov(2, 2, cov22)
    cov.set_ell_cov(4, 4, cov44)

    cov.set_ell_cov(0, 2, cov02)
    cov.set_ell_cov(0, 4, cov04)
    cov.set_ell_cov(4, 2, cov24.T)

    assert (cov.get_ell_cov(0,2).cov == cov02).all()
    assert (cov.get_ell_cov(2,0).cov == cov02.T).all()

    assert (cov.get_ell_cov(0,4).cov == cov04).all()
    assert (cov.get_ell_cov(4,0).cov == cov04.T).all()

    assert (cov.get_ell_cov(2,4).cov == cov24).all()
    assert (cov.get_ell_cov(4,2).cov == cov24.T).all()

    assert not (cov.get_ell_cov(0,0).cov == cov.get_ell_cov(0,0).cov.T).all()
    assert not (cov.get_ell_cov(2,2).cov == cov.get_ell_cov(2,2).cov.T).all()
    assert not (cov.get_ell_cov(4,4).cov == cov.get_ell_cov(4,4).cov.T).all()

    cov.symmetrize()

    assert (cov.get_ell_cov(0,2).cov == cov02).all()
    assert (cov.get_ell_cov(2,0).cov == cov02.T).all()

    assert (cov.get_ell_cov(0,4).cov == cov04).all()
    assert (cov.get_ell_cov(4,0).cov == cov04.T).all()

    assert (cov.get_ell_cov(2,4).cov == cov24).all()
    assert (cov.get_ell_cov(4,2).cov == cov24.T).all()

    assert (cov.get_ell_cov(0,0).cov == cov.get_ell_cov(0,0).cov.T).all()
    assert (cov.get_ell_cov(2,2).cov == cov.get_ell_cov(2,2).cov.T).all()
    assert (cov.get_ell_cov(4,4).cov == cov.get_ell_cov(4,4).cov.T).all()

    assert not (cov.get_ell_cov(0,2).cov == cov.get_ell_cov(0,2).cov.T).all()
    assert not (cov.get_ell_cov(2,4).cov == cov.get_ell_cov(2,4).cov.T).all()
    assert not (cov.get_ell_cov(4,0).cov == cov.get_ell_cov(4,0).cov.T).all()

    assert (cov.get_ell_cov(0,0).cov == (cov00 + cov00.T)/2).all()
    assert (cov.get_ell_cov(2,2).cov == (cov22 + cov22.T)/2).all()
    assert (cov.get_ell_cov(4,4).cov == (cov44 + cov44.T)/2).all()


def test_multipole_covariance_addition():
    cov1_00, cov1_22, cov1_44, cov1_02, cov1_04, cov1_24 = np.random.rand(6, 100, 100)
    cov2_00, cov2_22, cov2_44, cov2_02, cov2_04, cov2_24 = np.random.rand(6, 100, 100)

    cov1 = base.MultipoleCovariance()
    cov2 = base.MultipoleCovariance()

    cov1.set_ell_cov(0,0, cov1_00)
    cov1.set_ell_cov(2,2, cov1_22)
    cov1.set_ell_cov(4,4, cov1_44)

    cov1.set_ell_cov(0,2, cov1_02)
    cov1.set_ell_cov(0,4, cov1_04)
    cov1.set_ell_cov(4,2, cov1_24.T)

    cov2.set_ell_cov(0,0, cov2_00)
    cov2.set_ell_cov(2,2, cov2_22)
    cov2.set_ell_cov(4,4, cov2_44)

    cov2.set_ell_cov(0,2, cov2_02)
    cov2.set_ell_cov(0,4, cov2_04)
    cov2.set_ell_cov(4,2, cov2_24.T)

    addition = cov1 + cov2

    assert (addition.get_ell_cov(0,0).cov == cov1_00 + cov2_00).all()
    assert (addition.get_ell_cov(2,2).cov == cov1_22 + cov2_22).all()
    assert (addition.get_ell_cov(4,4).cov == cov1_44 + cov2_44).all()

    assert (addition.get_ell_cov(0,2).cov == cov1_02 + cov2_02).all()
    assert (addition.get_ell_cov(0,4).cov == cov1_04 + cov2_04).all()
    assert (addition.get_ell_cov(2,4).cov == cov1_24 + cov2_24).all()


# NOTE: Commented out since we will be chaning the relavent code to handle multi-tracer covariance
# def test_multipole_fourier_covariance_save_load_csv():
#     cov = thecov.base.MultipoleFourierCovariance()
#     cov.set_kbins(0., 0.4, 0.005)

#     cov00, cov22, cov44, cov02, cov04, cov24 = np.random.rand(6, cov.kbins, cov.kbins)

#     cov.set_ell_cov(0, 0, cov00)
#     cov.set_ell_cov(2, 2, cov22)
#     cov.set_ell_cov(4, 4, cov44)

#     cov.set_ell_cov(0, 2, cov02)
#     cov.set_ell_cov(0, 4, cov04)
#     cov.set_ell_cov(4, 2, cov24.T)

#     cov.savecsv('test1.txt')
#     cov.savecsv('test2.txt', ells_both_ways=True)

#     cov1 = thecov.base.MultipoleFourierCovariance.fromcsv('test1.txt')
#     cov2 = thecov.base.MultipoleFourierCovariance.fromcsv('test2.txt')

#     assert np.allclose(cov1.cov, cov.cov)
#     assert np.allclose(cov2.cov, cov.cov)

#     os.remove('test1.txt')
#     os.remove('test2.txt')


def test_sparse_ndarray_basic():
    # shape_out: (2, ), shape_in: (3, ) => dense shape (2,3)
    s = base.SparseNDArray((2,), (3,))
    # set an element at (0, 1)
    s[0, 1] = 5.0
    dense = s.to_dense()
    assert dense.shape == (2, 3)
    assert dense[0, 1] == 5.0