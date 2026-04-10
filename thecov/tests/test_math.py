import numpy as np
import thecov.math as math
import thecov.utils as utils
import thecov.binning as binning
import pytest

@pytest.mark.parametrize("kmin, kmax, num_kbins, k_shell_approx, binning_type", [
    (0.01, 1.0, 10, 0.05, "linear"),
    (0.01, 1.0, 10, 0.05, "log"),
])
def test_sample_kmodes(kmin, kmax, num_kbins, k_shell_approx, binning_type):
    if binning_type == "linear":
        k_binning = binning.LinearBinning(kmin, kmax, kbins=num_kbins)
    elif binning_type == "log":
        k_binning = binning.LogBinning(kmin, kmax, kbins=num_kbins)

    assert num_kbins == k_binning.kbins

    boxsize = 1000.0
    kmodes_sampled = 50
    kfun = 2 * np.pi / boxsize

    kmodes, Nmodes, weights = math.sample_kmodes(k_binning,
                                                 boxsize=boxsize,
                                                 max_modes=kmodes_sampled,
                                                 k_shell_approx=k_shell_approx,
                                                 sample_mode="monte-carlo")

    kedges = k_binning.kedges / kfun
    # check that the number of modes in each bin is correct
    # and that each mode's magnitude falls within the correct bin edges
    for i in range(k_binning.kbins):
        assert kmodes[i].shape[0] == int(min(Nmodes[i], kmodes_sampled))
        shell_magnitudes = kmodes[i][:, 3]
        assert np.all(shell_magnitudes >= kedges[i])
        assert np.all(shell_magnitudes < kedges[i+1])

    assert len(Nmodes) == num_kbins
    assert len(weights) == num_kbins


def test_evaluate_Ylms_matches_get_real_Ylm():
    pk_ellmax = 4
    Y_table = math.build_Ylm_table(pk_ellmax)

    # Expect rows for l=0,2,4
    assert len(Y_table) == 3
    # Row lengths should be 1,3,5 respectively
    assert [len(r) for r in Y_table] == [1, 3, 5]

    # sample three directions (unit vectors) as arrays
    kxh = np.array([1.0, 0.0, 0.0])
    kyh = np.array([0.0, 1.0, 0.0])
    kzh = np.array([0.0, 0.0, 1.0])

    evaluated = math.evaluate_Ylms(Y_table, pk_ellmax, kxh, kyh, kzh)

    # Compare each entry to calling the corresponding get_real_Ylm directly
    for (l, m) in utils.ellmiter(pk_ellmax, 1):
        l_idx = l // 2
        m_idx = (m + l) // 2
        direct = math.get_real_Ylm(l, m)(kxh, kyh, kzh)
        via_table = evaluated[l_idx][m_idx]
        assert np.allclose(via_table, np.array(direct))

def test_evaluate_Ylms_matches_expected():
    pk_ellmax = 4
    Y_table = math.build_Ylm_table(pk_ellmax)

    # sample three directions (unit vectors) as arrays
    kxh = np.array([1.0, 0.0, 0.0])
    kyh = np.array([0.0, 1.0, 0.0])
    kzh = np.array([0.0, 0.0, 1.0])

    evaluated = math.evaluate_Ylms(Y_table, pk_ellmax, kxh, kyh, kzh)

    # For l=0,m=0 the real Ylm should be a constant (1/sqrt(4pi))
    assert np.allclose(evaluated[0][0], 1/np.sqrt(4*np.pi))

    # For l=2,m=0 the real Ylm should be proportional to (3*cos^2(theta)-1)
    # where cos(theta) = kzh for our unit vector along z-axis
    expected_l2m0 = np.sqrt(5/(16*np.pi)) * (3*kzh**2 - 1)
    assert np.allclose(evaluated[1][1], expected_l2m0)


def test_k_binning_properties():
    kmin = 0.1
    kmax = 1.0
    dk = (kmax - kmin) / 10
    num_kbins = 10

    linear_binning = binning.LinearBinning(kmin, kmax, dk)
    assert linear_binning.kbins == num_kbins
    assert np.allclose(linear_binning.kedges, np.linspace(kmin, kmax, num_kbins + 1))
    assert np.allclose(linear_binning.kmid, np.linspace(kmin+linear_binning.dk/2, kmax-linear_binning.dk/2, num_kbins))

    log_binning = binning.LogBinning(kmin, kmax, kbins=num_kbins)
    assert log_binning.kbins == num_kbins
    # kedges should be geometrically spaced including both endpoints
    assert np.allclose(log_binning.kedges, np.geomspace(kmin, kmax, num_kbins + 1))
    # kmid should be geometric midpoints (sqrt of edge products)
    assert np.allclose(log_binning.kmid, np.sqrt(log_binning.kedges[:-1] * log_binning.kedges[1:]))