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
        k_binning = binning.LinearBinning(kmin, kmax, num_kbins=num_kbins)
    elif binning_type == "log":
        k_binning = binning.LogBinning(kmin, kmax, num_kbins=num_kbins)

    boxsize = 1000.0
    kmodes_sampled = 50
    kmodes, Nmodes, weights = math.sample_kmodes(k_binning,
                                                 boxsize=boxsize,
                                                 max_modes=kmodes_sampled,
                                                 k_shell_approx=k_shell_approx,
                                                 sample_mode="monte-carlo")
    
    # check that the number of modes in each bin is correct
    for i in range(num_kbins):
        assert kmodes[i].shape[0] == int(min(Nmodes[i], kmodes_sampled))
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
