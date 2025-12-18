import numpy as np
import thecov.math as math
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
