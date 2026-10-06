"""window_convolve (thecov.covariance_tools)."""
import numpy as np

from thecov.covariance_tools import window_convolve, mixing_kernel


def test_window_convolve():
    rng = np.random.default_rng(0)
    n = 30
    Q = rng.normal(size=(n, n))
    C = Q @ Q.T / n + np.eye(n)                                # a windowed 'Gaussian' covariance
    D = np.diag(np.diag(C))
    assert np.allclose(window_convolve(D, C), C)              # a diagonal box covariance maps to C exactly
    K = mixing_kernel(C)
    assert np.allclose(K, K.T)
    X = np.outer(np.ones(n), np.ones(n))
    w = np.linalg.eigvalsh(window_convolve(X, C))
    assert w.min() > -1e-10                                    # PSD in, PSD out
