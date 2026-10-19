"""Small tools for combining covariance terms.

window_convolve: apply the survey window to a covariance term computed in the narrow-window (local)
approximation, using the window's mode mixing as encoded in the windowed Gaussian covariance.

In the local approximation a term C_loc(k1, k2) is evaluated at sharp k-bins, i.e. as for a periodic box. The
window mixes nearby k (and multipoles): a measured multipole bin is a weighted sum of box modes, P_hat = K P_box,
so a box covariance maps to K C_box K^T. The Gaussian covariance of the same set-up is the same construction
applied to the (diagonal) box Gaussian covariance, C_G = K D K^T; in units of its own diagonal this gives the
mixing kernel K = corr(C_G)^(1/2) (the symmetric square root). This is exact for the Gaussian term by
construction and an approximation for other terms (it assumes their window mixing is that of the Gaussian
term); it matters for terms with structure on the scale of the window near the diagonal (the discreteness
4-point terms, the collapsed trispectrum), and removes their spurious bin-to-bin oscillating modes.
"""
from __future__ import annotations

import numpy as np


def mixing_kernel(C_gauss):
    """K = corr(C_gauss)^(1/2) (symmetric square root, via eigendecomposition)"""
    C = np.asarray(C_gauss, float)
    d = np.sqrt(np.diag(C))
    w, v = np.linalg.eigh(C / np.outer(d, d))
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def window_convolve(C_local, C_gauss, K=None):
    """window-convolved version of a local-approximation covariance term: D K (C_loc / D^2) K^T D with
    D = sqrt(diag C_gauss) and K = mixing_kernel(C_gauss)."""
    C_gauss = np.asarray(C_gauss, float)
    d = np.sqrt(np.diag(C_gauss))
    K = mixing_kernel(C_gauss) if K is None else K
    return np.outer(d, d) * (K @ (np.asarray(C_local) / np.outer(d, d)) @ K.T)
