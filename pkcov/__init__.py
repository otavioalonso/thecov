"""pkcov: Gaussian covariance of windowed power-spectrum multipoles in separation space.

Typical use
-----------
    from pkcov import Tracer, PowerSpectrumModel, GaussianCovariance

    lrg = Tracer('LRG', randoms_lrg, alpha=alpha_lrg)          # randoms: {'POSITION', 'WEIGHT'[, 'NZ']}
    model = PowerSpectrumModel()
    model.add(('LRG', 'LRG'), {0: (k, P0), 2: (k, P2), 4: (k, P4)})

    cov = GaussianCovariance([lrg], k_edges=np.arange(0.0, 0.31, 0.01), ells=(0, 2, 4), L_max=4)
    cov.compute_windows([('LRG', 'LRG')])       # pair counts (geometry), once per survey
    cov.set_model(model)                        # cosmology
    C, labels = cov.covariance([('LRG', 'LRG')])
"""
from .tracers import Tracer, Window, spectrum_window_pairs
from .kernels import PowerSpectrumModel, ShellKernels
from .windows import TripolarWindow, WindowLibrary
from .wigner import wigner_3j, gaunt_tensor, CouplingCoefficients, tri, tri_multi
from .covariance import GaussianCovariance

__all__ = ['Tracer', 'Window', 'spectrum_window_pairs', 'PowerSpectrumModel', 'ShellKernels',
           'TripolarWindow', 'WindowLibrary', 'wigner_3j', 'gaunt_tensor', 'CouplingCoefficients',
           'tri', 'tri_multi', 'GaussianCovariance']
