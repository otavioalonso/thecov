"""thecov: covariance of power spectrum multipoles with survey geometry.

Typical use
-----------
    from thecov import Tracer, PowerSpectrum, GaussianCovariance

    lrg = Tracer('LRG', randoms_lrg, alpha=alpha_lrg)          # randoms: {'POSITION', 'WEIGHT'[, 'NZ']}
    power = PowerSpectrum({0: (k, P0), 2: (k, P2), 4: (k, P4)}, kedges=np.arange(0.0, 0.31, 0.01))
    # measured spectra: PowerSpectrum.from_estimator(files), window-convolved, with the estimator's norm and shot noise

    cov = GaussianCovariance([lrg], power)      # k bins, multipoles, normalisation from the spectrum
    cov.compute_windows([('LRG', 'LRG')])       # kernel smoothing and pair counts (geometry), once per survey
    C, labels = cov.covariance([('LRG', 'LRG')])
"""
from .tracers import Tracer, Window, spectrum_window_pairs
from .power import PowerSpectrum, BinnedHankel
from .pair_counts import TripolarWindow, WindowLibrary, PairHistogram
from .wigner import wigner_3j, gaunt_tensor, CouplingCoefficients, tri, tri_multi
from .covariance import GaussianCovariance
from .smoothing import WindowSmoothing, DensityField
from .ssc import SuperSampleCovariance
from .tree import (response_coefficients, DiscretenessCovariance, TrispectrumCovariance, Bias, galileon_bias,
                   response_split, response_covariance)

__all__ = ['Tracer', 'Window', 'spectrum_window_pairs', 'PowerSpectrum', 'BinnedHankel',
           'TripolarWindow', 'WindowLibrary', 'PairHistogram', 'wigner_3j', 'gaunt_tensor', 'CouplingCoefficients',
           'tri', 'tri_multi', 'GaussianCovariance', 'WindowSmoothing', 'DensityField',
           'SuperSampleCovariance', 'response_coefficients', 'DiscretenessCovariance',
           'TrispectrumCovariance', 'Bias', 'galileon_bias', 'response_split', 'response_covariance']
