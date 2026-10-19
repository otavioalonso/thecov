"""Perturbation-theory (tree-level) model terms, and alternatives that are not used by default.

thecov's core computes what follows from the survey geometry (Gaussian term, SSC windows); the terms here need a
model of the clustering beyond the power spectrum:

  power             no-wiggle / IR-damped / FoG-dressed linear power (inputs for the terms below)
  responses         tree-level and dressed responses R_ell^(n)(k) of the multipoles to a long mode (used by the SSC)
  discreteness      B/nbar and P/nbar^2 terms in the local (tree-level) approximation
  trispectrum       tree-level connected trispectrum (T0)
  response_approach response-based T0 (collapsed term, long/hard split): under exploration
  windowing         window_convolve: window mode mixing of a term computed as for a periodic box
"""
from .power import eh_nowiggle, no_wiggle, ir_damped, fog, Dressed
from .responses import response_coefficients, response_multipoles
from .discreteness import DiscretenessCovariance
from .trispectrum import TrispectrumCovariance, Bias, galileon_bias
from .response_approach import squeezed_response, collapsed_multipoles, collapsed_covariance, response_split, response_covariance
from .windowing import window_convolve, mixing_kernel
