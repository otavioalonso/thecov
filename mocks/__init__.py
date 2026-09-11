"""Gaussian mock catalogues in a realistic window, for validating thecov end to end.

    python -m mocks.run_validation --n-mocks 300 --grid 256 --nproc 8 --out results/

Unlike every other test in the suite, this one probes the physical approximations behind the
formula (local plane-parallel; window slowly varying over a correlation length) rather than the
implementation: the mocks displace each galaxy along its own line of sight and the footprint is a
wide-angle cap with an irregular angular mask.
"""
from .field import Grid, GaussianField, cell_window
from .survey import Footprint, Catalogues, make_grid, make_mock, model_multipoles
from .estimator import MultipoleFields, ShellBinner, cross_multipole, shot_noise

__all__ = ['Grid', 'GaussianField', 'cell_window', 'Footprint', 'Catalogues', 'make_grid',
           'make_mock', 'model_multipoles', 'MultipoleFields', 'ShellBinner', 'cross_multipole',
           'shot_noise']
