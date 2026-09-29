r"""The mock multipoles computed with jaxpower itself (optional dependency).

thecov is meant to be used with pypower / jaxpower, so the validation should measure the mocks
with the same code. Their conventions, which the native estimator (estimator.py) reproduces:

* alpha = sum w_data / sum w_randoms, REALISED per catalogue (this imposes the integral constraint:
  the k = 0 mode of the FKP field vanishes);
* shot noise = sum_g w_g^2 + alpha^2 sum_r w_r^2, REALISED (removes every self-pair; with the
  expected value instead, its Poisson fluctuation adds a fully correlated ~ int nbar w^4 / I^2 to
  the monopole-auto blocks that no Gaussian formula contains);
* local line of sight at the first field ('firstpoint'), which carries the Legendre weight;
* normalisation: either jaxpower's default (data x randoms painted on a 10 Mpc/h mesh, realised)
  or a fixed value (`norm='fixed'`, the int nbar^2 w^2 of the mock survey, as thecov computes it).
"""
from __future__ import annotations

import numpy as np


def available() -> bool:
    try:
        import jaxpower  # noqa: F401
        return True
    except ImportError:
        return False


class JaxpowerEstimator:
    """Multipoles of several tracers on the mock grid, with jaxpower."""

    def __init__(self, grid, k_edges, ells, scheme='tsc', interlace=True):
        from jaxpower import MeshAttrs, BinMesh2SpectrumPoles
        self.grid = grid
        self.ells = tuple(ells)
        self.scheme = scheme
        self.interlacing = 2 if interlace else 0
        self.mattrs = MeshAttrs(meshsize=grid.N, boxsize=grid.L, boxcenter=grid.box_min + 0.5 * grid.L)
        self.bin = BinMesh2SpectrumPoles(self.mattrs, edges=np.asarray(k_edges, dtype=float), ells=self.ells)

    def __call__(self, gal, gal_w, ran, ran_w, spectra, fixed_norm=None):
        """gal, gal_w, ran, ran_w: dicts by tracer. Returns (data vector, {spectrum: norm used})."""
        from jaxpower import (ParticleField, FKPField, compute_mesh2_spectrum,
                              compute_fkp2_normalization, compute_fkp2_shotnoise)
        fkp, mesh = {}, {}
        for t in gal:
            d = ParticleField(np.asarray(gal[t]), np.asarray(gal_w[t]), attrs=self.mattrs)
            r = ParticleField(np.asarray(ran[t]), np.asarray(ran_w[t]), attrs=self.mattrs)
            fkp[t] = FKPField(d, r)
            mesh[t] = fkp[t].paint(resampler=self.scheme, interlacing=self.interlacing, compensate=True,
                                   out='complex')
        out, norms = [], {}
        for (X, Y) in spectra:
            meshes = (mesh[X],) if X == Y else (mesh[X], mesh[Y])
            fk = (fkp[X],) if X == Y else (fkp[X], fkp[Y])
            spec = compute_mesh2_spectrum(*meshes, bin=self.bin, los='firstpoint')
            if fixed_norm is None:
                norm = float(np.real(compute_fkp2_normalization(*fk)))
            else:
                norm = float(fixed_norm[(X, Y)])
            sn = compute_fkp2_shotnoise(*fk, bin=self.bin)
            spec = spec.clone(norm=[norm] * len(self.ells), num_shotnoise=sn)
            norms[(X, Y)] = norm
            for ell in self.ells:
                out.append(np.real(np.asarray(spec.get(ells=ell).value())))
        return np.concatenate(out), norms
