r"""A small synthetic data set in the DESI LSS catalogue and spectrum formats, to test the comparison
pipeline end to end where the answer is known (python -m desi_validation.synthetic <out_dir>).

It mimics the conventions that matter for the covariance:
  * two caps (NGC, SGC) of a masked footprint with holes;
  * data thinned by a fibre-assignment completeness c(theta) and up-weighted by WEIGHT_COMP = 1/c,
    times a per-object WEIGHT_SYS (so WEIGHT varies object by object);
  * randoms uniform on the sky inside the footprint (a fixed number per deg^2 per file), each taking
    the redshift and the WEIGHT of a randomly chosen data object;
  * NX = nbar(z) c(theta), WEIGHT_FKP = 1 / (1 + NX P0);
  * spectra measured with jaxpower (w_tot = WEIGHT x WEIGHT_FKP, realised alpha and shot noise,
    mesh normalisation) and written as lsstypes Mesh2SpectrumPoles, one directory per mock.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from mocks.field import Grid, GaussianField
from desi_validation.desi_compare import comoving_distance

TRACER, ZRANGE, P0 = 'LRG', (0.4, 0.6), 1e4
SURFACE_DENSITY = 150.0              # randoms per deg^2 per file (DESI: 2500)
CAP_DEG = 22.0
CAPS = {'NGC': np.array([0.3, 0.2, 0.93]), 'SGC': np.array([-0.2, -0.3, -0.93])}


def nbar_z(z):
    return 5e-4 * np.exp(-0.5 * ((z - 0.5) / 0.08) ** 2)


def _unit(v):
    return v / np.linalg.norm(v)


def in_footprint(u):
    """Cap membership per region and holes, for unit vectors u (N, 3)."""
    reg = np.full(len(u), '', dtype=object)
    for name, ax in CAPS.items():
        a = _unit(ax)
        inside = u @ a > np.cos(np.radians(CAP_DEG))
        # three holes per cap
        e1 = _unit(np.cross(a, [0, 0, 1.0]) if abs(a[2]) < 0.99 else np.cross(a, [1.0, 0, 0]))
        e2 = np.cross(a, e1)
        for (x, y, r) in ((0.15, 0.05, 0.05), (-0.1, 0.18, 0.04), (0.05, -0.2, 0.06)):
            c = _unit(a + x * e1 + y * e2)
            inside &= u @ c < np.cos(r)
        reg[inside] = name
    return reg


def completeness(u):
    """Fibre-assignment completeness in [0.55, 1], varying on ~5 deg scales."""
    return 0.775 + 0.225 * np.sin(7 * u[:, 0] + 3 * u[:, 1]) * np.cos(5 * u[:, 2] + 2 * u[:, 0])


def z_of_chi(chi):
    zz = np.linspace(0, 1.2, 4001)
    return np.interp(chi, comoving_distance(zz), zz)


def radec(u):
    ra = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360
    dec = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))
    return ra, dec


def main(out, n_mocks=60, N=128, seed=0, write_catalog_mock=0, random_z='data', fresh_randoms=False, box_factor=1.0,
         regions=('NGC', 'SGC', 'GCcomb'), amplitude=1.0):
    """random_z: 'data' (each random takes the redshift of a random data object, as DESI does) or 'nz'
    (redshifts drawn from the smooth n(z)). fresh_randoms: a new random catalogue for every mock (as
    the DESI mocks have) instead of one catalogue, built from the first mock, shared by all."""
    rng = np.random.default_rng(seed)
    chi_lo, chi_hi = comoving_distance([ZRANGE[0] - 0.05, ZRANGE[1] + 0.05])
    L = 2 * chi_hi * 1.05 * box_factor
    grid = Grid(np.array([-L / 2] * 3), L, N)
    x, y, z = grid.coords()
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    u = np.stack(np.broadcast_arrays(x / r, y / r, z / r), -1).reshape(-1, 3)
    rr = np.broadcast_to(r, (N,) * 3).ravel()
    zc = z_of_chi(rr)
    reg = in_footprint(u)
    comp = completeness(u)
    lam_true = np.where((reg != '') & (zc > ZRANGE[0]) & (zc < ZRANGE[1]), nbar_z(zc), 0.0)
    # randoms: uniform on the sky within the footprint, z and WEIGHT from a random data object
    area_sr = {}
    for name in CAPS:
        # Monte-Carlo solid angle of each cap with its holes
        v = rng.normal(size=(2_000_000, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
        area_sr[name] = 4 * np.pi * np.mean(in_footprint(v) == name)
    def make_randoms(data):
        out = {}
        for name in CAPS:
            n = rng.poisson(SURFACE_DENSITY * area_sr[name] * (180 / np.pi) ** 2)
            v = np.zeros((0, 3))
            while len(v) < n:
                w = rng.normal(size=(4 * n + 1000, 3)); w /= np.linalg.norm(w, axis=1)[:, None]
                v = np.vstack([v, w[in_footprint(w) == name]])
            v = v[:n]
            dsel = data[name]
            donor = rng.integers(0, len(dsel['Z']), n)
            ra, dec = radec(v)
            zr = dsel['Z'][donor]
            if random_z == 'nz':      # smooth n(z) chi^2 dchi/dz, by inverse CDF
                zg = np.linspace(*ZRANGE, 4001)
                pdf = nbar_z(zg) * comoving_distance(zg) ** 2 * np.gradient(comoving_distance(zg), zg)
                cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(zg))])
                zr = np.interp(rng.random(n), cdf / cdf[-1], zg)
            nx = nbar_z(zr) * completeness(v)
            out[name] = dict(RA=ra, DEC=dec, Z=zr, WEIGHT=dsel['WEIGHT'][donor],
                             WEIGHT_COMP=dsel['WEIGHT_COMP'][donor], WEIGHT_SYS=dsel['WEIGHT_SYS'][donor],
                             NX=nx, WEIGHT_FKP=1 / (1 + nx * P0))
        return out

    def make_data(m):
        rngm = np.random.default_rng(1000 + m)
        f = GaussianField(grid, lambda k: amplitude * 2.5e4 * (k / 0.05) / (1 + (k / 0.05) ** 2) ** 2, rngm)
        d = f.delta().ravel()
        lam = lam_true * (1 + 1.8 * np.clip(d, -0.55, None)) * comp * grid.V_cell
        cnt = rngm.poisson(lam)
        idx = np.repeat(np.flatnonzero(cnt), cnt[cnt > 0])
        pos = (np.stack(np.unravel_index(idx, (N,) * 3), 1) + rngm.random((len(idx), 3))) * grid.cell + grid.box_min
        rp = np.linalg.norm(pos, axis=1)
        up = pos / rp[:, None]
        zp = z_of_chi(rp)
        regp = in_footprint(up)
        ok = (regp != '') & (zp > ZRANGE[0]) & (zp < ZRANGE[1])
        out = {}
        for name in CAPS:
            s = ok & (regp == name)
            c = completeness(up[s])
            wsys = rngm.gamma(25.0, 1 / 25.0, s.sum())           # per-object, mean 1, rms 0.2
            ra, dec = radec(up[s])
            nx = nbar_z(zp[s]) * c
            out[name] = dict(RA=ra, DEC=dec, Z=zp[s], WEIGHT_COMP=1 / c, WEIGHT_SYS=wsys,
                             WEIGHT=wsys / c, NX=nx, WEIGHT_FKP=1 / (1 + nx * P0))
        return out

    import h5py
    def write_cat(fn, cat):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with h5py.File(fn, 'w') as f:
            g = f.create_group('LSS')
            for k_, v_ in cat.items():
                g.create_dataset(k_, data=np.asarray(v_))

    cat_dir = os.path.join(out, 'catalogs')
    spec_dir = os.path.join(out, 'spectra')
    from jaxpower import (MeshAttrs, BinMesh2SpectrumPoles, ParticleField, FKPField, compute_mesh2_spectrum,
                          compute_fkp2_normalization, compute_fkp2_shotnoise)
    from desi_validation.desi_compare import sky_to_cartesian
    ma = MeshAttrs(meshsize=N, boxsize=L, boxcenter=0.0)
    b = BinMesh2SpectrumPoles(ma, edges={'step': 0.001, 'max': 0.2}, ells=(0, 2, 4))
    randoms = None
    for m in range(n_mocks):
        data = make_data(m)
        if randoms is None or fresh_randoms:    # shared: built from the first mock's redshifts
            randoms = make_randoms(data)
        if m == write_catalog_mock:
            for name in CAPS:
                write_cat(os.path.join(cat_dir, f'{TRACER}_{name}_clustering.dat.h5'), data[name])
                write_cat(os.path.join(cat_dir, f'{TRACER}_{name}_0_clustering.ran.h5'), randoms[name])
        for name in regions:
            names = list(CAPS) if name == 'GCcomb' else [name]
            dd = {k_: np.concatenate([data[n][k_] for n in names]) for k_ in ('RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP')}
            rr_ = {k_: np.concatenate([randoms[n][k_] for n in names]) for k_ in ('RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP')}
            wd = dd['WEIGHT'] * dd['WEIGHT_FKP']
            wr = rr_['WEIGHT'] * rr_['WEIGHT_FKP']
            if name == 'GCcomb':      # renormalise each cap's randoms to the global alpha
                a_g = wd.sum() / wr.sum()
                off = 0
                for n in names:
                    nr = len(randoms[n]['Z']); ndd = data[n]['Z']
                    a_n = (data[n]['WEIGHT'] * data[n]['WEIGHT_FKP']).sum() / (randoms[n]['WEIGHT'] * randoms[n]['WEIGHT_FKP']).sum()
                    wr[off:off + nr] *= a_n / a_g
                    off += nr
            fkp = FKPField(ParticleField(sky_to_cartesian(dd['RA'], dd['DEC'], dd['Z']), wd, attrs=ma),
                           ParticleField(sky_to_cartesian(rr_['RA'], rr_['DEC'], rr_['Z']), wr, attrs=ma))
            mesh = fkp.paint(resampler='tsc', interlacing=2, compensate=True, out='complex')
            sp = compute_mesh2_spectrum(mesh, bin=b, los='firstpoint')
            sp = sp.clone(norm=compute_fkp2_normalization(fkp, bin=b), num_shotnoise=compute_fkp2_shotnoise(fkp, bin=b))
            fn = os.path.join(spec_dir, f'mock{m}',
                              f'mesh2_spectrum_poles_{TRACER}_z{ZRANGE[0]}-{ZRANGE[1]}_{name}_weight-default-FKP.h5')
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            sp.write(fn)
        if m % 10 == 0:
            print(f'mock {m}: N_data NGC {len(data["NGC"]["Z"])}, SGC {len(data["SGC"]["Z"])}', flush=True)
    return dict(cat_dir=cat_dir, spec_dir=spec_dir, L=L, N=N, surface_density=SURFACE_DENSITY)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('out'); p.add_argument('n_mocks', type=int, nargs='?', default=60)
    p.add_argument('--random-z', default='data', choices=['data', 'nz'])
    p.add_argument('--fresh-randoms', action='store_true')
    p.add_argument('--regions', default='NGC,SGC,GCcomb')
    p.add_argument('--amplitude', type=float, default=1.0)
    p.add_argument('--N', type=int, default=128); p.add_argument('--box-factor', type=float, default=1.0)
    a = p.parse_args()
    print(main(a.out, n_mocks=a.n_mocks, N=a.N, random_z=a.random_z, fresh_randoms=a.fresh_randoms,
               box_factor=a.box_factor, regions=a.regions.split(','), amplitude=a.amplitude))
