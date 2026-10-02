r"""thecov against the covariance of DESI mocks: loading, weight diagnostics, tracer set-up, comparison.

Used by desi_covariance_comparison.ipynb. Everything that depends on NERSC paths is in `Paths`; every
DESI-specific convention is stated where it is used and checked on the catalogues themselves
(`weight_diagnostics`), because getting a convention wrong biases the covariance silently.

What the covariance needs from the catalogues
---------------------------------------------
The mock spectra ("weight-default-FKP") are measured with the total weight

    w_tot = WEIGHT x WEIGHT_FKP                      (data and randoms alike)

and alpha = sum_d w_tot / sum_r w_tot (pypower / jaxpower). WEIGHT is a PER-OBJECT weight
(completeness x imaging systematics x redshift failures; randoms inherit the WEIGHT of a randomly
chosen data object), not a smooth function of position. The Gaussian covariance then needs

  * the clustering window  W(x) = m(x)^2,  m(x) = E[sum_g w_tot,g delta_D(x - x_g)]
    = alpha x (random density) x (local MEAN random weight): a smooth field. Using each random's
    own weight twice instead (NZ x WEIGHT at the random itself) gives <w^2> instead of <w>^2;
  * the shot-noise window  S(x) = E[sum_g w_g^2 delta_D] + alpha^2 sum_r w_r^2 delta_D, whose
    integral is the realised shot-noise numerator of the estimator (num_shotnoise);
  * the normalisation the estimator divided by (its `norm`), since C scales as 1 / norm^2.

`build_tracer` constructs m(x) without relying on what NX means: DESI randoms are uniform on the sky
inside the footprint (a fixed surface density per random file) and take their redshifts from the
data, so their density is rho_r(z) = dN_r/dz / (Omega chi^2 dchi/dz) with Omega = N_r(all z) /
(surface density), and m = alpha x rho_r(z) x <w_tot>_local, the local mean over neighbouring
randoms (excluding the random itself). NX enters only as a cross-check.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

TRACER_SPECS = {
    'BGS': ('BGS_BRIGHT-21.35', (0.1, 0.4)),
    'LRG1': ('LRG', (0.4, 0.6)),
    'LRG2': ('LRG', (0.6, 0.8)),
    'LRG3': ('LRG', (0.8, 1.1)),
    'LRG+ELG': ('LRG+ELG_LOPnotqso', (0.8, 1.1)),
    'ELG1': ('ELG_LOPnotqso', (0.8, 1.1)),
    'ELG2': ('ELG_LOPnotqso', (1.1, 1.6)),
    'QSO': ('QSO', (0.8, 2.1)),
}

# FKP P0 of the DESI LSS catalogues (only used to cross-check WEIGHT_FKP against NX)
FKP_P0 = {'LRG': 1e4, 'ELG': 4e3, 'QSO': 6e3, 'BGS': 7e3}

DATA_COLUMNS = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP', 'NX',
                'WEIGHT_COMP', 'WEIGHT_SYS', 'WEIGHT_ZFAIL', 'FRAC_TLOBS_TILES', 'NTILE', 'TARGETID']
SLIM_RANDOM_COLUMNS = ['TARGETID', 'TARGETID_DATA', 'WEIGHT', 'NX', 'RA', 'DEC', 'Z', 'WEIGHT_FKP']


@dataclass
class Paths:
    """Where things live. Catalogues: one mock of the same release as the spectra (holi v3:
    holi_v3/altmtl{mock}/loa-v1/mock{mock}/LSScats). Tracer names differ between the catalogues and
    the spectra (e.g. ELGnotqso vs ELG_LOPnotqso): `catalog_names` maps spectra -> catalogue names
    (missing entries: same name). Random files are found by globbing, so their numbering (from 0 or
    from 1) does not matter."""
    kind: str = 'holi_v3'
    mock: int = 173
    catalog_dir: str = ('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/{kind}/altmtl{mock}/loa-v1/'
                        'mock{mock}/LSScats')
    data_name: str = '{tracer}_{region}_clustering.dat.h5'
    randoms_name: str = '{tracer}_{region}_{i}_clustering.ran.h5'
    spectra_dir: str = ('/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/'
                        'full_shape/base/holi-v3-altmtl')
    spectra_name: str = 'mesh2_spectrum_poles_{tracer}_z{zmin}-{zmax}_{region}_weight-default-FKP.h5'
    h5_group: str = 'LSS'
    catalog_names: dict = field(default_factory=lambda: {'ELG_LOPnotqso': 'ELGnotqso',
                                                         'LRG+ELG_LOPnotqso': 'LRG+ELGnotqso'})
    # Slim random catalogues (holi v3) store only TARGETID, TARGETID_DATA, WEIGHT, NX: RA/DEC come
    # from a parent random file with TARGETID, RA, DEC (template with {tracer}, {region}, {i},
    # {kind}, {mock}; {i} is the index of the slim file); Z is that of the data object TARGETID_DATA.
    random_positions: str = None
    # 'clustering_statistics' (default if importable): read the catalogues with
    # clustering_statistics.tools.read_clustering_catalog, exactly as the spectra pipeline does
    # (INDWEIGHT = WEIGHT x WEIGHT_FKP, randoms completed from the parent randoms of cs_parent_version);
    # 'files': read the h5 files above directly.
    loader: str = 'auto'
    cs_version: str = 'holi-v3-altmtl'
    cs_parent_version: str = 'data-dr2-v2'
    cs_extra: dict = field(default_factory=dict)      # further catalog options, override the fiducial ones

    def _dir(self):
        return self.catalog_dir.format(kind=self.kind, mock=self.mock)

    def catalog_tracer(self, tracer):
        return self.catalog_names.get(tracer, tracer)

    def data_fn(self, tracer, region):
        return os.path.join(self._dir(), self.data_name.format(tracer=self.catalog_tracer(tracer), region=region))

    def randoms_fn(self, tracer, region, i=0):
        return os.path.join(self._dir(), self.randoms_name.format(tracer=self.catalog_tracer(tracer), region=region, i=i))

    def randoms_fns(self, tracer, region):
        """The random files that exist, in order of their index (the {i} field of randoms_name)."""
        pre, post = self.randoms_name.split('{i}')
        pre = pre.format(tracer=self.catalog_tracer(tracer), region=region)
        idx = lambda fn: int(os.path.basename(fn)[len(pre):-len(post)])
        fns = [fn for fn in glob.glob(self.randoms_fn(tracer, region, '*'))
               if os.path.basename(fn)[len(pre):-len(post)].isdigit()]
        return sorted(fns, key=idx)

    def random_positions_fn(self, tracer, region, i):
        if self.random_positions is None:
            return None
        return self.random_positions.format(tracer=self.catalog_tracer(tracer), region=region, i=i,
                                            kind=self.kind, mock=self.mock)

    def spectra_fns(self, tracer, zrange, region):
        pattern = os.path.join(self.spectra_dir, 'mock*',
                               self.spectra_name.format(tracer=tracer, zmin=zrange[0], zmax=zrange[1],
                                                        region=region))
        return sorted(glob.glob(pattern))

    def check(self, tracer_bins=None, regions=('NGC', 'SGC')):
        """Print, per bin and region, whether the data file exists and how many random files do."""
        for b in tracer_bins or TRACER_SPECS:
            t = TRACER_SPECS[b][0]
            for r in regions:
                d = self.data_fn(t, r)
                print(f"{b:8s} {r}: data {'ok' if os.path.exists(d) else 'MISSING'} ({os.path.basename(d)}), "
                      f"{len(self.randoms_fns(t, r))} random files")


# --------------------------------------------------------------------------- reading
# alternative names of the columns used here (first match wins; matching is case-insensitive)
COLUMN_ALIASES = {'Z': ('Z', 'Z_not4clus', 'Z_RSD', 'RSDZ', 'Z_COSMO', 'REDSHIFT'),
                  'NX': ('NX', 'NZ'),
                  'RA': ('RA',), 'DEC': ('DEC',)}
REQUIRED_COLUMNS = ('RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP')


def _column_names(fn, group):
    """(reader, names): the column names of the catalogue and a function name -> array."""
    if fn.endswith('.fits'):
        import fitsio
        f = fitsio.FITS(fn)
        return (lambda c: np.asarray(f[1][c].read())), list(f[1].get_colnames())
    try:
        from mpytools import Catalog
    except ImportError:
        Catalog = None
    if Catalog is not None:
        try:
            cat = Catalog.read(fn, group=group)
        except Exception:
            cat = Catalog.read(fn)
        return (lambda c: np.asarray(cat[c])), list(cat.columns())
    import h5py
    f = h5py.File(fn, 'r')
    g = f[group] if group in f else f
    return (lambda c: np.asarray(g[c][...])), [k for k in g.keys() if isinstance(g[k], h5py.Dataset)]


def read_catalog(fn, columns=None, group='LSS', required=REQUIRED_COLUMNS):
    """dict of numpy columns: those of `columns` that exist, found under the names of COLUMN_ALIASES
    (case-insensitive) and returned under the canonical name. mpytools if available, else h5py (for
    .h5) or fitsio (for .fits). Raises KeyError, listing the file's columns, if a `required` one is
    missing."""
    read, names = _column_names(fn, group)
    lower = {n.lower(): n for n in names}
    out = {}
    for c in (list(columns) if columns is not None else names):
        for alias in COLUMN_ALIASES.get(c, (c,)):
            if alias.lower() in lower:
                out[c] = read(lower[alias.lower()])
                break
    missing = [c for c in required if c not in out]
    if missing:
        raise KeyError(f'{fn}: no column for {missing} (tried {[COLUMN_ALIASES.get(c, (c,)) for c in missing]}); '
                       f'columns are {sorted(names)}')
    return out


def comoving_distance(z, cosmo=None):
    """Comoving distance in Mpc/h with the DESI fiducial cosmology (cosmoprimo if available; else a
    direct integral with the same parameters, accurate to ~1e-4)."""
    z = np.asarray(z, dtype=float)
    try:
        from cosmoprimo.fiducial import DESI
        return np.asarray((cosmo or DESI()).comoving_radial_distance(z))
    except ImportError:
        h, omb, omc, mnu = 0.6736, 0.02237, 0.1200, 0.06
        om = (omb + omc + mnu / 93.14) / h ** 2
        orad = 4.18e-5 / h ** 2 * (1 + 0.2271 * 2.044) / (1 + 0.2271 * 3.044) * (1 + 0.2271 * 3.044)
        ol = 1 - om - orad
        zz = np.linspace(0, max(3.5, float(z.max()) * 1.01), 20001)
        Ez = np.sqrt(om * (1 + zz) ** 3 + orad * (1 + zz) ** 4 + ol)
        chi = np.concatenate([[0], np.cumsum(0.5 * (1 / Ez[1:] + 1 / Ez[:-1]) * np.diff(zz))]) * 2997.92458
        return np.interp(z, zz, chi)


def sky_to_cartesian(ra, dec, z, cosmo=None):
    chi = comoving_distance(z, cosmo)
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([chi * np.cos(dec) * np.cos(ra), chi * np.cos(dec) * np.sin(ra), chi * np.sin(dec)], axis=1)


@dataclass
class RegionCatalogs:
    """Data and randoms of one region, z-cut, with the quantities the covariance needs."""
    region: str
    data: dict
    randoms: dict
    n_randoms_all_z: int                 # randoms in the files before the z cut (-> footprint area)
    n_random_files: int
    info: dict = field(default_factory=dict)


def random_index(paths, tracer, region, fn):
    pre, post = paths.randoms_name.split('{i}')
    pre = pre.format(tracer=paths.catalog_tracer(tracer), region=region)
    return int(os.path.basename(fn)[len(pre):-len(post)])


def _lookup(keys, values_keys, values):
    """values[j] where values_keys[j] == keys[i], for every i (KeyError if one is absent)."""
    order = np.argsort(values_keys, kind='stable')
    sk = values_keys[order]
    j = np.clip(np.searchsorted(sk, keys), 0, len(sk) - 1)
    bad = sk[j] != keys
    if np.any(bad):
        raise KeyError(f'{bad.sum()} of {len(keys)} ids not found (e.g. {keys[bad][:3]})')
    return {c: v[order][j] for c, v in values.items()}


def complete_slim_randoms(rand, data, paths, tracer, region, i, group='LSS'):
    """Fill RA, DEC, Z and WEIGHT_FKP of a slim random catalogue (TARGETID, TARGETID_DATA, WEIGHT, NX):
    Z from the data object it was drawn from, RA/DEC from paths.random_positions by TARGETID,
    WEIGHT_FKP = 1 / (1 + NX P0) (checked against the data's own WEIGHT_FKP)."""
    out = dict(rand)
    info = {}
    if 'Z' not in out:
        if 'TARGETID_DATA' not in out or 'TARGETID' not in data:
            raise KeyError('slim randoms need TARGETID_DATA (randoms) and TARGETID (data) to get Z')
        out['Z'] = _lookup(out['TARGETID_DATA'], data['TARGETID'], {'Z': data['Z']})['Z']
    if 'WEIGHT_FKP' not in out:
        P0 = FKP_P0[next(t for t in FKP_P0 if tracer.startswith(t))]
        out['WEIGHT_FKP'] = 1.0 / (1.0 + np.asarray(out['NX'], float) * P0)
        dev = np.max(np.abs(data['WEIGHT_FKP'] - 1.0 / (1.0 + data['NX'] * P0)) / data['WEIGHT_FKP'])
        info['WEIGHT_FKP_formula_max_rel_dev_on_data'] = float(dev)
        if dev > 1e-3:
            print(f'WARNING: data WEIGHT_FKP != 1/(1 + NX P0) with P0={P0:g} (max rel dev {dev:.2g}); '
                  'the randoms\' WEIGHT_FKP rebuilt this way may not be those of the spectra')
    if 'RA' not in out or 'DEC' not in out:
        fn = paths.random_positions_fn(tracer, region, i)
        if fn is None:
            raise KeyError('slim random catalogue without RA/DEC: set paths.random_positions to the parent random '
                           'files (TARGETID, RA, DEC); dc.find_random_sources(paths, tracer) lists candidates')
        par = read_catalog(fn, ['TARGETID', 'RA', 'DEC'], group, required=('TARGETID', 'RA', 'DEC'))
        out.update(_lookup(out['TARGETID'], par['TARGETID'], {'RA': par['RA'], 'DEC': par['DEC']}))
        info['positions_from'] = fn
    return out, info


def find_random_sources(paths, tracer, roots=None, max_files=40):
    """List random-like files (name contains 'ran') near the catalogues, and in the DA2 LSS
    directories, with whether they hold TARGETID + RA + DEC (candidates for paths.random_positions)."""
    ct = paths.catalog_tracer(tracer)
    d = paths._dir()
    roots = roots or [d, os.path.dirname(d), os.path.dirname(os.path.dirname(d)),
                      '/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/*',
                      '/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1']
    seen = 0
    for root in roots:
        for fn in sorted(glob.glob(os.path.join(root, f'*{ct}*ran*')) + glob.glob(os.path.join(root, '*random*'))):
            if seen >= max_files:
                return
            try:
                _, names = _column_names(fn, paths.h5_group)
                up = {n.upper() for n in names}
                ok = {'TARGETID', 'RA', 'DEC'} <= up
                print(f"{'OK ' if ok else '   '} {fn}  [{', '.join(sorted(names)[:12])}{', ...' if len(names) > 12 else ''}]")
            except Exception as ex:
                print(f'    {fn}  (unreadable: {type(ex).__name__})')
            seen += 1


def _have_clustering_statistics():
    import importlib.util
    return importlib.util.find_spec('clustering_statistics') is not None


def healpix_area(ra, dec, n_per_pixel=30):
    """Footprint solid angle (sr) from the occupancy of healpix pixels by the randoms, at the resolution
    giving ~n_per_pixel randoms per occupied pixel; edge pixels are counted by their filling."""
    import healpy as hp
    nside = 1
    while True:
        npix_occ = len(np.unique(hp.ang2pix(nside * 2, ra, dec, lonlat=True)))
        if len(ra) / npix_occ < n_per_pixel:
            break
        nside *= 2
    occ = np.bincount(hp.ang2pix(nside, ra, dec, lonlat=True), minlength=hp.nside2npix(nside))
    full = np.median(occ[occ > 0])
    return float(np.sum(np.minimum(occ / full, 1.0)) * hp.nside2pixarea(nside))


def load_region_cs(paths: Paths, tracer_bin, region, n_random_files=1):
    """Data and randoms through clustering_statistics.tools.read_clustering_catalog, with the options
    of the spectra pipeline (propose_fiducial for full_shape, weight='default-FKP'). The catalogues are
    read over all redshifts (to count the randoms of the whole footprint, which sets its area) and cut
    here; if the reader refuses that, the bin's zrange is used and the area comes from healpix."""
    from clustering_statistics.tools import get_catalog_fn, read_clustering_catalog, propose_fiducial
    tracer, zr = TRACER_SPECS[tracer_bin]
    keep = ['RA', 'DEC', 'Z', 'INDWEIGHT']
    opts = dict(version=paths.cs_version, imock=paths.mock, tracer=tracer, region=region, zrange=tuple(zr),
                nran=n_random_files, concatenate=True, keep_columns=keep, weight='default-FKP') | paths.cs_extra
    opts = propose_fiducial(kind='catalog', tracer=tracer, zrange=tuple(zr), analysis='full_shape') | opts
    expand = {'parent_randoms_fn': get_catalog_fn(kind='parent_randoms', version=paths.cs_parent_version,
                                                  tracer=tracer, nran=n_random_files)}
    info = {'loader': 'clustering_statistics', 'options': {k: str(v) for k, v in opts.items()}}

    def read_any(kind, zrange):
        """keep_columns plus NX and WEIGHT if the reader provides them, else keep_columns only."""
        kw = dict(expand=expand) if kind == 'randoms' else {}
        for cols in (keep + ['NX'], keep):
            try:
                cat = read_clustering_catalog(kind=kind, **kw, **dict(opts, zrange=zrange, keep_columns=cols))
                return {c: np.asarray(cat[c]) for c in cols}
            except (KeyError, ValueError):
                if cols is keep:
                    raise

    try:
        data, randoms = read_any('data', (0.0, 10.0)), read_any('randoms', (0.0, 10.0))
        n_all = len(randoms['Z'])
    except Exception as ex:
        info['wide_zrange_failed'] = f'{type(ex).__name__}: {ex}'
        data, randoms = read_any('data', tuple(zr)), read_any('randoms', tuple(zr))
        n_all = len(randoms['Z'])
        info['omega_sr'] = healpix_area(randoms['RA'], randoms['DEC'])
        print(f'[{tracer_bin} {region}] catalogues read with the bin zrange only; footprint area from healpix '
              f'({info["omega_sr"] * (180 / np.pi) ** 2:.0f} deg^2)')

    P0 = FKP_P0[next(t for t in FKP_P0 if tracer.startswith(t))]

    def canon(cat):
        # The total weight is INDWEIGHT (= WEIGHT x WEIGHT_FKP, what the spectra use). For the
        # diagnostics only, split it with WEIGHT_FKP = 1 / (1 + NX P0) where NX > 0 (else WEIGHT_FKP = 1).
        nx = np.asarray(cat.get('NX', np.zeros_like(cat['INDWEIGHT'])), float)
        cat['WEIGHT_FKP'] = np.where(nx > 0, 1.0 / (1.0 + nx * P0), 1.0)
        cat['WEIGHT'] = cat['INDWEIGHT'] / cat['WEIGHT_FKP']
        return cat
    info['weights'] = f'INDWEIGHT; split for diagnostics with WEIGHT_FKP = 1/(1 + NX x {P0:g})'
    data, randoms = canon(data), canon(randoms)

    info.update(_pix_counts_all(randoms['RA'], randoms['DEC']))

    def cut(cat):
        m = (cat['Z'] > zr[0]) & (cat['Z'] < zr[1])
        return {c: v[m] for c, v in cat.items()}
    return RegionCatalogs(region, cut(data), cut(randoms), n_all, n_random_files, info)


def load_region(paths: Paths, tracer_bin, region, n_random_files=1, columns=DATA_COLUMNS):
    """One region (NGC or SGC) of one tracer bin: data and randoms cut to the redshift range.

    The region is given by the file; no RA/DEC region cut is applied (the example script's
    select_region is redundant on per-region files, and its SGC branch, `not (array) & (array)`,
    raises for arrays). Slim random catalogues are completed with complete_slim_randoms. With
    paths.loader 'clustering_statistics' (or 'auto' and the package importable), load_region_cs."""
    if paths.loader == 'clustering_statistics' or (paths.loader == 'auto' and _have_clustering_statistics()):
        return load_region_cs(paths, tracer_bin, region, n_random_files)
    tracer, (zmin, zmax) = TRACER_SPECS[tracer_bin]
    data = read_catalog(paths.data_fn(tracer, region), columns, paths.h5_group)
    rfns = paths.randoms_fns(tracer, region)[:n_random_files]
    if len(rfns) < n_random_files:
        raise FileNotFoundError(f'{n_random_files} random files wanted, {len(rfns)} found: '
                                f'{paths.randoms_fn(tracer, region, "*")}')
    rand, info = [], {}
    for fn in rfns:
        r = read_catalog(fn, list(dict.fromkeys(columns + SLIM_RANDOM_COLUMNS)), paths.h5_group, required=('WEIGHT',))
        if not all(c in r for c in ('RA', 'DEC', 'Z', 'WEIGHT_FKP')):
            r, inf = complete_slim_randoms(r, data, paths, tracer, region, random_index(paths, tracer, region, fn),
                                           paths.h5_group)
            info.setdefault('slim_randoms', []).append(inf)
        rand.append(r)
    common = [c for c in rand[0] if all(c in r for r in rand)]
    randoms = {c: np.concatenate([r[c] for r in rand]) for c in common}
    n_all = int(len(randoms['Z']))
    info.update(_pix_counts_all(randoms['RA'], randoms['DEC']))

    def cut(cat):
        m = (cat['Z'] > zmin) & (cat['Z'] < zmax)
        return {c: v[m] for c, v in cat.items()}
    return RegionCatalogs(region, cut(data), cut(randoms), n_all, n_random_files, info)


# --------------------------------------------------------------------------- weights
def total_weight(cat, scheme='default-FKP'):
    """The weight the spectra were measured with: 'default-FKP' = WEIGHT x WEIGHT_FKP."""
    if scheme == 'default-FKP':
        return np.asarray(cat['WEIGHT'], float) * np.asarray(cat['WEIGHT_FKP'], float)
    if scheme == 'default':
        return np.asarray(cat['WEIGHT'], float)
    if scheme == 'FKP':
        return np.asarray(cat['WEIGHT_FKP'], float)
    raise ValueError(scheme)


def weight_diagnostics(rc: RegionCatalogs, tracer_bin, scheme='default-FKP', nz_bins=20):
    """Everything about the weights and densities that the covariance depends on, as a dict of
    numbers and small tables. Nothing here is assumed; each entry is measured."""
    tracer = TRACER_SPECS[tracer_bin][0]
    d, r = rc.data, rc.randoms
    wd, wr = total_weight(d, scheme), total_weight(r, scheme)
    out = {'region': rc.region, 'N_data': len(wd), 'N_randoms': len(wr),
           'data_columns': sorted(d), 'random_columns': sorted(r)}
    # alpha conventions
    out['alpha_unweighted'] = len(wd) / len(wr)
    out['alpha_WEIGHT'] = float(np.sum(d['WEIGHT']) / np.sum(r['WEIGHT']))
    out['alpha_total'] = float(np.sum(wd) / np.sum(wr))
    # is WEIGHT the product of its components?
    comps = [c for c in ('WEIGHT_COMP', 'WEIGHT_SYS', 'WEIGHT_ZFAIL') if c in d]
    if comps:
        prod = np.prod([np.asarray(d[c], float) for c in comps], axis=0)
        out['WEIGHT_over_product_of_' + '_'.join(comps)] = (
            float(np.median(d['WEIGHT'] / prod)), float(np.std(d['WEIGHT'] / prod)))
    # FKP weights against NX
    for name, cat in (('data', d), ('randoms', r)):
        if 'NX' in cat and 'WEIGHT_FKP' in cat:
            nx = np.asarray(cat['NX'], float)
            out[f'NX_zero_fraction_{name}'] = float(np.mean(nx <= 0))
            ok = nx > 0
            p0 = (1.0 / np.asarray(cat['WEIGHT_FKP'], float)[ok] - 1.0) / nx[ok]
            out[f'P0_implied_{name}'] = (float(np.median(p0)), float(np.percentile(p0, 5)), float(np.percentile(p0, 95)))
    out['P0_expected'] = next((v for key, v in FKP_P0.items() if tracer.startswith(key)), None)
    # weight moments: the per-object spread that makes <w^2> != <w>^2
    for name, w in (('data', wd), ('randoms', wr)):
        out[f'{name}_w_mean'] = float(np.mean(w))
        out[f'{name}_w2_over_wmean2'] = float(np.mean(w ** 2) / np.mean(w) ** 2)
    for name, w in (('data', d['WEIGHT']), ('randoms', r['WEIGHT'])):
        w = np.asarray(w, float)
        out[f'{name}_WEIGHT_w2_over_wmean2'] = float(np.mean(w ** 2) / np.mean(w) ** 2)
    # shot noise: realised numerator vs what the randoms predict
    a = out['alpha_total']
    sn_real = float(np.sum(wd ** 2) + a ** 2 * np.sum(wr ** 2))
    sn_rand = float((1 + a) * a * np.sum(wr ** 2))
    out['shotnoise_numerator_realised'] = sn_real
    out['shotnoise_numerator_from_randoms'] = sn_rand
    out['shotnoise_scale'] = sn_real / sn_rand
    # per-z-bin table: data vs random weight moments, NX vs the random density
    zb = np.linspace(d['Z'].min(), d['Z'].max(), nz_bins + 1)
    tab = []
    for z0, z1 in zip(zb[:-1], zb[1:]):
        md = (d['Z'] >= z0) & (d['Z'] < z1)
        mr = (r['Z'] >= z0) & (r['Z'] < z1)
        if md.sum() < 10 or mr.sum() < 10:
            continue
        row = dict(z=0.5 * (z0 + z1), wd=float(np.mean(wd[md])), wr=float(np.mean(wr[mr])),
                   w2d=float(np.mean(wd[md] ** 2)), w2r=float(np.mean(wr[mr] ** 2)),
                   ratio_counts=float(np.sum(wd[md]) / (a * np.sum(wr[mr]))))
        if 'NX' in r:
            row['NX_mean_randoms'] = float(np.mean(r['NX'][mr]))
        tab.append(row)
    out['per_z'] = tab
    return out


PIX_NSIDE_BASE = 64


def _pix_counts_all(ra, dec, nside_base=PIX_NSIDE_BASE):
    """Random counts at all redshifts per healpix pixel (NESTED, nside_base), for the area of sky
    patches (random_density_patches). Empty if healpy is missing."""
    try:
        import healpy as hp
    except ImportError:
        return {}
    pix = hp.ang2pix(nside_base, ra, dec, lonlat=True, nest=True)
    return {'pix_nside_base': nside_base, 'pix_counts_all': np.bincount(pix, minlength=hp.nside2npix(nside_base))}


def random_density_patches(rc: RegionCatalogs, nside=8, surface_density_deg2=2500.0, dz=0.01, min_fill=0.25,
                           cosmo=None):
    """Like random_density, but with n(z) measured separately in each sky patch (healpix pixel at
    `nside`): rho_r(z, patch) = dN_patch/dz / (Omega_patch chi^2 dchi/dz), Omega_patch from the patch's
    random count at all redshifts. Captures n(z) that varies across the sky (e.g. randoms taking their
    redshifts from the data of their own imaging region), which a single rho_r(z) per cap cannot.
    Patches with less than `min_fill` of a full pixel's randoms use the cap-wide rho_r(z)."""
    import healpy as hp
    if 'pix_counts_all' not in rc.info:
        raise ValueError('no all-z healpix counts in the catalogue info (reload with healpy installed)')
    base, cnt = rc.info['pix_nside_base'], rc.info['pix_counts_all']
    f = (base // nside) ** 2
    cnt_p = cnt.reshape(-1, f).sum(1)                            # NESTED: children are contiguous
    omega_p = cnt_p / (surface_density_deg2 * rc.n_random_files) * (np.pi / 180.0) ** 2
    full = surface_density_deg2 * rc.n_random_files * hp.nside2pixarea(nside, degrees=True)
    r = rc.randoms
    pix = hp.ang2pix(nside, r['RA'], r['DEC'], lonlat=True, nest=True)
    z = r['Z']
    edges = np.arange(z.min() - 1e-9, z.max() + dz, dz)
    zb = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
    nzb = len(edges) - 1
    counts = np.bincount(pix * nzb + zb, minlength=len(cnt_p) * nzb).reshape(len(cnt_p), nzb)
    chi = comoving_distance(edges, cosmo)
    shell = (chi[1:] ** 3 - chi[:-1] ** 3) / 3.0
    with np.errstate(divide='ignore', invalid='ignore'):
        rho_p = counts / (omega_p[:, None] * shell[None, :])
    rho_glob, _ = random_density(rc, surface_density_deg2, dz=dz, cosmo=cosmo)
    good = cnt_p[pix] >= min_fill * full
    rho = np.where(good, rho_p[pix, zb], rho_glob)
    return rho, dict(n_patches=int(np.sum(cnt_p >= min_fill * full)), frac_randoms_in_good=float(good.mean()))


def nz_variation(rc, nside=8, dz=0.02, min_fill=0.5, surface_density_deg2=2500.0):
    """How much the randoms' n(z) shape varies across the sky: per patch, n_p(z) / n(z) (both normalised
    to unit integral over the bin), weighted rms over patches and z; and the same for the split at
    DEC = 32.375 deg (BASS/MzLS north vs DECaLS, relevant for NGC)."""
    import healpy as hp
    r = rc.randoms
    z = r['Z']
    edges = np.arange(z.min() - 1e-9, z.max() + dz, dz)
    zb = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
    nzb = len(edges) - 1
    glob = np.bincount(zb, minlength=nzb) / len(z)
    pix = hp.ang2pix(nside, r['RA'], r['DEC'], lonlat=True, nest=True)
    npix = hp.nside2npix(nside)
    counts = np.bincount(pix * nzb + zb, minlength=npix * nzb).reshape(npix, nzb)
    tot = counts.sum(1)
    full = surface_density_deg2 * rc.n_random_files * hp.nside2pixarea(nside, degrees=True) * len(z) / max(rc.n_randoms_all_z, 1)
    good = tot >= min_fill * full
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = counts[good] / tot[good, None] / glob[None, :]
    wts = counts[good]
    dev = np.sqrt(np.nansum(wts * (ratio - 1) ** 2) / np.sum(wts))
    # expected from Poisson noise alone
    noise = np.sqrt(np.nansum(wts / np.maximum(counts[good], 1)) / np.sum(wts))
    north = r['DEC'] > 32.375
    out = dict(rms_patch_nz_deviation=float(dev), poisson_expectation=float(noise), n_patches=int(good.sum()),
               north_fraction=float(north.mean()))
    if 0.02 < north.mean() < 0.98:
        nn = np.bincount(zb[north], minlength=nzb) / north.sum()
        ns = np.bincount(zb[~north], minlength=nzb) / (~north).sum()
        zc = 0.5 * (edges[1:] + edges[:-1])
        with np.errstate(divide='ignore', invalid='ignore'):
            out['north_over_south_nz'] = {f'{zc[i]:.3f}': float(nn[i] / ns[i]) for i in range(nzb) if ns[i] > 0}
            out['mean_z_north_minus_south'] = float(np.mean(z[north]) - np.mean(z[~north]))
    return out


def random_density(rc: RegionCatalogs, surface_density_deg2=2500.0, dz=0.005, cosmo=None):
    """Unweighted density of the randoms at each random, rho_r(z) = dN/dz / (Omega chi^2 dchi/dz),
    with Omega = (randoms in the files, all z) / (surface density per file x files). DESI randoms are
    uniform on the sky inside the footprint (2500 per deg^2 per file), so this is exact up to the
    z binning -- no nearest-neighbour density estimate, hence no edge bias."""
    omega = rc.info.get('omega_sr')
    if omega is None:
        omega = rc.n_randoms_all_z / (surface_density_deg2 * rc.n_random_files) * (np.pi / 180.0) ** 2
    z = rc.randoms['Z']
    edges = np.arange(z.min() - 1e-9, z.max() + dz, dz)
    counts, _ = np.histogram(z, edges)
    chi = comoving_distance(edges, cosmo)
    vol = omega * (chi[1:] ** 3 - chi[:-1] ** 3) / 3.0
    rho = np.where(vol > 0, counts / vol, 0.0)
    idx = np.clip(np.digitize(z, edges) - 1, 0, len(rho) - 1)
    return rho[idx], omega


def mesh_normalization(rc, cellsizes=(10.0, 5.0, 2.5, 1.25, 0.6), scheme='default-FKP', cosmo=None, offset=0.0):
    """Data x randoms and randoms x randoms in cells of side `cellsize` (sparse cell indices):
    DR = alpha sum_c D_c R_c / V_c (the pypower/jaxpower normalisation; at 10 Mpc/h it should reproduce
    the spectra's `norm`) and RR = alpha^2 sum_c (R_c^2 - sum_{r in c} w_r^2) / V_c (no self pairs).
    Both tend to int m^2 as the cells shrink if data and randoms cover the same footprint; DR / RR < 1
    in small cells means the randoms occupy places where the data cannot be (inconsistent vetoes),
    which also makes a window built from the randoms too uniform. Returns {cellsize: {'DR', 'RR'}}."""
    wd, wr = total_weight(rc.data, scheme), total_weight(rc.randoms, scheme)
    alpha = wd.sum() / wr.sum()
    pd = sky_to_cartesian(rc.data['RA'], rc.data['DEC'], rc.data['Z'], cosmo)
    pr = sky_to_cartesian(rc.randoms['RA'], rc.randoms['DEC'], rc.randoms['Z'], cosmo)
    lo = np.minimum(pd.min(0), pr.min(0)) - 1.0
    n = np.int64(1 << 20)                       # > cells per axis for any survey
    out = {}
    for cs in cellsizes:
        def keys(p):
            i = np.floor((p - lo + offset) / cs).astype(np.int64)
            return (i[:, 0] * n + i[:, 1]) * n + i[:, 2]
        ud, invd = np.unique(keys(pd), return_inverse=True)
        Dc = np.bincount(invd, weights=wd)
        ur, invr = np.unique(keys(pr), return_inverse=True)
        Rc = np.bincount(invr, weights=wr)
        R2c = np.bincount(invr, weights=wr ** 2)
        _, i_d, i_r = np.intersect1d(ud, ur, assume_unique=True, return_indices=True)
        out[float(cs)] = {'DR': float(alpha * np.sum(Dc[i_d] * Rc[i_r]) / cs ** 3),
                          'RR': float(alpha ** 2 * np.sum(Rc ** 2 - R2c) / cs ** 3)}
    return out


def local_mean_weight(pos, w, k=32):
    """Mean weight of the k nearest OTHER randoms at each random (a smooth <w>(x), unbiased by the
    random's own weight; a mean, not a density, so footprint edges do not bias it)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    out = np.empty(len(pos))
    step = 500000
    for i0 in range(0, len(pos), step):
        _, idx = tree.query(pos[i0:i0 + step], k=k + 1, workers=-1)
        out[i0:i0 + step] = w[idx[:, 1:]].mean(axis=1)
    return out


def local_mean_weight_angular(ra, dec, w, k=128, query=None, chunk=200000):
    """Mean of w over the k nearest OTHER randoms on the sky (all redshifts), at the randoms `query`
    (boolean mask or None = all). Completeness-type weights vary with sky position only, often with
    sharp boundaries (number of overlapping tiles); an angular average resolves them far better than
    a 3D one at the same number of neighbours (~0.1 deg instead of ~15 Mpc/h)."""
    from scipy.spatial import cKDTree
    ra_, dec_ = np.radians(ra), np.radians(dec)
    u = np.stack([np.cos(dec_) * np.cos(ra_), np.cos(dec_) * np.sin(ra_), np.sin(dec_)], axis=1)
    tree = cKDTree(u)
    qi = np.arange(len(u)) if query is None else np.flatnonzero(query)
    out = np.empty(len(qi))
    for i0 in range(0, len(qi), chunk):
        _, idx = tree.query(u[qi[i0:i0 + chunk]], k=k + 1, workers=-1)
        out[i0:i0 + chunk] = w[idx[:, 1:]].mean(axis=1)
    return out


def fill_map(rc, nside=512, surface_density_deg2=2500.0):
    """Fraction of each healpix pixel (RING, `nside`) inside the footprint, from the randoms' counts
    against a full pixel's expectation (surface density x files x pixel area). Veto masks smaller than
    the pixel dilute it. The randoms are cut in z, which does not depend on angle, so the expectation
    is scaled by the fraction of the randoms kept. Use many random files: the Poisson noise is
    1 / sqrt(count) per pixel (~330 randoms per nside-512 pixel with 10 files)."""
    import healpy as hp
    r = rc.randoms
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, np.asarray(r['RA'], float), np.asarray(r['DEC'], float), lonlat=True)
    expected = (surface_density_deg2 * rc.n_random_files * hp.nside2pixarea(nside, degrees=True)
                * len(r['Z']) / max(rc.n_randoms_all_z, len(r['Z'])))
    return np.bincount(pix, None, npix) / expected


def _m_values(rc, nw, sel, a_reg, rho, p, scheme='default-FKP', k_mean=32, k_ang=128, nside=8,
              surface_density_deg2=2500.0, cosmo=None, fill=None):
    """m at the randoms `sel` for a construction `nw` (see build_tracer)."""
    r = rc.randoms
    if nw == 'fill':
        # m smoothed over the veto holes: the clustering window is m(x) m(x + r) for r within a
        # correlation length, i.e. the square of m averaged over holes much smaller than r, not the
        # square of its value between the holes (W = m^2 at a point). The randoms only sample the
        # unmasked area, so alpha sum_r w_r (f m0) = int (f m0)^2: the diluted window.
        if fill is None:
            raise ValueError("nw='fill' needs a fill map (fill_map of a region with many random files)")
        import healpy as hp
        f = fill[hp.ang2pix(hp.npix2nside(len(fill)), r['RA'][sel], r['DEC'][sel], lonlat=True)]
        base = a_reg * rho[sel] * local_mean_weight(p[sel], total_weight(r, scheme)[sel], k=k_mean)
        return base * np.clip(f, 0.0, 1.0)
    if nw == 'random-density':
        return a_reg * rho[sel] * local_mean_weight(p[sel], total_weight(r, scheme)[sel], k=k_mean)
    if nw == 'patch':
        rho_p, _ = random_density_patches(rc, nside=nside, surface_density_deg2=surface_density_deg2, cosmo=cosmo)
        return a_reg * rho_p[sel] * local_mean_weight(p[sel], total_weight(r, scheme)[sel], k=k_mean)
    if nw == 'angular':
        wfkp = np.asarray(r['WEIGHT_FKP'], float)
        wa = local_mean_weight_angular(r['RA'], r['DEC'], np.asarray(r['WEIGHT'], float), k=k_ang, query=sel)
        return a_reg * rho[sel] * wa * wfkp[sel]
    if nw == 'nx':
        # NX is the completeness-weighted density (m / (NX WEIGHT_FKP) = 0.99 on holi): no smoothing at all
        if 'NX' not in r:
            raise ValueError("nw='nx' needs the randoms' NX column")
        return np.asarray(r['NX'], float)[sel] * np.asarray(r['WEIGHT_FKP'], float)[sel]
    if nw == 'none':
        return None
    raise ValueError(nw)


def window_moments(rc, modes=('random-density', 'patch', 'angular'), n_max=1_000_000, surface_density_deg2=2500.0,
                   scheme='default-FKP', seed=0, cosmo=None, **kwargs):
    """1 / V_eff = int m^4 / (int m^2)^2 for several constructions of m (integrals as sums over randoms
    of f / rho_r). The Gaussian clustering covariance scales with it; the amplitude of m cancels. Also
    'own-weight' (each random's own weight in place of a local mean: biased high by the weight
    scatter, an upper bound). Returns {mode: 1/V_eff}."""
    rng = np.random.default_rng(seed)
    r = rc.randoms
    sel = rng.random(len(r['Z'])) < min(1.0, n_max / len(r['Z']))
    wd, wr = total_weight(rc.data, scheme), total_weight(r, scheme)
    a_reg = wd.sum() / wr.sum()
    rho, _ = random_density(rc, surface_density_deg2, cosmo=cosmo)
    try:                                        # integration measure: the local random density if available
        meas, _ = random_density_patches(rc, surface_density_deg2=surface_density_deg2, cosmo=cosmo)
    except (ImportError, ValueError):
        meas = rho
        modes = [m_ for m_ in modes if m_ != 'patch']
    p = sky_to_cartesian(r['RA'], r['DEC'], r['Z'], cosmo)
    out = {}
    for mode in list(modes) + ['own-weight']:
        m = a_reg * rho[sel] * wr[sel] if mode == 'own-weight' else _m_values(
            rc, mode, sel, a_reg, rho, p, scheme, surface_density_deg2=surface_density_deg2, cosmo=cosmo, **kwargs)
        out[mode] = float(np.sum(m ** 4 / meas[sel]) / np.sum(m ** 2 / meas[sel]) ** 2 * sel.mean())
    return out


def _median_ratio(m, r, sel):
    """median of m / (NX WEIGHT_FKP) over the randoms with NX > 0 (None without NX or m)."""
    if m is None or 'NX' not in r:
        return None
    d = np.asarray(r['NX'], float)[sel] * np.asarray(r['WEIGHT_FKP'], float)[sel]
    ok = d > 0
    return float(np.median(m[ok] / d[ok])) if ok.any() else None


def build_tracer(name, regions, scheme='default-FKP', n_randoms_max=4_000_000, surface_density_deg2=2500.0,
                 k_mean=32, k_ang=128, nside=8, shotnoise='realised', nw='random-density', seed=0, cosmo=None, verbose=True,
                 shotnoise_target=None, fill=None):
    """A thecov Tracer from one or several regions (several: the combined NGC+SGC catalogue of a
    single estimate, with each region's randoms renormalised to the global alpha, as in
    catalogs.normalize_and_concatenate).

    nw        : 'random-density' (default) -- m = alpha_region x rho_r(z) x <w_tot>_local (3D, k_mean);
                'patch' -- as 'random-density' with rho_r(z) measured per sky patch (healpix `nside`),
                for n(z) that varies across the sky (random_density_patches);
                'angular' -- m = alpha_region x rho_r(z) x <WEIGHT>_sky (k_ang nearest randoms on the
                sky, all z) x the random's own WEIGHT_FKP: resolves sharp completeness boundaries;
                'nx' -- m = NX x WEIGHT_FKP at each random, no smoothing (relies on NX being the
                completeness-weighted mean density, i.e. median m / (NX WEIGHT_FKP) ~ 1, as on holi);
                'fill' -- as 'random-density' times the fraction of the local healpix pixel inside the
                footprint (`fill`: {region: fill_map}, from many random files): m averaged over
                veto holes much smaller than the correlation length, as the clustering window needs;
                'none' -- NZ = NX and each random's own weight (the naive set-up; biased by
                <w^2>/<w>^2 when WEIGHT varies per object; kept for comparison).
    shotnoise : 'realised' -- scale S so that its integral is sum_d w^2 + alpha^2 sum_r w^2 of the
                catalogue; 'randoms' -- (1 + alpha) alpha sum_r w^2.
    shotnoise_target : if given, the integral of S instead (use the spectra's mean `num_shotnoise`:
                the catalogue value depends on how many random files are loaded here, through
                alpha^2 sum_r w^2 ~ alpha sum_d w^2, while the spectra used their own number).
    Returns (tracer, info).
    """
    from thecov import Tracer
    rng = np.random.default_rng(seed)
    wsum_d = [float(np.sum(total_weight(rc.data, scheme))) for rc in regions]
    wsum_r = [float(np.sum(total_weight(rc.randoms, scheme))) for rc in regions]
    alpha_glob = sum(wsum_d) / sum(wsum_r)
    pos, w, mw, nz = [], [], [], []
    sn_real = 0.0
    info = {'regions': {}}
    n_tot = sum(len(rc.randoms['Z']) for rc in regions)
    keep_frac = min(1.0, n_randoms_max / n_tot)
    for rc, sd, sr in zip(regions, wsum_d, wsum_r):
        a_reg = sd / sr
        r = rc.randoms
        wr = total_weight(r, scheme)
        wd = total_weight(rc.data, scheme)
        sn_real += float(np.sum(wd ** 2) + a_reg ** 2 * np.sum(wr ** 2))
        p = sky_to_cartesian(r['RA'], r['DEC'], r['Z'], cosmo)
        rho, omega = random_density(rc, surface_density_deg2, cosmo=cosmo)
        sel = rng.random(len(wr)) < keep_frac          # subsample: m is unchanged, alpha rescales
        m = _m_values(rc, nw, sel, a_reg, rho, p, scheme, k_mean=k_mean, k_ang=k_ang, nside=nside,
                      surface_density_deg2=surface_density_deg2, cosmo=cosmo,
                      fill=None if fill is None else fill[rc.region])
        # renormalise this region's random weights to the global alpha (only matters for >1 region)
        wsc = wr[sel] * (a_reg / alpha_glob)
        pos.append(p[sel]); w.append(wsc)
        if m is not None:
            mw.append(m)
        nz.append(np.asarray(r['NX'], float)[sel] if 'NX' in r else np.zeros(sel.sum()))
        info['regions'][rc.region] = dict(alpha=a_reg, omega_deg2=omega * (180 / np.pi) ** 2,
                                          n_randoms_used=int(sel.sum()), n_randoms=len(wr),
                                          median_m_over_NXwFKP=_median_ratio(m, r, sel))
    pos, w, nz = np.concatenate(pos), np.concatenate(w), np.concatenate(nz)
    # the subsample carries ~keep_frac of the weight; renormalise exactly: alpha sum_r w = sum_d w
    alpha = sum(wsum_d) / float(np.sum(w))
    randoms = {'POSITION': pos, 'WEIGHT': w}
    if nw == 'none':
        randoms['NZ'] = nz
    else:
        randoms['NW'] = np.concatenate(mw)
    sn_rand = (1 + alpha) * alpha * float(np.sum(w ** 2))
    scale = sn_real / sn_rand if shotnoise == 'realised' else 1.0
    if shotnoise_target is not None:
        scale = float(shotnoise_target) / sn_rand
    tr = Tracer(name, randoms, alpha, shotnoise_scale=scale)
    info.update(alpha=alpha, shotnoise_scale=scale, shotnoise_numerator=sn_real, keep_frac=keep_frac,
                nw=nw, n_randoms=len(w))
    if verbose:
        print(f"[{name}] {len(w)} randoms (fraction {keep_frac:.3f}), alpha={alpha:.4g}, "
              f"shot-noise scale {scale:.4f}, NW from '{nw}'")
    return tr, info


# --------------------------------------------------------------------------- spectra
def read_spectra(fns, kmax=0.4, rebin=5, ells=(0, 2, 4), kmin=0.0):
    """Mock spectra as the example script selects them (every `rebin` bins, kmin <= k <= kmax).
    Bins with k L <~ 1 (the first few) are outside the validity of any windowed Gaussian
    covariance (it assumes k >> 1/L) and can make it non-positive-definite: use kmin ~ 0.02 as the
    DESI fits do. Returns
    dict(vectors (N_mock, n), k, k_edges, ells, norm (N_mock,), shotnoise (N_mock,),
    num_shotnoise (N_mock,), nmodes)."""
    import lsstypes as types
    vecs, norms, sns, nsns = [], [], [], []
    k = edges = nmodes = None
    for fn in fns:
        s = types.read(fn)
        if rebin and rebin > 1:
            s = s.select(k=slice(0, None, rebin))
        s = s.select(k=(kmin, kmax))
        vecs.append(np.concatenate([np.real(np.asarray(s.get(ell).value())) for ell in ells]))
        p0 = s.get(ells[0])
        norms.append(float(np.ravel(p0.values('norm'))[0]))
        sns.append(float(np.ravel(p0.values('shotnoise'))[0]))
        nsns.append(float(np.ravel(p0.values('num_shotnoise'))[0]))
        if k is None:
            k = np.asarray(p0.coords('k'))
            e = np.asarray(p0.edges('k'))
            edges = np.concatenate([e[:, 0], e[-1:, 1]])
            nmodes = np.asarray(p0.values('nmodes'))
    return dict(vectors=np.array(vecs), k=k, k_edges=edges, ells=tuple(ells), norm=np.array(norms),
                shotnoise=np.array(sns), num_shotnoise=np.array(nsns), nmodes=nmodes, files=list(fns))


def model_from_mocks(spec, scale=1.0):
    """P_l(k) for thecov from the mean of the mocks (window-convolved, shot noise subtracted: an
    approximation at low k) times `scale`, on a grid that reaches k = 0 and the last bin edge.

    The estimator divides |F|^2 by its own `norm`, so its mean is (int m^2 / norm) x the (window-
    convolved) P; the power spectrum thecov needs is the mean times norm / int m^2. The two differ by
    the smoothing of the mesh normalisation, which is large (~20%) for a footprint with fine veto
    masks, and the clustering terms of the covariance go as its square."""
    nb = len(spec['k'])
    mean = spec['vectors'].mean(axis=0)
    kk = np.concatenate([[0.0], spec['k'], [spec['k_edges'][-1]]])
    poles = {}
    for i, ell in enumerate(spec['ells']):
        p = scale * mean[i * nb:(i + 1) * nb]
        poles[ell] = (kk, np.concatenate([[p[0]], p, [p[-1]]]))
    return poles


def thecov_covariance(tracer, spec, n_sub=20000, n_near=1_000_000, ds=2.0, ds_pair=10.0, s_split=80.0,
                      L_max=4, norm=None, windows_file=None, verbose=True, model_norm_correction=True, **kwargs):
    """thecov's matrix for the data vector of `spec` (same k bins and ells), normalised by the
    estimator's `norm` (default: the mean over the mocks). The model is the mocks' mean, times
    norm / int m^2 if model_norm_correction (see model_from_mocks). Returns (C, cov)."""
    from thecov import GaussianCovariance, PowerSpectrumModel
    name = tracer.name
    cov = GaussianCovariance([tracer], spec['k_edges'], ells=spec['ells'], L_max=L_max, ds=ds, ds_pair=ds_pair,
                             shot_noise=True, n_sub=n_sub, n_near=n_near, s_split=s_split, **kwargs)
    sp = [(name, name)]
    if windows_file and os.path.exists(windows_file):
        cov.load_windows(windows_file)
    else:
        cov.compute_windows(sp, verbose=verbose)
        if windows_file:
            cov.save_windows(windows_file)
    norm = float(np.mean(spec['norm'])) if norm is None else float(norm)
    I_thecov = cov.I(name, name)
    model = PowerSpectrumModel()
    model.add((name, name), model_from_mocks(spec, scale=norm / I_thecov if model_norm_correction else 1.0))
    cov.set_model(model)
    cov.set_normalization(name, name, norm)
    C, _ = cov.covariance(sp, ells=spec['ells'])
    cov.I_randoms = I_thecov
    return C, cov


def suggest_n_near(tracer, s_split=80.0, target_pairs=2e9):
    """n_near giving ~target_pairs near pairs: pairs ~ n_near^2 V(s_split) / V_survey, with V_survey
    = N_randoms / (mean random density) estimated from the weights' sampling identity."""
    from scipy.spatial import cKDTree
    sub = tracer.pos[:: max(1, tracer.size // 20000)]
    d, _ = cKDTree(tracer.pos).query(sub, k=33)
    rho = 32 / (4 / 3 * np.pi * d[:, -1] ** 3)
    V = tracer.size / np.median(rho)
    vs = 4 / 3 * np.pi * s_split ** 3
    return int(np.clip(np.sqrt(target_pairs * V / vs), 2e5, tracer.size)), V


# --------------------------------------------------------------------------- comparison
def eigen_directions(C, C_mock, n=6):
    """Eigenvalues of thecov's correlation matrix (smallest first) and the mock/thecov variance ratio
    along each eigenvector. Near-null directions (eigenvalue << 1: narrow bins, strongly correlated
    by the window) are where unmodelled effects show up first and dominate the full-vector chi^2."""
    D = np.sqrt(np.diag(C))
    e, v = np.linalg.eigh(C / np.outer(D, D))
    ratio = np.einsum('ij,ik,kj->j', v, C_mock / np.outer(D, D), v) / e
    return e[:n], ratio[:n], v[:, :n]


def compare(C, C_mock, vectors, nb, ells, kmax_list=(0.1, 0.2, 0.3, 0.4), k=None):
    """chi^2 of the mocks with C (whole vector and per k_max), per-multipole chi^2, variance ratios,
    eigenvalue range vs Marchenko-Pastur. Returns a dict."""
    N, n = vectors.shape
    mean = vectors.mean(0)
    out = {'N_mock': N, 'n': n}

    def chi2(idx):
        Cs = C[np.ix_(idx, idx)]
        try:
            L = np.linalg.cholesky(Cs)
        except np.linalg.LinAlgError:
            raise ValueError('covariance not positive definite (min eigenvalue/max = '
                             f'{np.linalg.eigvalsh(Cs)[0] / np.linalg.eigvalsh(Cs)[-1]:.2g}): '
                             'raise kmin (bins with k L <~ 1) or the pair-count sampling') from None
        z = np.linalg.solve(L, (vectors[:, idx] - mean[idx]).T)
        m = np.sum(z ** 2, axis=0).mean()
        e = len(idx) * (1 - 1 / N)
        return m / e, (m - e) / (e * np.sqrt(2 / (len(idx) * N)))
    out['chi2_all'] = chi2(np.arange(n))
    if k is not None:
        for km in kmax_list:
            idx = np.concatenate([np.arange(i * nb, i * nb + nb)[k <= km] for i in range(len(ells))])
            out[f'chi2_kmax{km}'] = chi2(idx)
            for i, ell in enumerate(ells):
                out[f'chi2_l{ell}_kmax{km}'] = chi2(np.arange(i * nb, i * nb + nb)[k <= km])
    out['var_ratio'] = np.diag(C_mock) / np.diag(C)
    L = np.linalg.cholesky(C)
    M = np.linalg.solve(L, np.linalg.solve(L, C_mock).T).T
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    q = n / (N - 1)
    out['eig'] = (float(ev.min()), float(ev.max()))
    out['mp'] = ((1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2) if q < 1 else None
    return out


def combine_regions(Cs, norms):
    """Covariance of the norm-weighted average of independent regions' spectra:
    P = sum_r n_r P_r / sum_r n_r  ->  C = sum_r n_r^2 C_r / (sum_r n_r)^2."""
    n = np.asarray(norms, float)
    return sum(ni ** 2 * Ci for ni, Ci in zip(n, Cs)) / n.sum() ** 2
