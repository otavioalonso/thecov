"""Reading the original NERSC files for paper_nersc.ipynb.

You should not need to open this file. The notebook calls

    s = load_holi('LRG1', 'NGC')          # holi v3 altmtl mocks, one tracer and cap
    s = load_abacus('complete', 'NGC')    # AbacusSummit LRG1 (complete or altmtl)
    nz = load_nz('LRG1', 'NGC')           # n(z) of every holi mock
    ran = load_randoms('LRG1', 'NGC')     # veto-hole maps from the randoms

and gets objects with plainly named arrays (see the classes below).

Where things come from
    mock P_ell(k)          DESI mesh2_spectrum_poles_*.h5, one per mock (read with lsstypes)
    n(z), sky area         DESI LSS data catalogues of every mock
    veto holes             randoms of holi mock 173 (DESI clustering_statistics reader)
    Gaussian covariances   thecov's output files ~/thecov_desi/<sim>/<bin>/cov_*.npz
    non-Gaussian terms     thecov's ssc_check output ssc_<label>.npz
The paths are the module variables below; the notebook can change them (paperdata.OUT = ...).

Everything read from the catalogues and spectra is kept in CACHE_DIR (one .npz per quantity), so a second session
starts in seconds. Delete a file there, or set CACHE_DIR = None, to read the originals again.
"""
from __future__ import annotations

import glob
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

HOME = os.path.expanduser('~')

# ------------------------------------------------------------------ paths of the original files
SPECTRA_BASE = '/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base'
SPECTRA_DIR = {'holi': 'holi-v3-altmtl',
               'abacus_complete': 'abacus-2ndgen-dr2-complete',
               'abacus_altmtl': 'abacus-2ndgen-dr2-altmtl'}
SPECTRUM_FILE = 'mock*/mesh2_spectrum_poles_{tracer}_z{zmin}-{zmax}_{cap}_weight-default-FKP.h5'
CATALOG_DIR = '/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/holi_v3/altmtl{mock}/loa-v1/mock{mock}/LSScats'
DATA_FILE = '{tracer}_{cap}_clustering.dat.h5'
RANDOMS_MOCK = 173
TRACERS = {'LRG1': ('LRG', 0.4, 0.6), 'QSO': ('QSO', 0.8, 2.1)}      # name -> (DESI tracer, zmin, zmax)

THECOV_CACHE = {'holi': HOME + '/thecov_desi/holi_v3_mock173',
                'abacus_complete': HOME + '/thecov_desi/abacus-2ndgen-dr2-complete_mock0',
                'abacus_altmtl': HOME + '/thecov_desi/abacus-2ndgen-dr2-altmtl_mock0'}
COV_FILE = '{cache}/{tracer}/cov_{cap}_{window}_k0.02-0.3_r{rebin}_sn1_mc1_v2.npz'
OUT = '/global/cfs/cdirs/desicollab/users/oalves/thecov_validation'
NONGAUSS_FILE = {'holi': 'ssc_holi-kcore2-{tracer}.npz', 'abacus_complete': 'ssc_abacus-kcore-complete-LRG1.npz'}
KERNEL_COV_OVERRIDE = {}      # {('holi', 'LRG1', 'NGC'): '<cov_..._r5_...npz>'} pins a xi-kernel covariance file

# ------------------------------------------------------------------ choices (those of the DESI fits / covariance runs)
KMIN, KMAX = 0.02, 0.3
FINE_BINS_PER_DK = {0.005: 5, 0.01: 10, 0.02: 20}      # the files have 0.001 h/Mpc bins
NZ_BINS = 20
N_RANDOM_FILES = 10
SURFACE_DENSITY = 2500.0          # randoms per deg^2 per file
WORKERS = 32
CACHE_DIR = os.environ.get('SCRATCH', HOME) + '/paper_nersc_cache'

sys.path.insert(0, HOME + '/thecov')


# ================================================================== what the notebook gets
@dataclass
class Sample:
    """Mocks and covariance models of one simulation / tracer / cap / bin width.

    Data vectors are P0, P2, P4 one after the other (n_k values each); use s.P0(v), s.P2(v), s.P4(v) to cut them."""
    name: str                    # e.g. 'holi LRG1 NGC'
    k: np.ndarray                # bin centres [h/Mpc]
    dk: float                    # bin width [h/Mpc]
    mocks: np.ndarray            # (n_mocks, 3 n_k)
    mock_ids: np.ndarray
    nmodes: np.ndarray           # Fourier modes per k bin
    cov_gauss: np.ndarray = None          # Gaussian, xi-kernel window (the paper's Gaussian model)
    cov_gauss_local: np.ndarray = None    # Gaussian, local window m^2
    cov_gauss_ownweight: np.ndarray = None   # Gaussian, local window with each random's own weight
    cov_ssc: np.ndarray = None            # super-sample covariance (with the local-average term)
    cov_ssc_no_local_average: np.ndarray = None
    cov_disc: np.ndarray = None           # discreteness 4-point terms
    cov_t0_tree: np.ndarray = None        # tree-level connected trispectrum (rejected by the mocks)
    cov_t0_response: np.ndarray = None    # response-based connected trispectrum (rejected by the mocks)
    b1: float = None
    f: float = None
    plin_k: np.ndarray = None             # linear P(k) used by the non-Gaussian terms (BAO-damped, normalised)
    plin: np.ndarray = None

    @property
    def cov_final(self):
        """the recommended covariance: Gaussian + SSC + discreteness (nothing fitted)"""
        return self.cov_gauss + self.cov_ssc + self.cov_disc

    @property
    def cov_mocks(self):
        """sample covariance of the mocks"""
        return np.cov(self.mocks.T)

    @property
    def n_mocks(self):
        return len(self.mocks)

    @property
    def n_k(self):
        return len(self.k)

    def P0(self, v):
        return v[..., :self.n_k]

    def P2(self, v):
        return v[..., self.n_k:2 * self.n_k]

    def P4(self, v):
        return v[..., 2 * self.n_k:]

    def ell(self, v, ell):
        return {0: self.P0, 2: self.P2, 4: self.P4}[ell](v)

    def below(self, kmax):
        """indices of the data vector with k <= kmax (all multipoles)"""
        s = np.flatnonzero(self.k <= kmax + 1e-9)
        return np.concatenate([s, s + self.n_k, s + 2 * self.n_k])


@dataclass
class NofZ:
    z_edges: np.ndarray
    counts: np.ndarray           # (n_mocks, NZ_BINS) completeness-weighted galaxy counts
    mock_ids: np.ndarray
    area_sr: float


@dataclass
class Randoms:
    pixels: np.ndarray           # occupied healpix pixels (NEST, nside below)
    counts: np.ndarray           # randoms per pixel
    expected: float              # randoms expected in a pixel with no veto
    nside: int
    scale_deg: np.ndarray        # hole statistics per angular scale (hole_fraction.py)
    R_PP: np.ndarray

    def fill_fraction_map(self, nside):
        """healpix map (RING) of the unvetoed fraction at resolution nside; UNSEEN outside the footprint"""
        import healpy as hp
        parent, inv = np.unique(self.pixels >> 2 * int(np.log2(self.nside // nside)), return_inverse=True)
        m = np.full(hp.nside2npix(nside), hp.UNSEEN)
        m[parent] = np.bincount(inv, weights=self.counts) / (self.expected * (self.nside // nside) ** 2)
        return hp.reorder(m, n2r=True)


# ================================================================== loaders
@lru_cache(None)
def load_holi(tracer, cap, dk=0.005, drop_defective=True):
    """holi v3 altmtl mocks of tracer ('LRG1', 'QSO') and cap ('NGC', 'SGC'), with every covariance model"""
    sp = _spectra('holi', tracer, cap)
    keep = ~np.isin(sp['ids'], excluded_mocks(tracer)) if drop_defective else np.ones(len(sp['ids']), bool)
    s = Sample(name=f'holi {tracer} {cap}', k=sp[f'k{dk}'], dk=dk, mocks=sp[f'V{dk}'][keep], mock_ids=sp['ids'][keep],
               nmodes=sp[f'nmodes{dk}'])
    s.cov_gauss = _cov('holi', tracer, cap, 'kernel', dk)
    s.cov_gauss_local = _cov('holi', tracer, cap, 'random-density', dk)
    if os.path.exists(_cov_file('holi', tracer, cap, 'none', dk)):
        s.cov_gauss_ownweight = _cov('holi', tracer, cap, 'none', dk)
    _add_nongauss(s, 'holi', tracer, cap, dk)
    return s


@lru_cache(None)
def load_abacus(variant, cap, dk=0.005):
    """AbacusSummit LRG1, variant 'complete' or 'altmtl'. Both have the local-window Gaussian covariance;
    'complete' also has the xi-kernel Gaussian and the non-Gaussian terms."""
    sim = 'abacus_' + variant
    sp = _spectra(sim, 'LRG1', cap)
    s = Sample(name=f'Abacus {variant} LRG1 {cap}', k=sp[f'k{dk}'], dk=dk, mocks=sp[f'V{dk}'], mock_ids=sp['ids'],
               nmodes=sp[f'nmodes{dk}'])
    s.cov_gauss_local = _cov(sim, 'LRG1', cap, 'random-density', dk)
    if variant == 'complete':
        s.cov_gauss = _cov(sim, 'LRG1', cap, 'kernel', dk)
        _add_nongauss(s, sim, 'LRG1', cap, dk)
    return s


@dataclass
class MockSpectra:
    """the measured multipoles of every mock, as the DESI pipeline wrote them"""
    name: str
    k: np.ndarray                # bin centres [h/Mpc]
    k_edges: np.ndarray
    nmodes: np.ndarray
    mocks: np.ndarray            # (n_mocks, 3 n_k): P0, P2, P4
    mock_ids: np.ndarray
    norm: np.ndarray             # each mock's estimator normalisation
    num_shotnoise: np.ndarray    # each mock's shot-noise numerator (sum_d w^2 + alpha^2 sum_r w^2)


def load_mock_spectra(sim, tracer, cap, dk=0.005, drop_defective=True):
    """sim: 'holi', 'abacus_complete' or 'abacus_altmtl'. drop_defective removes the holi mocks with a broken n(z)."""
    sp = _spectra(sim, tracer, cap)
    keep = np.ones(len(sp['ids']), bool)
    if drop_defective and sim == 'holi':
        keep = ~np.isin(sp['ids'], excluded_mocks(tracer))
    return MockSpectra(name=f'{sim} {tracer} {cap}', k=sp[f'k{dk}'], k_edges=sp[f'k_edges{dk}'], nmodes=sp[f'nmodes{dk}'],
                       mocks=sp[f'V{dk}'][keep], mock_ids=sp['ids'][keep], norm=sp[f'norm{dk}'][keep],
                       num_shotnoise=sp[f'num_shotnoise{dk}'][keep])


def load_catalogs(tracer, cap, n_random_files=1, mock=RANDOMS_MOCK):
    """data and randoms of one holi mock, cut to the tracer's redshift range (DESI reader, as the covariance runs):
    an object with .data and .randoms (dicts of columns RA, DEC, Z, WEIGHT, WEIGHT_FKP, NX)"""
    from desi_validation import desi_compare as dc
    paths = dc.Paths(kind='holi_v3', mock=mock, catalog_dir=CATALOG_DIR)
    paths.loader, paths.cs_version, paths.cs_parent_version = 'auto', 'holi-v3-altmtl', 'data-dr2-v2'
    return dc.load_region(paths, tracer, cap, n_random_files=n_random_files)


def load_nz(tracer, cap):
    """n(z) of every holi mock that has a spectrum"""
    z = _nz(tracer)
    return NofZ(z_edges=z['edges'], counts=z[f'{cap}/NZ'], mock_ids=z[f'{cap}/mocks'], area_sr=float(z[f'{cap}/area']))


def load_randoms(tracer, cap):
    """random counts at healpix nside 2048 (fill fraction) and the hole statistics R_PP"""
    r = _randoms(tracer, cap)
    return Randoms(pixels=r['pix'], counts=r['n'], expected=float(r['E']), nside=int(r['nside']),
                   scale_deg=r['scale_deg'], R_PP=r['R_PP'])


@lru_cache(None)
def excluded_mocks(tracer, threshold=6.0):
    """holi mocks whose n(z) has a bin more than `threshold` robust sigma from the median (either cap)"""
    bad = set()
    for cap in ('NGC', 'SGC'):
        nz = load_nz(tracer, cap)
        dev = nz.counts / np.median(nz.counts, axis=0) - 1
        dev = dev - np.median(dev, axis=0)
        robust_sigma = 1.4826 * np.median(np.abs(dev), axis=0)
        bad |= set(nz.mock_ids[(np.abs(dev) > threshold * robust_sigma).any(axis=1)].tolist())
    return tuple(sorted(bad))


# ================================================================== reading (details)
def _parallel_map(func, items):
    if WORKERS <= 1:
        return [func(x) for x in items]
    with multiprocessing.get_context('fork').Pool(WORKERS) as pool:
        return pool.map(func, items, chunksize=4)


def _cached(name, compute):
    fn = f'{CACHE_DIR}/{name}.npz' if CACHE_DIR else None
    if fn and os.path.exists(fn):
        with np.load(fn) as z:
            return {key: z[key] for key in z.files}
    t0 = time.time()
    out = compute()
    if fn:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(fn, **out)
    print(f'{name}: read from the original files in {time.time() - t0:.0f} s' + (f', kept in {fn}' if fn else ''))
    return out


def _read_one_spectrum(fn):
    import lsstypes
    try:
        s = lsstypes.read(fn)
        out = {}
        for dk, nfine in FINE_BINS_PER_DK.items():
            sel = s.select(k=slice(0, None, nfine)).select(k=(KMIN, KMAX))
            p0 = sel.get(0)
            edges = np.asarray(p0.edges('k'))
            out[dk] = (np.concatenate([np.real(np.asarray(sel.get(ell).value())) for ell in (0, 2, 4)]),
                       np.asarray(p0.coords('k')), np.asarray(p0.values('nmodes')),
                       float(np.ravel(p0.values('norm'))[0]), float(np.ravel(p0.values('num_shotnoise'))[0]),
                       np.concatenate([edges[:, 0], edges[-1:, 1]]))
        return out
    except Exception:
        import traceback
        return traceback.format_exc()


def _spectra(sim, tracer, cap):
    name, zmin, zmax = TRACERS[tracer]
    pattern = f'{SPECTRA_BASE}/{SPECTRA_DIR[sim]}/' + SPECTRUM_FILE.format(tracer=name, zmin=zmin, zmax=zmax, cap=cap)

    def compute():
        fns = sorted(glob.glob(pattern))
        if not fns:
            raise FileNotFoundError(pattern)
        res = _parallel_map(_read_one_spectrum, fns)
        bad = [(fn, r) for fn, r in zip(fns, res) if isinstance(r, str)]
        if bad:
            raise RuntimeError(f'{len(bad)} unreadable files, e.g. {bad[0][0]}:\n{bad[0][1]}')
        out = {'ids': np.array([int(os.path.basename(os.path.dirname(fn))[4:]) for fn in fns])}
        for dk in FINE_BINS_PER_DK:
            out[f'V{dk}'] = np.array([r[dk][0] for r in res])
            out[f'k{dk}'], out[f'nmodes{dk}'] = res[0][dk][1], res[0][dk][2]
            out[f'norm{dk}'] = np.array([r[dk][3] for r in res])
            out[f'num_shotnoise{dk}'] = np.array([r[dk][4] for r in res])
            out[f'k_edges{dk}'] = res[0][dk][5]
        return out
    return _cached(f'pk2_{sim}_{tracer}_{cap}', compute)


def _nongauss_file(sim, tracer):
    return f'{OUT}/' + NONGAUSS_FILE[sim].format(tracer=tracer)


@lru_cache(None)
def _kernel_cov_file(sim, tracer, cap):
    """several xi-kernel variants can sit in the cache: take the one equal to the Gaussian covariance that the
    non-Gaussian run used (ssc_check stores it as <cap>/C)"""
    if (sim, tracer, cap) in KERNEL_COV_OVERRIDE:
        return KERNEL_COV_OVERRIDE[sim, tracer, cap]
    pattern = COV_FILE.format(cache=THECOV_CACHE[sim], tracer=tracer, cap=cap, window=f'{tracer}_{cap}_kernel_*', rebin=5)
    with np.load(_nongauss_file(sim, tracer)) as ng:
        ref = ng[f'{cap}/C']
    match = []
    for fn in sorted(glob.glob(pattern), key=os.path.getmtime):
        with np.load(fn) as z:
            if z['C'].shape == ref.shape and np.allclose(z['C'], ref, rtol=1e-9, atol=0):
                match.append(fn)
    if not match:
        raise RuntimeError(f'no file {pattern} equals the Gaussian covariance of {_nongauss_file(sim, tracer)}: '
                           f'set paperdata.KERNEL_COV_OVERRIDE[{(sim, tracer, cap)}]')
    print(f'xi-kernel covariance {sim} {tracer} {cap}: {match[-1]}')
    return match[-1]


def kernel_tracer_name(sim, tracer, cap):
    """thecov tracer name of the cached xi-kernel run (it records the kernel settings), or None"""
    try:
        fn = os.path.basename(_kernel_cov_file(sim, tracer, cap))
    except (OSError, RuntimeError, KeyError):
        return None
    return fn[len(f'cov_{cap}_'):fn.index('_k0.02-')]


def stored_gaussian(sim, tracer, cap, window, dk=0.005):
    """the Gaussian covariance a pipeline run stored ('kernel', 'random-density' or 'none'), or None"""
    try:
        return _cov(sim, tracer, cap, window, dk)
    except (OSError, RuntimeError, KeyError):
        return None


def _cov_file(sim, tracer, cap, window, dk):
    rebin = FINE_BINS_PER_DK[dk]
    if window == 'kernel':
        return _kernel_cov_file(sim, tracer, cap).replace('_r5_', f'_r{rebin}_')
    return COV_FILE.format(cache=THECOV_CACHE[sim], tracer=tracer, cap=cap, window=window, rebin=rebin)


def _cov(sim, tracer, cap, window, dk):
    with np.load(_cov_file(sim, tracer, cap, window, dk)) as z:
        return z['C']


def _rebin_matrix(nmodes, factor):
    """R with R @ P_fine = mode-weighted average over groups of `factor` bins, for each multipole"""
    nb = len(nmodes)
    R1 = np.zeros((nb // factor, nb))
    for i in range(nb // factor):
        w = nmodes[i * factor:(i + 1) * factor]
        R1[i, i * factor:(i + 1) * factor] = w / w.sum()
    return np.kron(np.eye(3), R1)


def _add_nongauss(s, sim, tracer, cap, dk):
    """the non-Gaussian terms of ssc_check, computed at dk = 0.005 and averaged over bins for wider dk"""
    from thecov.covariance_tools import window_convolve
    ng = np.load(_nongauss_file(sim, tracer))
    factor = int(round(dk / 0.005))
    R = np.eye(len(ng[f'{cap}/C'])) if factor == 1 else _rebin_matrix(_spectra(sim, tracer, cap)['nmodes0.005'], factor)
    rb = lambda M: R @ M @ R.T                                   # noqa: E731
    s.cov_ssc = rb(ng[f'{cap}/C_ssc'])
    s.cov_ssc_no_local_average = rb(ng[f'{cap}/C_ssc_noLA'])
    s.cov_disc = rb(ng[f'{cap}/C_disc'])
    if f'{cap}/C_T0_response' in ng.files:
        s.cov_t0_response = rb(ng[f'{cap}/C_T0_response'])
    if f'{cap}/C_T0_snake' in ng.files:          # window-convolved with the Gaussian covariance, as in ssc_check
        s.cov_t0_tree = rb(window_convolve(ng[f'{cap}/C_T0_snake'] + ng[f'{cap}/C_T0_star'], ng[f'{cap}/C']))
    s.b1, s.f = float(ng[f'{cap}/params'][0]), float(ng[f'{cap}/params'][3])
    s.plin_k, s.plin = ng[f'{cap}/plin_k'], ng[f'{cap}/plin_P_damped_normalised']


def _nz_one_mock(args):
    from desi_validation import desi_compare as dc
    mock, tracer, cap, edges = args
    fn = CATALOG_DIR.format(mock=mock) + '/' + DATA_FILE.format(tracer=tracer, cap=cap)
    try:
        d = dc.read_catalog(fn, ['Z', 'WEIGHT'], required=('Z',))
    except (OSError, KeyError):
        return None
    return np.histogram(d['Z'], edges, weights=d.get('WEIGHT'))[0]


def _nz(tracer):
    from desi_validation import desi_compare as dc

    def compute():
        name, zmin, zmax = TRACERS[tracer]
        edges = np.linspace(zmin, zmax, NZ_BINS + 1)
        out = {'edges': edges}
        for cap in ('NGC', 'SGC'):
            ids = _spectra('holi', tracer, cap)['ids']
            rows = _parallel_map(_nz_one_mock, [(m, name, cap, edges) for m in ids])
            ok = np.array([r is not None for r in rows])
            if not ok.all():
                print(f'{tracer} {cap}: no catalogue for mocks {ids[~ok][:10]}')
            out[f'{cap}/mocks'] = ids[ok]
            out[f'{cap}/NZ'] = np.array([r for r in rows if r is not None], float)
            d = dc.read_catalog(CATALOG_DIR.format(mock=ids[ok][0]) + '/' + DATA_FILE.format(tracer=name, cap=cap),
                                ['RA', 'DEC'], required=('RA', 'DEC'))
            out[f'{cap}/area'] = dc.healpix_area(d['RA'], d['DEC'])
        return out
    return _cached(f'nz_{tracer}', compute)


def _randoms(tracer, cap, nside=2048, n_eval=2_000_000):
    import healpy as hp
    from desi_validation import desi_compare as dc
    from desi_validation.hole_fraction import hole_statistics

    def compute():
        paths = dc.Paths(kind='holi_v3', mock=RANDOMS_MOCK, catalog_dir=CATALOG_DIR)
        paths.loader, paths.cs_version, paths.cs_parent_version = 'auto', 'holi-v3-altmtl', 'data-dr2-v2'
        rc = dc.load_region(paths, tracer, cap, n_random_files=N_RANDOM_FILES)
        ra, dec, z = (np.asarray(rc.randoms[c], float) for c in ('RA', 'DEC', 'Z'))
        E = SURFACE_DENSITY * rc.n_random_files * hp.nside2pixarea(nside, degrees=True) * len(z) / max(rc.n_randoms_all_z, len(z))
        pix, n = np.unique(hp.ang2pix(nside, ra, dec, lonlat=True, nest=True), return_counts=True)
        rho, _ = dc.random_density(rc, SURFACE_DENSITY)
        ev = np.zeros(len(z), bool)
        ev[np.random.default_rng(0).choice(len(z), size=min(n_eval, len(z) // 10), replace=False)] = True
        w = dc.total_weight(rc.randoms)
        wloc = dc.local_mean_weight(dc.sky_to_cartesian(ra[ev], dec[ev], z[ev]), w[ev], k=32)
        st = hole_statistics(ra, dec, rho, {'local': wloc}, E, nside_max=nside, eval_mask=ev)['local']
        return dict(pix=pix, n=n, E=E, nside=nside, scale_deg=np.array([x['scale_arcmin'] / 60 for x in st]),
                    R_PP=np.array([x['R_PP'] for x in st]))
    return _cached(f'randoms_{tracer}_{cap}_{N_RANDOM_FILES}files', compute)
