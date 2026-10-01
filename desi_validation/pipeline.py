"""One tracer bin end to end: catalogues -> weight diagnostics -> mock spectra -> thecov tracers ->
covariances (NGC, SGC, GCcomb) -> validation statistics. Used by the notebook's loop over all bins.

Everything expensive is cached in <out_dir>/<bin>/: window pair counts (windows_<name>.npz) and the
covariance matrices (cov_<region>_<mode>_<binning>.npz).
"""
from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from . import desi_compare as dc
from . import validation as va

CAPS = ('NGC', 'SGC')
# bump when what is cached changes (v2: S window scaled to the spectra's num_shotnoise)
CACHE_VERSION = 'v2'


def windows_path(out_dir, tracer_bin, name):
    return os.path.join(out_dir, tracer_bin, f'windows_{name}_{CACHE_VERSION}.npz')


@dataclass
class Config:
    kmin: float = 0.02
    kmax: float = 0.3
    rebin: int = 5                    # 0.001 -> 0.005 h/Mpc bins
    ells: tuple = (0, 2, 4)
    n_random_files: int = 1
    n_randoms_max: int = 4_000_000     # randoms kept for thecov (all regions)
    surface_density: float = 2500.0   # randoms per deg^2 per file
    target_near_pairs: float = 2e9
    n_sub_far: int = 20000
    regions: tuple = ('NGC', 'SGC', 'GCcomb')
    shotnoise_from_files: bool = True    # S window integral = the spectra's mean num_shotnoise
    model_norm_correction: bool = True   # model P = mean P_hat x norm / int m^2
    naive: bool = False               # also build the naive NZ x WEIGHT covariance (comparison)
    coarse_check: bool = True         # repeat the validation with bins twice as wide

    def binning_tag(self, rebin=None):
        return (f'k{self.kmin:g}-{self.kmax:g}_r{rebin or self.rebin}_sn{int(self.shotnoise_from_files)}'
                f'_mc{int(self.model_norm_correction)}_{CACHE_VERSION}')


# --------------------------------------------------------------------------- spectra helpers
def mock_ids(spec):
    return [os.path.basename(os.path.dirname(f)) for f in spec['files']]


def gccomb_definition(spec, tol=1e-3):
    """How the GCcomb spectra were made, from the mocks present in all three regions:
    'average' if every GCcomb vector equals the norm-weighted average of its NGC and SGC vectors
    (then C_GCcomb = sum_r n_r^2 C_r / (sum n_r)^2 exactly), 'joint' if it is one estimate on the
    concatenated catalogues (then thecov runs on the combined geometry). Returns (mode, max deviation
    in units of the GCcomb standard deviation)."""
    if not all(r in spec for r in ('NGC', 'SGC', 'GCcomb')):
        return None, None
    ids = {r: {m: i for i, m in enumerate(mock_ids(spec[r]))} for r in ('NGC', 'SGC', 'GCcomb')}
    common = sorted(set(ids['NGC']) & set(ids['SGC']) & set(ids['GCcomb']))
    if not common:
        return None, None
    iN, iS, iG = ([ids[r][m] for m in common] for r in ('NGC', 'SGC', 'GCcomb'))
    nN, nS = spec['NGC']['norm'][iN][:, None], spec['SGC']['norm'][iS][:, None]
    avg = (nN * spec['NGC']['vectors'][iN] + nS * spec['SGC']['vectors'][iS]) / (nN + nS)
    dev = np.max(np.abs(spec['GCcomb']['vectors'][iG] - avg) / spec['GCcomb']['vectors'].std(0))
    return ('average' if dev < tol else 'joint'), float(dev)


def spectra_checks(spec):
    """Per region: is the subtracted shot noise the realised one (shotnoise = num_shotnoise / norm per
    mock)? and the scatter of norm and num_shotnoise over the mocks."""
    out = {}
    for r, s in spec.items():
        rel = np.abs(s['shotnoise'] - s['num_shotnoise'] / s['norm']) / s['shotnoise']
        out[r] = dict(shotnoise_is_realised=bool(np.max(rel) < 1e-6), max_rel_dev=float(np.max(rel)),
                      norm_mean=float(s['norm'].mean()), norm_rel_std=float(s['norm'].std() / s['norm'].mean()),
                      num_shotnoise_mean=float(s['num_shotnoise'].mean()),
                      num_shotnoise_var=float(s['num_shotnoise'].var(ddof=1)), N=len(s['vectors']))
    return out


def poisson_var_num_shotnoise(rc, scheme='default-FKP'):
    """Poisson part of Var over mocks of sum_d w^2 + alpha^2 sum_r w^2 (randoms redrawn per mock):
    sum_d w^4 + alpha^4 sum_r w^4. Clustering of the weighted counts adds to it."""
    wd, wr = dc.total_weight(rc.data, scheme), dc.total_weight(rc.randoms, scheme)
    a = wd.sum() / wr.sum()
    return float(np.sum(wd ** 4)), float(a ** 4 * np.sum(wr ** 4) * rc.n_random_files)


# --------------------------------------------------------------------------- covariances
def covariance_cached(tracer, spec, cfg: Config, path, n_near=None, log=print):
    if os.path.exists(path):
        with np.load(path) as f:
            if f['C'].shape[0] == spec['vectors'].shape[1] and np.allclose(f['k'], spec['k']):
                return f['C'], dict(I_randoms=float(f['I_randoms']), cached=True)
    if n_near is None:
        n_near, _ = dc.suggest_n_near(tracer, target_pairs=cfg.target_near_pairs)
    t0 = time.time()
    wfile = os.path.join(os.path.dirname(path), f'windows_{tracer.name}_{CACHE_VERSION}.npz')
    C, cov = dc.thecov_covariance(tracer, spec, n_sub=cfg.n_sub_far, n_near=n_near, verbose=False, windows_file=wfile,
                                  model_norm_correction=cfg.model_norm_correction)
    np.savez(path, C=C, k=spec['k'], I_randoms=cov.I_randoms)
    log(f'    {os.path.basename(path)}: n_near {n_near}, {time.time() - t0:.0f} s')
    return C, dict(I_randoms=cov.I_randoms, cached=False, cov=cov)


def run_bin(paths, tracer_bin, cfg: Config, out_dir, log=print, keep=False):
    """Validate one tracer bin. Returns a dict with diagnostics, spectra checks, covariances and the
    validation output per (region, mode); `keep` also returns the catalogues, tracers and spectra."""
    tracer, zr = dc.TRACER_SPECS[tracer_bin]
    od = os.path.join(out_dir, tracer_bin)
    os.makedirs(od, exist_ok=True)
    res = dict(bin=tracer_bin, tracer=tracer, zrange=zr, config=asdict(cfg))
    t0 = time.time()

    # spectra first (cheap; tells which regions exist)
    spec = {}
    for r in cfg.regions:
        fns = paths.spectra_fns(tracer, zr, r)
        if fns:
            spec[r] = dc.read_spectra(fns, kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=cfg.ells)
    if not spec:
        log(f'{tracer_bin}: no spectra found'); return res
    res['spectra_checks'] = spectra_checks(spec)
    res['gccomb'] = gccomb_definition(spec)
    counts = ', '.join(f'{r} ({len(s["vectors"])} mocks)' for r, s in spec.items())
    log(f"{tracer_bin}: spectra {counts}; GCcomb is {res['gccomb'][0]} (max dev {res['gccomb'][1]})")

    # catalogues and diagnostics
    caps = [r for r in CAPS if r in spec or 'GCcomb' in spec]
    regs = {r: dc.load_region(paths, tracer_bin, r, n_random_files=cfg.n_random_files) for r in caps}
    res['diagnostics'] = {r: {k: v for k, v in dc.weight_diagnostics(rc, tracer_bin).items()
                              if k not in ('data_columns', 'random_columns')} for r, rc in regs.items()}
    res['poisson_var_num_shotnoise'] = {r: poisson_var_num_shotnoise(rc) for r, rc in regs.items()}

    # tracers and covariances
    modes = ['random-density'] + (['none'] if cfg.naive else [])
    covs, info, tracers = {}, {}, {}
    for r in caps:
        for mode in modes:
            name = f'{tracer_bin}_{r}' + ('_naive' if mode == 'none' else '')
            sn = spec[r]['num_shotnoise'].mean() if (cfg.shotnoise_from_files and r in spec) else None
            tr, inf = dc.build_tracer(name, [regs[r]], nw=mode, n_randoms_max=cfg.n_randoms_max // 2,
                                      surface_density_deg2=cfg.surface_density, verbose=False, shotnoise_target=sn)
            tracers[(r, mode)], info[(r, mode)] = tr, inf
            if r in spec:
                C, ci = covariance_cached(tr, spec[r], cfg, os.path.join(od, f'cov_{r}_{mode}_{cfg.binning_tag()}.npz'), log=log)
                covs[(r, mode)] = C
                inf.update(I_over_norm=ci['I_randoms'] / spec[r]['norm'].mean(),
                           shotnoise_catalogue_over_files=inf['shotnoise_numerator'] / spec[r]['num_shotnoise'].mean())
    if 'GCcomb' in spec and all((r, 'random-density') in covs for r in CAPS):
        norms = [spec[r]['norm'].mean() for r in CAPS]
        covs[('GCcomb', 'combined-regions')] = dc.combine_regions([covs[(r, 'random-density')] for r in CAPS], norms)
        if res['gccomb'][0] == 'joint':
            sn = spec['GCcomb']['num_shotnoise'].mean() if cfg.shotnoise_from_files else None
            tr, inf = dc.build_tracer(f'{tracer_bin}_GCcomb', [regs[r] for r in CAPS], nw='random-density',
                                      n_randoms_max=cfg.n_randoms_max, surface_density_deg2=cfg.surface_density, verbose=False,
                                      shotnoise_target=sn)
            tracers[('GCcomb', 'random-density')], info[('GCcomb', 'random-density')] = tr, inf
            C, ci = covariance_cached(tr, spec['GCcomb'], cfg, os.path.join(od, f'cov_GCcomb_random-density_{cfg.binning_tag()}.npz'), log=log)
            covs[('GCcomb', 'random-density')] = C
            inf.update(I_over_norm=ci['I_randoms'] / spec['GCcomb']['norm'].mean())
    res['tracer_info'] = {f'{r}__{m}': {k: v for k, v in i.items() if k != 'regions'} | {'regions': i['regions']}
                          for (r, m), i in info.items()}

    # validation
    res['validation'] = {}
    for (r, mode), C in covs.items():
        s = spec[r]
        try:
            res['validation'][(r, mode)] = va.validate(s['vectors'], C, s['k'], cfg.ells)
        except ValueError as ex:
            log(f'  {r} {mode}: {ex}')
    if cfg.coarse_check:
        for r in spec:
            key = (r, 'random-density') if (r, 'random-density') in tracers else None
            if key is None:
                continue
            s2 = dc.read_spectra(paths.spectra_fns(tracer, zr, r), kmin=cfg.kmin, kmax=cfg.kmax, rebin=2 * cfg.rebin, ells=cfg.ells)
            C2, _ = covariance_cached(tracers[key], s2, cfg, os.path.join(od, f'cov_{r}_random-density_{cfg.binning_tag(2 * cfg.rebin)}.npz'), log=log)
            try:
                res['validation'][(r, 'random-density, bins x2')] = va.validate(s2['vectors'], C2, s2['k'], cfg.ells)
            except ValueError as ex:
                log(f'  {r} bins x2: {ex}')
    res['covariances'] = covs
    log(f'{tracer_bin}: done in {time.time() - t0:.0f} s')
    if keep:
        res.update(regions=regs, tracers=tracers, spectra=spec)
    else:
        del regs, tracers
        gc.collect()
        res['spectra'] = spec
    return res


def summary_table(results, mode='random-density'):
    """Rows (bin, region, numbers) for every validated case of the given mode."""
    rows = []
    for res in results:
        for (r, m), out in res.get('validation', {}).items():
            if m == mode:
                rows.append(dict(bin=res['bin'], region=r, **va.summary_row(out)))
    return rows


def to_json(results, path):
    def clean(o):
        if isinstance(o, dict):
            return {(k if isinstance(k, str) else '__'.join(map(str, k)) if isinstance(k, tuple) else str(k)): clean(v)
                    for k, v in o.items() if not (isinstance(k, str) and k.startswith('_'))}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist() if o.ndim <= 1 else None
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    slim = [{k: v for k, v in r.items() if k not in ('covariances', 'spectra', 'regions', 'tracers')} for r in results]
    with open(path, 'w') as f:
        json.dump(clean(slim), f, indent=1, default=str)
