"""Window-shape tests for the LRG1 excess, in one script (no notebook needed).

    python -m desi_validation.run_window_shape                 # everything, LRG1 + QSO control
    python -m desi_validation.run_window_shape --skip-mesh     # without the FFT-mesh test (faster)
    python -m desi_validation.run_window_shape --isotropic     # also the P2 = P4 = 0 mesh/thecov check
    python -m desi_validation.run_window_shape --skip-mesh --control \
        --cs-version <complete version> --spectra-dir <dir with mock*/ spectra>   # e.g. complete mocks

Run from the repository root (or set THECOV_DIR), in an interactive allocation, with the notebook's
environment. Output: a report on stdout (paste it back) and OUT/window_shape_report.json.

Tests
  Constructions of m compared (--modes, first = reference): 'random-density' (alpha rho_r(z) <w>_local,
  thecov's default), 'nx' (NX x WEIGHT_FKP at each random, no smoothing), 'patch' (rho_r(z) per sky
  patch), 'angular'.
  1. n(z) variation across the sky (nz_variation); m of each construction / the reference at the
     same randoms (percentiles, and per z quintile); 1/V_eff of each window; LRG1 and QSO.
  2. thecov with each construction of m: <chi2>/n and variance ratios per multipole and k range.
  3. FFT-mesh exact Gaussian Var(P0) with each m, against thecov, the mocks, and the counts-based
     window B of the earlier scratch runs (if their JSON files exist).
  4. (--isotropic) the same mesh vs thecov comparison with P2 = P4 = 0, to see whether the 2-3% offset
     between the mesh and thecov is the mesh's global line of sight.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl, mesh_gaussian as mg, validation as va  # noqa: E402

T0 = time.time()
KRANGES = ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3))
MODES = ['random-density', 'nx']


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def header(t):
    print('\n' + '=' * 100 + f'\n{t}\n' + '=' * 100, flush=True)


def setup():
    paths = dc.Paths(kind='holi_v3', mock=173)
    paths.catalog_names.update({'ELG_LOPnotqso': 'ELGnotqso', 'LRG+ELG_LOPnotqso': 'LRG+ELGnotqso'})
    paths.loader = 'auto'
    paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
    cfg = pl.Config()
    out = os.path.expanduser(f'~/thecov_desi/{paths.kind}_mock{paths.mock}')
    return paths, cfg, out


def kranges_var(out, ells):
    """mean variance ratio (mock / thecov) per multipole and k range from a validate() output"""
    k = np.asarray(out['k']); nb = len(k)
    r2 = np.asarray(out['_sigma_ratio']) ** 2
    res = {}
    for j, ell in enumerate(ells):
        for lo, hi in KRANGES:
            s = (k >= lo) & (k < hi)
            if s.any():
                res[f'P{ell} {lo}-{hi}'] = float(r2[j * nb:(j + 1) * nb][s].mean())
    return res


# ----------------------------------------------------------------------------- test 1
def m_ratio_stats(rc, cfg, n_max=300_000, seed=0):
    """m of each construction / m of the reference, at the same randoms: percentiles, and median in z bins"""
    r = rc.randoms
    sel = np.random.default_rng(seed).random(len(r['Z'])) < min(1.0, n_max / len(r['Z']))
    wd, wr = dc.total_weight(rc.data), dc.total_weight(r)
    a_reg = wd.sum() / wr.sum()
    rho, _ = dc.random_density(rc, cfg.surface_density)
    p = dc.sky_to_cartesian(r['RA'], r['DEC'], r['Z'])
    kw = dict(surface_density_deg2=cfg.surface_density)
    m0 = dc._m_values(rc, MODES[0], sel, a_reg, rho, p, **kw)
    z = r['Z'][sel]
    zb = np.quantile(z, np.linspace(0, 1, 6))
    out = {}
    for mode in MODES[1:]:
        q = dc._m_values(rc, mode, sel, a_reg, rho, p, **kw) / m0
        out[mode] = dict(p16_50_84=np.percentile(q, [16, 50, 84]).tolist(),
                         median_in_z=[float(np.median(q[(z >= lo) & (z <= hi)])) for lo, hi in zip(zb[:-1], zb[1:])],
                         z_edges=zb.tolist())
    return out


def test_nz_and_veff(regs, cfg, label):
    rows = {}
    for r, rc in regs.items():
        row = {}
        for nside in (4, 8):
            nv = dc.nz_variation(rc, nside=nside, surface_density_deg2=cfg.surface_density)
            row[f'nz_rms_nside{nside}'] = nv['rms_patch_nz_deviation']
            row[f'nz_poisson_nside{nside}'] = nv['poisson_expectation']
            row['north_fraction'] = nv['north_fraction']
            if 'mean_z_north_minus_south' in nv:
                row['dz_north_minus_south'] = nv['mean_z_north_minus_south']
                row['north_over_south_nz'] = nv['north_over_south_nz']
        vm = dc.window_moments(rc, modes=MODES, surface_density_deg2=cfg.surface_density)
        for mode in MODES[1:]:
            row[f'veff_{mode}_over_{MODES[0]}'] = vm[mode] / vm[MODES[0]]
        row['m_ratio'] = m_ratio_stats(rc, cfg)
        rows[r] = row
        for mode, mr in row['m_ratio'].items():
            log(f"{label} {r}: m[{mode}] / m[{MODES[0]}] at the randoms: 16/50/84% {np.round(mr['p16_50_84'], 4).tolist()}; "
                f"median in 5 z quintiles {np.round(mr['median_in_z'], 4).tolist()}")
        log(f"{label} {r}: n(z) rms across patches {row['nz_rms_nside8']:.4f} (Poisson {row['nz_poisson_nside8']:.4f}, nside 8), "
            f"{row['nz_rms_nside4']:.4f} (Poisson {row['nz_poisson_nside4']:.4f}, nside 4); "
            + (f"<z> north - south {row['dz_north_minus_south']:+.4f} (north fraction {row['north_fraction']:.2f}); "
               if 'dz_north_minus_south' in row else '')
            + f'1/V_eff relative to {MODES[0]}: ' + ', '.join(f"{m}: {row[f'veff_{m}_over_{MODES[0]}']:.4f}" for m in MODES[1:]))
    return rows


# ----------------------------------------------------------------------------- test 3/4
def mesh_test(D, b, r, cfg, OUT, isotropic=False, every=3, kmax=0.2):
    spec = D['spectra'][r]
    norm, num_sn = float(spec['norm'].mean()), float(spec['num_shotnoise'].mean())
    nb, edges = len(spec['k']), spec['k_edges']
    shells = [i for i in range(nb) if edges[i + 1] <= kmax + 1e-9][::every]
    k = spec['k'][shells]
    tr0 = D['tracers'][(r, 'random-density')]
    mesh = mg.Mesh(tr0.pos, log=log)
    los = tr0.pos.mean(0)
    var_mock = spec['vectors'].var(0, ddof=1)[shells]
    res = dict(k=k.tolist(), var_mock=var_mock.tolist())
    specs = {'aniso': spec}
    if isotropic:
        s_iso = copy.deepcopy(spec)
        s_iso['vectors'][:, nb:] = 0.0                           # P2 = P4 = 0 in the model (mean vector)
        specs['iso'] = s_iso
    for mode in MODES:
        tr = D['tracers'][(r, mode)]
        W, S, intW = mg.windows_from_tracer(mesh, tr, num_sn)
        for tag, sp in specs.items():
            v = mg.p0_variance(mesh, W, S, intW, sp, norm, shells, los, log=log, label=f'{r} {mode} {tag}')
            if tag == 'aniso':
                C = D['covariances'][(r, mode)]
            else:
                wfile = pl.windows_path(OUT, b, tr.name)
                C, _ = dc.thecov_covariance(tr, sp, n_sub=cfg.n_sub_far, n_near=10 ** 6, verbose=False, windows_file=wfile,
                                            model_norm_correction=cfg.model_norm_correction)
            dT = np.diag(C)[shells]
            res[f'{mode}_{tag}'] = dict(mesh=v.mean(0).tolist(), split_diff=float(np.mean(v[0] / v[1]) - 1),
                                        thecov=dT.tolist(), mesh_over_thecov=(v.mean(0) / dT).tolist())
        del W, S
    # counts-based window B from the earlier scratch runs, if available
    for name, fn, key in (('B12', f'mesh_gaussian_{r}.json', None), ('B1.5', f'mesh_resolution_{r}.json', '1.5')):
        path = os.path.join(OUT, b, fn)
        if os.path.exists(path):
            prev = json.load(open(path))
            kb = np.asarray(prev['k'])
            vb = np.asarray(prev['B']['diag']).mean(0) if key is None else np.asarray(prev['diag'][key])
            res[name] = [float(vb[np.argmin(np.abs(kb - kk))]) if np.min(np.abs(kb - kk)) < 1e-3 else float('nan') for kk in k]
    return res


def print_mesh(r, res):
    k = np.asarray(res['k'])
    vm = np.asarray(res['var_mock'])
    ref = res[f'{MODES[0]}_aniso']
    t0 = np.asarray(ref['thecov'])
    short = {'random-density': 'rd', 'patch': 'patch', 'nx': 'nx', 'angular': 'ang'}
    cols = []
    for mode in MODES:
        a = res[f'{mode}_aniso']
        cols.append((f'mesh_{short.get(mode, mode)}/thecov', np.asarray(a['mesh']) / t0))
        if mode != MODES[0]:
            cols.append((f'thecov_{short.get(mode, mode)}/thecov', np.asarray(a['thecov']) / t0))
    for name in ('B12', 'B1.5'):
        if name in res:
            cols.append((f'{name}/thecov', np.asarray(res[name]) / t0))
    for mode in MODES:
        cols.append((f'mock/thecov_{short.get(mode, mode)}', vm / np.asarray(res[f'{mode}_aniso']['thecov'])))
    print(f"\n{r}: P0 variance ratios; 'thecov' = thecov with m from {MODES[0]}; mesh split differences "
          + ', '.join(f"{mode} {res[f'{mode}_aniso']['split_diff']:+.4f}" for mode in MODES))
    print(f"{'k range':>12s}" + ''.join(f'{c:>20s}' for c, _ in cols))
    for lo, hi in KRANGES[:3]:
        s = (k >= lo) & (k < hi)
        if s.any():
            print(f'{lo:5.2f}-{hi:<5.2f}  ' + ''.join(f'{np.nanmean(v[s]):20.4f}' for _, v in cols))
    if f'{MODES[0]}_iso' in res:
        print('   isotropic model (P2 = P4 = 0), mesh / thecov with the same m: '
              + ', '.join(f"{mode} {np.mean(res[f'{mode}_iso']['mesh_over_thecov']):.4f} "
                          f"(anisotropic {np.mean(res[f'{mode}_aniso']['mesh_over_thecov']):.4f})" for mode in MODES))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bins', nargs='+', default=['LRG1'], help='bins validated with both constructions of m')
    ap.add_argument('--control', nargs='*', default=['QSO'], help='bins for test 1 only (n(z) / V_eff)')
    ap.add_argument('--regions', nargs='+', default=['NGC', 'SGC'])
    ap.add_argument('--skip-mesh', action='store_true')
    ap.add_argument('--isotropic', action='store_true')
    ap.add_argument('--modes', nargs='+', default=['random-density', 'nx'],
                    help="constructions of m to compare, the first is the reference (random-density, nx, patch, angular)")
    ap.add_argument('--cs-version', default=None,
                    help="clustering_statistics catalogue version (default holi-v3-altmtl), e.g. a complete-mock version")
    ap.add_argument('--spectra-dir', default=None, help='directory with mock*/ spectra of that version')
    ap.add_argument('--mock', type=int, default=None, help='mock whose catalogues set the window (default 173)')
    ap.add_argument('--synthetic', default=None, help=argparse.SUPPRESS)     # testing on the synthetic set
    args = ap.parse_args()
    global MODES
    MODES = list(args.modes)

    paths, cfg, OUT = setup()
    if args.cs_version or args.spectra_dir or args.mock is not None:
        if args.cs_version:
            paths.cs_version = args.cs_version
        if args.spectra_dir:
            paths.spectra_dir = args.spectra_dir
        if args.mock is not None:
            paths.mock = args.mock
        # separate caches: windows and covariances depend on the catalogues
        OUT = os.path.expanduser(f'~/thecov_desi/{paths.cs_version}_mock{paths.mock}')
        log(f'catalogues: version {paths.cs_version}, mock {paths.mock}; spectra: {paths.spectra_dir}; caches: {OUT}')
    if args.synthetic:
        paths = dc.Paths(catalog_dir=args.synthetic + '/catalogs', spectra_dir=args.synthetic + '/spectra', loader='files')
        dc.TRACER_SPECS.clear(); dc.TRACER_SPECS['TEST'] = ('LRG', (0.4, 0.6))
        cfg = dataclasses.replace(cfg, surface_density=150.0, kmax=0.1, target_near_pairs=3e8, n_sub_far=5000)
        OUT = args.synthetic + '/results_script'
        args.bins, args.control = ['TEST'], []
    import importlib.util
    if importlib.util.find_spec('healpy') is None:
        sys.exit('healpy is required (patches)')
    if not mg.self_test(log=log):
        sys.exit('mesh self-test failed')
    report = dict(config=dataclasses.asdict(cfg), out=OUT)
    cfg2 = dataclasses.replace(cfg, nw_modes=tuple(MODES), coarse_check=False,
                               regions=tuple(args.regions) + ('GCcomb',))

    results = {}
    for b in args.bins:
        header(f'{b}: thecov with m from ' + ', '.join(MODES) + ' (new pair counts for windows not cached yet)')
        D = pl.run_bin(paths, b, cfg2, OUT, log=log, keep=True)
        if 'validation' not in D:
            tracer, zr = dc.TRACER_SPECS[b]
            pattern = os.path.join(paths.spectra_dir, 'mock*', paths.spectra_name.format(tracer=tracer, zmin=zr[0], zmax=zr[1], region='NGC'))
            log(f'{b}: no spectra matching {pattern} -- check --spectra-dir (full path) and the file names; skipping {b}')
            continue
        results[b] = D
        header(f'Test 1 [{b}]: n(z) variation across the sky and 1/V_eff of the window')
        report.setdefault('test1', {})[b] = test_nz_and_veff({r: D['regions'][r] for r in args.regions}, cfg, b)

        header(f'Test 2 [{b}]: validation against the mocks for each construction of m')
        t2 = {}
        for (r, mode), out in D['validation'].items():
            kv = kranges_var(out, cfg.ells)
            t2[f'{r} {mode}'] = dict(chi2=out['chi2_ratio'], chi2_sigma=out['chi2_sigma'], var_ratio=out['var_ratio_mean'],
                                    z_mean=out['corr_resid_mean'], ev_max=out['ev_max'], **kv,
                                    n=out['n'], N=out['N'], chi2_i=np.asarray(out['_chi2']).tolist(),
                                    mock_ids=pl.mock_ids(D['spectra'][r]))
        report.setdefault('test2', {})[b] = t2
        ex = {}
        for r in list(args.regions) + ['GCcomb']:
            key = (r, MODES[0]) if (r, MODES[0]) in D['covariances'] else (r, 'combined-regions')
            if key in D['covariances'] and r in D['spectra']:
                o = va.excess_structure(D['spectra'][r]['vectors'], D['covariances'][key])
                ex[r] = dict(top=o['top_eigenvalues'][:4], noise_edge=o['noise_edge'],
                             sigma_A=o['templates']['amplitude (mean vector)']['sigma'],
                             chi2_after=o['templates']['amplitude (mean vector)']['chi2_after'])
                log(f"{b} {r} [{key[1]}] excess: top whitened eigenvalues {np.round(ex[r]['top'], 2).tolist()} "
                    f"(noise edge {ex[r]['noise_edge']:.2f}); common amplitude sigma_A {ex[r]['sigma_A']:.4f}, "
                    f"<chi2>/n with it {ex[r]['chi2_after']:.4f}")
        report.setdefault('excess', {})[b] = ex
        keys = [k_ for k_ in next(iter(t2.values())) if k_.startswith('P')]
        print(f"{'case':38s}{'<chi2>/n':>16s}{'var':>7s}{'z':>7s}" + ''.join(f'{k_:>13s}' for k_ in keys))
        for case, row in t2.items():
            print(f"{case:38s}{row['chi2']:8.4f} ({row['chi2_sigma']:+5.1f}){row['var_ratio']:7.3f}{row['z_mean']:+7.2f}"
                  + ''.join(f'{row[k_]:13.3f}' for k_ in keys))

        if not args.skip_mesh:
            header(f'Test 3 [{b}]: exact Gaussian Var(P0) on an FFT mesh with each m, vs thecov, mocks and window B'
                   + (' + isotropic check' if args.isotropic else ''))
            report.setdefault('test3', {})[b] = {}
            for r in args.regions:
                res = mesh_test(D, b, r, cfg, OUT, isotropic=args.isotropic)
                report['test3'][b][r] = res
                print_mesh(r, res)
        D.pop('regions', None); D.pop('tracers', None)

    for b in args.control:
        header(f'Test 1 [{b}, control]: n(z) variation and 1/V_eff')
        regs = {r: dc.load_region(paths, b, r, n_random_files=cfg.n_random_files) for r in args.regions}
        report.setdefault('test1', {})[b] = test_nz_and_veff(regs, cfg, b)
        del regs

    fn = os.path.join(OUT, 'window_shape_report.json')
    pl.to_json([report], fn)
    header(f'done in {time.time() - T0:.0f} s; numbers saved to {fn}')


if __name__ == '__main__':
    main()
