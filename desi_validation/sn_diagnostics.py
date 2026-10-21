"""Inputs for diagnosing the Poisson (discreteness) non-Gaussian covariance against the holi mocks.

The local B/n and P/n^2 terms (thecov.DiscretenessCovariance) come out 2-4x below what the mocks need. This
script gathers, for one tracer bin and cap, what is needed to find out why, in one small file
(<out>/sn_diag_<bin>_<cap>.npz, a few MB):

  weights/*   moments of the data and random weights (WEIGHT, WEIGHT_FKP, their product) of the catalogue mock
  J/*         the window integrals of the terms, int S m^2 and int S^2 (unnormalised), as thecov computes them
              from the randoms (S = alpha scale w_r^2 sampled by the randoms, local value with the global
              <w^2>/<w>), and from the data: sum_g w_g^2 m(x_g)^2, and sum_g w_g^2 S_loc(x_g) with the local
              <w^2>/<w> of the k nearest other galaxies; plus int m^2, int S, alpha, shot-noise scale
  pairs/*     unweighted and weighted 1 + xibar(<r) of the data at r = 0.25 ... 16 Mpc/h (DD / DR): pairs
              closer than the clustering model allows (one-halo, duplicates) make the counts non-Poisson
  hi/*        every mock's P0, P2, P4 in 0.01 h/Mpc bins up to --kmax (beyond the 0.3 of the main analysis), with
              norm and num_shotnoise
  hi/C_*      thecov on those bins: local Gaussian (G), SSC with the local average (ssc), and the local
              discreteness B/n and P/n^2 (disc_B, disc_P), with the parameters of the main run (ssc_<label>.npz)

Run (debug queue, one job per bin and cap):
  python -u -m desi_validation.sn_diagnostics --bin LRG1 --cap NGC
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402

T0 = time.time()
TRACERS = {'LRG1': ('LRG', 0.4, 0.6), 'QSO': ('QSO', 0.8, 2.1)}
SPECTRA = ('/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base/holi-v3-altmtl/'
           'mock*/mesh2_spectrum_poles_{name}_z{zmin}-{zmax}_{cap}_weight-default-FKP.h5')
R_PAIRS = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def moments(w):
    w = np.asarray(w, float)
    return np.array([len(w), w.sum(), (w ** 2).sum(), (w ** 3).sum(), (w ** 4).sum()])


def read_spectrum(args):
    fn, kmin, kmax, step = args
    import lsstypes
    try:
        s = lsstypes.read(fn).select(k=slice(0, None, step)).select(k=(kmin, kmax))
        p0 = s.get(0)
        e = np.asarray(p0.edges('k'))
        return (np.concatenate([np.real(np.asarray(s.get(l).value())) for l in (0, 2, 4)]), np.asarray(p0.coords('k')),
                np.concatenate([e[:, 0], e[-1:, 1]]), float(np.ravel(p0.values('norm'))[0]),
                float(np.ravel(p0.values('num_shotnoise'))[0]), int(os.path.basename(os.path.dirname(fn))[4:]))
    except Exception as exc:                                     # noqa: BLE001
        return f'{fn}: {exc}'


def pair_excess(pos_d, w_d, pos_r, w_r, n_max_r=2_000_000, seed=0):
    """1 + xibar(<r) = (DD / DR) (sum w_r / sum w_d), unweighted and weighted (self-pairs excluded)"""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    sel = rng.random(len(pos_r)) < min(1.0, n_max_r / len(pos_r))
    pos_r, w_r = pos_r[sel], w_r[sel]
    td, tr = cKDTree(pos_d), cKDTree(pos_r)
    out = {}
    for tag, wd, wr in (('unweighted', None, None), ('weighted', w_d, w_r)):
        dd = td.count_neighbors(td, R_PAIRS, weights=None if wd is None else (wd, wd), cumulative=True)
        dd = dd - (len(pos_d) if wd is None else np.sum(wd ** 2))          # self-pairs
        dr = td.count_neighbors(tr, R_PAIRS, weights=None if wd is None else (wd, wr), cumulative=True)
        sd = len(pos_d) if wd is None else wd.sum()
        sr = len(pos_r) if wr is None else wr.sum()
        out[tag] = (dd / sd) / (dr / sr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1', choices=tuple(TRACERS))
    ap.add_argument('--cap', default='NGC', choices=('NGC', 'SGC'))
    ap.add_argument('--mock', type=int, default=173, help='catalogue mock (randoms, footprint, weights)')
    ap.add_argument('--ssc', default=None, help='ssc_check output with the parameters (default <out>/ssc_holi-kcore2-<bin>.npz)')
    ap.add_argument('--dir', default=os.path.expanduser('~/thecov_desi/holi_v3_mock173'), help='ssc window pair counts')
    ap.add_argument('--out', default='/global/cfs/cdirs/desicollab/users/oalves/thecov_validation')
    ap.add_argument('--kmin', type=float, default=0.02)
    ap.add_argument('--kmax', type=float, default=0.6)
    ap.add_argument('--dk-step', type=int, default=10, help='fine bins (0.001 h/Mpc) per bin')
    ap.add_argument('--knn', type=int, default=64, help='data neighbours for the local <w^2>/<w>')
    ap.add_argument('--n-near', type=int, default=300000)
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()

    from thecov import GaussianCovariance, PowerSpectrum, SuperSampleCovariance, DiscretenessCovariance
    from thecov.tree.power import Dressed
    from scipy.spatial import cKDTree
    b, r = args.bin, args.cap
    res = {}
    fn_out = os.path.join(args.out, f'sn_diag_{b}_{r}.npz')

    def save():
        np.savez_compressed(fn_out, **res)
        log(f'saved {fn_out} ({os.path.getsize(fn_out) / 1e6:.1f} MB)')

    # ------------------------------------------------------------------ spectra up to kmax
    name, zmin, zmax = TRACERS[b]
    fns = sorted(glob.glob(SPECTRA.format(name=name, zmin=zmin, zmax=zmax, cap=r)))
    log(f'{len(fns)} spectrum files')
    import multiprocessing
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        sp = pool.map(read_spectrum, [(fn, args.kmin, args.kmax, args.dk_step) for fn in fns], chunksize=4)
    bad = [x for x in sp if isinstance(x, str)]
    sp = [x for x in sp if not isinstance(x, str)]
    if bad:
        log(f'{len(bad)} unreadable, e.g. {bad[0]}')
    V = np.array([x[0] for x in sp])
    k, k_edges = sp[0][1], sp[0][2]
    res.update({'hi/V': V, 'hi/k': k, 'hi/k_edges': k_edges, 'hi/norm': np.array([x[3] for x in sp]),
                'hi/num_shotnoise': np.array([x[4] for x in sp]), 'hi/ids': np.array([x[5] for x in sp])})
    log(f'spectra: {V.shape[0]} mocks, {len(k)} bins {k_edges[0]:.3f}-{k_edges[-1]:.3f}')
    save()

    # ------------------------------------------------------------------ catalogue mock, tracer as ssc_check
    cfg = pl.Config()
    paths = dc.Paths(kind='holi_v3', mock=args.mock)
    paths.loader, paths.cs_version, paths.cs_parent_version = 'auto', 'holi-v3-altmtl', 'data-dr2-v2'
    rc = dc.load_region(paths, b, r, n_random_files=cfg.n_random_files)
    tr, info = dc.build_tracer(f'{b}_{r}_ssc', [rc], nw='random-density', n_randoms_max=cfg.n_randoms_max // 2,
                               surface_density_deg2=cfg.surface_density, verbose=False)
    sn_scale = tr.shotnoise / ((1 + tr.alpha) * tr.alpha * float(np.sum(tr.weights ** 2)))
    log(f'tracer: {tr.size} randoms, alpha {tr.alpha:.4f}, shot-noise scale {sn_scale:.4f}')
    wd = np.asarray(dc.total_weight(rc.data), float)
    wr_all = np.asarray(dc.total_weight(rc.randoms), float)
    for cat, tag in ((rc.data, 'data'), (rc.randoms, 'randoms')):
        for col in ('WEIGHT', 'WEIGHT_FKP'):
            if col in cat:
                res[f'weights/{tag}_{col}'] = moments(cat[col])
        res[f'weights/{tag}_total'] = moments(dc.total_weight(cat))
    res['weights/n_random_files'] = np.array(rc.n_random_files)
    pos_d = dc.sky_to_cartesian(rc.data['RA'], rc.data['DEC'], rc.data['Z'])

    # ------------------------------------------------------------------ window integrals
    a, sc, w, m = tr.alpha, sn_scale, np.asarray(tr.weights, float), np.asarray(tr.weighted_density, float)
    ratio = np.sum(w ** 2) / np.sum(w)
    m_d, _ = tr.density_and_weight_at(pos_d)                                   # m at each galaxy (nearest random)
    # local <w^2>/<w> of the data: k nearest other galaxies, at each galaxy and at each random
    td = cKDTree(pos_d)
    _, idx = td.query(pos_d, k=args.knn + 1, workers=-1)
    idx = idx[:, 1:]
    ratio_d = np.sum(wd[idx] ** 2, 1) / np.sum(wd[idx], 1)
    _, idx = td.query(tr.positions, k=args.knn, workers=-1)
    ratio_r = np.sum(wd[idx] ** 2, 1) / np.sum(wd[idx], 1)
    del idx
    J = {
        'alpha': a, 'shotnoise_scale': sc, 'ratio_global_randoms': ratio,
        'ratio_global_data': np.sum(wd ** 2) / np.sum(wd),
        'int_m2': a * np.sum(w * m),                             # thecov's I_local
        'int_S_randoms': a * sc * np.sum(w ** 2), 'int_S_data': np.sum(wd ** 2),
        'Smm_randoms': a * np.sum(sc * w ** 2 * m ** 2),         # thecov J_SWW x norm^2
        'SS_randoms_global': a * np.sum(sc * w ** 2 * sc * ratio * m),   # thecov J_SS x norm^2
        'SS_randoms_local': a * np.sum(sc * w ** 2 * sc * ratio_r * m),
        'Smm_data': np.sum(wd ** 2 * m_d ** 2),
        'SS_data_local': np.sum(wd ** 2 * m_d * ratio_d),
        'SS_data_global': np.sum(wd ** 2 * m_d) * np.sum(wd ** 2) / np.sum(wd),
        'outside_fraction_data': np.mean(m_d == 0),
    }
    for key, val in J.items():
        res[f'J/{key}'] = np.array(val)
    log('integrals: ' + ', '.join(f'{key} {val:.4g}' for key, val in J.items()))
    log(f'local <w^2>/<w> of the data: at galaxies {np.percentile(ratio_d, [5, 50, 95])}, '
        f'global {J["ratio_global_data"]:.4f} (randoms {ratio:.4f})')
    res['J/ratio_d_percentiles'] = np.percentile(ratio_d, [1, 5, 25, 50, 75, 95, 99])
    save()

    # ------------------------------------------------------------------ small-scale pair excess
    pe = pair_excess(pos_d, wd, dc.sky_to_cartesian(rc.randoms['RA'], rc.randoms['DEC'], rc.randoms['Z']), wr_all)
    res['pairs/r'] = R_PAIRS
    for tag, val in pe.items():
        res[f'pairs/{tag}'] = val
        log(f'1 + xibar(<r) {tag}: ' + ', '.join(f'{x:g}: {y:.3f}' for x, y in zip(R_PAIRS, val)))
    del rc, pos_d, td
    save()

    # ------------------------------------------------------------------ thecov terms on the high-k bins
    ssc_fn = args.ssc or os.path.join(args.out, f'ssc_holi-kcore2-{b}.npz')
    z = np.load(ssc_fn)
    b1, b2, bs2, f, zeff = z[f'{r}/params']
    ir_sigma, sigma_v, A_norm = z[f'{r}/damping']
    kl, Pl, PirA = z[f'{r}/plin_k'], z[f'{r}/plin_P'], z[f'{r}/plin_P_damped_normalised']
    dil = np.interp(k, z[f'{r}/k'], z[f'{r}/dilution'])          # constant beyond the main run's kmax
    norm = float(np.mean(res['hi/norm']))
    nb = len(k)
    mean = V.mean(0)
    kk = np.r_[0.0, k, k_edges[-1]]
    power = PowerSpectrum({ell: (kk, np.r_[mean[i * nb], mean[i * nb:(i + 1) * nb], mean[(i + 1) * nb - 1]])
                           for i, ell in enumerate((0, 2, 4))}, kedges=k_edges, window_convolved=True, norm=norm,
                          shotnoise=tr.shotnoise)
    cov = GaussianCovariance([tr], power, n_sub=cfg.n_sub_far, n_near=args.n_near,
                             ds=2.0, ds_pair=10.0, s_split=80.0, smoothing=None)
    disc = DiscretenessCovariance(cov, (kl, PirA), b1=b1, f=f, b2=b2, bs2=bs2, sigma_fog=sigma_v,
                                  **(dict(n_mu=16, n_phi=24) if sigma_v else {}))
    comps = disc.components([(tr.name, tr.name)], ells=(0, 2, 4))
    res['hi/C_disc_B'], res['hi/C_disc_P'] = comps['B'], comps['P']
    res['hi/J_disc'] = np.array(disc.window_integrals(tr.name))
    res['hi/params'] = np.array([b1, b2, bs2, f, zeff, sigma_v, A_norm])
    log('discreteness done')
    save()
    ssc = SuperSampleCovariance(cov, (kl, A_norm * Pl), b1=b1, f=f, b2=b2, bs2=bs2, local_average=True,
                                p_dressed=Dressed(kl, PirA, sigma_v) if ir_sigma else None, n_near=args.n_near,
                                dilution=dil, discreteness=True, norm_kind='data-randoms')
    wfile = os.path.join(args.dir, f'ssc_windows_{b}_{r}.npz')
    if os.path.exists(wfile):
        ssc.load_windows(wfile)
    ssc.compute_windows(tr.name, verbose=True)
    res['hi/C_ssc'], _ = ssc.covariance([(tr.name, tr.name)], ells=(0, 2, 4))
    log('SSC done')
    save()
    cov.compute_windows([(tr.name, tr.name)], verbose=False)
    res['hi/C_G'], _ = cov.covariance([(tr.name, tr.name)])
    log('Gaussian (local window) done')
    save()


if __name__ == '__main__':
    main()
