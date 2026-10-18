"""Where does the excess "P/n^2" (two-shared-object) covariance of a catalogue come from? Catalogue-only diagnostics.

    python -u -m desi_validation.pair_diagnostics --bin QSO --region NGC --mocks 173 174 175 --data

The a^2 term of the closed-form thinning covariance (thinning_closed_form.py) is
    quad(k, k') = [ <|W2(k - k')|^2> + <|W2(k + k')|^2> - 2 sum w^4 ] / norm^2,   W2 = sum_g w_g^2 e^{-i k.x_g},
i.e. 2 sum_{g != h} w_g^2 w_h^2 <e^{i q.(x_g - x_h)}> over the pairs of the catalogue (q = k -+ k', shell-averaged). Its
smooth-density (Poisson, unclustered) expectation is the same expression for the randoms scaled to the same sum w^2,
quad_R (= thecov's Gaussian SS term, g2 of the closed-form fit). The difference quad - quad_R is the clustered-pair part,
the galaxies' own P/n^2. Pairs at separation r << 1/q contribute a constant (q-independent) amount; clustered pairs on
large scales contribute P(|k - k'|), which decays with |i - j|. This script measures, per catalogue:
  1. duplicates: exact repeated positions (and TARGETIDs if loaded);
  2. close pairs: xi(r) for 0.01 < r < 20 Mpc/h from weighted pair counts (unit weights and w^2 w^2), and the angular
     xi(theta) at fibre-collision scales (unit and WEIGHT weights), against the randoms;
  3. the weights: <w^2>/<w>^2 of data and randoms (total weight and WEIGHT only);
  4. quad and quad_R with the analysis weights (WEIGHT x FKP) and with FKP-only weights; their difference by band and
     by |i - j|, and the part of it carried by pairs closer than r_c (2 x excess w^4 pair count / norm^2, j0(q r) ~ 1).
Output: <out>/pairs_<bin>_<region>_<tag>.npz, saved after every catalogue.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.fft as sfft
from scipy.spatial import cKDTree

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc, pipeline as pl  # noqa: E402
from desi_validation.grf_randoms import NTH, log  # noqa: E402

KR = ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3))
R_EDGES = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
TH_EDGES_DEG = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1])
OFFS = (0, 1, 2, 3, 5, 8, 15, 25)


def paths_for(data, mock, data_dir):
    if data:
        paths = dc.Paths(kind='data', mock=0)
        paths.catalog_dir = data_dir
        paths.cs_version = paths.cs_parent_version = 'data-dr2-v2'
    else:
        paths = dc.Paths(kind='holi_v3', mock=mock)
        paths.cs_version, paths.cs_parent_version = 'holi-v3-altmtl', 'data-dr2-v2'
    paths.catalog_names.update({'ELG_LOPnotqso': 'ELGnotqso', 'LRG+ELG_LOPnotqso': 'LRG+ELGnotqso'})
    paths.loader = 'auto'
    return paths


def pair_counts(x1, w1, x2, w2, edges):
    """ordered pairs (i in 1, j in 2) with edges[b] < |x_i - x_j| <= edges[b+1], weights w1_i w2_j; edges[0] = 0 bin
    includes zero-distance pairs (self pairs when 1 = 2)."""
    t1, t2 = cKDTree(x1), cKDTree(x2)
    cum = t1.count_neighbors(t2, edges[1:], weights=(w1, w2), cumulative=True)
    return np.diff(np.concatenate([[0.0], np.asarray(cum, float)]))


def quad_terms(fields, xr_th_list, n, io, nb, nsh, rows, opts):
    """quad(i, j) for several weighted point sets: fields = list of (theta coords, w2 values, self sum w^4).
    Returns array (n_fields, nb, nb) of [2 (Q - sum w^4)] (not divided by norm^2); rows i outside `rows` left 0."""
    import finufft
    PW = []
    for thx, v, _ in fields:
        W2 = finufft.nufft3d1(*thx, v.astype(complex), (n,) * 3, isign=-1, **opts)
        pw = (np.abs(W2) ** 2).astype(np.float32)
        del W2
        PW.append(sfft.rfftn(pw, workers=NTH))
        del pw
    io3 = io.reshape(n, n, n)
    out = np.zeros((len(fields), nb, nb))
    for i in rows:
        IND = sfft.rfftn((io3 == i).astype(np.float32), workers=NTH)
        for f in range(len(fields)):
            corr = sfft.irfftn(np.conj(IND) * PW[f], s=(n,) * 3, workers=NTH).ravel()
            out[f, i] = np.bincount(io, corr, nb + 1)[:nb]
        del IND, corr
    Q = out / np.maximum(np.outer(nsh, nsh), 1)[None]
    return np.array([2 * (Q[f] - fields[f][2]) for f in range(len(fields))])


def one_catalogue(args, mock, data, grid, s_edges):
    import finufft
    t0 = time.time()
    b, r = args.bin, args.region
    paths = paths_for(data, mock, args.data_dir)
    rc = dc.load_region(paths, b, r, n_random_files=args.random_files)
    D, R = rc.data, rc.randoms
    wd, wr = dc.total_weight(D), dc.total_weight(R)
    wd_w, wr_w = dc.total_weight(D, 'default'), dc.total_weight(R, 'default')
    wd_f, wr_f = dc.total_weight(D, 'FKP'), dc.total_weight(R, 'FKP')
    xd = dc.sky_to_cartesian(D['RA'], D['DEC'], D['Z'])
    xr = dc.sky_to_cartesian(R['RA'], R['DEC'], R['Z'])
    tid = np.asarray(D['TARGETID']) if 'TARGETID' in D else None
    ra_d, dec_d = np.asarray(D['RA'], float), np.asarray(D['DEC'], float)
    ra_r, dec_r = np.asarray(R['RA'], float), np.asarray(R['DEC'], float)
    del rc, D, R
    o = dict(n_gal=len(wd), n_ran=len(wr))
    name = 'data' if data else f'mock {mock}'
    # 1. duplicates
    key = np.round(xd, 4)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    o['dup_pos'] = int(np.sum(cnt[cnt > 1] - 1))
    o['dup_tid'] = int(len(tid) - len(np.unique(tid))) if tid is not None else -1
    # 3. weights
    rat = lambda w: float(np.mean(w ** 2) / np.mean(w) ** 2)
    o['w2_over_w_sq'] = np.array([rat(wd), rat(wr), rat(wd_w), rat(wr_w)])
    log(f'{name}: {len(wd)} objects, {len(wr)} randoms ({time.time() - t0:.0f} s); duplicate positions {o["dup_pos"]}, '
        f'duplicate TARGETIDs {o["dup_tid"]}; <w^2>/<w>^2 data {o["w2_over_w_sq"][0]:.4f} randoms {o["w2_over_w_sq"][1]:.4f} '
        f'(WEIGHT only: {o["w2_over_w_sq"][2]:.4f} / {o["w2_over_w_sq"][3]:.4f})')
    # 2. pair counts in r: data-data with unit and w^2 w^2 weights; randoms subsample for the expectation
    rng = np.random.default_rng(1)
    sub = rng.choice(len(wr), size=min(len(wr), args.ran_factor * len(wd)), replace=False)
    xs, ws = xr[sub], wr[sub]
    one_d, one_s = np.ones(len(wd)), np.ones(len(sub))
    DD1 = pair_counts(xd, one_d, xd, one_d, R_EDGES); DD1[0] -= len(wd)
    DD4 = pair_counts(xd, wd ** 2, xd, wd ** 2, R_EDGES); DD4[0] -= float(np.sum(wd ** 4))
    RR1 = pair_counts(xs, one_s, xs, one_s, R_EDGES); RR1[0] -= len(sub)
    RR4 = pair_counts(xs, ws ** 2, xs, ws ** 2, R_EDGES); RR4[0] -= float(np.sum(ws ** 4))
    f1 = (len(wd) * (len(wd) - 1.0)) / (len(sub) * (len(sub) - 1.0))
    f4 = (np.sum(wd ** 2) ** 2 - np.sum(wd ** 4)) / (np.sum(ws ** 2) ** 2 - np.sum(ws ** 4))
    o.update(DD1=DD1, DD4=DD4, RR1e=RR1 * f1, RR4e=RR4 * f4)
    log(f'  pair counts in r done ({time.time() - t0:.0f} s)')
    # angular, unit and WEIGHT weights
    u = lambda ra, dec: np.stack([np.cos(np.radians(dec)) * np.cos(np.radians(ra)), np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
                                  np.sin(np.radians(dec))], 1)
    ch = 2 * np.sin(np.radians(TH_EDGES_DEG) / 2)
    suba = sub[: min(len(sub), 3 * len(wd))]
    ud, us = u(ra_d, dec_d), u(ra_r[suba], dec_r[suba])
    one_a = np.ones(len(suba))
    fa = (len(wd) * (len(wd) - 1.0)) / (len(suba) * (len(suba) - 1.0))
    A1 = pair_counts(ud, one_d, ud, one_d, ch); A1[0] -= len(wd)
    Aw = pair_counts(ud, wd_w, ud, wd_w, ch); Aw[0] -= float(np.sum(wd_w ** 2))
    B1 = pair_counts(us, one_a, us, one_a, ch); B1[0] -= len(suba)
    Bw = pair_counts(us, wr_w[suba], us, wr_w[suba], ch); Bw[0] -= float(np.sum(wr_w[suba] ** 2))
    fw = (np.sum(wd_w) ** 2 - np.sum(wd_w ** 2)) / (np.sum(wr_w[suba]) ** 2 - np.sum(wr_w[suba] ** 2))
    o.update(A1=A1, Aw=Aw, B1e=B1 * fa, Bwe=Bw * fw)
    del ud, us
    log(f'  angular pair counts done ({time.time() - t0:.0f} s)')
    # 4. quad terms on the grid
    lo, L, n = grid['lo'], grid['L'], grid['n']
    th = lambda x: [np.ascontiguousarray(2 * np.pi * (x[:, i] - lo[i]) / L) for i in range(3)]
    thd, thr = th(xd), th(xr)
    fields = []
    for w_d, w_r in ((wd, wr), (wd_f, wr_f)):
        a2 = float(np.sum(w_d ** 2) / np.sum(w_r ** 2))
        fields.append((thd, w_d ** 2, float(np.sum(w_d ** 4))))
        fields.append((thr, a2 * w_r ** 2, float(np.sum((a2 * w_r ** 2) ** 2))))
    q = quad_terms(fields, None, n, grid['io'], grid['nb'], grid['nsh'], grid['rows'], grid['opts'])
    o['quad_raw'] = q                              # (4, nb, nb): data / randoms for total weights, then FKP-only; not / norm^2
    # close-pair parts in the same (not / norm^2) units: 2 x excess w^4 pair count below r_c, j0(q r) ~ 1
    o['close_excess'] = np.array([2 * np.sum((DD4 - RR4 * f4)[R_EDGES[1:] <= rc_]) for rc_ in args.rc])
    o['close_total'] = np.array([2 * np.sum(DD4[R_EDGES[1:] <= rc_]) for rc_ in args.rc])
    o['sum_w2'], o['sum_w4'] = float(np.sum(wd ** 2)), float(np.sum(wd ** 4))
    log(f'  quad terms done ({time.time() - t0:.0f} s)')
    return o


def report(name, o, k, args, G_units=None):
    print(f'\n== {name}: {o["n_gal"]} objects; duplicates: positions {o["dup_pos"]}, TARGETIDs {o["dup_tid"]}; '
          f'<w^2>/<w>^2 data {o["w2_over_w_sq"][0]:.4f} vs randoms {o["w2_over_w_sq"][1]:.4f} (WEIGHT only {o["w2_over_w_sq"][2]:.4f} / {o["w2_over_w_sq"][3]:.4f})')
    print('   r [Mpc/h]       ' + ' '.join(f'{a:g}-{b:g}'.rjust(9) for a, b in zip(R_EDGES[:-1], R_EDGES[1:])))
    print('   DD (pairs)      ' + ' '.join(f'{x / 2:9.0f}' for x in o['DD1']))
    print('   1 + xi  (unit)  ' + ' '.join(f'{d / max(e, 1e-30):9.2f}' if e > 0 else f'{"(DD " + str(int(d / 2)) + ")":>9s}' for d, e in zip(o['DD1'], o['RR1e'])))
    print('   1 + xi  (w^4)   ' + ' '.join(f'{d / max(e, 1e-30):9.2f}' if e > 0 else f'{"-":>9s}' for d, e in zip(o['DD4'], o['RR4e'])))
    print('   theta [deg]     ' + ' '.join(f'{a:g}-{b:g}'.rjust(11) for a, b in zip(TH_EDGES_DEG[:-1], TH_EDGES_DEG[1:])))
    print('   1+w(th) unit    ' + ' '.join(f'{d / max(e, 1e-30):11.3f}' for d, e in zip(o['A1'], o['B1e'])))
    print('   1+w(th) WEIGHT  ' + ' '.join(f'{d / max(e, 1e-30):11.3f}' for d, e in zip(o['Aw'], o['Bwe'])))
    q = o['quad_raw']
    rows = np.flatnonzero(k > 0.15)
    for lab, (fd, fr) in (('WEIGHT x FKP', (0, 1)), ('FKP only', (2, 3))):
        D, R = q[fd], q[fr]
        ex = D - R
        def band(M):
            return ' | '.join(f'{np.mean(np.diag(M)[(k >= lo_) & (k < hi_)]):.4g}' for lo_, hi_ in KR[1:])
        rel = ' | '.join(f'{np.mean(np.diag(ex)[(k >= lo_) & (k < hi_)]) / np.mean(np.diag(R)[(k >= lo_) & (k < hi_)]):+.3f}' for lo_, hi_ in KR[1:])
        print(f'   pair term [{lab}]: clustered / smooth-density part (diag, bands 0.05-0.1 | 0.1-0.2 | 0.2-0.3): {rel}')
        offs = [np.mean([ex[i, i + d] for i in rows if i + d < len(k)]) for d in OFFS]
        offs_R = [np.mean([R[i, i + d] for i in rows if i + d < len(k)]) for d in OFFS]
        print(f'      clustered part vs |i-j| = {OFFS} (k > 0.15), / smooth diag: ' + ' '.join(f'{x / np.mean(np.diag(R)[rows]):+.3f}' for x in offs)
              + '   [smooth part itself: ' + ' '.join(f'{x / np.mean(np.diag(R)[rows]):.3f}' for x in offs_R) + ']')
        if lab == 'WEIGHT x FKP':
            exd = np.mean(np.diag(ex)[rows])
            print('      close pairs r < ' + ', '.join(f'{rc_:g}' for rc_ in args.rc) + ' Mpc/h: excess / clustered diag part = '
                  + ' '.join(f'{x / exd:+.3f}' for x in o['close_excess']) + ' (all close pairs: ' + ' '.join(f'{x / exd:.3f}' for x in o['close_total']) + ')')
            if G_units is not None:
                norm, dG = G_units[:2]
                if G_units[2] is not None:
                    cf = G_units[2]
                    print(f'      cross-check, thinning_closed_form.npz (other grid): quad diag k 0.15-0.3 / diag G = '
                          f'{np.mean(np.diag(cf["quad"])[rows] / dG[rows]):.4f} (here {np.mean(np.diag(D)[rows]) / norm ** 2 / np.mean(dG[rows]):.4f}); '
                          f'P/n^2 = quad - g2 (thecov fit) = {np.mean(np.diag(cf["Pn2"])[rows] / dG[rows]) if "Pn2" in cf else float("nan"):+.4f}; '
                          f'g2 = {np.mean(np.diag(cf["g2"])[rows] / dG[rows]) if "g2" in cf else float("nan"):.4f} vs smooth part here '
                          f'{np.mean(np.diag(R)[rows]) / norm ** 2 / np.mean(dG[rows]):.4f}')
                print(f'      in units of diag G (mock 173 thecov): clustered P/n^2 diag k 0.15-0.3 = {exd / norm ** 2 / np.mean(dG[rows]):+.4f}; '
                      f'smooth (thecov SS-term) part {np.mean(np.diag(R)[rows]) / norm ** 2 / np.mean(dG[rows]):.4f}; close-pair excess (r < {args.rc[-1]:g}) '
                      f'{o["close_excess"][-1] / norm ** 2 / np.mean(dG[rows]):+.4f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='QSO')
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--mocks', type=int, nargs='*', default=[173])
    ap.add_argument('--data', action='store_true')
    ap.add_argument('--data-dir', default='/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2')
    ap.add_argument('--random-files', type=int, default=2)
    ap.add_argument('--ran-factor', type=int, default=5, help='randoms subsample for the pair counts, x N_data')
    ap.add_argument('--rc', type=float, nargs='+', default=[0.1, 0.5, 1.0])
    ap.add_argument('--kmin-rows', type=float, default=0.05, help='rows of the quad matrix computed (k >= this)')
    ap.add_argument('--pad', type=float, default=1.1)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--out', default=os.path.expanduser('~/thecov/thinning'))
    args = ap.parse_args()
    b, r = args.bin, args.region
    tracer, zr = dc.TRACER_SPECS[b]
    cfg = pl.Config()
    p0 = paths_for(False, 173, args.data_dir)
    s1 = dc.read_spectra(p0.spectra_fns(tracer, zr, r)[:1], kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=(0,))
    k, edges = s1['k'], np.asarray(s1['k_edges'])
    nb = len(k)
    # grid from mock 173's randoms (the same box for every catalogue of this cap)
    rc0 = dc.load_region(p0, b, r, n_random_files=1)
    xr0 = dc.sky_to_cartesian(rc0.randoms['RA'], rc0.randoms['DEC'], rc0.randoms['Z'])
    del rc0
    lo, hi = xr0.min(0) - 50, xr0.max(0) + 50
    L = args.pad * float(np.max(hi - lo))
    lo = 0.5 * (lo + hi) - 0.5 * L
    n = 2 * int(np.ceil(edges[-1] / (2 * np.pi / L))) + 4
    n += n % 2
    del xr0
    ko = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KO = np.meshgrid(ko, ko, ko, indexing='ij')
    kom = np.sqrt(KO[0] ** 2 + KO[1] ** 2 + KO[2] ** 2).ravel()
    del KO
    io = np.digitize(kom, edges) - 1
    del kom
    io[(io < 0) | (io >= nb)] = nb
    nsh = np.bincount(io, minlength=nb + 1)[:nb]
    grid = dict(lo=lo, L=L, n=n, io=io, nb=nb, nsh=nsh, rows=np.flatnonzero(k >= args.kmin_rows),
                opts=dict(eps=args.eps, nthreads=NTH, modeord=1))
    log(f'{b} {r}: box {L:.0f} Mpc/h, grid {n}^3, {nb} k bins, quad rows k >= {args.kmin_rows}')
    # mock 173's thecov units, if the thinning outputs exist
    G_units = None
    od173 = os.path.join(args.out, f'{b}_{r}_mock173')
    if os.path.exists(os.path.join(od173, 'thecov_p1.npz')) and os.path.exists(os.path.join(od173, 'thinning.npz')):
        g = np.load(os.path.join(od173, 'thinning.npz'))
        cfn = os.path.join(od173, 'thinning_closed_form.npz')
        G_units = (float(g['norm']), np.diag(np.load(os.path.join(od173, 'thecov_p1.npz'))['C_G'])[:nb],
                   dict(np.load(cfn)) if os.path.exists(cfn) else None)
    cats = [(m, False) for m in args.mocks] + ([(0, True)] if args.data else [])
    tag = '-'.join(str(m) for m in args.mocks) + ('_data' if args.data else '')
    fn = os.path.join(args.out, f'pairs_{b}_{r}_{tag}.npz')
    res = {}
    for m, data in cats:
        name = 'data' if data else f'mock{m}'
        try:
            o = one_catalogue(args, m, data, grid, edges)
        except Exception as e:  # noqa: BLE001
            log(f'{name}: FAILED {type(e).__name__}: {e}')
            continue
        res[name] = o
        report(name, o, k, args, G_units if (not data and m == 173) else None)
        np.savez(fn, k=k, r_edges=R_EDGES, th_edges=TH_EDGES_DEG, rc=np.array(args.rc), offs=np.array(OFFS),
                 **{f'{nm}/{key}': v for nm, oo in res.items() for key, v in oo.items()})
        log(f'saved {fn}')


if __name__ == '__main__':
    main()
