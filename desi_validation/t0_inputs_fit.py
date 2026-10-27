"""Catalogue-consistent inputs of the tree-level terms (HANDOFF 19): (b1, sigma_v, A) from ONE mock's own P0, P2 through
the DESI window matrix, and J4 = int W^4 / norm^2 as a function of the resolution the window is painted at.

    python -u -m desi_validation.t0_inputs_fit --stage fit --bin LRG1 --region NGC --mock 173 --params <ssc npz>
    python -u -m desi_validation.t0_inputs_fit --stage j4  --bin LRG1 --region NGC --mock 173 --random-files 18

fit: model P_l(k') = Kaiser-FoG(b1, f, sigma_v) x A P_ir(k') on the window matrix's theory grid, observed P_hat_l(k) = W P_l,
rebinned exactly as read_spectra rebins the data (every `rebin` bins, nmodes-weighted), compared with the mock's own P0, P2 at
kmin <= k <= kmax with Gaussian diagonal errors (P_l + SN)^2 2 / nmodes x (2l+1). Grid scan in (b1, sigma_v), A analytic;
also A = 1 fixed, and the pure-Kaiser (no window) fit of ssc_check.fit_b1 for reference. Output <out>/t0_inputs_fit.npz
and the slim window matrix (observable k <= kmax_dump, theory k <= 0.6) for offline work.
j4: randoms split into 4 disjoint quarters, each painted as W_q = 4 alpha sum w / V_c on cubic cells of side `cells`;
sum_c W_1 W_2 W_3 W_4 V_c and sum_c W_1 W_2 V_c are unbiased for int W^4 and int W^2 at that resolution (no Poisson bias of
the randoms). Ratios across cell sizes show how much the local (randoms' resolution) J4 of TrispectrumCovariance
exceeds the value at the scale of the modes (compare with the Gaussian's I_k / int W^2 = 0.77-0.85 for LRG1 NGC).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc  # noqa: E402
from desi_validation.grf_randoms import log  # noqa: E402
from desi_validation.thinning_cov import SPECTRA_BASE, setup_mock  # noqa: E402

WINDOW_DIR = '/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/cutsky_window_matrix_validation/holi-v3-altmtl/mock0'


def kaiser_fog_multipoles(k, b1, f, sigma_v, ells=(0, 2, 4), n_mu=64):
    """{l: K_l(k)} with P_l = K_l P: (b1 + f mu^2)^2 exp(-(k mu sigma_v)^2) projected on L_l (Gauss-Legendre in mu)"""
    x, w = np.polynomial.legendre.leggauss(n_mu)
    K = (b1 + f * x[None, :] ** 2) ** 2 * np.exp(-(np.asarray(k)[:, None] * x[None, :] * sigma_v) ** 2)
    return {l: (2 * l + 1) / 2 * np.sum(w * np.polynomial.legendre.Legendre.basis(l)(x) * K, axis=1) for l in ells}


def _coords(obj, name='k'):
    for fn in (lambda: np.asarray(obj.coords(name)), lambda: np.asarray(getattr(obj, name))):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            continue
    raise AttributeError(f'no {name} coordinates on {type(obj)}')


def _ells_of(obj):
    for fn in (lambda: [int(p.ell) for p in obj.projs], lambda: [int(p) for p in obj.ells],
               lambda: [int(getattr(p, 'ell', p)) for p in obj.projs]):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            continue
    return None


def dump_window(fn, out, kmax_dump=0.32, ktheory_max=0.6):
    """Read the lsstypes window matrix; print its structure; return dict(M, k_obs, ells_obs, k_th, ells_th) (slim)."""
    import lsstypes as types
    wm = types.read(fn)
    log(f'window matrix {os.path.basename(fn)}: type {type(wm).__name__}')
    log('  dir: ' + ' '.join(a for a in dir(wm) if not a.startswith('_')))
    for side in ('observable', 'theory'):
        try:
            o = getattr(wm, side)
            log(f'  {side}: type {type(o).__name__}; dir: ' + ' '.join(a for a in dir(o) if not a.startswith('_'))[:600])
        except Exception as e:  # noqa: BLE001
            log(f'  {side}: {e!r}')
    M = np.asarray(wm.value())
    log(f'  value shape {M.shape}')
    obs, th = wm.observable, wm.theory
    ells_o, ells_t = _ells_of(obs), _ells_of(th)
    log(f'  observable ells {ells_o}, theory ells {ells_t}')
    k_o = [np.asarray(obs.get(l).coords('k')) for l in ells_o]
    k_t = [np.asarray(th.get(l).coords('k')) for l in ells_t]
    log(f'  observable k per ell: {[len(x) for x in k_o]} (first {k_o[0][:3]}, last {k_o[0][-1]:.4f}); '
        f'theory k per ell: {[len(x) for x in k_t]} (first {k_t[0][:3]}, last {k_t[0][-1]:.4f})')
    try:
        edges_o = np.asarray(obs.get(ells_o[0]).edges('k'))
        log(f'  observable k edges (first 3): {edges_o[:3].tolist()}')
    except Exception as e:  # noqa: BLE001
        edges_o = None
        log(f'  observable edges: {e!r}')
    try:
        nmodes_o = np.asarray(obs.get(ells_o[0]).values('nmodes'))
    except Exception:  # noqa: BLE001
        nmodes_o = None
    # slim: rows with k_obs <= kmax_dump, columns with k_th <= ktheory_max (block layout by ell, then k)
    rows = np.concatenate([np.flatnonzero(x <= kmax_dump) + sum(len(y) for y in k_o[:i]) for i, x in enumerate(k_o)])
    cols = np.concatenate([np.flatnonzero(x <= ktheory_max) + sum(len(y) for y in k_t[:i]) for i, x in enumerate(k_t)])
    assert M.shape == (sum(len(x) for x in k_o), sum(len(x) for x in k_t)), (M.shape, [len(x) for x in k_o], [len(x) for x in k_t])
    slim = dict(M=M[np.ix_(rows, cols)].astype(np.float32), rows=rows, cols=cols, ells_obs=np.array(ells_o), ells_th=np.array(ells_t),
                k_obs=np.concatenate(k_o), k_th=np.concatenate(k_t), n_obs=np.array([len(x) for x in k_o]), n_th=np.array([len(x) for x in k_t]),
                edges_obs=edges_o if edges_o is not None else np.zeros(0), nmodes_obs=nmodes_o if nmodes_o is not None else np.zeros(0))
    np.savez(os.path.join(out, 'wmatrix_slim.npz'), **slim)
    return dict(M=M, k_obs=k_o, ells_obs=ells_o, k_th=k_t, ells_th=ells_t, edges_obs=edges_o, nmodes_obs=nmodes_o)


def read_one_spectrum(fn, ells=(0, 2, 4)):
    """full-resolution multipoles of one file + k, nmodes, norm, shotnoise (no rebin, no k cut)"""
    import lsstypes as types
    s = types.read(fn)
    p0 = s.get(ells[0])
    return dict(P={l: np.real(np.asarray(s.get(l).value())) for l in ells}, k=np.asarray(p0.coords('k')),
                nmodes=np.asarray(p0.values('nmodes')), norm=float(np.ravel(p0.values('norm'))[0]),
                shotnoise=float(np.ravel(p0.values('shotnoise'))[0]), num_shotnoise=float(np.ravel(p0.values('num_shotnoise'))[0]))


def rebin(v, nmodes, rebin_n):
    """nmodes-weighted mean of consecutive groups of `rebin_n` bins (drops the incomplete tail)"""
    n = len(v) // rebin_n * rebin_n
    v, w = np.asarray(v[:n]).reshape(-1, rebin_n), np.asarray(nmodes[:n], float).reshape(-1, rebin_n)
    return np.sum(v * w, 1) / np.sum(w, 1), np.sum(w, 1)


def stage_fit(args):
    paths, cfg, od = setup_mock(args)
    out = args.out_dir or od
    os.makedirs(out, exist_ok=True)
    b, r = args.bin, args.region
    tracer, zr = dc.TRACER_SPECS[b]
    wfn = os.path.join(WINDOW_DIR, f'window_mesh2_spectrum_poles_{tracer}_z{zr[0]}-{zr[1]}_{r}_weight-default-FKP_smooth_mesh.h5')
    W = dump_window(wfn, out)
    fn = os.path.join(paths.spectra_dir, f'mock{args.mock}', paths.spectra_name.format(tracer=tracer, zmin=zr[0], zmax=zr[1], region=r))
    S = read_one_spectrum(fn)
    log(f'spectrum {os.path.basename(fn)}: {len(S["k"])} bins, k {S["k"][0]:.4f}..{S["k"][-1]:.4f}, norm {S["norm"]:.4g}, shotnoise {S["shotnoise"]:.1f}')
    # observable grid of the window must be the spectrum's grid
    k_o = W['k_obs'][0]
    n_common = min(len(k_o), len(S['k']))
    assert np.allclose(k_o[:n_common], S['k'][:n_common], rtol=1e-3), (k_o[:5], S['k'][:5])
    zt = np.load(os.path.expanduser(args.params)); key = lambda q: zt[f'{r}/{q}']
    b1p, b2p, bs2p, fp, zeff = (float(x) for x in key('params')); ir_sigma, sv_p, A_p = (float(x) for x in key('damping'))
    kl, Pl, PirA = key('plin_k'), key('plin_P'), key('plin_P_damped_normalised')
    Pir = PirA / A_p                                                   # IR-damped linear power with A = 1
    log(f'production inputs: b1 {b1p:.3f} b2 {b2p:.3f} bs2 {bs2p:.3f} f {fp:.3f} sigma_v {sv_p:.3f} A {A_p:.4f}')
    f = fp
    ells_t, k_t = W['ells_th'], W['k_th']
    # data: rebinned as read_spectra, k window of the fit
    nm = S['nmodes']
    kd, _ = rebin(S['k'], nm, args.rebin)
    d = {l: rebin(S['P'][l], nm, args.rebin)[0] for l in (0, 2)}
    nmb = rebin(np.ones_like(nm), nm, args.rebin)[1]
    sel = (kd >= args.kmin) & (kd <= args.kmax)
    sn = S['shotnoise']
    sig = {0: (d[0] + sn) * np.sqrt(2 / nmb), 2: (d[0] + sn) * np.sqrt(2 / nmb) * np.sqrt(5)}   # Gaussian diagonal, l = 2 ~ sqrt(5) x l = 0
    n_o = [len(x) for x in W['k_obs']]
    M = W['M']

    def model(b1, sigma_v):
        vec = []
        for l, k in zip(ells_t, k_t):
            if l in (0, 2, 4):
                Kl = kaiser_fog_multipoles(k, b1, f, sigma_v, ells=(l,))[l]
                vec.append(Kl * np.interp(k, kl, Pir))
            else:
                vec.append(np.zeros(len(k)))                              # odd / wide-angle theory rows: not modelled
        conv = M @ np.concatenate(vec)
        out_ = {}
        for i, l in enumerate(W['ells_obs']):
            if l in (0, 2):
                v = conv[sum(n_o[:i]):sum(n_o[:i + 1])][:len(nm)]
                out_[l] = rebin(v, nm, args.rebin)[0]
        return out_

    def chi2_A(b1, sigma_v, A=None):
        m = model(b1, sigma_v)
        num = sum(np.sum(d[l][sel] * m[l][sel] / sig[l][sel] ** 2) for l in (0, 2))
        den = sum(np.sum(m[l][sel] ** 2 / sig[l][sel] ** 2) for l in (0, 2))
        A_ = num / den if A is None else A
        c2 = sum(np.sum((d[l][sel] - A_ * m[l][sel]) ** 2 / sig[l][sel] ** 2) for l in (0, 2))
        return c2, A_
    b1s, svs = np.arange(1.5, 2.61, 0.01), np.arange(0.0, 8.01, 0.25)
    C2 = np.empty((len(b1s), len(svs))); AA = np.empty_like(C2); C2_A1 = np.empty_like(C2)
    t0 = time.time()
    for i, b1 in enumerate(b1s):
        for j, sv in enumerate(svs):
            C2[i, j], AA[i, j] = chi2_A(b1, sv)
            C2_A1[i, j], _ = chi2_A(b1, sv, A=1.0)
    i, j = np.unravel_index(np.argmin(C2), C2.shape); i1, j1 = np.unravel_index(np.argmin(C2_A1), C2.shape)
    log(f'window-convolved fit to mock {args.mock} P0, P2 at {args.kmin} <= k <= {args.kmax} ({sel.sum()} bins per l; {time.time() - t0:.0f} s):')
    log(f'  A free : b1 {b1s[i]:.3f}  sigma_v {svs[j]:.2f}  A {AA[i, j]:.4f}  chi2 {C2[i, j]:.1f}  (sqrt(A) b1 = {np.sqrt(AA[i, j]) * b1s[i]:.3f})')
    log(f'  A = 1  : b1 {b1s[i1]:.3f}  sigma_v {svs[j1]:.2f}  chi2 {C2_A1[i1, j1]:.1f}')
    # 1-sigma ranges (delta chi2 < 1 after profiling) for A free
    prof = C2.min(axis=1); ok = prof <= prof.min() + 1.0
    log(f'  b1 range (dchi2 < 1, A free): {b1s[ok].min():.3f} - {b1s[ok].max():.3f}')
    # pure-Kaiser reference fit (ssc_check.fit_b1) on the same data, k <= 0.06
    from desi_validation.ssc_check import fit_b1, fit_sigma_v, fit_amplitude
    sel6 = kd <= 0.06 + 1e-9
    b1_k, d_k = fit_b1(kd[sel6], d[0][sel6], d[2][sel6], lambda k: np.interp(k, kl, Pir), f)
    log(f'  reference (no window, ssc_check.fit_b1 at k <= 0.06): b1 {b1_k:.3f}, amplitude d = {d_k:.3f}')
    # convolved vs unconvolved P2/P0 ratio of the best model, to show the window effect
    m = model(b1s[i], svs[j]); Kl = kaiser_fog_multipoles(kd, b1s[i], f, svs[j])
    for kk_ in (0.03, 0.05, 0.08, 0.12):
        jx = np.argmin(np.abs(kd - kk_))
        log(f'  k = {kd[jx]:.3f}: P2/P0 data {d[2][jx] / d[0][jx]:+.3f}  convolved model {m[2][jx] / m[0][jx]:+.3f}  unconvolved Kaiser-FoG {Kl[2][jx] / Kl[0][jx]:+.3f}')
    np.savez(os.path.join(out, 't0_inputs_fit.npz'), b1_grid=b1s, sigma_v_grid=svs, chi2=C2, A=AA, chi2_A1=C2_A1,
             best=np.array([b1s[i], svs[j], AA[i, j], C2[i, j]]), best_A1=np.array([b1s[i1], svs[j1], 1.0, C2_A1[i1, j1]]),
             reference_kaiser=np.array([b1_k, d_k]), k=kd, P0=d[0], P2=d[2], sig0=sig[0], sig2=sig[2], sel=sel,
             model0=m[0], model2=m[2], production=np.array([b1p, b2p, bs2p, fp, sv_p, A_p]), mock=args.mock)
    log(f'saved {out}/t0_inputs_fit.npz')


def stage_j4(args):
    paths, cfg, od = setup_mock(args)
    out = args.out_dir or od
    os.makedirs(out, exist_ok=True)
    b, r = args.bin, args.region
    rc = dc.load_region(paths, b, r, n_random_files=args.random_files)
    wd, wr = dc.total_weight(rc.data), dc.total_weight(rc.randoms)
    xr = dc.sky_to_cartesian(rc.randoms['RA'], rc.randoms['DEC'], rc.randoms['Z'])
    alpha = float(np.sum(wd) / np.sum(wr))
    del rc
    lo, hi = xr.min(0), xr.max(0)
    rng = np.random.default_rng(args.seed)
    q = rng.integers(0, 4, len(wr))
    log(f'{b} {r} mock {args.mock}: {len(wr)} randoms ({args.random_files} files), alpha {alpha:.4f}, extent {np.round(hi - lo).tolist()}')
    res = {}
    for cell in args.cells:
        n = np.ceil((hi - lo) / cell).astype(int) + 1
        idx = np.floor((xr - lo) / cell).astype(int)
        flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), n)
        Vc = cell ** 3
        Wq = [np.bincount(flat[q == a], 4 * alpha * wr[q == a], minlength=int(np.prod(n))) / Vc for a in range(4)]
        Wall = np.bincount(flat, alpha * wr, minlength=int(np.prod(n))) / Vc
        I2_cross, I4_cross = float(np.sum(Wq[0] * Wq[1]) * Vc), float(np.sum(Wq[0] * Wq[1] * Wq[2] * Wq[3]) * Vc)
        I2_naive, I4_naive = float(np.sum(Wall ** 2) * Vc), float(np.sum(Wall ** 4) * Vc)
        I1 = float(np.sum(Wall) * Vc)
        res[cell] = (I1, I2_cross, I4_cross, I2_naive, I4_naive)
        log(f'  cell {cell:5.1f}: int W {I1:.4e}  int W^2 cross {I2_cross:.4e} (naive {I2_naive:.4e})  int W^4 cross {I4_cross:.4e} '
            f'(naive {I4_naive:.4e})  J4 x I2^2 = int W^4 / (int W^2)^2 = {I4_cross / I2_cross ** 2:.4e}')
        del Wq, Wall, flat, idx
    cells = np.array(sorted(res))
    arr = np.array([res[c] for c in cells])
    np.savez(os.path.join(out, 'j4_cells.npz'), cells=cells, I1=arr[:, 0], I2=arr[:, 1], I4=arr[:, 2], I2_naive=arr[:, 3], I4_naive=arr[:, 4], alpha=alpha)
    log(f'saved {out}/j4_cells.npz')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['fit', 'j4'], required=True)
    ap.add_argument('--bin', default='LRG1'); ap.add_argument('--region', default='NGC'); ap.add_argument('--mock', type=int, default=173)
    ap.add_argument('--cs-version', default=None)
    ap.add_argument('--params', default=None, help='ssc_check npz with <region>/params, damping, plin_* (fit)')
    ap.add_argument('--kmin', type=float, default=0.02); ap.add_argument('--kmax', type=float, default=0.15)
    ap.add_argument('--rebin', type=int, default=5)
    ap.add_argument('--random-files', type=int, default=18)
    ap.add_argument('--cells', type=float, nargs='+', default=[3.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args(argv)
    (stage_fit if args.stage == 'fit' else stage_j4)(args)


if __name__ == '__main__':
    main()
