"""What the veto holes look like, locally and after the pair averaging of the xi-kernel window (LRG1).

    python -u -m desi_validation.plot_holes --bin LRG1 --region NGC --out holes_LRG1.png

Panels: (a) the randoms in a thin shell around chi0 in a small sky patch (the holes are the empty regions);
(b) the radial weight 4 pi r^2 K_k(r) of the kernels (first, middle, last k bin); (c, d) the smoothed window
(K_k * m)(x) / m_bar on the shell for the middle and last bins, by direct sums over the randoms in 3D,
with K_k truncated at --rmax and renormalised there. m_bar: the patch median of the smoothed window far from
edges, so 1 = no dilution. The local window thecov uses by default is m (= m_bar between the holes, 0 inside).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from desi_validation import desi_compare as dc  # noqa: E402
from desi_validation.run_gaussian_footprint import setup  # noqa: E402
from thecov.smoothing import extend_power, xi0_from_power, bin_kernels  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--random-files', type=int, default=10)
    ap.add_argument('--size', type=float, default=1.5, help='patch side [deg]')
    ap.add_argument('--npix', type=int, default=150, help='grid points per side for the smoothed maps')
    ap.add_argument('--rmax', type=float, default=60.0, help='kernel truncation [Mpc/h]')
    ap.add_argument('--center', type=float, nargs=2, default=None, help='RA DEC of the patch (default: footprint median)')
    ap.add_argument('--out', default='holes.png')
    ap.add_argument('--hole-files', type=int, default=18, help='random files for the dense hole view (all z)')
    ap.add_argument('--zoom', type=float, default=0.3, help='side of the zoomed hole view [deg]')
    args = ap.parse_args()
    import types
    paths, cfg, OUT = setup(types.SimpleNamespace(cs_version=None, spectra_dir=None, mock=None, synthetic=None))
    b, r = args.bin, args.region
    tracer, zr = dc.TRACER_SPECS[b]
    s = dc.read_spectra(paths.spectra_fns(tracer, zr, r), kmin=cfg.kmin, kmax=cfg.kmax, rebin=cfg.rebin, ells=cfg.ells)
    k, edges, nb = s['k'], np.asarray(s['k_edges']), len(s['k'])
    rr = np.concatenate([np.linspace(0.01, 1, 100), np.linspace(1.02, args.rmax, 1500)])
    kk, pk = extend_power(k, s['vectors'][:, :nb].mean(0))
    K = bin_kernels(edges, rr, xi0_from_power(kk, pk, rr))
    sel = {'first': 0, 'middle': nb // 2, 'last': nb - 1}

    rc = dc.load_region(paths, b, r, n_random_files=args.random_files)
    rc_nall = rc.n_randoms_all_z
    ra, dec, z = (np.asarray(rc.randoms[c]) for c in ('RA', 'DEC', 'Z'))
    w = dc.total_weight(rc.randoms)
    ra0, dec0 = args.center or (float(np.median(ra)), float(np.median(dec)))
    zmid = 0.5 * sum(zr)
    x = dc.sky_to_cartesian(ra, dec, z)
    chi0 = float(np.linalg.norm(dc.sky_to_cartesian(np.array([ra0]), np.array([dec0]), np.array([zmid]))[0]))
    dra = args.size / 2 / np.cos(np.radians(dec0))
    pad = np.degrees(args.rmax / chi0) * 1.2
    near = (np.abs(ra - ra0) < dra + pad / np.cos(np.radians(dec0))) & (np.abs(dec - dec0) < args.size / 2 + pad) & \
           (np.abs(np.linalg.norm(x, axis=1) - chi0) < args.rmax)
    xs, ws = x[near], w[near]
    proj = (np.abs(ra - ra0) < dra) & (np.abs(dec - dec0) < args.size / 2)      # all depths: the holes are angular
    print(f'{b} {r}: patch ({ra0:.2f}, {dec0:.2f}) {args.size} deg at chi0 = {chi0:.0f} Mpc/h; '
          f'{near.sum()} randoms within the kernel reach, {proj.sum()} in the patch (all z)', flush=True)

    # The holes are angular and m(x) = nbar(chi) f(theta) separates, so on the shell (K_k * m) / nbar = (K2_k * f)(theta)
    # with the projected kernel K2(s) = int dl K(sqrt(s^2 + l^2)); summed over ALL randoms of the patch (all z) this has
    # little Poisson noise. f is normalised by the randoms' known surface density (cfg.surface_density per file).
    nbar2 = len(ra) / (rc_nall / (cfg.surface_density * args.random_files))      # randoms per deg^2 in the z range
    ll = np.linspace(-args.rmax, args.rmax, 801)
    ss = np.linspace(0, args.rmax, 600)
    from scipy.signal import fftconvolve
    pix = 0.01                                                     # deg (~0.23 Mpc/h at chi0)
    reach = (np.abs(ra - ra0) < dra + pad / np.cos(np.radians(dec0))) & (np.abs(dec - dec0) < args.size / 2 + pad)
    cosd = np.cos(np.radians(dec0))
    u, v = (ra[reach] - ra0) * cosd, dec[reach] - dec0               # flat-sky offsets [deg]
    half = args.size / 2 + pad
    ed = np.arange(-half, half + pix, pix)
    C, _, _ = np.histogram2d(u, v, bins=(ed, ed))
    C = C.T / (nbar2 * pix ** 2)                                     # local fill fraction f per pixel (noisy)
    cen = 0.5 * (ed[1:] + ed[:-1])
    th = np.hypot(*np.meshgrid(cen, cen)) * np.radians(1) * chi0     # separation on the shell [Mpc/h]
    keep = (np.abs(cen) <= args.size / 2)
    maps = {}
    for name in ('middle', 'last'):
        K3 = lambda r_: np.interp(r_, rr, K[sel[name]], right=0.0)
        K2 = np.array([np.trapezoid(K3(np.sqrt(s_ ** 2 + ll ** 2)), ll) for s_ in ss])
        img = np.interp(th, ss, K2, right=0.0)
        img /= img.sum()                                             # unit sum on the grid
        sm = fftconvolve(C, img, mode='same')
        maps[name] = sm[np.ix_(keep, keep)]
        print(f'  {name} bin (k = {k[sel[name]]:.3f}): smoothed fill: min {maps[name].min():.3f}, '
              f'median {np.median(maps[name]):.3f}, max {maps[name].max():.3f}', flush=True)
    g_ra = ra0 + cen[keep] / cosd
    g_dec = dec0 + cen[keep]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 9.4))
    sr, sd = ra[proj], dec[proj]
    a = ax[0, 0]
    a.scatter(sr, sd, s=0.15, c='k', lw=0, rasterized=True)
    a.set_title(f'(a) randoms in the patch ({args.random_files} files, all z): holes are the gaps', fontsize=10)
    a = ax[0, 1]
    for name, c_ in zip(sel, ('#888888', '#1f5fa8', '#c0392b')):
        a.plot(rr, 4 * np.pi * rr ** 2 * K[sel[name]], color=c_, label=f'{name} bin, k = {k[sel[name]]:.3f}')
    a.axhline(0, color='0.6', lw=0.6)
    a.set_xscale('log'); a.set_xlabel('r [Mpc/h]'); a.set_ylabel(r'$4\pi r^2 K_k(r)$')
    a.set_title('(b) kernel radial weight (unit integral)', fontsize=10); a.legend(frameon=False, fontsize=8)
    ext = (g_ra[0], g_ra[-1], g_dec[0], g_dec[-1])
    for a, name, lab in ((ax[1, 0], 'middle', '(c)'), (ax[1, 1], 'last', '(d)')):
        im = a.imshow(maps[name], origin='lower', extent=ext, aspect='auto', cmap='viridis', vmin=0.7, vmax=1.05)
        a.set_title(f'{lab} angular fill smoothed by the kernel, $(K_k * m)/\\bar n$, k = {k[sel[name]]:.3f}', fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046)
    for a in (ax[0, 0], ax[1, 0], ax[1, 1]):
        a.set_xlim(g_ra[-1], g_ra[0]); a.set_ylim(g_dec[0], g_dec[-1])
        a.set_xlabel('RA [deg]'); a.set_ylabel('DEC [deg]')
    fig.suptitle(f'{b} {r}: veto holes and the pair-averaged window ({args.size} deg patch, '
                 f'{args.size * np.radians(1) * chi0:.0f} Mpc/h across)', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print('saved', args.out)

    # dense hole view: all random files, NO redshift cut (the holes are angular)
    spec0 = dc.TRACER_SPECS[b]
    dc.TRACER_SPECS[b] = (spec0[0], (0.0, 10.0))         # load_region cuts to the z range: open it
    try:
        rcA = dc.load_region(paths, b, r, n_random_files=args.hole_files)
    except FileNotFoundError:
        rcA = dc.load_region(paths, b, r, n_random_files=args.random_files)
    dc.TRACER_SPECS[b] = spec0
    raA, decA = np.asarray(rcA.randoms['RA']), np.asarray(rcA.randoms['DEC'])
    nfA = rcA.n_random_files if hasattr(rcA, 'n_random_files') else args.hole_files
    del rcA
    figH, axH = plt.subplots(1, 2, figsize=(11, 5.2))
    dzr = args.zoom / 2 / np.cos(np.radians(dec0))
    pw = (np.abs(raA - ra0) < dra) & (np.abs(decA - dec0) < args.size / 2)
    H, xe, ye = np.histogram2d(raA[pw], decA[pw], bins=(int(2 * dra * 60), int(args.size * 60)))
    im = axH[0].imshow(H.T, origin='lower', extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect='auto', cmap='gray_r')
    axH[0].set_xlim(xe[-1], xe[0]); figH.colorbar(im, ax=axH[0], fraction=0.046, label='randoms per arcmin$^2$ pixel')
    axH[0].add_patch(plt.Rectangle((ra0 - dzr, dec0 - args.zoom / 2), 2 * dzr, args.zoom, fill=False, color='r', lw=1))
    axH[0].set_title(f'{args.size} deg patch, {nfA} files: counts per 1 arcmin pixel', fontsize=10)
    pz = (np.abs(raA - ra0) < dzr) & (np.abs(decA - dec0) < args.zoom / 2)
    axH[1].scatter(raA[pz], decA[pz], s=1.2, c='k', lw=0)
    axH[1].set_xlim(ra0 + dzr, ra0 - dzr); axH[1].set_ylim(dec0 - args.zoom / 2, dec0 + args.zoom / 2)
    axH[1].set_title(f'zoom {args.zoom} deg ({args.zoom * np.radians(1) * chi0:.1f} Mpc/h): {pz.sum()} randoms', fontsize=10)
    for a in axH:
        a.set_xlabel('RA [deg]'); a.set_ylabel('DEC [deg]')
    figH.tight_layout(); figH.savefig(args.out.replace('.png', '_dense.png'), dpi=170)
    print('saved', args.out.replace('.png', '_dense.png'), f'({pz.sum()} randoms in the zoom)', flush=True)


    # comparison from the dense randoms (all files, all z): raw fill vs kernel-smoothed fill, same randoms
    nbarA = cfg.surface_density * nfA                                 # all z: exactly the generated density
    rA = (np.abs(raA - ra0) < dra + pad / cosd) & (np.abs(decA - dec0) < half)
    uA, vA = (raA[rA] - ra0) * cosd, decA[rA] - dec0
    CA = np.histogram2d(uA, vA, bins=(ed, ed))[0].T / (nbarA * pix ** 2)
    raw_ed = np.arange(-args.size / 2, args.size / 2 + 1e-9, 1 / 60)   # 1 arcmin pixels for the raw view
    inside = (np.abs(uA) < args.size / 2) & (np.abs(vA) < args.size / 2)
    RAW = np.histogram2d(uA[inside], vA[inside], bins=(raw_ed, raw_ed))[0].T / (nbarA / 3600)
    figC, axC = plt.subplots(1, 3, figsize=(16, 5.0))
    extC = (ra0 - args.size / 2 / cosd, ra0 + args.size / 2 / cosd, dec0 - args.size / 2, dec0 + args.size / 2)
    im = axC[0].imshow(RAW, origin='lower', extent=extC, aspect='auto', cmap='gray', vmin=0, vmax=1.6)
    figC.colorbar(im, ax=axC[0], fraction=0.046, label='local fill (randoms / expected)')
    axC[0].set_title(f'raw fill, 1 arcmin pixels (~{nbarA / 3600:.0f} randoms expected per pixel)', fontsize=10)
    for a, name in zip(axC[1:], ('middle', 'last')):
        K3 = lambda r_: np.interp(r_, rr, K[sel[name]], right=0.0)
        K2 = np.array([np.trapezoid(K3(np.sqrt(s_ ** 2 + ll ** 2)), ll) for s_ in ss])
        img = np.interp(th, ss, K2, right=0.0)
        smA = fftconvolve(CA, img / img.sum(), mode='same')[np.ix_(keep, keep)]
        print(f'  dense {name} bin: smoothed fill min {smA.min():.3f}, median {np.median(smA):.3f}, max {smA.max():.3f}', flush=True)
        im = a.imshow(smA, origin='lower', extent=extC, aspect='auto', cmap='viridis', vmin=0.7, vmax=1.05)
        figC.colorbar(im, ax=a, fraction=0.046, label='smoothed fill')
        a.set_title(f'kernel-smoothed fill, k = {k[sel[name]]:.3f}', fontsize=10)
    for a in axC:
        a.set_xlim(extC[1], extC[0]); a.set_xlabel('RA [deg]'); a.set_ylabel('DEC [deg]')
    figC.suptitle(f'{b} {r}: the same {nfA} random files (all z), before and after the pair averaging', fontsize=11)
    figC.tight_layout(); figC.savefig(args.out.replace('.png', '_compare.png'), dpi=170)
    print('saved', args.out.replace('.png', '_compare.png'), flush=True)


if __name__ == '__main__':
    main()
