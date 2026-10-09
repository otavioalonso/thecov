"""Figure: the veto holes. (a) fill fraction f of the LRG1 NGC footprint (randoms per pixel over the expectation for a
full pixel, nside 256, ~14'); (b) a 4 x 4 deg zoom at nside 2048 (1.7'), where the individual holes (bright stars,
bad imaging) are resolved; (c) R_PP = <f^2>/<f>^2 (m^4-weighted) against the pixel scale: the extra Gaussian variance
the hole-fraction inhomogeneity can add relative to the local window m^2, flat from ~7' to ~1 deg.
Needs products/paper_maps_LRG1.npz and hole_fraction_<b>.json (desi_validation/paper_export.sh)."""
import json
import os
from common import *

fn = os.path.join(PROD, 'paper_maps_LRG1.npz')
if not os.path.exists(fn):
    sys.exit(f'{fn} missing: run desi_validation/paper_export.sh at NERSC, then make_products.py')
import healpy as hp

z = load('paper_maps_LRG1')
R = 'NGC'
pix, n, E, nside = z[f'{R}/pix'], z[f'{R}/n'].astype(float), float(z[f'{R}/E']), int(z[f'{R}/nside'])


def fill_map(ns):
    shift = 2 * int(np.log2(nside // ns))
    m = np.full(hp.nside2npix(ns), hp.UNSEEN)
    c = np.bincount(pix >> shift, weights=n, minlength=hp.nside2npix(ns))
    occ = c > 0
    m[occ] = c[occ] / (E * 4 ** (shift // 2))
    return hp.reorder(m, n2r=True)


fig = plt.figure(figsize=(TEXTWIDTH, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1.05], wspace=0.6)
m256 = fill_map(256)
th, ph = hp.pix2ang(256, np.flatnonzero(m256 != hp.UNSEEN), lonlat=True)
ra_c = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(th))))) % 360
proj = hp.projector.CartesianProj(rot=(ra_c, 0), lonra=[-60, 60], latra=[-15, 85], xsize=900)
img = proj.projmap(m256, lambda x, y, zz: hp.vec2pix(256, x, y, zz))
ax = fig.add_subplot(gs[0])
im = ax.imshow(np.ma.masked_less(img, -1e30), origin='lower', extent=(ra_c + 60, ra_c - 60, -15, 85), cmap=SEQUENTIAL,
               vmin=0.6, vmax=1.0, interpolation='nearest', aspect='auto')
ax.set_xlabel('RA [deg]')
ax.set_ylabel('Dec [deg]')
ax.set_title('(a) fill fraction, nside 256', fontsize=8)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, ticks=[0.6, 0.8, 1.0])
# zoom on a well-populated patch
m2048 = fill_map(2048)
good = np.flatnonzero(m2048 != hp.UNSEEN)
tz, pz = hp.pix2ang(2048, good, lonlat=True)
ra0, dec0 = np.median(tz), np.median(pz)
gp = hp.projector.GnomonicProj(rot=(ra0, dec0), xsize=500, reso=4 * 60 / 500)
zoom = gp.projmap(m2048, lambda x, y, zz: hp.vec2pix(2048, x, y, zz))
ax.add_patch(matplotlib.patches.Rectangle((ra0 - 2, dec0 - 2), 4, 4, fill=False, color=RED, lw=1.0))
ax = fig.add_subplot(gs[1])
ax.imshow(np.ma.masked_less(zoom, -1e30), origin='lower', extent=(2, -2, -2, 2), cmap=SEQUENTIAL, vmin=0, vmax=1.2,
          interpolation='nearest')
ax.set_xlabel(r'$\Delta$RA [deg]')
ax.set_ylabel(r'$\Delta$Dec [deg]')
ax.set_title(f'(b) zoom, nside 2048', fontsize=8)
ax.grid(False)
ax = fig.add_subplot(gs[2])
for b, lw in (('LRG1', 1.4), ('QSO', 0.9)):
    jf = os.path.join(PROD, f'hole_fraction_{b}.json')
    if not os.path.exists(jf):
        continue
    res = json.load(open(jf))
    for cap, ls in (('NGC', '-'), ('SGC', '--')):
        st = res[cap]['stats']['local-mean-weight']
        ax.plot([s['scale_arcmin'] / 60 for s in st], [s['R_PP'] for s in st], ls=ls, lw=lw, marker='o', ms=2.5,
                color=BLUE if b == 'LRG1' else ORANGE, label=f'{b} {cap}')
ax.set_xscale('log')
ax.axhline(1, color=INK2, lw=0.7)
ax.set_xlabel('pixel scale [deg]')
ax.set_ylabel(r'$R_{PP} = \langle f^2\rangle / \langle f\rangle^2$')
ax.set_title('(c) hole-fraction variance', fontsize=8)
ax.legend(fontsize=6.3)
save(fig, 'holes')
