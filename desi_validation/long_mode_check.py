"""The survey-mean long mode of the mocks against thecov's sigma^2 (SSC review 4.3, radial part; 2026-10-27).

    python -m desi_validation.long_mode_check --nz nz_scatter_LRG1.npz --ssc-json ssc_small.json --bundle t0_inputs.npz

Inputs: nz_scatter_<bin>.npz (WEIGHT-weighted n(z) counts per mock, 20 shells), the sigma2 / params entries of the ssc
npz (as json), the t0 bundle (excluded mocks, mock vectors, C_ssc, dilution, P_lin). Prints
  - Var over mocks of the M-weighted survey-mean galaxy overdensity D (Poisson 1.1/N removed) against
    g0^2 s_00 + 2 g0 g2 s_02 + g2^2 s_22 (observed long mode g0 delta_0 + g2 delta_2);
  - the shell-resolved clustering covariance of n(z);
  - an upper bound on the mock-fair SSC: all long-mode variance taken as locally radial (nu = 1: response
    (R^(0) + R^(2)) / (b1 + f) per unit galaxy overdensity, randoms following the data so lam_W = 2), i.e.
    Var(D) x c c^T, against the clustering part of thecov's C_ssc (C_ssc minus its projection on P_hat, the Poisson LA);
    the n(z) data constrain the survey-mean variance only, not its split over the n = 0, 2 projections.
Result (LRG1 NGC, 989 mocks): std(D) 0.448 % (clustering 0.430 %) vs model 0.476 %: variance ratio 0.82.
"""
import argparse, json, sys, os
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from thecov.tree.responses import response_coefficients  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nz', required=True); ap.add_argument('--ssc-json', required=True); ap.add_argument('--bundle', required=True)
    ap.add_argument('--tracer', default='LRG1'); ap.add_argument('--region', default='NGC')
    args = ap.parse_args()
    d = json.load(open(args.ssc_json)); r = args.region
    sig = {tuple(kk.split('|')): v for kk, v in zip(d[f'{r}/sigma2_keys'], d[f'{r}/sigma2'])}
    S2 = lambda X, Y: sig.get((X, Y), sig.get((Y, X)))
    Z = np.load(args.bundle); excl = set(Z[f'excluded_{args.tracer}'].tolist()) if f'excluded_{args.tracer}' in Z.files else set()
    nz = np.load(args.nz); NZ, ids, edges = nz[f'{r}/NZ'], nz[f'{r}/mocks'], nz['edges']
    keep = np.array([(i not in excl) and np.all(np.isfinite(NZ[j])) and NZ[j].sum() > 0 for j, i in enumerate(ids)]); NZ = NZ[keep]
    b1, b2, bs2, f = d[f'{r}/params'][:4]; g0, g2 = b1 + f / 3, 2 * f / 3
    tot = NZ.sum(1); D = tot / tot.mean() - 1; var = D.var(ddof=1); pois = 1.1 / tot.mean()
    model = g0 ** 2 * S2("('M', 0)", "('M', 0)") + 2 * g0 * g2 * S2("('M', 0)", "('M', 2)") + g2 ** 2 * S2("('M', 2)", "('M', 2)")
    print(f'{args.tracer} {r}, {keep.sum()} mocks: std(D) {np.sqrt(var) * 100:.3f} % (Poisson {np.sqrt(pois) * 100:.3f} %, clustering '
          f'{np.sqrt(max(var - pois, 0)) * 100:.3f} %) vs thecov {np.sqrt(model) * 100:.3f} %: variance ratio {(var - pois) / model:.2f}')
    dz = NZ / NZ.mean(0) - 1; Cclu = np.cov(dz.T) - np.diag(1.1 / NZ.mean(0))
    print('shell clustering std [%]:', np.round(np.sqrt(np.clip(np.diag(Cclu), 0, None)) * 100, 2))
    F = lambda key: Z[f'F/{args.tracer}/{r}/{key}'].astype(float); k = F('k'); nb = len(k); dil = np.broadcast_to(F('dilution'), (nb,))
    spl = CubicSpline(np.log(F('plin_k')), F('plin_P_damped_normalised')); Pb, dPb = spl(np.log(k)), spl(np.log(k), 1)
    a, c = response_coefficients(b1, f, b2, bs2); ELL = (0, 2, 4)
    R = {(l, n): a[i, j] * Pb + c[i, j] * dPb for i, l in enumerate(ELL) for j, n in enumerate(ELL)}
    V = Z[f'MV/{args.tracer}/{r}/vectors'].astype(float); mids = Z[f'MV/{args.tracer}/{r}/mocks']; Vc = V[~np.isin(mids, list(excl))]; Pm = Vc.mean(0)
    cvec = np.concatenate([dil * (R[(l, 0)] + R[(l, 2)]) / (b1 + f) - 2 * Pm[i * nb:(i + 1) * nb] for i, l in enumerate(ELL)])
    Crad = (var - pois) * np.outer(cvec, cvec)
    ssc = F('C_ssc'); p = Pm / np.linalg.norm(Pm); Cclust = ssc - (p @ ssc @ p) * np.outer(p, p)
    G = None
    for l in range(3):
        s_ = slice(l * nb, (l + 1) * nb)
        print(f'P{2 * l}: diag ratio radial-bound / thecov clustering SSC at k > 0.1: {np.mean(np.diag(Crad)[s_][k > 0.1] / np.diag(Cclust)[s_][k > 0.1]):.2f}')


if __name__ == '__main__':
    main()
