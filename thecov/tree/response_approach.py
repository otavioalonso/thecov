"""Response approach to the T0 covariance (under exploration; not used by default).

Below a scale k_split the tree-level trispectrum is kept; for squeezed configurations (one long, one hard bin) the
hard power responds to the realised long mode; the collapsed term T ~ B B / P_L replaces the hard-hard tree level.
Functions: squeezed_response, collapsed_multipoles (per k pair), collapsed_covariance (binned, window factor J4),
response_split (LL / LH / HH blocks and the completion), response_covariance (their sum).
"""
from __future__ import annotations

import numpy as np

from ._common import fill_symmetric
from .trispectrum import Z1, Z2


def squeezed_response(k, q, P, bias, f):
    """R(k, q) P(k) = B_ggm(k, -k + q, -q) / P_L(q) at tree level for vectors k (hard) and q (soft), n^ = z^:
    2 Z2(k, -q) Z1(k) P(k) + 2 Z2(-k + q, -q) Z1(-k + q) P(|k - q|), evaluated at finite q (its bulk-flow 1/q terms
    cancel between the two terms, so it is finite as q -> 0; for q -> 0 it is the SSC response). P: callable of |k|."""
    kb = -k + q
    return (2 * Z2(k, -q, bias, f) * Z1(k, bias, f) * P(np.sqrt(np.sum(k * k, 0)))
            + 2 * Z2(kb, -q, bias, f) * Z1(kb, bias, f) * P(np.sqrt(np.sum(kb * kb, 0))))


def collapsed_multipoles(k1, k2, P, bias, f, k_split, ells=(0, 2, 4), n_mu=20, n_psi=40, n_q=32, sigma_fog=0.0,
                         P_long=None):
    """Response-approach collapsed term: for |k1 -+ k2| = p < k_split the four-point function factorises into two
    squeezed bispectra, T ~ B(k1, -k2, -p) B(-k1, k2, p) / P_L(p) = R(k1, p)^2 P(k1)^2 P_L(p), plus the same for
    p = k1 + k2 (equal by parity for even l): T_coll = 2 [R(k1, p) P(k1)]^2 P_L(p). IR safe (each factor is),
    positive configuration by configuration. Returns {(l1, l2): (2 l1 + 1)(2 l2 + 1) < L L T_coll >} with the
    same normalisation as multipoles (the mu12 average restricted to p < k_split, done in ln p).
    P: the hard power (e.g. normalised, BAO-damped), P_long: the long-mode power (default P)."""
    P_long = P if P_long is None else P_long
    out = {(l1, l2): 0.0 for l1 in ells for l2 in ells}
    pmin = max(abs(k1 - k2), 1e-5)
    if pmin >= k_split:
        return out
    t, wt = np.polynomial.legendre.leggauss(n_q)
    lo, hi = np.log(pmin), np.log(k_split)
    pq = np.exp(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
    wq = 0.5 * (hi - lo) * wt * pq ** 2 / (2 * k1 * k2)            # dmu12 / 2 = p dp / (2 k1 k2)
    mu = np.clip((k1 ** 2 + k2 ** 2 - pq ** 2) / (2 * k1 * k2), -1, 1)
    xg, wg = np.polynomial.legendre.leggauss(n_mu)
    psi = 2 * np.pi * (np.arange(n_psi) + 0.5) / n_psi
    M, C1, PS = np.meshgrid(mu, xg, psi, indexing='ij')
    W = np.meshgrid(wq, wg / 2, np.full(n_psi, 1 / n_psi), indexing='ij')
    W = W[0] * W[1] * W[2]
    S1 = np.sqrt(1 - C1 ** 2)
    SM = np.sqrt(np.clip(1 - M ** 2, 0, None))
    e1 = np.stack([S1, np.zeros_like(S1), C1])
    ea = np.stack([C1, np.zeros_like(C1), -S1])
    eb = np.stack([np.zeros_like(C1), np.ones_like(C1), np.zeros_like(C1)])
    e2 = M * e1 + SM * (np.cos(PS) * ea + np.sin(PS) * eb)
    kv1, kv2 = k1 * e1, k2 * e2
    pv = kv1 - kv2
    T = 2 * squeezed_response(kv1, pv, P, bias, f) ** 2 * P_long(np.sqrt(np.sum(pv * pv, 0)))
    if sigma_fog:
        T = T * np.exp(-(kv1[2] * sigma_fog) ** 2 - (kv2[2] * sigma_fog) ** 2)
    mu1, mu2 = C1, e2[2]
    L = {l: (np.polynomial.legendre.Legendre.basis(l)(mu1), np.polynomial.legendre.Legendre.basis(l)(mu2)) for l in ells}
    return {(l1, l2): (2 * l1 + 1) * (2 * l2 + 1) * float(np.sum(W * L[l1][0] * L[l2][1] * T)) for l1 in ells for l2 in ells}


def collapsed_covariance(t0, spectra, k_split, ells=None):
    """Response-approach collapsed term (collapsed_multipoles) of a TrispectrumCovariance t0, for all bin pairs with |k_i - k_j| < k_split,
    window factor J4, same layout as components(); k-bin average with the n_k nodes."""
    cov = t0.cov
    ells = cov.ells if ells is None else tuple(ells)
    A = str(spectra[0][0])
    J4 = t0.window_integral(A)
    kn, kw = t0._nodes()
    nb, nl = len(kn), len(ells)
    out = np.zeros((nl * nb, nl * nb))
    for i in range(nb):
        for j in range(i, nb):
            acc = {key: 0.0 for key in ((l1, l2) for l1 in ells for l2 in ells)}
            for wa, ka in zip(kw[i], kn[i]):
                for wb, kb in zip(kw[j], kn[j]):
                    if abs(ka - kb) >= k_split:
                        continue
                    m = collapsed_multipoles(ka, kb, t0.P, t0.bias, t0.f, k_split, ells=ells,
                                             n_mu=max(t0.n_mu, 16), n_psi=max(t0.n_psi, 32), sigma_fog=t0.sigma_fog)
                    for key, v in m.items():
                        acc[key] += wa * wb * v
            for a, l1 in enumerate(ells):
                for b, l2 in enumerate(ells):
                    out[a * nb + i, b * nb + j] = out[b * nb + j, a * nb + i] = J4 * acc[(l1, l2)]
    return out


def response_split(C_T, k, k_split, C_base, ells=(0, 2, 4), C_collapsed=None):
    """Split a trispectrum covariance into the response-approach pieces at the scale k_split.

    Bins with k < k_split are 'long' (L), the others 'hard' (H). Returns
      LL : both long (tree level is valid there),
      LH : long x hard, the squeezed couplings X (the hard power responds to the realised long-mode power),
      HH : both hard (tree level; dropped by the response approach),
      HH_collapsed : with C_collapsed (collapsed_covariance), its HH block,
      completion : the hard-hard variance implied by the squeezed couplings. Model the hard power as
           P_H = P_H0 + J (P_L - <P_L>), with (P_L, P_H0) distributed as the rest of the model
           [[A, B0], [B0^T, D0]] (A = (C_base + LL)_LL, B0 = (C_base)_LH): then Cov(P_L, P_H) = B0 + A J^T = B0 + X
           gives J = X^T A^-1 and the extra hard-hard covariance X^T A^-1 X + X^T A^-1 B0 + B0^T A^-1 X.
           It is O(P^4) (beyond tree level) and makes the total positive whenever the rest of the model is.
    C_base: the rest of the covariance model (Gaussian + SSC + discreteness), full matrix in (ell, bin) order.
    LL + LH + HH = C_T."""
    k = np.asarray(k, float)
    L = np.tile(k < k_split, len(ells))
    H = ~L
    LL, HH = np.outer(L, L), np.outer(H, H)
    out = {'LL': np.where(LL, C_T, 0.0), 'HH': np.where(HH, C_T, 0.0), 'LH': np.where(~(LL | HH), C_T, 0.0)}
    if C_collapsed is not None:
        out['HH_collapsed'] = np.where(HH, C_collapsed, 0.0)
    comp = np.zeros_like(C_T)
    if L.any() and H.any():
        C_base = np.asarray(C_base)
        A = (C_base + out['LL'])[np.ix_(L, L)]
        X, B0 = C_T[np.ix_(L, H)], C_base[np.ix_(L, H)]
        AiX = np.linalg.solve(A, X)
        cross = B0.T @ AiX
        comp[np.ix_(H, H)] = X.T @ AiX + cross + cross.T
    out['completion'] = comp
    return out


def response_covariance(parts, hard=False):
    """The response-based T0 covariance from response_split's parts: long-long (tree level) + squeezed couplings +
    their completion + the collapsed term (+ the tree-level hard-hard block if hard=True, for diagnostics; the
    response approach drops it)."""
    C = parts['LL'] + parts['LH'] + parts['completion'] + parts.get('HH_collapsed', 0.0)
    return C + parts['HH'] if (hard or 'HH_collapsed' not in parts) else C
