"""The statistics used by the figures of paper_nersc.ipynb. Every function takes mock vectors (n_mocks x n) and/or
covariance matrices (n x n) and returns plain arrays."""
from __future__ import annotations

import numpy as np


def correlation(C):
    """correlation matrix C_ij / sqrt(C_ii C_jj)"""
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def chi2_of_each_mock(mocks, C):
    """chi^2 of every mock about the mean of the mocks, rescaled so that its expectation is n (not n (N-1)/N)"""
    L = np.linalg.cholesky(C)
    x = np.linalg.solve(L, (mocks - mocks.mean(0)).T)
    N = len(mocks)
    return (x ** 2).sum(0) * N / (N - 1)


def mean_chi2_per_dof(mocks, C):
    """<chi^2>/n over the mocks: 1 if C is right"""
    return chi2_of_each_mock(mocks, C).mean() / mocks.shape[1]


def inverse_sqrt(C):
    """C^{-1/2} (symmetric square root)"""
    w, U = np.linalg.eigh(C)
    return (U / np.sqrt(w)) @ U.T


def whitened_mock_covariance(mocks, C):
    """C^{-1/2} S C^{-1/2} with S the mocks' covariance: the identity matrix if C is right"""
    W = inverse_sqrt(C)
    return W @ np.cov(mocks.T) @ W


def whitened_trace(mocks, C):
    """tr(C^-1 S)/n: the average ratio of mock to model variance over all directions"""
    return np.trace(np.linalg.solve(C, np.cov(mocks.T))) / len(C)


def null_eigenvalues(n_mocks, n, n_draws=200, seed=1):
    """sorted (largest first) eigenvalues of the sample covariance of n_mocks unit Gaussian vectors, n_draws times:
    what whitened_mock_covariance gives when the model is exactly right"""
    rng = np.random.default_rng(seed)
    return np.array([np.sort(np.linalg.eigvalsh(np.cov(rng.standard_normal((n_mocks, n)).T)))[::-1]
                     for _ in range(n_draws)])


def wishart_sigma(C, n_mocks):
    """standard deviation of each element of the sample covariance of n_mocks Gaussian vectors with covariance C"""
    d = np.diag(C)
    return np.sqrt((np.outer(d, d) + C ** 2) / (n_mocks - 1))


def variance_ratio_error(C, n_mocks, idx):
    """expected scatter of mean(S_ii / C_ii) over the bins idx (accounts for correlated neighbouring bins)"""
    r2 = correlation(C)[np.ix_(idx, idx)] ** 2
    return np.sqrt(2 * r2.sum() / (n_mocks - 1)) / len(idx)


def excess_directions(mocks_train, C):
    """directions (in whitened space) along which mocks_train vary most relative to C, largest first.
    Returns (variance along each direction in the training mocks, directions as columns, C^{-1/2})."""
    W = inverse_sqrt(C)
    lam, U = np.linalg.eigh(W @ np.cov(mocks_train.T) @ W)
    order = np.argsort(-lam)
    return lam[order], U[:, order], W


def project(mocks, W, directions):
    """coordinates of the mocks (about their mean) along whitened directions: unit variance if the model is right"""
    return ((mocks - mocks.mean(0)) @ W) @ directions


def fit_amplitudes(mocks, C_gauss, templates):
    """maximum-likelihood amplitudes a_i of C = C_gauss + sum_i a_i T_i for the mocks (Wishart likelihood), with
    Fisher errors. templates: {name: matrix}. Returns ({name: amplitude}, {name: error}). a_i = 1 is the prediction."""
    from scipy.optimize import minimize
    S, N, T = np.cov(mocks.T), len(mocks), list(templates.values())

    def minus2lnL(a):
        try:
            L = np.linalg.cholesky(C_gauss + sum(ai * Ti for ai, Ti in zip(a, T)))
        except np.linalg.LinAlgError:
            return 1e300
        X = np.linalg.solve(L, S)
        return (N - 1) * (np.trace(np.linalg.solve(L, X.T)) + 2 * np.log(np.diag(L)).sum())
    start = np.ones(len(T)) if minus2lnL(np.ones(len(T))) < 1e299 else 0.1 * np.ones(len(T))
    a = minimize(minus2lnL, start, method='Powell', options=dict(xtol=1e-4, ftol=1e-6, maxiter=20000)).x
    Ci = np.linalg.inv(C_gauss + sum(ai * Ti for ai, Ti in zip(a, T)))
    A = [Ci @ Ti for Ti in T]
    F = np.array([[0.5 * (N - 1) * np.trace(x @ y) for y in A] for x in A])
    err = np.sqrt(np.diag(np.linalg.inv(F)))
    return dict(zip(templates, a)), dict(zip(templates, err))


def parameter_variance_ratio(sample, C, kmax):
    """fit each mock with C for: A (overall amplitude), A2 (extra quadrupole amplitude), alpha (dilation) and a
    monopole constant; return the variance of each over the mocks divided by the variance C predicts (4 numbers)"""
    k, mean = sample.k, sample.mocks.mean(0)
    dlnk = lambda P: -np.gradient(P, np.log(k))                  # noqa: E731
    P0, P2, P4 = sample.P0(mean), sample.P2(mean), sample.P4(mean)
    zero, one = np.zeros_like(k), np.ones_like(k)
    T = np.stack([np.r_[P0, P2, P4], np.r_[zero, P2, zero], np.r_[dlnk(P0), dlnk(P2), dlnk(P4)], np.r_[one, zero, zero]], 1)
    i = sample.below(kmax)
    T, Ci = T[i], np.linalg.inv(C[np.ix_(i, i)])
    Finv = np.linalg.inv(T.T @ Ci @ T)
    estimates = (Finv @ T.T @ Ci @ (sample.mocks[:, i] - mean[i]).T).T
    return estimates.var(0, ddof=1) / np.diag(Finv)
