"""Gaussian mixture models fitted by EM (numpy), and the single-Gaussian baseline.

The M-step for each covariance is (S_k + reg * I) / N_k, where S_k is the responsibility-weighted
scatter matrix and N_k the effective count. For reg > 0 this is the exact maximizer of the
expected complete-data log-likelihood plus the log-prior penalty -reg/2 * tr(Sigma_k^{-1})
(an improper inverse-Wishart prior), so EM increases the penalized log-likelihood monotonically;
for reg = 0 it is plain maximum-likelihood EM.
"""

import numpy as np
from scipy.special import logsumexp


def component_log_densities(x, weights, means, covs):
    """log pi_k + log N(x_i; mu_k, Sigma_k), shape (n, K)."""
    d = x.shape[1]
    chol = np.linalg.cholesky(covs)                                   # (K, d, d)
    sol = np.linalg.inv(chol) @ (x[None] - means[:, None]).transpose(0, 2, 1)  # (K, d, n)
    log_det = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(1)
    log_n = -0.5 * (d * np.log(2.0 * np.pi) + log_det[:, None] + (sol**2).sum(1))
    return (np.log(weights)[:, None] + log_n).T


def log_density(x, params):
    return logsumexp(component_log_densities(x, *params), axis=1)


def fit_gaussian(x):
    """Maximum-likelihood Gaussian (covariance normalized by n), as a one-component mixture."""
    mean = x.mean(0)
    cov = (x - mean).T @ (x - mean) / len(x)
    return np.ones(1), mean[None], cov[None]


def _kmeanspp_centers(x, k, rng):
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    return np.asarray(centers)


def _m_step(x, resp, reg):
    nk = resp.sum(0)
    weights = nk / len(x)
    means = resp.T @ x / nk[:, None]
    diff = x[None] - means[:, None]                                   # (K, n, d)
    scatter = (resp.T[:, :, None] * diff).transpose(0, 2, 1) @ diff
    covs = (scatter + reg * np.eye(x.shape[1])) / nk[:, None, None]
    return weights, means, covs


def _penalty(covs, reg):
    return -0.5 * reg * np.trace(np.linalg.inv(covs), axis1=1, axis2=2).sum()


def em(x, n_components, rng, n_iter=1000, tol=1e-8, reg=1e-6):
    """One EM run from a k-means++ initialization (each point starts in the component of its
    nearest seed).

    Returns (params, objective history), where the objective is the average log-likelihood
    plus the penalty divided by n (equal to the average log-likelihood when reg = 0).
    """
    centers = _kmeanspp_centers(x, n_components, rng)
    nearest = np.argmin(((x[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    resp = np.eye(n_components)[nearest] + 1e-12
    resp /= resp.sum(1, keepdims=True)
    params = _m_step(x, resp, reg)
    history = []
    for it in range(n_iter):
        logp = component_log_densities(x, *params)
        log_norm = logsumexp(logp, axis=1, keepdims=True)
        history.append(log_norm.mean() + _penalty(params[2], reg) / len(x))
        if it == n_iter - 1 or (it > 0 and history[-1] - history[-2] < tol):
            break
        params = _m_step(x, np.exp(logp - log_norm), reg)
    return params, np.asarray(history)


def fit_gmm(x, n_components, rng, n_init=5, **kwargs):
    """Best of n_init EM runs by final (penalized) training objective."""
    best = None
    for _ in range(n_init):
        params, hist = em(x, n_components, rng, **kwargs)
        if best is None or hist[-1] > best[1][-1]:
            best = (params, hist)
    return best
