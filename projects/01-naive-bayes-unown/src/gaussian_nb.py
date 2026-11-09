"""Gaussian naive Bayes (diagonal-covariance Gaussian per class) with variance smoothing.

Model: c ~ Categorical(pi), x_j | c ~ N(mu[c, j], var[c, j]) independently.
The MLE of var is zero for pixels that are constant within a class (for example
background pixels that are always 0), so every variance is floored by adding a
fixed eps > 0 once, at evaluation time.
"""
import numpy as np

from .bernoulli_nb import CHUNK


def fit(X, y, n_classes):
    """MLE: per-class mean and (biased, 1/N_c) variance of every pixel, and class frequencies."""
    D = X.shape[1]
    mu = np.zeros((n_classes, D))
    var = np.zeros((n_classes, D))
    n_c = np.bincount(y, minlength=n_classes).astype(np.float64)
    for c in range(n_classes):
        Xc = np.asarray(X[y == c], dtype=np.float64)
        mu[c] = Xc.mean(axis=0)
        var[c] = ((Xc - mu[c]) ** 2).mean(axis=0)
    return mu, var, np.log(n_c / n_c.sum())


def log_likelihood(X, mu, var, eps):
    """log p(x | c) = sum_j log N(x_j; mu[c, j], var[c, j] + eps), shape (N, K)."""
    v = var + eps
    prec = 1.0 / v
    const = -0.5 * (X.shape[1] * np.log(2 * np.pi) + np.log(v).sum(axis=1) + (mu ** 2 * prec).sum(axis=1))
    out = np.empty((X.shape[0], mu.shape[0]))
    for s in range(0, X.shape[0], CHUNK):
        x = np.asarray(X[s:s + CHUNK], dtype=np.float64)
        # sum_j (x_j - mu_j)^2 / v_j expanded as x^2 . prec - 2 x . (mu prec) + mu^2 . prec
        out[s:s + CHUNK] = -0.5 * ((x ** 2) @ prec.T) + x @ (mu * prec).T + const
    return out


def log_joint(X, mu, var, log_pi, eps):
    """log p(x, c), shape (N, K)."""
    return log_likelihood(X, mu, var, eps) + log_pi
