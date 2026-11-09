"""Bernoulli naive Bayes with independent Beta(alpha, alpha) priors on the pixel probabilities.

Model: c ~ Categorical(pi), and x_j | c ~ Bernoulli(theta[c, j]) independently for j = 1..D.
Parameters are stored class-major: theta has shape (K, D) and log_pi has shape (K,).

Everything is computed in the log domain. The convention 0 * log 0 = 0 is applied
explicitly, so the maximum likelihood estimate (alpha = 1), which contains exact
zeros and ones, yields -inf (an impossible pixel value) instead of NaN.

A pixel mask (1 = observed, 0 = hidden) can be given as an (N, D) array or as a
single (D,) array shared by all images. Hidden pixels are marginalized out, which
for naive Bayes simply drops their factors from p(x | c).
"""
import numpy as np
from scipy.special import logsumexp

CHUNK = 8192  # rows processed at a time, to bound memory on the 60000-image training set


def fit(X, y, n_classes, alpha=1.0):
    """MAP estimate of theta under iid Beta(alpha, alpha) priors, and the MLE of pi.

    theta[c, j] = (N_cj + alpha - 1) / (N_c + 2 (alpha - 1)), the mode of the
    Beta(N_cj + alpha, N_c - N_cj + alpha) posterior. alpha = 1 (flat prior) is the MLE.
    The formula is the posterior mode only for alpha >= 1, so smaller values are rejected.
    """
    if alpha < 1:
        raise ValueError("the closed-form posterior mode requires alpha >= 1")
    n_c = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.zeros((n_classes, X.shape[1]))
    for c in range(n_classes):
        counts[c] = X[y == c].sum(axis=0, dtype=np.int64)
    theta = (counts + alpha - 1.0) / (n_c[:, None] + 2.0 * (alpha - 1.0))
    log_pi = np.log(n_c / n_c.sum())
    return theta, log_pi


def _mask_rows(mask, start, stop, shape):
    if mask is None:
        return np.ones(shape)
    mask = np.asarray(mask, dtype=np.float64)
    return np.broadcast_to(mask if mask.ndim == 1 else mask[start:stop], shape)


def log_likelihood(X, theta, mask=None):
    """log p(x_E | c) for every image and class, shape (N, K); E = observed pixels."""
    log_on = np.log(theta, out=np.zeros_like(theta), where=theta > 0)
    log_off = np.log1p(-theta, out=np.zeros_like(theta), where=theta < 1)
    is_zero = (theta == 0).astype(np.float64)
    is_one = (theta == 1).astype(np.float64)
    N = X.shape[0]
    out = np.empty((N, theta.shape[0]))
    for s in range(0, N, CHUNK):
        x = np.asarray(X[s:s + CHUNK], dtype=np.float64)
        m = _mask_rows(mask, s, s + CHUNK, x.shape)
        on = x * m          # observed and on
        off = m - on        # observed and off
        ll = on @ log_on.T + off @ log_off.T
        n_impossible = on @ is_zero.T + off @ is_one.T
        ll[n_impossible > 0] = -np.inf
        out[s:s + CHUNK] = ll
    return out


def log_joint(X, theta, log_pi, mask=None):
    """log p(x_E, c), shape (N, K)."""
    return log_likelihood(X, theta, mask) + log_pi


def normalize(log_joint_values):
    """Turn log p(x, c) into log p(c | x) row by row (NaN rows where p(x) = 0)."""
    with np.errstate(invalid="ignore"):
        return log_joint_values - logsumexp(log_joint_values, axis=1, keepdims=True)


def log_posterior(X, theta, log_pi, mask=None):
    """log p(c | x_E), shape (N, K)."""
    return normalize(log_joint(X, theta, log_pi, mask))


def log_marginal(X, theta, log_pi, mask=None):
    """log p(x_E) = log sum_c p(c) p(x_E | c), shape (N,)."""
    return logsumexp(log_joint(X, theta, log_pi, mask), axis=1)


def predict(log_joint_values):
    """Most probable class; -1 where every class has probability zero."""
    pred = np.argmax(log_joint_values, axis=1)
    pred[np.all(np.isneginf(log_joint_values), axis=1)] = -1
    return pred


def predictive_on_prob(X, theta, log_pi, mask):
    """p(x_j = 1 | x_E) = sum_c p(c | x_E) theta[c, j] for every pixel j, shape (N, D)."""
    return np.exp(log_posterior(X, theta, log_pi, mask)) @ theta


def impute(X, theta, log_pi, mask):
    """Keep observed pixels and replace hidden ones by p(x_j = 1 | x_E)."""
    p = predictive_on_prob(X, theta, log_pi, mask)
    return np.where(np.broadcast_to(mask, p.shape) > 0, X, p)


def hidden_log_likelihood(X, theta, log_pi, mask):
    """log p(x_H | x_E) for the hidden pixels H jointly, shape (N,).

    p(x_H | x_E) = sum_c p(c | x_E) prod_{j in H} p(x_j | c), the exact predictive
    probability of the true hidden pixels (the pixels are dependent given x_E
    through the unknown class, so this is not a product of per-pixel marginals).
    """
    hidden = 1.0 - np.asarray(mask, dtype=np.float64)
    return logsumexp(log_posterior(X, theta, log_pi, mask) + log_likelihood(X, theta, hidden), axis=1)


def sample(theta, log_pi, n, rng, classes=None):
    """Ancestral sampling: c ~ pi (unless classes are given), then x_j ~ Bernoulli(theta[c, j])."""
    if classes is None:
        classes = rng.choice(len(log_pi), size=n, p=np.exp(log_pi))
    x = (rng.random((len(classes), theta.shape[1])) < theta[classes]).astype(np.uint8)
    return x, classes
