"""Gaussian naive Bayes checked against direct per-class statistics and scipy densities."""
import numpy as np
from scipy.stats import norm

from src import gaussian_nb as gnb


def test_mle_equals_direct_per_class_mean_and_variance():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=300)
    X = rng.random((300, 9)).astype(np.float32)
    X[y == 2, 3] = 0.0  # a constant pixel: zero variance in class 2
    mu, var, log_pi = gnb.fit(X, y, 4)
    for c in range(4):
        rows = [X[i].astype(np.float64) for i in range(len(y)) if y[i] == c]
        np.testing.assert_allclose(mu[c], np.mean(rows, axis=0), rtol=1e-12)
        np.testing.assert_allclose(var[c], np.var(rows, axis=0, ddof=0), rtol=1e-10, atol=1e-15)
    assert var[2, 3] == 0.0
    np.testing.assert_allclose(np.exp(log_pi), np.bincount(y, minlength=4) / len(y))


def test_log_likelihood_matches_scipy():
    rng = np.random.default_rng(1)
    mu = rng.random((3, 7))
    var = rng.random((3, 7)) * 0.1
    var[0, 0] = 0.0
    X = rng.random((20, 7))
    for eps in (1e-6, 1e-2):
        got = gnb.log_likelihood(X, mu, var, eps)
        want = norm.logpdf(X[:, None, :], mu[None], np.sqrt(var + eps)[None]).sum(axis=2)
        np.testing.assert_allclose(got, want, rtol=1e-9)
