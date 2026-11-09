"""EM for Gaussian mixtures: density evaluation, monotonicity and parameter recovery."""

import numpy as np
from scipy.stats import multivariate_normal

from src import data, gmm

MEANS = np.array([[-4.0, 0.0], [3.0, 3.0], [3.0, -3.0]])
COVS = np.array([[[1.0, 0.5], [0.5, 0.8]], [[0.3, 0.0], [0.0, 1.5]], [[0.6, -0.2], [-0.2, 0.4]]])
WEIGHTS = np.array([0.5, 0.3, 0.2])


def sample_mixture(n, rng):
    comp = rng.choice(3, size=n, p=WEIGHTS)
    return np.stack([rng.multivariate_normal(MEANS[c], COVS[c]) for c in comp])


def test_log_density_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 2)) * 3
    ref = np.log(sum(w * multivariate_normal(m, c).pdf(x) for w, m, c in zip(WEIGHTS, MEANS, COVS)))
    np.testing.assert_allclose(gmm.log_density(x, (WEIGHTS, MEANS, COVS)), ref, rtol=1e-10)


def test_single_gaussian_is_mle_and_equals_em_with_one_component():
    rng = np.random.default_rng(1)
    x = sample_mixture(500, rng)
    w, m, c = gmm.fit_gaussian(x)
    np.testing.assert_allclose(m[0], x.mean(0))
    np.testing.assert_allclose(c[0], np.cov(x.T, bias=True))
    (w1, m1, c1), _ = gmm.em(x, 1, rng, reg=0.0)
    np.testing.assert_allclose(m1, m)
    np.testing.assert_allclose(c1, c)


def test_em_log_likelihood_is_monotone():
    rng = np.random.default_rng(2)
    x, _, _ = data.sample(150, 0.05, rng)
    for k in (2, 5, 12):
        _, hist = gmm.em(x, k, rng, n_iter=300, tol=-np.inf, reg=0.0)
        assert len(hist) == 300
        assert np.all(np.diff(hist) >= -1e-10)
        assert hist[-1] > hist[0]


def test_em_penalized_objective_is_monotone_and_matches_definition():
    rng = np.random.default_rng(3)
    x, _, _ = data.sample(150, 0.02, rng)
    reg = 1e-3
    params, hist = gmm.em(x, 8, rng, n_iter=200, tol=-np.inf, reg=reg)
    assert np.all(np.diff(hist) >= -1e-10)
    penalty = -0.5 * reg * sum(np.trace(np.linalg.inv(c)) for c in params[2])
    np.testing.assert_allclose(hist[-1], gmm.log_density(x, params).mean() + penalty / len(x), rtol=1e-12)


def test_em_recovers_well_separated_clusters():
    rng = np.random.default_rng(4)
    x = sample_mixture(6000, rng)
    (w, m, c), _ = gmm.fit_gmm(x, 3, rng, n_init=3, reg=0.0)
    order = [int(np.argmin(((m - mu) ** 2).sum(1))) for mu in MEANS]
    assert sorted(order) == [0, 1, 2]
    np.testing.assert_allclose(w[order], WEIGHTS, atol=0.02)
    np.testing.assert_allclose(m[order], MEANS, atol=0.06)
    np.testing.assert_allclose(c[order], COVS, atol=0.08)
