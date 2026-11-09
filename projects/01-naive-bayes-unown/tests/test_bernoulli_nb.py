"""Bernoulli naive Bayes routines checked against brute force on tiny problems."""
import itertools

import numpy as np
import pytest

from src import bernoulli_nb as bnb


def tiny_data(rng, n=15, d=4, k=2):
    y = np.arange(n) % k
    X = (rng.random((n, d)) < 0.4).astype(np.uint8)
    X[y == 0, 0] = 0  # a pixel never on in class 0
    X[y == 1, 1] = 1  # a pixel always on in class 1
    return X, y


def random_model(rng, d, k):
    theta = rng.uniform(0.05, 0.95, size=(k, d))
    pi = rng.dirichlet(np.ones(k))
    return theta, np.log(pi)


@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0, 5.0])
def test_map_is_mode_of_beta_posterior_by_grid_search(alpha):
    X, y = tiny_data(np.random.default_rng(0))
    theta, log_pi = bnb.fit(X, y, 2, alpha)
    grid = np.linspace(1e-7, 1 - 1e-7, 2_000_001)
    for c in range(2):
        Xc = X[y == c]
        for j in range(X.shape[1]):
            n1 = Xc[:, j].sum()
            n0 = len(Xc) - n1
            # unnormalized log posterior: Bernoulli likelihood times Beta(alpha, alpha) prior
            log_post = (n1 + alpha - 1) * np.log(grid) + (n0 + alpha - 1) * np.log1p(-grid)
            assert abs(theta[c, j] - grid[np.argmax(log_post)]) < 1e-6
    np.testing.assert_allclose(np.exp(log_pi), np.bincount(y) / len(y))


def test_alpha_below_one_is_rejected():
    X, y = tiny_data(np.random.default_rng(0))
    with pytest.raises(ValueError):
        bnb.fit(X, y, 2, 0.5)


def test_posterior_rows_sum_to_one_and_match_direct_bayes():
    rng = np.random.default_rng(1)
    d, k = 6, 5
    theta, log_pi = random_model(rng, d, k)
    X = (rng.random((40, d)) < 0.5).astype(np.uint8)
    log_post = bnb.log_posterior(X, theta, log_pi)
    np.testing.assert_allclose(np.exp(log_post).sum(axis=1), 1.0, rtol=1e-12)
    # direct Bayes rule in probability space
    lik = np.prod(np.where(X[:, None, :] == 1, theta[None], 1 - theta[None]), axis=2)
    joint = np.exp(log_pi) * lik
    np.testing.assert_allclose(np.exp(log_post), joint / joint.sum(axis=1, keepdims=True), rtol=1e-10)


def test_posterior_is_stable_for_long_images():
    rng = np.random.default_rng(2)
    theta, log_pi = random_model(rng, 784, 28)
    X = (rng.random((10, 784)) < 0.5).astype(np.uint8)
    log_post = bnb.log_posterior(X, theta, log_pi)  # p(x | c) underflows to 0 in probability space
    assert np.all(np.isfinite(np.exp(log_post)))
    np.testing.assert_allclose(np.exp(log_post).sum(axis=1), 1.0, rtol=1e-12)


def test_zero_one_parameters_give_minus_inf_not_nan():
    theta = np.array([[0.0, 1.0, 0.3], [0.5, 0.5, 0.5]])
    X = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    ll = bnb.log_likelihood(X, theta)
    with np.errstate(divide="ignore"):
        direct = np.log(np.prod(np.where(X[:, None, :] == 1, theta[None], 1 - theta[None]), axis=2))
    assert not np.any(np.isnan(ll))
    np.testing.assert_allclose(ll, direct)
    # an image impossible under every class has an undefined posterior and no prediction
    log_joint = np.array([[-np.inf, -np.inf], [-1.0, -np.inf]])
    np.testing.assert_array_equal(bnb.predict(log_joint), [-1, 0])


def enumerate_joint(theta, log_pi):
    """p(x, c) for every x in {0,1}^D: arrays (2^D, D) and (2^D, K)."""
    d = theta.shape[1]
    xs = np.array(list(itertools.product([0, 1], repeat=d)))
    p = np.exp(log_pi) * np.prod(np.where(xs[:, None, :] == 1, theta[None], 1 - theta[None]), axis=2)
    return xs, p


def test_imputation_matches_enumeration():
    rng = np.random.default_rng(3)
    d, k = 4, 3
    theta, log_pi = random_model(rng, d, k)
    xs, p_joint = enumerate_joint(theta, log_pi)
    p_x = p_joint.sum(axis=1)
    for x, observed in itertools.product(xs, itertools.product([0, 1], repeat=d)):
        mask = np.array(observed)
        consistent = np.all((xs == x) | (mask == 0), axis=1)  # completions agreeing with x on E
        p_e = p_x[consistent].sum()
        p_on = (p_x[consistent][:, None] * xs[consistent]).sum(axis=0) / p_e
        got = bnb.predictive_on_prob(x[None], theta, log_pi, mask)[0]
        hidden = mask == 0
        np.testing.assert_allclose(got[hidden], p_on[hidden], rtol=1e-12)
        # joint predictive of the hidden pixels: p(x_H | x_E) = p(x) / p(x_E)
        p_this = p_x[np.all(xs == x, axis=1)][0]
        got_ll = bnb.hidden_log_likelihood(x[None], theta, log_pi, mask)[0]
        np.testing.assert_allclose(got_ll, np.log(p_this / p_e), rtol=1e-12, atol=1e-14)
        filled = bnb.impute(x[None], theta, log_pi, mask)[0]
        np.testing.assert_allclose(filled, np.where(hidden, p_on, x))


def test_per_image_and_shared_masks_agree():
    rng = np.random.default_rng(4)
    theta, log_pi = random_model(rng, 10, 3)
    X = (rng.random((7, 10)) < 0.5).astype(np.uint8)
    shared = (rng.random(10) < 0.5).astype(np.uint8)
    a = bnb.predictive_on_prob(X, theta, log_pi, shared)
    b = bnb.predictive_on_prob(X, theta, log_pi, np.tile(shared, (7, 1)))
    np.testing.assert_allclose(a, b)
    # with nothing observed the prediction is the prior predictive sum_c pi_c theta_c
    none = bnb.predictive_on_prob(X, theta, log_pi, np.zeros(10))
    np.testing.assert_allclose(none, np.tile(np.exp(log_pi) @ theta, (7, 1)))


def test_sampler_matches_model_frequencies():
    rng = np.random.default_rng(5)
    theta, log_pi = random_model(rng, 5, 3)
    n = 200_000
    x, c = bnb.sample(theta, log_pi, n, np.random.default_rng(6))
    pi = np.exp(log_pi)
    freq = np.bincount(c, minlength=3) / n
    assert np.all(np.abs(freq - pi) < 5 * np.sqrt(pi * (1 - pi) / n))
    for k in range(3):
        nk = (c == k).sum()
        se = np.sqrt(theta[k] * (1 - theta[k]) / nk)
        assert np.all(np.abs(x[c == k].mean(axis=0) - theta[k]) < 5 * se)
