import math

import numpy as np
import torch
from scipy.stats import norm

from src.data import win_matrix
from src.model import log_joint, log_likelihood, log_prior, make_log_joint, value_and_grad


def _random_games(n_players=6, n_games=40, seed=0):
    rng = np.random.default_rng(seed)
    winners = rng.integers(0, n_players, n_games)
    losers = (winners + rng.integers(1, n_players, n_games)) % n_players
    return winners, losers


def test_log_likelihood_matches_loop_over_games():
    winners, losers = _random_games()
    wins = win_matrix(winners, losers, 6)
    z = torch.randn(5, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 2
    expected = []
    for row in z.numpy():
        total = 0.0
        for w, l in zip(winners, losers):
            total += math.log(1.0 / (1.0 + math.exp(-(row[w] - row[l]))))
        expected.append(total)
    assert np.allclose(log_likelihood(z, wins).numpy(), expected, atol=1e-10)


def test_log_prior_and_joint_match_scipy():
    winners, losers = _random_games()
    wins = win_matrix(winners, losers, 6)
    z = torch.randn(3, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    for m in (0.0, 5.0):
        prior = norm.logpdf(z.numpy(), loc=m).sum(1)
        assert np.allclose(log_prior(z, m).numpy(), prior, atol=1e-10)
        assert np.allclose(log_joint(z, wins, m).numpy(), prior + log_likelihood(z, wins).numpy())


def test_likelihood_is_stable_for_extreme_skills():
    wins = win_matrix([0], [1], 2)
    z = torch.tensor([[-400.0, 400.0]], dtype=torch.float64)
    assert torch.isfinite(log_likelihood(z, wins)).all()
    assert abs(log_likelihood(z, wins).item() + 800.0) < 1e-9


def test_autograd_gradient_matches_finite_differences():
    winners, losers = _random_games(n_players=5, n_games=30, seed=3)
    f = make_log_joint(win_matrix(winners, losers, 5), prior_mean=0.7)
    z = torch.randn(2, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    _, grad = value_and_grad(f, z)
    h = 1e-6
    fd = torch.zeros_like(z)
    for k in range(5):
        e = torch.zeros(5, dtype=torch.float64)
        e[k] = h
        fd[:, k] = (f(z + e) - f(z - e)) / (2 * h)
    assert torch.allclose(grad, fd, atol=1e-6)


def test_prior_shift_translates_the_joint():
    # log p(z + m 1, games; prior mean m) = log p(z, games; prior mean 0) for every z,
    # so the posterior under prior mean m is the original posterior shifted by m.
    winners, losers = _random_games(seed=5)
    wins = win_matrix(winners, losers, 6)
    z = torch.randn(4, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    assert torch.allclose(log_joint(z + 5.0, wins, 5.0), log_joint(z, wins, 0.0), atol=1e-10)


def test_average_skill_has_exact_posterior_n_0_1_over_n():
    # The likelihood is invariant to z -> z + c 1, so log p(z, games) = -N zbar^2 / 2 + h(z - zbar 1):
    # the average skill zbar is independent of all differences, with posterior N(0, 1/N).
    winners, losers = _random_games(seed=7)
    f = make_log_joint(win_matrix(winners, losers, 6))
    z = torch.randn(4, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    zbar = z.mean(-1)
    for c in (-1.3, 0.4, 2.0):
        expected = -6 * (zbar + c) ** 2 / 2 + 6 * zbar ** 2 / 2
        assert torch.allclose(f(z + c) - f(z), expected, atol=1e-9)
