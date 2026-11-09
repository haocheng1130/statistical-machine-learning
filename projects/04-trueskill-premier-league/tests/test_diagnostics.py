import numpy as np

from src.diagnostics import ess, split_rhat


def _ar1(rho, n, m, seed):
    rng = np.random.default_rng(seed)
    x = np.empty((n, m))
    x[0] = rng.standard_normal(m) / np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.standard_normal(m)
    return x


def test_ess_of_independent_draws_is_close_to_sample_size():
    x = np.random.default_rng(0).standard_normal((2000, 4))
    assert abs(ess(x) / 8000 - 1) < 0.1


def test_ess_of_ar1_matches_integrated_autocorrelation_time():
    rho = 0.8
    x = _ar1(rho, 20000, 4, seed=1)
    expected = x.size * (1 - rho) / (1 + rho)  # tau = (1 + rho) / (1 - rho)
    assert abs(ess(x) / expected - 1) < 0.15


def test_rhat_near_one_for_mixed_chains_and_large_for_separated_chains():
    x = np.random.default_rng(2).standard_normal((1000, 4))
    assert split_rhat(x) < 1.01
    shifted = x + np.array([0.0, 0.0, 1.5, 1.5])
    assert split_rhat(shifted) > 1.2
    trend = x + np.linspace(0, 3, 1000)[:, None]  # non-stationary within each chain
    assert split_rhat(trend) > 1.2
