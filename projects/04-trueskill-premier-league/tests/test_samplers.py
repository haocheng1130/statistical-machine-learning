import numpy as np
import torch

from src.data import two_player_wins
from src.diagnostics import ess
from src.exact import grid_posterior
from src.model import make_log_joint, value_and_grad
from src.samplers import hmc, leapfrog, mala

MEAN = torch.tensor([1.0, -2.0], dtype=torch.float64)
COV = torch.tensor([[1.0, 0.8], [0.8, 1.0]], dtype=torch.float64)
GAUSS = torch.distributions.MultivariateNormal(MEAN, covariance_matrix=COV).log_prob
TOY = make_log_joint(two_player_wins(20, 0))


def _start(n_chains=4, dim=2, seed=0):
    return torch.randn(n_chains, dim, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _check_moments(draws, mean, cov, n_se=5.0, sd_rtol=0.07, corr_atol=0.05):
    x = draws.numpy()
    flat = x.reshape(-1, x.shape[-1])
    sd = np.sqrt(np.diag(cov))
    for k in range(x.shape[-1]):
        mcse = flat[:, k].std() / np.sqrt(ess(x[:, :, k]))
        assert abs(flat[:, k].mean() - mean[k]) < n_se * mcse
    assert np.allclose(flat.std(0), sd, rtol=sd_rtol)
    corr = np.corrcoef(flat.T)[0, 1]
    assert abs(corr - cov[0, 1] / (sd[0] * sd[1])) < corr_atol


def test_leapfrog_is_reversible():
    z0 = torch.tensor([[1.3, -0.4]], dtype=torch.float64)
    p0 = torch.tensor([[0.7, 1.1]], dtype=torch.float64)
    _, g0 = value_and_grad(TOY, z0)
    z1, p1, _, g1 = leapfrog(z0, p0, g0, 0.1, 25, TOY)
    z2, p2, _, _ = leapfrog(z1, -p1, g1, 0.1, 25, TOY)
    assert torch.allclose(z2, z0, atol=1e-10) and torch.allclose(p2, -p0, atol=1e-10)


def test_leapfrog_energy_error_is_small_and_second_order():
    z0 = torch.tensor([[1.3, -0.4]], dtype=torch.float64)
    p0 = torch.tensor([[0.7, 1.1]], dtype=torch.float64)
    logp0, g0 = value_and_grad(TOY, z0)
    h0 = -logp0 + 0.5 * (p0 ** 2).sum(-1)
    errors = []
    for eps in (0.02, 0.01):  # same trajectory length 1.0
        z1, p1, logp1, _ = leapfrog(z0, p0, g0, eps, round(1.0 / eps), TOY)
        errors.append(abs((-logp1 + 0.5 * (p1 ** 2).sum(-1) - h0).item()))
    assert errors[0] < 1e-3
    assert 3.0 < errors[0] / errors[1] < 5.0  # O(eps^2) global energy error


def test_mala_recovers_gaussian_mean_and_covariance():
    res = mala(GAUSS, _start(), n_warmup=1000, n_draws=6000, step_size=0.1, seed=1)
    assert abs(res.accept_rate.mean().item() - 0.574) < 0.05  # dual averaging hits the target
    _check_moments(res.draws, MEAN.numpy(), COV.numpy())


def test_hmc_recovers_gaussian_mean_and_covariance():
    res = hmc(GAUSS, _start(), n_warmup=500, n_draws=1500, step_size=0.1, n_leapfrog=10, seed=2,
              jitter=0.5)
    _check_moments(res.draws, MEAN.numpy(), COV.numpy())


def test_mala_and_hmc_match_exact_two_player_posterior():
    post = grid_posterior(TOY)
    mean, cov = post.mean.numpy(), post.cov.numpy()
    res_mala = mala(TOY, _start(seed=3), n_warmup=1000, n_draws=6000, step_size=0.1, seed=3)
    _check_moments(res_mala.draws, mean, cov)
    res_hmc = hmc(TOY, _start(seed=4), n_warmup=500, n_draws=1500, step_size=0.1, n_leapfrog=10,
                  seed=4, jitter=0.5)
    _check_moments(res_hmc.draws, mean, cov)


def test_omitting_the_proposal_factor_halves_the_variance():
    # Without the 1/(4 eta) factor the Hastings correction almost vanishes; the Langevin
    # proposal is then nearly reversible for p, and the extra p(z')/p(z) filter makes the
    # chain target approximately p^2, whose covariance is half of p's.
    target = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64),
                                                    covariance_matrix=torch.eye(2, dtype=torch.float64))
    z0 = _start(n_chains=8, seed=5)
    kwargs = dict(n_warmup=500, n_draws=15000, step_size=0.01, adapt=False, seed=5)
    wrong = mala(target.log_prob, z0, log_q_scale=1.0, **kwargs).draws.reshape(-1, 2)
    right = mala(target.log_prob, z0, **kwargs).draws.reshape(-1, 2)
    assert np.allclose(wrong.var(0).numpy(), 0.5, atol=0.07)
    assert np.allclose(right.var(0).numpy(), 1.0, atol=0.12)
