import math

import torch

from src.vi import (FullRankGaussian, MeanFieldGaussian, elbo_estimate, fit_vi, log_evidence_is,
                    prob_greater)

MEAN = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
COV = torch.tensor([[1.0, 0.7, 0.2], [0.7, 1.5, -0.3], [0.2, -0.3, 0.6]], dtype=torch.float64)
TARGET = torch.distributions.MultivariateNormal(MEAN, covariance_matrix=COV)


def _set_full_rank(q, mean, cov):
    scale = torch.linalg.cholesky(cov)
    with torch.no_grad():
        q.mean.copy_(mean)
        q.raw.copy_(torch.tril(scale, -1) + torch.diag(scale.diagonal().log()))


def test_entropy_and_sampler_match_the_gaussian_family():
    gen = torch.Generator().manual_seed(0)
    mf = MeanFieldGaussian(3)
    with torch.no_grad():
        mf.mean.copy_(MEAN)
        mf.log_sd.copy_(torch.tensor([0.1, -0.5, 0.3]))
    fc = FullRankGaussian(3)
    _set_full_rank(fc, MEAN, COV)
    assert torch.allclose(fc.covariance(), COV, atol=1e-12)
    for q in (mf, fc):
        ref = torch.distributions.MultivariateNormal(MEAN, covariance_matrix=q.covariance())
        assert abs(q.entropy().item() - ref.entropy().item()) < 1e-10
        x = q.sample(200000, gen).detach()
        assert torch.allclose(x.mean(0), MEAN, atol=0.01)
        assert torch.allclose(torch.cov(x.T), q.covariance(), atol=0.02)


def test_elbo_estimate_matches_closed_form_for_gaussian_target():
    # For a target with known normalizer Z, ELBO(q) = log Z - KL(q || p) in closed form.
    log_z = 3.0
    log_joint = lambda z: TARGET.log_prob(z) + log_z
    fc = FullRankGaussian(3)
    q_mean = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    q_cov = torch.tensor([[0.5, 0.1, 0.0], [0.1, 0.8, 0.0], [0.0, 0.0, 0.4]], dtype=torch.float64)
    _set_full_rank(fc, q_mean, q_cov)
    q_dist = torch.distributions.MultivariateNormal(q_mean, covariance_matrix=q_cov)
    exact = log_z - torch.distributions.kl_divergence(q_dist, TARGET).item()
    est, se = elbo_estimate(fc, log_joint, 200000, torch.Generator().manual_seed(1))
    assert abs(est - exact) < 4 * se + 1e-3


def test_svi_recovers_known_optima_for_gaussian_target():
    # Full-rank: q* = p. Mean-field: q* has the true mean and variances 1 / Lambda_ii,
    # where Lambda is the target precision (the mean-field fit of a Gaussian).
    fc = FullRankGaussian(3)
    fit_vi(fc, TARGET.log_prob, n_iters=4000, lr=0.05, n_samples=16, seed=2, final_lr_fraction=0.001)
    assert torch.allclose(fc.mean.detach(), MEAN, atol=0.02)
    assert torch.allclose(fc.covariance(), COV, atol=0.03)
    mf = MeanFieldGaussian(3)
    fit_vi(mf, TARGET.log_prob, n_iters=4000, lr=0.05, n_samples=16, seed=3, final_lr_fraction=0.001)
    precision = torch.linalg.inv(COV)
    assert torch.allclose(mf.mean.detach(), MEAN, atol=0.02)
    assert torch.allclose(mf.covariance().diagonal(), 1 / precision.diagonal(), rtol=0.03)


def test_prob_greater_matches_monte_carlo():
    gen = torch.Generator().manual_seed(5)
    x = MEAN + torch.randn(400000, 3, generator=gen, dtype=torch.float64) @ torch.linalg.cholesky(COV).T
    for i, j in [(0, 1), (2, 0), (1, 2)]:
        mc = (x[:, i] > x[:, j]).double().mean().item()
        assert abs(prob_greater(MEAN, COV, i, j).item() - mc) < 0.004
    diag = torch.diag(COV.diagonal())  # factorized case
    y = MEAN + torch.randn(400000, 3, generator=gen, dtype=torch.float64) * COV.diagonal().sqrt()
    mc = (y[:, 0] > y[:, 1]).double().mean().item()
    exact = 0.5 * (1 + math.erf((MEAN[0] - MEAN[1]).item() / math.sqrt(2 * (COV[0, 0] + COV[1, 1]).item())))
    assert abs(prob_greater(MEAN, diag, 0, 1).item() - exact) < 1e-12
    assert abs(exact - mc) < 0.004


def test_importance_sampling_recovers_known_evidence():
    log_joint = lambda z: TARGET.log_prob(z) - 7.5
    log_z, kish = log_evidence_is(log_joint, MEAN + 0.1, COV * 1.3, 100000, seed=4)
    assert abs(log_z + 7.5) < 0.01 and kish > 50000
