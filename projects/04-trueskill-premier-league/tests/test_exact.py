import math

import numpy as np
import torch
from scipy import integrate
from scipy.special import expit, log_expit
from scipy.stats import norm

from src.data import two_player_wins
from src.exact import grid_posterior, kl_gaussian_to_grid
from src.model import make_log_joint

MEAN = torch.tensor([0.5, -0.3], dtype=torch.float64)
COV = torch.tensor([[1.0, 0.6], [0.6, 0.8]], dtype=torch.float64)


def _gaussian(log_const):
    dist = torch.distributions.MultivariateNormal(MEAN, covariance_matrix=COV)
    return lambda z: dist.log_prob(z) + log_const


def test_grid_recovers_gaussian_moments_and_normalizer():
    post = grid_posterior(_gaussian(log_const=2.5), half_width=8.0, n=601)
    assert abs(post.log_z - 2.5) < 1e-8
    assert torch.allclose(post.mean, MEAN, atol=1e-8)
    assert torch.allclose(post.cov, COV, atol=1e-8)
    d_sd = math.sqrt(COV[0, 0] + COV[1, 1] - 2 * COV[0, 1])
    exact = norm.cdf((MEAN[0] - MEAN[1]).item() / d_sd)
    assert abs(post.prob_a_greater - exact) < 2e-3  # diagonal cells limit the accuracy


def test_one_game_evidence_and_win_probability():
    # By symmetry the prior probability that A beats B is 1/2, so log Z = log(1/2).
    post = grid_posterior(make_log_joint(two_player_wins(1, 0)))
    assert abs(post.log_z - math.log(0.5)) < 1e-8
    # d = z_A - z_B ~ N(0, 2) a priori; P(d > 0 | A won) by 1-D quadrature.
    dens = lambda d: norm.pdf(d, scale=math.sqrt(2)) * expit(d)
    num, _ = integrate.quad(dens, 0, np.inf)
    den, _ = integrate.quad(dens, -np.inf, np.inf)
    assert abs(post.prob_a_greater - num / den) < 2e-3


def test_kl_on_grid_matches_closed_form_gaussian_kl():
    post = grid_posterior(_gaussian(log_const=0.0), half_width=8.0, n=601)
    q_mean = torch.tensor([0.2, 0.1], dtype=torch.float64)
    q_cov = torch.tensor([[0.5, 0.0], [0.0, 0.3]], dtype=torch.float64)
    q = torch.distributions.MultivariateNormal(q_mean, covariance_matrix=q_cov)
    p = torch.distributions.MultivariateNormal(MEAN, covariance_matrix=COV)
    exact = torch.distributions.kl_divergence(q, p).item()
    assert abs(kl_gaussian_to_grid(post, q_mean, q_cov) - exact) < 1e-6
    assert abs(kl_gaussian_to_grid(post, MEAN, COV)) < 1e-6


def test_tail_probability_after_twenty_wins_matches_quadrature():
    # The posterior of d = z_A - z_B is proportional to N(d; 0, 2) sigmoid(d)^20, a 1-D integral.
    post = grid_posterior(make_log_joint(two_player_wins(20, 0)))
    log_dens = lambda d: norm.logpdf(d, scale=math.sqrt(2)) + 20 * log_expit(d)
    dens = lambda d: np.exp(log_dens(d))
    lower, _ = integrate.quad(dens, -np.inf, 0, epsabs=1e-16, epsrel=1e-10)
    total, _ = integrate.quad(dens, -np.inf, np.inf, epsrel=1e-12)
    exact = lower / total  # about 1.07e-6
    assert abs((1 - post.prob_a_greater) / exact - 1) < 0.02
