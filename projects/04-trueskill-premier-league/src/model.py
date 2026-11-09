"""TrueSkill-style paired-comparison model.

Skills z in R^N with independent prior z_i ~ N(m, 1) and
    p(i beats j | z) = sigmoid(z_i - z_j),
games independent given z. Games are summarized by the win matrix W, where
W[i, j] counts the wins of i over j, so that
    log p(games | z) = sum_{i,j} W[i, j] * log sigmoid(z_i - z_j).
All functions accept a batch of skill vectors z of shape (..., N).
"""
import math

import torch
import torch.nn.functional as F


def log_prior(z: torch.Tensor, prior_mean: float = 0.0) -> torch.Tensor:
    """Log density of independent N(prior_mean, 1) skills, shape (...,)."""
    n = z.shape[-1]
    return -0.5 * ((z - prior_mean) ** 2).sum(-1) - 0.5 * n * math.log(2 * math.pi)


def log_likelihood(z: torch.Tensor, wins: torch.Tensor) -> torch.Tensor:
    """Sum over games of log sigmoid(z_winner - z_loser), shape (...,)."""
    diff = z[..., :, None] - z[..., None, :]  # diff[..., i, j] = z_i - z_j
    # log sigmoid(x) = -softplus(-x), numerically stable for large |x|.
    return -(wins * F.softplus(-diff)).sum((-1, -2))


def log_joint(z: torch.Tensor, wins: torch.Tensor, prior_mean: float = 0.0) -> torch.Tensor:
    """Unnormalized log posterior log p(z, games), shape (...,)."""
    return log_prior(z, prior_mean) + log_likelihood(z, wins)


def make_log_joint(wins: torch.Tensor, prior_mean: float = 0.0):
    """Return z -> log p(z, games) for a fixed data set."""
    return lambda z: log_joint(z, wins, prior_mean)


def value_and_grad(log_density, z: torch.Tensor):
    """Evaluate log_density at a batch z of shape (C, N) and its gradient by autograd.

    Rows of z are independent (e.g. parallel chains), so the gradient of the
    summed values gives every row's own gradient.
    """
    z = z.detach().requires_grad_(True)
    value = log_density(z)
    (grad,) = torch.autograd.grad(value.sum(), z)
    return value.detach(), grad.detach()
