"""Exact (grid) posterior summaries for two players.

With two skills the posterior can be normalized by a Riemann sum on a fine
grid; for a smooth density whose mass lies well inside the grid this is accurate
to many digits and serves as ground truth for the approximate methods.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class GridPosterior:
    za: torch.Tensor  # (G, G) grid of z_A (first index)
    zb: torch.Tensor  # (G, G) grid of z_B (second index)
    log_density: torch.Tensor  # (G, G) unnormalized log density on the grid
    log_z: float  # log normalizing constant (log evidence)
    weights: torch.Tensor  # (G, G) probability mass of each cell
    mean: torch.Tensor  # (2,)
    cov: torch.Tensor  # (2, 2)
    prob_a_greater: float  # P(z_A > z_B)

    def expect(self, values: torch.Tensor) -> float:
        return float((self.weights * values).sum())


def grid_posterior(log_density, center=0.0, half_width=6.0, n=801) -> GridPosterior:
    """Normalize a 2-D unnormalized log density z -> log p(z) on a square grid."""
    axis = torch.linspace(center - half_width, center + half_width, n, dtype=torch.float64)
    za, zb = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([za.reshape(-1), zb.reshape(-1)], dim=1)
    log_p = log_density(points).reshape(n, n)
    cell = (axis[1] - axis[0]).item() ** 2
    log_z = (torch.logsumexp(log_p.reshape(-1), 0) + math.log(cell)).item()
    weights = torch.exp(log_p - torch.logsumexp(log_p.reshape(-1), 0))
    mean = torch.stack([(weights * za).sum(), (weights * zb).sum()])
    da, db = za - mean[0], zb - mean[1]
    cov = torch.stack([
        torch.stack([(weights * da * da).sum(), (weights * da * db).sum()]),
        torch.stack([(weights * da * db).sum(), (weights * db * db).sum()]),
    ])
    # Cells on the diagonal z_A = z_B are split evenly between the two events.
    greater = (za > zb).double() + 0.5 * (za == zb).double()
    prob = float((weights * greater).sum())
    return GridPosterior(za, zb, log_p, log_z, weights, mean, cov, prob)


def kl_gaussian_to_grid(post: GridPosterior, mean, cov) -> float:
    """KL(q || p) for q = N(mean, cov), by quadrature on the posterior's grid."""
    points = torch.stack([post.za.reshape(-1), post.zb.reshape(-1)], dim=1)
    dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    log_q = dist.log_prob(points).reshape(post.za.shape)
    cell = ((post.za[1, 0] - post.za[0, 0]) ** 2).item()
    q_mass = torch.exp(log_q) * cell
    log_p = post.log_density - post.log_z
    return float((q_mass * (log_q - log_p)).sum())
