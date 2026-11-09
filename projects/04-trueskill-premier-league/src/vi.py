"""Gaussian variational inference fitted by stochastic gradients (reparameterization).

ELBO(q) = E_q[log p(z, data)] + H(q) <= log p(data). The expectation is estimated
with reparameterized draws z = mu + L eps, eps ~ N(0, I); the Gaussian entropy is
used in closed form, H(q) = sum_i log L_ii + (N / 2)(1 + log 2 pi).
"""
import math
import time

import torch


class MeanFieldGaussian:
    """q(z) = prod_i N(z_i; mu_i, sigma_i^2), parameters (mu, log sigma)."""

    def __init__(self, dim, init_mean=0.0, init_log_sd=0.0, dtype=torch.float64):
        self.mean = torch.full((dim,), float(init_mean), dtype=dtype, requires_grad=True)
        self.log_sd = torch.full((dim,), float(init_log_sd), dtype=dtype, requires_grad=True)
        self.dim = dim

    def parameters(self):
        return [self.mean, self.log_sd]

    def sample(self, n, generator=None):
        eps = torch.randn((n, self.dim), generator=generator, dtype=self.mean.dtype)
        return self.mean + eps * self.log_sd.exp()

    def entropy(self):
        return self.log_sd.sum() + 0.5 * self.dim * (1 + math.log(2 * math.pi))

    def covariance(self):
        return torch.diag(torch.exp(2 * self.log_sd)).detach()


class FullRankGaussian:
    """q(z) = N(z; mu, L L^T), L lower triangular with a positive (exp) diagonal."""

    def __init__(self, dim, init_mean=0.0, init_log_sd=0.0, dtype=torch.float64):
        self.mean = torch.full((dim,), float(init_mean), dtype=dtype, requires_grad=True)
        raw = torch.zeros((dim, dim), dtype=dtype)
        raw.diagonal().fill_(float(init_log_sd))
        self.raw = raw.requires_grad_(True)
        self.dim = dim

    def parameters(self):
        return [self.mean, self.raw]

    def scale_tril(self):
        return torch.tril(self.raw, -1) + torch.diag(self.raw.diagonal().exp())

    def sample(self, n, generator=None):
        eps = torch.randn((n, self.dim), generator=generator, dtype=self.mean.dtype)
        return self.mean + eps @ self.scale_tril().T

    def entropy(self):
        return self.raw.diagonal().sum() + 0.5 * self.dim * (1 + math.log(2 * math.pi))

    def covariance(self):
        scale = self.scale_tril()
        return (scale @ scale.T).detach()


def gaussian_log_density(z, mean, cov):
    """log N(z; mean, cov) for a batch z of shape (S, N)."""
    return torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).log_prob(z)


def elbo_estimate(q, log_joint, n_samples, generator=None):
    """Monte Carlo ELBO estimate and its standard error (no gradient)."""
    with torch.no_grad():
        values = log_joint(q.sample(n_samples, generator)) + q.entropy()
    return values.mean().item(), (values.std() / math.sqrt(n_samples)).item()


def fit_vi(q, log_joint, n_iters, lr, n_samples, seed=0, optimizer="adam", momentum=0.9,
           final_lr_fraction=1.0):
    """Maximize the ELBO by stochastic gradient ascent on the reparameterized estimate.

    The learning rate decays geometrically from `lr` to `lr * final_lr_fraction`.
    Returns the per-iteration ELBO estimates and the wall-clock time.
    """
    gen = torch.Generator().manual_seed(seed)
    if optimizer == "adam":
        opt = torch.optim.Adam(q.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(q.parameters(), lr=lr, momentum=momentum)
    gamma = final_lr_fraction ** (1.0 / max(n_iters, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    history = torch.empty(n_iters, dtype=torch.float64)
    start = time.perf_counter()
    for t in range(n_iters):
        opt.zero_grad()
        elbo = log_joint(q.sample(n_samples, gen)).mean() + q.entropy()
        (-elbo).backward()
        opt.step()
        sched.step()
        history[t] = elbo.item()
    return history, time.perf_counter() - start


def prob_greater(mean, cov, i, j):
    """P(z_i > z_j) under z ~ N(mean, cov): Phi((mu_i - mu_j) / sd(z_i - z_j))."""
    var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
    return torch.special.ndtr((mean[i] - mean[j]) / torch.sqrt(var))


def log_evidence_is(log_joint, mean, cov, n_samples, seed=0, batch=20000):
    """Importance-sampling estimate of log p(data) with proposal N(mean, cov).

    Returns (log evidence, Kish effective sample size of the weights).
    """
    gen = torch.Generator().manual_seed(seed)
    dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    scale = torch.linalg.cholesky(cov)
    log_w = []
    for start in range(0, n_samples, batch):
        size = min(batch, n_samples - start)
        z = mean + torch.randn((size, mean.shape[0]), generator=gen, dtype=mean.dtype) @ scale.T
        log_w.append(log_joint(z) - dist.log_prob(z))
    log_w = torch.cat(log_w)
    log_z = torch.logsumexp(log_w, 0) - math.log(n_samples)
    w = torch.exp(log_w - log_w.max())
    kish = (w.sum() ** 2 / (w ** 2).sum()).item()
    return log_z.item(), kish
