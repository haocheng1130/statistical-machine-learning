"""Gradient-based MCMC: Metropolis-adjusted Langevin (MALA) and Hamiltonian Monte Carlo.

Both samplers run C chains in parallel (z has shape (C, N)) and share one step
size. During warm-up the step size is tuned by dual averaging (Hoffman and
Gelman, 2014) toward a target mean acceptance probability; afterwards it is
frozen, so the retained draws come from a fixed, valid Metropolis-Hastings kernel.
"""
import math
import time
from dataclasses import dataclass

import torch

from .model import value_and_grad


class DualAveraging:
    """Step-size adaptation of Hoffman and Gelman (2014, Sec. 3.2)."""

    def __init__(self, step_size, target, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = math.log(10.0 * step_size)
        self.target, self.gamma, self.t0, self.kappa = target, gamma, t0, kappa
        self.h_bar, self.log_bar, self.m = 0.0, 0.0, 0

    def update(self, accept_prob: float) -> float:
        """Record one acceptance probability and return the next step size."""
        self.m += 1
        w = 1.0 / (self.m + self.t0)
        self.h_bar = (1 - w) * self.h_bar + w * (self.target - accept_prob)
        log_step = self.mu - math.sqrt(self.m) / self.gamma * self.h_bar
        decay = self.m ** (-self.kappa)
        self.log_bar = decay * log_step + (1 - decay) * self.log_bar
        return math.exp(log_step)

    def final(self) -> float:
        return math.exp(self.log_bar)


def _accept_prob(log_alpha):
    """min(1, exp(log_alpha)), treating NaN (numerical failure) as rejection."""
    log_alpha = torch.nan_to_num(log_alpha, nan=-math.inf)
    return torch.exp(log_alpha.clamp(max=0.0)), log_alpha


@dataclass
class ChainResult:
    draws: torch.Tensor  # (n_draws, C, N), warm-up removed
    accept_rate: torch.Tensor  # (C,) fraction of accepted proposals after warm-up
    step_size: float  # frozen step size used after warm-up
    n_grad: int  # gradient evaluations per chain, warm-up included
    seconds: float  # wall-clock time for all chains


def mala(log_density, z0, n_warmup, n_draws, step_size, target_accept=0.574,
         adapt=True, log_q_scale=None, seed=0) -> ChainResult:
    """Metropolis-adjusted Langevin algorithm.

    Proposal: z' = z + eta * grad log p(z) + sqrt(2 eta) * xi, xi ~ N(0, I), i.e.
        q(z' | z) = N(z'; z + eta * grad log p(z), 2 eta I),
        log q(z' | z) = -||z' - z - eta * grad log p(z)||^2 / (4 eta) + const.
    `log_q_scale` replaces the factor 1/(4 eta) and exists only to reproduce the
    pitfall of omitting it (log_q_scale=1); leave it as None for correct MALA.
    """
    gen = torch.Generator().manual_seed(seed)
    z = z0.clone()
    logp, grad = value_and_grad(log_density, z)
    eta = step_size
    adapter = DualAveraging(eta, target_accept) if adapt and n_warmup > 0 else None
    draws = torch.empty((n_draws,) + tuple(z.shape), dtype=z.dtype)
    accepted = torch.zeros(z.shape[0], dtype=z.dtype)
    start = time.perf_counter()
    for t in range(n_warmup + n_draws):
        if t == n_warmup and adapter is not None:
            eta = adapter.final()
        noise = torch.randn(z.shape, generator=gen, dtype=z.dtype)
        z_prop = z + eta * grad + math.sqrt(2 * eta) * noise
        logp_prop, grad_prop = value_and_grad(log_density, z_prop)
        scale = 1.0 / (4 * eta) if log_q_scale is None else log_q_scale
        log_q_forward = -scale * ((z_prop - z - eta * grad) ** 2).sum(-1)
        log_q_backward = -scale * ((z - z_prop - eta * grad_prop) ** 2).sum(-1)
        prob, log_alpha = _accept_prob(logp_prop - logp + log_q_backward - log_q_forward)
        u = torch.rand(z.shape[0], generator=gen, dtype=z.dtype)
        accept = torch.log(u) < log_alpha
        z = torch.where(accept[:, None], z_prop, z)
        logp = torch.where(accept, logp_prop, logp)
        grad = torch.where(accept[:, None], grad_prop, grad)
        if t < n_warmup:
            if adapter is not None:
                eta = adapter.update(prob.mean().item())
        else:
            draws[t - n_warmup] = z
            accepted += accept
    seconds = time.perf_counter() - start
    return ChainResult(draws, accepted / max(n_draws, 1), eta, n_warmup + n_draws + 1, seconds)


def leapfrog(z, p, grad, step_size, n_steps, log_density):
    """Leapfrog integration of Hamiltonian dynamics with H(z, p) = -log p(z) + |p|^2 / 2.

    `grad` is the gradient of log p at the starting z. Returns the final
    (z, p, log p(z), grad log p(z)); costs n_steps gradient evaluations.
    """
    p = p + 0.5 * step_size * grad
    for i in range(n_steps):
        z = z + step_size * p
        logp, grad = value_and_grad(log_density, z)
        if i < n_steps - 1:
            p = p + step_size * grad
    p = p + 0.5 * step_size * grad
    return z, p, logp, grad


def hmc(log_density, z0, n_warmup, n_draws, step_size, n_leapfrog, target_accept=0.65,
        adapt=True, jitter=0.1, seed=0) -> ChainResult:
    """Hamiltonian Monte Carlo with identity mass matrix.

    Each iteration draws a fresh momentum p ~ N(0, I), runs `n_leapfrog`
    leapfrog steps with step size eps * U(1 - jitter, 1 + jitter), and accepts
    with probability min(1, exp(H(z, p) - H(z', p'))). The random jitter avoids
    periodic trajectories; momentum negation is omitted because H is even in p.
    """
    gen = torch.Generator().manual_seed(seed)
    z = z0.clone()
    logp, grad = value_and_grad(log_density, z)
    eps = step_size
    adapter = DualAveraging(eps, target_accept) if adapt and n_warmup > 0 else None
    draws = torch.empty((n_draws,) + tuple(z.shape), dtype=z.dtype)
    accepted = torch.zeros(z.shape[0], dtype=z.dtype)
    start = time.perf_counter()
    for t in range(n_warmup + n_draws):
        if t == n_warmup and adapter is not None:
            eps = adapter.final()
        p = torch.randn(z.shape, generator=gen, dtype=z.dtype)
        jittered = eps * (1 + jitter * (2 * torch.rand((), generator=gen, dtype=z.dtype).item() - 1))
        z_new, p_new, logp_new, grad_new = leapfrog(z, p, grad, jittered, n_leapfrog, log_density)
        h_old = -logp + 0.5 * (p ** 2).sum(-1)
        h_new = -logp_new + 0.5 * (p_new ** 2).sum(-1)
        prob, log_alpha = _accept_prob(h_old - h_new)
        u = torch.rand(z.shape[0], generator=gen, dtype=z.dtype)
        accept = torch.log(u) < log_alpha
        z = torch.where(accept[:, None], z_new, z)
        logp = torch.where(accept, logp_new, logp)
        grad = torch.where(accept[:, None], grad_new, grad)
        if t < n_warmup:
            if adapter is not None:
                eps = adapter.update(prob.mean().item())
        else:
            draws[t - n_warmup] = z
            accepted += accept
    seconds = time.perf_counter() - start
    n_grad = (n_warmup + n_draws) * n_leapfrog + 1
    return ChainResult(draws, accepted / max(n_draws, 1), eps, n_grad, seconds)
