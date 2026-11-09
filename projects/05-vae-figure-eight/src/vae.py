"""Variational autoencoder with a diagonal-Gaussian encoder and a Gaussian decoder.

Generative model:  z ~ N(0, I_d),  x | z ~ N(mu_theta(z), diag(sigma_theta(z)^2)).
Inference model:   q_phi(z | x) = N(m_phi(x), diag(s_phi(x)^2)).

The bound functions below take an ``encoder`` and a ``decoder`` callable, each returning
(mean, log_std) with the same leading shape as its input. This lets the same code run on the
neural-network VAE and on a linear-Gaussian model whose marginal likelihood is known exactly.
"""

import math

import numpy as np
import torch
from torch import nn

LOG_2PI = math.log(2.0 * math.pi)


# ----------------------------------------------------------------------------------------
# Diagonal Gaussians
# ----------------------------------------------------------------------------------------

def diag_gaussian_log_density(x, mean, log_std):
    """log N(x; mean, diag(exp(log_std)^2)), summed over the last axis."""
    z = (x - mean) * torch.exp(-log_std)
    return (-0.5 * LOG_2PI - log_std - 0.5 * z**2).sum(-1)


def sample_diag_gaussian(mean, log_std, n_samples):
    """Reparameterized samples mean + exp(log_std) * eps, shape (n_samples,) + mean.shape.

    The randomness enters only through eps ~ N(0, I), so the samples are differentiable
    functions of mean and log_std.
    """
    eps = torch.randn((n_samples,) + tuple(mean.shape), dtype=mean.dtype, device=mean.device)
    return mean + torch.exp(log_std) * eps


def kl_diag_gaussians(mean_q, log_std_q, mean_p, log_std_p):
    """KL(N(mean_q, s_q^2) || N(mean_p, s_p^2)) for diagonal Gaussians, summed over last axis."""
    var_ratio = torch.exp(2.0 * (log_std_q - log_std_p))
    mahal = ((mean_q - mean_p) * torch.exp(-log_std_p)) ** 2
    return 0.5 * (var_ratio + mahal - 1.0 - 2.0 * (log_std_q - log_std_p)).sum(-1)


def kl_to_standard_normal(mean, log_std):
    zero = torch.zeros_like(mean)
    return kl_diag_gaussians(mean, log_std, zero, zero)


# ----------------------------------------------------------------------------------------
# Bounds on log p(x)
# ----------------------------------------------------------------------------------------

def log_weights(encoder, decoder, x, n_samples):
    """log w_k = log p(z_k) + log p(x | z_k) - log q(z_k | x) with z_k ~ q(z | x).

    Returns a tensor of shape (n_samples, batch).
    """
    mean, log_std = encoder(x)
    z = sample_diag_gaussian(mean, log_std, n_samples)
    dec_mean, dec_log_std = decoder(z)
    log_lik = diag_gaussian_log_density(x, dec_mean, dec_log_std)
    log_prior = diag_gaussian_log_density(z, torch.zeros_like(z), torch.zeros_like(z))
    log_q = diag_gaussian_log_density(z, mean, log_std)
    return log_prior + log_lik - log_q


def elbo(encoder, decoder, x, n_samples=1, analytic_kl=True):
    """Monte Carlo estimate of the ELBO for every data point, shape (batch,).

    ELBO(x) = E_q[log p(x | z)] - KL(q(z | x) || p(z)).
    With analytic_kl=True the KL term is computed in closed form and only the
    reconstruction term is estimated by sampling; otherwise the whole integrand
    log p(x, z) - log q(z | x) is averaged over the samples. Both are unbiased.
    """
    if not analytic_kl:
        return log_weights(encoder, decoder, x, n_samples).mean(0)
    mean, log_std = encoder(x)
    z = sample_diag_gaussian(mean, log_std, n_samples)
    dec_mean, dec_log_std = decoder(z)
    recon = diag_gaussian_log_density(x, dec_mean, dec_log_std).mean(0)
    return recon - kl_to_standard_normal(mean, log_std)


@torch.no_grad()
def iwae_bound(encoder, decoder, x, n_samples, batch_size=64):
    """Importance-weighted bound log (1/K) sum_k w_k with K = n_samples, shape (batch,).

    E[IWAE_K] <= log p(x) and increases with K; IWAE_1 is the ELBO.
    """
    out = []
    for start in range(0, len(x), batch_size):
        lw = log_weights(encoder, decoder, x[start:start + batch_size], n_samples)
        out.append(torch.logsumexp(lw, 0) - math.log(n_samples))
    return torch.cat(out)


@torch.no_grad()
def log_marginal_quadrature(decoder, x, latent_dim, n_grid, half_width=5.0, batch_size=256):
    """log p(x) = log int N(z; 0, I) p(x | z) dz by a midpoint rule (latent_dim 1 or 2).

    The grid has n_grid points per axis on [-half_width, half_width]; in 2-D only the cells
    inside the disk of radius half_width are kept (the prior mass outside is exp(-L^2 / 2)).
    The result does not involve the encoder and is accurate when the grid resolves p(x | z)
    as a function of z. The Gaussian log-likelihood is expanded as a quadratic in x, so all
    grid points are handled by one matrix product, in float64 to avoid cancellation.
    """
    h = 2.0 * half_width / n_grid
    g = -half_width + h * (torch.arange(n_grid, dtype=torch.float64) + 0.5)
    if latent_dim == 1:
        z = g[:, None]
    elif latent_dim == 2:
        z = torch.cartesian_prod(g, g)
        z = z[(z**2).sum(-1) <= half_width**2]
    else:
        raise ValueError("quadrature is implemented for latent_dim 1 or 2")
    dec_mean, dec_log_std = (t.double() for t in decoder(z.to(x.dtype)))
    prec = torch.exp(-2.0 * dec_log_std)
    # log w_g + log N(x; m_g, diag(s_g^2)) = c_g + sum_d [-(prec_gd / 2) x_d^2 + m_gd prec_gd x_d]
    log_prior = -0.5 * (latent_dim * LOG_2PI + (z**2).sum(-1))
    const = log_prior + latent_dim * math.log(h) - (0.5 * LOG_2PI + dec_log_std + 0.5 * dec_mean**2 * prec).sum(-1)
    coef = torch.cat([-0.5 * prec, dec_mean * prec], dim=-1)
    out = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].double()
        feats = torch.cat([xb**2, xb], dim=-1)
        out.append(torch.logsumexp(feats @ coef.T + const, dim=1))
    return torch.cat(out).to(x.dtype)


# ----------------------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------------------

def _mlp(d_in, hidden):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers), d


class Encoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden=(100, 50)):
        super().__init__()
        self.body, d = _mlp(data_dim, hidden)
        self.mean = nn.Linear(d, latent_dim)
        self.log_std = nn.Linear(d, latent_dim)

    def forward(self, x):
        h = self.body(x)
        return self.mean(h), self.log_std(h)


class Decoder(nn.Module):
    """Gaussian decoder. ``noise`` selects how the observation noise is modelled:

    - "fixed":     sigma is a constant ``sigma_fixed`` (not trained);
    - "learned":   one trained log sigma per data dimension, shared by all z;
    - "learned_z": log sigma(z) is an output of the network (heteroscedastic).
    """

    def __init__(self, latent_dim, data_dim, noise="learned", sigma_fixed=0.1, hidden=(100, 50)):
        super().__init__()
        if noise not in ("fixed", "learned", "learned_z"):
            raise ValueError(f"unknown noise model {noise!r}")
        self.noise = noise
        self.body, d = _mlp(latent_dim, hidden)
        self.mean = nn.Linear(d, data_dim)
        init = torch.full((data_dim,), math.log(sigma_fixed))
        if noise == "fixed":
            self.register_buffer("log_sigma", init)
        elif noise == "learned":
            self.log_sigma = nn.Parameter(init)
        else:
            self.log_std = nn.Linear(d, data_dim)
            nn.init.zeros_(self.log_std.weight)
            nn.init.constant_(self.log_std.bias, math.log(sigma_fixed))

    def forward(self, z):
        h = self.body(z)
        mean = self.mean(h)
        if self.noise == "learned_z":
            return mean, self.log_std(h)
        return mean, self.log_sigma.expand_as(mean)


class VAE(nn.Module):
    def __init__(self, data_dim=2, latent_dim=2, noise="learned", sigma_fixed=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(data_dim, latent_dim)
        self.decoder = Decoder(latent_dim, data_dim, noise, sigma_fixed)

    @torch.no_grad()
    def sample(self, n, with_noise=True):
        """Ancestral samples from p(z) p(x | z); with_noise=False returns the decoder means."""
        z = torch.randn(n, self.latent_dim)
        mean, log_std = self.decoder(z)
        if not with_noise:
            return mean
        return mean + torch.exp(log_std) * torch.randn_like(mean)


# ----------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------

def train(model, x_train, x_val, n_iters, batch_size, lr, seed, lr_min_factor=0.01, eval_every=250,
          eval_samples=10):
    """Maximize the average ELBO with Adam on shuffled minibatches (one z sample per point).

    The learning rate follows a cosine schedule from lr to lr_min_factor * lr. Returns a dict
    of curves evaluated every ``eval_every`` iterations on the full training and validation
    sets (ELBO estimated with ``eval_samples`` samples per point) and the decoder noise
    standard deviation per data dimension, averaged over the encoded training points.
    """
    gen = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x_train)
    history = {"iteration": [], "train_elbo": [], "val_elbo": [], "sigma": []}
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iters, eta_min=lr_min_factor * lr)
    perm, pos = gen.permutation(n), 0
    for it in range(n_iters + 1):
        if it % eval_every == 0 or it == n_iters:
            with torch.no_grad():
                history["iteration"].append(it)
                history["train_elbo"].append(elbo(model.encoder, model.decoder, x_train, eval_samples).mean().item())
                history["val_elbo"].append(elbo(model.encoder, model.decoder, x_val, eval_samples).mean().item())
                _, log_std = model.decoder(model.encoder(x_train)[0])
                history["sigma"].append(torch.exp(log_std).mean(0).tolist())
        if it == n_iters:
            break
        if pos + batch_size > n:
            perm, pos = gen.permutation(n), 0
        idx = torch.as_tensor(perm[pos:pos + batch_size])
        pos += batch_size
        opt.zero_grad()
        loss = -elbo(model.encoder, model.decoder, x_train[idx]).mean()
        loss.backward()
        opt.step()
        sched.step()
    return history
