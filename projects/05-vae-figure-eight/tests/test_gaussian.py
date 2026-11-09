"""Diagonal-Gaussian primitives: log density, KL divergence and the reparameterized sampler."""

import numpy as np
import torch
from scipy.stats import multivariate_normal

from src.vae import diag_gaussian_log_density, kl_diag_gaussians, kl_to_standard_normal, sample_diag_gaussian


def t(values):
    return torch.tensor(values, dtype=torch.float64)


def test_log_density_matches_scipy():
    gen = torch.Generator().manual_seed(0)
    mean = torch.randn(3, generator=gen, dtype=torch.float64)
    log_std = 0.5 * torch.randn(3, generator=gen, dtype=torch.float64)
    x = torch.randn(10, 3, generator=gen, dtype=torch.float64)
    ours = diag_gaussian_log_density(x, mean, log_std).numpy()
    ref = multivariate_normal(mean.numpy(), np.diag(np.exp(2 * log_std.numpy()))).logpdf(x.numpy())
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)


def test_kl_closed_form_matches_monte_carlo():
    torch.manual_seed(1)
    m_q, ls_q = t([0.3, -1.0, 2.0]), t([-0.5, 0.2, 0.0])
    m_p, ls_p = t([0.0, 0.5, 1.0]), t([0.3, -0.2, 0.4])
    z = sample_diag_gaussian(m_q, ls_q, 400_000)
    log_ratio = diag_gaussian_log_density(z, m_q, ls_q) - diag_gaussian_log_density(z, m_p, ls_p)
    mc, se = log_ratio.mean().item(), log_ratio.std().item() / np.sqrt(len(log_ratio))
    closed = kl_diag_gaussians(m_q, ls_q, m_p, ls_p).item()
    assert abs(mc - closed) < 4 * se
    assert closed > 0


def test_kl_special_cases():
    m, ls = t([0.4, -0.3]), t([0.1, -0.7])
    assert kl_diag_gaussians(m, ls, m, ls).item() == 0.0
    zero = torch.zeros(2, dtype=torch.float64)
    np.testing.assert_allclose(kl_to_standard_normal(m, ls).item(), kl_diag_gaussians(m, ls, zero, zero).item())
    # 0.5 * sum(mu^2 + s^2 - 1 - log s^2), the textbook formula
    s2 = torch.exp(2 * ls)
    textbook = (0.5 * (m**2 + s2 - 1 - torch.log(s2))).sum().item()
    np.testing.assert_allclose(kl_to_standard_normal(m, ls).item(), textbook)


def test_sampler_moments():
    torch.manual_seed(2)
    mean, log_std = t([1.0, -2.0, 0.5]), t([0.0, -1.0, 0.7])
    n = 200_000
    z = sample_diag_gaussian(mean, log_std, n)
    assert z.shape == (n, 3)
    std = torch.exp(log_std)
    assert torch.all((z.mean(0) - mean).abs() < 4 * std / np.sqrt(n))
    var_se = std**2 * np.sqrt(2.0 / n)
    assert torch.all((z.var(0) - std**2).abs() < 4 * var_se)
    corr = np.corrcoef(z.numpy().T)
    assert np.all(np.abs(corr[np.triu_indices(3, 1)]) < 4 / np.sqrt(n))


def test_sampler_is_reparameterized():
    """Gradients of E[z^2] flow through the samples: d/dmu = 2 mu, d/dlog_std = 2 s^2."""
    torch.manual_seed(3)
    mean = t([0.7, -0.4]).requires_grad_()
    log_std = t([-0.3, 0.2]).requires_grad_()
    z = sample_diag_gaussian(mean, log_std, 400_000)
    (z**2).mean(0).sum().backward()
    np.testing.assert_allclose(mean.grad.numpy(), 2 * mean.detach().numpy(), atol=0.02)
    np.testing.assert_allclose(log_std.grad.numpy(), 2 * np.exp(2 * log_std.detach().numpy()), atol=0.02)
