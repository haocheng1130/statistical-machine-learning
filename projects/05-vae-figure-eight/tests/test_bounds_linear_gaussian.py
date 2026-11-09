"""ELBO, IWAE and latent quadrature on a linear-Gaussian VAE, where log p(x) is exact.

Model: z ~ N(0, I_d), x | z ~ N(W z + b, sigma^2 I_D). This is probabilistic PCA, with
marginal x ~ N(b, W W^T + sigma^2 I) and posterior z | x ~ N(M^{-1} W^T (x - b), sigma^2 M^{-1}),
M = W^T W + sigma^2 I.
"""

import math

import numpy as np
import torch
from scipy.stats import multivariate_normal

from src.vae import elbo, iwae_bound, kl_to_standard_normal, log_marginal_quadrature

F64 = torch.float64
SIGMA = 0.3


def make_model(d, orthogonal=False, seed=0):
    gen = torch.Generator().manual_seed(seed)
    D = 3
    W = torch.randn(D, d, generator=gen, dtype=F64)
    if orthogonal:
        q, _ = torch.linalg.qr(W)
        W = q * torch.tensor([1.5, 0.8][:d], dtype=F64)
    b = torch.randn(D, generator=gen, dtype=F64)

    def decoder(z):
        mean = z @ W.T + b
        return mean, torch.full_like(mean, math.log(SIGMA))

    return W, b, decoder


def exact_log_marginal(W, b, x):
    cov = (W @ W.T + SIGMA**2 * torch.eye(len(b), dtype=F64)).numpy()
    return multivariate_normal(b.numpy(), cov).logpdf(x.numpy())


def exact_posterior(W, b, x):
    M = W.T @ W + SIGMA**2 * torch.eye(W.shape[1], dtype=F64)
    mean = torch.linalg.solve(M, W.T @ (x - b).T).T
    return mean, SIGMA**2 * torch.linalg.inv(M)


def make_encoder(mean, log_std):
    """A fixed diagonal-Gaussian q(z | x) for each row of x (same q for every row)."""
    return lambda x: (mean.expand(len(x), -1), log_std.expand(len(x), -1))


def closed_form_elbo(W, b, x, m, log_s):
    """E_q[log p(x | z)] - KL(q || p) for q = N(m, diag(exp(2 log_s))), in closed form."""
    D = len(b)
    s2 = torch.exp(2 * log_s)
    resid = ((x - m @ W.T - b) ** 2).sum(-1)
    trace = (s2 * (W**2).sum(0)).sum(-1)
    recon = -D * (0.5 * math.log(2 * math.pi) + math.log(SIGMA)) - (resid + trace) / (2 * SIGMA**2)
    return recon - kl_to_standard_normal(m, log_s)


def kl_full(m_q, s2_q, m_p, cov_p):
    """KL(N(m_q, diag(s2_q)) || N(m_p, cov_p))."""
    d = len(m_q)
    prec = torch.linalg.inv(cov_p)
    diff = m_p - m_q
    return 0.5 * (torch.trace(prec @ torch.diag(s2_q)) + diff @ prec @ diff - d
                  + torch.logdet(cov_p) - torch.log(s2_q).sum())


def setup(d=2, orthogonal=False):
    W, b, decoder = make_model(d, orthogonal)
    x = torch.tensor([[0.5, -1.0, 2.0]], dtype=F64)
    return W, b, decoder, x


def test_elbo_gap_equals_kl_to_posterior():
    W, b, decoder, x = setup()
    m, log_s = torch.tensor([[0.2, -0.4]], dtype=F64), torch.tensor([[-1.0, -0.5]], dtype=F64)
    lp = exact_log_marginal(W, b, x)[()]
    bound = closed_form_elbo(W, b, x, m, log_s).item()
    post_mean, post_cov = exact_posterior(W, b, x)
    gap = kl_full(m[0], torch.exp(2 * log_s[0]), post_mean[0], post_cov).item()
    assert bound < lp
    np.testing.assert_allclose(lp - bound, gap, rtol=1e-10)


def test_monte_carlo_elbo_is_unbiased():
    torch.manual_seed(0)
    W, b, decoder, x = setup()
    m, log_s = torch.tensor([[0.2, -0.4]], dtype=F64), torch.tensor([[-1.0, -0.5]], dtype=F64)
    encoder = make_encoder(m, log_s)
    target = closed_form_elbo(W, b, x, m, log_s).item()
    xs = x.expand(100_000, -1)
    for analytic in (True, False):
        est = elbo(encoder, decoder, xs, n_samples=1, analytic_kl=analytic)
        se = est.std().item() / math.sqrt(len(est))
        assert abs(est.mean().item() - target) < 4 * se + 1e-12


def test_exact_posterior_makes_both_bounds_tight():
    W, b, decoder, x = setup(orthogonal=True)
    post_mean, post_cov = exact_posterior(W, b, x)
    off_diag = post_cov - torch.diag(torch.diagonal(post_cov))
    assert off_diag.abs().max() < 1e-12  # orthogonal columns => diagonal posterior
    post_log_std = 0.5 * torch.log(torch.diagonal(post_cov))[None]
    encoder = make_encoder(post_mean, post_log_std)
    lp = exact_log_marginal(W, b, x)[()]
    np.testing.assert_allclose(closed_form_elbo(W, b, x, post_mean, post_log_std).item(), lp, atol=1e-10)
    for k in (1, 50):
        np.testing.assert_allclose(iwae_bound(encoder, decoder, x, k).item(), lp, atol=1e-10)


def test_iwae_increases_with_k_towards_log_marginal():
    torch.manual_seed(1)
    W, b, decoder, x = setup()
    post_mean, _ = exact_posterior(W, b, x)
    # A deliberately poor q: shifted mean and too-wide variances.
    m, log_s = post_mean + 0.3, torch.tensor([[-0.2, 0.1]], dtype=F64)
    encoder = make_encoder(m, log_s)
    lp = exact_log_marginal(W, b, x)[()]
    xs = x.expand(200, -1)  # 200 independent replicates of the estimator
    means, ses = [], []
    for k in (1, 10, 100, 1000, 10000):
        est = iwae_bound(encoder, decoder, xs, k, batch_size=20)
        means.append(est.mean().item())
        ses.append(est.std().item() / math.sqrt(len(est)))
    means, ses = np.array(means), np.array(ses)
    # IWAE_1 is the ELBO
    assert abs(means[0] - closed_form_elbo(W, b, x, m, log_s).item()) < 4 * ses[0]
    # monotone increase in K and convergence to log p(x) from below
    assert np.all(np.diff(means) > -2 * (ses[1:] + ses[:-1]))
    assert np.all(means < lp + 4 * ses)
    assert lp - means[0] > 1.0
    assert abs(lp - means[-1]) < 0.01


def test_quadrature_matches_exact_marginal():
    for d in (1, 2):
        W, b, decoder = make_model(d, seed=d)
        gen = torch.Generator().manual_seed(10 + d)
        x = b + torch.randn(5, 3, generator=gen, dtype=F64) @ torch.diag(torch.tensor([1.0, 0.8, 0.6], dtype=F64))
        n_grid = 4000 if d == 1 else 500
        quad = log_marginal_quadrature(decoder, x, d, n_grid, half_width=6.0).numpy()
        np.testing.assert_allclose(quad, exact_log_marginal(W, b, x), atol=1e-6)
