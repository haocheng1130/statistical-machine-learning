"""Neural-network VAE: noise models, ancestral sampling, training, and the latent quadrature."""

import math

import numpy as np
import torch

from src import data, vae


def test_noise_models():
    torch.manual_seed(0)
    z = torch.randn(7, 2)
    fixed = vae.Decoder(2, 2, "fixed", sigma_fixed=0.1)
    assert "log_sigma" not in dict(fixed.named_parameters())
    np.testing.assert_allclose(torch.exp(fixed(z)[1]).detach().numpy(), 0.1, rtol=1e-6)
    learned = vae.Decoder(2, 2, "learned")
    assert dict(learned.named_parameters())["log_sigma"].shape == (2,)
    log_std = learned(z)[1]
    assert log_std.shape == (7, 2) and torch.all(log_std == log_std[0])
    per_z = vae.Decoder(2, 2, "learned_z")
    with torch.no_grad():
        per_z.log_std.weight.normal_()
    assert per_z(z)[1].std(0).min() > 0


def test_ancestral_sampling_adds_decoder_noise():
    model = vae.VAE(2, 2, "fixed", sigma_fixed=0.1)
    torch.manual_seed(1)
    means = model.sample(20_000, with_noise=False)
    torch.manual_seed(1)
    z = torch.randn(20_000, 2)
    with torch.no_grad():
        np.testing.assert_allclose(means.numpy(), model.decoder(z)[0].numpy(), rtol=1e-6)
    torch.manual_seed(1)
    noisy = model.sample(20_000, with_noise=True)
    resid = (noisy - means).numpy()
    np.testing.assert_allclose(resid.std(0), 0.1, rtol=0.03)


def test_quadrature_matches_prior_monte_carlo():
    """For a broad likelihood, log mean_s p(x | z_s) with z_s ~ p(z) is an accurate reference."""
    torch.manual_seed(2)
    dec = vae.Decoder(2, 2, "fixed", sigma_fixed=0.3).double()
    with torch.no_grad():
        dec.mean.weight.mul_(5.0)  # spread the decoder means so p(x | z) varies with z
    x = torch.tensor([[0.0, 0.0], [0.3, -0.2], [-0.5, 0.4]], dtype=torch.float64)
    quad = vae.log_marginal_quadrature(dec, x, 2, n_grid=400).numpy()
    z = torch.randn(1_000_000, 2, dtype=torch.float64)
    with torch.no_grad():
        mean, log_std = dec(z)
        log_lik = vae.diag_gaussian_log_density(x[:, None, :], mean[None], log_std[None])
    mc = (torch.logsumexp(log_lik, 1) - math.log(len(z))).numpy()
    np.testing.assert_allclose(quad, mc, atol=5e-3)


def test_training_improves_elbo_and_iwae_is_below_exact():
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    x = torch.tensor(data.sample(100, 0.05, rng)[0], dtype=torch.float32)
    model = vae.VAE(2, 2, "fixed", sigma_fixed=0.1)
    hist = vae.train(model, x, x, n_iters=600, batch_size=300, lr=3e-3, seed=0, eval_every=200)
    assert hist["train_elbo"][-1] > hist["train_elbo"][0] + 5
    enc, dec = model.encoder, model.decoder
    exact = vae.log_marginal_quadrature(dec, x, 2, n_grid=640)
    finer = vae.log_marginal_quadrature(dec, x, 2, n_grid=1280)
    np.testing.assert_allclose(exact.numpy(), finer.numpy(), atol=1e-3)
    with torch.no_grad():
        bound = vae.elbo(enc, dec, x, n_samples=200).mean().item()
    iwae = vae.iwae_bound(enc, dec, x, 2000).mean().item()
    assert bound < iwae <= exact.mean().item() + 1e-3
