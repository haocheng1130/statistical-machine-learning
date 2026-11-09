"""The data generator and the exact data density."""

import numpy as np

from src import data


def test_sampler_arcs_and_noise():
    rng = np.random.default_rng(0)
    x, label, theta = data.sample(2000, 0.05, rng)
    assert x.shape == (6000, 2)
    np.testing.assert_array_equal(np.bincount(label), [2000, 2000, 2000])
    offset = theta - 2 * np.pi * label / 3
    assert offset.min() >= 0 and offset.max() <= np.pi
    resid = x - data.curve(theta)
    np.testing.assert_allclose(resid.std(0), 0.05, rtol=0.05)
    np.testing.assert_allclose(data.curve(np.array([0.0, np.pi / 2]))[1], [1.0, 0.0], atol=1e-15)


def test_quadrature_is_converged():
    rng = np.random.default_rng(1)
    x, _, _ = data.sample(100, 0.02, rng)
    x = np.vstack([x, rng.uniform(-1.3, 1.3, size=(100, 2))])
    a = data.log_density(x, 0.02, n_panels=128)
    b = data.log_density(x, 0.02, n_panels=512)
    np.testing.assert_allclose(a, b, atol=1e-8)


def test_density_integrates_to_one():
    h = 0.01
    gx = np.arange(-1.7, 1.7, h) + h / 2
    gy = np.arange(-1.1, 1.1, h) + h / 2
    xx, yy = np.meshgrid(gx, gy)
    p = np.exp(data.log_density(np.stack([xx.ravel(), yy.ravel()], 1), 0.1, n_panels=32))
    np.testing.assert_allclose(p.sum() * h * h, 1.0, atol=1e-4)


def test_density_matches_empirical_frequencies():
    """Probability of small boxes: integral of the density vs frequency in a large sample."""
    rng = np.random.default_rng(2)
    noise = 0.05
    x, _, _ = data.sample(300_000, noise, rng)
    n = len(x)
    for cx, cy in [(0.0, 0.0), (0.9, 0.3), (-0.6, -0.45), (0.95, -0.1), (0.3, 0.3)]:
        w = 0.1
        inside = (np.abs(x[:, 0] - cx) < w / 2) & (np.abs(x[:, 1] - cy) < w / 2)
        freq = inside.mean()
        s = (np.arange(40) + 0.5) / 40 * w - w / 2
        sx, sy = np.meshgrid(cx + s, cy + s)
        prob = np.exp(data.log_density(np.stack([sx.ravel(), sy.ravel()], 1), noise)).mean() * w * w
        se = np.sqrt(prob * (1 - prob) / n)
        assert prob > 1e-3
        assert abs(freq - prob) < 4 * se
