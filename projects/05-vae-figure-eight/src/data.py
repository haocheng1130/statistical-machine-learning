"""Synthetic figure-eight ("banana") data and its exact density.

Each point is generated as

    c ~ Uniform{0, ..., n_arcs - 1}           (arc label, never shown to the models)
    theta ~ Uniform[2 pi c / n_arcs, 2 pi c / n_arcs + pi]
    x = f(theta) + noise_std * eps,           eps ~ N(0, I_2)

with f(theta) = (sin theta, sin theta cos theta). The curve f traces a figure eight
(lemniscate of Gerono) with a self-crossing at the origin, and the arcs overlap: with three
arcs of length pi on a curve of period 2 pi, half of the curve is covered by two arcs.

Because the noise is isotropic Gaussian, the data density is a one-dimensional integral

    p(x) = sum_c (1 / n_arcs) * (1 / pi) * int_{arc c} N(x; f(theta), noise_std^2 I) dtheta,

which we evaluate with composite Gauss-Legendre quadrature on each arc.
"""

import numpy as np
from scipy.special import logsumexp


def curve(theta):
    """Points on the figure-eight curve, shape theta.shape + (2,)."""
    theta = np.asarray(theta, dtype=float)
    return np.stack([np.sin(theta), np.sin(theta) * np.cos(theta)], axis=-1)


def arc_start(label, n_arcs):
    return 2.0 * np.pi * np.asarray(label) / n_arcs


def sample(n_per_arc, noise_std, rng, n_arcs=3):
    """Draw n_per_arc points from each arc.

    Returns (x, label, theta) in random order, with x of shape (n, 2).
    """
    label = np.repeat(np.arange(n_arcs), n_per_arc)
    theta = arc_start(label, n_arcs) + rng.uniform(0.0, np.pi, size=label.size)
    x = curve(theta) + noise_std * rng.standard_normal((label.size, 2))
    perm = rng.permutation(label.size)
    return x[perm], label[perm], theta[perm]


def _quadrature(n_arcs, n_panels, order=16):
    """Composite Gauss-Legendre rule: nodes theta_m and log weights such that
    sum_m w_m g(theta_m) ~ E_theta[g(theta)], with n_panels panels of ``order`` nodes per arc."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    width = np.pi / n_panels
    left = width * np.arange(n_panels)
    t = (left[:, None] + (nodes[None, :] + 1.0) * width / 2.0).ravel()  # nodes in [0, pi]
    w = np.tile(weights * width / 2.0, n_panels)
    theta = np.concatenate([t + arc_start(c, n_arcs) for c in range(n_arcs)])
    log_w = np.log(np.tile(w, n_arcs) / (np.pi * n_arcs))
    return theta, log_w


def log_density(x, noise_std, n_arcs=3, n_panels=128, chunk=1024):
    """Exact log p(x) of the data distribution (up to quadrature error), shape (n,)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    theta, log_w = _quadrature(n_arcs, n_panels)
    f = curve(theta)
    log_norm = -np.log(2.0 * np.pi * noise_std**2)
    out = np.empty(len(x))
    for start in range(0, len(x), chunk):
        xb = x[start:start + chunk]
        sq = ((xb[:, None, :] - f[None, :, :]) ** 2).sum(-1)
        out[start:start + chunk] = logsumexp(log_w[None, :] + log_norm - sq / (2.0 * noise_std**2), axis=1)
    return out
