"""Checkerboard Gibbs sampling for the grid Ising model.

Given its neighbors, pixel i has p(x_i = +1 | x_N(i)) = sigmoid(2 f_i) with local
field f_i = h_i + sum_j J_ij x_j. Pixels of one checkerboard color are
conditionally independent given the other color, so each half-sweep samples a
whole color exactly. Posterior means are estimated with the Rao-Blackwellized
average of E[x_i | x_N(i)] = tanh(f_i) over the sweeps after burn-in.
"""
import numpy as np
from scipy.special import expit

from .ising import checkerboard, couplings, neighbor_field


def gibbs_sweep(x, h, Jh, Jv, rng):
    """One sweep (black then white pixels); returns the new state and E[x | neighbors]."""
    black = checkerboard(h.shape)
    cond_mean = np.zeros_like(h, dtype=float)
    for mask in (black, ~black):
        field = h + neighbor_field(x, Jh, Jv)
        draw = np.where(rng.random(h.shape) < expit(2.0 * field), 1.0, -1.0)
        x = np.where(mask, draw, x)
        cond_mean = np.where(mask, np.tanh(field), cond_mean)
    return x, cond_mean


def gibbs_means(h, J, n_sweeps, burn_in, rng, init=None):
    """Estimate E[x_i] for all pixels; the chain starts at init (default sign(h))."""
    Jh, Jv = couplings(J, h.shape)
    x = np.where(h >= 0, 1.0, -1.0) if init is None else np.array(init, dtype=float)
    acc = np.zeros_like(h, dtype=float)
    for t in range(burn_in + n_sweeps):
        x, cond_mean = gibbs_sweep(x, h, Jh, Jv, rng)
        if t >= burn_in:
            acc += cond_mean
    return acc / n_sweeps
