"""Ising model on an H x W grid with 4-neighbor couplings and an external field.

The (posterior) distribution over x in {-1, +1}^(H x W) is

    p(x) proportional to exp( sum_i h_i x_i + sum_{(i,j) in E} J_ij x_i x_j ),

where E holds the horizontal edges (r, c)-(r, c+1) and the vertical edges
(r, c)-(r+1, c). For denoising, h = beta * y. Couplings are stored as two
arrays: Jh with shape (H, W-1) and Jv with shape (H-1, W). A zero coupling is
equivalent to removing the edge, which lets the same code handle trees.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_expit


def couplings(J, shape):
    """Return (Jh, Jv) arrays for a scalar coupling or a (Jh, Jv) pair."""
    H, W = shape
    if np.isscalar(J):
        return np.full((H, W - 1), float(J)), np.full((H - 1, W), float(J))
    Jh, Jv = (np.asarray(a, dtype=float) for a in J)
    if Jh.shape != (H, W - 1) or Jv.shape != (H - 1, W):
        raise ValueError(f"coupling shapes {Jh.shape}, {Jv.shape} do not fit grid {shape}")
    return Jh, Jv


def neighbor_field(x, Jh, Jv):
    """sum_{j in N(i)} J_ij x_j for every pixel i (x may be real-valued, e.g. means)."""
    s = np.zeros_like(x, dtype=float)
    s[:, :-1] += Jh * x[:, 1:]
    s[:, 1:] += Jh * x[:, :-1]
    s[:-1, :] += Jv * x[1:, :]
    s[1:, :] += Jv * x[:-1, :]
    return s


def log_potential(x, h, Jh, Jv):
    """Unnormalized log-probability sum_i h_i x_i + sum_E J_ij x_i x_j.

    x may carry leading batch dimensions (..., H, W); the result has shape (...).
    The energy minimized by MAP inference is the negative of this quantity.
    """
    unary = (h * x).sum(axis=(-2, -1))
    horiz = (Jh * x[..., :, :-1] * x[..., :, 1:]).sum(axis=(-2, -1))
    vert = (Jv * x[..., :-1, :] * x[..., 1:, :]).sum(axis=(-2, -1))
    return unary + horiz + vert


def decide(score, y, tol=1e-9):
    """Pixelwise decision sign(score); exact ties (|score| <= tol) keep the observation y.

    score is any quantity whose sign gives the preferred state: a log-odds
    log b(+1) - log b(-1), or a posterior mean E[x_i].
    """
    out = np.sign(score)
    tie = np.abs(score) <= tol
    out[tie] = y[tie]
    return out


def pseudo_likelihood_coupling(x, J_max=10.0):
    """Maximum pseudo-likelihood estimate of a uniform coupling J from one clean image.

    For the field-free Ising prior p(x) proportional to exp(J sum_E x_i x_j),
    p(x_i | x_N(i)) = sigmoid(2 J x_i s_i) with s_i = sum_{j in N(i)} x_j, so the
    log pseudo-likelihood sum_i log sigmoid(2 J x_i s_i) is concave in J.
    """
    ones_h, ones_v = couplings(1.0, x.shape)
    xs = x * neighbor_field(x, ones_h, ones_v)
    res = minimize_scalar(lambda J: -log_expit(2.0 * J * xs).sum(),
                          bounds=(0.0, J_max), method="bounded", options={"xatol": 1e-8})
    return float(res.x)


def checkerboard(shape):
    """Boolean mask of the 'black' squares; the grid graph is bipartite in this coloring."""
    H, W = shape
    return (np.add.outer(np.arange(H), np.arange(W)) % 2) == 0
