"""Vectorized log-domain loopy belief propagation on the 4-neighbor grid.

Messages are stored as normalized log-probabilities in an array of shape
(4, H, W, 2): logm[d, r, c, s] is log m_{j -> i}(x_i = STATES[s]) for the
message arriving at pixel i = (r, c) from its neighbor j in direction d
(FROM_LEFT, FROM_RIGHT, FROM_ABOVE, FROM_BELOW). Messages from outside the
grid are fixed at the uniform value log(1/2) and do not affect anything.

One iteration updates every message synchronously from the previous ones:

    m_new(x_i) = REDUCE_{x_j} exp(J_ij x_i x_j) exp(h_j x_j) prod_{k in N(j)\\i} m_{k->j}(x_j),

with REDUCE = sum (sum-product) or max (max-product), followed by
normalization (sum_{x_i} m_new(x_i) = 1) and damping

    m <- eta * m_new + (1 - eta) * m_old,

a convex combination of two normalized messages (eta = 1: plain BP).
"""
import numpy as np

from .ising import couplings

STATES = np.array([-1.0, 1.0])  # index 0 <-> x = -1, index 1 <-> x = +1
FROM_LEFT, FROM_RIGHT, FROM_ABOVE, FROM_BELOW = range(4)
LOG_HALF = np.log(0.5)


def init_messages(shape):
    """Uniform messages m(x) = 1/2 for every directed edge."""
    return np.full((4, *shape, 2), LOG_HALF)


def normalize(logm):
    """Normalize log-messages over the last (state) axis so that they sum to one."""
    return logm - np.logaddexp(logm[..., 0], logm[..., 1])[..., None]


def pair_message(cavity, J, mode):
    """Message to x_i given the sender's cavity log-values over x_j.

    cavity: (..., 2) array log[ psi_j(x_j) prod_{k != i} m_{k->j}(x_j) ];
    J: coupling array broadcastable to cavity[..., 0]. Returns normalized log m(x_i).
    """
    reduce = np.logaddexp if mode == "sum" else np.maximum
    a_minus, a_plus = cavity[..., 0], cavity[..., 1]
    to_plus = reduce(J + a_plus, -J + a_minus)    # x_i = +1
    to_minus = reduce(-J + a_plus, J + a_minus)   # x_i = -1
    return normalize(np.stack([to_minus, to_plus], axis=-1))


def log_beliefs(logm, h):
    """Normalized log-beliefs log b(x_i) proportional to h_i x_i + sum_j log m_{j->i}(x_i)."""
    return normalize(h[..., None] * STATES + logm.sum(axis=0))


def update_messages(logm, h, Jh, Jv, mode):
    """One synchronous, undamped update of all messages (returns a new array)."""
    if mode not in ("sum", "max"):
        raise ValueError(f"mode must be 'sum' or 'max', got {mode!r}")
    total = h[..., None] * STATES + logm.sum(axis=0)
    new = np.full_like(logm, LOG_HALF)
    # The sender's cavity excludes the message it received from the target.
    new[FROM_LEFT][:, 1:] = pair_message((total - logm[FROM_RIGHT])[:, :-1], Jh, mode)
    new[FROM_RIGHT][:, :-1] = pair_message((total - logm[FROM_LEFT])[:, 1:], Jh, mode)
    new[FROM_ABOVE][1:, :] = pair_message((total - logm[FROM_BELOW])[:-1, :], Jv, mode)
    new[FROM_BELOW][:-1, :] = pair_message((total - logm[FROM_ABOVE])[1:, :], Jv, mode)
    return new


def damp(logm_new, logm_old, eta):
    """Log of eta * m_new + (1 - eta) * m_old (stays normalized)."""
    if eta == 1.0:
        return logm_new
    return np.logaddexp(np.log(eta) + logm_new, np.log1p(-eta) + logm_old)


def loopy_bp(h, J, mode="sum", eta=0.5, max_iter=200, tol=1e-6, callback=None):
    """Run damped loopy BP; return a dict with log-odds, beliefs, and convergence info.

    h: (H, W) external field (h = beta * y for denoising).
    J: scalar coupling or (Jh, Jv) arrays.
    mode: 'sum' (marginals) or 'max' (max-marginals).
    eta: damping weight on the new message, in (0, 1].
    Stops when the largest change of any log-message is below tol.
    callback(t, log_odds) is called after every iteration t = 1, 2, ...
    """
    if not 0.0 < eta <= 1.0:
        raise ValueError("eta must be in (0, 1]")
    Jh, Jv = couplings(J, h.shape)
    logm = init_messages(h.shape)
    residuals = []
    converged = False
    for t in range(1, max_iter + 1):
        new = damp(update_messages(logm, h, Jh, Jv, mode), logm, eta)
        residuals.append(float(np.max(np.abs(new - logm))))
        logm = new
        if callback is not None:
            lb = log_beliefs(logm, h)
            callback(t, lb[..., 1] - lb[..., 0])
        if residuals[-1] < tol:
            converged = True
            break
    lb = log_beliefs(logm, h)
    return {
        "log_odds": lb[..., 1] - lb[..., 0],
        "p_plus": np.exp(lb[..., 1]),
        "messages": logm,
        "n_iter": t,
        "converged": converged,
        "residuals": np.array(residuals),
    }
