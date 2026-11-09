"""Naive mean-field inference for the grid Ising model.

q(x) = prod_i q_i(x_i) with means mu_i = E_q[x_i]. The coordinate-ascent update
is mu_i <- tanh(h_i + sum_{j in N(i)} J_ij mu_j). Pixels of one checkerboard
color share no edges, so updating a whole color at once is still exact
coordinate ascent and the ELBO never decreases.
"""
import numpy as np

from .ising import checkerboard, couplings, neighbor_field


def elbo(mu, h, Jh, Jv):
    """E_q[log p~(x)] + H(q) for the factorized q with means mu (a lower bound on log Z)."""
    expected = (h * mu).sum() + (Jh * mu[:, :-1] * mu[:, 1:]).sum() + (Jv * mu[:-1] * mu[1:]).sum()
    p = np.clip((1.0 + mu) / 2.0, 1e-300, 1.0)
    q = np.clip((1.0 - mu) / 2.0, 1e-300, 1.0)
    entropy = -(p * np.log(p) + q * np.log(q)).sum()
    return expected + entropy


def mean_field(h, J, max_iter=500, tol=1e-6, track_elbo=False):
    """Checkerboard coordinate ascent from mu = tanh(h); returns means and convergence info."""
    Jh, Jv = couplings(J, h.shape)
    black = checkerboard(h.shape)
    mu = np.tanh(h)
    elbos = [elbo(mu, h, Jh, Jv)] if track_elbo else None
    converged = False
    for t in range(1, max_iter + 1):
        old = mu.copy()
        for mask in (black, ~black):
            mu = np.where(mask, np.tanh(h + neighbor_field(mu, Jh, Jv)), mu)
        if track_elbo:
            elbos.append(elbo(mu, h, Jh, Jv))
        if np.max(np.abs(mu - old)) < tol:
            converged = True
            break
    return {"mean": mu, "n_iter": t, "converged": converged,
            "elbo": None if elbos is None else np.array(elbos)}
