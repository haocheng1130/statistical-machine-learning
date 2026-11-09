"""Exact references: brute-force enumeration (tiny grids) and graph-cut MAP (any grid).

Graph cut: for J_ij >= 0 the MAP problem of the grid Ising model is a minimum
s-t cut (Greig, Porteous and Seheult, 1989). Pixels on the source side get
x = +1. Up to a constant, the energy -log p~(x) equals

    sum_i 2 max(h_i, 0) [x_i = -1] + 2 max(-h_i, 0) [x_i = +1] + sum_E 2 J_ij [x_i != x_j],

so edge s->i has capacity 2 max(h_i, 0), edge i->t has 2 max(-h_i, 0), and every
grid edge has capacity 2 J_ij in both directions. scipy's max-flow needs integer
capacities, so all capacities are multiplied by a common scale and rounded.
"""
import functools
import itertools

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from .ising import couplings, log_potential


@functools.lru_cache(maxsize=8)
def all_states(shape):
    """All 2^(H*W) configurations as a read-only array of shape (2^n, H, W)."""
    n = shape[0] * shape[1]
    if n > 20:
        raise ValueError("brute force is limited to 20 pixels")
    X = np.array(list(itertools.product([-1.0, 1.0], repeat=n))).reshape(-1, *shape)
    X.flags.writeable = False
    return X


def brute_force(h, J):
    """Exact P(x_i = +1), max-marginal log-odds, MAP state, and log Z by enumeration."""
    Jh, Jv = couplings(J, h.shape)
    X = all_states(tuple(h.shape))
    logp = log_potential(X, h, Jh, Jv)
    log_z = np.logaddexp.reduce(logp)
    prob = np.exp(logp - log_z)
    p_plus = np.tensordot(prob, (X > 0).astype(float), axes=1)
    best_plus = np.where(X > 0, logp[:, None, None], -np.inf).max(axis=0)
    best_minus = np.where(X < 0, logp[:, None, None], -np.inf).max(axis=0)
    return {
        "p_plus": p_plus,
        "max_log_odds": best_plus - best_minus,
        "map": X[np.argmax(logp)],
        "log_z": log_z,
    }


def graph_cut_map(h, J, max_scale=1e6):
    """Exact MAP configuration of the grid Ising model with nonnegative couplings."""
    Jh, Jv = couplings(J, h.shape)
    if (Jh < 0).any() or (Jv < 0).any():
        raise ValueError("graph-cut MAP requires nonnegative couplings")
    H, W = h.shape
    n = H * W
    src, snk = n, n + 1
    idx = np.arange(n).reshape(H, W)
    cap_src = 2.0 * np.maximum(h, 0.0).ravel()
    cap_snk = 2.0 * np.maximum(-h, 0.0).ravel()
    # The flow value is bounded by the total terminal capacity; keep it inside int32.
    bound = max(cap_src.sum(), cap_snk.sum(), 2.0 * max(Jh.max(initial=0), Jv.max(initial=0)), 1e-12)
    scale = min(max_scale, np.floor((2**31 - 1) / (2.0 * bound)))

    heads = [np.full(n, src), idx.ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel(),
             idx[:-1].ravel(), idx[1:].ravel()]
    tails = [idx.ravel(), np.full(n, snk), idx[:, 1:].ravel(), idx[:, :-1].ravel(),
             idx[1:].ravel(), idx[:-1].ravel()]
    caps = [cap_src, cap_snk, 2 * Jh.ravel(), 2 * Jh.ravel(), 2 * Jv.ravel(), 2 * Jv.ravel()]
    rows, cols = np.concatenate(heads), np.concatenate(tails)
    data = np.rint(np.concatenate(caps) * scale).astype(np.int32)
    keep = data > 0
    capacity = csr_array((data[keep], (rows[keep], cols[keep])), shape=(n + 2, n + 2))

    flow = maximum_flow(capacity, src, snk).flow
    residual = (capacity - flow).tocsr()
    residual.data[residual.data < 0] = 0
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, src, directed=True, return_predecessors=False)
    x = -np.ones(n + 2)
    x[reachable] = 1.0
    return x[:n].reshape(H, W)
