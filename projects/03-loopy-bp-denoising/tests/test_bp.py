"""Loopy BP against brute-force enumeration and a naive message-passing loop."""
import numpy as np
import pytest

from src.bp import (FROM_ABOVE, FROM_BELOW, FROM_LEFT, FROM_RIGHT, LOG_HALF, damp,
                    init_messages, loopy_bp, pair_message, update_messages)
from src.exact import brute_force
from src.ising import couplings


def comb_tree(shape, rng, scale=1.0):
    """Random couplings on a spanning tree of the grid: all vertical edges plus row 0."""
    H, W = shape
    Jh = np.zeros((H, W - 1))
    Jh[0] = rng.normal(scale=scale, size=W - 1)
    Jv = rng.normal(scale=scale, size=(H - 1, W))
    return Jh, Jv


@pytest.mark.parametrize("eta", [1.0, 0.5])
def test_sum_product_exact_on_tree(eta):
    rng = np.random.default_rng(1)
    h = rng.normal(size=(3, 4))
    J = comb_tree(h.shape, rng)
    exact = brute_force(h, J)
    out = loopy_bp(h, J, mode="sum", eta=eta, max_iter=2000, tol=1e-13)
    assert out["converged"]
    np.testing.assert_allclose(out["p_plus"], exact["p_plus"], atol=1e-10)


def test_max_product_map_on_chain():
    rng = np.random.default_rng(2)
    for _ in range(10):
        h = rng.normal(size=(1, 12))
        J = (rng.normal(size=(1, 11)), np.zeros((0, 12)))  # mixed-sign couplings
        exact = brute_force(h, J)
        out = loopy_bp(h, J, mode="max", eta=1.0, max_iter=100, tol=1e-13)
        # Max-product beliefs are the exact max-marginals; their argmax is the MAP.
        np.testing.assert_allclose(out["log_odds"], exact["max_log_odds"], atol=1e-10)
        np.testing.assert_array_equal(np.sign(out["log_odds"]), exact["map"])


def test_max_product_max_marginals_on_tree():
    rng = np.random.default_rng(3)
    h = rng.normal(size=(3, 4))
    J = comb_tree(h.shape, rng)
    exact = brute_force(h, J)
    out = loopy_bp(h, J, mode="max", eta=0.7, max_iter=2000, tol=1e-13)
    np.testing.assert_allclose(out["log_odds"], exact["max_log_odds"], atol=1e-9)


def test_loopy_3x3_close_to_exact_for_weak_coupling():
    rng = np.random.default_rng(4)
    errors = []
    for _ in range(20):
        h = rng.normal(scale=0.5, size=(3, 3))
        exact = brute_force(h, 0.3)
        out = loopy_bp(h, 0.3, mode="sum", eta=0.5, max_iter=2000, tol=1e-12)
        assert out["converged"]
        errors.append(np.abs(out["p_plus"] - exact["p_plus"]).max())
    assert max(errors) < 0.02     # close ...
    assert max(errors) > 1e-4     # ... but not exact: the grid has cycles


def naive_update(logm, h, J, mode):
    """Message update written directly from the BP equation with explicit neighbor lists."""
    H, W = h.shape
    Jh, Jv = couplings(J, h.shape)
    states = (-1.0, 1.0)
    # incoming[(i, j)] = log m_{j -> i}; direction index of j as seen from i
    offsets = {FROM_LEFT: (0, -1), FROM_RIGHT: (0, 1), FROM_ABOVE: (-1, 0), FROM_BELOW: (1, 0)}

    def coupling(a, b):
        (r1, c1), (r2, c2) = sorted([a, b])
        return Jh[r1, c1] if r1 == r2 else Jv[r1, c1]

    def neighbors(node):
        r, c = node
        return {d: (r + dr, c + dc) for d, (dr, dc) in offsets.items()
                if 0 <= r + dr < H and 0 <= c + dc < W}

    new = np.full_like(logm, LOG_HALF)
    for r in range(H):
        for c in range(W):
            i = (r, c)
            for d, j in neighbors(i).items():
                vals = []
                for xi in states:
                    terms = []
                    for sj, xj in enumerate(states):
                        t = h[j] * xj + coupling(i, j) * xi * xj
                        for dk, k in neighbors(j).items():
                            if k != i:
                                t += logm[dk, j[0], j[1], sj]
                        terms.append(t)
                    vals.append(np.logaddexp(*terms) if mode == "sum" else max(terms))
                vals = np.array(vals)
                new[d, r, c] = vals - np.logaddexp(*vals)
    return new


@pytest.mark.parametrize("mode", ["sum", "max"])
def test_vectorized_update_matches_naive_loop(mode):
    rng = np.random.default_rng(5)
    h = rng.normal(size=(3, 4))
    J = (rng.normal(size=(3, 3)), rng.normal(size=(2, 4)))
    # random normalized messages (boundary entries stay uniform)
    logm = init_messages(h.shape)
    raw = rng.normal(size=logm.shape)
    raw -= np.logaddexp(raw[..., 0], raw[..., 1])[..., None]
    logm[FROM_LEFT][:, 1:] = raw[FROM_LEFT][:, 1:]
    logm[FROM_RIGHT][:, :-1] = raw[FROM_RIGHT][:, :-1]
    logm[FROM_ABOVE][1:] = raw[FROM_ABOVE][1:]
    logm[FROM_BELOW][:-1] = raw[FROM_BELOW][:-1]
    Jh, Jv = couplings(J, h.shape)
    np.testing.assert_allclose(update_messages(logm, h, Jh, Jv, mode),
                               naive_update(logm, h, J, mode), atol=1e-12)


@pytest.mark.parametrize("mode", ["sum", "max"])
def test_pair_message_formula(mode):
    rng = np.random.default_rng(6)
    cavity = rng.normal(size=(5, 2))
    J = rng.normal(size=5)
    out = pair_message(cavity, J, mode)
    for n in range(5):
        raw = []
        for xi in (-1.0, 1.0):
            terms = [J[n] * xi * xj + cavity[n, sj] for sj, xj in enumerate((-1.0, 1.0))]
            raw.append(np.log(np.exp(terms).sum()) if mode == "sum" else max(terms))
        expected = np.array(raw) - np.log(np.exp(raw).sum())
        np.testing.assert_allclose(out[n], expected, atol=1e-12)


@pytest.mark.parametrize("mode", ["sum", "max"])
def test_messages_are_normalized(mode):
    rng = np.random.default_rng(7)
    h = rng.normal(scale=3.0, size=(5, 6))
    Jh, Jv = couplings(2.0, h.shape)
    logm = init_messages(h.shape)
    for _ in range(5):
        new = update_messages(logm, h, Jh, Jv, mode)
        np.testing.assert_allclose(np.exp(new).sum(axis=-1), 1.0, atol=1e-12)
        logm = damp(new, logm, 0.3)
        np.testing.assert_allclose(np.exp(logm).sum(axis=-1), 1.0, atol=1e-12)


def test_damping_is_convex_combination_of_messages():
    rng = np.random.default_rng(8)
    a, b = rng.dirichlet([1, 1], size=10), rng.dirichlet([1, 1], size=10)
    for eta in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(np.exp(damp(np.log(a), np.log(b), eta)),
                                   eta * a + (1 - eta) * b, atol=1e-14)
    np.testing.assert_array_equal(damp(np.log(a), np.log(b), 1.0), np.log(a))


def test_damping_keeps_fixed_points():
    rng = np.random.default_rng(9)
    h = rng.normal(size=(4, 4))
    Jh, Jv = couplings(0.4, h.shape)
    fixed = loopy_bp(h, 0.4, mode="sum", eta=1.0, max_iter=5000, tol=1e-14)["messages"]
    for eta in (0.2, 0.7):
        again = damp(update_messages(fixed, h, Jh, Jv, "sum"), fixed, eta)
        np.testing.assert_allclose(again, fixed, atol=1e-12)
    slow = loopy_bp(h, 0.4, mode="sum", eta=0.2, max_iter=5000, tol=1e-14)
    np.testing.assert_allclose(slow["messages"], fixed, atol=1e-10)
