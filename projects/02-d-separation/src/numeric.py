"""Exact numerical independence tests for binary Bayesian networks.

The joint distribution is built by enumerating all 2^n states, so these
routines are meant for networks with at most ~20 binary nodes.
"""

from __future__ import annotations

import numpy as np


def random_cpts(dag, rng, low=0.0, high=1.0):
    """Random conditional probability tables for binary nodes.

    ``cpts[v][pa_values]`` is p(v = 1 | parents = pa_values), with the parents
    in the order of ``dag.parents[v]``; entries are Uniform(low, high).
    """
    return {
        v: rng.uniform(low, high, size=(2,) * len(dag.parents[v]))
        for v in dag.nodes
    }


def all_states(n):
    """All 2^n binary vectors in C order (first node = most significant bit)."""
    idx = np.arange(2**n)[:, None]
    return (idx >> np.arange(n - 1, -1, -1)) & 1


def joint_distribution(dag, cpts):
    """Exact joint p(x_1, ..., x_n) as an array of shape (2,) * n.

    Axis i corresponds to ``dag.nodes[i]``; the joint is the product of the
    local conditionals p(x_v | x_parents(v)) evaluated at every state.
    """
    n = len(dag.nodes)
    pos = {v: i for i, v in enumerate(dag.nodes)}
    states = all_states(n)
    joint = np.ones(2**n)
    for v in dag.nodes:
        pa_cols = [pos[u] for u in dag.parents[v]]
        theta = cpts[v][tuple(states[:, pa_cols].T)]  # p(v = 1 | parents)
        joint *= np.where(states[:, pos[v]] == 1, theta, 1.0 - theta)
    return joint.reshape((2,) * n)


def marginal(joint, dag, keep):
    """Marginal over the nodes in ``keep``, with axes in the order of ``keep``."""
    if len(set(keep)) != len(keep):
        raise ValueError("a node appears twice")
    pos = {v: i for i, v in enumerate(dag.nodes)}
    axes = [pos[v] for v in keep]
    drop = tuple(i for i in range(joint.ndim) if i not in axes)
    m = joint.sum(axis=drop)
    # After summing, the remaining axes are in increasing original order.
    remaining = sorted(axes)
    return np.transpose(m, [remaining.index(a) for a in axes])


def ci_deviation(joint, dag, x, y, evidence):
    """max over values of |p(x, y | z) - p(x | z) p(y | z)|.

    The maximum runs over all values of x, y and of the evidence z with
    p(z) > 0. It is zero (up to rounding) exactly when x and y are
    conditionally independent given the evidence in this distribution.
    """
    evidence = list(evidence)
    pxyz = marginal(joint, dag, [x, y] + evidence)
    pz = pxyz.sum(axis=(0, 1))
    pxz = pxyz.sum(axis=1)
    pyz = pxyz.sum(axis=0)
    ok = pz > 0
    pz_safe = np.where(ok, pz, 1.0)
    dev = pxyz / pz_safe - (pxz / pz_safe)[:, None] * (pyz / pz_safe)[None, :]
    return float(np.max(np.abs(dev[:, :, ok])))


def conditional(joint, dag, query, given=None):
    """p(query | given) for dicts {node: value}."""
    given = dict(given or {})
    nodes = list(query) + [v for v in given if v not in query]
    m = marginal(joint, dag, nodes)
    num = m[tuple({**given, **query}[v] for v in nodes)]
    rest = [v for v in nodes if v in given]
    den = marginal(joint, dag, rest)[tuple(given[v] for v in rest)] if rest else 1.0
    return float(num / den)
