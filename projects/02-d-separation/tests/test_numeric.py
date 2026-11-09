"""Numerical side: exact joints and conditional-independence deviations."""

import itertools

import numpy as np
import pytest

from src.dag import DAG, example_dag, random_dag
from src.dsep import d_separated_from
from src.numeric import (
    ci_deviation, conditional, joint_distribution, marginal, random_cpts,
)

CHAIN = DAG("xmy", [("x", "m"), ("m", "y")])
FORK = DAG("xmy", [("m", "x"), ("m", "y")])
COLLIDER = DAG("xmyd", [("x", "m"), ("y", "m"), ("m", "d")])


def loop_deviation(dag, cpts, x, y, Z):
    """Reference implementation with explicit loops over all states."""
    nodes = dag.nodes
    prob = {}
    for state in itertools.product((0, 1), repeat=len(nodes)):
        s = dict(zip(nodes, state))
        p = 1.0
        for v in nodes:
            t = cpts[v][tuple(s[u] for u in dag.parents[v])]
            p *= t if s[v] else 1 - t
        prob[state] = p

    def P(fix):
        return sum(p for st, p in prob.items()
                   if all(dict(zip(nodes, st))[k] == v for k, v in fix.items()))

    worst = 0.0
    for z in itertools.product((0, 1), repeat=len(Z)):
        zfix = dict(zip(Z, z))
        pz = P(zfix)
        if pz == 0:
            continue
        for a in (0, 1):
            for b in (0, 1):
                pxy = P({**zfix, x: a, y: b}) / pz
                px = P({**zfix, x: a}) / pz
                py = P({**zfix, y: b}) / pz
                worst = max(worst, abs(pxy - px * py))
    return worst


def test_joint_is_normalised_and_factorises():
    g = DAG("ab", [("a", "b")])
    cpts = {"a": np.array(0.3), "b": np.array([0.2, 0.9])}
    J = joint_distribution(g, cpts)
    expected = np.array([[0.7 * 0.8, 0.7 * 0.2], [0.3 * 0.1, 0.3 * 0.9]])
    np.testing.assert_allclose(J, expected)


def test_marginal_axis_order():
    rng = np.random.default_rng(0)
    g = example_dag()
    J = joint_distribution(g, random_cpts(g, rng))
    assert J.sum() == pytest.approx(1.0)
    m = marginal(J, g, ["H", "A"])
    pos = {v: i for i, v in enumerate(g.nodes)}
    ref = np.einsum(J, list(range(9)), [pos["H"], pos["A"]])
    np.testing.assert_allclose(m, ref)


def test_deviation_matches_loop_reference():
    rng = np.random.default_rng(1)
    for _ in range(10):
        g = random_dag(5, 0.5, rng)
        cpts = random_cpts(g, rng)
        J = joint_distribution(g, cpts)
        x, y, *Z = [int(v) for v in rng.choice(5, size=4, replace=False)]
        Z = Z[: int(rng.integers(0, 3))]
        assert ci_deviation(J, g, x, y, Z) == pytest.approx(
            loop_deviation(g, cpts, x, y, Z), abs=1e-14)


@pytest.mark.parametrize("dag", [CHAIN, FORK])
def test_chain_fork_numeric(dag):
    J = joint_distribution(dag, random_cpts(dag, np.random.default_rng(2)))
    assert ci_deviation(J, dag, "x", "y", []) > 1e-3
    assert ci_deviation(J, dag, "x", "y", ["m"]) < 1e-14


def test_collider_numeric():
    J = joint_distribution(COLLIDER, random_cpts(COLLIDER, np.random.default_rng(3)))
    assert ci_deviation(J, COLLIDER, "x", "y", []) < 1e-14
    assert ci_deviation(J, COLLIDER, "x", "y", ["m"]) > 1e-3
    assert ci_deviation(J, COLLIDER, "x", "y", ["d"]) > 1e-4


def test_explaining_away_closed_form():
    # x -> m <- y with a deterministic OR: p(x=1 | m=1, y=1) falls back to p(x=1).
    g = DAG("xym", [("x", "m"), ("y", "m")])
    cpts = {"x": np.array(0.3), "y": np.array(0.4),
            "m": np.array([[0.0, 1.0], [1.0, 1.0]])}
    J = joint_distribution(g, cpts)
    p_m1 = 1 - 0.7 * 0.6
    assert conditional(J, g, {"x": 1}, {"m": 1}) == pytest.approx(0.3 / p_m1)
    assert conditional(J, g, {"x": 1}, {"m": 1, "y": 1}) == pytest.approx(0.3)
    assert conditional(J, g, {"x": 1}, {"m": 1, "y": 0}) == pytest.approx(1.0)


def test_numeric_agrees_with_dseparation_on_random_dags():
    rng = np.random.default_rng(4)
    for _ in range(60):
        n = int(rng.integers(3, 8))
        g = random_dag(n, 0.45, rng)
        J = joint_distribution(g, random_cpts(g, rng))
        x = int(rng.integers(n))
        others = [v for v in range(n) if v != x]
        Z = [int(v) for v in rng.choice(others, size=int(rng.integers(0, 3)), replace=False)]
        sep = d_separated_from(g, x, Z)
        for y in others:
            if y in Z:
                continue
            dev = ci_deviation(J, g, x, y, Z)
            if y in sep:
                assert dev < 1e-13
            else:
                assert dev > 1e-12


def test_example_factorization_identities():
    # p(C|A) = sum_L p(L) p(C|A,L) and p(G|A,C) = sum_{B,E} p(B|A) p(E|B) p(G|C,E).
    g = example_dag()
    cpts = random_cpts(g, np.random.default_rng(5))
    J = joint_distribution(g, cpts)
    assert g.parents["C"] == ["A", "L"] and g.parents["G"] == ["C", "E"]
    bern = lambda p, v: p if v else 1 - p  # noqa: E731
    for a in (0, 1):
        p_c = sum(bern(cpts["L"], l) * cpts["C"][a, l] for l in (0, 1))
        assert conditional(J, g, {"C": 1}, {"A": a}) == pytest.approx(p_c, abs=1e-14)
        for c in (0, 1):
            p_g = sum(bern(cpts["B"][a], b) * bern(cpts["E"][b], e) * cpts["G"][c, e]
                      for b in (0, 1) for e in (0, 1))
            assert conditional(J, g, {"G": 1}, {"A": a, "C": c}) == pytest.approx(p_g, abs=1e-14)
