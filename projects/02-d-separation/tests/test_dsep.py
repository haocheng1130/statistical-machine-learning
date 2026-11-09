"""Graph-side checks: canonical triples, Bayes ball vs trail enumeration."""

import numpy as np
import pytest

from src.dag import DAG, example_dag, lattice_dag, random_dag
from src.dsep import (
    active_trails_bruteforce, d_separated_bruteforce, d_separated_from,
    is_active_trail, reachable,
)

BOTH = [d_separated_from, d_separated_bruteforce]

CHAIN = DAG("xmy", [("x", "m"), ("m", "y")])
FORK = DAG("xmy", [("m", "x"), ("m", "y")])
COLLIDER = DAG("xmyd", [("x", "m"), ("y", "m"), ("m", "d")])
# Collider whose effect is only seen through a grandchild.
COLLIDER_DEEP = DAG("xmyde", [("x", "m"), ("y", "m"), ("m", "d"), ("d", "e")])


@pytest.mark.parametrize("sep", BOTH)
@pytest.mark.parametrize("dag", [CHAIN, FORK])
def test_chain_and_fork(sep, dag):
    assert "y" not in sep(dag, "x", [])  # open when m is hidden
    assert "y" in sep(dag, "x", ["m"])  # blocked when m is observed


@pytest.mark.parametrize("sep", BOTH)
def test_collider(sep):
    assert "y" in sep(COLLIDER, "x", [])  # blocked when m is hidden
    assert "y" not in sep(COLLIDER, "x", ["m"])  # observing m opens it
    assert "y" not in sep(COLLIDER, "x", ["d"])  # so does observing a child


@pytest.mark.parametrize("sep", BOTH)
def test_collider_observed_grandchild(sep):
    assert "y" in sep(COLLIDER_DEEP, "x", [])
    assert "y" not in sep(COLLIDER_DEEP, "x", ["e"])
    # Observing d blocks x -> m -> d (chain) but the collider at m is open.
    assert sep(COLLIDER_DEEP, "x", ["d"]) == {"e"}


def test_cycle_rejected():
    with pytest.raises(ValueError):
        DAG("ab", [("a", "b"), ("b", "a")])


def test_is_active_trail_example():
    g = example_dag()
    trail = ["H", "F", "D", "E", "G", "C", "L"]  # passes the collider E -> G <- C
    assert is_active_trail(g, trail, ["G", "A"])
    assert not is_active_trail(g, trail, ["A"])  # G hidden and H not observed
    assert not is_active_trail(g, trail, ["G", "F"])  # F blocks the chain


def test_example_dag_queries():
    g = example_dag()
    for sep in BOTH:
        assert sep(g, "H", ["G", "A"]) == set()
        assert sep(g, "H", ["G", "F"]) == {"A", "B", "C", "D", "E", "L"}
        assert "G" not in sep(g, "A", ["C"])
        assert "L" not in sep(g, "A", ["C"])  # explaining away through C
        assert "L" in sep(g, "A", [])


def test_shortest_trails_are_active():
    g = example_dag()
    trails = active_trails_bruteforce(g, "H", ["G", "A"])
    for y, t in trails.items():
        assert t[0] == "H" and t[-1] == y
        assert is_active_trail(g, t, ["G", "A"])


def test_bayes_ball_matches_bruteforce_on_random_dags():
    rng = np.random.default_rng(123)
    for _ in range(400):
        n = int(rng.integers(3, 10))
        g = random_dag(n, float(rng.uniform(0.1, 0.7)), rng)
        x = int(rng.integers(n))
        others = [v for v in range(n) if v != x]
        k = int(rng.integers(0, min(4, n - 1) + 1))
        Z = [int(v) for v in rng.choice(others, size=k, replace=False)]
        assert d_separated_from(g, x, Z) == d_separated_bruteforce(g, x, Z)


def test_dseparation_is_symmetric():
    rng = np.random.default_rng(7)
    for _ in range(100):
        n = int(rng.integers(3, 9))
        g = random_dag(n, 0.4, rng)
        Z = [int(v) for v in rng.choice(n, size=int(rng.integers(0, 3)), replace=False)]
        free = [v for v in range(n) if v not in Z]
        for x in free:
            for y in d_separated_from(g, x, Z):
                assert x in d_separated_from(g, y, Z)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_lattice_bayes_ball_matches_bruteforce(n):
    lat = lattice_dag(n, n)
    Z = [(k, k) for k in range(1, n + 1)]
    for s in lat.nodes:
        if s in Z:
            continue
        assert set(active_trails_bruteforce(lat, s, Z)) == set(reachable(lat, [s], Z)) - {s}
    assert d_separated_from(lat, (3, 2), Z) == set()
