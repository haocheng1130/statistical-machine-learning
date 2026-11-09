"""Two independent implementations of d-separation.

1. ``reachable``: the linear-time "Bayes ball" reachability algorithm
   (Koller and Friedman, Probabilistic Graphical Models, Algorithm 3.1).
   It walks over (node, direction) states and handles colliders with the
   ancestor set of the evidence.
2. ``active_trails_bruteforce``: enumerate simple undirected paths (trails)
   and apply the three triple rules directly, using explicit descendant sets.

Both return the nodes that are d-connected to the source given the evidence.
"""

from __future__ import annotations

from collections import deque

UP, DOWN = "up", "down"  # UP: arrived from a child; DOWN: arrived from a parent


def reachable(dag, sources, evidence):
    """Bayes-ball reachability.

    Returns a dict mapping every node that is d-connected to ``sources`` given
    ``evidence`` (and not itself in the evidence) to the length of the shortest
    active walk that reaches it. Sources map to 0.
    """
    sources, evidence = set(sources), set(evidence)
    if sources & evidence:
        raise ValueError("a source node is also in the evidence")
    # Phase 1: the evidence and its ancestors. A collider Y is active exactly
    # when Y is in this set (Y observed or one of its descendants observed).
    anc = dag.ancestors(evidence)

    # Phase 2: breadth-first search over (node, direction) states.
    queue = deque((s, UP, 0) for s in sources)
    visited = set()
    hops = {}
    while queue:
        y, d, k = queue.popleft()
        if (y, d) in visited:
            continue
        visited.add((y, d))
        if y not in evidence and y not in hops:
            hops[y] = k
        if d == UP and y not in evidence:
            # Arrived from a child: y is a chain or fork node, open if unobserved.
            queue.extend((p, UP, k + 1) for p in dag.parents[y])
            queue.extend((c, DOWN, k + 1) for c in dag.children[y])
        elif d == DOWN:
            if y not in evidence:
                # Arrived from a parent: continue the chain downwards.
                queue.extend((c, DOWN, k + 1) for c in dag.children[y])
            if y in anc:
                # y is a collider on this trail and it is active.
                queue.extend((p, UP, k + 1) for p in dag.parents[y])
    return hops


def d_separated_from(dag, x, evidence):
    """Nodes d-separated from x given evidence (evidence and x excluded)."""
    connected = reachable(dag, [x], evidence)
    return set(dag.nodes) - set(connected) - set(evidence) - {x}


def _triple_is_active(dag, a, m, b, evidence, desc_observed):
    """Rule for the consecutive triple a - m - b on a trail."""
    if a in dag.parents[m] and b in dag.parents[m]:  # collider a -> m <- b
        return desc_observed[m]
    return m not in evidence  # chain or fork


def is_active_trail(dag, trail, evidence):
    """True if the given trail (list of adjacent nodes) is active given evidence."""
    evidence = set(evidence)
    for u, v in zip(trail[:-1], trail[1:]):
        if v not in dag.neighbors(u):
            raise ValueError(f"{u} and {v} are not adjacent")
    if len(set(trail)) != len(trail):
        raise ValueError("a trail cannot repeat nodes")
    if trail[0] in evidence or trail[-1] in evidence:
        return False
    desc_observed = {v: bool(dag.descendants(v) & evidence) for v in dag.nodes}
    return all(
        _triple_is_active(dag, a, m, b, evidence, desc_observed)
        for a, m, b in zip(trail[:-2], trail[1:-1], trail[2:])
    )


def active_trails_bruteforce(dag, x, evidence):
    """Enumerate trails from x by depth-first search, triple by triple.

    A trail is extended only while every interior triple is active, so every
    active trail from x is visited exactly once. Returns a dict mapping each
    d-connected node y (y not in the evidence, y != x) to a shortest active
    trail from x to y.
    """
    evidence = set(evidence)
    if x in evidence:
        raise ValueError("the source node is in the evidence")
    desc_observed = {v: bool(dag.descendants(v) & evidence) for v in dag.nodes}
    best = {}

    def extend(path, on_path):
        last = path[-1]
        for w in dag.neighbors(last):
            if w in on_path:
                continue
            if len(path) >= 2 and not _triple_is_active(
                dag, path[-2], last, w, evidence, desc_observed
            ):
                continue
            path.append(w)
            on_path.add(w)
            if w not in evidence and (w not in best or len(path) < len(best[w])):
                best[w] = list(path)
            extend(path, on_path)
            path.pop()
            on_path.remove(w)

    extend([x], {x})
    return best


def d_separated_bruteforce(dag, x, evidence):
    """Nodes d-separated from x given evidence, by trail enumeration."""
    connected = active_trails_bruteforce(dag, x, evidence)
    return set(dag.nodes) - set(connected) - set(evidence) - {x}
