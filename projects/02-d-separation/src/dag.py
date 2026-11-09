"""A minimal directed acyclic graph plus the graphs studied in this project."""

from __future__ import annotations


class DAG:
    """Directed acyclic graph stored as parent and child adjacency lists.

    Node names can be any hashable objects. Parent lists keep the order in
    which edges were given; conditional probability tables rely on that order.
    """

    def __init__(self, nodes, edges):
        self.nodes = list(nodes)
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("duplicate node names")
        self.parents = {v: [] for v in self.nodes}
        self.children = {v: [] for v in self.nodes}
        for u, v in edges:
            if u not in self.parents or v not in self.parents:
                raise ValueError(f"edge {u}->{v} uses an unknown node")
            if u == v or u in self.parents[v]:
                raise ValueError(f"self-loop or repeated edge {u}->{v}")
            self.parents[v].append(u)
            self.children[u].append(v)
        self.order = self._topological_order()

    @property
    def edges(self):
        return [(u, v) for v in self.nodes for u in self.parents[v]]

    def _topological_order(self):
        indeg = {v: len(self.parents[v]) for v in self.nodes}
        queue = [v for v in self.nodes if indeg[v] == 0]
        order = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for w in self.children[u]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    queue.append(w)
        if len(order) != len(self.nodes):
            raise ValueError("graph has a directed cycle")
        return order

    def neighbors(self, v):
        return self.parents[v] + self.children[v]

    def descendants(self, v):
        """All nodes reachable from v by a directed path, including v."""
        seen, stack = {v}, [v]
        while stack:
            u = stack.pop()
            for w in self.children[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    def ancestors(self, nodes):
        """All nodes with a directed path into the given set, including the set."""
        seen, stack = set(nodes), list(nodes)
        while stack:
            u = stack.pop()
            for w in self.parents[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen


# The nine-node example network. Its factorization is
# p(A) p(L) p(B|A) p(C|A,L) p(E|B) p(G|C,E) p(D|E) p(F|D) p(H|G,F).
EXAMPLE_NODES = ["A", "L", "B", "C", "E", "G", "D", "F", "H"]
EXAMPLE_EDGES = [
    ("A", "B"), ("A", "C"), ("L", "C"), ("B", "E"), ("C", "G"),
    ("E", "G"), ("E", "D"), ("D", "F"), ("G", "H"), ("F", "H"),
]


def example_dag():
    return DAG(EXAMPLE_NODES, EXAMPLE_EDGES)


def lattice_label(r, c):
    """Row r = 1, 2, ... is named A, B, ...; column c is the number."""
    return f"{chr(ord('A') + r - 1)}{c}"


def lattice_dag(n_rows, n_cols):
    """Truncated lattice: node (r, c) points to (r, c+1) and (r+1, c).

    Nodes are (row, column) tuples, 1-based, row 1 at the bottom. This is the
    lower-left n_rows x n_cols corner of an infinite lattice that grows upwards
    and to the right.
    """
    nodes = [(r, c) for r in range(1, n_rows + 1) for c in range(1, n_cols + 1)]
    edges = []
    for r, c in nodes:
        if c < n_cols:
            edges.append(((r, c), (r, c + 1)))
        if r < n_rows:
            edges.append(((r, c), (r + 1, c)))
    return DAG(nodes, edges)


def random_dag(n_nodes, edge_prob, rng):
    """Random DAG: shuffle a node order and add each forward edge with edge_prob."""
    order = rng.permutation(n_nodes)
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                edges.append((int(order[i]), int(order[j])))
    return DAG(list(range(n_nodes)), edges)

