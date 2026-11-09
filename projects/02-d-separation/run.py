"""Regenerate every figure in figures/ and every number in results/.

Usage: python run.py [--quick]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from src.dag import example_dag, lattice_dag, lattice_label, random_dag
from src.dsep import active_trails_bruteforce, d_separated_from, reachable
from src.numeric import ci_deviation, conditional, joint_distribution, random_cpts
from src import plotting

ROOT = Path(__file__).resolve().parent
FIG, RES = ROOT / "figures", ROOT / "results"

EXAMPLE_QUERIES = [("H", ["G", "A"]), ("H", ["G", "F"]), ("A", ["C"])]
EXAMPLE_POS = {
    "A": (0, 4), "L": (2, 4), "B": (-1, 3), "C": (1, 3), "E": (-1, 2),
    "G": (1, 1.6), "D": (-1, 1), "F": (0, 0.3), "H": (1.6, 0.6),
}
LATTICE_QUERY = (3, 2)  # node C2


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small smoke run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--param-draws", type=int, default=200,
                    help="random CPT draws for the nine-node network")
    ap.add_argument("--lattice-max", type=int, default=20,
                    help="largest n for the n x n lattice truncations")
    ap.add_argument("--lattice-bruteforce-max", type=int, default=5,
                    help="largest n for the brute-force cross-check on lattices")
    ap.add_argument("--lattice-numeric-n", type=int, default=4,
                    help="n x n lattice used for the exact numerical check")
    ap.add_argument("--lattice-draws", type=int, default=20)
    ap.add_argument("--lattice-figure-n", type=int, default=6)
    ap.add_argument("--n-dags", type=int, default=4000, help="random DAGs in the stress test")
    ap.add_argument("--queries-per-dag", type=int, default=5)
    ap.add_argument("--min-nodes", type=int, default=4)
    ap.add_argument("--max-nodes", type=int, default=12)
    ap.add_argument("--max-evidence", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="numerical CI threshold on the max absolute deviation")
    args = ap.parse_args()
    if args.quick:
        args.param_draws, args.lattice_max, args.lattice_bruteforce_max = 20, 8, 4
        args.lattice_draws, args.n_dags = 3, 150
    return args


def save_json(obj, name):
    with open(RES / name, "w") as f:
        json.dump(obj, f, indent=2)


def trail_str(trail, dag):
    """Render a trail with arrow directions, e.g. H <- F <- D."""
    out = str(trail[0])
    for u, v in zip(trail[:-1], trail[1:]):
        out += (" -> " if v in dag.children[u] else " <- ") + str(v)
    return out


# ---------------------------------------------------------------------------
# 1. The nine-node network
# ---------------------------------------------------------------------------
def experiment_example(args):
    g = example_dag()
    rng = np.random.default_rng(args.seed)
    joints = [joint_distribution(g, random_cpts(g, rng)) for _ in range(args.param_draws)]
    queries = []
    for x, Z in EXAMPLE_QUERIES:
        hops = reachable(g, [x], Z)
        trails = active_trails_bruteforce(g, x, Z)
        sep_bb = sorted(d_separated_from(g, x, Z))
        sep_bf = sorted(set(g.nodes) - set(trails) - set(Z) - {x})
        targets = [y for y in g.nodes if y != x and y not in Z]
        devs = {y: [ci_deviation(J, g, x, y, Z) for J in joints] for y in targets}
        queries.append({
            "source": x, "evidence": Z,
            "d_separated_bayes_ball": sep_bb,
            "d_separated_bruteforce": sep_bf,
            "d_connected": {
                y: {"bayes_ball_hops": hops[y],
                    "shortest_active_trail": trail_str(trails[y], g)}
                for y in sorted(trails, key=lambda v: (hops[v], v))
            },
            "numeric_min_deviation_over_draws": {y: min(d) for y, d in devs.items()},
            "numeric_median_deviation_over_draws": {y: float(np.median(d))
                                                    for y, d in devs.items()},
            "numeric_max_deviation_over_draws": {y: max(d) for y, d in devs.items()},
            "numeric_says_independent": sorted(
                y for y, d in devs.items() if max(d) < args.tol),
        })

    # A versus G given C through the factorization. With the first CPT draw,
    # p(G | A, C) = sum_{B,E} p(B|A) p(E|B) p(G|C,E); compare with the joint.
    rng = np.random.default_rng(args.seed)
    cpts = random_cpts(g, rng)
    J = joint_distribution(g, cpts)
    pb = lambda b, a: cpts["B"][a] if b else 1 - cpts["B"][a]  # noqa: E731
    pe = lambda e, b: cpts["E"][b] if e else 1 - cpts["E"][b]  # noqa: E731
    factor = {}
    for a in (0, 1):
        for c in (0, 1):
            from_formula = sum(pb(b, a) * pe(e, b) * cpts["G"][c, e]
                               for b in (0, 1) for e in (0, 1))
            from_joint = conditional(J, g, {"G": 1}, {"A": a, "C": c})
            factor[f"A={a},C={c}"] = {"p(G=1|A,C) formula": from_formula,
                                      "p(G=1|A,C) joint": from_joint}
    pa_c = {f"C={c}": conditional(J, g, {"A": 1}, {"C": c}) for c in (0, 1)}
    pa_cg = {f"C={c},G={gg}": conditional(J, g, {"A": 1}, {"C": c, "G": gg})
             for c in (0, 1) for gg in (0, 1)}
    factorization = {"p(G=1|A,C)": factor, "p(A=1|C)": pa_c, "p(A=1|C,G)": pa_cg}

    # Figure: three panels, one per query.
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12, 4.6))
    titles = [r"Query $H$, evidence $\{G, A\}$", r"Query $H$, evidence $\{G, F\}$",
              r"Query $A$, evidence $\{C\}$"]
    show_trail = ["L", None, "G"]
    for ax, q, title, end in zip(axes, queries, titles, show_trail):
        x, Z = q["source"], q["evidence"]
        roles = {v: ("source" if v == x else "evidence" if v in Z else
                     "separated" if v in q["d_separated_bayes_ball"] else "connected")
                 for v in g.nodes}
        hl = []
        if end is not None:
            t = active_trails_bruteforce(g, x, Z)[end]
            hl = list(zip(t[:-1], t[1:]))
        plotting.draw_dag(ax, g, EXAMPLE_POS, roles, highlight=hl)
        ax.set_title(title)
        ax.set_xlim(-1.6, 2.6)
        ax.set_ylim(-0.3, 4.6)
    plotting.role_legend(fig, loc="lower center", ncol=5)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.1, top=0.92, wspace=0.05)
    fig.savefig(FIG / "example_dag_queries.png", dpi=150)
    plotting.plt.close(fig)

    connected = [d for q in queries for y, d in q["numeric_min_deviation_over_draws"].items()
                 if y not in q["d_separated_bayes_ball"]]
    separated = [d for q in queries for y, d in q["numeric_max_deviation_over_draws"].items()
                 if y in q["d_separated_bayes_ball"]]
    out = {
        "param_draws": args.param_draws, "tol": args.tol, "queries": queries,
        "min_deviation_d_connected": min(connected),
        "max_deviation_d_separated": max(separated),
        "factorization_A_G_given_C": factorization,
    }
    save_json(out, "example_dag.json")
    return out


# ---------------------------------------------------------------------------
# 2. Explaining away on the collider A -> C <- L
# ---------------------------------------------------------------------------
def experiment_explaining_away(args):
    g = example_dag()
    cpts = random_cpts(g, np.random.default_rng(args.seed))
    p_a, p_l, leak, strength = 0.2, 0.2, 0.05, 0.8
    cpts["A"] = np.array(p_a)
    cpts["L"] = np.array(p_l)
    # Noisy-OR: each active cause independently triggers C with probability 0.8.
    cpts["C"] = np.array([[1 - (1 - leak) * (1 - strength) ** (a + l) for l in (0, 1)]
                          for a in (0, 1)])
    J = joint_distribution(g, cpts)
    out = {
        "cpt": {"p(A=1)": p_a, "p(L=1)": p_l,
                "p(C=1|A,L)": {f"A={a},L={l}": float(cpts["C"][a, l])
                               for a in (0, 1) for l in (0, 1)}},
        "p(A=1)": conditional(J, g, {"A": 1}),
        "p(A=1|L=1)": conditional(J, g, {"A": 1}, {"L": 1}),
        "p(A=1|C=1)": conditional(J, g, {"A": 1}, {"C": 1}),
        "p(A=1|C=1,L=0)": conditional(J, g, {"A": 1}, {"C": 1, "L": 0}),
        "p(A=1|C=1,L=1)": conditional(J, g, {"A": 1}, {"C": 1, "L": 1}),
        "deviation_A_L_given_nothing": ci_deviation(J, g, "A", "L", []),
        "deviation_A_L_given_C": ci_deviation(J, g, "A", "L", ["C"]),
        "deviation_A_L_given_H": ci_deviation(J, g, "A", "L", ["H"]),
        "d_separated_A_L_given_nothing": "L" in d_separated_from(g, "A", []),
        "d_separated_A_L_given_C": "L" in d_separated_from(g, "A", ["C"]),
        "d_separated_A_L_given_H": "L" in d_separated_from(g, "A", ["H"]),
    }
    save_json(out, "explaining_away.json")
    return out


# ---------------------------------------------------------------------------
# 3. The lattice with shaded diagonal
# ---------------------------------------------------------------------------
def lattice_evidence(n, pattern):
    if pattern == "full_diagonal":
        return [(k, k) for k in range(1, n + 1)]
    return [(k, k) for k in range(1, 4)]  # "first_three": only A1, B2, C3


def experiment_lattice(args):
    rows = []
    for n in range(3, args.lattice_max + 1):
        lat = lattice_dag(n, n)
        for pattern in ("full_diagonal", "first_three"):
            Z = lattice_evidence(n, pattern)
            unshaded = [v for v in lat.nodes if v not in Z]
            hops = reachable(lat, [LATTICE_QUERY], Z)
            sep = d_separated_from(lat, LATTICE_QUERY, Z)
            # All pairs of unshaded nodes, by Bayes ball from every source.
            sep_pairs = sum(len(d_separated_from(lat, s, Z)) for s in unshaded) // 2
            row = {
                "n": n, "evidence": pattern, "unshaded_nodes": len(unshaded),
                "d_connected_to_C2": len(hops) - 1, "d_separated_from_C2": len(sep),
                "max_hops_from_C2": max(hops.values()),
                "d_separated_unshaded_pairs": sep_pairs, "bruteforce_agrees": "",
            }
            if n <= args.lattice_bruteforce_max:
                row["bruteforce_agrees"] = all(
                    set(active_trails_bruteforce(lat, s, Z))
                    == set(reachable(lat, [s], Z)) - {s}
                    for s in unshaded)
            rows.append(row)
    with open(RES / "lattice_truncations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Exact numerical check on a small truncation.
    n = args.lattice_numeric_n
    lat = lattice_dag(n, n)
    rng = np.random.default_rng(args.seed + 1)
    numeric = {}
    joints = [joint_distribution(lat, random_cpts(lat, rng)) for _ in range(args.lattice_draws)]
    for pattern in ("full_diagonal", "first_three"):
        Z = lattice_evidence(n, pattern)
        targets = [v for v in lat.nodes if v not in Z and v != LATTICE_QUERY]
        devs = {lattice_label(*y): min(ci_deviation(J, lat, LATTICE_QUERY, y, Z) for J in joints)
                for y in targets}
        numeric[pattern] = {
            "min_over_draws_deviation_per_node": devs,
            "smallest": min(devs.values()),
            "numeric_says_independent": sorted(k for k, d in devs.items() if d < args.tol),
            "d_separated_from_C2": sorted(lattice_label(*v)
                                          for v in d_separated_from(lat, LATTICE_QUERY, Z)),
        }
    save_json({"n": n, "draws": args.lattice_draws, "tol": args.tol, **numeric},
              "lattice_numeric.json")

    # Figure: hop distance from C2 for both evidence patterns.
    n = args.lattice_figure_n
    lat = lattice_dag(n, n)
    labels = {v: lattice_label(*v) for v in lat.nodes}
    fig = plotting.plt.figure(figsize=(12, 6.2))
    axes = [fig.add_axes((0.01, 0.1, 0.43, 0.8)), fig.add_axes((0.46, 0.1, 0.43, 0.8))]
    cax = fig.add_axes((0.91, 0.2, 0.015, 0.6))
    all_hops = [reachable(lat, [LATTICE_QUERY], lattice_evidence(n, p))
                for p in ("full_diagonal", "first_three")]
    vmax = max(max(h.values()) for h in all_hops)
    names = {"full_diagonal": "whole diagonal observed",
             "first_three": "only A1, B2, C3 observed"}
    for ax, pattern, hops in zip(axes, ("full_diagonal", "first_three"), all_hops):
        Z = lattice_evidence(n, pattern)
        roles = plotting.draw_lattice_hops(ax, lat, LATTICE_QUERY, Z, hops, labels, vmax)
        n_sep = sum(r == "separated" for r in roles.values())
        ax.set_title(f"{names[pattern]}: {n_sep} nodes d-separated from C2")
        ax.set_xlim(0.4, n + 0.6)
        ax.set_ylim(0.4, n + 0.6)
    sm = plotting.plt.cm.ScalarMappable(cmap=plotting.BLUES,
                                        norm=plotting.plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Bayes-ball hops from C2 (shortest active walk)")
    plotting.role_legend(fig, roles=("source", "evidence", "separated"), trail=False,
                         loc="lower center", ncol=3)
    fig.savefig(FIG / "lattice_dconnection.png", dpi=150)
    plotting.plt.close(fig)
    return rows, numeric


# ---------------------------------------------------------------------------
# 4. Random-DAG stress test: Bayes ball vs brute force vs exact numerics
# ---------------------------------------------------------------------------
def experiment_stress(args):
    rng = np.random.default_rng(args.seed + 2)
    n_queries = set_agree = 0
    recs = []  # (separated by Bayes ball, separated by brute force, deviation, trail length)
    for _ in range(args.n_dags):
        n = int(rng.integers(args.min_nodes, args.max_nodes + 1))
        g = random_dag(n, float(rng.uniform(0.15, 0.6)), rng)
        J = joint_distribution(g, random_cpts(g, rng))
        for _ in range(args.queries_per_dag):
            x = int(rng.integers(n))
            others = [v for v in range(n) if v != x]
            k = int(rng.integers(0, min(args.max_evidence, n - 2) + 1))
            Z = [int(v) for v in rng.choice(others, size=k, replace=False)]
            sep_bb = d_separated_from(g, x, Z)
            trails = active_trails_bruteforce(g, x, Z)
            sep_bf = set(g.nodes) - set(trails) - set(Z) - {x}
            n_queries += 1
            set_agree += sep_bb == sep_bf
            for y in others:
                if y in Z:
                    continue
                dev = ci_deviation(J, g, x, y, Z)
                recs.append((y in sep_bb, y in sep_bf, dev,
                             len(trails[y]) - 1 if y in trails else 0))
    sep_bb = np.array([r[0] for r in recs])
    sep_bf = np.array([r[1] for r in recs])
    dev = np.array([r[2] for r in recs])
    tlen = np.array([r[3] for r in recs])
    num_indep = dev < args.tol
    con = ~sep_bb
    by_len = {}
    for L in np.unique(tlen[con]):
        d = dev[con & (tlen == L)]
        by_len[int(L)] = {"pairs": int(d.size), "median": float(np.median(d)),
                          "min": float(d.min())}
    out = {
        "n_dags": args.n_dags, "queries": n_queries, "pairs": int(dev.size),
        "nodes_range": [args.min_nodes, args.max_nodes], "tol": args.tol,
        "query_sets_bayes_ball_equals_bruteforce": set_agree,
        "pairs_bayes_ball_equals_bruteforce": int((sep_bb == sep_bf).sum()),
        "pairs_d_separated": int(sep_bb.sum()), "pairs_d_connected": int(con.sum()),
        "pairs_numeric_agrees_with_dsep": int((num_indep == sep_bb).sum()),
        "max_deviation_d_separated": float(dev[sep_bb].max()),
        "min_deviation_d_connected": float(dev[con].min()),
        "d_connected_below_1e-6": int((dev[con] < 1e-6).sum()),
        "d_connected_below_tol": [{"deviation": float(d), "shortest_trail_length": int(L)}
                                  for d, L in zip(dev[con], tlen[con]) if d < args.tol],
        "deviation_by_trail_length": by_len,
    }
    save_json(out, "stress_test.json")
    plotting.stress_figure(FIG / "stress_test.png", dev[sep_bb], dev[con], tlen[con],
                           args.tol, floor=1e-18)
    return out


def main():
    args = parse_args()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    ex = experiment_example(args)
    for q in ex["queries"]:
        print(f"[example] {q['source']} given {q['evidence']}: d-separated "
              f"{q['d_separated_bayes_ball']} (brute force {q['d_separated_bruteforce']}, "
              f"numeric {q['numeric_says_independent']})")
    ea = experiment_explaining_away(args)
    print(f"[explaining away] p(A=1)={ea['p(A=1)']:.4f}  p(A=1|C=1)={ea['p(A=1|C=1)']:.4f}  "
          f"p(A=1|C=1,L=1)={ea['p(A=1|C=1,L=1)']:.4f}")
    rows, numeric = experiment_lattice(args)
    print(f"[lattice] d-separated from C2 for n=3..{args.lattice_max}: "
          f"{sorted({r['d_separated_from_C2'] for r in rows})}; numeric independent: "
          f"{[numeric[p]['numeric_says_independent'] for p in numeric]}")
    st = experiment_stress(args)
    print(f"[stress] {st['query_sets_bayes_ball_equals_bruteforce']}/{st['queries']} query sets "
          f"agree; numeric agrees on {st['pairs_numeric_agrees_with_dsep']}/{st['pairs']} pairs; "
          f"max dev (sep) {st['max_deviation_d_separated']:.2e}, "
          f"min dev (conn) {st['min_deviation_d_connected']:.2e}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
