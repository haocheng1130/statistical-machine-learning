"""Matplotlib drawings of DAGs, lattices and the stress-test results."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COLORS = {
    "source": "#eb6834",
    "evidence": "#52514e",
    "connected": "#2a78d6",
    "separated": "#1baf7a",
    "edge": "#898781",
    "ink": "#0b0b0b",
    "grid": "#e1e0d9",
}
ROLE_LABELS = {
    "source": "query node",
    "evidence": "observed (evidence)",
    "connected": "d-connected to query",
    "separated": "d-separated from query",
}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95"]
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})


def _arrow(ax, p, q, radius_pt, color, lw, zorder=1):
    ax.annotate(
        "", xy=q, xytext=p, zorder=zorder,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12,
                        shrinkA=radius_pt, shrinkB=radius_pt + 1),
    )


def draw_dag(ax, dag, pos, roles, labels=None, highlight=(), node_size=900,
             node_colors=None, font_size=10):
    """Draw a DAG with nodes coloured by role.

    roles: dict node -> one of COLORS' role keys; node_colors overrides the fill.
    highlight: list of (u, v) node pairs (either direction) drawn as a thick
    trail in the source colour.
    """
    radius_pt = np.sqrt(node_size) / 2
    hl = {frozenset(e) for e in highlight}
    for u, v in dag.edges:
        on = frozenset((u, v)) in hl
        _arrow(ax, pos[u], pos[v], radius_pt,
               COLORS["source"] if on else COLORS["edge"], 2.6 if on else 1.1,
               zorder=2 if on else 1)
    node_colors = node_colors or {}
    for v in dag.nodes:
        fill = node_colors.get(v, COLORS[roles[v]])
        ax.scatter(*pos[v], s=node_size, color=fill, edgecolor="white",
                   linewidth=1.5, zorder=3)
        text = labels[v] if labels else str(v)
        ink = _ink_on(fill) if v in node_colors else "white"
        ax.text(*pos[v], text, ha="center", va="center", color=ink,
                fontsize=font_size, fontweight="bold", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")


def _ink_on(rgba):
    r, g, b = rgba[:3]
    return COLORS["ink"] if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def role_legend(ax_or_fig, roles=("source", "evidence", "connected", "separated"),
                trail=True, **kw):
    handles = [Line2D([], [], marker="o", ls="", ms=10, color=COLORS[r],
                      label=ROLE_LABELS[r]) for r in roles]
    if trail:
        handles.append(Line2D([], [], color=COLORS["source"], lw=2.6,
                              label="shortest active trail"))
    return ax_or_fig.legend(handles=handles, frameon=False, **kw)


def lattice_positions(dag):
    return {(r, c): (c, r) for r, c in dag.nodes}


def draw_lattice_hops(ax, dag, query, evidence, hops, labels, vmax):
    """Lattice with unobserved nodes coloured by Bayes-ball hop distance."""
    pos = lattice_positions(dag)
    roles, fills = {}, {}
    for v in dag.nodes:
        if v == query:
            roles[v] = "source"
        elif v in evidence:
            roles[v] = "evidence"
        elif v in hops:
            roles[v] = "connected"
            fills[v] = BLUES(hops[v] / vmax)
        else:
            roles[v] = "separated"
    draw_dag(ax, dag, pos, roles, labels=labels, node_size=560,
             node_colors=fills, font_size=7.5)
    return roles


def stress_figure(path, dev_sep, dev_con, trail_len, tol, floor):
    """Histogram of deviations by d-separation status, and decay with trail length."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    lo = np.log10(np.maximum(dev_sep, floor))
    hi = np.log10(np.maximum(dev_con, floor))
    bins = np.linspace(np.log10(floor) - 0.25, 0.25, 70)
    ax.hist(lo, bins=bins, color=COLORS["separated"], label=f"d-separated pairs (n={len(lo):,})")
    ax.hist(hi, bins=bins, color=COLORS["connected"], label=f"d-connected pairs (n={len(hi):,})")
    ax.axvline(np.log10(tol), color=COLORS["ink"], lw=1, ls="--")
    ax.text(np.log10(tol) - 0.3, 1.5, f"threshold {tol:g}", rotation=90,
            ha="right", va="bottom", fontsize=9, color=COLORS["ink"])
    ax.set_xlabel(r"$\log_{10}\ \max\ |p(x,y\,|\,z) - p(x\,|\,z)\,p(y\,|\,z)|$")
    ax.set_ylabel("number of tested triples (x, y, Z)")
    ax.set_yscale("log")
    ax.set_title("Exact deviation from conditional independence")
    ax.legend(loc="upper center", framealpha=1, facecolor="white", edgecolor="none")

    ax = axes[1]
    lengths = np.unique(trail_len)
    data = [np.log10(dev_con[trail_len == k]) for k in lengths]
    ax.boxplot(data, positions=lengths, widths=0.6, patch_artist=True,
               flierprops=dict(marker="o", ms=2.5, mec="none", alpha=0.5,
                               mfc=COLORS["connected"]),
               boxprops=dict(facecolor="#cde2fb", edgecolor=COLORS["connected"]),
               medianprops=dict(color=COLORS["connected"], lw=2),
               whiskerprops=dict(color=COLORS["connected"]),
               capprops=dict(color=COLORS["connected"]))
    ax.axhline(np.log10(tol), color=COLORS["ink"], lw=1, ls="--")
    ax.text(lengths[-1] + 0.45, np.log10(tol) + 0.15, f"threshold {tol:g}",
            ha="right", va="bottom", fontsize=9, color=COLORS["ink"])
    ax.set_xlabel("length of the shortest active trail (edges)")
    ax.set_ylabel(r"$\log_{10}$ deviation (d-connected pairs)")
    ax.set_title("Dependence weakens along longer trails")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
