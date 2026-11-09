"""Figures for the report (matplotlib, Agg backend)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COLORS = {  # one fixed color per method, used in every figure
    "hmc": "#2a78d6",
    "mala": "#1baf7a",
    "mfvi": "#eb6834",
    "fcvi": "#6250d6",
    "mala_omitted": "#e34948",
    "svi_short": "#eda100",
}
LABELS = {
    "hmc": "HMC",
    "mala": "MALA",
    "mfvi": "Mean-field VI",
    "fcvi": "Full-cov. VI",
    "mala_omitted": "MALA, 1/(4η) omitted",
    "svi_short": "Mean-field VI, 200 SGD steps",
}
MASSES = (0.5, 0.9, 0.99)  # highest-density regions drawn as contours

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _hpd_levels(weights, masses=MASSES):
    """Density thresholds whose super-level sets contain the given probability masses."""
    w = np.sort(weights.ravel())[::-1]
    cum = np.cumsum(w)
    return sorted(w[np.searchsorted(cum, m)] for m in masses)


def _gaussian_ellipse(ax, mean, cov, color, style, label=None):
    """Draw the HPD ellipses of N(mean, cov) for MASSES (radius sqrt(-2 log(1 - m)))."""
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(t), np.sin(t)])
    for k, m in enumerate(MASSES):
        r = math.sqrt(-2 * math.log(1 - m))
        pts = mean[:, None] + vecs @ (r * np.sqrt(vals)[:, None] * circle)
        ax.plot(pts[0], pts[1], color=color, ls=style, lw=1.4, label=label if k == 0 else None)


def _equal_skill(ax, lim):
    ax.plot(lim, lim, color=INK_2, lw=0.8, ls=":", label="equal skill")


def toy_posteriors(path, cases):
    """cases: list of dicts with name, post (GridPosterior), hmc draws (S, 2), mfvi/fcvi (mean, cov)."""
    fig, axes = plt.subplots(2, len(cases), figsize=(3.3 * len(cases), 6.6), sharex=True, sharey=True)
    lim = (-3.5, 3.5)
    for col, case in enumerate(cases):
        post = case["post"]
        za, zb, w = post.za.numpy(), post.zb.numpy(), post.weights.numpy()
        levels = _hpd_levels(w)
        for row in range(2):
            ax = axes[row, col]
            ax.contour(za, zb, w, levels=levels, colors=INK, linewidths=1.0)
            _equal_skill(ax, lim)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
        draws = case["hmc"]
        axes[0, col].scatter(draws[:, 0], draws[:, 1], s=2, color=COLORS["hmc"], alpha=0.35,
                             lw=0, label="HMC draws")
        axes[0, col].set_title(f"A–B record {case['name']}: exact + HMC")
        for key, style in (("mfvi", "--"), ("fcvi", "-")):
            mean, cov = case[key]
            _gaussian_ellipse(axes[1, col], mean, cov, COLORS[key], style, LABELS[key])
        axes[1, col].set_title(f"A–B record {case['name']}: exact + Gaussian VI")
        axes[1, col].set_xlabel("skill of A, $z_A$")
    for row in range(2):
        axes[row, 0].set_ylabel("skill of B, $z_B$")
        legend_ax = axes[row, min(1, len(cases) - 1)]  # the A-wins-all panel has an empty upper left
        legend_ax.plot([], [], color=INK, lw=1.0, label="exact posterior")
        legend_ax.legend(loc="upper left", fontsize=7, markerscale=4)
    fig.suptitle("Two-player posteriors: contours enclose 50%, 90%, 99% of the mass", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def mala_pitfall(path, grid_axis, exact_marginal, squared_marginal, draws_right, draws_wrong,
                 sd_ratio_right, sd_ratio_wrong, toy_name):
    """Left: marginal of z_A for a toy case. Right: per-team sd ratios vs HMC on real data."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    bins = np.linspace(grid_axis[0], grid_axis[-1], 81)
    ax.hist(draws_right, bins=bins, density=True, color=COLORS["mala"], alpha=0.45, label="MALA (correct)")
    ax.hist(draws_wrong, bins=bins, density=True, color=COLORS["mala_omitted"], alpha=0.45,
            label=LABELS["mala_omitted"])
    ax.plot(grid_axis, exact_marginal, color=INK, lw=1.4, label="exact posterior p")
    ax.plot(grid_axis, squared_marginal, color=INK_2, lw=1.4, ls="--", label="p² (normalized)")
    ax.set_xlim(-3, 4)
    ax.set_xlabel("skill of A, $z_A$")
    ax.set_ylabel("density")
    ax.set_title(f"Two players, record {toy_name}: marginal of $z_A$")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    x = np.arange(len(sd_ratio_right))
    order = np.argsort(sd_ratio_wrong)
    ax.scatter(x, sd_ratio_right[order], s=14, color=COLORS["mala"], label="MALA (correct)")
    ax.scatter(x, sd_ratio_wrong[order], s=14, color=COLORS["mala_omitted"], marker="s",
               label=LABELS["mala_omitted"])
    ax.axhline(1.0, color=INK_2, lw=0.8)
    ax.axhline(1 / math.sqrt(2), color=INK_2, lw=0.8, ls="--")
    ax.text(len(x) - 1, 1 / math.sqrt(2) + 0.01, "1/√2", ha="right", va="bottom", color=INK_2, fontsize=8)
    ax.set_ylim(0.5, 1.15)
    ax.set_xlabel("team (sorted by ratio)")
    ax.set_ylabel("posterior sd / HMC posterior sd")
    ax.set_title("Premier League: marginal sd relative to HMC")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


CHAIN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def mcmc_traces(path, chains, pair, pair_names, n_show):
    """Traces of the average skill and of one skill difference, for MALA and HMC.

    chains: dict method -> draws (n, C, N); pair: indices (i, j) of the difference.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.2), sharex=True, sharey="row")
    for col, method in enumerate(("mala", "hmc")):
        draws = chains[method][:n_show]
        quantities = (draws.mean(-1), draws[:, :, pair[0]] - draws[:, :, pair[1]])
        titles = ("average skill over teams", f"{pair_names[0]} minus {pair_names[1]}")
        for row, (q, title) in enumerate(zip(quantities, titles)):
            ax = axes[row, col]
            for c in range(q.shape[1]):
                ax.plot(q[:, c], lw=0.5, alpha=0.8, color=CHAIN_COLORS[c % len(CHAIN_COLORS)],
                        label=f"chain {c + 1}")
            ax.set_title(f"{LABELS[method]}: {title}")
    for ax in axes[1]:
        ax.set_xlabel("iteration after warm-up")
    axes[0, 0].set_ylabel("average skill")
    axes[1, 0].set_ylabel("skill difference")
    axes[0, 1].legend(fontsize=7, loc="upper right", ncol=4)
    fig.suptitle(f"Trace plots of the first {n_show} iterations after warm-up", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def elbo_curves(path, histories, log_evidence, window=100):
    """histories: dict method -> per-iteration ELBO estimates (numpy)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for key, h in histories.items():
        w = min(window, max(1, len(h) // 10))
        smooth = np.convolve(h, np.ones(w) / w, mode="valid")
        it = np.arange(w, len(h) + 1)
        ax.plot(it, smooth, color=COLORS[key], lw=1.5, label=LABELS[key])
    ax.axhline(log_evidence, color=INK, lw=1.0, ls="--", label="log evidence (importance sampling)")
    ax.set_xscale("log")
    ax.set_ylim(log_evidence - 45, log_evidence + 5)
    ax.set_xlabel("iteration")
    ax.set_ylabel(f"ELBO (running mean over {window} iterations)")
    ax.set_title("Stochastic VI on the Premier League data")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def team_skills(path, names, summaries, order, methods):
    """Forest plot: posterior mean ± 2 sd per team and method, teams sorted by `order`."""
    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    n = len(order)
    offsets = np.linspace(0.27, -0.27, len(methods))  # first method on top, as in the legend
    for off, key in zip(offsets, methods):
        mean, sd = summaries[key]
        y = np.arange(n)[::-1] + off
        ax.errorbar(mean[order], y, xerr=2 * sd[order], fmt="o", ms=3, lw=1.0, color=COLORS[key],
                    label=LABELS[key])
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_ylim(-0.8, n - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("skill $z$ (posterior mean ± 2 sd)")
    ax.set_title("Premier League team skills, 2500 decisive matches (Dec 2014 – May 2023)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def uncertainty(path, ref_sd, sd_by_method, ref_pairs, pairs_by_method, joint, names_ij):
    """(a) marginal sd vs HMC, (b) pairwise P(z_i > z_j) vs HMC, (c) joint of two teams."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    markers = {"mfvi": "o", "fcvi": "^", "mala_omitted": "s", "svi_short": "D"}
    ax = axes[0]
    for key, sd in sd_by_method.items():
        ax.scatter(ref_sd, sd, s=14, color=COLORS[key], marker=markers[key], label=LABELS[key])
    lim = (0, max(ref_sd.max(), max(s.max() for s in sd_by_method.values())) * 1.05)
    ax.plot(lim, lim, color=INK_2, lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("HMC posterior sd")
    ax.set_ylabel("approximate posterior sd")
    ax.set_title("(a) marginal standard deviations")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    for key, p in pairs_by_method.items():
        ax.scatter(ref_pairs, p - ref_pairs, s=5, color=COLORS[key], marker=markers[key], alpha=0.7,
                   lw=0, label=LABELS[key])
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("HMC estimate of $P(z_i > z_j)$")
    ax.set_ylabel("approximation − HMC")
    ax.set_title("(b) all 528 pairwise probabilities")
    ax.legend(fontsize=7, loc="lower left", markerscale=2)
    ax = axes[2]
    draws, gaussians = joint
    ax.scatter(draws[:, 0], draws[:, 1], s=2, color=COLORS["hmc"], alpha=0.3, lw=0, label="HMC draws")
    for key, (mean, cov) in gaussians.items():
        _gaussian_ellipse(ax, mean, cov, COLORS[key], "--" if key == "mfvi" else "-", LABELS[key])
    lo = min(draws.min(), *(m.min() - 3 * np.sqrt(c.diagonal().max()) for m, c in gaussians.values()))
    hi = max(draws.max(), *(m.max() + 3 * np.sqrt(c.diagonal().max()) for m, c in gaussians.values()))
    _equal_skill(ax, (lo, hi))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"skill of {names_ij[0]}")
    ax.set_ylabel(f"skill of {names_ij[1]}")
    ax.set_title(f"(c) joint posterior, {names_ij[0]} vs {names_ij[1]}")
    ax.legend(fontsize=7, loc="upper left", markerscale=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
