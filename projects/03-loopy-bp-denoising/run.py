"""Regenerate every figure in figures/ and every number in results/.

Binary image denoising with an Ising prior: loopy BP (sum- and max-product)
compared with naive mean field, Gibbs sampling, and the exact MAP (graph cut).

    python run.py            # full run, a few minutes on a laptop CPU
    python run.py --quick    # smoke run (overrides grid sizes, seeds, and sweeps)
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from src.bp import loopy_bp  # noqa: E402
from src.data import flip_noise, load_binary_image, matched_beta  # noqa: E402
from src.exact import brute_force, graph_cut_map  # noqa: E402
from src.gibbs import gibbs_means  # noqa: E402
from src.ising import couplings, decide, log_potential, pseudo_likelihood_coupling  # noqa: E402
from src.meanfield import mean_field  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG, RES = ROOT / "figures", ROOT / "results"

METHODS = {  # key: (label, color, marker); colors in a fixed categorical order
    "bp_sum": ("Loopy BP, sum-product", "#2a78d6", "o"),
    "bp_max": ("Loopy BP, max-product", "#eb6834", "s"),
    "mean_field": ("Naive mean field", "#1baf7a", "^"),
    "gibbs": ("Gibbs sampling", "#eda100", "D"),
    "map": ("Exact MAP (graph cut)", "#e87ba4", "v"),
}
GRAY, INK, MUTED = "#8a8986", "#0b0b0b", "#52514e"
ETA_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue ramp, light = little damping


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="fast smoke run (overrides several options)")
    p.add_argument("--image", default="data/trc1l3gqu9651.png", help="image path, relative to the project")
    p.add_argument("--size", type=int, default=96, help="image side length after resizing")
    p.add_argument("--threshold", type=int, default=120, help="gray level above which a pixel is white")
    p.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4], help="flip rates")
    p.add_argument("--J", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.5, 2.0, 3.0, 5.0],
                   help="coupling grid")
    p.add_argument("--seeds", type=int, default=3, help="noise realizations per flip rate")
    p.add_argument("--eta", type=float, default=0.5, help="BP damping: weight on the new message")
    p.add_argument("--max-iter", type=int, default=300, help="iteration cap for BP and mean field")
    p.add_argument("--tol", type=float, default=1e-4, help="stop when the largest update is below tol")
    p.add_argument("--gibbs-sweeps", type=int, default=1000, help="Gibbs sweeps after burn-in")
    p.add_argument("--gibbs-burn-in", type=int, default=200)
    p.add_argument("--conv-eps", type=float, default=0.2, help="flip rate of the convergence study")
    p.add_argument("--conv-J", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--conv-etas", type=float, nargs="+", default=[1.0, 0.8, 0.5, 0.2])
    p.add_argument("--conv-iters", type=int, default=200)
    p.add_argument("--gallery-eps", type=float, nargs="+", default=[0.05, 0.3])
    p.add_argument("--gallery-J", type=float, nargs="+", default=[0.1, 0.5, 1.0, 5.0])
    p.add_argument("--patch-grid", type=int, default=6, help="patch study uses patch-grid^2 patches")
    p.add_argument("--patch-size", type=int, default=4, help="side of the brute-force patches")
    p.add_argument("--patch-eps", type=float, default=0.2)
    p.add_argument("--patch-gibbs-sweeps", type=int, default=5000)
    args = p.parse_args()
    if args.quick:
        args.eps, args.J, args.seeds = [0.05, 0.2], [0.2, 0.5, 1.0, 2.0], 1
        args.gibbs_sweeps, args.gibbs_burn_in, args.patch_gibbs_sweeps = 200, 50, 500
        args.conv_iters, args.patch_grid = 60, 3
    return args


def noise_rng(seed, eps, *extra):
    return np.random.default_rng([seed, round(eps * 10_000), *extra])


def run_method(method, y, beta, J, args, rng=None):
    """Denoise y with one method; returns the decision image and run information."""
    h = beta * y
    t0 = time.perf_counter()
    info = {"n_iter": np.nan, "converged": np.nan}
    if method in ("bp_sum", "bp_max"):
        out = loopy_bp(h, J, mode=method[3:], eta=args.eta, max_iter=args.max_iter, tol=args.tol)
        xhat = decide(out["log_odds"], y)
        info = {"n_iter": out["n_iter"], "converged": out["converged"]}
    elif method == "mean_field":
        out = mean_field(h, J, max_iter=args.max_iter, tol=args.tol)
        xhat = decide(out["mean"], y)
        info = {"n_iter": out["n_iter"], "converged": out["converged"]}
    elif method == "gibbs":
        xhat = decide(gibbs_means(h, J, args.gibbs_sweeps, args.gibbs_burn_in, rng), y)
    elif method == "map":
        xhat = graph_cut_map(h, J)
    else:
        raise ValueError(method)
    info["runtime_s"] = time.perf_counter() - t0
    return xhat, info


# --------------------------------------------------------------------------- experiments

def grid_experiment(x, args, J_pl):
    """Pixel error over flip rate x coupling, for matched beta and beta = 1."""
    rows = []
    settings = [(J, "grid") for J in args.J] + [(J_pl, "pseudo-likelihood")]
    for eps in args.eps:
        t_eps = time.perf_counter()
        for seed in range(args.seeds):
            y = flip_noise(x, eps, noise_rng(seed, eps))
            rows.append(dict(eps=eps, seed=seed, method="noisy", error=np.mean(y != x)))
            for j_idx, (J, source) in enumerate(settings):
                for beta_mode in ("matched", "one") if source == "grid" else ("matched",):
                    beta = matched_beta(eps) if beta_mode == "matched" else 1.0
                    # The exact MAP runs first so that max-product BP can be compared with it.
                    methods = (["map", "bp_sum", "bp_max", "mean_field", "gibbs"] if beta_mode == "matched"
                               else ["map", "bp_sum", "bp_max"])
                    Jh, Jv = couplings(J, x.shape)
                    for method in methods:
                        xhat, info = run_method(method, y, beta, J, args, noise_rng(seed, eps, j_idx, 1))
                        row = dict(eps=eps, seed=seed, J=J, J_source=source, beta_mode=beta_mode,
                                   beta=beta, method=method, error=np.mean(xhat != x), **info)
                        if method == "map":
                            xmap = xhat
                        if method == "bp_max":
                            row["diff_from_map"] = np.mean(xhat != xmap)
                            row["logp_gap"] = (log_potential(xmap, beta * y, Jh, Jv)
                                               - log_potential(xhat, beta * y, Jh, Jv))
                        rows.append(row)
        print(f"  grid: eps={eps} done in {time.perf_counter() - t_eps:.0f}s", flush=True)
    return pd.DataFrame(rows)


def convergence_experiment(x, args):
    """Message residual and pixel error per iteration for several damping weights."""
    eps = args.conv_eps
    y = flip_noise(x, eps, noise_rng(0, eps))
    h = matched_beta(eps) * y
    rows = []
    for mode in ("sum", "max"):
        for J in args.conv_J:
            for eta in args.conv_etas:
                errors = []
                out = loopy_bp(h, J, mode=mode, eta=eta, max_iter=args.conv_iters, tol=0.0,
                               callback=lambda t, lo: errors.append(np.mean(decide(lo, y) != x)))
                for t, (res, err) in enumerate(zip(out["residuals"], errors), start=1):
                    rows.append(dict(mode=mode, J=J, eta=eta, iteration=t, residual=res, error=err))
    return pd.DataFrame(rows)


def tiled_couplings(J, n_blocks, k):
    """Uniform coupling J inside k x k blocks of an (n_blocks k) x (n_blocks k) grid, 0 across blocks."""
    side = n_blocks * k
    Jh, Jv = couplings(J, (side, side))
    Jh[:, k - 1::k] = 0.0
    Jv[k - 1::k, :] = 0.0
    return Jh, Jv


def patch_experiment(x, args):
    """Approximate vs exact marginals on small patches, all solved at once as disconnected blocks."""
    eps, k, nb = args.patch_eps, args.patch_size, args.patch_grid
    beta = matched_beta(eps)
    y = flip_noise(x, eps, noise_rng(0, eps))
    rng = np.random.default_rng(2025)
    corners = rng.integers(0, x.shape[0] - k + 1, size=(nb * nb, 2))
    h_tiled = np.zeros((nb * k, nb * k))
    y_tiled = np.zeros_like(h_tiled)
    for p, (r, c) in enumerate(corners):
        br, bc = divmod(p, nb)
        y_tiled[br * k:(br + 1) * k, bc * k:(bc + 1) * k] = y[r:r + k, c:c + k]
    h_tiled = beta * y_tiled
    rows = []
    for j_idx, J in enumerate(args.J):
        Jt = tiled_couplings(J, nb, k)
        bp_sum = loopy_bp(h_tiled, Jt, "sum", eta=args.eta, max_iter=5000, tol=1e-10)
        bp_max = loopy_bp(h_tiled, Jt, "max", eta=args.eta, max_iter=5000, tol=1e-10)
        approx = {
            "bp_sum": bp_sum["p_plus"],
            "mean_field": (1 + mean_field(h_tiled, Jt, max_iter=5000, tol=1e-10)["mean"]) / 2,
            "gibbs": (1 + gibbs_means(h_tiled, Jt, args.patch_gibbs_sweeps, 500,
                                      np.random.default_rng([7, j_idx]))) / 2,
        }
        bp_max_dec = decide(bp_max["log_odds"], y_tiled)
        for p in range(nb * nb):
            br, bc = divmod(p, nb)
            sl = (slice(br * k, (br + 1) * k), slice(bc * k, (bc + 1) * k))
            exact = brute_force(h_tiled[sl], J)
            for method, p_plus in approx.items():
                err = np.abs(p_plus[sl] - exact["p_plus"])
                rows.append(dict(J=J, patch=p, method=method, mean_abs_err=err.mean(), max_abs_err=err.max(),
                                 bp_converged=bp_sum["converged"] if method == "bp_sum" else np.nan))
            # Agreement with the MAP is only defined when the MAP is unique (no exact max-marginal ties).
            unique = bool(np.abs(exact["max_log_odds"]).min() > 1e-9)
            rows.append(dict(J=J, patch=p, method="bp_max", map_unique=unique,
                             map_agreement=np.mean(bp_max_dec[sl] == exact["map"]) if unique else np.nan,
                             bp_converged=bp_max["converged"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- summaries

def best_by_J(df):
    """For each (eps, method, beta_mode): the grid J with the lowest mean error over seeds."""
    g = df[df.J_source == "grid"].groupby(["eps", "method", "beta_mode", "J"])["error"]
    stats = g.agg(["mean", "std"]).reset_index()
    best = stats.loc[stats.groupby(["eps", "method", "beta_mode"])["mean"].idxmin()]
    return best.rename(columns={"mean": "error_mean", "std": "error_std"})


def to_json(obj):
    """Convert numpy scalars to Python types and NaN to None (strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def summarize(x, grid, conv, patches, J_pl, args, runtime):
    best = best_by_J(grid)
    noisy = grid[grid.method == "noisy"].groupby("eps")["error"].mean()
    at_pl = (grid[grid.J_source == "pseudo-likelihood"].groupby(["eps", "method"])["error"].mean())
    table = []
    for eps in args.eps:
        row = {"eps": eps, "beta_star": round(matched_beta(eps), 4), "noisy": round(noisy[eps], 4)}
        for method in METHODS:
            b = best[(best.eps == eps) & (best.method == method) & (best.beta_mode == "matched")].iloc[0]
            row[f"{method}_best_J"] = b.J
            row[f"{method}_best_error"] = round(b.error_mean, 4)
            row[f"{method}_best_error_std"] = round(b.error_std, 4)
            row[f"{method}_error_at_J_pl"] = round(at_pl[(eps, method)], 4)
        for method in ("bp_sum", "bp_max", "map"):
            b = best[(best.eps == eps) & (best.method == method) & (best.beta_mode == "one")].iloc[0]
            row[f"{method}_beta1_best_J"] = b.J
            row[f"{method}_beta1_best_error"] = round(b.error_mean, 4)
        table.append(row)
    table = pd.DataFrame(table)
    table.to_csv(RES / "method_table.csv", index=False)

    matched = grid[(grid.beta_mode == "matched") & (grid.J_source == "grid")]
    bpmax = matched[matched.method == "bp_max"]
    vs_map = bpmax.groupby("J").agg(mean_pixel_diff=("diff_from_map", "mean"),
                                    frac_runs_identical=("diff_from_map", lambda s: float(np.mean(s == 0))),
                                    mean_logp_gap=("logp_gap", "mean"))
    bp_rows = grid[grid.method.isin(["bp_sum", "bp_max"]) & (grid.J_source == "grid")]
    conv_frac = bp_rows.groupby(["method", "beta_mode", "J"])["converged"].mean().unstack("J")

    conv_summary = []
    for (mode, J, eta), d in conv.groupby(["mode", "J", "eta"], sort=False):
        below = d[d.residual < args.tol]
        conv_summary.append(dict(
            mode=mode, J=J, eta=eta,
            first_iter_below_tol=int(below.iteration.iloc[0]) if len(below) else None,
            final_residual=float(d.residual.iloc[-1]), final_error=round(float(d.error.iloc[-1]), 4),
            min_error=round(float(d.error.min()), 4), iter_of_min_error=int(d.iteration.iloc[d.error.argmin()])))

    patch_summary = (patches.groupby(["J", "method"])
                     .agg(mean_abs_err=("mean_abs_err", "mean"), max_abs_err=("max_abs_err", "max"),
                          map_agreement=("map_agreement", "mean"), n_unique_map=("map_unique", "sum"),
                          bp_converged=("bp_converged", "mean"))
                     .reset_index())
    patch_summary.to_csv(RES / "patch_summary.csv", index=False)

    summary = {
        "image": {"size": int(x.shape[0]), "fraction_white": round(float(np.mean(x > 0)), 4),
                  "J_pseudo_likelihood": round(J_pl, 4)},
        "settings": {k: v for k, v in vars(args).items()},
        "method_table": table.to_dict(orient="records"),
        "mean_runtime_s_per_image": matched.groupby("method")["runtime_s"].mean().round(4).to_dict(),
        "mean_iterations": matched[matched.method.isin(["bp_sum", "bp_max", "mean_field"])]
        .groupby("method")["n_iter"].mean().round(1).to_dict(),
        "bp_fraction_converged_by_J": {f"{m}|{b}": {str(J): round(v, 3) for J, v in r.items()}
                                        for (m, b), r in conv_frac.iterrows()},
        "max_product_vs_exact_map_by_J": {str(J): {k: round(float(v), 5) for k, v in r.items()}
                                          for J, r in vs_map.iterrows()},
        "convergence_study": conv_summary,
        "patch_study": patch_summary.round(5).to_dict(orient="records"),
        "total_runtime_s": round(runtime, 1),
    }
    with open(RES / "summary.json", "w") as f:
        json.dump(to_json(summary), f, indent=2, allow_nan=False)
    return table, best


# --------------------------------------------------------------------------- figures

def style_axes(ax):
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=MUTED, labelsize=8)


def log_x(ax, ticks=(0.1, 0.2, 0.5, 1, 2, 5)):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def show(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_examples(x, args, J_show):
    eps = args.conv_eps
    y = flip_noise(x, eps, noise_rng(0, eps))
    beta = matched_beta(eps)
    fig, axes = plt.subplots(1, 7, figsize=(15, 2.7))
    show(axes[0], x, "clean")
    show(axes[1], y, f"noisy, eps={eps}\nerror {np.mean(y != x):.1%}")
    for ax, method in zip(axes[2:], METHODS):
        xhat, _ = run_method(method, y, beta, J_show, args, noise_rng(0, eps, 99))
        show(ax, xhat, f"{METHODS[method][0]}\nerror {np.mean(xhat != x):.1%}")
    fig.suptitle(f"Denoising one noisy image (beta = beta* = {beta:.3f}, J = {J_show})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "denoising_examples.png", dpi=150)
    plt.close(fig)


def fig_gallery(x, args):
    cols = len(args.gallery_J) + 1
    fig, axes = plt.subplots(len(args.gallery_eps), cols, figsize=(2.2 * cols, 2.9 * len(args.gallery_eps)),
                             squeeze=False)
    for r, eps in enumerate(args.gallery_eps):
        y = flip_noise(x, eps, noise_rng(0, eps))
        show(axes[r, 0], y, f"noisy, eps={eps}\nerror {np.mean(y != x):.1%}")
        for c, J in enumerate(args.gallery_J, start=1):
            xhat, _ = run_method("bp_max", y, matched_beta(eps), J, args)
            show(axes[r, c], xhat, f"J={J}\nerror {np.mean(xhat != x):.1%}")
    fig.suptitle("Max-product loopy BP with matched beta: the coupling J sets the amount of smoothing",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "coupling_gallery.png", dpi=150)
    plt.close(fig)


def fig_convergence(conv, args):
    combos = [(m, J) for m in ("sum", "max") for J in args.conv_J]
    fig, axes = plt.subplots(2, len(combos), figsize=(3.6 * len(combos), 6), sharex=True)
    for c, (mode, J) in enumerate(combos):
        for eta, color in zip(args.conv_etas, ETA_COLORS):
            d = conv[(conv["mode"] == mode) & (conv.J == J) & (conv.eta == eta)]
            axes[0, c].semilogy(d.iteration, np.maximum(d.residual, 1e-16), color=color, lw=1.5,
                                label=f"eta={eta}")
            axes[1, c].plot(d.iteration, 100 * d.error, color=color, lw=1.5, label=f"eta={eta}")
        axes[0, c].set_title(f"{mode}-product, J={J}", fontsize=10)
        axes[0, c].axhline(2 * J, color=GRAY, ls=":", lw=1)
        axes[0, c].text(args.conv_iters, 2 * J, "2J = largest possible change", color=MUTED, fontsize=7,
                        ha="right", va="bottom")
        axes[1, c].set_xlabel("iteration")
        for ax in axes[:, c]:
            style_axes(ax)
    axes[0, 0].set_ylabel("max |change of log-message|")
    axes[1, 0].set_ylabel("pixel error (%)")
    axes[0, 0].legend(fontsize=8, frameon=False, title="damping weight", title_fontsize=8)
    fig.suptitle(f"Synchronous loopy BP at eps={args.conv_eps}, beta=beta*: residual (top) and "
                 "pixel error (bottom) per iteration", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "convergence_damping.png", dpi=150)
    plt.close(fig)


def mean_curve(grid, method, beta_mode, eps):
    d = grid[(grid.method == method) & (grid.beta_mode == beta_mode) & (grid.eps == eps)
             & (grid.J_source == "grid")]
    return d.groupby("J")["error"].mean()


def fig_beta_choice(grid, args, J_pl):
    n = len(args.eps) + 1
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), squeeze=False)
    axes = axes.ravel()
    noisy = grid[grid.method == "noisy"].groupby("eps")["error"].mean()
    for ax, eps in zip(axes, args.eps):
        for method in ("bp_sum", "bp_max"):
            label, color, marker = METHODS[method]
            for beta_mode, ls in (("matched", "-"), ("one", "--")):
                s = mean_curve(grid, method, beta_mode, eps)
                ax.plot(s.index, 100 * s.values, ls=ls, color=color, marker=marker, ms=4, lw=1.5,
                        label=f"{label}, beta={'beta*' if beta_mode == 'matched' else '1'}")
        ax.axhline(100 * noisy[eps], color=GRAY, ls=":", lw=1.2, label="no denoising")
        ax.axvline(J_pl, color=GRAY, lw=0.8, ls="-.")
        log_x(ax)
        ax.set_title(f"eps={eps}  (beta*={matched_beta(eps):.2f})", fontsize=10)
        ax.set_xlabel("coupling J")
        ax.set_ylabel("pixel error (%)")
        ax.set_ylim(0, None)
        style_axes(ax)
    # Last panel: the same curves against J / beta for the noisiest level.
    ax = axes[len(args.eps)]
    eps = max(args.eps)
    for method in ("bp_sum", "bp_max"):
        label, color, marker = METHODS[method]
        for beta_mode, ls in (("matched", "-"), ("one", "--")):
            beta = matched_beta(eps) if beta_mode == "matched" else 1.0
            s = mean_curve(grid, method, beta_mode, eps)
            ax.plot(s.index / beta, 100 * s.values, ls=ls, color=color, marker=marker, ms=4, lw=1.5)
    log_x(ax, (0.1, 0.2, 0.5, 1, 2, 5, 10, 20))
    ax.set_title(f"eps={eps}, plotted against J / beta", fontsize=10)
    ax.set_xlabel("J / beta")
    ax.set_ylabel("pixel error (%)")
    style_axes(ax)
    for extra in axes[n:]:
        extra.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Pixel error vs coupling J; solid: beta matched to the noise, dashed: beta = 1; "
                 "dash-dot line: pseudo-likelihood J", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "error_vs_coupling.png", dpi=150)
    plt.close(fig)


def fig_methods(grid, args):
    n = len(args.eps)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.4), squeeze=False)
    noisy = grid[grid.method == "noisy"].groupby("eps")["error"].mean()
    for ax, eps in zip(axes[0], args.eps):
        for method, (label, color, marker) in METHODS.items():
            s = mean_curve(grid, method, "matched", eps)
            ax.plot(s.index, 100 * s.values, color=color, marker=marker, ms=4, lw=1.5, label=label)
        ax.axhline(100 * noisy[eps], color=GRAY, ls=":", lw=1.2, label="no denoising")
        log_x(ax)
        ax.set_ylim(0, None)
        ax.set_title(f"eps={eps}", fontsize=10)
        ax.set_xlabel("coupling J")
        style_axes(ax)
    axes[0, 0].set_ylabel("pixel error (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("All inference methods on the same posterior (beta = beta*)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "methods_vs_coupling.png", dpi=150)
    plt.close(fig)


def fig_exactness(patches, grid, args):
    ps = patches.groupby(["J", "method"]).agg(mae=("mean_abs_err", "mean"), worst=("max_abs_err", "max"),
                                              agree=("map_agreement", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for method in ("bp_sum", "mean_field", "gibbs"):
        label, color, marker = METHODS[method]
        d = ps[ps.method == method]
        axes[0].plot(d.J, d.mae, color=color, marker=marker, ms=4, lw=1.5, label=f"{label} (mean)")
        axes[0].plot(d.J, d.worst, color=color, marker=marker, ms=4, lw=1.2, ls="--", label=f"{label} (worst pixel)")
    log_x(axes[0])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("coupling J")
    axes[0].set_ylabel("|P(x=+1) - exact P(x=+1)|")
    axes[0].set_title(f"{args.patch_size}x{args.patch_size} patches (exact by enumeration)", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    d = ps[ps.method == "bp_max"]
    axes[1].plot(d.J, 100 * (1 - d.agree), color=METHODS["bp_max"][1], marker="s", ms=4, lw=1.5,
                 label=f"{args.patch_size}x{args.patch_size} patches with a unique MAP")
    full = grid[(grid.method == "bp_max") & (grid.J_source == "grid") & (grid.beta_mode == "matched")]
    full = full.groupby("J")["diff_from_map"].mean()
    axes[1].plot(full.index, 100 * full.values, color=METHODS["bp_max"][1], marker="s", ms=4, lw=1.5,
                 ls="--", label=f"full {args.size}x{args.size} image (all eps)")
    log_x(axes[1])
    axes[1].set_xlabel("coupling J")
    axes[1].set_ylabel("pixels differing from exact MAP (%)")
    axes[1].set_title("Max-product BP vs exact MAP", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False, loc="center right")
    for ax in axes:
        style_axes(ax)
    fig.suptitle(f"Inference error alone (beta = beta*; patches at eps={args.patch_eps})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "inference_accuracy.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    t_start = time.perf_counter()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    x = load_binary_image(ROOT / args.image, size=args.size, threshold=args.threshold)
    J_pl = pseudo_likelihood_coupling(x)
    print(f"image {x.shape}, white fraction {np.mean(x > 0):.3f}, pseudo-likelihood J = {J_pl:.3f}")

    print("convergence study ...", flush=True)
    conv = convergence_experiment(x, args)
    conv.to_csv(RES / "convergence.csv", index=False)
    print("noise x coupling grid ...", flush=True)
    grid = grid_experiment(x, args, J_pl)
    grid.to_csv(RES / "grid.csv", index=False)
    print("patch study ...", flush=True)
    patches = patch_experiment(x, args)
    patches.to_csv(RES / "patches.csv", index=False)

    print("figures ...", flush=True)
    best = best_by_J(grid)
    show_eps = args.conv_eps if args.conv_eps in args.eps else args.eps[0]
    J_show = float(best[(best.eps == show_eps) & (best.method == "bp_sum") & (best.beta_mode == "matched")].J.iloc[0])
    fig_examples(x, args, J_show)
    fig_gallery(x, args)
    fig_convergence(conv, args)
    fig_beta_choice(grid, args, J_pl)
    fig_methods(grid, args)
    fig_exactness(patches, grid, args)

    table, _ = summarize(x, grid, conv, patches, J_pl, args, time.perf_counter() - t_start)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(f"total runtime {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
