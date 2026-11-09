"""Figures for the VAE density-modelling study (matplotlib, Agg backend)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

# Categorical slots (fixed order) and a one-hue sequential ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, DATA_GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
CYCLIC = "twilight_shifted"


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "figure.constrained_layout.use": True,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.6,
    })


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def _data_axes(ax):
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.85, 0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def data_figure(x, label, log_density_grid, extent, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    for c in np.unique(label):
        ax.scatter(*x[label == c].T, s=4, color=SERIES[c], label=f"arc {c}", linewidths=0)
    _data_axes(ax)
    ax.legend(loc="upper left", ncol=3, markerscale=3, fontsize=8)
    ax.set_title("Training data (colour = generating arc; never shown to the models)")
    ax = axes[1]
    im = ax.imshow(log_density_grid, origin="lower", extent=extent, cmap=SEQ, vmin=-6, vmax=3, aspect="equal")
    _data_axes(ax)
    ax.grid(False)
    ax.set_title("True log-density $\\log p(x)$ (clipped at $-6$)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="nats")
    _save(fig, path)


def training_figure(histories, names, sigma_true, sigma_fixed, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    for i, (name, h) in enumerate(zip(names, histories)):
        ax.plot(h["iteration"], h["train_elbo"], color=SERIES[i], label=f"{name} (train)")
        ax.plot(h["iteration"], h["val_elbo"], color=SERIES[i], ls="--", lw=1.0, label=f"{name} (validation)")
    low = -2.0
    ax.set_ylim(low, 1.0)
    off = [f"{name}: {h['val_elbo'][-1]:.2f}" for name, h in zip(names, histories) if h["val_elbo"][-1] < low]
    if off:
        ax.text(0.02, 0.97, "final validation ELBO off scale\n" + "\n".join(off), transform=ax.transAxes,
                ha="left", va="top", fontsize=7, color=INK_2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("ELBO per point (nats)")
    ax.set_title("ELBO during training (seed 0)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax = axes[1]
    for i, (name, h) in enumerate(zip(names, histories)):
        s = np.asarray(h["sigma"])
        if np.allclose(s, s[0]):
            continue
        ax.plot(h["iteration"], s[:, 0], color=SERIES[i], label=f"{name}: $\\sigma_1$")
        ax.plot(h["iteration"], s[:, 1], color=SERIES[i], ls="--", lw=1.0, label=f"{name}: $\\sigma_2$")
    ax.axhline(sigma_true, color=INK, lw=1.0, ls=":", label=f"true noise sd = {sigma_true}")
    ax.axhline(sigma_fixed, color=INK_2, lw=1.0, ls="-.", label=f"fixed decoder sd = {sigma_fixed}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("decoder noise sd (mean over encoded data)")
    ax.set_title("Learned observation noise")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, path)


def _prior_circles(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    for r in (1, 2, 3):
        ax.plot(r * np.cos(t), r * np.sin(t), color=GRID, lw=0.8, zorder=0)


def latent_figure(encoded, names, label, theta, path):
    """encoded: list of (n, 2) arrays of encoder means for the training data."""
    fig, axes = plt.subplots(2, len(encoded), figsize=(3.6 * len(encoded), 7))
    for j, (m, name) in enumerate(zip(encoded, names)):
        ax = axes[0, j]
        _prior_circles(ax)
        for c in np.unique(label):
            ax.scatter(*m[label == c].T, s=4, color=SERIES[c], label=f"arc {c}", linewidths=0)
        ax.set_title(f"{name}: colour = arc")
        ax = axes[1, j]
        _prior_circles(ax)
        sc = ax.scatter(*m.T, s=4, c=np.mod(theta, 2 * np.pi), cmap=CYCLIC, vmin=0, vmax=2 * np.pi, linewidths=0)
        ax.set_title(f"{name}: colour = curve position $\\theta$")
        for ax in axes[:, j]:
            ax.set_xlim(-3.3, 3.3)
            ax.set_ylim(-3.3, 3.3)
            ax.set_aspect("equal")
            ax.set_xlabel("$z_1$")
            ax.set_ylabel("$z_2$")
    axes[0, 0].legend(loc="upper left", markerscale=3, fontsize=7)
    fig.colorbar(sc, ax=axes[1, :].tolist(), shrink=0.8, label="$\\theta$ mod $2\\pi$")
    _save(fig, path)


def samples_figure(means, noisy, names, x_train, path):
    fig, axes = plt.subplots(2, len(names), figsize=(3.6 * len(names), 4.6))
    for j, name in enumerate(names):
        for i, (pts, kind) in enumerate([(means[j], "decoder means only"), (noisy[j], "with likelihood noise")]):
            ax = axes[i, j]
            ax.scatter(*x_train.T, s=2, color=DATA_GRAY, linewidths=0, label="training data")
            ax.scatter(*pts.T, s=2, color=SERIES[j], linewidths=0, label="model samples")
            _data_axes(ax)
            ax.set_title(f"{name}\n{kind}", fontsize=9)
    axes[0, 0].legend(loc="upper left", markerscale=3, fontsize=7)
    _save(fig, path)


def _grid_lines(gx, gy, ax, **kw):
    segs = np.stack([gx, gy], axis=2)
    ax.add_collection(LineCollection(segs, **kw))
    ax.add_collection(LineCollection(segs.transpose(1, 0, 2), **kw))


def grid_warp_figure(warps, names, path):
    """warps[j] = dict with the encoder image (ex, ey) of a data-space grid, the decoder image
    (dx, dy) of a latent grid, the encoded training means and the training data."""
    fig, axes = plt.subplots(len(names), 2, figsize=(9.5, 3.6 * len(names)))
    for j, (w, name) in enumerate(zip(warps, names)):
        ax = axes[j, 0]
        _grid_lines(w["ex"], w["ey"], ax, colors=SERIES[j], linewidths=0.7)
        ax.scatter(*w["encoded"].T, s=2, color=INK, linewidths=0, zorder=3)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.set_title(f"{name}: encoder mean of a grid on\n"
                     "$[-1.3, 1.3] \\times [-0.8, 0.8]$ (dots: encoded data)", fontsize=9)
        ax = axes[j, 1]
        _grid_lines(w["dx"], w["dy"], ax, colors=SERIES[j], linewidths=0.7)
        ax.scatter(*w["x_train"].T, s=2, color=INK, linewidths=0, zorder=3)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(f"{name}: decoder mean of a grid on\n$[-3, 3]^2$ in latent space (dots: data)", fontsize=9)
    _save(fig, path)


def density_figure(maps, titles, extent, path):
    n = len(maps)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        im = ax.imshow(maps[k], origin="lower", extent=extent, cmap=SEQ, vmin=-6, vmax=3, aspect="equal")
        _data_axes(ax)
        ax.grid(False)
        ax.set_title(titles[k], fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="$\\log p(x)$, nats (clipped at $-6$)")
    _save(fig, path)


def heldout_figure(gmm_table, k_star, rows, truth, path):
    """gmm_table: dict K -> (train, val, test); rows: list of (label, color, dict metric -> (mean, lo, hi))."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ks = sorted(gmm_table)
    for idx, (part, ls) in enumerate([("train", "-"), ("validation", "--"), ("test", "-")]):
        vals = [gmm_table[k][idx] for k in ks]
        ax.plot(ks, vals, ls=ls, marker="o", ms=3.5, color=[INK_2, INK_2, SERIES[0]][idx],
                lw=[1.0, 1.0, 1.8][idx], label=f"GMM, {part}")
    ax.axhline(truth, color=INK, ls=":", lw=1.0, label="true density (test)")
    ax.axvline(k_star, color=GRID, lw=2.0, zorder=0)
    ax.annotate(f"selected K = {k_star}", (k_star, -1.45), xytext=(-4, 0), textcoords="offset points",
                ha="right", fontsize=8, color=INK_2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=7)
    ax.set_xlabel("number of mixture components K")
    ax.set_ylabel("average log-likelihood (nats)")
    ax.set_ylim(-1.6, 1.4)
    ax.set_title("Gaussian mixture fitted by EM")
    ax.legend(fontsize=7, loc="center right")

    ax = axes[1]
    markers = {"ELBO": ("s", "white"), "IWAE": ("D", None), "exact": ("o", None)}
    labels = []
    for i, (lab, color, metrics) in enumerate(rows):
        y = len(rows) - 1 - i
        labels.append((y, lab))
        for name, (mean, lo, hi) in metrics.items():
            mk, face = markers.get(name, ("o", None))
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt=mk, ms=6, color=color,
                        mfc=face or color, mec=color, capsize=2, lw=1.0)
    ax.axvline(truth, color=INK, ls=":", lw=1.0)
    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([lab for _, lab in labels], fontsize=8)
    ax.set_xlabel("test log-likelihood per point (nats)")
    ax.set_title("Held-out density: square = ELBO, diamond = IWAE, circle = exact")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def iwae_figure(ks, curves, exact, names, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, name in enumerate(names):
        gap = exact[i] - np.asarray(curves[i])
        ax.plot(ks, gap, marker="o", ms=4, color=SERIES[i], label=name)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(-1e-2, 30)
    ax.axhline(0.0, color=INK_2, lw=0.8)
    ax.set_xlabel("number of importance samples K")
    ax.set_ylabel("exact log p(x) - IWAE$_K$ (nats, symlog)")
    ax.set_title("Looseness of the importance-weighted bound (seed 0, test subset)")
    ax.legend(fontsize=8)
    _save(fig, path)


def latent1_figure(z_line, x_line, x_train, enc_mean, label, theta, x_test, gap, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    ax = axes[0]
    ax.scatter(*x_train.T, s=2, color=DATA_GRAY, linewidths=0)
    pts = x_line.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ramp = LinearSegmentedColormap.from_list("z_ramp", ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    lc = LineCollection(segs, cmap=ramp, array=z_line[:-1], linewidths=2.0)
    lc.set_clim(z_line.min(), z_line.max())
    ax.add_collection(lc)
    _data_axes(ax)
    fig.colorbar(lc, ax=ax, shrink=0.8, label="latent z")
    ax.set_title("Decoder mean $\\mu(z)$ for $z \\in [-4, 4]$")
    ax = axes[1]
    for c in np.unique(label):
        sel = label == c
        ax.scatter(np.mod(theta[sel], 2 * np.pi), enc_mean[sel], s=3, color=SERIES[c], linewidths=0, label=f"arc {c}")
    ax.set_xlabel("curve position $\\theta$ mod $2\\pi$")
    ax.set_ylabel("encoder mean $m(x)$")
    ax.set_title("Encoder: one latent coordinate along the loop")
    ax.legend(markerscale=3, fontsize=7)
    ax = axes[2]
    order = np.argsort(gap)
    g = np.clip(gap[order], 1e-3, None)
    sc = ax.scatter(*x_test[order].T, s=4, c=g, cmap="magma_r", norm=LogNorm(vmin=1e-3, vmax=max(g.max(), 1.0)),
                    linewidths=0)
    _data_axes(ax)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="exact log p(x) - IWAE (nats)")
    ax.set_title("Test points where the IWAE estimate fails")
    _save(fig, path)
