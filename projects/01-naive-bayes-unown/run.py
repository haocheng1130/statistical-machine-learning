"""Naive Bayes on Unown glyphs: classify, generate, and fill in missing pixels.

Regenerates every figure in figures/ and every number in results/ (fixed seeds).
Usage, from the project folder:  python run.py
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
from scipy.special import logsumexp  # noqa: E402

from src import bernoulli_nb as bnb  # noqa: E402
from src import data, masks  # noqa: E402
from src import gaussian_nb as gnb  # noqa: E402
from src.plots import show_montage  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
K = data.N_CLASSES
D = data.IMAGE_SHAPE[0] * data.IMAGE_SHAPE[1]
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66"
MASK_COLORS = {"random": BLUE, "lattice": ORANGE, "raster": AQUA}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="data/unown_mnist.npz", help="data file, relative to the project folder")
    p.add_argument("--seed", type=int, default=0, help="seed for the validation split, masks and samples")
    p.add_argument("--n-val", type=int, default=10000, help="training images held out for model selection")
    p.add_argument("--alpha-minus-one", type=float, nargs="+", default=np.logspace(-8, 3, 23).tolist(),
                   help="prior pseudo-counts alpha - 1 of the Beta(alpha, alpha) sweep (alpha = 1 is always added)")
    p.add_argument("--eps", type=float, nargs="+", default=np.logspace(-9, 0, 19).tolist(),
                   help="variance floors of the Gaussian naive Bayes sweep")
    p.add_argument("--reference-alpha", type=float, default=2.0, help="add-one (Laplace) prior, shown for reference")
    p.add_argument("--reference-eps", type=float, default=1e-5, help="small variance floor, shown for reference")
    p.add_argument("--lattice-sizes", type=int, nargs="+", default=list(range(6, 27)),
                   help="m of the m x m observation lattices; random and raster masks observe the same m^2 pixels")
    p.add_argument("--example-lattice", type=int, default=15,
                   help="lattice size of the example figure (15 x 15 = 225 pixels, 29%%)")
    p.add_argument("--n-examples", type=int, default=30, help="test images in the imputation example figure")
    p.add_argument("--samples-per-class", type=int, default=3, help="class-conditional samples shown per class")
    p.add_argument("--n-generated", type=int, default=10000, help="ancestral samples for the neighbour statistic")
    return p.parse_args()


def finite_or_str(v):
    """JSON-safe number: keep integers, round finite floats, spell out infinities and NaN."""
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return round(v, 4) if np.isfinite(v) else str(v)


# ----------------------------------------------------------------------------- Bernoulli NB: prior sweep

def bnb_scores(B, y, theta, log_pi):
    idx = np.arange(len(y))
    lj = bnb.log_joint(B, theta, log_pi)
    undefined = np.all(np.isneginf(lj), axis=1)           # p(x) = 0: posterior is 0/0
    true_impossible = np.isneginf(lj[idx, y]) & ~undefined  # p(c_true | x) = 0
    true_log_post = bnb.normalize(lj)[idx, y]
    with np.errstate(divide="ignore"):
        log_px = logsumexp(lj, axis=1)
    return {
        "accuracy": float(np.mean(bnb.predict(lj) == y)),
        "mean_log_p_c_given_x": float(np.mean(np.where(undefined, -np.inf, true_log_post))),
        "mean_log_p_x": float(np.mean(log_px)),
        "frac_undefined_posterior": float(np.mean(undefined)),
        "frac_true_class_impossible": float(np.mean(true_impossible)),
        "n_theta_zero_or_one": int(np.sum((theta == 0) | (theta == 1))),
    }


def alpha_sweep(splits, pseudo_counts):
    rows = []
    for a in [0.0] + list(pseudo_counts):
        for split, (B_fit, y_fit, B_eval, y_eval) in splits.items():
            theta, log_pi = bnb.fit(B_fit, y_fit, K, 1.0 + a)
            rows.append({"alpha": 1.0 + a, "alpha_minus_one": a, "split": split,
                         **bnb_scores(B_eval, y_eval, theta, log_pi)})
    return pd.DataFrame(rows)


def plot_alpha_sweep(df, a_gen, a_cls, ref_alpha, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    sweep = df[df.alpha_minus_one > 0]
    mle = df[(df.alpha_minus_one == 0) & (df.split == "test")].iloc[0]
    panels = [("accuracy", "accuracy", "(a) classification accuracy"),
              ("mean_log_p_c_given_x", "mean log p(c | x)  [nats]", "(b) class log-likelihood"),
              ("mean_log_p_x", "mean log p(x)  [nats / image]", "(c) generative log-likelihood")]
    for ax, (col, ylabel, title) in zip(axes, panels):
        for split, color, ls in [("test", BLUE, "-"), ("validation", ORANGE, "--")]:
            s = sweep[sweep.split == split]
            ax.plot(s.alpha_minus_one, s[col], ls, color=color, lw=2, marker="o", ms=4, label=split)
        ax.axvline(ref_alpha - 1, color=GRAY, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(r"prior pseudo-count $\alpha - 1$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].text(ref_alpha - 1, axes[0].get_ylim()[0], r" $\alpha=2$", color=GRAY, va="bottom", fontsize=8)
    axes[0].legend(loc="lower left", frameon=False)
    mle_text = (f"$\\alpha = 1$ (MLE), test:\naccuracy {mle.accuracy:.4f}\n"
                f"{mle.frac_undefined_posterior:.2%} images with p(x) = 0\n"
                f"mean log p(c|x) = $-\\infty$, mean log p(x) = $-\\infty$")
    axes[1].text(0.03, 0.04, mle_text, transform=axes[1].transAxes, fontsize=8, va="bottom",
                 bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.9))
    axes[0].axvline(a_cls, color=BLUE, lw=0.8, ls="-.", alpha=0.6)
    axes[2].axvline(a_gen, color=BLUE, lw=0.8, ls="-.", alpha=0.6)
    # panel (c) zooms on the optimum; the inset shows the full range
    zoom = sweep[sweep.alpha_minus_one <= 4]
    axes[2].set_xlim(zoom.alpha_minus_one.min() / 3, zoom.alpha_minus_one.max() * 3)
    axes[2].set_ylim(zoom.mean_log_p_x.min() - 0.05, zoom.mean_log_p_x.max() + 0.25)
    ins = axes[2].inset_axes([0.12, 0.06, 0.42, 0.3])
    for split, color, ls in [("test", BLUE, "-"), ("validation", ORANGE, "--")]:
        s = sweep[sweep.split == split]
        ins.plot(s.alpha_minus_one, s.mean_log_p_x, ls, color=color, lw=1.2)
    ins.set_xscale("log")
    ins.tick_params(labelsize=6)
    ins.text(0.04, 0.08, "full range", transform=ins.transAxes, fontsize=7)
    ins.grid(alpha=0.25)
    fig.suptitle("Bernoulli naive Bayes: effect of the Beta($\\alpha$, $\\alpha$) prior "
                 "(dotted: $\\alpha = 2$; dash-dot: value selected on validation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- Gaussian NB

def gnb_sweep(splits, eps_values):
    rows = []
    for split, (F_fit, y_fit, F_eval, y_eval) in splits.items():
        mu, var, log_pi = gnb.fit(F_fit, y_fit, K)
        for eps in eps_values:
            lj = gnb.log_joint(F_eval, mu, var, log_pi, eps)
            true_log_post = bnb.normalize(lj)[np.arange(len(y_eval)), y_eval]
            rows.append({"eps": eps, "split": split, "accuracy": float(np.mean(lj.argmax(axis=1) == y_eval)),
                         "mean_log_p_c_given_x": float(np.mean(true_log_post)),
                         "frac_zero_variance": float(np.mean(var == 0))})
    return pd.DataFrame(rows)


def plot_gnb(df, eps_sel, eps_ref, bnb_ref, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for split, color, ls in [("test", BLUE, "-"), ("validation", ORANGE, "--")]:
        s = df[df.split == split]
        axes[0].plot(s.eps, s.accuracy, ls, color=color, lw=2, marker="o", ms=4, label=f"Gaussian NB, {split}")
        axes[1].plot(s.eps, -s.mean_log_p_c_given_x, ls, color=color, lw=2, marker="o", ms=4,
                     label=f"Gaussian NB, {split}")
    label = f"Bernoulli NB, test ($\\alpha - 1$ = {bnb_ref['alpha_minus_one']:.2g})"
    axes[0].axhline(bnb_ref["accuracy"], color=GRAY, lw=1.5, ls=":", label=label)
    axes[1].axhline(-bnb_ref["mean_log_p_c_given_x"], color=GRAY, lw=1.5, ls=":", label=label)
    for ax in axes:
        ax.axvline(eps_sel, color=BLUE, lw=0.8, ls="-.", alpha=0.6)
        ax.axvline(eps_ref, color=GRAY, lw=0.8, ls="-.", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"variance floor $\varepsilon$ (pixel intensities in [0, 1])")
        ax.grid(alpha=0.25)
    axes[1].set_yscale("log")
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("cross-entropy  -mean log p(c | x)  [nats]")
    axes[0].set_title("(a) classification accuracy", fontsize=11)
    axes[1].set_title("(b) class cross-entropy (lower is better)", fontsize=11)
    axes[0].legend(loc="lower center", frameon=False, fontsize=8)
    fig.suptitle("Gaussian naive Bayes on raw intensities vs Bernoulli naive Bayes on binarized pixels\n"
                 f"(dash-dot: $\\varepsilon$ selected on validation = {eps_sel:g} (blue), "
                 f"reference $\\varepsilon$ = {eps_ref:g} (gray))", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- samples

def isolated_fraction(B):
    """Fraction of on-pixels none of whose four neighbours is on."""
    x = B.reshape(-1, *data.IMAGE_SHAPE).astype(bool)
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    any_neighbour = p[:, :-2, 1:-1] | p[:, 2:, 1:-1] | p[:, 1:-1, :-2] | p[:, 1:-1, 2:]
    return float((x & ~any_neighbour).sum() / x.sum())


def plot_samples(real, theta, samples, spc, path):
    n_cols = K // 2
    rows = []
    for half in range(2):
        cls = np.arange(half * n_cols, (half + 1) * n_cols)
        rows.append(("test image", real[cls]))
        rows.append((r"$\theta_c$ (template)", theta[cls]))
        for s in range(spc):
            rows.append((f"sample {s + 1}", samples[cls * spc + s]))
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 0.8 * len(rows) + 0.9))
    fig.subplots_adjust(left=0.15, right=0.99, top=0.93, bottom=0.01, hspace=0.3)
    for ax, (label, imgs) in zip(axes, rows):
        show_montage(ax, imgs, n_cols, label=label)
    axes[0].set_title("classes 0-13", fontsize=10)
    axes[len(rows) // 2].set_title("classes 14-27", fontsize=10)
    fig.suptitle("Class templates and samples x ~ p(x | c) from Bernoulli naive Bayes (one column per class)",
                 fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- imputation

def imputation_scores(B, y, theta, log_pi, mask, rng):
    """Per-image imputation scores on the hidden pixels H of each image."""
    X = B.astype(np.float64)
    H = 1.0 - np.broadcast_to(np.asarray(mask, dtype=np.float64), X.shape)
    n_hidden = H.sum(axis=1)
    log_post = bnb.log_posterior(B, theta, log_pi, mask)
    p = np.exp(log_post) @ theta                    # p(x_j = 1 | x_E)
    p_known = theta[y]                              # p(x_j = 1 | c_true): class-known reference
    p_prior = np.exp(log_pi) @ theta                # p(x_j = 1): nothing observed

    def hidden_ce(q):
        return -((X * np.log(q) + (1 - X) * np.log1p(-q)) * H).sum(axis=1) / n_hidden

    ce = hidden_ce(p)
    ce_known = hidden_ce(p_known)
    # pitfall: plug in the template of a single class drawn from p(c | x_E) instead of averaging
    drawn = (np.exp(log_post).cumsum(axis=1) > rng.random((len(y), 1))).argmax(axis=1)
    return {
        "correct": (log_post.argmax(axis=1) == y).astype(float),
        "log_p_c_given_xE": log_post[np.arange(len(y)), y],
        "hidden_ce": ce,
        "hidden_ce_class_known": ce_known,
        "hidden_ce_prior": hidden_ce(np.broadcast_to(p_prior, X.shape)),
        "excess_ce": ce - ce_known,
        "hidden_ce_drawn_class": hidden_ce(theta[drawn]),
        "hidden_pixel_error": (((p > 0.5) != (X > 0.5)) * H).sum(axis=1) / n_hidden,
        "hidden_joint_ll_per_pixel": bnb.hidden_log_likelihood(B, theta, log_pi, mask) / n_hidden,
    }


def mask_for(kind, m, n_images, rng):
    if kind == "random":
        return masks.random_masks(n_images, m * m, D, rng)
    if kind == "lattice":
        return masks.lattice_mask(m, data.IMAGE_SHAPE)
    return masks.raster_mask(m * m, D)


def imputation_sweep(B, y, theta, log_pi, sizes, seed):
    rows, per_image = [], {}
    for m in sizes:
        rng = np.random.default_rng([seed, 2, m])
        for kind in ("random", "lattice", "raster"):
            sc = imputation_scores(B, y, theta, log_pi, mask_for(kind, m, len(B), rng), rng)
            per_image[(kind, m)] = sc
            row = {"mask": kind, "m": m, "n_observed": m * m, "frac_observed": m * m / D}
            row["class_accuracy"] = sc["correct"].mean()
            for key in ("log_p_c_given_xE", "hidden_ce", "hidden_ce_class_known", "hidden_ce_prior", "excess_ce",
                        "hidden_ce_drawn_class", "hidden_pixel_error", "hidden_joint_ll_per_pixel"):
                row[key] = sc[key].mean()
            row["class_accuracy_se"] = sc["correct"].std() / np.sqrt(len(B))
            row["hidden_ce_se"] = sc["hidden_ce"].std() / np.sqrt(len(B))
            rows.append(row)
    return pd.DataFrame(rows), per_image


def paired(a, b):
    d = a - b
    return {"diff": finite_or_str(d.mean()), "se": finite_or_str(d.std() / np.sqrt(len(d)))}


def plot_imputation_sweep(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for kind, color in MASK_COLORS.items():
        s = df[df["mask"] == kind]
        style = dict(color=color, lw=2, marker="o", ms=4, label=kind)
        axes[0].plot(100 * s.frac_observed, s.class_accuracy, **style)
        axes[1].plot(100 * s.frac_observed, s.hidden_ce, **style)
        axes[2].plot(100 * s.frac_observed, s.excess_ce, **style)
    s = df[df["mask"] == "random"]
    axes[1].plot(100 * s.frac_observed, s.hidden_ce_class_known, ls="--", color=GRAY, lw=1.5,
                 label="random mask, true class known")
    axes[2].axhline(0, color=GRAY, lw=1)
    axes[0].set_ylabel("accuracy of argmax p(c | x_E)")
    axes[1].set_ylabel("-mean log p(x_j | x_E) over hidden j  [nats]")
    axes[2].set_ylabel("hidden cross-entropy minus class-known  [nats]")
    axes[0].set_title("(a) class identified from the observed pixels", fontsize=11)
    axes[1].set_title("(b) imputation cross-entropy of hidden pixels", fontsize=11)
    axes[2].set_title("(c) excess over imputing with the true class", fontsize=11)
    for ax in axes:
        ax.set_xlabel("observed pixels [%]")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].legend(loc="upper center", frameon=False, fontsize=8)
    fig.suptitle("Imputation with Bernoulli naive Bayes on 10,000 test images: random vs evenly spaced masks",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_masks(sizes, seed, path):
    rng = np.random.default_rng([seed, 4])
    fig, axes = plt.subplots(len(sizes), 3, figsize=(7.5, 2.9 * len(sizes)))
    for r, m in enumerate(sizes):
        for c, kind in enumerate(("random", "lattice", "raster")):
            ax = axes[r, c]
            ax.imshow(mask_for(kind, m, 1, rng).reshape(data.IMAGE_SHAPE), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{kind}, {m * m} px ({m * m / D:.0%})", fontsize=9)
    fig.suptitle("Observation masks (black = observed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_imputation_examples(B, theta, log_pi, m, n_examples, seed, path):
    images = B[:n_examples]
    random_mask = masks.random_masks(n_examples, m * m, D, np.random.default_rng([seed, 3]))
    lattice = np.broadcast_to(masks.lattice_mask(m, data.IMAGE_SHAPE), images.shape)
    filled_random = bnb.impute(images, theta, log_pi, random_mask)
    filled_lattice = bnb.impute(images, theta, log_pi, lattice)
    n_cols = 10
    rows = []
    for start in range(0, n_examples, n_cols):
        sl = slice(start, start + n_cols)
        rows += [("test image", images[sl], None),
                 ("random: observed", images[sl], 1 - random_mask[sl]),
                 ("random: imputed", filled_random[sl], None),
                 ("lattice: observed", images[sl], 1 - lattice[sl]),
                 ("lattice: imputed", filled_lattice[sl], None)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 0.75 * len(rows) + 0.6))
    for ax, (label, imgs, hidden) in zip(axes, rows):
        show_montage(ax, imgs, n_cols, label=label, hidden=hidden)
    fig.suptitle(f"First {n_examples} test images, {m * m} of {D} pixels observed ({m * m / D:.0%}); "
                 "blue = hidden, gray level = p(x_j = 1 | x_E)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    args = parse_args()
    t0 = time.time()
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    X_train, y_train, X_test, y_test = data.load_unown(ROOT / args.data)
    B_train, B_test = data.binarize(X_train), data.binarize(X_test)
    perm = np.random.default_rng([args.seed, 0]).permutation(len(y_train))
    val_idx, fit_idx = perm[:args.n_val], perm[args.n_val:]
    summary = {"data": {"n_train": len(y_train), "n_test": len(y_test), "n_validation": args.n_val,
                        "binarize_rule": f"pixel on iff uint8 intensity >= {data.BINARIZE_THRESHOLD} (x/255 > 0.5)",
                        "frac_on_train": finite_or_str(B_train.mean())}}

    # 1. Bernoulli NB: MLE vs MAP and the Beta prior
    splits = {"validation": (B_train[fit_idx], y_train[fit_idx], B_train[val_idx], y_train[val_idx]),
              "test": (B_train, y_train, B_test, y_test)}
    df_alpha = alpha_sweep(splits, args.alpha_minus_one)
    df_alpha.to_csv(RES_DIR / "alpha_sweep.csv", index=False)
    val = df_alpha[(df_alpha.split == "validation") & (df_alpha.alpha_minus_one > 0)]
    a_gen = float(val.loc[val.mean_log_p_x.idxmax(), "alpha_minus_one"])
    a_cls = float(val.loc[val.accuracy.idxmax(), "alpha_minus_one"])
    plot_alpha_sweep(df_alpha, a_gen, a_cls, args.reference_alpha, FIG_DIR / "alpha_sweep.png")

    def test_row(a):
        theta, log_pi = bnb.fit(B_train, y_train, K, 1.0 + a)
        return {"alpha_minus_one": a, **bnb_scores(B_test, y_test, theta, log_pi)}

    bernoulli_rows = {"mle": test_row(0.0), "laplace": test_row(args.reference_alpha - 1.0),
                      "selected_for_classification": test_row(a_cls), "selected_for_generation": test_row(a_gen)}
    summary["bernoulli_test"] = {k: {kk: (vv if kk == "alpha_minus_one" else finite_or_str(vv)) for kk, vv in v.items()}
                                 for k, v in bernoulli_rows.items()}
    theta_mle, _ = bnb.fit(B_train, y_train, K, 1.0)
    summary["bernoulli_test"]["mle"]["n_theta_zero"] = int((theta_mle == 0).sum())
    summary["bernoulli_test"]["mle"]["n_theta_one"] = int((theta_mle == 1).sum())
    summary["bernoulli_test"]["mle"]["n_test_undefined"] = int(round(bernoulli_rows["mle"]["frac_undefined_posterior"] * len(y_test)))
    summary["bernoulli_test"]["mle"]["n_test_true_class_impossible"] = int(round(bernoulli_rows["mle"]["frac_true_class_impossible"] * len(y_test)))
    # pitfall: evaluating x log(theta) + (1 - x) log(1 - theta) literally gives 0 * (-inf) = NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        naive = (B_test @ np.log(theta_mle).T + (1 - B_test.astype(float)) @ np.log(1 - theta_mle).T)
    summary["bernoulli_test"]["mle"]["naive_0log0_frac_nan_rows"] = finite_or_str(np.isnan(naive).all(axis=1).mean())
    summary["bernoulli_test"]["mle"]["naive_0log0_accuracy"] = finite_or_str(np.mean(naive.argmax(axis=1) == y_test))
    summary["bernoulli_test"]["mle"]["test_frac_class_0"] = finite_or_str(np.mean(y_test == 0))
    summary["selection"] = {"alpha_minus_one_generation": a_gen, "alpha_minus_one_classification": a_cls}

    # 2. Gaussian NB with variance smoothing
    F_train, F_test = data.to_unit_interval(X_train), data.to_unit_interval(X_test)
    gsplits = {"validation": (F_train[fit_idx], y_train[fit_idx], F_train[val_idx], y_train[val_idx]),
               "test": (F_train, y_train, F_test, y_test)}
    eps_values = sorted(set(args.eps) | {args.reference_eps})
    df_eps = gnb_sweep(gsplits, eps_values)
    df_eps.to_csv(RES_DIR / "gaussian_eps_sweep.csv", index=False)
    gval = df_eps[df_eps.split == "validation"]
    eps_sel = float(gval.loc[gval.accuracy.idxmax(), "eps"])
    eps_ce = float(gval.loc[gval.mean_log_p_c_given_x.idxmax(), "eps"])
    gtest = df_eps[df_eps.split == "test"].set_index("eps")
    summary["selection"]["eps"] = eps_sel
    summary["gaussian_test"] = {
        name: {"eps": e, "accuracy": finite_or_str(gtest.loc[e, "accuracy"]),
               "mean_log_p_c_given_x": finite_or_str(gtest.loc[e, "mean_log_p_c_given_x"])}
        for name, e in [("selected", eps_sel), ("reference_small_eps", args.reference_eps),
                        ("min_validation_cross_entropy", eps_ce)]}
    summary["gaussian_test"]["frac_zero_variance"] = finite_or_str(gtest.frac_zero_variance.iloc[0])
    plot_gnb(df_eps, eps_sel, args.reference_eps, bernoulli_rows["selected_for_classification"],
             FIG_DIR / "gaussian_vs_bernoulli.png")
    del F_train, F_test, gsplits

    # 3. Samples from the generative model (prior selected for generation)
    theta, log_pi = bnb.fit(B_train, y_train, K, 1.0 + a_gen)
    rng = np.random.default_rng([args.seed, 1])
    cond_samples, _ = bnb.sample(theta, log_pi, None, rng,
                                 classes=np.repeat(np.arange(K), args.samples_per_class))
    real = B_test[[np.flatnonzero(y_test == c)[0] for c in range(K)]]
    plot_samples(real, theta, cond_samples, args.samples_per_class, FIG_DIR / "samples.png")
    generated, _ = bnb.sample(theta, log_pi, args.n_generated, rng)
    summary["samples"] = {"n_generated": args.n_generated,
                          "isolated_on_pixels_test": float(f"{isolated_fraction(B_test):.3g}"),
                          "isolated_on_pixels_samples": float(f"{isolated_fraction(generated):.3g}"),
                          "on_rate_test": finite_or_str(B_test.mean()),
                          "on_rate_samples": finite_or_str(generated.mean())}

    # 4. Imputation: random vs evenly spaced masks over the observed fraction
    df_imp, per_image = imputation_sweep(B_test, y_test, theta, log_pi, args.lattice_sizes, args.seed)
    df_imp.to_csv(RES_DIR / "imputation_sweep.csv", index=False)
    plot_imputation_sweep(df_imp, FIG_DIR / "imputation_vs_fraction.png")
    m = args.example_lattice
    plot_imputation_examples(B_test, theta, log_pi, m, args.n_examples, args.seed, FIG_DIR / "imputation_examples.png")
    plot_masks([m - 1, m], args.seed, FIG_DIR / "masks.png")
    imp = {"paired_random_minus_lattice": {}, "table": []}
    for mm in args.lattice_sizes:
        r, g = per_image[("random", mm)], per_image[("lattice", mm)]
        imp["paired_random_minus_lattice"][f"{mm * mm} px"] = {
            "class_accuracy": paired(r["correct"], g["correct"]),
            "excess_ce": paired(r["excess_ce"], g["excess_ce"])}
    cols = ["class_accuracy", "hidden_ce", "hidden_ce_class_known", "hidden_ce_prior", "excess_ce",
            "hidden_ce_drawn_class",
            "hidden_pixel_error", "hidden_joint_ll_per_pixel"]
    for row in df_imp[df_imp.m.isin([6, 7, 8, 10, 12, 14, 15, 20, 26])].to_dict("records"):
        imp["table"].append({"mask": row["mask"], "n_observed": row["n_observed"],
                             "frac_observed": finite_or_str(row["frac_observed"]),
                             **{c: finite_or_str(row[c]) for c in cols}})
    summary["imputation"] = imp

    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(RES_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
