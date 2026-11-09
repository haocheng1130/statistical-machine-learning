"""Regenerate every figure in figures/ and every number in results/.

    python run.py            # full run
    python run.py --quick    # smoke run (fewer iterations, one seed, smaller evaluation)
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import data, gmm, plots, vae

ROOT = Path(__file__).resolve().parent

# The four VAE variants: the decoder-noise ablation (fixed / learned / learned per z)
# and the latent-dimension ablation (1 vs 2), all with the same networks.
CONFIGS = [
    {"name": "fixed sd 0.1", "latent_dim": 2, "noise": "fixed"},
    {"name": "learned sd", "latent_dim": 2, "noise": "learned"},
    {"name": "learned sd(z)", "latent_dim": 2, "noise": "learned_z"},
    {"name": "1-D latent", "latent_dim": 1, "noise": "learned"},
]

# Plotting window in data space (log-density maps).
EXTENT = (-1.35, 1.35, -0.85, 0.85)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="fast smoke run")
    p.add_argument("--noise-std", type=float, default=0.02, help="data noise sd")
    p.add_argument("--n-train", type=int, default=300, help="training points per arc")
    p.add_argument("--n-val", type=int, default=300, help="validation points per arc")
    p.add_argument("--n-test", type=int, default=1000, help="test points per arc")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=3, help="VAE training seeds 0..seeds-1")
    p.add_argument("--iters", type=int, default=20000, help="Adam iterations")
    p.add_argument("--batch-size", type=int, default=900)
    p.add_argument("--lr", type=float, default=3e-3, help="initial learning rate (cosine decay)")
    p.add_argument("--sigma-fixed", type=float, default=0.1, help="decoder sd of the fixed-noise VAE")
    p.add_argument("--elbo-samples", type=int, default=100, help="samples per point for ELBO evaluation")
    p.add_argument("--iwae-k", type=int, default=5000, help="importance samples for the IWAE estimate")
    p.add_argument("--quad-grid-2d", type=int, default=640, help="quadrature points per axis, 2-D latent")
    p.add_argument("--quad-grid-1d", type=int, default=8000, help="quadrature points, 1-D latent")
    p.add_argument("--quad-half-width", type=float, default=5.0, help="latent quadrature range [-L, L]")
    p.add_argument("--gmm-ks", type=str, default="1,2,3,4,6,8,12,16,20,24,32,48,64")
    p.add_argument("--gmm-restarts", type=int, default=5)
    p.add_argument("--gmm-seed", type=int, default=0)
    p.add_argument("--map-step", type=float, default=0.025, help="grid step of the log-density maps")
    p.add_argument("--workers", type=int, default=2, help="parallel processes for VAE runs")
    args = p.parse_args()
    if args.quick:
        args.iters, args.seeds, args.n_test = 1500, 1, 200
        args.iwae_k, args.elbo_samples = 200, 20
        args.quad_grid_2d, args.quad_grid_1d = 320, 4000
        args.gmm_ks, args.gmm_restarts, args.map_step = "1,4,16", 2, 0.05
    return args


def map_grid(step):
    gx = np.arange(EXTENT[0] + step / 2, EXTENT[1], step)
    gy = np.arange(EXTENT[2] + step / 2, EXTENT[3], step)
    xx, yy = np.meshgrid(gx, gy)
    return np.stack([xx.ravel(), yy.ravel()], axis=1), xx.shape


def summarize(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


# ----------------------------------------------------------------------------------------
# One VAE configuration and seed (runs in a worker process)
# ----------------------------------------------------------------------------------------

def fit_and_evaluate(job):
    cfg, seed, arr, a = job["cfg"], job["seed"], job["arrays"], job["args"]
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    x_train, x_val, x_test = (torch.tensor(arr[k], dtype=torch.float32) for k in ("train", "val", "test"))
    d = cfg["latent_dim"]
    model = vae.VAE(2, d, cfg["noise"], a["sigma_fixed"])
    t0 = time.perf_counter()
    history = vae.train(model, x_train, x_val, a["iters"], a["batch_size"], a["lr"], seed)
    train_seconds = time.perf_counter() - t0
    enc, dec = model.encoder, model.decoder
    n_grid = a["quad_grid_1d"] if d == 1 else a["quad_grid_2d"]

    def exact(x, grid=n_grid):
        return vae.log_marginal_quadrature(dec, x, d, grid, a["quad_half_width"]).numpy()

    with torch.no_grad():
        mean, log_std = enc(x_train)
        kl_dims = (0.5 * (mean**2 + torch.exp(2 * log_std) - 1 - 2 * log_std)).mean(0).numpy()
        sigma = torch.exp(dec(mean)[1]).mean(0).numpy()
        test_elbo = vae.elbo(enc, dec, x_test, a["elbo_samples"]).numpy()
        train_elbo = vae.elbo(enc, dec, x_train, a["elbo_samples"]).numpy()
    test_iwae = vae.iwae_bound(enc, dec, x_test, a["iwae_k"]).numpy()
    test_exact, train_exact = exact(x_test), exact(x_train)

    out = {
        "config": cfg["name"], "seed": seed, "latent_dim": d, "noise": cfg["noise"],
        "train_elbo": float(train_elbo.mean()), "train_exact": float(train_exact.mean()),
        "test_elbo": float(test_elbo.mean()), "test_iwae": float(test_iwae.mean()),
        "test_exact": float(test_exact.mean()),
        "test_exact_se": float(test_exact.std(ddof=1) / np.sqrt(len(test_exact))),
        "test_median_exact": float(np.median(test_exact)), "test_median_iwae": float(np.median(test_iwae)),
        "frac_test_iwae_gap_gt_1": float(np.mean(test_exact - test_iwae > 1.0)),
        "sigma_1": float(sigma[0]), "sigma_2": float(sigma[1]),
        "kl_per_dim_sorted": sorted(map(float, kl_dims), reverse=True),
        "train_seconds": train_seconds, "history": history,
    }
    if seed == 0:
        sub = x_test[: a["n_subset"]]
        ks = a["iwae_ks"]
        out["iwae_curve"] = [float(vae.iwae_bound(enc, dec, sub, k).mean()) for k in ks]
        out["iwae_curve_exact"] = float(test_exact[: a["n_subset"]].mean())
        fine = exact(sub[:200], 2 * n_grid)
        out["quad_refine_max_abs_diff"] = float(np.abs(fine - test_exact[:200]).max())
        out["quad_refine_mean_diff"] = float((fine - test_exact[:200]).mean())
        grid = torch.tensor(a["map_points"], dtype=torch.float32)
        out["density_map"] = exact(grid)
        out["test_exact_points"], out["test_iwae_points"] = test_exact, test_iwae
        out["state_dict"] = {k: v.clone() for k, v in model.state_dict().items()}
    return out


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------

def main():
    args = parse_args()
    t_start = time.perf_counter()
    for sub in ("data", "figures", "results"):
        (ROOT / sub).mkdir(exist_ok=True)
    plots.style()

    # Data ------------------------------------------------------------------------------
    rng = np.random.default_rng(args.data_seed)
    splits = {}
    for name, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        x, label, theta = data.sample(n, args.noise_std, rng)
        splits[name] = (x, label, theta)
        pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "arc": label, "theta": theta}).to_csv(
            ROOT / "data" / f"{name}.csv", index=False)
    x_tr, lab_tr, th_tr = splits["train"]
    x_va, x_te = splits["val"][0], splits["test"][0]
    truth = {k: float(data.log_density(splits[k][0], args.noise_std).mean()) for k in splits}
    print(f"true log-likelihood: {truth}")

    pts, shape = map_grid(args.map_step)
    true_map = data.log_density(pts, args.noise_std).reshape(shape)
    plots.data_figure(x_tr, lab_tr, true_map, EXTENT, ROOT / "figures" / "data.png")

    # Baselines: single Gaussian and GMMs -------------------------------------------------
    ks = [int(k) for k in args.gmm_ks.split(",")]
    gmm_rng = np.random.default_rng(args.gmm_seed)
    gmm_rows, gmm_params = [], {}
    for k in ks:
        params, hist = gmm.fit_gmm(x_tr, k, gmm_rng, n_init=args.gmm_restarts)
        gmm_params[k] = params
        gmm_rows.append({"K": k, "em_iterations": len(hist),
                         "train": float(gmm.log_density(x_tr, params).mean()),
                         "val": float(gmm.log_density(x_va, params).mean()),
                         "test": float(gmm.log_density(x_te, params).mean())})
    gmm_df = pd.DataFrame(gmm_rows)
    gmm_df.to_csv(ROOT / "results" / "gmm_selection.csv", index=False)
    k_star = int(gmm_df.loc[gmm_df["val"].idxmax(), "K"])
    gauss = gmm.fit_gaussian(x_tr)
    baselines = {
        "gaussian": {"train": float(gmm.log_density(x_tr, gauss).mean()),
                     "test": float(gmm.log_density(x_te, gauss).mean())},
        "gmm": {"K": k_star, **{c: float(gmm_df.loc[gmm_df.K == k_star, c].iloc[0]) for c in ("train", "val", "test")}},
    }
    print(f"baselines: {baselines}")

    # VAEs ------------------------------------------------------------------------------
    iwae_ks = [k for k in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if k <= 2 * args.iwae_k]
    shared = {"sigma_fixed": args.sigma_fixed, "iters": args.iters, "batch_size": args.batch_size,
              "lr": args.lr, "elbo_samples": args.elbo_samples, "iwae_k": args.iwae_k,
              "quad_grid_1d": args.quad_grid_1d, "quad_grid_2d": args.quad_grid_2d,
              "quad_half_width": args.quad_half_width, "n_subset": min(500, len(x_te)),
              "iwae_ks": iwae_ks, "map_points": pts}
    arrays = {"train": x_tr, "val": x_va, "test": x_te}
    jobs = [{"cfg": c, "seed": s, "arrays": arrays, "args": shared} for c in CONFIGS for s in range(args.seeds)]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            runs = list(ex.map(fit_and_evaluate, jobs))
    else:
        runs = [fit_and_evaluate(j) for j in jobs]
    for r in runs:
        print(f"  {r['config']:>14s} seed {r['seed']}: test ELBO {r['test_elbo']:.3f}  IWAE {r['test_iwae']:.3f}  "
              f"exact {r['test_exact']:.3f}  sd ({r['sigma_1']:.3f}, {r['sigma_2']:.3f})  {r['train_seconds']:.0f}s")

    scalar_keys = ["config", "seed", "latent_dim", "noise", "train_elbo", "train_exact", "test_elbo", "test_iwae",
                   "test_exact", "test_exact_se", "test_median_exact", "test_median_iwae", "frac_test_iwae_gap_gt_1",
                   "sigma_1", "sigma_2"]
    run_df = pd.DataFrame([{**{k: r[k] for k in scalar_keys},
                            **{f"kl_dim{i + 1}": v for i, v in enumerate(r["kl_per_dim_sorted"])}} for r in runs])
    run_df.to_csv(ROOT / "results" / "vae_runs.csv", index=False)
    first = {r["config"]: r for r in runs if r["seed"] == 0}
    names = [c["name"] for c in CONFIGS]

    vae_summary = {}
    for name in names:
        sel = run_df[run_df.config == name]
        vae_summary[name] = {m: summarize(sel[m]) for m in
                             ("train_elbo", "train_exact", "test_elbo", "test_iwae", "test_exact",
                              "test_median_exact", "test_median_iwae", "frac_test_iwae_gap_gt_1", "sigma_1", "sigma_2",
                              "kl_dim1", "kl_dim2") if m in sel and sel[m].notna().all()}
        r0 = first[name]
        vae_summary[name]["seed0_quadrature_refinement"] = {"max_abs_diff": r0["quad_refine_max_abs_diff"],
                                                            "mean_diff": r0["quad_refine_mean_diff"]}
    pd.DataFrame([{"config": n, "K": k, "iwae": v, "exact": first[n]["iwae_curve_exact"]}
                  for n in names for k, v in zip(iwae_ks, first[n]["iwae_curve"])]).to_csv(
        ROOT / "results" / "iwae_convergence.csv", index=False)

    # One row per density model: test log-likelihood per point (VAE: mean over seeds).
    table = [{"model": "true density", "train": truth["train"], "test_exact": truth["test"]},
             {"model": "single Gaussian", "train": baselines["gaussian"]["train"],
              "test_exact": baselines["gaussian"]["test"]},
             {"model": f"GMM (K = {k_star})", "train": baselines["gmm"]["train"],
              "test_exact": baselines["gmm"]["test"]}]
    for n in names:
        v = vae_summary[n]
        table.append({"model": f"VAE, {n}", "train": v["train_exact"]["mean"], "test_elbo": v["test_elbo"]["mean"],
                      "test_iwae": v["test_iwae"]["mean"], "test_exact": v["test_exact"]["mean"],
                      "test_exact_seed_min": v["test_exact"]["min"], "test_exact_seed_max": v["test_exact"]["max"]})
    pd.DataFrame(table).to_csv(ROOT / "results" / "density_comparison.csv", index=False, float_format="%.4f")

    summary = {
        "settings": {k: v for k, v in vars(args).items() if k not in ("workers",)},
        "true_log_likelihood": truth,
        "train_marginal_sd": [float(v) for v in x_tr.std(0)],
        "baselines": baselines,
        "vae": vae_summary,
    }
    with open(ROOT / "results" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Figures ---------------------------------------------------------------------------
    models = {}
    for c in CONFIGS:
        m = vae.VAE(2, c["latent_dim"], c["noise"], args.sigma_fixed)
        m.load_state_dict(first[c["name"]]["state_dict"])
        models[c["name"]] = m
    xt = torch.tensor(x_tr, dtype=torch.float32)

    plots.training_figure([first[n]["history"] for n in names], names, args.noise_std, args.sigma_fixed,
                          ROOT / "figures" / "training_curves.png")

    two_d = [n for n in names if models[n].latent_dim == 2]
    with torch.no_grad():
        encoded = [models[n].encoder(xt)[0].numpy() for n in two_d]
    plots.latent_figure(encoded, two_d, lab_tr, th_tr, ROOT / "figures" / "latent_space.png")

    torch.manual_seed(1234)
    means = [models[n].sample(1500, with_noise=False).numpy() for n in names]
    noisy = [models[n].sample(1500, with_noise=True).numpy() for n in names]
    plots.samples_figure(means, noisy, names, x_tr, ROOT / "figures" / "samples.png")

    warp_names = ["fixed sd 0.1", "learned sd"]
    warps = []
    gx, gy = np.meshgrid(np.linspace(-1.3, 1.3, 27), np.linspace(-0.8, 0.8, 17), indexing="ij")
    lx, ly = np.meshgrid(np.linspace(-3, 3, 25), np.linspace(-3, 3, 25), indexing="ij")
    with torch.no_grad():
        for n in warp_names:
            m = models[n]
            e = m.encoder(torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32))[0].numpy()
            dd = m.decoder(torch.tensor(np.stack([lx.ravel(), ly.ravel()], 1), dtype=torch.float32))[0].numpy()
            warps.append({"ex": e[:, 0].reshape(gx.shape), "ey": e[:, 1].reshape(gx.shape),
                          "dx": dd[:, 0].reshape(lx.shape), "dy": dd[:, 1].reshape(lx.shape),
                          "encoded": m.encoder(xt)[0].numpy(), "x_train": x_tr})
    plots.grid_warp_figure(warps, warp_names, ROOT / "figures" / "grid_warps.png")

    maps = [true_map, gmm.log_density(pts, gmm_params[k_star]).reshape(shape)]
    titles = [f"true density (test avg {truth['test']:.3f})",
              f"GMM, K = {k_star} (test avg {baselines['gmm']['test']:.3f})"]
    for n in names:
        maps.append(first[n]["density_map"].reshape(shape))
        titles.append(f"VAE, {n} (test avg {first[n]['test_exact']:.3f})")
    plots.density_figure(maps, titles, EXTENT, ROOT / "figures" / "density_maps.png")

    rows = [("true density", plots.INK, {"exact": (truth["test"],) * 3}),
            (f"GMM (K = {k_star})", plots.INK_2, {"exact": (baselines["gmm"]["test"],) * 3}),
            ("single Gaussian", plots.INK_2, {"exact": (baselines["gaussian"]["test"],) * 3})]
    for i, n in enumerate(names):
        s = vae_summary[n]
        rows.append((f"VAE, {n}", plots.SERIES[i],
                     {m: (s[f"test_{k}"]["mean"], s[f"test_{k}"]["min"], s[f"test_{k}"]["max"])
                      for m, k in (("ELBO", "elbo"), ("IWAE", "iwae"), ("exact", "exact"))}))
    plots.heldout_figure({r["K"]: (r["train"], r["val"], r["test"]) for r in gmm_rows}, k_star, rows,
                         truth["test"], ROOT / "figures" / "heldout_loglik.png")

    plots.iwae_figure(iwae_ks, [first[n]["iwae_curve"] for n in names],
                      [first[n]["iwae_curve_exact"] for n in names], names, ROOT / "figures" / "iwae_gap.png")

    m1 = models["1-D latent"]
    with torch.no_grad():
        z_line = torch.linspace(-4, 4, 801)[:, None]
        x_line = m1.decoder(z_line)[0].numpy()
        enc1 = m1.encoder(xt)[0].numpy()[:, 0]
    r1 = first["1-D latent"]
    plots.latent1_figure(z_line.numpy()[:, 0], x_line, x_tr, enc1, lab_tr, th_tr, x_te,
                         r1["test_exact_points"] - r1["test_iwae_points"], ROOT / "figures" / "latent1.png")

    print(f"done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
