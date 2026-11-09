"""How much does the approximate inference algorithm matter for TrueSkill ratings?

Regenerates every figure in figures/ and every number in results/:
  1. two-player toy posteriors: exact grid vs MALA, HMC, mean-field and full-covariance VI;
  2. 2500 decisive Premier League matches: HMC (reference), MALA, MALA with the
     1/(4 eta) proposal factor omitted, mean-field VI, full-covariance VI, and a
     short, unconverged SGD run of mean-field VI;
  3. shifting the prior mean.
Usage: python run.py [--quick] [--threads 2]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from scipy.stats import kendalltau

from src import plots
from src.data import load_premier_league, two_player_wins, win_matrix
from src.diagnostics import ess, split_rhat, summarize_chains
from src.exact import grid_posterior, kl_gaussian_to_grid
from src.model import make_log_joint
from src.samplers import hmc, mala
from src.vi import (FullRankGaussian, MeanFieldGaussian, elbo_estimate, fit_vi, log_evidence_is,
                    prob_greater)

PROJECT_DIR = Path(__file__).resolve().parent
FIG_DIR, RES_DIR = PROJECT_DIR / "figures", PROJECT_DIR / "results"
TOY_CASES = (("1-0", 1, 0), ("20-0", 20, 0), ("20-20", 20, 20))
PITFALL_CASE = "20-0"
PAIR = ("Arsenal", "Liverpool")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="fast smoke run with ~10x fewer iterations")
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (default: torch default)")
    p.add_argument("--seed", type=int, default=0, help="base random seed")
    p.add_argument("--data", type=Path, default=PROJECT_DIR / "data" / "premier-league-matches.csv")
    p.add_argument("--n-games", type=int, default=2500, help="most recent decisive matches to use")
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--mala-warmup", type=int, default=2000)
    p.add_argument("--mala-draws", type=int, default=40000)
    p.add_argument("--hmc-warmup", type=int, default=1000)
    p.add_argument("--hmc-draws", type=int, default=20000, help="long run: HMC is the reference")
    p.add_argument("--hmc-leapfrog", type=int, default=15)
    p.add_argument("--hmc-jitter", type=float, default=0.5, help="relative step-size jitter")
    p.add_argument("--wrong-eta", type=float, default=0.01,
                   help="fixed step size of the MALA run that omits the 1/(4 eta) factor")
    p.add_argument("--vi-iters", type=int, default=20000)
    p.add_argument("--vi-lr", type=float, default=0.02, help="initial Adam learning rate")
    p.add_argument("--vi-final-lr", type=float, default=1e-5, help="learning rate at the last iteration")
    p.add_argument("--vi-samples", type=int, default=32, help="Monte Carlo draws per SVI step")
    p.add_argument("--short-svi-iters", type=int, default=200)
    p.add_argument("--short-svi-lr", type=float, default=1e-4)
    p.add_argument("--short-svi-samples", type=int, default=50)
    p.add_argument("--elbo-samples", type=int, default=100000, help="draws for final ELBO estimates")
    p.add_argument("--is-samples", type=int, default=200000, help="draws for the log-evidence estimate")
    p.add_argument("--toy-mala-draws", type=int, default=10000)
    p.add_argument("--toy-hmc-draws", type=int, default=2500)
    p.add_argument("--toy-vi-iters", type=int, default=5000)
    p.add_argument("--prior-shift", type=float, default=5.0, help="alternative prior mean")
    args = p.parse_args()
    if args.quick:
        args.mala_warmup, args.mala_draws = 500, 4000
        args.hmc_warmup, args.hmc_draws = 200, 500
        args.vi_iters, args.elbo_samples, args.is_samples = 2000, 10000, 20000
        args.toy_mala_draws, args.toy_hmc_draws, args.toy_vi_iters = 2000, 500, 1000
    return args


def rnd(x, k=4):
    return float(round(float(x), k))


def sig(x, digits=3):
    """Round to significant digits (for small tail probabilities)."""
    return float(f"{float(x):.{digits}g}")


def prior_draws(n_chains, dim, seed):
    """Chain initializations drawn from the N(0, I) prior (over-dispersed relative to the posterior)."""
    return torch.randn(n_chains, dim, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def vi_fit(cls, log_joint, dim, args, iters, seed, prior_mean=0.0):
    q = cls(dim)
    history, seconds = fit_vi(q, log_joint, iters, args.vi_lr, args.vi_samples, seed=seed,
                              final_lr_fraction=args.vi_final_lr / args.vi_lr)
    elbo, elbo_se = elbo_estimate(q, log_joint, args.elbo_samples, torch.Generator().manual_seed(seed + 1))
    return {"q": q, "mean": q.mean.detach().numpy().copy(), "cov": q.covariance().numpy(),
            "history": history.numpy(), "seconds": seconds, "elbo": elbo, "elbo_se": elbo_se}


def chain_stats(res):
    draws = res.draws.numpy()
    rhat, ess_values = summarize_chains(draws)
    return {"draws": draws, "flat": draws.reshape(-1, draws.shape[-1]), "rhat": rhat, "ess": ess_values,
            "seconds": res.seconds, "n_grad": res.n_grad, "accept": float(res.accept_rate.mean()),
            "step": res.step_size}


def pairwise_from_draws(flat):
    n = flat.shape[1]
    out = np.empty((n, n))
    for i in range(n):
        out[i] = (flat[:, [i]] > flat).mean(0)
    return out


def pairwise_gaussian(mean, cov):
    var = np.diag(cov)[:, None] + np.diag(cov)[None, :] - 2 * cov
    np.fill_diagonal(var, 1.0)
    return ndtr((mean[:, None] - mean[None, :]) / np.sqrt(var))


# --------------------------------------------------------------------------- toy experiments
def run_toy(args):
    rows, fig_cases, pitfall = [], [], None
    for k, (name, a_wins, b_wins) in enumerate(TOY_CASES):
        f = make_log_joint(two_player_wins(a_wins, b_wins))
        post = grid_posterior(f)
        squared = grid_posterior(lambda z: 2 * f(z))  # p^2, the target of the pitfall chain

        def add(method, mean, cov, extra=None):
            sd = np.sqrt(np.diag(cov))
            row = {"case": name, "method": method, "mean_A": rnd(mean[0]), "mean_B": rnd(mean[1]),
                   "sd_A": rnd(sd[0]), "sd_B": rnd(sd[1]), "corr": rnd(cov[0, 1] / (sd[0] * sd[1]))}
            row.update(extra or {})
            rows.append(row)

        add("exact", post.mean.numpy(), post.cov.numpy(),
            {"P_A_better": rnd(post.prob_a_greater, 5), "P_B_better": sig(1 - post.prob_a_greater),
             "log_evidence": rnd(post.log_z)})
        add("exact p^2 (normalized)", squared.mean.numpy(), squared.cov.numpy(),
            {"P_A_better": rnd(squared.prob_a_greater, 5), "P_B_better": sig(1 - squared.prob_a_greater)})
        z0 = prior_draws(args.chains, 2, args.seed + 100 + k)
        runs = {
            "mala": mala(f, z0, args.toy_mala_draws // 10, args.toy_mala_draws, 0.1, seed=args.seed + 10 + k),
            "hmc": hmc(f, z0, args.toy_hmc_draws // 5, args.toy_hmc_draws, 0.1, 10, jitter=args.hmc_jitter,
                       seed=args.seed + 20 + k),
            "mala_omitted": mala(f, z0, args.toy_mala_draws // 10, args.toy_mala_draws, args.wrong_eta,
                                 adapt=False, log_q_scale=1.0, seed=args.seed + 30 + k),
        }
        stats = {}
        for method, res in runs.items():
            s = stats[method] = chain_stats(res)
            p_hat = (s["flat"][:, 0] > s["flat"][:, 1]).mean()
            add(method, s["flat"].mean(0), np.cov(s["flat"].T),
                {"P_A_better": rnd(p_hat, 5), "P_B_better": sig(1 - p_hat), "min_ess": round(float(s["ess"].min())),
                 "max_rhat": rnd(s["rhat"].max()), "accept_rate": rnd(s["accept"], 3),
                 "step_size": rnd(s["step"], 4)})
        fits = {}
        for method, cls in (("mfvi", MeanFieldGaussian), ("fcvi", FullRankGaussian)):
            fit = fits[method] = vi_fit(cls, f, 2, args, args.toy_vi_iters, seed=args.seed + 40 + k)
            mean_t, cov_t = torch.tensor(fit["mean"]), torch.tensor(fit["cov"])
            p_fit = prob_greater(mean_t, cov_t, 0, 1).item()
            add(method, fit["mean"], fit["cov"],
                {"P_A_better": rnd(p_fit, 5), "P_B_better": sig(1 - p_fit),
                 "elbo": rnd(fit["elbo"]), "kl_to_exact": rnd(kl_gaussian_to_grid(post, mean_t, cov_t))})
        flat = stats["hmc"]["flat"]
        fig_cases.append({"name": name, "post": post, "hmc": flat[:: max(1, len(flat) // 1500)],
                          "mfvi": (fits["mfvi"]["mean"], fits["mfvi"]["cov"]),
                          "fcvi": (fits["fcvi"]["mean"], fits["fcvi"]["cov"])})
        if name == PITFALL_CASE:
            axis = post.za[:, 0].numpy()
            width = axis[1] - axis[0]
            pitfall = {"axis": axis, "exact": post.weights.sum(1).numpy() / width,
                       "squared": squared.weights.sum(1).numpy() / width,
                       "right": stats["mala"]["flat"][:, 0], "wrong": stats["mala_omitted"]["flat"][:, 0]}
    pd.DataFrame(rows).to_csv(RES_DIR / "toy_summary.csv", index=False)
    plots.toy_posteriors(FIG_DIR / "toy_posteriors.png", fig_cases)
    return pitfall


# --------------------------------------------------------------------------- Premier League
def run_premier_league(args, pitfall):
    names, winners, losers, info = load_premier_league(args.data, args.n_games)
    n = len(names)
    wins = win_matrix(winners, losers, n)
    f = make_log_joint(wins)
    ia, ib = names.index(PAIR[0]), names.index(PAIR[1])
    z0 = prior_draws(args.chains, n, args.seed + 1)

    chains = {
        "hmc": chain_stats(hmc(f, z0, args.hmc_warmup, args.hmc_draws, 0.05, args.hmc_leapfrog,
                               jitter=args.hmc_jitter, seed=args.seed + 2)),
        "mala": chain_stats(mala(f, z0, args.mala_warmup, args.mala_draws, 0.01, seed=args.seed + 3)),
        "mala_omitted": chain_stats(mala(f, z0, args.mala_warmup, args.mala_draws, args.wrong_eta,
                                         adapt=False, log_q_scale=1.0, seed=args.seed + 4)),
    }
    fits = {
        "mfvi": vi_fit(MeanFieldGaussian, f, n, args, args.vi_iters, seed=args.seed + 5),
        "fcvi": vi_fit(FullRankGaussian, f, n, args, args.vi_iters, seed=args.seed + 6),
    }
    # The optimizer settings of a quick, unconverged fit: plain SGD with momentum.
    q_short = MeanFieldGaussian(n)
    hist, sec = fit_vi(q_short, f, args.short_svi_iters, args.short_svi_lr, args.short_svi_samples,
                       seed=args.seed + 7, optimizer="sgd", momentum=0.9)
    e, se = elbo_estimate(q_short, f, args.elbo_samples, torch.Generator().manual_seed(args.seed + 8))
    fits["svi_short"] = {"mean": q_short.mean.detach().numpy().copy(), "cov": q_short.covariance().numpy(),
                         "history": hist.numpy(), "seconds": sec, "elbo": e, "elbo_se": se}

    # Log evidence by importance sampling with the full-covariance fit as proposal.
    fc_mean, fc_cov = torch.tensor(fits["fcvi"]["mean"]), torch.tensor(fits["fcvi"]["cov"])
    log_z, kish = log_evidence_is(f, fc_mean, fc_cov, args.is_samples, seed=args.seed + 9)
    log_z_repeat, _ = log_evidence_is(f, fc_mean, fc_cov * 1.21, args.is_samples, seed=args.seed + 10)

    # Per-method summaries.
    summary = {}
    for key, s in chains.items():
        summary[key] = {"mean": s["flat"].mean(0), "cov": np.cov(s["flat"].T), "pairs": pairwise_from_draws(s["flat"])}
    for key, fit in fits.items():
        summary[key] = {"mean": fit["mean"], "cov": fit["cov"], "pairs": pairwise_gaussian(fit["mean"], fit["cov"])}
    for s in summary.values():
        cov = s["cov"]
        s["sd"] = np.sqrt(np.diag(cov))
        # sd of every skill difference z_i - z_j, and of the average skill (1/N) sum_i z_i
        s["diff_sd"] = np.sqrt(np.diag(cov)[:, None] + np.diag(cov)[None, :] - 2 * cov)
        s["avg_sd"] = math.sqrt(cov.sum()) / n
    ref = summary["hmc"]
    upper = np.triu_indices(n, 1)
    ref_rank = np.argsort(np.argsort(-ref["mean"])) + 1

    method_rows = []
    for key in ("hmc", "mala", "mala_omitted", "mfvi", "fcvi", "svi_short"):
        s = summary[key]
        rank = np.argsort(np.argsort(-s["mean"])) + 1
        err = np.abs(s["pairs"][upper] - ref["pairs"][upper])
        row = {"method": key, "label": plots.LABELS[key],
               "kendall_tau_vs_hmc": rnd(kendalltau(s["mean"], ref["mean"]).statistic),
               "teams_with_other_rank": int((rank != ref_rank).sum()),
               "max_abs_mean_diff": rnd(np.abs(s["mean"] - ref["mean"]).max()),
               "sd_ratio_median": rnd(np.median(s["sd"] / ref["sd"]), 3),
               "sd_ratio_min": rnd((s["sd"] / ref["sd"]).min(), 3),
               "sd_ratio_max": rnd((s["sd"] / ref["sd"]).max(), 3),
               "diff_sd_ratio_median": rnd(np.median(s["diff_sd"][upper] / ref["diff_sd"][upper]), 3),
               "diff_sd_ratio_min": rnd((s["diff_sd"][upper] / ref["diff_sd"][upper]).min(), 3),
               "diff_sd_ratio_max": rnd((s["diff_sd"][upper] / ref["diff_sd"][upper]).max(), 3),
               "average_skill_mean": rnd(s["mean"].mean()), "average_skill_sd": rnd(s["avg_sd"]),
               "pairwise_mean_abs_err": rnd(err.mean()), "pairwise_max_abs_err": rnd(err.max()),
               f"P_{PAIR[0]}_better": rnd(s["pairs"][ia, ib])}
        if key in chains:
            c = chains[key]
            indicator = (c["draws"][:, :, ia] > c["draws"][:, :, ib]).astype(float)
            p = indicator.mean()
            row.update({
                f"P_{PAIR[0]}_better_mcse": rnd(math.sqrt(p * (1 - p) / ess(indicator))),
                "seconds": rnd(c["seconds"], 2), "grad_evals_per_chain": c["n_grad"],
                "step_size": rnd(c["step"], 5), "accept_rate": rnd(c["accept"], 3),
                "max_rhat": rnd(c["rhat"].max()), "min_ess": round(float(c["ess"].min())),
                "median_ess": round(float(np.median(c["ess"]))),
                "min_ess_per_sec": rnd(c["ess"].min() / c["seconds"], 1),
                "min_ess_per_1000_grads": rnd(1000 * c["ess"].min() / (c["n_grad"] * args.chains), 2),
                "ess_average_skill": round(ess(c["draws"].mean(-1))),
                "rhat_average_skill": rnd(split_rhat(c["draws"].mean(-1))),
                f"ess_{PAIR[0]}_minus_{PAIR[1]}": round(ess(c["draws"][:, :, ia] - c["draws"][:, :, ib])),
            })
        else:
            fit = fits[key]
            row.update({"seconds": rnd(fit["seconds"], 2), "elbo": rnd(fit["elbo"], 3),
                        "elbo_se": rnd(fit["elbo_se"], 3), "kl_to_posterior": rnd(log_z - fit["elbo"], 3)})
        method_rows.append(row)
    methods = pd.DataFrame(method_rows)
    methods.to_csv(RES_DIR / "methods.csv", index=False)

    # Per-team table, with rank uncertainty from the HMC draws.
    flat = chains["hmc"]["flat"]
    draw_ranks = np.argsort(np.argsort(-flat, axis=1), axis=1) + 1
    team = pd.DataFrame({"team": names,
                         "wins": np.bincount(winners, minlength=n), "losses": np.bincount(losers, minlength=n)})
    for key in ("hmc", "mala", "mfvi", "fcvi", "mala_omitted", "svi_short"):
        team[f"{key}_mean"] = np.round(summary[key]["mean"], 4)
        team[f"{key}_sd"] = np.round(summary[key]["sd"], 4)
        team[f"{key}_rank"] = np.argsort(np.argsort(-summary[key]["mean"])) + 1
    team["hmc_P_top10"] = np.round((draw_ranks <= 10).mean(0), 4)
    team["hmc_rank_q05"] = np.quantile(draw_ranks, 0.05, axis=0).astype(int)
    team["hmc_rank_q95"] = np.quantile(draw_ranks, 0.95, axis=0).astype(int)
    team = team.sort_values("hmc_rank")
    team.to_csv(RES_DIR / "team_posteriors.csv", index=False)

    top10 = {key: [names[i] for i in np.argsort(-summary[key]["mean"])[:10]] for key in summary}
    pl = {"data": info, "average_skill_exact": {"mean": 0.0, "sd": rnd(1 / math.sqrt(n))},
          "log_evidence_is": rnd(log_z, 4), "is_kish_ess": round(kish),
          "is_samples": args.is_samples, "log_evidence_is_inflated_proposal": rnd(log_z_repeat, 4),
          "top10": top10}

    # Figures.
    order = np.argsort(-ref["mean"])
    plots.team_skills(FIG_DIR / "team_skills.png", names,
                      {k: (summary[k]["mean"], summary[k]["sd"]) for k in summary}, order,
                      ("hmc", "mfvi", "fcvi"))
    plots.elbo_curves(FIG_DIR / "elbo_curves.png",
                      {k: fits[k]["history"] for k in ("svi_short", "mfvi", "fcvi")}, log_z)
    plots.mcmc_traces(FIG_DIR / "mcmc_traces.png", {k: chains[k]["draws"] for k in ("mala", "hmc")},
                      (ia, ib), PAIR, n_show=min(1000, args.hmc_draws))
    plots.uncertainty(FIG_DIR / "uncertainty.png", ref["sd"],
                      {k: summary[k]["sd"] for k in ("mfvi", "fcvi", "mala_omitted", "svi_short")},
                      ref["pairs"][upper], {k: summary[k]["pairs"][upper] for k in ("mfvi", "fcvi", "mala_omitted")},
                      (flat[:: max(1, len(flat) // 3000)][:, [ia, ib]],
                       {k: (fits[k]["mean"][[ia, ib]], fits[k]["cov"][np.ix_([ia, ib], [ia, ib])])
                        for k in ("mfvi", "fcvi")}), PAIR)
    plots.mala_pitfall(FIG_DIR / "mala_pitfall.png", pitfall["axis"], pitfall["exact"], pitfall["squared"],
                       pitfall["right"], pitfall["wrong"], summary["mala"]["sd"] / ref["sd"],
                       summary["mala_omitted"]["sd"] / ref["sd"], PITFALL_CASE)
    return pl, wins, names, fits["mfvi"]


# --------------------------------------------------------------------------- prior shift
def run_prior_shift(args, pl_wins, names, mf_base):
    m = args.prior_shift
    out = {"prior_mean": m}
    # Exact, two players (record 20-0): the whole posterior moves by m.
    wins = two_player_wins(20, 0)
    base = grid_posterior(make_log_joint(wins))
    shifted = grid_posterior(make_log_joint(wins, prior_mean=m), center=m)
    out["toy_20-0_exact"] = {
        "mean_shift": [rnd(x, 6) for x in (shifted.mean - base.mean).tolist()],
        "max_abs_sd_change": sig((shifted.cov.diagonal().sqrt() - base.cov.diagonal().sqrt()).abs().max()),
        "P_B_better": [sig(1 - base.prob_a_greater), sig(1 - shifted.prob_a_greater)],
    }
    # Premier League: mean-field VI refitted from the same initialization (mean 0) with new random
    # draws, so the differences below include the optimization noise of SVI.
    fit = vi_fit(MeanFieldGaussian, make_log_joint(pl_wins, prior_mean=m), len(names), args, args.vi_iters,
                 seed=args.seed + 11)
    shift = fit["mean"] - mf_base["mean"]
    sd_base, sd_new = np.sqrt(np.diag(mf_base["cov"])), np.sqrt(np.diag(fit["cov"]))
    p_base, p_new = pairwise_gaussian(mf_base["mean"], mf_base["cov"]), pairwise_gaussian(fit["mean"], fit["cov"])
    ia, ib = names.index(PAIR[0]), names.index(PAIR[1])
    out["premier_league_mfvi"] = {
        "mean_shift_average": rnd(shift.mean()), "max_abs_mean_shift_error": sig(np.abs(shift - m).max()),
        "max_abs_sd_change": sig(np.abs(sd_new - sd_base).max()),
        "max_abs_pairwise_P_change": sig(np.abs(p_new - p_base).max()),
        f"P_{PAIR[0]}_better": [rnd(p_base[ia, ib]), rnd(p_new[ia, ib])],
    }
    with open(RES_DIR / "prior_shift.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    start = time.perf_counter()
    timings = {}
    t = time.perf_counter()
    pitfall = run_toy(args)
    timings["toy"] = rnd(time.perf_counter() - t, 1)
    t = time.perf_counter()
    pl, pl_wins, names, mf_base = run_premier_league(args, pitfall)
    timings["premier_league"] = rnd(time.perf_counter() - t, 1)
    t = time.perf_counter()
    run_prior_shift(args, pl_wins, names, mf_base)
    timings["prior_shift"] = rnd(time.perf_counter() - t, 1)
    timings["total"] = rnd(time.perf_counter() - start, 1)
    pl["runtime_seconds"] = timings
    pl["settings"] = {k: (str(v.relative_to(PROJECT_DIR)) if isinstance(v, Path) else v)
                      for k, v in vars(args).items()}
    pl["settings"]["torch_threads"] = torch.get_num_threads()
    with open(RES_DIR / "summary.json", "w") as fh:
        json.dump(pl, fh, indent=2)
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
