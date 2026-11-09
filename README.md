# Statistical Machine Learning

Lecture notes on statistical and probabilistic machine learning, together with five small research projects that study topics from the notes in code. Each project is self-contained, verified by tests against exact answers, and reproducible with a single command.

## Lecture notes

[**Statistical Machine Learning — Lecture Notes**](notes/STA314_STA414_Statistical_ML_Lecture_Notes.pdf) (PDF, 146 pages) combines two University of Toronto courses into one connected text: STA 314 (Statistical Methods for Machine Learning I) and STA 414/2104 (Probabilistic Machine Learning).

The notes have three parts:

- **Foundations:** risk and the bias–variance trade-off, likelihood and Bayesian inference, and gradient-based optimization.
- **Prediction:** linear regression and model selection, shrinkage and Bayesian linear regression, splines, classification, support vector machines, kernels and Gaussian processes, trees and ensembles, and neural networks.
- **Probabilistic models and inference:** graphical models, exact inference and message passing, Monte Carlo and MCMC, variational inference, latent variable models (EM, PCA, VAEs), attention, and diffusion models.

The emphasis is on ideas and algorithms rather than proofs, and every algorithm is written as a boxed procedure. LaTeX sources and build instructions are in [`notes/`](notes/).

## Projects

| | Project | Question | Methods | Notes |
|---|---|---|---|---|
| 1 | [Naive Bayes on Unown glyphs](projects/01-naive-bayes-unown/) | What can a model that treats pixels as independent given the class do: classify, generate, and fill in missing pixels? | Bernoulli and Gaussian naive Bayes, MAP with a Beta prior, probabilistic imputation | Ch. 2, 7, 12 |
| 2 | [Reading conditional independence off a graph](projects/02-d-separation/) | Do the graphical d-separation rules agree with exact numerical independence tests? | Bayes-ball reachability, trail enumeration, exact joints with random tables | Ch. 12 |
| 3 | [Loopy BP for image denoising](projects/03-loopy-bp-denoising/) | How should the coupling and the evidence strength of an Ising model be set, and does the inference method matter? | Sum- and max-product loopy BP, mean field, Gibbs sampling, exact MAP by graph cut | Ch. 12–15 |
| 4 | [TrueSkill ratings of Premier League teams](projects/04-trueskill-premier-league/) | How much does the approximate inference algorithm matter for a Bayesian rating model? | Exact grid posteriors, MALA, HMC, mean-field and full-covariance variational inference, R-hat and ESS | Ch. 14, 15 |
| 5 | [A VAE on a noisy figure eight](projects/05-vae-figure-eight/) | What does a VAE learn on simple 2-D data, and how good a density model is it? | VAE with reparameterization, IWAE bound, exact latent quadrature, Gaussian mixture by EM | Ch. 14–16 |

Each project folder contains:

- `README.md`: a short research report (question, methods, experiments, results, discussion);
- `run.py`: regenerates every figure in `figures/` and every number in `results/`;
- `src/`: the implementation;
- `tests/`: pytest checks against brute force, closed forms, or exact posteriors.

## Getting started

```bash
conda create -n sml python=3.12
conda activate sml
pip install -r requirements.txt

cd projects/04-trueskill-premier-league
python -m pytest -q tests     # verify the implementation
python run.py                 # reproduce figures and results
```

Every project runs on a laptop CPU. Full runs take from under a minute to several minutes, and the longer ones provide a `--quick` flag for a fast smoke test. Running `--quick` overwrites the committed figures and results, so run the full `run.py` again afterwards to restore them. Each project README states its runtime.

## Repository layout

```
statistical-machine-learning/
├── notes/        LaTeX sources and PDF of the lecture notes
├── projects/     five research projects (01–05)
├── requirements.txt
└── LICENSE
```

## License

MIT; see [LICENSE](LICENSE).
