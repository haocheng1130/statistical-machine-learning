# Lecture Notes: Statistical Machine Learning

[`STA314_STA414_Statistical_ML_Lecture_Notes.pdf`](STA314_STA414_Statistical_ML_Lecture_Notes.pdf) (146 pages) combines two University of Toronto courses into one connected set of notes: STA 314 (Statistical Methods for Machine Learning I, Fall 2024) and STA 414/2104 (Probabilistic Machine Learning, Winter 2025).

The notes have three parts:

- **Foundations:** the learning problem, likelihood and Bayesian inference, and optimization.
- **Prediction:** linear regression, shrinkage, splines, classification, SVMs, kernels and Gaussian processes, trees and ensembles, and neural networks.
- **Probabilistic models and inference:** graphical models, exact inference, MCMC, variational inference, latent variable models, attention, and diffusion models.

The emphasis is on ideas and algorithms rather than proofs. Every algorithm is written as a boxed procedure, and remarks titled *Connection* link related topics across chapters.

## Build

Requires a TeX distribution with `latexmk` (e.g. TeX Live 2025).

```bash
latexmk -pdf -outdir=build main.tex
cp build/main.pdf STA314_STA414_Statistical_ML_Lecture_Notes.pdf
```

Sources: `main.tex`, `preamble.tex`, and one file per chapter in `chapters/`.
