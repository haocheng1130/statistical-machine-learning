"""MCMC convergence diagnostics for one scalar quantity.

Input arrays have shape (n_draws, n_chains). Both diagnostics use split chains
(each chain cut into halves), following Gelman et al. (BDA3, Sec. 11.4-11.5)
and the effective-sample-size estimator of Stan (Vehtari et al., 2021) without
rank normalization.
"""
import numpy as np


def _split(x: np.ndarray) -> np.ndarray:
    n = x.shape[0] // 2
    return np.concatenate([x[:n], x[x.shape[0] - n:]], axis=1)


def split_rhat(x: np.ndarray) -> float:
    """Potential scale reduction factor computed on split chains."""
    x = _split(np.asarray(x, dtype=float))
    n = x.shape[0]
    chain_means = x.mean(0)
    w = x.var(0, ddof=1).mean()
    b = n * chain_means.var(ddof=1)
    var_plus = (n - 1) / n * w + b / n
    return float(np.sqrt(var_plus / w))


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance of each column at lags 0..n-1, via FFT."""
    n = x.shape[0]
    centered = x - x.mean(0)
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(centered, n=size, axis=0)
    return np.fft.irfft(f * np.conj(f), n=size, axis=0)[:n] / n


def ess(x: np.ndarray) -> float:
    """Effective sample size of the pooled draws (Geyer's initial monotone sequence)."""
    x = _split(np.asarray(x, dtype=float))
    n, m = x.shape
    acov = _autocovariance(x)
    chain_var = acov[0] * n / (n - 1)
    w = chain_var.mean()
    var_plus = (n - 1) / n * w + (x.mean(0).var(ddof=1) if m > 1 else 0.0)
    rho = 1.0 - (w - acov.mean(1)) / var_plus
    rho[0] = 1.0
    # Sums of adjacent pairs are positive for a reversible chain; truncate at the
    # first non-positive pair and make the sequence monotone (Geyer, 1992).
    n_pairs = n // 2
    pairs = rho[: 2 * n_pairs].reshape(n_pairs, 2).sum(1)
    positive = pairs > 0
    k = n_pairs if positive.all() else int(np.argmin(positive))
    pairs = np.minimum.accumulate(pairs[:k])
    tau = -1.0 + 2.0 * pairs.sum()
    tau = max(tau, 1.0 / np.log10(n * m))  # cap ESS at n m log10(n m), as Stan does
    return float(n * m / tau)


def summarize_chains(draws: np.ndarray):
    """Per-coordinate split R-hat and ESS for draws of shape (n_draws, n_chains, dim)."""
    dim = draws.shape[-1]
    rhat = np.array([split_rhat(draws[:, :, k]) for k in range(dim)])
    ess_values = np.array([ess(draws[:, :, k]) for k in range(dim)])
    return rhat, ess_values
