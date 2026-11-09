"""Mean field, Gibbs sampling, graph-cut MAP, and model helpers against exact answers."""
from pathlib import Path

import numpy as np
from scipy.special import expit

from src.data import flip_noise, load_binary_image, matched_beta
from src.exact import all_states, brute_force, graph_cut_map
from src.gibbs import gibbs_means, gibbs_sweep
from src.ising import (couplings, decide, log_potential, neighbor_field,
                       pseudo_likelihood_coupling)
from src.meanfield import elbo, mean_field

DATA = Path(__file__).resolve().parents[1] / "data" / "trc1l3gqu9651.png"


def test_brute_force_independent_pixels():
    h = np.random.default_rng(0).normal(size=(2, 3))
    exact = brute_force(h, 0.0)
    np.testing.assert_allclose(exact["p_plus"], expit(2 * h), atol=1e-12)
    np.testing.assert_allclose(exact["log_z"], np.log(2 * np.cosh(h)).sum(), atol=1e-12)


def test_neighbor_field_and_log_potential_against_loops():
    rng = np.random.default_rng(1)
    H, W = 3, 4
    h = rng.normal(size=(H, W))
    Jh, Jv = rng.normal(size=(H, W - 1)), rng.normal(size=(H - 1, W))
    x = rng.choice([-1.0, 1.0], size=(H, W))
    edges = [((r, c), (r, c + 1), Jh[r, c]) for r in range(H) for c in range(W - 1)]
    edges += [((r, c), (r + 1, c), Jv[r, c]) for r in range(H - 1) for c in range(W)]
    field = np.zeros((H, W))
    for a, b, J in edges:
        field[a] += J * x[b]
        field[b] += J * x[a]
    np.testing.assert_allclose(neighbor_field(x, Jh, Jv), field, atol=1e-12)
    expected = (h * x).sum() + sum(J * x[a] * x[b] for a, b, J in edges)
    assert abs(log_potential(x, h, Jh, Jv) - expected) < 1e-12


def test_decide_breaks_ties_with_observation():
    score = np.array([2.0, -0.5, 0.0, 1e-12])
    y = np.array([-1.0, 1.0, -1.0, -1.0])
    np.testing.assert_array_equal(decide(score, y), [1.0, -1.0, -1.0, -1.0])


def test_graph_cut_equals_brute_force_map():
    rng = np.random.default_rng(2)
    for _ in range(25):
        h = rng.normal(size=(3, 4))
        J = (rng.uniform(0, 1.5, size=(3, 3)), rng.uniform(0, 1.5, size=(2, 4)))
        np.testing.assert_array_equal(graph_cut_map(h, J), brute_force(h, J)["map"])


def test_graph_cut_on_full_image_is_locally_optimal():
    # On the full grid: no single-pixel flip of the graph-cut MAP increases log p~.
    x = load_binary_image(DATA)
    y = flip_noise(x, 0.2, np.random.default_rng(3))
    h, J = matched_beta(0.2) * y, 0.6
    Jh, Jv = couplings(J, h.shape)
    xmap = graph_cut_map(h, J)
    gain = -2 * xmap * (h + neighbor_field(xmap, Jh, Jv))  # change of log p~ when flipping x_i
    assert gain.max() <= 1e-9
    assert log_potential(xmap, h, Jh, Jv) >= log_potential(y, h, Jh, Jv)


def test_mean_field_closed_form_and_fixed_point():
    rng = np.random.default_rng(4)
    h = rng.normal(size=(4, 5))
    np.testing.assert_allclose(mean_field(h, 0.0)["mean"], np.tanh(h), atol=1e-14)
    out = mean_field(h, 0.5, tol=1e-12, max_iter=10000, track_elbo=True)
    Jh, Jv = couplings(0.5, h.shape)
    np.testing.assert_allclose(out["mean"], np.tanh(h + neighbor_field(out["mean"], Jh, Jv)),
                               atol=1e-10)
    assert np.all(np.diff(out["elbo"]) >= -1e-12)          # coordinate ascent
    assert out["elbo"][-1] <= brute_force(h, 0.5)["log_z"] + 1e-12  # lower bound on log Z
    assert abs(out["elbo"][-1] - elbo(out["mean"], h, Jh, Jv)) < 1e-12


def test_gibbs_marginals_match_exact():
    rng = np.random.default_rng(5)
    h = rng.normal(scale=0.5, size=(3, 3))
    exact = brute_force(h, 0.4)
    means = gibbs_means(h, 0.4, n_sweeps=20000, burn_in=200, rng=rng)
    np.testing.assert_allclose((1 + means) / 2, exact["p_plus"], atol=0.02)


def test_all_states_enumerates_every_configuration():
    X = all_states((2, 3))
    assert X.shape == (64, 2, 3)
    assert len({tuple(s.ravel()) for s in X}) == 64


def test_matched_beta_reproduces_flip_likelihood():
    for eps in (0.05, 0.2, 0.4):
        b = matched_beta(eps)
        z = 2 * np.cosh(b)
        assert abs(np.exp(b) / z - (1 - eps)) < 1e-12   # y == x
        assert abs(np.exp(-b) / z - eps) < 1e-12        # y != x


def test_flip_noise_rate_and_image():
    x = load_binary_image(DATA)
    assert x.shape == (96, 96) and set(np.unique(x)) == {-1.0, 1.0}
    y = flip_noise(x, 0.1, np.random.default_rng(6))
    rate = (y != x).mean()
    assert abs(rate - 0.1) < 4 * np.sqrt(0.1 * 0.9 / x.size)


def test_pseudo_likelihood_recovers_coupling():
    # Sample the field-free Ising prior with J = 0.3 (disordered phase) and refit J.
    rng = np.random.default_rng(7)
    h = np.zeros((64, 64))
    Jh, Jv = couplings(0.3, h.shape)
    x = rng.choice([-1.0, 1.0], size=h.shape)
    estimates = []
    for t in range(1, 501):
        x, _ = gibbs_sweep(x, h, Jh, Jv, rng)
        if t % 100 == 0:
            estimates.append(pseudo_likelihood_coupling(x))
    assert abs(np.mean(estimates) - 0.3) < 0.03
