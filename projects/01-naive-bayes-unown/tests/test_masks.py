"""Observation masks have the advertised size and spacing."""
import numpy as np

from src import masks


def test_random_masks_have_exact_counts_and_uniform_coverage():
    m = masks.random_masks(4000, 100, 784, np.random.default_rng(0))
    assert np.all(m.sum(axis=1) == 100)
    coverage = m.mean(axis=0)
    p = 100 / 784
    assert np.all(np.abs(coverage - p) < 5 * np.sqrt(p * (1 - p) / 4000))


def test_lattice_mask_is_an_evenly_spaced_product_grid():
    for m in range(2, 28):
        mask = masks.lattice_mask(m).reshape(28, 28)
        assert mask.sum() == m * m
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        assert len(rows) == m and len(cols) == m
        gaps = np.diff(rows)
        assert gaps.max() - gaps.min() <= 1  # spacing differs by at most one pixel
        np.testing.assert_array_equal(mask, np.outer(mask.any(axis=1), mask.any(axis=0)))


def test_raster_mask_count_and_stripes():
    for n in (50, 196, 235, 676):
        assert masks.raster_mask(n).sum() == n
    # every 4th pixel of a 28-wide image: 7 whole columns
    stripes = masks.raster_mask(196).reshape(28, 28)
    assert np.all(stripes[:, ::4] == 1) and stripes.sum() == 196
