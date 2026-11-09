"""Observation masks (1 = observed pixel, 0 = hidden pixel) for imputation experiments."""
import numpy as np


def random_masks(n_images, n_observed, n_pixels, rng):
    """One independent uniformly random subset of exactly n_observed pixels per image, shape (N, D)."""
    order = np.argsort(rng.random((n_images, n_pixels)), axis=1)
    mask = np.zeros((n_images, n_pixels), dtype=np.uint8)
    np.put_along_axis(mask, order[:, :n_observed], 1, axis=1)
    return mask


def lattice_mask(m, shape=(28, 28)):
    """Evenly spaced m x m lattice: rows and columns at the centers of m equal bands, shape (D,)."""
    rows = np.floor((np.arange(m) + 0.5) * shape[0] / m).astype(int)
    cols = np.floor((np.arange(m) + 0.5) * shape[1] / m).astype(int)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[np.ix_(rows, cols)] = 1
    return mask.ravel()


def raster_mask(n_observed, n_pixels=784):
    """n_observed pixels evenly spaced in row-major (raster) order, shape (D,).

    This is 'evenly spaced' in one dimension only: when the stride divides the
    image width (for example every 4th pixel of a 28-pixel row) the observed
    pixels form whole columns rather than a 2-D grid.
    """
    mask = np.zeros(n_pixels, dtype=np.uint8)
    mask[np.floor(np.arange(n_observed) * n_pixels / n_observed).astype(int)] = 1
    return mask
