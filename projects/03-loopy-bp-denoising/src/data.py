"""Image loading, flip noise, and the likelihood-matched evidence strength."""
from pathlib import Path

import numpy as np
from PIL import Image


def load_binary_image(path: Path, size: int = 96, threshold: int = 120) -> np.ndarray:
    """Load an image as a size x size array with values in {-1, +1} (+1 = white).

    The image is converted to 8-bit grayscale, resized with a Lanczos filter,
    and thresholded: gray level > threshold becomes +1, everything else -1.
    """
    with Image.open(path) as img:
        gray = img.convert("L").resize((size, size), Image.LANCZOS)
    return np.where(np.asarray(gray) > threshold, 1.0, -1.0)


def flip_noise(x: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Flip every pixel of a {-1, +1} image independently with probability eps."""
    flips = rng.random(x.shape) < eps
    return np.where(flips, -x, x)


def matched_beta(eps: float) -> float:
    """Evidence strength for which exp(beta * x * y) is the flip-noise likelihood.

    p(y | x) = (1 - eps) if y == x else eps, which is proportional to
    exp(beta * x * y) with beta = 0.5 * log((1 - eps) / eps).
    """
    return 0.5 * np.log((1.0 - eps) / eps)
