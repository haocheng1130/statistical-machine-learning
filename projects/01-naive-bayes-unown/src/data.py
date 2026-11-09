"""Loading and preprocessing of the Unown-MNIST glyph data."""
import numpy as np

IMAGE_SHAPE = (28, 28)
N_CLASSES = 28
# A pixel is "on" iff its uint8 intensity is >= 128, i.e. iff intensity / 255 > 0.5.
# The glyphs are bright strokes on a black (0) background, so "on" means ink.
BINARIZE_THRESHOLD = 128


def load_unown(path):
    """Return (X_train, y_train, X_test, y_test) with images flattened to (N, 784) uint8."""
    with np.load(path) as d:
        X_train = d["X_train"].reshape(len(d["X_train"]), -1)
        X_test = d["X_test"].reshape(len(d["X_test"]), -1)
        return X_train, d["Y_train"].astype(np.int64), X_test, d["Y_test"].astype(np.int64)


def binarize(X):
    """Map uint8 intensities to {0, 1} (uint8) with the fixed threshold above."""
    return (X >= BINARIZE_THRESHOLD).astype(np.uint8)


def to_unit_interval(X):
    """Map uint8 intensities to [0, 1] (float32, to keep the 60000 x 784 array small)."""
    return X.astype(np.float32) / np.float32(255.0)
