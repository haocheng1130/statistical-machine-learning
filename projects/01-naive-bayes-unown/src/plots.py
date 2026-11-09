"""Small helpers for drawing image montages with matplotlib."""
import matplotlib.pyplot as plt
import numpy as np

from .data import IMAGE_SHAPE

PAD_COLOR = "#c8c8c8"     # gaps between thumbnails
HIDDEN_COLOR = "#86b6ef"  # pixels that were not observed


def montage(images, n_cols, pad=2, hidden=None):
    """Tile (N, D) images into one 2-D array.

    Padding is NaN; pixels flagged in `hidden` ((N, D) or (D,) 0/1 array) are set
    to -1, so that both can be given their own colors (cmap 'bad' and 'under').
    """
    images = np.asarray(images, dtype=float)
    if hidden is not None:
        images = np.where(np.broadcast_to(hidden, images.shape) > 0, -1.0, images)
    h, w = IMAGE_SHAPE
    n_rows = int(np.ceil(len(images) / n_cols))
    out = np.full((n_rows * (h + pad) + pad, n_cols * (w + pad) + pad), np.nan)
    for i, img in enumerate(images):
        r, c = divmod(i, n_cols)
        top, left = pad + r * (h + pad), pad + c * (w + pad)
        out[top:top + h, left:left + w] = img.reshape(h, w)
    return out


def show_montage(ax, images, n_cols, label=None, hidden=None):
    """Draw a montage (ink = 1 = black, background = 0 = white) with an optional row label."""
    cmap = plt.get_cmap("gray_r").copy()
    cmap.set_bad(PAD_COLOR)
    cmap.set_under(HIDDEN_COLOR)
    ax.imshow(montage(images, n_cols, hidden=hidden), cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label is not None:
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
