import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def run_kmeans(img_bgr, K=3):
    return kmeans_raw(img_bgr, K)
    # return kmeans_with_spatial(img_bgr, K)

def kmeans_raw(img_bgr, K=3):    
    img = img_bgr
    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    label_img = labels.reshape(h, w)
    return label_img


def kmeans_with_spatial(img_bgr, K=3, spatial_weight=0.5):
    h, w, _ = img_bgr.shape

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    xs = xs / w
    ys = ys / h

    features = np.concatenate([
        img_bgr.reshape(-1, 3) / 255.0,
        spatial_weight * xs.reshape(-1, 1),
        spatial_weight * ys.reshape(-1, 1)
    ], axis=1).astype(np.float32)

    _, labels, _ = cv2.kmeans(
        features, K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        5, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(h, w)
    return labels

def plot_kmeans_labels(labels, ax=None, title="labels"):
    """
    Display a labeled image on given axes with tab10 colormap
    and a discrete colorbar showing actual label values.

    labels: 2D numpy array of integer labels
    ax: matplotlib.axes.Axes object (optional)
    title: plot title
    """
    unique_labels = np.unique(labels)

    if ax is None:
        ax = plt.gca()  # use current axes if not provided

    # Discrete colormap for labels
    cmap = ListedColormap(plt.get_cmap("tab10").colors[: len(unique_labels)])
    norm = BoundaryNorm(np.arange(len(unique_labels) + 1) - 0.5, len(unique_labels))

    # Show image
    im = ax.imshow(labels, cmap=cmap, norm=norm, interpolation="nearest")

    # Colorbar (attached to the figure containing the axes)
    cbar = ax.figure.colorbar(im, ax=ax, ticks=range(len(unique_labels)))
    cbar.ax.set_yticklabels(unique_labels)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax, im, cbar


def extract_kmeans_label(orig, lbl, include_cls):
    mask = get_include_class(lbl, include_cls)
    data = orig.copy()
    data[~mask] = 0
    return data


def get_include_class(lbl, include):
    return np.isin(lbl, include)


def get_exclude_class(lbl, exclude):
    return ~np.isin(lbl, exclude)
