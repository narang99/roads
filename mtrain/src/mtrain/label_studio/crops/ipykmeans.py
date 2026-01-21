"k means step code"
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from pathlib import Path


def run_kmeans(img_bgr, K=3):
    img = img_bgr
    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    label_img = labels.reshape(h, w)
    return label_img


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


class KMeansDatasetExplorer:
    """
    Interactive KMeans explorer over a dataset of images.

    Features:
    - Navigate through images (Prev / Next / Skip)
    - Adjust K per image
    - Visualize original + labels
    - Extract selected label(s)
    - Save result per image
    """

    def __init__(self, images):
        """
        images: list of paths OR directory path
        """
        self.images = self._load_images(images)
        if len(self.images) == 0:
            raise ValueError("No images found.")


        # Dataset state
        self.idx = 0
        self.img_bgr = None
        self.labels = None
        self.current_K = None

        # UI
        self._build_widgets()
        self._load_current_image()

    def _load_images(self, images):
        if isinstance(images, (str, Path)):
            images = sorted(Path(images).glob("*"))
        return [Path(p) for p in images]

    def _build_widgets(self):
        import ipywidgets as widgets

        self.k_slider = widgets.IntSlider(
            value=3, min=2, max=15, step=1, description="K", continuous_update=False
        )

        self.class_input = widgets.Text(
            value="0", description="Label(s)", placeholder="e.g. 0 or 1,2"
        )

        self.prev_button = widgets.Button(description="◀ Prev")
        self.next_button = widgets.Button(description="Next ▶")
        self.skip_button = widgets.Button(description="Skip ⏭")
        self.save_button = widgets.Button(description="Save", button_style="success")

        self.status = widgets.HTML()
        self.out = widgets.Output()

        self.k_slider.observe(self._on_k_change, names="value")
        self.save_button.on_click(self._on_save)
        self.next_button.on_click(lambda b: self._move(+1))
        self.prev_button.on_click(lambda b: self._move(-1))
        self.skip_button.on_click(lambda b: self._move(+1, skipped=True))

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------
    def _move(self, step, skipped=False):
        self.idx = np.clip(self.idx + step, 0, len(self.images) - 1)
        self._load_current_image()

    # --------------------------------------------------------
    # Load current image
    # --------------------------------------------------------
    def _load_current_image(self):
        img_path = self.images[self.idx]
        self.img_bgr = cv2.imread(str(img_path))
        if self.img_bgr is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Reset per-image state
        self.labels = None
        self.current_K = None

        self.status.value = (
            f"<b>Image {self.idx + 1}/{len(self.images)}</b>: {img_path.name}"
        )

        # Trigger clustering
        self._on_k_change()

    # --------------------------------------------------------
    # K slider → recompute + redraw
    # --------------------------------------------------------
    def _on_k_change(self, change=None):
        self.save_button.disabled = True

        K = self.k_slider.value
        labels = run_kmeans(self.img_bgr, K=K)

        self.labels = labels
        self.current_K = K

        with self.out:
            self.out.clear_output(wait=True)
            self._plot_current()

        self.save_button.disabled = False

    # --------------------------------------------------------
    # Plot original + labels
    # --------------------------------------------------------
    def _plot_current(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Original
        axes[0].imshow(cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Labels
        plot_kmeans_labels(
            self.labels, ax=axes[1], title=f"KMeans (K={self.current_K})"
        )

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # Save current extraction
    # --------------------------------------------------------
    def _on_save(self, button):
        if self.labels is None:
            print("Nothing to save.")
            return

        try:
            include_cls = [int(x.strip()) for x in self.class_input.value.split(",")]
        except ValueError:
            print("Invalid label input.")
            return

        crop = extract_kmeans_label(self.img_bgr, self.labels, include_cls)
        mask = get_include_class(self.labels, include_cls)

        out_path = self.images[self.idx].parent / "proc.png"
        mask_path = self.images[self.idx].parent / "mask.json"
        with open(mask_path, "w") as f:
            json.dump(mask.tolist(), f)

        cv2.imwrite(str(out_path), crop)

        with self.out:
            print(f"Saved → {out_path}")

    # --------------------------------------------------------
    # Display UI
    # --------------------------------------------------------
    def display(self):
        from IPython.display import display
        import ipywidgets as widgets

        nav = widgets.HBox([self.prev_button, self.next_button, self.skip_button])

        ui = widgets.VBox(
            [
                self.status,
                nav,
                self.k_slider,
                self.class_input,
                self.save_button,
                self.out,
            ]
        )

        display(ui)
