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
        self.original_img = None
        self.meta_data = None
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

        self.perspective_checkbox = widgets.Checkbox(
            value=False, description="Perspective?"
        )

        self.kind_dropdown = widgets.Dropdown(
            options=["scatter_small_white", "scatter_small_color", "big", "cluster"],
            value="scatter_small_white",
            description="Kind",
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

        # Load meta.json and original image if available
        meta_path = img_path.parent / "meta.json"
        self.meta_data = None
        self.original_img = None

        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.meta_data = json.load(f)

            # Load original image if path exists
            if "original" in self.meta_data and "path" in self.meta_data["original"]:
                original_path = Path(self.meta_data["original"]["path"])
                if original_path.exists():
                    self.original_img = cv2.imread(str(original_path))

            # Load existing perspective and kind values if present
            if self.meta_data:
                self.perspective_checkbox.value = self.meta_data.get(
                    "perspective", False
                )
                self.kind_dropdown.value = self.meta_data.get(
                    "kind", "scatter_small_white"
                )

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
        # Determine number of subplots based on available images
        num_plots = 3 if self.original_img is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

        # axes[0].imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        # axes[1].imshow(cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB))
        # plot_kmeans_labels(
        #     self.labels, ax=axes[2], title=f"KMeans (K={self.current_K})"
        # )
        # Original image from meta.json if available
        plot_idx = 0
        if self.original_img is not None:
            img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            if self.meta_data:
                bb = self.meta_data["bounding_box"]
                img = cv2.rectangle(
                    img,
                    (bb["c"], bb["r"]),
                    (bb["c"] + bb["c_len"], bb["r"] + bb["r_len"]),
                    color=(255,0,0),
                    thickness=5,
                )
            axes[plot_idx].imshow(img)
            axes[plot_idx].set_title("Original Image")
            axes[plot_idx].axis("off")
            plot_idx += 1

        # Crop fragment
        axes[plot_idx].imshow(cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB))
        axes[plot_idx].set_title("Crop Fragment")
        axes[plot_idx].axis("off")
        plot_idx += 1

        # Labels
        plot_kmeans_labels(
            self.labels, ax=axes[plot_idx], title=f"KMeans (K={self.current_K})"
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
        meta_path = self.images[self.idx].parent / "meta.json"

        # Save mask
        with open(mask_path, "w") as f:
            json.dump(mask.tolist(), f)

        # Save processed image
        cv2.imwrite(str(out_path), crop)

        # Update meta.json with perspective and kind values
        if self.meta_data is not None:
            self.meta_data["perspective"] = self.perspective_checkbox.value
            self.meta_data["kind"] = self.kind_dropdown.value

            with open(meta_path, "w") as f:
                json.dump(self.meta_data, f)

        with self.out:
            print(f"Saved → {out_path}")
            print(
                f"Updated meta.json with perspective={self.perspective_checkbox.value}, kind={self.kind_dropdown.value}"
            )

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
                self.perspective_checkbox,
                self.kind_dropdown,
                self.save_button,
                self.out,
            ]
        )

        display(ui)
