"k means step code"

import cachetools
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mtrain.scale import calculate_perspective_scale_factor, scale_fragment_with_mask
from mtrain.label_studio.crops.kmeans import (
    run_kmeans,
    plot_kmeans_labels,
    extract_kmeans_label,
    get_include_class,
)
import functools
from mtrain import horizon
from mtrain import superpose
from pathlib import Path


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

    def __init__(self, images, backdrop):
        self.images = self._load_images(images)
        if len(self.images) == 0:
            raise ValueError("No images found.")

        self.backdrop_bgr = cv2.imread(str(backdrop))
        if self.backdrop_bgr is None:
            raise ValueError(f"Could not load backdrop image: {backdrop}")

        self.idx = 0
        self.orig_crop_bgr = None
        self.original_img_bgr = None
        self.meta_data = None
        self.labels = None
        self.current_K = None

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
            value=True, description="Perspective?"
        )

        self.kind_dropdown = widgets.Dropdown(
            options=["scatter_small_white", "scatter_small_color", "big", "cluster"],
            value="scatter_small_white",
            description="Kind",
        )

        self.backdrop_position = widgets.Text(
            value="600,200",
            description="Backdrop pos",
            placeholder="r,c (e.g. 100,200)",
        )

        self.show_padded_fragments = widgets.Checkbox(
            value=False, description="Show Padded Fragments"
        )

        self.use_morphology = widgets.Checkbox(
            value=True, description="Use Morphology (Close Gaps)"
        )

        self.use_blur = widgets.Text(
            value="0", description="Blur Kernel size (0 for no blur)",
        )
        

        self.prev_button = widgets.Button(description="◀ Prev")
        self.next_button = widgets.Button(description="Next ▶")
        self.save_button = widgets.Button(description="Save", button_style="success")
        self.refresh_button = widgets.Button(description="Refresh")

        self.status = widgets.HTML()
        self.out = widgets.Output()

        self.k_slider.observe(self._on_k_change, names="value")
        self.show_padded_fragments.observe(self._on_padding_toggle, names="value")
        self.use_morphology.observe(self._on_morphology_toggle, names="value")
        self.use_blur.observe(self._redraw, names="value")

        self.refresh_button.on_click(lambda _: self._redraw())
        self.save_button.on_click(self._on_save)
        self.next_button.on_click(lambda b: self._move(+1))
        self.prev_button.on_click(lambda b: self._move(-1))

    def _move(self, step):
        self.idx = np.clip(self.idx + step, 0, len(self.images) - 1)
        self._load_current_image()

    def _load_current_image(self):
        img_path = self.images[self.idx]
        self.orig_crop_bgr = cv2.imread(str(img_path))
        if self.orig_crop_bgr is None:
            raise ValueError(f"Could not load image: {img_path}")
        meta_path = img_path.parent / "meta.json"
        if not meta_path.exists():
            raise ValueError(f"meta.json not found: {meta_path}")
        with open(meta_path, "r") as f:
            self.meta_data = json.load(f)
        if "original" not in self.meta_data or "path" not in self.meta_data["original"]:
            raise ValueError(f"Original image path not found in meta.json: {meta_path}")
        original_path = Path(self.meta_data["original"]["path"])
        if not original_path.exists():
            raise ValueError(f"Original image file not found: {original_path}")
        self.original_img_bgr = _load_original_image_cached(str(original_path))
        if self.original_img_bgr is None:
            raise ValueError(f"Could not load original image: {original_path}")
        self.kind_dropdown.value = self.meta_data.get("kind", "scatter_small_white")
        self.labels = None
        self.current_K = None
        self.status.value = f"<b>Image {self.idx + 1}/{len(self.images)}</b>: {img_path} ; {original_path}"
        self._on_k_change()

    def _on_k_change(self, change=None):
        self.save_button.disabled = True
        K = self.k_slider.value
        labels = run_kmeans(self.orig_crop_bgr.copy(), K=K)
        self.labels = labels
        self.current_K = K

        self._redraw()
        self.save_button.disabled = False

    def _redraw(self):
        with self.out:
            self.out.clear_output(wait=True)
            self._plot_current()

    def _on_padding_toggle(self, change=None):
        if self.labels is not None:
            self._redraw()

    def _on_morphology_toggle(self, change=None):
        if self.labels is not None:
            self._redraw()

    def _plot_original_img_with_box_on_frag(self, ax):
        # Top row, left: Original image with bounding box
        img_rgb = cv2.cvtColor(self.original_img_bgr, cv2.COLOR_BGR2RGB)
        # img_rgb = self.original_img_bgr.copy()
        bb = self.meta_data["bounding_box"]

        # Create padded rectangle around the bounding box
        rect_coords = _create_padded_rectangle(
            img_rgb.shape,
            bb["r"],
            bb["c"],
            bb["r"] + bb["r_len"],
            bb["c"] + bb["c_len"],
            padding=10,
        )
        rect_r1, rect_c1, rect_r2, rect_c2 = rect_coords

        ax.imshow(img_rgb)
        ax.set_title("Original Image")
        ax.axis("off")
        rect = Rectangle(
            (rect_c1, rect_r1),
            rect_c2 - rect_c1,
            rect_r2 - rect_r1,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        _draw_horizon_line(img_rgb, ax)

    def _get_processed_frag(self):
        # Get processed fragment with current selected labels
        include_cls = [int(x.strip()) for x in self.class_input.value.split(",")]
        mask = get_include_class(self.labels, include_cls)

        # Apply morphology if enabled
        if self.use_morphology.value:
            mask = _apply_morphology_closing(mask)

        # Apply the processed mask to the fragment
        processed_fragment_bgr = self.orig_crop_bgr.copy()
        processed_fragment_bgr[~mask] = 0
        return processed_fragment_bgr, mask

    def _scale_fragment(self, pos_r, processed_fragment_bgr, mask):
        # Calculate perspective scaling for the fragment
        bb = self.meta_data["bounding_box"]
        orig_center_y = bb["r"] + bb["r_len"] // 2
        new_center_y = pos_r + processed_fragment_bgr.shape[0] // 2

        # Calculate horizon lines for both images
        orig_horizon_y = _calculate_horizon(self.original_img_bgr)
        new_horizon_y = _calculate_horizon(self.backdrop_bgr)

        # we should only scale if perspective is true
        if self.perspective_checkbox.value:
            scale_factor = calculate_perspective_scale_factor(
                orig_center_y,
                new_center_y,
                orig_horizon_y,
                new_horizon_y,
                self.original_img_bgr.shape[0],
                self.backdrop_bgr.shape[0],
            )
        else:
            scale_factor = 1.0
        scaled_fragment_bgr, mask = scale_fragment_with_mask(
            processed_fragment_bgr, mask, scale_factor
        )
        return scaled_fragment_bgr, scale_factor, mask

    def _plot_backdrop(self, scaled_fragment_bgr, pos_r, pos_c, scale_factor, ax):
        # Place scaled fragment on backdrop and get actual coordinates
        backdrop_with_frag_bgr = self.backdrop_bgr.copy()
        
        start0, end0, start1, end1 = superpose.direct_copy(
            backdrop_with_frag_bgr, scaled_fragment_bgr, pos_r, pos_c
        )
        blurred = self._blur(backdrop_with_frag_bgr)
        backdrop_with_frag_bgr[start0:end0, start1:end1] = blurred[start0:end0, start1:end1]

        backdrop_rgb = cv2.cvtColor(backdrop_with_frag_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(backdrop_rgb)
        title = f"Backdrop + Fragment (Scale: {scale_factor:.2f})"
        ax.set_title(title)
        ax.axis("off")

        _draw_padded_rectangle(backdrop_rgb, start0, start1, end0, end1, 30, ax)
        _draw_horizon_line(backdrop_rgb, ax)
        return backdrop_rgb, start0, start1, end0, end1

    def _plot_only_backdrop_and_notice_invalid_superpose_frag(
        self, ax_backdrop, ax_superposed_frag
    ):
        backdrop_rgb = cv2.cvtColor(self.backdrop_bgr, cv2.COLOR_BGR2RGB)
        ax_backdrop.imshow(backdrop_rgb)
        ax_backdrop.set_title("Backdrop (Invalid Position)")
        ax_backdrop.axis("off")
        _draw_horizon_line(backdrop_rgb, ax_backdrop)
        ax_superposed_frag.axis("off")
        ax_superposed_frag.set_title("Invalid Position")

    def _blur(self, img):
        try:
            ksize = int(self.use_blur.value)
        except:
            print("parse error, bad blur size, keeping 0")
            ksize = 0

        if ksize > 0:
            return cv2.blur(img, (ksize,ksize))
        else:
            return img

    def _plot_superposed_fragment(self, backdrop_rgb, start0, start1, end0, end1, ax):
        # Top row, right: Show superposed fragment (exact or with padding)
        if self.show_padded_fragments.value:
            superposed_crop = _extract_fragment_with_padding(
                backdrop_rgb, start0, start1, end0, end1, padding=20
            )
            ax.set_title("Superposed Fragment (with Padding)")
        else:
            superposed_crop = backdrop_rgb[start0:end0, start1:end1]
            ax.set_title("Superposed Fragment (Exact)")
        ax.imshow(superposed_crop)
        ax.axis("off")

    def _plot_original_cropped_fragment(self, ax):
        # Bottom row, middle: Original crop fragment (exact or with padding from original image)
        if self.show_padded_fragments.value:
            # Extract the original fragment with padding from the original image
            bb = self.meta_data["bounding_box"]
            original_rgb = cv2.cvtColor(self.original_img_bgr, cv2.COLOR_BGR2RGB)
            original_fragment_padded = _extract_fragment_with_padding(
                original_rgb,
                bb["r"],
                bb["c"],
                bb["r"] + bb["r_len"],
                bb["c"] + bb["c_len"],
                padding=20,
            )
            ax.imshow(original_fragment_padded)
            ax.set_title("Original Fragment (with Padding)")
        else:
            ax.imshow(cv2.cvtColor(self.orig_crop_bgr, cv2.COLOR_BGR2RGB))
            ax.set_title("Original Crop Fragment (Exact)")
        ax.axis("off")

    def _safe_process_fragment(self):
        try:
            processed_fragment_bgr, processed_mask = self._get_processed_frag()
        except (ValueError, IndexError):
            # invalid parsing (user input bad)
            processed_fragment_bgr, processed_mask = None, None
        return processed_fragment_bgr, processed_mask

    def _parse_positions(self):
        try:
            pos_parts = self.backdrop_position.value.split(",")
            pos_r, pos_c = int(pos_parts[0].strip()), int(pos_parts[1].strip())
        except (ValueError, IndexError):
            pos_r, pos_c = None, None
        return pos_r, pos_c

    def _all_none_if_one_none(self, a, b, c, d):
        if a is None or b is None or c is None or d is None:
            return None, None, None, None
        else:
            return a, b, c, d

    def _plot_processed_fragment(self, processed_fragment, ax):
        # Bottom row, right: Processed crop fragment (with selected labels only)
        if processed_fragment is not None:
            # ax.imshow(processed_fragment)
            ax.imshow(cv2.cvtColor(processed_fragment, cv2.COLOR_BGR2RGB))
            title = "Processed Fragment (Selected Labels)"
            if self.use_morphology.value:
                title += " + Morphology"
            ax.set_title(title)
            ax.axis("off")
        else:
            ax.axis("off")
            ax.set_title("Invalid Labels")

    def _plot_current(self):
        # 2x3 grid: top row (original + backdrop), bottom row (labels + crop + superposed crop)
        processed_fragment_bgr, processed_mask = self._safe_process_fragment()
        pos_r, pos_c = self._parse_positions()
        processed_fragment_bgr, processed_mask, pos_r, pos_c = (
            self._all_none_if_one_none(
                processed_fragment_bgr, processed_mask, pos_r, pos_c
            )
        )

        # start plotting
        _, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 0,0: original image with superposed fragment
        self._plot_original_img_with_box_on_frag(axes[0, 0])

        # 0,1: backdrop with superposed fragment
        # 0,2: 0,1 but zoomed
        if processed_fragment_bgr is not None:
            # Parse backdrop position
            scaled_fragment_bgr, scale_factor, _ = self._scale_fragment(
                pos_r, processed_fragment_bgr, processed_mask
            )
            backdrop_rgb, start0, start1, end0, end1 = self._plot_backdrop(
                scaled_fragment_bgr, pos_r, pos_c, scale_factor, axes[0, 1]
            )
            self._plot_superposed_fragment(
                backdrop_rgb, start0, start1, end0, end1, axes[0, 2]
            )
        else:
            # parsing error in scale or scaling error
            self._plot_only_backdrop_and_notice_invalid_superpose_frag(
                axes[0, 1], axes[0, 2]
            )

        # 1,0: kmeans output
        plot_kmeans_labels(
            self.labels, ax=axes[1, 0], title=f"KMeans (K={self.current_K})"
        )
        # 1,1: kmeans processed crop given user input
        self._plot_original_cropped_fragment(axes[1, 1])

        # 1:2: Processed fragment, raw
        self._plot_processed_fragment(processed_fragment_bgr, axes[1, 2])

        plt.tight_layout()
        plt.show()

    def _on_save(self, button):
        if self.labels is None:
            print("Nothing to save.")
            return

        try:
            include_cls = [int(x.strip()) for x in self.class_input.value.split(",")]
        except ValueError:
            print("Invalid label input.")
            return

        mask = get_include_class(self.labels, include_cls)

        if self.use_morphology.value:
            mask = _apply_morphology_closing(mask)

        crop = self.orig_crop_bgr.copy()
        crop[~mask] = 0

        out_path = self.images[self.idx].parent / "proc.png"
        mask_path = self.images[self.idx].parent / "mask.json"
        meta_path = self.images[self.idx].parent / "meta.json"

        with open(mask_path, "w") as f:
            json.dump(mask.tolist(), f)
        cv2.imwrite(str(out_path), crop)

        self.meta_data["perspective"] = self.perspective_checkbox.value
        self.meta_data["kind"] = self.kind_dropdown.value

        with open(meta_path, "w") as f:
            json.dump(self.meta_data, f)

        with self.out:
            print(f"Saved → {out_path}")
            print(
                f"Updated meta.json with perspective={self.perspective_checkbox.value}, kind={self.kind_dropdown.value}"
            )

    def display(self):
        from IPython.display import display
        import ipywidgets as widgets

        nav = widgets.HBox([self.prev_button, self.next_button, self.refresh_button])
        footer = widgets.HBox([self.save_button, self.refresh_button])
        ui = widgets.VBox(
            [
                self.status,
                nav,
                self.k_slider,
                self.class_input,
                self.perspective_checkbox,
                self.kind_dropdown,
                self.backdrop_position,
                self.show_padded_fragments,
                self.use_morphology,
                self.use_blur,
                footer,
                self.out,
            ]
        )
        display(ui)


@cachetools.cached(cache={}, key=id)
def _calculate_horizon(img):
    return horizon.using_road_mask(img)


def _create_padded_rectangle(img_shape, r1, c1, r2, c2, padding=20):
    """
    Create rectangle coordinates with padding, bounded by image dimensions.

    img_shape: (height, width) of the image
    r1, c1, r2, c2: original rectangle coordinates
    padding: padding to add around the rectangle

    Returns: (rect_r1, rect_c1, rect_r2, rect_c2) with padding applied
    """
    rect_r1 = max(0, r1 - padding)
    rect_c1 = max(0, c1 - padding)
    rect_r2 = min(img_shape[0], r2 + padding)
    rect_c2 = min(img_shape[1], c2 + padding)

    return rect_r1, rect_c1, rect_r2, rect_c2


def _extract_fragment_with_padding(img, r1, c1, r2, c2, padding=20):
    padded_coords = _create_padded_rectangle(img.shape, r1, c1, r2, c2, padding)
    rect_r1, rect_c1, rect_r2, rect_c2 = padded_coords
    return img[rect_r1:rect_r2, rect_c1:rect_c2]


def _apply_morphology_closing(mask, kernel_size=3, iterations=1):
    # Convert boolean mask to uint8
    mask_uint8 = mask.astype(np.uint8) * 255

    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply closing (dilation followed by erosion)
    closed_mask = cv2.morphologyEx(
        mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations
    )

    # Convert back to boolean
    return closed_mask > 0


@functools.lru_cache(maxsize=10)
def _load_original_image_cached(path: str):
    return cv2.imread(path)


def _draw_horizon_line(img, ax):
    orig_horizon_y = _calculate_horizon(img)
    ax.axhline(y=orig_horizon_y, color="blue", linestyle="--", linewidth=2, alpha=0.7)
    ax.text(
        10,
        orig_horizon_y + 10,
        "Horizon",
        color="blue",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )


def _draw_padded_rectangle(img, start0, start1, end0, end1, padding, ax):
    rect_coords_padded = _create_padded_rectangle(
        img.shape, start0, start1, end0, end1, padding=padding
    )
    rect_r1, rect_c1, rect_r2, rect_c2 = rect_coords_padded
    rect = Rectangle(
        (rect_c1, rect_r1),
        rect_c2 - rect_c1,
        rect_r2 - rect_r1,
        linewidth=3,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
