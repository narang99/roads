"k means step code"

import cachetools
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mtrain.seg.cityscapes import CityScapesCls, get_cached_seg_former
import functools

from pathlib import Path


def copy_at(background, frag, start0, start1):
    mask = np.any(frag != 0, axis=-1)
    end0 = start0 + mask.shape[0]
    end1 = start1 + mask.shape[1]
    roi = background[start0:end0, start1:end1]
    roi_row_len = roi.shape[0]
    roi_col_len = roi.shape[1]
    mask = mask[:roi_row_len, :roi_col_len]
    frag = frag[:roi_row_len, :roi_col_len]
    roi[mask] = frag[mask]
    return start0, start0 + roi_row_len, start1, start1 + roi_col_len


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


def create_padded_rectangle(img_shape, r1, c1, r2, c2, padding=20):
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


def extract_fragment_with_padding(img, r1, c1, r2, c2, padding=20):
    """
    Extract a fragment from an image with padding around the specified region.
    
    img: source image (BGR or RGB)
    r1, c1, r2, c2: region coordinates
    padding: padding to add around the region
    
    Returns: cropped image with padding
    """
    padded_coords = create_padded_rectangle(img.shape, r1, c1, r2, c2, padding)
    rect_r1, rect_c1, rect_r2, rect_c2 = padded_coords
    return img[rect_r1:rect_r2, rect_c1:rect_c2]


def apply_morphology_closing(mask, kernel_size=3, iterations=1):
    """
    Apply morphological closing to fill gaps in a binary mask.
    
    mask: binary mask (boolean array)
    kernel_size: size of the morphological kernel
    iterations: number of iterations to apply
    
    Returns: processed mask with gaps filled
    """
    # Convert boolean mask to uint8
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply closing (dilation followed by erosion)
    closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Convert back to boolean
    return closed_mask > 0


@cachetools.cached(cache={}, key=id)
def calculate_horizon(img):
    # return calculate_horizon_simple(img.shape[0])
    # return calculate_horizon_vanishing_point(img)
    return calculate_horizon_road_edges(img)


def calculate_horizon_simple(img_height):
    """
    Simple horizon calculation: 30% from top (1/3 y).
    
    img_height: height of the image in pixels
    
    Returns: horizon line y-coordinate
    """
    return int(img_height * 0.3)


def calculate_horizon_vanishing_point(img):
    """
    Alternative horizon calculation using vanishing point detection.
    Uses Hough line transform to find dominant lines and estimate vanishing point.
    
    img: BGR image
    
    Returns: horizon line y-coordinate
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None:
        # Fallback to simple calculation
        return calculate_horizon_simple(img)
    
    # Find horizontal-ish lines and estimate horizon
    horizontal_lines = []
    for line in lines:
        rho, theta = line[0]
        # Consider lines that are roughly horizontal (within 30 degrees)
        if abs(theta - np.pi/2) < np.pi/6:
            y = rho / np.sin(theta) if np.sin(theta) != 0 else img.shape[0] * 0.3
            if 0 < y < img.shape[0]:
                horizontal_lines.append(y)
    
    if horizontal_lines:
        # Use median of horizontal lines as horizon estimate
        return int(np.median(horizontal_lines))
    else:
        # Fallback to simple calculation
        return calculate_horizon_simple(img)


def calculate_horizon_road_edges(img):
    model = get_cached_seg_former()
    pred = model.predict_bgr_image(img)
    road_mask = model.get_mask(pred, CityScapesCls.ROAD)
    return _calculate_horizon_road_edges(img, road_mask)


def _calculate_horizon_road_edges(img, road_mask):
    """
    Calculate horizon using the highest road coordinate.
    
    img: BGR image
    road_mask: boolean mask of the road area
    
    Returns: horizon line y-coordinate
    """
    # Find all road pixels
    road_coords = np.where(road_mask)
    
    if len(road_coords[0]) == 0:
        # No road pixels found, fallback to simple calculation
        return calculate_horizon_simple(img.shape[0])
    
    # Get the minimum y-coordinate (highest point) of road pixels
    highest_road_y = np.min(road_coords[0])
    
    return int(highest_road_y)


def line_intersection(line1, line2):
    """
    Find intersection point of two lines defined by their endpoints.
    
    line1: (x1, y1, x2, y2)
    line2: (x3, y3, x4, y4)
    
    Returns: (x, y) intersection point or None if lines are parallel
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Calculate line directions
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        # Lines are parallel
        return None
    
    # Calculate intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    # Calculate intersection coordinates
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


def calculate_perspective_scale_factor(orig_y, new_y, orig_horizon_y, new_horizon_y):
    """
    Calculate scale factor based on perspective and horizon lines.
    Objects farther from horizon (closer to it) should be smaller.
    
    orig_y: y-coordinate in original image
    new_y: y-coordinate in new image  
    orig_horizon_y: y-coordinate of horizon line in original image
    new_horizon_y: y-coordinate of horizon line in new image
    
    Returns: scale factor (1.0 = no scaling, <1.0 = smaller, >1.0 = larger)
    """
    # Distance from respective horizons
    orig_dist_from_horizon = abs(orig_y - orig_horizon_y)
    new_dist_from_horizon = abs(new_y - new_horizon_y)
    
    # Avoid division by zero
    if orig_dist_from_horizon == 0:
        orig_dist_from_horizon = 1
    if new_dist_from_horizon == 0:
        new_dist_from_horizon = 1
    
    # Scale factor based on relative distances from respective horizons
    scale_factor = new_dist_from_horizon / orig_dist_from_horizon
    
    # Clamp to reasonable range
    scale_factor = np.clip(scale_factor, 0.1, 3.0)
    
    return scale_factor


def scale_fragment_with_mask(fragment, mask, scale_factor):
    """
    Scale a fragment and its mask by the given factor.
    
    fragment: BGR image fragment
    mask: boolean mask for the fragment
    scale_factor: scaling factor
    
    Returns: scaled_fragment, scaled_mask
    """
    if abs(scale_factor - 1.0) < 0.01:
        # No meaningful scaling needed
        return fragment, mask
    
    # Calculate new dimensions
    old_h, old_w = fragment.shape[:2]
    new_w = int(old_w * scale_factor)
    new_h = int(old_h * scale_factor)
    
    if new_w <= 0 or new_h <= 0:
        # Return minimal fragment if scaling results in zero size
        return np.zeros((1, 1, 3), dtype=fragment.dtype), np.zeros((1, 1), dtype=bool)
    
    # Scale fragment
    scaled_fragment = cv2.resize(fragment, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Scale mask
    mask_uint8 = mask.astype(np.uint8) * 255
    scaled_mask_uint8 = cv2.resize(mask_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scaled_mask = scaled_mask_uint8 > 127
    
    return scaled_fragment, scaled_mask


@functools.lru_cache(maxsize=10)
def _load_original_image_cached(path: str):
    return cv2.imread(path)


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
        """
        images: list of paths OR directory path
        backdrop: path to backdrop image
        """
        self.images = self._load_images(images)
        if len(self.images) == 0:
            raise ValueError("No images found.")
        
        # Load backdrop image
        self.backdrop = cv2.imread(str(backdrop))
        if self.backdrop is None:
            raise ValueError(f"Could not load backdrop image: {backdrop}")

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

        self.backdrop_position = widgets.Text(
            value="600,200", description="Backdrop pos", placeholder="r,c (e.g. 100,200)"
        )

        self.show_padded_fragments = widgets.Checkbox(
            value=False, description="Show Padded Fragments"
        )

        self.use_morphology = widgets.Checkbox(
            value=True, description="Use Morphology (Close Gaps)"
        )

        self.prev_button = widgets.Button(description="◀ Prev")
        self.next_button = widgets.Button(description="Next ▶")
        self.skip_button = widgets.Button(description="Skip ⏭")
        self.save_button = widgets.Button(description="Save", button_style="success")

        self.status = widgets.HTML()
        self.out = widgets.Output()

        self.k_slider.observe(self._on_k_change, names="value")
        self.show_padded_fragments.observe(self._on_padding_toggle, names="value")
        self.use_morphology.observe(self._on_morphology_toggle, names="value")
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

        # Load meta.json and original image
        meta_path = img_path.parent / "meta.json"
        if not meta_path.exists():
            raise ValueError(f"meta.json not found: {meta_path}")
            
        with open(meta_path, "r") as f:
            self.meta_data = json.load(f)

        # Load original image
        if "original" not in self.meta_data or "path" not in self.meta_data["original"]:
            raise ValueError(f"Original image path not found in meta.json: {meta_path}")
            
        original_path = Path(self.meta_data["original"]["path"])
        if not original_path.exists():
            raise ValueError(f"Original image file not found: {original_path}")
            
        self.original_img = _load_original_image_cached(str(original_path))
        if self.original_img is None:
            raise ValueError(f"Could not load original image: {original_path}")


        # Load existing perspective and kind values
        self.perspective_checkbox.value = self.meta_data.get("perspective", False)
        self.kind_dropdown.value = self.meta_data.get("kind", "scatter_small_white")

        # Reset per-image state
        self.labels = None
        self.current_K = None

        self.status.value = (
            f"<b>Image {self.idx + 1}/{len(self.images)}</b>: {img_path.name} ; {original_path}"
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
    # Padding toggle → redraw
    # --------------------------------------------------------
    def _on_padding_toggle(self, change=None):
        if self.labels is not None:
            with self.out:
                self.out.clear_output(wait=True)
                self._plot_current()

    # --------------------------------------------------------
    # Morphology toggle → redraw
    # --------------------------------------------------------
    def _on_morphology_toggle(self, change=None):
        if self.labels is not None:
            with self.out:
                self.out.clear_output(wait=True)
                self._plot_current()

    # --------------------------------------------------------
    # Plot original + labels
    # --------------------------------------------------------
    def _plot_current(self):
        # 2x3 grid: top row (original + backdrop), bottom row (labels + crop + superposed crop)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        from matplotlib.patches import Rectangle

        # Top row, left: Original image with bounding box
        img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        bb = self.meta_data["bounding_box"]
        
        # Create padded rectangle around the bounding box
        rect_coords = create_padded_rectangle(
            img.shape, bb["r"], bb["c"], 
            bb["r"] + bb["r_len"], bb["c"] + bb["c_len"], 
            padding=10
        )
        rect_r1, rect_c1, rect_r2, rect_c2 = rect_coords
        
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        
        # Draw rectangle with padding around the bounding box
        rect = Rectangle((rect_c1, rect_r1), rect_c2 - rect_c1, rect_r2 - rect_r1,
                       linewidth=3, edgecolor='red', facecolor='none')
        axes[0, 0].add_patch(rect)
        
        # Draw horizon line on original image
        orig_horizon_y = calculate_horizon(self.original_img)
        axes[0, 0].axhline(y=orig_horizon_y, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        axes[0, 0].text(10, orig_horizon_y + 10, 'Horizon', color='blue', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Top row, middle and right: Backdrop with fragment
        try:
            # Parse backdrop position
            pos_parts = self.backdrop_position.value.split(',')
            pos_r, pos_c = int(pos_parts[0].strip()), int(pos_parts[1].strip())
            
            # Get processed fragment with current selected labels
            include_cls = [int(x.strip()) for x in self.class_input.value.split(",")]
            mask = get_include_class(self.labels, include_cls)
            
            # Apply morphology if enabled
            if self.use_morphology.value:
                mask = apply_morphology_closing(mask)
            
            processed_fragment = extract_kmeans_label(self.img_bgr, self.labels, include_cls)
            # Apply the processed mask to the fragment
            processed_fragment = self.img_bgr.copy()
            processed_fragment[~mask] = 0
            
            # Calculate perspective scaling for the fragment
            bb = self.meta_data["bounding_box"]
            orig_center_y = bb["r"] + bb["r_len"] // 2  # Center y of fragment in original image
            new_center_y = pos_r + processed_fragment.shape[0] // 2  # Center y in backdrop
            
            # Calculate horizon lines for both images
            orig_horizon_y = calculate_horizon(self.original_img)
            new_horizon_y = calculate_horizon(self.backdrop)
            
            scale_factor = calculate_perspective_scale_factor(orig_center_y, new_center_y, orig_horizon_y, new_horizon_y)
            scaled_fragment, scaled_mask = scale_fragment_with_mask(processed_fragment, mask, scale_factor)
            
            # Place scaled fragment on backdrop and get actual coordinates
            backdrop_with_frag = self.backdrop.copy()
            start0, end0, start1, end1 = copy_at(backdrop_with_frag, scaled_fragment, pos_r, pos_c)
            
            # Display full backdrop
            backdrop_rgb = cv2.cvtColor(backdrop_with_frag, cv2.COLOR_BGR2RGB)
            axes[0, 1].imshow(backdrop_rgb)
            title = f"Backdrop + Fragment (Scale: {scale_factor:.2f})"
            axes[0, 1].set_title(title)
            axes[0, 1].axis("off")
            
            # Draw padded rectangle around the placed fragment for visibility
            rect_coords_padded = create_padded_rectangle(
                self.backdrop.shape, start0, start1, end0, end1, padding=30
            )
            rect_r1, rect_c1, rect_r2, rect_c2 = rect_coords_padded
            rect = Rectangle((rect_c1, rect_r1), rect_c2 - rect_c1, rect_r2 - rect_r1,
                           linewidth=3, edgecolor='red', facecolor='none')
            axes[0, 1].add_patch(rect)
            
            # Draw horizon line on backdrop
            backdrop_horizon_y = calculate_horizon(self.backdrop)
            axes[0, 1].axhline(y=backdrop_horizon_y, color='blue', linestyle='--', linewidth=2, alpha=0.7)
            axes[0, 1].text(10, backdrop_horizon_y + 10, 'Horizon', color='blue', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # Top row, right: Show superposed fragment (exact or with padding)
            if self.show_padded_fragments.value:
                superposed_crop = extract_fragment_with_padding(backdrop_rgb, start0, start1, end0, end1, padding=20)
                axes[0, 2].set_title("Superposed Fragment (with Padding)")
            else:
                superposed_crop = backdrop_rgb[start0:end0, start1:end1]
                axes[0, 2].set_title("Superposed Fragment (Exact)")
            
            axes[0, 2].imshow(superposed_crop)
            axes[0, 2].axis("off")
            
        except (ValueError, IndexError):
            # If position parsing fails, show backdrop without fragment
            backdrop_rgb = cv2.cvtColor(self.backdrop, cv2.COLOR_BGR2RGB)
            axes[0, 1].imshow(backdrop_rgb)
            axes[0, 1].set_title("Backdrop (Invalid Position)")
            axes[0, 1].axis("off")
            
            # Draw horizon line on backdrop even in error case
            backdrop_horizon_y = calculate_horizon(self.backdrop)
            axes[0, 1].axhline(y=backdrop_horizon_y, color='blue', linestyle='--', linewidth=2, alpha=0.7)
            axes[0, 1].text(10, backdrop_horizon_y + 10, 'Horizon', color='blue', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            axes[0, 2].axis("off")
            axes[0, 2].set_title("Invalid Position")

        # Bottom row, left: KMeans labels
        plot_kmeans_labels(
            self.labels, ax=axes[1, 0], title=f"KMeans (K={self.current_K})"
        )

        # Bottom row, middle: Original crop fragment (exact or with padding from original image)
        if self.show_padded_fragments.value:
            # Extract the original fragment with padding from the original image
            bb = self.meta_data["bounding_box"]
            original_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            original_fragment_padded = extract_fragment_with_padding(
                original_rgb, bb["r"], bb["c"], 
                bb["r"] + bb["r_len"], bb["c"] + bb["c_len"], 
                padding=20
            )
            axes[1, 1].imshow(original_fragment_padded)
            axes[1, 1].set_title("Original Fragment (with Padding)")
        else:
            axes[1, 1].imshow(cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title("Original Crop Fragment (Exact)")
        axes[1, 1].axis("off")

        # Bottom row, right: Processed crop fragment (with selected labels only)
        try:
            include_cls = [int(x.strip()) for x in self.class_input.value.split(",")]
            mask = get_include_class(self.labels, include_cls)
            
            # Apply morphology if enabled
            if self.use_morphology.value:
                mask = apply_morphology_closing(mask)
            
            # Apply the processed mask to the fragment
            processed_fragment = self.img_bgr.copy()
            processed_fragment[~mask] = 0
            
            axes[1, 2].imshow(processed_fragment)
            # axes[1, 2].imshow(cv2.cvtColor(processed_fragment, cv2.COLOR_BGR2RGB))
            title = "Processed Fragment (Selected Labels)"
            if self.use_morphology.value:
                title += " + Morphology"
            axes[1, 2].set_title(title)
            axes[1, 2].axis("off")
        except (ValueError, IndexError):
            axes[1, 2].axis("off")
            axes[1, 2].set_title("Invalid Labels")

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

        mask = get_include_class(self.labels, include_cls)
        
        # Apply morphology if enabled
        if self.use_morphology.value:
            mask = apply_morphology_closing(mask)
        
        # Apply the processed mask to create the crop
        crop = self.img_bgr.copy()
        crop[~mask] = 0

        out_path = self.images[self.idx].parent / "proc.png"
        mask_path = self.images[self.idx].parent / "mask.json"
        meta_path = self.images[self.idx].parent / "meta.json"

        # Save mask
        with open(mask_path, "w") as f:
            json.dump(mask.tolist(), f)

        # Save processed image
        cv2.imwrite(str(out_path), crop)

        # Update meta.json with perspective and kind values
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
                self.backdrop_position,
                self.show_padded_fragments,
                self.use_morphology,
                self.save_button,
                self.out,
            ]
        )

        display(ui)
