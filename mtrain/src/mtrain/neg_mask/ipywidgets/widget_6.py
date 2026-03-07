import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import ipywidgets as widgets
import numpy as np
from IPython.display import display
from PIL import Image

from mtrain.neg_mask.crops import Bbox
from mtrain.neg_mask.ipywidgets.utils import arr_to_png_bytes
from mtrain.neg_mask.ipywidgets.done_tracker import DoneTracker
from mtrain.neg_mask.ipywidgets.save_crops import save_crop_level, get_sample_dir

# ------------------------------------------------------------------
# Label constants (only trash and other for training)
# ------------------------------------------------------------------
LABEL_OTHER = 0
LABEL_TRASH = 1

_LABEL_FOLDER: dict[int, str] = {
    LABEL_OTHER: "other",
    LABEL_TRASH: "trash",
}


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class BboxAnnotation:
    """Represents annotation state for a single bounding box."""

    bbox: Bbox
    bbox_idx: int
    original_pred: int
    original_prob: float
    human_label: Optional[int] = None  # None = not annotated by human
    is_hard: bool = False
    is_annotated: bool = False

    @property
    def final_label(self) -> int:
        """Get the final label (human if available, otherwise model prediction)."""
        return self.human_label if self.human_label is not None else self.original_pred

    @property
    def is_hard_example(self) -> bool:
        """Check if this is a hard example (manually marked or model disagreement)."""
        return self.is_hard or (
            self.human_label is not None and self.human_label != self.original_pred
        )


# ------------------------------------------------------------------
# Mass Annotation Widget
# ------------------------------------------------------------------


class MassAnnotationWidget:
    """
    Widget for efficient mass annotation using model predictions.

    Shows entire image with probability overlays, allows clicking regions
    to review and annotate, saves only hard examples.
    """

    def __init__(
        self, output_dir: str | Path, learner, crop_pad: int = 220
    ):
        self._output_dir = Path(output_dir)
        self._learner = learner
        self._crop_pad = crop_pad

        # Create output directories
        for sub in ("crop_level/trash", "crop_level/other"):
            (self._output_dir / sub).mkdir(parents=True, exist_ok=True)

        # Done tracking
        self._done_tracker = DoneTracker(self._output_dir)

        # State - populated by ui()
        self._name: str = ""
        self._source_dir: Path | None = None
        self._image: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
        self._mask: np.ndarray = np.zeros((1, 1), dtype=bool)
        self._bboxes: List[Bbox] = []
        self._annotations: Dict[int, BboxAnnotation] = {}

        # Probability masks
        self._trash_prob_mask: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self._other_prob_mask: np.ndarray = np.zeros((1, 1), dtype=np.float32)

        # UI state
        self._selected_bbox_idx: Optional[int] = None
        self._show_trash_overlay = True
        self._show_other_overlay = True

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ui(
        self, source_dir: Path, bboxes: List[Bbox], image: np.ndarray, mask: np.ndarray
    ):
        """Display the mass annotation UI for one image."""
        self._name = source_dir.name
        self._source_dir = source_dir
        self._bboxes = bboxes
        self._image = image
        self._mask = mask.astype(bool)
        self._selected_bbox_idx = None
        self._annotations = {}

        # Run inference on all bboxes
        self._run_whole_image_inference()

        # Build and display UI
        self._build_ui()
        self._render()

    # ------------------------------------------------------------------
    # Inference processing
    # ------------------------------------------------------------------

    def _run_whole_image_inference(self):
        """Run inference on all bounding boxes and create probability masks."""
        if not self._bboxes or self._learner is None:
            h, w = self._mask.shape
            self._trash_prob_mask = np.zeros((h, w), dtype=np.float32)
            self._other_prob_mask = np.zeros((h, w), dtype=np.float32)
            return

        from mtrain.neg_mask.model.predict.predict_8ch import run_inference

        # Prepare crop data for batch inference
        crop_data_list = [(self._image, self._mask, bbox) for bbox in self._bboxes]

        # Run batch inference
        all_probs = run_inference(self._learner, crop_data_list)  # Shape: [N, C]

        # Create annotations from predictions
        for i, (bbox, probs) in enumerate(zip(self._bboxes, all_probs)):
            pred_class = probs.argmax().item()
            pred_prob = probs[pred_class].item()

            self._annotations[i] = BboxAnnotation(
                bbox=bbox, bbox_idx=i, original_pred=pred_class, original_prob=pred_prob
            )

        # Reconstruct probability masks
        self._reconstruct_probability_masks(all_probs.numpy())

    def _reconstruct_probability_masks(self, all_probs: np.ndarray):
        """Map bbox predictions back to full image coordinates."""
        h, w = self._mask.shape
        self._trash_prob_mask = np.zeros((h, w), dtype=np.float32)
        self._other_prob_mask = np.zeros((h, w), dtype=np.float32)

        for i, bbox in enumerate(self._bboxes):
            # Get probabilities for this bbox
            other_prob = all_probs[i, LABEL_OTHER].item()
            trash_prob = all_probs[i, LABEL_TRASH].item()

            # Apply to bbox region in full image
            bbox_mask = self._mask[bbox.y : bbox.y2, bbox.x : bbox.x2]
            self._other_prob_mask[bbox.y : bbox.y2, bbox.x : bbox.x2][bbox_mask] = (
                other_prob
            )
            self._trash_prob_mask[bbox.y : bbox.y2, bbox.x : bbox.x2][bbox_mask] = (
                trash_prob
            )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the widget UI components."""
        # Output widgets
        self._out_main_image = widgets.Output()
        self._out_tight_crop = widgets.Output()
        self._out_medium_crop = widgets.Output()

        # Toggle controls
        self._trash_overlay_toggle = widgets.ToggleButton(
            value=True,
            description="Hide Trash",  # Starts active, so shows "Hide"
            button_style="danger",  # Red style when active
            icon="eye",
            layout=widgets.Layout(width="120px"),
        )
        self._trash_overlay_toggle.observe(self._on_overlay_toggle, names="value")

        self._other_overlay_toggle = widgets.ToggleButton(
            value=True,
            description="Hide Other",  # Starts active, so shows "Hide"
            button_style="info",  # Blue style when active
            icon="eye",
            layout=widgets.Layout(width="120px"),
        )
        self._other_overlay_toggle.observe(self._on_overlay_toggle, names="value")

        # Annotation controls
        self._annotation_radio = widgets.RadioButtons(
            options=[("Other", LABEL_OTHER), ("Trash", LABEL_TRASH)],
            value=LABEL_OTHER,
            description="Label:",
            disabled=True,
        )
        self._annotation_radio.observe(self._on_annotation_change, names="value")

        self._hard_checkbox = widgets.Checkbox(
            value=False, description="Hard Example", disabled=True, indent=False
        )
        self._hard_checkbox.observe(self._on_hard_toggle, names="value")

        # Action buttons
        self._save_btn = widgets.Button(
            description="Save Hard Examples",
            button_style="success",
            icon="save",
            layout=widgets.Layout(width="180px"),
        )
        self._save_btn.on_click(self._on_save)

        # Status displays
        self._selection_status = widgets.Label(value="Click on a region to select it")
        self._model_status = widgets.Label(value="")
        self._stats_status = widgets.Label(value="")


        # Layout assembly
        overlay_controls = widgets.HBox(
            [self._trash_overlay_toggle, self._other_overlay_toggle],
            layout=widgets.Layout(gap="10px"),
        )

        annotation_panel = widgets.VBox(
            [
                widgets.HTML("<b>Selected Region:</b>"),
                self._selection_status,
                self._model_status,
                widgets.HTML("<b>Tight Crop:</b>"),
                self._out_tight_crop,
                widgets.HTML("<b>Medium Crop:</b>"),
                self._out_medium_crop,
                widgets.HTML("<b>Annotation:</b>"),
                self._annotation_radio,
                self._hard_checkbox,
                self._save_btn,
                self._stats_status,
            ],
            layout=widgets.Layout(padding="10px", width="350px"),
        )

        main_image_panel = widgets.VBox(
            [overlay_controls, self._out_main_image],
            layout=widgets.Layout(padding="10px"),
        )

        # Create tabs
        main_tab = widgets.HBox(
            [main_image_panel, annotation_panel], layout=widgets.Layout(gap="20px")
        )

        display(main_tab)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_overlay_toggle(self, change):
        """Handle overlay toggle changes."""
        self._show_trash_overlay = self._trash_overlay_toggle.value
        self._show_other_overlay = self._other_overlay_toggle.value

        # Update button appearance based on state
        if self._show_trash_overlay:
            self._trash_overlay_toggle.description = "Hide Trash"
            self._trash_overlay_toggle.button_style = "danger"
        else:
            self._trash_overlay_toggle.description = "Show Trash"
            self._trash_overlay_toggle.button_style = ""

        if self._show_other_overlay:
            self._other_overlay_toggle.description = "Hide Other"
            self._other_overlay_toggle.button_style = "info"
        else:
            self._other_overlay_toggle.description = "Show Other"
            self._other_overlay_toggle.button_style = ""

        self._render_main_image()

    def _on_annotation_change(self, change):
        """Handle annotation radio button changes."""
        if self._selected_bbox_idx is not None:
            annotation = self._annotations[self._selected_bbox_idx]
            new_label = change["new"]
            annotation.human_label = new_label
            annotation.is_annotated = True
            self._update_selection_display()
            self._update_stats()

    def _on_hard_toggle(self, change):
        """Handle hard example checkbox changes."""
        if self._selected_bbox_idx is not None:
            annotation = self._annotations[self._selected_bbox_idx]
            annotation.is_hard = change["new"]
            self._update_stats()

    def _on_save(self, button):
        """Handle save button clicks."""
        self._save_hard_examples()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self):
        """Render all UI components."""
        self._render_main_image()
        self._update_stats()
        self._clear_selection()

    def _render_main_image(self):
        """Render the main image with probability overlays."""
        # Start with original image
        display_image = self._image.copy()

        # Create prediction overlay based on which class has higher probability
        if (
            (self._show_trash_overlay or self._show_other_overlay)
            and self._trash_prob_mask is not None
            and self._other_prob_mask is not None
        ):
            overlay = self._create_prediction_overlay()
            display_image = self._blend_overlay(display_image, overlay)

        # Highlight selected bbox if any
        if self._selected_bbox_idx is not None:
            bbox = self._bboxes[self._selected_bbox_idx]
            cv2.rectangle(
                display_image,
                (bbox.x, bbox.y),
                (bbox.x2, bbox.y2),
                (255, 255, 0),  # Yellow highlight
                3,
            )

        # Enable click interaction by making image clickable
        # Note: This is a simplified approach - in practice, you'd need to handle
        # image coordinates and clicks through custom JavaScript widgets
        self._out_main_image.clear_output(wait=True)
        with self._out_main_image:
            img_widget = widgets.Image(
                value=arr_to_png_bytes(display_image),
                format="png",
                layout=widgets.Layout(max_width="800px"),
            )
            display(img_widget)

            # Add click handler instruction
            click_instruction = widgets.HTML(
                value="<i>Click implementation would require custom JavaScript widget for coordinate detection</i>"
            )
            display(click_instruction)

    def _create_prediction_overlay(self) -> np.ndarray:
        """Create overlay showing the predicted class (highest probability) for each pixel."""
        h, w = self._mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Find pixels where we have predictions
        has_prediction = (self._trash_prob_mask > 0) | (self._other_prob_mask > 0)

        # For pixels with predictions, color based on which class has higher probability
        if self._show_trash_overlay and self._show_other_overlay:
            # Show both: red for trash, green for other
            trash_wins = (
                self._trash_prob_mask >= self._other_prob_mask
            ) & has_prediction
            other_wins = (
                self._other_prob_mask > self._trash_prob_mask
            ) & has_prediction
            overlay[trash_wins] = (255, 0, 0)  # Red for trash
            overlay[other_wins] = (0, 255, 0)  # Green for other
        elif self._show_trash_overlay:
            # Show only trash predictions
            trash_regions = (
                self._trash_prob_mask >= self._other_prob_mask
            ) & has_prediction
            overlay[trash_regions] = (255, 0, 0)  # Red for trash
        elif self._show_other_overlay:
            # Show only other predictions
            other_regions = (
                self._other_prob_mask > self._trash_prob_mask
            ) & has_prediction
            overlay[other_regions] = (0, 255, 0)  # Green for other

        return overlay

    def _create_probability_overlay(
        self, prob_mask: np.ndarray, color: tuple, alpha: float = 0.4
    ) -> np.ndarray:
        """Create a colored overlay from a probability mask."""
        h, w = prob_mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Apply color where probability > 0
        mask = prob_mask > 0
        overlay[mask] = color

        return overlay

    def _blend_overlay(
        self, base_image: np.ndarray, overlay: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """Blend an overlay onto the base image."""
        # Create alpha mask from overlay
        overlay_mask = (overlay.sum(axis=2) > 0).astype(np.float32)
        overlay_mask = np.stack([overlay_mask] * 3, axis=2)

        # Blend
        result = base_image.copy().astype(np.float32)
        overlay_f = overlay.astype(np.float32)

        result = result * (1 - alpha * overlay_mask) + overlay_f * alpha * overlay_mask

        return result.astype(np.uint8)

    # ------------------------------------------------------------------
    # Selection and detail view
    # ------------------------------------------------------------------

    def select_bbox(self, bbox_idx: int):
        """Select a bounding box for annotation (public method for manual selection)."""
        if 0 <= bbox_idx < len(self._bboxes):
            self._selected_bbox_idx = bbox_idx
            self._update_selection_display()
            self._render_main_image()

    def _clear_selection(self):
        """Clear the current selection."""
        self._selected_bbox_idx = None
        self._annotation_radio.disabled = True
        self._hard_checkbox.disabled = True
        self._selection_status.value = "Click on a region to select it"
        self._model_status.value = ""

        # Clear crop displays
        for output in [self._out_tight_crop, self._out_medium_crop]:
            output.clear_output()

    def _update_selection_display(self):
        """Update the display for the currently selected bbox."""
        if self._selected_bbox_idx is None:
            self._clear_selection()
            return

        bbox_idx = self._selected_bbox_idx
        annotation = self._annotations[bbox_idx]
        bbox = annotation.bbox

        # Enable annotation controls
        self._annotation_radio.disabled = False
        self._hard_checkbox.disabled = False

        # Update status
        self._selection_status.value = (
            f"Region {bbox_idx + 1}/{len(self._bboxes)} selected"
        )
        pred_name = "Trash" if annotation.original_pred == LABEL_TRASH else "Other"
        self._model_status.value = (
            f"Model: {pred_name} ({annotation.original_prob:.2f})"
        )

        # Update controls to match current state
        current_label = annotation.final_label
        self._annotation_radio.value = current_label
        self._hard_checkbox.value = annotation.is_hard_example

        # Render crop details
        self._render_crop_details(bbox)

    def _render_crop_details(self, bbox: Bbox):
        """Render tight and medium crop details for the selected bbox."""
        from mtrain.neg_mask.model.crop_level_dataset import CropLevelDataset2Chan
        from mtrain.neg_mask.model.predict.predict_8ch import _prepare_8_channel_tensor

        # Get the 8-channel tensor and denormalize to get crops
        tensor_8ch = _prepare_8_channel_tensor(self._image, self._mask, bbox)
        pairs = CropLevelDataset2Chan.denormalize(tensor_8ch)

        tight_img, tight_mask = pairs[0]
        medium_img, medium_mask = pairs[1]

        # Render tight crop
        self._out_tight_crop.clear_output(wait=True)
        with self._out_tight_crop:
            display(
                widgets.Image(
                    value=arr_to_png_bytes(tight_img.astype(np.uint8)),
                    format="png",
                    layout=widgets.Layout(width="150px"),
                )
            )

        # Render medium crop
        self._out_medium_crop.clear_output(wait=True)
        with self._out_medium_crop:
            display(
                widgets.Image(
                    value=arr_to_png_bytes(medium_img.astype(np.uint8)),
                    format="png",
                    layout=widgets.Layout(width="150px"),
                )
            )

    # ------------------------------------------------------------------
    # Statistics and status updates
    # ------------------------------------------------------------------

    def _update_stats(self):
        """Update statistics display."""
        total_regions = len(self._annotations)
        hard_count = sum(1 for ann in self._annotations.values() if ann.is_hard_example)
        annotated_count = sum(
            1 for ann in self._annotations.values() if ann.is_annotated
        )

        self._stats_status.value = f"Total: {total_regions} | Hard: {hard_count} | Annotated: {annotated_count}"

    # ------------------------------------------------------------------
    # Done tracking
    # ------------------------------------------------------------------

    def is_done(self, name: str) -> bool:
        """Return True if this image name has already been processed."""
        return self._done_tracker.is_done(name)

    # ------------------------------------------------------------------
    # Saving logic
    # ------------------------------------------------------------------

    def _save_hard_examples(self):
        """Save only hard examples in widget_2.py format."""
        hard_annotations = [
            ann for ann in self._annotations.values() if ann.is_hard_example
        ]

        if not hard_annotations:
            self._save_btn.description = "No hard examples to save"
            self._save_btn.button_style = "warning"
            return

        saved_count = 0
        saved_examples = []

        # Save each hard example
        for annotation in hard_annotations:
            try:
                saved_path = self._save_single_crop(annotation)
                saved_examples.append(
                    {
                        "bbox_idx": annotation.bbox_idx,
                        "path": str(saved_path),
                        "final_label": annotation.final_label,
                        "original_pred": annotation.original_pred,
                        "is_disagreement": annotation.human_label is not None
                        and annotation.human_label != annotation.original_pred,
                    }
                )
                saved_count += 1
            except Exception as e:
                print(f"Error saving bbox {annotation.bbox_idx}: {e}")

        # Save summary files
        save_summary_files(
            saved_examples, self._name, self._annotations, self._output_dir
        )

        # Mark image as done
        self._done_tracker.mark_done(self._name)

        # Update UI
        self._save_btn.description = f"Saved {saved_count} hard examples"
        self._save_btn.button_style = "success"

        print(
            f"Successfully saved {saved_count} hard examples for image '{self._name}'"
        )

    def _save_single_crop(self, annotation: BboxAnnotation) -> Path:
        """Save a single crop using save_crop_level and add mass annotation specific metadata."""
        bbox = annotation.bbox
        final_label = annotation.final_label

        # Use save_crop_level for the basic saving
        save_crop_level(
            self._image,
            self._mask,
            self._crop_pad,
            self._output_dir,
            self._name,
            annotation.bbox_idx,
            bbox,
            final_label,
            _LABEL_FOLDER,
            pred_label=annotation.original_pred,
            pred_prob=annotation.original_prob,
            source_dir=self._source_dir,
        )
        sample_dir = get_sample_dir(
            final_label, self._name, annotation.bbox_idx, self._output_dir, _LABEL_FOLDER
        )
        write_extra_metadata(sample_dir, annotation)
        return sample_dir



def write_extra_metadata(sample_dir, annotation):
    # Add mass annotation specific metadata
    meta_file = sample_dir / "meta.json"

    # Read existing metadata and add our fields
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
    else:
        meta = {}

    # Add mass annotation specific fields
    meta["hard_example"] = True
    meta["manual_annotation"] = annotation.human_label is not None

    if (
        annotation.human_label is not None
        and annotation.human_label != annotation.original_pred
    ):
        meta["human_label"] = int(annotation.human_label)

    # Write back the enhanced metadata
    meta_file.write_text(json.dumps(meta, indent=2))

    return sample_dir


def save_summary_files(saved_examples, name, annotations, output_dir):
    """Save summary files for tracking."""
    # Save hard examples list
    hard_examples_file = output_dir / "hard_examples_list.txt"
    with hard_examples_file.open("a") as f:
        for example in saved_examples:
            f.write(f"{name}_{example['bbox_idx']}\n")

    # Save annotation summary
    from datetime import datetime

    summary = {
        "image_name": name,
        "timestamp": datetime.now().isoformat(),
        "total_regions": len(annotations),
        "hard_examples": len(saved_examples),
        "disagreements": sum(1 for ex in saved_examples if ex["is_disagreement"]),
        "saved_examples": saved_examples,
    }

    summary_file = output_dir / "annotation_summary.json"
    summaries = []
    if summary_file.exists():
        summaries = json.loads(summary_file.read_text())
    summaries.append(summary)
    summary_file.write_text(json.dumps(summaries, indent=2))


def read_saved_dataset(dataset_dir: str | Path):
    """
    Read the dataset saved by MassAnnotationWidget and yield (image, mask, label, metadata).

    This function reads the crop_level directory structure and yields data in the format
    that would be fed to the model, along with ground truth labels and metadata.

    Args:
        dataset_dir: Path to the output directory where the widget saved data

    Yields:
        tuple: (image_array, mask_array, label, metadata_dict)
            - image_array: RGB crop image as numpy array
            - mask_array: Binary mask as numpy array
            - label: Ground truth label (LABEL_TRASH or LABEL_OTHER)
            - metadata_dict: Dictionary containing meta.json data
    """
    dataset_path = Path(dataset_dir)
    crop_level_dir = dataset_path / "crop_level"

    if not crop_level_dir.exists():
        print(f"No crop_level directory found in {dataset_path}")
        return

    # Iterate through all label folders (trash, other, unknown)
    for label_folder in crop_level_dir.iterdir():
        if not label_folder.is_dir():
            continue

        # Map folder name to label constant
        label_name = label_folder.name
        if label_name == "trash":
            label = LABEL_TRASH
        elif label_name == "other":
            label = LABEL_OTHER
        else:
            continue  # Skip unknown folders

        print(f"\n=== Processing {label_name.upper()} examples ===")

        # Iterate through all samples in this label folder
        for sample_dir in label_folder.iterdir():
            if not sample_dir.is_dir():
                continue

            # Load image, mask, and metadata
            image_path = sample_dir / "image.jpg"
            mask_path = sample_dir / "mask.png"
            meta_path = sample_dir / "meta.json"

            if not all(p.exists() for p in [image_path, mask_path, meta_path]):
                print(f"Skipping incomplete sample: {sample_dir}")
                continue

            try:
                # Load image and mask
                image_array = np.array(Image.open(image_path))
                mask_array = np.array(Image.open(mask_path))

                # Load metadata
                with open(meta_path) as f:
                    metadata = json.load(f)

                # Print hard example status from metadata
                is_hard = metadata.get("hard_example", False)
                model_disagreement = metadata.get("model_disagreement", False)
                model_pred_label = metadata.get("model_pred", {}).get(
                    "label", "unknown"
                )
                model_pred_prob = metadata.get("model_pred", {}).get("prob", 0.0)

                print(f"Sample: {sample_dir.name}")
                print(f"  - Hard Example: {is_hard}")
                print(f"  - Model Disagreement: {model_disagreement}")
                print(
                    f"  - Model Prediction: {model_pred_label} (prob: {model_pred_prob:.3f})"
                )
                print(f"  - Ground Truth: {label} ({label_name})")

                yield image_array, mask_array, label, metadata

            except Exception as e:
                print(f"Error loading sample {sample_dir}: {e}")
                continue
