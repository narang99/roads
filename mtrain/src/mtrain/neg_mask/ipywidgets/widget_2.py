from mtrain.neg_mask.ipywidgets.save_crops import save_crop_level
import itertools
import json
from pathlib import Path

import cv2
import ipywidgets as widgets
import numpy as np
from IPython.display import display
from PIL import Image
from mtrain.smallnet.unet.extract.draw import overlay_mask_on_img
from mtrain.neg_mask.crops import Bbox, bbox_only_mask, padded_crop
from mtrain.neg_mask.ipywidgets.utils import arr_to_png_bytes
from mtrain.neg_mask.ipywidgets.done_tracker import DoneTracker, SkippedTracker
from mtrain.neg_mask.ipywidgets.bbox_processing import iter_crops, get_region_crops, apply_label_to_out_mask, get_crops_for_image

# ------------------------------------------------------------------
# Label constants
#   Used in out_mask.png (image-level dataset) as pixel values:
#     0 = background
#     1 = trash
#     2 = other
#     3 = unknown  (user pressed Skip)
#   Also used to pick the subfolder in crop_level/.
# ------------------------------------------------------------------
LABEL_TRASH = 1
LABEL_OTHER = 2
LABEL_UNKNOWN = 3

_LABEL_FOLDER: dict[int, str] = {
    LABEL_TRASH: "trash",
    LABEL_OTHER: "other",
    LABEL_UNKNOWN: "unknown",
}

_BBOX_INSET = 5  # pixels the drawn rectangle is inset from the actual bbox edge



# ==================================================================
# Widget — UI only, delegates all data work to helpers above
# ==================================================================


class LabelWidget:
    """
    Mask-relabeling widget.  One instance lives across notebook cells;
    call .ui() once per image.

    Output layout
    -------------
    out/
      dataset/
        {image_name}/
          image.jpg      — original RGB image
          in_mask.png    — original binary mask (0 / 255)
          out_mask.png   — per-crop label mask  (uint8, see constants above)
      crop_level/
        {trash|other|unknown}/
          {image_name}_{crop_idx}/
            image.jpg    — padded crop of the original image
            mask.png     — binary mask (0 / 1); only pixels inside bbox kept

    Parameters
    ----------
    output_dir : str | Path
    crop_pad   : context padding in pixels added around each bbox (default 40)
    """

    def __init__(
        self,
        output_dir: str | Path,
        crop_pad: int = 220,
        learner=None,
    ):
        self._out_dir = Path(output_dir)
        self._crop_pad = crop_pad
        self._learner = learner

        for sub in (
            "dataset",
            "crop_level/trash",
            "crop_level/other",
            "crop_level/unknown",
        ):
            (self._out_dir / sub).mkdir(parents=True, exist_ok=True)

        self._done_tracker = DoneTracker(self._out_dir)
        self._skipped_tracker = SkippedTracker(self._out_dir)

        # State — populated by ui()
        self._name: str = ""
        self._source_dir: Path | None = None
        self._bboxes: list[Bbox] = []
        self._image: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
        self._mask: np.ndarray = np.zeros((1, 1), dtype=bool)
        self._out_mask: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self._bbox_idx: int = 0

        # Model prediction cache: (label, prob) or None; keyed by bbox_idx
        self._current_model_pred: tuple[int, float] | None = None
        self._last_pred_idx: int = -1

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ui(
        self, source_dir: Path, bboxes: list[Bbox], image: np.ndarray, mask: np.ndarray
    ):
        """Display the labeling UI for one image.  Call once per cell."""
        self._name = source_dir.name
        self._source_dir = source_dir
        self._bboxes = bboxes
        self._image = image
        self._mask = mask.astype(bool)
        self._out_mask = np.zeros(mask.shape, dtype=np.uint8)
        self._bbox_idx = 0
        self._current_model_pred = None
        self._last_pred_idx = -1

        self._build_ui()
        self._render()

    # ------------------------------------------------------------------
    # UI construction  (fresh widgets on every ui() call)
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._out_crop = widgets.Output()
        self._out_full = widgets.Output()
        self._out_model_input = widgets.Output()

        self._overlay_toggle = widgets.ToggleButton(
            value=True,
            description="Overlay on",
            icon="eye",
            layout=widgets.Layout(width="120px"),
        )
        self._overlay_toggle.observe(lambda _: self._render(), names="value")

        self._blend_toggle = widgets.ToggleButton(
            value=False,
            description="Full blend",
            icon="adjust",
            layout=widgets.Layout(width="120px"),
        )
        self._blend_toggle.observe(lambda _: self._render(), names="value")

        self._bbox_toggle = widgets.ToggleButton(
            value=True,
            description="BBox on",
            icon="square-o",
            layout=widgets.Layout(width="120px"),
        )
        self._bbox_toggle.observe(lambda _: self._render(), names="value")

        self._btn_trash = widgets.Button(
            description="Trash (1)",
            button_style="danger",
            layout=widgets.Layout(width="120px"),
        )
        self._btn_other = widgets.Button(
            description="Other (2)",
            button_style="info",
            layout=widgets.Layout(width="120px"),
        )
        self._btn_skip = widgets.Button(
            description="Skip",
            button_style="warning",
            layout=widgets.Layout(width="120px"),
        )

        self._btn_other_all = widgets.Button(
            description="Other All",
            button_style="info",
            layout=widgets.Layout(width="120px"),
        )
        self._btn_skip_all = widgets.Button(
            description="Skip Image",
            button_style="warning",
            layout=widgets.Layout(width="120px"),
        )

        self._btn_trash.on_click(lambda _: self._on_label(LABEL_TRASH))
        self._btn_other.on_click(lambda _: self._on_label(LABEL_OTHER))
        self._btn_skip.on_click(lambda _: self._on_label(LABEL_UNKNOWN))
        self._btn_other_all.on_click(lambda _: self._on_other_all())
        self._btn_skip_all.on_click(lambda _: self._on_skip_all())

        self._status = widgets.Label(value="")
        self._model_status = widgets.Label(value="")


        # Keep original side-by-side layout for crop and full views
        views = widgets.HBox(
            [self._out_crop, self._out_full], layout=widgets.Layout(gap="12px")
        )

        # Create tabs with main view and model input
        main_view_tab = widgets.VBox([views], layout=widgets.Layout(padding="10px"))
        model_input_tab = widgets.VBox(
            [self._out_model_input], layout=widgets.Layout(padding="10px")
        )

        tabs = widgets.Tab(children=[main_view_tab, model_input_tab])
        tabs.set_title(0, "Main View")
        tabs.set_title(1, "Model Input")

        toggle_row = widgets.HBox(
            [self._overlay_toggle, self._blend_toggle, self._bbox_toggle],
            layout=widgets.Layout(gap="8px", margin="8px 0 0 0"),
        )
        btn_row = widgets.HBox(
            [
                self._btn_trash,
                self._btn_other,
                self._btn_skip,
                self._btn_other_all,
                self._btn_skip_all,
            ],
            layout=widgets.Layout(gap="8px", margin="8px 0 0 0"),
        )
        display(
            widgets.VBox([toggle_row, btn_row, self._status, self._model_status, tabs])
        )

    def _set_buttons(self, disabled: bool):
        for btn in (
            self._btn_trash,
            self._btn_other,
            self._btn_skip,
            self._btn_other_all,
            self._btn_skip_all,
        ):
            btn.disabled = disabled

    # ------------------------------------------------------------------
    # Done tracking
    # ------------------------------------------------------------------

    def is_done(self, name: str) -> bool:
        """Return True if this image name has already been fully labelled."""
        return self._done_tracker.is_done(name)

    def is_skipped(self, name: str) -> bool:
        """Return True if this image was fully skipped via Skip Image."""
        return self._skipped_tracker.is_skipped(name)

    # ------------------------------------------------------------------
    # Model prediction
    # ------------------------------------------------------------------

    def _run_model_pred(
        self, crop_img: np.ndarray, crop_mask: np.ndarray, bbox
    ) -> tuple[int, float]:
        from mtrain.neg_mask.model.predict.predict_8ch import run_inference

        all_probs = run_inference(
            self._learner, [(self._image, self._mask, bbox)]
        )  # [1, C]
        class_idx = all_probs[0].argmax().item()
        prob = all_probs[0, int(class_idx)].item()
        # trash_pred_idx=0 matches the default in predict_trash
        label = LABEL_TRASH if class_idx == 1 else LABEL_OTHER
        return label, prob

    def _visualize_model_input(self, bbox) -> list[tuple[str, np.ndarray]]:
        """Create visualization of the 8-channel tensor input to the model."""
        from mtrain.neg_mask.model.predict.predict_8ch import _prepare_8_channel_tensor
        from mtrain.neg_mask.model.crop_level_dataset import CropLevelDataset2Chan

        # Get the 8-channel tensor that would be fed to the model
        tensor_8ch = _prepare_8_channel_tensor(self._image, self._mask, bbox)

        # Denormalize to get back the image/mask pairs
        pairs = CropLevelDataset2Chan.denormalize(
            tensor_8ch
        )  # [(img, mask), (img, mask)]

        # Return individual images with labels for better visualization
        tight_img, tight_mask = pairs[0]
        medium_img, medium_mask = pairs[1]

        # Convert masks to 3-channel for visualization
        tight_mask_vis = np.stack([tight_mask] * 3, axis=-1) * 255
        medium_mask_vis = np.stack([medium_mask] * 3, axis=-1) * 255

        return [
            ("Tight Crop (RGB)", tight_img.astype(np.uint8)),
            ("Medium Crop (RGB)", medium_img.astype(np.uint8)),
            ("Tight Mask", tight_mask_vis.astype(np.uint8)),
            ("Medium Mask", medium_mask_vis.astype(np.uint8)),
        ]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self):
        if self._bbox_idx >= len(self._bboxes):
            # self._status.value = f"{self._name}  —  DONE"
            # alpha = 0.4 if self._overlay_toggle.value else 0.0
            # full_overlaid = overlay_mask_on_img(self._image, self._mask, alpha).copy()
            # self._out_full.clear_output(wait=True)
            # with self._out_full:
            #     display(widgets.Image(value=arr_to_png_bytes(full_overlaid), format="png"))
            return

        bbox = self._bboxes[self._bbox_idx]
        alpha = (
            (1.0 if self._blend_toggle.value else 0.4)
            if self._overlay_toggle.value
            else 0.0
        )
        self._status.value = (
            f"{self._name}  —  crop {self._bbox_idx + 1} / {len(self._bboxes)}"
        )

        # Crop view: padded crop with bbox rectangle
        crop_img, y1c, x1c = padded_crop(self._image, bbox, self._crop_pad)
        crop_mask = bbox_only_mask(self._mask, bbox, self._crop_pad)

        # Model prediction (cached per bbox so toggles don't re-run inference)
        if self._learner is not None and self._bbox_idx != self._last_pred_idx:
            self._current_model_pred = self._run_model_pred(
                crop_img, crop_mask, self._bboxes[self._bbox_idx]
            )
            self._last_pred_idx = self._bbox_idx
        if self._learner is not None and self._current_model_pred is not None:
            pred_label, pred_prob = self._current_model_pred
            pred_name = "Trash" if pred_label == LABEL_TRASH else "Other"
            self._model_status.value = f"Model: {pred_name}  ({pred_prob:.2f})"
        else:
            self._model_status.value = ""
        crop_overlaid = overlay_mask_on_img(
            crop_img, crop_mask.astype(bool), alpha
        ).copy()
        full_overlaid = overlay_mask_on_img(
            self._image, self._mask.astype(bool), alpha
        ).copy()

        if self._bbox_toggle.value:
            _HOT_PINK = (255, 105, 180)
            inset = 0
            cv2.rectangle(
                crop_overlaid,
                (bbox.x - x1c + inset, bbox.y - y1c + inset),
                (bbox.x2 - x1c - inset, bbox.y2 - y1c - inset),
                _HOT_PINK,
                1,
            )
            cv2.rectangle(
                full_overlaid,
                (bbox.x + inset, bbox.y + inset),
                (bbox.x2 - inset, bbox.y2 - inset),
                _HOT_PINK,
                1,
            )

        self._out_crop.clear_output(wait=True)
        with self._out_crop:
            display(
                widgets.Image(
                    value=arr_to_png_bytes(crop_overlaid),
                    format="png",
                    layout=widgets.Layout(width="700px"),
                )
            )

        self._out_full.clear_output(wait=True)
        with self._out_full:
            display(widgets.Image(value=arr_to_png_bytes(full_overlaid), format="png"))

        # Render model input visualization
        if self._learner is not None:
            model_inputs = self._visualize_model_input(bbox)
            self._out_model_input.clear_output(wait=True)
            with self._out_model_input:
                # Create 2x2 grid layout
                top_row_widgets = []
                bottom_row_widgets = []

                for i, (label, img_array) in enumerate(model_inputs):
                    section = widgets.VBox(
                        [
                            widgets.HTML(
                                value=f"<b>{label}</b>",
                                layout=widgets.Layout(text_align="center"),
                            ),
                            widgets.Image(
                                value=arr_to_png_bytes(img_array),
                                format="png",
                                layout=widgets.Layout(width="300px"),
                            ),
                        ]
                    )

                    if i < 2:  # First two go in top row
                        top_row_widgets.append(section)
                    else:  # Last two go in bottom row
                        bottom_row_widgets.append(section)

                top_row = widgets.HBox(
                    top_row_widgets,
                    layout=widgets.Layout(justify_content="space-around"),
                )
                bottom_row = widgets.HBox(
                    bottom_row_widgets,
                    layout=widgets.Layout(justify_content="space-around"),
                )

                display(widgets.VBox([top_row, bottom_row]))
        else:
            self._out_model_input.clear_output(wait=True)
            with self._out_model_input:
                display(
                    widgets.HTML(
                        value="<i>No model loaded - model input visualization unavailable</i>"
                    )
                )

    # ------------------------------------------------------------------
    # Button handler
    # ------------------------------------------------------------------

    def _on_skip_all(self):
        for bbox in self._bboxes[self._bbox_idx :]:
            apply_label_to_out_mask(self._out_mask, self._mask, bbox, LABEL_UNKNOWN)
            self._save_crop_level(bbox, LABEL_UNKNOWN)
            self._bbox_idx += 1
        self._skipped_tracker.mark_skipped(self._name)
        self._finish()

    def _on_other_all(self):
        for bbox in self._bboxes[self._bbox_idx :]:
            apply_label_to_out_mask(self._out_mask, self._mask, bbox, LABEL_OTHER)
            self._save_crop_level(bbox, LABEL_OTHER)
            self._bbox_idx += 1
        self._finish()

    def _on_label(self, label: int):
        if self._bbox_idx >= len(self._bboxes):
            return

        bbox = self._bboxes[self._bbox_idx]
        apply_label_to_out_mask(self._out_mask, self._mask, bbox, label)
        self._save_crop_level(bbox, label)

        self._bbox_idx += 1
        if self._bbox_idx >= len(self._bboxes):
            self._finish()
        else:
            self._render()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_crop_level(self, bbox: Bbox, label: int):
        if self._current_model_pred is not None:
            pred_label, pred_prob = self._current_model_pred
        else:
            pred_label, pred_prob = None, None
        save_crop_level(
            self._image,
            self._mask,
            self._crop_pad,
            self._out_dir,
            self._name,
            self._bbox_idx,
            bbox,
            label,
            _LABEL_FOLDER,
            pred_label,
            pred_prob,
            self._source_dir,
        )

    def _finish(self):
        self._set_buttons(disabled=True)

        sample_dir = self._out_dir / "dataset" / self._name
        sample_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(self._image).save(sample_dir / "image.jpg")
        Image.fromarray((self._mask.astype(np.uint8))).save(sample_dir / "in_mask.png")
        Image.fromarray(self._out_mask).save(sample_dir / "out_mask.png")

        self._done_tracker.mark_done(self._name)

        for out in (self._out_crop, self._out_full, self._out_model_input):
            out.clear_output()
        self._status.value = f"✓ Saved — {self._name}  ({len(self._bboxes)} crops)"


def parse_top_level_ds(root: Path):
    for d in root.glob("*"):
        if not d.is_dir():
            continue
        yield d / "image.jpg", d / "in_mask.png", d / "out_mask.png"


def parse_crop_level_ds(root: Path):
    def _label_iter(label):
        return zip((root / label).glob("*"), itertools.repeat(label))

    other = _label_iter("other")
    trash = _label_iter("trash")
    unknown = _label_iter("unknown")
    zipped = itertools.zip_longest(trash, other, unknown)
    it = itertools.chain.from_iterable(zipped)
    it = filter(lambda p: p is not None, it)
    for d, label in it:
        yield label, d / "image.jpg", d / "mask.png"
