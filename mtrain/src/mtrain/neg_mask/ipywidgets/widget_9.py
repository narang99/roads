import numpy as np
import ipywidgets as widgets
from IPython.display import display
from typing import Literal

from mtrain.random import random_filename
from mtrain.neg_mask.crops import get_region_crops, padded_crop, bbox_only_mask
from mtrain.disk import DiskBooleanMask, DiskImage
from mtrain.neg_mask.ipywidgets.utils import arr_to_png_bytes
from mtrain.utils import overlay_mask_on_img
from mtrain.neg_mask.ipywidgets.done_tracker import DoneTracker


class MissedExampleWidget:
    def __init__(self, dirs, dest_dir, pad=224):
        self.dirs = dirs
        self.pad = pad
        self.dest_dir = dest_dir

        # Done tracking
        self._done_tracker = DoneTracker(dest_dir)

        # State
        self._dir_idx = 0
        self._crop_idx = 0
        self._crops: list[tuple[np.ndarray, np.ndarray]] = []

        self._advance_to_next_dir()
        self._build_ui()
        self._render()

    def get_crops_for_single_dir(self, index):
        miss_mask_path = self.dirs[index] / "negmask-miss.png"
        image_path = self.dirs[index] / "image.jpg"
        if not miss_mask_path.exists():
            return None
        mask = DiskBooleanMask.load(miss_mask_path)
        image = DiskImage.load(image_path)
        bboxes = list(get_region_crops(mask))

        res = []
        for bbox in bboxes:
            crop, _, _ = padded_crop(image, bbox, self.pad)
            crop_mask = bbox_only_mask(mask, bbox, self.pad)
            res.append((crop, crop_mask))
        return res

    def save_to_dest_dir(self, crop, crop_mask, label: Literal["trash", "other"]):
        train_dir = self.dest_dir / "train"
        mask_dir = self.dest_dir / "masks"
        train_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        fname = random_filename()
        fname = f"{label}_{fname}"

        DiskImage.save(crop, train_dir / f"{fname}.jpg")
        DiskBooleanMask.save(crop_mask, mask_dir / f"{fname}.png")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _advance_to_next_dir(self) -> bool:
        """Find the next dir that has crops and isn't done, update state. Returns False when exhausted."""
        while self._dir_idx < len(self.dirs):
            dir_name = self.dirs[self._dir_idx].name
            if self._done_tracker.is_done(dir_name):
                self._dir_idx += 1
                continue
            crops = self.get_crops_for_single_dir(self._dir_idx)
            if crops:
                self._crops = crops
                self._crop_idx = 0
                return True
            self._dir_idx += 1
        self._crops = []
        return False

    def _advance_crop(self):
        self._crop_idx += 1
        if self._crop_idx >= len(self._crops):
            # Mark current directory as done
            if self._dir_idx < len(self.dirs):
                dir_name = self.dirs[self._dir_idx].name
                self._done_tracker.mark_done(dir_name)
            self._dir_idx += 1
            self._advance_to_next_dir()
        self._render()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._out_original = widgets.Output()
        self._out_overlay = widgets.Output()
        self._out_mask = widgets.Output()
        self._status = widgets.Label(value="")

        self._btn_trash = widgets.Button(
            description="Trash",
            button_style="danger",
            layout=widgets.Layout(width="100px"),
        )
        self._btn_other = widgets.Button(
            description="Other",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )
        self._btn_skip = widgets.Button(
            description="Skip",
            button_style="warning",
            layout=widgets.Layout(width="100px"),
        )

        self._btn_trash.on_click(lambda _: self._on_label("trash"))
        self._btn_other.on_click(lambda _: self._on_label("other"))
        self._btn_skip.on_click(lambda _: self._advance_crop())

        btn_row = widgets.HBox(
            [self._btn_trash, self._btn_other, self._btn_skip],
            layout=widgets.Layout(gap="8px", margin="8px 0"),
        )
        
        # Three panels side-by-side
        img_row = widgets.HBox(
            [self._out_original, self._out_overlay, self._out_mask],
            layout=widgets.Layout(gap="12px")
        )
        
        display(widgets.VBox([self._status, btn_row, img_row]))

    def _set_buttons(self, disabled: bool):
        for btn in (self._btn_trash, self._btn_other, self._btn_skip):
            btn.disabled = disabled

    def _render(self):
        if not self._crops or self._crop_idx >= len(self._crops):
            self._out_original.clear_output()
            self._out_overlay.clear_output()
            self._out_mask.clear_output()
            self._status.value = "Done — no more crops."
            self._set_buttons(disabled=True)
            return

        crop, crop_mask = self._crops[self._crop_idx]
        dir_name = self.dirs[self._dir_idx].name
        self._status.value = (
            f"Dir {self._dir_idx + 1}/{len(self.dirs)}: {dir_name}  —  "
            f"crop {self._crop_idx + 1}/{len(self._crops)}"
        )

        # Panel 1: Original crop
        self._out_original.clear_output(wait=True)
        with self._out_original:
            display(
                widgets.Image(
                    value=arr_to_png_bytes(crop),
                    format="png",
                    layout=widgets.Layout(max_width="400px"),
                )
            )

        # Panel 2: Crop with overlay
        crop_with_overlay = overlay_mask_on_img(crop, crop_mask.astype(bool), alpha=0.4, color=[0, 255, 0])
        
        self._out_overlay.clear_output(wait=True)
        with self._out_overlay:
            display(
                widgets.Image(
                    value=arr_to_png_bytes(crop_with_overlay),
                    format="png",
                    layout=widgets.Layout(max_width="400px"),
                )
            )

        # Panel 3: Mask only (convert to 3-channel for display)
        mask_viz = (crop_mask[:, :, None] * np.array([255, 255, 255])).astype(np.uint8)
        
        self._out_mask.clear_output(wait=True)
        with self._out_mask:
            display(
                widgets.Image(
                    value=arr_to_png_bytes(mask_viz),
                    format="png",
                    layout=widgets.Layout(max_width="400px"),
                )
            )


    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_label(self, label: Literal["trash", "other"]):
        if not self._crops or self._crop_idx >= len(self._crops):
            return
        crop, crop_mask = self._crops[self._crop_idx]
        self.save_to_dest_dir(crop, crop_mask, label)
        self._advance_crop()

