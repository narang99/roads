from pathlib import Path

import numpy as np
import ipywidgets as widgets
from IPython.display import display

from mtrain.disk import DiskImage, DiskBooleanMask
from mtrain.smallnet.unet.extract.draw import overlay_mask_on_img
from mtrain.neg_mask.model.predict import predict_and_reconstruct_mask, get_trash_mask
from mtrain.neg_mask.ipywidgets.utils import arr_to_png_bytes


class EvalWidget:
    """
    Evaluation widget for the neg_mask classifier.

    Iterates over (image_path, mask_path) pairs, runs predict_and_reconstruct_mask,
    and shows results across three tabs:

      Trash   — image overlaid with predicted trash regions
      Others  — image overlaid with predicted non-trash regions (what the model is missing)
      Compare — row 1: original mask, row 2: new (filtered) trash mask

    Parameters
    ----------
    learn     : fastai Learner (neg_mask classifier)
    iterator  : iterable of (image_path, mask_path)
    threshold : trash probability threshold (default 0.25)
    crop_pad  : padding around each bbox crop for inference (default 220)
    bbox_pad  : connected-component bbox expansion (default 20)
    """

    def __init__(self, learn, iterator, threshold=0.25, crop_pad=220, bbox_pad=20):
        self._learn = learn
        self._iter = iter(iterator)
        self._threshold = threshold
        self._crop_pad = crop_pad
        self._bbox_pad = bbox_pad
        self._idx = 0

        self._build_ui()
        self._load_next()

    def _build_ui(self):
        self._status = widgets.Label(value="Loading…")

        self._out_trash = widgets.Output()
        self._out_others = widgets.Output()
        self._out_compare = widgets.Output()

        self._overlay_toggle = widgets.ToggleButton(
            value=True,
            description="Overlay on",
            icon="eye",
            layout=widgets.Layout(width="120px"),
        )
        self._overlay_toggle.observe(lambda _: self._render(), names="value")

        self._alpha_slider = widgets.SelectionSlider(
            options=[0.2, 0.4, 0.8, 1.0],
            value=0.4,
            description="Alpha:",
            continuous_update=False,
        )
        self._alpha_slider.observe(lambda _: self._render(), names="value")

        self._threshold_slider = widgets.FloatSlider(
            value=self._threshold,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Threshold:",
            continuous_update=False,
        )
        self._threshold_slider.observe(lambda _: self._repredict(), names="value")

        tabs = widgets.Tab(children=[self._out_trash, self._out_others, self._out_compare])
        for i, title in enumerate(("Trash", "Others", "Compare")):
            tabs.set_title(i, title)

        self._btn_next = widgets.Button(
            description="Next",
            button_style="primary",
            layout=widgets.Layout(width="120px"),
        )
        self._btn_next.on_click(lambda _: self._load_next())

        controls = widgets.HBox(
            [self._overlay_toggle, self._alpha_slider, self._threshold_slider],
            layout=widgets.Layout(gap="8px"),
        )
        display(widgets.VBox(
            [self._status, controls, tabs, self._btn_next],
            layout=widgets.Layout(gap="8px"),
        ))

    def _load_next(self):
        self._btn_next.disabled = True
        self._status.value = "Running prediction…"

        try:
            image_path, mask_path = next(self._iter)
        except StopIteration:
            self._status.value = "✓ Done — no more samples."
            for out in (self._out_trash, self._out_others, self._out_compare):
                out.clear_output()
            return

        image = DiskImage.load(image_path)
        orig_mask = DiskBooleanMask.load_as_bool(mask_path).astype(np.uint8)

        reconstructed = predict_and_reconstruct_mask(
            self._learn, image, orig_mask,
            bbox_pad=self._bbox_pad,
            crop_pad=self._crop_pad,
        )
        trash_mask = get_trash_mask(reconstructed, self._threshold)

        self._image = image
        self._orig_mask = orig_mask
        self._reconstructed = reconstructed
        self._trash_mask = trash_mask
        self._name = Path(image_path).name

        self._render()

        self._status.value = f"[{self._idx}]  {self._name}"
        self._idx += 1
        self._btn_next.disabled = False

    def _repredict(self):
        self._trash_mask = get_trash_mask(self._reconstructed, self._threshold_slider.value)
        self._render()

    def _render(self):
        trash_only = (self._trash_mask == 1).astype(bool)
        self._others_only = (self._trash_mask == 2).astype(bool)

        # Tab: Trash
        self._trash_only = trash_only
        self._render_trash()

        # Tab: Others
        self._render_others()

        # Tab: Compare
        alpha = self._alpha_slider.value if self._overlay_toggle.value else 0.0
        orig_overlay = overlay_mask_on_img(self._image, self._orig_mask.astype(bool), alpha=alpha).astype(np.uint8)
        new_overlay = overlay_mask_on_img(self._image, self._trash_only, alpha=alpha).astype(np.uint8)
        self._out_compare.clear_output(wait=True)
        with self._out_compare:
            display(widgets.HBox([
                widgets.VBox([
                    widgets.Label(value="Original mask"),
                    widgets.Image(value=arr_to_png_bytes(orig_overlay), format="png"),
                ]),
                widgets.VBox([
                    widgets.Label(value="New mask (trash only)"),
                    widgets.Image(value=arr_to_png_bytes(new_overlay), format="png"),
                ]),
            ]))

    def _render_trash(self):
        if self._overlay_toggle.value:
            arr = overlay_mask_on_img(self._image, self._trash_only, alpha=self._alpha_slider.value).astype(np.uint8)
        else:
            arr = self._image
        self._out_trash.clear_output(wait=True)
        with self._out_trash:
            display(widgets.Image(value=arr_to_png_bytes(arr), format="png"))

    def _render_others(self):
        if self._overlay_toggle.value:
            arr = overlay_mask_on_img(self._image, self._others_only, alpha=self._alpha_slider.value).astype(np.uint8)
        else:
            arr = self._image
        self._out_others.clear_output(wait=True)
        with self._out_others:
            display(widgets.Image(value=arr_to_png_bytes(arr), format="png"))
