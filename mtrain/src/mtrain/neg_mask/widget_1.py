from mtrain.smallnet.unet.extract.draw import overlay_mask_on_img
import io
from pathlib import Path

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image

matplotlib.use("Agg")


class LabelWidget:
    """
    Interactive ipywidget for binary classification labelling of (image, mask) pairs.

    Parameters
    ----------
    iterator : iterable of (name, image, mask)
        - name  : str  — sample name (may repeat)
        - image : np.ndarray (H, W, 3) uint8 RGB
        - mask  : np.ndarray (H, W) bool/uint8 binary mask
    output_dir : str | Path
        Root output directory.  Structure created automatically:
            out/data/trash/{name}_{global_idx}/image.jpg + mask.png
            out/data/no/{name}_{global_idx}/image.jpg + mask.png
            out/skipped/{name}_{global_idx}/image.jpg + mask.png
    overlay_alpha : float
        Opacity of the red mask overlay in the Overlaid tab (default 0.45).
    """

    def __init__(self, iterator, output_dir: str | Path, overlay_alpha: float = 0.45):
        self._iter = iter(iterator)
        self._out_dir = Path(output_dir)
        self._alpha = overlay_alpha
        self._idx = 0
        self._current = None

        self._setup_dirs()
        self._build_ui()
        self._load_next()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_dirs(self):
        for sub in ("data/trash", "data/no", "skipped"):
            (self._out_dir / sub).mkdir(parents=True, exist_ok=True)

    def _build_ui(self):
        self._out_image = widgets.Output()
        self._out_overlaid = widgets.Output()
        self._out_mask = widgets.Output()

        tabs = widgets.Tab(
            children=[self._out_image, self._out_overlaid, self._out_mask]
        )
        for i, title in enumerate(("Image", "Overlaid", "Mask")):
            tabs.set_title(i, title)

        self._btn_trash = widgets.Button(
            description="Trash",
            button_style="danger",
            layout=widgets.Layout(width="120px"),
        )
        self._btn_no = widgets.Button(
            description="No",
            button_style="success",
            layout=widgets.Layout(width="120px"),
        )
        self._btn_skip = widgets.Button(
            description="Skip",
            button_style="warning",
            layout=widgets.Layout(width="120px"),
        )

        self._btn_trash.on_click(lambda _: self._on_label("trash"))
        self._btn_no.on_click(lambda _: self._on_label("no"))
        self._btn_skip.on_click(lambda _: self._on_skip())

        self._status = widgets.Label(value="Loading…")

        btn_row = widgets.HBox(
            [self._btn_trash, self._btn_no, self._btn_skip],
            layout=widgets.Layout(gap="8px", margin="8px 0 0 0"),
        )

        self._ui = widgets.VBox([self._status, tabs, btn_row])
        display(self._ui)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _load_next(self):
        try:
            self._current = next(self._iter)
            self._render()
            self._set_buttons(disabled=False)
        except StopIteration:
            self._current = None
            self._status.value = "✓ Done — no more items."
            for out in (self._out_image, self._out_overlaid, self._out_mask):
                out.clear_output()
            self._set_buttons(disabled=True)

    def _set_buttons(self, disabled: bool):
        for btn in (self._btn_trash, self._btn_no, self._btn_skip):
            btn.disabled = disabled

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self):
        assert self._current is not None
        name, image, mask = self._current
        self._status.value = f"[{self._idx}]  {name}"

        mask_bool = mask.astype(bool)
        overlaid = overlay_mask_on_img(image, mask_bool, self._alpha).astype(np.uint8)
        mask_display = (mask_bool.astype(np.uint8) * 255)

        panels = [
            (self._out_image, image, False),
            (self._out_overlaid, overlaid, False),
            (self._out_mask, mask_display, True),
        ]
        for output, arr, is_gray in panels:
            png = self._array_to_png(arr, cmap="gray" if is_gray else None)
            output.clear_output(wait=True)
            with output:
                display(widgets.Image(value=png, format="png"))

    def _make_overlay(self, image: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        red = np.zeros_like(image)
        red[..., 0] = 255
        alpha = self._alpha * mask_bool[..., np.newaxis]
        blended = image * (1 - alpha) + red * alpha
        return blended.astype(np.uint8)

    @staticmethod
    def _array_to_png(arr: np.ndarray, cmap=None) -> bytes:
        fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
        ax.imshow(arr, cmap=cmap)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save(self, folder: Path):
        assert self._current is not None
        name, image, mask = self._current
        sample_dir = folder / f"{name}_{self._idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(sample_dir / "image.jpg")
        mask_uint8 = (mask.astype(bool).astype(np.uint8) * 255)
        Image.fromarray(mask_uint8).save(sample_dir / "mask.png")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_label(self, label: str):
        if self._current is None:
            return
        self._save(self._out_dir / "data" / label)
        self._idx += 1
        self._load_next()

    def _on_skip(self):
        if self._current is None:
            return
        self._save(self._out_dir / "skipped")
        self._idx += 1
        self._load_next()
