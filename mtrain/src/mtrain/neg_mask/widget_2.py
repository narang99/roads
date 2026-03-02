import io
import itertools
from pathlib import Path
from dataclasses import dataclass

import cv2
import ipywidgets as widgets
import numpy as np
from IPython.display import display
from PIL import Image
from mtrain.smallnet.unet.extract.draw import overlay_mask_on_img

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

_THEME_BG: dict[str, str] = {
    "Dark": "#1e1e1e",
    "Light": "#f5f5f5",
}


@dataclass
class Bbox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


# ==================================================================
# Pure helper functions — no UI, no side effects
# ==================================================================

def padded_crop(arr: np.ndarray, bbox: Bbox, pad: int) -> tuple[np.ndarray, int, int]:
    """
    Crop `arr` around `bbox` with `pad` pixels on each side, clamped to array bounds.

    Returns
    -------
    crop   : the cropped sub-array (view, not a copy)
    y1c    : actual top row used  (needed to map bbox coords into crop space)
    x1c    : actual left col used
    """
    H, W = arr.shape[:2]
    y1c = max(0, bbox.y - pad)
    y2c = min(H, bbox.y2 + pad)
    x1c = max(0, bbox.x - pad)
    x2c = min(W, bbox.x2 + pad)
    return arr[y1c:y2c, x1c:x2c], y1c, x1c


def bbox_only_mask(mask: np.ndarray, bbox: Bbox, pad: int) -> np.ndarray:
    """
    Return a uint8 binary mask (0 / 255) with padding, where ONLY pixels
    that are (a) inside the bbox boundary AND (b) foreground in `mask` are kept.
    Foreground pixels in the padding region are zeroed out.

    Parameters
    ----------
    mask : bool or uint8 array, shape (H, W)
    bbox : bounding box
    pad  : context padding in pixels
    """
    crop, y1c, x1c = padded_crop(mask, bbox, pad)

    # Local bbox coordinates inside the padded crop
    ry1, ry2 = bbox.y - y1c, bbox.y2 - y1c
    rx1, rx2 = bbox.x - x1c, bbox.x2 - x1c

    result = np.zeros(crop.shape, dtype=np.uint8)
    result[ry1:ry2, rx1:rx2] = crop[ry1:ry2, rx1:rx2].astype(bool).astype(np.uint8) * 255
    return result


def apply_label_to_out_mask(
    out_mask: np.ndarray, mask: np.ndarray, bbox: Bbox, label: int
) -> None:
    """
    In-place: write `label` onto `out_mask` for every foreground pixel of
    `mask` that lies within `bbox`.
    """
    region_mask = mask[bbox.y:bbox.y2, bbox.x:bbox.x2].astype(bool)
    out_mask[bbox.y:bbox.y2, bbox.x:bbox.x2][region_mask] = label


def arr_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a numpy array (uint8 RGB or grayscale) as PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="png")
    buf.seek(0)
    return buf.getvalue()


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
            mask.png     — binary mask (0 / 255); only pixels inside bbox kept

    Parameters
    ----------
    output_dir : str | Path
    crop_pad   : context padding in pixels added around each bbox (default 40)
    """

    def __init__(self, output_dir: str | Path, crop_pad: int = 100, theme: str = "Dark"):
        self._out_dir = Path(output_dir)
        self._crop_pad = crop_pad
        self._bg = _THEME_BG[theme]

        for sub in ("dataset", "crop_level/trash", "crop_level/other", "crop_level/unknown"):
            (self._out_dir / sub).mkdir(parents=True, exist_ok=True)

        self._done: set[str] = self._load_done()

        # State — populated by ui()
        self._name: str = ""
        self._bboxes: list[Bbox] = []
        self._image: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
        self._mask: np.ndarray = np.zeros((1, 1), dtype=bool)
        self._out_mask: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self._bbox_idx: int = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ui(self, name: str, bboxes: list[Bbox], image: np.ndarray, mask: np.ndarray):
        """Display the labeling UI for one image.  Call once per cell."""
        self._name = name
        self._bboxes = bboxes
        self._image = image
        self._mask = mask.astype(bool)
        self._out_mask = np.zeros(mask.shape, dtype=np.uint8)
        self._bbox_idx = 0

        self._build_ui()
        self._render()

    # ------------------------------------------------------------------
    # UI construction  (fresh widgets on every ui() call)
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._out_crop = widgets.Output()
        self._out_full = widgets.Output()

        self._overlay_toggle = widgets.ToggleButton(
            value=True,
            description="Overlay on",
            icon="eye",
            layout=widgets.Layout(width="120px"),
        )
        self._overlay_toggle.observe(lambda _: self._render(), names="value")

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

        self._btn_trash.on_click(lambda _: self._on_label(LABEL_TRASH))
        self._btn_other.on_click(lambda _: self._on_label(LABEL_OTHER))
        self._btn_skip.on_click(lambda _: self._on_label(LABEL_UNKNOWN))

        self._status = widgets.Label(value="")

        self._out_crop.layout.background = self._bg
        self._out_full.layout.background = self._bg

        views = widgets.HBox([self._out_crop, self._out_full], layout=widgets.Layout(gap="12px"))
        btn_row = widgets.HBox(
            [self._btn_trash, self._btn_other, self._btn_skip, self._overlay_toggle],
            layout=widgets.Layout(gap="8px", margin="8px 0 0 0"),
        )
        display(widgets.VBox([self._status, views, btn_row]))

    def _set_buttons(self, disabled: bool):
        for btn in (self._btn_trash, self._btn_other, self._btn_skip):
            btn.disabled = disabled

    # ------------------------------------------------------------------
    # Done tracking
    # ------------------------------------------------------------------

    @property
    def _done_file(self) -> Path:
        return self._out_dir / "done_names.txt"

    def _load_done(self) -> set[str]:
        if self._done_file.exists():
            return set(self._done_file.read_text().splitlines())
        return set()

    def _mark_done(self, name: str) -> None:
        self._done.add(name)
        with self._done_file.open("a") as f:
            f.write(name + "\n")

    def is_done(self, name: str) -> bool:
        """Return True if this image name has already been fully labelled."""
        return name in self._done

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self):
        if self._bbox_idx >= len(self._bboxes):
            return

        bbox = self._bboxes[self._bbox_idx]
        alpha = 0.4 if self._overlay_toggle.value else 0.0
        self._status.value = f"{self._name}  —  crop {self._bbox_idx + 1} / {len(self._bboxes)}"

        # Crop view: padded crop with bbox rectangle
        crop_img, y1c, x1c = padded_crop(self._image, bbox, self._crop_pad)
        crop_mask, _, _ = padded_crop(self._mask, bbox, self._crop_pad)
        crop_overlaid = overlay_mask_on_img(crop_img, crop_mask, alpha).copy()
        _HOT_PINK = (255, 59, 48)
        if self._overlay_toggle.value:
            cv2.rectangle(
                crop_overlaid,
                (bbox.x - x1c, bbox.y - y1c), (bbox.x2 - x1c, bbox.y2 - y1c),
                _HOT_PINK, 1,
            )

        # Full image view: full overlay with bbox rectangle
        full_overlaid = overlay_mask_on_img(self._image, self._mask, alpha).copy()
        if self._overlay_toggle.value:
            cv2.rectangle(full_overlaid, (bbox.x, bbox.y), (bbox.x2, bbox.y2), _HOT_PINK, 3)

        self._out_crop.clear_output(wait=True)
        with self._out_crop:
            display(widgets.Image(value=arr_to_png_bytes(crop_overlaid), format="png",
                                  layout=widgets.Layout(width="500px")))

        self._out_full.clear_output(wait=True)
        with self._out_full:
            display(widgets.Image(value=arr_to_png_bytes(full_overlaid), format="png"))

    # ------------------------------------------------------------------
    # Button handler
    # ------------------------------------------------------------------

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
        folder = _LABEL_FOLDER[label]
        sample_dir = self._out_dir / "crop_level" / folder / f"{self._name}_{self._bbox_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        crop_img, _, _ = padded_crop(self._image, bbox, self._crop_pad)
        crop_mask = bbox_only_mask(self._mask, bbox, self._crop_pad)

        Image.fromarray(crop_img).save(sample_dir / "image.jpg")
        Image.fromarray(crop_mask).save(sample_dir / "mask.png")

    def _finish(self):
        self._set_buttons(disabled=True)

        sample_dir = self._out_dir / "dataset" / self._name
        sample_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(self._image).save(sample_dir / "image.jpg")
        Image.fromarray((self._mask.astype(np.uint8) * 255)).save(sample_dir / "in_mask.png")
        Image.fromarray(self._out_mask).save(sample_dir / "out_mask.png")

        self._mark_done(self._name)

        for out in (self._out_crop, self._out_full):
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
