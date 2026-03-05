import numpy as np
import cv2
from dataclasses import dataclass


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


def get_crops_for_image(image: np.ndarray, mask: np.ndarray, bbox_pad=20, crop_pad=220):
    bboxes = list(get_region_crops(image, mask, bbox_pad))
    yield from iter_crops(image, mask, bboxes, crop_pad)


def iter_crops(
    image: np.ndarray,
    mask: np.ndarray,
    bboxes: list[Bbox],
    pad: int = 220,
):
    """
    Yield (image_crop, mask_crop) for each bbox.

    image_crop : padded RGB crop of `image`
    mask_crop  : uint8 binary mask (0/1); only foreground pixels *inside* the
                 bbox are kept — the padding region is zeroed out
    """
    for bbox in bboxes:
        crop_img, _, _ = padded_crop(image, bbox, pad)
        crop_mask = bbox_only_mask(mask, bbox, pad)
        yield bbox, crop_img, crop_mask


def get_region_crops(img, mask, padding=20):
    _, labels = cv2.connectedComponents(mask)
    h, w = img.shape[:2]
    for label in range(1, labels.max() + 1):
        rows, cols = np.where(labels == label)
        r1 = max(0, rows.min())
        r2 = min(h, rows.max())
        c1 = max(0, cols.min())
        c2 = min(w, cols.max())
        yield Bbox(c1, r1, c2 - c1, r2 - r1)


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
    Return a uint8 binary mask (0 / 1) with padding, where ONLY pixels
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
    result[ry1:ry2, rx1:rx2] = crop[ry1:ry2, rx1:rx2].astype(bool).astype(np.uint8)
    return result
