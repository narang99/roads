"""
Bbox processing utilities shared across annotation widgets.

This module provides pure functions for extracting bounding boxes from masks,
generating crops, and applying labels to output masks.
"""

import cv2
import numpy as np
from mtrain.neg_mask.crops import Bbox, bbox_only_mask, padded_crop


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

def apply_label_to_out_mask(
    out_mask: np.ndarray, mask: np.ndarray, bbox: Bbox, label: int
) -> None:
    """
    In-place: write `label` onto `out_mask` for every foreground pixel of
    `mask` that lies within `bbox`.
    """
    region_mask = mask[bbox.y : bbox.y2, bbox.x : bbox.x2].astype(bool)
    out_mask[bbox.y : bbox.y2, bbox.x : bbox.x2][region_mask] = label
