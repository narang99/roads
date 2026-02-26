import random
from typing import Iterator
import numpy as np
import cv2
from dataclasses import dataclass
import math


@dataclass
class Bbox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h


def extract_crops_for_single_image(
    img: np.ndarray, mask: np.ndarray, bbox_heights: list[int], crop_height, crop_width
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    bboxes = get_bounding_boxes_connected(mask)
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    for bbox in bboxes:
        for bb_h in bbox_heights:
            try:
                en_bbox = get_engulfing_bbox_to_resize(
                    mask, bbox, bb_h, crop_height, crop_width
                )
            except BadDimensionsException:
                # print("bad dimensions, skipping. cause:", ex)
                yield (None, None)
            else:
                resized_img =  resize_bbox_in_img(
                    blurred, en_bbox, crop_height, crop_width
                )
                resized_mask =  resize_bbox_in_img(
                    mask, en_bbox, crop_height, crop_width
                )
                yield (resized_img, resized_mask)

# given a bbox, we want to resize it to some size.
# i have the final shape of the image i want (that would be my cell size)
# my original algorithm has 2 bboxes.
# it first decides the alpha based on their heights
# it then finds how much space the top, bottom, left, right take in the target
# it scales them to our size using the alpha
# then adds them to our bbox. This gives us a crop which can be resized to the targets actual size
# we want without target bbox
# we essentially have the height we want for the object
# we have the final cell size
# using bbox height and required height, we get the aspect ratio
# cell height - our calc height is leftover
# cell width - our calc width is leftover
# split leftover in 2 parts, randomly
# and then create a crop to resize

class BadDimensionsException(Exception):
    pass

def get_engulfing_bbox_to_resize(
    mask: np.ndarray,
    bbox: Bbox,
    target_bbox_height: int,
    target_cell_height: int,
    target_cell_width: int,
):
    aspect_ratio = target_bbox_height / bbox.h
    target_bbox_width = math.ceil(aspect_ratio * bbox.w)

    if target_bbox_height > target_cell_height:
        raise Exception(
            f"target_bbox_height > target_cell_height: {target_bbox_height} > {target_cell_height}"
        )
    if target_bbox_width > target_cell_width:
        raise BadDimensionsException(
            f"target_bbox_width > target_cell_width: {target_bbox_width} > {target_cell_width}"
        )

    leftover_height = target_cell_height - target_bbox_height
    leftover_height = math.ceil(leftover_height / aspect_ratio)

    leftover_width = target_cell_width - target_bbox_width
    leftover_width = math.ceil(leftover_width / aspect_ratio)

    top, bottom = _split_dist_in_2(leftover_height)
    left, right = _split_dist_in_2(leftover_width)

    x1 = max(bbox.x - left, 0)
    x2 = min(bbox.x2 + right, mask.shape[1])
    y1 = max(bbox.y - top, 0)
    y2 = min(bbox.y2 + bottom, mask.shape[0])

    return Bbox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _split_dist_in_2(dist: int) -> tuple[int, int]:
    splitter = random.randint(0, dist)
    return splitter, dist - splitter


def resize_bbox_in_img(
    img, bbox: Bbox, target_height, target_width, interp=cv2.INTER_AREA
):
    crop = img[bbox.y : bbox.y2, bbox.x : bbox.x2]
    return cv2.resize(crop, (target_height, target_width), interpolation=interp)


def get_bounding_boxes_connected(binary_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    boxes = []
    for label in range(1, num_labels):  # skip label 0 (background)
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        boxes.append(Bbox(x, y, w, h))

    return boxes
