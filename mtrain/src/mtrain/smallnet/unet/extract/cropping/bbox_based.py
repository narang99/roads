import random
from typing import Iterator, Literal
import albumentations as A
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
    img: np.ndarray, mask: np.ndarray, bbox_heights: list[int], crop_height, crop_width, min_area=0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    bboxes = get_bounding_boxes_connected(mask)
    blurred = img
    for bbox in bboxes:
        for bb_h in bbox_heights:
            res = get_engulfing_bbox_to_resize(
                mask, bbox, bb_h, crop_height, crop_width, min_area
            )
            if res is None:
                continue
            en_bbox, scaled_leftover_height, scaled_leftover_width = res
            # now the bbox can be bigger than image
            resized_img = resize_bbox_in_img(
                blurred,
                en_bbox,
                crop_height,
                crop_width,
                scaled_leftover_height,
                scaled_leftover_width,
                interp=cv2.INTER_AREA,
            )
            resized_mask = resize_bbox_in_img(
                mask,
                en_bbox,
                crop_height,
                crop_width,
                scaled_leftover_height,
                scaled_leftover_width,
                interp=cv2.INTER_NEAREST,
            )
            yield bb_h, (resized_img, resized_mask)


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
    min_area: int,
):
    if target_bbox_height > 5:
        jitter = random.randint(-5, 5)
    else:
        jitter = random.randint(-1, 0)

    target_bbox_height += jitter
    target_bbox_height = max(target_bbox_height, 0)
    target_bbox_height = min(target_bbox_height, target_cell_height)

    if target_bbox_height > bbox.h:
        # we dont size up
        return None
        # raise BadDimensionsException(
        #     "target bbox size cannot be greater than original size"
        # )

    aspect_ratio = target_bbox_height / bbox.h
    target_bbox_width = math.ceil(aspect_ratio * bbox.w)

    if target_bbox_width * target_bbox_height < min_area:
        return None

    if target_bbox_height > target_cell_height:
        return None
        # raise Exception(
        #     f"target_bbox_height > target_cell_height: {target_bbox_height} > {target_cell_height}"
        # )
    if target_bbox_width > target_cell_width:
        return None
        # raise BadDimensionsException(
        #     f"target_bbox_width > target_cell_width: {target_bbox_width} > {target_cell_width}"
        # )

    if aspect_ratio == 0:
        return None
        # raise BadDimensionsException("aspect ratio is zero")

    leftover_height = target_cell_height - target_bbox_height
    scaled_leftover_height = math.ceil(leftover_height / aspect_ratio)
    y1, y2, actual_total_height = split_leftovers(
        scaled_leftover_height, bbox.y, bbox.y2, 0, mask.shape[0]
    )

    leftover_width = target_cell_width - target_bbox_width
    scaled_leftover_width = math.ceil(leftover_width / aspect_ratio)
    x1, x2, actual_total_width = split_leftovers(
        scaled_leftover_width, bbox.x, bbox.x2, 0, mask.shape[1]
    )

    return Bbox(x=x1, y=y1, w=x2 - x1, h=y2 - y1), scaled_leftover_height, scaled_leftover_width


# def _find_target_heights(
#     bbox,
#     target_length,
#     target_cell_height,
#     target_cell_width,
#     along: Literal["h", "w"] = "h",
# ):
#     if along == "h":
#         primary_max = target_cell_height
#         orig_primary_length = bbox.h
#         secondary_max = target_cell_width
#         orig_secondary_length = bbox.w
#     else:
#         primary_max = target_cell_width
#         orig_primary_length = bbox.w
#         secondary_max = target_cell_height
#         orig_secondary_length = bbox.h

#     primary_length = max(target_length, 0)
#     primary_length = min(target_length, primary_max)

#     aspect_ratio = primary_length / orig_primary_length
#     if aspect_ratio == 0:
#         raise BadDimensionsException("aspect ratio is zero")

#     secondary_length = math.ceil(orig_secondary_length * aspect_ratio)
#     if secondary_length > secondary_max:
#         length_str = "target_bbox_width" if along == "h" else "target_bbox_height"
#         cell_str = "target_cell_width" if along == "h" else "target_cell_height"
#         raise BadDimensionsException(
#             f"{length_str} > {cell_str}: {secondary_length} > {secondary_max}"
#         )


#     target_bbox_height = max(target_bbox_height, 0)
#     target_bbox_height = min(target_bbox_height, target_cell_height)

#     if target_bbox_height > bbox.h:
#         # we dont size up
#         raise BadDimensionsException(
#             "target bbox size cannot be greater than original size"
#         )

#     aspect_ratio = target_bbox_height / bbox.h
#     target_bbox_width = math.ceil(aspect_ratio * bbox.w)

#     if target_bbox_height > target_cell_height:
#         raise Exception(
#             f"target_bbox_height > target_cell_height: {target_bbox_height} > {target_cell_height}"
#         )
#     if target_bbox_width > target_cell_width:
#         raise BadDimensionsException(
#             f"target_bbox_width > target_cell_width: {target_bbox_width} > {target_cell_width}"
#         )

#     if aspect_ratio == 0:
#         raise BadDimensionsException("aspect ratio is zero")


def split_leftovers(scaled_leftover, bbox_left, bbox_right, left_min, right_max):
    left_length = random.randint(0, scaled_leftover)

    left_idx = bbox_left - left_length
    left_idx = max(left_idx, left_min)

    actual_left_length = bbox_left - left_idx
    right_length = scaled_leftover - actual_left_length

    right_idx = bbox_right + right_length
    right_idx = min(right_idx, right_max)
    actual_right_length = bbox_right - right_idx

    actual_total_length = (
        actual_left_length + (bbox_right - bbox_left) + actual_right_length
    )

    return left_idx, right_idx, actual_total_length


def _split_dist_in_2(dist: int) -> tuple[int, int]:
    splitter = random.randint(0, dist)
    return splitter, dist - splitter


def resize_bbox_in_img(
    img,
    bbox: Bbox,
    target_height,
    target_width,
    scaled_leftover_height,
    scaled_leftover_width,
    interp=cv2.INTER_AREA,
):
    # the bbox can be fully the image
    # it is already cropped to img dims

    # if the bbox height or width is greater than that of the image
    # then we need to pad it
    crop = img[bbox.y : bbox.y2, bbox.x : bbox.x2]
    pad_if_needed = A.PadIfNeeded(target_height, target_width, position="center")
    return cv2.resize(crop, (target_width, target_height), interpolation=interp)
    # return pad_if_needed(image=res)["image"]


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
