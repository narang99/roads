from PIL import Image
import math
import numpy as np
import random
from mtrain.smallnet.unet.extract.cropping.utils import get_annotation_box
from mtrain.random import add_jitter_pixels


def get_num_annotations(coco, img_path):
    img_id = int(img_path.stem)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    return len(ann_ids)


def extract_single_crop(
    coco,
    img_path,
    mask_path,
    max_pad_scale,
    horiz_skew,
    vert_skew,
    min_padding=None,
    max_padding=None,
    ann_idx=None,
):
    # padding calculation
    # first the padding scale is used to calculate the max padding vertically and horizontally
    # then we use skew to define which side wins
    # horiz_skew: if positive, the left max padding is decreased, else right is decreased
    # similar for vert_skew

    # Pick a random annotation to center the crop around
    img_id = int(img_path.stem)
    img_array = np.array(Image.open(img_path))
    H, W = img_array.shape[:2]
    mask_array = np.array(Image.open(mask_path))
    x, y, w, h = get_annotation_box(coco, img_id, ann_idx)

    # create max_padding based on max_pad_scale
    coords = _get_coords(
        W, H, max_pad_scale, horiz_skew, vert_skew, x, y, w, h, min_padding, max_padding
    )
    crop_x1, crop_y1, crop_x2, crop_y2 = coords

    # Crop image and mask
    crop_img = img_array[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_mask = mask_array[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop_img, crop_mask


def _get_coords(
    W,
    H,
    max_pad_scale,
    horiz_skew,
    vert_skew,
    x,
    y,
    w,
    h,
    min_padding,
    max_padding,
):
    # create max_padding based on max_pad_scale
    # this however has a problem of making our paddings very dependent on the height and width of the object
    FIX_MAX_PADDING = 4000
    max_padding = FIX_MAX_PADDING if max_padding is None else max_padding
    horiz_max_pad = max_padding
    # horiz_max_pad = (
    #     max_padding if random_true_one_three_times() else int(max_pad_scale * w)
    # )
    vert_max_pad = max_padding
    # vert_max_pad = (
    #     max_padding if random_true_one_three_times() else int(max_pad_scale * h)
    # )
    horiz_max_pad = add_jitter_pixels(min(horiz_max_pad, max_padding))
    vert_max_pad = add_jitter_pixels(min(vert_max_pad, max_padding))

    # Random padding on each side
    max_left, max_right = _get_paddings(horiz_max_pad, horiz_skew)
    max_top, max_bottom = _get_paddings(vert_max_pad, vert_skew)

    min_padding = 0 if min_padding is None else min_padding
    pad_left = _get_random_pad(min_padding, max_left)
    pad_left = _get_random_pad(min_padding, max_left)
    pad_right = _get_random_pad(min_padding, max_right)
    pad_top = _get_random_pad(min_padding, max_top)
    pad_bottom = _get_random_pad(min_padding, max_bottom)

    # Calculate crop boundaries
    crop_x1 = max(0, int(x - pad_left))
    crop_y1 = max(0, int(y - pad_top))
    crop_x2 = min(W, int(x + w + pad_right))
    crop_y2 = min(H, int(y + h + pad_bottom))

    return crop_x1, crop_y1, crop_x2, crop_y2


def _get_random_pad(min_padding, right):
    left = min(add_jitter_pixels(min_padding), right)
    return random.randint(left, right)


def _get_paddings(max_padding, skew):
    # we return before_padding and after_padding
    before = max_padding
    if skew < 0:
        before = max_padding
        after = math.floor(before / (-skew))
    else:
        after = max_padding
        before = math.floor(after / skew)
    return before, after
