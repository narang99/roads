# instead of min padding, we use min length in this version
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
    horiz_skew,
    vert_skew,
    max_padding=None,
    ann_idx=None,
    min_length=None,
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
        W, H, horiz_skew, vert_skew, x, y, w, h, max_padding, min_length,
    )
    crop_x1, crop_y1, crop_x2, crop_y2 = coords

    # Crop image and mask
    crop_img = img_array[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_mask = mask_array[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop_img, crop_mask


def _get_coords(
    W,
    H,
    horiz_skew,
    vert_skew,
    x,
    y,
    w,
    h,
    max_padding,
    min_length,
):
    # create max_padding based on max_pad_scale
    # this however has a problem of making our paddings very dependent on the height and width of the object
    FIX_MAX_PADDING = 4000
    max_padding = FIX_MAX_PADDING if max_padding is None else max_padding
    horiz_max_pad = max_padding
    vert_max_pad = max_padding
    horiz_max_pad = add_jitter_pixels(min(horiz_max_pad, max_padding))
    vert_max_pad = add_jitter_pixels(min(vert_max_pad, max_padding))

    # Random padding on each side
    max_left, max_right = _get_paddings(horiz_max_pad, horiz_skew)
    max_top, max_bottom = _get_paddings(vert_max_pad, vert_skew)

    pad_left = random.randint(1, max_left)
    pad_right = random.randint(1, max_right)
    pad_left, pad_right = _scale_paddings(pad_left, pad_right, min_length)
    pad_top = random.randint(1, max_top)
    pad_bottom = random.randint(1, max_bottom)
    pad_top, pad_bottom = _scale_paddings(pad_top, pad_bottom, min_length)

    # Calculate crop boundaries
    crop_x1 = max(0, int(x - pad_left))
    crop_y1 = max(0, int(y - pad_top))
    crop_x2 = min(W, int(x + w + pad_right))
    crop_y2 = min(H, int(y + h + pad_bottom))

    return crop_x1, crop_y1, crop_x2, crop_y2


def _scale_paddings(lpad, rpad, length):
    if length is None:
        return lpad, rpad
    # scale lpad and rpad such that they add up to length
    clen = lpad + rpad
    lpad = math.floor((lpad * length) / clen)
    rpad = math.floor((rpad * length) / clen)
    lpad = max(1, lpad)
    rpad = max(1, rpad)
    return lpad, rpad

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
