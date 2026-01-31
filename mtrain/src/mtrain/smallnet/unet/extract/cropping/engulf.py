from PIL import Image
import math
import numpy as np
import random
from mtrain.smallnet.unet.extract.cropping.utils import get_annotation_box
from mtrain.random import random_true_one_three_times


def extract_single_crop(
    coco, img_path, mask_path, max_pad_scale, horiz_skew, vert_skew
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
    x, y, w, h = get_annotation_box(coco, img_id)

    # create max_padding based on max_pad_scale
    coords = _get_coords(W, H, max_pad_scale, horiz_skew, vert_skew, x, y, w, h)
    crop_x1, crop_y1, crop_x2, crop_y2 = coords

    # Crop image and mask
    crop_img = img_array[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_mask = mask_array[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop_img, crop_mask


def _get_coords(W, H, max_pad_scale, horiz_skew, vert_skew, x, y, w, h):
    # create max_padding based on max_pad_scale
    # this however has a problem of making our paddings very dependent on the height and width of the object
    FIX_MAX_PADDING = 2000
    horiz_max_pad = FIX_MAX_PADDING if random_true_one_three_times() else int(max_pad_scale * w)
    vert_max_pad = FIX_MAX_PADDING if random_true_one_three_times() else int(max_pad_scale * h)


    # Random padding on each side
    max_left, max_right = _get_paddings(horiz_max_pad, horiz_skew)
    max_top, max_bottom = _get_paddings(vert_max_pad, vert_skew)
    pad_left = random.randint(0, max_left)
    pad_right = random.randint(0, max_right)
    pad_top = random.randint(0, max_top)
    pad_bottom = random.randint(0, max_bottom)


    # Calculate crop boundaries
    crop_x1 = max(0, int(x - pad_left))
    crop_y1 = max(0, int(y - pad_top))
    crop_x2 = min(W, int(x + w + pad_right))
    crop_y2 = min(H, int(y + h + pad_bottom))

    return crop_x1, crop_y1, crop_x2, crop_y2


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
