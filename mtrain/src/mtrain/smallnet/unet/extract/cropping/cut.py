from PIL import Image
import numpy as np
import random
from mtrain.random import random_bool
from mtrain.smallnet.unet.extract.cropping.utils import get_annotation_box


def get_crop_mode():
    # which corner should be inside the object to start cropping from
    top_corner = random_bool()
    left_corner = random_bool()

    if top_corner and left_corner:
        return "TL"
    elif top_corner and not left_corner:
        return "TR"
    elif not top_corner and left_corner:
        return "BL"
    elif not top_corner and not left_corner:
        return "BR"
    raise Exception(f"top={top_corner} left={left_corner}")


def extract_crop_by_cutting_object(coco, img_path, mask_path, max_pad_scale):
    # we take a crop by starting or ending with a point inside the object
    # Pick a random annotation to center the crop around
    img_id = int(img_path.stem)
    img_array = np.array(Image.open(img_path))
    mask_array = np.array(Image.open(mask_path))
    H, W = img_array.shape[:2]
    x, y, w, h = get_annotation_box(coco, img_id)

    crop_mode = get_crop_mode()
    coords = _get_coords(crop_mode, W, H, max_pad_scale, x, y, w, h)
    crop_x1, crop_y1, crop_x2, crop_y2 = coords
    crop_img = img_array[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_mask = mask_array[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop_img, crop_mask


def _get_coords(crop_mode, W, H, max_pad_scale, x, y, w, h):
    # corner = (x + random.randint(0, int(0.6 * w)), y + random.randint(0, int(0.6 * h)))
    corner = _get_point_inside(x, y, w, h)

    horiz_pad = random.randint(0, int(max_pad_scale * w))
    vert_pad = random.randint(0, int(max_pad_scale * h))
    horiz_pad = max(horiz_pad, 100)
    vert_pad = max(vert_pad, 100)

    if crop_mode == "TL":
        # corner is in top left, the next coordinates are on the right and bottom
        crop_x1, crop_y1 = corner
        crop_x2 = x + w + horiz_pad
        crop_y2 = y + h + vert_pad
    elif crop_mode == "TR":
        crop_x2, crop_y1 = corner
        crop_x1 = x - horiz_pad
        crop_y2 = y + h + vert_pad
    elif crop_mode == "BL":
        crop_x1, crop_y2 = corner
        crop_x2 = x + w + horiz_pad
        crop_y1 = y - vert_pad
    elif crop_mode == "BR":
        crop_x2, crop_y2 = corner
        crop_x1 = x - horiz_pad
        crop_y1 = y - vert_pad
    else:
        raise Exception(f"bad crop mode {crop_mode}")

    crop_x1 = int(max(0, crop_x1))
    crop_y1 = int(max(0, crop_y1))
    crop_x2 = int(min(crop_x2, W))
    crop_y2 = int(min(crop_y2, H))

    return crop_x1, crop_y1, crop_x2, crop_y2


def _get_point_inside(x, y, w, h):
    w_pct_40 = int(0.4 * w)
    h_pct_40 = int(0.4 * h)
    w_pct_60 = int(0.6 * w)
    h_pct_60 = int(0.6 * h)

    return (
        x + random.randint(w_pct_40, w_pct_60),
        y + random.randint(h_pct_40, h_pct_60),
    )
