import numpy as np
import cv2
from mtrain.neg_mask.crops import Bbox
from mtrain.neg_mask.crops import get_largest_bbox, padded_bbox
import albumentations as A


def get_foviated_image_and_mask(img, mask, bbox, full_image_size, crop_size, bbox_pad):
    pre_tfm = A.Compose(
        [
            A.Resize(full_image_size, full_image_size, cv2.INTER_AREA),
            A.PadIfNeeded(
                full_image_size, full_image_size, border_mode=cv2.BORDER_CONSTANT
            ),
        ],
        additional_targets={"bbox_mask": "mask"},
    )

    post_tfm = A.Compose(
        [
            A.Resize(crop_size, crop_size, cv2.INTER_AREA),
            A.PadIfNeeded(crop_size, crop_size, border_mode=cv2.BORDER_CONSTANT),
        ],
        additional_targets={"bbox_mask": "mask"},
    )

    res = pre_tfm(
        image=img,
        mask=mask,
        bbox_mask=_get_bbox_mask(
            padded_bbox(bbox, 10, mask.shape), mask.shape
        ),
    )
    t_image, t_mask, t_bbox_mask = res["image"], res["mask"], res["bbox_mask"]
    t_bb = get_largest_bbox(t_bbox_mask)
    map_x, map_y = get_foviate_remaps(t_image.shape, t_bb, crop_size)
    re_img = cv2.remap(t_image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    re_mask = cv2.remap(t_mask, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    res = post_tfm(image=re_img, mask=re_mask)
    re_img, re_mask = res["image"], res["mask"]

    return (t_image, t_mask), (re_img, re_mask)


def _get_bbox_mask(bb, shape):
    zero = np.zeros(shape)
    zero[bb.y : bb.y2, bb.x : bb.x2] = 1
    return zero



# def save_foveated_crops(
#     crop_level_dir, root_dest_dir, full_image_size, crop_size, bbox_pad
# ):
#     it = get_foviated_clean_crops(CROP_LEVEL_DIR, full_image_size, crop_size, bbox_pad)
#     root_images_dir = mkdir(root_dest_dir / "train")
#     root_masks_dir = mkdir(root_dest_dir / "masks")
#     for item in tqdm(it):
#         (img, mask), (re_img, re_mask), label, name = item
#         fname = f"{label}_{name}"
#         DiskImage.save(re_img, root_images_dir / f"{fname}.jpg")
#         DiskBooleanMask.save(re_mask, root_masks_dir / f"{fname}.png")


def get_foviate_remaps(shape, bbox: Bbox, total_desired_length, sigma=40):
    # Generate maps for both axes
    map_x_1d = _get_remap_array(bbox.x, bbox.x2, shape[1], total_desired_length, sigma)
    map_y_1d = _get_remap_array(bbox.y, bbox.y2, shape[0], total_desired_length, sigma)

    # Create the 2D grid
    map_x, map_y = np.meshgrid(map_x_1d, map_y_1d)
    return map_x, map_y


def do_foviate_shrink(img, bbox: Bbox, total_desired_length, sigma=40):
    map_x, map_y = get_foviate_remaps(img.shape, bbox, total_desired_length, sigma)

    # Remap - this pulls the WHOLE 1024 image into the 224 canvas
    re_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return re_img


def _get_remap_array(left_point, right_point, original_length, desired_length, sigma):
    left_len, right_len = _get_downsampled_portions(
        left_point, right_point, original_length, desired_length
    )

    left_x_map_1d = _get_scaled_array(left_point, left_len, sigma=sigma)
    right_x_map_1d = _get_scaled_array(
        original_length - right_point, right_len, sigma=sigma
    )

    right_range = right_point + right_x_map_1d
    left_range = left_point - np.flip(left_x_map_1d)
    res = np.concatenate(
        [left_range, np.arange(left_point, right_point, 1), right_range]
    )
    return res.astype(np.float32)


def _get_downsampled_portions(bb_left, bb_right, length, desired_length):
    desired_left = int((bb_left * desired_length) / length)
    right_length = length - bb_right
    desired_right = int((right_length * desired_length) / length)
    return desired_left, desired_right


def _get_scaled_array(orig_size, downsampled_size, ampl=10.0, sigma=20.0):
    # walk k (224) steps to cover n (1024) steps
    # we create an array of size downsampled_size which describes the points to take from 0 -> orig_size

    out_steps = np.arange(0, downsampled_size, 1)
    weights = _get_weights(out_steps, ampl, sigma)
    total_weighted_steps = weights.sum()
    step_locs = weights.cumsum()

    return ((step_locs * orig_size) / total_weighted_steps).astype(np.float32)


def _get_weights(dist, ampl, sigma):
    return 1.0 + ampl * (1.0 - np.exp(-(dist**2) / (2 * sigma**2)))
