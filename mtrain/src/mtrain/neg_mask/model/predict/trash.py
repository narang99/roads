from mtrain.neg_mask.leveled_cropping import create_crop_level_sample, make_crop_level_pairs_v2, load_crop_level_sample_from_directory
from mtrain.neg_mask.model.predict.mk_tensors import mk_8_chan
import cv2
import numpy as np
from mtrain.neg_mask.model.predict import core
from mtrain.neg_mask.crops import get_region_crops, Bbox
from mtrain.utils import DiskBooleanMask
import torch


def run_inference(
    learner, crop_data_list, crop_size: int = 60, device=None
) -> torch.Tensor:
    tensors = [
        mk_8_chan(image, mask, bbox, crop_size) for image, mask, bbox in crop_data_list
    ]
    batch_tensor = torch.stack(tensors)  # Shape: [N, 8, crop_size+20, crop_size+20]

    return core.run_inference(learner, batch_tensor, device)


def get_unblurred(image, mask, crop_size=130, blur_kernel_sz=5, blur_sigma=3, bbox_pad=5):
    bboxes = list(get_region_crops(image, mask))
    if not bboxes:
        zero_mask = np.zeros(mask.shape, dtype=np.float32)
        return zero_mask, zero_mask
    

    unblurred = []
    for bbox in bboxes:
        sample = create_crop_level_sample(image, mask, bbox)
        level_pairs = make_crop_level_pairs_v2(sample, crop_size, crop_size+100, 1)
        tight_crop, _ = level_pairs.pairs[0]
        # we now have the tight_crop and the tight_mask
        # the mask can be used to get blurred artifacts
        artifacts = get_blurred_artifacts(tight_crop, bbox, blur_kernel_sz, blur_sigma, bbox_pad)
        unblurred.append(artifacts["unblurred"])
    
    return bboxes, unblurred


def predict_and_return_prob_masks(
    image, mask, learner, crop_size=130, device=None, blur_kernel_sz=5, blur_sigma=3, bbox_pad=5
):
    bboxes = list(get_region_crops(image, mask))
    if not bboxes:
        zero_mask = np.zeros(mask.shape, dtype=np.float32)
        return zero_mask, zero_mask

    unblurred = []
    for bbox in bboxes:
        artifacts = get_blurred_artifacts(image, bbox, blur_kernel_sz, blur_sigma, bbox_pad)
        crop = get_centered_crop(artifacts["unblurred"], bbox, crop_size)
        unblurred.append(crop)
    return crop
    
    # test_dl = learner.dls.test_dl(unblurred)
    # preds, _, decoded = learner.get_preds(dl=test_dl, with_decoded=True)
    # print(decoded)

    # input_tensor = dl.one_batch()[0]
    # mean, std = imagenet_stats


    # mask = mask.astype(bool)
    # crop_data_list = [(image, mask, bbox) for bbox in bboxes]
    # probs = run_inference(learner, crop_data_list, crop_size, device)
    # prob_masks = core.reconstruct_probability_masks(image, mask, probs, bboxes)

    # other_mask, trash_mask = prob_masks[0], prob_masks[1]
    # return other_mask, trash_mask


def get_centered_crop(array, bbox, crop_size=130):
    x, x2, y, y2 = bbox
    
    # 1. Find the center of the bounding box
    center_x = (x + x2) // 2
    center_y = (y + y2) // 2
    
    # 2. Calculate the crop boundaries
    half_size = crop_size // 2
    
    start_x = center_x - half_size
    end_x = start_x + crop_size
    start_y = center_y - half_size
    end_y = start_y + crop_size
    
    # 3. Handle Out-of-Bounds (Padding)
    # If the crop goes outside the original array, we need to pad
    pad_left = max(0, -start_x)
    pad_top = max(0, -start_y)
    pad_right = max(0, end_x - array.shape[1])
    pad_bottom = max(0, end_y - array.shape[0])
    
    # Adjust slicing coordinates to stay within array bounds
    slice_x1, slice_x2 = max(0, start_x), min(array.shape[1], end_x)
    slice_y1, slice_y2 = max(0, start_y), min(array.shape[0], end_y)
    
    # Extract the valid portion
    crop = array[slice_y1:slice_y2, slice_x1:slice_x2]
    
    # Apply padding if the crop was near the edge
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        # Assuming a 2D or 3D array (RGB)
        padding = [(pad_top, pad_bottom), (pad_left, pad_right)]
        if array.ndim == 3:
            padding.append((0, 0)) # Don't pad the color channels
            
        crop = np.pad(crop, padding, mode='constant', constant_values=0)
        
    return crop


# to use: B5P5K5S3
def get_blurred_artifacts(
    img, bbox: Bbox, blur_kernel_sz=5, blur_sigma=3, bbox_pad=5
):
    assert img is not None
    new_mask, box = get_padded_bbox_mask(img.shape, bbox, bbox_pad)

    blurred = cv2.GaussianBlur(img, (blur_kernel_sz, blur_kernel_sz), blur_sigma)
    only_mask_unblurred = blurred.copy()

    new_mask = new_mask.astype(bool)
    only_mask_unblurred[new_mask] = img[new_mask]

    return {
        "padded_mask": new_mask,
        "unblurred": only_mask_unblurred,
        "img": img,
    }


def get_padded_bbox_mask(shape, bbox: Bbox, padding=5):
    # 2. Get the standard bounding box
    img_h, img_w = shape[:2]

    # 3. Apply padding with boundary constraints
    x1 = max(0, bbox.x - padding)
    y1 = max(0, bbox.y - padding)
    x2 = min(img_w, bbox.x2 + padding)
    y2 = min(img_h, bbox.y2 + padding)

    # 4. Create the new mask
    padded_mask = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(padded_mask, (x1, y1), (x2, y2), 255, -1)

    return padded_mask, (x1, y1, x2, y2)

