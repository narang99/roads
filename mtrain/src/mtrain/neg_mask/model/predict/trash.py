import itertools
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from fastai.vision.all import (
    DataLoaders,
    default_device,
    CrossEntropyLossFlat,
    vision_learner,
    ProgressCallback,
    Precision,
    resnet18,
)
from mtrain.neg_mask.crops import padded_crop, bbox_only_mask, get_region_crops, Bbox
from mtrain.neg_mask.model.predict.core import reconstruct_probability_masks
from mtrain.neg_mask.model.datasets.blur_pad_dl import (
    BlurPadDataset,
    BlurPadInferDataset,
    blur_overwriter, BlurPadStepEdgeInferDataset, noise_adder,
)


def predict_and_return_prob_masks_using_step_edge(
    image,
    mask,
    learner,
    crop_size=130,
    device=None,
    max_noise=30,
    bbox_pad=5,
):
    crops, masks, bboxes, inner_bboxes = get_crops_masks_bboxes(
        image, mask, crop_size, bbox_pad
    )
    ds = BlurPadStepEdgeInferDataset(
        crops,
        masks,
        inner_bboxes,
        crop_size,
        noise_adder(max_noise),
    )
    return _predict_and_return_probs(ds, learner, image, mask, bboxes)



def predict_and_return_prob_masks_using_unblurred(
    image,
    mask,
    learner,
    crop_size=130,
    device=None,
    blur_kernel_sz=5,
    blur_sigma=3,
    bbox_pad=5,
):
    crops, masks, bboxes, inner_bboxes = get_crops_masks_bboxes(
        image, mask, crop_size, bbox_pad
    )
    ds = BlurPadInferDataset(
        crops,
        masks,
        inner_bboxes,
        crop_size,
        blur_overwriter(blur_kernel_sz, blur_sigma),
    )
    return _predict_and_return_probs(ds, learner, image, mask, bboxes)


def _predict_and_return_probs(ds, learner, image, mask, bboxes):
    dl = DataLoader(ds, 4)
    learner.eval()
    with torch.no_grad():
        res = [learner.model(b).softmax(dim=1) for b in dl]
        probs = torch.stack(list(itertools.chain.from_iterable(res)))
        other_mask, trash_mask = reconstruct_probability_masks(
            image, mask, probs, bboxes
        )
        return other_mask, trash_mask


def get_crops_masks_bboxes(image, mask, crop_size, bbox_pad):
    bboxes = list(get_region_crops(mask))
    crops, masks, result_bboxes = [], [], []
    for bbox in tqdm(bboxes):
        tight_img, new_y1, new_x1 = padded_crop(image, bbox, crop_size)
        tight_mask = bbox_only_mask(mask, bbox, crop_size)
        inner_bbox = Bbox(bbox.x - new_x1, bbox.y - new_y1, bbox.w, bbox.h)
        crops.append(tight_img)
        masks.append(tight_mask)
        result_bboxes.append(inner_bbox)
    return crops, masks, bboxes, result_bboxes


def get_padded_bbox_mask(mask, bbox, padding=10):
    # 1. Find the coordinates of all non-zero pixels
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    img_h, img_w = mask.shape[:2]

    # 3. Apply padding with boundary constraints
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    # 4. Create the new mask
    padded_mask = np.zeros_like(mask)
    cv2.rectangle(padded_mask, (x1, y1), (x2, y2), 255, -1)

    return padded_mask, (x1, y1, x2, y2)


def get_learner():
    # dummy dls
    train_ds = BlurPadDataset([], Path("./masks"), 130, False, max_noise=None)
    valid_ds = BlurPadDataset([], Path("./masks"), 130, True, max_noise=None)
    dls = DataLoaders.from_dsets(
        train_ds,
        valid_ds,
        device=default_device(),
        num_workers=4,
        bs=16,
        persistent_workers=True,
    )

    learn = vision_learner(
        dls,
        resnet18,
        metrics=[Precision()],
        loss_func=CrossEntropyLossFlat(),
        n_out=2,
        normalize=False,
        n_in=3,
    )
    learn = learn.remove_cb(ProgressCallback)
    return learn


# from mtrain.neg_mask.model.datasets.blur_pad_dl import (
#     get_coords_for_set,
#     get_center_crop_coords_from_mask, BlurPadInferDataset, blur_overwriter,
# )
# from mtrain.neg_mask.leveled_cropping import (
#     create_crop_level_sample,
#     make_crop_level_pairs_v2,
#     load_crop_level_sample_from_directory,
# )
# from mtrain.neg_mask.model.predict.mk_tensors import (
#     mk_8_chan,
#     get_crop_pairs_from_full_image,
# )
# import cv2
# import numpy as np
# from mtrain.neg_mask.model.predict import core
# from mtrain.neg_mask.crops import get_region_crops, Bbox
# from mtrain.utils import DiskBooleanMask
# import torch


# def run_inference(
#     learner, crop_data_list, crop_size: int = 60, device=None
# ) -> torch.Tensor:
#     tensors = [
#         mk_8_chan(image, mask, bbox, crop_size) for image, mask, bbox in crop_data_list
#     ]
#     batch_tensor = torch.stack(tensors)  # Shape: [N, 8, crop_size+20, crop_size+20]

#     return core.run_inference(learner, batch_tensor, device)


# def predict_and_return_prob_masks(
#     image,
#     mask,
#     learner,
#     crop_size=130,
#     device=None,
#     blur_kernel_sz=5,
#     blur_sigma=3,
#     bbox_pad=5,
# ):
#     bboxes = list(get_region_crops(mask))
#     if not bboxes:
#         zero_mask = np.zeros(mask.shape, dtype=np.float32)
#         return zero_mask, zero_mask

#     crops, masks = get_crops_and_masks(image, mask, bboxes, crop_size, bbox_pad)

#     ds = BlurPadInferDataset(crops, masks, crop_size, blur_overwriter(blur_kernel_sz, blur_sigma))
#     test_dl = learner.dls.test_dl(ds)
#     return learner.get_preds(dl=test_dl)


# def get_crops_and_masks(image, mask, bboxes, crop_size, bbox_pad):
#     crops, masks = [], []
#     for bbox in bboxes:
#         sample = create_crop_level_sample(image, mask, bbox)
#         pairs_obj = make_crop_level_pairs_v2(sample, crop_size, crop_size + 100)
#         crop, crop_mask = pairs_obj.pairs[0]
#         crop_mask, inner_bbox = get_padded_bbox_mask(crop_mask, bbox_pad)
#         crops.append(crop)
#         masks.append(mask)
#     return crops, masks


# # to use: B5P5K5S3
# def get_blurred_artifacts(img, mask, blur_kernel_sz=5, blur_sigma=3, bbox_pad=5):
#     assert img is not None
#     new_mask, box = get_padded_bbox_mask(mask, bbox_pad)

#     blurred = cv2.GaussianBlur(img, (blur_kernel_sz, blur_kernel_sz), blur_sigma)
#     only_mask_unblurred = blurred.copy()

#     new_mask = new_mask.astype(bool)
#     only_mask_unblurred[new_mask] = img[new_mask]

#     return {
#         "padded_mask": new_mask,
#         "unblurred": only_mask_unblurred,
#         "img": img,
#     }


# def get_padded_bbox_mask(mask, padding=10):
#     # 1. Find the coordinates of all non-zero pixels
#     coords = cv2.findNonZero(mask)
#     if coords is None:
#         return np.zeros_like(mask), None

#     # 2. Get the standard bounding box
#     x, y, w, h = cv2.boundingRect(coords)
#     img_h, img_w = mask.shape[:2]

#     # 3. Apply padding with boundary constraints
#     x1 = max(0, x - padding)
#     y1 = max(0, y - padding)
#     x2 = min(img_w, x + w + padding)
#     y2 = min(img_h, y + h + padding)

#     # 4. Create the new mask
#     padded_mask = np.zeros_like(mask)
#     cv2.rectangle(padded_mask, (x1, y1), (x2, y2), 255, -1)

#     return padded_mask, (x1, y1, x2, y2)
