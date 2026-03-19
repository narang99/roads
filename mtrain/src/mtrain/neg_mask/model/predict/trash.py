import itertools
from functools import partial
from pathlib import Path
import numpy as np
import cv2
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
from mtrain.neg_mask.crops import (
    padded_crop,
    bbox_only_mask,
    get_region_crops,
    Bbox,
    padded_bbox,
)
from mtrain.neg_mask.model.predict.core import reconstruct_probability_masks
from mtrain.neg_mask.model.datasets.blur_pad_dl import (
    BlurPadDataset,
    BlurPadInferDataset,
    BlurPadStepEdgeInferDataset,
    noise_adder,
    CropTfmsOutsideBbox,
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


def blur_and_noise_adder(blur_kernel_sz, blur_sigma, max_noise):
    def wrapped(crop, mask, inner_bbox):
        return (
            CropTfmsOutsideBbox(crop, inner_bbox)
            .overwrite_with_blur(blur_kernel_sz, blur_sigma)
            .add_noise(max_noise)
            .crop
        )

    return wrapped


def step_down_gauss_tfm(cropped_image, mask, inner_bbox, min_value):
    tfm = CropTfmsOutsideBbox(cropped_image, inner_bbox)
    tfm = tfm.step_down_gaussian(min_value)
    return tfm.crop

def step_down_edge_tfm(cropped_image, mask, inner_bbox, ratio):
    tfm = CropTfmsOutsideBbox(cropped_image, inner_bbox)
    tfm = tfm.step_down(ratio)
    return tfm.crop

def zero_overwriter(crop, crop_mask, inner_bbox):
    return CropTfmsOutsideBbox(crop, inner_bbox).overwrite_with_zeros().crop


def id_mutator(crop, mask, bbox):
    return crop

def predict_and_return_prob_masks_using_unblurred(
    image,
    mask,
    learner,
    crop_size=130,
    device=None,
    bbox_pad=5,
    mutator=id_mutator,
):
    # mutator = partial(step_down_gauss_tfm, min_value=gaussian_step_down_min_val)
    # mutator = partial(step_down_edge_tfm, ratio=0.5)
    crops, masks, bboxes, inner_bboxes = get_crops_masks_bboxes(
        image, mask, crop_size, bbox_pad
    )

    ds = BlurPadInferDataset(
        crops,
        masks,
        inner_bboxes,
        crop_size,
        mutator,
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
    bboxes = get_region_crops(mask)
    bboxes = [_padded_inner_bbox(bbox, bbox_pad, mask.shape) for bbox in bboxes]
    crops, masks, result_bboxes = [], [], []
    for bbox in bboxes:
        tight_img, new_y1, new_x1 = padded_crop(image, bbox, crop_size)
        tight_mask = bbox_only_mask(mask, bbox, crop_size)
        inner_bbox = Bbox(bbox.x - new_x1, bbox.y - new_y1, bbox.w, bbox.h)
        crops.append(tight_img)
        masks.append(tight_mask)
        result_bboxes.append(inner_bbox)
    return crops, masks, bboxes, result_bboxes


def _padded_inner_bbox(bbox, bbox_pad, shape):
    return padded_bbox(bbox, bbox_pad, shape)


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
    train_ds = BlurPadDataset([], Path("./masks"), 130, False)
    valid_ds = BlurPadDataset([], Path("./masks"), 130, True)
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