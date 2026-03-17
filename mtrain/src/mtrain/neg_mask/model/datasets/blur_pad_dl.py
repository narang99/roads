from mtrain.neg_mask.model.datasets.crop_dataset_base import (
    MASK_DS_IMAGENET_MEAN,
    MASK_DS_IMAGENET_STD,
)
from mtrain.disk import DiskImage, DiskBooleanMask
from tqdm import tqdm
from torchvision.transforms import v2
import cv2
import os
from typing import Optional, Callable, Tuple, List
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from PIL import Image
import numpy as np

# Import ImageNet normalization constants from existing codebase
from mtrain.neg_mask.crops import Bbox

DEFAULT_LABEL_BY_IDX = {
    "other": 0,
    "trash": 1,
}


def _id_after_tfms(t_crop, t_mask):
    return t_crop, t_mask

class BlurPadInferDataset(Dataset):
    LABEL_BY_IDX = DEFAULT_LABEL_BY_IDX

    def __init__(self, crops, masks, bboxes: list[Bbox], crop_size, crop_mutator):
        self.crop_size = crop_size
        self.crops, self.masks = crops, masks
        self.bboxes = [
            make_box_of_crop_size_centered_at_box(mask.shape, bbox, crop_size)
            for mask, bbox in zip(masks, bboxes)
        ]
        self.tfms = get_valid_tfms(self.crop_size)
        self.crop_mutator = crop_mutator

    @classmethod
    def label_func(cls, image_path):
        return BlurPadDataset.label_func(image_path)

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        image, mask = self.crops[index], self.masks[index]
        bbox, inner_bbox = self.bboxes[index]

        crop, cropped_mask = _get_crops(image, mask, bbox)
        crop = self.crop_mutator(crop, cropped_mask, inner_bbox)
        t_crop, t_mask = _with_tfms(crop, cropped_mask, self.tfms)

        return t_crop


class BlurPadStepEdgeInferDataset(BlurPadInferDataset):
    def __getitem__(self, index):
        image, mask = self.crops[index], self.masks[index]
        bbox, inner_bbox = self.bboxes[index]

        crop, cropped_mask = _get_crops(image, mask, bbox)
        crop = self.crop_mutator(crop, cropped_mask, inner_bbox)
        step_edge_mask = mask_to_step_mask(cropped_mask, inner_bbox)
        t_crop, t_mask = _with_tfms(crop, step_edge_mask, self.tfms)
        return t_crop * t_mask


def noise_adder(max_noise):
    def wrapped(crop, mask, inner_bbox):
        return (
            CropTfmsOutsideBbox(crop, inner_bbox)
            .add_noise(max_noise)
            .crop
        )
    return wrapped


def blur_overwriter(blur_kernel_sz, blur_sigma):
    def wrapped(crop, mask, inner_bbox):
        return (
            CropTfmsOutsideBbox(crop, inner_bbox)
            .overwrite_with_blur(blur_kernel_sz, blur_sigma)
            .crop
        )

    return wrapped


class BlurPadDataset(Dataset):
    LABEL_BY_IDX = DEFAULT_LABEL_BY_IDX

    def __init__(self, image_paths, mask_dir, crop_size, is_valid, max_noise=None):
        self.crop_size = crop_size
        self.image_paths, self.mask_paths, self.img_name_by_bbox = (
            get_image_mask_and_bbox(image_paths, mask_dir, crop_size)
        )
        self._labels = [label_func(i.name) for i in self.image_paths]
        self.is_valid = is_valid
        self._train_tfms = get_train_tfms(self.crop_size)
        self._valid_tfms = get_valid_tfms(self.crop_size)

        self.max_noise = max_noise

    @classmethod
    def label_func(cls, image_path):
        return label_func(Path(image_path).name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        tfms = self._valid_tfms if self.is_valid else self._train_tfms
        image, mask, bbox, inner_bbox = self.get_loaded_objects(index)

        crop, cropped_mask = _get_crops(image, mask, bbox)
        crop = (
            CropTfmsOutsideBbox(crop, inner_bbox)
            .overwrite_with_noise(self.max_noise)
            .crop
        )
        t_crop, t_mask = _with_tfms(crop, cropped_mask, tfms)

        label_tensor = get_label_tensor(self._labels[index], self.LABEL_BY_IDX)
        return t_crop, label_tensor

    def get_loaded_objects(self, index):
        image_path = self.image_paths[index]
        image, mask = (
            DiskImage.load(image_path),
            DiskBooleanMask.load(self.mask_paths[index]),
        )
        bbox, inner_bbox = self.img_name_by_bbox[image_path.stem]
        return image, mask, bbox, inner_bbox


class BlurPadGaussianDataset(BlurPadDataset):
    def __getitem__(self, index):
        # similar to the original
        # but we first mutate the mask to get a grayscale step edge mask
        # the mask would have 1 where the bbox is (as before)
        # it would have 0.3 elsewhere
        # we multiply this with the original image to decrease the intensity of pixels outside bbox

        tfms = self._valid_tfms if self.is_valid else self._train_tfms
        image, mask, bbox, inner_bbox = self.get_loaded_objects(index)
        crop, cropped_mask = _get_crops(image, mask, bbox)
        crop = CropTfmsOutsideBbox(crop, inner_bbox).add_noise(self.max_noise).crop
        step_edge_mask = mask_to_step_mask(cropped_mask, inner_bbox)
        t_crop, t_mask = _with_tfms(crop, step_edge_mask, tfms)
        combined = t_crop * t_mask
        label_tensor = get_label_tensor(self._labels[index], self.LABEL_BY_IDX)
        return combined, label_tensor


class BlurPad4ChanDataset(BlurPadDataset):
    def __getitem__(self, index):
        tfms = self._valid_tfms if self.is_valid else self._train_tfms
        image_path = self.image_paths[index]
        image, mask = (
            DiskImage.load(image_path),
            DiskBooleanMask.load(self.mask_paths[index]),
        )
        bbox, inner_bbox = self.img_name_by_bbox[image_path.stem]
        t_crop, t_mask = get_transformed_pair(
            image, mask, bbox, inner_bbox, self.max_noise, tfms
        )
        combined = torch.cat([t_crop, t_mask])
        label_tensor = get_label_tensor(self._labels[index], self.LABEL_BY_IDX)
        return combined, label_tensor


def mask_to_step_mask(mask, bbox: Bbox):
    # instead of gaussian
    # we just put 1s in mask box
    # 1/3rd everyone else abruptly
    # Get bounding box limits
    y_min, x_min = bbox.y, bbox.x
    y_max, x_max = bbox.y2, bbox.x2

    gray = np.ones(mask.shape, dtype=np.float32) * 0.8
    gray[y_min:y_max, x_min:x_max] = 1.0
    return gray


def mask_to_gaussian(mask, bbox: Bbox, min_value=0.3):
    # 1. Get bounding box limits
    y_min, x_min = bbox.y, bbox.x
    y_max, x_max = bbox.y2, bbox.x2

    # 2. Calculate Center (x0, y0)
    y0, x0 = (y_min + y_max) / 2, (x_min + x_max) / 2

    # 3. Calculate Sigma
    # Note: If you want 1 sigma to be the full width,
    # sigma = (max - min). If you want 1 sigma to be half-width,
    # use (max - min) / 2.
    ratio = 1.0
    sigma_y = ratio * (y_max - y_min)
    sigma_x = ratio * (x_max - x_min)

    sigma_x = max(sigma_x, 1e-6)
    sigma_y = max(sigma_y, 1e-6)

    # 4. Create the Coordinate Grid
    h, w = mask.shape
    y, x = np.ogrid[0:h, 0:w]

    # 5. Apply the 2D Gaussian Formula
    exponent = -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
    gaussian_kernel = np.exp(exponent)

    # 6. Rescale to [min_value, 1.0]
    # This ensures the peak is 1.0 and the "floor" is your min_value
    gaussian_kernel = min_value + (1 - min_value) * gaussian_kernel

    return gaussian_kernel


def get_label_tensor(label, label_by_idx):
    label_idx = label_by_idx[label]
    return torch.tensor(label_idx, dtype=torch.long)


def get_step_edge_pair(image, full_mask, bbox, inner_bbox, max_noise, tfms):
    crop, cropped_mask = _get_crops(image, full_mask, bbox)
    crop = CropTfmsOutsideBbox(crop, inner_bbox).add_noise(max_noise).crop
    step_edge_mask = mask_to_step_mask(cropped_mask, inner_bbox)
    return _with_tfms(crop, step_edge_mask, tfms)


def get_transformed_pair(image, full_mask, bbox, inner_bbox, max_noise, tfms):
    crop, cropped_mask = _get_crops(image, full_mask, bbox)
    crop = CropTfmsOutsideBbox(crop, inner_bbox).overwrite_with_noise(max_noise).crop
    return _with_tfms(crop, cropped_mask, tfms)


def _get_crops(image, full_mask, bbox):
    crop = image[bbox.y : bbox.y2, bbox.x : bbox.x2]
    cropped_mask = full_mask[bbox.y : bbox.y2, bbox.x : bbox.x2]
    return crop, cropped_mask


def _with_tfms(crop: np.ndarray, cropped_mask: np.ndarray, tfms):
    t_crop = tv_tensors.Image(crop).permute([2, 0, 1])
    t_cropped_mask = tv_tensors.Mask(cropped_mask.reshape(1, *cropped_mask.shape))
    t_crop, t_mask = tfms((t_crop, t_cropped_mask))
    return t_crop, t_mask


def _add_noise_if_needed(crop, inner_bbox, max_noise):
    if max_noise is not None:
        noisy = np.random.randint(0, max_noise, crop.shape, dtype=np.uint8)
        y, y2, x, x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
        noisy[y:y2, x:x2] = 0
        crop = crop + noisy
    return crop


def _overwrite_outside_bbox_with_noise_if_needed(crop, inner_bbox, max_noise):
    if max_noise is not None:
        noisy = np.random.randint(0, max_noise, crop.shape, dtype=np.uint8)
        y, y2, x, x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
        noisy[y:y2, x:x2] = crop[y:y2, x:x2]
        crop = noisy
    return crop


def _overwrite_outside_bbox_with_blur_if_needed(crop, inner_bbox, blur_params=None):
    if blur_params is not None:
        blur_kernel_sz, blur_sigma = blur_params
        blurred = cv2.GaussianBlur(crop, (blur_kernel_sz, blur_kernel_sz), blur_sigma)
        y, y2, x, x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
        blurred[y:y2, x:x2] = crop[y:y2, x:x2]
        crop = blurred
    return crop


class CropTfmsOutsideBbox:
    def __init__(self, crop, inner_bbox):
        self.crop, self.inner_bbox = crop, inner_bbox

    def overwrite_with_blur(self, blur_kernel_sz=None, blur_sigma=None):
        if blur_kernel_sz is None or blur_sigma is None:
            return self

        crop, inner_bbox = self.crop, self.inner_bbox
        blurred = cv2.GaussianBlur(crop, (blur_kernel_sz, blur_kernel_sz), blur_sigma)
        y, y2, x, x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
        blurred[y:y2, x:x2] = crop[y:y2, x:x2]
        return CropTfmsOutsideBbox(blurred, self.inner_bbox)

    def overwrite_with_noise(self, max_noise=None):
        if max_noise is None:
            return self

        crop, inner_bbox = self.crop, self.inner_bbox
        noisy = np.random.randint(0, max_noise, crop.shape, dtype=np.uint8)
        y, y2, x, x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
        noisy[y:y2, x:x2] = crop[y:y2, x:x2]
        return CropTfmsOutsideBbox(noisy, self.inner_bbox)

    def add_noise(self, max_noise):
        if max_noise is None:
            return self

        crop, inner_bbox = self.crop, self.inner_bbox
        noisy = np.random.randint(0, max_noise, crop.shape, dtype=np.uint8)
        y, y2, x, x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
        noisy[y:y2, x:x2] = 0
        crop = crop + noisy
        return CropTfmsOutsideBbox(crop, self.inner_bbox)


def get_image_mask_and_bbox(image_paths, mask_dir, crop_size):
    image_paths = list(image_paths)
    mask_paths = [mask_dir / f"{i.stem}.png" for i in image_paths]
    img_name_by_bbox = get_coords_for_set(mask_paths, crop_size)
    image_paths, mask_paths = get_filtered_images_and_masks(
        image_paths, mask_paths, img_name_by_bbox
    )
    return image_paths, mask_paths, img_name_by_bbox


def get_filtered_images_and_masks(image_paths, mask_paths, img_name_by_bbox):
    images, masks = [], []
    for image_path, mask_path in zip(image_paths, mask_paths):
        bbox = img_name_by_bbox.get(image_path.stem)
        if bbox is None:
            continue
        images.append(image_path)
        masks.append(mask_path)
    return images, masks


def get_coords_for_set(mask_paths, crop_size) -> dict[str, Bbox]:
    res = {}
    for m in tqdm(mask_paths, desc="Getting Crop Coords"):
        bbox, inner_bbox = get_center_crop_coords_from_mask(
            DiskBooleanMask.load(m), crop_size
        )
        if bbox is None:
            print(f"SKIP: empty mask {m}")
            continue
        res[m.stem] = (bbox, inner_bbox)
    return res


# redefined region crops, its okay for now
def get_region_crops(mask):
    _, labels = cv2.connectedComponents(mask)
    h, w = mask.shape
    for label in range(1, labels.max() + 1):
        rows, cols = np.where(labels == label)
        r1 = max(0, rows.min())
        r2 = min(h, rows.max())
        c1 = max(0, cols.min())
        c2 = min(w, cols.max())
        yield Bbox(c1, r1, c2 - c1, r2 - r1)


def get_center_crop_coords_from_mask(mask, crop_size):
    bboxes = list(get_region_crops(mask))
    if not bboxes:
        return None, None

    bbox = bboxes[0]
    return make_box_of_crop_size_centered_at_box(mask.shape, bbox, crop_size)


def make_box_of_crop_size_centered_at_box(image_shape, bbox:Bbox, crop_size):
    center_x, center_y = bbox.x + (bbox.w // 2), bbox.y + (bbox.h // 2)

    left_pad = crop_size // 2

    left_x = max(center_x - left_pad, 0)
    up_y = max(center_y - left_pad, 0)

    right_x = min(left_x + crop_size, image_shape[1])
    down_y = min(up_y + crop_size, image_shape[0])

    inner_bbox = Bbox(bbox.x - left_x, bbox.y - up_y, bbox.w, bbox.h)

    return Bbox(left_x, up_y, right_x - left_x, down_y - up_y), inner_bbox


def label_func(x):
    if x.startswith("other"):
        return "other"
    elif x.startswith("trash"):
        return "trash"
    else:
        raise Exception(f"bad file name {x}")


def mask_to_gaussian_torch(mask):
    """
    Converts a binary mask tensor to a Gaussian grayscale tensor.
    Input: torch.Tensor of shape (H, W)
    Output: torch.Tensor of shape (H, W)
    """
    device = mask.device

    # 1. Find the coordinates of the filled box (1s)
    # .nonzero() is the PyTorch equivalent of np.argwhere
    coords = torch.nonzero(mask)

    if coords.shape[0] == 0:
        return torch.zeros_like(mask, dtype=torch.float32)

    # Get bounding box limits (y is dim 0, x is dim 1)
    y_min, x_min = coords.min(dim=0).values
    y_max, x_max = coords.max(dim=0).values

    # 2. Calculate Center (y0, x0)
    y0 = (y_min + y_max) / 2.0
    x0 = (x_min + x_max) / 2.0

    # 3. Calculate Sigma
    # Sigma 1 falls on the bbox ends
    sigma_y = (y_max - y_min) / 2.0
    sigma_x = (x_max - x_min) / 2.0

    # Stability: prevent division by zero
    sigma_y = torch.clamp(sigma_y, min=1e-6)
    sigma_x = torch.clamp(sigma_x, min=1e-6)

    # 4. Create the Coordinate Grid
    h, w = mask.shape
    y_range = torch.arange(0, h, device=device, dtype=torch.float32)
    x_range = torch.arange(0, w, device=device, dtype=torch.float32)

    # indexing='ij' ensures y is the first dim and x is the second
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")

    # 5. Apply the 2D Gaussian Formula
    # f(x,y) = exp( -0.5 * ( ((x-x0)/sx)^2 + ((y-y0)/sy)^2 ) )
    # Note: the 2 in the denominator is handled by the 0.5 multiplier
    exponent = -0.5 * (((grid_x - x0) / sigma_x) ** 2 + ((grid_y - y0) / sigma_y) ** 2)

    return torch.exp(exponent)


def get_train_tfms(crop_size):
    return v2.Compose(
        [
            v2.Resize(size=crop_size - 1, max_size=crop_size),
            v2.CenterCrop(crop_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MASK_DS_IMAGENET_MEAN, std=MASK_DS_IMAGENET_STD),
            v2.ToPureTensor(),
        ]
    )


def get_valid_tfms(crop_size):
    return v2.Compose(
        [
            v2.Resize(size=crop_size - 1, max_size=crop_size),
            v2.CenterCrop(crop_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MASK_DS_IMAGENET_MEAN, std=MASK_DS_IMAGENET_STD),
            v2.ToPureTensor(),
        ]
    )
