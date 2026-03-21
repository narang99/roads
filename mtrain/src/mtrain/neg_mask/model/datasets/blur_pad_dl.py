import random
from mpmath.libmp.libmpi import gamma_min
from mtrain.neg_mask.model.datasets.crop_dataset_base import (
    MASK_DS_IMAGENET_MEAN,
    MASK_DS_IMAGENET_STD,
)
from mtrain.disk import DiskImage, DiskBooleanMask
from tqdm import tqdm
from torchvision.transforms import v2
import cv2
import os
from typing import Optional, Callable, Tuple, List, Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from PIL import Image
import numpy as np

# Import ImageNet normalization constants from existing codebase
from mtrain.neg_mask.crops import Bbox, padded_bbox

DEFAULT_LABEL_BY_IDX = {
    "other": 0,
    "trash": 1,
}


def _id_after_tfms(t_crop, t_mask):
    return t_crop, t_mask


class BlurPadInferDataset(Dataset):
    """Inference dataset equivalent for BlurPad and its variants

    The class is quite simple so that it can be used generically (its a slim wrapper for creating a dataset for inference)
    The params (crops, masks, bboxes) should be of equal length, each index contains the mask corresponding to the crop at that index
    The mask is assumed to have ONLY one object. And the corresponding passed bbox should be the bbox surrounding that single object
    This class can find the bbox itself,  but generally you pass many crops and masks for a single image
    Thus, the source of truth is simply one mask, on which cv2.connectedComponents is called
    We dont want to call it again, its quite slow. So we take the bboxes from the client. It is assumed it is generated using connectedComponents generaly

    The class would create a centered crop of appropriate size around the object (it uses the passed bbox for that)

    The class takes a crop_mutator, which is given the final crop and mask, and return the final crop
    Examples are adding noise

    NOTE: you should pass the padded bbox itself
    """

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
        print(f"Index {index}: crop shape before mutator: {crop.shape}")
        crop = self.crop_mutator(crop, cropped_mask, inner_bbox)
        print(f"Index {index}: crop shape after mutator: {crop.shape}")
        t_crop, t_mask = _with_tfms(crop, cropped_mask, self.tfms)
        print(f"Index {index}: t_crop shape after transforms: {t_crop.shape}")
        return t_crop


class BlurPadStepEdgeInferDataset(BlurPadInferDataset):
    """Step edge inference version of blurpadinferdataset. The equivalent of blurpadgaussian

    Check the doc of parent class for what to pass
    """

    def __getitem__(self, index):
        image, mask = self.crops[index], self.masks[index]
        bbox, inner_bbox = self.bboxes[index]

        crop, cropped_mask = _get_crops(image, mask, bbox)
        crop = self.crop_mutator(crop, cropped_mask, inner_bbox)
        step_edge_mask = get_step_edge_mask(cropped_mask.shape, inner_bbox)
        t_crop, t_mask = _with_tfms(crop, step_edge_mask, self.tfms)
        return t_crop * t_mask


def noise_adder(max_noise):
    def wrapped(crop, mask, inner_bbox):
        return CropTfmsOutsideBbox(crop, inner_bbox).add_noise(max_noise).crop

    return wrapped


def blur_overwriter(blur_kernel_sz, blur_sigma):
    def wrapped(crop, mask, inner_bbox):
        return (
            CropTfmsOutsideBbox(crop, inner_bbox)
            .overwrite_with_blur(blur_kernel_sz, blur_sigma)
            .crop
        )

    return wrapped


def _id_crop_mutator(crop, cropped_mask, inner_bbox):
    return crop


# class BlurPad2ChanDataset(Dataset):
#     LABEL_BY_IDX = DEFAULT_LABEL_BY_IDX

#     def __init__(
#         self, image_paths, mask_dir, crop_size, is_valid, small_pad=10, big_pad=128
#     ):
#         self.crop_size = crop_size
#         self.image_paths, self.mask_paths, self.img_name_by_small_bbox = (
#             get_image_mask_and_bbox(image_paths, mask_dir, crop_size, small_pad)
#         )
#         _, _, self.img_name_by_big_bbox = (
#             get_image_mask_and_bbox(image_paths, mask_dir, crop_size, big_pad)
#         )
#         self._labels = [label_func(i.name) for i in self.image_paths]
#         self.is_valid = is_valid
#         self._train_tfms = get_train_tfms(self.crop_size)
#         self._valid_tfms = get_valid_tfms(self.crop_size)

#     @classmethod
#     def label_func(cls, image_path):
#         return label_func(Path(image_path).name)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         tfms = self._valid_tfms if self.is_valid else self._train_tfms
#         image, mask, bbox, inner_bbox = self.get_loaded_objects(index)

#         crop, cropped_mask = _get_crops(image, mask, bbox)
#         crop = self.crop_mutator(crop, cropped_mask, inner_bbox)
#         t_crop, t_mask = _with_tfms(crop, cropped_mask, tfms)

#         label_tensor = get_label_tensor(self._labels[index], self.LABEL_BY_IDX)
#         return t_crop, label_tensor

#     def get_loaded_objects(self, index):
#         image_path = self.image_paths[index]
#         image, mask = (
#             DiskImage.load(image_path),
#             DiskBooleanMask.load(self.mask_paths[index]),
#         )
#         bbox, inner_bbox = self.img_name_by_bbox[image_path.stem]
#         return image, mask, bbox, inner_bbox


class BlurPadDataset(Dataset):
    """Dataset class used after creating data in using create_dataset in unblur.ipynb

    This class takes the set of images and the maskdir generated by create_dataset.
    It only cares about the inner bbox assigned by create_dataset to each image
    Everything outside that bbox is overwritten with noise. This is used for training 0pad models.
    I started out by putting noise in create_dataset itself, but the model would learn to find stuff using noise itself
    So I needed to change the noise in every iteration
    It does not matter what amount of blurring you do in create_dataset or anything, this will simply overwrite that with noise
    The padding around bbox is already done (by create dataset), we dont change anything.
    It is the responsibility of the masks in mask dir itself to make sure we have some padding around raw bboxes which are found in segmentation outputs
    """

    LABEL_BY_IDX = DEFAULT_LABEL_BY_IDX

    def __init__(
        self, image_paths, mask_dir, crop_size, is_valid, crop_mutator=_id_crop_mutator, bbox_pad=0, min_area=35, min_bbox_length=3, max_area=None,
    ):
        self.crop_size = crop_size
        self.min_area = min_area
        self.image_paths, self.mask_paths, self.img_name_by_bbox = (
            get_image_mask_and_bbox(image_paths, mask_dir, crop_size, bbox_pad, min_area, min_bbox_length, max_area)
        )
        self._labels = [label_func(i.name) for i in self.image_paths]
        self.is_valid = is_valid
        self._train_tfms = get_train_tfms(self.crop_size)
        self._valid_tfms = get_valid_tfms(self.crop_size)

        self.crop_mutator = crop_mutator

    @classmethod
    def label_func(cls, image_path):
        return label_func(Path(image_path).name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        tfms = self._valid_tfms if self.is_valid else self._train_tfms
        image, mask, bbox, inner_bbox = self.get_loaded_objects(index)

        crop, cropped_mask = _get_crops(image, mask, bbox)
        crop = self.crop_mutator(crop, cropped_mask, inner_bbox)
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


def random_tfm(cropped_image, mask, inner_bbox):
    tfm = CropTfmsOutsideBbox(cropped_image, inner_bbox)
    NOISE_OVERWRITE = 0
    BLUR_1_7 = 1
    BLUR_2_13 = 2
    STEP_EDGE_0_3 = 3
    STEP_EDGE_0_7 = 4
    STEP_GAUSE_0_3 = 5

    i = random.randint(0, 5)
    add_noise_change = random.randint(0,2)
    def _add_noise(tfm):
        if add_noise_change == 1:
            return tfm.add_noise(20)
        return tfm

    if i == NOISE_OVERWRITE:
        return tfm.overwrite_with_noise(30).crop
    elif i == BLUR_1_7:
        tfm = tfm.overwrite_with_blur(7, 1)
        return _add_noise(tfm).crop
    elif i == BLUR_2_13:
        tfm = tfm.overwrite_with_blur(13, 2)
        return _add_noise(tfm).crop
    elif i == STEP_EDGE_0_3:
        tfm = tfm.step_down(0.3)
        return _add_noise(tfm).crop
    elif i == STEP_EDGE_0_7:
        tfm = tfm.step_down(0.7)
        return _add_noise(tfm).crop
    elif i == STEP_GAUSE_0_3:
        tfm = tfm.step_down_gaussian(0.3)
        return _add_noise(tfm).crop
    else:
        raise Exception(f"invalid i={i}")

    
class BlurPadGaussianDataset(BlurPadDataset):
    """similar to blurpad but instead of overwriting with noise, we change the intensity of pixels around bbox, along with adding some noise

    The name is misleading, it should be StepEdgeDataset
    The dataset already assumes that you have blurred images in the image_paths. (the blur is assumed to be outside the bbox of interest)
    The mask is also assumed to have a bbox which is already padded with bbox_pad
    This class would take that mask, and add noise to the image outside the bbox of interest
    It would also decrease the intensity of pixels outside using mask_to_step_edge

    get_step_edge_mask creates a mask with 1 inside the box, 0.8 outside
    we just multiply it to the image to change the intensity
    """

    def __getitem__(self, index):
        # similar to the original
        # but we first mutate the mask to get a grayscale step edge mask
        # the mask would have 1 where the bbox is (as before)
        # it would have 0.3 elsewhere
        # we multiply this with the original image to decrease the intensity of pixels outside bbox

        tfms = self._valid_tfms if self.is_valid else self._train_tfms
        image, mask, bbox, inner_bbox = self.get_loaded_objects(index)
        crop, cropped_mask = _get_crops(image, mask, bbox)
        crop = self.crop_mutator(crop, cropped_mask, inner_bbox)
        step_edge_mask = get_step_edge_mask(cropped_mask.shape, inner_bbox)
        t_crop, t_mask = _with_tfms(crop, step_edge_mask, tfms)
        combined = t_crop * t_mask
        label_tensor = get_label_tensor(self._labels[index], self.LABEL_BY_IDX)
        return combined, label_tensor


class BlurPad4ChanDataset(BlurPadDataset):
    """like blurpad but add mask as the fourth channel

    I tried hoping the model would make a correlation, but convolutions seem to suck at making these correlations across channels.
    """

    def __init__(
        self,
        image_paths,
        mask_dir,
        crop_size,
        is_valid,
        crop_mutator=_id_crop_mutator,
        gaussian_mask_sigma_ratio_to_box_len=1.0,
        gaussian_mask_min_value=0.3,
    ):
        super().__init__(image_paths, mask_dir, crop_size, is_valid, crop_mutator)
        self.gaussian_mask_sigma_ratio_to_box_len = gaussian_mask_sigma_ratio_to_box_len
        self.gaussian_mask_min_value = gaussian_mask_min_value

    def __getitem__(self, index):
        tfms = self._valid_tfms if self.is_valid else self._train_tfms
        image_path = self.image_paths[index]
        image, mask = (
            DiskImage.load(image_path),
            DiskBooleanMask.load(self.mask_paths[index]),
        )
        bbox, inner_bbox = self.img_name_by_bbox[image_path.stem]
        crop, cropped_mask = _get_crops(image, mask, bbox)
        cropped_mask = get_gaussian_mask(
            cropped_mask.shape[:2],
            inner_bbox,
            self.gaussian_mask_sigma_ratio_to_box_len,
            self.gaussian_mask_min_value,
        )
        crop = self.crop_mutator(crop, cropped_mask, inner_bbox)
        t_crop, t_mask = _with_tfms(crop, cropped_mask, tfms)
        t_mask = t_mask.to(torch.float32)
        combined = torch.cat([t_crop, t_mask])
        # print("dtypes", t_crop.dtype, t_mask.dtype, combined.dtype)
        label_tensor = get_label_tensor(self._labels[index], self.LABEL_BY_IDX)
        return combined, label_tensor


def get_step_edge_mask(shape, bbox: Bbox, ratio=0.3):
    # instead of gaussian
    # we just put 1s in mask box
    # 1/3rd everyone else abruptly
    # Get bounding box limits
    y_min, x_min = bbox.y, bbox.x
    y_max, x_max = bbox.y2, bbox.x2

    gray = np.ones(shape, dtype=np.float32) * ratio
    gray[y_min:y_max, x_min:x_max] = 1.0
    return gray


def get_gaussian_mask(shape, bbox: Bbox, sigma_ratio_to_box_len=1.0, min_value=0.3):
    # 1. Get bounding box limits
    y_min, x_min = bbox.y, bbox.x
    y_max, x_max = bbox.y2, bbox.x2

    # 2. Calculate Center (x0, y0)
    y0, x0 = (y_min + y_max) / 2, (x_min + x_max) / 2

    # 3. Calculate Sigma
    # Note: If you want 1 sigma to be the full width,
    # sigma = (max - min). If you want 1 sigma to be half-width,
    # use (max - min) / 2.
    ratio = sigma_ratio_to_box_len
    sigma_y = ratio * (y_max - y_min)
    sigma_x = ratio * (x_max - x_min)

    sigma_x = max(sigma_x, 1e-6)
    sigma_y = max(sigma_y, 1e-6)

    # 4. Create the Coordinate Grid
    h, w = shape
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
    step_edge_mask = get_step_edge_mask(cropped_mask.shape, inner_bbox)
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

    def overwrite_with_blur(
        self, blur_kernel_sz=None, blur_sigma=None
    ) -> "CropTfmsOutsideBbox":
        if blur_kernel_sz is None or blur_sigma is None:
            return self

        crop, inner_bbox = self.crop, self.inner_bbox
        blurred = cv2.GaussianBlur(crop, (blur_kernel_sz, blur_kernel_sz), blur_sigma)
        y, y2, x, x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
        blurred[y:y2, x:x2] = crop[y:y2, x:x2]
        return CropTfmsOutsideBbox(blurred, self.inner_bbox)

    def overwrite_with_noise(self, max_noise=None) -> "CropTfmsOutsideBbox":
        if max_noise is None:
            return self

        crop, inner_bbox = self.crop, self.inner_bbox
        noisy = np.random.randint(0, max_noise, crop.shape, dtype=np.uint8)
        y, y2, x, x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
        noisy[y:y2, x:x2] = crop[y:y2, x:x2]
        return CropTfmsOutsideBbox(noisy, self.inner_bbox)

    def overwrite_with_zeros(self) -> "CropTfmsOutsideBbox":
        crop, inner_bbox = self.crop, self.inner_bbox
        dest = np.zeros(crop.shape, dtype=np.uint8)
        y, y2, x, x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
        dest[y:y2, x:x2] = crop[y:y2, x:x2]
        return CropTfmsOutsideBbox(dest, self.inner_bbox)

    def add_noise(self, max_noise) -> "CropTfmsOutsideBbox":
        if max_noise is None:
            return self

        crop, inner_bbox = self.crop, self.inner_bbox
        noisy = np.random.randint(0, max_noise, crop.shape, dtype=np.uint8)
        y, y2, x, x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
        noisy[y:y2, x:x2] = 0
        crop = crop + noisy
        return CropTfmsOutsideBbox(crop, self.inner_bbox)

    def step_down(self, ratio) -> "CropTfmsOutsideBbox":
        mask = get_step_edge_mask(self.crop.shape[:2], self.inner_bbox, ratio)
        crop = self.crop.astype(np.float32) * mask[..., None]
        return CropTfmsOutsideBbox(crop.astype(np.uint8), self.inner_bbox)

    def step_down_gaussian(self, min_val) -> "CropTfmsOutsideBbox":
        mask = get_gaussian_mask(self.crop.shape[:2], self.inner_bbox)
        crop = self.crop.astype(np.float32) * mask[..., None]
        return CropTfmsOutsideBbox(crop.astype(np.uint8), self.inner_bbox)


def get_image_mask_and_bbox(image_paths, mask_dir, crop_size, bbox_pad, min_area, min_bbox_length, max_area):
    image_paths = list(image_paths)
    mask_paths = [mask_dir / f"{i.stem}.png" for i in image_paths]
    img_name_by_bbox = get_coords_for_set(mask_paths, crop_size, bbox_pad, min_area, min_bbox_length, max_area)
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


def get_coords_for_set(mask_paths, crop_size, bbox_pad, min_area, min_bbox_length, max_area) -> dict[str, Bbox]:
    res = {}
    for m in tqdm(mask_paths, desc="Getting Crop Coords"):
        bbox, inner_bbox = get_center_crop_coords_from_mask(
            DiskBooleanMask.load(m), crop_size, bbox_pad, min_area, min_bbox_length, max_area
        )
        if bbox is None:
            print(f"SKIP: empty mask or too less area {m}")
            continue
        res[m.stem] = (bbox, inner_bbox)
    return res


# redefined region crops, its okay for now
def get_region_crops(mask) -> Iterator[Bbox]:
    _, labels = cv2.connectedComponents(mask)
    h, w = mask.shape
    for label in range(1, labels.max() + 1):
        rows, cols = np.where(labels == label)
        r1 = max(0, rows.min())
        r2 = min(h, rows.max())
        c1 = max(0, cols.min())
        c2 = min(w, cols.max())
        yield Bbox(c1, r1, c2 - c1, r2 - r1)


def get_max_area_bbox(bboxes: list[Bbox]):
    if not bboxes:
        raise Exception("bboxes cannot be empty for get_max_area_bbox")
    return max([(bb.h * bb.w, bb) for bb in bboxes])[1]

def get_center_crop_coords_from_mask(mask, crop_size, bbox_pad, min_area, min_bbox_length, max_area):
    bboxes = list(get_region_crops(mask))
    bboxes = [padded_bbox(bbox, bbox_pad, mask.shape) for bbox in bboxes]
    if not bboxes:
        return None, None

    bbox = max([(bb.h * bb.w, bb) for bb in bboxes])[1]
    if bbox.area() < min_area:
        print("bbox area too less", bbox.area(), min_area)
        return None, None
    if max_area is not None and bbox.area() > max_area:
        print("bbox area too high", bbox.area(), max_area)
        return None, None
        return None, None
    if bbox.h < min_bbox_length or bbox.w < min_bbox_length:
        return None, None
    return make_box_of_crop_size_centered_at_box(mask.shape, bbox, crop_size)


def make_box_of_crop_size_centered_at_box(image_shape, bbox: Bbox, crop_size):
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
