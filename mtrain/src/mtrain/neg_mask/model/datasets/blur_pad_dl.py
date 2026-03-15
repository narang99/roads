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


class BlurPadDataset(Dataset):
    LABEL_BY_IDX = {
        "other": 0,
        "trash": 1,
    }

    def __init__(
        self, image_paths, mask_dir, crop_size, is_valid, max_noise=None
    ):
        self.crop_size = crop_size
        self.image_paths, self.mask_paths, self.img_name_by_bbox = get_image_mask_and_bbox(
            image_paths, mask_dir, crop_size
        )
        self._labels = [label_func(i.name) for i in self.image_paths]
        self.is_valid = is_valid

        self._train_tfms = v2.Compose(
            [
                v2.Resize(size=self.crop_size - 1, max_size=self.crop_size),
                v2.CenterCrop(self.crop_size),
                v2.RandomHorizontalFlip(p=0.5),
                # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # v2.RandomRotation([0,10]),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=MASK_DS_IMAGENET_MEAN, std=MASK_DS_IMAGENET_STD),
                v2.ToPureTensor(),
            ]
        )
        self._valid_tfms = v2.Compose(
            [
                v2.Resize(size=self.crop_size - 1, max_size=self.crop_size),
                v2.CenterCrop(self.crop_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=MASK_DS_IMAGENET_MEAN, std=MASK_DS_IMAGENET_STD),
                v2.ToPureTensor(),
            ]
        )
        self.max_noise = max_noise

    @classmethod
    def label_func(cls, image_path):
        return label_func(Path(image_path).name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = DiskImage.load(image_path)
        full_mask = DiskBooleanMask.load(self.mask_paths[index])

        bbox, inner_bbox = self.img_name_by_bbox[image_path.stem]

        crop = image[bbox.y : bbox.y2, bbox.x : bbox.x2]
        cropped_mask = full_mask[bbox.y:bbox.y2, bbox.x:bbox.x2]

        if self.max_noise is not None:
            noisy = np.random.randint(
                0, self.max_noise, crop.shape, dtype=np.uint8
            )
            y,y2,x,x2 = inner_bbox.y, inner_bbox.y2, inner_bbox.x, inner_bbox.x2
            noisy[y:y2, x:x2] = crop[y:y2, x:x2]
            crop = noisy

        crop = tv_tensors.Image(crop).permute([2,0,1])
        cropped_mask = tv_tensors.Mask(cropped_mask.reshape(1, *cropped_mask.shape))

        label = self._labels[index]
        label_idx = self.LABEL_BY_IDX[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        tfms = self._valid_tfms if self.is_valid else self._train_tfms

        t_crop, t_mask = tfms((crop, cropped_mask))
    
        return t_crop, label_tensor


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
        bbox, inner_bbox = get_center_crop_coords_from_mask(DiskBooleanMask.load(m), crop_size)
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
    center_x, center_y = bbox.x + (bbox.w // 2), bbox.y + (bbox.h // 2)

    left_pad = crop_size // 2

    left_x = max(center_x - left_pad, 0)
    up_y = max(center_y - left_pad, 0)

    right_x = min(left_x + crop_size, mask.shape[1])
    down_y = min(up_y + crop_size, mask.shape[0])

    inner_bbox = Bbox(bbox.x - left_x, bbox.y - up_y, bbox.w, bbox.h)

    return Bbox(left_x, up_y, right_x - left_x, down_y - up_y), inner_bbox


def label_func(x):
    if x.startswith("other"):
        return "other"
    elif x.startswith("trash"):
        return "trash"
    else:
        raise Exception(f"bad file name {x}")
