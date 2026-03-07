from typing import Optional
import torch
import numpy as np
from pathlib import Path
from torchvision import tv_tensors
from torchvision.transforms import v2
from mtrain.neg_mask.leveled_cropping import (
    load_crop_level_sample_from_directory,
    make_crop_level_pairs_v2,
)
import albumentations as A


# Shared constants
MASK_DS_IMAGENET_MEAN = [0.485, 0.456, 0.406]
MASK_DS_IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizeIfLarger:
    """Conditionally resize only if image exceeds size threshold."""

    def __init__(self, size: int, max_size: int):
        self.size = size
        self.resize = v2.Resize(size, max_size=max_size, antialias=True)

    def __call__(self, img_and_mask):
        img, mask = img_and_mask
        h, w = img.shape[-2], img.shape[-1]
        if h > self.size or w > self.size:
            img = self.resize(img)
        h, w = mask.shape[-2], mask.shape[-1]
        if h > self.size or w > self.size:
            mask = self.resize(mask)
        return img, mask


def denormalize(
    combined: torch.Tensor, pairs_per_dir: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Denormalize multi-channel tensor back to list of (image, mask) pairs.

    Args:
        combined: Multi-channel tensor
        pairs_per_dir: Number of pairs

    Returns:
        List of (image_np, mask_np) tuples
    """
    mean = torch.tensor(MASK_DS_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(MASK_DS_IMAGENET_STD).view(3, 1, 1)

    result = []
    for i in range(pairs_per_dir):
        # Denormalize RGB channels
        rgb = (combined[i * 3 : i * 3 + 3] * std + mean).clamp(0, 1)
        img_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Get corresponding mask
        mask_np = combined[pairs_per_dir * 3 + i].numpy()

        result.append((img_np, mask_np))

    return result


def create_default_source_augmentations(train: bool = True) -> Optional[A.Compose]:
    """Create default albumentations for source image/mask augmentation."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),  # Good for aerial/satellite data
            A.RandomRotate90(p=0.5),  # 90° rotations preserve spatial relationships
            A.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.7
            ),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ],
        additional_targets={"mask": "mask"},
    )


def get_pairs_from_directory(
    directory,
    source_augmentations,
    full_image_size,
    tight_pad,
    medium_pad,
    medium_center_prob,
):
    sample = load_crop_level_sample_from_directory(directory, full_image_size)

    # Apply source-level augmentation BEFORE cropping
    if source_augmentations is not None:
        augmented = source_augmentations(
            image=sample.full_image, mask=sample.full_mask.astype(np.uint8)
        )
        sample.full_image = augmented["image"]
        sample.full_mask = augmented["mask"]

        # RECALCULATE bbox after augmentation since mask pixels may have moved
        from mtrain.neg_mask.crops import Bbox
        import cv2

        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            sample.full_mask.astype(np.uint8)
        )
        if labels.max() > 0:
            x, y, w, h = (
                stats[1, cv2.CC_STAT_LEFT],
                stats[1, cv2.CC_STAT_TOP],
                stats[1, cv2.CC_STAT_WIDTH],
                stats[1, cv2.CC_STAT_HEIGHT],
            )
            sample.bbox = Bbox(x=int(x), y=int(y), w=int(w), h=int(h))

    # Now create 3-level crops from potentially augmented source
    crop_pair = make_crop_level_pairs_v2(
        sample, tight_pad, medium_pad, medium_center_prob
    )
    return crop_pair.pairs


def validate_labels(dirs, allowed_labels):
    """Ensure all directories have valid labels."""
    for d in dirs:
        label = get_label(d)
        if label not in allowed_labels:
            raise ValueError(
                f"Label '{label}' not found for {d}. Expected: {allowed_labels}"
            )


def get_label(directory: Path) -> str:
    """Extract label from directory structure (parent folder name)."""
    return directory.parent.name


def run_tfms_on_pairs(pairs, tfms, index, direc_path):
    # Convert to tensors
    tensor_pairs = [
        (
            tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)),
            tv_tensors.Mask(torch.from_numpy(mask.reshape(1, *mask.shape))),
        )
        for img, mask in pairs
    ]

    # Apply transforms to each pair
    t_images, t_masks = [], []
    for img_tensor, mask_tensor in tensor_pairs:
        try:
            t_img, t_mask = tfms((img_tensor, mask_tensor))
        except:
            print(
                f"failure in: index={index} dir={direc_path} input_tensor_shapes: {img_tensor.shape} | {mask_tensor.shape}"
            )
            raise
        t_images.append(t_img)
        t_masks.append(t_mask)
    return t_images, t_masks


def create_strong_source_augmentations() -> A.Compose:
    """Create stronger augmentations for aggressive regularization."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=10, p=0.3),  # Small rotations
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.GaussNoise(std_range=(3.2, 7.1), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
        ],
        additional_targets={"mask": "mask"},
    )

def create_std_transforms() -> v2.Compose:
    """Create standard training transforms."""
    return v2.Compose(
        [
            ResizeIfLarger(129, 130),
            v2.CenterCrop(130),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MASK_DS_IMAGENET_MEAN, std=MASK_DS_IMAGENET_STD),
            v2.ToPureTensor(),
        ]
    )
