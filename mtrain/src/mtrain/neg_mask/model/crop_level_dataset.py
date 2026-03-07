from typing import Optional, Callable
import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import v2
from .crop_dataset_base import (
    create_default_source_augmentations,
    get_pairs_from_directory, run_tfms_on_pairs,
    get_label, validate_labels,
    denormalize, create_std_transforms
)
import albumentations as A


class CropLevelDataset(torch.utils.data.Dataset):
    """
    Concrete dataset for crop-level classification using leveled cropping.

    Creates 12-channel input from 3 scales: tight crop, medium crop, full image.
    Each sample directory should contain: image.jpg, mask.png, meta.json, source_dir symlink.

    Supports source-level augmentation (before cropping) using albumentations.
    """

    def __init__(
        self,
        dirs: list[Path | str],
        labels: list[str],  # ["other", "trash"]
        train: bool = False,
        tight_pad: int = 20,
        medium_pad: int = 130,
        full_image_size: int = 1024,
        medium_center_prob: float = 1.0,
        custom_transforms: Optional[v2.Compose] = None,
        source_augmentations: Optional[A.Compose] = None,
        use_default_source_augs: bool = True,
    ):
        self.dirs = [Path(d) for d in dirs]
        self.labels = labels
        self.label_by_idx = {label: i for i, label in enumerate(self.labels)}
        self.tight_pad = tight_pad
        self.medium_pad = medium_pad
        self.full_image_size = full_image_size
        self.medium_center_prob = medium_center_prob
        self.train = train

        # Setup source augmentations (applied before cropping)
        if source_augmentations is not None:
            self.source_augmentations = source_augmentations
        elif train and use_default_source_augs:
            self.source_augmentations = create_default_source_augmentations(train=True)
        else:
            self.source_augmentations = None

        # Define transforms
        if custom_transforms is not None:
            self.tfms = custom_transforms
        else:
            self.tfms = create_std_transforms()

        validate_labels(self.dirs, labels)

    def load_crop_pairs(self, directory: Path) -> list[tuple[np.ndarray, np.ndarray]]:
        """Load the 3-level crop pairs using leveled cropping with optional source augmentation."""
        return get_pairs_from_directory(
            directory,
            self.source_augmentations,
            self.full_image_size,
            self.tight_pad,
            self.medium_pad,
            self.medium_center_prob,
        )

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: 12-channel tensor [tight_rgb, medium_rgb, full_rgb, tight_mask, medium_mask, full_mask]
            y: Label tensor (class index)
        """
        directory = self.dirs[index]

        # Load 3-level crop pairs
        pairs = self.load_crop_pairs(directory)
        t_images, t_masks = run_tfms_on_pairs(pairs, self.tfms, index, self.dirs[index])

        # Stack: [img0, img1, img2, mask0, mask1, mask2] = 12 channels
        combined = torch.cat(t_images + t_masks, dim=0)

        # Get label
        label = get_label(directory)
        label_idx = self.label_by_idx[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return combined, label_tensor

    @classmethod
    def denormalize(
        cls, combined: torch.Tensor
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return denormalize(combined, 3)

    @property
    def num_classes(self) -> int:
        """Number of classification classes."""
        return len(self.labels)

    @property
    def num_channels(self) -> int:
        """Number of input channels (always 12 for crop level)."""
        return 12




class CropLevelDataset2Chan(CropLevelDataset):
    def load_crop_pairs(self, directory: Path) -> list[tuple[np.ndarray, np.ndarray]]:
        """Load the 3-level crop pairs using leveled cropping with optional source augmentation."""
        all_pairs = get_pairs_from_directory(
            directory,
            self.source_augmentations,
            self.full_image_size,
            self.tight_pad,
            self.medium_pad,
            self.medium_center_prob,
        )
        return all_pairs[:2]

    @property
    def num_channels(self) -> int:
        """Number of input channels (always 12 for crop level)."""
        return 8

    @classmethod
    def denormalize(
        cls, combined: torch.Tensor
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return denormalize(combined, 2)