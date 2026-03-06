from typing import Optional, Callable
import torch
import numpy as np
from pathlib import Path
from torchvision import tv_tensors
from torchvision.transforms import v2
from mtrain.neg_mask.leveled_cropping import load_crop_level_sample_from_directory, make_crop_level_pairs_v2
import albumentations as A


# Same constants as existing datasets
MASK_DS_IMAGENET_MEAN = [0.485, 0.456, 0.406]
MASK_DS_IMAGENET_STD = [0.229, 0.224, 0.225]

# Base evaluation transforms
CROP_LEVEL_EVAL_TFMS = v2.Compose([
    v2.Resize(130, antialias=True),
    v2.CenterCrop(130),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MASK_DS_IMAGENET_MEAN, std=MASK_DS_IMAGENET_STD),
    v2.ToPureTensor(),
])

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


def create_default_source_augmentations(train: bool = True) -> Optional[A.Compose]:
    """Create default albumentations for source image/mask augmentation."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),  # Good for aerial/satellite data
        A.RandomRotate90(p=0.5),  # 90° rotations preserve spatial relationships
        A.ColorJitter(
            brightness=0.1, 
            contrast=0.1, 
            saturation=0.1, 
            hue=0.05, 
            p=0.7
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.1, 
            p=0.5
        ),
    ], additional_targets={'mask': 'mask'})


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
        elif train:
            self.tfms = v2.Compose([
                ResizeIfLarger(129,130),
                v2.CenterCrop(130),
                # Add augmentations here when ready
                # v2.RandomHorizontalFlip(p=0.5),
                # v2.ColorJitter(brightness=0.1, contrast=0.1),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=MASK_DS_IMAGENET_MEAN, std=MASK_DS_IMAGENET_STD),
                v2.ToPureTensor(),
            ])
        else:
            self.tfms = CROP_LEVEL_EVAL_TFMS
            
        self._validate_labels()

    def _validate_labels(self):
        """Ensure all directories have valid labels."""
        for d in self.dirs:
            label = self._get_label(d)
            if label not in self.label_by_idx:
                raise ValueError(f"Label '{label}' not found for {d}. Expected: {self.labels}")

    def _get_label(self, directory: Path) -> str:
        """Extract label from directory structure (parent folder name)."""
        return directory.parent.name

    def _load_crop_pairs(self, directory: Path) -> list[tuple[np.ndarray, np.ndarray]]:
        """Load the 3-level crop pairs using leveled cropping with optional source augmentation."""
        sample = load_crop_level_sample_from_directory(directory, self.full_image_size)
        
        # Apply source-level augmentation BEFORE cropping
        if self.source_augmentations is not None:
            augmented = self.source_augmentations(
                image=sample.full_image, 
                mask=sample.full_mask.astype(np.uint8)
            )
            sample.full_image = augmented['image']
            sample.full_mask = augmented['mask']
            
            # RECALCULATE bbox after augmentation since mask pixels may have moved
            from mtrain.neg_mask.crops import Bbox
            import cv2
            _, labels, stats, _ = cv2.connectedComponentsWithStats(sample.full_mask.astype(np.uint8))
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
            sample, self.tight_pad, self.medium_pad, self.medium_center_prob
        )
        return crop_pair.pairs

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
        pairs = self._load_crop_pairs(directory)
        assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)} from {directory}"
        
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
                t_img, t_mask = self.tfms((img_tensor, mask_tensor))
            except:
                print(f"failure in: index={index} dir={self.dirs[index]} input_tensor_shapes: {img_tensor.shape} | {mask_tensor.shape}")
                raise
            t_images.append(t_img)
            t_masks.append(t_mask)
        
        # Stack: [img0, img1, img2, mask0, mask1, mask2] = 12 channels
        combined = torch.cat(t_images + t_masks, dim=0)
        
        # Get label
        label = self._get_label(directory)
        label_idx = self.label_by_idx[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        return combined, label_tensor

    @classmethod
    def denormalize(
        cls, combined: torch.Tensor, pairs_per_dir: int = 3
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Denormalize 12-channel tensor back to list of (image, mask) pairs.
        
        Args:
            combined: 12-channel tensor [img0, img1, img2, mask0, mask1, mask2]
            pairs_per_dir: Number of pairs (always 3 for crop level)
            
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

    @property
    def num_classes(self) -> int:
        """Number of classification classes."""
        return len(self.labels)

    @property
    def num_channels(self) -> int:
        """Number of input channels (always 12 for crop level)."""
        return 12


def create_crop_level_dataloaders(
    train_dirs: list[Path],
    valid_dirs: list[Path], 
    labels: list[str],
    batch_size: int = 32,
    num_workers: int = 0,
    source_augmentations: Optional[A.Compose] = None,
    use_default_source_augs: bool = True,
    **dataset_kwargs
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Convenience function to create train/valid dataloaders with source augmentation.
    
    Args:
        train_dirs: Training sample directories
        valid_dirs: Validation sample directories
        labels: Class labels (e.g., ["other", "trash"])
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        source_augmentations: Custom albumentations for source image/mask
        use_default_source_augs: Whether to use default augmentations if none provided
        **dataset_kwargs: Additional arguments for CropLevelDataset
        
    Returns:
        (train_dataloader, valid_dataloader)
    """
    train_ds = CropLevelDataset(
        train_dirs, labels, train=True,
        source_augmentations=source_augmentations,
        use_default_source_augs=use_default_source_augs,
        **dataset_kwargs
    )
    valid_ds = CropLevelDataset(
        valid_dirs, labels, train=False,
        source_augmentations=None,  # No augmentation for validation
        use_default_source_augs=False,
        **dataset_kwargs
    )
    
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_dl, valid_dl


# Example usage with custom augmentations
def create_strong_source_augmentations() -> A.Compose:
    """Create stronger augmentations for aggressive regularization."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=10, p=0.3),  # Small rotations
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.GaussNoise(std_range=(3.2, 7.1), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
    ], additional_targets={'mask': 'mask'})