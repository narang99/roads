from typing import Callable
from mtrain.utils import mkdir
from tqdm import tqdm
import json
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors
from pathlib import Path
import cv2
from mtrain.neg_mask.crops import Bbox, padded_crop, bbox_only_mask


MASK_DS_IMAGENET_MEAN = [0.485, 0.456, 0.406]
MASK_DS_IMAGENET_STD = [0.229, 0.224, 0.225]
MASK_DS_EVAL_TFMS = v2.Compose(
    [
        v2.Resize(130, antialias=True),
        v2.CenterCrop(130),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MASK_DS_IMAGENET_MEAN, std=MASK_DS_IMAGENET_STD),
        v2.ToPureTensor(),
    ]
)

class ResizeIfLarger:
    def __init__(self, size: int):
        self.size = size
        self.resize = v2.Resize(size, antialias=True)

    def __call__(self, img_and_mask):
        img, mask = img_and_mask
        h, w = img.shape[-2], img.shape[-1]
        if h > self.size or w > self.size:
            img = self.resize(img)
        h, w = mask.shape[-2], mask.shape[-1]
        if h > self.size or w > self.size:
            mask = self.resize(mask)
        return img, mask


class GenericMaskClassificationDataset(torch.utils.data.Dataset):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        dirs: list[Path | str],
        labels: list[
            str
        ],  # make sure these are always in the same order, new labels should be appended
        train: bool,
        pairs_per_dir: int,
        get_pairs: Callable[[Path], list[tuple[np.ndarray, np.ndarray]]],
    ):
        self.dirs = [d for d in map(Path, dirs)]
        self.labels = labels
        self.label_by_idx = {label: i for i, label in enumerate(self.labels)}
        self.pairs_per_dir = pairs_per_dir
        self.get_pairs = get_pairs
        train_tfms = v2.Compose(
            [
                ResizeIfLarger(200),
                #   v2.RandomCrop(130),  # Instead of CenterCrop
                #   v2.RandomHorizontalFlip(p=0.5),
                #   v2.RandomVerticalFlip(p=0.3),  # Good for aerial/satellite data
                #   v2.RandomRotation(degrees=15),  # Moderate rotation
                #   v2.RandomPerspective(distortion_scale=0.1, p=0.3),
                #   v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                # v2.Resize(200, antialias=True),
                # v2.RandomCrop(130),
                # v2.RandomHorizontalFlip(p=0.5),
                v2.CenterCrop(130),
                # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=MASK_DS_IMAGENET_MEAN, std=MASK_DS_IMAGENET_STD),
                v2.ToPureTensor(),
            ]
        )
        self.tfms = train_tfms if train else MASK_DS_EVAL_TFMS
        _validate_labels_in_dirs(self.dirs, self.label_by_idx)

    @property
    def num_in_channels(self):
        return self.pairs_per_dir * 4

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        d = self.dirs[index]

        pairs = self.get_pairs(d)
        pairs = [
            (
                tv_tensors.Image(torch.from_numpy(image).permute(2, 0, 1)),
                tv_tensors.Mask(torch.from_numpy(mask.reshape(1, *mask.shape))),
            )
            for image, mask in pairs
        ]
        t_images, t_masks= [], []
        for p in pairs:
            t_img, t_mask = self.tfms(p)
            t_images.append(t_img)
            t_masks.append(t_mask)
        t_res = []
        t_res.extend(t_images)
        t_res.extend(t_masks)

        combined = torch.cat(t_res, dim=0)

        label = _label_func(d)
        label_idx = self.label_by_idx[label]
        label_tensor = torch.Tensor([label_idx]).squeeze()

        return (combined, label_tensor)

    @classmethod
    def denormalize(
        cls, combined: torch.Tensor, pairs_per_dir: int = 1
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # Layout: [rgb0, rgb1, ..., mask0, mask1, ...]
        # all images first (3 channels each), then all masks (1 channel each)
        mean = torch.tensor(MASK_DS_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(MASK_DS_IMAGENET_STD).view(3, 1, 1)
        result = []
        for i in range(pairs_per_dir):
            rgb = (combined[i * 3 : i * 3 + 3] * std + mean).clamp(0, 1)
            img_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            result.append((img_np, combined[pairs_per_dir * 3 + i].numpy()))
        return result



class MaskClassificationDataset(torch.utils.data.Dataset):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        dirs: list[Path | str],
        labels: list[
            str
        ],  # make sure these are always in the same order, new labels should be appended
        train: bool,
    ):
        self.dirs = [d for d in map(Path, dirs) if _is_valid_dir(d)]
        self.labels = labels
        self.label_by_idx = {label: i for i, label in enumerate(self.labels)}
        train_tfms = v2.Compose(
            [
                v2.Resize(200, antialias=True),
                v2.RandomCrop(130),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=MASK_DS_IMAGENET_MEAN, std=MASK_DS_IMAGENET_STD),
                v2.ToPureTensor(),
            ]
        )
        self.tfms = train_tfms if train else MASK_DS_EVAL_TFMS
        _validate_labels_in_dirs(self.dirs, self.label_by_idx)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        d = self.dirs[index]

        # load individual as tensors
        img = tv_tensors.Image(Image.open(d / "image.jpg").convert("RGB"))
        mask = tv_tensors.Mask(Image.open(d / "mask.png").convert("L"))

        # tfms then combine
        t_img, t_mask = self.tfms([img, mask])
        combined = torch.cat([t_img, t_mask], dim=0)

        # label tensor — if mask is empty after transforms, force label to 0
        area = t_mask.sum()
        if area < 5:
            label_idx = 0
        else:
            label = _label_func(d)
            label_idx = self.label_by_idx[label]
        label_tensor = torch.Tensor([label_idx]).squeeze()

        return (combined, label_tensor)

    @classmethod
    def denormalize(cls, combined: torch.Tensor):
        """Return (img_np uint8 HxWx3, mask_np uint8 HxW) from a 4-channel tensor."""
        mean = torch.tensor(MASK_DS_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(MASK_DS_IMAGENET_STD).view(3, 1, 1)
        rgb = (combined[:3] * std + mean).clamp(0, 1)
        img_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return img_np, combined[3].numpy()


class MaskInferenceDataset(torch.utils.data.Dataset):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, img_mask_pairs):
        self.pairs = img_mask_pairs
        self.tfms = MASK_DS_EVAL_TFMS

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img, mask = self.pairs[index]
        img = tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1))
        mask = tv_tensors.Mask(torch.from_numpy(mask).unsqueeze(0))
        t_img, t_mask = self.tfms([img, mask])
        return torch.cat([t_img, t_mask], dim=0)


def make_crop_level_pairs(
    tight_pad: int = 20,
) -> Callable[[Path], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Factory returning a get_pairs callable for GenericMaskClassificationDataset.

    Each crop_level sample dir yields 3 (image, mask) pairs:
      0. Tight re-crop  — bbox of mask pixels in stored crop + tight_pad context
      1. Stored crop    — image.jpg + mask.png as saved by the labeling widget
    """

    def get_pairs(d: Path) -> list[tuple[np.ndarray, np.ndarray]]:
        img = np.array(Image.open(d / "image.jpg").convert("RGB"))
        mask = np.array(Image.open(d / "mask.png"))

        # Pair 0: tight re-crop around the single connected component
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if labels.max() > 0:
            x, y, w, h = (
                stats[1, cv2.CC_STAT_LEFT],
                stats[1, cv2.CC_STAT_TOP],
                stats[1, cv2.CC_STAT_WIDTH],
                stats[1, cv2.CC_STAT_HEIGHT],
            )
            bbox = Bbox(x=int(x), y=int(y), w=int(w), h=int(h))
            tight_img, _, _ = padded_crop(img, bbox, tight_pad)
            tight_mask = bbox_only_mask(mask, bbox, tight_pad)
        else:
            tight_img, tight_mask = img, mask

        # i dont need the full imager right now, take lite
        # later when doing hawker, i might add it

        # # Pair 2: full image from source_dir, object mask placed at crop_origin
        meta = json.loads((d / "meta.json").read_text())
        ox, oy = meta["crop_origin"]["x"], meta["crop_origin"]["y"]
        source_dir = (d / "source_dir").resolve()
        full_img = np.array(Image.open(source_dir / "image.jpg").convert("RGB"))
        fh, fw = full_img.shape[:2]
        full_mask = np.zeros((fh, fw), dtype=np.uint8)
        mh, mw = mask.shape[:2]
        y2, x2 = min(oy + mh, fh), min(ox + mw, fw)
        full_mask[oy:y2, ox:x2] = mask[:y2 - oy, :x2 - ox]

        # return [(tight_img, tight_mask), (img, mask), (full_img, full_mask)]
        return [(tight_img, tight_mask), (img, mask), (full_img, full_mask)]

    return get_pairs


def denormalize(combined: torch.Tensor):
    """Return (img_np uint8 HxWx3, mask_np uint8 HxW) from a 4-channel tensor."""
    mean = torch.tensor(MASK_DS_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(MASK_DS_IMAGENET_STD).view(3, 1, 1)
    rgb = (combined[:3] * std + mean).clamp(0, 1)
    img_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img_np, combined[3].numpy()


def _label_func(d: Path):
    return Path(d).parent.name


def _validate_labels_in_dirs(dirs, label_by_idx):
    for d in dirs:
        label = _label_func(d)
        if label not in label_by_idx:
            raise Exception(
                f"Label={label} not found for directory={d}. label_by_index={label_by_idx}"
            )


def _is_valid_dir(direc: Path):
    valid = (
        direc.is_dir()
        and (direc / "image.jpg").exists()
        and (direc / "mask.png").exists()
    )
    return valid


def persist_dataset_to_cache(ds, cache_dir):
    direc = mkdir(cache_dir)
    for i, (x, y) in tqdm(enumerate(ds)):
        torch.save((x, y), direc / f"{i}.pt")


class DirectTensorLoadFromCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, num_in_channels):
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        self.num_in_cannels = num_in_channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return torch.load(self.files[index])