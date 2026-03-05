import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors
from pathlib import Path


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

class MaskClassificationDataset(torch.utils.data.Dataset):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        dirs: list[Path | str],
        norm_stats: tuple[list[float], list[float]],
        labels: list[
            str
        ],  # make sure these are always in the same order, new labels should be appended
        train: bool,
    ):
        self.mean = torch.tensor(norm_stats[0])
        self.std = torch.tensor(norm_stats[1])
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

        # label tensor
        label = _label_func(d)
        label_idx = self.label_by_idx[label]
        label_tensor = torch.Tensor([label_idx]).squeeze()

        return (combined, label_tensor)

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

