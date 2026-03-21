import functools
from pathlib import Path
from fastai.basics import get_image_files
import numpy as np
from fastai.vision.all import (
    FocalLossFlat,
    DiceLoss,
    store_attr,
    unet_learner,
    SegmentationDataLoaders,
    Dice,
    ProgressCallback,
    aug_transforms,
    Resize,
    default_device,
)
import torch
from torch.nn import functional as F
from .common import get_arch


class CombinedLoss:
    "Dice and Focal combined"

    def __init__(
        self,
        axis=1,
        smooth=1.0,
        alpha=1.0,
    ):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):
        return x.argmax(dim=self.axis)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)


@functools.lru_cache(maxsize=5)
def get_smallnet_learner(
    tile_size, bs, data_dir, pth_path, arch="xresnet18", device=None
):
    data_dir = Path(data_dir)
    dls = SegmentationDataLoaders.from_label_func(
        ".",
        bs=bs,
        fnames=get_image_files(data_dir / "images"),
        label_func=lambda o: data_dir / "masks" / f"{o.stem}.png",
        codes=np.array(["background", "trash"]),
        item_tfms=Resize(tile_size),
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        batch_tfms=aug_transforms(),
    )

    arch = get_arch(arch)
    learner = unet_learner(
        dls,
        arch,
        n_out=2,
        pretrained=True,
        loss_func=CombinedLoss(),
        metrics=[Dice()],
    )
    learner = learner.remove_cb(ProgressCallback)
    if device is None:
        device = default_device()
    sd = torch.load(pth_path, device)
    learner.model.load_state_dict(sd)
    learner.model.eval()
    return learner
