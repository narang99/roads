import functools
from mtrain.neg_mask.model.datasets.blur_pad_dl import BlurPadDataset
from torchvision.models import resnet18
from pathlib import Path
from fastai.basics import DataLoaders, Precision, CrossEntropyLossFlat
from fastai.vision.all import (
    xresnet18,
    ProgressCallback,
    default_device,
    vision_learner,
)
import torch
from .common import get_arch


@functools.lru_cache(maxsize=5)
def get_negmask_learner(bs, crop_size, pth_path, arch="xresnet18"):
    arch = get_arch(arch)
    dls = dummy_unblur_dls(bs, crop_size)
    learn = vision_learner(
        dls,
        arch,
        metrics=[Precision()],
        loss_func=CrossEntropyLossFlat(),
        n_out=2,
        normalize=False,
        n_in=3,
        pretrained=True,
    )
    learn = learn.remove_cb(ProgressCallback)
    state_dict = torch.load(pth_path, map_location=default_device())
    learn.model.load_state_dict(state_dict, strict=True)
    learn.model.eval()
    return learn


def dummy_unblur_dls(bs, crop_size):
    train_ds = BlurPadDataset([], Path("./masks"), crop_size, False)
    valid_ds = BlurPadDataset([], Path("./masks"), crop_size, True)
    dls = DataLoaders.from_dsets(
        train_ds,
        valid_ds,
        device=default_device(),
        num_workers=4,
        bs=bs,
        persistent_workers=True,
    )
    return dls


def get_arch(arch_str):
    arch = xresnet18 if arch_str == "xresnet18" else resnet18
    return arch
