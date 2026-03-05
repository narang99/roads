from mtrain.neg_mask.model.dataset import MaskClassificationDataset
from pathlib import Path

LABELS = ["trash", "other"]


def load_our_learner(dls, model_arch, weights, pth_path=None):
    from fastai.vision.all import (
        vision_learner,
        CrossEntropyLossFlat,
        accuracy,
        F1Score,
        ProgressCallback,
    )

    learn = vision_learner(
        dls,
        model_arch,
        n_in=4,
        metrics=[accuracy, F1Score(average="macro")],
        loss_func=CrossEntropyLossFlat(weight=weights),
        n_out=len(LABELS),
        normalize=False,
    )
    learn = learn.remove_cb(ProgressCallback)
    if pth_path is not None:
        pth_path = Path(pth_path)
        if pth_path.suffix == ".pth":
            print("WARN: the path you should pass should not have suffix .pth")
        learn = learn.load(pth_path)
    return learn


def dummy_dls(device=None):
    from fastai.vision.all import DataLoaders, default_device

    if device is None:
        device = default_device()
    return DataLoaders.from_dsets(
        MaskClassificationDataset([], LABELS, True),
        MaskClassificationDataset([], LABELS, False),
    )
