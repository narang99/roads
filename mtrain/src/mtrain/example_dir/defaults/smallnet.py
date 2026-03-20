from fastai.vision.all import load_learner
from pathlib import Path
from ..learners import (
    SmallnetLearner,
    get_raw_smallnet_learner,
)


def default_smallnet_learners(models_dir, labels, bs) -> dict[str, SmallnetLearner]:
    models_dir = Path(models_dir)
    data_dir = models_dir / "dummy_smallnet_data"

    res = {}
    if "v1" in labels:
        res["v1"] = get_smallnet_v1(models_dir, bs)
    if "md" in labels:
        res["md"] = get_smallnet_md(models_dir, bs, data_dir)
    if "sm" in labels:
        res["sm"] = get_smallnet_md(models_dir, bs, data_dir)
    return res


def get_smallnet_v1(models_dir, bs):
    path = models_dir / "smallnet-100x100" / "export_iter_14.pkl"
    learner = load_learner(path)
    learner.eval()
    return SmallnetLearner("v1", learner, bs, 100, [50], 2)


def get_smallnet_md(models_dir, bs, data_dir):
    path = models_dir / "smallnet-128x128" / "xresnet18-iter17.pth"
    learner = get_raw_smallnet_learner(128, bs, data_dir, path, "xresnet18")
    return SmallnetLearner("md", learner, bs, 128, [64], 35)


def get_smallnet_sm(models_dir, bs, data_dir):
    path = models_dir / "smallnet-64x64" / "raw_torch_iter80.pth"
    learner = get_raw_smallnet_learner(64, bs, data_dir, path, "xresnet18")
    return SmallnetLearner("sm", learner, bs, 64, [32], 2)