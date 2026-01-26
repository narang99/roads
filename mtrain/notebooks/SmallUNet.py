# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: mtrain
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import os

# %%
PROJECT_CODE = os.environ["PROJECT_CODE"]
FILE_SIZE = int(os.environ["FILE_SIZE"])
FINE_TUNE_EPOCHS = int(os.environ["FINE_TUNE_EPOCHS"])
NUM_SAMPLES = int(os.environ["NUM_SAMPLES"])
FIT_ONE_CYCLE_EPOCHS = int(os.environ["FIT_ONE_CYCLE_EPOCHS"])
MODEL = os.environ.get("MODEL", "mobilenet_v3_small")
LOSS = os.environ.get("LOSS", None)

params = {
    "PROJECT_CODE": PROJECT_CODE,
    "FILE_SIZE": FILE_SIZE,
    "FINE_TUNE_EPOCHS": FINE_TUNE_EPOCHS,
}


TFMS = {
    "do_flip": True,
    "flip_vert": True,
    "max_rotate": 360,
    "max_zoom": 1.4,
    "max_lighting": 0.4,
    "max_warp": 0.2,
}

DS = Path("../../datasets/")
T004_DIR = DS / "T004-taco-crops"
T005_DIR = DS / "T005-with-taco-and-ddg"
T008_DIR = DS / "T008-unet"
EXP_BASE = T008_DIR / PROJECT_CODE
OUTS = EXP_BASE / "synth"
LOG_BASE = EXP_BASE / "log"
TACO_BASE_DIR = Path("/Users/hariomnarang/Desktop/personal/TACO/data/")
DDG_BASE_DIR = DS / "T006-ddg-garbage"
ANN_FILE = TACO_BASE_DIR / "annotations.json"
TEST_BIG_IMG = T004_DIR / "14325.jpeg"

LOG_BASE.mkdir(parents=True, exist_ok=True)
DS.exists(), TACO_BASE_DIR.exists(), ANN_FILE.exists()

# %% [markdown]
# # Generate Data

# %%
# generate data first

from mtrain.smallnet import extract_all_taco, dump_ddg_images

# %%
extract_all_taco(
    images_root=TACO_BASE_DIR,
    ann_file=ANN_FILE,
    out_dir=OUTS,
    box_size=FILE_SIZE,
    total_samples=NUM_SAMPLES,
    empty_samples_ratio=1,
)
dump_ddg_images(
    ddg_base_dir=DDG_BASE_DIR,
    out_pos_dir=OUTS / "pos",
    box_size=FILE_SIZE,
    max_samples=NUM_SAMPLES,
)

# %% [markdown]
# # Training

# %%
from fastai.vision.all import (
    DataBlock,
    CategoryBlock,
    SegmentationDataLoaders,
    ImageBlock,
    get_image_files,
    RandomSplitter,
    parent_label,
    aug_transforms,
    Resize,
)

dbl = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    batch_tfms=aug_transforms(**TFMS),
    item_tfms=Resize(FILE_SIZE),
)


dls = dbl.dataloaders(OUTS, path=LOG_BASE)
dls.valid.show_batch(max_n=4, nrows=1)

# %%
from fastai.vision.all import (
    vision_learner,
    unet_learner,
    resnet18,
    mobilenet_v3_small,
    mobilenet_v3_large,
    CrossEntropyLossFlat,
)
from fastai.callback.progress import CSVLogger, Recorder, ProgressCallback

# %%
# ModelCls = mobilenet_v3_small if MODEL == "mobilenet_v3_small" else mobilenet_v3_large
NAME_BY_MODEL_CLS = {
    "mobilenet_v3_small": mobilenet_v3_small,
    "mobilenet_v3_large": mobilenet_v3_large,
    "resnet18": resnet18,
}
NAME_BY_LOSS_FN = {
    "CrossEntropyLossFlat": CrossEntropyLossFlat,
}
# ModelCls = mobilenet_v3_small if MODEL == "mobilenet_v3_small" else mobilenet_v3_large
LossFn = NAME_BY_LOSS_FN.get("LOSS")
if LOSS == "CrossEntropyLossFlat":
    kw = {"loss_func": LossFn()}
else:
    kw = {}

ModelCls = NAME_BY_MODEL_CLS[MODEL]
learner = unet_learner(dls, ModelCls, normalize=True, pretrained=True, **kw)
# learner = vision_learner(dls, ModelCls, normalize=True, pretrained=True, **kw)
learner = learner.remove_cbs([ProgressCallback])
learner = learner.add_cb(CSVLogger)

# %%
learner.fine_tune(FINE_TUNE_EPOCHS)

# %%
learner.unfreeze()
learner.fit_one_cycle(FIT_ONE_CYCLE_EPOCHS, lr_max=slice(1e-6, 1e-4))

# %%
learner.remove_cb(CSVLogger)
learner.export()

# %%
from mtrain.smallnet.predict import tile_image_and_predict

res = tile_image_and_predict(TEST_BIG_IMG, learner, FILE_SIZE, add_labels=False)
plt.imsave(EXP_BASE / "res.png", res)
