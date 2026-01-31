from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import os
from mtrain.smallnet.unet.extract import generate_dataset
from fastai.vision.all import resnet18, mobilenet_v3_large, mobilenet_v3_small
from mtrain.smallnet.unet.train import get_dls
from fastai.vision.all import unet_learner, ProgressCallback, CSVLogger

## extra env, always fixed
TACO_DIR = Path(os.environ.get("TACO_DIR", "/content/data"))

## exp root dirs injected by runner
PROJECT_PERM_DIR = Path(os.environ["PROJECT_PERM_DIR"])
PROJECT_WORK_DIR = Path(os.environ["PROJECT_WORK_DIR"])

## exp params
DATA_COUNT = int(os.environ["DATA_COUNT"])
TILE_SIZE = int(os.environ["TILE_SIZE"])
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
FINE_TUNE_EPOCHS = int(os.environ["FINE_TUNE_EPOCHS"])
FIT_EPOCHS = int(os.environ.get("FIT_EPOCHS"))
MODEL = os.environ.get("MODEL", "resnet18")

# inferred
ANN_FILE = TACO_DIR / "annotations.json"
LOG_BASE = PROJECT_PERM_DIR / "log"
DATA_DIR = PROJECT_WORK_DIR / "data"

LOG_BASE.mkdir(exist_ok=True, parents=True)


######### synth gen

if DATA_DIR.exists():
    print(f"deleting data dir: {DATA_DIR}")
    shutil.rmtree(DATA_DIR)
generate_dataset(
    ann_file=ANN_FILE,
    taco_dir=TACO_DIR,
    output_path=DATA_DIR,
    tile_size=TILE_SIZE,
    num_samples=DATA_COUNT,
    workers=16,
)
print(DATA_DIR.ls())
print((DATA_DIR / "images").ls())
print((DATA_DIR / "masks").ls())


######### dls


dls = get_dls(BATCH_SIZE, LOG_BASE, TILE_SIZE, DATA_DIR / "images", DATA_DIR / "masks")
dls.train.show_batch(max_n=4, nrows=1, alpha=0.7)
show_batch_path = LOG_BASE / "show_batch.png"
plt.savefig(show_batch_path, bbox_inches="tight", dpi=200)
plt.close()

######### learner
model_by_cls = {
    "resnet18": resnet18,
    "mobilenet_v3_small": mobilenet_v3_small,
    "mobilenet_v3_large": mobilenet_v3_large,
}

learner = unet_learner(dls, model_by_cls[MODEL])
learner.remove_cb(ProgressCallback)
learner.add_cb(CSVLogger)


####### train
learner.fine_tune(FINE_TUNE_EPOCHS)


########## unfreeze and train
learner.unfreeze()
learner.fit_one_cycle(FIT_EPOCHS, lr_max=slice(1e-6, 1e-4))

########################### show results and save
learner.show_results(max_n=9, alpha=0.7)
show_results_path = LOG_BASE / "show_results.png"
plt.savefig(show_results_path, bbox_inches="tight", dpi=200)
plt.close()

########## export
learner.remove_cb(CSVLogger)
learner.export()
