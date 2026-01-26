from pathlib import Path
import shutil
import os
from mtrain.smallnet.unet.extract import generate_dataset
from fastai.vision.all import resnet18
from mtrain.smallnet.unet.train import get_dls
from fastai.vision.all import unet_learner, ProgressCallback, CSVLogger


DRIVE_BASE = Path(
    os.environ.get(
        "DRIVE_BASE", "/content/drive/Othercomputers/My MacBook Pro/gdrive-sync"
    )
)
TACO_DIR = Path(os.environ.get("TACO_DIR", "/content/data"))


PROJECT_CODE = os.environ["PROJECT_CODE"]
DATA_COUNT = int(os.environ["DATA_COUNT"])
TILE_SIZE = int(os.environ["TILE_SIZE"])
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
FINE_TUNE_EPOCHS = int(os.environ["FINE_TUNE_EPOCHS"])
FIT_EPOCHS = int(os.environ.get("FIT_EPOCHS"))

PROJECT_DIR = DRIVE_BASE / "garbage" / "experiments" / PROJECT_CODE
ANN_FILE = TACO_DIR / "annotations.json"
LOG_BASE = PROJECT_DIR / "log"
DATA_DIR = Path("/content/out")

PROJECT_DIR.mkdir(exist_ok=True, parents=True)
LOG_BASE.mkdir(exist_ok=True, parents=True)


######### synth gen

if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)
generate_dataset(ANN_FILE, TACO_DIR, DATA_DIR, TILE_SIZE, num_samples=DATA_COUNT)


######### dls


dls = get_dls(BATCH_SIZE, LOG_BASE, TILE_SIZE, DATA_DIR / "images", DATA_DIR / "masks")

######### learner

learner = unet_learner(dls, resnet18)
learner.remove_cb(ProgressCallback)
learner.add_cb(CSVLogger)


####### train
learner.fine_tune(FINE_TUNE_EPOCHS)


########## unfreeze and train
learner.unfreeze()
learner.fit_one_cycle(FIT_EPOCHS, lr_max=slice(1e-6, 1e-4))


########## export
learner.remove_cb(CSVLogger)
learner.export()
