import shutil
from mtrain.smallnet.unet.extract.cropping import create_crops_dataset
from mtrain.smallnet.unet.extract.taco_to_fastai import extract_taco_dataset
from pathlib import Path
from pycocotools.coco import COCO


if __name__ == '__main__':
    TACO_DIR = Path("/Users/hariomnarang/Desktop/personal/TACO/data")
    ANN_FILE = TACO_DIR / "annotations.json"
    EXP_BASE = Path("../../datasets/T007-uncentered/")
    OUT_DIR = EXP_BASE / "data"
    BIN_OUT = OUT_DIR / "binary"
    WORKERS = 8
    CROP_SIZE = 50
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coco = COCO(ANN_FILE)
    EXT_DIR = OUT_DIR / "EXT"
    CROPS_DIR = OUT_DIR / "engulfed-higher-skewing-50"
    engulfed_dir = CROPS_DIR


    shutil.rmtree(EXT_DIR, ignore_errors=True)
    shutil.rmtree(CROPS_DIR, ignore_errors=True)

    extract_taco_dataset(
        ann_file=ANN_FILE,
        taco_dir=TACO_DIR,
        out_dir=EXT_DIR,
        should_collapse_mask_to_binary=True,
    )

    create_crops_dataset(
        ANN_FILE,
        EXT_DIR,
        engulfed_dir,
        mode="engulf",
        workers=WORKERS,
        crop_size=CROP_SIZE,
        crops_per_image=10,
        max_pad_scale=10,
    )
