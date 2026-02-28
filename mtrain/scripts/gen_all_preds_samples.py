
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from tqdm import tqdm
from fastai.vision.all import load_learner
from mtrain.utils import mkdir, show, draw_grid
from mtrain.smallnet.unet.predict import predict_unet_only_mask, overlay_mask_on_img
import torch
from mtrain.smallnet.unet.predict import predict_unet_only_mask, overlay_mask_on_img, strided_predict_unet_only_mask
from PIL import Image
import shutil
from tqdm import tqdm


def run_model_on_images(images, learner, sz, results_dir):
    for im in tqdm(images):
        dest = mkdir(results_dir / im.stem)
        shutil.copy(im, dest / f"image{im.suffix}")
        img = plt.imread(im)
        mask_dest = dest / "mask.png"
        res_dest = dest / "res.jpg"
        if mask_dest.exists() and res_dest.exists():
            continue
        mask = strided_predict_unet_only_mask(img, sz, learner, [50])
        # mask = predict_unet_only_mask(img, sz, learner)
        overlaid = overlay_mask_on_img(img, mask)
        Image.fromarray(mask, "L").save(dest / "mask.png")
        Image.fromarray(overlaid, "RGB").save(dest / "res.jpg")



DS = Path("../../datasets")
BASE = DS / "test-samples"
NEG_MASKING_DIR = mkdir(BASE / "neg-masking")
NEG_MASK_WORK_DIR = mkdir(BASE / "neg-masking" / "V1")

IMAGE_DIR = BASE / "positive-samples"
images = list(IMAGE_DIR.glob("*.jpg"))
len(images)

images = list((DS / "samples_mapillary").rglob("*.jpg"))

INITIAL_RESULTS_DIR = mkdir(NEG_MASK_WORK_DIR / "samples_mapillary" / "100")
# generate the masks first
learner100 = load_learner(
    "/Users/hariomnarang/Desktop/gdrive-sync/garbage/experiments/enguled-bbox-levels-crops-v3/log/export_iter_14.pkl"
)
SIZE = 100

run_model_on_images(images, learner100, SIZE, INITIAL_RESULTS_DIR)