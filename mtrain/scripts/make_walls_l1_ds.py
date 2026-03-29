from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from tqdm import tqdm
import json
from mtrain.neg_mask.leveled_cropping import load_crop_level_sample_from_directory
from mtrain.neg_mask.model.datasets.foviate_shrink import get_foviated_image_and_mask
from mtrain.neg_mask.crops import get_region_crops, bbox_only_mask, get_largest_bbox
import albumentations as A
from mtrain.utils import *
from mtrain.smallnet.unet.extract.draw import show_extracted_dataset
import shutil
from mtrain.seg import mapillary as mapi
from mtrain.example_dir import ExampleDir
from mtrain.smallnet.unet.extract.taco_to_fastai import extract_taco_dataset
from pycocotools.coco import COCO
from mtrain.neg_mask.walls import get_trash_mask_regions_fully_enclosed_in_mapi_region
import shutil


def save_to_l1(edir: ExampleDir, l1_dest_dir: Path):
    # we want the relative path to be symlinked
    a = edir.load_all_assets("md", "md")
    image = edir.load_and_resize_image(edir.image_path)
    mapi_mask = mapi.get_mask_with_labels(
        a["mapi_pred"], [mapi.Label.WALL, mapi.Label.BUILDING]
    )
    regions = get_trash_mask_regions_fully_enclosed_in_mapi_region(a["mask"], mapi_mask)
    for i, (region, bbox) in enumerate(regions):
        dest = mkdir(l1_dest_dir / f"{edir.d.name}-{i}")
        DiskImage.save(image, dest / "image.jpg")
        DiskBooleanMask.save(region, dest / "mask.png")


if __name__ == "__main__":
    WALLS_MAPILLARY = Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/training/walls-mapillary"
    )
    RAW_DATA = WALLS_MAPILLARY / "raw_data"
    WALLS_L1 = WALLS_MAPILLARY / "L1"

    dirs = globL(RAW_DATA, "*")
    edirs = [ExampleDir(d, {}, {}) for d in dirs]
    for edir in tqdm(edirs):
        save_to_l1(edir, WALLS_L1)
