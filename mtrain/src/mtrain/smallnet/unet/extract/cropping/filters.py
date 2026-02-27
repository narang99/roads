from .bbox_based import get_bounding_boxes_connected
from tqdm import tqdm
import shutil
import numpy as np
from PIL import Image
from mtrain.utils import mkdir
from pathlib import Path


def filter_small_objects_in_ds(ds_dir: Path, dest_dir: Path, min_area: int):
    ims = list((ds_dir / "images").glob("*"))
    ims_and_masks = [(i, ds_dir / "masks" / f"{i.stem}.png") for i in ims]
    for _, m in ims_and_masks:
        if not m.exists():
            raise Exception(f"mask file {m} does not exist")
    dest_images = mkdir(dest_dir / "images")
    dest_masks = mkdir(dest_dir / "masks")

    for i, m in tqdm(ims_and_masks):
        mask = np.array(Image.open(m).convert("L"))
        mask = filter_small_objects_in_mask(mask, min_area)
        shutil.copy(i, dest_images / i.name)
        Image.fromarray(mask, "L").save(dest_masks / m.name)
    


def filter_small_objects_in_mask(mask, min_area: int):
    bboxes = get_bounding_boxes_connected(mask)
    new_mask = mask.copy()
    for bbox in bboxes:
        if bbox.h * bbox.w < min_area:
            new_mask[bbox.y:bbox.y2, bbox.x:bbox.x2] = 0
    return new_mask