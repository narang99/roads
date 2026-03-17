import json
from mtrain.neg_mask.taco.query import get_cat_id_to_cat, get_cat_ids_of_image_id
from PIL import Image, ExifTags
import functools
import numpy as np
import cv2
from pathlib import Path
from pycocotools.coco import COCO
from mtrain.tqdm import Progress
from mtrain.cache import DEFAULT_SYNTH_CACHE


@DEFAULT_SYNTH_CACHE.decorator(
    output_arg="out_dir",
    key_args=["ann_file", "taco_dir", "should_collapse_mask_to_binary", "num_samples"],
)
def extract_taco_dataset(
    ann_file: Path,
    taco_dir: Path,
    out_dir: Path,
    should_collapse_mask_to_binary: bool,
    num_samples: int = -1,
):
    coco = COCO(ann_file)
    images_out, masks_out = out_dir / "images", out_dir / "masks"
    meta_file = out_dir / "meta.json"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    img_ids = coco.getImgIds()
    img_ids = _get_subset(img_ids, num_samples)
    progress = Progress(len(img_ids), "Extract TACO", 5)

    for i, img_id in enumerate(img_ids):
        img, mask = extract_mask_for_image_id(img_id, coco, taco_dir)
        mask = _collapse_mask_if_binary_needed(mask, should_collapse_mask_to_binary)
        fname = str(img_id)
        Image.fromarray(img, "RGB").save(images_out / f"{fname}.jpeg")
        Image.fromarray(mask, "L").save(masks_out / f"{fname}.png")

        progress(i)
    
    persist_meta_for_all_images(meta_file, coco)


def persist_meta_for_all_images(meta_path, coco):
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    img_ids = coco.getImgIds()
    cat_id_to_name, cat_id_to_sup = get_cat_id_to_cat(coco)

    img_id_to_cats = {}
    img_id_to_sups = {}
    for img_id in img_ids:
        cats, sups = _get_cats_of_img_id(coco, img_id, cat_id_to_name, cat_id_to_sup)
        img_id_to_cats[str(img_id)] = cats
        img_id_to_sups[str(img_id)] = sups
    
    res = {
        "img_id_to_cats": img_id_to_cats,
        "img_id_to_sups": img_id_to_sups
    }

    with open(meta_path, "w") as f:
        json.dump(res, f)


def _get_cats_of_img_id(coco, img_id, cat_id_to_name, cat_id_to_sup) -> tuple[list[str], list[str]]:
    cat_ids = get_cat_ids_of_image_id(coco, img_id)
    sups = list(set([cat_id_to_sup[cid] for cid in cat_ids]))
    cats = list(set([cat_id_to_name[cid] for cid in cat_ids]))
    return cats, sups

def _get_subset(img_ids, num_samples):
    if num_samples == -1:
        return img_ids
    else:
        return img_ids[:num_samples]

def _collapse_mask_if_binary_needed(mask, need_binary):
    if need_binary:
        mask = mask != 0
        return mask.astype(np.uint8)
    else:
        return mask


def extract_mask_for_image_id(img_id, coco, taco_dir):
    image_path = taco_dir / coco.loadImgs(img_id)[0]["file_name"]
    annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
    anns_sel = coco.loadAnns(annIds)
    img_array = load_image(image_path)
    h, w = img_array.shape[:2]
    mask = anns_to_mask(anns_sel, h, w)
    return img_array, mask


def load_image(image_path):
    # Obtain Exif orientation tag code
    orientation = get_orientation_tag()

    img = Image.open(image_path)

    # Load and process image metadata
    if img._getexif() and orientation:
        exif = dict(img._getexif().items())
        # Rotate portrait and upside down images if necessary
        if orientation in exif:
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            if exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            if exif[orientation] == 8:
                img = img.rotate(90, expand=True)

    img = img.convert("RGB")
    return np.array(img)


@functools.lru_cache(maxsize=1)
def get_orientation_tag():
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == "Orientation":
            return orientation
    return None


def anns_to_mask(anns_sel, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns_sel:
        for seg in ann["segmentation"]:
            poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly], _fixed_cat_id(ann["category_id"]))
    return mask

def _fixed_cat_id(cat_id):
    return cat_id+1