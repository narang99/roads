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


def anns_to_mask(anns_sel, height, width, value=1):
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns_sel:
        for seg in ann["segmentation"]:
            poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly], value)
    return mask
