# main code to extract crops from an image
from tqdm import tqdm
import string
import random
from .coco import CocoAnnotationFile
from .bbox import merge_overlapping_bboxes, get_non_trash_boxes, crop_with_bbox
from .tfms import PaddedResize
import cv2
from typing import Optional
import numpy as np
from pathlib import Path


def extract_all(
    images_root: Path,
    ann_file: Path,
    out_dir: Path,
    box_size: int,
    total_samples: int,
    empty_samples_ratio: int = 1,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    negs = out_dir / "neg"
    pos = out_dir / "pos"
    negs.mkdir(parents=True, exist_ok=True)
    pos.mkdir(parents=True, exist_ok=True)
    existing = len(list(pos.glob("*.png"))) + len(list(negs.glob("*.png")))
    print("existing samples:", existing)
    total_samples -= existing
    total_samples = max(total_samples, 0)
    print("remaining (will generate):", total_samples)

    coco_ann = CocoAnnotationFile(ann_file)

    images = list(images_root.rglob("*.JPG"))
    random.shuffle(images)

    pbar = tqdm(total=total_samples, desc="Generating samples", unit="iteration")
    i = 0
    total_cycles = 0
    while i < total_samples:
        for img in images:
            n_dumped = _single_image_dump(img, coco_ann, box_size, empty_samples_ratio, pos, negs)
            i += n_dumped
            pbar.update(n_dumped)

            if i >= total_samples:
                break
        total_cycles += 1
    print("total cycles:", total_cycles)



def _single_image_dump(img, coco_ann, box_size, empty_samples_ratio, pos, negs) -> int:
    trashes, non_trashes = extract_crops_from_image(
        img, coco_ann, box_size, empty_samples_ratio
    )
    if trashes is None or non_trashes is None:
        return 0

    i = 0
    for t in trashes:
        fname = random_filename(6)
        path = pos / fname
        cv2.imwrite(path, t)
        i += 1
    for t in non_trashes:
        fname = random_filename(6)
        path = negs / fname
        cv2.imwrite(path, t)
        i += 1
    return i


def random_filename(k):
    chars = string.ascii_lowercase + string.digits
    name = "".join(random.choices(chars, k=k))
    return f"{name}.png"


def extract_crops_from_image(
    image_path,
    coco_ann: CocoAnnotationFile,
    box_size: int,
    empty_samples_ratio: int = 1,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    max_box_size = 1000
    # empty_samples_ratio: if there are n positive boxes, n*ratio negative boxes are returned
    boxes = coco_ann.extract_bboxes_with_path(image_path)
    if not boxes:
        return None, None
    boxes = merge_overlapping_bboxes(boxes)
    image = cv2.imread(image_path)

    resizer = PaddedResize(box_size)

    crops = [crop_with_bbox(image, b, max_box_size)[0] for b in boxes]
    crops = [resizer(c) for c in crops]

    num_non_trash = empty_samples_ratio * len(crops)
    non_trash = get_non_trash_boxes(image, boxes, max_box_size, num_non_trash)
    non_trash = [resizer(n) for n in non_trash]


    return crops, non_trash
