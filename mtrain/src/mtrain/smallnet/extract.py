# main code to extract crops from an image
import string
import random
from .coco import CocoAnnotationFile
from .bbox import merge_overlapping_bboxes, get_non_trash_boxes, crop_with_bbox
from .tfms import PaddedResize
import cv2
from typing import Optional
import numpy as np
from pathlib import Path
from mtrain.cache import DEFAULT_SYNTH_CACHE, SuffixIn
import sys

def log_progress(i, total, step=10):
    prev = (i * 100) // total
    curr = ((i + 1) * 100) // total
    if curr // step > prev // step:
        print(f"{(curr // step) * step}% done", flush=True)
        sys.stdout.flush()


def dump_ddg_images(
    ddg_base_dir: Path,
    out_pos_dir: Path,
    box_size: int,
    max_samples: int,
    tfms=None,
):
    if tfms is None:
        tfms = PaddedResize(box_size)
    print(f"DDG Dumping: samples={max_samples} size={box_size} out_dir={out_pos_dir}")
    dump_all_images_as_pos_after_tfms(
        list(ddg_base_dir.glob("*.jpg")), out_pos_dir, max_samples, tfms,
    )


def dump_all_images_as_pos_after_tfms(
    images: list[Path],
    out_pos_dir: Path,
    max_samples: int,
    tfms,
):
    out_pos_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(images)
    print("shuffled set size", len(images))
    print("after samples size", len(images[:max_samples]))
    for i, im in enumerate(images[:max_samples]):
        log_progress(i, len(images[:max_samples]))
        fname = random_filename(6)
        path = out_pos_dir / fname
        img = cv2.imread(im)
        img = tfms(img)
        cv2.imwrite(path, img)


@DEFAULT_SYNTH_CACHE.decorator(
    key_args=[
        "ann_file",
        "images_root",
        "box_size",
        "empty_samples_ratio",
    ],
    output_arg="out_dir",
    num_samples_arg="total_samples",
    is_asset=SuffixIn([".jpg", ".jpeg", ".png"]),
)
def extract_all_taco(
    images_root: Path,
    ann_file: Path,
    out_dir: Path,
    box_size: int,
    total_samples: int,
    empty_samples_ratio: int = 1,
    tfms=None,
):
    if tfms is None:
        tfms = PaddedResize(box_size)
    print(f"TACO Extraction: samples={total_samples} size={box_size} out_dir={out_dir}")
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

    i = 0
    total_cycles = 0
    while i < total_samples:
        for img in images:
            n_dumped = _single_image_dump(
                img, coco_ann, empty_samples_ratio, pos, negs, tfms
            )
            i += n_dumped
            log_progress(i, total_samples)

            if i >= total_samples:
                break
        total_cycles += 1
    print("total cycles:", total_cycles)


def _single_image_dump(img, coco_ann, empty_samples_ratio, pos, negs, tfms) -> int:
    trashes, non_trashes = extract_crops_from_image(
        img, coco_ann, tfms, empty_samples_ratio
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
    tfms,
    empty_samples_ratio: int = 1,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    max_box_size = 1000
    # empty_samples_ratio: if there are n positive boxes, n*ratio negative boxes are returned
    boxes = coco_ann.extract_bboxes_with_path(image_path)
    if not boxes:
        return None, None
    boxes = merge_overlapping_bboxes(boxes)
    image = cv2.imread(image_path)

    # resizer = PaddedResize(box_size)

    crops = [crop_with_bbox(image, b, max_box_size)[0] for b in boxes]
    crops = [tfms(c) for c in crops]

    num_non_trash = empty_samples_ratio * len(crops)
    non_trash = get_non_trash_boxes(image, boxes, max_box_size, num_non_trash)
    non_trash = [tfms(n) for n in non_trash]

    return crops, non_trash
