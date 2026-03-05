from dataclasses import dataclass
from tqdm import tqdm
from uuid import uuid4
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from pycocotools.coco import COCO

from .extract import load_image, anns_to_mask
from .query import get_images_with_multiple_classes


@dataclass
class TacoSample:
    image: np.ndarray   # RGB uint8, shape (size, size, 3)
    mask: np.ndarray    # binary uint8, shape (size, size)
    cat_name: str
    img_id: int
    ann_id: int


def iter_taco_samples(
    coco: COCO,
    taco_dir: Path,
    size: int = 220,
) -> Iterator[TacoSample]:
    """
    Yield one TacoSample per annotation across all multi-class TACO images.

    The image is the full TACO image resized to (size x size).
    The mask isolates only the polygons of that single annotation, also resized.
    """
    taco_dir = Path(taco_dir)

    for img_info in get_images_with_multiple_classes(coco):
        img_id = img_info["id"]
        try:
            img_array = load_image(taco_dir / img_info["file_name"])
        except Exception as e:
            print(f"  [skip] img_id={img_id}: {e}")
            continue

        h, w = img_array.shape[:2]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        img_resized = cv2.resize(img_array, (size, size), interpolation=cv2.INTER_LANCZOS4)

        for ann in anns:
            mask = anns_to_mask([ann], h, w)
            mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            cat_name = coco.loadCats([ann["category_id"]])[0]["name"]
            yield TacoSample(
                image=img_resized,
                mask=mask_resized,
                cat_name=cat_name,
                img_id=img_id,
                ann_id=ann["id"],
            )


def build_taco_classification_dataset(
    coco: COCO,
    taco_dir: Path,
    output_dir: Path,
    size: int = 220,
) -> None:
    """
    Write a MaskClassificationDataset-compatible dataset from TACO.

    Output layout
    -------------
    output_dir/
      <category_name>/
        <img_id>_<ann_id>/
          image.jpg
          mask.png
    """
    output_dir = Path(output_dir)
    print(f"Building dataset → {output_dir}")

    for sample in tqdm(iter_taco_samples(coco, taco_dir, size)):
        sample_dir = output_dir / sample.cat_name / f"{sample.img_id}_{sample.ann_id}_{uuid4()}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / "image.jpg"), cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sample_dir / "mask.png"), sample.mask)

    print("Done.")
