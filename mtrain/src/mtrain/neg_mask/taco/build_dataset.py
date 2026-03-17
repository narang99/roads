from dataclasses import dataclass
from tqdm import tqdm
from uuid import uuid4
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from pycocotools.coco import COCO

from .extract import load_image, anns_to_mask
from ..crops import Bbox, get_region_crops
from ..leveled_cropping import CropLevelSample, CropLevelPairs, create_crop_level_sample, make_crop_level_pairs_v2


@dataclass
class TacoSample:
    """TACO-specific metadata for a CropLevelSample."""
    
    crop_sample: CropLevelSample
    cat_name: str
    img_id: int
    ann_id: int


@dataclass
class TacoPairs:
    """Multi-scale (image, mask) pairs for one annotation, ordered by increasing context."""

    pairs: list[tuple[np.ndarray, np.ndarray]]  # [tight crop, mid crop, full resized]
    cat_name: str
    img_id: int
    ann_id: int


def iter_taco_samples(
    coco: COCO,
    taco_dir: Path,
    starting_size: int = 1000,
) -> Iterator[TacoSample]:
    """
    Yield one TacoSample per annotation across all TACO images.
    Images and masks are downsampled to starting_size × starting_size at load time;
    bbox coordinates are scaled accordingly.
    """
    taco_dir = Path(taco_dir)

    for img_info in coco.loadImgs(coco.getImgIds()):
        img_id = img_info["id"]
        try:
            img_array = load_image(taco_dir / img_info["file_name"])
        except Exception as e:
            print(f"  [skip] img_id={img_id}: {e}")
            continue

        h, w = img_array.shape[:2]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        img_array = cv2.resize(
            img_array, (starting_size, starting_size), interpolation=cv2.INTER_LANCZOS4
        )

        for ann in anns:
            mask = anns_to_mask([ann], h, w)
            mask = cv2.resize(
                mask, (starting_size, starting_size), interpolation=cv2.INTER_NEAREST
            )
            cat_name = coco.loadCats([ann["category_id"]])[0]["supercategory"]

            bboxes = list(get_region_crops(mask))
            if not bboxes:
                print(
                    f"  [skip] img_id={img_id} ann_id={ann['id']}: empty mask after resize"
                )
                continue

            crop_sample = create_crop_level_sample(
                full_image=img_array,
                full_mask=mask,
                bbox=bboxes[0]
            )
            
            yield TacoSample(
                crop_sample=crop_sample,
                cat_name=cat_name,
                img_id=img_id,
                ann_id=ann["id"],
            )


def make_taco_pairs(
    sample: TacoSample,
    full_image_size: int = 220,
    small_image_pad: int = 20,
    medium_image_pad: int = 200,
) -> TacoPairs:
    """
    Compute three (image, mask) pairs at different scales from a raw TacoSample.

      Pair 0 — tight crop: original image cropped around bbox + small_image_pad
      Pair 1 — mid crop:   original image cropped around bbox + medium_image_pad
      Pair 2 — full:       full image resized to full_image_size × full_image_size
    """
    # Use the leveled cropping utility for pairs 0 and 1
    crop_pairs = make_crop_level_pairs_v2(
        sample.crop_sample, 
        tight_pad=small_image_pad, 
        medium_pad=medium_image_pad
    )
    
    # Override pair 2 with resized full image (TACO-specific behavior)
    img2 = cv2.resize(
        sample.crop_sample.full_image,
        (full_image_size, full_image_size),
        interpolation=cv2.INTER_LANCZOS4,
    )
    mask2 = cv2.resize(
        sample.crop_sample.full_mask, 
        (full_image_size, full_image_size), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Replace the third pair with resized version
    pairs = crop_pairs.pairs[:2] + [(img2, mask2)]

    return TacoPairs(
        pairs=pairs,
        cat_name=sample.cat_name,
        img_id=sample.img_id,
        ann_id=sample.ann_id,
    )


def iter_taco_pairs(
    coco: COCO,
    taco_dir: Path,
    starting_size: int = 1000,
    full_image_size: int = 220,
    small_image_pad: int = 20,
    medium_image_pad: int = 200,
) -> Iterator[TacoPairs]:
    for sample in iter_taco_samples(coco, taco_dir, starting_size):
        yield make_taco_pairs(
            sample, full_image_size, small_image_pad, medium_image_pad
        )


def build_taco_classification_dataset(
    coco: COCO,
    taco_dir: Path,
    output_dir: Path,
    starting_size: int = 1000,
    full_image_size: int = 220,
    small_image_pad: int = 20,
    medium_image_pad: int = 200,
) -> None:
    """
    Write a multi-scale dataset from TACO annotations.

    Output layout
    -------------
    output_dir/
      <category_name>/
        <img_id>_<ann_id>_<uuid>/
          img_pair_0.jpg / mask_pair_0.png  — tight crop from original image (bbox + small_image_pad)
          img_pair_1.jpg / mask_pair_1.png  — fixed crop from original full-res image
          img_pair_2.jpg / mask_pair_2.png  — full resized image + full mask
    """
    output_dir = Path(output_dir)
    print(f"Building dataset → {output_dir}")

    def save_pair(img: np.ndarray, mask: np.ndarray, d: Path, i: int) -> None:
        cv2.imwrite(str(d / f"img_pair_{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(d / f"mask_pair_{i}.png"), mask)

    for pairs in tqdm(
        iter_taco_pairs(
            coco, taco_dir, starting_size, full_image_size, small_image_pad, medium_image_pad
        )
    ):
        sample_dir = (
            output_dir / pairs.cat_name / f"{pairs.img_id}_{pairs.ann_id}_{uuid4()}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        for i, (img, mask) in enumerate(pairs.pairs):
            save_pair(img, mask, sample_dir, i)

    print("Done.")
