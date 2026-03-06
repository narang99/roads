from typing import Callable, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image

from .crops import Bbox, padded_crop, bbox_only_mask


@dataclass
class CropLevelSample:
    """Raw crop-level data at original resolution."""
    
    full_image: np.ndarray  # RGB uint8, full source image
    full_mask: np.ndarray   # binary uint8, mask in full image coords
    bbox: Bbox             # bbox of mask pixels in full image coords


@dataclass 
class CropLevelPairs:
    """Multi-scale (image, mask) pairs for one crop annotation, ordered by increasing context."""
    
    pairs: list[tuple[np.ndarray, np.ndarray]]  # [tight crop, stored crop, full image]


def load_crop_level_sample_from_directory(d: Path, full_image_size: Optional[int] = None) -> CropLevelSample:
    """
    Load raw crop-level data from a sample directory structure.
    
    Args:
        d: Path to sample directory containing image.jpg, mask.png, meta.json
        full_image_size: If provided, resize full image and mask if either dimension exceeds this size
    """
    # Load stored crop to find mask bbox
    stored_mask = np.array(Image.open(d / "mask.png"))
    
    # Find bbox of mask pixels in stored crop
    _, labels, stats, _ = cv2.connectedComponentsWithStats(stored_mask.astype(np.uint8))
    if labels.max() > 0:
        crop_bbox_x, crop_bbox_y, crop_bbox_w, crop_bbox_h = (
            stats[1, cv2.CC_STAT_LEFT],
            stats[1, cv2.CC_STAT_TOP], 
            stats[1, cv2.CC_STAT_WIDTH],
            stats[1, cv2.CC_STAT_HEIGHT],
        )
    else:
        crop_bbox_x, crop_bbox_y = 0, 0
        crop_bbox_h, crop_bbox_w = stored_mask.shape
    
    # Load metadata and full image
    meta = json.loads((d / "meta.json").read_text())
    ox, oy = meta["crop_origin"]["x"], meta["crop_origin"]["y"]
    source_dir = (d / "source_dir").resolve()
    full_img = np.array(Image.open(source_dir / "image.jpg").convert("RGB"))
    
    # Create full mask by placing stored mask at crop origin
    fh, fw = full_img.shape[:2]
    full_mask = np.zeros((fh, fw), dtype=np.uint8)
    mh, mw = stored_mask.shape[:2]
    y2, x2 = min(oy + mh, fh), min(ox + mw, fw)
    full_mask[oy:y2, ox:x2] = stored_mask[:y2 - oy, :x2 - ox]
    
    # Convert crop-relative bbox to full image coordinates
    full_bbox = Bbox(
        x=ox + crop_bbox_x,
        y=oy + crop_bbox_y, 
        w=crop_bbox_w,
        h=crop_bbox_h
    )
    
    # Resize if full_image_size is provided and image is larger
    if full_image_size is not None and (fh > full_image_size or fw > full_image_size):
        # Calculate scale factor to resize to full_image_size
        scale = full_image_size / max(fh, fw)
        new_h, new_w = int(fh * scale), int(fw * scale)
        
        # Resize image and mask
        full_img = cv2.resize(full_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        full_mask = cv2.resize(full_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Scale bbox coordinates
        full_bbox = Bbox(
            x=int(full_bbox.x * scale),
            y=int(full_bbox.y * scale),
            w=int(full_bbox.w * scale), 
            h=int(full_bbox.h * scale)
        )
    
    return CropLevelSample(
        full_image=full_img,
        full_mask=full_mask,
        bbox=full_bbox
    )


def create_crop_level_sample(
    full_image: np.ndarray,
    full_mask: np.ndarray,
    bbox: Bbox
) -> CropLevelSample:
    """Create a CropLevelSample directly from full image, mask, and bbox."""
    return CropLevelSample(
        full_image=full_image,
        full_mask=full_mask,
        bbox=bbox
    )


def make_crop_level_pairs_v2(
    sample: CropLevelSample,
    tight_pad: int = 20,
    medium_pad: int = 200,
) -> CropLevelPairs:
    """
    Compute three (image, mask) pairs at different scales from a CropLevelSample.
    
      Pair 0 — tight crop: bbox + tight_pad context
      Pair 1 — medium crop: bbox + medium_pad context  
      Pair 2 — full image: complete source image with full mask
    """
    # Pair 0: tight crop around bbox
    if sample.bbox.w > 0 and sample.bbox.h > 0:
        tight_img, _, _ = padded_crop(sample.full_image, sample.bbox, tight_pad)
        tight_mask = bbox_only_mask(sample.full_mask, sample.bbox, tight_pad)
    else:
        tight_img, tight_mask = sample.full_image, sample.full_mask
    
    # Pair 1: medium crop around bbox  
    if sample.bbox.w > 0 and sample.bbox.h > 0:
        medium_img, _, _ = padded_crop(sample.full_image, sample.bbox, medium_pad)
        medium_mask = bbox_only_mask(sample.full_mask, sample.bbox, medium_pad)
    else:
        medium_img, medium_mask = sample.full_image, sample.full_mask
    
    # Pair 2: full image
    full_img, full_mask = sample.full_image, sample.full_mask
    
    return CropLevelPairs(
        pairs=[(tight_img, tight_mask), (medium_img, medium_mask), (full_img, full_mask)]
    )


def make_crop_level_pairs_factory_v2(
    tight_pad: int = 20,
    medium_pad: int = 200,
    full_image_size: int = 1024,
) -> Callable[[Path], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Factory returning a get_pairs callable for GenericMaskClassificationDataset.
    New version that reads like iter_taco_pairs structure.
    
    Args:
        tight_pad: Padding for tight crop
        medium_pad: Padding for medium crop
        full_image_size: If provided, resize full image and mask if either dimension exceeds this size
    """
    def get_pairs(d: Path) -> list[tuple[np.ndarray, np.ndarray]]:
        sample = load_crop_level_sample_from_directory(d, full_image_size)
        pairs_obj = make_crop_level_pairs_v2(sample, tight_pad, medium_pad)
        return pairs_obj.pairs
    
    return get_pairs