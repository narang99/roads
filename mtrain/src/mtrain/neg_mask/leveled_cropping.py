from mtrain.disk import DiskBooleanMask
from typing import Callable, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image

from .crops import Bbox, padded_crop, bbox_only_mask, configurable_padded_crop, Paddings, configurable_bbox_only_mask


@dataclass
class CropLevelSample:
    """Raw crop-level data at original resolution."""

    full_image: np.ndarray  # RGB uint8, full source image
    full_mask: np.ndarray  # binary uint8, mask in full image coords
    bbox: Bbox  # bbox of mask pixels in full image coords


@dataclass
class CropLevelPairs:
    """Multi-scale (image, mask) pairs for one crop annotation, ordered by increasing context."""

    pairs: list[tuple[np.ndarray, np.ndarray]]  # [tight crop, stored crop, full image]


def load_crop_level_sample_from_directory(
    d: Path, full_image_size: Optional[int] = None
) -> CropLevelSample:
    """
    Load raw crop-level data from a sample directory structure.

    Args:
        d: Path to sample directory containing image.jpg, mask.png, meta.json
        full_image_size: If provided, resize full image and mask if either dimension exceeds this size
    """
    # Load stored crop to find mask bbox
    stored_mask = DiskBooleanMask.load(d / "mask.png")

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
    full_mask[oy:y2, ox:x2] = stored_mask[: y2 - oy, : x2 - ox]

    # Convert crop-relative bbox to full image coordinates
    full_bbox = Bbox(
        x=ox + crop_bbox_x, y=oy + crop_bbox_y, w=crop_bbox_w, h=crop_bbox_h
    )

    # Resize if full_image_size is provided and image is larger
    full_img, full_mask, full_bbox = _resize_image_and_mask_if_needed(
        full_img, full_mask, full_bbox, full_image_size
    )

    return CropLevelSample(full_image=full_img, full_mask=full_mask, bbox=full_bbox)


def _resize_image_and_mask_if_needed(full_img, full_mask, full_bbox, full_image_size):
    fh, fw = full_img.shape[:2]
    if full_image_size is not None and (fh > full_image_size or fw > full_image_size):
        # Calculate scale factor to resize to full_image_size
        scale = full_image_size / max(fh, fw)
        new_h, new_w = int(fh * scale), int(fw * scale)

        # Resize image and mask
        full_img = cv2.resize(
            full_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
        )
        full_mask = cv2.resize(
            full_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )
        # Scale bbox coordinates
        full_bbox = Bbox(
            x=int(full_bbox.x * scale),
            y=int(full_bbox.y * scale),
            w=int(full_bbox.w * scale),
            h=int(full_bbox.h * scale),
        )
    return full_img, full_mask, full_bbox


def create_crop_level_sample(
    full_image: np.ndarray,
    full_mask: np.ndarray,
    bbox: Bbox,
    full_image_size: Optional[int] = None,
) -> CropLevelSample:
    """Create a CropLevelSample directly from full image, mask, and bbox."""
    full_image, full_mask, bbox = _resize_image_and_mask_if_needed(
        full_image, full_mask, bbox, full_image_size
    )
    return CropLevelSample(full_image=full_image, full_mask=full_mask, bbox=bbox)


def _centered_medium_crop(
    image, mask, bbox, medium_pad
):
    crop_img, _, _ = padded_crop(image, bbox, medium_pad)
    crop_mask = bbox_only_mask(mask, bbox, medium_pad)
    return crop_img, crop_mask


def _random_medium_crop(
    image: np.ndarray,
    mask: np.ndarray, 
    bbox: Bbox,
    medium_pad: int,
    center_prob: float
) -> tuple[np.ndarray, np.ndarray]:
    """Create medium crop with random bbox placement based on probability."""
    import random
    
    # for now, we just do medium crop everytime
    # altho this crop is working, the downstream tfms in torch doi a centercrop
    # they do this to maintain ratio of the image
    # in this case we lose out on our object
    if random.random() < center_prob:
        # Use original centered crop
        return _centered_medium_crop(image, mask, bbox, medium_pad)
    # else:
    #     return _centered_medium_crop(image, mask, bbox, medium_pad)
    
    total_vert_pad = 2*medium_pad
    total_horiz_pad = 2*medium_pad

    pad_left = random.randint(5, total_horiz_pad)
    pad_right = total_horiz_pad - pad_left
    pad_top = random.randint(5, total_vert_pad)
    pad_bottom = total_vert_pad - pad_left
    pads = Paddings(pad_left, pad_right, pad_top, pad_bottom)
    crop_img, _, _ = configurable_padded_crop(image, bbox, pads)
    crop_mask = configurable_bbox_only_mask(mask, bbox, pads)
    return crop_img, crop_mask
    
    # Random placement with constraints
    img_h, img_w = image.shape[:2]
    crop_size = medium_pad * 2 + max(bbox.w, bbox.h)  # Minimum crop size needed
    
    # Calculate valid placement bounds (ensure FULL bbox stays inside crop)
    min_x = max(0, bbox.x2 - crop_size)  # Crop must include right edge of bbox
    max_x = min(img_w - crop_size, bbox.x)  # Crop must include left edge of bbox
    min_y = max(0, bbox.y2 - crop_size)  # Crop must include bottom edge of bbox 
    max_y = min(img_h - crop_size, bbox.y)  # Crop must include top edge of bbox
    
    if max_x >= min_x and max_y >= min_y:
        # Random placement possible
        crop_x = random.randint(min_x, max_x)
        crop_y = random.randint(min_y, max_y)
        
        # Extract crop at random position
        crop_img = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
        
        # Create mask with only bbox region visible in crop coordinates
        crop_mask = np.zeros((crop_size, crop_size), dtype=mask.dtype)
        
        # Map bbox to crop coordinates and extract region
        bbox_in_crop_x1 = max(0, bbox.x - crop_x)
        bbox_in_crop_y1 = max(0, bbox.y - crop_y)
        bbox_in_crop_x2 = min(crop_size, bbox.x2 - crop_x)
        bbox_in_crop_y2 = min(crop_size, bbox.y2 - crop_y)
        
        if bbox_in_crop_x2 > bbox_in_crop_x1 and bbox_in_crop_y2 > bbox_in_crop_y1:
            # Copy only the bbox region from original mask
            src_x1, src_y1 = bbox.x, bbox.y
            src_x2, src_y2 = bbox.x2, bbox.y2
            
            # Adjust source coordinates if crop extends beyond bbox
            if crop_x > bbox.x:
                src_x1 = crop_x
            if crop_y > bbox.y:
                src_y1 = crop_y
            if crop_x + crop_size < bbox.x2:
                src_x2 = crop_x + crop_size
            if crop_y + crop_size < bbox.y2:
                src_y2 = crop_y + crop_size
                
            bbox_region = mask[src_y1:src_y2, src_x1:src_x2]
            crop_mask[bbox_in_crop_y1:bbox_in_crop_y2, bbox_in_crop_x1:bbox_in_crop_x2] = bbox_region
        
        return crop_img, crop_mask
    else:
        # Fallback to centered when constraints can't be met
        crop_img, _, _ = padded_crop(image, bbox, medium_pad)
        crop_mask = bbox_only_mask(mask, bbox, medium_pad)
        return crop_img, crop_mask


def make_crop_level_pairs_v2(
    sample: CropLevelSample,
    tight_pad: int = 20,
    medium_pad: int = 200,
    medium_center_prob: float = 1.0,
) -> CropLevelPairs:
    """
    Compute three (image, mask) pairs at different scales from a CropLevelSample.

      Pair 0 — tight crop: bbox + tight_pad context
      Pair 1 — medium crop: bbox + medium_pad context (with random placement)
      Pair 2 — full image: complete source image with bbox-only mask
      
    Args:
        sample: CropLevelSample with full image, mask, and bbox
        tight_pad: Padding for tight crop
        medium_pad: Padding for medium crop
        medium_center_prob: Probability of centering bbox in medium crop (1.0 = always center)
    """
    # Pair 0: tight crop around bbox (always centered)
    if sample.bbox.w > 0 and sample.bbox.h > 0:
        tight_img, _, _ = padded_crop(sample.full_image, sample.bbox, tight_pad)
        tight_mask = bbox_only_mask(sample.full_mask, sample.bbox, tight_pad)
    else:
        tight_img, tight_mask = sample.full_image, sample.full_mask

    # Pair 1: medium crop with optional random placement
    if sample.bbox.w > 0 and sample.bbox.h > 0:
        medium_img, medium_mask = _random_medium_crop(
            sample.full_image, sample.full_mask, sample.bbox, 
            medium_pad, medium_center_prob
        )
    else:
        medium_img, medium_mask = sample.full_image, sample.full_mask

    # Pair 2: full image with only bbox region activated
    full_img = sample.full_image
    if sample.bbox.w > 0 and sample.bbox.h > 0:
        # Create mask with only the bbox region
        full_mask_bbox_only = np.zeros_like(sample.full_mask)
        bbox_region = sample.full_mask[sample.bbox.y:sample.bbox.y2, sample.bbox.x:sample.bbox.x2]
        full_mask_bbox_only[sample.bbox.y:sample.bbox.y2, sample.bbox.x:sample.bbox.x2] = bbox_region
        full_mask = full_mask_bbox_only
    else:
        full_mask = sample.full_mask

    return CropLevelPairs(
        pairs=[
            (tight_img, tight_mask),
            (medium_img, medium_mask),
            (full_img, full_mask),
        ]
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
