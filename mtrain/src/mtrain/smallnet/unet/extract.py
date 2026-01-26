import random
from pycocotools.coco import COCO
from pathlib import Path
from mtrain.utils import random_filename
from PIL import Image
import itertools
from tqdm import tqdm


def generate_dataset(ann_file, taco_dir, output_path, tile_size, num_samples):
    coco = COCO(ann_file)
    images_dir = Path(output_path) / "images"
    masks_dir = Path(output_path) / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    imgs = coco.loadImgs(coco.getImgIds())
    if num_samples is not None:
        imgs = imgs[:num_samples]

    random.shuffle(imgs)
    it = itertools.islice(itertools.cycle(imgs), num_samples)
    for img_info in tqdm(it, total=num_samples):
        im, mask, bbox = generate_random_crops_with_masks(
            coco, taco_dir, img_info, tile_size, 1000, num_crops=1
        )
        fname = random_filename(6)
        Image.fromarray(im).save(images_dir / f"{fname}.png")
        Image.fromarray(mask).save(masks_dir / f"{fname}.png")


def generate_random_crops_with_masks(
    coco, dataset_path, img_info, crop_size, max_padding, num_crops=5
):
    """
    Generate random crops of image containing garbage along with corresponding masks.

    Args:
        coco: COCO object
        dataset_path: Path to TACO images
        img_info: Image info dict from COCO
        crop_size: Size of crops (width, height) or single int for square
        max_padding: Maximum padding around bounding box
        num_crops: Number of random crops to generate per image

    Returns:
        List of tuples: (crop_array, mask_array, crop_bbox)
        where crop_bbox is [x, y, w, h] of the crop in original image
    """
    import os
    import numpy as np
    from PIL import Image, ImageDraw
    import random

    if isinstance(crop_size, int):
        crop_w, crop_h = crop_size, crop_size
    else:
        crop_w, crop_h = crop_size

    img_id = img_info["id"]
    img_filename = img_info["file_name"]

    # Load image
    img_path = os.path.join(dataset_path, img_filename)
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]

    # Get annotations
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    if len(anns) == 0:
        return []

    # Create full mask for the image
    full_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for ann in anns:
        if "segmentation" in ann:
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape(-1, 2)
                mask_img = Image.fromarray(full_mask)
                draw = ImageDraw.Draw(mask_img)
                draw.polygon([tuple(p) for p in poly], fill=1)
                full_mask = np.array(mask_img)

    # Generate random crops
    # crops_data = []

    # Pick a random annotation to center the crop around
    ann = random.choice(anns)
    x, y, w, h = ann["bbox"]

    # Random padding on each side
    pad_left = random.randint(0, max_padding)
    pad_right = random.randint(0, max_padding)
    pad_top = random.randint(0, max_padding)
    pad_bottom = random.randint(0, max_padding)

    # Calculate crop boundaries
    crop_x1 = max(0, int(x - pad_left))
    crop_y1 = max(0, int(y - pad_top))
    crop_x2 = min(img_width, int(x + w + pad_right))
    crop_y2 = min(img_height, int(y + h + pad_bottom))

    # Ensure minimum crop size
    if crop_x2 - crop_x1 < crop_w:
        crop_x2 = min(img_width, crop_x1 + crop_w)
    if crop_y2 - crop_y1 < crop_h:
        crop_y2 = min(img_height, crop_y1 + crop_h)

    # Crop image and mask
    crop_img = img_array[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_mask = full_mask[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to target crop size
    crop_img_pil = Image.fromarray(crop_img)
    crop_mask_pil = Image.fromarray(crop_mask)

    crop_img_resized = crop_img_pil.resize((crop_w, crop_h), Image.BILINEAR)
    crop_mask_resized = crop_mask_pil.resize((crop_w, crop_h), Image.NEAREST)

    crop_img_array = np.array(crop_img_resized)
    crop_mask_array = np.array(crop_mask_resized)

    crop_bbox = [crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1]

    return (crop_img_array, crop_mask_array, crop_bbox)
