import random
from mtrain.smallnet.tfms import PaddedResize
import shutil
from itertools import islice
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pathlib import Path
import os
import numpy as np
from mtrain.cache import DEFAULT_SYNTH_CACHE
from PIL import Image, ImageDraw
import random
from mtrain.utils import random_filename
from PIL import Image
import itertools
from mtrain.tqdm import Progress
import tempfile


def generate_dataset(ann_file, taco_dir, output_path, tile_size, num_samples):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        extract_images_and_masks(
            ann_file=ann_file,
            taco_dir=taco_dir,
            output_path=tmp,
            num_samples=num_samples,
        )
        print(tmp.ls())
        create_crops_from_extracted(ann_file, tmp, output_path, tile_size, num_samples)

    # coco = COCO(ann_file)
    # images_dir = Path(output_path) / "images"
    # masks_dir = Path(output_path) / "masks"
    # images_dir.mkdir(parents=True, exist_ok=True)
    # masks_dir.mkdir(parents=True, exist_ok=True)

    # imgs = coco.loadImgs(coco.getImgIds())
    # if num_samples is not None:
    #     imgs = imgs[:num_samples]

    # random.shuffle(imgs)
    # it = itertools.islice(itertools.cycle(imgs), num_samples)
    # print("starting data generation")
    # for i, img_info in enumerate(it):
    #     progress(i, num_samples)
    #     im, mask, bbox = generate_random_crops_with_masks(
    #         coco, taco_dir, img_info, tile_size, 1000, num_crops=1
    #     )
    #     fname = random_filename(6)
    #     Image.fromarray(im).save(images_dir / f"{fname}.png")
    #     Image.fromarray(mask).save(masks_dir / f"{fname}.png")


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


@DEFAULT_SYNTH_CACHE.decorator(
    output_arg="output_path",
    num_samples_arg="num_samples",
    key_args=["ann_file", "taco_dir", "num_samples"],
)
def extract_images_and_masks(ann_file, taco_dir, output_path, num_samples=None):
    """
    Extract images and their corresponding masks from TACO dataset.

    Args:
        ann_file: Path to COCO annotations file
        taco_dir: Path to TACO dataset images
        output_path: Directory to save extracted images and masks
        num_samples: Number of samples to extract (None for all)
    """
    coco = COCO(ann_file)
    images_dir = Path(output_path) / "images"
    masks_dir = Path(output_path) / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    imgs = coco.loadImgs(coco.getImgIds())
    if num_samples is not None:
        imgs = imgs[:num_samples]

    category_ids = sorted({ann["category_id"] for ann in coco.loadAnns(coco.getAnnIds())})
    catid2maskid = {cat_id: i+1 for i, cat_id in enumerate(category_ids)}  # 0 = background
    codes = ["background"] + [coco.loadCats(cat_id)[0]["name"] for cat_id in category_ids]

    progress = Progress(len(imgs))
    print("Starting image and mask extraction")
    for i, img_info in enumerate(imgs):
        progress(i)

        img_id = img_info["id"]
        img_filename = img_info["file_name"]

        # Load image
        img_path = os.path.join(taco_dir, img_filename)
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        # Get annotations and create mask
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Create mask
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for ann in anns:
            if "segmentation" in ann:
                mask_value = catid2maskid[ann["category_id"]]
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape(-1, 2)
                    mask_img = Image.fromarray(mask)
                    draw = ImageDraw.Draw(mask_img)
                    draw.polygon([tuple(p) for p in poly], fill=mask_value)
                    mask = np.array(mask_img)

        # Save with original filename stem
        print("img iddddddd", img_id)
        fname = f"{img_id}.png" 
        Image.fromarray(img_array).save(images_dir / fname)
        Image.fromarray(mask).save(masks_dir / fname)
    with open(output_path / "codes.txt", "w") as f:
        f.write("\n".join(codes))


def collapse_to_binary_dataset(orig_dir, binary_dir, background_label=0):
    
    orig_dir = Path(orig_dir)
    binary_dir = Path(binary_dir)
    if binary_dir.exists():
        print(f"output directory {binary_dir} exists, nuking")
        shutil.rmtree(binary_dir)

    images_dir = orig_dir / "images"
    masks_dir = orig_dir / "masks"
    codes_file = orig_dir / "codes.txt"

    # Create new directories
    (binary_dir / "images").mkdir(parents=True, exist_ok=True)
    (binary_dir / "masks").mkdir(parents=True, exist_ok=True)

    # Optional: read original codes
    if codes_file.exists():
        with open(codes_file) as f:
            codes = [line.strip() for line in f]
        print(f"Original codes: {codes}")

    # Process masks
    for mask_path in masks_dir.glob("*.png"):
        mask = np.array(Image.open(mask_path))
        mask_binary = (mask != background_label).astype(np.uint8)
        Image.fromarray(mask_binary).save(binary_dir / "masks" / mask_path.name)

    # Copy images
    for img_path in images_dir.glob("*.*"):
        shutil.copy(img_path, binary_dir / "images" / img_path.name)

    # Write new codes.txt
    with open(binary_dir / "codes.txt", "w") as f:
        f.write("background\nforeground\n")

    print(f"Binary dataset created at: {binary_dir}")


def create_crops_from_extracted(ann_file, input_dir, output_dir, tile_size, num_crops, max_padding=1000):
    """
    Create random crops from previously extracted images and masks.

    Args:
        input_dir: Directory containing images/ and masks/ subdirectories
        output_dir: Directory to save cropped images and masks
        tile_size: Size of crops (int for square)
        num_crops: Total number of crops to generate
    """
    coco = COCO(ann_file)
    input_images_dir = Path(input_dir) / "images"
    input_masks_dir = Path(input_dir) / "masks"

    output_images_dir = Path(output_dir) / "images"
    output_masks_dir = Path(output_dir) / "masks"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(input_images_dir.glob("*.png"))

    if len(image_files) == 0:
        print("No images found in input directory")
        return

    print(f"Creating {num_crops} crops from {len(image_files)} images")

    # Cycle through images to generate crops
    random.shuffle(image_files)
    it = itertools.islice(itertools.cycle(image_files), num_crops)

    progress = Progress(num_crops)
    for i, img_path in enumerate(it):
        progress(i)

        mask_path = input_masks_dir / img_path.name

        # Load image and mask
        # img = np.array(Image.open(img_path))
        # mask = np.array(Image.open(mask_path))

        # Generate crop
        crop_img, crop_mask, _ = _extract_single_crop(coco, img_path, mask_path, tile_size, max_padding=max_padding)
        # crop_img, crop_mask = generate_random_crop(
        #     img, mask, tile_size, max_padding=1000
        # )

        # Save with random filename
        fname = img_path.stem
        Image.fromarray(crop_img).save(output_images_dir / f"{fname}.png")
        Image.fromarray(crop_mask).save(output_masks_dir / f"{fname}.png")


def _extract_single_crop(coco, img_path, mask_path, crop_size, max_padding):
    # Pick a random annotation to center the crop around

    img_id = int(img_path.stem)
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]

    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

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

    # Crop image and mask
    crop_img = img_array[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_mask = mask_array[crop_y1:crop_y2, crop_x1:crop_x2]
    resizer = PaddedResize(crop_size)
    crop_img_resized = resizer(crop_img)
    crop_mask_resized = resizer(crop_mask)

    crop_bbox = [crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1]
    return (crop_img_resized, crop_mask_resized, crop_bbox)



def generate_random_crop(img_array, mask_array, crop_size, max_padding):
    """
    Generate a single random crop from image and mask where mask has objects.

    Args:
        img_array: Image as numpy array
        mask_array: Mask as numpy array
        crop_size: Size of crop (int for square)
        max_padding: Maximum padding around objects

    Returns:
        Tuple of (cropped_image, cropped_mask)
    """
    if isinstance(crop_size, int):
        crop_w, crop_h = crop_size, crop_size
    else:
        crop_w, crop_h = crop_size

    img_height, img_width = img_array.shape[:2]

    # Find objects in mask
    if mask_array.max() == 0:
        # No objects, return random crop
        crop_x = random.randint(0, max(0, img_width - crop_w))
        crop_y = random.randint(0, max(0, img_height - crop_h))
    else:
        # Find bounding box of mask objects
        rows = np.any(mask_array > 0, axis=1)
        cols = np.any(mask_array > 0, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Add random padding
        pad_left = random.randint(0, max_padding)
        pad_right = random.randint(0, max_padding)
        pad_top = random.randint(0, max_padding)
        pad_bottom = random.randint(0, max_padding)

        crop_x1 = max(0, int(x_min - pad_left))
        crop_y1 = max(0, int(y_min - pad_top))
        crop_x2 = min(img_width, int(x_max + pad_right))
        crop_y2 = min(img_height, int(y_max + pad_bottom))

        # Ensure minimum crop size
        if crop_x2 - crop_x1 < crop_w:
            crop_x2 = min(img_width, crop_x1 + crop_w)
        if crop_y2 - crop_y1 < crop_h:
            crop_y2 = min(img_height, crop_y1 + crop_h)

        crop_x, crop_y = crop_x1, crop_y1

    # Extract crop
    crop_img = img_array[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    crop_mask = mask_array[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

    # Resize if needed
    if crop_img.shape[0] != crop_h or crop_img.shape[1] != crop_w:
        crop_img_pil = Image.fromarray(crop_img)
        crop_mask_pil = Image.fromarray(crop_mask)

        crop_img_pil = crop_img_pil.resize((crop_w, crop_h), Image.BILINEAR)
        crop_mask_pil = crop_mask_pil.resize((crop_w, crop_h), Image.NEAREST)

        crop_img = np.array(crop_img_pil)
        crop_mask = np.array(crop_mask_pil)

    return crop_img, crop_mask


def show_extracted_dataset(d, n=8):
    ims = d / "images"
    msks = d / "masks"
    res = []
    for im in islice(ims.glob("*"), n):
        msk = msks / im.name
        res.append((im, msk))

    num = min(n, len(res))
    _,ax = plt.subplots(num, 2, figsize=(10, 4*num))
    for i in range(num):
        r0 = np.array(Image.open(res[i][0]))
        r1 = np.array(Image.open(res[i][1]))
        ax[i][0].imshow(r0)
        ax[i][1].imshow(r1)
    plt.tight_layout()
    plt.show()