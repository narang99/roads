import random
import cv2
from mtrain.smallnet.tfms import PaddedResize
import shutil
from itertools import islice
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pathlib import Path
import os
import numpy as np
from mtrain.cache import DEFAULT_SYNTH_CACHE
from PIL import Image
import itertools
from mtrain.tqdm import Progress
import tempfile
from mtrain.random import random_filename
from mtrain.chunk import chunk_list
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


@DEFAULT_SYNTH_CACHE.decorator(
    output_arg="output_path",
    key_args=["ann_file", "taco_dir", "num_samples", "tile_size"],
)
def generate_dataset(
    ann_file, taco_dir, output_path, tile_size, num_samples, workers=8
):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ex1 = tmp / "multi-level"
        extract_images_and_masks(
            ann_file=ann_file,
            taco_dir=taco_dir,
            output_path=ex1,
            num_samples=num_samples,
            workers=workers,
        )
        print("extraction: done")
        ex2 = tmp / "binary-level"
        collapse_to_binary_dataset(ex1, ex2, workers=workers)
        print("binary collapse: done")
        # ann_file, input_dir, output_dir, tile_size, num_crops, max_padding=1000, workers=8
        create_crops_from_extracted(
            ann_file=ann_file,
            input_dir=ex2,
            output_dir=output_path,
            tile_size=tile_size,
            num_crops=num_samples,
            workers=4,
        )
        print("crops: done")


@DEFAULT_SYNTH_CACHE.decorator(
    output_arg="output_path",
    key_args=["ann_file", "taco_dir", "num_samples"],
)
def extract_images_and_masks(
    ann_file, taco_dir, output_path, num_samples=None, workers=8
):
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
    BAD_SAMPLE_IDS = [359, 229]
    imgs = [i for i in imgs if i["id"] not in BAD_SAMPLE_IDS]
    random.shuffle(imgs)
    if num_samples is not None:
        imgs = imgs[:num_samples]

    category_ids = sorted(
        {ann["category_id"] for ann in coco.loadAnns(coco.getAnnIds())}
    )
    catid2maskid = {
        cat_id: i + 1 for i, cat_id in enumerate(category_ids)
    }  # 0 = background
    codes = ["background"] + [
        coco.loadCats(cat_id)[0]["name"] for cat_id in category_ids
    ]

    print("Starting image and mask extraction")
    hp = PoolHelper(coco, taco_dir, catid2maskid, images_dir, masks_dir)
    with _get_pool(len(imgs), workers) as p:
        p.map(hp, list(enumerate(chunk_list(imgs, workers))))

    with open(output_path / "codes.txt", "w") as f:
        f.write("\n".join(codes))


class PoolHelper:
    def __init__(self, coco, taco_dir, catid2maskid, images_dir, masks_dir):
        self.coco = coco
        self.taco_dir = taco_dir
        self.catid2maskid = catid2maskid
        self.images_dir = images_dir
        self.masks_dir = masks_dir

    def __call__(self, id_and_chunk):
        cid, chunk = id_and_chunk
        _extract_images_and_mask_chunk(
            cid,
            chunk,
            self.coco,
            self.taco_dir,
            self.catid2maskid,
            self.images_dir,
            self.masks_dir,
        )


def _extract_images_and_mask_chunk(
    chunk_id, img_infos, coco, taco_dir, catid2maskid, images_dir, masks_dir
):
    progress = Progress(len(img_infos), f"EXTRACT; CHUNK={chunk_id}", step=5)
    for i, img_info in enumerate(img_infos):
        _extract_images_and_mask_single(
            img_info, coco, taco_dir, catid2maskid, images_dir, masks_dir
        )
        progress(i)


def _extract_images_and_mask_single(
    img_info, coco, taco_dir, catid2maskid, images_dir, masks_dir
):
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
                # seg is your flat list of coordinates
                poly = np.array(seg).reshape(-1, 2)  # shape (num_points, 2)
                # OpenCV expects an array of shape (num_points, 1, 2) and type int32
                pts = poly.reshape((-1, 1, 2)).astype(np.int32)
                # fill the polygon in-place on the mask
                cv2.fillPoly(mask, [pts], color=mask_value)

    # Save with original filename stem
    fname = str(img_id)
    Image.fromarray(img_array).save(images_dir / f"{fname}.jpeg")
    _save_mask(mask, masks_dir / f"{fname}.png")
    return img_array, mask


def collapse_to_binary_dataset(orig_dir, binary_dir, background_label=0, workers=8):
    print(f"starting binary collapse of dataset, input={orig_dir} output={binary_dir}")
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
    all_masks_paths = list(masks_dir.glob("*.png"))
    progress = Progress(len(all_masks_paths), "CVT MASK: ")
    mph = MpCollapseHelper(background_label, binary_dir)
    with _get_pool(len(all_masks_paths), workers) as p:
        p.map(mph, list(enumerate(chunk_list(all_masks_paths, workers))))

    # Copy images
    images_list = list(images_dir.glob("*.*"))
    progress = Progress(len(images_list), "COPY_IMAGE")
    for i, img_path in enumerate(images_list):
        progress(i)
        shutil.copy(img_path, binary_dir / "images" / img_path.name)

    # Write new codes.txt
    with open(binary_dir / "codes.txt", "w") as f:
        f.write("background\nforeground\n")

    print(f"Binary dataset created at: {binary_dir}")


class MpCollapseHelper:
    def __init__(self, bg_label, binary_dir):
        self._bg_label = bg_label
        self._bin_dir = binary_dir

    def __call__(self, id_and_mask_paths):
        i, mask_paths = id_and_mask_paths
        progress = Progress(len(mask_paths), f"CVT MASK: {i}", step=5)
        for p in mask_paths:
            progress(i)
            mask = np.array(Image.open(p))
            mask_binary = mask != self._bg_label
            _save_mask(mask_binary, self._bin_dir / "masks" / p.name)


def _save_mask(mask, path):
    mask = mask.astype(np.uint8)
    Image.fromarray(mask, mode="L").save(path, format="PNG", compress_level=9)


def _corres_mask_name(img_path):
    return f"{Path(img_path).stem}.png"


@DEFAULT_SYNTH_CACHE.decorator(
    output_arg="output_dir",
    key_args=["ann_file", "tile_size", "num_crops", "max_padding"],
)
def create_crops_from_extracted(
    ann_file, input_dir, output_dir, tile_size, num_crops, max_padding=1000, workers=8
):
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

    if output_dir.exists():
        print(f"CROPS: output directory {output_dir} exists, nuking")
        shutil.rmtree(output_dir)
    output_images_dir = Path(output_dir) / "images"
    output_masks_dir = Path(output_dir) / "masks"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(input_images_dir.glob("*.jpeg"))

    if len(image_files) == 0:
        print("No images found in input directory")
        return

    print(f"Creating {num_crops} crops from {len(image_files)} images")

    # Cycle through images to generate crops
    random.shuffle(image_files)
    it = itertools.islice(itertools.cycle(image_files), num_crops)
    images = list(it)

    # progress = Progress(num_crops)

    mpcrp = MpCropHelper(
        input_masks_dir,
        tile_size,
        max_padding,
        coco,
        output_images_dir,
        output_masks_dir,
    )
    with _get_pool(len(images), workers) as p:
        p.map(mpcrp, enumerate(chunk_list(images, workers)))

    # for i, img_path in enumerate(it):
    #     progress(i)

    #     mask_path = input_masks_dir / _corres_mask_name(img_path)

    #     # Generate crop
    #     crop_img, crop_mask, _ = _extract_single_crop(
    #         coco, img_path, mask_path, tile_size, max_padding=max_padding
    #     )
    #     # Save with random filename
    #     img_fname = img_path.name
    #     Image.fromarray(crop_img).save(output_images_dir / img_fname)
    #     _save_mask(crop_mask, output_masks_dir / _corres_mask_name(img_fname))


class MpCropHelper:
    def __init__(
        self,
        input_masks_dir,
        tile_size,
        max_padding,
        coco,
        output_images_dir,
        output_masks_dir,
    ):
        self.input_masks_dir = input_masks_dir
        self.tile_size = tile_size
        self.max_padding = max_padding
        self.coco = coco
        self.output_images_dir = output_images_dir
        self.output_masks_dir = output_masks_dir

    def __call__(self, i_and_chunk):
        chunk_id, paths = i_and_chunk
        progress = Progress(len(paths), f"MPCrop {chunk_id}:")
        for i, img_path in enumerate(paths):
            mask_path = self.input_masks_dir / _corres_mask_name(img_path)

            # Generate crop
            try:
                crop_img, crop_mask, _ = _extract_single_crop(
                    self.coco,
                    img_path,
                    mask_path,
                    self.tile_size,
                    max_padding=self.max_padding,
                )
            except Exception as ex:
                print(
                    f"failure in generating crop for {img_path}, reason={ex}. ignoring"
                )
                continue
            # Save with random filename
            img_fname = img_path.name
            Image.fromarray(crop_img).save(self.output_images_dir / img_fname)
            _save_mask(crop_mask, self.output_masks_dir / _corres_mask_name(img_fname))
            progress(i)


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


def show_extracted_dataset(d, n=8, img_id=None):
    ims = d / "images"
    msks = d / "masks"
    res = []
    it = ims.glob("*")
    if img_id is not None:

        def _same_as_img_id(im_path):
            try:
                return int(Path(im_path).stem) == int(img_id)
            except:
                return False

        it = filter(_same_as_img_id, it)

    for im in islice(it, n):
        msk = msks / _corres_mask_name(im)
        res.append((im, msk))

    num = min(n, len(res))
    print(f"results: {num}")
    _, ax = plt.subplots(num, 2, figsize=(10, 3 * num))
    if num == 1:
        r0 = np.array(Image.open(res[0][0]))
        r1 = np.array(Image.open(res[0][1]))
        ax[0].imshow(r0)
        ax[1].imshow(r1)
    else:
        for i in range(num):
            r0 = np.array(Image.open(res[i][0]))
            r1 = np.array(Image.open(res[i][1]))
            ax[i][0].imshow(r0)
            ax[i][1].imshow(r1)
    plt.tight_layout()
    plt.show()


def collate(all_extracted_dirs, output_dir, codes):
    # we do not collate the codes.txt files for now, they are lost
    # assume the user passes the codes
    # randomise input file names and mask names coherently
    all_extracted_dirs = list(map(Path, all_extracted_dirs))
    output_dir = Path(output_dir)

    if output_dir.exists():
        print(f"COLLATE: output directory {output_dir} exists, nuking")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    images_dir, masks_dir = output_dir / "images", output_dir / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    with open(output_dir / "codes.txt", "w") as f:
        f.write("\n".join(codes))

    progress = Progress(len(all_extracted_dirs), "Multiple Extraction")
    for i, exd in enumerate(all_extracted_dirs):
        _randomise_and_dump_single_extracted_dir(exd, images_dir, masks_dir)
        progress(i)


def _randomise_and_dump_single_extracted_dir(
    extracted_dir, dest_images_dir, dest_masks_dir
):
    src_images_dir, src_masks_dir = extracted_dir / "images", extracted_dir / "masks"
    images = list(src_images_dir.glob("*"))

    for i, image in enumerate(images):
        mask = src_masks_dir / _corres_mask_name(image)
        if not mask.exists():
            raise Exception(f"mask for image={image} does not exist at {mask}")
        fname = random_filename()
        shutil.copy(image, dest_images_dir / f"{fname}{image.suffix}")
        shutil.copy(mask, dest_masks_dir / f"{fname}{mask.suffix}")


def _get_pool(samples_len, workers):
    if samples_len > 500:
        return Pool(workers)
    else:
        return ThreadPool(1)
