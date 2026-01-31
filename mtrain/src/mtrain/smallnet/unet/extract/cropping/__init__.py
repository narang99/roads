from PIL import Image
import functools
from typing import Optional, Literal
from mtrain.tqdm import Progress
from mtrain.chunk import chunk_list
import random
from mtrain.random import (
    random_filename,
    random_true_one_three_times,
    add_jitter_pixels,
)
from mtrain.smallnet.tfms import PaddedResize
from mtrain.smallnet.unet.extract.cropping import engulf, cut
from pycocotools.coco import COCO
from multiprocessing import Pool


CropType = Optional[Literal["engulf", "cut"]]


def create_crops_dataset(
    ann_file,
    in_dir,
    out_dir,
    images_to_sample=-1,
    max_pad_scale=4,
    crops_per_image=5,
    crop_size=50,
    max_skew=3,
    mode: CropType = None,
    workers: int = 4,
):
    coco = COCO(ann_file)
    in_images_dir, in_masks_dir = in_dir / "images", in_dir / "masks"
    out_images_dir, out_masks_dir = out_dir / "images", out_dir / "masks"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    images = list(in_images_dir.rglob("*.jpeg"))
    random.shuffle(images)
    images = _arr_subset(images, images_to_sample)

    chunker = ChunkCropper(
        in_masks_dir,
        coco,
        max_pad_scale,
        crops_per_image,
        max_skew,
        mode,
        crop_size,
        out_images_dir,
        out_masks_dir,
    )
    with Pool(workers) as p:
        p.map(chunker, chunk_list(images, workers))


class ChunkCropper:
    def __init__(
        self,
        in_masks_dir,
        coco,
        max_pad_scale,
        crops_per_image,
        max_skew,
        mode,
        crop_size,
        out_images_dir,
        out_masks_dir,
    ):
        self.in_masks_dir = in_masks_dir
        self.coco = coco
        self.max_pad_scale = max_pad_scale
        self.crops_per_image = crops_per_image
        self.max_skew = max_skew
        self.mode = mode
        self.crop_size = crop_size
        self.out_images_dir = out_images_dir
        self.out_masks_dir = out_masks_dir

    def __call__(self, img_paths):
        progress = Progress(len(img_paths), "Create Crops", 5)
        for i, img_path in enumerate(img_paths):
            mask_path = self.in_masks_dir / f"{img_path.stem}.png"
            res = extract_crops_for_single_image(
                self.coco,
                img_path,
                mask_path,
                self.max_pad_scale,
                self.crops_per_image,
                self.max_skew,
                self.mode,
            )
            resizer = PaddedResize(self.crop_size)
            for img, mask in res:
                try:
                    img, mask = resizer(img), resizer(mask)
                except Exception as ex:
                    print("image shape", img.shape, "mask shape", mask.shape)
                    print("reason", str(ex))
                    raise
                _save_crop(img, mask, self.out_images_dir, self.out_masks_dir)
            progress(i)


def _arr_subset(arr, n):
    if n < 0:
        return arr
    else:
        return arr[:n]


def _save_crop(img, mask, out_images_dir, out_masks_dir):
    fname = random_filename()
    Image.fromarray(img, "RGB").save(out_images_dir / f"{fname}.jpeg")
    Image.fromarray(mask, "L").save(out_masks_dir / f"{fname}.png")


def extract_crops_for_single_image(
    coco,
    img_path,
    mask_path,
    max_pad_scale,
    num_samples,
    max_skew=3,
    mode: CropType = None,
):
    # you should generally have a high number of num_samples, it helps
    # in providing a variety of data points
    if num_samples == 0:
        return []
    if num_samples < 0:
        raise Exception(f"num samples has to be positive, passed = {num_samples}")

    # first add the sample without any skewing

    engulf_extractor = functools.partial(
        engulf.extract_single_crop,
        coco=coco,
        img_path=img_path,
        mask_path=mask_path,
        max_pad_scale=max_pad_scale,
    )
    res = []
    # one without skew for each
    anns_len = engulf.get_num_annotations(coco, img_path)
    for idx in range(anns_len):
        res.append(
            engulf_extractor(
                horiz_skew=1,
                vert_skew=1,
                max_padding=add_jitter_pixels(100),
                ann_idx=idx,
            )
        )

    # one centered small for each
    for idx in range(anns_len):
        res.append(
            engulf_extractor(
                horiz_skew=1,
                vert_skew=1,
                max_padding=add_jitter_pixels(4000),
                min_padding=add_jitter_pixels(2000),
                ann_idx=idx,
            ),
        )

    # add skews and randomisation
    for _ in range(num_samples):
        horiz_skew = random.choice([-1, 1]) * random.uniform(1, max_skew)
        vert_skew = random.choice([-1, 1]) * random.uniform(1, max_skew)

        # cut with a probability of 1/3
        if mode is None:
            mode = "cut" if random_true_one_three_times() else "engulf"
        if mode == "engulf":
            res.append(
                engulf.extract_single_crop(
                    coco, img_path, mask_path, max_pad_scale, horiz_skew, vert_skew
                )
            )
        else:
            res.append(
                cut.extract_crop_by_cutting_object(
                    coco, img_path, mask_path, max_pad_scale
                )
            )
    return res
