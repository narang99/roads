from PIL import Image
import numpy as np
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
from mtrain.smallnet.unet.extract.cropping import engulf, cut, v2_engulf
from pycocotools.coco import COCO
from multiprocessing import Pool
from . import bbox_based


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
    min_padding=None,
    min_length=None,
    bbox_heights=None,
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
        min_padding,
        min_length,
        bbox_heights,
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
        min_padding,
        min_length,
        bbox_heights=None,
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
        self.min_padding = min_padding
        self.min_length = min_length
        small_crops = list(range(5, min(crop_size, 20), 3))
        if crop_size > 20:
            big_crops = list(range(20, crop_size, 5))
        else:
            big_crops = []
        crop_levels = big_crops + small_crops
        self.bbox_heights = (
            crop_levels if bbox_heights is None else bbox_heights
        )
        print("Chunker: bbox heights:", self.bbox_heights)

    def __call__(self, img_paths):
        success, failures = 0, 0
        progress = Progress(len(img_paths), "Create Crops", 5)
        for i, img_path in enumerate(img_paths):
            mask_path = self.in_masks_dir / f"{img_path.stem}.png"
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            fname_prefix = random_filename()
            it = bbox_based.extract_crops_for_single_image(
                img,
                mask,
                self.bbox_heights,
                self.crop_size,
                self.crop_size,
            )
            for idx, (rsz_img, rsz_mask) in enumerate(it):
                if rsz_img is None:
                    failures += 1
                else:
                    fname = f"{fname_prefix}_{idx}"
                    _save_crop(
                        rsz_img,
                        rsz_mask,
                        self.out_images_dir,
                        self.out_masks_dir,
                        fname=fname,
                    )
                    success += 1
            progress(i)
        print(f"Success: {success}")
        print(f"Failures: {failures}")

        # resizer = PaddedResize(self.crop_size)
        # img, mask = resizer(img), resizer(mask)
        # _save_crop(img, mask, self.out_images_dir, self.out_masks_dir)

        # res = extract_crops_for_single_image(
        #     self.coco,
        #     img_path,
        #     mask_path,
        #     self.max_pad_scale,
        #     self.crops_per_image,
        #     self.max_skew,
        #     self.mode,
        #     min_length=self.min_length,
        # )
        # resizer = PaddedResize(self.crop_size)
        # for img, mask in res:
        #     try:
        #         img, mask = resizer(img), resizer(mask)
        #     except Exception as ex:
        #         print("image shape", img.shape, "mask shape", mask.shape)
        #         print("reason", str(ex))
        #         raise
        #     _save_crop(img, mask, self.out_images_dir, self.out_masks_dir)
        # progress(i)


def _arr_subset(arr, n):
    if n < 0:
        return arr
    else:
        return arr[:n]


def _save_crop(img, mask, out_images_dir, out_masks_dir, fname=None):
    fname = random_filename() if fname is None else fname
    Image.fromarray(img, "RGB").save(out_images_dir / f"{fname}.jpeg")
    Image.fromarray(mask, "L").save(out_masks_dir / f"{fname}.png")


def extract_crops_for_single_image(
    coco,
    img_path,
    mask_path,
    max_pad_scale,
    num_samples,
    max_skew=5,
    mode: CropType = None,
    min_padding=None,
    min_length=None,
):
    # you should generally have a high number of num_samples, it helps
    # in providing a variety of data points
    if num_samples == 0:
        return []
    if num_samples < 0:
        raise Exception(f"num samples has to be positive, passed = {num_samples}")

    engulf_extractor = functools.partial(
        v2_engulf.extract_single_crop,
        coco=coco,
        img_path=img_path,
        mask_path=mask_path,
    )
    res = []
    # TODO: what is this?
    # first add the sample without any skewing
    # one without skew for each
    # for idx in range(anns_len):
    #     res.append(
    #         engulf_extractor(
    #             horiz_skew=1,
    #             vert_skew=1,
    #             max_padding=add_jitter_pixels(100),
    #             ann_idx=idx,
    #             min_length=min_length,
    #         )
    #     )

    anns_len = engulf.get_num_annotations(coco, img_path)
    # one centered small for each
    for idx in range(anns_len):
        yield engulf_extractor(
            horiz_skew=1,
            vert_skew=1,
            max_padding=add_jitter_pixels(100),
            ann_idx=idx,
        )

    # one for extreme skew for each id
    skews = {
        "h": [1, -10, 10],
        "v": [1, -10, 10],
    }
    skew_min_length = max(min_length, 1000) if min_length is not None else 1000
    for hskew in skews["h"]:
        for vskew in skews["v"]:
            if hskew == 1 and vskew == 1:
                continue

            for idx in range(anns_len):
                yield engulf_extractor(
                    horiz_skew=hskew,
                    vert_skew=vskew,
                    ann_idx=idx,
                    min_length=skew_min_length,
                    coords_fn=v2_engulf.skewed_coords,
                )

    # # add skews and randomisation
    for _ in range(num_samples):
        random_min_skew = random.randint(1, max_skew)
        horiz_skew = random.choice([-1, 1]) * random.uniform(random_min_skew, max_skew)
        vert_skew = random.choice([-1, 1]) * random.uniform(random_min_skew, max_skew)

        # cut with a probability of 1/3
        if mode is None:
            mode = "cut" if random_true_one_three_times() else "engulf"
        if mode == "engulf":
            yield engulf_extractor(
                horiz_skew=horiz_skew,
                vert_skew=vert_skew,
                min_length=min_length,
            )
        else:
            yield cut.extract_crop_by_cutting_object(
                coco, img_path, mask_path, max_pad_scale
            )
    return res
