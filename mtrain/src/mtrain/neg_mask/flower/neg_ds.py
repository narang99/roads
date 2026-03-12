# create a negative dataset
from mtrain.neg_mask.leveled_cropping import (
    load_crop_level_sample_from_directory,
    make_crop_level_pairs_v2,
)
from pathlib import Path


def get_crop_level_trash_from_root(
    crop_level_trash_dir: Path, crop_size: int, start_image_size: int = 1024
):
    return get_crop_level_trash_from_dirs(
        crop_level_trash_dir.glob("*"), crop_size, start_image_size
    )


def get_crop_level_trash_from_dirs(
    labelled_dirs, crop_size: int, start_image_size: int = 1024
):
    for d in labelled_dirs:
        if not d.is_dir():
            continue
        if (
            not (d / "source_dir" / "image.jpg").exists()
            or not (d / "meta.json").exists()
        ):
            continue
        try:
            sample = load_crop_level_sample_from_directory(d, start_image_size)
            level_pairs = make_crop_level_pairs_v2(sample, crop_size, crop_size + 100, 1)
        except Exception as ex:
            print(f"WARN: failure in getting crop; dir={d.name} reason={ex}")
        
        crop = level_pairs.pairs[0][0]

        yield d, crop
