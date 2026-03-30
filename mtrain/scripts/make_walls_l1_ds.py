from itertools import batched
import os
from pathlib import Path
from tqdm import tqdm
from mtrain.utils import *
from mtrain.seg import mapillary as mapi
from mtrain.example_dir import ExampleDir
from mtrain.neg_mask.walls import get_trash_mask_regions_fully_enclosed_in_mapi_region
from functools import partial
from mtrain.tqdm import Progress
from mtrain.neg_mask.crops import Bbox
from dataclasses import dataclass, asdict
import random
from collections import defaultdict
import multiprocessing


multiprocessing.set_start_method("spawn", force=True)


NEGMASK_DATASET_DIR = Path(
    "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/training"
)


@dataclass
class ModelResult:
    bbox: Bbox
    label: str

    @classmethod
    def from_json(cls, json_path) -> "ModelResult":
        with open(json_path) as f:
            mres = json.load(f)
            bb = mres["bbox"]
            bb = Bbox(x=bb["x"], y=bb["y"], w=bb["w"], h=bb["h"])
            mres = ModelResult(bbox=bb, label=mres["label"])
            return mres


def multi_save(edirs: list[ExampleDir], l1_dest_dir: Path):
    print("got edirs", len(edirs))
    progress = Progress(len(edirs), f"Thread: {os.getpid()}", 5)
    for i, edir in enumerate(edirs):
        try:
            save_to_l1(edir, l1_dest_dir)
            progress(i)
        except Exception as ex:
            print(f"unexpected failure, skipping {edir.d}, ex={ex}")


def save_to_l1(edir: ExampleDir, l1_dest_dir: Path):
    # we want the relative path to be symlinked
    a = edir.load_all_assets("md", "md")
    image = edir.load_and_resize_image(edir.image_path)
    mapi_mask = mapi.get_mask_with_labels(
        a["mapi_pred"], [mapi.Label.WALL, mapi.Label.BUILDING]
    )
    if (l1_dest_dir / f"{edir.d.name}-0").exists():
        # skip, i know this check is not enough
        # but we have a lot of data and it is going to take time
        return
    regions = get_trash_mask_regions_fully_enclosed_in_mapi_region(a["mask"], mapi_mask)
    regions = binned_sample(regions, 5)
    for i, (region, bbox) in enumerate(regions):
        dest = mkdir(l1_dest_dir / f"{edir.d.name}-{i}")
        image_dest = dest / "image.jpg"
        if i == 0:
            DiskImage.save(image, image_dest)
        else:
            # Symlink to the first region's image
            target = Path("..") / f"{edir.d.name}-0" / "image.jpg"
            if image_dest.exists():
                image_dest.unlink()
            image_dest.symlink_to(target)
        DiskBooleanMask.save(region, dest / "mask.png")
        mres = ModelResult(
            bbox=Bbox(x=int(bbox.x), y=int(bbox.y), w=int(bbox.w), h=int(bbox.h)),
            label="other",
        )
        with open(dest / "model.json", "w") as f:
            json.dump(asdict(mres), f)


def binned_sample(tuples, n_total=5, n_bins=10):
    if not tuples:
        return []

    areas = [bbox.h * bbox.w for _, bbox in tuples]
    lo, hi = min(areas), max(areas)

    # All same area — just random sample directly
    if lo == hi:
        return random.sample(tuples, min(n_total, len(tuples)))

    bins = defaultdict(list)
    for item in tuples:
        _, bbox = item
        area = bbox.w * bbox.h
        bin_idx = min(int((area - lo) / (hi - lo) * n_bins), n_bins - 1)
        bins[bin_idx].append(item)

    non_empty = list(bins.values())
    chosen_bins = random.sample(non_empty, min(n_total, len(non_empty)))
    return [random.choice(bucket) for bucket in chosen_bins]


if __name__ == "__main__":
    WALLS_MAPILLARY = Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/training/walls-mapillary"
    )
    RAW_DATA = WALLS_MAPILLARY / "raw_data"
    WALLS_L1 = WALLS_MAPILLARY / "L1"

    dirs = globL(RAW_DATA, "*")
    edirs = [ExampleDir(d, {}, {}) for d in dirs]

    func = partial(multi_save, l1_dest_dir=WALLS_L1)
    with multiprocessing.Pool(processes=4) as p:
        batches = list(batched(edirs, 200))
        print("starting, total =", len(edirs))
        p.map(func, batches)
