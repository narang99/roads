# bulk create segmentations
from typing import Literal
import json
import argparse
from mtrain.cache import DEFAULT_SYNTH_CACHE
from mtrain.seg import mapillary, elevated_vegetation
from pathlib import Path
from tqdm import tqdm


def create_mapillary_segments(
    image_dir: Path, out_dir: Path, exts: tuple
):
    _create_segments(image_dir=image_dir, out_dir=out_dir, exts=exts, module="mapillary")


def create_elev_segments(
    image_dir: Path, out_dir: Path, exts: tuple
):
    _create_segments(image_dir=image_dir, out_dir=out_dir, exts=exts, module="elev")


@DEFAULT_SYNTH_CACHE.decorator(output_arg="out_dir", key_args=["image_dir", "exts", "module"])
def _create_segments(
    image_dir: Path, out_dir: Path, exts: tuple, module: Literal["mapillary", "elev"]
):
    exts = set(exts)
    images = list(image_dir.rglob("*"))
    print(f"total number of all files: {len(images)}")
    for img in tqdm(images):
        if img.suffix not in exts:
            continue
        # find the parent directory relative to root directory
        path_comp = img.parent.resolve().relative_to(image_dir.resolve())
        dest = out_dir / path_comp / f"{img.stem}.json"
        if dest.exists():
            continue
        mod = mapillary if module == "mapillary" else elevated_vegetation
        mask = mod.cached_predict(img)
        # maintain relative directory semantics
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            json.dump(mask.tolist(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate segmentations for images")
    parser.add_argument("input", type=Path, help="Input directory containing images")
    parser.add_argument("output", type=Path, help="Output directory for segmentations")
    parser.add_argument(
        "--exts",
        type=str,
        default=".jpg,.png",
        help="Comma-separated file extensions (default: .jpg,.png)",
    )
    parser.add_argument(
        "--type",
        choices=["mapillary", "elev"],
        default="mapillary",
        help="Segmentation type (default: mapillary)",
    )

    args = parser.parse_args()
    exts = tuple(args.exts.split(","))
    args.output.mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()

    print(f"IMAGE DIRECTORY: {image_dir}")
    print(f"OUTPUT DIRECTORY: {out_dir}")

    if args.type == "mapillary":
        create_mapillary_segments(image_dir=image_dir, out_dir=out_dir, exts=exts)
    else:
        create_elev_segments(image_dir=image_dir, out_dir=out_dir, exts=exts)
