"""
For each sample in crop_level/, find the original image directory
(from the CLIP files) and create a symlink `source_dir` inside it.

Usage:
    python link_crop_sources.py [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

DS = Path(__file__).parents[2] / "datasets"
BASE = DS / "test-samples"
NEG_MASKING_V1 = BASE / "neg-masking" / "V1"
TRASH = NEG_MASKING_V1 / "trash"
ROCKS = NEG_MASKING_V1 / "rocks"

CLIP_FILE_NAMES = [
    "clip_bottles.txt",
    "clip_litter.txt",
    "clip_plastic.txt",
    "clip_tobacco_packs.txt",
    "clip_delhi_litter.txt",
]
CLIP_FILES = [TRASH / c for c in CLIP_FILE_NAMES]

# crop_level dirs to scan
CLASSIFICATION_ROOTS = [
    ROCKS / "classification" / "crop_level",
    TRASH / "data" / "crop_level",  # add more here if needed
]


def build_name_to_dir_index(clip_files: list[Path]) -> dict[str, Path]:
    """Return {dir_name: dir_path} from all CLIP files + fallback dir scans."""
    index: dict[str, Path] = {}

    # Primary: CLIP files
    for clip_file in clip_files:
        if not clip_file.exists():
            print(f"  [warn] CLIP file not found: {clip_file}", file=sys.stderr)
            continue
        for line in clip_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            img_path = Path(parts[1].strip())
            dir_path = img_path.parent
            index[dir_path.name] = dir_path

    # Fallback: scan known source roots for any dir containing image.jpg
    fallback_roots = [
        NEG_MASKING_V1 / "samples_mapillary",
        TRASH / "personal",
        TRASH / "delhi_litter",
    ]
    for root in fallback_roots:
        if not root.exists():
            continue
        for img in root.rglob("image.jpg"):
            d = img.parent
            index.setdefault(d.name, d)

    return index


def extract_image_name(crop_dir_name: str) -> str:
    """
    Strip the trailing _{N} bbox index from a crop directory name.
    e.g. '3436227063145344_15'  -> '3436227063145344'
         'd8faccd7-c4c5-41fb_3' -> 'd8faccd7-c4c5-41fb'
    """
    m = re.match(r"^(.+)_(\d+)$", crop_dir_name)
    if m:
        return m.group(1)
    return crop_dir_name  # fallback: no suffix found


def process_crop_level(root: Path, index: dict[str, Path], dry_run: bool) -> None:
    if not root.exists():
        print(f"  [skip] does not exist: {root}")
        return

    label_dirs = [d for d in root.iterdir() if d.is_dir()]
    total = ok = missing = already = 0

    for label_dir in label_dirs:
        for crop_dir in sorted(label_dir.iterdir()):
            if not crop_dir.is_dir():
                continue
            total += 1
            image_name = extract_image_name(crop_dir.name)
            symlink_path = crop_dir / "source_dir"

            if symlink_path.exists() or symlink_path.is_symlink():
                already += 1
                continue

            source = index.get(image_name)
            if source is None:
                print(f"  [miss] {crop_dir.name}  (looked up '{image_name}')")
                missing += 1
                continue

            if dry_run:
                print(f"  [dry]  {symlink_path} -> {source}")
            else:
                symlink_path.symlink_to(source)
            ok += 1

    print(
        f"{root.relative_to(DS)}: {total} samples — "
        f"{ok} linked, {already} already done, {missing} missing"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Building index from CLIP files…")
    index = build_name_to_dir_index(CLIP_FILES)
    print(f"  {len(index)} unique source dirs indexed\n")

    for root in CLASSIFICATION_ROOTS:
        process_crop_level(root, index, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
