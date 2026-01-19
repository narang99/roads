#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
YOLO Dataset Splitting Script
Splits a YOLO dataset into train/val/test sets with proper directory structure.
Run --help for doc
"""

import argparse
import os
import random
import shutil
from pathlib import Path
import json


EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


def create_yolo_directories(base_path):
    directories = [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]

    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)


def _file_image_for_label(label_path, image_stem_by_path: dict):
    base = Path(label_path).stem
    if base in image_stem_by_path:
        return Path(image_stem_by_path[base])
    raise Exception(f"image file not found for label={label_path}")


def _get_all_image_files(source_images_dir) -> dict:
    stem_by_path = {}
    for ext in EXTS:
        for img in Path(source_images_dir).rglob(f"*{ext}"):
            stem_by_path[img.stem] = img
        for img in Path(source_images_dir).rglob(f"*{ext.upper()}"):
            stem_by_path[img.stem] = img
    return stem_by_path


def _get_splitted(label_files, train_ratio, val_ratio, test_ratio):
    # Shuffle the list for random split

    total_files = len(label_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count

    print(f"Total images: {total_files}")
    print(f"Train: {train_count} ({train_ratio:.1%})")
    print(f"Val: {val_count} ({val_ratio:.1%})")
    print(f"Test: {test_count} ({test_ratio:.1%})")

    # Split files
    train_files = label_files[:train_count]
    val_files = label_files[train_count : train_count + val_count]
    test_files = label_files[train_count + val_count :]

    # Copy files to respective directories
    sets = [("train", train_files), ("val", val_files), ("test", test_files)]
    return sets


def _move_single_label(label_file, image_dest_dir, label_dest_dir, image_stem_by_path, dry_run):
    image_path = _file_image_for_label(label_file, image_stem_by_path)
    image_dest = Path(image_dest_dir) / image_path.name
    label_file = Path(label_file)
    label_dest = Path(label_dest_dir) / label_file.name

    if label_dest.exists() and image_dest.exists():
        return True

    if image_path.exists() and label_file.exists():
        if dry_run:
            print(f"mv: {image_path}->{image_dest}    {label_file}->{label_dest}")
        else:
            shutil.copy2(image_path, image_dest)
            shutil.copy2(label_file, label_dest)
        return True
    else:
        return False


def split_dataset(
    source_images_dir,
    source_labels_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    dry_run=False,
):
    source_labels_dir = Path(source_labels_dir)
    source_images_dir = Path(source_images_dir)
    output_dir = Path(output_dir)
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, val, and test ratios must sum to 1.0")

    # Create output directories
    create_yolo_directories(output_dir)
    label_files = list(source_labels_dir.glob("*.txt"))
    random.shuffle(label_files)

    image_stem_by_path = _get_all_image_files(source_images_dir)
    if not image_stem_by_path:
        raise ValueError(f"No image files found in {source_images_dir}")

    sets = _get_splitted(label_files, train_ratio, val_ratio, test_ratio)

    missing_labels = []
    for set_name, files in sets:
        print(f"\nProcessing {set_name} set...")
        image_out = output_dir / "images" / set_name
        label_out = output_dir / "labels" / set_name

        for label_file in files:
            moved = _move_single_label(
                label_file, image_out, label_out, image_stem_by_path, dry_run
            )
            if not moved:
                missing_labels.append(label_file)

    if missing_labels:
        print(f"\nWarning: {len(missing_labels)} labels had missing images")
        for label in missing_labels[:5]:  # Show first 5
            print(f"  {label}")
        if len(missing_labels) > 5:
            print(f"  ... and {len(missing_labels) - 5} more")
    else:
        print(f"Nothing missing. Total labels prepared = {len(label_files)}")

    print(f"\nDataset split completed! Output directory: {output_dir}")


def update_yaml_config(output_dir, comma_separated_classes):
    """Update data.yaml file with new paths"""
    classes = comma_separated_classes.split(",")
    names = json.dumps(classes)
    output_dir = Path(output_dir).resolve()

    yaml_content = f"""# YOLOv8 dataset configuration
path: .
train: images/train
val: images/val
test: images/test

# Classes
nc: {len(classes)}
names: {names}
"""

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Updated data.yaml at: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset into train/val/test sets"
    )
    parser.add_argument(
        "--images", required=True, help="Path to source images directory"
    )
    parser.add_argument(
        "--labels", required=True, help="Path to source labels directory"
    )
    parser.add_argument(
        "--classes", required=True, help="Comma separated string of classes of boxes, in the order in which source labels are used"
    )
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.05, help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Validate input directories
    if not os.path.exists(args.images):
        raise ValueError(f"Images directory does not exist: {args.images}")
    if not os.path.exists(args.labels):
        raise ValueError(f"Labels directory does not exist: {args.labels}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"total labels={len(args.labels)}")
    # Split dataset
    split_dataset(
        args.images,
        args.labels,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.dry_run,
    )

    # Update yaml configuration
    update_yaml_config(args.output, args.classes)


if __name__ == "__main__":
    main()
