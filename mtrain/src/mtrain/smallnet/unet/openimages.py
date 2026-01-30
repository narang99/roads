"""
OpenImages Dataset Extractor
Extracts images and masks from OpenImages dataset structure
Optimized for memory usage with minimal filesystem access
"""

import os
import csv
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm
import fiftyone
from mtrain.cache import DEFAULT_SYNTH_CACHE
import json

# openimages dataset explore and extract
# data structure
# |-- data
# |   |-- 00a36f96e31731c4.jpg
# |   |-- 02deba0102b5ce2a.jpg
# |   |-- 031244297d177089.jpg
# |   |-- 03abc39ad2c14097.jpg
# |   |-- 049720d842de2d3e.jpg
# |   `-- ... (95 more files)
# |-- labels
# |   |-- masks
# |   |   |-- 0
# |   |   |-- 1
# |   |   |-- 2
# |   |   |-- 3
# |   |   |-- 4
# |   |   `-- 5
# |   |-- detections.csv
# |   |-- points.csv
# |   `-- segmentations.csv
# `-- metadata
#     |-- classes.csv
#     |-- hierarchy.json
#     |-- image_ids.csv
#     |-- point_classes.csv
#     `-- segmentation_classes.csv


# algorithm
# go through segmentations.csv
# each image can have multiple masks
# you have to find that mask and & it with this mask for our purposes
#!/usr/bin/env python3
class OpenImagesExtractor:
    def __init__(self, dataset_root, output_dir):
        """
        Initialize the extractor

        Args:
            dataset_root: Root directory of OpenImages dataset
            output_dir: Output directory for organized data
        """
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)

        # Paths
        self.data_dir = self.dataset_root / "data"
        self.labels_dir = self.dataset_root / "labels"
        self.masks_dir = self.labels_dir / "masks"
        self.segmentations_csv = self.labels_dir / "segmentations.csv"

        # Output paths
        self.output_images_dir = self.output_dir / "images"
        self.output_masks_dir = self.output_dir / "masks"

        # Create output directories
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_masks_dir.mkdir(parents=True, exist_ok=True)

        # In-memory metadata
        self.image_to_masks = None
        self.mask_path_cache = {}

    def build_mask_path_cache(self):
        """Build complete cache of all mask file locations in memory"""
        print("Building mask location cache...")

        mask_path_cache = {
            f.name: f for f in self.masks_dir.rglob("*.png") if f.is_file()
        }

        print(f"Cached {len(mask_path_cache)} mask file locations")
        return mask_path_cache

    def load_all_metadata(self):
        """Load all metadata into memory"""
        print("Loading all metadata into memory...")

        # Load segmentations and group by image_id
        print("  - Loading segmentations.csv...")
        image_to_masks = defaultdict(list)

        with open(self.segmentations_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["ImageID"]
                mask_path = row["MaskPath"]
                image_to_masks[image_id].append(mask_path)

        print(f"  - Found {len(image_to_masks)} unique images with masks")

        # Build mask path cache
        mask_path_cache = self.build_mask_path_cache()

        return image_to_masks, mask_path_cache

    def combine_masks(self, mask_paths):
        """
        Combine multiple masks using OR operation
        Returns combined binary mask
        """
        combined_mask = None

        for mask_path in mask_paths:
            # Load mask
            mask = np.array(Image.open(mask_path))

            # Convert to binary (0 or 1)
            binary_mask = (mask > 0).astype(np.uint8)

            if combined_mask is None:
                combined_mask = binary_mask
            else:
                # OR operation: any pixel that is 1 in any mask becomes 1
                combined_mask = np.logical_or(combined_mask, binary_mask).astype(
                    np.uint8
                )

        return combined_mask

    def process_image(self, image_id, mask_filenames):
        """Process a single image and its masks"""
        try:
            # Source image path
            src_image_path = self.data_dir / f"{image_id}.jpg"

            if not src_image_path.exists():
                print(f"Warning: Image {image_id}.jpg not found, skipping")
                return False

            # Copy image to output directory as JPEG
            dst_image_path = self.output_images_dir / f"{image_id}.jpeg"
            shutil.copy2(src_image_path, dst_image_path)

            # Get mask paths from cache
            mask_paths = []
            for mask_filename in mask_filenames:
                if mask_filename in self.mask_path_cache:
                    mask_paths.append(self.mask_path_cache[mask_filename])
                else:
                    print(f"Warning: Mask {mask_filename} not found in cache")

            if not mask_paths:
                print(f"Warning: No masks found for image {image_id}")
                return False

            # Combine masks if multiple
            if len(mask_paths) == 1:
                # Just load the single mask
                src_mask = np.array(Image.open(mask_paths[0]))
                combined_mask = (src_mask > 0).astype(np.uint8)
            else:
                # Combine multiple masks
                combined_mask = self.combine_masks(mask_paths)

            # Save combined mask with same base name as image
            # Convert to binary image (0 or 255 for visibility)
            mask_image = Image.fromarray(combined_mask * 255)
            dst_mask_path = self.output_masks_dir / f"{image_id}.png"
            mask_image.save(dst_mask_path)

            return True

        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            return False

    def extract(self):
        """Main extraction process"""
        # Load all metadata into memory
        self.image_to_masks, self.mask_path_cache = self.load_all_metadata()

        # Process all images
        images = list(self.data_dir.rglob("*.jpg"))
        total_images = len(images)
        print(f"\nProcessing {total_images} images...")

        success_count = 0
        for image_path in images:
            image_id = image_path.stem
            mask_filenames = self.image_to_masks.get(image_id)
            if not mask_filenames:
                print(f"no mask file: {image_id}")
            if self.process_image(image_id, mask_filenames):
                success_count += 1

        print(f"\nExtraction complete!")
        print(f"Images saved to: {self.output_images_dir}")
        print(f"Masks saved to: {self.output_masks_dir}")
        print(f"Successfully processed: {success_count}/{total_images} images")


def extract_images(dataset_root, output_dir):
    extracter = OpenImagesExtractor(dataset_root, output_dir)
    extracter.extract()


def download_and_extract(n_samples_by_classes, output_dir):
    n_samples_by_classes = {
        n: list(sorted(classes)) for n, classes in n_samples_by_classes.items()
    }
    n_samples_by_classes_str = json.dumps(n_samples_by_classes, sort_keys=True)
    _download_and_extract_cacheable(
        n_samples_by_classes_str=n_samples_by_classes_str, output_dir=output_dir
    )


@DEFAULT_SYNTH_CACHE.decorator(
    output_arg="output_dir", key_args=["n_samples_by_classes_str"]
)
def _download_and_extract_cacheable(n_samples_by_classes_str, output_dir):
    # fresh download cuz we dont know what labels are what
    fiftyone_data_loc = Path.home() / "fiftyone" / "open-images-v7"
    if fiftyone_data_loc.exists():
        shutil.rmtree(fiftyone_data_loc)

    n_samples_by_classes = json.loads(n_samples_by_classes_str)
    for n_sample, classes in n_samples_by_classes.items():
        print(n_sample, classes)
        fiftyone.zoo.load_zoo_dataset(
            "open-images-v7",
            label_types=["segmentations"],
            classes=classes,
            max_samples=int(n_sample),
            shuffle=True,
            splits=["validation"],
        )

    extract_images(fiftyone_data_loc / "validation", output_dir)
