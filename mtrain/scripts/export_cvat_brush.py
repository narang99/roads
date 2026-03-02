"""
Convert image + mask arrays to a CVAT-uploadable COCO dataset.

Usage:
    from masks_to_cvat import export_to_cvat

    label_map = {
        1: "cat",
        2: "dog",
    }

    def my_data():
        yield "cat001.jpg", image_array, mask_array  # mask values match label_map keys
        yield "dog042.jpg", image_array2, mask_array2

    export_to_cvat(my_data(), label_map, output_dir="./cvat_upload")

Output structure:
    cvat_upload/
    ├── images/
    │   ├── cat001.jpg
    │   └── dog042.jpg
    └── annotations.json       <-- upload this via CVAT > Actions > Upload Annotations > COCO 1.0

Notes:
    - mask_array: H x W numpy array where each pixel value is a key in label_map (0 = background, ignored)
    - If multiple disconnected regions share the same label in one image, each is a separate annotation instance
    - image_rgb_array: H x W x 3 uint8 numpy array
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import cv2
from tqdm import tqdm


def split_connected_components(mask, min_area=200):
    mask = (mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(mask)
    components = []
    for i in range(1, num):
        comp = (labels == i).astype(np.uint8)
        if comp.sum() >= min_area:
            components.append(comp)
    return components


def _encode_binary_mask(binary_mask: np.ndarray) -> dict:
    """Encode a binary H x W uint8 mask to COCO RLE format."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")  # make JSON serializable
    return rle


def export_to_cvat(
    iterator,
    label_map: dict,
    output_dir: str | Path,
    images_subdir: str = "images",
    total: int | None = None,
) -> Path:
    """
    Export images and masks to a CVAT-compatible COCO dataset.

    Args:
        iterator:       Yields (image_name, image_rgb_array, mask_array) tuples.
                        image_name:      str, e.g. "frame001.jpg"
                        image_rgb_array: np.ndarray H x W x 3, dtype uint8
                        mask_array:      np.ndarray H x W, dtype int, pixel values = label_map keys
        label_map:      Dict mapping mask pixel values to label name strings.
                        e.g. {1: "cat", 2: "dog"}
                        Value 0 is always treated as background and skipped.
        output_dir:     Directory to write images/ and annotations.json into.
        images_subdir:  Name of the subdirectory for images (default: "images").
        total:          Total number of items in the iterator, for tqdm progress bar.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / images_subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build category list from label_map, sorted by id for determinism
    # COCO category ids must be >= 1
    # We remap label_map keys to stable 1-based category ids
    sorted_label_values = sorted(k for k in label_map.keys() if k != 0)
    value_to_cat_id = {v: i + 1 for i, v in enumerate(sorted_label_values)}
    categories = [
        {"id": value_to_cat_id[v], "name": label_map[v], "supercategory": ""}
        for v in sorted_label_values
    ]

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img_id, (image_name, image_rgb_array, mask_array) in enumerate(
        tqdm(iterator, total=total, desc="Exporting", unit="img")
    ):
        # --- save image ---
        img_pil = Image.fromarray(image_rgb_array.astype(np.uint8))
        img_path = images_dir / image_name
        # preserve original extension; default to jpg if none
        ext = Path(image_name).suffix.lower()
        fmt = "JPEG" if ext in ("", ".jpg", ".jpeg") else ext.lstrip(".").upper()
        img_pil.save(img_path, format=fmt)

        h, w = mask_array.shape[:2]

        coco_images.append(
            {
                "id": img_id,
                "file_name": image_name,
                "height": h,
                "width": w,
            }
        )

        # --- encode each label instance as a separate annotation ---
        unique_values = np.unique(mask_array)
        for mask_val in unique_values:
            if mask_val == 0:
                continue  # background

            if mask_val not in label_map:
                print(
                    f"  Warning: mask value {mask_val} in '{image_name}' not in label_map, skipping."
                )
                continue

            cat_id = value_to_cat_id[mask_val]
            binary_mask = (mask_array == mask_val).astype(np.uint8)

            components = split_connected_components(binary_mask)

            for comp in components:
                rle = _encode_binary_mask(comp)
                area = float(mask_utils.area(rle))
                bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]

                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 1,  # 1 = RLE mask format (not polygon)
                    }
                )
                ann_id += 1


    coco_output = {
        "info": {"description": "Exported for CVAT upload"},
        "licenses": [],
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    annotations_path = output_dir / "annotations.json"
    with annotations_path.open("w") as f:
        json.dump(coco_output, f, indent=2)

    print(f"\nDone.")
    print(f"  {len(coco_images)} images  →  {images_dir}")
    print(f"  {len(coco_annotations)} annotations  →  {annotations_path}")
    print(f"\nUpload steps:")
    print(f"  1. Create a CVAT task and upload all files from: {images_dir}")
    print(f"  2. Open the task → Actions → Upload Annotations → COCO 1.0")
    print(f"  3. Upload: {annotations_path.resolve()}")

    return output_dir  # Path object


# ---------------------------------------------------------------------------
# Example / quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    def fake_data():
        """Synthetic example: 2 images, each with 2 labeled regions."""
        rng = np.random.default_rng(0)
        for i in range(2):
            img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[50:200, 50:200] = 1  # cat region
            mask[300:420, 400:580] = 2  # dog region
            yield f"image_{i:03d}.jpg", img, mask

    label_map = {1: "cat", 2: "dog"}
    export_to_cvat(fake_data(), label_map, output_dir="/tmp/cvat_test")
