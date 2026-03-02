from pathlib import Path
import numpy as np
import cv2
import json
import zipfile
import tempfile
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm
from typing import Iterator
from mtrain.utils import mkdir


DatasetIter = Iterator[tuple[str, np.ndarray, np.ndarray]]

TEMPLATE_LABELS = [
    {
        "name": "rock",
        "color": "#5AC8FA",
        "attributes": [],
        "type": "any",
        "sublabels": [],
    },
    {
        "name": "trash",
        "color": "#FF3B30",
        "attributes": [],
        "type": "any",
        "sublabels": [],
    },
    {
        "name": "colored_marker",
        "color": "#007AFF",
        "attributes": [],
        "type": "any",
        "sublabels": [],
    },
    {
        "name": "shop",
        "color": "#5AC8FA",
        "attributes": [],
        "type": "any",
        "sublabels": [],
    },
    {
        "name": "leaves",
        "color": "#AF52DE",
        "attributes": [],
        "type": "any",
        "sublabels": [],
    },
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def split_connected_components(mask, min_area=200):
    mask = (mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(mask)
    return [
        (labels == i).astype(np.uint8)
        for i in range(1, num)
        if (labels == i).sum() >= min_area
    ]


def _encode_binary_mask(binary_mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _save_image(arr: np.ndarray, path: Path):
    ext = path.suffix.lower()
    fmt = "JPEG" if ext in ("", ".jpg", ".jpeg") else ext.lstrip(".").upper()
    Image.fromarray(arr.astype(np.uint8)).save(path, format=fmt)


def _build_coco_categories(label_map: dict) -> tuple[dict, list]:
    sorted_vals = sorted(k for k in label_map if k != 0)
    # COCO category ids must be >= 1, so we remap mask pixel values to 1-based ids
    value_to_cat_id = {v: i + 1 for i, v in enumerate(sorted_vals)}
    categories = [
        {"id": value_to_cat_id[v], "name": label_map[v], "supercategory": ""}
        for v in sorted_vals
    ]
    return value_to_cat_id, categories


def _encode_image_annotations(
    mask_array: np.ndarray,
    img_id: int,
    ann_id: int,
    label_map: dict,
    value_to_cat_id: dict,
    image_name: str,
) -> tuple[list, int]:
    annotations = []
    for mask_val in np.unique(mask_array):
        if mask_val == 0:
            continue
        if mask_val not in label_map:
            print(
                f"  Warning: mask value {mask_val} in '{image_name}' not in label_map, skipping."
            )
            continue
        binary_mask = (mask_array == mask_val).astype(np.uint8)
        for comp in split_connected_components(binary_mask):
            rle = _encode_binary_mask(comp)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": value_to_cat_id[mask_val],
                    "segmentation": rle,
                    "area": float(mask_utils.area(rle)),
                    "bbox": mask_utils.toBbox(rle).tolist(),
                    "iscrowd": 1,  # 1 = RLE mask format (not polygon)
                }
            )
            ann_id += 1
    return annotations, ann_id


def _write_coco_json(path: Path, categories: list, images: list, annotations: list):
    coco = {
        "info": {"description": "Exported for CVAT upload"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    with path.open("w") as f:
        json.dump(coco, f, indent=2)


def _write_manifest(data_dir: Path, entries: list[dict]):
    # CVAT requires entries sorted lexicographically to match sorting_method in task.json
    sorted_entries = sorted(entries, key=lambda e: e["name"])
    with (data_dir / "manifest.jsonl").open("w") as f:
        f.write(json.dumps({"version": "1.1"}) + "\n")
        f.write(json.dumps({"type": "images"}) + "\n")
        for entry in sorted_entries:
            f.write(json.dumps(entry) + "\n")


def _build_task_json(
    task_name: str,
    labels: list,
    stop_frame: int,
    image_quality: int,
    chunk_size: int,
) -> dict:
    return {
        "name": task_name,
        "bug_tracker": "",
        "status": "annotation",
        "subset": "",
        "labels": labels,
        "version": "1.0",
        "data": {
            "chunk_size": chunk_size,
            "image_quality": image_quality,
            "start_frame": 0,
            "stop_frame": stop_frame,
            "storage_method": "cache",
            "sorting_method": "lexicographical",
            "chunk_type": "imageset",
            "deleted_frames": [],
            "storage": "local",
        },
        "jobs": [
            {
                "status": "annotation",
                "type": "annotation",
                "start_frame": 0,
                "stop_frame": stop_frame,
            }
        ],
    }


def _zip_dir(source_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source_dir))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_annotations_to_cvat(
    iterator: DatasetIter,
    label_map: dict,
    output_dir: str | Path,
    total: int | None = None,
) -> Path:
    """
    Export masks from the iterator to a COCO-format annotations.json for CVAT upload.

    Args:
        iterator:   Yields (image_name, image_rgb_array, mask_array) tuples.
                    image_name:  str, e.g. "frame001.jpg"
                    mask_array:  np.ndarray H x W, dtype int, pixel values = label_map keys
        label_map:  Dict mapping mask pixel values to label name strings.
                    e.g. {1: "cat", 2: "dog"}. Value 0 is background and skipped.
        output_dir: Directory to write annotations.json into.
        total:      Total number of items in the iterator, for tqdm progress bar.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    mkdir(output_dir)

    value_to_cat_id, categories = _build_coco_categories(label_map)
    coco_images, coco_annotations = [], []
    ann_id = 1

    for img_id, (image_name, _, mask_array) in enumerate(
        tqdm(iterator, total=total, desc="Exporting annotations", unit="img")
    ):
        h, w = mask_array.shape[:2]
        coco_images.append(
            {"id": img_id, "file_name": image_name, "height": h, "width": w}
        )
        new_anns, ann_id = _encode_image_annotations(
            mask_array, img_id, ann_id, label_map, value_to_cat_id, image_name
        )
        coco_annotations.extend(new_anns)

    annotations_path = output_dir / "annotations.json"
    _write_coco_json(annotations_path, categories, coco_images, coco_annotations)

    print(f"\nDone. {len(coco_images)} images, {len(coco_annotations)} annotations.")
    print(f"  annotations.json  →  {annotations_path.resolve()}")
    print(
        f"\nUpload: task → Actions → Upload Annotations → COCO 1.0 → {annotations_path.resolve()}"
    )

    return output_dir


def export_cvat_task_backup(
    iterator: DatasetIter,
    task_name: str,
    output_dir: str | Path,
    task_labels: list[dict] | None = None,
    image_quality: int = 70,
    chunk_size: int = 72,
    total: int | None = None,
) -> Path:
    """
    Export images from the iterator to a CVAT task backup zip.

    Produces output_dir/task.zip, which can be imported in CVAT to create a new task.
    Run export_annotations_to_cvat separately (with a fresh iterator) to generate
    the COCO annotations.json to upload into that task.

    Args:
        iterator:       Yields (image_name, image_rgb_array, mask_array) tuples.
        task_name:      Name shown for the task in CVAT.
        output_dir:     Directory to write task.zip into.
        task_labels:    Label definitions for task.json. Defaults to TEMPLATE_LABELS.
        image_quality:  JPEG quality stored in task.json (default 70).
        chunk_size:     Chunk size stored in task.json (default 72).
        total:          Total number of items in the iterator, for tqdm progress bar.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    mkdir(output_dir)
    if task_labels is None:
        task_labels = TEMPLATE_LABELS

    manifest_entries = []
    num_images = 0

    with tempfile.TemporaryDirectory(prefix="cvat_backup_") as tmp:
        staging_dir = Path(tmp)
        data_dir = mkdir(staging_dir / "data")

        for image_name, image_rgb, _ in tqdm(
            iterator, total=total, desc="Exporting images", unit="img"
        ):
            _save_image(image_rgb, data_dir / image_name)
            h, w = image_rgb.shape[:2]
            stem, ext = Path(image_name).stem, Path(image_name).suffix or ".jpg"
            manifest_entries.append(
                {"name": stem, "extension": ext, "width": w, "height": h}
            )
            num_images += 1

        stop_frame = max(num_images - 1, 0)
        _write_manifest(data_dir, manifest_entries)

        with (staging_dir / "task.json").open("w") as f:
            json.dump(
                _build_task_json(
                    task_name, task_labels, stop_frame, image_quality, chunk_size
                ),
                f,
                indent=4,
            )
        with (staging_dir / "annotations.json").open("w") as f:
            json.dump([{"version": 0, "tags": [], "shapes": [], "tracks": []}], f)

        task_zip_path = output_dir / "task.zip"
        _zip_dir(staging_dir, task_zip_path)

    print(f"\nDone. {num_images} images  →  {task_zip_path.resolve()}")
    print("\nUpload: CVAT → Create task from backup → upload task.zip")

    return output_dir
