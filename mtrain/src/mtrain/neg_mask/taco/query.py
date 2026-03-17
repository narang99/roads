from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pycocotools.coco import COCO

from .extract import load_image, fixed_cat_id


def get_total_classes(coco: COCO) -> int:
    """Returns the total number of categories in the TACO dataset."""
    return len(coco.getCatIds())


def get_cat_id_to_cat(coco: COCO) -> tuple[dict, dict]:
    cat_ids = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {fixed_cat_id(c["id"]): c["name"] for c in cat_ids}
    cat_id_to_sup = {fixed_cat_id(c["id"]): c["supercategory"] for c in cat_ids}

    return cat_id_to_name, cat_id_to_sup


def print_all_categories(coco: COCO) -> None:
    """Prints all categories and their supercategories."""
    cats = coco.loadCats(coco.getCatIds())
    for cat in cats:
        print(f"{fixed_cat_id(cat['id']):>3}  {cat['supercategory']} > {cat['name']}")


def show_image_with_boxes(coco: COCO, taco_dir: Path, img_info: dict) -> dict:
    """
    Given a taco_dir and an image info dict (from get_images_with_multiple_classes),
    prints the number of classes present and displays the image with labeled bounding boxes.
    """
    img_array = load_image(taco_dir / img_info["file_name"])

    ann_ids = coco.getAnnIds(imgIds=img_info["id"])
    anns = coco.loadAnns(ann_ids)

    cat_ids_present = {ann["category_id"] for ann in anns}
    cats = {c["id"]: c for c in coco.loadCats(list(cat_ids_present))}
    print(
        f"Classes in image ({len(cat_ids_present)}): {', '.join(c['name'] for c in cats.values())}"
    )
    _, cat_id_to_sup = get_cat_id_to_cat(coco)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_array)
    ax.axis("off")

    cmap = plt.cm.get_cmap("tab20", len(cat_ids_present))
    cat_color = {cat_id: cmap(i) for i, cat_id in enumerate(cat_ids_present)}

    cat_name_to_ann_ids = defaultdict(list)
    for ann in anns:
        x, y, w, h = ann["bbox"]
        color = cat_color[ann["category_id"]]
        sup_name = cat_id_to_sup[fixed_cat_id(ann["category_id"])]
        cat_name_to_ann_ids[sup_name].append(ann["id"])

        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y - 4,
            f"{sup_name}:{ann['id']}",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor=color, edgecolor="none", pad=2, alpha=0.8),
        )

    plt.tight_layout()
    plt.show()

    return cat_name_to_ann_ids


def get_images_with_multiple_classes(coco: COCO) -> list[dict]:
    """
    Returns a list of image info dicts for images that contain annotations
    from more than one category.
    """
    img_to_cats: dict[int, set[int]] = {}
    for ann in coco.dataset["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        img_to_cats.setdefault(img_id, set()).add(cat_id)

    multi_class_img_ids = [
        img_id for img_id, cats in img_to_cats.items() if len(cats) > 1
    ]
    return coco.loadImgs(multi_class_img_ids)


def get_cat_ids_of_image_id(coco: COCO, img_id: int) -> list[int]:
    annIds = coco.getAnnIds(imgIds=img_id)
    anns_sel = coco.loadAnns(annIds)
    cat_ids = [fixed_cat_id(a["category_id"]) for a in anns_sel]
    return cat_ids

