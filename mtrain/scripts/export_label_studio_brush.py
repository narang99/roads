from uuid import uuid4
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from label_studio_converter import brush


def split_connected_components(mask, min_area=200):
    # mask: binary (0/1)
    mask = (mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(mask)
    components = []
    for i in range(1, num):
        comp = (labels == i).astype(np.uint8)
        if comp.sum() >= min_area:
            components.append(comp)
    return components


# make a single annotation only right now (we do clusters later if needed)
def _make_json_part(image_path, masks, label, tmp):
    anns = []
    for mask in masks:
        dest = tmp / f"{uuid4()}.png"
        Image.fromarray(mask.astype(bool)).save(dest)
        ann = brush.image2annotation(
            dest, label, "brush", "image", model_version="v1", score=0.9
        )
        anns.append(ann["result"][0])
    path = str(Path(image_path).resolve())[1:]
    return {
        # "predictions": anns,
        "predictions": [
            {
                "model_version": "v1",
                "score": 0.88,
                "result": anns,
            }
        ],
        "data": {
            "image": f"/data/local-files/?d={path}",
        },
    }


def make_json_part(image_path, mask, label, tmp):
    masks = split_connected_components(mask)
    return _make_json_part(image_path, masks, label, tmp)