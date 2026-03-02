import json
from PIL import Image
import cv2
from uuid import uuid4
from pathlib import Path
import numpy as np
from label_studio_converter import brush
from mtrain.utils import json_to_content


def get_image_path(content) -> Path:
    ext_p = content["data"]["image"].split("?d=")[1]
    ext_p = f"/{ext_p}"
    return Path(ext_p)


def extract_single_id_json(iid, in_json_path_or_content, out_json_path):
    content = json_to_content(in_json_path_or_content)
    # if the content is list, assume the full file
    # else assume the small half file

    if isinstance(content, list):
        res = [c for c in content if c["id"] == iid]
    else:
        res = [in_json_path_or_content]
    if len(res) == 0:
        raise Exception(
            f"no result found for id={iid} in JSON={in_json_path_or_content}"
        )
    if len(res) == 1:
        with open(out_json_path, "w") as f:
            json.dump(res[0], f)
        return res[0]
    else:
        raise Exception(
            f"corrupted data: {in_json_path_or_content}, found multiple results for image_id={iid} data={res}"
        )



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