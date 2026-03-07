from mtrain.neg_mask.crops import Bbox, bbox_only_mask, padded_crop

import json
from PIL import Image
from pathlib import Path


def save_crop_level(
    image,
    mask,
    crop_pad,
    out_dir: Path,
    name: str,
    bbox_idx: int,
    bbox: Bbox,
    label: int,
    label_by_folder: dict,
    pred_label=None,
    pred_prob=None,
    source_dir=None,
):
    folder = label_by_folder[label]
    sample_dir = out_dir / "crop_level" / folder / f"{name}_{bbox_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    crop_img, y1c, x1c = padded_crop(image, bbox, crop_pad)
    crop_mask = bbox_only_mask(mask, bbox, crop_pad)

    Image.fromarray(crop_img).save(sample_dir / "image.jpg")
    Image.fromarray(crop_mask).save(sample_dir / "mask.png")
    meta: dict = {"crop_origin": {"x": int(x1c), "y": int(y1c)}}
    if pred_label is not None and pred_prob is not None:
        meta["model_pred"] = {"label": pred_label, "prob": round(pred_prob, 4)}
        if pred_label != label:
            meta["model_disagreement"] = True

    (sample_dir / "meta.json").write_text(json.dumps(meta))
    if source_dir is not None:
        symlink = sample_dir / "source_dir"
        if not symlink.exists() and not symlink.is_symlink():
            symlink.symlink_to(source_dir)
