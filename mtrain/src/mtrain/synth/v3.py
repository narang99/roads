# use segmask to find places where we can keep garbage.
# use scaling, random blurring and random luminosity changes
import json
from pathlib import Path
from mtrain.data_prep.prep_crops import SingleCrop, PrepareCrops
from mtrain.seg import cityscapes
from mtrain.seg.cityscapes import CityScapesCls, get_cached_seg_former
from mtrain.synth.v1 import random_multiset_total
from mtrain import superpose
from mtrain.random import many_random_start_points
from mtrain import luminosity
from mtrain import scale
from mtrain import horizon
import numpy as np
import random
import cv2
from tqdm import tqdm
from mtrain.yolo.tfms import mk_yolo_label_from_numpy_coords
from typing import Optional
from dataclasses import dataclass



@dataclass
class ImageAndSeg:
    image: Path
    seg: Optional[Path]


def get_image_and_seg_for(image_root: Path, seg_root: Path) -> list[ImageAndSeg]:
    res = []
    for p in image_root.glob("*.jpg"):
        if (seg_root / f"{p.stem}.json").exists():
            seg = seg_root / f"{p.stem}.json"
        else:
            seg = None
        res.append(ImageAndSeg(image=p, seg=seg))
    return res


def generate(image_and_segs: list[ImageAndSeg], frag_root: Path, out_dir: Path, total=100):
    frags = [
        crp for crp in PrepareCrops(frag_root).get_all_existing_crops() if crp.proc
    ]
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    label_out = out_dir / "labels"
    label_out.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(total)):
        p = random.choice(image_and_segs)
        img_bgr, boxes = generate_single(p.image, frags, p.seg)
        labels = [
            mk_yolo_label_from_numpy_coords(
                0, b[0], b[1], b[2], b[3], img_bgr.shape[0], img_bgr.shape[1]
            )
            for b in boxes
        ]
        labels = "\n".join(labels)

        cv2.imwrite(img_out / f"{i}.png", img_bgr)
        with open(label_out / f"{i}.txt", "w") as f:
            f.write(labels)

    print("done")


def generate_single(
    p: Path, frags: list[SingleCrop], seg_path: Optional[Path], max_num_objects: int = 10
) -> tuple[np.ndarray, tuple]:
    frags = random_multiset_total(frags, max_total=max_num_objects)

    img_bgr = cv2.imread(p)
    if seg_path:
        with open(seg_path) as f:
            seg = np.asarray(json.load(f))
    else:
        seg = None

    mask = _get_placement_mask(p, seg)
    sps = [sp for sp in many_random_start_points(img_bgr, 30) if mask[*sp]]

    points = []
    for frag, sp in zip(frags, sps):
        (s0, s1), (e0, e1), _ = _copy_at(img_bgr, frag, sp[0], sp[1], seg)
        points.append((s0, e0, s1, e1))
    return img_bgr, points


def _get_placement_mask(img_path: Path, seg):
    pred = seg
    if pred is None:
        img = cv2.imread(img_path)
        model = get_cached_seg_former()
        pred = model.predict_bgr_image(img)

    # i might use other labels later
    clazzes = [
        CityScapesCls.SIDEWALK,
        CityScapesCls.ROAD,
        CityScapesCls.TERRAIN,
    ]
    masks = [cityscapes.get_mask(pred, clazz) for clazz in clazzes]
    mask = np.logical_or.reduce(masks)
    return mask


def _copy_at(img_bgr: np.ndarray, frag: SingleCrop, r: int, c: int, seg: Optional[np.ndarray]):
    meta = frag.parse_meta()
    frag_bgr = cv2.imread(frag.proc)

    tparams = _calculate_tfms(img_bgr, frag_bgr, meta, r, c, seg)

    frag_bgr = _scale_fragment(frag_bgr, tparams["scale"])
    if tparams["luminosity"] != -1:
        frag_bgr = _scale_luminosity(
            img_bgr, frag.raw, frag_bgr, r, c, tparams["luminosity"]
        )

    s0, e0, s1, e1 = superpose.direct_copy(img_bgr, frag_bgr, r, c)

    if tparams["blur"] > 0:
        kernel = (tparams["blur"], tparams["blur"])
        blurred = cv2.blur(img_bgr, kernel)
        img_bgr[s0:e0, s1:e1] = blurred[s0:e0, s1:e1]

    return (s0, s1), (e0, e1), tparams


def _scale_fragment(frag_bgr, factor):
    dummy_mask = frag_bgr == 0
    frag_bgr, _ = scale.scale_fragment_with_mask(frag_bgr, dummy_mask, factor)
    return frag_bgr


def _calculate_tfms(img_bgr, frag_bgr, meta, r, c, seg):
    scale_factor = _calculate_scale(img_bgr, meta, r, seg)
    blur_k = random.randint(0, 2)
    return {
        "scale": scale_factor,
        "blur": blur_k,
        "luminosity": _get_lum_scale(),
    }


def _get_lum_scale():
    lum_scale = random.uniform(1, 1.5)
    do_lum = random.randint(0, 1)
    return lum_scale if do_lum else -1


def _scale_luminosity(img_bgr, raw_path, frag_bgr, r, c, lum_scale):
    params = _calculate_lum_params(img_bgr, raw_path, r, c, lum_scale)
    return luminosity.do_std_matching_without_mask(frag_bgr, params)


def _calculate_lum_params(img_bgr, raw_path, r, c, lum_scale):
    raw_frag = cv2.imread(raw_path)
    roi = img_bgr[r : r + raw_frag.shape[0], c : c + raw_frag.shape[1]]
    std_params = luminosity.get_std_matcher_params(raw_frag, roi)
    std_params.scale_target_mean(lum_scale)
    return std_params


def _calculate_scale(img_bgr, meta, r, seg: Optional[np.ndarray]):
    fbox = meta["bounding_box"]

    hrz_curr = horizon.highest_point_in_road_mask(img_bgr, cityscapes.get_mask(seg, CityScapesCls.ROAD))

    # hrz_curr = horizon.using_road_mask(img_bgr)
    hrz_frag = meta["original"]["horizon"]
    factor = scale.calculate_perspective_scale_factor(
        fbox["r"], r, hrz_frag, hrz_curr, meta["original"]["r_len"], img_bgr.shape[0]
    )
    return factor
