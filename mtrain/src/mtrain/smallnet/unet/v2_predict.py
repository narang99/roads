import cv2
from PIL import Image
from itertools import batched, chain
from tqdm import tqdm
from mtrain.smallnet.tile import split_image_into_tiles
from pathlib import Path
import numpy as np


def _shift_mask(mask, orig_h, orig_w, shift_y, shift_x):
    H, W = orig_h, orig_w
    aligned = np.zeros((H, W), dtype=mask.dtype)
    aligned[shift_y:, shift_x:] = mask
    return aligned


def predict_unet(img_path, sz, learner, alpha=0.4, strides=None):
    img_path = Path(img_path)
    if strides is None:
        strides = []
    img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    strideds = [(0, img_arr)]
    for stride in strides:
        strideds.append((stride, img_arr[stride:, stride:]))
    fms = []
    for stride, strided in strideds:
        fm = _predict_unet_img_arr(strided, sz, learner)
        fms.append((stride, fm))
    H, W = img_arr.shape[:2]
    safe_fms = []
    for stride, mask in fms:
        safe_fms.append(_shift_mask(mask, H, W, stride, stride))
    full_mask = np.logical_or.reduce(safe_fms)
    res = img_arr.copy()
    res[full_mask] = (
        (1 - alpha) * res[full_mask].astype(np.float32) + alpha * np.array([255, 0, 0])
    ).astype(np.uint8)
    return res, full_mask


def _predict_unet_img_arr(img_arr, sz, learner):
    arr_and_coord = split_image_into_tiles(img_arr, sz)
    mask_and_coord = _run_predict(arr_and_coord, learner, 8)
    res = img_arr.copy()
    H, W = res.shape[:2]
    full_mask = np.zeros((H, W)).astype(bool)
    for mask, (y, x) in tqdm(mask_and_coord):
        ny, nx = min(y + sz, H), min(x + sz, W)
        roi = full_mask[y:ny, x:nx]
        mask = mask[: ny - y, : nx - x]
        mask = mask.astype(bool)
        roi[mask] = True

    return full_mask


def _do_batch(batch, learner):
    images = [Image.fromarray(arr) for (arr, _) in batch]

    tdl = learner.dls.test_dl(test_items=images, with_labels=False)
    preds, _ = learner.get_preds(dl=tdl)
    masks = preds.argmax(dim=1)

    return [(mask.numpy(), pos) for mask, (_, pos) in zip(masks, batch)]


def _run_predict(mask_and_coord, learner, bs=8):
    batched_results = (
        _do_batch(batch, learner) for batch in tqdm(batched(mask_and_coord, bs))
    )
    return list(chain.from_iterable(batched_results))
