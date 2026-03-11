from contextlib import contextmanager
from fastai.vision.all import default_device
import torch
from dataclasses import dataclass
import time
from collections import defaultdict
from itertools import batched, chain
from PIL import Image
import numpy as np
from mtrain.smallnet.tile import split_image_into_tiles


@contextmanager
def timed(label="Block"):
    start_time = time.perf_counter() # Use perf_counter for precision
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"'{label}' time: {duration:.3f} seconds")

@dataclass
class TileTag:
    img_idx: int
    stride: int
    pos: tuple  # (y, x)


def _do_batch(batch, learner):
    images = [Image.fromarray(arr) for (arr, _) in batch]

    with torch.no_grad():
        tdl = learner.dls.test_dl(test_items=images, with_labels=False, num_workers=4, device=default_device())
        preds, _ = learner.get_preds(dl=tdl)
        masks = preds.argmax(dim=1).cpu().numpy()

    return [(mask, tag) for mask, (_, tag) in zip(masks, batch)]


def predict_unet_only_mask(img_arr, sz, learner, bs):
    arr_and_coord = split_image_into_tiles(img_arr, sz)
    mask_and_coord = list(chain.from_iterable((
        _do_batch(batch, learner) for batch in batched(arr_and_coord, bs)
    )))
    H, W = img_arr.shape[:2]
    res = np.zeros((H, W), dtype=np.bool)
    for mask, (y, x) in mask_and_coord:
        ny, nx = min(y + sz, H), min(x + sz, W)
        res[y:ny, x:nx] = mask[: ny - y, : nx - x].astype(bool)

    return res


def _strided_tiles(img_arr, tile_size, strides, img_idx):
    all_strides = [(s, img_arr[s:, s:]) for s in strides]
    for stride, strided in all_strides:
        for arr, pos in split_image_into_tiles(strided, tile_size):
            yield arr, TileTag(img_idx=img_idx, stride=stride, pos=pos)


def _recombine_strided(tagged_masks, img_arr, tile_size, all_strides):
    H, W = img_arr.shape[:2]

    strided_masks = defaultdict(list)
    for mask, tag in tagged_masks:
        strided_masks[tag.stride].append((mask, tag.pos))

    safe_fms = []
    for stride in all_strides:
        sh, sw = img_arr[stride:, stride:].shape[:2]
        strided_mask = np.zeros((sh, sw), dtype=np.bool_)
        for mask, (y, x) in strided_masks.get(stride, []):
            ny, nx = min(y + tile_size, sh), min(x + tile_size, sw)
            strided_mask[y:ny, x:nx] = mask[: ny - y, : nx - x].astype(bool)
        safe_fms.append(_shift_mask(strided_mask, H, W, stride, stride))

    return np.logical_or.reduce(safe_fms)


def strided_predict_unet_only_mask(
    img_arrs,
    tile_size,
    learner,
    strides=None,
    bs=128,
):
    if strides is None:
        strides = []
    strides = [0] + strides

    all_tiles = list(chain.from_iterable(
        _strided_tiles(img_arr, tile_size, strides, img_idx)
        for img_idx, img_arr in enumerate(img_arrs)
    ))

    all_results = list(chain.from_iterable(
        _do_batch(batch, learner) for batch in batched(all_tiles, bs)
    ))

    img_by_tagged_masks = defaultdict(list)
    for mask, tag in all_results:
        img_by_tagged_masks[tag.img_idx].append((mask, tag))

    combined  = [
        _recombine_strided(img_by_tagged_masks[i], img_arr, tile_size, strides)
        for i, img_arr in enumerate(img_arrs)
    ]

    return combined


def _shift_mask(mask, orig_h, orig_w, shift_y, shift_x):
    H, W = orig_h, orig_w
    aligned = np.zeros((H, W), dtype=mask.dtype)
    aligned[shift_y:, shift_x:] = mask
    return aligned
