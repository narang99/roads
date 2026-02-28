from itertools import batched, chain
from PIL import Image
import numpy as np
from mtrain.smallnet.tile import split_image_into_tiles


def _do_batch(batch, learner):
    images = [Image.fromarray(arr) for (arr, _) in batch]

    tdl = learner.dls.test_dl(test_items=images, with_labels=False)
    preds, _ = learner.get_preds(dl=tdl)
    masks = preds.argmax(dim=1)

    return [(mask.numpy(), pos) for mask, (_, pos) in zip(masks, batch)]


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


def strided_predict_unet_only_mask(
    img_arr,
    tile_size,
    learner,
    strides=None,
    bs=4,
):
    if strides is None:
        strides = []
    strideds = [(0, img_arr.copy())]
    for stride in strides:
        strideds.append((stride, img_arr[stride:, stride:]))
    fms = []
    for stride, strided in strideds:
        fm = predict_unet_only_mask(strided, tile_size, learner, bs)
        fms.append((stride, fm))
    H, W = img_arr.shape[:2]

    safe_fms = []
    for stride, mask in fms:
        safe_fms.append(_shift_mask(mask, H, W, stride, stride))
    full_mask = np.logical_or.reduce(safe_fms)
    return full_mask


def _shift_mask(mask, orig_h, orig_w, shift_y, shift_x):
    H, W = orig_h, orig_w
    aligned = np.zeros((H, W), dtype=mask.dtype)
    aligned[shift_y:, shift_x:] = mask
    return aligned