import cv2
from itertools import batched, chain
from PIL import Image
from tqdm import tqdm
import numpy as np
from mtrain.smallnet.tile import split_image_into_tiles
from mtrain.seg.cityscapes import cached_predict, get_mask_with_labels, CityScapesCls


def predict_unet_with_neg_mask(img_path, sz, learner, alpha=0.4):
    img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = predict_unet_only_mask(img_arr, sz, learner)
    pred = cached_predict(img_path)
    neg_mask = get_mask_with_labels(
        pred, [CityScapesCls.ROAD, CityScapesCls.SIDEWALK, CityScapesCls.TERRAIN]
    )
    print(mask.shape, neg_mask.shape, mask.dtype, neg_mask.dtype)

    mask &= neg_mask
    return overlay_mask_on_img(img_arr, mask, alpha), mask


def predict_unet(img_path, sz, learner, alpha=0.4):
    img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    arr_and_coord = split_image_into_tiles(img_arr, sz)
    mask_and_coord = [
        (learner.predict(arr)[0].numpy(), (y, x))
        for (arr, (y, x)) in tqdm(arr_and_coord)
    ]
    res = img_arr.copy()
    H, W = res.shape[:2]
    for mask, (y, x) in tqdm(mask_and_coord):
        ny, nx = min(y + sz, H), min(x + sz, W)
        roi = res[y:ny, x:nx]
        mask = mask[: ny - y, : nx - x]
        mask = mask.astype(bool)

        roi[mask] = (
            (1 - alpha) * roi[mask].astype(np.float32) + alpha * np.array([255, 0, 0])
        ).astype(np.uint8)

    return res


def overlay_mask_on_img(img_arr, mask, alpha=0.4):
    if mask.dtype != "bool":
        print(
            f"WARN: mask dtype is {mask.dtype}. This can have adverse performance, please pass it astype(bool)"
        )
    res = img_arr.copy()
    res[mask] = (
        (1 - alpha) * res[mask].astype(np.float32) + alpha * np.array([255, 0, 0])
    ).astype(np.uint8)
    return res

def _do_batch(batch, learner):
    images = [Image.fromarray(arr) for (arr, _) in batch]

    tdl = learner.dls.test_dl(test_items=images, with_labels=False)
    preds, _ = learner.get_preds(dl=tdl)
    masks = preds.argmax(dim=1)

    return [(mask.numpy(), pos) for mask, (_, pos) in zip(masks, batch)]


def predict_unet_only_mask(img_arr, sz, learner, bs):
    arr_and_coord = split_image_into_tiles(img_arr, sz)
    mask_and_coord = list(chain.from_iterable((
        _do_batch(batch, learner) for batch in tqdm(batched(arr_and_coord, bs))
    )))
    # mask_and_coord = [
    #     (learner.predict(arr)[0].numpy(), (y, x))
    #     for (arr, (y, x)) in tqdm(arr_and_coord)
    # ]
    H, W = img_arr.shape[:2]
    res = np.zeros((H, W), dtype=np.bool)
    for mask, (y, x) in tqdm(mask_and_coord):
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