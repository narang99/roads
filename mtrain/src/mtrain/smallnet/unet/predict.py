import cv2
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

    mask &= (neg_mask)
    return overlay_mask_on_img(img_arr, mask, alpha), mask


def predict_unet(img_path, sz, learner, alpha=0.4):
    img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    arr_and_coord = split_image_into_tiles(img_arr, sz)
    mask_and_coord = [
        (learner.predict(arr)[0].numpy(), (y, x)) for (arr, (y, x)) in tqdm(arr_and_coord)
    ]
    res = img_arr.copy()
    H, W = res.shape[:2]
    for mask, (y, x) in tqdm(mask_and_coord):
        ny, nx = min(y+sz, H), min(x+sz, W)
        roi = res[y:ny, x:nx]
        mask = mask[:ny-y, :nx-x]
        mask = mask.astype(bool)

        roi[mask] = (
            (1 - alpha) * roi[mask].astype(np.float32) +
            alpha * np.array([255, 0, 0])
        ).astype(np.uint8)

    return res


def overlay_mask_on_img(img_arr, mask, alpha=0.4):
    res = img_arr.copy()
    res[mask] = (
        (1 - alpha) * res[mask].astype(np.float32) +
        alpha * np.array([255, 0, 0])
    ).astype(np.uint8)
    return res

def predict_unet_only_mask(img_arr, sz, learner):
    arr_and_coord = split_image_into_tiles(img_arr, sz)
    mask_and_coord = [
        (learner.predict(arr)[0].numpy(), (y, x)) for (arr, (y, x)) in tqdm(arr_and_coord)
    ]
    H, W = img_arr.shape[:2]
    res = np.zeros((H, W), dtype=np.bool)
    for mask, (y, x) in tqdm(mask_and_coord):
        ny, nx = min(y+sz, H), min(x+sz, W)
        res[y:ny, x:nx] = mask[:ny-y, :nx-x].astype(bool)

    return res