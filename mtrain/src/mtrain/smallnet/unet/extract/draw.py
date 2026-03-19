import random
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math

def show(crops, figsize=None, ncols=2):
    crops = list(crops)
    rows = math.ceil(len(crops) / ncols)
    if figsize is None:
        figsize = (10 * rows, 10 * rows)
        print("figsize", figsize)
    _, axs = plt.subplots(rows, ncols, figsize=figsize)
    axs = axs.flatten()
    for i, c in enumerate(crops):
        axs[i].imshow(c)
    plt.tight_layout()
    plt.show()

def get_batch_from_extracted_dataset(d, n=8, img_id=None, filterer=None):
    ims, msks = d / "images", d / "masks"
    res = []
    it = ims.glob("*")
    if img_id is not None:
        it = _filter_only_img_id_in_path(it, img_id)
    if filterer is not None:
        it = filter(filterer, it)
    ims = list(it)
    random.shuffle(ims)
    for im in ims[:n]:
        msk = msks / f"{im.stem}.png"
        res.append((im, msk))
    return res

def show_extracted_dataset(d, n=8, img_id=None, mode=None, filterer=None):
    res = get_batch_from_extracted_dataset(d, n, img_id, filterer)
    _show_images_and_masks(n, res)


def _show_images_and_masks(num_to_show, res):
    num = min(num_to_show, len(res))
    print(f"results: {num}")
    _, ax = plt.subplots(num, 2, figsize=(10, 3 * num))
    if num == 1:
        r0 = np.array(Image.open(res[0][0]))
        r1 = np.array(Image.open(res[0][1]))
        ax[0].imshow(r0)
        ax[1].imshow(r1)
    else:
        for i in range(num):
            r0 = np.array(Image.open(res[i][0]))
            r1 = np.array(Image.open(res[i][1]))
            ax[i][0].imshow(r0)
            ax[i][1].imshow(r1)
    plt.tight_layout()
    plt.show()


def _filter_only_img_id_in_path(it, img_id):
    def _same_as_img_id(im_path):
        try:
            return int(Path(im_path).stem) == int(img_id)
        except:
            return False

    it = filter(_same_as_img_id, it)
    return it


def overlay_mask_on_img(img_arr, mask, alpha=0.4, color=[255,0,0]):
    res = img_arr.copy()
    res[mask] = (
        (1 - alpha) * res[mask].astype(np.float32) +
        alpha * np.array(color)
    ).astype(np.uint8)
    return res
