import random
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def show_extracted_dataset(d, n=8, img_id=None, mode=None):
    ims, msks = d / "images", d / "masks"
    res = []
    it = ims.glob("*")
    if img_id is not None:
        it = _filter_only_img_id_in_path(it, img_id)
    ims = list(it)
    random.shuffle(ims)
    for im in ims[:n]:
        msk = msks / f"{im.stem}.png"
        res.append((im, msk))
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
