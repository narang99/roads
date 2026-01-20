from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.image

sample_p = Path("/Users/hariomnarang/Downloads/project-6-at-2026-01-17-07-16-eb8e5c49.json")
with open(sample_p) as f:
    sample = json.load(f)

def get_image_path(content):
    ext_p = content["data"]["image"].split("?d=")[1]
    ext_p = f"/{ext_p}"
    return Path(ext_p)

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])

def rle_to_mask(rle: list[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    # print('RLE params:', num, 'values,', word_size, 'word_size,', rle_sizes, 'rle_sizes')

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image

def crop_from_mask(image, mask, pad=0):
    """
    image: H x W x C
    mask:  H x W (bool or 0/1)
    pad:   optional padding in pixels
    """
    mask = mask.astype(bool)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # Optional padding
    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(image.shape[0] - 1, y2 + pad)
    x2 = min(image.shape[1] - 1, x2 + pad)

    crop = image[y1:y2+1, x1:x2+1]
    crop_mask = mask[y1:y2+1, x1:x2+1]

    # Zero background inside the crop (optional but usually desired)
    crop = crop * crop_mask[..., None]

    return crop, crop_mask




class ImageSaver:
    def __init__(self, dest):
        self.d = dest
        self.d.mkdir(exist_ok=True, parents=True)

    def save(self, img_arr, index):
        p = self.d / f"{index}.png"
        matplotlib.image.imsave(p, img_arr)
        return p

def extract_from_single_result(content):
    p = get_image_path(content)
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = []
    for i, annot in enumerate(content["annotations"]):
        for res in annot["result"]:
            mask = rle_to_mask(res["value"]["rle"], res["original_height"], res["original_width"])
            crop, _ = crop_from_mask(img, mask)
            if crop is None:
                print("warning: found empty mask for i =", i)
                continue
            result.append(crop)
    return result

def run_kmeans(img_bgr, K=3):
    img = img_bgr
    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )
    label_img = labels.reshape(h, w)
    return label_img


def get_include_class(lbl, include):
    return np.isin(lbl, include)

def get_exclude_class(lbl, exclude):
    return ~np.isin(lbl, exclude)
