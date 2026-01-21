from dataclasses import dataclass
from mtrain.label_studio.json_export import get_image_path
import cv2
import numpy as np


class _InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)


def _access_bit(data, num):
    """from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def _bytes2bit(data):
    """get bit string from bytes data"""
    return "".join([str(_access_bit(data, i)) for i in range(len(data) * 8)])


def _rle_to_mask(rle: list[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = _InputStream(_bytes2bit(rle))

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

    crop = image[y1 : y2 + 1, x1 : x2 + 1]
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]

    # Zero background inside the crop (optional but usually desired)
    crop = crop * crop_mask[..., None]

    return crop, crop_mask


# class ImageSaver:
#     def __init__(self, dest):
#         self.d = dest
#         self.d.mkdir(exist_ok=True, parents=True)

#     def save(self, img_arr, index):
#         p = self.d / f"{index}.png"
#         matplotlib.image.imsave(p, img_arr)
#         return p


@dataclass
class BoundingBox:
    r: int
    c: int
    r_len: int
    c_len: int


@dataclass
class OriginalStats:
    r_len: int
    c_len: int

@dataclass
class ExtractedFragment:
    mask: np.ndarray
    bounding_box: BoundingBox
    original: OriginalStats
    crop: np.ndarray


def extract_from_single_result(content) -> list[ExtractedFragment]:
    p = get_image_path(content)
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = []
    for i, annot in enumerate(content["annotations"]):
        for res in annot["result"]:
            mask = _rle_to_mask(
                res["value"]["rle"], res["original_height"], res["original_width"]
            )
            crop, _ = crop_from_mask(img, mask)
            if crop is None:
                print("warning: found empty mask for i =", i)
                continue
            x, y, w, h = cv2.boundingRect(mask)
            result.append(
                ExtractedFragment(
                    mask=mask,
                    crop=crop,
                    bounding_box=BoundingBox(r=y, c=x, r_len=h, c_len=w),
                    original=OriginalStats(img.shape[0], img.shape[1])
                )
            )
    return result

