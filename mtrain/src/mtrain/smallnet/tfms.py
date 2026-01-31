# converting a cropped image from TACO to similar to road side images
# this is the main module for make synthetic data that is
# similar to actual real roadside street view images captured from cameras
import cv2
import functools


def PaddedResize(target_size):
    return functools.partial(resize_and_pad, target_size=target_size, pad_value=0)


def resize_and_pad(img, target_size, pad_value=0):
    padded, _ = resize_and_pad_raw(img, target_size, pad_value=0)
    return padded


def resize_and_pad_raw(img, target_size, pad_value=0):
    """
    Resize image so that the larger dimension becomes `target_size`,
    then pad the other dimension to get
    (target_size x target_size).

    Args:
        img (np.ndarray): Input image
        target_size (int): Output size (e.g. 50)
        pad_value (int or tuple): Padding value (default = 0 / black)

    Returns:
        np.ndarray: (target_size x target_size) image
    """
    h, w = img.shape[:2]

    # Scale so max dimension == target_size
    scale = target_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute padding
    pad_x = target_size - new_w
    pad_y = target_size - new_h

    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )

    meta = {
        "orig_h": h,
        "orig_w": w,
        "scale": scale,
        "new_h": new_h,
        "new_w": new_w,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
    }

    return padded, meta
