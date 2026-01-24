import cv2
import numpy as np
from mtrain.random import random_bool, random_true_one_three_times
import random


def random_color_shift(img):
    if random_true_one_three_times():
        # return image back 1/3rd times
        return img
    shifter = _random_hsv_shift if random_bool() else _random_lab_shift
    return shifter(img)


def _random_hsv_shift(img, max_hue_shift=20, max_saturation_shift=30, max_value_shift=30):
    """
    Randomly shift the base color of a fragment using HSV color space.

    Parameters:
        img : np.ndarray
            Input fragment (zeros for background)
        max_hue_shift : int
            Max shift in hue channel (0-179)
        max_saturation_shift : int
            Max shift in saturation channel
        max_value_shift : int
            Max shift in value/brightness channel

    Returns:
        np.ndarray : HSV-modified fragment
    """
    # Create mask of non-zero fragment
    mask = (img > 0).any(axis=2)  # True where fragment exists

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    h, s, v = cv2.split(hsv)

    # Random shifts
    h_shift = np.random.randint(-max_hue_shift, max_hue_shift + 1)
    s_shift = np.random.randint(-max_saturation_shift, max_saturation_shift + 1)
    v_shift = np.random.randint(-max_value_shift, max_value_shift + 1)

    # Apply shifts only on fragment
    h[mask] = (h[mask] + h_shift) % 180
    s[mask] = np.clip(s[mask] + s_shift, 0, 255)
    v[mask] = np.clip(v[mask] + v_shift, 0, 255)

    hsv_mod = cv2.merge([h, s, v]).astype(np.uint8)
    img_mod = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)
    return img_mod

def _random_lab_shift(img, max_a_shift=10, max_b_shift=10, max_l_shift=10):
    """
    Randomly shift the base color of a fragment using LAB color space.

    Parameters:
        img : np.ndarray
            Input fragment (zeros for background)
        max_a_shift : int
            Max shift in A channel (green-red)
        max_b_shift : int
            Max shift in B channel (blue-yellow)
        max_l_shift : int
            Max shift in Lightness channel

    Returns:
        np.ndarray : LAB-modified fragment
    """
    mask = (img > 0).any(axis=2)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int32)
    l, a, b = cv2.split(lab)

    # Random shifts
    l_shift = np.random.randint(-max_l_shift, max_l_shift + 1)
    a_shift = np.random.randint(-max_a_shift, max_a_shift + 1)
    b_shift = np.random.randint(-max_b_shift, max_b_shift + 1)

    # Apply shifts only on fragment
    l[mask] = np.clip(l[mask] + l_shift, 0, 255)
    a[mask] = np.clip(a[mask] + a_shift, 0, 255)
    b[mask] = np.clip(b[mask] + b_shift, 0, 255)

    lab_mod = cv2.merge([l, a, b]).astype(np.uint8)
    img_mod = cv2.cvtColor(lab_mod, cv2.COLOR_LAB2BGR)
    return img_mod