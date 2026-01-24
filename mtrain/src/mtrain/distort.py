import numpy as np
import random
from mtrain.random import random_bool
from scipy.ndimage import gaussian_filter
import cv2
from mtrain.random import random_true_one_three_times


rng = np.random.default_rng()  # use numpy's new Generator API


def random_warp(img):
    if random_true_one_three_times():
        # return image back 1/3rd times
        return img
    # warper = _perspective_warp if random_bool() else _perspective_warp
    warper = random_warp_masked
    return _warp_n_times(img, warper, random.randint(0, 2))


def _warp_n_times(img, warper, n):
    res = img
    for _ in range(n):
        res = warper(res)
    return res


# def _perspective_warp(img, max_shift=5):
#     h, w = img.shape[:2]

#     mask = (img > 0).astype(np.uint8)

#     src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
#     dst_pts = src_pts + rng.integers(-max_shift, max_shift + 1, src_pts.shape).astype(
#         np.float32
#     )

#     M = cv2.getPerspectiveTransform(src_pts, dst_pts)

#     warped_img = cv2.warpPerspective(
#         img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
#     )
#     warped_mask = cv2.warpPerspective(
#         mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
#     )

#     warped_img[warped_mask == 0] = 0
#     return warped_img


# def _elastic_warp(img, alpha=5, sigma=2):
#     """
#     Apply elastic deformation to a fragment image.

#     img: np.ndarray, input fragment (zeros for background)
#     alpha: float, scaling factor for maximum displacement
#     sigma: float, smoothness of deformation (higher = smoother)

#     returns: np.ndarray, warped fragment
#     """
#     h, w = img.shape[:2]

#     # 1. Create random displacement fields
#     dx = np.random.uniform(-1, 1, (h, w)) * alpha
#     dy = np.random.uniform(-1, 1, (h, w)) * alpha

#     # 2. Smooth them with Gaussian
#     dx = gaussian_filter(dx, sigma=sigma)
#     dy = gaussian_filter(dy, sigma=sigma)

#     # 3. Create meshgrid of coordinates
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
#     map_x = (x + dx).astype(np.float32)
#     map_y = (y + dy).astype(np.float32)

#     # 4. Warp the image
#     warped = cv2.remap(
#         img,
#         map_x,
#         map_y,
#         interpolation=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=0,
#     )

#     return warped

def random_warp_masked(img, max_shift=5):
    img, mask = _random_warp_masked(img)
    mask = trim_mask(mask, 3)
    mask = mask == 255
    img[~mask] = 0
    return img

def _random_warp_masked(img, max_shift=5):
    """
    Perspective warp that preserves zero background by warping
    image and mask together (Strategy B).

    Parameters
    ----------
    img : np.ndarray (H, W, 3)
        Fragment image with zero-padded background
    rng : np.random.Generator
        Random generator (np.random.default_rng)
    max_shift : int
        Max pixel shift for perspective warp

    Returns
    -------
    warped_img : np.ndarray
        Warped fragment with clean zero background
    """

    h, w = img.shape[:2]

    # --- 1. Build fragment mask ---
    mask = (img > 0).any(axis=2).astype(np.uint8) * 255

    # --- 2. Perspective transform points ---
    src_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])

    dst_pts = src_pts + rng.integers(
        -max_shift, max_shift + 1, size=src_pts.shape
    ).astype(np.float32)

    # --- 3. Compute transform ---
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # --- 4. Warp image ---
    warped_img = cv2.warpPerspective(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # --- 5. Warp mask ---
    warped_mask = cv2.warpPerspective(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # --- 6. Enforce mask (kill any background bleed) ---
    warped_img[warped_mask == 0] = 0

    return warped_img, warped_mask

def trim_mask(mask, pixels=2, kernel_shape=cv2.MORPH_ELLIPSE):
    """
    Shrink a binary mask inward by a fixed number of pixels.

    Parameters
    ----------
    mask : np.ndarray (H, W), bool or uint8
        Binary mask (foreground = 1 or 255)
    pixels : int
        How many pixels to trim inward
    kernel_shape : cv2 morphology shape
        MORPH_ELLIPSE (natural), MORPH_RECT, or MORPH_CROSS

    Returns
    -------
    trimmed_mask : np.ndarray (uint8)
        Eroded binary mask
    """

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Ensure binary 0 / 255
    mask = (mask > 0).astype(np.uint8) * 255

    # Kernel roughly corresponds to pixel trim size
    k = 2 * pixels + 1
    kernel = cv2.getStructuringElement(kernel_shape, (k, k))

    trimmed = cv2.erode(mask, kernel, iterations=1)

    return trimmed