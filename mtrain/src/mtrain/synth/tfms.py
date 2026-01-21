import random
import numpy as np
import cv2


def merge_overlapping_boxes(boxes):
    """
    Merge overlapping bounding boxes.

    Args:
        boxes: list of tuples [(start0, end0, start1, end1), ...]

    Returns:
        merged_boxes: list of tuples with overlapping boxes merged
    """
    if not boxes:
        return []

    # Convert to numpy array for easier manipulation
    boxes = [list(box) for box in boxes]
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        # Start with current box
        current = boxes[i]
        used[i] = True

        # Keep merging until no more overlaps found
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue

                # Check if boxes overlap
                if boxes_overlap(current, boxes[j]):
                    # Merge boxes
                    current = [
                        min(current[0], boxes[j][0]),  # min start0
                        max(current[1], boxes[j][1]),  # max end0
                        min(current[2], boxes[j][2]),  # min start1
                        max(current[3], boxes[j][3]),  # max end1
                    ]
                    used[j] = True
                    changed = True

        merged.append(tuple(current))

    return merged


def boxes_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.

    Args:
        box1, box2: [start0, end0, start1, end1]

    Returns:
        bool: True if boxes overlap
    """
    start0_1, end0_1, start1_1, end1_1 = box1
    start0_2, end0_2, start1_2, end1_2 = box2

    # Check for no overlap (easier to detect)
    if end0_1 <= start0_2 or end0_2 <= start0_1:  # No vertical overlap
        return False
    if end1_1 <= start1_2 or end1_2 <= start1_1:  # No horizontal overlap
        return False

    return True


def decrease_luminosity_lab(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Decrease image luminosity using LAB color space.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image (uint8).
    scale : float
        Scaling factor in (0, 1].
        1.0 = no change, 0.5 = half brightness.

    Returns
    -------
    np.ndarray
        Brightness-adjusted BGR image.
    """
    if not (0 < scale <= 1):
        raise ValueError("scale must be in (0, 1]")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Scale L* channel
    L = lab[:, :, 0].astype(np.float32)
    L *= scale
    lab[:, :, 0] = np.clip(L, 0, 255).astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def random_luminosity_scale(img):
    scale = random.uniform(0.6, 1.0)
    return decrease_luminosity_lab(img, scale)


def random_blur(img):
    kernel_size = random.randint(2, 7)
    return cv2.blur(img, kernel_size)


def resize_fragment_with_manual_horizon(
    fragment_img,
    fragment_base_y,
    fragment_horizon_y,
    target_base_y,
    target_horizon_y,
):
    """
    Resizes fragment so it matches perspective between two images
    using manually provided horizon lines.
    """
    if fragment_base_y <= fragment_horizon_y:
        raise ValueError("Fragment base must be below fragment horizon")
    if target_base_y <= target_horizon_y:
        raise ValueError("Target base must be below target horizon")
    scale = (target_base_y - target_horizon_y) / (fragment_base_y - fragment_horizon_y)
    if scale <= 0:
        raise ValueError("Invalid scale computed")
    new_w = int(fragment_img.shape[1] * scale)
    new_h = int(fragment_img.shape[0] * scale)
    return new_w, new_h, scale
