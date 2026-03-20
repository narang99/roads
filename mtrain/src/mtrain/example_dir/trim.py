import numpy as np
import cv2


def extract_regions(mask: np.ndarray) -> list[dict]:
    """Extract connected components from binary mask (from gen_preds.py)"""
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    regions = []
    for i in range(1, num_labels):  # skip background (0)
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions.append(
            {
                "label": i,
                "area": int(area),
                "bbox": (int(x), int(y), int(w), int(h)),
                "centroid": (float(cx), float(cy)),
                "contour": contours[0] if contours else None,
                "component_mask": component_mask,
            }
        )
    return regions


def get_mask_with_area_in_range(mask, lo, hi):
    """Filter mask by area threshold (from gen_preds.py)"""
    mask = mask.astype(bool)
    regions = extract_regions(mask)

    def _is_valid_area(area):
        if lo is not None:
            if area < lo:
                return False
        if hi is not None:
            if area > hi:
                return False
        return True
            

    regions = [r for r in regions if _is_valid_area(r["area"])]
    res = np.zeros(mask.shape, bool)
    for r in regions:
        res |= r["component_mask"].astype(bool)
    return res
