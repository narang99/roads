import cv2
import numpy as np
from mtrain.seg.cityscapes import CityScapesCls, get_cached_seg_former


def using_simple_ratio(img_height):
    return int(img_height * 0.3)


def using_vanishing_point(img):
    # vibe coded
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        # Fallback to simple calculation
        return using_simple_ratio(img)

    # Find horizontal-ish lines and estimate horizon
    horizontal_lines = []
    for line in lines:
        rho, theta = line[0]
        # Consider lines that are roughly horizontal (within 30 degrees)
        if abs(theta - np.pi / 2) < np.pi / 6:
            y = rho / np.sin(theta) if np.sin(theta) != 0 else img.shape[0] * 0.3
            if 0 < y < img.shape[0]:
                horizontal_lines.append(y)

    if horizontal_lines:
        # Use median of horizontal lines as horizon estimate
        return int(np.median(horizontal_lines))
    else:
        # Fallback to simple calculation
        return using_simple_ratio(img)


def using_road_mask(img):
    # works well, needs caching though
    model = get_cached_seg_former()
    pred = model.predict_bgr_image(img)
    road_mask = model.get_mask(pred, CityScapesCls.ROAD)
    return highest_point_in_road_mask(img, road_mask)


def highest_point_in_road_mask(img, road_mask):
    # vibe coded
    # Find all road pixels
    road_coords = np.where(road_mask)

    if len(road_coords[0]) == 0:
        # No road pixels found, fallback to simple calculation
        return using_simple_ratio(img.shape[0])

    # Get the minimum y-coordinate (highest point) of road pixels
    highest_road_y = np.min(road_coords[0])

    return int(highest_road_y)
