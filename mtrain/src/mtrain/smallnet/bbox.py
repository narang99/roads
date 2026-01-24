import random
import numpy as np


def crop_with_bbox(image, bbox, max_padding, min_padding=20):
    """
    Create a crop containing the bounding box with random padding.

    Args:
        image: PIL Image or numpy array
        bbox: Bounding box [x, y, w, h]
        max_padding: Maximum padding on each side

    Returns:
        Cropped image, new bbox coordinates in crop
    """
    img_width, img_height = image.shape[1], image.shape[0]
    x, y, w, h = bbox

    # Random padding on each side
    pad_left = random.randint(min_padding, max_padding)
    pad_right = random.randint(min_padding, max_padding)
    pad_top = random.randint(min_padding, max_padding)
    pad_bottom = random.randint(min_padding, max_padding)

    # Calculate crop boundaries
    crop_x1 = max(0, x - pad_left)
    crop_y1 = max(0, y - pad_top)
    crop_x2 = min(img_width, x + w + pad_right)
    crop_y2 = min(img_height, y + h + pad_bottom)

    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # New bbox coordinates in cropped image
    new_bbox = [x - crop_x1, y - crop_y1, w, h]

    return cropped, new_bbox


def boxes_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


# def get_non_trash_boxes(image, trash_bboxes, box_size, num_samples=10):
#     """
#     Generate square bounding boxes for regions without trash.

#     Args:
#         image: numpy array
#         trash_bboxes: List of bounding boxes with trash [x, y, w, h]
#         box_size: Side length of square crop boxes in pixels
#         num_samples: Number of non-trash boxes to generate

#     Returns:
#         List of square bounding boxes [x, y, w, h] that don't contain trash
#     """
#     img_width, img_height = image.shape[1], image.shape[0]

#     non_trash_boxes = []
#     attempts = 0
#     max_attempts = num_samples * 100

#     while len(non_trash_boxes) < num_samples and attempts < max_attempts:
#         # Random position for square box
#         x = random.randint(0, max(0, img_width - box_size))
#         y = random.randint(0, max(0, img_height - box_size))

#         candidate_box = [x, y, box_size, box_size]

#         # Check if overlaps with any trash box
#         has_trash = any(
#             boxes_overlap(candidate_box, trash_box) for trash_box in trash_bboxes
#         )

#         if not has_trash:
#             non_trash_boxes.append(candidate_box)

#         attempts += 1

#     return [single_box_to_crop(image, box) for box in non_trash_boxes]


def get_non_trash_boxes(image, trash_bboxes, max_box_size, num_samples=10):
    """
    Generate square bounding boxes for regions without trash.
    Uses the maximum possible size up to max_box_size.

    Args:
        image: PIL Image or numpy array
        trash_bboxes: List of bounding boxes with trash [x, y, w, h]
        max_box_size: Maximum side length of square crop boxes in pixels
        num_samples: Number of non-trash boxes to generate

    Returns:
        List of square bounding boxes [x, y, w, h] that don't contain trash
    """
    img_width, img_height = image.shape[1], image.shape[0]

    def boxes_overlap(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    def find_max_box_size(x, y, max_size):
        """Find the largest box size that doesn't overlap with trash."""
        for size in range(max_size, 0, -1):
            # Check if box fits in image
            if x + size > img_width or y + size > img_height:
                continue

            candidate_box = [x, y, size, size]

            # Check if overlaps with any trash box
            has_trash = any(
                boxes_overlap(candidate_box, trash_box) for trash_box in trash_bboxes
            )

            if not has_trash:
                return size

        return 0

    non_trash_boxes = []
    attempts = 0
    max_attempts = num_samples * 100

    while len(non_trash_boxes) < num_samples and attempts < max_attempts:
        # Random position
        x = random.randint(0, max(0, img_width - 1))
        y = random.randint(0, max(0, img_height - 1))

        # Find maximum box size at this position
        box_size = find_max_box_size(x, y, max_box_size)

        if box_size > 0:
            non_trash_boxes.append([x, y, box_size, box_size])

        attempts += 1

    return [single_box_to_crop(image, b) for b in non_trash_boxes]


def merge_overlapping_bboxes(bboxes):
    """
    Merge bounding boxes that have any non-zero overlap.
    """
    if len(bboxes) == 0:
        return []

    # Convert to [x1, y1, x2, y2] format
    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in bboxes])

    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        current = boxes[i].copy()
        used[i] = True

        # Keep merging until no more overlaps found
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue

                # Check for any overlap
                x1 = max(current[0], boxes[j][0])
                y1 = max(current[1], boxes[j][1])
                x2 = min(current[2], boxes[j][2])
                y2 = min(current[3], boxes[j][3])

                has_overlap = (x2 > x1) and (y2 > y1)

                if has_overlap:
                    # Merge boxes
                    current[0] = min(current[0], boxes[j][0])
                    current[1] = min(current[1], boxes[j][1])
                    current[2] = max(current[2], boxes[j][2])
                    current[3] = max(current[3], boxes[j][3])
                    used[j] = True
                    changed = True

        # Convert back to [x, y, w, h]
        merged.append(
            [current[0], current[1], current[2] - current[0], current[3] - current[1]]
        )

    return merged


def single_box_to_crop(image, box):
    # box: [x,y,w,h]
    x, y, w, h = box
    return image[y : y + h, x : x + w]
