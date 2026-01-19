def numpy_slice_to_yolo_box(start0, end0, start1, end1, img_height, img_width):
    """
    Convert numpy array slice coordinates to YOLO bounding box format.

    Args:
        start0, end0: row indices (y-axis)
        start1, end1: column indices (x-axis)
        img_height: total image height
        img_width: total image width

    Returns:
        x_center, y_center, width, height (all normalized to 0-1)
    """
    # Calculate box dimensions in pixels
    box_width = end1 - start1
    box_height = end0 - start0

    # Calculate center coordinates in pixels
    x_center_px = start1 + box_width / 2
    y_center_px = start0 + box_height / 2

    # Normalize to 0-1 range
    x_center = x_center_px / img_width
    y_center = y_center_px / img_height
    width = box_width / img_width
    height = box_height / img_height

    return x_center, y_center, width, height


def mk_yolo_label_from_numpy_coords(
    label_cls, start0, end0, start1, end1, img_height, img_width
):
    x_center, y_center, width, height = numpy_slice_to_yolo_box(
        start0, end0, start1, end1, img_height, img_width
    )
    return f"{label_cls} {x_center} {y_center} {width} {height}"