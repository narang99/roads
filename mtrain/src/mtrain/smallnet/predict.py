import cv2
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

def tile_image_and_predict(image_path, learner):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tiles = split_image_into_tiles(img)

    result = []
    for tile in tqdm(tiles):
        t, _ = tile
        if learner.predict(t)[0] == "pos":
            result.append(True)
        else:
            result.append(False)

    return highlight_tiles_bbox_with_index(img, tiles, result)

def split_image_into_tiles(
    img: np.ndarray,
    tile_size: int = 50
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split an image into tiles of size tile_size x tile_size.
    Edge tiles may be smaller.

    Parameters
    ----------
    img : np.ndarray
        Image array of shape (H, W) or (H, W, C)
    tile_size : int
        Size of each tile (default 50)

    Returns
    -------
    tiles : list of (tile, (y, x))
        tile: cropped image patch
        (y, x): top-left coordinate of the tile in original image
    """
    H, W = img.shape[:2]
    tiles = []

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile = img[y:y + tile_size, x:x + tile_size]
            tiles.append((tile, (y, x)))

    return tiles

def highlight_tiles_bbox_with_index(
    img: np.ndarray,
    tiles: List[Tuple[np.ndarray, Tuple[int, int]]],
    positives: List[bool],
    tile_size: int = 50,
    color=(0, 255, 0),
    thickness=2,
    font_scale=0.5,
    text_color=(255, 255, 255),
) -> np.ndarray:
    """
    Draw bounding boxes and tile index for positive tiles.
    """
    img_out = img.copy()
    H, W = img.shape[:2]

    for idx, ((_, (y, x)), is_pos) in enumerate(zip(tiles, positives)):
        if not is_pos:
            continue

        y2 = min(y + tile_size, H)
        x2 = min(x + tile_size, W)

        # Draw rectangle
        cv2.rectangle(img_out, (x, y), (x2, y2), color, thickness)

        # Draw tile index
        label = str(idx)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Background for text (better visibility)
        cv2.rectangle(img_out, (x, y), (x + tw + 4, y + th + 4), color, -1)

        cv2.putText(
            img_out,
            label,
            (x + 2, y + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return img_out