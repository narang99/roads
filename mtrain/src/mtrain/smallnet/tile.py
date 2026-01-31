import numpy as np


def split_image_into_tiles(
    img: np.ndarray, tile_size: int = 50
) -> list[tuple[np.ndarray, tuple[int, int]]]:
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
            tile = img[y : y + tile_size, x : x + tile_size]
            tiles.append((tile, (y, x)))

    return tiles
