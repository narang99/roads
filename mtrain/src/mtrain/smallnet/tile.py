import numpy as np


def split_image_into_tiles(
    img: np.ndarray, tile_size: int = 50
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """
    Split an image into tiles of size tile_size x tile_size.
    Edge tiles are full-sized and may overlap with previous tiles.

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

    # Calculate number of full tiles that fit
    y_positions = list(range(0, H, tile_size))
    x_positions = list(range(0, W, tile_size))
    
    # Add final positions to ensure full tiles at edges
    if y_positions[-1] + tile_size < H:
        y_positions.append(H - tile_size)
    if x_positions[-1] + tile_size < W:
        x_positions.append(W - tile_size)

    for y in y_positions:
        for x in x_positions:
            tile = img[y : y + tile_size, x : x + tile_size]
            tiles.append((tile, (y, x)))

    return tiles
