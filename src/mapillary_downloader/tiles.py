"""
Tile calculation module.

Calculate map tiles that cover a geographic bounding box.
Uses mercantile for tile math.
"""

import mercantile
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Tile:
    """Map tile coordinates."""
    z: int  # zoom level
    x: int  # tile x coordinate
    y: int  # tile y coordinate
    
    def as_tuple(self) -> tuple[int, int, int]:
        """Return as (z, x, y) tuple."""
        return (self.z, self.x, self.y)


def get_tiles_for_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int = 14
) -> list[Tile]:
    """
    Get all tiles that cover a bounding box at the specified zoom level.
    
    Mapillary coverage tiles work best at zoom 14 for individual images.
    
    Args:
        min_lon: Western boundary (minimum longitude)
        min_lat: Southern boundary (minimum latitude)
        max_lon: Eastern boundary (maximum longitude)
        max_lat: Northern boundary (maximum latitude)
        zoom: Zoom level (default 14 for Mapillary image data)
    
    Returns:
        List of Tile objects covering the bounding box
    """
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom))
    return [Tile(z=t.z, x=t.x, y=t.y) for t in tiles]


def tile_to_bbox(tile: Tile) -> tuple[float, float, float, float]:
    """
    Convert a tile to its bounding box.
    
    Args:
        tile: Tile coordinates
    
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    bounds = mercantile.bounds(tile.x, tile.y, tile.z)
    return (bounds.west, bounds.south, bounds.east, bounds.north)


def tile_to_small_bboxes(
    tile: Tile,
    max_area_degrees: float = 0.01
) -> list[tuple[float, float, float, float]]:
    """
    Split a tile into smaller bounding boxes for Mapillary API.
    
    Mapillary's bbox search requires areas smaller than 0.01 degrees square.
    This splits larger tiles into compliant sub-boxes.
    
    Args:
        tile: Tile to split
        max_area_degrees: Maximum bbox dimension (default 0.01 per Mapillary API)
    
    Returns:
        List of bounding box tuples that cover the tile
    """
    min_lon, min_lat, max_lon, max_lat = tile_to_bbox(tile)
    
    bboxes = []
    
    # Calculate step size (slightly smaller than max to be safe)
    step = max_area_degrees * 0.9
    
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            bboxes.append((
                lon,
                lat,
                min(lon + step, max_lon),
                min(lat + step, max_lat)
            ))
            lon += step
        lat += step
    
    return bboxes


def count_tiles_for_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int = 14
) -> int:
    """
    Count tiles needed for a bounding box without generating them all.
    
    Useful for progress estimation.
    """
    tiles = mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom)
    return sum(1 for _ in tiles)
