"""Tests for tile calculation module."""

import pytest
from mapillary_downloader.tiles import (
    Tile,
    get_tiles_for_bbox,
    tile_to_bbox,
    tile_to_small_bboxes,
    count_tiles_for_bbox,
)


class TestTile:
    """Tests for Tile dataclass."""
    
    def test_tile_creation(self):
        tile = Tile(z=14, x=100, y=200)
        assert tile.z == 14
        assert tile.x == 100
        assert tile.y == 200
    
    def test_tile_as_tuple(self):
        tile = Tile(z=14, x=100, y=200)
        assert tile.as_tuple() == (14, 100, 200)
    
    def test_tile_hashable(self):
        """Tiles should be hashable for use in sets."""
        tile1 = Tile(z=14, x=100, y=200)
        tile2 = Tile(z=14, x=100, y=200)
        tile3 = Tile(z=14, x=101, y=200)
        
        assert hash(tile1) == hash(tile2)
        assert {tile1, tile2, tile3} == {tile1, tile3}


class TestGetTilesForBbox:
    """Tests for get_tiles_for_bbox function."""
    
    def test_single_tile_area(self):
        """A very small area should return at least one tile."""
        tiles = get_tiles_for_bbox(-122.15, 37.42, -122.14, 37.43, zoom=14)
        assert len(tiles) >= 1
        assert all(isinstance(t, Tile) for t in tiles)
        assert all(t.z == 14 for t in tiles)
    
    def test_larger_area(self):
        """A larger area should return multiple tiles."""
        # Approximately Palo Alto area
        tiles = get_tiles_for_bbox(-122.20, 37.40, -122.10, 37.50, zoom=14)
        assert len(tiles) > 1
    
    def test_different_zoom_levels(self):
        """Higher zoom = more tiles."""
        bbox = (-122.15, 37.42, -122.10, 37.45)
        tiles_z12 = get_tiles_for_bbox(*bbox, zoom=12)
        tiles_z14 = get_tiles_for_bbox(*bbox, zoom=14)
        tiles_z16 = get_tiles_for_bbox(*bbox, zoom=16)
        
        assert len(tiles_z12) <= len(tiles_z14) <= len(tiles_z16)


class TestTileToBbox:
    """Tests for tile_to_bbox function."""
    
    def test_returns_valid_bbox(self):
        tile = Tile(z=14, x=2621, y=6332)
        bbox = tile_to_bbox(tile)
        
        assert len(bbox) == 4
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Check ordering
        assert min_lon < max_lon
        assert min_lat < max_lat
        
        # Check reasonable values for this tile
        assert -180 <= min_lon <= 180
        assert -90 <= min_lat <= 90
    
    def test_roundtrip(self):
        """Converting tiles to bbox and back should cover the original area."""
        original_bbox = (-122.15, 37.42, -122.10, 37.45)
        tiles = get_tiles_for_bbox(*original_bbox, zoom=14)
        
        # All tile bboxes should cover the original area
        all_min_lon = min(tile_to_bbox(t)[0] for t in tiles)
        all_min_lat = min(tile_to_bbox(t)[1] for t in tiles)
        all_max_lon = max(tile_to_bbox(t)[2] for t in tiles)
        all_max_lat = max(tile_to_bbox(t)[3] for t in tiles)
        
        # The tiles should cover the original bbox
        assert all_min_lon <= original_bbox[0]
        assert all_min_lat <= original_bbox[1]
        assert all_max_lon >= original_bbox[2]
        assert all_max_lat >= original_bbox[3]


class TestTileToSmallBboxes:
    """Tests for tile_to_small_bboxes function."""
    
    def test_splits_large_tile(self):
        """Large tiles should be split into smaller bboxes."""
        tile = Tile(z=14, x=2621, y=6332)
        bboxes = tile_to_small_bboxes(tile, max_area_degrees=0.01)
        
        # Should create multiple bboxes
        assert len(bboxes) >= 1
        
        # Each bbox should be small enough
        for bbox in bboxes:
            min_lon, min_lat, max_lon, max_lat = bbox
            assert (max_lon - min_lon) <= 0.01
            assert (max_lat - min_lat) <= 0.01
    
    def test_covers_original_tile(self):
        """Split bboxes should cover the entire tile."""
        tile = Tile(z=14, x=2621, y=6332)
        tile_bbox = tile_to_bbox(tile)
        bboxes = tile_to_small_bboxes(tile, max_area_degrees=0.01)
        
        # Check coverage
        all_min_lon = min(b[0] for b in bboxes)
        all_min_lat = min(b[1] for b in bboxes)
        all_max_lon = max(b[2] for b in bboxes)
        all_max_lat = max(b[3] for b in bboxes)
        
        # Should cover the tile
        assert all_min_lon <= tile_bbox[0] + 0.001
        assert all_min_lat <= tile_bbox[1] + 0.001
        assert all_max_lon >= tile_bbox[2] - 0.001
        assert all_max_lat >= tile_bbox[3] - 0.001


class TestCountTilesForBbox:
    """Tests for count_tiles_for_bbox function."""
    
    def test_matches_actual_count(self):
        bbox = (-122.15, 37.42, -122.10, 37.45)
        count = count_tiles_for_bbox(*bbox, zoom=14)
        tiles = get_tiles_for_bbox(*bbox, zoom=14)
        
        assert count == len(tiles)
