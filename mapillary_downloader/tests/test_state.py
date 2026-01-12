"""Tests for SQLite state management module."""

import pytest
import tempfile
from pathlib import Path

from mapillary_downloader.state import StateManager
from mapillary_downloader.tiles import Tile


@pytest.fixture
def state_manager():
    """Create a temporary state manager for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_state.db"
        yield StateManager(db_path)


class TestStateManagerMetadata:
    """Tests for metadata storage."""
    
    def test_set_and_get_metadata(self, state_manager):
        state_manager.set_metadata("city_name", "Palo Alto, CA")
        assert state_manager.get_metadata("city_name") == "Palo Alto, CA"
    
    def test_get_nonexistent_metadata(self, state_manager):
        assert state_manager.get_metadata("nonexistent") is None
    
    def test_update_metadata(self, state_manager):
        state_manager.set_metadata("key", "value1")
        state_manager.set_metadata("key", "value2")
        assert state_manager.get_metadata("key") == "value2"


class TestStateManagerTiles:
    """Tests for tile tracking."""
    
    def test_add_tiles(self, state_manager):
        tiles = [
            Tile(14, 100, 200),
            Tile(14, 101, 200),
            Tile(14, 102, 200),
        ]
        state_manager.add_tiles(tiles)
        
        pending = state_manager.get_pending_tiles()
        assert len(pending) == 3
    
    def test_add_duplicate_tiles(self, state_manager):
        """Adding the same tile twice should not create duplicates."""
        tile = Tile(14, 100, 200)
        state_manager.add_tiles([tile])
        state_manager.add_tiles([tile])
        
        pending = state_manager.get_pending_tiles()
        assert len(pending) == 1
    
    def test_tile_lifecycle(self, state_manager):
        tile = Tile(14, 100, 200)
        state_manager.add_tiles([tile])
        
        # Initially pending
        pending = state_manager.get_pending_tiles()
        assert len(pending) == 1
        
        # Mark started
        state_manager.mark_tile_started(tile)
        pending = state_manager.get_pending_tiles()
        assert len(pending) == 1  # In-progress is still "pending"
        
        # Mark completed
        state_manager.mark_tile_completed(tile, image_count=42)
        pending = state_manager.get_pending_tiles()
        assert len(pending) == 0
        
        # Check stats
        stats = state_manager.get_tile_stats()
        assert stats["completed"]["count"] == 1
        assert stats["completed"]["images"] == 42
    
    def test_tile_failure(self, state_manager):
        tile = Tile(14, 100, 200)
        state_manager.add_tiles([tile])
        state_manager.mark_tile_started(tile)
        state_manager.mark_tile_failed(tile, "API error")
        
        pending = state_manager.get_pending_tiles()
        assert len(pending) == 0
        
        stats = state_manager.get_tile_stats()
        assert stats["failed"]["count"] == 1


class TestStateManagerImages:
    """Tests for image tracking."""
    
    def test_add_images(self, state_manager):
        tile = Tile(14, 100, 200)
        images = [
            {"id": "123", "captured_at": 1000000, "lat": 37.5, "lon": -122.1},
            {"id": "456", "captured_at": 1000001, "lat": 37.51, "lon": -122.11},
        ]
        state_manager.add_images(images, tile)
        
        # Should be discovered but not pending download yet
        assert state_manager.get_total_discovered_images() == 2
        assert len(state_manager.get_pending_downloads()) == 0

        # Mark for download
        state_manager.mark_all_pending_images_for_download()
        
        pending = state_manager.get_pending_images()
        assert len(pending) == 2
        assert "123" in pending
        assert "456" in pending
    
    def test_add_duplicate_images(self, state_manager):
        """Adding the same image twice should not create duplicates."""
        tile = Tile(14, 100, 200)
        images = [{"id": "123"}]
        state_manager.add_images(images, tile)
        state_manager.add_images(images, tile)
        
        assert state_manager.get_total_discovered_images() == 1
        
        state_manager.mark_all_pending_images_for_download()
        pending = state_manager.get_pending_images()
        assert len(pending) == 1
    
    def test_image_lifecycle(self, state_manager):
        tile = Tile(14, 100, 200)
        state_manager.add_images([{"id": "123"}], tile)
        state_manager.mark_all_pending_images_for_download() # Explicitly mark for download
        
        # Initially pending
        pending = state_manager.get_pending_images()
        assert len(pending) == 1
        
        # Mark downloading
        state_manager.mark_image_downloading("123", "https://example.com/img.jpg")
        pending = state_manager.get_pending_images()
        assert len(pending) == 0  # No longer pending
        
        # Mark downloaded
        state_manager.mark_image_downloaded("123", "/path/to/image.jpg")
        
        stats = state_manager.get_image_stats()
        assert stats.get("downloaded", 0) == 1
    
    def test_image_failure(self, state_manager):
        tile = Tile(14, 100, 200)
        state_manager.add_images([{"id": "123"}], tile)
        state_manager.mark_all_pending_images_for_download() # Explicitly mark for download

        state_manager.mark_image_failed("123", "Download failed")
        
        stats = state_manager.get_image_stats()
        assert stats.get("failed", 0) == 1
    
    def test_get_pending_images_limit(self, state_manager):
        tile = Tile(14, 100, 200)
        images = [{"id": str(i)} for i in range(100)]
        state_manager.add_images(images, tile)
        # Mark them
        state_manager.mark_all_pending_images_for_download()
        
        pending = state_manager.get_pending_images(limit=10)
        assert len(pending) == 10
    
    def test_get_total_images(self, state_manager):
        tile = Tile(14, 100, 200)
        images = [{"id": str(i)} for i in range(50)]
        state_manager.add_images(images, tile)
        
        # Check discovery count
        assert state_manager.get_total_discovered_images() == 50
        
        # Check download count (should be 0)
        assert state_manager.get_total_images() == 0

        # Mark for download
        state_manager.mark_all_pending_images_for_download()
        assert state_manager.get_total_images() == 50
    
    def test_mark_random_images_for_download(self, state_manager):
        tile = Tile(14, 100, 200)
        # Create 100 images
        images = [{"id": str(i)} for i in range(100)]
        state_manager.add_images(images, tile)
        
        # Request 10 random images
        state_manager.mark_random_images_for_download(10)
        
        pending = state_manager.get_pending_images()
        assert len(pending) == 10
        assert state_manager.get_total_discovered_images() == 100
        
        # Request 10 more (should be different ones if not exhausted, but here we just check total)
        # Wait, the logic is "NOT IN download_requests". So subsequent calls add more.
        state_manager.mark_random_images_for_download(10)
        pending = state_manager.get_pending_images()
        assert len(pending) == 20 


class TestStateManagerRecovery:
    """Tests for crash recovery."""
    
    def test_reset_in_progress(self, state_manager):
        tile = Tile(14, 100, 200)
        state_manager.add_tiles([tile])
        state_manager.mark_tile_started(tile)
        
        # Simulate crash recovery
        state_manager.reset_in_progress()
        
        # Tile should be pending again
        pending = state_manager.get_pending_tiles()
        assert len(pending) == 1
    
    def test_reset_downloading_images(self, state_manager):
        tile = Tile(14, 100, 200)
        state_manager.add_images([{"id": "123"}], tile)
        state_manager.mark_all_pending_images_for_download() # Explicitly mark for download

        state_manager.mark_image_downloading("123", "https://example.com/img.jpg")
        
        # Simulate crash recovery
        state_manager.reset_in_progress()
        
        # Image should be pending again
        pending = state_manager.get_pending_images()
        assert len(pending) == 1
