"""
SQLite-based state management for resumable downloads.

Tracks:
- Which tiles have been processed
- Which images have been discovered
- Which images have been downloaded
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

from .tiles import Tile


@dataclass
class ImageRecord:
    """Record of an image in the state database."""
    image_id: str
    tile_z: int
    tile_x: int
    tile_y: int
    thumb_url: Optional[str]
    status: str  # 'pending', 'downloading', 'downloaded', 'failed'
    local_path: Optional[str]
    error_message: Optional[str]
    captured_at: Optional[int]
    lat: Optional[float]
    lon: Optional[float]


class StateManager:
    """
    Manages download state using SQLite.
    
    All methods are synchronous since SQLite operations are fast
    and we want to keep the state consistent.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize state manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Tiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tiles (
                    z INTEGER,
                    x INTEGER,
                    y INTEGER,
                    status TEXT DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    image_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    PRIMARY KEY (z, x, y)
                )
            """)
            
            # Images table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    tile_z INTEGER,
                    tile_x INTEGER,
                    tile_y INTEGER,
                    thumb_url TEXT,
                    status TEXT DEFAULT 'pending',
                    downloaded_at TEXT,
                    local_path TEXT,
                    error_message TEXT,
                    captured_at INTEGER,
                    lat REAL,
                    lon REAL
                )
            """)
            
            # Metadata table for city/config info
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tiles_status 
                ON tiles(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_images_status 
                ON images(status)
            """)
    
    # --- Metadata methods ---
    
    def set_metadata(self, key: str, value: str):
        """Set a metadata value."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value)
            )
    
    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = ?",
                (key,)
            ).fetchone()
            return row["value"] if row else None
    
    # --- Tile methods ---
    
    def add_tiles(self, tiles: list[Tile]):
        """Add tiles to track (if not already present)."""
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO tiles (z, x, y, status) 
                VALUES (?, ?, ?, 'pending')
                """,
                [(t.z, t.x, t.y) for t in tiles]
            )
    
    def get_pending_tiles(self) -> list[Tile]:
        """Get all tiles that haven't been completed."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT z, x, y FROM tiles WHERE status IN ('pending', 'in_progress')"
            ).fetchall()
            return [Tile(z=r["z"], x=r["x"], y=r["y"]) for r in rows]
    
    def mark_tile_started(self, tile: Tile):
        """Mark a tile as being processed."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE tiles 
                SET status = 'in_progress', started_at = ?
                WHERE z = ? AND x = ? AND y = ?
                """,
                (datetime.utcnow().isoformat(), tile.z, tile.x, tile.y)
            )
    
    def mark_tile_completed(self, tile: Tile, image_count: int):
        """Mark a tile as completed."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE tiles 
                SET status = 'completed', completed_at = ?, image_count = ?
                WHERE z = ? AND x = ? AND y = ?
                """,
                (datetime.utcnow().isoformat(), image_count, tile.z, tile.x, tile.y)
            )
    
    def mark_tile_failed(self, tile: Tile, error: str):
        """Mark a tile as failed."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE tiles 
                SET status = 'failed', error_message = ?
                WHERE z = ? AND x = ? AND y = ?
                """,
                (error, tile.z, tile.x, tile.y)
            )
    
    def get_tile_stats(self) -> dict:
        """Get statistics about tile processing."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) as count, SUM(image_count) as images
                FROM tiles GROUP BY status
                """
            ).fetchall()
            return {
                r["status"]: {"count": r["count"], "images": r["images"] or 0}
                for r in rows
            }
    
    # --- Image methods ---
    
    def add_images(self, images: list[dict], tile: Tile):
        """
        Add discovered images to the database.
        
        Args:
            images: List of image dicts with at least 'id' key
            tile: Tile the images belong to
        """
        with self._get_connection() as conn:
            for img in images:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO images 
                    (image_id, tile_z, tile_x, tile_y, status, captured_at, lat, lon)
                    VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
                    """,
                    (
                        str(img.get("id")),
                        tile.z, tile.x, tile.y,
                        img.get("captured_at"),
                        img.get("lat"),
                        img.get("lon")
                    )
                )
    
    def get_pending_images(self, limit: int = 100) -> list[str]:
        """Get image IDs that haven't been downloaded yet."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT image_id FROM images WHERE status = 'pending' LIMIT ?",
                (limit,)
            ).fetchall()
            return [r["image_id"] for r in rows]
    
    def mark_image_downloading(self, image_id: str, thumb_url: str):
        """Mark an image as being downloaded."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE images SET status = 'downloading', thumb_url = ? WHERE image_id = ?",
                (thumb_url, image_id)
            )
    
    def mark_image_downloaded(self, image_id: str, local_path: str):
        """Mark an image as successfully downloaded."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE images 
                SET status = 'downloaded', downloaded_at = ?, local_path = ?
                WHERE image_id = ?
                """,
                (datetime.utcnow().isoformat(), local_path, image_id)
            )
    
    def mark_image_failed(self, image_id: str, error: str):
        """Mark an image download as failed."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE images SET status = 'failed', error_message = ? WHERE image_id = ?",
                (error, image_id)
            )
    
    def get_image_stats(self) -> dict:
        """Get statistics about image downloads."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM images GROUP BY status"
            ).fetchall()
            return {r["status"]: r["count"] for r in rows}
    
    def get_total_images(self) -> int:
        """Get total number of discovered images."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM images").fetchone()
            return row["count"]
    
    def reset_in_progress(self):
        """Reset any in-progress items to pending (for crash recovery)."""
        with self._get_connection() as conn:
            conn.execute("UPDATE tiles SET status = 'pending' WHERE status = 'in_progress'")
            conn.execute("UPDATE images SET status = 'pending' WHERE status = 'downloading'")
