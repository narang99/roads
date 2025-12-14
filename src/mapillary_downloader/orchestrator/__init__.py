"""
Main orchestrator for downloading Mapillary images.

Coordinates all modules: geocoding, tiles, state, and API client.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, Optional
import typing

import httpx

from .api_client import (
    MapillaryAPIError,
    RateLimitError,
    download_image,
    fetch_image_metadata,
    fetch_images_in_bbox,
)
from .geocoder import get_city_bbox
from .state import StateManager
from .tiles import Tile, get_tiles_for_bbox, tile_to_small_bboxes

# Configure logger for this module
logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Optional[Callable[[str, int, int, str], None]]


def print_bbox_progress(current: int, total: int, images_found: int):
    """Print progress bar for bbox processing."""
    if total <= 1:
        return  # No progress bar needed for single bbox
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = current / total * 100
    print(
        f"\r    Bbox [{bar}] {pct:5.1f}% ({current}/{total}) - {images_found} images",
        end="",
        flush=True,
    )
    if current == total:
        print()  # Newline when complete

async def _get_city_bbox_from_state(state, city_name, client) -> tuple[float, float, float, float]:
    bbox_str = state.get_metadata("bbox")
    if bbox_str:
        bbox = typing.cast(
            tuple[float, float, float, float],
            tuple(map(float, bbox_str.split(",")))
        )
        logger.info(f"geocode 1, 1, Using cached bounding box for {city_name}")
    else:
        logger.info(f"geocode 0, 1, Geocoding {city_name}...")
        bbox_obj = await get_city_bbox(client, city_name)
        bbox = bbox_obj.as_tuple()
        state.set_metadata("city_name", city_name)
        state.set_metadata("bbox", ",".join(map(str, bbox)))
        logger.info(f"geocode 1, 1, Bounding box: {bbox}")
    return bbox

class CityImageDownloader:
    """
    Downloads all street-level images for a city from Mapillary.

    Supports resumable downloads via SQLite state tracking.
    """

    def __init__(
        self,
        access_token: str,
        output_dir: Path,
        state_db_path: Optional[Path] = None,
        image_size: str = "thumb_1024_url",
        rate_limit_delay: float = 0.1,
        save_metadata: bool = True,
    ):
        """
        Initialize the downloader.

        Args:
            access_token: Mapillary API access token
            output_dir: Directory to save images
            state_db_path: Path to SQLite state DB (default: output_dir/state.db)
            image_size: Which thumbnail to download (thumb_256_url, thumb_1024_url, etc.)
            rate_limit_delay: Delay between API calls (seconds)
            save_metadata: Whether to save image metadata as JSON
        """
        self.access_token = access_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state_db_path = state_db_path or (self.output_dir / "state.db")
        self.state = StateManager(self.state_db_path)

        self.image_size = image_size
        self.rate_limit_delay = rate_limit_delay
        self.save_metadata = save_metadata

        # Directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        if save_metadata:
            self.metadata_dir = self.output_dir / "metadata"
            self.metadata_dir.mkdir(exist_ok=True)

    def _make_logger(self, progress_callback: ProgressCallback):
        """Create a logging function."""

        def log(phase: str, current: int, total: int, message: str):
            if progress_callback:
                progress_callback(phase, current, total, message)
            else:
                print(f"[{phase}] {message} ({current}/{total})")

        return log

    @staticmethod
    def print_bbox_info(bbox: tuple[float, float, float, float]):
        """
        Print bounding box info with Google Maps URLs.

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        print("\n📍 Bounding box:")
        print(
            f"   SW corner: https://www.google.com/maps/?q={min_lat:.6f},{min_lon:.6f}"
        )
        print(
            f"   NE corner: https://www.google.com/maps/?q={max_lat:.6f},{max_lon:.6f}"
        )
        print()

    async def download_city(
        self, city_name: str, progress_callback: ProgressCallback = None
    ):
        """
        Download all images for a city.

        Args:
            city_name: Name of the city (e.g., "Palo Alto, CA")
            progress_callback: Optional callback(phase, current, total, message)
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Phase 1: Check if we're resuming or starting fresh
            saved_city = self.state.get_metadata("city_name")

            if saved_city and saved_city != city_name:
                raise ValueError(
                    f"State database contains data for '{saved_city}', "
                    f"but you requested '{city_name}'. "
                    f"Use a different output directory or delete the state.db file."
                )

            bbox = await _get_city_bbox_from_state(self.state, city_name, client)
            self.print_bbox_info(bbox)
            await self.download_bbox(
                client=client, bbox=bbox, progress_callback=progress_callback
            )

    async def download_bbox(
        self,
        client: httpx.AsyncClient,
        bbox: tuple[float, float, float, float],
        progress_callback: ProgressCallback = None,
        image_limit: Optional[int] = None,
    ):
        """
        Download all images within a bounding box.

        This is the core download logic shared by download_city and test_download.

        Args:
            client: httpx AsyncClient
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            progress_callback: Optional progress callback
            image_limit: Optional limit on number of images to download
        """
        log = self._make_logger(progress_callback)

        # Store original bbox for optimization in _process_tile
        self._original_bbox = bbox

        # Phase 1: Calculate and register tiles
        tiles = get_tiles_for_bbox(*bbox)
        self.state.add_tiles(tiles)
        total_tiles = len(tiles)
        log("tiles", 0, total_tiles, f"Need to process {total_tiles} tiles")

        # Reset any in-progress items from crashed runs
        self.state.reset_in_progress()

        # Phase 2: Process tiles to discover images
        pending_tiles = self.state.get_pending_tiles()
        processed_tiles = total_tiles - len(pending_tiles)

        for i, tile in enumerate(pending_tiles):
            log(
                "discover",
                processed_tiles + i,
                total_tiles,
                f"Processing tile {tile.z}/{tile.x}/{tile.y}",
            )

            await self._process_tile(
                client, tile, progress_callback=progress_callback
            )

            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)

        log("discover", total_tiles, total_tiles, "Tile processing complete")

        # Phase 3: Download images
        total_images = self.state.get_total_images()
        if total_images == 0:
            log("complete", 0, 0, "No images found in this area")
            return

        image_stats = self.state.get_image_stats()
        downloaded = image_stats.get("downloaded", 0)

        effective_total = (
            min(image_limit, total_images) if image_limit else total_images
        )

        log(
            "download",
            downloaded,
            effective_total,
            f"Starting image download ({downloaded} already done)",
        )

        images_downloaded = downloaded
        while True:
            if image_limit and images_downloaded >= image_limit:
                break

            pending = self.state.get_pending_images(limit=50)
            if not pending:
                break

            for image_id in pending:
                if image_limit and images_downloaded >= image_limit:
                    break

                log(
                    "download",
                    images_downloaded,
                    effective_total,
                    f"Downloading {image_id}",
                )

                await self._download_image(client, image_id)
                await asyncio.sleep(self.rate_limit_delay)
                images_downloaded += 1

        # Final stats
        final_stats = self.state.get_image_stats()
        log(
            "complete",
            final_stats.get("downloaded", 0),
            effective_total,
            f"Download complete! {final_stats}",
        )

    async def _process_tile(
        self,
        client: httpx.AsyncClient,
        tile: Tile,
        progress_callback: ProgressCallback = None,
    ):
        """Process a single tile to discover images."""
        self.state.mark_tile_started(tile)

        try:
            # Split tile into smaller bboxes for API compliance
            bboxes = tile_to_small_bboxes(tile)

            # Optimization: if original bbox is smaller than tile, use it directly
            # This avoids fetching images outside the requested area
            all_images = []
            total_bboxes = len(bboxes)
            idx = 0

            while idx < total_bboxes:
                bbox = bboxes[idx]
                try:
                    images = await fetch_images_in_bbox(
                        client,
                        self.access_token,
                        *bbox,
                        fields=["id", "captured_at", "computed_geometry"],
                    )

                    # Parse geometry to get lat/lon
                    for img in images:
                        geom = img.get("computed_geometry", {})
                        coords = geom.get("coordinates", [None, None])
                        img["lon"] = coords[0] if coords else None
                        img["lat"] = coords[1] if len(coords) > 1 else None

                    all_images.extend(images)

                    # Success - update progress and move to next bbox
                    idx += 1
                    print_bbox_progress(idx, total_bboxes, len(all_images))
                    await asyncio.sleep(self.rate_limit_delay)

                except RateLimitError:
                    # Log error and wait before retrying same bbox
                    logger.exception(
                        f"Rate limit hit on bbox {idx + 1}/{total_bboxes}, waiting 60s before retry"
                    )
                    print(
                        f"\n    ⚠ Rate limit hit, waiting 60s before retrying bbox {idx + 1}/{total_bboxes}..."
                    )
                    await asyncio.sleep(60)
                    # Don't increment idx - will retry same bbox

            # Deduplicate by image ID
            seen = set()
            unique_images = []
            for img in all_images:
                img_id = str(img.get("id"))
                if img_id not in seen:
                    seen.add(img_id)
                    unique_images.append(img)

            # Save to state
            self.state.add_images(unique_images, tile)
            self.state.mark_tile_completed(tile, len(unique_images))

        except Exception as e:
            logger.exception(f"Failed to process tile {tile.z}/{tile.x}/{tile.y}")
            self.state.mark_tile_failed(tile, str(e))
            raise

    async def _download_image(self, client: httpx.AsyncClient, image_id: str):
        """Download a single image."""
        try:
            # Fetch metadata
            metadata = await fetch_image_metadata(
                client,
                self.access_token,
                image_id,
                fields=[
                    "id",
                    "captured_at",
                    "computed_geometry",
                    "computed_compass_angle",
                    self.image_size,
                    "is_pano",
                    "camera_type",
                    "make",
                    "model",
                    "height",
                    "width",
                ],
            )

            image_url = metadata.get(self.image_size)
            if not image_url:
                self.state.mark_image_failed(
                    image_id, f"No {self.image_size} available"
                )
                return

            self.state.mark_image_downloading(image_id, image_url)

            # Download image
            output_path = self.images_dir / f"{image_id}.jpg"
            await download_image(client, image_url, output_path)

            self.state.mark_image_downloaded(image_id, str(output_path))

            # Save metadata
            if self.save_metadata:
                meta_path = self.metadata_dir / f"{image_id}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

        except RateLimitError:
            # Don't mark as failed, will retry later
            logger.exception(
                f"Rate limit hit while downloading image {image_id}, will retry later"
            )
            await asyncio.sleep(60)
        except MapillaryAPIError as e:
            logger.exception(f"API error downloading image {image_id}")
            self.state.mark_image_failed(image_id, str(e))
        except Exception as e:
            logger.exception(f"Unexpected error downloading image {image_id}")
            self.state.mark_image_failed(image_id, str(e))


async def download_city_images(
    city_name: str,
    access_token: str,
    output_dir: str = "./output",
    image_size: str = "thumb_1024_url",
    rate_limit_delay: float = 0.1,
    save_metadata: bool = True,
):
    """
    Convenience function to download images for a city.

    Args:
        city_name: City name (e.g., "Palo Alto, CA")
        access_token: Mapillary access token
        output_dir: Output directory
        image_size: Thumbnail size field
        rate_limit_delay: Delay between API calls
        save_metadata: Whether to save JSON metadata
    """
    downloader = CityImageDownloader(
        access_token=access_token,
        output_dir=Path(output_dir),
        image_size=image_size,
        rate_limit_delay=rate_limit_delay,
        save_metadata=save_metadata,
    )
    await downloader.download_city(city_name)
