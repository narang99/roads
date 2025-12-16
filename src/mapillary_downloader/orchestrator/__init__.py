"""
Main orchestrator for downloading Mapillary images.

Coordinates all modules: geocoding, tiles, state, and API client.
"""

import asyncio
import json
import logging
import typing
from pathlib import Path
from typing import Callable, Optional

import httpx
from tqdm import tqdm

from mapillary_downloader.orchestrator.chunk_dwn import (
    discover_phase,
    download_single_image_with_retry,
)
from mapillary_downloader.orchestrator.pretty import print_bbox_info

from ..geocoder import get_city_bbox
from ..state import StateManager

# Configure logger for this module
logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Optional[Callable[[str, int, int, str], None]]


async def _get_city_bbox_from_state(
    state, city_name, client
) -> tuple[float, float, float, float]:
    bbox_str = state.get_metadata("bbox")
    if bbox_str:
        bbox = typing.cast(
            tuple[float, float, float, float], tuple(map(float, bbox_str.split(",")))
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

    async def download_city(
        self,
        city_name: str,
        progress_callback: ProgressCallback = None,
        post_discovery_hook: Optional[Callable[["StateManager"], None]] = None,
    ):
        """
        Download all images for a city.

        Args:
            city_name: Name of the city (e.g., "Palo Alto, CA")
            progress_callback: Optional callback(phase, current, total, message)
            post_discovery_hook: Optional callback to run after discovery but before download
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
            print_bbox_info(bbox)
            await self.download_bbox(
                client=client,
                bbox=bbox,
                progress_callback=progress_callback,
                post_discovery_hook=post_discovery_hook,
            )

    async def download_bbox(
        self,
        client: httpx.AsyncClient,
        bbox: tuple[float, float, float, float],
        progress_callback: ProgressCallback = None,
        image_limit: Optional[int] = None,
        post_discovery_hook: Optional[Callable[["StateManager"], None]] = None,
    ):
        """
        Download all images within a bounding box.

        This is the core download logic shared by download_city and test_download.

        Args:
            client: httpx AsyncClient
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            progress_callback: Optional progress callback
            image_limit: Optional limit on number of images to download
            post_discovery_hook: Optional callback to run after discovery
        """
        self.state.reset_in_progress()
        await discover_phase(
            bbox, self.state, client, self.access_token, self.rate_limit_delay
        )

        # Selection Phase
        if post_discovery_hook:
            post_discovery_hook(self.state)
        else:
            # Default behavior: download everything discovered
            self.state.mark_all_pending_images_for_download()

        await self._download_phase(image_limit, client)

    async def _download_phase(self, image_limit, client):
        total_images = self.state.get_total_images()
        if total_images == 0:
            logger.info("nothing to download for the given bbox. exiting.")
            return
        downloaded = self.state.get_num_downloaded_images()
        effective_total = (
            min(image_limit, total_images) if image_limit else total_images
        )
        logger.info(
            f"starting image download at {downloaded} / {effective_total} (total={total_images}, limit={image_limit})"
        )

        images_processed = downloaded
        while images_processed < effective_total:
            batch_size = max(0, min(50, effective_total - images_processed))
            logger.info(f"downloading batch, size={batch_size}")
            tried_downloading = await self._download_batch(client, batch_size)
            images_processed += tried_downloading
            logger.info(f"download: {images_processed} / {effective_total}")
            if tried_downloading == 0:
                # done everything
                break

        final_downloaded = self.state.get_num_downloaded_images()
        logger.info(
            f"finished image download: {final_downloaded} / {effective_total} (total={total_images}, limit={image_limit})"
        )

    async def _download_batch(self, client, batch_size) -> int:
        pending = self.state.get_pending_images(limit=batch_size)
        if not pending:
            return 0

        tasks = [self._download_image(client, image_id) for image_id in pending]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await f
        return len(pending)

    async def _download_image(self, client: httpx.AsyncClient, image_id: str):
        """Download a single image."""
        metadata = await download_single_image_with_retry(
            image_id,
            self.image_size,
            self.state,
            self.images_dir,
            client,
            self.access_token,
        )
        if self.save_metadata:
            meta_path = self.metadata_dir / f"{image_id}.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)


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
